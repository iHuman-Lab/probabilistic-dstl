"""Lane change and on-ramp merge: receding-horizon pdSTL planning in traffic.

The road, traffic, task and limits come from a YAML in
configs/scenarios/lane/.
"""

import logging
from typing import NamedTuple

import torch

from models.estimation import observe_gaussian, sample_gaussian
from models.rollouts import lane_rollout
from planning.environment import (
    build_lane_merge_environment,
    lane_contains_footprint,
    lane_deadline_missed,
    lane_goal_reached,
    lane_has_collision,
    lane_local_window,
    lane_target_contains,
)
from planning.runners import output_path, setup_problem

logger = logging.getLogger(__name__)


class LaneState(NamedTuple):
    """Posterior beliefs, estimated task progress, and hidden simulation truth.

    The first six fields retain the previous state tuple's belief interface.
    Physical outcomes use the separate true states and true task progress.
    `completed`/`true_completed` latch the step the dwell task finished.
    """

    ego_mean: torch.Tensor
    ego_cov: torch.Tensor
    streak: int
    traffic_mean: torch.Tensor
    traffic_cov: torch.Tensor
    entry: int | None
    ego_true: torch.Tensor
    traffic_true: torch.Tensor
    true_streak: int
    true_entry: int | None
    completed: int | None = None
    true_completed: int | None = None


def _observation_covariance(cfg, key, reference):
    std = torch.as_tensor(
        cfg.get("observation", {}).get(key, 0.1),
        dtype=reference.dtype,
        device=reference.device,
    )
    if std.shape not in (torch.Size([]), torch.Size([4])):
        raise ValueError(f"observation.{key} must be scalar or length 4")
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError(f"observation.{key} must be finite and positive")
    return torch.diag(std.expand(4).square())


def _generator(s):
    return torch.Generator(device=s.dyn.device).manual_seed(
        s.cfg.get("seed", 0)
    )


def build(cfg, device=None):
    """Dynamics, planner and road environment for one lane config."""
    s = setup_problem(cfg, device=device)
    s.env = build_lane_merge_environment(cfg, s.dyn.device)
    s.ego_observation_cov = _observation_covariance(cfg, "ego_std", s.state[0])
    s.traffic_observation_cov = _observation_covariance(
        cfg, "traffic_std", s.state[0]
    )
    return s


def simulate(
    s,
    *,
    live=False,
    live_optimization=False,
    optimization_every=20,
    verbose=False,
    max_steps=None,
):
    """Execute one lane scenario closed loop (receding-horizon planning).

    Args:
        s: The problem from `build`.
        live, live_optimization, optimization_every: Live execution view.
        verbose: Log progress.
        max_steps: Override the configured number of executed steps.

    Returns:
        The MPCResult.
    """
    generator = _generator(s)
    initial_state = initial_state_of(s, generator=generator)
    label = "Lane merge" if s.env.metadata.get("ramp") else "Lane change"
    on_step, on_iteration, callback_every = _observers(
        s,
        label=label,
        live=live or live_optimization,
        optimization_every=optimization_every,
        initial_state=initial_state,
    )
    result = trial(
        s,
        initial_state=initial_state,
        generator=generator,
        max_steps=max_steps,
        verbose=verbose,
        on_step=on_step,
        on_iteration=on_iteration,
        callback_every=callback_every,
    )
    lower, upper = (
        result.window_plans[-1].hard_interval
        if result.window_plans
        else (float("nan"), float("nan"))
    )
    print(
        f"{label}: {result.stopped_reason} after "
        f"{len(result.window_plans)} steps; last window "
        f"[rho_lower, rho_upper] = [{lower:.4f}, {upper:.4f}]",
        flush=True,
    )
    return result


def trial(
    s,
    *,
    initial_state=None,
    ego_disturbances=None,
    traffic_disturbances=None,
    ego_observation_noise=None,
    traffic_observation_noise=None,
    generator=None,
    make_spec=None,
    max_steps=None,
    verbose=False,
    on_step=None,
    on_iteration=None,
    callback_every=1,
):
    """Run one trial with separate truth and beliefs, optionally fixing noise.

    Process disturbances and observation noise are additive tensors indexed
    by executed step. A local seeded generator controls all other draws.
    Initial beliefs are priors; the first measurement follows the first action.
    """
    generator = _generator(s) if generator is None else generator
    initial_state = (
        initial_state_of(s, generator=generator)
        if initial_state is None
        else initial_state
    )
    if not isinstance(initial_state, LaneState):
        raise ValueError(
            "initial_state must be a LaneState with separate truth"
        )
    traffic_q = s.cfg["traffic_q_std"] ** 2 * torch.eye(
        4, dtype=s.dyn.B.dtype, device=s.dyn.device
    )
    acceleration = s.cfg["accel_bounds"]
    accel_bounds = (acceleration["longitudinal"], acceleration["lateral"])
    names = [car["name"] for car in s.cfg["traffic"]]
    result = s.planner.run_receding_horizon(
        initial_state,
        make_rollout=lambda state, step: lane_rollout(
            s.dyn,
            state[0],
            state[1],
            state[3],
            state[4],
            traffic_q,
            names,
            s.cfg["speed_bounds"],
            accel_bounds,
        ),
        make_spec=lambda state, step: _specification(
            s, make_spec, state, step
        ),
        execute=_executor(
            s,
            traffic_q,
            ego_disturbances,
            traffic_disturbances,
            verbose,
            ego_observation_noise=ego_observation_noise,
            traffic_observation_noise=traffic_observation_noise,
            generator=generator,
        ),
        is_done=_outcome(s),
        max_steps=s.cfg["T_SIM"] if max_steps is None else max_steps,
        init_guess=initial_controls(s),
        verbose=verbose,
        on_step=on_step,
        on_iteration=on_iteration,
        callback_every=callback_every,
        capture_planning_failures=True,
    )
    if result.stopped_reason == "max_steps":
        result.stopped_reason = "step_limit"
    return result


def initial_state_of(s, *, generator=None):
    """Sample physical initial states from the unchanged Gaussian priors."""
    traffic = s.cfg["traffic"]
    mean = torch.tensor(
        [[car["x0"], car["y"], car["speed"], 0.0] for car in traffic],
        dtype=s.dyn.B.dtype,
        device=s.dyn.device,
    )
    covariance = torch.eye(4, device=s.dyn.device).expand(len(traffic), -1, -1)
    covariance = covariance.clone() * s.cfg["traffic_cov_scale"]
    ego_true = sample_gaussian(*s.state, generator=generator)
    traffic_true = sample_gaussian(mean, covariance, generator=generator)
    streak = success_counter(s.env, s.state[0], 0)
    true_streak = success_counter(s.env, ego_true, 0)
    return LaneState(
        *s.state,
        streak,
        mean,
        covariance,
        0 if streak else None,
        ego_true,
        traffic_true,
        true_streak,
        0 if true_streak else None,
    )


def initial_controls(s):
    """The configured first-window guess: accelerate and steer into the lane."""
    warm = s.cfg["warm_start"]
    controls = torch.zeros(
        s.cfg["H"], 2, dtype=s.dyn.B.dtype, device=s.dyn.device
    )
    controls[:, 0] = warm["longitudinal"]
    switch = warm["switch_step"]
    controls[:switch, 1] = warm["lateral"]
    controls[switch : 2 * switch, 1] = -warm["lateral"]
    return controls


def success_counter(environment, mean, counter):
    """Consecutive executed steps inside the target lane."""
    inside = lane_target_contains(environment.metadata, mean)
    return counter + 1 if inside else 0


def _observers(s, *, label, live, optimization_every, initial_state):
    """Live-view callbacks, or (None, None, 1) without a live view."""
    if not live:
        return None, None, 1
    from visualization.live_plots import create_live_view

    _, _, observer = create_live_view(
        s.env,
        initial_state,
        dt=s.cfg["dt"],
        dynamics=s.dyn,
        max_iters=s.planner.cfg["max_iters"],
        label=label,
    )

    def on_step(step, state, plan):
        observer.finish_window(step, plan)
        observer(step, state, plan)

    return on_step, observer.on_iteration, optimization_every


def present(result, s, *, stem, show=True, save=True):
    """Figures and animation of the run; saved under `stem` if `save`."""
    from visualization.animation import animate_mpc
    from visualization.figures import plot_lane_merge

    plot_lane_merge(
        result,
        s.env,
        dt=s.cfg["dt"],
        save_path=output_path(stem, ".png") if save else None,
        show=show,
    )
    animate_mpc(
        result,
        s.env,
        dt=s.cfg["dt"],
        filename=output_path(stem, ".gif") if save else None,
        show=show,
    )


def _executor(
    s,
    traffic_q,
    ego_disturbances,
    traffic_disturbances,
    verbose,
    *,
    ego_observation_noise=None,
    traffic_observation_noise=None,
    generator=None,
):
    def draw(sequence, step, reference, covariance):
        if sequence is not None:
            return sequence[step].to(reference)
        return sample_gaussian(
            torch.zeros_like(reference), covariance, generator
        )

    def execute(state, control, step):
        # Hidden physical states evolve under exactly the prediction dynamics.
        ego_true = s.dyn.A @ state.ego_true + s.dyn.B @ control
        ego_true = ego_true + draw(ego_disturbances, step, ego_true, s.dyn.Q)
        traffic_true = state.traffic_true @ s.dyn.A.T
        traffic_true = traffic_true + draw(
            traffic_disturbances, step, traffic_true, traffic_q
        )
        mean, covariance = s.dyn.step(state.ego_mean, state.ego_cov, control)
        traffic_mean = state.traffic_mean @ s.dyn.A.T
        traffic_cov = s.dyn.A @ state.traffic_cov @ s.dyn.A.T + traffic_q
        observation = ego_true + draw(
            ego_observation_noise, step, ego_true, s.ego_observation_cov
        )
        traffic_observation = traffic_true + draw(
            traffic_observation_noise,
            step,
            traffic_true,
            s.traffic_observation_cov,
        )
        mean, covariance = observe_gaussian(
            mean, covariance, observation, s.ego_observation_cov
        )
        traffic_mean, traffic_cov = observe_gaussian(
            traffic_mean,
            traffic_cov,
            traffic_observation,
            s.traffic_observation_cov,
        )
        counter = success_counter(s.env, mean, state.streak)
        entry = step + 1 if counter == 1 else state.entry if counter else None
        true_counter = success_counter(s.env, ego_true, state.true_streak)
        true_entry = (
            step + 1
            if true_counter == 1
            else state.true_entry
            if true_counter
            else None
        )
        completed = _latch(s, state.completed, mean, step, counter)
        true_completed = _latch(
            s, state.true_completed, ego_true, step, true_counter
        )
        if verbose and step % 5 == 0:
            logger.info(
                "Lane step %d: true position %s", step, ego_true[:2].tolist()
            )
        return LaneState(
            mean,
            covariance,
            counter,
            traffic_mean,
            traffic_cov,
            entry,
            ego_true,
            traffic_true,
            true_counter,
            true_entry,
            completed,
            true_completed,
        )

    return execute


def _latch(s, completed, mean, step, counter):
    """The step the dwell task first finished, kept once it has."""
    if completed is not None:
        return completed
    return (
        step + 1 if lane_goal_reached(s.env, mean, step + 1, counter) else None
    )


def _outcome(s):
    def is_done(state, step):
        if lane_has_collision(
            s.env, state.ego_true[:2], state.traffic_true[:, :2]
        ):
            return "collision"
        if not lane_contains_footprint(
            s.env, float(state.ego_true[0]), float(state.ego_true[1])
        ):
            return "road_violation"
        if (
            not s.cfg["speed_bounds"][0]
            <= float(state.ego_true[2])
            <= s.cfg["speed_bounds"][1]
        ):
            return "speed_violation"
        if state.true_completed is not None:
            # Keep monitoring truth while the ego settles in the target lane.
            settle = s.env.metadata["task"]["settle_steps"]
            return "success" if step >= state.true_completed + settle else None
        if lane_deadline_missed(
            s.env, state.ego_true, step, state.true_streak
        ):
            return "deadline_missed"
        return None

    return is_done


def _specification(s, factory, state, step):
    # Estimated completion can precede true completion. Keep a valid
    # zero-remaining-dwell formula while the physical monitor keeps running.
    credit = min(state.streak, s.env.metadata["task"]["dwell_steps"] + 1)
    completed = state.completed is not None
    if not completed and lane_deadline_missed(
        s.env, state.ego_mean, step, credit
    ):
        # Truth completed but the estimate did not: nothing left to time, so
        # plan to stay and settle while the physical monitor finishes.
        completed = True
    environment = lane_local_window(
        s.env, step, streak=credit, completed=completed
    )
    return (
        factory(environment, state, step)
        if factory is not None
        else environment.get_specification(s.cfg["H"])
    )
