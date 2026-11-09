"""Lane change and on-ramp merge: receding-horizon pdSTL planning in traffic.

The road, traffic, task and limits come from a YAML in
configs/scenarios/lane/.
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch

from models.estimation import observe_gaussian, sample_gaussian
from models.rollouts import lane_rollout
from planning.environment import (
    build_lane_merge_environment,
    lane_contains_footprint,
    lane_deadline_missed,
    lane_goal_reached,
    lane_has_collision,
    lane_headway_overlap,
    lane_headway_violated,
    lane_lead_gap,
    lane_local_window,
    lane_target_contains,
    lane_traffic_acceleration,
    lane_yield_contains,
    lane_yield_possible,
    lane_yield_reached,
)
from planning.runners import output_path, save_result, setup_problem
from utils import cpu_model, load_config

logger = logging.getLogger(__name__)


class LaneState(NamedTuple):
    """Posterior beliefs, estimated task progress, and hidden simulation truth.

    The first six fields retain the previous state tuple's belief interface.
    Physical outcomes use the separate true states and true task progress.
    Merge and yield have separate estimated and true dwell/completion fields.
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
    yield_streak: int = 0
    true_yield_streak: int = 0
    yield_completed: int | None = None
    true_yield_completed: int | None = None


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


def run(
    config_path,
    *,
    show=True,
    save=True,
    live=False,
    live_optimization=False,
    optimization_every=20,
    verbose=False,
    max_steps=None,
):
    """Execute one lane scenario and show its figure and results table.

    Args:
        config_path: Lane YAML; outputs are named after its file stem.
        show, save: Display and/or save the figure and table images.
        live, live_optimization, optimization_every: Live execution view.
        verbose: Log progress.
        max_steps: Override the configured number of executed steps.

    Returns:
        The MPCResult.
    """
    s = build(load_config(config_path))
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
    stem = Path(config_path).stem
    if show or save:
        _present(result, s, stem=stem, show=show, save=save)
    save_result(result, stem, save)
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
        raise TypeError(
            "initial_state must be a LaneState with separate truth"
        )
    traffic_q = s.cfg["traffic_q_std"] ** 2 * torch.eye(
        4, dtype=s.dyn.B.dtype, device=s.dyn.device
    )
    acceleration = s.cfg["accel_bounds"]
    accel_bounds = (acceleration["longitudinal"], acceleration["lateral"])
    names = [car["name"] for car in s.cfg["traffic"]]

    def rollout_at(state, step):
        return lane_rollout(
            s.dyn,
            state[0],
            state[1],
            state[3],
            state[4],
            traffic_q,
            names,
            s.cfg["speed_bounds"],
            accel_bounds,
        )

    def spec_at(state, step):
        return _specification(s, make_spec, state, step)

    started = perf_counter()
    first_guess = best_first_guess(s, initial_state, rollout_at, spec_at)
    search_time = perf_counter() - started
    result = s.planner.run_receding_horizon(
        initial_state,
        make_rollout=rollout_at,
        make_spec=spec_at,
        execute=_executor(
            s,
            traffic_q,
            ego_disturbances,
            traffic_disturbances,
            verbose,
            ego_observation_noise=ego_observation_noise,
            traffic_observation_noise=traffic_observation_noise,
            generator=generator,
            filter_q=_filter_q(s, traffic_q),
        ),
        is_done=_outcome(s),
        max_steps=s.cfg["T_SIM"] if max_steps is None else max_steps,
        init_guess=first_guess,
        verbose=verbose,
        on_step=on_step,
        on_iteration=on_iteration,
        callback_every=callback_every,
        capture_planning_failures=True,
    )
    if result.stopped_reason == "max_steps":
        result.stopped_reason = "step_limit"
    result.warm_start_search_time = search_time
    return result


def initial_state_of(s, *, generator=None):
    """Sample physical initial states from the unchanged Gaussian priors.

    Traffic keeps its lane: its true lateral speed starts at zero, so a car
    cannot drift across the lane over the run. The belief prior on that speed
    is unchanged, so the planner still does not know it is zero.

    Args:
        s: Lane problem from `build`.
        generator: Torch generator for the samples.

    Returns:
        LaneState with beliefs at the priors and sampled true states.
    """
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
    traffic_true[:, 3] = 0.0
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
        yield_streak=_yield_counter(s, s.state[0], 0),
        true_yield_streak=_yield_counter(s, ego_true, 0),
    )


def initial_controls(s, warm=None):
    """Controls for one warm start (default: the first configured one).

    A warm start holds one longitudinal acceleration and steers `lateral`
    for `switch_step` steps, then `-lateral` for as long, then straight.
    """
    warm = s.cfg["warm_starts"][0] if warm is None else warm
    controls = torch.zeros(
        s.cfg["H"], 2, dtype=s.dyn.B.dtype, device=s.dyn.device
    )
    controls[:, 0] = warm["longitudinal"]
    switch = warm["switch_step"]
    controls[:switch, 1] = warm["lateral"]
    controls[switch : 2 * switch, 1] = -warm["lateral"]
    return controls


def best_first_guess(s, state, rollout_at, spec_at):
    """Solve the first window from every warm start; return the best plan.

    "Best" is the highest exact lower score, then the highest smooth score
    (so two plans that both score 0 are still told apart). On a full tie the
    earlier warm start wins, so the YAML order states the preference openly.
    This search runs once before the loop, outside any replanning time.
    """
    best, best_score = None, None
    for warm in s.cfg["warm_starts"]:
        try:
            plan = s.planner.optimize_window(
                rollout_at(state, 0),
                spec=spec_at(state, 0),
                init_guess=initial_controls(s, warm),
            )
        except (RuntimeError, ValueError):
            continue  # the loop itself reports a window it cannot plan
        score = (plan.hard_interval[0], plan.smooth_lower)
        if best is None or score > best_score:
            best, best_score = plan, score
    return initial_controls(s) if best is None else best.controls


def success_counter(environment, mean, counter):
    """Consecutive executed steps inside the target lane."""
    inside = lane_target_contains(environment.metadata, mean)
    return counter + 1 if inside else 0


def _yield_counter(s, state, counter):
    if s.env.metadata["task"]["spec"] != "merge_or_yield":
        return 0
    return counter + 1 if lane_yield_contains(s.env, state) else 0


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


def _present(result, s, *, stem, show, save):
    """Display or save the selected run's figure and table."""
    from visualization.figures import plot_lane_merge

    plot_lane_merge(
        result,
        s.env,
        dt=s.cfg["dt"],
        save_path=output_path(stem, ".png") if save else None,
        show=show,
    )


def _filter_q(s, traffic_q):
    """Traffic process noise the Kalman filter assumes.

    The true traffic noise `traffic_q` plus white-noise longitudinal
    acceleration of std `traffic_filter_accel_std` (m/s^2, default 0), so
    the estimate can follow a car that brakes. The planner's prediction and
    the true traffic keep `traffic_q`.

    Args:
        s: A built lane problem.
        traffic_q: [4, 4] true traffic process noise.

    Returns:
        [4, 4] filter process noise.
    """
    sigma = s.cfg.get("traffic_filter_accel_std", 0.0)
    dt = s.cfg["dt"]
    block = sigma**2 * torch.tensor(
        [[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]],
        dtype=traffic_q.dtype,
        device=traffic_q.device,
    )
    q = traffic_q.clone()
    q[0::2, 0::2] += block  # x and vx
    return q


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
    filter_q=None,
):
    filter_q = traffic_q if filter_q is None else filter_q

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
        traffic_true = traffic_true + _scripted_braking(s, state, step)
        traffic_true = traffic_true + draw(
            traffic_disturbances, step, traffic_true, traffic_q
        )
        mean, covariance = s.dyn.step(state.ego_mean, state.ego_cov, control)
        traffic_mean = state.traffic_mean @ s.dyn.A.T
        traffic_cov = s.dyn.A @ state.traffic_cov @ s.dyn.A.T + filter_q
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
        yield_counter = _yield_counter(s, mean, state.yield_streak)
        true_yield_counter = _yield_counter(
            s, ego_true, state.true_yield_streak
        )
        yield_completed = _yield_latch(
            s, state.yield_completed, mean, step + 1, yield_counter
        )
        true_yield_completed = _yield_latch(
            s,
            state.true_yield_completed,
            ego_true,
            step + 1,
            true_yield_counter,
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
            yield_counter,
            true_yield_counter,
            yield_completed,
            true_yield_completed,
        )

    return execute


def _scripted_braking(s, state, step):
    """Position/velocity change from configured brakes; never reverses."""
    dt = s.cfg["dt"]
    accel = torch.tensor(
        lane_traffic_acceleration(s.cfg["traffic"], step * dt),
        dtype=state.traffic_true.dtype,
        device=state.traffic_true.device,
    )
    accel = torch.maximum(accel, -state.traffic_true[:, 2].clamp(min=0) / dt)
    change = torch.zeros_like(state.traffic_true)
    change[:, 0] = 0.5 * accel * dt**2
    change[:, 2] = accel * dt
    return change


def _latch(s, completed, mean, step, counter):
    """The step the dwell task first finished, kept once it has."""
    if completed is not None:
        return completed
    return (
        step + 1 if lane_goal_reached(s.env, mean, step + 1, counter) else None
    )


def _yield_latch(s, completed, state, step, counter):
    if (
        completed is not None
        or s.env.metadata["task"]["spec"] != "merge_or_yield"
    ):
        return completed
    return step if lane_yield_reached(s.env, state, step, counter) else None


def _physical_violation(s, state):
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
    if lane_headway_violated(s.env, state.ego_true, state.traffic_true):
        return "headway_violation"
    return None


def _outcome(s):
    task = s.env.metadata["task"]

    def is_done(state, step):
        violation = _physical_violation(s, state)
        if violation:
            return violation
        yielded = state.true_yield_completed
        merged = state.true_completed
        if yielded is not None and (merged is None or yielded < merged):
            if not lane_yield_contains(s.env, state.ego_true):
                return "yield_retention_violation"
            settle = task["settle_steps"]
            return "yielded" if step >= yielded + settle else None
        if merged is not None:
            return _merged_outcome(s, state, step)
        if lane_deadline_missed(
            s.env, state.ego_true, step, state.true_streak
        ):
            if task["spec"] == "merge_or_yield" and lane_yield_possible(
                s.env, step, state.true_yield_streak
            ):
                return None
            return "deadline_missed"
        return None

    return is_done


def _merged_outcome(s, state, step):
    """Truth after the dwell: stay in the lane, settle by the merge deadline.

    The spec lets the dwell finish as late as the window allows and only
    asks for low lateral speed after that, so an unsettled car is a failure
    only once the window's last finish time has passed.
    """
    task = s.env.metadata["task"]
    if not lane_target_contains(s.env.metadata, state.ego_true):
        return "lane_retention_violation"
    settled = bool(state.ego_true[3].abs() <= task["settle_lateral_speed"])
    if not settled:
        late = step >= task["start_end_steps"][1]
        return "settling_violation" if late else None
    if step < state.true_completed + task["settle_steps"]:
        return None
    return "success"


def _specification(s, factory, state, step):
    # Estimated completion can precede true completion. Keep a valid
    # zero-remaining-dwell formula while the physical monitor keeps running.
    credit = min(state.streak, s.env.metadata["task"]["dwell_steps"] + 1)
    completed = state.completed is not None
    missed = not completed and lane_deadline_missed(
        s.env, state.ego_mean, step, credit
    )
    yielded = False
    if s.env.metadata["task"]["spec"] == "merge_or_yield":
        yielded = state.yield_completed is not None and (
            state.completed is None or state.yield_completed < state.completed
        )
        if missed and (
            lane_target_contains(s.env.metadata, state.ego_mean)
            or not lane_yield_possible(s.env, step, state.yield_streak)
        ):
            # Already in the lane, or nothing left to yield to: plan to stay.
            # The physical monitor still decides success or failure.
            completed, missed = True, False
    elif missed:
        # An expired estimated window leaves only lane retention to plan;
        # the separate physical monitor still decides success or failure.
        completed = True
    environment = lane_local_window(
        s.env,
        step,
        streak=credit,
        completed=completed and not yielded,
        yield_streak=state.yield_streak,
        yield_completed=yielded,
        merge_unavailable=missed,
    )
    return (
        factory(environment, state, step)
        if factory is not None
        else environment.get_specification(s.cfg["H"])
    )


# --- Results -----------------------------------------------------------------


def summarize(result, env, dt):
    """One run's outcome, safety margins, scores and timings, by name.

    Physical values come from the true states. Times are in milliseconds;
    "replan" times cover building the spec and solving one window.
    """
    ego = np.asarray(
        [state.ego_true[:2].cpu().numpy() for state in result.states]
    )
    traffic = np.asarray(
        [state.traffic_true[:, :2].cpu().numpy() for state in result.states]
    )
    vehicle = env.metadata["ego_vehicle"]
    half_extents = np.asarray(
        [
            [
                (vehicle["width"] + car["width"]) / 2,
                (vehicle["height"] + car["height"]) / 2,
            ]
            for car in env.metadata["traffic"]
        ]
    )
    # Signed distance between footprints (negative when they overlap).
    separation = np.abs(traffic - ego[:, None, :]) - half_extents
    clearance = np.linalg.norm(np.maximum(separation, 0), axis=-1)
    clearance += np.minimum(separation.max(axis=-1), 0)
    final = result.states[-1]
    completion, branch = min(
        [
            (step, name)
            for step, name in [
                (final.true_completed, "merge"),
                (final.true_yield_completed, "yield"),
            ]
            if step is not None
        ],
        default=(None, None),
    )
    optimizer = [plan.planning_time * 1000 for plan in result.window_plans]
    scores = [plan.hard_interval[0] for plan in result.window_plans]
    alphas = [plan.alpha for plan in result.window_plans]
    replanning = [plan.replanning_time * 1000 for plan in result.window_plans]
    if result.failed_replanning_time is not None:
        replanning.append(result.failed_replanning_time * 1000)
    later = replanning[1:]
    rows = {
        "Outcome": result.stopped_reason.replace("_", " "),
        "Realized branch": branch,
        "Completion time [s]": None if completion is None else completion * dt,
        "Minimum traffic clearance [m]": float(clearance.min()),
    }
    if "headway" in env.metadata["task"]:  # reported even when not enforced
        low, high = lane_headway_overlap(env.metadata)
        gaps = [
            lane_lead_gap(env, state.ego_true, state.traffic_true)
            for state in result.states
            if low <= state.ego_true[1] <= high
        ]
        rows["Minimum lead gap in lane [m]"] = min(gaps) if gaps else None
    rows["Lowest window score"] = min(scores) if scores else None
    if alphas and alphas[0] is not None:
        rows["Windows meeting alpha"] = float(
            np.mean([score >= alphas[0] for score in scores])
        )
    rows.update(
        {
            "Warm-start search [ms]": getattr(
                result, "warm_start_search_time", 0.0
            )
            * 1000,
            "Optimizer median [ms]": float(np.median(optimizer))
            if optimizer
            else None,
            "First replan [ms]": replanning[0] if replanning else None,
            "Later replans median [ms]": float(np.median(later))
            if later
            else None,
            "Later replans p95 [ms]": float(np.percentile(later, 95))
            if later
            else None,
            "Later replans maximum [ms]": max(later) if later else None,
        }
    )
    return rows


def batch(config_path, seeds, *, save=True, budget_ms=200.0, max_steps=None):
    """Run one scenario once per seed, without plots, and summarize.

    One unrecorded one-step trial of the first seed runs first, so CUDA
    kernel loading is not timed in the first replan (harmless on CPU).

    Args:
        config_path: Lane YAML.
        seeds: Seeds to run; each changes the noise draws only.
        save: Write `outputs/<stem>_batch.json` (per-trial rows + table).
        budget_ms: Replanning time to compare against (the 0.2 s step).
        max_steps: Override the configured number of executed steps.

    Returns:
        (per-trial rows, summary table) as dictionaries.
    """
    seeds = [int(seed) for seed in seeds]
    trial(_seeded(config_path, seeds[0]), max_steps=1)  # warm-up, not kept
    trials, later = [], []
    for seed in seeds:
        s = _seeded(config_path, seed)
        cfg = s.cfg
        result = trial(s, max_steps=max_steps)
        row = {"seed": int(seed), **summarize(result, s.env, cfg["dt"])}
        row["window_scores"] = [
            p.hard_interval[0] for p in result.window_plans
        ]
        row["replan_ms"] = [
            p.replanning_time * 1000 for p in result.window_plans
        ]
        trials.append(row)
        later += row["replan_ms"][1:]
        print(f"seed {seed}: {row['Outcome']}", flush=True)
    branches = Counter(row["Realized branch"] or "none" for row in trials)
    alpha = s.planner.cfg["alpha"]
    scores = [score for row in trials for score in row["window_scores"]]
    completions = [
        row["Completion time [s]"]
        for row in trials
        if row["Completion time [s]"] is not None
    ]
    table = {
        "scenario": Path(config_path).stem,
        "trials": len(trials),
        "outcomes": dict(Counter(row["Outcome"] for row in trials)),
        "branches": {
            name: branches[name] for name in ("merge", "yield", "none")
        },
        "completion time median [s]": (
            float(np.median(completions)) if completions else None
        ),
        "minimum clearance [m]": min(
            row["Minimum traffic clearance [m]"] for row in trials
        ),
        "minimum lead gap in lane [m]": min(
            (
                row["Minimum lead gap in lane [m]"]
                for row in trials
                if row.get("Minimum lead gap in lane [m]") is not None
            ),
            default=None,
        ),
        "lowest window score": min(
            (
                row["Lowest window score"]
                for row in trials
                if row["Lowest window score"] is not None
            ),
            default=None,
        ),
        "windows meeting alpha": (
            None
            if alpha is None
            else float(np.mean([score >= alpha for score in scores]))
        ),
        "first replan median [ms]": float(
            np.median([row["First replan [ms]"] for row in trials])
        ),
        "later replans median [ms]": float(np.median(later)),
        "later replans p95 [ms]": float(np.percentile(later, 95)),
        "later replans max [ms]": float(max(later)),
        f"fraction over {budget_ms:g} ms": float(
            np.mean([t > budget_ms for t in later])
        ),
        "machine": _machine(s),
    }
    for key, value in table.items():
        print(f"{key}: {value}")
    if save:
        path = output_path(f"{Path(config_path).stem}_batch", ".json")
        path.write_text(
            json.dumps({"table": table, "trials": trials}, indent=1)
        )
    return trials, table


def _seeded(config_path, seed):
    """Build one lane config with its seed replaced.

    Args:
        config_path: Lane YAML.
        seed: Seed for the trial's noise draws.

    Returns:
        The built problem (see `build`).
    """
    cfg = load_config(config_path)
    cfg["seed"] = seed
    return build(cfg)


def _machine(s):
    """Hardware and planner settings the batch timings depend on.

    Args:
        s: A built lane problem.

    Returns:
        Dict with CPU model, torch threads, the CPUs the process may run on,
        device, GPU name (None off CUDA), torch version, horizon H and the
        planner's max_iters.
    """
    device = torch.device(s.dyn.device)
    return {
        "cpu": cpu_model(),
        "threads": torch.get_num_threads(),
        "allowed cpus": sorted(os.sched_getaffinity(0)),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "torch": torch.__version__,
        "H": s.cfg["H"],
        "max_iters": s.planner.cfg["max_iters"],
    }


# --- Paper outputs -------------------------------------------------------------

LANE_DIR = "configs/scenarios/lane"
# Paper figure: one simulation frame per scene, (name, title, time in s).
# The first is drawn full width, the next two side by side. The exact specs
# at t = 0 (absolute deadlines, kept by every replan), for the caption:
#   A: G[0,6] safe AND M(1 s, [2.6, 4.0] s)
#   C: G[0,6] safe AND (M(0.6 s, [0.8, 3.4] s) OR F[2.4,4.0] G[0,inf) stop)
#   M(d, I) = F_I (G[0,d] lane AND F[d,d] x <= x_end
#                  AND G[d,inf) (lane AND |v_y| <= 0.3))
#   x_end = ramp end - half the car length: 157.75 m (A), 137.75 m (C).
#   "safe" includes the gore: no part of the car in the main lane before
#   x = 20 m (A) or 0 m (C).
FIGURE_SCENES = (
    # (scene, title, frame time in s or None for when the task was done on
    #  truth, ego ghost times in s, score label). The formulas go in the
    #  caption.
    ("merge", "(a) Merge", 4.8, (0, 1.6, 3.2), "(a) merge"),
    ("merge_or_yield", "(b) Dense traffic", None, (0, 0.8), "(b) dense"),
    ("merge_or_yield_gap", "(c) One open gap", None, (0, 1.2), "(c) gap"),
)
# The braking car whose brake time is marked in the score panel.
FIGURE_EVENT_CAR = "lead"
# Rows of the paper table (scene, row name).
TABLE_SCENES = (
    ("merge", "Merge"),
    ("headway", "Merge + headway"),
    ("merge_or_yield", "Merge or yield, dense"),
    ("merge_or_yield_gap", "Merge or yield, gap"),
)


def paper_figure(scenes=FIGURE_SCENES, *, save=True, show=False):
    """The lane paper figure from saved runs, in both page layouts.

    Run each scene once with `run(...)` first (it saves outputs/<name>.pt).
    The first scene is the wide frame, the others short frames, and a last
    panel plots each run's per-window lower score. The event car's brake
    time in the first scene is marked in that panel.

    Args:
        scenes: Rows like FIGURE_SCENES.
        save: Write outputs/lane_paper.pdf (two-column) and
            outputs/lane_paper_column.pdf (one-column), each with a PNG.
        show: Display the figures.

    Returns:
        {"two_column": (figure, axes), "one_column": (figure, axes)}.
    """
    from visualization.figures import plot_lane_paper

    panels = []
    for name, title, time, ghosts, label in scenes:
        cfg = load_config(f"{LANE_DIR}/lane_{name}.yaml")
        path = output_path(f"lane_{name}", ".pt")
        if not path.exists():
            raise FileNotFoundError(f"run lane_{name}.yaml first ({path})")
        result = torch.load(path, weights_only=False)
        step = _figure_step(result, cfg["dt"], time)
        env = build_lane_merge_environment(cfg)
        panels.append((title, result, env, step, ghosts, label))
    cfg = load_config(f"{LANE_DIR}/lane_{scenes[0][0]}.yaml")
    car = next(c for c in cfg["traffic"] if c["name"] == FIGURE_EVENT_CAR)
    letter = scenes[0][4].split()[0]
    figures = {}
    for key, stem, column in (
        ("two_column", "lane_paper", False),
        ("one_column", "lane_paper_column", True),
    ):
        figures[key] = plot_lane_paper(
            panels,
            dt=cfg["dt"],
            alpha=cfg["planner"].get("alpha"),
            events=(
                (car["brake"]["at"], f"{FIGURE_EVENT_CAR} brakes {letter}"),
            ),
            column=column,
            save_path=output_path(stem, ".pdf") if save else None,
            show=show,
        )
    return figures


def _figure_step(result, dt, time):
    """The step a figure frame shows.

    Args:
        result: MPCResult of the run.
        dt: Step length in seconds.
        time: Frame time in seconds, or None for when the task was done on
            truth (the last step if it never was).

    Returns:
        Step index into result.states.
    """
    last = len(result.states) - 1
    if time is not None:
        return min(round(time / dt), last)
    final = result.states[-1]
    done = final.true_completed or final.true_yield_completed
    return last if done is None else min(done, last)


def latex_table(scenes=TABLE_SCENES, *, save=True):
    """A LaTeX tabular from saved batch results (run `batch` per scene first).

    Columns: completed trials, merged/yielded trials, closest traffic
    clearance, closest lead gap, lowest window score, percent of windows with
    score >= alpha, and later replanning median / p95 in milliseconds. The
    first-replan medians go in a LaTeX comment under the table.
    """
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        (
            r"Scene & Completed & Merged/yielded & Clearance (m) & "
            r"Lead gap (m) & Lowest $\widehat{r}^{\varphi,\downarrow}_0$ & "
            r"$\widehat{r}^{\varphi,\downarrow}_0 \geq \alpha$ (\%) & "
            r"Replan med./p95 (ms) \\"
        ),
        r"\midrule",
    ]
    first = []
    for name, row_name in scenes:
        path = output_path(f"lane_{name}_batch", ".json")
        table = json.loads(path.read_text())["table"]
        done = sum(
            count
            for outcome, count in table["outcomes"].items()
            if outcome in ("success", "yielded")
        )
        gap = table.get("minimum lead gap in lane [m]")
        lowest = table["lowest window score"]
        meeting = table["windows meeting alpha"]
        branches = table["branches"]
        lines.append(
            f"{row_name} & {done}/{table['trials']} & "
            f"{branches['merge']}/{branches['yield']} & "
            f"{table['minimum clearance [m]']:.1f} & "
            f"{'--' if gap is None else f'{gap:.1f}'} & "
            f"{'--' if lowest is None else f'{lowest:.2f}'} & "
            f"{'--' if meeting is None else f'{100 * meeting:.1f}'} & "
            f"{table['later replans median [ms]']:.0f} / "
            f"{table['later replans p95 [ms]']:.0f} \\\\"
        )
        first.append(f"{row_name} {table['first replan median [ms]']:.0f} ms")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"% first replan median: {', '.join(first)}",
    ]
    text = "\n".join(lines)
    if save:
        output_path("lane_table", ".tex").write_text(text + "\n")
    print(text)
    return text
