"""Reach-avoid: stay safe, avoid every obstacle, visit regions, reach a goal.

The regions and settings come from a YAML in configs/scenarios/reach_avoid/.
"""

import torch

from models.dynamics import SingleIntegrator
from planning.environment import (
    build_reach_avoid_environment,
    shortest_route,
)
from planning.runners import (
    control_unit,
    monte_carlo,
    output_path,
    setup_problem,
    solve,
)


def build(cfg, device=None):
    """Environment, pdSTL spec, and initial guess for one config.

    The optimizer starts on the shortest collision-free route: gradient
    ascent cannot move a path across an obstacle, and far from a region its
    probability has no gradient. The route keeps only a small clearance;
    pdSTL optimization adds the uncertainty-aware separation.
    """
    s = setup_problem(cfg, device=device)
    s.env = build_reach_avoid_environment(cfg)
    s.spec = s.env.get_specification(cfg["H"])
    s.route = shortest_route(s.env, cfg["x0_mean"][:2], **cfg.get("route", {}))
    s.init_guess = straight_line_guess(cfg, s.dyn, s.route)
    return s


def optimize(s, *, live=False, optimization_every=5, verbose=False):
    """Plan one open-loop control sequence for a built scenario.

    Args:
        s: The problem from `build`.
        live, optimization_every: Stream sampled iterates to a live view.
        verbose: Log optimizer progress.

    Returns:
        The PlanResult with the highest exact lower robustness.
    """
    title = _title(s)
    visual = s.cfg.get("visualization", {})
    on_iteration, finish_live = (
        _live_view(s, title, visual) if live else (None, None)
    )
    result = solve(
        s,
        label=title,
        verbose=verbose,
        on_iteration=on_iteration,
        callback_every=optimization_every if live else 1,
    )
    if finish_live is not None:
        finish_live(result)
    return result


def validate(s, result):
    """Monte Carlo P(phi) of the plan under noise; stored on the result.

    Returns:
        (rate, lower, upper) with a 95% Clopper-Pearson interval.
    """
    settings = s.cfg.get("monte_carlo", {})
    result.monte_carlo = monte_carlo(s, result, **settings)
    rate, lower, upper = result.monte_carlo
    print(
        f"{_title(s)}: Monte Carlo P(phi) = {rate:.3f} "
        f"[{lower:.3f}, {upper:.3f}], N = {settings.get('samples', 5000)}",
        flush=True,
    )
    return result.monte_carlo


def _title(s):
    return s.cfg.get("name", "Reach–Avoid")


def straight_line_guess(cfg, dyn, waypoints, kp=4.0, kd=4.0):
    """Controls whose mean path follows straight lines through waypoints.

    Args:
        cfg: Scenario config (x0_mean, H, dt).
        dyn: Single- or double-integrator dynamics.
        waypoints: [[x, y], ...] after the start.
        kp, kd: PD gains of the double-integrator tracker.

    Returns:
        [H, 2] controls within u_max.
    """
    reference = _straight_line_reference(cfg, waypoints, dyn.B.dtype)
    velocity = (reference[1:] - reference[:-1]) / cfg["dt"]
    if isinstance(dyn, SingleIntegrator):
        return velocity.clamp(-dyn.u_max, dyn.u_max).to(dyn.device)
    state = torch.tensor(cfg["x0_mean"], dtype=dyn.B.dtype)
    controls = []
    for target, target_velocity in zip(reference[1:], velocity):
        error = kp * (target - state[:2]) + kd * (target_velocity - state[2:])
        control = error.clamp(-dyn.u_max, dyn.u_max)
        state = dyn.A.cpu() @ state + dyn.B.cpu() @ control
        controls.append(control)
    return torch.stack(controls).to(dyn.device)


def _straight_line_reference(cfg, waypoints, dtype, arrival=0.8):
    """Positions along x0 -> waypoints at constant speed, done at arrival*H."""
    points = torch.tensor([cfg["x0_mean"][:2], *waypoints], dtype=dtype)
    segments = points[1:] - points[:-1]
    ends = torch.cat((points.new_zeros(1), segments.norm(dim=1).cumsum(0)))
    progress = (torch.arange(cfg["H"] + 1) / (arrival * cfg["H"])).clamp(max=1)
    distance = ends[-1] * progress
    index = (torch.searchsorted(ends, distance, right=True) - 1).clamp(
        0, len(segments) - 1
    )
    length = (ends[index + 1] - ends[index]).clamp_min(1e-9)
    fraction = ((distance - ends[index]) / length).unsqueeze(1)
    return points[index] + fraction * segments[index]


def _live_view(s, title, visual):
    from visualization.live_plots import create_reach_avoid_live_view

    _, _, on_iteration, finish = create_reach_avoid_live_view(
        s.env,
        lambda controls: s.planner.evaluate_controls(
            s.rollout, controls, spec=s.spec
        ),
        dt=s.cfg["dt"],
        title=title,
        max_iters=s.planner.cfg["max_iters"],
        ellipse_every=visual.get("ellipse_every", 4),
        alpha=s.planner.cfg["alpha"],
    )
    return on_iteration, finish


def present(result, s, *, stem, show=True, save=True):
    """Figure and animation of the plan; saved under `stem` if `save`."""
    title = _title(s)
    from visualization.animation import animate_reach_avoid
    from visualization.figures import plot_reach_avoid

    visual = s.cfg.get("visualization", {})
    plot_reach_avoid(
        result,
        s.env,
        dt=s.cfg["dt"],
        u_max=s.dyn.u_max,
        control_unit=control_unit(s.dyn),
        title=title,
        ellipse_every=visual.get("ellipse_every", 4),
        ellipse_confidence=visual.get("ellipse_confidence", 0.95),
        save_path=output_path(stem, ".png") if save else None,
        show=show,
    )
    animate_reach_avoid(
        result,
        s.env,
        dt=s.cfg["dt"],
        title=title,
        fps=visual.get("animation_fps", 6),
        filename=output_path(stem, ".gif") if save else None,
        show=show,
    )
