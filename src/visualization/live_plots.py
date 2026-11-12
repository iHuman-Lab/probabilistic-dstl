"""Live views while planning: reach-avoid iterates and lane execution."""

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from visualization.figures import (
    COLORS,
    RHO_INTERVAL,
    RHO_LOWER,
    _draw_ego_vehicle,
    _draw_environment,
    _ellipse,
    _ellipse_indices,
    _ellipse_parameters,
    _ego_truth,
    _move_ego_vehicle,
    _move_obstacle,
    _np,
    _style,
    _unique_legend,
)


def _refresh(fig, interactive):
    if interactive:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)


def _record_interval(record):
    """An iterate's exact interval, or NaNs when it was not scored
    (`select_by: margin` or `last` skips in-loop exact scoring)."""
    if record.hard_interval is None:
        return (float("nan"), float("nan"))
    return record.hard_interval


def create_reach_avoid_live_view(
    env,
    evaluate_controls,
    *,
    dt,
    title="Reach–Avoid",
    max_iters,
    ellipse_every=8,
    alpha=None,
):
    """Live candidate trajectory and its lower robustness per iteration."""
    fig, (ax_map, ax_scores) = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        layout="constrained",
        gridspec_kw={"width_ratios": (1.45, 1)},
    )
    _draw_environment(ax_map, env)
    (path,) = ax_map.plot(
        [],
        [],
        color=COLORS["mean"],
        linewidth=2.2,
        label="Candidate belief mean",
        zorder=8,
    )
    start = ax_map.scatter(
        [], [], color="black", s=55, marker="o", label="Start", zorder=10
    )
    terminal = ax_map.scatter(
        [],
        [],
        color=COLORS["mean"],
        s=55,
        marker="s",
        label="Terminal belief mean",
        zorder=10,
    )
    (score_line,) = ax_scores.plot(
        [],
        [],
        color=COLORS["mean"],
        linewidth=1.7,
        label=f"{RHO_LOWER} (lower bound)",
    )
    if alpha is not None:
        ax_scores.axhline(
            alpha,
            color=COLORS["obstacle"],
            linestyle="--",
            linewidth=1.1,
            label=f"threshold α = {alpha:.2f}",
        )
    ax_scores.set(
        xlabel="optimizer iteration",
        ylabel=f"lower robustness {RHO_LOWER}",
        ylim=(-0.03, 1.03),
        xlim=(0, max(1, max_iters)),
    )
    _style(ax_scores, probability=True)
    _unique_legend(ax_scores, loc="best")
    interactive = matplotlib.get_backend().lower() != "agg"
    if interactive:
        fig.show()
    ellipses = []
    iterations, scores = [], []

    def render(iteration, controls, hard_interval):
        plan = evaluate_controls(controls)
        mean = _np(plan.rollout.aux["mean_trace"])[0]
        covariance = _np(plan.rollout.aux["cov_trace"])[0]
        path.set_data(mean[:, 0], mean[:, 1])
        start.set_offsets(mean[:1, :2])
        terminal.set_offsets(mean[-1:, :2])
        while ellipses:
            ellipses.pop().remove()
        for index in _ellipse_indices(mean, ellipse_every):
            ellipses.append(_ellipse(ax_map, mean[index], covariance[index]))
        ax_map.set_title(
            f"{title} | iteration {iteration + 1}/{max_iters} | "
            f"{RHO_LOWER} = {hard_interval[0]:.3f}",
            fontweight="bold",
        )
        _refresh(fig, interactive)

    def on_iteration(iteration, record):
        interval = _record_interval(record)
        iterations.append(iteration + 1)
        scores.append(interval[0])
        score_line.set_data(iterations, scores)
        render(iteration, record.controls, interval)
        print(
            f"{title} iteration {iteration + 1}/{max_iters}: "
            f"loss {record.loss:.4f}, "
            f"rho_lower {interval[0]:.4f}",
            flush=True,
        )

    def finish(result):
        history = result.hard_lower_history
        score_line.set_data(np.arange(len(history)), history)
        render(
            result.selected_iteration, result.controls, result.hard_interval
        )

    return fig, (ax_map, ax_scores), on_iteration, finish


def _beliefs(state):
    """(mean, covariance, colour) of the ego and every traffic vehicle."""
    ego = [(state[0], state[1], COLORS["mean"])]
    traffic = [
        (mean, covariance, COLORS["traffic"])
        for mean, covariance in zip(state[3], state[4])
    ]
    return ego + traffic


def _draw_belief_outlines(ax, state):
    outlines = []
    for mean, covariance, color in _beliefs(state):
        width, height, angle = _ellipse_parameters(_np(covariance))
        outline = patches.Ellipse(
            _np(mean)[:2],
            width,
            height,
            angle=angle,
            fill=False,
            edgecolor=color,
            alpha=0.7,
            linewidth=1.2,
        )
        ax.add_patch(outline)
        outlines.append(outline)
    return outlines


def _update_belief_outlines(outlines, state):
    for outline, (mean, covariance, _) in zip(outlines, _beliefs(state)):
        width, height, angle = _ellipse_parameters(_np(covariance))
        outline.center = _np(mean)[:2]
        outline.width = width
        outline.height = height
        outline.angle = angle


def _candidate_trace(dynamics, state, controls):
    point = _np(state[0]).copy()
    candidate = [point[:2].copy()]
    for control in _np(controls):
        point = _np(dynamics.A) @ point + _np(dynamics.B) @ control
        candidate.append(point[:2].copy())
    return np.asarray(candidate)


def _mark_lane_window(ax, env, dt):
    task = env.metadata["task"]
    start, end = task["start_end_steps"]
    ax.axvspan(
        start * dt,
        (end - task["dwell_steps"]) * dt,
        color=COLORS["goal"],
        alpha=0.12,
        label="Dwell-start window",
    )
    ax.axvline(
        end * dt,
        color=COLORS["goal"],
        linestyle="--",
        linewidth=1,
        label="Completion deadline",
    )
    return end * dt


def create_live_view(
    env, initial_state, *, dt, dynamics=None, max_iters=40, label="MPC"
):
    """One road-centred figure for candidate plans and executed steps."""
    fig = plt.figure(figsize=(14, 6), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(2.8, 1.0))
    ax_map = fig.add_subplot(grid[:, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    ax_score = fig.add_subplot(grid[1, 1])
    ax_windows = fig.add_subplot(grid[2, 1])
    moving = _draw_environment(ax_map, env, lane=True)
    initial = _np(_ego_truth(initial_state))[:2]
    _move_obstacle(moving, env, 0, initial_state)
    ego_patch = _draw_ego_vehicle(ax_map, env, initial)
    executed = [initial]
    window_scores = []
    (line_exec,) = ax_map.plot(
        [initial[0]],
        [initial[1]],
        color=COLORS["executed"],
        linewidth=2,
        marker="o",
        markersize=3,
        label="Executed",
    )
    (line_plan,) = ax_map.plot(
        [], [], color=COLORS["planned"], linewidth=1.5, label="Plan"
    )
    (line_loss,) = ax_loss.plot(
        [],
        [],
        color=COLORS["planned"],
        marker="o",
        markersize=3,
        label="Optimization loss",
    )
    (line_smooth,) = ax_score.plot(
        [],
        [],
        color=COLORS["score"],
        marker="o",
        markersize=3,
        label=f"smooth {RHO_LOWER}",
    )
    (line_hard,) = ax_score.plot(
        [],
        [],
        color=COLORS["goal"],
        marker="o",
        markersize=3,
        label=RHO_LOWER,
    )
    (line_windows,) = ax_windows.plot(
        [],
        [],
        color=COLORS["score"],
        linewidth=1.5,
        marker="o",
        markersize=3,
        label=f"{RHO_LOWER} per window",
    )
    ax_loss.set(xlabel="iteration", ylabel="loss", xlim=(0, max_iters))
    ax_score.set(xlabel="iteration", ylabel=RHO_LOWER, xlim=(0, max_iters))
    ax_windows.set(xlabel="executed time [s]", ylabel=RHO_LOWER)
    final_time = _mark_lane_window(ax_windows, env, dt)
    ax_windows.set_xlim(0, final_time)
    for ax in (ax_loss, ax_score, ax_windows):
        _style(ax, probability=ax is ax_windows)
        _unique_legend(ax, loc="best")
    _unique_legend(
        ax_map, loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=5
    )
    ax_map.set_xlim(initial[0] - 15, initial[0] + 45)
    belief_outlines = _draw_belief_outlines(ax_map, initial_state)
    interactive = matplotlib.get_backend().lower() != "agg"
    if interactive:
        fig.show()
    current_state = initial_state
    iterations, losses, smooth, hard = [], [], [], []
    current_window = None

    def on_iteration(window, iteration, record):
        nonlocal current_window
        if current_window != window:
            current_window = window
            iterations.clear()
            losses.clear()
            smooth.clear()
            hard.clear()
            ax_map.set_title(
                f"Planning window {window + 1} at t={window * dt:.1f} s"
            )
        iterations.append(iteration + 1)
        losses.append(record.loss)
        smooth.append(record.smooth_lower)
        hard.append(_record_interval(record)[0])
        line_loss.set_data(iterations, losses)
        line_smooth.set_data(iterations, smooth)
        line_hard.set_data(iterations, hard)
        for ax in (ax_loss, ax_score):
            ax.relim()
            ax.autoscale_view(scalex=False)
        if dynamics is not None:
            candidate = _candidate_trace(
                dynamics, current_state, record.controls
            )
            line_plan.set_data(candidate[:, 0], candidate[:, 1])
        print(
            f"{label} window {window + 1}, iteration {iteration + 1}/"
            f"{max_iters}: loss {record.loss:.4f}, "
            f"rho_lower {_record_interval(record)[0]:.4f}",
            flush=True,
        )
        _refresh(fig, interactive)

    def finish_window(window, plan):
        print(
            f"{label} window {window + 1} complete after "
            f"{len(plan.loss_history)} iterations: {RHO_INTERVAL} = "
            f"[{plan.hard_interval[0]:.4f}, {plan.hard_interval[1]:.4f}]",
            flush=True,
        )

    def on_step(step, state, plan):
        nonlocal current_state
        current_state = state
        executed.append(_np(_ego_truth(state))[:2])
        points = np.asarray(executed)
        line_exec.set_data(points[:, 0], points[:, 1])
        trace = _np(plan.rollout.aux["mean_trace"])[0]
        line_plan.set_data(trace[:, 0], trace[:, 1])
        window_scores.append(plan.hard_interval[0])
        line_windows.set_data(
            dt * np.arange(1, len(window_scores) + 1), window_scores
        )
        ax_windows.set_xlim(0, max(final_time, dt * (len(window_scores) + 1)))
        _move_obstacle(moving, env, step + 1, state)
        _move_ego_vehicle(ego_patch, points[-1])
        ax_map.set_xlim(points[-1, 0] - 15, points[-1, 0] + 45)
        _update_belief_outlines(belief_outlines, state)
        ax_map.set_title(
            f"Executed step {step + 1} at t={(step + 1) * dt:.1f} s"
        )
        _refresh(fig, interactive)

    on_step.on_iteration = on_iteration
    on_step.finish_window = finish_window
    return fig, (ax_map, ax_loss, ax_score, ax_windows), on_step
