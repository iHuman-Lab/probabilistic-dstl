"""Publication figures from completed pdSTL planning results."""

from math import log, sqrt
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from matplotlib.transforms import Affine2D

from planning.environment import lane_lead_gap

COLORS = {
    "mean": "tab:blue",
    "planned": "tab:orange",
    "executed": "tab:blue",
    "goal": "tab:green",
    "obstacle": "tab:red",
    "workspace": "tab:gray",
    "score": "tab:purple",
    "ellipse": "tab:blue",
    "traffic": "tab:gray",
    "target": "goldenrod",
    # Yield zone: the light-fill / dark-edge purple of the 2D visit region.
    "yield_fill": "#c5b0d5",
    "yield_edge": "#9467bd",
    # Non-drivable ramp end: the light-fill / dark-edge red of 2D obstacles.
    "blocked_fill": "#ff9896",
    "blocked_edge": "#d62728",
    # Target lane: the light green of the 2D goal region.
    "lane_fill": "#98df8a",
}


def _chi2_radius(confidence):
    """Mahalanobis radius of a 2-D Gaussian's `confidence` region."""
    return sqrt(-2 * log(1 - confidence))


CHI95_RADIUS = _chi2_radius(0.95)

# pdSTL lower and upper stochastic robustness of the spec φ. They are a
# robustness interval, not bounds on the probability that φ holds.
# mathtext has no \underline; \underset draws the lower-bound bar.
RHO_LOWER = r"$\underset{\,\_}{\rho}_\varphi$"
RHO_UPPER = r"$\overline{\rho}_\varphi$"
RHO_INTERVAL = rf"[{RHO_LOWER}, {RHO_UPPER}]"


class VehiclePatch(patches.Polygon):
    """Top-down car silhouette within its configured collision footprint."""

    def __init__(self, x, y, width, height, **kwargs):
        self._x, self._y = x, y
        self._width, self._height = width, height
        super().__init__(self._vertices(x, y), closed=True, **kwargs)

    def _vertices(self, x, y):
        w, h = self._width, self._height
        return [
            (x, y + 0.20 * h),
            (x + 0.10 * w, y + 0.05 * h),
            (x + 0.72 * w, y + 0.05 * h),
            (x + w, y + 0.28 * h),
            (x + w, y + 0.72 * h),
            (x + 0.72 * w, y + 0.95 * h),
            (x + 0.10 * w, y + 0.95 * h),
            (x, y + 0.80 * h),
        ]

    def set_xy(self, xy):
        coordinates = np.asarray(xy)
        if coordinates.ndim == 2:
            return super().set_xy(coordinates)
        self._x, self._y = coordinates
        return super().set_xy(self._vertices(self._x, self._y))

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height


def _np(value):
    return (
        value.detach().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )


def _unique_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(
            unique.values(),
            unique.keys(),
            fontsize=9,
            framealpha=0.9,
            **kwargs,
        )


def _finish(fig, save_path, show):
    """Let the user inspect the figure, then save both publication formats."""
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show(block=True)
    if save_path is not None:
        stem = Path(save_path).with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return fig


def _style(ax, *, probability=False):
    ax.grid(color=COLORS["workspace"], alpha=0.16, linewidth=0.6)
    ax.tick_params(labelsize=9)
    if probability:
        ax.set_ylim(-0.03, 1.03)


def _ellipse_parameters(covariance, confidence=0.95):
    """Width, height, and angle of a joint two-dimensional belief ellipse."""
    covariance = np.asarray(covariance)[:2, :2]
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * _chi2_radius(confidence) * np.sqrt(values)
    return width, height, angle


def _ellipse(ax, mean, covariance, *, label=None, confidence=0.95):
    width, height, angle = _ellipse_parameters(covariance, confidence)
    ellipse = patches.Ellipse(
        mean[:2],
        width,
        height,
        angle=angle,
        facecolor=COLORS["ellipse"],
        edgecolor=COLORS["mean"],
        alpha=0.27,
        linewidth=0.8,
        label=label,
    )
    ax.add_patch(ellipse)
    return ellipse


def _ellipse_indices(mean, every, minimum_distance=0.35):
    """Temporal ellipse cadence with spatial thinning near stationary states."""
    candidates = sorted(
        set(range(0, len(mean), max(1, every))) | {len(mean) - 1}
    )
    selected = []
    for index in candidates:
        if (
            not selected
            or np.linalg.norm(mean[index, :2] - mean[selected[-1], :2])
            >= minimum_distance
        ):
            selected.append(index)
        elif index == len(mean) - 1:
            selected[-1] = index
    return selected


def _draw_task_region(ax, region):
    """A goal or target box, tagged with its name."""
    color = COLORS[region.role]
    ax.add_patch(
        patches.Rectangle(
            (region.xmin, region.ymin),
            region.xmax - region.xmin,
            region.ymax - region.ymin,
            facecolor=color,
            edgecolor=color,
            alpha=0.35,
            label=region.role.capitalize(),
        )
    )
    ax.text(
        (region.xmin + region.xmax) / 2,
        (region.ymin + region.ymax) / 2,
        region.name,
        color=color,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=9,
    )


def _draw_environment(ax, env, *, lane=False):
    """Draw configured geometry; return patches for moving traffic, if any."""
    moving = None
    if lane:
        road = env.metadata["road"]
        ramp = env.metadata.get("ramp")
        if ramp is None:
            ax.axhspan(
                road["y_min"],
                road["y_max"],
                color=COLORS["workspace"],
                alpha=0.08,
            )
            for y in (road["y_min"], road["y_max"]):
                ax.axhline(y, color=COLORS["workspace"], linewidth=1.3)
            ax.axhline(
                road["lane_divider"],
                color=COLORS["workspace"],
                linestyle="--",
                linewidth=1.1,
                label="Lane divider",
            )
        else:
            extent = env.single_region("goal").x
            start, end = ramp["start_x"], ramp["end_x"]
            x = np.asarray([extent[0], start, end, extent[1]])
            lower = np.asarray(
                [
                    road["y_min"],
                    road["y_min"],
                    road["lane_divider"],
                    road["lane_divider"],
                ]
            )
            ax.fill_between(
                [start, end],
                [road["y_min"], road["lane_divider"]],
                road["y_min"],
                color=COLORS["obstacle"],
                alpha=0.38,
                label="Non-drivable",
            )
            ax.plot(x, lower, color=COLORS["workspace"], linewidth=1.3)
            ax.axhline(road["y_max"], color=COLORS["workspace"], linewidth=1.3)
            gore = ramp.get("gore_x")
            if gore is not None:
                ax.plot(
                    [extent[0], gore],
                    [road["lane_divider"], road["lane_divider"]],
                    color=COLORS["workspace"],
                    linewidth=1.3,
                    label="Gore (no merging)",
                )
            ax.plot(
                [extent[0] if gore is None else gore, start],
                [road["lane_divider"], road["lane_divider"]],
                color=COLORS["workspace"],
                linestyle="--",
                linewidth=1.1,
                label="Ramp merge line",
            )
            ax.scatter(
                [end],
                [road["lane_divider"]],
                marker="x",
                color=COLORS["workspace"],
                label="Ramp end",
            )
        goal = env.single_region("goal")
        ax.add_patch(
            patches.Rectangle(
                (goal.xmin, goal.ymin),
                goal.xmax - goal.xmin,
                goal.ymax - goal.ymin,
                facecolor=COLORS["goal"],
                edgecolor="none",
                alpha=0.10,
                label="Goal lane",
            )
        )
        if "yield" in env.regions:
            region = env.region("yield")
            ax.add_patch(
                patches.Rectangle(
                    (region.xmin, region.ymin),
                    region.xmax - region.xmin,
                    region.ymax - region.ymin,
                    facecolor=COLORS["yield_fill"],
                    edgecolor=COLORS["yield_edge"],
                    alpha=0.4,
                    label="Yield region",
                )
            )
        moving = {}
        for index, vehicle in enumerate(env.metadata["traffic"]):
            name = vehicle["name"]
            patch = VehiclePatch(
                vehicle["x0"] - vehicle["width"] / 2,
                vehicle["y"] - vehicle["height"] / 2,
                vehicle["width"],
                vehicle["height"],
                facecolor=COLORS["traffic"],
                edgecolor=COLORS["traffic"],
                alpha=0.65,
                label="Traffic" if index == 0 else "_nolegend_",
            )
            ax.add_patch(patch)
            moving[name] = patch
    for region in env.regions.values():
        if region.role == "workspace" and not lane:
            ax.add_patch(
                patches.Rectangle(
                    (region.xmin, region.ymin),
                    region.xmax - region.xmin,
                    region.ymax - region.ymin,
                    fill=False,
                    edgecolor=COLORS["workspace"],
                    linewidth=1.3,
                )
            )
            ax.set_xlim(region.xmin, region.xmax)
            ax.set_ylim(region.ymin, region.ymax)
        elif region.role in ("goal", "target") and not lane:
            _draw_task_region(ax, region)
        elif region.role == "obstacle":
            ax.add_patch(
                patches.Rectangle(
                    (region.xmin, region.ymin),
                    region.xmax - region.xmin,
                    region.ymax - region.ymin,
                    facecolor=COLORS["obstacle"],
                    edgecolor=COLORS["obstacle"],
                    alpha=0.45,
                    hatch="//",
                    label="Obstacle",
                )
            )
    ax.set_xlabel("x [m]" if not lane else "x position [m]")
    ax.set_ylabel("y [m]" if not lane else "y position [m]")
    ax.set_aspect(1.4 if lane else "equal", adjustable="box")
    _style(ax)
    return moving


def _draw_ego_vehicle(ax, env, position):
    """Show the ego footprint at its current sampled position."""
    size = env.metadata["ego_vehicle"]
    patch = VehiclePatch(
        position[0] - size["width"] / 2,
        position[1] - size["height"] / 2,
        size["width"],
        size["height"],
        facecolor=COLORS["executed"],
        edgecolor=COLORS["executed"],
        alpha=0.7,
        label="Ego",
        zorder=6,
    )
    ax.add_patch(patch)
    return patch


def _move_ego_vehicle(patch, position):
    patch.set_xy(
        (
            position[0] - patch.get_width() / 2,
            position[1] - patch.get_height() / 2,
        )
    )


def _ego_truth(state):
    """Physical ego state; older saved tuples used their mean as truth."""
    return getattr(state, "ego_true", state[0])


def _traffic_truth(state):
    return getattr(state, "traffic_true", state[3])


def _move_obstacle(patches_by_name, env, step, state=None):
    """Move each lane traffic patch to its state (or constant-speed) position."""
    if patches_by_name is None:
        return
    for index, vehicle in enumerate(env.metadata["traffic"]):
        if state is not None and len(state) > 3:
            center = _np(_traffic_truth(state)[index])[:2]
        else:
            center = (
                vehicle["x0"] + vehicle["speed"] * env.metadata["dt"] * step,
                vehicle["y"],
            )
        patches_by_name[vehicle["name"]].set_xy(
            (
                center[0] - vehicle["width"] / 2,
                center[1] - vehicle["height"] / 2,
            )
        )


def _selected_windows(count, maximum=5):
    if count <= maximum:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, maximum, dtype=int)))


def plot_reach_avoid(
    result,
    env,
    *,
    dt=1.0,
    u_max=1.0,
    control_unit="m/s",
    title="Reach–Avoid",
    ellipse_every=4,
    ellipse_confidence=0.95,
    save_path=None,
    show=False,
):
    """Map, controls, and lower-robustness history of a reach-avoid plan.

    Args:
        result: Selected PlanResult.
        env: Its Environment.

    Returns:
        (figure, (map, controls, scores) axes).
    """
    fig = plt.figure(figsize=(13, 7), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.55, 1.0))
    ax = fig.add_subplot(grid[:, 0])
    ax_controls = fig.add_subplot(grid[0, 1])
    ax_scores = fig.add_subplot(grid[1, 1])
    _draw_environment(ax, env)
    mean = _np(result.rollout.aux["mean_trace"])[0]
    covariance = _np(result.rollout.aux["cov_trace"])[0]
    ax.plot(
        mean[:, 0],
        mean[:, 1],
        color=COLORS["mean"],
        linewidth=2.2,
        label="Plan",
        zorder=8,
    )
    ax.scatter(*mean[0, :2], marker="o", s=55, color="black", zorder=10)
    ax.scatter(
        *mean[-1, :2], marker="s", s=55, color=COLORS["mean"], zorder=10
    )
    indices = _ellipse_indices(mean, ellipse_every)
    for index in indices:
        _ellipse(
            ax,
            mean[index],
            covariance[index],
            label=f"{100 * ellipse_confidence:g}% ellipse"
            if index == indices[0]
            else None,
            confidence=ellipse_confidence,
        )
    ax.set_title(title, fontweight="bold")
    _unique_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)

    control_time = dt * np.arange(len(result.controls))
    controls = _np(result.controls)
    for dimension, label in enumerate(("u_x", "u_y")):
        ax_controls.step(
            control_time,
            controls[:, dimension],
            where="post",
            linewidth=1.6,
            label=label,
        )
    ax_controls.set(
        ylabel=f"control [{control_unit}]",
        ylim=(-1.1 * u_max, 1.1 * u_max),
        title="Selected controls",
    )
    _style(ax_controls)
    _unique_legend(ax_controls, loc="best")

    ax_scores.plot(
        np.arange(len(result.hard_lower_history)),
        result.hard_lower_history,
        color=COLORS["mean"],
        linewidth=1.7,
        label=f"{RHO_LOWER} (lower bound)",
    )
    if result.alpha is not None:
        ax_scores.axhline(
            result.alpha,
            color=COLORS["obstacle"],
            linestyle="--",
            linewidth=1.1,
            label=f"threshold α = {result.alpha:.2f}",
        )
    selected = result.selected_iteration + 1
    ax_scores.scatter(
        [selected],
        [result.hard_interval[0]],
        color=COLORS["mean"],
        marker="s",
        zorder=5,
        label="Selected plan",
    )
    if result.monte_carlo is not None:
        rate, lower, upper = result.monte_carlo
        ax_scores.errorbar(
            [selected],
            [rate],
            yerr=[[rate - lower], [upper - rate]],
            color=COLORS["score"],
            marker="D",
            markersize=5,
            capsize=3,
            linestyle="none",
            zorder=6,
            label=r"Monte Carlo $P(\varphi)$",
        )
    change = f"{result.hard_interval[0]:.3f}"
    if result.initial_hard_lower is not None:
        change = f"{result.initial_hard_lower:.3f} → {change}"
    ax_scores.set(
        xlabel="iteration (0 = initial guess)",
        ylabel=f"lower robustness {RHO_LOWER}",
        ylim=(-0.03, 1.03),
        title=f"{RHO_LOWER}: {change} · {result.alpha_status()}",
    )
    _style(ax_scores, probability=True)
    _unique_legend(ax_scores, loc="best")
    _finish(fig, save_path, show)
    return fig, (ax, ax_controls, ax_scores)


def _mark_lane_task_window(ax, env, dt):
    task = env.metadata["task"]
    windows = [(task, "Dwell-start window", "Completion deadline", "goal")]
    if task["spec"] == "merge_or_yield":
        windows.append(
            (
                task["yield"],
                "Yield dwell-start window",
                "Yield completion deadline",
                "target",
            )
        )
    for window, label, deadline, color in windows:
        start, end = window["start_end_steps"]
        latest_start = end - window["dwell_steps"]
        ax.axvspan(
            start * dt,
            latest_start * dt,
            color=COLORS[color],
            alpha=0.10,
            label=label,
        )
        ax.axvline(
            end * dt,
            color=COLORS[color],
            linestyle="--",
            linewidth=1,
            label=deadline,
        )


def _plot_execution(result, env, *, dt, save_path, show):
    fig = plt.figure(figsize=(14, 6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.55, 1))
    ax_map = fig.add_subplot(grid[0, :])
    ax_score = fig.add_subplot(grid[1, 0])
    ax_control = fig.add_subplot(grid[1, 1])
    moving = _draw_environment(ax_map, env, lane=True)
    means = np.asarray([_np(_ego_truth(state))[:2] for state in result.states])
    ax_map.plot(
        means[:, 0],
        means[:, 1],
        color=COLORS["executed"],
        linewidth=2,
        label="Executed",
        zorder=5,
    )
    ax_map.scatter(
        [means[0, 0]],
        [means[0, 1]],
        color=COLORS["executed"],
        s=26,
        label="Start",
        zorder=6,
    )
    for index in _selected_windows(len(result.window_plans)):
        plan = _np(result.window_plans[index].rollout.aux["mean_trace"])[0]
        ax_map.plot(
            plan[:, 0],
            plan[:, 1],
            color=COLORS["planned"],
            linewidth=1.1,
            alpha=0.6,
            label="Plan",
        )
    _draw_ego_vehicle(ax_map, env, means[-1])
    for index, _vehicle in enumerate(env.metadata["traffic"]):
        positions = np.asarray(
            [_np(_traffic_truth(state)[index])[:2] for state in result.states]
        )
        ax_map.plot(
            positions[:, 0],
            positions[:, 1],
            color=COLORS["traffic"],
            linestyle=":",
            linewidth=1.2,
            label="Traffic path" if index == 0 else "_nolegend_",
        )
    _move_obstacle(moving, env, len(result.states) - 1, result.states[-1])
    # Frame the ego path and every car's final position.
    traffic = _np(_traffic_truth(result.states[-1]))[:, 0]
    right = max(means[:, 0].max() + 18, traffic.max() + 4)
    if "yield" in env.regions:
        right = max(right, env.region("yield").xmax + 2)
    ax_map.set_xlim(min(means[:, 0].min(), traffic.min()) - 4, right)
    elapsed = dt * np.arange(len(means))
    labeled = None
    for second in range(int(np.floor(elapsed[-1])) + 1):
        index = int(np.argmin(np.abs(elapsed - second)))
        # Skip labels that would overlap while the ego is (nearly) stopped.
        if labeled is None or abs(means[index, 0] - labeled) > 3.0:
            ax_map.annotate(
                f"t={second} s",
                means[index],
                xytext=(4, 8),
                textcoords="offset points",
                fontsize=8,
            )
            labeled = means[index, 0]
    reason = result.stopped_reason.replace("_", " ")
    ax_map.text(
        0.98,
        0.95,
        f"Outcome: {reason}",
        transform=ax_map.transAxes,
        fontsize=9,
        ha="right",
        va="top",
    )
    _unique_legend(
        ax_map, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=6
    )
    intervals = np.asarray(
        [plan.hard_interval for plan in result.window_plans]
    )
    time = dt * np.arange(len(intervals))
    if len(intervals):
        ax_score.fill_between(
            time,
            intervals[:, 0],
            intervals[:, 1],
            color=COLORS["goal"],
            alpha=0.24,
            label=RHO_INTERVAL,
        )
        ax_score.plot(time, intervals[:, 0], color=COLORS["goal"])
        ax_score.plot(
            time, intervals[:, 1], color=COLORS["goal"], linestyle="--"
        )
        ax_score.plot(
            time,
            [plan.smooth_lower for plan in result.window_plans],
            color=COLORS["score"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"smooth {RHO_LOWER}",
        )
        ax_score.text(
            0.98,
            0.05,
            f"last window {RHO_INTERVAL} = "
            f"[{intervals[-1, 0]:.3f}, {intervals[-1, 1]:.3f}]",
            transform=ax_score.transAxes,
            ha="right",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    _mark_lane_task_window(ax_score, env, dt)
    ax_score.set(xlabel="executed time [s]", ylabel="robustness per window")
    _style(ax_score, probability=True)
    if len(intervals):
        # The smooth surrogate can be negative; keep it on the axis.
        smooth = min(plan.smooth_lower for plan in result.window_plans)
        ax_score.set_ylim(bottom=min(-0.03, smooth - 0.05))
    _unique_legend(
        ax_score, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3
    )
    controls = _np(result.applied_controls)
    for dimension in range(controls.shape[1]):
        ax_control.step(
            dt * np.arange(len(controls)),
            controls[:, dimension],
            where="post",
            linewidth=1.4,
            label=("Longitudinal" if dimension == 0 else "Lateral"),
        )
    ax_control.set(xlabel="time [s]", ylabel="applied control [m/s²]")
    _style(ax_control)
    _unique_legend(ax_control, loc="best")
    _finish(fig, save_path, show)
    return fig, (ax_map, ax_score, ax_control)


def plot_lane_merge(result, env, *, dt, save_path=None, show=False):
    """Show the combined figure, the snapshot figure and a results table."""
    from experiments.lane import summarize

    combined = _plot_execution(
        result, env, dt=dt, save_path=save_path, show=show
    )
    snapshots = plot_lane_snapshots(
        result,
        env,
        dt=dt,
        save_path=_sibling(save_path, "snapshots"),
        show=show,
    )
    rows = []
    for label, value in summarize(result, env, dt).items():
        if value is None:
            value = "N/A"
        elif isinstance(value, float):
            value = f"{value:.3f}"
        rows.append((label, value))
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.set_axis_off()
    ax.set_title(env.metadata["task"]["spec"].replace("_", " ").capitalize())
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colWidths=[0.7, 0.3],
        bbox=[0, 0, 1, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    _finish(fig, _sibling(save_path, "table"), show)
    return {"combined": combined, "snapshots": snapshots, "table": (fig, ax)}


def _sibling(save_path, kind):
    """`outputs/x.png` -> `outputs/x_<kind>.png` (None stays None)."""
    if save_path is None:
        return None
    stem = Path(save_path).with_suffix("")
    return stem.with_name(f"{stem.name}_{kind}.png")


def plot_lane_snapshots(
    result,
    env,
    *,
    dt,
    number=5,
    view=(-60.0, 60.0),
    save_path=None,
    show=False,
):
    """Top-down panels of the whole scene at evenly spaced moments.

    Every traffic car is drawn at its true position with its name, together
    with the ego, its path so far and the plan made at that moment.

    Args:
        result: MPCResult from a lane trial.
        env: The trial's lane environment.
        dt: Step length in seconds.
        number: How many moments to show (always including first and last).
        view: Longitudinal window around the ego, (behind, ahead) in metres.
        save_path, show: PNG/PDF output path and interactive display.

    Returns:
        (figure, axes).
    """
    count = len(result.states)
    steps = sorted({int(k) for k in np.linspace(0, count - 1, number)})
    path = np.asarray([_np(_ego_truth(state))[:2] for state in result.states])
    road = env.metadata["road"]
    headway = "headway" in env.metadata["task"]
    fig, axes = plt.subplots(
        len(steps),
        1,
        figsize=(11, 1.9 * len(steps) + 0.6),
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    for ax, step in zip(axes, steps):
        state = result.states[step]
        moving = _draw_environment(ax, env, lane=True)
        _move_obstacle(moving, env, step, state)
        cars = _np(_traffic_truth(state))
        for car, position in zip(env.metadata["traffic"], cars):
            ax.annotate(
                car["name"],
                (position[0], position[1] + car["height"] / 2),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLORS["traffic"],
                annotation_clip=True,
            )
        ax.plot(
            *path[: step + 1].T,
            color=COLORS["executed"],
            linewidth=1.8,
            label="Executed",
        )
        title = f"t = {step * dt:.1f} s"
        if step < len(result.window_plans):
            plan = result.window_plans[step]
            mean = _np(plan.rollout.aux["mean_trace"])[0]
            ax.plot(
                mean[:, 0],
                mean[:, 1],
                color=COLORS["planned"],
                linestyle="--",
                linewidth=1.3,
                label="Current plan",
            )
            title += f",  plan {RHO_LOWER} = {plan.hard_interval[0]:.2f}"
        if headway:
            gap = lane_lead_gap(env, state.ego_true, state.traffic_true)
            lead = env.metadata["task"]["headway"]["lead_vehicle"]
            title += f",  gap to {lead} = {gap:.1f} m"
        _draw_ego_vehicle(ax, env, path[step])
        ax.set_xlim(path[step, 0] + view[0], path[step, 0] + view[1])
        ax.set_ylim(road["y_min"] - 0.6, road["y_max"] + 1.4)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("y (m)")
        ax.set_xlabel("")
    axes[-1].set_xlabel("x (m)")
    _unique_legend(
        axes[0], loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=6
    )
    _finish(fig, save_path, show)
    return fig, axes


# --- Paper figure ---------------------------------------------------------------

PAPER_WIDTH = 7.16  # in, IEEE two-column text width
COLUMN_WIDTH = 3.5  # in, IEEE column width
PAPER_FONT = 7  # pt


def _draw_road(ax, env):
    """Shaded lanes (target lane green), edges, divider, ramp taper."""
    road = env.metadata["road"]
    ramp = env.metadata.get("ramp")
    edge = {"color": "0.2", "linewidth": 1.1}
    ax.axhspan(
        road["lane_divider"],
        road["y_max"],
        color=COLORS["lane_fill"],
        alpha=0.5,
        lw=0,
        label="Target lane",
    )
    if ramp is None:
        ax.axhspan(
            road["y_min"], road["lane_divider"], color="0.5", alpha=0.06, lw=0
        )
    else:
        ax.fill_between(
            [-1e3, ramp["start_x"], ramp["end_x"]],
            [road["y_min"], road["y_min"], road["lane_divider"]],
            road["lane_divider"],
            color="0.5",
            alpha=0.06,
            lw=0,
        )
    divider = {"color": "0.3", "linewidth": 0.9, "linestyle": "--"}
    ax.axhline(road["y_max"], **edge)
    if ramp is None:
        ax.axhline(road["y_min"], **edge)
        ax.axhline(road["lane_divider"], **divider)
    else:
        start, end = ramp["start_x"], ramp["end_x"]
        middle = road["lane_divider"]
        ax.plot(
            [-1e3, start, end, 1e3], [road["y_min"]] * 2 + [middle] * 2, **edge
        )
        gore = ramp.get("gore_x")
        if gore is None:
            ax.plot([-1e3, start], [middle] * 2, **divider)
        else:
            ax.plot([-1e3, gore], [middle] * 2, **edge)  # separate roads
            ax.plot([gore, start], [middle] * 2, **divider)
        ax.fill_between(
            [start, end],
            [road["y_min"], middle],
            road["y_min"],
            facecolor=COLORS["blocked_fill"],
            edgecolor=COLORS["blocked_edge"],
            hatch="//",
            alpha=0.9,
            lw=0.8,
            label="Ramp end",
        )
    if "yield" in env.regions:
        zone = env.region("yield")
        ax.add_patch(
            patches.Rectangle(
                (zone.xmin, zone.ymin),
                zone.xmax - zone.xmin,
                zone.ymax - zone.ymin,
                facecolor=COLORS["yield_fill"],
                edgecolor=COLORS["yield_edge"],
                alpha=0.6,
                lw=0.8,
                label="Yield zone",
            )
        )


def _draw_car(
    ax, center, size, color, *, heading=0.0, edge=None, label=None, alpha=0.9
):
    """A filled car footprint turned to `heading` (rad), outlined in a
    darker shade of its colour; `edge` outlines a car the text refers to."""
    outline = 0.6 * np.asarray(matplotlib.colors.to_rgb(color))
    car = ax.add_patch(
        VehiclePatch(
            center[0] - size["width"] / 2,
            center[1] - size["height"] / 2,
            size["width"],
            size["height"],
            facecolor=color,
            edgecolor=edge or tuple(outline),
            linewidth=0.9 if edge else 0.6,
            alpha=alpha,
            label=label,
            zorder=4,
        )
    )
    car.set_transform(
        Affine2D().rotate_around(center[0], center[1], heading) + ax.transData
    )


def _frame_strip(
    ax, title, result, env, step, xlim, ylim, show_uncertainty=False
):
    """One simulation frame: what the planner saw and did at `step`.

    Every car at its true position, the ego path driven so far, and the
    plan made at this step (dashed), optionally with the 95% position
    region of its predicted belief. The ego is turned to its heading.
    """
    _draw_road(ax, env)
    path = np.asarray([_np(_ego_truth(state))[:2] for state in result.states])
    ax.plot(
        *path[: step + 1].T,
        color=COLORS["executed"],
        lw=1.1,
        zorder=3,
        label="Driven path",
    )
    if step < len(result.window_plans):
        aux = result.window_plans[step].rollout.aux
        mean = _np(aux["mean_trace"])[0]
        covariance = _np(aux["cov_trace"])[0]
        # The predicted 95% position region along the plan, as one band.
        for k in range(1, len(mean), 2) if show_uncertainty else ():
            width, height, angle = _ellipse_parameters(covariance[k][:2, :2])
            ax.add_patch(
                patches.Ellipse(
                    mean[k, :2],
                    width,
                    height,
                    angle=angle,
                    facecolor=COLORS["ellipse"],
                    edgecolor="none",
                    alpha=0.18,
                    zorder=2,
                    label="Predicted 95% region",
                )
            )
        ax.plot(
            mean[:, 0],
            mean[:, 1],
            color=COLORS["planned"],
            lw=1.0,
            ls="--",
            zorder=3,
            label="Current plan",
        )
    cars = _np(_traffic_truth(result.states[step]))
    for car, center in zip(env.metadata["traffic"], cars):
        # Where the car's front will be in 0.5 s at its current speed.
        front = center[0] + car["width"] / 2
        ax.annotate(
            "",
            xy=(front + center[2] * 0.5, center[1]),
            xytext=(front, center[1]),
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 0.6,
                "color": "0.45",
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 5,
            },
            annotation_clip=True,
            zorder=3,
        )
        _draw_car(ax, center[:2], car, COLORS["traffic"], label="Traffic")
    ax.plot(
        [],
        [],
        color="0.45",
        lw=0.6,
        marker=">",
        ms=3,
        label="0.5 s motion",
    )
    velocity = _np(_ego_truth(result.states[step]))[2:4]
    _draw_car(
        ax,
        path[step],
        env.metadata["ego_vehicle"],
        COLORS["executed"],
        heading=float(np.arctan2(velocity[1], max(velocity[0], 0.5))),
        label="Ego",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.6")
    ax.tick_params(labelsize=PAPER_FONT, length=2, pad=1, color="0.6")
    ax.set_title(title, loc="left", fontsize=PAPER_FONT, pad=1.5)


SCORE_STYLES = (("tab:blue", "-"), ("tab:orange", "--"), ("tab:gray", ":"))


def _time_label(ax, center, size, time):
    """A time tag such as "1.6 s" just above a drawn ego car.

    Args:
        ax: Axes of the frame.
        center: Car centre (x, y).
        size: Car size dict with "height".
        time: Time in seconds.
    """
    ax.text(
        center[0],
        center[1] + size["height"] / 2 + 0.25,
        f"{time:g} s",
        fontsize=PAPER_FONT - 1,
        color=COLORS["executed"],
        ha="center",
        va="bottom",
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.12",
            "fc": "white",
            "ec": "none",
            "alpha": 0.85,
        },
    )


def _ego_ghosts(ax, result, env, step, times, dt):
    """Faded ego cars at earlier times, each tagged with its time.

    Args:
        ax: Axes of one frame.
        result: MPCResult of the run.
        env: Lane environment.
        step: Step of the frame; only earlier times are drawn.
        times: Times in seconds to draw.
        dt: Step length in seconds.
    """
    drawn = []
    for time in times:
        k = round(time / dt)
        state = _np(_ego_truth(result.states[k])) if k < step else None
        # Skip a ghost that would sit on top of one already drawn.
        if state is None or any(abs(state[0] - x) < 5.5 for x in drawn):
            continue
        drawn.append(state[0])
        _draw_car(
            ax,
            state[:2],
            env.metadata["ego_vehicle"],
            COLORS["executed"],
            heading=float(np.arctan2(state[3], max(state[2], 0.5))),
            alpha=0.4,
        )
        _time_label(ax, state[:2], env.metadata["ego_vehicle"], k * dt)
    now = _np(_ego_truth(result.states[step]))
    _time_label(ax, now[:2], env.metadata["ego_vehicle"], step * dt)


def _score_panel(ax, panels, dt, alpha, events, title):
    """Each run's per-window lower score over time, with alpha and events.

    Args:
        ax: Axes for the panel.
        panels: (title, result, env, step, ghosts, label) per run.
        dt: Step length in seconds.
        alpha: Threshold line, or None.
        events: (time, text) pairs drawn as vertical lines.
        title: Panel title.
    """
    end = 0.0
    for index, (_t, result, _e, _s, _g, label) in enumerate(panels):
        color, style = SCORE_STYLES[index]
        scores = [plan.hard_interval[0] for plan in result.window_plans]
        times = np.arange(len(scores)) * dt
        end = max(end, len(scores) * dt)
        ax.step(
            times,
            scores,
            where="post",
            color=color,
            ls=style,
            lw=1.0,
            label=label,
        )
        final = result.states[-1]
        done = final.true_completed or final.true_yield_completed
        if done is not None:
            # The first run's marker is hollow so a marker on top shows.
            first = index == 0
            ax.plot(
                done * dt,
                scores[min(done, len(scores) - 1)],
                "o",
                ms=5 if first else 3,
                color=color,
                mfc="none" if first else color,
                zorder=5,
            )
    end += 0.2
    if alpha is not None:
        ax.axhline(alpha, color="0.3", lw=0.6, ls=(0, (1, 1)))
        ax.text(
            end,
            alpha - 0.04,
            r"$\alpha$",
            fontsize=PAPER_FONT,
            ha="right",
            va="top",
            color="0.3",
        )
    for time, text in events:
        ax.axvline(time, color="0.6", lw=0.6)
        ax.text(
            time + 0.08,
            0.6,
            text,
            fontsize=PAPER_FONT - 1,
            va="bottom",
            color="0.4",
        )
    ax.plot([], [], "o", color="0.3", ms=3, label="task done")
    ax.set_xlim(0, end)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time (s)", fontsize=PAPER_FONT, labelpad=0)
    ax.set_ylabel(
        r"$\hat{r}^{\varphi,\downarrow}_0$", fontsize=PAPER_FONT, labelpad=1
    )
    ax.tick_params(labelsize=PAPER_FONT, length=2, pad=1)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.legend(
        fontsize=PAPER_FONT - 1,
        frameon=False,
        loc="lower right",
        handlelength=1.6,
        borderaxespad=0.1,
        labelspacing=0.2,
    )
    ax.set_title(title, loc="left", fontsize=PAPER_FONT, pad=1.5)


def _paper_layout(column, road_height, span, short_span, stretch):
    """Panel rectangles for the two-column or the one-column figure.

    Every frame has the same metres per inch along x. Two-column: the wide
    frame on top, then two short frames and the score panel in one row.
    One-column: the wide frame, then the two short frames side by side,
    then the score panel across the full width.

    Args:
        column: One-column layout instead of two-column.
        road_height: Shown road height in metres.
        span: Metres across the wide frame.
        short_span: Metres in each short frame (two-column only; one-column
            short frames take half the width each).
        stretch: How much taller than true scale the road is drawn.

    Returns:
        (width, height, rectangles, short_span, legend_columns): sizes in
        inches; rectangles (left, bottom, width, height) for the wide frame,
        the two short frames and the score panel.
    """
    margin, gap = 0.04, 0.14
    title_h, ticks_h, label_h = 0.16, 0.13, 0.15
    fig_w = COLUMN_WIDTH if column else PAPER_WIDTH
    width = fig_w - 2 * margin
    per_inch = span / width  # metres per inch along x, every frame
    strip = road_height / per_inch * stretch
    axis_row = ticks_h + label_h  # x ticks and "x (m)" / "time (s)"
    if column:
        legend_h, score_h, ylabel_w = 0.3, 0.8, 0.3
        short_w = (width - gap) / 2
        short_span = short_w * per_inch
        score = (margin + ylabel_w, axis_row, width - ylabel_w, score_h)
        short_bottom = axis_row + score_h + title_h + 0.06 + axis_row
    else:
        legend_h = 0.17
        short_w = short_span / per_inch
        short_bottom = axis_row
        left = margin + 2 * (short_w + gap) + 0.22
        score = (left, axis_row, fig_w - left - margin, strip)
    shorts = [
        (margin + index * (short_w + gap), short_bottom, short_w, strip)
        for index in range(2)
    ]
    wide_bottom = short_bottom + strip + title_h + 0.06 + ticks_h
    height = wide_bottom + strip + title_h + legend_h + margin
    wide = (margin, wide_bottom, width, strip)
    legend_columns = 4 if column else 8
    return fig_w, height, [wide, *shorts, score], short_span, legend_columns


def plot_lane_paper(
    panels,
    *,
    dt,
    alpha=None,
    events=(),
    column=False,
    save_path=None,
    show=False,
    span=None,
    short_span=40.0,
    start=-5.0,
    stretch=None,
    show_uncertainty=False,
):
    """Paper figure: one wide frame, two short frames, score panel.

    Two-column (IEEE text width) or one-column (`column=True`) layout; see
    `_paper_layout`. Every frame shares one metres-per-inch scale along x;
    the road is drawn `stretch` times taller than true scale so the cars
    stay readable. Each frame shows faded ego cars at earlier times, tagged
    with the time. The last panel is each run's per-window lower score.
    Font is DejaVu Sans, embedded as TrueType.

    Args:
        panels: (title, MPCResult, environment, step, ghost times [s],
            score label) for the wide frame, then up to two short frames.
        dt: Step length in seconds.
        alpha: Score threshold drawn in the score panel, or None.
        events: (time, text) pairs marked in the score panel.
        column: One-column layout.
        save_path: Output path; a PDF and a 300 dpi PNG are written.
        show: Display the figure.
        span: Metres across the wide frame (default 130, one-column 105).
        short_span: Metres in each short frame (two-column only).
        start: Left edge of every frame in metres.
        stretch: How much taller than true scale the road is drawn
            (default 1.8, one-column 2.2).
        show_uncertainty: Also draw the plan's predicted 95% region.

    Returns:
        (figure, axes): the frame axes in panel order, then the score axes.
    """
    if not 1 <= len(panels) <= 3:
        raise ValueError("plot_lane_paper needs one wide and up to two short")
    if span is None:
        span = 105.0 if column else 130.0
    if stretch is None:
        stretch = 2.2 if column else 1.8
    road = panels[0][2].metadata["road"]
    ylim = (road["y_min"] - 0.4, road["y_max"] + 0.4)
    fig_w, height, rects, short_span, ncol = _paper_layout(
        column, ylim[1] - ylim[0], span, short_span, stretch
    )
    style = {
        "mathtext.fontset": "dejavusans",
        "font.family": "DejaVu Sans",
        # Embed TrueType, not Type 3, fonts (IEEE PDF checks reject Type 3).
        "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(fig_w, height))

        def axes_at(left, bottom, size, tall):
            return fig.add_axes(
                [left / fig_w, bottom / height, size / fig_w, tall / height]
            )

        axes = []
        for index, panel in enumerate(panels):
            title, result, env, step, ghosts, _label = panel
            ax = axes_at(*rects[index])
            xlim = (start, start + (short_span if index else span))
            _frame_strip(
                ax, title, result, env, step, xlim, ylim, show_uncertainty
            )
            _ego_ghosts(ax, result, env, step, ghosts, dt)
            if index:
                ax.set_xlabel("x (m)", fontsize=PAPER_FONT, labelpad=0)
            axes.append(ax)
        axes[0].set_xticks(np.arange(0, start + span, 20))
        score = axes_at(*rects[3])
        letter = chr(ord("a") + len(panels))
        _score_panel(
            score,
            panels,
            dt,
            alpha,
            events,
            f"({letter}) Exact lower score per window",
        )
        axes.append(score)
        handles = {}
        for ax in axes[:-1]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(label, handle)
        fig.legend(
            handles.values(),
            handles.keys(),
            loc="upper center",
            ncol=ncol,
            fontsize=PAPER_FONT - 1,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
            handlelength=1.4,
            columnspacing=1.0,
            labelspacing=0.3,
        )
        if save_path is not None:
            stem = Path(save_path).with_suffix("")
            stem.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(stem.with_suffix(".pdf"))
            fig.savefig(stem.with_suffix(".png"), dpi=300)
        if show and matplotlib.get_backend().lower() != "agg":
            plt.show(block=True)
        plt.close(fig)
    return fig, axes


# --- Timing benchmark ---------------------------------------------------------

TIMING_COLORS = {"parallel": "tab:blue", "recurrent": "tab:orange"}
TIMING_LINES = {"cpu": ("-", "o"), "cuda": ("--", "s")}
TIMING_DEVICE_NAMES = {"cpu": "CPU", "cuda": "GPU"}


def _timing_series(rows, experiment, x_key, y_key):
    """{(method, device): (xs, ys)} for one experiment, sorted by x."""
    series = {}
    for row in rows:
        if row["experiment"] != experiment:
            continue
        key = (row["method"], row["device"].split(":")[0])
        series.setdefault(key, []).append((row[x_key], row[y_key]))
    return {key: tuple(zip(*sorted(points))) for key, points in series.items()}


def plot_timing(result, *, save_path=None, show=False):
    """One-column figure of the evaluation benchmark.

    (a) milliseconds per smooth forward and backward pass against trace
    length at batch 1; (b) candidate plans scored per second against batch
    size. Colour: parallel (blue) or recurrent (orange); line: CPU (solid)
    or GPU (dashed).

    Args:
        result: Output of `experiments.timing.benchmark`.
        save_path: Output path; a PDF and a 300 dpi PNG are written.
        show: Display the figure.

    Returns:
        (figure, (axes_a, axes_b)).
    """
    style = {
        "mathtext.fontset": "dejavusans",
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
    }
    panels = (
        ("A", "steps", "ms", "(a) Time per iteration", "trace length T", "ms"),
        (
            "B",
            "batch",
            "plans per s",
            f"(b) Throughput, T = {_batch_steps(result)}",
            "candidate plans per call",
            "plans / s",
        ),
    )
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1, 2, figsize=(COLUMN_WIDTH, 1.75), constrained_layout=True
        )
        for ax, (experiment, x_key, y_key, title, xlabel, ylabel) in zip(
            axes, panels
        ):
            series = _timing_series(result["rows"], experiment, x_key, y_key)
            for (method, device), (xs, ys) in series.items():
                line, marker = TIMING_LINES[device]
                ax.plot(
                    xs,
                    ys,
                    ls=line,
                    marker=marker,
                    ms=2.5,
                    lw=1.0,
                    color=TIMING_COLORS[method],
                    label=f"{method}, {TIMING_DEVICE_NAMES[device]}",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            # Tick each measured size, written plainly (31, 61, ... 4096).
            measured = sorted({x for xs, _ys in series.values() for x in xs})
            ax.set_xticks(measured, [f"{x:g}" for x in measured])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_title(title, loc="left", fontsize=PAPER_FONT, pad=1.5)
            ax.set_xlabel(xlabel, fontsize=PAPER_FONT, labelpad=0)
            ax.set_ylabel(ylabel, fontsize=PAPER_FONT, labelpad=1)
            ax.tick_params(labelsize=PAPER_FONT - 1, length=2, pad=1)
            ax.grid(alpha=0.25, lw=0.4, which="major")
            for side in ("right", "top"):
                ax.spines[side].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=2,
            fontsize=PAPER_FONT - 1,
            frameon=False,
            handlelength=1.8,
            columnspacing=0.8,
        )
        if save_path is not None:
            stem = Path(save_path).with_suffix("")
            stem.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(stem.with_suffix(".pdf"))
            fig.savefig(stem.with_suffix(".png"), dpi=300)
        if show and matplotlib.get_backend().lower() != "agg":
            plt.show(block=True)
        plt.close(fig)
    return fig, tuple(axes)


def _batch_steps(result):
    """Trace length of the batch experiment."""
    steps = {
        row["steps"] for row in result["rows"] if row["experiment"] == "B"
    }
    return steps.pop() if len(steps) == 1 else "varies"
