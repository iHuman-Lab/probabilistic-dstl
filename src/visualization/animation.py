"""GIF of a reach-avoid plan, step by step."""

from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from visualization.figures import (
    COLORS,
    RHO_LOWER,
    _draw_environment,
    _ellipse,
    _ellipse_parameters,
    _np,
)


def _finish_animation(fig, movie, filename, fps, show):
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show(block=True)
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        movie.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def _frames(last_step, fps):
    return [0] * fps + list(range(1, last_step + 1)) + [last_step] * fps


def animate_reach_avoid(
    result,
    env,
    *,
    dt,
    title="Reach–Avoid",
    filename=None,
    fps=6,
    show=False,
):
    """Reveal one optimized belief trajectory inside its environment."""
    mean = _np(result.rollout.aux["mean_trace"])[0]
    covariance = _np(result.rollout.aux["cov_trace"])[0]
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    _draw_environment(ax, env)
    ax.plot(
        mean[:, 0],
        mean[:, 1],
        color=COLORS["mean"],
        alpha=0.15,
    )
    ax.scatter(
        [mean[0, 0]],
        [mean[0, 1]],
        marker="o",
        s=55,
        color="black",
        zorder=10,
    )
    (path,) = ax.plot(
        [],
        [],
        color=COLORS["mean"],
        linewidth=2.2,
        zorder=8,
    )
    (point,) = ax.plot(
        [],
        [],
        marker="s",
        color=COLORS["mean"],
        markersize=6,
        zorder=10,
    )
    ellipse = _ellipse(ax, mean[0], covariance[0])
    status = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_title(
        f"{title} | {RHO_LOWER} = {result.hard_interval[0]:.3f}",
        fontweight="bold",
    )

    def update(step):
        path.set_data(mean[: step + 1, 0], mean[: step + 1, 1])
        point.set_data([mean[step, 0]], [mean[step, 1]])
        width, height, angle = _ellipse_parameters(covariance[step])
        ellipse.set_center(mean[step, :2])
        ellipse.set_width(width)
        ellipse.set_height(height)
        ellipse.set_angle(angle)
        status.set_text(
            f"t = {step * dt:.1f} s  ·  predicted step {step}/{len(mean) - 1}"
        )
        return path, point, ellipse, status

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=_frames(len(mean) - 1, fps),
        interval=1000 / fps,
        blit=False,
    )
    _finish_animation(fig, movie, filename, fps, show)
    return fig, ax, movie
