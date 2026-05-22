import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch

# Tableau 10 colors
PALETTE = {
    "ego": {"fill": "#1f77b4", "stroke": "#1f77b4"},  # Tableau Blue
    "plan": {"fill": "#ff7f0e", "stroke": "#ff7f0e"},  # Tableau Orange
    "visit": {"fill": "#c5b0d5", "stroke": "#9467bd"},  # Tableau Green (Light/Dark)
    "obs_static": {"fill": "#ff9896", "stroke": "#d62728"},  # Tableau Red (Light/Dark)
    "obs_moving": {
        "fill": "#ff9896",
        "stroke": "#d62728",
    },  # Tableau Purple (Light/Dark)
    "lane": {"fill": "#c7c7c7", "stroke": "#7f7f7f"},  # Tableau Gray (Light/Dark)
    "goal": {"fill": "#98df8a", "stroke": "#2ca02c"},  # Tableau Green (Light/Dark)
    "road": {"fill": "#F2F2F7"},  # Light Gray Background
}


def cov_ellipse_params(cov, k=1.96):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * k * np.sqrt(vals)
    return theta, width, height


def plot_covariance_ellipse(
    ax,
    mean,
    cov,
    k=1.96,
    facecolor="blue",
    edgecolor="blue",
    alpha=0.4,
    zorder=10,
    label=None,
):
    """Draws a confidence ellipse for a 2D Gaussian belief."""
    theta, width, height = cov_ellipse_params(cov, k)

    ellipse = patches.Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    ax.add_patch(ellipse)


def draw_road_backdrop(ax, env):
    """Draw road background, goal lane fill, and lane markings onto ax.

    Returns (road_lo, road_hi) so callers can set ylim from road extent.
    """
    road_lo = min(lm["y"] for lm in env.lane_markings) if env.lane_markings else -2.0
    road_hi = max(lm["y"] for lm in env.lane_markings) if env.lane_markings else 6.0
    ax.axhspan(road_lo, road_hi, color=PALETTE["road"]["fill"], zorder=0)
    if env.goal:
        gy0, gy1 = env.goal["y"]
        ax.axhspan(
            gy0,
            gy1,
            color=PALETTE["goal"]["fill"],
            alpha=0.22,
            zorder=1,
            label="Goal Lane",
        )
    for lane in env.lane_markings:
        style = "--" if lane["style"] == "dashed" else "-"
        lw = 1.5 if lane["style"] == "dashed" else 2.0
        ax.axhline(
            lane["y"],
            color=PALETTE["lane"]["stroke"],
            linestyle=style,
            linewidth=lw,
            alpha=0.9,
            zorder=2,
        )
    return road_lo, road_hi


def draw_env_on_ax(
    ax,
    env,
    *,
    draw_moving_path=True,
    obs_static_label="Obstacle",
    visit_label="Visit Region",
    moving_obs_label="Moving Obstacle Path",
    moving_obs_snapshots=False,
    x_mask=None,
):
    """Render environment geometry: road, goal, obstacles, visit regions, moving paths."""
    if env.lane_markings:
        draw_road_backdrop(ax, env)
    else:
        if env.goal:
            gx, gy = env.goal["x"], env.goal["y"]
            ax.add_patch(
                patches.Rectangle(
                    (gx[0], gy[0]),
                    gx[1] - gx[0],
                    gy[1] - gy[0],
                    facecolor=PALETTE["goal"]["fill"],
                    edgecolor=PALETTE["goal"]["stroke"],
                    alpha=0.4,
                    zorder=3,
                    label="Goal",
                )
            )

    for region in env.visit_regions:
        vx, vy = region["x"], region["y"]
        ax.add_patch(
            patches.Rectangle(
                (vx[0], vy[0]),
                vx[1] - vx[0],
                vy[1] - vy[0],
                facecolor=PALETTE["visit"]["fill"],
                edgecolor=PALETTE["visit"]["stroke"],
                alpha=0.4,
                zorder=3,
                label=visit_label,
            )
        )

    for obs in env.obstacles:
        ox, oy = obs["x"], obs["y"]
        ax.add_patch(
            patches.Rectangle(
                (ox[0], oy[0]),
                ox[1] - ox[0],
                oy[1] - oy[0],
                facecolor=PALETTE["obs_static"]["fill"],
                edgecolor=PALETTE["obs_static"]["stroke"],
                alpha=0.6,
                hatch="//",
                zorder=4,
                label=obs_static_label,
            )
        )

    for obs in env.circle_obstacles:
        ax.add_patch(
            patches.Circle(
                obs["center"],
                obs["radius"],
                facecolor=PALETTE["obs_static"]["fill"],
                edgecolor=PALETTE["obs_static"]["stroke"],
                alpha=0.6,
                hatch="//",
                zorder=4,
            )
        )

    if draw_moving_path or moving_obs_snapshots:
        for obs in env.moving_obstacles:
            xt = np.asarray(
                obs["x_traj"].detach().cpu()
                if isinstance(obs["x_traj"], torch.Tensor)
                else obs["x_traj"]
            )
            yt = np.asarray(
                obs["y_traj"].detach().cpu()
                if isinstance(obs["y_traj"], torch.Tensor)
                else obs["y_traj"]
            )
            mask = np.ones(len(xt), dtype=bool)
            if x_mask is not None:
                mask = (xt >= x_mask[0]) & (xt <= x_mask[1])
            if draw_moving_path:
                ax.plot(
                    xt[mask],
                    yt[mask],
                    color=PALETTE["obs_moving"]["stroke"],
                    linestyle="--",
                    alpha=0.4,
                    label=moving_obs_label,
                )
            w, h = obs["width"], obs["height"]
            if moving_obs_snapshots:
                snap_step = max(1, len(xt) // 5)
                for k_i in range(0, len(xt), snap_step):
                    if mask[k_i]:
                        ax.add_patch(
                            patches.Rectangle(
                                (xt[k_i] - w / 2, yt[k_i] - h / 2),
                                w,
                                h,
                                facecolor=PALETTE["obs_moving"]["fill"],
                                edgecolor=PALETTE["obs_moving"]["stroke"],
                                alpha=0.3,
                                zorder=4,
                            )
                        )


def _compute_env_bounds(mean_np, env):
    x_min, x_max = np.min(mean_np[:, 0]), np.max(mean_np[:, 0])
    y_min, y_max = np.min(mean_np[:, 1]), np.max(mean_np[:, 1])
    for lane in env.lane_markings:
        x_min = min(x_min, min(lane["x"]))
        x_max = max(x_max, max(lane["x"]))
        y_min = min(y_min, lane["y"])
        y_max = max(y_max, lane["y"])
    if env.goal:
        x_min = min(x_min, env.goal["x"][0])
        x_max = max(x_max, env.goal["x"][1])
        y_min = min(y_min, env.goal["y"][0])
        y_max = max(y_max, env.goal["y"][1])
    for obs in env.obstacles:
        x_min = min(x_min, obs["x"][0])
        x_max = max(x_max, obs["x"][1])
        y_min = min(y_min, obs["y"][0])
        y_max = max(y_max, obs["y"][1])
    for obs in env.circle_obstacles:
        x_min = min(x_min, obs["center"][0] - obs["radius"])
        x_max = max(x_max, obs["center"][0] + obs["radius"])
        y_min = min(y_min, obs["center"][1] - obs["radius"])
        y_max = max(y_max, obs["center"][1] + obs["radius"])
    for region in env.visit_regions:
        x_min = min(x_min, region["x"][0])
        x_max = max(x_max, region["x"][1])
        y_min = min(y_min, region["y"][0])
        y_max = max(y_max, region["y"][1])
    return x_min, x_max, y_min, y_max


def draw_ego_rect(ax, x, y, heading_deg, rw, rh, alpha, zorder=7):
    t_aff = (
        transforms.Affine2D()
        .translate(-rw / 2, -rh / 2)
        .rotate_deg(heading_deg)
        .translate(x, y)
    )
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            rw,
            rh,
            transform=t_aff + ax.transData,
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            linewidth=1.2,
            alpha=alpha,
            zorder=zorder,
        )
    )


def heading_deg(mean_np, t, T):
    dx = mean_np[min(t + 1, T), 0] - mean_np[max(t - 1, 0), 0]
    dy = mean_np[min(t + 1, T), 1] - mean_np[max(t - 1, 0), 1]
    return np.degrees(np.arctan2(dy, dx))


def plot_trajectory(mean_np, cov_np, env):
    T = mean_np.shape[0] - 1
    x_min, x_max, y_min, y_max = _compute_env_bounds(mean_np, env)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x_min - 1.0, x_max + 1.0)
    ax.set_ylim(y_min - 1.0, y_max + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [m]", fontsize=20, fontweight="bold")
    ax.set_ylabel("$y$ [m]", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)

    draw_env_on_ax(ax, env, moving_obs_snapshots=True)

    if env.goal:
        gx, gy = env.goal["x"], env.goal["y"]
        ax.text(
            (gx[0] + gx[1]) / 2,
            (gy[0] + gy[1]) / 2,
            "G",
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
            color=PALETTE["goal"]["stroke"],
            zorder=30,
        )

    for region in env.visit_regions:
        vx, vy = region["x"], region["y"]
        ax.text(
            (vx[0] + vx[1]) / 2,
            (vy[0] + vy[1]) / 2,
            "V",
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
            color=PALETTE["visit"]["stroke"],
            zorder=30,
        )

    ax.plot(
        mean_np[:, 0],
        mean_np[:, 1],
        color=PALETTE["ego"]["stroke"],
        linewidth=2.5,
        alpha=0.9,
        label="Trajectory",
        zorder=25,
    )
    for t in range(0, T + 1, 2):
        plot_covariance_ellipse(
            ax,
            mean_np[t, :2],
            cov_np[t, :2, :2],
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.25,
            zorder=15,
            label="Uncertainty" if t == 0 else None,
        )

    start_pos = mean_np[0, :2]
    ax.text(
        start_pos[0] - 0.5,
        start_pos[1],
        "S",
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="center",
        color=PALETTE["ego"]["stroke"],
        zorder=30,
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            ncol=1,
            fontsize=17,
            framealpha=0.95,
            edgecolor="#cccccc",
        )

    plt.show()
    plt.close(fig)


def plot_controls(u_np):
    T = u_np.shape[0]
    time_steps = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_steps, u_np[:, 0], color=PALETTE["ego"]["stroke"], linewidth=1.8)
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("$u_x$", fontsize=18, fontweight="bold")
    axes[0].tick_params(labelsize=16)
    axes[0].grid(True, alpha=0.35)
    axes[1].plot(time_steps, u_np[:, 1], color=PALETTE["plan"]["stroke"], linewidth=1.8)
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[1].set_ylabel("$u_y$", fontsize=18, fontweight="bold")
    axes[1].set_xlabel("Time Step", fontsize=18, fontweight="bold")
    axes[1].tick_params(labelsize=16)
    axes[1].grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_metrics(history, p_sat_trace):
    if history is None and p_sat_trace is None:
        return
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if p_sat_trace is not None:
        ax.plot(
            p_sat_trace,
            color=PALETTE["goal"]["stroke"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=r"$P_{\downarrow}(\varphi)$",
        )
        ax.set_ylabel(r"$P_{\downarrow}(\varphi)$", fontsize=18, fontweight="bold")
    else:
        ax.plot(history, color=PALETTE["lane"]["stroke"], linewidth=2, label="Loss")
        ax.set_ylabel("Loss", fontsize=18, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=16,
        framealpha=0.95,
        edgecolor="#cccccc",
    )
    fig.subplots_adjust(bottom=0.25)
    plt.show()
    plt.close(fig)


def visualize_results(
    mean_trace,
    cov_trace,
    u_trace,
    env,
    history=None,
    p_sat_trace=None,
    robot_dims=None,
):
    mean_np = mean_trace.cpu().squeeze().numpy()
    cov_np = cov_trace.cpu().squeeze().numpy()
    u_np = u_trace.cpu().squeeze().numpy()
    plot_trajectory(mean_np, cov_np, env)
    plot_controls(u_np)
    plot_metrics(history, p_sat_trace)


def plot_lc_trajectory(
    ax,
    mean_trace,
    cov_trace,
    env,
    dt,
    robot_dims,
    title=None,
    show_legend=True,
    xlim=None,
):
    mean_np = mean_trace.cpu().squeeze().numpy()  # [T+1, ≥2]
    cov_np = cov_trace.cpu().squeeze().numpy()  # [T+1, D, D]
    T = mean_np.shape[0] - 1

    road_lo = min(lm["y"] for lm in env.lane_markings) if env.lane_markings else -2.0
    road_hi = max(lm["y"] for lm in env.lane_markings) if env.lane_markings else 6.0
    x_lo = mean_np[:, 0].min() - 1.5
    x_hi = mean_np[:, 0].max() + 1.5
    y_lo, y_hi = road_lo - 1.2, road_hi + 1.2

    if xlim:
        ax.set_xlim(xlim)
        x_lo = min(x_lo, xlim[0])
        x_hi = max(x_hi, xlim[1])
    else:
        ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("$y$ [m]", fontsize=24)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title, fontsize=22, fontweight="bold")

    draw_env_on_ax(
        ax,
        env,
        obs_static_label="Stopped Vehicle",
        visit_label="Merge Zone",
        moving_obs_label="Other Vehicle Path",
        moving_obs_snapshots=True,
        x_mask=(x_lo, x_hi),
    )

    # Uncertainty tube
    step_ell = max(1, T // 16)
    for t in range(0, T + 1, step_ell):
        plot_covariance_ellipse(
            ax,
            mean_np[t, :2],
            cov_np[t, :2, :2],
            k=2.45,
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.16,
            zorder=5,
            label="95% CI" if t == 0 else None,
        )

    ax.plot(
        mean_np[:, 0],
        mean_np[:, 1],
        color=PALETTE["ego"]["stroke"],
        linewidth=2.2,
        alpha=0.9,
        zorder=7,
        label="Ego Trajectory",
    )
    if robot_dims:
        rw, rh = robot_dims
        for t in range(0, T + 1, max(1, T // 10)):
            draw_ego_rect(
                ax,
                mean_np[t, 0],
                mean_np[t, 1],
                heading_deg(mean_np, t, T),
                rw,
                rh,
                alpha=0.30,
                zorder=6,
            )

    ax.plot(
        mean_np[0, 0],
        mean_np[0, 1],
        "o",
        color=PALETTE["ego"]["stroke"],
        markersize=6,
        zorder=9,
    )
    ax.plot(
        mean_np[-1, 0],
        mean_np[-1, 1],
        "s",
        color=PALETTE["ego"]["stroke"],
        markersize=6,
        zorder=9,
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    if show_legend:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=len(by_label),
            fontsize=18,
            framealpha=0.95,
        )

    return by_label


def plot_lc_snapshots(
    ax,
    mean_trace,
    cov_trace,
    env,
    dt,
    robot_dims,
    title=None,
    show_legend=True,
    show_xlabel=True,
    xlim=None,
):
    mean_np = mean_trace.cpu().squeeze().numpy()  # [T+1, ≥2]
    cov_np = cov_trace.cpu().squeeze().numpy()  # [T+1, D, D]
    T = mean_np.shape[0] - 1

    road_lo = min(lm["y"] for lm in env.lane_markings) if env.lane_markings else -2.0
    road_hi = max(lm["y"] for lm in env.lane_markings) if env.lane_markings else 6.0
    x_lo = mean_np[:, 0].min() - 1.5
    x_hi = mean_np[:, 0].max() + 1.5
    y_lo, y_hi = road_lo - 1.2, road_hi + 1.2

    N_SNAP = 6
    snap_t = np.linspace(0, T, N_SNAP, dtype=int)

    if xlim:
        ax.set_xlim(xlim)
        x_lo = min(x_lo, xlim[0])
        x_hi = max(x_hi, xlim[1])
    else:
        ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    if show_xlabel:
        ax.set_xlabel("$x$ [m]", fontsize=24)
    ax.set_ylabel("$y$ [m]", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title, fontsize=22, fontweight="bold")

    draw_env_on_ax(ax, env, x_mask=(x_lo, x_hi))

    # Thin full trajectory as reference line
    ax.plot(
        mean_np[:, 0],
        mean_np[:, 1],
        color=PALETTE["ego"]["stroke"],
        linewidth=1.2,
        alpha=0.35,
        zorder=4,
    )

    # Legend patches
    ego_patch = patches.Patch(
        facecolor=PALETTE["ego"]["fill"],
        edgecolor=PALETTE["ego"]["stroke"],
        linewidth=1.2,
        label="Ego Vehicle",
    )
    obs_patch = patches.Patch(
        facecolor=PALETTE["obs_moving"]["fill"],
        edgecolor=PALETTE["obs_moving"]["stroke"],
        linewidth=1.2,
        label="Other Vehicle",
    )
    ci_patch = patches.Patch(
        facecolor=PALETTE["ego"]["fill"],
        edgecolor=PALETTE["ego"]["stroke"],
        linewidth=0.8,
        alpha=0.25,
        label="95% CI",
    )
    goal_patch = patches.Patch(
        facecolor=PALETTE["goal"]["fill"],
        edgecolor="none",
        alpha=0.5,
        label="Goal Lane",
    )
    legend_handles = [ego_patch, obs_patch]
    if env.obstacles:
        static_patch = patches.Patch(
            facecolor=PALETTE["obs_static"]["fill"],
            edgecolor=PALETTE["obs_static"]["stroke"],
            linewidth=1.2,
            label="Stopped Vehicle",
        )
        legend_handles.append(static_patch)
    if env.visit_regions:
        visit_patch = patches.Patch(
            facecolor=PALETTE["visit"]["fill"],
            edgecolor=PALETTE["visit"]["stroke"],
            alpha=0.5,
            label="Merge Zone",
        )
        legend_handles.append(visit_patch)
    legend_handles.extend([ci_patch, goal_patch])

    # Faint intermediate covariance tube
    for t in range(0, T + 1, max(1, T // 20)):
        plot_covariance_ellipse(
            ax,
            mean_np[t, :2],
            cov_np[t, :2, :2],
            k=2.45,
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.06,
            zorder=4,
        )

    for ki, t in enumerate(snap_t):
        frac = ki / (N_SNAP - 1)
        alpha = 0.35 + 0.55 * frac
        t_sec = t * dt

        plot_covariance_ellipse(
            ax,
            mean_np[t, :2],
            cov_np[t, :2, :2],
            k=2.45,
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.10 + 0.15 * frac,
            zorder=5,
        )

        ex, ey = mean_np[t, 0], mean_np[t, 1]
        if robot_dims:
            rw, rh = robot_dims
            draw_ego_rect(ax, ex, ey, heading_deg(mean_np, t, T), rw, rh, alpha=alpha, zorder=7)
            label_y_off = rh / 2 + 0.45
        else:
            ax.plot(
                ex,
                ey,
                "o",
                color=PALETTE["ego"]["stroke"],
                markersize=6,
                alpha=alpha,
                zorder=7,
            )
            label_y_off = 0.5

        ax.annotate(
            f"$t={t_sec:.1f}\\,$s",
            xy=(ex, ey + label_y_off),
            fontsize=12,
            ha="center",
            va="bottom",
            color=PALETTE["ego"]["stroke"],
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

        for obs in env.moving_obstacles:
            xt = np.asarray(
                obs["x_traj"].detach().cpu()
                if isinstance(obs["x_traj"], torch.Tensor)
                else obs["x_traj"]
            )
            yt = np.asarray(
                obs["y_traj"].detach().cpu()
                if isinstance(obs["y_traj"], torch.Tensor)
                else obs["y_traj"]
            )
            if t < len(xt) and x_lo <= xt[t] <= x_hi:
                ax.add_patch(
                    patches.Rectangle(
                        (xt[t] - obs["width"] / 2, yt[t] - obs["height"] / 2),
                        obs["width"],
                        obs["height"],
                        facecolor=PALETTE["obs_moving"]["fill"],
                        edgecolor=PALETTE["obs_moving"]["stroke"],
                        linewidth=1.0,
                        alpha=alpha,
                        zorder=6,
                    )
                )

    if show_legend:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(legend_handles),
            fontsize=22,
            framealpha=0.95,
            edgecolor="#cccccc",
        )

    return legend_handles


def visualize_lane_change(
    mean_trace,
    cov_trace,
    u_trace,
    env,
    p_sat_trace=None,
    dt=0.2,
    robot_dims=None,
    xlim=None,
):
    mean_np = mean_trace.cpu().squeeze().numpy()  # [T+1, ≥2]
    u_np = u_trace.cpu().squeeze().numpy()  # [T,  2]
    T = mean_np.shape[0] - 1
    time_u = np.arange(T) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    h_traj = plot_lc_trajectory(
        ax1, mean_trace, cov_trace, env, dt, robot_dims, show_legend=False, xlim=xlim
    )
    h_snap = plot_lc_snapshots(
        ax2, mean_trace, cov_trace, env, dt, robot_dims, show_legend=False, xlim=xlim
    )

    combined_handles = []
    seen = set()
    for lbl, h in h_traj.items():
        if lbl not in seen:
            seen.add(lbl)
            combined_handles.append(h)
    for h in h_snap:
        lbl = h.get_label()
        if lbl not in seen:
            seen.add(lbl)
            combined_handles.append(h)

    fig.legend(
        handles=combined_handles,
        loc="lower center",
        ncol=min(len(combined_handles), 4),
        fontsize=16,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    plt.close(fig)

    n_rows = 3 if p_sat_trace is not None else 2
    fig3, axes = plt.subplots(n_rows, 1, figsize=(10, 2.6 * n_rows), sharex=True)

    axes[0].plot(time_u, u_np[:, 0], color=PALETTE["ego"]["stroke"], linewidth=1.8)
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("$a_x$ [m/s²]", fontsize=16)
    axes[0].set_title(
        "Control Inputs and Satisfaction Probability", fontsize=18, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.35)
    axes[0].tick_params(labelsize=14)

    axes[1].plot(time_u, u_np[:, 1], color=PALETTE["plan"]["stroke"], linewidth=1.8)
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[1].set_ylabel("$a_y$ [m/s²]", fontsize=16)
    axes[1].grid(True, alpha=0.35)
    axes[1].tick_params(labelsize=14)

    if p_sat_trace is not None:
        p_sat_arr = np.asarray(p_sat_trace)
        axes[2].plot(
            time_u[: len(p_sat_arr)],
            p_sat_arr,
            color=PALETTE["goal"]["stroke"],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=r"$P_{\downarrow}(\varphi)$",
        )
        axes[2].axhline(
            0.85,
            color="k",
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
            label="Threshold ($\\alpha = 0.85$)",
        )
        axes[2].set_ylim(0, 1.05)
        axes[2].set_ylabel(r"$P_{\downarrow}(\varphi)$", fontsize=16)
        axes[2].legend(fontsize=14, loc="lower right", framealpha=0.9)
        axes[2].grid(True, alpha=0.35)
        axes[2].tick_params(labelsize=14)

    axes[-1].set_xlabel("Time [s]", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close(fig3)
