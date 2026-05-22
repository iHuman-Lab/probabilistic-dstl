import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from visualization.planning import PALETTE, cov_ellipse_params, draw_env_on_ax, draw_road_backdrop


def setup_mpc_live_plot(env):
    """Create the two-panel live figure for MPC execution."""
    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax_map = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])

    if env.bounds:
        ax_map.set_xlim(*env.bounds["x"])
        ax_map.set_ylim(*env.bounds["y"])
    ax_map.set_aspect("equal")
    ax_map.grid(True, alpha=0.3)
    ax_map.set_title("MPC Live Execution")

    draw_env_on_ax(ax_map, env)

    (line_exec,) = ax_map.plot([], [], color=PALETTE["ego"]["stroke"], marker="o", label="Executed Path")
    (line_plan,) = ax_map.plot([], [], color=PALETTE["plan"]["stroke"], linestyle="--", alpha=0.8, label="Planned Window")
    ax_map.legend(loc="upper left")

    ax_p.set_xlim(0, 100)
    ax_p.set_ylim(0, 1.1)
    ax_p.set_title("Window Satisfaction Prob")
    ax_p.set_xlabel("Step")
    ax_p.set_ylabel("P(Sat)")
    ax_p.grid(True)
    (line_p,) = ax_p.plot([], [], color=PALETTE["goal"]["stroke"], marker="o", markersize=3)

    return fig, ax_map, ax_p, line_exec, line_plan, line_p


def setup_lane_change_live_plot(env, label="", xlim=None):
    """Create the live-execution figure for lane-change MPC."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.grid(True, alpha=0.3, zorder=3)
    ax.set_title(f"Lane Change MPC ({label}) — Live Execution")
    ax.set_ylabel("$y$ [m]")
    ax.set_xlabel("$x$ [m]")

    road_lo, road_hi = draw_road_backdrop(ax, env)

    if env.success:
        ax.axhspan(
            env.success["y_min"], env.success["y_max"],
            color=PALETTE["goal"]["fill"], alpha=0.15, zorder=1,
        )

    divider_y = next(
        (lm["y"] for lm in env.lane_markings if lm["style"] == "dashed"), None
    )
    if divider_y is not None:
        _blend = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.02, (road_lo + divider_y) / 2, "Lane 1",
            transform=_blend, color=PALETTE["lane"]["stroke"],
            fontsize=8, va="center", ha="left",
        )
        ax.text(
            0.02, (divider_y + road_hi) / 2, "Lane 2",
            transform=_blend, color=PALETTE["lane"]["stroke"],
            fontsize=8, va="center", ha="left",
        )

    (ego_dot,) = ax.plot(
        [], [], color=PALETTE["ego"]["stroke"], marker="o", markersize=8,
        label="Ego", zorder=10,
    )
    (ego_trail,) = ax.plot(
        [], [], color=PALETTE["ego"]["stroke"], alpha=0.4, linewidth=1.5, zorder=9,
    )
    (plan_line,) = ax.plot(
        [], [], color=PALETTE["plan"]["stroke"], linestyle="--",
        alpha=0.8, linewidth=1.5, label="Plan", zorder=8,
    )
    ego_cov_patch = patches.Ellipse(
        (0, 0), width=0, height=0, angle=0,
        facecolor=PALETTE["ego"]["fill"], edgecolor=PALETTE["ego"]["stroke"],
        alpha=0.2, label="Uncertainty", zorder=7,
    )
    ax.add_patch(ego_cov_patch)

    obs0 = env.moving_obstacles[0]
    obs_pos0 = env.moving_obstacle_position(0)
    obs_rect = patches.Rectangle(
        (obs_pos0[0] - obs0["width"] / 2, obs_pos0[1] - obs0["height"] / 2),
        obs0["width"], obs0["height"],
        facecolor=PALETTE["obs_moving"]["fill"], edgecolor=PALETTE["obs_moving"]["stroke"],
        alpha=0.8, label="Other Car", zorder=9,
    )
    ax.add_patch(obs_rect)
    ax.legend(loc="upper right", fontsize=8)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(road_lo - 1, road_hi + 1)

    return fig, ax, ego_dot, ego_trail, plan_line, ego_cov_patch, obs_rect


def update_mpc_live_plot(fig, line_exec, line_plan, line_p, ax_p,
                         real_mean_trace, best_mean, p_sat_trace):
    xs = [m[0].item() for m in real_mean_trace]
    ys = [m[1].item() for m in real_mean_trace]
    line_exec.set_data(xs, ys)

    plan_np = best_mean.detach().cpu().squeeze().numpy()
    line_plan.set_data(plan_np[:, 0], plan_np[:, 1])

    steps = list(range(len(p_sat_trace)))
    line_p.set_data(steps, p_sat_trace)
    if p_sat_trace:
        ax_p.set_xlim(0, max(len(p_sat_trace) + 5, 20))

    plt.pause(0.001)


def update_lane_change_plot(
    ego_dot, ego_trail, plan_line, ego_cov_patch, obs_rect,
    real_mean_trace, curr_cov, p_mean, obs_pos, obs_cfg,
):
    ego_pos = real_mean_trace[-1].cpu().numpy()
    ego_x, ego_y = ego_pos[0], ego_pos[1]

    ego_dot.set_data([ego_x], [ego_y])
    ego_trail.set_data(
        [m[0].item() for m in real_mean_trace],
        [m[1].item() for m in real_mean_trace],
    )

    theta, w_e, h_e = cov_ellipse_params(curr_cov[:2, :2].cpu().numpy())
    ego_cov_patch.set_center((ego_x, ego_y))
    ego_cov_patch.set_width(w_e)
    ego_cov_patch.set_height(h_e)
    ego_cov_patch.set_angle(theta)

    plan_np = p_mean.detach().cpu().squeeze().numpy()
    plan_line.set_data(plan_np[:, 0], plan_np[:, 1])
    obs_rect.set_xy((obs_pos[0] - obs_cfg["width"] / 2, obs_pos[1] - obs_cfg["height"] / 2))
    plt.pause(0.001)


def make_mpc_live_callback(env):
    fig, ax_map, ax_p, line_exec, line_plan, line_p = setup_mpc_live_plot(env)
    real_trace, p_sat_so_far = [], []

    def callback(step, curr_mean, curr_cov, best_mean, best_p):
        real_trace.append(curr_mean.detach())
        p_sat_so_far.append(best_p)
        update_mpc_live_plot(fig, line_exec, line_plan, line_p, ax_p,
                             real_trace, best_mean, p_sat_so_far)

    return callback


def make_lane_change_live_callback(env):
    fig_live, _ax, ego_dot, ego_trail, plan_line, ego_cov_patch, obs_rect = (
        setup_lane_change_live_plot(env, label=env.label, xlim=env.plot_xlim)
    )
    real_trace = []
    obs0 = env.moving_obstacles[0]

    def callback(step, curr_mean, curr_cov, best_mean, best_p):
        real_trace.append(curr_mean.detach())
        obs_pos = env.moving_obstacle_position(step)
        if obs_pos is None:
            obs_pos = [0.0, 0.0]
        update_lane_change_plot(
            ego_dot, ego_trail, plan_line, ego_cov_patch, obs_rect,
            real_trace, curr_cov, best_mean, obs_pos, obs0,
        )

    return callback
