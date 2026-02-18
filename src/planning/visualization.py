import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import torch


PALETTE = {
    "ego": {"fill": "#1f77b4", "stroke": "#1f77b4"},       # Deep Blue
    "goal": {"fill": "#2ca02c", "stroke": "#2ca02c"},      # Deep Green
    "visit": {"fill": "#bcbd22", "stroke": "#bcbd22"},     # Deep Olive
    "obs_static": {"fill": "#d62728", "stroke": "#d62728"},# Deep Red
    "obs_moving": {"fill": "#d62728", "stroke": "#d62728"},# Deep Red
    "lane": {"fill": "#7f7f7f", "stroke": "#7f7f7f"},      # Deep Gray
    "plan": {"fill": "#ff7f0e", "stroke": "#ff7f0e"},      # Deep Orange
}


def plot_covariance_ellipse(ax, mean, cov, k=1.96, facecolor="blue", edgecolor="blue", alpha=2.0, zorder=10):
    """
    Draws a confidence ellipse for a 2D Gaussian belief.
    k=2.45 corresponds to ~95% confidence.
    """
    # Eigendecomposition for ellipse orientation/scale
    vals, vecs = np.linalg.eigh(cov)

    # Sort eigenvalues/vectors (largest first)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Calculate angle (degrees) and width/height
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * k * np.sqrt(vals)

    ellipse = patches.Ellipse(
        xy=mean, width=width, height=height, angle=theta, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=zorder
    )
    ax.add_patch(ellipse)


def visualize_results(
    mean_trace, cov_trace, u_trace, env, history=None, p_sat_trace=None, robot_dims=None, layout="default"
):
    """
    Generates the verification outputs:
    1. Spatial Trajectory with Covariance Ellipses
    2. Optimization Objective Convergence
    3. Control Input Evolution
    """
    # Convert tensors to numpy (ensure they are on CPU)
    mean_np = mean_trace.cpu().squeeze().numpy()
    cov_np = cov_trace.cpu().squeeze().numpy()
    u_np = u_trace.cpu().squeeze().numpy()

    T = mean_np.shape[0] - 1

    # Common Axis Limits (calculated from full trajectory)
    x_min, x_max = np.min(mean_np[:, 0]), np.max(mean_np[:, 0])
    y_min, y_max = np.min(mean_np[:, 1]), np.max(mean_np[:, 1])
    
    # Expand limits to include environment
    for lane in env.lane_markings:
        x_min = min(x_min, min(lane["x"]))
        x_max = max(x_max, max(lane["x"]))
        y_min = min(y_min, lane["y"])
        y_max = max(y_max, lane["y"])
    if env.goal:
        x_max = max(x_max, env.goal["x"][1])
        y_max = max(y_max, env.goal["y"][1])
    
    # Add margins
    x_lims = (x_min - 1.0, x_max + 1.0)
    y_lims = (y_min - 1.0, y_max + 1.0)

    # Helper to draw environment and state
    def draw_scene(ax, t_idx=None, title=None):
        if title:
            ax.set_title(title)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Draw Environment (Static)
        for lane in env.lane_markings:
            lx = lane["x"]
            ly = lane["y"]
            style = "--" if lane["style"] == "dashed" else "-"
            ax.plot(lx, [ly, ly], color=PALETTE["lane"]["stroke"], linestyle=style, linewidth=2, alpha=0.7)

        if env.goal:
            gx, gy = env.goal["x"], env.goal["y"]
            ax.add_patch(patches.Rectangle((gx[0], gy[0]), gx[1]-gx[0], gy[1]-gy[0], fill=False, edgecolor=PALETTE["goal"]["stroke"], linestyle="-", linewidth=2, label="Goal"))

        for region in env.visit_regions:
            vx, vy = region["x"], region["y"]
            ax.add_patch(patches.Rectangle((vx[0], vy[0]), vx[1]-vx[0], vy[1]-vy[0], facecolor=PALETTE["visit"]["fill"], edgecolor=PALETTE["visit"]["stroke"], alpha=0.6))

        for obs in env.obstacles:
            ox, oy = obs["x"], obs["y"]
            ax.add_patch(patches.Rectangle((ox[0], oy[0]), ox[1]-ox[0], oy[1]-oy[0], facecolor=PALETTE["obs_static"]["fill"], edgecolor=PALETTE["obs_static"]["stroke"], alpha=0.5, hatch="//"))

        for obs in env.circle_obstacles:
            ax.add_patch(patches.Circle(obs["center"], obs["radius"], facecolor=PALETTE["obs_static"]["fill"], edgecolor=PALETTE["obs_static"]["stroke"], alpha=0.5, hatch="//"))

        # Draw Moving Obstacles
        for obs in env.moving_obstacles:
            xt = obs["x_traj"]
            yt = obs["y_traj"]
            if isinstance(xt, torch.Tensor): xt = xt.detach().cpu().numpy()
            if isinstance(yt, torch.Tensor): yt = yt.detach().cpu().numpy()
            
            if t_idx is None:
                # Full Trajectory View: Show path and start/end
                ax.plot(xt, yt, color=PALETTE["obs_moving"]["stroke"], linestyle="--", alpha=0.4, label="Moving Obs Path")
                # Draw a few snapshots along the path
                step = max(1, len(xt) // 5)
                for k in range(0, len(xt), step):
                    cx, cy = xt[k], yt[k]
                    w, h = obs["width"], obs["height"]
                    ax.add_patch(patches.Rectangle((cx - w/2, cy - h/2), w, h, facecolor=PALETTE["obs_moving"]["fill"], edgecolor=PALETTE["obs_moving"]["stroke"], alpha=0.3))
            else:
                # Snapshot View: Show path faint, current position solid
                ax.plot(xt, yt, color=PALETTE["obs_moving"]["stroke"], linestyle="--", alpha=0.15)
                idx = min(t_idx, len(xt) - 1)
                cx, cy = xt[idx], yt[idx]
                w, h = obs["width"], obs["height"]
                ax.add_patch(patches.Rectangle((cx - w/2, cy - h/2), w, h, facecolor=PALETTE["obs_moving"]["fill"], edgecolor=PALETTE["obs_moving"]["stroke"], alpha=0.5, label="Moving Obs"))

        # Draw Ego Trajectory
        if t_idx is None:
            # Full Trajectory
            ax.plot(mean_np[:, 0], mean_np[:, 1], color=PALETTE["ego"]["stroke"], linestyle="-", linewidth=2, alpha=0.8, label="Trajectory", zorder=25)
            # Draw Covariance Ellipses (Subsampled for clarity)
            for t in range(0, T + 1, 5):
                pos_cov = cov_np[t, :2, :2]
                plot_covariance_ellipse(ax, mean_np[t, :2], pos_cov, facecolor=PALETTE["ego"]["fill"], edgecolor=PALETTE["ego"]["stroke"], alpha=0.1, zorder=15)
        else:
            # Snapshot Trajectory (History)
            ax.plot(mean_np[:t_idx+1, 0], mean_np[:t_idx+1, 1], color=PALETTE["ego"]["stroke"], linestyle="-", linewidth=2, alpha=0.8, label="Trajectory")
            
            # Draw Ego Vehicle at t_idx
            cx, cy = mean_np[t_idx, 0], mean_np[t_idx, 1]
            if robot_dims:
                l, w = robot_dims
                # Calculate heading
                if t_idx < T:
                    dx = mean_np[t_idx+1, 0] - cx
                    dy = mean_np[t_idx+1, 1] - cy
                else:
                    dx = cx - mean_np[t_idx-1, 0]
                    dy = cy - mean_np[t_idx-1, 1]
                theta = np.degrees(np.arctan2(dy, dx))
                
                rect = patches.Rectangle((0,0), l, w, facecolor=PALETTE["ego"]["fill"], edgecolor=PALETTE["ego"]["stroke"], alpha=0.7, label="Ego")
                t_start = transforms.Affine2D().translate(-l/2, -w/2).rotate_deg(theta).translate(cx, cy)
                rect.set_transform(t_start + ax.transData)
                ax.add_patch(rect)
            else:
                ax.plot(cx, cy, marker="o", color=PALETTE["ego"]["stroke"], markersize=8, label="Ego")

            # Draw Covariance at t_idx
            pos_cov = cov_np[t_idx, :2, :2]
            plot_covariance_ellipse(ax, mean_np[t_idx, :2], pos_cov, facecolor=PALETTE["ego"]["fill"], edgecolor=PALETTE["ego"]["stroke"], alpha=0.6, zorder=20)

        # Legend (only if requested or first time, handled by caller mostly, but we can dedupe)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=8)

    # --- Layout Logic ---
    if layout == "snapshots":
        fig = plt.figure(figsize=(24, 14))
        # Grid Layout: 
        # Rows 0-1: Left=Full Trajectory (2x2 space), Right=4 Snapshots (2x2)
        # Row 2: Metrics
        gs = fig.add_gridspec(3, 4)

        # Full Trajectory
        ax_full = fig.add_subplot(gs[0:2, 0:2])
        draw_scene(ax_full, t_idx=None, title="Full Trajectory Realization")

        # Snapshots
        snapshot_indices = np.linspace(0, T, 4, dtype=int)
        for i, t_idx in enumerate(snapshot_indices):
            row = i // 2
            col = 2 + (i % 2)
            ax = fig.add_subplot(gs[row, col])
            draw_scene(ax, t_idx=t_idx, title=f"Step {t_idx}")

        # Metrics Locations
        if p_sat_trace is not None:
            ax_sat = fig.add_subplot(gs[2, 0])
            ax_loss = fig.add_subplot(gs[2, 1])
            ax_ctrl = fig.add_subplot(gs[2, 2:])
        else:
            ax_sat = None
            ax_loss = fig.add_subplot(gs[2, 0:2])
            ax_ctrl = fig.add_subplot(gs[2, 2:])

    else:
        # Default Layout (Single Shot / Standard)
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])

        # Full Trajectory (Left Column)
        ax_full = fig.add_subplot(gs[:, 0])
        draw_scene(ax_full, t_idx=None, title="Trajectory & Beliefs")

        # Metrics Locations (Right Column)
        if p_sat_trace is not None:
            ax_sat = fig.add_subplot(gs[0, 1])
            ax_loss = None # Share space or skip
            ax_ctrl = fig.add_subplot(gs[1, 1])
        else:
            ax_sat = None
            ax_loss = fig.add_subplot(gs[0, 1])
            ax_ctrl = fig.add_subplot(gs[1, 1])

    # --- Plot Metrics ---
    # Satisfaction Plot
    if ax_sat is not None and p_sat_trace is not None:
        ax_sat.plot(p_sat_trace, color=PALETTE["goal"]["stroke"], marker="o", linewidth=2, markersize=4)
        ax_sat.set_title("MPC: Satisfaction Probability per Step")
        ax_sat.set_xlabel("Simulation Step")
        ax_sat.set_ylabel("P(Sat)")
        ax_sat.grid(True, alpha=0.5)

    # Loss Plot
    if ax_loss is not None:
        if history is not None:
            label = "Step" if p_sat_trace is not None else "Iteration"
            title = "Final Loss per Step" if p_sat_trace is not None else "Optimization Convergence"
            ax_loss.plot(history, color=PALETTE["lane"]["stroke"], linewidth=2)
            ax_loss.set_title(title)
            ax_loss.set_xlabel(label)
            ax_loss.set_ylabel("Loss J")
            ax_loss.grid(True, alpha=0.5)
        else:
            ax_loss.text(0.5, 0.5, "No Loss History", ha="center")

    # Control Plot
    if ax_ctrl is not None:
        time_steps = np.arange(T)
        ax_ctrl.plot(time_steps, u_np[:, 0], color=PALETTE["obs_static"]["stroke"], linestyle="--", label="$u_x$")
        ax_ctrl.plot(time_steps, u_np[:, 1], color=PALETTE["ego"]["stroke"], linestyle="--", label="$u_y$")
        ax_ctrl.set_title("Control Inputs")
        ax_ctrl.set_xlabel("Time Step")
        ax_ctrl.set_ylabel("Control Output")
        ax_ctrl.legend()
        ax_ctrl.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()
