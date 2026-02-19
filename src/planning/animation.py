import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation

# Tableau 10 colors
PALETTE = {
    "ego":        {"fill": "#1f77b4", "stroke": "#1f77b4"},  # Blue
    "plan":       {"fill": "#ff7f0e", "stroke": "#ff7f0e"},  # Orange
    "visit":      {"fill": "#2ca02c", "stroke": "#2ca02c"},  # Green
    "obs_static": {"fill": "#d62728", "stroke": "#d62728"},  # Red
    "obs_moving": {"fill": "#d62728", "stroke": "#d62728"},  # Red
    "lane":       {"fill": "#7f7f7f", "stroke": "#7f7f7f"},  # Gray
    "goal":       {"fill": "#bcbd22", "stroke": "#bcbd22"},  # Olive
}


def animate_results(
    mean_trace, cov_trace, env, filename="trajectory.gif", plan_traces=None, step=1, dt=0.2, robot_dims=None, title="Motion Planning", bounds=None
):
    """
    Animates the robot's trajectory with covariance ellipses.

    Args:
        mean_trace: Tensor [Batch, Time, Dim]
        cov_trace: Tensor [Batch, Time, Dim, Dim]
        env: Environment object
        filename: Output filename (e.g., 'trajectory.gif')
        plan_traces: List of Tensors (optional), each [Batch, Horizon, Dim] representing the sliding window plan at that step.
        step: Frame skip step size (default 1). Increase to speed up animation generation.
        dt: Time step size (default 0.2)
        robot_dims: Tuple (length, width) for drawing the ego vehicle as a rectangle.
    """
    mean_np = mean_trace.detach().cpu().squeeze().numpy()
    cov_np = cov_trace.detach().cpu().squeeze().numpy()

    T = mean_np.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))

    if bounds:
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
    else:
        ax.set_xlim(-5, 15)
        ax.set_ylim(-4, 8)
        
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # Draw Lane Markings
    for lane in env.lane_markings:
        lx = lane["x"]
        ly = lane["y"]
        style = "--" if lane["style"] == "dashed" else "-"
        ax.plot(lx, [ly, ly], color=PALETTE["lane"]["stroke"], linestyle=style, linewidth=2, alpha=0.7)

    # Draw Goal
    if env.goal:
        gx = env.goal["x"]
        gy = env.goal["y"]
        ax.add_patch(
            patches.Rectangle(
                (gx[0], gy[0]),
                gx[1] - gx[0],
                gy[1] - gy[0],
                facecolor=PALETTE["goal"]["fill"],
                edgecolor=PALETTE["goal"]["stroke"],
                alpha=0.3,
                label="Goal Lane",
                zorder=5,
            )
        )

    # Draw Visit Regions
    for region in env.visit_regions:
        vx = region["x"]
        vy = region["y"]
        ax.add_patch(
            patches.Rectangle(
                (vx[0], vy[0]),
                vx[1] - vx[0],
                vy[1] - vy[0],
                facecolor=PALETTE["visit"]["fill"],
                edgecolor=PALETTE["visit"]["stroke"],
                alpha=0.3,
                label="Visit Region",
                zorder=5,
            )
        )

    # Draw Obstacles
    for obs in env.obstacles:
        ox = obs["x"]
        oy = obs["y"]
        ax.add_patch(
            patches.Rectangle(
                (ox[0], oy[0]),
                ox[1] - ox[0],
                oy[1] - oy[0],
                facecolor=PALETTE["obs_static"]["fill"],
                edgecolor=PALETTE["obs_static"]["stroke"],
                alpha=0.8,
                hatch="//",
                label="Obstacle",
                zorder=5,
            )
        )

    # Draw Circle Obstacles
    for obs in env.circle_obstacles:
        c = patches.Circle(
            obs["center"], 
            obs["radius"], 
            facecolor=PALETTE["obs_static"]["fill"], 
            edgecolor=PALETTE["obs_static"]["stroke"],
            alpha=0.5,
            label="Obstacle",
            hatch="//",
            zorder=5
        )
        ax.add_patch(c)

    # Moving Obstacles Patches
    moving_patches = []
    for obs in env.moving_obstacles:
        # Initialize at t=0
        w, h = obs["width"], obs["height"]
        rect = patches.Rectangle((0,0), w, h, facecolor=PALETTE["obs_moving"]["fill"], edgecolor=PALETTE["obs_moving"]["stroke"], alpha=0.5, label="Moving Obs", zorder=6)
        ax.add_patch(rect)
        moving_patches.append((rect, obs))

    if robot_dims:
        robot_rect = patches.Rectangle((0, 0), robot_dims[0], robot_dims[1], facecolor=PALETTE["ego"]["fill"], edgecolor=PALETTE["ego"]["stroke"], alpha=0.7, label="Ego Vehicle", zorder=10)
        ax.add_patch(robot_rect)
        robot_dot = None
    else:
        (robot_dot,) = ax.plot([], [], color=PALETTE["ego"]["stroke"], marker="o", markersize=8, label="Robot Mean", zorder=10)

    (trail,) = ax.plot([], [], color=PALETTE["ego"]["stroke"], linewidth=2, alpha=0.6, zorder=9)
    
    (plan_line,) = ax.plot(
        [],
        [],
        color=PALETTE["plan"]["stroke"],
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Planned Horizon",
    )

    ellipse = patches.Ellipse(
        (0, 0), width=0, height=0, angle=0, facecolor=PALETTE["ego"]["fill"], edgecolor=PALETTE["ego"]["stroke"], alpha=0.25, zorder=8
    )
    ax.add_patch(ellipse)

    # Time text
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        if robot_dot:
            robot_dot.set_data([], [])
        if robot_dims:
            robot_rect.set_visible(False)
        trail.set_data([], [])
        plan_line.set_data([], [])
        ellipse.set_width(0)
        ellipse.set_height(0)
        time_text.set_text("")
        
        actors = [trail, ellipse, time_text, plan_line]
        if robot_dot: actors.append(robot_dot)
        if robot_dims: actors.append(robot_rect)
        actors.extend([p[0] for p in moving_patches])
        return actors

    def update(frame):
        x, y = mean_np[frame, 0], mean_np[frame, 1]

        # Camera Follow (Zoom/Snap)
        if bounds is None: # Only auto-follow if bounds not explicitly set
            ax.set_xlim(x - 8.0, x + 12.0)
            ax.set_ylim(-4.0, 8.0)

        if robot_dot:
            robot_dot.set_data([x], [y])
        
        if robot_dims:
            robot_rect.set_visible(True)
            if frame < len(mean_np) - 1:
                dx = mean_np[frame+1, 0] - x
                dy = mean_np[frame+1, 1] - y
            else:
                dx = x - mean_np[frame-1, 0]
                dy = y - mean_np[frame-1, 1]
            
            theta = np.degrees(np.arctan2(dy, dx))
            
            # Update Rectangle Transform (Rotate around center)
            w, h = robot_dims[0], robot_dims[1]
            t = transforms.Affine2D().translate(-w/2, -h/2).rotate_deg(theta).translate(x, y)
            robot_rect.set_transform(t + ax.transData)

        trail.set_data(mean_np[: frame + 1, 0], mean_np[: frame + 1, 1])

        # Update Covariance Ellipse
        cov = cov_np[frame, :2, :2]
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Calculate angle and size
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 2.45 * np.sqrt(vals)

        ellipse.set_center((x, y))
        ellipse.set_width(width)
        ellipse.set_height(height)
        ellipse.set_angle(theta)

        time_text.set_text(f"Time Step: {frame}")

        # Update Sliding Window Plan
        if plan_traces is not None and frame < len(plan_traces):
            plan = plan_traces[frame].detach().cpu().squeeze().numpy()
            plan_line.set_data(plan[:, 0], plan[:, 1])
        else:
            plan_line.set_data([], [])

        # Update Moving Obstacles
        for rect, obs in moving_patches:
            # Ensure frame is within bounds of trajectory
            idx = min(frame, len(obs["x_traj"]) - 1)
            cx = obs["x_traj"][idx]
            cy = obs["y_traj"][idx]
            rect.set_xy((cx - obs["width"]/2, cy - obs["height"]/2))

        actors = [trail, ellipse, time_text, plan_line]
        if robot_dot: actors.append(robot_dot)
        if robot_dims: actors.append(robot_rect)
        actors.extend([p[0] for p in moving_patches])
        return actors

    frames = range(0, T, step)
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=False, interval=100
    )

    # Save
    if filename:
        print(f"Saving animation to {filename}...")
        try:
            if filename.endswith(".gif"):
                ani.save(filename, writer="pillow", fps=20)
            else:
                ani.save(filename, writer="ffmpeg", fps=20)
            print("Animation saved successfully.")
        except Exception as e:
            print(f"Warning: Could not save animation (ffmpeg/pillow issue?): {e}")
            print("Displaying plot instead.")
            plt.show()
    else:
        plt.show()
