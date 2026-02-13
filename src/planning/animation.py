import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


def animate_results(
    mean_trace, cov_trace, env, filename="trajectory.gif", plan_traces=None, step=1, dt=0.2
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
    """
    # Convert tensors to numpy (ensure they are on CPU)
    # mean_trace shape: [1, T, 2] -> [T, 2]
    mean_np = mean_trace.detach().cpu().squeeze().numpy()
    cov_np = cov_trace.detach().cpu().squeeze().numpy()

    T = mean_np.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Setup bounds (match visualization.py)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Probabilistic STL Motion Planning Animation")

    # Draw Goal
    if env.goal:
        gx = env.goal["x"]
        gy = env.goal["y"]
        ax.add_patch(
            patches.Rectangle(
                (gx[0], gy[0]),
                gx[1] - gx[0],
                gy[1] - gy[0],
                color="green",
                alpha=0.3,
                label="Goal",
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
                color="yellow",
                alpha=0.5,
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
                color="red",
                alpha=0.5,
                label="Obstacle",
                zorder=5,
            )
        )

    # Draw Circle Obstacles
    for obs in env.circle_obstacles:
        c = patches.Circle(
            obs["center"], obs["radius"], color="red", alpha=0.5, label="Obstacle", zorder=5
        )
        ax.add_patch(c)

    # Dynamic Elements
    # Robot position dot
    (robot_dot,) = ax.plot([], [], "bo", markersize=8, label="Robot Mean", zorder=10)
    # Trajectory trail
    (trail,) = ax.plot([], [], "b-", linewidth=1, alpha=0.5, zorder=9)
    # Planned Sliding Window (Ghost)
    (plan_line,) = ax.plot(
        [],
        [],
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Planned Horizon",
    )

    # Covariance Ellipse (initially hidden)
    ellipse = patches.Ellipse(
        (0, 0), width=0, height=0, angle=0, color="blue", alpha=0.3, zorder=8
    )
    ax.add_patch(ellipse)

    # Time text
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        robot_dot.set_data([], [])
        trail.set_data([], [])
        plan_line.set_data([], [])
        ellipse.set_width(0)
        ellipse.set_height(0)
        time_text.set_text("")
        return robot_dot, trail, ellipse, time_text, plan_line

    def update(frame):
        # Update Mean
        x, y = mean_np[frame, 0], mean_np[frame, 1]
        robot_dot.set_data([x], [y])

        # Update Trail
        trail.set_data(mean_np[: frame + 1, 0], mean_np[: frame + 1, 1])

        # Update Covariance Ellipse
        # Extract 2x2 position covariance
        cov = cov_np[frame, :2, :2]

        # Eigendecomposition
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Calculate angle and size (95% confidence)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 2.45 * np.sqrt(vals)

        ellipse.set_center((x, y))
        ellipse.set_width(width)
        ellipse.set_height(height)
        ellipse.set_angle(theta)

        time_text.set_text(f"Time Step: {frame}")

        # Update Sliding Window Plan
        if plan_traces is not None and frame < len(plan_traces):
            # plan_traces[frame] is likely [1, H, 2] or [H, 2]
            plan = plan_traces[frame].detach().cpu().squeeze().numpy()
            plan_line.set_data(plan[:, 0], plan[:, 1])
        else:
            plan_line.set_data([], [])

        return robot_dot, trail, ellipse, time_text, plan_line

    # Create Animation
    # Use range(0, T, step) to skip frames for speed
    frames = range(0, T, step)
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=False, interval=50
    )

    # Save or Show
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
