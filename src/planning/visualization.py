import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_covariance_ellipse(ax, mean, cov, k=2.45, color="blue", alpha=0.2):
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
        xy=mean, width=width, height=height, angle=theta, color=color, alpha=alpha
    )
    ax.add_patch(ellipse)


def visualize_results(
    mean_trace, cov_trace, u_trace, env, history=None, p_sat_trace=None
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

    fig = plt.figure(figsize=(16, 10))

    # Determine layout based on whether we have MPC satisfaction data
    if p_sat_trace is not None:
        gs = fig.add_gridspec(3, 2)  # 3 Rows
        ax_sat = fig.add_subplot(gs[0, 1])
        ax_loss = fig.add_subplot(gs[1, 1])
        ax_ctrl = fig.add_subplot(gs[2, 1])
    else:
        gs = fig.add_gridspec(2, 2)  # 2 Rows (Skip Sat Pane)
        ax_sat = None
        ax_loss = fig.add_subplot(gs[0, 1])
        ax_ctrl = fig.add_subplot(gs[1, 1])

    # --- 1. Spatial Trajectory & Beliefs ---
    ax1 = fig.add_subplot(gs[:, 0])  # Spans all rows on the left
    ax1.set_title("Robot Trajectory & Beliefs")

    # Draw Goal
    if env.goal:
        gx = env.goal["x"]
        gy = env.goal["y"]
        ax1.add_patch(
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
        ax1.add_patch(
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
        ax1.add_patch(
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
        c = plt.Circle(
            obs["center"], obs["radius"], color="red", alpha=0.5, label="Obstacle", zorder=5
        )
        ax1.add_patch(c)

    # Draw Mean Path
    ax1.plot(
        mean_np[:, 0],
        mean_np[:, 1],
        "b-o",
        label="Mean Path",
        markersize=3,
        linewidth=1,
    )

    # Draw Covariance Ellipses (every 5 steps for clarity)
    for t in range(0, T + 1, 5):
        # Extract 2x2 position covariance (works for Single or Double Integrator)
        # Assuming state is [x, y, ...] so pos is indices 0,1
        pos_cov = cov_np[t, :2, :2]
        plot_covariance_ellipse(ax1, mean_np[t, :2], pos_cov)

    ax1.set_xlim(-2, 12)
    ax1.set_ylim(-2, 12)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # --- 2. Satisfaction Probability (MPC) ---
    if ax_sat is not None:
        ax_sat.plot(p_sat_trace, "g-o", linewidth=2, markersize=4)
        ax_sat.set_title("MPC: Satisfaction Probability per Step")
        ax_sat.set_xlabel("Simulation Step")
        ax_sat.set_ylabel("P(Sat)")
        ax_sat.grid(True, alpha=0.5)

    # --- 3. Optimization Loss ---
    if history is not None:
        # For MPC, this is Loss per Step. For Single Shot, it's Loss per Iteration.
        label = "Step" if p_sat_trace is not None else "Iteration"
        title = (
            "Final Loss per Step"
            if p_sat_trace is not None
            else "Optimization Convergence"
        )

        ax_loss.plot(history, "k-", linewidth=2)
        ax_loss.set_title(title)
        ax_loss.set_xlabel(label)
        ax_loss.set_ylabel("Loss J")
        ax_loss.grid(True, alpha=0.5)
    else:
        ax_loss.text(0.5, 0.5, "No Loss History", ha="center")

    # --- 4. Control Inputs ---
    time_steps = np.arange(T)
    ax_ctrl.plot(time_steps, u_np[:, 0], "r--", label="$u_x$")
    ax_ctrl.plot(time_steps, u_np[:, 1], "b--", label="$u_y$")
    ax_ctrl.set_title("Control Inputs")
    ax_ctrl.set_xlabel("Time Step")
    ax_ctrl.set_ylabel("Control Output")
    ax_ctrl.legend()
    ax_ctrl.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()
