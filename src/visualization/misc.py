import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

# =====================================
# CONFIGURATION
# =====================================

PALETTE = {
    "ego": {"fill": "#1f77b4", "stroke": "#1f77b4"},
    "obs": {"fill": "#C0BFBF", "stroke": "#060505"},
    "goal": {"fill": "#98df8a", "stroke": "#2ca02c"},
    "lane": {"fill": "#d9d9d9"},
    "road": "#F2F2F7",
}

np.random.seed(4)

T = 50
N_DET = 30
N_PROB = 30

# =====================================
# ENVIRONMENT GEOMETRY
# =====================================
# =====================================
# NATURAL ENVIRONMENT GEOMETRY
# =====================================

# =====================================
# HIGH-DENSITY ENVIRONMENT
# =====================================

obstacle_specs = [
    # Boundary Curbs (Fills the top and bottom edges)
    (0, 2.8, 10, 0.6),  # Top boundary wall
    (0, -3.4, 10, 0.6),  # Bottom boundary wall
    # Staggered "Slalom" Blocks
    (1.5, 1.10, 3, 1.7),  # Block 1: Top-mid
    (3.5, -2.8, 1.5, 2.0),  # Block 2: Bottom
    (6.0, 0.8, 1.2, 2.0),  # Block 3: Top-mid
    (6.5, -2.5, 3.5, 1.0),  # Block 4: Bottom-right
]


# NEW: Circle obstacles (x, y, radius)
# Added circles at the beginning (low x values)
circle_specs = [
    (2.5, -0.5, 0.6),  # Top start
    (8.3, 0.65, 0.7),
]

lane_specs = [
    (0, 3.0, 10, 0.2),
    (0, -3.2, 10, 0.2),
]

goal_spec = (9.5, -0.8, 0.5, 1.6)

# =====================================
# MEAN PATH
# =====================================

t = np.linspace(0, 1, T)
mean_path = np.vstack([10 * t, 0.75 * np.sin(2 * np.pi * t)]).T

# =====================================
# TRAJECTORY GENERATION
# =====================================


def generate_rollouts(mean_path, n_rollouts, lateral_scale):
    T = mean_path.shape[0]
    tau = np.linspace(0, 1, T)
    rollouts = []

    for _ in range(n_rollouts):
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.normal(0, lateral_scale)

        growth = tau**0.5
        lateral_offset = amplitude * growth * np.sin(3 * np.pi * tau + phase)

        traj = mean_path.copy()
        traj[:, 1] += lateral_offset
        rollouts.append(traj)

    return rollouts


mc_det = generate_rollouts(mean_path, N_DET, 0.6)
mc_prob = generate_rollouts(mean_path, N_PROB, 0.25)

# =====================================
# COLLISION CHECK
# =====================================


def check_collision(traj, obstacle_specs, circle_specs):
    mask = np.zeros(len(traj), dtype=bool)
    # Rectangle collisions
    for ox, oy, w, h in obstacle_specs:
        inside = (
            (traj[:, 0] >= ox)
            & (traj[:, 0] <= ox + w)
            & (traj[:, 1] >= oy)
            & (traj[:, 1] <= oy + h)
        )
        mask |= inside

    # Circle collisions (Distance formula)
    for cx, cy, r in circle_specs:
        dist = np.sqrt((traj[:, 0] - cx) ** 2 + (traj[:, 1] - cy) ** 2)
        mask |= dist <= r

    return mask


# =====================================
# AXIS SETUP
# =====================================


def setup_axis(ax):
    ax.set_ylim(-3.4, 3.4)
    ax.set_xlim(0, 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Rectangular Obstacles
    for x, y, w, h in obstacle_specs:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=PALETTE["obs"]["fill"],
                edgecolor=PALETTE["obs"]["stroke"],
                linewidth=1.2,
                alpha=0.95,
                # hatch="//",
            ),
        )

    # Circular Obstacles
    for cx, cy, r in circle_specs:
        ax.add_patch(
            Circle(
                (cx, cy),
                r,
                facecolor=PALETTE["obs"]["fill"],
                edgecolor=PALETTE["obs"]["stroke"],
                linewidth=1.2,
                alpha=0.95,
            )
        )

    # Goal region
    gx, gy, gw, gh = goal_spec
    ax.add_patch(
        Rectangle(
            (gx, gy),
            gw,
            gh,
            facecolor=PALETTE["goal"]["fill"],
            edgecolor=PALETTE["goal"]["stroke"],
            linewidth=1.5,
            alpha=0.9,
        )
    )


# =====================================
# PLOTTING
# =====================================

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

for ax in axes:
    setup_axis(ax)

# -------- Deterministic --------
ax = axes[0]

for traj in mc_det:
    ax.plot(traj[:, 0], traj[:, 1], color="#d62728", alpha=0.25, linewidth=1)

    # Updated to include circle_specs
    collision_mask = check_collision(traj, obstacle_specs, circle_specs)
    ax.scatter(
        traj[collision_mask, 0],
        traj[collision_mask, 1],
        marker="x",
        color="#d62728",
        s=18,
    )

ax.plot(mean_path[:, 0], mean_path[:, 1], color="black", linewidth=3)
ax.scatter(mean_path[0, 0], mean_path[0, 1], s=60, color="black", zorder=5)
ax.annotate(
    "Start",
    (mean_path[0, 0], mean_path[0, 1]),
    textcoords="offset points",
    xytext=(20, -15),
    ha="center",
    fontsize=14,
)
ax.set_title("Deterministic STL", fontsize=11)

# -------- Probabilistic --------
ax = axes[1]

for traj in mc_prob:
    ax.plot(
        traj[:, 0], traj[:, 1], color=PALETTE["ego"]["stroke"], alpha=0.25, linewidth=1
    )

ax.plot(mean_path[:, 0], mean_path[:, 1], color="black", linewidth=3)
ax.scatter(mean_path[0, 0], mean_path[0, 1], s=60, color="black", zorder=5)
ax.annotate(
    "Start",
    (mean_path[0, 0], mean_path[0, 1]),
    textcoords="offset points",
    xytext=(20, -15),
    ha="center",
    fontsize=14,
)

# Covariance ellipses
tau = np.linspace(0, 1, T)
for i in range(0, T, 3):
    growth = tau[i] ** 0.75
    width = 0.15 + 1.1 * growth
    ax.add_patch(
        Ellipse(
            xy=mean_path[i],
            width=width,
            height=width,
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.25,
            linewidth=0.5,
        )
    )

ax.set_title("Probabilistic dSTL", fontsize=11)


plt.tight_layout()
plt.savefig("comparison.png", dpi=400)
plt.show()
