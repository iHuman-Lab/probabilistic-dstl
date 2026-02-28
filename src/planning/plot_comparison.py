"""
Side-by-side trajectory plots comparing:
  - Deterministic baseline (q_std≈0): mean path only, η certificate
  - pdSTL (ours): mean path + 2σ belief tube, P∈[lower, upper] certificate

Run after Example 7 has produced:
  saved_data/single_shot.pt
  saved_data/single_shot_det.pt
  saved_data/single_shot_comparison.pt

Usage:
  cd src
  python planning/plot_comparison.py
"""

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

RESULTS_DIR = "saved_data"
device = torch.device("cpu")

ENV_CONFIG = {
    "obstacles": [
        {"x": (3.0, 6.0), "y": (0.0, 3.0)},
        {"x": (3.0, 6.0), "y": (4.0, 7.0)},
        {"x": (3.0, 6.0), "y": (7.5, 10.0)},
    ],
    "goal": {"x": (10.0, 12.0), "y": (2.0, 4.0)},
    "visit_regions": [{"x": (8.0, 10.0), "y": (7.0, 9.0)}],
    "bounds": {"x": (0.0, 12.0), "y": (0.0, 10.5)},
}

DT = 0.2
X0 = np.array([0.0, 5.0])


def draw_env(ax):
    for i, obs in enumerate(ENV_CONFIG["obstacles"]):
        x0, x1 = obs["x"]
        y0, y1 = obs["y"]
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1,
                edgecolor="darkred",
                facecolor="lightcoral",
                alpha=0.7,
                label="Obstacle" if i == 0 else "_nolegend_",
            )
        )

    g = ENV_CONFIG["goal"]
    ax.add_patch(
        patches.Rectangle(
            (g["x"][0], g["y"][0]),
            g["x"][1] - g["x"][0],
            g["y"][1] - g["y"][0],
            linewidth=2,
            edgecolor="darkgreen",
            facecolor="lightgreen",
            alpha=0.5,
            label="Goal",
        )
    )

    for i, v in enumerate(ENV_CONFIG["visit_regions"]):
        ax.add_patch(
            patches.Rectangle(
                (v["x"][0], v["y"][0]),
                v["x"][1] - v["x"][0],
                v["y"][1] - v["y"][0],
                linewidth=2,
                edgecolor="darkblue",
                facecolor="lightblue",
                alpha=0.4,
                label="Visit Region" if i == 0 else "_nolegend_",
            )
        )

    b = ENV_CONFIG["bounds"]
    ax.set_xlim(b["x"][0] - 0.3, b["x"][1] + 0.3)
    ax.set_ylim(b["y"][0] - 0.3, b["y"][1] + 0.3)


def draw_mc_overlay(ax, u_seq, n_display=20, true_q_std=0.03, seed=42):
    """Overlay stochastic sample trajectories."""
    T = u_seq.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(n_display):
        xt = X0.copy()
        traj = [xt]
        noise = rng.standard_normal((T, 2)) * true_q_std * np.sqrt(DT)
        for t in range(T):
            xt = xt + u_seq[t] * DT + noise[t]
            traj.append(xt.copy())
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], "-", alpha=0.12, linewidth=0.7, color="gray")


def plot_comparison():
    prob_path = os.path.join(RESULTS_DIR, "single_shot.pt")
    det_path = os.path.join(RESULTS_DIR, "single_shot_det.pt")
    cmp_path = os.path.join(RESULTS_DIR, "single_shot_comparison.pt")

    assert os.path.exists(prob_path), f"Missing {prob_path} — run Example 3 first"
    assert os.path.exists(det_path), f"Missing {det_path} — run Example 7 first"
    assert os.path.exists(cmp_path), f"Missing {cmp_path} — run Example 7 first"

    prob_data = torch.load(prob_path, map_location=device, weights_only=False)
    det_data = torch.load(det_path, map_location=device, weights_only=False)
    cmp_data = torch.load(cmp_path, map_location=device, weights_only=False)

    # Trajectories — squeeze batch dim
    mean_prob = prob_data["mean_trace"].squeeze(0).numpy()  # [T+1, 2]
    cov_prob = prob_data["cov_trace"].squeeze(0).numpy()    # [T+1, 2, 2]
    mean_det = det_data["mean_trace"].squeeze(0).numpy()    # [T+1, 2]
    u_prob = prob_data["u_trace"].squeeze(0).numpy()        # [T, 2]
    u_det = det_data["u_trace"].squeeze(0).numpy()          # [T, 2]

    # Metrics from comparison run
    results = cmp_data["results"]
    n_trials = cmp_data["n_trials"]
    eta_det = results["deterministic"]["planned_metric"]
    p_lower = results["probabilistic"]["planned_metric"]
    p_upper = results["probabilistic"].get("planned_metric_upper", p_lower)
    det_success = results["deterministic"]["success_rate"]
    prob_success = results["probabilistic"]["success_rate"]

    # 2σ tube from covariance diagonal
    std_x = np.sqrt(np.maximum(cov_prob[:, 0, 0], 0))
    std_y = np.sqrt(np.maximum(cov_prob[:, 1, 1], 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Deterministic ───────────────────────────────────────────────
    ax = axes[0]
    draw_env(ax)
    draw_mc_overlay(ax, u_det)
    ax.plot(mean_det[:, 0], mean_det[:, 1], "-", color="firebrick", linewidth=2,
            label="Mean trajectory")
    ax.plot(*mean_det[0], "go", markersize=10, label="Start")
    ax.plot(*mean_det[-1], "r*", markersize=14, label="End")
    ax.set_title(
        f"Deterministic Baseline (q_std≈0)\n"
        f"$\\eta = {eta_det:.3f}$ [planning]   "
        f"MC success: {det_success*100:.1f}%  (N={n_trials})",
        fontsize=11,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Right: pdSTL ──────────────────────────────────────────────────────
    ax = axes[1]
    draw_env(ax)
    draw_mc_overlay(ax, u_prob)
    ax.fill_between(
        mean_prob[:, 0],
        mean_prob[:, 1] - 2 * std_y,
        mean_prob[:, 1] + 2 * std_y,
        alpha=0.2,
        color="royalblue",
        label="2σ belief tube",
    )
    ax.plot(mean_prob[:, 0], mean_prob[:, 1], "-", color="royalblue", linewidth=2,
            label="Mean trajectory")
    ax.plot(*mean_prob[0], "go", markersize=10, label="Start")
    ax.plot(*mean_prob[-1], "r*", markersize=14, label="End")
    ax.set_title(
        f"pdSTL (Ours)\n"
        f"$P \\in [{p_lower:.3f},\\, {p_upper:.3f}]$ [planning]   "
        f"MC success: {prob_success*100:.1f}%  (N={n_trials})",
        fontsize=11,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Deterministic Baseline vs. pdSTL: Planned Trajectories with MC Samples",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "stlcg_comparison_trajectories.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
