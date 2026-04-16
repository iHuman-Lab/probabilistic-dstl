"""
Empirical O(T) complexity benchmark for the Probabilistic STL planner.

Measures wall-clock time for a single forward-backward pass as a function
of the planning horizon T, and plots the result with a linear fit.

Run from the src/ directory:
    python -m planning.benchmark_complexity
"""

import os
import time

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from planning.dynamics import SingleIntegrator
from planning.planner import TorchGaussianBelief
from planning.runners import build_environment, load_scenario_config
from pdstl.base import BeliefTrajectory


def build_spec_and_env(T, device):
    """Build the same environment and spec used in run_single_shot, parameterised by T."""
    cfg, _ = load_scenario_config("configs/scenarios/single_shot.yaml")
    env = build_environment(cfg, device)
    return env, env.get_specification(T), cfg


def single_iteration_time(T, device, n_warmup=3, n_trials=10):
    """Return mean wall-clock time (seconds) for one forward+backward pass at horizon T."""
    _, spec, cfg = build_spec_and_env(T, device)
    _, planner_cfg = load_scenario_config("configs/scenarios/single_shot.yaml")
    dyn = SingleIntegrator(dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"], device=device)

    x0_mean = torch.tensor(cfg["x0_mean"], device=device)
    x0_cov = torch.eye(len(cfg["x0_mean"]), device=device) * cfg["x0_cov_scale"]

    v_params = nn.Parameter(
        torch.randn(T, 2, device=device) * 0.1 + torch.tensor([0.5, 0.0], device=device),
        requires_grad=True,
    )
    optimizer = optim.Adam([v_params], lr=planner_cfg["lr"])

    def one_pass():
        optimizer.zero_grad()

        # --- Dynamics rollout ---
        mean_trace, cov_trace = dyn(v_params, x0_mean, x0_cov)

        # --- Wrap for STL evaluation ---
        beliefs = [
            TorchGaussianBelief(mean_trace[:, t, :], cov_trace[:, t])
            for t in range(T + 1)
        ]
        traj = BeliefTrajectory(beliefs)

        # --- STL evaluation ---
        stl_trace = spec(traj)
        p_all = stl_trace[0, 0, 0]

        # --- Loss ---
        u_seq = dyn.bound_control(v_params)
        loss_u = torch.sum(u_seq ** 2)
        u_diff = u_seq[1:] - u_seq[:-1]
        loss_du = torch.sum(u_diff ** 2)
        loss_phi = -torch.log(p_all + 1e-4)
        J = planner_cfg["w_u"] * loss_u + planner_cfg["w_du"] * loss_du + planner_cfg["w_phi"] * loss_phi

        # --- Backward ---
        J.backward()
        optimizer.step()

        return J.item()

    # Warmup (JIT / caching effects)
    for _ in range(n_warmup):
        one_pass()

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_pass()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), float(np.std(times))


def run_benchmark(T_values=None, device_str="cpu"):
    if T_values is None:
        T_values = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]

    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"{'T':>6}  {'mean (ms)':>12}  {'std (ms)':>10}")
    print("-" * 35)

    means_ms, stds_ms = [], []
    for T in T_values:
        mu, sigma = single_iteration_time(T, device)
        means_ms.append(mu * 1e3)
        stds_ms.append(sigma * 1e3)
        print(f"{T:>6}  {mu*1e3:>12.3f}  {sigma*1e3:>10.3f}")

    return np.array(T_values), np.array(means_ms), np.array(stds_ms)


def plot_results(T_values, means_ms, stds_ms, save_path="benchmark_complexity.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Linear fit ---
    coeffs = np.polyfit(T_values, means_ms, 1)
    slope, intercept = coeffs
    T_fit = np.linspace(T_values[0], T_values[-1], 200)
    y_fit = slope * T_fit + intercept

    # --- Left panel: linear scale ---
    ax = axes[0]
    ax.errorbar(
        T_values, means_ms, yerr=stds_ms,
        fmt="o", color="#2563EB", capsize=4, label="Measured",
    )
    ax.plot(T_fit, y_fit, "--", color="#DC2626",
            label=f"Linear fit  (slope={slope:.3f} ms/step)")
    ax.set_xlabel("Planning horizon $T$ (steps)", fontsize=12)
    ax.set_ylabel("Wall-clock time per iteration (ms)", fontsize=12)
    ax.set_title("Computation time vs. $T$ (linear scale)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    # --- Right panel: log-log scale ---
    ax = axes[1]
    ax.errorbar(
        T_values, means_ms, yerr=stds_ms,
        fmt="o", color="#2563EB", capsize=4, label="Measured",
    )
    log_T = np.log10(T_values)
    log_t = np.log10(means_ms)
    log_coeffs = np.polyfit(log_T, log_t, 1)
    exponent = log_coeffs[0]
    T_log_fit = np.logspace(np.log10(T_values[0]), np.log10(T_values[-1]), 200)
    y_log_fit = 10 ** np.polyval(log_coeffs, np.log10(T_log_fit))
    ax.loglog(T_log_fit, y_log_fit, "--", color="#DC2626",
              label=f"Power-law fit  (exponent={exponent:.2f})")
    ax.set_xlabel("Planning horizon $T$ (steps)", fontsize=12)
    ax.set_ylabel("Wall-clock time per iteration (ms)", fontsize=12)
    ax.set_title("Computation time vs. $T$ (log-log scale)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"\nFigure saved to: {save_path}")
    if os.environ.get("DISPLAY"):
        plt.show()

    print(f"\nLinear fit:  time ≈ {slope:.4f}·T + {intercept:.4f}  (ms)")
    print(f"Log-log fit exponent: {exponent:.3f}  (1.0 = perfect O(T))")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--T", nargs="+", type=int,
        default=[10, 25, 50, 75, 100, 150, 200, 300, 400, 500],
        help="Planning horizons to benchmark",
    )
    parser.add_argument("--save", default="benchmark_complexity.pdf")
    args = parser.parse_args()

    T_values, means_ms, stds_ms = run_benchmark(args.T, args.device)
    plot_results(T_values, means_ms, stds_ms, save_path=args.save)
