from asyncio import run
import numpy as np
import torch
from models.dynamics import (
    GaussianBelief,
    linear_system,
    piecewise_signal,
    sinusoidial_input,
)
from pdstl.base import BeliefTrajectory
from pdstl.operators import GreaterThan, Always
from visualization.robustness import plot_stl_formula_bounds, plot_piecewise_stl
from utils import skip_run
from planning.runners import run_single_shot, run_mpc


# HELPERS
def create_belief_trajectory(mean_trace, var_trace, confidence_level=1.0):
    """Create belief trajectory from mean and variance traces."""
    mean = torch.tensor(mean_trace, dtype=torch.float32).reshape(1, -1, 1)
    var = torch.tensor(var_trace, dtype=torch.float32).reshape(1, -1, 1)

    beliefs = []
    for i in range(len(mean_trace)):
        m = mean[:, i : i + 1, :]
        v = var[:, i : i + 1, :]
        beliefs.append(GaussianBelief(m, v, confidence_level=confidence_level))

    return BeliefTrajectory(beliefs)


def to_steps(interval_sec, t):
    """Convert time interval [a,b] in seconds to steps."""
    dt = float(t[1] - t[0])
    a = int(round(interval_sec[0] / dt))
    b = np.inf if np.isinf(interval_sec[1]) else int(round(interval_sec[1] / dt))
    return [a, b]


def print_trace(name, trace, time, step=1):
    """Print trace values."""
    if isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()
    trace = np.asarray(trace)

    if trace.ndim == 3:
        trace = trace[0]

    print(f"\n{name}:")
    print(f"{'t':>4} {'time':>6} {'lower':>10} {'upper':>10}")
    print("-" * 34)

    for i in range(0, len(time), step):
        print(f"{i:>4} {time[i]:>6.1f} {trace[i, 0]:>10.4f} {trace[i, 1]:>10.4f}")


# =============================================================================
# EXAMPLE 1: Always Operator
# =============================================================================

with skip_run("skip", "Example 1: Always") as check, check():
    t = np.linspace(0, 10, 100)
    mean, var = linear_system(
        a=0.01, b=1.0, g=2.0, q=2.5, mu=50, P=0.15, t=t, control_func=sinusoidial_input
    )

    beliefs = create_belief_trajectory(mean, var, confidence_level=1.0)

    threshold = 50.0
    interval_sec = [1, 2]
    interval_steps = to_steps(interval_sec, t)

    phi = GreaterThan(threshold)
    spec = Always(phi, interval=interval_steps)

    pred_trace = phi(beliefs)
    oper_trace = spec(beliefs)

    print(f"\n{'=' * 50}")
    print(f"Example 1: □[{interval_sec[0]}, {interval_sec[1]}](x ≥ {threshold})")
    print(f"{'=' * 50}")
    print_trace("Predicate", pred_trace, t, step=10)
    print_trace("Always", oper_trace, t, step=10)

    plot_stl_formula_bounds(
        t,
        oper_trace,
        mean_trace=mean,
        var_trace=var,
        predicate_trace=pred_trace,
        thresholds=threshold,
        formula_str=f"□[{interval_sec[0]}, {interval_sec[1]}](x ≥ {threshold})",
        interval=interval_steps,
        operator_type="always",
    )

# =============================================================================
# EXAMPLE 2: Discrete Piecewise Signal
# =============================================================================

with skip_run("run", "Example 3: Piecewise") as check, check():
    t, mean, var = piecewise_signal()

    print(f"\n{'=' * 50}")
    print("Example 3: Discrete Piecewise Signal")
    print(f"{'=' * 50}")
    print("\nSignal values:")
    print(f"{'t':<4} {'μ(t)':<8} {'σ²(t)':<8} {'σ(t)':<8}")
    print("-" * 32)
    for i in range(len(t)):
        print(f"{i:<4} {mean[i]:<8.0f} {var[i]:<8.0f} {np.sqrt(var[i]):<8.2f}")

    beliefs = create_belief_trajectory(mean, var, confidence_level=1.0)

    threshold = 50.0
    interval_steps = [1, 2]

    phi = GreaterThan(threshold)
    spec_always = Always(phi, interval=interval_steps)

    pred_trace = phi(beliefs)
    always_trace = spec_always(beliefs)

    print_trace("Predicate P(x ≥ 50)", pred_trace, t)
    print_trace("Always □[1,2]P(φ)", always_trace, t)

    plot_piecewise_stl(
        t,
        always_trace,
        mean_trace=mean,
        var_trace=var,
        predicate_trace=pred_trace,
        thresholds=threshold,
        formula_str=f"□[1, 2](x ≥ {threshold})",
        interval=interval_steps,
        operator_type="always",
    )
# =============================================================================
# EXAMPLE 3: Single Shot Motion Planning
# =============================================================================

with skip_run("skip", "Example 3: Single Shot Motion Planning") as check, check():
    run_single_shot(max_iterations=1000)


# =============================================================================
# EXAMPLE 4: MPC Receding Horizon Motion Planning
# =============================================================================
with skip_run("skip", "Example 4: MPC Receding Horizon") as check, check():
    run_mpc()
