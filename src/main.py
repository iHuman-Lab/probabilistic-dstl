import numpy as np

from models.dynamics import linear_system, piecewise_signal, sinusoidial_input
from pdstl.operators import Always, GreaterThan
from planning.runners import (
    run_lane_change,
    run_lane_change_aggressive,
    run_mpc,
    run_single_shot,
)
from utils import create_belief_trajectory, load_config, skip_run, to_steps
from visualization.robustness import plot_piecewise_stl, plot_stl_formula_bounds

_demos = load_config("configs/stl_demos.yaml")

# =============================================================================
# EXAMPLE 1: Always Operator
# =============================================================================

with skip_run("skip", "Example 1: Always") as check, check():
    d = _demos["example1"]
    t = np.linspace(0, d["t_end"], d["n_steps"])
    mean, var = linear_system(
        a=d["a"], b=d["b"], g=d["g"], q=d["q"],
        mu=d["mu"], P=d["P"], t=t, control_func=sinusoidial_input,
    )

    beliefs = create_belief_trajectory(mean, var)
    interval_steps = to_steps(d["interval_sec"], t)

    phi = GreaterThan(d["threshold"])
    spec = Always(phi, interval=interval_steps)

    pred_trace = phi(beliefs)
    oper_trace = spec(beliefs)

    plot_stl_formula_bounds(
        t, oper_trace,
        mean_trace=mean, var_trace=var, predicate_trace=pred_trace,
        thresholds=d["threshold"],
        formula_str=f"□[{d['interval_sec'][0]}, {d['interval_sec'][1]}](x ≥ {d['threshold']})",
        interval=interval_steps, operator_type="always",
    )

# =============================================================================
# EXAMPLE 2: Discrete Piecewise Signal
# =============================================================================

with skip_run("skip", "Example 2: Piecewise") as check, check():
    d = _demos["example2"]
    t, mean, var = piecewise_signal()

    beliefs = create_belief_trajectory(mean, var)

    phi = GreaterThan(d["threshold"])
    spec_always = Always(phi, interval=d["interval_steps"])

    pred_trace = phi(beliefs)
    always_trace = spec_always(beliefs)

    plot_piecewise_stl(
        t, always_trace,
        mean_trace=mean, var_trace=var, predicate_trace=pred_trace,
        thresholds=d["threshold"],
        formula_str=f"□[{d['interval_steps'][0]}, {d['interval_steps'][1]}](x ≥ {d['threshold']})",
        interval=d["interval_steps"], operator_type="always",
    )

# =============================================================================
# EXAMPLE 3: Single Shot Motion Planning
# =============================================================================

with skip_run("skip", "Example 3: Single Shot Motion Planning") as check, check():
    run_single_shot(max_iterations=500, force_run=True)

# =============================================================================
# EXAMPLE 4: MPC Receding Horizon Motion Planning
# =============================================================================

with skip_run("skip", "Example 4: MPC Receding Horizon") as check, check():
    run_mpc()

# =============================================================================
# EXAMPLE 5: Lane Change with Moving Obstacle
# =============================================================================

with skip_run("skip", "Example 5: Lane Change") as check, check():
    run_lane_change()

# =============================================================================
# EXAMPLE 6: Aggressive Lane Change
# =============================================================================

with skip_run("skip", "Example 6: Aggressive Lane Change") as check, check():
    run_lane_change_aggressive()
