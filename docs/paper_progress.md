# Probabilistic STL Motion Planning — Paper Progress Notes

## Overview

This project develops a **gradient-based motion planning framework** under uncertainty using **Probabilistic Signal Temporal Logic (P-STL)**. The core idea: instead of planning over deterministic trajectories, we plan over *Gaussian belief trajectories* and optimize control inputs to maximize the probability of satisfying a temporal logic specification.

---

## Problem Formulation

### System Model
The robot/agent state evolves as a **Gaussian belief**:

- **State distribution at time t**: `x(t) ~ N(μ(t), Σ(t))`
- Two dynamics models are implemented:
  - **SingleIntegrator**: `x_{t+1} = x_t + u_t · dt`, covariance grows as `Σ_{t+1} = Σ_t + Q` (position-controlled, 2D)
  - **DoubleIntegrator**: `x_{t+1} = A·x_t + B·u_t`, with full linear covariance propagation `Σ_{t+1} = A·Σ_t·Aᵀ + Q` (acceleration-controlled, 4D state: [px, py, vx, vy])
- Controls are bounded via smooth squashing: `u = u_max · tanh(v)` where `v` is the unconstrained optimization variable

### Probabilistic STL (P-STL)
Classical STL predicates `μ(x) ≥ 0` are replaced with **probability measures**:

- **Predicate robustness**: `ρ(φ, x(t)) → P(φ satisfied | x(t) ~ N(μ, Σ))`
- Computed using the Gaussian CDF `Φ(z)` analytically
- The framework returns **[lower, upper] probability bounds** `[B, T, 2]` at each timestep

**Operators** (in `src/pdstl/operators.py`):
- `Always(φ, [a,b])` — `□[a,b]φ`: min over the interval
- `Eventually(φ, [a,b])` — `◇[a,b]φ`: max over the interval
- `And(φ₁, φ₂)` — conjunction: element-wise min
- `Or(φ₁, φ₂)` — disjunction: element-wise max
- `Negation(φ)` — negation: `1 - p`
- `GreaterThan(threshold)` — scalar predicate with conservative probability bounds

Smooth approximations use log-sum-exp (`Minish`/`Maxish`) for gradient flow.

---

## Probabilistic Predicates

Defined in `src/planning/environment.py`:

### Goal / Region Predicates
**`RectangularGoalPredicate`** — `P(x ∈ R)` for a rectangular region `[x_min, x_max] × [y_min, y_max]`:
```
P_goal(t) = min( P(x ≥ x_min), P(x ≤ x_max), P(y ≥ y_min), P(y ≤ y_max) )
```
Uses Gaussian CDF on each face independently; intersection via min.

### Obstacle Predicates
**`RectangularObstaclePredicate`** — `P(x ∉ O)`:
```
P_safe(t) = max( P(x ≤ x_min), P(x ≥ x_max), P(y ≤ y_min), P(y ≥ y_max) )
```
Safe if outside *any* face; union via max.

**`CircularObstaclePredicate`** — `P(||x - c|| > r)`:
- Projects Gaussian uncertainty along the radial direction
- `σ_proj² = dᵀ Σ d` where `d` is the unit vector from center to mean
- `P_safe = 1 - Φ(r; dist, σ_proj²)`

**`MovingRectangularObstaclePredicate`** — same as rectangular but obstacle bounds shift over time `[T+1]` following a pre-defined trajectory.

---

## Optimization Algorithm

### Objective Function (`ProbabilisticSTLPlanner.solve`)

At each gradient step, the total loss is:

```
J = w_u · ||u||² + w_du · ||Δu||² + w_φ · (-log(P_sat + ε)) + w_dist · dist_goal² + w_obs · Σ relu(r - d)² + w_visit · Σ min_t(dist_visit²)
```

| Term | Weight | Purpose |
|------|--------|---------|
| Control effort | `w_u = 0.1` | Minimize energy |
| Smoothness | `w_du = 0.1` | Penalize jerky control |
| STL satisfaction | `w_φ = 10.0` | Main objective: maximize `P(φ)` via `-log(P)` |
| Goal guidance | `w_dist = 5.0` | Heuristic gradient toward goal when P is low |
| Obstacle repulsion | `w_obs = 5.0` | Smooth repulsion from obstacle centers |
| Visit region pull | `w_visit = 5.0` | Attracts trajectory to visit regions (for `◇`) |

**Optimizer**: Adam, `lr = 0.05`

**Convergence**: Early stop when `P(sat) ≥ α = 0.95` for 50 consecutive iterations, or loss plateau (`|ΔJ| < 1e-4`).

**Warm-start**: MPC uses the previous solution shifted by one step as the initial guess (inverse tanh mapping back to `v` space).

---

## Experiments / Examples

### Example 1 & 2: STL Semantics Verification
- Demonstrates the `Always` operator on a 1D Gaussian signal from a linear dynamical system with sinusoidal input
- Example 2 uses a discrete piecewise signal to validate predicate computation
- Visualized with probability bound plots over time

### Example 3: Single-Shot Motion Planning
- **Environment**: 2D workspace with static rectangular obstacles, a goal region, and workspace bounds
- **Dynamics**: `SingleIntegrator`, `dt=0.2`, `T=30` steps
- **Spec**: `φ = Always(safe) ∧ Eventually(goal)`
- Runs gradient descent for up to 1000 iterations; result cached in `saved_data/single_shot.pt`
- **Outputs**: trajectory plot, covariance ellipses, control inputs, loss/P(sat) history — saved as PDFs

### Example 4: MPC Receding Horizon Planning
- Uses a **receding horizon** approach: solve → execute 1 step → re-solve with new initial state
- Warm-starts each solve from the previous solution
- Demonstrates closed-loop replanning under uncertainty

### Example 5: Lane Change with Moving Obstacle (Normal)
- **Environment**: 3-lane road scenario with lane markings
- **Moving obstacle**: vehicle in the adjacent lane following a constant-velocity trajectory
- **Spec**: `Always(safe from moving obs) ∧ Eventually(target lane) ∧ Always(road bounds)`
- Uses `DoubleIntegrator` dynamics for more realistic vehicle motion
- **Visualization**: animated GIF + PDF figures with covariance ellipses and obstacle positions over time

### Example 6: Aggressive Lane Change
- Same structure as Example 5 but with **tighter timing constraints** or faster obstacle
- Demonstrates the planner's ability to handle higher-risk maneuvers under uncertainty
- Comparison figures: `lane_change_compare_*.pdf`

---

## Software Architecture

```
src/
├── pdstl/                   # Core P-STL library
│   ├── base.py              # Belief, BeliefTrajectory, OnlineBeliefTrajectory
│   ├── operators.py         # STL_Formula, Always, Eventually, And, Or, GreaterThan, etc.
│   └── propagate.py         # Belief propagation utilities
├── planning/
│   ├── dynamics.py          # SingleIntegrator, DoubleIntegrator
│   ├── environment.py       # Environment, all probabilistic predicates
│   ├── planner.py           # ProbabilisticSTLPlanner (optimization loop)
│   ├── runners.py           # run_single_shot, run_mpc, run_lane_change, run_lane_change_aggressive
│   ├── visualization.py     # visualize_results, visualize_lane_change, plot_covariance_ellipse
│   └── animation.py         # animate_results (GIF generation)
├── models/
│   └── dynamics.py          # GaussianBelief, linear_system, piecewise_signal, sinusoidal_input
├── visualization/
│   └── robustness.py        # plot_stl_formula_bounds, plot_piecewise_stl
└── main.py                  # Top-level experiment runner (skip_run gating)
```

**Key design choices**:
- `STL_Formula` inherits `torch.nn.Module` — all operations are differentiable, enabling backprop through the STL evaluator
- `BeliefTrajectory` wraps a list of `Belief` objects; `TorchGaussianBelief` provides a differentiable wrapper for the planner's internal use
- `skip_run` context manager gates which examples run without modifying code
- Results are cached with `.pt` files to avoid re-running expensive optimizations

---

## Key Results (Qualitative)

- Single-shot planner reliably finds collision-free, goal-reaching trajectories with probabilistic guarantees
- MPC receding-horizon extends this to closed-loop replanning
- Lane-change scenarios show the planner avoids a moving vehicle by timing the lane change appropriately, with uncertainty ellipses growing as the horizon extends
- Aggressive vs. normal lane change comparison highlights the trade-off between aggressiveness and safety margin

---

## Outstanding Items / Future Work

- [ ] Formal quantitative evaluation: success rate over random seeds/initial conditions
- [ ] Comparison baseline (e.g., deterministic STL planner, sampling-based)
- [ ] Scalability analysis: timing vs. state/control dimension
- [ ] Tighter probabilistic bounds (currently lower=upper for most predicates — true interval arithmetic)
- [ ] Noise-aware replanning: update `x0_cov` from actual observation uncertainty in MPC loop
- [ ] Paper figures: finalize trajectory/control/metrics PDFs for submission
