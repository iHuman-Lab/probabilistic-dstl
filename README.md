# Probabilistic STL Motion Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Gradient-based motion planner for uncertain dynamical systems using Probabilistic Signal Temporal Logic (STL). Given a Gaussian belief over the initial state, the planner finds a control sequence that satisfies a formal STL specification with high probability — avoiding obstacles, reaching goals, and respecting temporal constraints.

## Overview

Classical motion planners treat the robot's state as known exactly. This system instead propagates a **Gaussian belief** (mean + covariance) forward through the dynamics, and optimises controls so that the resulting belief trajectory satisfies an STL specification with probability ≥ α.

The planner supports:
- **Single-shot** planning: optimise a full open-loop trajectory in one pass
- **MPC (receding-horizon)**: replan at every step as the robot executes
- **Lane-change**: MPC with a moving obstacle and road constraints

## Project Structure

```
src/
├── pdstl/              # Probabilistic STL library (operators, predicates, propagation)
├── planning/
│   ├── dynamics.py     # Single- and double-integrator belief dynamics
│   ├── environment.py  # Workspace geometry + STL specification builder
│   ├── planner.py      # Gradient-descent optimiser (Planner.solve)
│   ├── runners.py      # Thin scenario wiring (load config → solve → visualise)
│   └── log_utils.py    # Logging helpers
├── visualization/
│   ├── planning.py     # Static trajectory and environment plots
│   ├── live_plots.py   # Live MPC figures that update step-by-step
│   ├── animation.py    # GIF/video export
│   └── robustness.py   # STL robustness bound plots
└── main.py             # Runnable examples (toggle with skip_run)

configs/
├── planning.yaml           # Optimiser defaults (weights, convergence)
└── scenarios/              # Per-scenario YAML (dynamics, horizon, environment)
```

## Installation

```bash
conda create -n pdstl python=3.10
conda activate pdstl
pip install -r requirements.txt
```

## Quick Start

Edit `src/main.py` to select which examples to run using the `skip_run` toggle, then:

```bash
python src/main.py
```

Each scenario saves its result to `saved_data/` and writes an animated GIF. Re-running loads the cached result unless `force_run=True`.

## Examples

| Example | Description | Runner |
|---------|-------------|--------|
| Single Shot | Full trajectory optimisation, static obstacles | `run_single_shot()` |
| MPC | Receding-horizon replanning to a goal | `run_mpc()` |
| Lane Change | MPC with a moving vehicle, road geometry | `run_lane_change()` |
| Aggressive Lane Change | Higher-speed variant | `run_lane_change_aggressive()` |

## Key Concepts

**Probabilistic STL** — An STL formula φ is evaluated on a belief trajectory. Instead of asking "does the trajectory satisfy φ?", the planner asks "does P(φ) ≥ α?" where α is a user-set probability threshold (default 0.95).

**Belief dynamics** — The dynamics propagate a Gaussian (μ, Σ) forward. Process noise grows the covariance; the planner accounts for this uncertainty when checking obstacle avoidance and goal-reaching constraints.

**Gradient descent** — Controls are parameterised as unconstrained values passed through `tanh` to enforce bounds. The loss combines STL satisfaction probability, control effort, smoothness, and heuristic terms (goal distance, obstacle repulsion). PyTorch autodiff computes gradients through the full belief rollout.

## Dependencies

- [PyTorch](https://pytorch.org) — differentiable optimisation
- [NumPy](https://numpy.org) — numerical operations
- [Matplotlib](https://matplotlib.org) — visualisation and animation
- [PyYAML](https://pyyaml.org) — scenario configuration

## License

MIT
