# Probabilistic STL Motion Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Gradient-based motion planner for uncertain dynamical systems with formal correctness guarantees. The planner optimises a control sequence so that the resulting **Gaussian belief trajectory** satisfies a **Probabilistic Signal Temporal Logic (STL)** specification with probability ≥ α — handling obstacle avoidance, goal reaching, and temporal constraints all within one differentiable optimisation problem.



---


```

---

## Scenarios

| Scenario | Dynamics | Mode | Description |
|----------|----------|------|-------------|
| **Single Shot** | Single integrator | Open-loop | Optimise one full trajectory past static obstacles to a goal |
| **MPC** | Single integrator | Receding-horizon | Replan at every step; terminate when goal is reached |
| **Lane Change** | Double integrator | Receding-horizon | Change lanes around a moving vehicle on a two-lane road |
| **Aggressive Lane Change** | Double integrator | Receding-horizon | Higher-speed lane change with tighter margins |

All scenarios produce a live animation during execution and save an animated GIF and cached result.

---

## Project Structure

```
src/
├── pdstl/                  # Probabilistic STL library
│   ├── base.py             # Belief, BeliefTrajectory abstractions
│   └── operators.py        # GreaterThan, Always, Eventually, And, Or, Until, Negation
│
├── planning/
│   ├── dynamics.py         # SingleIntegrator, DoubleIntegrator — propagate (μ, Σ) forward
│   ├── environment.py      # Workspace geometry + STL spec builder (obstacles, goal, lanes)
│   ├── planner.py          # Gradient-descent optimiser; Planner.solve() is the entry point
│   ├── runners.py          # Thin scenario wiring: load config → solve → visualise
│   └── log_utils.py        # Structured logging helpers
│
├── visualization/
│   ├── planning.py         # Static trajectory and environment plots
│   ├── live_plots.py       # Live figures that update at each MPC step
│   ├── animation.py        # Animated GIF / video export
│   └── robustness.py       # STL robustness bound plots
│
├── models/
│   └── dynamics.py         # GaussianBelief, test signal generators
│
└── main.py                 # Six runnable examples (toggle with skip_run context manager)

configs/
├── planning.yaml           # Optimiser defaults (weights, convergence thresholds)
└── scenarios/              # Per-scenario YAML: dynamics, horizon, obstacles, goal
    ├── single_shot.yaml
    ├── mpc.yaml
    ├── lane_change.yaml
    └── lane_change_aggressive.yaml
```

---

## Installation

```bash
conda create -n pdstl python=3.10
conda activate pdstl
pip install -r requirements.txt

```

---

## Running Examples

Open `src/main.py` and use the `skip_run` context manager to enable the examples you want:

```python
with skip_run("run", "Example 3: Single Shot Motion Planning"):
    run_single_shot()

with skip_run("run", "Example 4: MPC Receding Horizon"):
    run_mpc()

with skip_run("run", "Example 5: Lane Change"):
    run_lane_change()
```

Then run:

```bash
python src/main.py
```


---

## Dependencies

- [PyTorch](https://pytorch.org) — differentiable belief propagation and optimisation
- [NumPy](https://numpy.org) — numerical operations
- [Matplotlib](https://matplotlib.org) — static plots, live MPC figures, GIF export
- [PyYAML](https://pyyaml.org) — scenario configuration

---


