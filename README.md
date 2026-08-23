# Probabilistic dSTL

> *"I am 94.7% sure the robot won't crash. Probably."*

**pdSTL** is a Python library for evaluating and optimizing [Signal Temporal Logic (STL)](https://en.wikipedia.org/wiki/Signal_temporal_logic) specifications over **probabilistic (Gaussian belief) trajectories** — because the real world is uncertain and your specs shouldn't pretend otherwise.

Instead of asking *"does the robot always stay in the lane?"*, pdSTL asks the more honest question: ***"what is the probability that the robot always stays in the lane?"***

---

## What is this?

STL lets you write temporal requirements like:

```
□[1s, 5s] (x ≥ 50)      # "x must always be ≥ 50 between 1 and 5 seconds"
◇[0, 10] (goal reached)  # "reach the goal within 10 seconds"
```

Classical STL checks these against **deterministic** signals. But real robots live in an uncertain world — sensors are noisy, dynamics are approximate, wind exists.

**pdSTL** propagates Gaussian uncertainty through STL operators, giving you a **satisfaction probability** for every spec at every timestep. You can then use that probability as an objective to optimize trajectories that are *robustly safe* under uncertainty.

---

## Features

- **Probabilistic STL evaluation** — compute P(spec satisfied) for Gaussian belief trajectories
- **Gradient-based motion planning** — maximize satisfaction probability via PyTorch autograd
- **MPC (Receding Horizon)** — roll the planner forward in real time
- **Lane change scenarios** — dodge moving obstacles while staying in the lane, stochastically
- **Pluggable belief system** — bring your own `Belief` subclass; Gaussian is just the default

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/iHuman-Lab/probabilistic-dstl
cd probabilistic-dstl
pip install -e .

# 2. Run the demos (toggle skip/run flags in src/main.py)
python src/main.py
```

### Evaluate a spec over a belief trajectory

```python
import numpy as np
from models.dynamics import linear_system, sinusoidial_input
from pdstl.operators import Always, GreaterThan
from utils import create_belief_trajectory, to_steps

t = np.linspace(0, 10, 100)
mean, var = linear_system(
    a=0.01, b=1.0, g=2.0, q=2.5, mu=50.0, P=0.15, t=t, control_func=sinusoidial_input
)

beliefs = create_belief_trajectory(mean, var)

phi = GreaterThan(threshold=50.0)
spec = Always(phi, interval=to_steps([1, 2], t))

p_sat = spec(beliefs)  # probability of satisfaction at each timestep
```

---

## Examples

| # | Scenario | Description |
|---|----------|-------------|
| 1 | Always operator | Evaluate □[1,2](x ≥ 50) on a linear stochastic system |
| 2 | Piecewise signal | Same spec, but on a discrete piecewise signal |
| 3 | Single-shot planning | Optimize a full trajectory satisfying an STL spec |
| 4 | MPC receding horizon | Re-plan every step as the robot moves |
| 5 | Lane change | Merge lanes while a car moves into your path |
| 6 | Aggressive lane change | Same, but faster (and scarier) |

Toggle which examples run by changing `"skip"` → `"run"` in [src/main.py](src/main.py).

---

## Project Structure

```
src/
├── pdstl/          # Core: Belief base classes, STL operators, propagation
├── models/         # Dynamical systems (linear, double integrator, etc.)
├── planning/       # Gradient-based planner, MPC runner, environments
├── visualization/  # Robustness plots, animations, live MPC callbacks
├── baselines/      # Deterministic STL baseline for comparison
└── main.py         # Demo entry point
configs/            # YAML configs for scenarios and hyperparameters
data/               # Video recordings of hardware experiments
saved_data/         # Cached optimization results (.pt files)
```

---

## Device Configuration

The library defaults to **CPU** (some machines expose CUDA even when it can't initialize). To use a GPU:

```bash
PDSTL_DEVICE=cuda python src/main.py
# or
PDSTL_USE_CUDA=1 python src/main.py
```

---

## Requirements

- Python 3.8+
- PyTorch (for autograd-based planning)
- NumPy, PyYAML, python-dotenv

```bash
pip install -r requirements.txt
```

---

## Citation

If this library is useful in your research, please consider citing the associated work (details forthcoming).

---

## License

MIT — do whatever you want, but don't blame us if the robot crashes. (We did say *probabilistic*.)
