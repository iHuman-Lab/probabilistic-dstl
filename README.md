# Probabilistic differentiable STL

pdSTL combines probability bounds for atomic events into Boolean and temporal
intervals. A belief supplies `probability_bounds(event) -> [B, 2]`; pdSTL does
not assume a distribution. Gaussian beliefs give exact axis and half-space
probabilities, Boolean operators use Fréchet bounds, and `Always` / `Eventually`
take the minimum / maximum over their window.

The result for a spec φ is the stochastic robustness interval
**[ρ̲_φ, ρ̄_φ]** (lower and upper). `beta=None` evaluates it exactly;
`beta > 0` gives a smooth outer interval (smooth ρ̲_φ ≤ exact ρ̲_φ) for
gradients.

## Code layout

| Piece | Role |
|---|---|
| `src/pdstl/` | the semantics: predicates, Boolean and temporal operators |
| `src/baselines/det_stl.py` | independent STLCG-style recurrent mean robustness and formula conversion |
| `src/models/` | dynamics, Gaussian beliefs, belief rollouts, trajectory sampling |
| `src/planning/environment.py` | builds environments and their pdSTL specs (reach–avoid, lane) |
| `src/planning/planner.py` | the optimization: one window, or receding horizon |
| `src/planning/runners.py` | shared glue: set up, `solve`, `monte_carlo`, save |
| `src/experiments/reach_avoid.py`, `lane.py` | one file per experiment: `build(cfg)` and `run(config_path)` |
| `src/visualization/` | `figures`, `animation`, `live_plots` |
| `configs/scenarios/<experiment>/*.yaml` | the numbers for each case |
| `src/main.py` | runs the experiments; flip `"run"`/`"skip"` per block |

## Planning

`Planner.optimize_window` is direct shooting: from an initial control
sequence u⁽⁰⁾ it takes Adam steps on

    −w_φ · smooth ρ̲_φ(u) + w_u Σ‖u‖² + w_du Σ‖Δu‖²

At every iteration, it evaluates the exact ρ̲_φ and retains the iterate with
the highest value; a later iterate wins an exact tie. It runs all
`max_iters` updates, since the smooth bound tightens as β anneals toward
`beta_end` and the exact bound can dip before it recovers. `alpha` only
reports whether the selected plan meets the required satisfaction level. The
initial guess's exact and smooth ρ̲_φ are recorded (`initial_hard_lower`,
`initial_smooth_lower`), so each run prints what the optimization improved.

Optional deterministic guidance adds
`w_margin * max(0, margin_target - rho_aux)` to that objective, where
`rho_aux` is STLCG-style robustness of the same formula on the belief mean,
smoothed with log-sum-exp at `margin_beta` (default 10; `null` is hard
robustness). It is off by default (`w_margin: 0`); add `w_margin: 10` to a
scenario's `planner:` settings to enable it. It matters from poor initial
guesses (e.g. zero controls), where the exact lower bound is 0 and its
gradient vanishes: the hinge supplies a direction, then switches itself off
once the mean plan satisfies the formula. From a good initial guess it leaves
the selected plan unchanged. It shares the rollout, predicates, and temporal
windows, does not change pdSTL semantics or exact-score candidate selection,
and is guidance only, not a probability bound.

The recurrent baseline supports all current pdSTL operators, including
inclusive `Until`, nested finite windows, and suffix windows. Its public
traces use forward time and omit incomplete origins. Suffix reductions seed
each observed value once, avoiding duplicated terminal padding in smooth
robustness (including Until prefixes). Arbitrary probability-only predicates
need an explicit geometric auxiliary evaluator. For a standalone
deterministic comparison:

```python
from baselines.det_stl import from_pdstl

det_spec = from_pdstl(spec)  # raw mean robustness
rho = det_spec.robustness(prediction.belief_trajectory)
```

The recurrence and reductions are adapted from
[StanfordASL/STLCG](https://github.com/StanfordASL/stlcg), with its MIT notice
retained in the module. This adapter covers the STL grammar used here; it
does not reproduce upstream expression visualization, experimental AGM
robustness, `Then`, or integral extensions.

`Planner.run_receding_horizon` is the MPC loop used by the lane experiment:
plan a window, execute its first control, shift, and repeat until an outcome
(success, collision, road or speed violation, deadline missed, planning failure,
step limit).

## Reach–avoid

Each file in `configs/scenarios/reach_avoid/` is one case of the same problem.
Add, move or remove regions:

```yaml
bounds: {x_range: [0, 10], y_range: [0, 10]}
obstacles:
  - {x_range: [3, 5], y_range: [4, 6]}
goal: {x_range: [7, 8], y_range: [8, 9], interval: [1, 60]}
visit_regions:
  - dwell: 5
    any_of:
      - {name: t1, x_range: [1, 2], y_range: [6, 7]}
      - {name: t2, x_range: [7, 8], y_range: [4.5, 5.5]}
```

The task is G[1,H](bounds ∧ ¬obstacles) ∧ F[goal interval](goal) ∧, for each
visit region, F[interval] G[0,dwell](region). `interval` (in steps) and `name`
are optional; `any_of` lists alternatives, any one of which suffices (the goal
accepts it too). In code: `Environment.set_bounds`, `add_obstacle`, `set_goal`,
`add_visit_region`.

`src/experiments/reach_avoid.py` builds the environment and spec, starts the
optimizer on the shortest collision-free route (`shortest_route`, configured
by `route: {clearance, resolution}`; the clearance is kept small, 0.05 m, so
the route only avoids starting on an obstacle and pdSTL adds the
uncertainty-aware separation), optimizes, and then checks the plan
by **Monte Carlo**: it samples noisy trajectories under the plan's controls and
scores each with the same spec (with zero variance every atom is 0 or 1, so the
operators reduce to Boolean logic). It prints ρ̲_φ next to the empirical P(φ)
with a 95% interval, and the figure shows both.

## Lane change and merge

`configs/scenarios/lane/lane_change.yaml` and `lane_merge.yaml`: traffic `x0`
(position, m) and `speed` (m/s), ego `x0_mean: [x, y, vx, vy]`. Collision and
road checks use the full vehicle footprint plus safety margins. A merge
succeeds only when the target-lane dwell finishes inside the time window and
before the ramp end. Each planning window reports its exact [ρ̲_φ, ρ̄_φ].

The merge configs use a parallel on-ramp, `ramp: {gore_x, start_x, end_x}`:
no part of the ego may be in the main lane upstream of `gore_x` (the ramp is
a separate road there), the ramp lane is full width from `gore_x` to
`start_x` (50 m) and tapers to zero at `end_x` (90 m, about the 300 ft AASHTO
parallel-entrance taper). Without `gore_x` the ramp lane runs alongside the
main lane all the way upstream, as before.

Each window's formula is reach-and-stay: safe over the whole horizon, finish
the dwell in time, then remain in the target lane with lateral speed within
`task.settle_lateral_speed` (default 0.3 m/s) until the horizon end. Once the
dwell is done, windows drop the timing and ramp-end terms and only require
staying safe, in lane and settled. The closed loop keeps running for
`task.settle_seconds` (default 1.0 s) after the true dwell completes, still
monitoring truth, and reports `success` only after that settling period.

Lane simulation keeps physical truth separate from the planner's beliefs.
Initial ego and traffic states are sampled from the configured Gaussian
priors, except that true traffic starts with zero lateral speed, so cars keep
their lane instead of drifting across it (the belief prior on that speed is
unchanged). After each action, the simulator advances those true states with
process noise, generates independent noisy observations of `[x, y, vx, vy]`,
and applies a full-state Kalman update to the predicted beliefs. The first
planning window uses the prior; the first observation follows its action.

`observation.ego_std` and `observation.traffic_std` configure measurement
standard deviations as a positive scalar or four values (units m, m, m/s,
m/s). Both default to `0.1` per component. These are illustrative sensor
settings, independent of `q_std` and `traffic_q_std` (process noise).

`traffic_filter_accel_std` (m/s^2, default 0) adds white-noise longitudinal
acceleration to the traffic noise the Kalman filter assumes, so its estimate
follows a car that brakes without warning. The merge configs set it to the
brake size, 2.0. The true traffic and the planner's constant-velocity
prediction keep `traffic_q_std`.

The planner uses posterior means/covariances and estimates dwell progress
from the posterior mean. The physical monitor independently tracks true dwell
progress and evaluates success, collision, road and speed violations on truth.
Executed paths and vehicle drawings show truth; live belief ellipses remain
centered on the estimates. Saved `LaneState` records contain both.

Speed bounds constrain the nominal plan. Physical Gaussian disturbances are
not clipped: a sampled speed outside the bounds reports `speed_violation`.
This preserves the linear-Gaussian plant/estimator model. Each future planning
window remains an open-loop prediction, without anticipated measurement
updates; task-progress estimates are not a probabilistic history filter.

`lane.trial` uses a local generator seeded from the scenario, and accepts
explicit initial states, process disturbances, observation noise, or a
`torch.Generator` for reproducible trials without changing global RNG state.
Custom initial states must be `LaneState` records with truth and beliefs.

## Scope: what the numbers mean

1. **ρ̲_φ is a robustness, not P(φ).** `Always` takes the minimum over time of
   per-step bounds. That bounds each single step, not the whole trajectory, so
   ρ̲_φ can exceed the true probability that φ holds. On narrow passage,
   Monte Carlo gives P(φ) = 0.950, 95% interval
   [0.944, 0.956], against ρ̲_φ = 0.957. A sound probability
   bound over time (Fréchet: max(0, Σ lower − (n − 1))) exists but is much more
   conservative (0.55 there).
2. **ρ̄_φ is not a sound upper probability either:** `Eventually` takes the
   maximum over time for the upper value too. Read [ρ̲_φ, ρ̄_φ] as a
   robustness interval.
3. **Covariance does not depend on the controls** in the linear-Gaussian model,
   so the planner only moves the mean. Covariance steering is the natural
   extension.
4. **The start is outside the theory.** Far from the goal, ρ̲_φ is exactly 0
   (the Fréchet lower bound of a conjunction is 0 until its parts are likely),
   so a gradient method needs a starting path. The route only chooses where to
   start; ρ̲_φ and the Monte Carlo rate evaluate the plan that comes out.
5. **The lane first-window guesses are hand-shaped manoeuvres** (`warm_starts:`
   in the lane YAMLs), the same kind of choice as item 4. The first window is
   solved from each and the best plan seeds the loop; on a tie the earlier one
   wins, so the YAML order is the stated preference.
6. **Lane values are per window and open loop.** Success, collision and the
   other outcomes are sampled closed-loop executions, not guarantees.
7. **Reproducibility.** Reach–avoid is deterministic; lane and Monte Carlo are
   seeded from the YAML. Report numbers from CPU runs (`PDSTL_DEVICE=cpu`);
   CUDA can differ in the last digits.

## Other belief models

The planner, environment and formulas never see the distribution. A sample
(Monte Carlo) belief would add a `SampleBelief` whose bounds are the fraction
of particles in the event, widened by a finite-sample (e.g. Hoeffding) margin,
and a rollout that propagates particles with fixed noise draws and fills
`aux["mean_trace"]` and `aux["cov_trace"]` for the figures.

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

Set the desired block in `src/main.py` to `"run"` and the others to `"skip"`.
The on-ramp merge has one block per STL specification, each its own YAML:

| Block | YAML | Specification | What to look for |
| --- | --- | --- | --- |
| `LaneMerge` | `lane_merge.yaml` | A: safe, merge in the window, dwell, stay settled | merges; closes on the braking lead |
| `LaneHeadway` | `lane_headway.yaml` | B: A + gap to `lead` ≥ 22 m once touching the lane | same scene, keeps the gap |
| `LaneMergeOrYield` | `lane_merge_or_yield.yaml` | C: A's merge, or stop in the yield region | dense traffic: stops |
| `LaneMergeOrYieldGap` | `lane_merge_or_yield_gap.yaml` | C, same rule | one open gap: merges |
| `LaneBatch` | all four | several seeds, no plots | outcome and timing table |

A and B share their scene and seed, as do the two C scenes; only the rule (or
only the traffic) differs. In A/B the `lead` car brakes at 4 s (`brake:` in
the YAML); the planner is not told and replans from measurements. Each run
shows the run figure, a snapshot figure naming every car, and a results
table, and saves them with the `.pt` result in `outputs/`. Leave
`LANE_VIEW` live drawing off when reading timings: drawing runs inside the
replanning timer. `lane.batch` saves `outputs/<name>_batch.json`.

For the paper, block `LanePaper` builds one full-width figure (7.16 in wide,
about 2.3 in tall) from the saved runs. Each panel is the simulation at one
chosen moment: shaded lanes, every car with a 0.5 s motion arrow, the path
driven so far, the plan made at that moment, and the ego turned to its
heading. (a) merge is full width; (b) dense and (c) one-gap merge-or-yield
share a row. Titles give each spec in short form; the exact specs are in the
comment above `FIGURE_SCENES` in `lane.py`, for the caption. The road is
drawn 1.4x taller than true scale. The plan's predicted 95% region is off by
default (`show_uncertainty=True` draws it): at these noise levels it is too
small to see. Headway is reported in `outputs/lane_table.tex`, written from
the batch results: completed trials, closest clearance and lead gap, lowest
window score, replanning median/p95.

Two planner settings keep the closed loop steady; the lane YAMLs set both
(defaults elsewhere are unchanged):

- `warm_tail: zero`: each new window's guess ends in a zero control instead of
  repeating the last one, so a leftover acceleration is not held to the end of
  the next plan (it was the cause of mid-merge score dips).
- `tie_break: smooth`: if several iterates share the best exact score (often
  all 0 after a surprise), keep the one with the best smooth score instead of
  the last one.

## Development checks

```bash
pip install -e ".[dev]"
ruff format --check src tests
flake8 src
MPLBACKEND=Agg pytest -q -m "not slow"   # -m slow runs the full scenarios
```

## License

MIT. See [LICENSE](LICENSE).
