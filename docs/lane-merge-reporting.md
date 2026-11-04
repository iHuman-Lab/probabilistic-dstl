# Lane merge: tasks for the paper results

Branch: `lane-merge-clean`. All lane work stays on this branch for now: do
not merge it with `experiments/two-gap` or into `opt-planning`, and do not
pull changes from those branches. Read `docs/lane-merge-plan.md` and the
README section "Lane change and merge" first. Work in the order below; each task
lists what to change, the checks it must pass, and what not to do.

## What the paper needs from this experiment

Two claims, nothing else:

1. Closed-loop pdSTL satisfies nested STL specifications under noise and
   unannounced events (the lead brakes at 4 s; dense vs open traffic).
2. Replanning cost per 0.2 s control step.

There is no method comparison here (the 2D experiment carries that). The
paper shows one figure (`lane.paper_figure()`, already done) and one table
(`lane.latex_table()`), both built from saved runs.

## Ground rules

- Do not tune scenes, seeds or thresholds to make a result look better. If
  a trial fails, it is reported as a failure, with its outcome name.
- Count task completion on the **true** state only (`stopped_reason` of
  `success` or `yielded`), never on the estimate.
- ρ̲ (`hard_interval[0]`) is a per-window lower **robustness score** under
  the predicted belief. Never label it a probability of success.
- Keep defaults that other experiments use unchanged (`configs/planning.yaml`
  defaults, `Planner` behaviour when a setting is not given).
- Keep functions small, with docstrings (Args/Returns), ruff clean, and add
  a pytest test for every change.

## Task 1: faster merge spec (one Eventually instead of an Or over starts)

Where: `_lane_merge_task` in `src/planning/environment.py`, the `else:`
branch (no dwell credit yet). Today it builds one candidate per possible
dwell start `s` in `[start, latest_start]` and joins them with `Or`:

    branch = OR_s ( G[s,s+d] lane  AND  G[s+d,s+d] before_end
                    AND  G[s+d, H] (lane AND settled) )

Replace it with a single temporal operator over the start time:

    branch = F[start, latest_start] ( G[0,d] lane  AND  G[d,d] before_end
                                      AND  G[d, inf) (lane AND settled) )

and build `complete` and `stay` the same way (`F[start, latest_start]` of the
completion part, and of `G[d, inf)(lane AND settled)`). Keep the grouping of
the conjunctions exactly as today (`(lane AND before_end) AND stay`), so the
Fréchet bounds are identical at every start. `G[d, inf)` runs to the end of
the trace, which is the horizon end used today.

Leave the credited branch (`if credit:`) as it is: it needs the lane to hold
from now until a variable finish, which is not a plain future-only window.

Checks:

- New test: for random Gaussian lane beliefs (several dozen, with the ego
  crossing into the lane at different times), the old and new `branch`
  formulas give the **same lower bound** (`probability_interval(...)[0]`,
  exact, `beta=None`) to 1e-6. The upper bound may differ (Eventually takes
  a max, a chain of Or takes a capped sum); assert new upper >= new lower.
  Keep the old construction as a private helper inside the test, not in
  `src/`.
- Time one `spec.smooth_lower(...)` + backward on the first window of
  `lane_merge.yaml`, old vs new; report both numbers in the PR text.
- All lane tests, including `-m slow`, still pass.

## Task 2: two more table columns

Where: `lane.batch` and `lane.latex_table` in `src/experiments/lane.py`.

1. In `batch`, add to the summary table:
   - `"branches"`: counts of `row["Realized branch"]` over trials, e.g.
     `{"merge": 18, "yield": 2, None: 0}` (store `None` as `"none"`).
   - `"windows meeting alpha"`: the fraction over **all** windows of all
     trials with score >= alpha, where alpha is `s.planner.cfg["alpha"]`;
     `None` when alpha is not set.
   - `"machine"`: add the GPU name (`torch.cuda.get_device_name()` when the
     device is CUDA), the torch version and the planner's `max_iters`
     (already there) and `H`.
2. In `latex_table`, add the columns "Merged/yielded" (`18/2`) and
   "ρ̲ ≥ α (%)" (one decimal). Keep "Completed", "Clearance", "Lead gap",
   "Lowest ρ̲" and "Replan med./p95 (ms)". Put the first-replan median in a
   commented LaTeX line under the table (`% first replan median: ... ms`),
   not in a column.

Checks: extend `test_paper_figure_is_full_width_to_scale_and_table_is_latex`
(fake batch JSON) for the new columns, and
`test_batch_runs_each_seed_and_saves_one_table` for the new table keys.

## Task 3: a timing warm-up

The first CUDA call in a process pays for kernel loading, which would land
in the first replan of the first seed. In `batch`, before the seed loop,
build the first scene and run one `optimize_window` that is not recorded.
On CPU it is harmless. Add one line to the docstring saying so.

Check: the batch test still passes; no timing values change on CPU beyond
noise.

## Task 4: run the batch and the figure (on the RTX machine)

1. In `src/main.py` set `LANE_VIEW = {"live_optimization": False}` (it is).
2. Run blocks 5, 7 and 8 once (they save `outputs/lane_<name>.pt`).
3. Change block 9 to `seeds=range(20)` and run it, first with
   `PDSTL_DEVICE=cuda`, then once more with `PDSTL_DEVICE=cpu` and rename
   the CPU outputs to `outputs/lane_<name>_batch_cpu.json` (both go in the
   paper text: GPU in the table, CPU in one sentence).
4. Run block 10 (`LanePaper`).
5. Do not commit `outputs/` (it is git-ignored). Copy
   `outputs/lane_paper.pdf` and `outputs/lane_table.tex` into the PR
   description or a release note instead.

Report back, per scene: outcomes (all names and counts), merged/yielded,
lowest window score, replan median/p95/max, fraction over 200 ms, first
replan median, and the `machine` block.

## Task 5: decide what the timing allows us to say

Compare the GPU replan median and p95 with the 0.2 s step:

- p95 <= 200 ms: the planner keeps up with the control step on this
  machine. Say exactly that, with the hardware.
- otherwise: report the ratio (for example "median 0.9 s, 4.5x the 0.2 s
  step") and that the simulation pauses while planning. Do not describe the
  runs as real time.

Write the result as two sentences at the end of the PR description; the
author puts them in the paper.

## Done when

- Tasks 1-3 merged on `lane-merge-clean` with tests, ruff clean.
- Task 4 outputs exist and the report in Task 4 is filled in.
- Task 5's two sentences are written.
