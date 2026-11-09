# Lane merge: implementation plan

Branch: `lane-merge`, from `opt-planning` at `a503854d9fab1794b138dec353e18d901e0a16ca`.

Current phase: use the existing run/skip blocks for one selected scenario,
with its figure and a compact results table. Batch execution, smoke campaigns,
persistent trace logging, and formal reporting below are deferred until the
interactive examples have been tested. Earlier generated reports are archived
locally under the ignored `outputs/lane-merge-archive/` directory.

## Goal

Use the existing closed-loop merge to demonstrate multiple STL specifications, including nested temporal tasks, and measure online replanning time. No method comparison.

Build A and B first. Add C once those run correctly.

## Reuse

Keep the existing dynamics, traffic beliefs, estimator, MPC loop, warm-start shifting, geometric predicates, and truth-based safety checks. Extend the lane experiment rather than creating a second implementation.

## Specifications

Select the task in YAML:

```yaml
task:
  spec: merge  # merge | headway | merge_or_yield
```

Default to `merge` when the field is absent, preserving existing configurations.

| Task | Requirement | Work |
| --- | --- | --- |
| A: merge | Stay safe; merge within the existing window, dwell, then stay in lane. | Preserve the current formula and behavior. |
| B: headway | A plus a minimum gap behind a designated lead vehicle while entering or occupying the target lane. | Add a relative-position predicate and a truth check. |
| C: merge_or_yield | Stay safe; either complete A's merge task or reach a yield region and remain stopped for a dwell. | Add the yield branch, its progress tracking, and outcome. |

A includes the nested requirement “eventually remain in lane for a duration” (F G). B adds an always requirement. C adds OR between two temporal tasks; its stopping branch also uses F G.

Preserve the current absolute-time interpretation: the configured window's upper endpoint is the deadline for completing the merge dwell. Do not restart deadlines on each replan. Apply the ramp-end condition at merge completion, rather than prohibiting later travel beyond the ramp.

## Decisions for implementation

- Headway uses a fixed named lead vehicle; no dynamic lead selection.
- Configure bumper-to-bumper `min_gap`; convert it to center separation using both vehicle lengths in `RelativeAxisInterval`.
- Activate headway when the ego footprint first overlaps the target lane, and whenever that overlap continues. Use the same condition in planning and truth monitoring.
- Define yielding with an explicit rectangular region upstream of the ramp taper, a deadline, a nonzero stopped-speed tolerance on both velocity components, and a dwell duration.
- Give C feasible braking initial conditions. Check stopping distance and full-footprint road containment before running it.
- Either branch is acceptable in C. Do not introduce a hidden preference for merging.
- Keep estimated progress for planning separate from true progress for outcomes.

## Step 1: get A and B working

1. Add spec selection to `lane_subformulas`, exposing the relevant subformulas for logging.
2. Add headway parameters, predicate, and truth-based violation detection. Preserve headway after merge completion.
3. Keep the current A configuration as a regression case. Add A/B experiment YAMLs using identical traffic, initial conditions, noise, and seeds. Choose a feasible lead gap where the added rule can affect behavior.
4. Keep checking lane retention and settling on truth after merge completion.
5. Run one trial of each, then five seeded smoke trials per task.

## Step 2: add C

1. Build the stopping branch using existing position and speed predicates; combine it with the merge branch under shared safety.
2. Track yield dwell and deadlines across replans. Missing the merge deadline must not end C while yielding remains possible.
3. Extend the task-completion handling so a completed yield maintains the stopped task instead of switching to the existing stay-in-target-lane formula.
4. Add a dense-traffic YAML and a `yielded` outcome. Determine the realized branch from truth, not a smooth OR score.
5. Verify both a merge-success case and a yield-success case.

## Step 3: batch results and timing

Add a small runner with configurable N and seeds. Disable plotting and live callbacks during timing.

Save per trial:
- Outcome and realized branch.
- Merge or yield completion time.
- Minimum traffic clearance; B's minimum lead bumper gap.
- Exact lower bound per planning window and whether it meets alpha.
- Existing `planning_time` plus full replanning time, including formula construction and initial evaluation.

Produce one results table per batch: task, N, merge/yield/failure counts, completion time, minimum clearance, and replanning median/p95/max plus fraction exceeding 200 ms. Keep first-solve timing separate. Record hardware, device, horizon, and iteration budget. Retain failed trials.

The current execution interval is 0.2 s. Measure performance against that interval; a synchronous simulation alone does not establish real-time execution. Do not treat per-window bounds as a whole-trial probability guarantee.

## Files and checks

Main changes: `src/planning/environment.py`, `src/experiments/lane.py`, lane YAMLs, a small batch runner, `src/main.py`, and focused lane tests. Touch planner timing only as needed.

Check spec selection, existing A behavior, headway activation/clearance, nested dwell deadlines, C's OR branches, truth-based outcomes, and timing summaries. Run existing lane/planner tests and one complete smoke batch.

Done when all three YAMLs run through the same MPC pipeline and produce the results table and per-window logs.

Changing rules during a single drive is a later extension. The first version selects one specification per trial.
