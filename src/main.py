"""Run each experiment by setting its block to "run" or "skip".

Figures are shown, then saved to outputs/ when closed.
"""

from experiments import lane, reach_avoid, timing
from utils import skip_run

REACH_AVOID = "configs/scenarios/reach_avoid"
LANE = "configs/scenarios/lane"
LIVE = {"live": True, "optimization_every": 1}  # plot and print each iterate
LANE_VIEW = {"live_optimization": False}  # True: watch every replan live

# 1. Reach-avoid: one obstacle
with skip_run("skip", "ReachAvoid") as check, check():
    reach_avoid.run(f"{REACH_AVOID}/obstacle.yaml", **LIVE)


# 2. Reach-avoid (stlpy NarrowPassage): goal A or B past four obstacles
with skip_run("skip", "NarrowPassage") as check, check():
    reach_avoid.run(f"{REACH_AVOID}/narrow_passage.yaml", **LIVE)


# 3. Reach-avoid (stlpy EitherOr): dwell in t1 or t2, then reach the goal
with skip_run("skip", "EitherOr") as check, check():
    reach_avoid.run(f"{REACH_AVOID}/either_or.yaml", **LIVE)


# 4. Lane change into the faster lane
with skip_run("skip", "LaneChange") as check, check():
    lane.run(f"{LANE}/lane_change.yaml", **LANE_VIEW)


# --- On-ramp merge: one scene per STL specification -------------------------
# Each block shows the run (map, scores, controls), a snapshot figure with
# every car named, and a results table. LANE_VIEW draws each replan live;
# the live view slows planning, so leave it off when reading timings.

# 5. A, merge: G safe AND F[window] G[dwell] in_lane, then stay settled.
#    The lead brakes at 4 s; nothing asks the ego to keep a distance.
with skip_run("run", "LaneMerge") as check, check():
    lane.run(f"{LANE}/lane_merge.yaml", **LANE_VIEW)


# 6. B, headway: A AND G(touching the main lane -> gap to lead >= 22 m).
#    Same scene and seed as A; only the rule differs.
with skip_run("skip", "LaneHeadway") as check, check():
    lane.run(f"{LANE}/lane_headway.yaml", **LANE_VIEW)


# 7. C, merge or yield, dense traffic: no gap, so the ego stops on the ramp.
with skip_run("skip", "LaneMergeOrYield") as check, check():
    lane.run(f"{LANE}/lane_merge_or_yield.yaml", **LANE_VIEW)


# 8. C, merge or yield, one open gap: same rule and ego, so now it merges.
with skip_run("skip", "LaneMergeOrYieldGap") as check, check():
    lane.run(f"{LANE}/lane_merge_or_yield_gap.yaml", **LANE_VIEW)


# 9. Batch: several seeds per scenario, no plots; prints and saves a table
#    (outcomes, gaps, scores, replanning times) to outputs/<name>_batch.json.
with skip_run("skip", "LaneBatch") as check, check():
    for name in ("merge", "headway", "merge_or_yield", "merge_or_yield_gap"):
        lane.batch(f"{LANE}/lane_{name}.yaml", seeds=range(20))


# 10. Paper outputs: the figure of merge, dense and gap with the per-window
#     lower score of each, from the runs saved by blocks 5, 7 and 8, in two
#     layouts (two-column outputs/lane_paper.pdf, one-column
#     outputs/lane_paper_column.pdf), and the LaTeX table of all four scenes from block 9 (outputs/lane_table.tex).
with skip_run("skip", "LanePaper") as check, check():
    lane.paper_figure()
    lane.latex_table()


# 11. Evaluation cost: the formula evaluated for all origins at once vs the
#     recurrent register, smooth forward + backward, against trace length and
#     batch size, on CPU and GPU (outputs/timing_benchmark.json, .pdf). For
#     the paper's CPU numbers run pinned: taskset -c 10 with OMP_NUM_THREADS=1.
with skip_run("skip", "TimingBenchmark") as check, check():
    timing.benchmark()
