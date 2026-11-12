"""Run each experiment by setting its block to "run" or "skip".

Figures are shown, then saved to outputs/ when closed.
"""

from experiments import lane, risk_aversion
from planning.runners import save_result
from utils import load_config, skip_run

LANE = "configs/scenarios/lane"

# Lane merge. lane.simulate(..., live_optimization=True) draws each replan
# (slower; leave it off for timings).

# 1. Merge (A); the lead brakes at 4 s.
with skip_run("run", "LaneMerge") as check, check():
    name = "lane_merge"
    cfg = load_config(f"{LANE}/{name}.yaml")
    scenario = lane.build(cfg)
    result = lane.simulate(scenario)
    lane.present(result, scenario, stem=name)
    save_result(result, name, save=True)


# 2. Merge or yield (C), dense traffic: the ego yields.
with skip_run("skip", "LaneMergeOrYield") as check, check():
    name = "lane_merge_or_yield"
    cfg = load_config(f"{LANE}/{name}.yaml")
    scenario = lane.build(cfg)
    result = lane.simulate(scenario)
    lane.present(result, scenario, stem=name)
    save_result(result, name, save=True)


# 3. Merge or yield (C), one open gap: the ego merges.
with skip_run("skip", "LaneMergeOrYieldGap") as check, check():
    name = "lane_merge_or_yield_gap"
    cfg = load_config(f"{LANE}/{name}.yaml")
    scenario = lane.build(cfg)
    result = lane.simulate(scenario)
    lane.present(result, scenario, stem=name)
    save_result(result, name, save=True)


# 4. Batch: 20 seeds per scene, incl. headway (B); outputs/<name>_batch.json.
with skip_run("skip", "LaneBatch") as check, check():
    for name in ("merge", "headway", "merge_or_yield", "merge_or_yield_gap"):
        lane.batch(f"{LANE}/lane_{name}.yaml", seeds=range(20))


# 5. Paper figure (from blocks 1-3) and LaTeX table (from block 4).
with skip_run("skip", "LanePaper") as check, check():
    lane.paper_figure()
    lane.latex_table()


# 6. Risk aversion, STLCG: plan on the mean path, then Monte Carlo.
with skip_run("skip", "RiskAversionSTLCG") as check, check():
    cfg = risk_aversion.with_noise(load_config(risk_aversion.CONFIG))
    candidates, picks = risk_aversion.optimize(cfg, "stlcg")
    selected = risk_aversion.validate(cfg, picks)
    risk_aversion.save_outputs("stlcg", candidates, selected, picks)


# 7. Risk aversion, pdSTL: plan on the belief, then Monte Carlo.
with skip_run("skip", "RiskAversionPdSTL") as check, check():
    cfg = risk_aversion.with_noise(load_config(risk_aversion.CONFIG))
    candidates, picks = risk_aversion.optimize(cfg, "pdstl")
    selected = risk_aversion.validate(cfg, picks)
    risk_aversion.save_outputs("pdstl", candidates, selected, picks)


# 8. Risk aversion table and figure from blocks 6-7 (outputs/risk_aversion/).
with skip_run("skip", "RiskAversionReport") as check, check():
    cfg = risk_aversion.with_noise(load_config(risk_aversion.CONFIG))
    stlcg = risk_aversion.load_outputs("stlcg")
    pdstl = risk_aversion.load_outputs("pdstl")
    risk_aversion.table(cfg, stlcg, pdstl)
    risk_aversion.figure(cfg, stlcg, pdstl)
