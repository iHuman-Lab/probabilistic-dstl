"""Run each experiment by setting its block to "run" or "skip".

Every block reads top to bottom: config -> scenario -> result -> outputs.
Figures are shown, then saved to outputs/ when closed.
"""

from experiments import lane, reach_avoid, risk_aversion
from planning.runners import save_result
from utils import load_config, skip_run

REACH_AVOID = "configs/scenarios/reach_avoid"
LANE = "configs/scenarios/lane"
LIVE = {"live": True, "optimization_every": 1}  # plot and print each iterate


# 1. Reach-avoid: one obstacle
with skip_run("skip", "ReachAvoid") as check, check():
    name = "obstacle"
    cfg = load_config(f"{REACH_AVOID}/{name}.yaml")
    scenario = reach_avoid.build(cfg)  # environment, pdSTL spec, initial guess
    result = reach_avoid.optimize(scenario, **LIVE)  # plan by gradient ascent
    reach_avoid.validate(scenario, result)  # Monte Carlo P(phi) under noise
    reach_avoid.present(result, scenario, stem=name)  # figure + animation
    save_result(result, name, save=True)  # outputs/{name}.pt


# 2. Reach-avoid (stlpy NarrowPassage): goal A or B past four obstacles
with skip_run("skip", "NarrowPassage") as check, check():
    name = "narrow_passage"
    cfg = load_config(f"{REACH_AVOID}/{name}.yaml")
    scenario = reach_avoid.build(cfg)  # environment, pdSTL spec, initial guess
    result = reach_avoid.optimize(scenario, **LIVE)  # plan by gradient ascent
    reach_avoid.validate(scenario, result)  # Monte Carlo P(phi) under noise
    reach_avoid.present(result, scenario, stem=name)  # figure + animation
    save_result(result, name, save=True)  # outputs/{name}.pt


# 3. Reach-avoid (stlpy EitherOr): dwell in t1 or t2, then reach the goal
with skip_run("run", "EitherOr") as check, check():
    name = "either_or"
    cfg = load_config(f"{REACH_AVOID}/{name}.yaml")
    scenario = reach_avoid.build(cfg)  # environment, pdSTL spec, initial guess
    result = reach_avoid.optimize(scenario, **LIVE)  # plan by gradient ascent
    reach_avoid.validate(scenario, result)  # Monte Carlo P(phi) under noise
    reach_avoid.present(result, scenario, stem=name)  # figure + animation
    save_result(result, name, save=True)  # outputs/{name}.pt


# 4. Lane change into the faster lane
with skip_run("skip", "LaneChange") as check, check():
    name = "lane_change"
    cfg = load_config(f"{LANE}/{name}.yaml")
    scenario = lane.build(cfg)  # road, traffic, dynamics, planner
    result = lane.simulate(scenario, live_optimization=True)  # closed-loop MPC
    lane.present(result, scenario, stem=name)  # figures + animation
    save_result(result, name, save=True)  # outputs/{name}.pt


# 5. On-ramp merge into the main lane
with skip_run("skip", "LaneMerge") as check, check():
    name = "lane_merge"
    cfg = load_config(f"{LANE}/{name}.yaml")
    scenario = lane.build(cfg)  # road, traffic, dynamics, planner
    result = lane.simulate(scenario, live_optimization=True)  # closed-loop MPC
    lane.present(result, scenario, stem=name)  # figures + animation
    save_result(result, name, save=True)  # outputs/{name}.pt


# 6. Risk aversion, STLCG: margin on the mean path, noise ignored
with skip_run("skip", "RiskAversionSTLCG") as check, check():
    cfg = load_config(risk_aversion.CONFIG)
    cfg = risk_aversion.with_noise(cfg)  # process noise q per level
    candidates, picks = risk_aversion.optimize(
        cfg, "stlcg"
    )  # plan, keep one/level
    selected = risk_aversion.validate(cfg, picks)  # Monte Carlo of kept plans
    risk_aversion.save_outputs("stlcg", candidates, selected, picks)


# 7. Risk aversion, pdSTL: score on the belief, noise-aware
with skip_run("skip", "RiskAversionPdSTL") as check, check():
    cfg = load_config(risk_aversion.CONFIG)
    cfg = risk_aversion.with_noise(cfg)  # process noise q per level
    candidates, picks = risk_aversion.optimize(
        cfg, "pdstl"
    )  # plan, keep one/level
    selected = risk_aversion.validate(cfg, picks)  # Monte Carlo of kept plans
    risk_aversion.save_outputs("pdstl", candidates, selected, picks)


# 8. Risk aversion, report: compare the saved STLCG and pdSTL results
with skip_run("skip", "RiskAversionReport") as check, check():
    cfg = load_config(risk_aversion.CONFIG)
    cfg = risk_aversion.with_noise(cfg)
    stlcg = risk_aversion.load_outputs("stlcg")  # selected rows + plans
    pdstl = risk_aversion.load_outputs("pdstl")
    risk_aversion.table(
        cfg, stlcg, pdstl
    )  # outputs/risk_aversion/table.csv, table.md
    risk_aversion.figure(
        cfg, stlcg, pdstl
    )  # outputs/risk_aversion/figure.png, figure.pdf
