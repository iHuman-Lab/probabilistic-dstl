"""Lane change and merge: semantics, planning, and figures."""

import os
from functools import reduce
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest
import torch
import yaml
from matplotlib import colors

from experiments import lane
from models.beliefs import GaussianBeliefTrajectory
from models.rollouts import LaneBeliefTrajectory, lane_rollout
from pdstl.operators import Always, Or
from pdstl.predicates import AxisInterval, LessThan, RelativeAxisInterval
from planning.environment import (
    build_lane_merge_environment,
    lane_contains_footprint,
    lane_goal_reached,
    lane_has_collision,
    lane_headway_violated,
    lane_lead_gap,
    lane_local_window,
    lane_road_predicate,
    lane_subformulas,
    lane_traffic_acceleration,
    lane_yield_contains,
    lane_yield_reached,
)
from planning.planner import IterationRecord, MPCResult, Planner
from utils import load_config
from visualization.animation import animate_mpc
from visualization.figures import (
    COLORS,
    RHO_INTERVAL,
    RHO_LOWER,
    _draw_environment,
    plot_lane_merge,
    plot_lane_snapshots,
)
from visualization.live_plots import create_live_view


def test_collision_uses_footprints_and_margins():
    env = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_change.yaml")
    )
    ego = torch.tensor([0.0, 0.0])
    traffic = torch.tensor([[5.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    assert lane_has_collision(env, ego, traffic)
    traffic[0, 0] += 0.001
    assert not lane_has_collision(env, ego, traffic)


def test_spec_selection_and_paired_scenarios():
    a = load_config("configs/scenarios/lane/lane_merge.yaml")
    b = load_config("configs/scenarios/lane/lane_headway.yaml")
    assert {k: v for k, v in a.items() if k != "task"} == {
        k: v for k, v in b.items() if k != "task"
    }
    # A carries the same headway settings (reported, not enforced).
    assert {k: v for k, v in a["task"].items() if k != "spec"} == {
        k: v for k, v in b["task"].items() if k != "spec"
    }
    del a["task"]["spec"]
    assert build_lane_merge_environment(a).metadata["task"]["spec"] == "merge"
    a["task"]["spec"] = "typo"
    with pytest.raises(ValueError, match="unknown lane task spec"):
        build_lane_merge_environment(a)
    b["task"]["headway"]["lead_vehicle"] = "missing"
    with pytest.raises(ValueError, match="lead_vehicle"):
        build_lane_merge_environment(b)


def test_paired_ab_initial_truth_and_noise_draws():
    states = []
    for name in ("merge", "headway"):
        s = lane.build(
            load_config(f"configs/scenarios/lane/lane_{name}.yaml"),
            device="cpu",
        )
        generator = lane._generator(s)
        state = lane.initial_state_of(s, generator=generator)
        execute = lane._executor(
            s,
            s.cfg["traffic_q_std"] ** 2 * torch.eye(4),
            None,
            None,
            False,
            generator=generator,
        )
        trace = [state]
        for step in range(3):
            state = execute(state, torch.tensor([0.5, 0.1]), step)
            trace.append(state)
        states.append(trace)
    for a, b in zip(*states):
        for field in (
            "ego_mean",
            "ego_cov",
            "ego_true",
            "traffic_mean",
            "traffic_cov",
            "traffic_true",
        ):
            torch.testing.assert_close(
                getattr(a, field), getattr(b, field), rtol=0, atol=0
            )


def _deterministic_lane_beliefs(env, ego, traffic):
    return LaneBeliefTrajectory(
        ego,
        torch.zeros(*ego.shape, 4),
        traffic,
        torch.zeros(*traffic.shape, 4),
        [car["name"] for car in env.metadata["traffic"]],
    )


@pytest.mark.parametrize(
    "y,violated", [(0.899, False), (0.9, True), (2.4, True), (3.6, True)]
)
def test_headway_activation_formula_and_truth(y, violated):
    cfg = load_config("configs/scenarios/lane/lane_headway.yaml")
    cfg["task"]["headway"]["min_gap"] = 40.0
    env = build_lane_merge_environment(cfg)
    ego = torch.tensor([[[0.0, y, 0.0, 0.0]]]).expand(1, 2, 4).clone()
    traffic = torch.zeros(1, 2, 3, 4)
    traffic[..., 0] = 44.49  # 39.99 m bumper gap, including both lengths.
    formula = lane_subformulas(lane_local_window(env, 30, completed=True), 1)[
        "headway"
    ]
    exact = formula.probability_interval(
        _deterministic_lane_beliefs(env, ego, traffic)
    )
    assert exact[0].item() == float(not violated)
    assert lane_headway_violated(env, ego[0, 0], traffic[0, 0]) == violated
    assert lane_lead_gap(env, ego[0, 0], traffic[0, 0]) == pytest.approx(39.99)
    traffic[..., 0] = 44.5
    assert not lane_headway_violated(env, ego[0, 0], traffic[0, 0])


def test_physical_post_merge_retention_and_headway():
    cfg = load_config("configs/scenarios/lane/lane_headway.yaml")
    s = lane.build(cfg, device="cpu")
    state = lane.initial_state_of(s)
    # Past the gore, 20 m behind the lead: a 15.5 m bumper gap, under 22 m.
    ahead = state.traffic_true.clone()
    ahead[:, 0] += 40.0
    lead = float(ahead[0, 0])
    true = torch.tensor([lead - 20.0, 3.6, 18.0, 0.0])
    state = state._replace(
        ego_true=true, traffic_true=ahead, true_completed=20
    )
    assert lane._outcome(s)(state, 21) == "headway_violation"
    true = true.clone()
    true[1] = 0.8
    assert lane._outcome(s)(state._replace(ego_true=true), 21) == (
        "lane_retention_violation"
    )
    traffic = state.traffic_true.clone()
    traffic[:, 0] = 100
    drifting = torch.tensor([40.0, 3.6, 18.0, 0.4])
    drift = state._replace(ego_true=drifting, traffic_true=traffic)
    deadline = s.env.metadata["task"]["start_end_steps"][1]
    # Unsettled is allowed until the latest dwell finish, then it fails.
    assert lane._outcome(s)(drift, deadline - 1) is None
    assert lane._outcome(s)(drift, deadline) == "settling_violation"


def _c_environment():
    return build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_merge_or_yield.yaml")
    )


@pytest.mark.parametrize(
    "window,steps", [([2.4, 4.8], (12, 24)), ([2.41, 4.79], (13, 23))]
)
def test_merge_and_yield_use_the_same_time_conversion(window, steps):
    cfg = load_config("configs/scenarios/lane/lane_merge_or_yield.yaml")
    cfg["task"]["start_window_seconds"] = window
    cfg["task"]["yield"]["window_seconds"] = window
    task = build_lane_merge_environment(cfg).metadata["task"]
    assert task["start_end_steps"] == task["yield"]["start_end_steps"] == steps


def test_positive_dwell_requires_a_transition():
    cfg = load_config("configs/scenarios/lane/lane_merge_or_yield.yaml")
    cfg["task"]["dwell_seconds"] = 1e-12
    cfg["task"]["yield"]["dwell_seconds"] = 1e-12
    task = build_lane_merge_environment(cfg).metadata["task"]
    assert task["dwell_steps"] == task["yield"]["dwell_steps"] == 1
    cfg["task"]["dwell_seconds"] = 0
    assert (
        build_lane_merge_environment(cfg).metadata["task"]["dwell_steps"] == 0
    )


def test_credited_merge_dwell_must_finish_by_deadline():
    env = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_merge.yaml")
    )
    end = env.metadata["task"]["start_end_steps"][1]
    window = lane_local_window(env, end - 2, streak=4)
    ego = torch.zeros(1, 7, 4)
    ego[..., 1] = env.metadata["task"]["target_center"]
    ego[0, 3:, 1] = 0
    traffic = torch.zeros(1, 7, 3, 4)
    beliefs = _deterministic_lane_beliefs(env, ego, traffic)
    assert (
        lane_subformulas(window, 6)["complete"].probability_interval(beliefs)[
            0
        ]
        == 1
    )
    with pytest.raises(ValueError, match="no remaining feasible dwell start"):
        lane_subformulas(lane_local_window(env, end - 1, streak=4), 6)


@pytest.mark.parametrize("spec", ["merge", "headway"])
def test_run_table_uses_truth_and_separates_first_timing(spec):
    cfg = load_config(f"configs/scenarios/lane/lane_{spec}.yaml")
    s = lane.build(cfg, device="cpu")
    base = lane.initial_state_of(s, generator=lane._generator(s))
    traffic = torch.tensor(
        [
            [50.0, 3.6, 0.0, 0.0],
            [100.0, 3.6, 0.0, 0.0],
            [150.0, 3.6, 0.0, 0.0],
            [-100.0, 3.6, 0.0, 0.0],
            [-150.0, 3.6, 0.0, 0.0],
        ]
    )
    states = [
        base._replace(
            ego_true=torch.tensor([float(step), 3.6, 0.0, 0.0]),
            traffic_true=traffic,
            completed=2,
            true_completed=1,
        )
        for step in range(4)
    ]
    plans = [
        SimpleNamespace(
            planning_time=optimizer,
            replanning_time=replan,
            hard_interval=(score, 1.0),
            alpha=0.95,
        )
        for optimizer, replan, score in [
            (0.01, 2.0, 0.5),
            (0.02, 0.1, 0.96),
            (0.03, 0.3, 1.0),
        ]
    ]
    result = MPCResult(states, torch.zeros(3, 2), plans, "success")
    rows = lane.summarize(result, s.env, cfg["dt"])
    assert rows["Realized branch"] == "merge"
    assert rows["Completion time [s]"] == pytest.approx(0.2)
    assert rows["Minimum traffic clearance [m]"] == pytest.approx(42.5)
    assert rows["Optimizer median [ms]"] == pytest.approx(20)
    assert rows["First replan [ms]"] == 2000
    assert rows["Later replans median [ms]"] == pytest.approx(200)
    assert rows["Later replans p95 [ms]"] == pytest.approx(290)
    assert rows["Later replans maximum [ms]"] == 300
    assert rows["Lowest window score"] == 0.5
    assert rows["Windows meeting alpha"] == pytest.approx(2 / 3)
    # Both paired scenes report the gap; only B enforces it.
    assert rows["Minimum lead gap in lane [m]"] == pytest.approx(42.5)


@pytest.mark.parametrize("outcome", ["step_limit", "planning_failure"])
def test_incomplete_run_figure_and_table(outcome, tmp_path):
    s = lane.build(
        load_config("configs/scenarios/lane/lane_merge_or_yield.yaml"),
        device="cpu",
    )
    result = lane.trial(s, max_steps=0)
    result.stopped_reason = outcome
    if outcome == "planning_failure":
        result.failed_replanning_time = 0.05
    rows = lane.summarize(result, s.env, s.cfg["dt"])
    assert rows["Completion time [s]"] is None
    assert rows["Realized branch"] is None
    assert rows["Later replans median [ms]"] is None
    if outcome == "planning_failure":
        assert rows["First replan [ms]"] == 50
    figures = plot_lane_merge(
        result, s.env, dt=s.cfg["dt"], save_path=tmp_path / "yield.png"
    )
    map_ax, score_ax, _ = figures["combined"][1]
    assert "Yield region" in {patch.get_label() for patch in map_ax.patches}
    assert map_ax.get_xlim()[1] >= s.env.region("yield").xmax
    assert "Yield completion deadline" in {
        text.get_text() for text in score_ax.get_legend().get_texts()
    }
    cells = figures["table"][1].tables[0].get_celld()
    assert outcome.replace("_", " ") in {
        cell.get_text().get_text() for cell in cells.values()
    }
    assert {file.name for file in tmp_path.iterdir()} == {
        "yield.png",
        "yield.pdf",
        "yield_snapshots.png",
        "yield_snapshots.pdf",
        "yield_table.png",
        "yield_table.pdf",
    }


def test_yield_geometry_speed_and_validation():
    env = _c_environment()
    stopped = torch.tensor([10.0, 0.0, 0.0, 0.0])
    assert lane_yield_contains(env, stopped)
    for dim, value in [(0, 5.0), (1, 1.0), (2, 0.401), (3, 0.301)]:
        outside = stopped.clone()
        outside[dim] = value
        assert not lane_yield_contains(env, outside)
    cfg = load_config("configs/scenarios/lane/lane_merge_or_yield.yaml")
    cfg["task"]["yield"]["x_range"] = [10.0, cfg["ramp"]["start_x"] + 1.0]
    with pytest.raises(ValueError, match="upstream"):
        build_lane_merge_environment(cfg)


@pytest.mark.parametrize(
    "branch,y,expected",
    [
        ("merge", 3.6, 1),
        ("yield", 0.0, 1),
        ("neither", 1.0, 0),
    ],
)
def test_c_or_branches_share_safety(branch, y, expected):
    env = _c_environment()
    ego = torch.zeros(1, 31, 4)
    ego[..., 0] = 10
    ego[..., 1] = y
    traffic = torch.zeros(1, 31, 3, 4)
    traffic[..., 0] = 100
    formula = lane_subformulas(env, 30)
    assert {"merge_complete", "yield_complete"} <= formula.keys()
    beliefs = _deterministic_lane_beliefs(env, ego, traffic)
    assert formula["overall"].probability_interval(beliefs)[0] == expected
    if expected:
        assert (
            formula[f"{branch}_complete"].probability_interval(beliefs)[0] == 1
        )
    ego[0, 0, 1] = -1.0
    unsafe = _deterministic_lane_beliefs(env, ego, traffic)
    assert formula["overall"].probability_interval(unsafe)[0] == 0


def test_c_absolute_yield_deadline_and_accumulated_dwell():
    env = _c_environment()
    task = env.metadata["task"]["yield"]
    end = task["start_end_steps"][1]
    dwell = task["dwell_steps"]
    ego = torch.zeros(1, 7, 4)
    ego[..., 0] = 10
    traffic = torch.zeros(1, 7, 3, 4)
    traffic[..., 0] = 100
    # Three credited samples leave two transitions in a four-transition dwell.
    window = lane_local_window(env, end - 2, yield_streak=dwell - 1)
    parts = lane_subformulas(window, 6)
    assert "merge_complete" not in parts
    ego[0, 3:, 2] = 1.0
    beliefs = _deterministic_lane_beliefs(env, ego, traffic)
    assert parts["yield_complete"].probability_interval(beliefs)[0] == 1
    assert lane_yield_reached(env, ego[0, 0], end, dwell + 1)
    assert not lane_yield_reached(env, ego[0, 0], end + 1, dwell + 2)
    with pytest.raises(ValueError, match="neither merge nor yield"):
        lane_subformulas(
            lane_local_window(env, end - 1, yield_streak=dwell - 1), 6
        )


def test_c_truth_outcomes_deadlines_and_yield_retention():
    cfg = load_config("configs/scenarios/lane/lane_merge_or_yield.yaml")
    s = lane.build(cfg, device="cpu")
    state = lane.initial_state_of(s)
    truth = torch.tensor([10.0, 0.0, 0.0, 0.0])
    traffic = state.traffic_true.clone()
    traffic[:, 0] = 100
    state = state._replace(ego_true=truth, traffic_true=traffic)
    end = s.env.metadata["task"]["start_end_steps"][1]
    assert lane._outcome(s)(state, end) is None  # merge expired, yield lives
    # Estimated success must not produce a physical outcome.
    estimated = state._replace(yield_completed=end, yield_streak=100)
    assert lane._outcome(s)(estimated, end) is None
    yield_end = s.env.metadata["task"]["yield"]["start_end_steps"][1]
    assert lane._outcome(s)(estimated, yield_end) == "deadline_missed"
    yielded = state._replace(true_yield_completed=end)
    assert lane._outcome(s)(yielded, end + 1) is None
    assert lane._outcome(s)(yielded, end + 5) == "yielded"
    drift = truth.clone()
    drift[3] = 0.4
    assert lane._outcome(s)(yielded._replace(ego_true=drift), end + 1) == (
        "yield_retention_violation"
    )


def test_c_estimated_and_true_progress_are_separate():
    cfg = load_config("configs/scenarios/lane/lane_merge_or_yield.yaml")
    s = lane.build(cfg, device="cpu")
    state = lane.initial_state_of(s)
    estimate = torch.tensor([10.0, 0.0, 0.0, 0.0])
    truth = estimate.clone()
    truth[1] = 1.0
    state = state._replace(
        ego_mean=estimate, ego_true=truth, yield_streak=0, true_yield_streak=0
    )
    steps = cfg["T_SIM"]
    observation = torch.zeros(steps, 4)
    observation[:, 1] = -1.0
    executor = lane._executor(
        s,
        cfg["traffic_q_std"] ** 2 * torch.eye(4),
        torch.zeros(steps, 4),
        torch.zeros(steps, 3, 4),
        False,
        ego_observation_noise=observation,
        traffic_observation_noise=torch.zeros(steps, 3, 4),
    )
    start = s.env.metadata["task"]["yield"]["start_end_steps"][0]
    dwell = s.env.metadata["task"]["yield"]["dwell_steps"]
    for step in range(start - 1, start + dwell):
        state = executor(state, torch.zeros(2), step)
    assert state.yield_completed == start + dwell
    assert state.true_yield_completed is None
    assert state.true_yield_streak == 0


def test_c_completed_yield_formula_maintains_stop_after_deadlines():
    env = _c_environment()
    window = lane_local_window(env, 40, yield_completed=True)
    ego = torch.zeros(1, 5, 4)
    ego[..., 0] = 10
    traffic = torch.zeros(1, 5, 3, 4)
    traffic[..., 0] = 100
    formula = lane_subformulas(window, 4)["overall"]
    assert (
        formula.probability_interval(
            _deterministic_lane_beliefs(env, ego, traffic)
        )[0]
        == 1
    )
    ego[0, -1, 2] = 0.401
    assert (
        formula.probability_interval(
            _deterministic_lane_beliefs(env, ego, traffic)
        )[0]
        == 0
    )


def test_road_contains_the_whole_vehicle_and_taper_front_corner():
    straight = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_change.yaml")
    )
    assert lane_contains_footprint(straight, 0.0, -0.9)
    assert not lane_contains_footprint(straight, 0.0, -0.901)

    merge = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_merge.yaml")
    )
    road, ramp = merge.metadata["road"], merge.metadata["ramp"]
    x = ramp["start_x"] + 20.0
    front = x + merge.metadata["ego_vehicle"]["width"] / 2
    slope = (road["lane_divider"] - road["y_min"]) / (
        ramp["end_x"] - ramp["start_x"]
    )
    center_y = road["y_min"] + slope * (front - ramp["start_x"]) + 0.9
    assert lane_contains_footprint(merge, x, center_y)
    assert not lane_contains_footprint(merge, x, center_y - 0.001)


def test_dwell_finishes_inside_window_and_merge_front_is_bounded():
    change = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_change.yaml")
    )
    task = change.metadata["task"]
    completion = task["start_end_steps"][0] + task["dwell_steps"]
    mean = torch.tensor([50.0, task["target_center"], 0.0, 0.0])
    assert not lane_goal_reached(
        change, mean, completion - 1, task["dwell_steps"]
    )
    assert lane_goal_reached(change, mean, completion, task["dwell_steps"] + 1)

    merge = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_merge.yaml")
    )
    limit = (
        merge.metadata["ramp"]["end_x"]
        - merge.metadata["ego_vehicle"]["width"] / 2
    )
    mean[0] = limit
    assert lane_goal_reached(merge, mean, completion, task["dwell_steps"] + 1)
    mean[0] += 0.001
    assert not lane_goal_reached(
        merge, mean, completion, task["dwell_steps"] + 1
    )


def test_completion_formula_matches_sampled_dwell():
    env = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_change.yaml")
    )
    task = env.metadata["task"]
    steps = env.metadata["task"]["start_end_steps"][1] + 1
    mean = torch.zeros(1, steps, 4)
    start = task["start_end_steps"][0]
    finish = start + task["dwell_steps"]
    mean[0, start : finish + 1, 1] = task["target_center"]
    covariance = torch.zeros(1, steps, 4, 4)
    traffic = torch.zeros(1, steps, 3, 4)
    traffic_covariance = torch.zeros(1, steps, 3, 4, 4)
    beliefs = LaneBeliefTrajectory(
        mean,
        covariance,
        traffic,
        traffic_covariance,
        [vehicle["name"] for vehicle in env.metadata["traffic"]],
    )
    interval = lane_subformulas(env, steps - 1)[
        "complete"
    ].probability_interval(beliefs)
    torch.testing.assert_close(interval, torch.ones(2))
    assert lane_goal_reached(
        env,
        mean[0, finish],
        finish,
        task["dwell_steps"] + 1,
    )


def test_completed_spec_requires_staying_settled_not_ramp_timing():
    env = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_merge.yaml")
    )
    task = env.metadata["task"]
    steps = 6
    mean = torch.zeros(1, steps, 4)
    mean[..., 0] = env.metadata["ramp"]["end_x"] + 20.0
    mean[..., 1] = task["target_center"]
    cars = len(env.metadata["traffic"])
    traffic = torch.zeros(1, steps, cars, 4)
    traffic[..., 0] = -100.0

    def overall(ego_mean, step):
        beliefs = LaneBeliefTrajectory(
            ego_mean,
            torch.zeros(1, steps, 4, 4),
            traffic,
            torch.zeros(1, steps, cars, 4, 4),
            [vehicle["name"] for vehicle in env.metadata["traffic"]],
        )
        window = lane_local_window(env, step, completed=True)
        return lane_subformulas(window, steps - 1)[
            "overall"
        ].probability_interval(beliefs)

    # Past the ramp end and after the deadline, staying in lane still holds.
    end = task["start_end_steps"][1]
    torch.testing.assert_close(overall(mean, end + 3), torch.ones(2))
    drifting = mean.clone()
    drifting[..., 3] = task["settle_lateral_speed"] + 0.5
    torch.testing.assert_close(overall(drifting, end + 3), torch.zeros(2))


@pytest.fixture
def lane_config(tmp_path):
    cfg = load_config("configs/scenarios/lane/lane_change.yaml")
    cfg["H"] = 8
    cfg["task"]["start_window_seconds"] = [0.6, 1.0]
    cfg["task"]["dwell_seconds"] = 0.2
    cfg["planner"]["max_iters"] = 2
    cfg["seed"] = 0
    path = tmp_path / "lane_change.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return cfg, path


def test_relative_beliefs_and_separate_formulae(lane_config):
    cfg, _ = lane_config
    setup = lane.build(cfg)
    state = lane.initial_state_of(setup)
    names = [car["name"] for car in cfg["traffic"]]
    rollout = lane_rollout(
        setup.dyn,
        state[0],
        state[1],
        state[3],
        state[4],
        cfg["traffic_q_std"] ** 2 * torch.eye(4, device=setup.dyn.device),
        names,
        cfg["speed_bounds"],
        (cfg["accel_bounds"]["longitudinal"], cfg["accel_bounds"]["lateral"]),
    )
    prediction = rollout(
        setup.planner._control_parameters(lane.initial_controls(setup))
    )
    belief = prediction.belief_trajectory
    relative = belief.relative_belief("lead")
    torch.testing.assert_close(
        relative.trace.mean[0, 0], state[3][0, :2] - state[0][:2]
    )
    torch.testing.assert_close(
        relative.trace.covariance[0, 0],
        state[4][0, :2, :2] + state[1][:2, :2],
    )
    event = RelativeAxisInterval("lead", -5.0, 5.0, dim=0)
    torch.testing.assert_close(
        belief.probability_bounds(event),
        relative.probability_bounds(AxisInterval(-5.0, 5.0, dim=0)),
    )
    with pytest.raises(ValueError, match="lane belief"):
        GaussianBeliefTrajectory(
            prediction.aux["mean_trace"], prediction.aux["cov_trace"]
        ).probability_bounds(event)
    local = lane_local_window(setup.env, 0)
    formulae = lane_subformulas(local, cfg["H"])
    assert {"road", "safe", "complete", "overall"} <= formulae.keys()
    assert sum(name.startswith("safety_") for name in formulae) == 3
    lower, upper = formulae["overall"].probability_interval(belief)
    assert 0 <= lower <= upper + 1e-6 <= 1 + 1e-6


def test_lane_save_and_live_observer(
    lane_config, tmp_path, monkeypatch, capsys
):
    from planning import runners

    cfg, path = lane_config
    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    result = lane.run(str(path), max_steps=1, show=False, save=True)
    for stem in ("lane_change", "lane_change_snapshots", "lane_change_table"):
        for suffix in (".png", ".pdf"):
            assert (tmp_path / f"{stem}{suffix}").exists()
    assert (tmp_path / "lane_change.pt").exists()  # the result, for analysis
    assert {file.suffix for file in tmp_path.iterdir()} == {
        ".yaml",
        ".png",
        ".pdf",
        ".pt",
    }

    setup = lane.build(cfg)
    fig, axes, observe = create_live_view(
        setup.env,
        result.states[0],
        dt=cfg["dt"],
        dynamics=setup.dyn,
        max_iters=2,
    )
    ax_map = axes[0]
    labels = [patch.get_label() for patch in ax_map.patches]
    assert {"Ego", "Traffic"} <= set(labels)
    traffic_patches = [
        item
        for item in ax_map.patches
        if item.get_label() in {"Traffic", "_nolegend_"}
    ]
    for truth, patch in zip(result.states[0].traffic_true, traffic_patches):
        assert patch.get_x() + patch.get_width() / 2 == pytest.approx(
            truth[0].item()
        )
    legend = [item.get_text() for item in ax_map.get_legend().get_texts()]
    assert len(legend) == len(set(legend))
    window_labels = [
        item.get_text() for item in axes[3].get_legend().get_texts()
    ]
    assert "Dwell-start window" in window_labels
    assert "Completion deadline" in window_labels
    executed = next(
        line for line in ax_map.lines if line.get_label() == "Executed"
    )
    planned = next(line for line in ax_map.lines if line.get_label() == "Plan")
    assert list(executed.get_xdata()) == [result.states[0].ego_true[0].item()]
    plan = result.window_plans[0]
    observe.on_iteration(
        0,
        0,
        IterationRecord(
            plan.controls,
            plan.final_loss,
            plan.smooth_lower,
            plan.hard_interval,
            plan.smoothing_beta,
        ),
    )
    assert len(planned.get_xdata()) == cfg["H"] + 1
    first_candidate = list(planned.get_ydata())
    observe.on_iteration(
        0,
        1,
        IterationRecord(
            plan.controls * 0.5,
            plan.final_loss,
            plan.smooth_lower,
            plan.hard_interval,
            plan.smoothing_beta,
        ),
    )
    assert list(planned.get_ydata()) != first_candidate
    observe(0, result.states[1], plan)
    assert len(executed.get_xdata()) == 2
    plt.close(fig)

    viewed = lane.run(
        str(path), max_steps=1, show=False, save=False, live=True
    )
    torch.testing.assert_close(
        viewed.applied_controls, result.applied_controls
    )
    for live_state, state in zip(viewed.states, result.states):
        torch.testing.assert_close(live_state[0], state[0])
        torch.testing.assert_close(live_state[3], state[3])
    assert [p.hard_interval for p in viewed.window_plans] == [
        p.hard_interval for p in result.window_plans
    ]
    assert "Lane change window 1, iteration 1/2" in capsys.readouterr().out
    plt.close("all")


@pytest.mark.slow
def test_default_lane_completes_with_physical_controls():
    cfg = load_config("configs/scenarios/lane/lane_change.yaml")
    result = lane.run(
        "configs/scenarios/lane/lane_change.yaml", show=False, save=False
    )
    assert result.stopped_reason == "success"
    assert 1 <= len(result.window_plans) <= cfg["T_SIM"]
    assert (
        len(result.window_plans[0].loss_history) <= cfg["planner"]["max_iters"]
    )
    lower, upper = result.window_plans[0].hard_interval
    assert 0 <= lower <= upper + 1e-6 <= 1 + 1e-6
    for index, plan in enumerate(result.window_plans):
        torch.testing.assert_close(
            result.applied_controls[index], plan.controls[0]
        )
        torch.testing.assert_close(
            plan.controls, plan.rollout.aux["applied_controls"]
        )
    speeds = torch.stack([state.ego_true[2] for state in result.states])
    assert bool(((speeds >= 0) & (speeds <= 25)).all())
    assert bool((result.applied_controls[:, 0].abs() <= 2.5).all())
    assert bool((result.applied_controls[:, 1].abs() <= 2.0).all())
    setup = lane.build(cfg)
    first_state = lane.initial_state_of(setup)
    replay_rollout = lane_rollout(
        setup.dyn,
        first_state[0],
        first_state[1],
        first_state[3],
        first_state[4],
        cfg["traffic_q_std"] ** 2 * torch.eye(4, device=setup.dyn.device),
        [car["name"] for car in cfg["traffic"]],
        cfg["speed_bounds"],
        (cfg["accel_bounds"]["longitudinal"], cfg["accel_bounds"]["lateral"]),
    )
    first = result.window_plans[0]
    spec = lane_local_window(setup.env, 0).get_specification(cfg["H"])
    replay = setup.planner.evaluate_controls(
        replay_rollout, first.controls, spec=spec
    )
    torch.testing.assert_close(
        replay.rollout.aux["mean_trace"], first.rollout.aux["mean_trace"]
    )
    assert replay.hard_interval == pytest.approx(first.hard_interval, abs=1e-6)
    task = build_lane_merge_environment(cfg).metadata["task"]
    witness = max(result.states[-1].true_entry, task["start_end_steps"][0])
    assert witness <= task["start_end_steps"][1]
    assert len(result.window_plans) >= witness + task["dwell_steps"]
    _assert_settled_in_lane(result, task, cfg)


def test_deadline_missed_is_reported(lane_config):
    cfg, path = lane_config
    cfg["task"]["start_window_seconds"] = [0.0, 0.2]
    path.write_text(yaml.safe_dump(cfg))
    result = lane.run(str(path), max_steps=5, show=False, save=False)
    assert result.stopped_reason == "deadline_missed"
    assert len(result.window_plans) <= 2


@pytest.mark.slow
def test_on_ramp_merge_tapers_and_completes():
    cfg = load_config("configs/scenarios/lane/lane_merge.yaml")
    env = build_lane_merge_environment(cfg)
    ramp = env.metadata["ramp"]
    road = env.metadata["road"]
    assert ramp["start_x"] < ramp["end_x"]
    assert env.metadata["traffic"][0]["y"] == road["lane_divider"] * 2
    result = lane.run(
        "configs/scenarios/lane/lane_merge.yaml", show=False, save=False
    )
    assert result.stopped_reason == "success"
    assert len(result.window_plans) <= cfg["T_SIM"]
    assert float(result.states[-1].ego_true[1]) >= road["lane_divider"]
    lower, upper = result.window_plans[-1].hard_interval
    assert 0 <= lower <= upper + 1e-6 <= 1 + 1e-6
    _assert_settled_in_lane(result, env.metadata["task"], cfg)


@pytest.mark.slow
def test_c_open_traffic_completes_merge_branch():
    cfg = load_config("configs/scenarios/lane/lane_merge.yaml")
    cfg["task"]["spec"] = "merge_or_yield"
    cfg["task"]["yield"] = load_config(
        "configs/scenarios/lane/lane_merge_or_yield.yaml"
    )["task"]["yield"]
    s = lane.build(cfg, device="cpu")
    result = lane.trial(s)
    assert result.stopped_reason == "success"
    assert result.states[-1].true_completed is not None
    assert result.states[-1].true_yield_completed is None
    _assert_settled_in_lane(result, s.env.metadata["task"], cfg)


def _assert_settled_in_lane(result, task, cfg):
    """The run kept going past the dwell and left the ego settled in lane."""
    final = result.states[-1]
    assert final.true_completed is not None
    assert len(result.window_plans) == (
        final.true_completed + task["settle_steps"]
    )
    y, vy = float(final.ego_true[1]), float(final.ego_true[3])
    assert abs(y - task["target_center"]) <= task["target_tolerance"]
    assert abs(vy) <= task["settle_lateral_speed"] + 0.1
    top = cfg["road"]["y_max"] - cfg["ego_vehicle"]["height"] / 2
    plan = result.window_plans[-1].rollout.aux["mean_trace"][0]
    assert float(plan[:, 1].max()) <= top


def test_lane_figures_have_time_labels_red_traffic_and_final_score(
    tmp_path,
):
    config = load_config("configs/scenarios/lane/lane_change.yaml")
    config["H"] = 8
    config["T_SIM"] = 1
    config["task"]["start_window_seconds"] = [0.2, 1.0]
    config["task"]["dwell_seconds"] = 0.2
    config["planner"]["max_iters"] = 1
    path = tmp_path / "lane_change.yaml"
    path.write_text(yaml.safe_dump(config))
    result = lane.run(str(path), max_steps=1, show=False, save=False)
    problem = lane.build(config, device="cpu")
    figures = plot_lane_merge(result, problem.env, dt=config["dt"])
    trajectory = figures["combined"][1][0]
    trajectory_text = {text.get_text() for text in trajectory.texts}
    assert "t=0 s" in trajectory_text
    # The map frames the ego path and every car's final position.
    left, right = trajectory.get_xlim()
    cars = result.states[-1].traffic_true[:, 0]
    assert left <= min(0.0, float(cars.min()))
    assert right >= float(cars.max())
    traffic = [
        patch
        for patch in trajectory.patches
        if patch.get_label() in {"Traffic", "_nolegend_"}
    ]
    expected = colors.to_rgb(COLORS["traffic"])
    assert traffic and all(
        patch.get_facecolor()[:3] == expected for patch in traffic
    )
    combined_axes = figures["combined"][1]
    score_labels = {
        text.get_text() for text in combined_axes[1].get_legend().get_texts()
    }
    assert {RHO_INTERVAL, f"smooth {RHO_LOWER}"} <= score_labels
    assert {"Dwell-start window", "Completion deadline"} <= score_labels
    assert any(
        text.get_text().startswith("last window")
        for text in combined_axes[1].texts
    )
    animation_fig, animation_axes, movie = animate_mpc(
        result, problem.env, dt=config["dt"]
    )
    animation_anchor = (
        animation_axes[0]
        .get_legend()
        .get_bbox_to_anchor()
        .transformed(animation_axes[0].transAxes.inverted())
    )
    assert animation_anchor.y0 > 1
    movie._draw_was_started = True
    plt.close(animation_fig)
    plt.close("all")


@pytest.mark.parametrize(
    "rear_past_gore,y,inside",
    [
        (-0.01, 0.9, True),  # in the ramp lane: fine before the gore
        (-0.01, 0.91, False),  # reaches the main lane before the gore
        (0.0, 0.91, True),  # whole car past the gore
        (0.0, 3.6, True),
    ],
)
def test_gore_forbids_the_main_lane_upstream(rear_past_gore, y, inside):
    env = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_merge.yaml")
    )
    half_width = env.metadata["ego_vehicle"]["width"] / 2
    x = env.metadata["ramp"]["gore_x"] + half_width + rear_past_gore
    assert lane_contains_footprint(env, x, y) == inside
    # The planner's road atom agrees with the truth check.
    ego = torch.tensor([[[x, y, 0.0, 0.0]]])
    road = lane_road_predicate(env.metadata)
    traffic = torch.full((1, 1, len(env.metadata["traffic"]), 4), -100.0)
    trace = road(_deterministic_lane_beliefs(env, ego, traffic))
    assert trace[0, 0, 0].item() == float(inside)


def test_ramp_without_gore_keeps_the_old_road():
    cfg = load_config("configs/scenarios/lane/lane_merge.yaml")
    del cfg["ramp"]["gore_x"]
    env = build_lane_merge_environment(cfg)
    assert lane_contains_footprint(env, 0.0, 0.91)
    cfg["ramp"]["gore_x"] = cfg["ramp"]["start_x"] + 1.0
    with pytest.raises(ValueError, match="gore_x <= start_x"):
        build_lane_merge_environment(cfg)


def test_merge_shades_only_the_taper_triangle():
    config = load_config("configs/scenarios/lane/lane_merge.yaml")
    problem = lane.build(config, device="cpu")
    fig, ax = plt.subplots()
    _draw_environment(ax, problem.env, lane=True)
    shaded = [
        collection
        for collection in ax.collections
        if collection.get_label() == "Non-drivable"
    ]
    assert len(shaded) == 1
    vertices = shaded[0].get_paths()[0].vertices
    assert vertices[:, 0].min() == config["ramp"]["start_x"]
    assert vertices[:, 0].max() == config["ramp"]["end_x"]
    plt.close(fig)


def test_live_map_legend_is_outside_axes():
    config = load_config("configs/scenarios/lane/lane_change.yaml")
    problem = lane.build(config, device="cpu")
    fig, axes, _ = create_live_view(
        problem.env,
        lane.initial_state_of(problem),
        dt=config["dt"],
        dynamics=problem.dyn,
    )
    anchor = (
        axes[0]
        .get_legend()
        .get_bbox_to_anchor()
        .transformed(axes[0].transAxes.inverted())
    )
    assert anchor.y0 > 1
    plt.close(fig)


# --- Clean-up: warm tail, yield rule, braking, warm starts, fallback, plots ---


def test_zero_warm_tail_appends_a_zero_control():
    controls = torch.tensor([[1.0, 0.5], [2.0, -0.3], [3.0, -0.3]])
    repeat = Planner.shift_controls(controls)
    zero = Planner.shift_controls(controls, "zero")
    torch.testing.assert_close(repeat[-1], controls[-1])
    torch.testing.assert_close(zero[-1], torch.zeros(2))
    torch.testing.assert_close(zero[:2], controls[1:])
    for name in ("merge", "headway", "merge_or_yield", "merge_or_yield_gap"):
        cfg = load_config(f"configs/scenarios/lane/lane_{name}.yaml")
        assert lane.build(cfg, device="cpu").planner.cfg["warm_tail"] == "zero"
    with pytest.raises(ValueError, match="warm_tail"):
        cfg["planner"]["warm_tail"] = "hold"
        lane.build(cfg, device="cpu")


def test_yield_branch_is_one_always_from_the_last_start():
    cfg = load_config("configs/scenarios/lane/lane_merge_or_yield.yaml")
    env = build_lane_merge_environment(cfg)
    parts = lane_subformulas(lane_local_window(env, 0), cfg["H"])
    task = env.metadata["task"]["yield"]
    last = task["start_end_steps"][1] - task["dwell_steps"]
    choice = parts["overall"].subformula2  # safe AND (merge OR yield)
    assert type(choice).__name__ == "Or"
    yield_branch = choice.subformula2
    assert type(yield_branch).__name__ == "Always"
    assert yield_branch.interval == [last, cfg["H"]]


def test_scripted_braking_slows_only_the_true_car_and_never_reverses():
    traffic = [
        {"name": "lead", "brake": {"at": 1.0, "duration": 0.5, "decel": 2.0}},
        {"name": "other"},
    ]
    assert lane_traffic_acceleration(traffic, 0.8) == [0.0, 0.0]
    assert lane_traffic_acceleration(traffic, 1.0) == [-2.0, 0.0]
    assert lane_traffic_acceleration(traffic, 1.5) == [0.0, 0.0]
    cfg = load_config("configs/scenarios/lane/lane_merge.yaml")
    s = lane.build(cfg, device="cpu")
    state = lane.initial_state_of(s, generator=lane._generator(s))
    brake = cfg["traffic"][0]["brake"]
    step = round(brake["at"] / cfg["dt"])
    change = lane._scripted_braking(s, state, step)
    assert change[0, 2] == pytest.approx(-brake["decel"] * cfg["dt"])
    assert torch.all(change[1:] == 0)
    stopped = state._replace(traffic_true=state.traffic_true * 0)
    assert torch.all(lane._scripted_braking(s, stopped, step)[:, 2] == 0)
    bad = load_config("configs/scenarios/lane/lane_merge.yaml")
    bad["traffic"][0]["brake"] = {"at": 1.0}
    with pytest.raises(ValueError, match="brake"):
        build_lane_merge_environment(bad)


def test_best_first_guess_keeps_the_better_plan_and_ties_go_first():
    cfg = load_config("configs/scenarios/lane/lane_merge.yaml")
    s = lane.build(cfg, device="cpu")
    scores = iter([0.2, 0.9, 0.5, 0.5])

    def fake(rollout, *, spec, init_guess):
        return SimpleNamespace(
            hard_interval=(next(scores), 1.0),
            smooth_lower=0.0,
            controls=init_guess,
        )

    s.planner.optimize_window = fake
    first = {"longitudinal": 0.1, "lateral": 0.0, "switch_step": 5}
    second = {"longitudinal": -0.2, "lateral": 0.0, "switch_step": 5}
    s.cfg["warm_starts"] = [first, second]
    pick = lane.best_first_guess(s, None, lambda *a: None, lambda *a: None)
    torch.testing.assert_close(pick, lane.initial_controls(s, second))
    pick = lane.best_first_guess(s, None, lambda *a: None, lambda *a: None)
    torch.testing.assert_close(pick, lane.initial_controls(s, first))


def test_c_missed_estimate_plans_to_stay_instead_of_failing():
    cfg = load_config("configs/scenarios/lane/lane_merge.yaml")
    cfg["task"]["spec"] = "merge_or_yield"
    cfg["task"]["yield"] = load_config(
        "configs/scenarios/lane/lane_merge_or_yield.yaml"
    )["task"]["yield"]
    s = lane.build(cfg, device="cpu")
    task = s.env.metadata["task"]
    in_lane = torch.tensor([75.0, 3.6, 15.0, 0.0])
    state = lane.initial_state_of(s)._replace(
        ego_mean=in_lane, ego_true=in_lane, streak=0, completed=None
    )
    spec = lane._specification(s, None, state, task["start_end_steps"][1])
    assert "Lateral speed settled" in str(spec)


def test_snapshots_name_every_car_and_show_the_lead_gap():
    cfg = load_config("configs/scenarios/lane/lane_headway.yaml")
    cfg["planner"]["max_iters"] = 1
    s = lane.build(cfg, device="cpu")
    result = lane.trial(s, max_steps=2)
    _fig, axes = plot_lane_snapshots(
        result, s.env, dt=cfg["dt"], number=3, view=(-120.0, 120.0)
    )
    names = {car["name"] for car in cfg["traffic"]}
    assert len(names) == 5
    for ax in axes:
        assert names <= {text.get_text() for text in ax.texts}
        assert "gap to lead" in ax.get_title(loc="left")
    figures = plot_lane_merge(result, s.env, dt=cfg["dt"])
    left, right = figures["combined"][1][0].get_xlim()
    final = result.states[-1].traffic_true[:, 0]
    assert left <= float(final.min()) and right >= float(final.max())


def test_credited_dwell_may_finish_later_so_settling_can_follow():
    # In the lane with one dwell step done, still drifting sideways when the
    # earliest finish arrives, settled two steps later: a later finish works.
    cfg = load_config("configs/scenarios/lane/lane_merge.yaml")
    env = build_lane_merge_environment(cfg)
    task = env.metadata["task"]
    start, _ = task["start_end_steps"]
    steps = 12
    ego = torch.zeros(1, steps, 4)
    ego[..., 1] = task["target_center"]
    earliest = task["dwell_steps"]
    ego[0, : earliest + 2, 3] = task["settle_lateral_speed"] + 0.2
    traffic = torch.zeros(1, steps, len(env.metadata["traffic"]), 4)
    traffic[..., 0] = -100.0
    beliefs = _deterministic_lane_beliefs(env, ego, traffic)
    window = lane_local_window(env, start, streak=1)
    branch = lane_subformulas(window, steps - 1)["branch"]
    assert branch.probability_interval(beliefs)[0] == 1


@pytest.mark.parametrize("accel_std,tracks", [(0.0, False), (2.0, True)])
def test_filter_follows_a_braking_lead_only_with_acceleration_noise(
    accel_std, tracks
):
    cfg = load_config("configs/scenarios/lane/lane_merge.yaml")
    cfg["traffic_filter_accel_std"] = accel_std
    s = lane.build(cfg, device="cpu")
    state = lane.initial_state_of(s, generator=lane._generator(s))
    state = state._replace(traffic_true=state.traffic_mean.clone())
    steps, cars = 35, len(cfg["traffic"])
    traffic_q = cfg["traffic_q_std"] ** 2 * torch.eye(4)
    execute = lane._executor(
        s,
        traffic_q,
        torch.zeros(steps, 4),
        torch.zeros(steps, cars, 4),
        False,
        ego_observation_noise=torch.zeros(steps, 4),
        traffic_observation_noise=torch.zeros(steps, cars, 4),
        filter_q=lane._filter_q(s, traffic_q),
    )
    for step in range(steps):  # the lead brakes from 4 s to 6 s
        state = execute(state, torch.zeros(2), step)
    error = abs(float(state.traffic_mean[0, 2] - state.traffic_true[0, 2]))
    assert (error < 0.3) == tracks
    # Without the setting the filter noise is the true traffic noise.
    if accel_std == 0.0:
        torch.testing.assert_close(lane._filter_q(s, traffic_q), traffic_q)


def _or_over_starts_branch(metadata, horizon):
    """The merge branch as built before the Eventually form: one Or per start.

    Args:
        metadata: Lane metadata of an uncredited local window.
        horizon: Last step of the planning window.

    Returns:
        OR_s (G[s,s+d] lane AND G[s+d,s+d] before_end
              AND G[s+d,H] (lane AND settled)).
    """
    task = metadata["task"]
    step, dwell = metadata["step"], task["dwell_steps"]
    first, deadline = task["start_end_steps"]
    center, tolerance = task["target_center"], task["target_tolerance"]
    target = AxisInterval(center - tolerance, center + tolerance, dim=1)
    speed = task["settle_lateral_speed"]
    settled = AxisInterval(-speed, speed, dim=3)
    front = metadata["ramp"]["end_x"] - metadata["ego_vehicle"]["width"] / 2
    before_end = LessThan(front, dim=0)
    candidates = []
    for start in range(max(0, first - step), deadline - dwell - step + 1):
        finish = start + dwell
        completion = Always(target, [start, finish]) & Always(
            before_end, [finish, finish]
        )
        candidates.append(
            completion & Always(target & settled, [finish, horizon])
        )
    return reduce(Or, candidates)


def _or_over_finishes_branch(metadata, horizon):
    """The in-lane merge branch as built before the Until form.

    Args:
        metadata: Lane metadata of a local window with dwell credit.
        horizon: Last step of the planning window.

    Returns:
        OR_f (G[0,f] lane AND G[f,f] before_end
              AND G[f,H] (lane AND settled)).
    """
    task = metadata["task"]
    step, dwell = metadata["step"], task["dwell_steps"]
    first, deadline = task["start_end_steps"]
    credit = min(metadata["streak"], step - first + 1, dwell + 1)
    center, tolerance = task["target_center"], task["target_tolerance"]
    target = AxisInterval(center - tolerance, center + tolerance, dim=1)
    speed = task["settle_lateral_speed"]
    settled = AxisInterval(-speed, speed, dim=3)
    front = metadata["ramp"]["end_x"] - metadata["ego_vehicle"]["width"] / 2
    before_end = LessThan(front, dim=0)
    candidates = []
    last = min(deadline - step, horizon)
    for finish in range(dwell - (credit - 1), last + 1):
        completion = Always(target, [0, finish]) & Always(
            before_end, [finish, finish]
        )
        candidates.append(
            completion & Always(target & settled, [finish, horizon])
        )
    return reduce(Or, candidates)


def _random_beliefs(env, generator, horizon, in_lane):
    """Random ego beliefs near the lane and the ramp end, traffic far away.

    Args:
        env: Lane environment.
        generator: Torch generator.
        horizon: Last step of the window.
        in_lane: Stay in the target lane throughout, instead of entering
            it at a random step.

    Returns:
        LaneBeliefTrajectory with one sample.
    """
    steps, cars = horizon + 1, len(env.metadata["traffic"])
    switch = int(torch.randint(0, steps, (1,), generator=generator))
    speed = 1.5 + 1.5 * torch.rand(1, generator=generator)
    ego = torch.zeros(1, steps, 4)
    # x starts 40 m before the ramp-end limit, so before_end matters.
    front = env.metadata["ramp"]["end_x"] - 2.25
    ego[0, :, 0] = front - 40.0 + speed * torch.arange(steps)
    ego[0, 0 if in_lane else switch :, 1] = env.metadata["task"][
        "target_center"
    ]
    ego[..., 1] += 0.4 * torch.randn(1, steps, generator=generator)
    ego[..., 3] = 0.3 * torch.randn(1, steps, generator=generator)
    variance = 0.01 + 0.3 * torch.rand(1, steps, 4, generator=generator)
    return LaneBeliefTrajectory(
        ego,
        torch.diag_embed(variance),
        torch.full((1, steps, cars, 4), -100.0),
        torch.zeros(1, steps, cars, 4, 4),
        [car["name"] for car in env.metadata["traffic"]],
    )


@pytest.mark.parametrize("step, streak", [(14, 2), (16, 4), (20, 1)])
def test_in_lane_until_has_the_or_over_finishes_lower_bound(step, streak):
    env = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_merge.yaml")
    )
    horizon = 30
    window = lane_local_window(env, step, streak)
    new = lane_subformulas(window, horizon)["branch"]
    old = _or_over_finishes_branch(window.metadata, horizon)
    generator = torch.Generator().manual_seed(step)
    lowers = []
    for _ in range(24):
        beliefs = _random_beliefs(env, generator, horizon, in_lane=True)
        lower, upper = new.probability_interval(beliefs).tolist()
        assert lower == pytest.approx(
            old.probability_interval(beliefs)[0].item(), abs=1e-6
        )
        assert upper >= lower
        lowers.append(lower)
    assert max(lowers) > 0.1  # the comparison is not only zeros


@pytest.mark.parametrize("step", [0, 6, 14])
def test_merge_eventually_has_the_or_over_starts_lower_bound(step):
    env = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_merge.yaml")
    )
    horizon = 30
    window = lane_local_window(env, step)
    new = lane_subformulas(window, horizon)["branch"]
    old = _or_over_starts_branch(window.metadata, horizon)
    generator = torch.Generator().manual_seed(step)
    lowers = []
    for _ in range(24):
        # Ego crosses into the lane at a random step, at a random speed.
        beliefs = _random_beliefs(env, generator, horizon, in_lane=False)
        lower, upper = new.probability_interval(beliefs).tolist()
        assert lower == pytest.approx(
            old.probability_interval(beliefs)[0].item(), abs=1e-6
        )
        assert upper >= lower
        lowers.append(lower)
    assert max(lowers) > 0.1  # the comparison is not only zeros


@pytest.mark.parametrize("alpha", [0.95, None])
def test_batch_runs_each_seed_and_saves_one_table(
    lane_config, tmp_path, monkeypatch, alpha
):
    from planning import runners

    cfg, path = lane_config
    cfg["planner"]["alpha"] = alpha
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    trials, table = lane.batch(str(path), seeds=[0, 1], max_steps=2)
    assert [row["seed"] for row in trials] == [0, 1]
    assert table["trials"] == 2
    assert sum(table["outcomes"].values()) == 2
    assert len(trials[0]["replan_ms"]) == len(trials[0]["window_scores"])
    # Branch counts cover every trial; None is stored as "none".
    assert set(table["branches"]) == {"merge", "yield", "none"}
    assert sum(table["branches"].values()) == 2
    # Fraction over all windows of all trials, not a mean of trial means.
    scores = trials[0]["window_scores"] + trials[1]["window_scores"]
    expected = (
        None
        if alpha is None
        else sum(score >= alpha for score in scores) / len(scores)
    )
    assert table["windows meeting alpha"] == pytest.approx(expected)
    machine = table["machine"]
    assert machine["torch"] == torch.__version__
    assert machine["H"] == cfg["H"]
    assert machine["max_iters"] == cfg["planner"]["max_iters"]
    assert (machine["gpu"] is None) == (machine["device"] == "cpu")
    # What the timings ran on: threads and the CPUs the process was allowed.
    assert machine["threads"] == torch.get_num_threads()
    assert machine["allowed cpus"] == sorted(os.sched_getaffinity(0))
    assert machine["cpu"]
    saved = (tmp_path / "lane_change_batch.json").read_text()
    assert '"table"' in saved and '"trials"' in saved


def test_paper_figure_is_full_width_to_scale_and_table_is_latex(
    tmp_path, monkeypatch
):
    import json

    from planning import runners
    from visualization.figures import (
        COLUMN_WIDTH,
        PAPER_WIDTH,
        plot_lane_paper,
    )

    panels = []
    for name, title, _time, ghosts, label in lane.FIGURE_SCENES:
        cfg = load_config(f"configs/scenarios/lane/lane_{name}.yaml")
        cfg["planner"]["max_iters"] = 1
        cfg["warm_starts"] = cfg["warm_starts"][:1]
        s = lane.build(cfg, device="cpu")
        result = lane.trial(s, max_steps=3)
        panels.append((title, result, s.env, 2, ghosts, label))
    fig, axes = plot_lane_paper(panels, dt=cfg["dt"], alpha=0.95)
    width, height = fig.get_size_inches()
    assert width == PAPER_WIDTH and height < 2.8
    assert len(axes) == 4  # three frames and the score panel
    # Every frame shows the plan; the 95% region only when asked for.
    for ax in axes[:3]:
        labels = set(ax.get_legend_handles_labels()[1])
        assert {"Current plan", "Ego", "Traffic"} <= labels
        assert "Predicted 95% region" not in labels
    # Ghost at 0 s (earlier than the frame) plus the frame ego, each tagged
    # above the car, outside it.
    tags = [t for t in axes[1].texts if t.get_text()]  # arrows have no text
    assert [t.get_text() for t in tags] == ["0 s", "0.4 s"]
    ego_y = float(panels[1][1].states[2].ego_true[1])
    assert tags[1].get_position()[1] > ego_y + 0.9  # half the car height
    # One short title line per frame; the formulas go in the caption.
    assert [ax.get_title(loc="left") for ax in axes[:3]] == [
        row[1] for row in lane.FIGURE_SCENES
    ]
    # The score panel has one line per run, labelled as given.
    score_labels = axes[3].get_legend_handles_labels()[1]
    assert score_labels[:3] == [row[4] for row in lane.FIGURE_SCENES]
    _fig, axes = plot_lane_paper(panels, dt=cfg["dt"], show_uncertainty=True)
    assert "Predicted 95% region" in axes[0].get_legend_handles_labels()[1]
    # Same metres per inch in every frame (to scale across frames).
    scales = [
        (ax.get_xlim()[1] - ax.get_xlim()[0])
        / (ax.get_position().width * width)
        for ax in axes[:3]
    ]
    assert max(scales) == pytest.approx(min(scales))
    # One-column version: column width, score panel under the frames, and
    # still one scale across all frames.
    fig, axes = plot_lane_paper(panels, dt=cfg["dt"], column=True)
    width, height = fig.get_size_inches()
    assert width == COLUMN_WIDTH and height < 3.8
    assert axes[3].get_position().y1 < axes[1].get_position().y0
    scales = [
        (ax.get_xlim()[1] - ax.get_xlim()[0])
        / (ax.get_position().width * width)
        for ax in axes[:3]
    ]
    assert max(scales) == pytest.approx(min(scales))
    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    for name, _row in lane.TABLE_SCENES:
        table = {
            "trials": 5,
            "outcomes": {"success": 3, "yielded": 1, "collision": 1},
            "branches": {"merge": 3, "yield": 1, "none": 1},
            "minimum clearance [m]": 3.21,
            "minimum lead gap in lane [m]": None,
            "lowest window score": 0.9,
            "windows meeting alpha": 0.98765,
            "first replan median [ms]": 512.4,
            "later replans median [ms]": 120.0,
            "later replans p95 [ms]": 180.0,
        }
        if name == "merge":
            table["windows meeting alpha"] = None  # alpha not set
        path = tmp_path / f"lane_{name}_batch.json"
        path.write_text(json.dumps({"table": table, "trials": []}))
    text = lane.latex_table()
    assert text.startswith(r"\begin{tabular}{lccccccc}")
    assert r"Merged/yielded" in text
    assert r"$\widehat{r}^{\varphi,\downarrow}_0 \geq \alpha$ (\%)" in text
    assert (
        "Merge + headway & 4/5 & 3/1 & 3.2 & -- & 0.90 & 98.8 & 120 / 180"
        in text
    )
    assert "Merge & 4/5 & 3/1 & 3.2 & -- & 0.90 & -- & 120 / 180" in text
    # The first-replan median is a comment under the table, not a column.
    last = text.splitlines()[-1]
    assert last.startswith("% first replan median: Merge 512 ms")
    assert text.index(r"\end{tabular}") < text.index(last)
    assert (tmp_path / "lane_table.tex").exists()
    with pytest.raises(FileNotFoundError, match="run lane_merge.yaml"):
        lane.paper_figure()


def test_figure_step_is_the_time_or_when_the_task_was_done():
    def run(done, yielded, steps=10):
        final = SimpleNamespace(
            true_completed=done, true_yield_completed=yielded
        )
        return SimpleNamespace(states=[None] * (steps - 1) + [final])

    assert lane._figure_step(run(None, None), 0.2, 1.0) == 5
    assert lane._figure_step(run(None, None), 0.2, 9.0) == 9  # clipped
    assert lane._figure_step(run(6, None), 0.2, None) == 6
    assert lane._figure_step(run(None, 4), 0.2, None) == 4
    assert lane._figure_step(run(None, None), 0.2, None) == 9  # never done
