"""Lane change and merge: semantics, planning, and figures."""

from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pytest
import torch
import yaml
from PIL import Image

from models.beliefs import GaussianBeliefTrajectory
from models.rollouts import LaneBeliefTrajectory, lane_rollout
from pdstl.predicates import AxisInterval, RelativeAxisInterval
from experiments import lane
from planning.environment import (
    build_lane_merge_environment,
    lane_contains_footprint,
    lane_goal_reached,
    lane_has_collision,
    lane_local_window,
    lane_subformulas,
)
from planning.planner import IterationRecord
from planning.runners import save_result
from utils import load_config
from visualization.animation import animate_mpc
from visualization.live_plots import create_live_view
from visualization.figures import (
    COLORS,
    RHO_INTERVAL,
    RHO_LOWER,
    _draw_environment,
    plot_lane_merge,
)


def run_lane(path, *, show=False, save=False, **kwargs):
    """The main.py flow: config -> build -> simulate -> output."""
    cfg = load_config(path)
    stem = Path(path).stem
    scenario = lane.build(cfg)
    result = lane.simulate(scenario, **kwargs)
    if show or save:
        lane.present(result, scenario, stem=stem, show=show, save=save)
    save_result(result, stem, save)
    return result


def test_collision_uses_footprints_and_margins():
    env = build_lane_merge_environment(
        load_config("configs/scenarios/lane/lane_change.yaml")
    )
    ego = torch.tensor([0.0, 0.0])
    traffic = torch.tensor([[5.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    assert lane_has_collision(env, ego, traffic)
    traffic[0, 0] += 0.001
    assert not lane_has_collision(env, ego, traffic)


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
    x = 40.0
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
    traffic = torch.zeros(1, steps, 3, 4)
    traffic[..., 0] = -100.0

    def overall(ego_mean, step):
        beliefs = LaneBeliefTrajectory(
            ego_mean,
            torch.zeros(1, steps, 4, 4),
            traffic,
            torch.zeros(1, steps, 3, 4, 4),
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
    import planning.runners as runners

    cfg, path = lane_config
    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    result = run_lane(str(path), max_steps=1, show=False, save=True)
    assert (tmp_path / "lane_change.pt").exists()
    assert (tmp_path / "lane_change.gif").exists()
    for stem in ("lane_change", "lane_change_trajectory"):
        for suffix in (".png", ".pdf"):
            assert (tmp_path / f"{stem}{suffix}").exists()
    assert not (tmp_path / "lane_change_controls.png").exists()
    assert not (tmp_path / "lane_change_scores.png").exists()
    with Image.open(tmp_path / "lane_change.gif") as movie:
        assert movie.n_frames >= 2
        first = movie.convert("RGB").tobytes()
        movie.seek(movie.n_frames - 1)
        assert movie.convert("RGB").tobytes() != first

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

    viewed = run_lane(
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


def test_default_lane_completes_with_physical_controls():
    cfg = load_config("configs/scenarios/lane/lane_change.yaml")
    result = run_lane(
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
    result = run_lane(str(path), max_steps=5, show=False, save=False)
    assert result.stopped_reason == "deadline_missed"
    assert len(result.window_plans) <= 2


def test_on_ramp_merge_tapers_and_completes():
    cfg = load_config("configs/scenarios/lane/lane_merge.yaml")
    env = build_lane_merge_environment(cfg)
    ramp = env.metadata["ramp"]
    road = env.metadata["road"]
    assert ramp["start_x"] < ramp["end_x"]
    assert env.metadata["traffic"][0]["y"] == road["lane_divider"] * 2
    result = run_lane(
        "configs/scenarios/lane/lane_merge.yaml", show=False, save=False
    )
    assert result.stopped_reason == "success"
    assert len(result.window_plans) <= cfg["T_SIM"]
    assert float(result.states[-1].ego_true[1]) >= road["lane_divider"]
    lower, upper = result.window_plans[-1].hard_interval
    assert 0 <= lower <= upper + 1e-6 <= 1 + 1e-6
    _assert_settled_in_lane(result, env.metadata["task"], cfg)


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
    result = run_lane(str(path), max_steps=1, show=False, save=False)
    problem = lane.build(config, device="cpu")
    figures = plot_lane_merge(result, problem.env, dt=config["dt"])
    trajectory = figures["trajectory"][1]
    trajectory_text = {text.get_text() for text in trajectory.texts}
    assert "t=0 s" in trajectory_text
    assert any("last window" in text for text in trajectory_text)
    assert -3.0 < trajectory.get_xlim()[0] <= 0.0
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
