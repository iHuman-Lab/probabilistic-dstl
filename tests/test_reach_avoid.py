"""The generic reach-avoid scenario and its two stlpy examples."""

import math
from functools import reduce
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from scipy.stats import chi2

from experiments import reach_avoid
from models.rollouts import create_gaussian_belief_trajectory
from pdstl.operators import Always, And, Eventually, Or
from planning.environment import (
    Environment,
    RectangleRegion,
    build_reach_avoid_environment,
    inside,
    outside,
    shortest_route,
)
from planning.planner import IterationRecord
from planning.runners import monte_carlo, save_result
from visualization.animation import animate_reach_avoid
from visualization.live_plots import create_reach_avoid_live_view
from utils import load_config
from visualization.figures import RHO_LOWER, plot_reach_avoid

SCENARIOS = {
    name: f"configs/scenarios/reach_avoid/{name}.yaml"
    for name in ("obstacle", "narrow_passage", "either_or")
}
CONFIG = {
    "bounds": {"x_range": [0, 10], "y_range": [0, 10]},
    "obstacles": [{"name": "wall", "x_range": [3, 5], "y_range": [4, 6]}],
    "goal": {
        "any_of": [
            {"name": "a", "x_range": [7, 8], "y_range": [8, 9]},
            {"name": "b", "x_range": [8, 9], "y_range": [1, 2]},
        ]
    },
    "visit_regions": [
        {
            "dwell": 2,
            "any_of": [
                {"name": "t1", "x_range": [1, 2], "y_range": [6, 7]},
                {"name": "t2", "x_range": [7, 8], "y_range": [4, 5]},
            ],
        }
    ],
}


def run_reach_avoid(path, *, show=False, save=False, **plan):
    """The main.py flow: config -> build -> optimize -> validate -> output."""
    cfg = load_config(path)
    stem = Path(path).stem
    scenario = reach_avoid.build(cfg)
    result = reach_avoid.optimize(scenario, **plan)
    reach_avoid.validate(scenario, result)
    if show or save:
        reach_avoid.present(result, scenario, stem=stem, show=show, save=save)
    save_result(result, stem, save)
    return result


def _problem(name, **planner):
    problem = reach_avoid.build(load_config(SCENARIOS[name]), device="cpu")
    problem.planner.cfg.update(planner)
    return problem


def _inside(points, region):
    return (
        (points[:, 0] >= region.xmin)
        & (points[:, 0] <= region.xmax)
        & (points[:, 1] >= region.ymin)
        & (points[:, 1] <= region.ymax)
    )


@pytest.fixture(scope="module")
def short():
    """A 20-iteration either-or solve from its route."""
    problem = _problem("either_or", max_iters=20)
    spec = problem.spec
    result = problem.planner.optimize_window(
        problem.rollout, spec=spec, init_guess=problem.init_guess
    )
    return problem, spec, result


# --- Environment and specification -------------------------------------------


def test_regions_validate_their_role_bounds_and_names():
    with pytest.raises(ValueError, match="role must be one of"):
        RectangleRegion("r", "lava", 0, 1, 0, 1)
    with pytest.raises(ValueError, match="min < max"):
        RectangleRegion("r", "goal", 1, 1, 0, 1)
    environment = Environment()
    environment.add_region(RectangleRegion("g", "goal", 0, 1, 0, 1))
    with pytest.raises(ValueError, match="already exists"):
        environment.add_region(RectangleRegion("g", "goal", 2, 3, 2, 3))
    with pytest.raises(ValueError, match="no region named"):
        environment.region("nowhere")


def test_builder_reads_bounds_obstacles_goal_and_visit_regions():
    environment = build_reach_avoid_environment(CONFIG)
    roles = {name: r.role for name, r in environment.regions.items()}
    assert roles == {
        "bounds": "workspace",
        "wall": "obstacle",
        "a": "goal",
        "b": "goal",
        "t1": "target",
        "t2": "target",
    }
    assert environment.goal == (None, ["a", "b"])
    assert environment.visits == [(None, 2, ["t1", "t2"])]


def test_unnamed_regions_get_default_names():
    environment = Environment()
    environment.set_bounds([0, 10], [0, 10])
    environment.add_obstacle([1, 2], [1, 2])
    environment.set_goal([8, 9], [8, 9], interval=[5, 10])
    environment.add_visit_region(
        any_of=[
            {"x_range": [3, 4], "y_range": [6, 7]},
            {"x_range": [6, 7], "y_range": [3, 4]},
        ],
        dwell=2,
    )
    assert list(environment.regions) == [
        "bounds",
        "obstacle 1",
        "goal",
        "visit 1 a",
        "visit 1 b",
    ]
    assert environment.goal == ([5, 10], ["goal"])


def test_specification_is_built_from_the_config():
    environment = build_reach_avoid_environment(CONFIG)
    region = environment.region
    H = 12
    safe = And(inside(region("bounds")), outside(region("wall")))
    stays = Or(
        Always(inside(region("t1")), [0, 2]),
        Always(inside(region("t2")), [0, 2]),
    )
    expected = reduce(
        And,
        [
            Always(safe, [1, H]),
            Eventually(Or(inside(region("a")), inside(region("b"))), [1, H]),
            Eventually(stays, [1, H - 2]),
        ],
    )
    steps = torch.linspace(0, 1, H + 1).unsqueeze(-1)
    mean = torch.tensor([1.5, 1.5]) + steps * torch.tensor([6.0, 7.0])
    beliefs = create_gaussian_belief_trajectory(
        mean, torch.full_like(mean, 0.05)
    )
    spec = environment.get_specification(H)
    torch.testing.assert_close(
        spec.probability_interval(beliefs),
        expected.probability_interval(beliefs),
    )
    torch.testing.assert_close(
        spec.smooth_lower(beliefs, 50.0), expected.smooth_lower(beliefs, 50.0)
    )
    with pytest.raises(ValueError, match="horizon must be a positive"):
        environment.get_specification(0)


def test_goal_and_visit_intervals_set_the_time_windows():
    config = {
        **CONFIG,
        "goal": {"x_range": [7, 8], "y_range": [8, 9], "interval": [8, 12]},
        "visit_regions": [
            {
                "x_range": [1, 2],
                "y_range": [6, 7],
                "dwell": 2,
                "interval": [2, 5],
            }
        ],
    }
    environment = build_reach_avoid_environment(config)
    region = environment.region
    H = 12
    safe = And(inside(region("bounds")), outside(region("wall")))
    expected = reduce(
        And,
        [
            Always(safe, [1, H]),
            Eventually(inside(region("goal")), [8, 12]),
            Eventually(Always(inside(region("visit 1")), [0, 2]), [2, 5]),
        ],
    )
    steps = torch.linspace(0, 1, H + 1).unsqueeze(-1)
    mean = torch.tensor([1.5, 1.5]) + steps * torch.tensor([6.0, 7.0])
    beliefs = create_gaussian_belief_trajectory(
        mean, torch.full_like(mean, 0.05)
    )
    torch.testing.assert_close(
        environment.get_specification(H).probability_interval(beliefs),
        expected.probability_interval(beliefs),
    )
    config["visit_regions"][0]["interval"] = [2, 11]
    with pytest.raises(ValueError, match="must fit"):
        build_reach_avoid_environment(config).get_specification(H)


# --- Examples ------------------------------------------------------------------


def _polyline(start, route, spacing=0.02):
    corners = torch.tensor([start, *route])
    return torch.cat(
        [
            torch.stack([a + s * (b - a) for s in torch.linspace(0, 1, 200)])
            for a, b in zip(corners, corners[1:])
        ]
    )


def _passes(path, point, tolerance=0.1):
    return bool(((path - torch.tensor(point)).norm(dim=1) <= tolerance).any())


def _grown(region, margin):
    return RectangleRegion(
        "grown",
        "obstacle",
        region.xmin - margin,
        region.xmax + margin,
        region.ymin - margin,
        region.ymax + margin,
    )


@pytest.mark.parametrize(
    "obstacles",
    [
        [{"x_range": [3, 5], "y_range": [4, 6]}],
        [
            {"x_range": [0, 6], "y_range": [4, 5]},
            {"x_range": [6, 7], "y_range": [2, 3]},
        ],
    ],
)
def test_route_follows_the_obstacles_and_visits_a_target_and_a_goal(
    obstacles,
):
    environment = build_reach_avoid_environment(
        {**CONFIG, "obstacles": obstacles}
    )
    start = [1.0, 1.0]
    route = shortest_route(environment, start, clearance=0.2, resolution=0.05)
    path = _polyline(start, route)
    for obstacle in environment.by_role("obstacle"):
        assert not _inside(path, _grown(obstacle, 0.15)).any()
    for role in ("target", "goal"):
        regions = environment.by_role(role)
        assert any(_passes(path, region.centre) for region in regions)


def test_route_settings_come_from_the_config(monkeypatch):
    seen = {}

    def spy(environment, start, **settings):
        seen.update(settings)
        return shortest_route(environment, start, **settings)

    monkeypatch.setattr(reach_avoid, "shortest_route", spy)
    cfg = load_config(SCENARIOS["obstacle"])
    cfg["route"] = {"clearance": 0.3, "resolution": 0.1}
    problem = reach_avoid.build(cfg, device="cpu")
    assert seen == {"clearance": 0.3, "resolution": 0.1}
    obstacle = problem.env.by_role("obstacle")[0]
    path = _polyline(cfg["x0_mean"][:2], problem.route)
    assert not _inside(path, _grown(obstacle, 0.25)).any()


def test_route_reports_an_enclosed_goal():
    environment = build_reach_avoid_environment(
        {
            "bounds": {"x_range": [0, 10], "y_range": [0, 10]},
            "obstacles": [
                {"x_range": [5, 6], "y_range": [5, 9]},
                {"x_range": [8, 9], "y_range": [5, 9]},
                {"x_range": [5, 9], "y_range": [8, 9]},
                {"x_range": [5, 9], "y_range": [5, 6]},
            ],
            "goal": {"x_range": [6.8, 7.2], "y_range": [6.8, 7.2]},
        }
    )
    with pytest.raises(ValueError, match="no route"):
        shortest_route(environment, [1.0, 1.0])


@pytest.mark.parametrize("name", SCENARIOS)
def test_initial_guess_tracks_the_route(name):
    problem = _problem(name)
    guess = reach_avoid.straight_line_guess(
        problem.cfg, problem.dyn, problem.route
    )
    torch.testing.assert_close(guess, problem.init_guess)
    assert guess.abs().max() <= problem.dyn.u_max
    plan = problem.planner.evaluate_controls(
        problem.rollout, guess, spec=problem.spec
    )
    end = plan.rollout.aux["mean_trace"][0, -1, :2]
    torch.testing.assert_close(
        end, torch.tensor(problem.route[-1]), atol=0.5, rtol=0
    )


def test_short_solve_is_bounded_improving_and_replayable(short):
    problem, spec, result = short
    initial = problem.planner.evaluate_controls(
        problem.rollout, problem.init_guess, spec=spec
    )
    replay = problem.planner.evaluate_controls(
        problem.rollout, result.controls, spec=spec
    )
    assert result.hard_interval[0] >= initial.hard_interval[0]
    assert result.controls.abs().max() <= problem.dyn.u_max
    assert replay.hard_interval == pytest.approx(result.hard_interval)
    assert len(result.hard_lower_history) == 21


def test_figure_shows_one_plan_and_one_lower_robustness_line(short, tmp_path):
    problem, _, result = short
    fig, (axis, controls, scores) = plot_reach_avoid(
        result,
        problem.env,
        control_unit="m/s²",
        title="Either–Or",
        ellipse_confidence=0.9,
        save_path=str(tmp_path / "plan.png"),
    )
    assert (tmp_path / "plan.png").exists()
    assert (tmp_path / "plan.pdf").exists()
    labels = [t.get_text() for t in axis.get_legend().get_texts()]
    assert labels == ["Obstacle", "Goal", "Target", "Plan", "90% ellipse"]
    assert controls.get_ylabel() == "control [m/s²]"
    assert [line.get_label() for line in scores.lines] == [
        f"{RHO_LOWER} (lower bound)",
        "threshold α = 0.90",
    ]
    assert scores.get_title().startswith(RHO_LOWER)
    ellipse = next(p for p in axis.patches if p.get_label() == "90% ellipse")
    variance = torch.linalg.eigvalsh(
        result.rollout.aux["cov_trace"][0, 0, :2, :2]
    ).max()
    assert ellipse.width == pytest.approx(
        2 * (chi2.ppf(0.9, 2) * variance.item()) ** 0.5, rel=1e-5
    )
    plt.close(fig)


def test_live_view_and_animation(short):
    problem, spec, result = short
    fig, axes, observe, finish = create_reach_avoid_live_view(
        problem.env,
        lambda controls: problem.planner.evaluate_controls(
            problem.rollout, controls, spec=spec
        ),
        dt=problem.cfg["dt"],
        title="Either–Or",
        max_iters=problem.planner.cfg["max_iters"],
        alpha=problem.planner.cfg["alpha"],
    )
    observe(
        0,
        IterationRecord(
            result.controls,
            result.final_loss,
            result.smooth_lower,
            result.hard_interval,
            result.smoothing_beta,
        ),
    )
    assert len(axes[0].lines[0].get_xdata()) == problem.cfg["H"] + 1
    score_line = axes[1].lines[0]
    assert score_line.get_label() == f"{RHO_LOWER} (lower bound)"
    assert len(score_line.get_xdata()) == 1
    # An iterate planned with `select_by: margin` or `last` has no exact
    # score; the live view must still draw it.
    observe(
        1,
        IterationRecord(
            result.controls,
            result.final_loss,
            float("nan"),
            None,
            result.smoothing_beta,
        ),
    )
    assert len(score_line.get_xdata()) == 2
    assert math.isnan(score_line.get_ydata()[-1])
    finish(result)
    assert len(score_line.get_xdata()) == len(result.hard_lower_history)
    plt.close(fig)
    fig, axis, movie = animate_reach_avoid(
        result, problem.env, dt=problem.cfg["dt"], title="Either–Or"
    )
    movie._draw_frame(1)
    assert len(axis.lines[1].get_xdata()) == 2
    movie._draw_was_started = True
    plt.close(fig)


def test_monte_carlo_is_one_for_a_sure_spec_and_zero_for_an_impossible_one(
    short,
):
    problem, _, result = short
    H = problem.cfg["H"]
    anywhere = RectangleRegion("anywhere", "workspace", -99, 99, -99, 99)
    far = RectangleRegion("far", "goal", 50, 51, 50, 51)
    for spec, expected in (
        (Always(inside(anywhere), [0, H]), 1.0),
        (Eventually(inside(far), [0, H]), 0.0),
    ):
        s = SimpleNamespace(dyn=problem.dyn, state=problem.state, spec=spec)
        rate, lower, upper = monte_carlo(s, result, samples=300)
        assert rate == expected
        assert lower <= rate <= upper


def test_figure_shows_the_monte_carlo_rate(short):
    problem, _, result = short
    result.monte_carlo = (0.9, 0.85, 0.95)
    fig, (_, _, scores) = plot_reach_avoid(result, problem.env)
    labels = [t.get_text() for t in scores.get_legend().get_texts()]
    assert r"Monte Carlo $P(\varphi)$" in labels
    result.monte_carlo = None
    plt.close(fig)


def _meets_alpha_and_clear(name, result):
    problem = _problem(name)
    points = result.rollout.aux["mean_trace"][0]
    assert result.threshold_met
    rate, lower, upper = result.monte_carlo
    assert 0 <= lower <= rate <= upper <= 1
    for obstacle in problem.env.by_role("obstacle"):
        assert not _inside(points, obstacle).any(), obstacle.name
    return problem, points


@pytest.mark.slow
def test_obstacle_case_meets_alpha_around_its_obstacle():
    result = run_reach_avoid(SCENARIOS["obstacle"], show=False, save=False)
    problem, points = _meets_alpha_and_clear("obstacle", result)
    assert _inside(points, problem.env.region("goal")).any()


@pytest.mark.slow
def test_narrow_passage_meets_alpha_past_four_obstacles_to_a_goal():
    result = run_reach_avoid(
        SCENARIOS["narrow_passage"], show=False, save=False
    )
    problem, points = _meets_alpha_and_clear("narrow_passage", result)
    goals = problem.env.by_role("goal")
    assert any(_inside(points, goal).any() for goal in goals)


@pytest.mark.slow
def test_either_or_dwells_in_one_target_then_reaches_the_goal():
    result = run_reach_avoid(SCENARIOS["either_or"], show=False, save=False)
    problem, points = _meets_alpha_and_clear("either_or", result)

    def longest_stay(region):
        best = run = 0
        for is_inside in _inside(points, region).tolist():
            run = run + 1 if is_inside else 0
            best = max(best, run)
        return best

    # always[0, 5] spans six samples.
    assert max(map(longest_stay, problem.env.by_role("target"))) >= 6
    assert _inside(points, problem.env.region("goal")).any()
