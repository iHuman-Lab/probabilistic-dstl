"""Two-gap experiment: config, gaps, selection rule and Monte Carlo seed."""

import numpy as np
import pytest
import torch

from experiments import two_gap


@pytest.fixture(scope="module")
def cfg():
    return two_gap.load()


def test_config_loads_with_three_increasing_uncertainty_levels(cfg):
    assert cfg["dynamics"] == "double_integrator"
    assert list(cfg["noise"]) == ["low", "medium", "high"]
    levels = list(cfg["noise"].values())
    assert levels == sorted(levels)
    assert set(cfg["gaps"]) == {"slit", "around"}


def _mean_path(s, controls):
    state, means = s.state[0], [s.state[0]]
    for control in controls:
        state = s.dyn.A @ state + s.dyn.B @ control
        means.append(state)
    return torch.stack(means)


@pytest.mark.parametrize("gap", ["slit", "around"])
def test_each_initial_guess_passes_through_its_gap(cfg, gap):
    s = two_gap.problem(cfg, "stlcg", None)
    controls = two_gap.initial_guess(s.cfg, s.dyn, cfg["gaps"][gap])
    assert controls.shape == (cfg["H"], 2)
    assert controls.abs().max() < s.dyn.u_max
    assert two_gap.gap_used(cfg, _mean_path(s, controls).numpy()) == gap


@pytest.mark.parametrize("gap", ["slit", "around"])
def test_initial_guess_comes_to_rest_in_the_goal_without_overshoot(cfg, gap):
    s = two_gap.problem(cfg, "stlcg", None)
    controls = two_gap.initial_guess(s.cfg, s.dyn, cfg["gaps"][gap])
    path = _mean_path(s, controls)
    goal = cfg["goal"]
    end = cfg["gaps"][gap][-1]
    assert goal["x_range"][0] <= path[-1, 0] <= goal["x_range"][1]
    assert goal["y_range"][0] <= path[-1, 1] <= goal["y_range"][1]
    assert path[-1, 2:].abs().max() < 0.05  # at rest
    assert path[:, 0].max() <= end[0] + 0.1  # no run past the goal


def test_reference_starts_and_ends_at_rest(cfg):
    ref = two_gap.reference(cfg, cfg["gaps"]["slit"])
    assert ref.shape == (cfg["H"] + 1, 2)
    torch.testing.assert_close(ref[0], torch.tensor(cfg["x0_mean"][:2]))
    step = ref[1:] - ref[:-1]
    assert step[0].norm() < 1e-3 and step[-1].norm() < 1e-6


def test_stlcg_plans_the_mean_without_noise_and_pdstl_with_it(cfg):
    stlcg = two_gap.problem(cfg, "stlcg", "high")
    assert torch.count_nonzero(stlcg.dyn.Q) == 0
    assert torch.count_nonzero(stlcg.state[1]) == 0
    pdstl = two_gap.problem(cfg, "pdstl", "high")
    q = cfg["noise"]["high"]
    torch.testing.assert_close(pdstl.dyn.Q, q**2 * torch.eye(4))


def _row(gap, robustness, score, effort):
    return {
        "gap": gap,
        "robustness": robustness,
        "score": score,
        "effort": effort,
    }


def test_selection_meets_the_requirement_then_least_effort(cfg):
    target = cfg["methods"]["pdstl"]["planner"]["score_threshold"]
    margin = cfg["methods"]["stlcg"]["planner"]["margin_target"]
    slit = _row("slit", margin + 0.25, target - 0.05, 10.0)
    around = _row("around", margin + 0.40, target + 0.005, 20.0)
    detour = _row("around", margin + 0.50, target + 0.009, 30.0)
    # STLCG: every plan clears the margin on the mean, so the cheapest.
    assert two_gap.select(cfg, "stlcg", [slit, around, detour]) is slit
    # pdSTL: two plans reach the score threshold; the cheaper of them wins,
    # not the highest score.
    assert two_gap.select(cfg, "pdstl", [slit, around, detour]) is around


def test_selection_keeps_the_best_criterion_when_none_qualifies(cfg):
    a = _row("slit", 0.01, 0.80, 10.0)
    b = _row("around", 0.03, 0.90, 20.0)
    assert two_gap.select(cfg, "stlcg", [a, b]) is b
    assert two_gap.select(cfg, "pdstl", [a, b]) is b


def test_clearance_is_distance_to_the_nearest_block():
    blocks = [{"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}]
    paths = torch.tensor([[[2.0, 0.5, 0, 0], [1.5, 0.5, 0, 0]]])
    torch.testing.assert_close(
        two_gap.clearance(paths, blocks), torch.tensor([0.5])
    )


def test_monte_carlo_draws_are_fixed_by_the_config_seed(cfg):
    assert cfg["monte_carlo"]["seed"] == 7
    s = two_gap.problem(cfg, "stlcg", None)
    controls = two_gap.initial_guess(s.cfg, s.dyn, cfg["gaps"]["slit"])

    def run(seed):
        small = {**cfg, "monte_carlo": {"samples": 200, "seed": seed}}
        return two_gap.evaluate(small, controls, "medium")

    assert run(7) == run(7)  # every plan sees the same draws
    assert run(7)["clearance_median"] != run(8)["clearance_median"]


def test_first_arrival_is_the_first_step_inside_the_goal(cfg):
    goal = cfg["goal"]
    inside = [goal["x_range"][0] + 0.1, goal["y_range"][0] + 0.1]
    outside = [goal["x_range"][1] + 0.5, goal["y_range"][0] + 0.1]
    mean = np.array([outside, outside, inside, outside, inside])
    assert two_gap.first_arrival(cfg, mean) == 2
    assert two_gap.first_arrival(cfg, mean[:2]) is None


def test_effort_is_the_sum_of_squared_controls():
    assert two_gap.effort(torch.tensor([[1.0, 2.0], [0.0, 3.0]])) == 14.0


def test_report_builds_table_and_figure_from_saved_results(
    cfg, tmp_path, monkeypatch
):
    """`report` alone, on results written the way `run` writes them.

    The plans are the initial guesses (no optimization), so the test needs
    no saved outputs and runs in CI.
    """
    monkeypatch.setattr(two_gap, "OUTPUTS", tmp_path)
    s = two_gap.problem(cfg, "stlcg", None)
    guess = {
        g: two_gap.initial_guess(s.cfg, s.dyn, w)
        for g, w in cfg["gaps"].items()
    }
    plans, rows = {}, []
    for method in cfg["methods"]:
        for level in cfg["noise"]:
            gap = "slit" if method == "stlcg" else "around"
            plans[f"{method}/{level}"] = guess[gap]
            rows.append(
                {
                    "method": method,
                    "noise": level,
                    "gap": gap,
                    "effort": two_gap.effort(guess[gap]),
                    "success": 0.9,
                    "ci_low": 0.89,
                    "ci_high": 0.91,
                    "score_eval": 0.8,
                    "clearance_p05": 0.1,
                }
            )
    torch.save(plans, tmp_path / "plans.pt")
    two_gap._write(rows, "selected.csv")
    two_gap.report(cfg)
    for name in ("table.csv", "table.md", "figure.png", "figure.pdf"):
        assert (tmp_path / name).stat().st_size > 0
    assert len((tmp_path / "table.md").read_text().splitlines()) == 2 + 6


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_noise_gives_the_stated_position_uncertainty(cfg, level):
    after = cfg["uncertainty"]["after_s"]
    sigma = cfg["uncertainty"]["position_std_m"][level]
    s = two_gap.evaluation_problem(cfg, level)
    mean, cov = s.state
    for _ in range(round(after / cfg["dt"])):
        mean, cov = s.dyn.step(mean, cov, torch.zeros(2))
    assert cov[1, 1].sqrt().item() == pytest.approx(sigma, rel=1e-4)
