import json

import pytest
import torch

from experiments import timing
from models.beliefs import GaussianBeliefTrajectory
from pdstl.operators import Always, Eventually


def test_merge_like_formula_evaluates_one_origin_and_methods_agree():
    generator = torch.Generator().manual_seed(0)
    for steps in (11, 31):
        mean, covariance = timing.random_beliefs(3, steps, "cpu", generator)
        formula = timing.merge_like_formula(steps, Always, Eventually)
        out = formula(GaussianBeliefTrajectory(mean, covariance), beta=10.0)
        assert out.shape == (3, 1, 2)  # one origin per plan, as in planning
        assert (
            timing.largest_difference(steps, 2, "cpu", 10.0, generator) < 1e-5
        )


def test_iteration_time_includes_a_backward_pass():
    generator = torch.Generator().manual_seed(1)
    formula = timing.merge_like_formula(11)
    seconds = timing.iteration_seconds(formula, 2, 11, "cpu", 10.0, generator)
    assert seconds > 0


def test_median_ms_warms_up_then_repeats_at_least_min_reps():
    calls = []

    def run():
        calls.append(None)
        return 0.002

    ms, reps = timing.median_ms(run, min_reps=3, max_reps=5, budget=0.0)
    assert reps == 3 and len(calls) == 4  # one untimed warm-up
    assert ms == pytest.approx(2.0)


def test_benchmark_rows_cover_both_experiments_and_save(tmp_path, monkeypatch):
    from planning import runners

    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    result = timing.benchmark(
        horizons=(11, 21), batches=(1, 4), batch_steps=11, devices=["cpu"]
    )
    rows = result["rows"]
    # A: each horizon x method at batch 1; B: each batch x method.
    assert len(rows) == 2 * 2 + 2 * 2
    assert {r["method"] for r in rows} == {"parallel", "recurrent"}
    assert all(r["batch"] == 1 for r in rows if r["experiment"] == "A")
    assert all(r["steps"] == 11 for r in rows if r["experiment"] == "B")
    for row in rows:
        assert row["plans per s"] == pytest.approx(
            row["batch"] / (row["ms"] / 1e3)
        )
    assert result["largest difference"] < 1e-5
    assert result["machine"]["gpu"] is None
    saved = json.loads((tmp_path / "timing_benchmark.json").read_text())
    assert saved["rows"] == rows
    assert (tmp_path / "timing_benchmark.pdf").exists()
    assert (tmp_path / "timing_benchmark.png").exists()
