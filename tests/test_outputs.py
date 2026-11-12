"""Saved figures, animations, and main.py switches."""

import matplotlib
import matplotlib.pyplot as plt
import yaml
from PIL import Image

from tests.test_reach_avoid import run_reach_avoid
from utils import load_config, skip_run
from visualization.figures import _finish


def test_preview_happens_before_publication_files_are_saved(
    tmp_path, monkeypatch
):
    fig, _ = plt.subplots()
    calls = []
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "QtAgg")
    monkeypatch.setattr(plt, "show", lambda **kwargs: calls.append("show"))
    monkeypatch.setattr(
        fig,
        "savefig",
        lambda path, **kwargs: calls.append(path.suffix),
    )
    _finish(fig, tmp_path / "preview.png", show=True)
    assert calls == ["show", ".png", ".pdf"]


def test_reach_avoid_saves_figure_animation_and_result(
    tmp_path, monkeypatch, capsys
):
    import planning.runners as runners

    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    cfg = load_config("configs/scenarios/reach_avoid/obstacle.yaml")
    cfg["H"] = 4
    cfg["planner"]["max_iters"] = 2
    cfg["monte_carlo"] = {"samples": 200}
    path = tmp_path / "obstacle.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_reach_avoid(
        str(path), show=False, save=True, live=True, optimization_every=1
    )
    for suffix in (".pt", ".png", ".pdf", ".gif"):
        assert (tmp_path / f"obstacle{suffix}").exists()
    with Image.open(tmp_path / "obstacle.gif") as movie:
        assert movie.n_frames >= 2
        first = movie.convert("RGB").tobytes()
        movie.seek(movie.n_frames - 1)
        assert movie.convert("RGB").tobytes() != first
    progress = capsys.readouterr().out
    assert "Reach–Avoid iteration 1/2" in progress
    assert "Reach–Avoid: rho_lower " in progress
    assert "Monte Carlo P(phi) = " in progress
    plt.close("all")


def test_project_blocks_can_be_run_or_skipped():
    visited = []
    with skip_run("run", "active") as check, check():
        visited.append("active")
    with skip_run("skip", "inactive") as check, check():
        visited.append("inactive")
    assert visited == ["active"]
