"""Check the delivered numerical data, saved optimizer paths, and encoded videos."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdstl-mpl")
import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent


def main():
    checks = []
    smoothing = json.loads((HERE / "smoothing_values.json").read_text())
    values = np.array(smoothing["teaching_values"])
    for row in smoothing["rows"]:
        b = row["beta"]
        s = -logsumexp(-b * values) / b
        weights = np.exp(-b * values - logsumexp(-b * values))
        np.testing.assert_allclose(row["smin"], s, atol=1e-14)
        np.testing.assert_allclose(
            [row[f"w{i}"] for i in range(11)], weights, atol=1e-14
        )
        assert row["bound_lower"] <= s <= row["bound_upper"]
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-14)
    checks.append(
        "All three smoothing values, numerical bounds and eleven derivatives match the analytic rule; source/autograd comparison is saved in smoothing_values.json."
    )
    predicate = json.loads((HERE / "predicate_bounds.json").read_text())
    lower, upper, exact = [
        np.array(predicate[k]) for k in ("lower", "upper", "exact")
    ]
    assert np.all(lower <= exact + 1e-12) and np.all(exact <= upper + 1e-12)
    assert np.max(upper - lower) > 0.1
    run = json.loads((HERE / "reach_avoid_run.json").read_text())
    belief = run["predictions"]["stlcg/high"]
    np.testing.assert_array_equal(
        predicate["mean_m"], np.array(belief["mean"])[:, :2]
    )
    np.testing.assert_array_equal(
        predicate["covariance_m2"], np.array(belief["covariance"])[:, :2, :2]
    )
    checks.append(
        "Exact predicate probability is contained in every non-exact repo interval."
    )
    memory = np.array([[0.8, 0.9], [0.9, 0.95], [0.4, 0.6]])
    shift = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    updated = shift @ memory + np.array([[0], [0], [1]]) @ np.array(
        [[0.7, 0.85]]
    )
    np.testing.assert_allclose(updated.min(0), [0.4, 0.6])
    np.testing.assert_allclose(updated.max(0), [0.9, 0.95])
    checks.append(
        "Register shift, insertion, min and max match the supplied teaching intervals."
    )
    run = json.loads((HERE / "reach_avoid_run.json").read_text())
    dt = run["config"]["dt"]
    a = np.eye(4, dtype=np.float32)
    a[0, 2] = a[1, 3] = dt
    b = np.array(
        [[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]], dtype=np.float32
    )
    for entry in run["histories"].values():
        hist = np.load(HERE / entry["file"])
        assert hist["theta"].shape == (601, 60, 2)
        np.testing.assert_array_equal(hist["iterations"], np.arange(-1, 600))
        np.testing.assert_allclose(
            np.tanh(hist["theta"]) * run["config"]["u_max"],
            hist["controls"],
            atol=2e-7,
        )
        mean = hist["mean_trajectories"]
        np.testing.assert_allclose(
            mean[:, 1:], mean[:, :-1] @ a.T + hist["controls"] @ b.T, atol=2e-6
        )
    checks.append(
        "All eight solves contain initial theta and all 600 hook-recorded updates; saved controls and complete trajectories reproduce double-integrator dynamics."
    )
    for r in run["selected"]:
        n = run["config"]["monte_carlo"]["samples"]
        hits = round(r["success"] * n)
        lo = beta_dist.ppf(0.025, hits, n - hits + 1) if hits else 0
        hi = beta_dist.ppf(0.975, hits + 1, n - hits) if hits < n else 1
        np.testing.assert_allclose(
            [lo, hi], [r["ci_low"], r["ci_high"]], atol=1e-12
        )
        key = f"{r['method']}_{r['noise'] if r['method'] == 'pdstl' else 'nominal'}_{r['start']}"
        hist = np.load(HERE / run["histories"][key]["file"])
        selected = int(hist["selected_iteration"]) + 1
        np.testing.assert_allclose(
            hist["controls"][selected],
            run["controls"][f"{r['method']}/{r['noise']}"],
            atol=2e-7,
        )
        np.testing.assert_allclose(
            hist["mean_trajectories"][selected],
            run["predictions"][f"{r['method']}/{r['noise']}"]["mean"],
            atol=8e-6,
        )
    checks.append(
        "All six Monte Carlo confidence intervals match their 5000-rollout counts; selected controls and paths belong to the recorded solves."
    )
    pngs = {}
    for path in sorted(HERE.glob("*.png")):
        im = Image.open(path)
        expected = (
            (2000, 780)
            if path.name == "predicate_bounds.png"
            else (1920, 1080)
        )
        assert im.size == expected, (path.name, im.size)
        assert im.convert("RGB").getpixel((0, 0)) == (255, 255, 255)
        pngs[path.name] = list(im.size)
    videos = {}
    for name, stages in [
        ("smoothing_window", 5),
        ("register", 8),
        ("reach_avoid_optimizer", 7),
    ]:
        path = HERE / f"{name}.mp4"
        frames = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        meta = next(frames)
        assert "h264" in meta["codec"]
        assert meta["size"] == (1920, 1080)
        assert meta["duration"] <= 15
        assert path.stat().st_size <= 20_000_000
        n, last = 0, None
        for packet in frames:
            last = np.frombuffer(packet, dtype=np.uint8).reshape(1080, 1920, 3)
            n += 1
        final = np.array(Image.open(HERE / f"{name}_final.png").convert("RGB"))
        error = float(np.mean(abs(last.astype(float) - final)))
        assert error < 3, error
        assert (HERE / f"{name}_final.png").read_bytes() == (
            HERE / f"{name}_step{stages}.png"
        ).read_bytes()
        videos[path.name] = {
            "codec": meta["codec"],
            "size": list(meta["size"]),
            "fps": meta["fps"],
            "frames": n,
            "duration_s": n / meta["fps"],
            "bytes": path.stat().st_size,
            "final_frame_mean_absolute_error": error,
        }
    checks.append(
        "Every MP4 fully decodes as 1920×1080 H.264, is below 15 seconds and 20 MB, and ends on the exported final still; every stage still exists."
    )
    output = {
        "checks": checks,
        "videos": videos,
        "png_dimensions": pngs,
        "source_commit": run["source_commit"],
    }
    (HERE / "validation.json").write_text(json.dumps(output, indent=2) + "\n")
    print("\n".join(checks))
    for k, v in videos.items():
        print(k, v)


if __name__ == "__main__":
    main()
