"""Deterministic slide media. Run with --source pointing to the verified main archive.

Requires the project dependencies plus imageio-ffmpeg. Uses the bundled Rubik font.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdstl-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from PIL import Image
from scipy.special import logsumexp, ndtr
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, PURPLE, GRAY, INK = (
    "#1F77B4",
    "#EC672C",
    "#7B3FA0",
    "#6F6F6F",
    "#1F1F1F",
)
L = np.array(
    [0.82, 0.75, 0.62, 0.48, 0.40, 0.45, 0.58, 0.70, 0.90, 0.88, 0.85]
)
BETAS = [5, 20, 100]
font_manager.fontManager.addfont(str(HERE / "Rubik-Regular.ttf"))
font_manager.fontManager.addfont(str(HERE / "Rubik-Bold.ttf"))
plt.rcParams.update(
    {
        "font.family": "Rubik",
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRAY,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    }
)


def frame(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def save_frame(fig, name):
    rgb = frame(fig)
    Image.fromarray(rgb).save(HERE / name)
    return rgb


class Video:
    def __init__(self, name, fps=24):
        import imageio_ffmpeg

        self.path, self.fps = HERE / name, fps
        self.proc = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1920x1080",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
        )
        self.n = 0

    def add(self, rgb, seconds=None):
        assert rgb.shape == (1080, 1920, 3)
        count = 1 if seconds is None else round(seconds * self.fps)
        data = rgb.astype(np.uint8).tobytes()
        for _ in range(count):
            self.proc.stdin.write(data)
        self.n += count

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError("H.264 encoding failed")
        assert self.path.stat().st_size <= 20_000_000
        assert self.n / self.fps <= 15
        print(
            f"{self.path.name}: {self.n / self.fps:g} s, {self.path.stat().st_size} bytes",
            flush=True,
        )


def title(fig, heading, subtitle):
    fig.text(0.065, 0.94, heading, fontsize=32, weight="bold", va="top")
    fig.text(0.065, 0.875, subtitle, fontsize=19, color=GRAY, va="top")


def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.13, color=GRAY)
    ax.set_axisbelow(True)


def smoothing(source):
    sys.path.insert(0, str(source / "src"))
    from pdstl.operators import Minish, _reduce_bounds

    rows, checks = [], []
    for beta in BETAS:
        x = torch.tensor(
            np.stack([L, L], -1)[None], dtype=torch.float64, requires_grad=True
        )
        repo_value = _reduce_bounds(Minish(), x, beta)[0, 0, 0]
        derivative = torch.autograd.grad(repo_value, x)[0][0, :, 0].numpy()
        smin = -logsumexp(-beta * L) / beta
        weights = np.exp(-beta * L - logsumexp(-beta * L))
        lower = L.min() - np.log(len(L)) / beta
        np.testing.assert_allclose(
            repo_value.detach().item(), smin, atol=1e-14
        )
        np.testing.assert_allclose(derivative, weights, atol=1e-14)
        assert lower <= smin <= L.min()
        rows.append(
            dict(
                beta=beta,
                smin=smin,
                bound_lower=lower,
                bound_upper=float(L.min()),
                **{f"w{j}": float(w) for j, w in enumerate(weights)},
            )
        )
        checks.append(
            {
                "beta": beta,
                "repo_lower": repo_value.item(),
                "max_derivative_error": float(abs(derivative - weights).max()),
            }
        )
    with (HERE / "smoothing_values.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "smoothing_values.json").write_text(
        json.dumps(
            {
                "teaching_values": L.tolist(),
                "m": 11,
                "rows": rows,
                "repo_checks": checks,
            },
            indent=2,
        )
        + "\n"
    )
    video = Video("smoothing_window.mp4")
    captions = [
        "An 11-step window of lower-endpoint scores",
        "hard min: whole gradient on one time step",
        r"$\beta = 5$: gradient spread over several time steps",
        r"$\beta = 20$: gradient concentrates near the minimum",
        r"$\beta = 100$: nearly the hard-min spike again",
    ]
    for stage in range(5):
        fig, (top, bot) = plt.subplots(
            2, 1, figsize=(19.2, 10.8), dpi=100, sharex=True
        )
        fig.subplots_adjust(
            left=0.11, right=0.94, bottom=0.20, top=0.79, hspace=0.34
        )
        title(
            fig,
            r"Smoothing shares the gradient; $\beta$ sets how much",
            r"Lower-endpoint scores and $\beta$: teaching values",
        )
        for ax in (top, bot):
            clean_axis(ax)
            ax.set_xlim(-0.3, 10.3)
            ax.set_xticks(range(11))
            ax.axvspan(-0.3, 10.3, color=BLUE, alpha=0.035)
        top.set_ylim(0, 1.05)
        top.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        bot.set_ylim(0, 1)
        bot.set_yticks([0, 0.25, 0.5, 0.75, 1])
        top.set_ylabel(r"score $L_j$ (unitless)")
        bot.set_ylabel("derivative $\\partial/\\partial L_j$\n(unitless)")
        bot.set_xlabel("time step j (steps)")
        top.plot(range(11), L, "o-", color=BLUE, lw=3, ms=9)
        if stage >= 1:
            top.axhline(0.4, color=GRAY, ls="--", lw=2)
            top.text(
                10.05,
                0.435,
                "hard min = 0.400000",
                color=GRAY,
                ha="right",
                fontsize=17,
            )
            top.scatter(
                [4],
                [0.4],
                s=250,
                facecolor="white",
                edgecolor=BLUE,
                linewidth=4,
                zorder=6,
            )
        if stage == 1:
            weights = np.eye(11)[4]
            bot.plot(
                range(11),
                weights,
                "o-",
                color=GRAY,
                lw=2.5,
                ms=7,
                clip_on=False,
            )
            bot.vlines(4, 0, 1, color=GRAY, lw=3)
        if stage >= 2:
            row = rows[stage - 2]
            weights = [row[f"w{j}"] for j in range(11)]
            top.axhline(row["smin"], color=BLUE, lw=2.5)
            top.text(
                10.05,
                row["smin"] - 0.06,
                f"smooth min = {row['smin']:.6f}",
                color=BLUE,
                ha="right",
                va="top",
                fontsize=17,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
            )
            bot.plot(
                range(11), weights, "o-", color=BLUE, lw=3, ms=9, clip_on=False
            )
            bound = f"m = 11:  {row['bound_lower']:.6f} ≤ {row['smin']:.6f} ≤ 0.400000"
        else:
            bound = r"$m = 11$:  $0.400000 - \log(11)/\beta \leq \mathrm{smin} \leq 0.400000$   ($\beta > 0$)"
        fig.text(0.11, 0.105, captions[stage], fontsize=22, weight="bold")
        fig.text(0.11, 0.052, bound, fontsize=19, color=GRAY)
        rgb = save_frame(fig, f"smoothing_window_step{stage + 1}.png")
        video.add(rgb, 1.8)
        if stage == 4:
            Image.fromarray(rgb).save(HERE / "smoothing_window_final.png")
        plt.close(fig)
    video.close()


def predicate(source):
    sys.path.insert(0, str(source / "src"))
    from models.rollouts import create_gaussian_belief_trajectory
    from pdstl.predicates import OutsideRectangle

    # Reuse the actual Task E run. The independent-axis dynamics make the
    # exact joint analytic; the repo uses only Frechet marginal composition.
    run = json.loads((HERE / "reach_avoid_run.json").read_text())
    prediction = run["predictions"]["stlcg/high"]
    mean = np.array(prediction["mean"])[:, :2]
    covariance = np.array(prediction["covariance"])[:, :2, :2]
    np.testing.assert_array_equal(covariance[:, 0, 1], 0)
    variance = np.diagonal(covariance, axis1=1, axis2=2).copy()
    t = np.arange(len(mean)) * run["config"]["dt"]
    obstacle = next(
        b for b in run["config"]["obstacles"] if b["name"] == "floor"
    )
    xr, yr = obstacle["x_range"], obstacle["y_range"]
    belief = create_gaussian_belief_trajectory(
        torch.tensor(mean), torch.tensor(variance)
    )
    rectangle = OutsideRectangle(xr, yr)
    bounds = rectangle(belief, beta=None)[0].numpy()
    sx, sy = np.sqrt(variance).T
    px = ndtr((xr[1] - mean[:, 0]) / sx) - ndtr((xr[0] - mean[:, 0]) / sx)
    py = ndtr((yr[1] - mean[:, 1]) / sy) - ndtr((yr[0] - mean[:, 1]) / sy)
    exact = 1 - px * py
    np.testing.assert_allclose(
        bounds[:, 0], 1 - np.minimum(px, py), atol=1e-12
    )
    np.testing.assert_allclose(
        bounds[:, 1], np.minimum(1, 2 - px - py), atol=1e-12
    )
    assert np.all(bounds[:, 0] <= exact + 1e-12) and np.all(
        exact <= bounds[:, 1] + 1e-12
    )
    fig, ax = plt.subplots(figsize=(20, 7.8), dpi=100)
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.18, top=0.71)
    title(
        fig,
        "Predicates become probability intervals over time",
        "OutsideRectangle: exact Gaussian joint vs Boole–Fréchet bounds | same reach–avoid run",
    )
    clean_axis(ax)
    ax.vlines(t, bounds[:, 0], bounds[:, 1], color=GRAY, alpha=0.55, lw=4)
    ax.plot(t, exact, color=INK, lw=3, label="exact P(outside)")
    ax.plot(t, bounds[:, 0], "_", color=BLUE, ms=17, mew=3, label="lower L")
    ax.plot(t, bounds[:, 1], "_", color=ORANGE, ms=17, mew=3, label="upper U")
    ax.set(
        xlim=(0, t[-1]),
        ylim=(0, 1),
        xlabel="time (s)",
        ylabel="P(outside rectangle) (unitless)",
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.21), ncol=3, frameon=False
    )
    fig.text(
        0.075,
        0.045,
        "Selected STLCG plan; high noise q = 0.012244 (50 cm at 10 s); independent Gaussian axes; floor rectangle [4, 5.5] × [0, 3] m",
        fontsize=17,
        color=GRAY,
    )
    save_frame(fig, "predicate_bounds.png")
    plt.close(fig)
    data = {
        "label": "Actual belief from the single reach-avoid run, selected STLCG plan under high evaluation noise",
        "source_commit": run["source_commit"],
        "source_data": "reach_avoid_run.json:predictions.stlcg/high",
        "process_noise_q": run["config"]["noise"]["high"],
        "time_s": t.tolist(),
        "mean_m": mean.tolist(),
        "covariance_m2": covariance.tolist(),
        "rectangle_m": {"x": xr, "y": yr},
        "exact": exact.tolist(),
        "lower": bounds[:, 0].tolist(),
        "upper": bounds[:, 1].tolist(),
        "code": "pdstl.predicates.OutsideRectangle.robustness_trace -> Negation(And(AxisInterval, AxisInterval)); models.beliefs.GaussianBelief.probability",
    }
    (HERE / "predicate_bounds.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )


def register_frame(stage, progress=1):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    title(
        fig,
        "A shift register makes temporal readout recurrent",
        "Window [0, 2] steps | lower L and upper U | teaching values",
    )
    ax = fig.add_axes([0.05, 0.14, 0.9, 0.65])
    ax.set(xlim=(0, 16), ylim=(-1.5, 7.2))
    ax.axis("off")
    xcol, yrow = [4, 6], [5.3, 3.5, 1.7]
    for x, text, color in zip(xcol, ["L", "U"], [BLUE, ORANGE]):
        ax.text(
            x, 6.6, text, color=color, fontsize=30, ha="center", weight="bold"
        )
    for y, label in zip(yrow, ["k + 2", "k + 1", "k"]):
        ax.text(1.7, y, label, color=GRAY, fontsize=23, va="center")
        for x, color in zip(xcol, [BLUE, ORANGE]):
            active = stage == 4 and x == 4 or stage == 5 and x == 6
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.8, y - 0.65),
                    1.6,
                    1.3,
                    boxstyle="round,pad=0.05",
                    fc=color if active else "#F7F7F7",
                    alpha=0.14 if active else 1,
                    ec=GRAY,
                    lw=1.3,
                )
            )

    def row(idx, y, alpha=1):
        vals = [[0.7, 0.85], [0.4, 0.6], [0.9, 0.95], [0.8, 0.9]][idx]
        ax.text(
            2.65,
            y,
            f"r{idx}",
            color=GRAY,
            alpha=alpha,
            ha="right",
            va="center",
            fontsize=20,
        )
        for x, val, color in zip(xcol, vals, [BLUE, ORANGE]):
            ax.text(
                x,
                y,
                f"{val:.2f}",
                fontsize=28,
                color=color,
                alpha=alpha,
                ha="center",
                va="center",
            )

    if stage == 0:
        row(3, yrow[0])
        row(2, yrow[1])
        row(1, yrow[2])
        row(0, -0.7)
    elif stage == 1:
        row(3, yrow[0] + 0.75 * progress, 1 - progress)
        row(2, yrow[1])
        row(1, yrow[2])
        row(0, -0.7)
    elif stage == 2:
        row(2, yrow[1] + 1.8 * progress)
        row(1, yrow[2] + 1.8 * progress)
        row(0, -0.7)
    elif stage == 3:
        row(2, yrow[0])
        row(1, yrow[1])
        row(0, -0.7 + 2.4 * progress)
    else:
        row(2, yrow[0])
        row(1, yrow[1])
        row(0, yrow[2])
    if stage <= 3:
        ax.text(8.7, 5.5, r"$H_k =$", fontsize=31)
        ax.text(
            10.2, 5.5, r"$M$", fontsize=31, color=BLUE if stage == 2 else GRAY
        )
        ax.text(10.9, 5.5, r"$H_{k+1} +$", fontsize=31)
        ax.text(
            13.2,
            5.5,
            r"$e_2$",
            fontsize=31,
            color=ORANGE if stage == 3 else GRAY,
        )
        ax.text(14, 5.5, r"$\hat r_k$", fontsize=31)
        ax.text(
            8.7,
            3.7,
            "shift M"
            if stage == 2
            else "insert e₂"
            if stage == 3
            else "shift, then insert",
            fontsize=25,
            color=BLUE if stage == 2 else ORANGE if stage == 3 else GRAY,
        )
    if stage in (4, 5, 6):
        ax.annotate(
            "",
            xy=(8.2, 2.3),
            xytext=(8.2, 5.7),
            arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 3},
        )
        ax.text(9, 4.8, "min down each column", fontsize=25)
        if stage >= 4:
            ax.text(9, 3.5, "L = 0.40", color=BLUE, fontsize=28)
        if stage >= 5:
            ax.text(12, 3.5, "U = 0.60", color=ORANGE, fontsize=28)
        if stage >= 6:
            ax.text(
                9,
                1.5,
                "output [0.40, 0.60]",
                color=PURPLE,
                fontsize=31,
                weight="bold",
            )
    if stage == 7:
        ax.text(9, 4.6, "Same memory, max readout", fontsize=25)
        ax.text(
            9,
            2.7,
            "output [0.90, 0.95]",
            color=PURPLE,
            fontsize=31,
            weight="bold",
        )
    if stage == 8:
        ax.text(8.7, 5.3, r"$H_k = M H_{k+1} + e_2\,\hat r_k$", fontsize=31)
        ax.text(
            9, 3.8, "M shifts surviving rows upward", color=BLUE, fontsize=22
        )
        ax.text(
            9,
            2.8,
            "e₂ inserts r₀ into the bottom row",
            color=ORANGE,
            fontsize=22,
        )
        ax.text(9, 1.1, "M = [0 1 0; 0 0 1; 0 0 0]", color=BLUE, fontsize=21)
        ax.text(
            9, 0, r"$e_2 = [0, 0, 1]^{\mathsf{T}}$", color=ORANGE, fontsize=21
        )
    captions = [
        "Start: H₁ = [r₃; r₂; r₁]",
        "r₃ leaves upward",
        "r₂ and r₁ move up one fixed row",
        "r₀ enters the bottom row",
        "Always: min of the L column",
        "Always: min of the U column",
        "Always readout [0.40, 0.60]",
        "Eventually readout [0.90, 0.95]",
        "Shift and insert = one affine recurrence",
    ]
    fig.text(0.08, 0.07, captions[stage], fontsize=24, weight="bold")
    rgb = frame(fig)
    plt.close(fig)
    return rgb


def register():
    video = Video("register.mp4")
    video.add(register_frame(0), 0.5)
    for stage in range(1, 9):
        if stage <= 3:
            for p in np.linspace(0, 1, 8):
                video.add(register_frame(stage, p))
        rgb = register_frame(stage)
        Image.fromarray(rgb).save(HERE / f"register_step{stage}.png")
        video.add(rgb, 1.5)
    Image.fromarray(rgb).save(HERE / "register_final.png")
    video.close()


def room(ax, cfg):
    clean_axis(ax)
    ax.set(
        xlim=(-0.4, 11.6),
        ylim=(-0.4, 11.6),
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 11, 2))
    ax.add_patch(Rectangle((0, 0), 10, 10, fc="none", ec=GRAY, lw=1.2))
    for block in cfg["obstacles"]:
        x0, x1 = block["x_range"]
        y0, y1 = block["y_range"]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fc="#E8E8E8",
                ec=GRAY,
                hatch="//",
                lw=1,
            )
        )
    x0, x1 = cfg["goal"]["x_range"]
    y0, y1 = cfg["goal"]["y_range"]
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fc=PURPLE, ec=PURPLE, alpha=0.13)
    )
    ax.text(
        (x0 + x1) / 2,
        y1 + 0.22,
        "goal",
        ha="center",
        color=PURPLE,
        fontsize=17,
    )
    ax.plot(*cfg["x0_mean"][:2], "o", color=INK, ms=6)


def draw_path(ax, mean, cov, method):
    mean, cov = np.asarray(mean)[:, :2], np.asarray(cov)[:, :2, :2]
    color, ls = (BLUE, "-") if method == "pdstl" else (ORANGE, "--")
    ax.plot(mean[:, 0], mean[:, 1], color=color, ls=ls, lw=3, zorder=4)
    for i in range(5, len(mean), 5):
        values, vectors = np.linalg.eigh(cov[i])
        width, height = 2 * np.sqrt(
            chi2.ppf(0.95, 2) * np.maximum(values[::-1], 0)
        )
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        ax.add_patch(
            Ellipse(
                mean[i],
                width,
                height,
                angle=angle,
                fill=False,
                ec=color,
                ls=ls,
                alpha=0.5,
                lw=1.3,
            )
        )


def reach_avoid():
    data = json.loads((HERE / "reach_avoid_run.json").read_text())
    cfg = data["config"]
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), dpi=100)
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.26, top=0.77, wspace=0.24
    )
    title(
        fig,
        "Reach–avoid: where does uncertainty change the plan?",
        "One CPU run | pointwise 95% position ellipses under execution noise for both plans",
    )
    for ax, level in zip(axes, cfg["noise"]):
        room(ax, cfg)
        for method in ("stlcg", "pdstl"):
            pred = data["predictions"][f"{method}/{level}"]
            draw_path(ax, pred["mean"], pred["covariance"], method)
        ax.set_title(
            f"{level}: q = {cfg['noise'][level]:.6f}\n({100 * cfg['uncertainty']['position_std_m'][level]:g} cm at 10 s)",
            fontsize=19,
            pad=13,
        )
    fig.legend(
        [
            Line2D([], [], color=BLUE, lw=3),
            Line2D([], [], color=ORANGE, ls="--", lw=3),
        ],
        ["pdSTL", "deterministic STLCG"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.14),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.065,
        0.09,
        r"Run formula: $G_{[1,60]}$ safe $\wedge$ $F_{[1,60]}$ goal; $\Delta t = 0.2$ s; horizon = 12 s.",
        fontsize=19,
    )
    fig.text(
        0.065,
        0.042,
        "No goal dwell in this experiment. Ellipses are pointwise regions; empirical satisfaction is reported separately.",
        fontsize=18,
        color=GRAY,
    )
    save_frame(fig, "reach_avoid_paths.png")
    fig.savefig(HERE / "reach_avoid_paths.pdf")
    plt.close(fig)
    rows = data["selected"]
    with (HERE / "reach_avoid_monte_carlo.csv").open("w") as f:
        cols = [
            "method",
            "noise",
            "q",
            "position_std_at_10s_cm",
            "gap",
            "planned_score",
            "noisy_evaluation_score",
            "hits",
            "rollouts",
            "empirical_satisfaction",
            "ci95_lower",
            "ci95_upper",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "method": r["method"],
                    "noise": r["noise"],
                    "q": cfg["noise"][r["noise"]],
                    "position_std_at_10s_cm": 100
                    * cfg["uncertainty"]["position_std_m"][r["noise"]],
                    "gap": r["gap"],
                    "planned_score": r["score"],
                    "noisy_evaluation_score": r["score_eval"],
                    "hits": round(r["success"] * 5000),
                    "rollouts": 5000,
                    "empirical_satisfaction": r["success"],
                    "ci95_lower": r["ci_low"],
                    "ci95_upper": r["ci_high"],
                }
            )
    lines = [
        "| Method | q (spread at 10 s) | Route | Planned score | Score under evaluation noise | MC successes | Empirical satisfaction (95% CI) |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        n = r["noise"]
        lines.append(
            f"| {r['method']} | {cfg['noise'][n]:.8f} ({100 * cfg['uncertainty']['position_std_m'][n]:g} cm) | {r['gap']} | {r['score']:.6f} | {r['score_eval']:.6f} | {round(r['success'] * 5000)}/5000 | {100 * r['success']:.2f}% [{100 * r['ci_low']:.2f}, {100 * r['ci_high']:.2f}] |"
        )
    (HERE / "reach_avoid_monte_carlo.md").write_text("\n".join(lines) + "\n")
    # Replay the selected high-noise candidate, including the terminal update
    # and the separately identified plan selected by the planner's rule.
    high = next(
        r for r in rows if r["method"] == "pdstl" and r["noise"] == "high"
    )
    key = f"pdstl_high_{high['start']}"
    hist = np.load(HERE / data["histories"][key]["file"])
    selected = int(hist["selected_iteration"])
    indices = [-1, 49, 149, 299, 449, 599]
    stages = [
        (i, "initial guess" if i == -1 else f"after update {i + 1}")
        for i in indices
    ]
    stages.append((selected, "selected output"))
    video = Video("reach_avoid_optimizer.mp4")
    for stage, (iteration, label) in enumerate(stages, 1):
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        title(
            fig,
            "Optimizer replay: one recorded pdSTL solve",
            f"High noise q = {cfg['noise']['high']:.6f} (50 cm at 10 s) | start: {high['start']} | seed 7",
        )
        ax = fig.add_axes([0.07, 0.17, 0.5, 0.61])
        room(ax, cfg)
        mean = hist["mean_trajectories"][iteration + 1]
        pred = data["predictions"]["pdstl/high"]
        draw_path(ax, mean, pred["covariance"], "pdstl")
        updates = iteration + 1
        fig.text(0.62, 0.70, label, fontsize=29, weight="bold")
        fig.text(
            0.62,
            0.60,
            f"iteration index: {iteration}\ncompleted updates: {updates}",
            fontsize=23,
            linespacing=1.6,
        )
        fig.text(
            0.62,
            0.43,
            "600 Adam updates recorded\n$\\theta$ and controls saved at every update\nFixed room axes throughout",
            fontsize=20,
            color=GRAY,
            linespacing=1.7,
        )
        fig.text(
            0.07,
            0.065,
            "The selected output can precede the terminal update. Same solve, same initial path, same objective.",
            fontsize=19,
            color=GRAY,
        )
        rgb = save_frame(fig, f"reach_avoid_optimizer_step{stage}.png")
        video.add(rgb, 1.8)
        plt.close(fig)
    Image.fromarray(rgb).save(HERE / "reach_avoid_optimizer_final.png")
    video.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--only",
        choices=["smoothing", "predicate", "register", "reach_avoid", "all"],
        default="all",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    for name in ("smoothing", "predicate", "register", "reach_avoid"):
        if args.only not in (name, "all"):
            continue
        print(f"rendering {name}", flush=True)
        if name in ("smoothing", "predicate"):
            globals()[name](source)
        else:
            globals()[name]()


if __name__ == "__main__":
    main()
