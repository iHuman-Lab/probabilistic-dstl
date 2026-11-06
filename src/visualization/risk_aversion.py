"""Figure for the risk-aversion experiment (experiments/risk_aversion.py).

Pure plotting: the caller supplies the planned belief paths and the Monte
Carlo success rates; nothing here loads results or runs a planner.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Rectangle

from visualization.figures import COLORS, _ellipse_parameters

LABELS = {"stlcg": "STLCG", "pdstl": "pdSTL"}
# The repo's figure palette (visualization.figures.COLORS): pdSTL in the
# belief blue, the deterministic plan in the planned orange.
METHOD_COLOR = {"pdstl": COLORS["mean"], "stlcg": COLORS["planned"]}


def _draw_room(ax, cfg):
    """Walls, obstacles and goal in the repo's style; start marked S."""
    (x0, x1), (y0, y1) = cfg["bounds"]["x_range"], cfg["bounds"]["y_range"]
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            ec=COLORS["workspace"],
            lw=1.3,
        )
    )
    for block in cfg["obstacles"]:
        (bx0, bx1), (by0, by1) = block["x_range"], block["y_range"]
        ax.add_patch(
            Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                fc=COLORS["obstacle"],
                ec=COLORS["obstacle"],
                alpha=0.45,
                hatch="//",
            )
        )
    (gx0, gx1), (gy0, gy1) = cfg["goal"]["x_range"], cfg["goal"]["y_range"]
    ax.add_patch(
        Rectangle(
            (gx0, gy0),
            gx1 - gx0,
            gy1 - gy0,
            fc=COLORS["goal"],
            ec=COLORS["goal"],
            alpha=0.35,
        )
    )
    ax.text(
        (gx0 + gx1) / 2,
        gy1 + 0.15,
        "G",
        color=COLORS["goal"],
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    sx, sy = cfg["x0_mean"][:2]
    ax.plot(sx, sy, "o", color="k", ms=3.5, zorder=5)
    ax.text(sx, sy + 0.25, "S", fontweight="bold", ha="center", va="bottom")


def _draw_plan(ax, mean, cov, method, every):
    """Mean path; pdSTL also gets its 95% belief ellipses every `every`.

    STLCG plans the noise-free mean, so it has no belief to draw; its
    executed spread shows in its Monte Carlo success instead.
    """
    color = METHOD_COLOR[method]
    ax.plot(*mean.T, color=color, lw=1.6, zorder=4)
    if method != "pdstl":
        return
    for k in [*range(every, len(mean) - 1, every), len(mean) - 1]:
        width, height, angle = _ellipse_parameters(cov[k], 0.95)
        ax.add_patch(
            Ellipse(
                mean[k],
                width,
                height,
                angle=angle,
                fc=color,
                ec=color,
                alpha=0.22,
                lw=0.6,
                zorder=3,
            )
        )


def _ticks(limits, step=2, last=True):
    """Ticks every `step` metres across `limits`; drop the upper end unless
    `last`."""
    lo, hi = limits
    ticks = np.arange(lo, hi + 1e-9, step)
    return ticks if last else ticks[ticks < hi]


def plot_risk_aversion(
    cfg, beliefs, success, out_dir, methods=("stlcg", "pdstl")
):
    """One panel per noise level: both plans and their MC success.

    `beliefs[(method, noise)]` is (mean [T, 2], cov [T, 2, 2]);
    `success[(method, noise)]` is the Monte Carlo success rate in [0, 1].
    Writes figure.png and figure.pdf to `out_dir` and returns the png path.

    Sized for a two-column IEEE figure (7.16 in) with the panels filling
    the width; the explanation of the marks belongs in the caption.
    """
    levels = list(cfg["noise"])
    every = cfg["figure"]["ellipse_every"]
    sigma = cfg["uncertainty"]["position_std_m"]
    after = cfg["uncertainty"]["after_s"]
    with plt.rc_context({"font.size": 8}):
        fig, axes = plt.subplots(
            1,
            len(levels),
            figsize=(7.16, 2.95),
            sharey=True,
            gridspec_kw={"wspace": 0.05},
        )
        for i, (ax, noise) in enumerate(zip(axes, levels)):
            _draw_room(ax, cfg)
            for method in methods:
                mean, cov = beliefs[(method, noise)]
                _draw_plan(ax, mean, cov, method, every)
            text = "\n".join(
                f"{LABELS[m]}: {100 * success[(m, noise)]:.1f}%"
                for m in reversed(methods)
            )
            ax.text(
                0.025,
                0.975,
                text,
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
                linespacing=1.35,
                bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.5},
            )
            ax.set_title(
                f"({'abc'[i]}) {noise} uncertainty: "
                f"{100 * sigma[noise]:.0f} cm after {after:g} s",
                fontsize=8,
                pad=3,
            )
            ax.set(
                xlim=cfg["bounds"]["x_range"],
                ylim=cfg["bounds"]["y_range"],
                aspect="equal",
                # the last panel alone shows the right wall's tick, so
                # labels of neighbouring panels do not meet
                xticks=_ticks(
                    cfg["bounds"]["x_range"], last=i == len(levels) - 1
                ),
                yticks=_ticks(cfg["bounds"]["y_range"]),
            )
            ax.set_xlabel("x [m]", labelpad=1)
            ax.grid(color=COLORS["workspace"], alpha=0.16, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7, length=2, pad=1.5)
        axes[0].set_ylabel("y [m]", labelpad=1)
        handles = [
            Line2D([], [], color=METHOD_COLOR["pdstl"], lw=1.6, label="pdSTL"),
            Patch(
                fc=METHOD_COLOR["pdstl"],
                ec=METHOD_COLOR["pdstl"],
                alpha=0.3,
                label="pdSTL 95% belief",
            ),
            Line2D([], [], color=METHOD_COLOR["stlcg"], lw=1.6, label="STLCG"),
            Patch(
                fc=COLORS["obstacle"],
                ec=COLORS["obstacle"],
                alpha=0.45,
                hatch="//",
                label="Obstacle",
            ),
            Patch(
                fc=COLORS["goal"], ec=COLORS["goal"], alpha=0.35, label="Goal"
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=5,
            fontsize=7.5,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.4,
            borderaxespad=0.1,
        )
        fig.subplots_adjust(left=0.05, right=0.998, bottom=0.1, top=0.88)
        for suffix in (".png", ".pdf"):
            fig.savefig(
                out_dir / f"figure{suffix}",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.02,
            )
        plt.close(fig)
    return out_dir / "figure.png"
