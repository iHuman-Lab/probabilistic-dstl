import matplotlib.pyplot as plt
import numpy as np
import torch

# Tableau 10
_BLUE  = "#1f77b4"
_RED   = "#d62728"
_GREEN = "#2ca02c"
_GRAY  = "#7f7f7f"


def _to_numpy(trace, T):
    """Convert trace to numpy [T, 2]."""
    if isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()
    trace = np.asarray(trace)

    if trace.ndim == 3:
        trace = trace[0]
    elif trace.ndim == 1:
        trace = np.stack([trace, trace], axis=-1)

    assert trace.shape == (T, 2), f"Expected ({T}, 2), got {trace.shape}"
    return trace


def plot_stl_formula_bounds(
    time,
    robustness_trace,
    mean_trace=None,
    var_trace=None,
    predicate_trace=None,
    thresholds=None,
    formula_str=None,
    interval=None,
    operator_type="always",
    figsize=(10, 8),
):
    time = np.asarray(time)
    T = len(time)

    oper = _to_numpy(robustness_trace, T)
    pred = _to_numpy(predicate_trace, T) if predicate_trace is not None else None

    op_symbol = "□" if operator_type == "always" else "◇"

    if pred is not None:
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        ax_signal, ax_pred, ax_oper = axes
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 0.7), sharex=True
        )
        ax_signal, ax_oper = axes
        ax_pred = None

    # Panel (a): Signal

    if mean_trace is not None and var_trace is not None:
        mean_trace = np.asarray(mean_trace)
        var_trace = np.asarray(var_trace)
        sigma = np.sqrt(np.maximum(var_trace, 0.0))

        ax_signal.fill_between(time, mean_trace - sigma, mean_trace + sigma, alpha=0.25, color=_BLUE)
        ax_signal.plot(time, mean_trace, color=_BLUE, lw=1.5, label="$\\mu(t)$")
        ax_signal.plot(time, mean_trace + sigma, color=_BLUE, lw=1, ls="--", alpha=0.7, label="$\\mu \\pm \\sigma$")
        ax_signal.plot(time, mean_trace - sigma, color=_BLUE, lw=1, ls="--", alpha=0.7)

        if thresholds is not None:
            thresholds = [thresholds] if not isinstance(thresholds, (list, tuple)) else thresholds
            for th in thresholds:
                ax_signal.axhline(th, color=_RED, ls="--", lw=1.5, label=f"$h = {th}$")

    ax_signal.set_ylabel("$x(t)$", fontsize=11)
    ax_signal.set_title("(a) Signal Trajectory", loc="left", fontweight="bold")
    ax_signal.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, framealpha=0.95)
    ax_signal.grid(True, alpha=0.3)

    if ax_pred is not None:
        ax_pred.fill_between(time, pred[:, 0], pred[:, 1], alpha=0.3, color=_GREEN)
        ax_pred.plot(time, pred[:, 0], color=_BLUE, lw=1.5, label="$P_{\\mathrm{lower}}$")
        ax_pred.plot(time, pred[:, 1], color=_RED, lw=1.5, label="$P_{\\mathrm{upper}}$")
        ax_pred.set_ylabel("$P(\\varphi)$", fontsize=11)
        ax_pred.set_ylim(-0.05, 1.05)
        ax_pred.set_title("(b) Predicate Satisfaction Probability", loc="left", fontweight="bold")
        ax_pred.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, framealpha=0.95)
        ax_pred.grid(True, alpha=0.3)

    ax_oper.fill_between(time, oper[:, 0], oper[:, 1], alpha=0.3, color=_GREEN)
    ax_oper.plot(time, oper[:, 0], color=_BLUE, lw=1.5, label="$P_{\\mathrm{lower}}$")
    ax_oper.plot(time, oper[:, 1], color=_RED, lw=1.5, label="$P_{\\mathrm{upper}}$")
    ax_oper.set_xlabel("Time [s]", fontsize=11)
    ax_oper.set_ylabel(f"$P({op_symbol}\\varphi)$", fontsize=11)
    ax_oper.set_ylim(-0.05, 1.05)
    ax_oper.set_title("(c) Temporal Operator Output", loc="left", fontweight="bold")
    ax_oper.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, framealpha=0.95)
    ax_oper.grid(True, alpha=0.3)

    if formula_str:
        fig.suptitle(formula_str, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()

    return fig, axes


def _draw_output_markers(ax, time, T, interval, oper):
    """Draw markers on output corresponding to window positions."""
    a = int(interval[0])
    b_inf = np.isinf(interval[1])
    b = T - 1 if b_inf else int(interval[1])

    window_width = b - a + 1

    if window_width >= T:
        n_windows = 1
    else:
        n_windows = min(4, max(1, T // window_width))

    max_start = max(0, T - 1 - b)
    if max_start <= 0:
        t_indices = np.array([0])
    else:
        t_indices = np.linspace(0, max_start, n_windows).astype(int)

    alphas = np.linspace(0.35, 0.85, len(t_indices))

    for i, t_idx in enumerate(t_indices):
        if t_idx >= T:
            continue
        ax.plot(time[t_idx], oper[t_idx, 0], "o", color="#ff7f0e", alpha=alphas[i], markersize=5)


def plot_piecewise_stl(
    time,
    robustness_trace,
    mean_trace=None,
    var_trace=None,
    predicate_trace=None,
    thresholds=None,
    formula_str=None,
    interval=None,
    operator_type="always",
    figsize=(10, 12),
):
    time = np.asarray(time)
    T = len(time)

    oper = _to_numpy(robustness_trace, T)
    pred = _to_numpy(predicate_trace, T) if predicate_trace is not None else None

    mean_trace = np.asarray(mean_trace) if mean_trace is not None else None
    var_trace = np.asarray(var_trace) if var_trace is not None else None
    sigma_trace = np.sqrt(var_trace) if var_trace is not None else None

    threshold = (
        thresholds
        if isinstance(thresholds, (int, float))
        else (thresholds[0] if thresholds else 50)
    )

    a, b = interval if interval else [0, 1]
    op_symbol = "" if operator_type == "always" else "◇"

    signal_color = _GRAY
    bound_color = _BLUE
    threshold_color = _RED
    lower_color = _BLUE
    upper_color = _RED

    # Create one figure with 3 subplots sharing the x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    if mean_trace is not None and sigma_trace is not None:
        for i in range(T):
            x_start = time[i]
            x_end = time[i + 1] if i < T - 1 else time[i] + 0.5

            # Mean line
            ax1.hlines(mean_trace[i], x_start, x_end, colors=signal_color, lw=2)

            # Uncertainty band (μ ± σ)
            upper_band = mean_trace[i] + sigma_trace[i]
            lower_band = mean_trace[i] - sigma_trace[i]
            ax1.hlines(
                upper_band, x_start, x_end, colors=bound_color, lw=1, ls="--", alpha=0.7
            )
            ax1.hlines(
                lower_band, x_start, x_end, colors=bound_color, lw=1, ls="--", alpha=0.7
            )
            ax1.fill_between(
                [x_start, x_end], lower_band, upper_band, alpha=0.15, color=bound_color
            )

            # Vertical transition
            if i < T - 1:
                ax1.vlines(
                    time[i + 1],
                    mean_trace[i],
                    mean_trace[i + 1],
                    colors=signal_color,
                    lw=2,
                )

            ax1.plot(time[i], mean_trace[i], "o", markersize=6)

        ax1.axhline(threshold, color=threshold_color, ls="--", lw=1.5)

        # Time labels
        for i in range(T):
            offset = 3 if mean_trace[i] < threshold else -12
            ax1.annotate(
                f"t$_{i}$",
                (time[i], mean_trace[i]),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=14,
            )

        ax1.annotate(
            f"h = {int(threshold)}",
            (time[-1] + 0.3, threshold),
            fontsize=14,
            color=threshold_color,
        )

    ax1.set_ylabel("x(t)", fontsize=18)
    ax1.set_xlim(-0.3, time[-1] + 1)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(True, alpha=0.3)

    if pred is not None:
        for i in range(T):
            x_start = time[i]
            x_end = time[i + 1] if i < T - 1 else time[i] + 0.5

            ax2.hlines(pred[i, 0], x_start, x_end, colors=lower_color, lw=2)
            ax2.hlines(pred[i, 1], x_start, x_end, colors=upper_color, lw=2)
            ax2.fill_between(
                [x_start, x_end], pred[i, 0], pred[i, 1], alpha=0.2, color=_GREEN
            )

            if i < T - 1:
                ax2.vlines(
                    time[i + 1],
                    pred[i, 0],
                    pred[i + 1, 0],
                    colors=lower_color,
                    lw=1,
                    alpha=0.5,
                )
                ax2.vlines(
                    time[i + 1],
                    pred[i, 1],
                    pred[i + 1, 1],
                    colors=upper_color,
                    lw=1,
                    alpha=0.5,
                )

            ax2.plot(time[i], pred[i, 0], "o", color=_BLUE, markersize=5)
            ax2.plot(time[i], pred[i, 1], "o", color=_RED,  markersize=5)

    ax2.set_ylabel("P(φ)", fontsize=18)
    ax2.set_ylim(-0.05, 1.05)

    ax2.plot([], [], color=_BLUE, lw=2, label=r"$P^{\downarrow}$")
    ax2.plot([], [], color=_RED,  lw=2, label=r"$P^{\uparrow}$")
    ax2.legend(loc="lower right", fontsize=14, framealpha=0.95)
    ax2.set_xlim(-0.3, time[-1] + 1)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(True, alpha=0.3)

    for i in range(T):
        x_start = time[i]
        x_end = time[i + 1] if i < T - 1 else time[i] + 0.5

        ax3.hlines(oper[i, 0], x_start, x_end, colors=lower_color, lw=2)
        ax3.hlines(oper[i, 1], x_start, x_end, colors=upper_color, lw=2)
        ax3.fill_between(
            [x_start, x_end], oper[i, 0], oper[i, 1], alpha=0.2, color=_GREEN
        )

        if i < T - 1:
            ax3.vlines(
                time[i + 1],
                oper[i, 0],
                oper[i + 1, 0],
                colors=lower_color,
                lw=1,
                alpha=0.5,
            )
            ax3.vlines(
                time[i + 1],
                oper[i, 1],
                oper[i + 1, 1],
                colors=upper_color,
                lw=1,
                alpha=0.5,
            )

        ax3.plot(time[i], oper[i, 0], "o", color=_BLUE, markersize=5)
        ax3.plot(time[i], oper[i, 1], "o", color=_RED,  markersize=5)

    ax3.set_xlabel("Time t", fontsize=18)
    ax3.set_ylabel(f"P({op_symbol}φ)", fontsize=18)
    ax3.set_ylim(-0.05, 1.05)
    ax3.plot([], [], color=_BLUE, lw=2, label=f"${op_symbol}P^{{\\downarrow}}$")
    ax3.plot([], [], color=_RED,  lw=2, label=f"${op_symbol}P^{{\\uparrow}}$")
    ax3.legend(loc="lower right", fontsize=14, framealpha=0.95)
    ax3.set_xlim(-0.3, time[-1] + 1)
    ax3.set_xticks(time)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return fig, (ax1, ax2, ax3)
