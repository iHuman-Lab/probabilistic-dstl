"""pdSTL operators on [B, N, 2] traces: Frechet Boolean, windowed temporal.

`beta=None` is exact: the pair is the pdSTL probability interval.
`beta > 0` is smooth and sound: an outer interval (lower <= exact lower,
upper >= exact upper), so `smooth_lower() >= alpha` certifies the exact bound.
"""

import math

import numpy as np
import torch

from pdstl.base import check_probability_bounds


def _checked_beta(beta):
    """None for exact evaluation, else a finite positive smoothing strength."""
    if beta is None:
        return None
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(
            f"beta must be a finite positive number or None, got {beta}"
        )
    return beta


class STL_Formula(torch.nn.Module):
    """Base formula: robustness_trace(beliefs) -> [B, N, 2] at complete origins."""

    @property
    def is_pointwise(self) -> bool:
        return False

    def robustness_trace(
        self, belief_trajectory, beta=None, keepdim=True, **kwargs
    ):
        raise NotImplementedError("robustness_trace not yet implemented")

    def forward(self, belief_trajectory, beta=None, **kwargs):
        return self.robustness_trace(
            belief_trajectory, beta=_checked_beta(beta), **kwargs
        )

    def probability_interval(self, belief_trajectory, origin=0, **kwargs):
        """The exact pdSTL probability interval [lower, upper] at one origin.

        Named in full because `interval` on a temporal operator is its window [a, b].
        """
        return self(belief_trajectory, beta=None, **kwargs)[0, origin]

    def smooth_lower(self, belief_trajectory, beta, origin=0, **kwargs):
        """The differentiable lower score at one origin: the scalar a planner descends."""
        return self(belief_trajectory, beta=beta, **kwargs)[0, origin, 0]

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Negation(self)


def _smooth_max(x, beta, dim, keepdim):
    """Normalized log-mean-exp: lies in [mean, max] and tends to max as beta grows."""
    return (
        torch.logsumexp(beta * x, dim=dim, keepdim=keepdim)
        - math.log(x.shape[dim])
    ) / beta


class Minish(torch.nn.Module):
    """Exact min (beta is None) or normalized smooth min along dim."""

    def forward(self, x, beta=None, dim=1, keepdim=True):
        if beta is None:
            return x.min(dim=dim, keepdim=keepdim)[0]
        return -_smooth_max(-x, beta, dim, keepdim)


class Maxish(torch.nn.Module):
    """Exact max (beta is None) or normalized smooth max along dim."""

    def forward(self, x, beta=None, dim=1, keepdim=True):
        if beta is None:
            return x.max(dim=dim, keepdim=keepdim)[0]
        return _smooth_max(x, beta, dim, keepdim)


def _reduce_bounds(operation, x, beta, dim=1, keepdim=True):
    """Min/max-reduce bound pairs, shifting by log(n)/beta to stay sound."""
    reduced = operation(x, beta, dim=dim, keepdim=keepdim)
    if beta is None:
        return reduced
    shift = math.log(x.shape[dim]) / beta
    if isinstance(operation, Minish):
        # Normalized soft-min overshoots min: pull the lower down.
        return _pair(reduced[..., 0] - shift, reduced[..., 1])
    # Normalized soft-max undershoots max: push the upper up.
    return _pair(reduced[..., 0], reduced[..., 1] + shift)


# --- Atomic events ------------------------------------------------------------


class Predicate(STL_Formula):
    """Atomic event; each belief supplies its probability bounds."""

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    @property
    def is_pointwise(self):
        return True

    def robustness_trace(self, belief_trajectory, validate=True, **kwargs):
        # A trajectory with a bounds cache scores each atom once: the bounds
        # do not depend on beta, so a smooth and an exact pass share them.
        cache = getattr(belief_trajectory, "bounds_cache", None)
        if cache is not None and self in cache:
            return cache[self]
        # A trajectory that can score every step at once does so; otherwise, step by step.
        score_all = getattr(belief_trajectory, "probability_bounds", None)
        if score_all is not None:
            trace = score_all(self)
        else:
            trace = torch.stack(
                [b.probability_bounds(self) for b in belief_trajectory], dim=1
            )
        if validate:
            check_probability_bounds(trace, self)
        if cache is not None:
            cache[self] = trace
        return trace

    def __str__(self):
        return self.name if self.name is not None else type(self).__name__


# --- Boolean operators ----------------------------------------------------------


_MIN, _MAX = Minish(), Maxish()


def _align(*traces):
    """Truncate traces to their common number of origins."""
    n = min(t.shape[1] for t in traces)
    return tuple(t[:, :n] for t in traces)


def _pair(lower, upper):
    return torch.stack([lower, upper], dim=-1)


def _conjunction(trace1, trace2, beta=None):
    """[max(0, L1 + L2 - 1), min(U1, U2)].

    Smooth mode keeps the unclamped excess (sound, with unit gradient) and a
    soft-min upper, which Negation turns into every OutsideRectangle lower.
    """
    excess = trace1[..., 0] + trace2[..., 0] - 1.0
    lower = torch.clamp(excess, min=0.0) if beta is None else excess
    upper = _MIN(
        _pair(trace1[..., 1], trace2[..., 1]), beta, dim=-1, keepdim=False
    )
    return _pair(lower, upper)


def _disjunction(trace1, trace2, beta=None):
    """[max(L1, L2), min(1, U1 + U2)]; both endpoints smooth when beta is set."""
    lower = _MAX(
        _pair(trace1[..., 0], trace2[..., 0]), beta, dim=-1, keepdim=False
    )
    total = trace1[..., 1] + trace2[..., 1]
    upper = _MIN(
        _pair(total, torch.ones_like(total)), beta, dim=-1, keepdim=False
    )
    return _pair(lower, upper)


def _negation(trace):
    """[1 - U, 1 - L]: exact in both modes, and it swaps which endpoint is the lower one."""
    return _pair(1.0 - trace[..., 1], 1.0 - trace[..., 0])


class Negation(STL_Formula):
    """¬φ."""

    def __init__(self, subformula):
        super().__init__()
        self.subformula = subformula

    @property
    def is_pointwise(self):
        return self.subformula.is_pointwise

    def robustness_trace(self, belief_trajectory, **kwargs):
        return _negation(self.subformula(belief_trajectory, **kwargs))

    def __str__(self):
        return f"¬({self.subformula})"


class _Binary(STL_Formula):
    combine, symbol = None, None

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    @property
    def is_pointwise(self):
        return self.subformula1.is_pointwise and self.subformula2.is_pointwise

    def robustness_trace(self, belief_trajectory, beta=None, **kwargs):
        trace1 = self.subformula1(belief_trajectory, beta=beta, **kwargs)
        trace2 = self.subformula2(belief_trajectory, beta=beta, **kwargs)
        return type(self).combine(*_align(trace1, trace2), beta=beta)

    def __str__(self):
        return f"({self.subformula1}) {self.symbol} ({self.subformula2})"


class And(_Binary):
    """φ₁ ∧ φ₂ (Frechet conjunction)."""

    combine, symbol = _conjunction, "∧"


class Or(_Binary):
    """φ₁ ∨ φ₂ (Frechet disjunction)."""

    combine, symbol = _disjunction, "∨"


class Implies(STL_Formula):
    """φ₁ ⇒ φ₂ := ¬φ₁ ∨ φ₂."""

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.equivalent = Or(Negation(subformula1), subformula2)

    @property
    def is_pointwise(self):
        return self.equivalent.is_pointwise

    def robustness_trace(self, belief_trajectory, **kwargs):
        return self.equivalent(belief_trajectory, **kwargs)

    def __str__(self):
        return f"({self.subformula1}) ⇒ ({self.subformula2})"


# --- Temporal operators ---------------------------------------------------------

# Window end meaning the last step of the trace (the plan end). Traces are
# finite, so G[a, END] at origin k is the finite window k + a .. N; only its
# length depends on k.
END = np.inf


def _window_text(interval):
    """[a, b] as text, with END written as "end"."""
    a, b = interval
    return f"[{a}, {'end' if b == END else b}]"


def _checked_interval(interval):
    """A temporal window [a, b]: whole numbers with 0 <= a <= b, or b = END."""
    if interval is None:
        return None
    a, b = interval
    if b != END and (not np.isfinite(b) or b != int(b)):
        raise ValueError(
            f"interval end must be a whole number or END, got {list(interval)}"
        )
    if not np.isfinite(a) or a != int(a) or a < 0:
        raise ValueError(
            f"interval must start at a whole number >= 0, got {list(interval)}"
        )
    if a > b:
        raise ValueError(f"interval must satisfy a <= b, got {list(interval)}")
    return [int(a), END if b == END else int(b)]


class Temporal_Operator(STL_Formula):
    """Window reduction (min: Always, max: Eventually); beta > 0 smooths.

    At origin k the window is k + a .. k + b, or k + a .. N (the trace end)
    for [a, END]. Every origin's window is reduced at once, exactly (min/max)
    or with the sound smooth reductions of `_reduce_bounds` over that window.
    """

    symbol = None  # subclasses set self.operation to Minish() or Maxish()

    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = _checked_interval(interval)
        self._interval = [0, END] if self.interval is None else self.interval
        self._to_end = self._interval[1] == END

    @property
    def lookahead(self):
        """Future steps one evaluation needs (a for [a, END], else b)."""
        a, b = self._interval
        return int(a) if self._to_end else int(b)

    def _window_trace(self, trace, beta):
        """[a, b]: reduce trace[t + a : t + b + 1] at every origin t."""
        a, b = int(self._interval[0]), int(self._interval[1])
        # [B, origins, 2, n] -> [B, origins, n, 2]
        windows = trace[:, a:].unfold(1, b - a + 1, 1).transpose(-1, -2)
        return _reduce_bounds(
            self.operation, windows, beta, dim=2, keepdim=False
        )

    def _tail_trace(self, trace, beta):
        """[a, END]: reduce trace[k + a :] at every origin k.

        Suffix reductions are cumulative from the trace end; with beta, each
        origin's window of n values gets the normalized soft min/max and the
        log(n)/beta shift, as `_reduce_bounds` gives a fixed window.
        """
        a = int(self._interval[0])
        minimum = isinstance(self.operation, Minish)
        suffix = trace[:, a:].flip(1)  # from the trace end backwards
        if beta is None:
            cumulative = torch.cummin if minimum else torch.cummax
            return cumulative(suffix, dim=1).values.flip(1)
        sign = -1.0 if minimum else 1.0
        log_sums = torch.logcumsumexp(sign * beta * suffix, dim=1).flip(1)
        count = torch.arange(
            suffix.shape[1], 0, -1, dtype=trace.dtype, device=trace.device
        ).view(1, -1, 1)
        smooth = (log_sums - torch.log(count)) / (sign * beta)
        shift = torch.log(count[..., 0]) / beta
        if minimum:
            # Normalized soft-min overshoots min: pull the lower down.
            return _pair(smooth[..., 0] - shift, smooth[..., 1])
        # Normalized soft-max undershoots max: push the upper up.
        return _pair(smooth[..., 0], smooth[..., 1] + shift)

    def robustness_trace(
        self, belief_trajectory, beta=None, keepdim=True, **kwargs
    ):
        trace = self.subformula(
            belief_trajectory, beta=beta, keepdim=keepdim, **kwargs
        )
        if trace.shape[1] - self.lookahead <= 0:
            raise ValueError(
                f"{self}: one evaluation origin needs {self.lookahead + 1} steps, "
                f"child trace has {trace.shape[1]}"
            )
        if self._to_end:
            return self._tail_trace(trace, beta)
        return self._window_trace(trace, beta)

    def __str__(self):
        if self.interval is None:
            return f"{self.symbol}({self.subformula})"
        return (
            f"{self.symbol}_{_window_text(self._interval)}({self.subformula})"
        )


class Always(Temporal_Operator):
    """□_[a,b] φ: minimum over the window."""

    symbol = "□"

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Minish()


class Eventually(Temporal_Operator):
    """♢_[a,b] φ: maximum over the window."""

    symbol = "♢"

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Maxish()


class Until(STL_Formula):
    """φ U_[a,b] ψ: max over witnesses τ of Frechet(min φ on [t, τ], ψ(τ))."""

    def __init__(self, left, right, interval=None):
        super().__init__()
        self.left = left
        self.right = right
        checked = _checked_interval(interval)
        self.interval = [0, END] if checked is None else checked
        self._interval = self.interval
        self.min_op = Minish()
        self.max_op = Maxish()

    @property
    def lookahead(self):
        a, b = self._interval
        return int(a) if b == END else int(b)

    def robustness_trace(
        self, belief_trajectory, beta=None, keepdim=True, **kwargs
    ):
        phi = self.left(belief_trajectory, beta=beta, keepdim=True, **kwargs)
        psi = self.right(belief_trajectory, beta=beta, keepdim=True, **kwargs)
        phi, psi = _align(phi, psi)

        T = phi.shape[1]
        a = int(self._interval[0])
        b = T - 1 if self._interval[1] == END else int(self._interval[1])
        n_valid = T - self.lookahead
        if n_valid <= 0:
            raise ValueError(
                f"{self}: one evaluation origin needs {self.lookahead + 1} steps, "
                f"child traces have {T}"
            )

        if self._interval[1] != END:
            return self._window_trace(phi, psi, n_valid, a, b, beta)
        results = []
        for t in range(n_valid):
            candidates = []
            for tau in range(t + a, min(t + b, T - 1) + 1):
                # Inclusive: φ must hold from t through the witness τ.
                prefix = _reduce_bounds(
                    self.min_op, phi[:, t : tau + 1, :], beta, keepdim=False
                )
                candidates.append(_conjunction(prefix, psi[:, tau, :], beta))
            results.append(
                _reduce_bounds(
                    self.max_op,
                    torch.stack(candidates, dim=1),
                    beta,
                    keepdim=False,
                )
            )
        return torch.stack(results, dim=1)

    def _window_trace(self, phi, psi, n_valid, a, b, beta):
        """Bounded [a, b] at every origin at once (the loop's values).

        Each origin's window phi[t : t + b + 1] gives every prefix min
        phi[t .. tau] as one cumulative reduction; the prefixes ending at
        tau = t + a .. t + b meet psi(tau), and the best one is kept.
        """

        def windows(trace):  # [B, n_valid, b + 1, 2]
            return (
                trace[:, : n_valid + b].unfold(1, b + 1, 1).transpose(-1, -2)
            )

        phi_windows, psi_windows = windows(phi), windows(psi)
        if beta is None:
            prefix = torch.cummin(phi_windows, dim=2).values
        else:
            # Prefix of length n: lower -logsumexp(-beta x) / beta (sound
            # shift included), upper the normalized soft-min.
            log_sums = torch.logcumsumexp(-beta * phi_windows, dim=2)
            length = torch.arange(
                1, b + 2, dtype=phi.dtype, device=phi.device
            ).view(1, 1, -1)
            prefix = _pair(
                -log_sums[..., 0] / beta,
                -(log_sums[..., 1] - torch.log(length)) / beta,
            )
        candidates = _conjunction(
            prefix[:, :, a:], psi_windows[:, :, a:], beta
        )
        return _reduce_bounds(
            self.max_op, candidates, beta, dim=2, keepdim=False
        )

    def __str__(self):
        return f"({self.left}) U_{_window_text(self._interval)} ({self.right})"
