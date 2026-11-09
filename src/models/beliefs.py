"""Gaussian beliefs: the exact probability of a pdSTL event under N(mean, covariance)."""

import torch

from pdstl.base import Belief, BeliefTrajectory
from pdstl.predicates import AxisInterval, HalfSpace, RelativeAxisInterval

INF = float("inf")
# Covariances checked per eigenvalue call: cuSOLVER on an RTX 3090 (torch
# 2.10) accepts 16384 4x4 matrices per call and rejects 32768.
EIGEN_CHUNK = 16384


def _cdf(z):
    """Phi(z), accurate deep into the lower tail in float32."""
    return torch.exp(torch.special.log_ndtr(z))


def interval_probability(low, high, mean, variance):
    """P(low <= X <= high) for scalar X ~ N(mean, variance). Either bound may be infinite.

    Each case is written as the tail that keeps its float32 resolution: an unbounded side is
    dropped rather than evaluated at infinity, which would make the gradient NaN. A zero
    variance compares deterministically instead of dividing by zero.
    """
    certain = variance <= 0
    sigma = torch.sqrt(
        torch.where(certain, torch.ones_like(variance), variance)
    )

    if low == -INF and high == INF:
        probability = torch.ones_like(mean)
    elif low == -INF:
        probability = _cdf((high - mean) / sigma)
    elif high == INF:
        probability = _cdf((mean - low) / sigma)
    else:
        a, b = (low - mean) / sigma, (high - mean) / sigma
        # Above the mean use the upper tail, Phi(-a) - Phi(-b); otherwise
        # Phi(b) - Phi(a). Picking the arguments first costs two CDFs.
        upper_tail = a > 0
        probability = _cdf(torch.where(upper_tail, -a, b)) - _cdf(
            torch.where(upper_tail, -b, a)
        )

    deterministic = ((mean >= low) & (mean <= high)).to(mean.dtype)
    return torch.where(certain, deterministic, probability.clamp(0.0, 1.0))


def _as_full_covariance(mean, covariance):
    """Expand a [..., D] diagonal covariance to [..., D, D]; pass a full one through."""
    if not torch.is_tensor(covariance):
        raise ValueError("GaussianBelief covariance must be a tensor")
    return (
        torch.diag_embed(covariance)
        if covariance.shape == mean.shape
        else covariance
    )


def _validate(mean, covariance):
    """Shape, finiteness, symmetry and positive semi-definiteness."""
    if not torch.is_tensor(mean) or mean.ndim < 2:
        raise ValueError("GaussianBelief mean must have shape [..., D]")
    if not torch.is_tensor(covariance):
        raise ValueError("GaussianBelief covariance must be a tensor")
    diagonal = covariance.shape == mean.shape
    expected = mean.shape + mean.shape[-1:]
    if not diagonal and covariance.shape != expected:
        raise ValueError(
            f"covariance must be {tuple(expected)}, got {tuple(covariance.shape)}"
        )
    if not torch.isfinite(mean).all():
        raise ValueError("GaussianBelief mean must be finite")
    if not torch.isfinite(covariance).all():
        raise ValueError("GaussianBelief covariance must be finite")
    if diagonal:
        # Diagonal variances are the eigenvalues. Avoid a large CUDA solver
        # batch when Monte Carlo trajectories have deterministic beliefs.
        if bool((covariance < -1e-6).any()):
            raise ValueError("covariance must be positive semi-definite")
        return
    # Tolerances, not equality: A P A^T accumulates round-off.
    if not torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-5):
        raise ValueError("covariance must be symmetric")
    # cuSOLVER rejects very large batches of matrices, so check in chunks.
    matrices = covariance.reshape(-1, *covariance.shape[-2:])
    for chunk in matrices.split(EIGEN_CHUNK):
        if bool((torch.linalg.eigvalsh(chunk) < -1e-6).any()):
            raise ValueError("covariance must be positive semi-definite")


class GaussianBelief(Belief):
    """X ~ N(mean, covariance), with mean [..., D] and covariance [..., D, D].

    A [..., D] diagonal covariance is accepted and expanded. The leading dimensions are free:
    [B, D] is one prediction step, [B, T, D] a whole trace scored in one call.
    """

    def __init__(self, mean, covariance, validate=True):
        if validate:
            _validate(mean, covariance)
        covariance = _as_full_covariance(mean, covariance)
        self.mean, self.covariance = mean, covariance

    def value(self):
        return self.mean

    def probability_bounds(self, event):
        """[..., 2] bounds. Gaussian marginals are exact, so lower == upper."""
        p = self.probability(event)
        return torch.stack((p, p), dim=-1)

    def probability(self, event):
        """[...] exact probability of an atomic event."""
        if isinstance(event, RelativeAxisInterval):
            raise ValueError(
                "relative events require a lane belief trajectory"
            )
        if isinstance(event, AxisInterval):
            variance = self._variance(
                event.dim
            )  # checks the index before indexing the mean
            return interval_probability(
                event.lower, event.upper, self.mean[..., event.dim], variance
            )
        if isinstance(event, HalfSpace):
            normal = self.mean.new_tensor(event.a)
            if normal.shape != self.mean.shape[-1:]:
                raise ValueError(
                    "half-space normal must match the state dimension"
                )
            # A PSD covariance can still project to a tiny negative value by round-off.
            variance = torch.einsum(
                "i,...ij,j->...", normal, self.covariance, normal
            ).clamp_min(0)
            return interval_probability(
                -INF, event.b, self.mean @ normal, variance
            )
        raise ValueError(f"GaussianBelief cannot evaluate {event}")

    def _variance(self, dim):
        if not 0 <= dim < self.mean.shape[-1]:
            raise ValueError(f"predicate dimension {dim} is outside the state")
        return self.covariance[..., dim, dim]


class GaussianBeliefTrajectory(BeliefTrajectory):
    """A whole Gaussian trace: mean [B, T, D], covariance [B, T, D, D].

    Holding the trace as tensors lets an event be scored at all T steps in one call; the
    per-step beliefs remain available for anything that iterates.
    """

    def __init__(self, mean, covariance):
        self.trace = GaussianBelief(mean, covariance)
        # Predicate -> [B, T, 2] bounds, filled as formulas score this trace
        # (the trace is never changed after construction).
        self.bounds_cache = {}
        covariance = self.trace.covariance
        super().__init__(
            GaussianBelief(mean[:, t], covariance[:, t], validate=False)
            for t in range(mean.shape[1])
        )

    def probability_bounds(self, event):
        """[B, T, 2] for every step at once."""
        return self.trace.probability_bounds(event)


class ProbabilityBelief(Belief):
    """Supplied probability bounds for named events, with gradients retained."""

    def __init__(self, bounds, value=None):
        self.bounds = {}
        for name, bound in bounds.items():
            bound = torch.as_tensor(bound)
            if bound.ndim != 2 or bound.shape[-1] != 2:
                raise ValueError(
                    f"bounds[{name!r}] must be [batch, 2], got {tuple(bound.shape)}"
                )
            self.bounds[name] = bound
        self._value = value

    def probability_bounds(self, predicate):
        name = getattr(predicate, "name", None)
        if name not in self.bounds:
            raise ValueError(
                f"no supplied bounds for event {name!r}; have {sorted(self.bounds)}"
            )
        return self.bounds[name]

    def value(self):
        return super().value() if self._value is None else self._value
