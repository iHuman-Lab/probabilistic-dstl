"""Belief contract and trajectory containers: Belief.probability_bounds(event) -> [B, 2]."""

from abc import ABC, abstractmethod

import torch

_ATOL = 1e-6  # round-off allowed outside [0, 1]


class Belief(ABC):
    """Abstract event-probability contract for any uncertainty model."""

    @abstractmethod
    def probability_bounds(self, predicate):
        """[B, 2] lower/upper probability of the event; ValueError if unsupported."""

    def value(self):
        """Representative state [B, D]."""
        raise NotImplementedError(f"{type(self).__name__} has no value()")


class BeliefTrajectory:
    """Ordered per-step beliefs from any concrete ``Belief`` subclass."""

    def __init__(self, beliefs):
        self.beliefs = list(beliefs)

    def __getitem__(self, t):
        return self.beliefs[t]

    def __len__(self):
        return len(self.beliefs)

    def __iter__(self):
        return iter(self.beliefs)

    def suffix(self, t):
        return type(self)(self.beliefs[t:])


def check_probability_bounds(trace, predicate=None):
    """Raise (never clamp) on non-finite, unordered, or out-of-[0, 1] bounds."""
    # One pass when all is well; NaN fails every comparison, so it is caught.
    lower, upper = trace[..., 0], trace[..., 1]
    in_range = (trace >= -_ATOL) & (trace <= 1.0 + _ATOL)
    if bool(in_range.all() & (lower <= upper + _ATOL).all()):
        return
    who = "predicate"
    if predicate is not None:
        who = getattr(predicate, "name", None) or type(predicate).__name__

    bad = (
        ~torch.isfinite(trace).all(dim=-1)
        | (lower > upper + _ATOL)
        | (trace < -_ATOL).any(dim=-1)
        | (trace > 1.0 + _ATOL).any(dim=-1)
    )  # [B, T]

    if bool(bad.any()):
        t = int(bad.any(dim=0).nonzero()[0])
        raise ValueError(
            f"{who}: malformed probability bounds at step {t}: {trace[:, t].tolist()}"
        )
