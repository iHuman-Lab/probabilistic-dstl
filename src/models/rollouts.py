from typing import NamedTuple

import torch

from models.beliefs import (
    GaussianBeliefTrajectory,
    ProbabilityBelief,
    interval_probability,
)
from pdstl.base import BeliefTrajectory, check_probability_bounds
from pdstl.predicates import RelativeAxisInterval


class BeliefRollout(NamedTuple):
    """Predicted beliefs (what pdSTL reads) plus traces for costs and plots."""

    belief_trajectory: BeliefTrajectory
    aux: dict[str, torch.Tensor] | None = None


def create_gaussian_belief_trajectory(
    mean, covariance, dtype=None, device=None
):
    """Build Gaussian beliefs from [T], [T, D], or [B, T, D] traces."""
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    covariance = torch.as_tensor(
        covariance, dtype=mean.dtype, device=mean.device
    )
    if mean.ndim == 1:
        mean, covariance = mean.unsqueeze(-1), covariance.reshape(-1, 1)
    if mean.ndim == 2:
        mean, covariance = mean.unsqueeze(0), covariance.unsqueeze(0)
    if mean.ndim != 3:
        raise ValueError("mean trace must have shape [T], [T,D] or [B,T,D]")
    if covariance.shape[:2] != mean.shape[:2]:
        raise ValueError(
            "covariance must have the same batch and number of steps as mean"
        )
    return GaussianBeliefTrajectory(mean, covariance)


def create_belief_trajectory(traces, dtype=None, device=None):
    """Build beliefs from named [T, 2] supplied probability intervals."""
    if not traces:
        raise ValueError("at least one event trace is required")
    tensors = {}
    for name, trace in traces.items():
        tensor = torch.as_tensor(trace, dtype=dtype, device=device)
        if tensor.ndim != 2 or tensor.shape[-1] != 2:
            raise ValueError(
                f"probability bounds for {name!r} must have shape [T, 2], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] < 1:
            raise ValueError("probability bounds must cover at least one step")
        check_probability_bounds(tensor.unsqueeze(0))
        tensors[name] = tensor

    steps = {name: tensor.shape[0] for name, tensor in tensors.items()}
    if len(set(steps.values())) > 1:
        raise ValueError(
            f"every trace must cover the same number of steps, got {steps}"
        )
    return BeliefTrajectory(
        ProbabilityBelief(
            {name: tensor[t : t + 1] for name, tensor in tensors.items()}
        )
        for t in range(next(iter(steps.values())))
    )


def create_probability_belief_trajectory(
    predicate, bounds, dtype=None, device=None
):
    """Build beliefs from a single event's [T, 2] probability trace."""
    return create_belief_trajectory(
        {predicate.name: bounds}, dtype=dtype, device=device
    )


def gaussian_rollout(dynamics, mean0, cov0):
    """Return rollout(v) -> BeliefRollout of Gaussian beliefs from (mean0, cov0)."""

    def rollout(v):
        mean_trace, cov_trace = dynamics(v, mean0, cov0)
        return BeliefRollout(
            create_gaussian_belief_trajectory(mean_trace[0], cov_trace[0]),
            {"mean_trace": mean_trace, "cov_trace": cov_trace},
        )

    return rollout


def sample_trajectories(dynamics, mean0, cov0, controls, samples, generator):
    """Noisy trajectories x' = A x + B u + w, w ~ N(0, Q), under fixed controls.

    Args:
        dynamics: Linear dynamics with A, B, Q.
        mean0, cov0: Gaussian initial state.
        controls: [H, m] physical controls.
        samples: Number of trajectories N.
        generator: torch.Generator for reproducible draws.

    Returns:
        [N, H+1, D] sampled states.
    """

    def draw(covariance):
        # Cholesky of a PSD matrix that may be singular (zero variances).
        jitter = 1e-9 * torch.eye(len(covariance), device=covariance.device)
        factor = torch.linalg.cholesky(covariance + jitter)
        noise = torch.randn(
            samples,
            len(covariance),
            generator=generator,
            device=covariance.device,
            dtype=covariance.dtype,
        )
        return noise @ factor.T

    state = mean0 + draw(cov0)
    states = [state]
    for control in controls:
        state = (
            state @ dynamics.A.T + control @ dynamics.B.T + draw(dynamics.Q)
        )
        states.append(state)
    return torch.stack(states, dim=1)


class LaneBeliefTrajectory(GaussianBeliefTrajectory):
    """Ego beliefs with vectorized relative beliefs for named traffic."""

    def __init__(self, mean, covariance, traffic_mean, traffic_cov, names):
        super().__init__(mean, covariance)
        self.traffic_mean = traffic_mean
        self.traffic_cov = traffic_cov
        self.names = tuple(names)
        self.relative_mean = traffic_mean[..., :2] - mean[..., :2].unsqueeze(2)
        self.relative_cov = traffic_cov[..., :2, :2] + covariance[
            ..., :2, :2
        ].unsqueeze(2)

    def relative_belief(self, vehicle):
        index = self.names.index(vehicle)
        mean = self.relative_mean[:, :, index]
        covariance = self.relative_cov[:, :, index]
        return GaussianBeliefTrajectory(mean, covariance)

    def probability_bounds(self, event):
        if isinstance(event, RelativeAxisInterval):
            index = self.names.index(event.vehicle)
            mean = self.relative_mean[:, :, index, event.dim]
            variance = self.relative_cov[:, :, index, event.dim, event.dim]
            probability = interval_probability(
                event.lower, event.upper, mean, variance
            )
            return torch.stack((probability, probability), dim=-1)
        return super().probability_bounds(event)


def lane_rollout(
    dynamics,
    ego_mean,
    ego_cov,
    traffic_mean,
    traffic_cov,
    traffic_q,
    names,
    speed_bounds,
    accel_bounds,
):
    """Predict ego and constant-velocity traffic with physical controls.

    Covariances and traffic do not depend on the controls, so they are
    predicted once here for the longest window and sliced per rollout.
    """
    traffic_a = dynamics.A
    covariances, traffic_means, traffic_covariances = [], [], []
    covariance, traffic_m, traffic_c = ego_cov, traffic_mean, traffic_cov
    cached_steps = 0

    def extend(steps):
        """Predict the control-free parts out to `steps` steps."""
        nonlocal covariance, traffic_m, traffic_c, cached_steps
        if not covariances:
            covariances.append(ego_cov)
            traffic_means.append(traffic_mean)
            traffic_covariances.append(traffic_cov)
        while cached_steps < steps:
            covariance = dynamics.A @ covariance @ dynamics.A.T + dynamics.Q
            traffic_m = traffic_m @ traffic_a.T
            traffic_c = traffic_a @ traffic_c @ traffic_a.T + traffic_q
            covariances.append(covariance)
            traffic_means.append(traffic_m)
            traffic_covariances.append(traffic_c)
            cached_steps += 1

    def rollout(parameters):
        steps = len(parameters)
        extend(steps)
        means = [ego_mean]
        # tanh and the lateral clamp act on each step alone: do all at once.
        proposed = dynamics.bound_control(parameters)
        lateral = proposed[:, 1].clamp(*accel_bounds[1])
        applied = []
        for k in range(steps):
            # The longitudinal limit keeps the next speed within bounds, so
            # it depends on the speed reached so far.
            speed = means[-1][2]
            lower = torch.clamp(
                (speed_bounds[0] - speed) / dynamics.dt, min=accel_bounds[0][0]
            )
            upper = torch.clamp(
                (speed_bounds[1] - speed) / dynamics.dt, max=accel_bounds[0][1]
            )
            longitudinal = torch.clamp(proposed[k, 0], min=lower, max=upper)
            control = torch.stack((longitudinal, lateral[k]))
            applied.append(control)
            means.append(dynamics.A @ means[-1] + dynamics.B @ control)
        mean_trace = torch.stack(means).unsqueeze(0)
        cov_trace = torch.stack(covariances[: steps + 1]).unsqueeze(0)
        traffic_mean_trace = torch.stack(traffic_means[: steps + 1]).unsqueeze(
            0
        )
        traffic_cov_trace = torch.stack(
            traffic_covariances[: steps + 1]
        ).unsqueeze(0)
        controls = torch.stack(applied)
        return BeliefRollout(
            LaneBeliefTrajectory(
                mean_trace,
                cov_trace,
                traffic_mean_trace,
                traffic_cov_trace,
                names,
            ),
            {
                "mean_trace": mean_trace,
                "cov_trace": cov_trace,
                "traffic_mean_trace": traffic_mean_trace,
                "traffic_cov_trace": traffic_cov_trace,
                "applied_controls": controls,
            },
        )

    return rollout
