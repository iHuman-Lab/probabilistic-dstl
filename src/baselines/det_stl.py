"""
Deterministic STL operators for fair comparison with our probabilistic framework.

This module implements stlcg-style real-valued (signed-distance) STL robustness
using the same RNN dynamic programming logic as stlcg, adapted to:
  - Accept BeliefTrajectory input (uses mean only, ignores covariance)
  - Work in forward time (same convention as our probabilistic operators)
  - Return [B, T+1, 1] robustness traces (vs [B, T+1, 2] probability bounds)

Accessing output[:, 0, 0] gives robustness at t=0, matching stl_trace[0, 0, 0]
used in the probabilistic planner.

Predicates use signed distance:
  - Positive = formula satisfied
  - Negative = formula violated
  - Magnitude = distance to satisfaction/violation boundary
"""

import numpy as np
import torch
import torch.nn as nn

from pdstl.operators import Maxish, Minish


# =============================================================================
# BASE CLASS
# =============================================================================


class DetSTL_Formula(nn.Module):
    """
    Base class for deterministic stlcg-style STL formulas.

    External interface (forward time):
      Input:  BeliefTrajectory (list of beliefs with .mean_full [B, D])
      Output: [B, T+1, 1] robustness trace, forward time

    Internal interface (robustness_trace):
      Input:  mu [B, T+1, D] mean trajectory, forward time
      Output: [B, T+1, 1] robustness trace, forward time
    """

    def robustness_trace(self, mu, scale=-1, **kwargs):
        """
        mu: [B, T+1, D] mean trajectory (forward time)
        Returns: [B, T+1, 1]
        """
        raise NotImplementedError

    def _extract_mean(self, belief_trajectory):
        means = [b.mean_full for b in belief_trajectory]
        return torch.stack(means, dim=1)  # [B, T+1, D]

    def forward(self, belief_trajectory, scale=-1, **kwargs):
        mu = self._extract_mean(belief_trajectory)
        return self.robustness_trace(mu, scale=scale, **kwargs)

    def robustness(self, belief_trajectory, scale=-1, **kwargs):
        """Robustness at t=0 (the planning-relevant quantity)."""
        return self.forward(belief_trajectory, scale=scale, **kwargs)[:, :1, :]

    def __and__(self, other):
        return DetAnd(self, other)

    def __or__(self, other):
        return DetOr(self, other)

    def __invert__(self):
        return DetNegation(self)


# =============================================================================
# TEMPORAL OPERATORS (stlcg RNN logic, forward-time interface)
# =============================================================================


class DetTemporalOperator(DetSTL_Formula):
    """
    Sliding-window temporal operator using stlcg's RNN dynamic programming.

    stlcg processes signals in reversed time order (element 0 = time T).
    Internally we flip the subformula trace to reversed, run the stlcg RNN,
    then flip the output back to forward time. This gives identical results to
    stlcg while keeping the external interface in forward time.
    """

    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if interval is None else interval
        self.rnn_dim = 1 if not interval else interval[-1]
        if self.rnn_dim == np.inf:
            self.rnn_dim = self._interval[0]
        self.steps = 1 if not interval else interval[-1] - interval[0] + 1
        self.operation = None
        # Shift matrices for the sliding window (identical to stlcg)
        M = np.diag(np.ones(self.rnn_dim - 1), k=1)
        self.register_buffer("M", torch.tensor(M, dtype=torch.float32), persistent=False)
        b = torch.zeros(self.rnn_dim, 1, dtype=torch.float32)
        b[-1] = 1.0
        self.register_buffer("b", b, persistent=False)

    def _initialize_rnn_cell(self, x):
        """x: [B, T+1, 1] time-reversed. Init hidden state from first (= time T) element."""
        h0 = torch.ones(x.shape[0], self.rnn_dim, x.shape[2], device=x.device) * x[:, :1, :]
        if (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0 = x[:, :1, :]
            return ((d0, h0), 0.0)
        return (h0, 0.0)

    def _rnn_cell(self, x, hc, scale=-1):
        """Single RNN step. Mirrors stlcg's Always/Eventually._rnn_cell exactly."""
        h0, c = hc
        if self.interval is None:
            input_ = torch.cat([h0, x], dim=1)  # [B, rnn_dim+1, 1]
            output = self.operation(input_, scale, dim=1, keepdim=True)
            state = (output, None)
        elif (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)  # [B, 2, 1]
            output = self.operation(dh, scale, dim=1, keepdim=True)
            state = ((output, torch.matmul(self.M, h0) + self.b * x), None)
        else:  # [a, b]
            state = (torch.matmul(self.M, h0) + self.b * x, None)
            h0x = torch.cat([h0, x], dim=1)  # [B, rnn_dim+1, 1]
            input_ = h0x[:, : self.steps, :]
            output = self.operation(input_, scale, dim=1, keepdim=True)
        return output, state

    def robustness_trace(self, mu, scale=-1, **kwargs):
        # 1. Subformula trace in forward time → [B, T+1, 1]
        sub_fwd = self.subformula.robustness_trace(mu, scale=scale, **kwargs)
        # 2. Reverse → time-reversed (stlcg convention)
        sub_rev = torch.flip(sub_fwd, dims=[1])
        # 3. Run stlcg RNN
        outputs = []
        hc = self._initialize_rnn_cell(sub_rev)
        for xi in torch.split(sub_rev, 1, dim=1):
            o, hc = self._rnn_cell(xi, hc, scale=scale)
            outputs.append(o)
        out_rev = torch.cat(outputs, dim=1)  # [B, T+1, 1] reversed
        # 4. Flip back to forward time
        return torch.flip(out_rev, dims=[1])


class DetAlways(DetTemporalOperator):
    """□[a,b] φ  —  minimum robustness over [a, b]. Uses Minish."""

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Minish()


class DetEventually(DetTemporalOperator):
    """♢[a,b] φ  —  maximum robustness over [a, b]. Uses Maxish."""

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Maxish()


# =============================================================================
# LOGICAL OPERATORS
# =============================================================================


class DetAnd(DetSTL_Formula):
    """φ ∧ ψ  —  min(ρ_φ, ρ_ψ) element-wise."""

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Minish()

    def robustness_trace(self, mu, scale=-1, **kwargs):
        r1 = self.subformula1.robustness_trace(mu, scale=scale, **kwargs)  # [B, T+1, 1]
        r2 = self.subformula2.robustness_trace(mu, scale=scale, **kwargs)
        xx = torch.cat([r1, r2], dim=-1)  # [B, T+1, 2]
        return self.operation(xx, scale, dim=-1, keepdim=True)  # [B, T+1, 1]


class DetOr(DetSTL_Formula):
    """φ ∨ ψ  —  max(ρ_φ, ρ_ψ) element-wise."""

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Maxish()

    def robustness_trace(self, mu, scale=-1, **kwargs):
        r1 = self.subformula1.robustness_trace(mu, scale=scale, **kwargs)
        r2 = self.subformula2.robustness_trace(mu, scale=scale, **kwargs)
        xx = torch.cat([r1, r2], dim=-1)
        return self.operation(xx, scale, dim=-1, keepdim=True)


class DetNegation(DetSTL_Formula):
    """¬φ  —  negate robustness."""

    def __init__(self, subformula):
        super().__init__()
        self.subformula = subformula

    def robustness_trace(self, mu, scale=-1, **kwargs):
        return -self.subformula.robustness_trace(mu, scale=scale, **kwargs)


# =============================================================================
# DETERMINISTIC PREDICATES (signed distance on mean trajectory)
# =============================================================================


class DetRectangularGoalPredicate(DetSTL_Formula):
    """
    Signed distance to being inside goal G = [x_min, x_max] × [y_min, y_max].

        ρ(t) = min(μ_x − x_min, x_max − μ_x, μ_y − y_min, y_max − μ_y)

    Positive iff mean is inside the goal rectangle.
    Mirrors RectangularGoalPredicate from environment.py.
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, mu, **kwargs):
        mu_x = mu[..., 0:1]  # [B, T+1, 1]
        mu_y = mu[..., 1:2]
        stacked = torch.cat(
            [
                mu_x - self.x_min,
                self.x_max - mu_x,
                mu_y - self.y_min,
                self.y_max - mu_y,
            ],
            dim=-1,
        )  # [B, T+1, 4]
        return stacked.min(dim=-1, keepdim=True)[0]  # [B, T+1, 1]


class DetRectangularObstaclePredicate(DetSTL_Formula):
    """
    Signed distance to being outside (safe from) obstacle O = [x_min, x_max] × [y_min, y_max].

        ρ(t) = max(x_min − μ_x, μ_x − x_max, y_min − μ_y, μ_y − y_max)

    Positive iff mean is outside the obstacle (safe).
    Mirrors RectangularObstaclePredicate from environment.py.
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, mu, **kwargs):
        mu_x = mu[..., 0:1]
        mu_y = mu[..., 1:2]
        stacked = torch.cat(
            [
                self.x_min - mu_x,
                mu_x - self.x_max,
                self.y_min - mu_y,
                mu_y - self.y_max,
            ],
            dim=-1,
        )
        return stacked.max(dim=-1, keepdim=True)[0]


class DetCircularObstaclePredicate(DetSTL_Formula):
    """
    Signed distance to being outside a circular obstacle.

        ρ(t) = ||μ(t) − center|| − radius

    Positive iff mean is outside the circle (safe).
    Mirrors CircularObstaclePredicate from environment.py.
    """

    def __init__(self, circle_def, device="cpu"):
        super().__init__()
        self.center = torch.tensor(
            circle_def["center"], dtype=torch.float32, device=device
        )
        self.radius = circle_def["radius"]

    def robustness_trace(self, mu, **kwargs):
        diff = mu[..., :2] - self.center  # [B, T+1, 2]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, T+1, 1]
        return dist - self.radius


class DetMovingRectangularObstaclePredicate(DetSTL_Formula):
    """
    Signed distance to being outside a moving rectangular obstacle.

        ρ(t) = max(x_min(t) − μ_x, μ_x − x_max(t), y_min(t) − μ_y, μ_y − y_max(t))

    x_traj, y_traj are obstacle center positions in forward time, shape [T+1].
    Positive iff mean is outside the obstacle at each time step.
    Mirrors MovingRectangularObstaclePredicate from environment.py.
    """

    def __init__(self, obs_def, device="cpu"):
        super().__init__()
        self.x_traj = torch.as_tensor(
            obs_def["x_traj"], dtype=torch.float32, device=device
        )
        self.y_traj = torch.as_tensor(
            obs_def["y_traj"], dtype=torch.float32, device=device
        )
        self.width = obs_def["width"]
        self.height = obs_def["height"]

    def robustness_trace(self, mu, **kwargs):
        x_min = self.x_traj - self.width / 2.0  # [T+1]
        x_max = self.x_traj + self.width / 2.0
        y_min = self.y_traj - self.height / 2.0
        y_max = self.y_traj + self.height / 2.0

        mu_x = mu[..., 0]  # [B, T+1]
        mu_y = mu[..., 1]

        stacked = torch.stack(
            [x_min - mu_x, mu_x - x_max, y_min - mu_y, mu_y - y_max], dim=-1
        )  # [B, T+1, 4]
        return stacked.max(dim=-1, keepdim=True)[0]  # [B, T+1, 1]


# =============================================================================
# SPECIFICATION BUILDER
# =============================================================================


def det_get_specification(env, T, t_goal_start=0, t_constraints_start=1):
    """
    Build a deterministic STL specification from an Environment object.

    Mirrors Environment.get_specification() but uses signed-distance robustness
    (stlcg-style) instead of probabilistic CDF-based bounds. Operates on the
    mean trajectory only; covariance is ignored.

    Args:
        env: Environment instance (from planning/environment.py)
        T: total time horizon in steps
        t_goal_start: step index for start of goal liveness window
        t_constraints_start: step index for start of safety window (default=1
            skips t=0 initial state, matching get_specification() convention)

    Returns:
        DetSTL_Formula: combined specification.
        Call as: spec(belief_trajectory) → [B, T+1, 1] (forward time)
        Planning objective: spec(belief_trajectory)[0, 0, 0]  (robustness at t=0)
    """
    specs = []

    # 1. Goal (liveness)
    if env.goal:
        goal_pred = DetRectangularGoalPredicate(env.goal)
        specs.append(DetEventually(goal_pred, interval=[t_goal_start, T]))

    # 2. Visit regions (liveness)
    for region in env.visit_regions:
        visit_pred = DetRectangularGoalPredicate(region)
        specs.append(DetEventually(visit_pred, interval=[0, T]))

    # 3. Obstacle safety
    obs_preds = []
    for obs in env.obstacles:
        obs_preds.append(DetRectangularObstaclePredicate(obs))
    for obs in env.circle_obstacles:
        obs_preds.append(DetCircularObstaclePredicate(obs, device=env.device))
    for obs in env.moving_obstacles:
        obs_preds.append(DetMovingRectangularObstaclePredicate(obs, device=env.device))

    if obs_preds:
        safe_formula = obs_preds[0]
        for p in obs_preds[1:]:
            safe_formula = DetAnd(safe_formula, p)
        specs.append(DetAlways(safe_formula, interval=[t_constraints_start, T]))

    # 4. Workspace bounds
    if env.bounds is not None:
        bounds_pred = DetRectangularGoalPredicate(env.bounds)
        specs.append(DetAlways(bounds_pred, interval=[t_constraints_start, T]))

    if not specs:
        raise ValueError("No constraints defined in environment.")

    combined = specs[0]
    for s in specs[1:]:
        combined = DetAnd(combined, s)

    return combined
