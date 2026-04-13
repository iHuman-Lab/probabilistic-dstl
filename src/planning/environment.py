import math

import torch

from pdstl.operators import Always, And, Eventually, STL_Formula


def extract_trajectory_stats(belief_trajectory, diagonal_only=True):
    """Stack mean and covariance tensors from a belief trajectory.

    Parameters
    ----------
    belief_trajectory : list of TorchGaussianBelief
    diagonal_only : bool
        If True, extract only the diagonal of full covariance matrices,
        returning var of shape [Batch, Time, Dim].
        If False, stack full covariance matrices as-is.

    Returns
    -------
    mu  : Tensor [Batch, Time, Dim]
    var : Tensor [Batch, Time, Dim] (diagonal_only=True)
          or [Batch, Time, Dim] / [Batch, Time, Dim, Dim] (diagonal_only=False)
    """
    means, vars_ = [], []
    for belief in belief_trajectory:
        means.append(belief.mean_full)
        if diagonal_only and belief.var_full.ndim > 2:
            vars_.append(torch.diagonal(belief.var_full, dim1=-2, dim2=-1))
        else:
            vars_.append(belief.var_full)
    return torch.stack(means, dim=1), torch.stack(vars_, dim=1)


class Environment:
    """
    Defines the workspace, obstacles, and goal regions.
    Generates the probabilistic STL specification based on the optimization problem PDF.
    """

    def __init__(self, device="cpu"):
        self.obstacles = []
        self.circle_obstacles = []
        self.moving_obstacles = []
        self.visit_regions = []
        self.lane_markings = []
        self.goal = None
        self.bounds = None
        self.device = device

    def add_obstacle(self, x_range, y_range):
        """
        Adds an axis-aligned rectangular obstacle O_i = [x_min, x_max] x [y_min, y_max].
        """
        self.obstacles.append(
            {
                "x": x_range,  # (min, max)
                "y": y_range,  # (min, max)
            }
        )

    def add_circle_obstacle(self, center, radius):
        """
        Adds a circular obstacle defined by center [x, y] and radius r.
        """
        self.circle_obstacles.append({"center": center, "radius": radius})

    def add_moving_obstacle(self, x_traj, y_traj, width, height):
        """
        Adds a moving rectangular obstacle.
        x_traj, y_traj: Tensors or lists of center positions over time [T+1]
        """
        self.moving_obstacles.append(
            {"x_traj": x_traj, "y_traj": y_traj, "width": width, "height": height}
        )

    def add_lane_marking(self, x_range, y_pos, style="dashed", color="white"):
        """
        Adds a visual lane marking (line).
        """
        self.lane_markings.append(
            {"x": x_range, "y": y_pos, "style": style, "color": color}
        )

    def add_visit_region(self, x_range, y_range):
        """
        Adds a rectangular region that must be visited at some point.
        """
        self.visit_regions.append({"x": x_range, "y": y_range})

    def set_goal(self, x_range, y_range):
        """
        Sets the goal region G = [x_g_min, x_g_max] x [y_g_min, y_g_max].
        """
        self.goal = {"x": x_range, "y": y_range}

    def set_bounds(self, x_range, y_range):
        """
        Sets hard workspace boundaries. The trajectory must always stay inside.
        """
        self.bounds = {"x": x_range, "y": y_range}

    def get_predicates(self):
        """ """
        preds = {"obstacles": [], "visit": [], "goal": None}

        if self.goal:
            preds["goal"] = RectangularGoalPredicate(self.goal)

        for region in self.visit_regions:
            preds["visit"].append(RectangularGoalPredicate(region))

        if self.obstacles or self.circle_obstacles or self.moving_obstacles:
            obs_preds = [RectangularObstaclePredicate(obs) for obs in self.obstacles]
            obs_preds.extend(
                [
                    CircularObstaclePredicate(obs, device=self.device)
                    for obs in self.circle_obstacles
                ]
            )
            obs_preds.extend(
                [
                    MovingRectangularObstaclePredicate(obs, device=self.device)
                    for obs in self.moving_obstacles
                ]
            )
            preds["obstacles"] = obs_preds

        return preds

    def get_specification(self, T, t_goal_start=0, t_constraints_start=1):
        """
        Generates the STL formula: phi = (Always Safe) & (Eventually Goal)

        Args:
            T (int): Total time horizon
            t_goal_start (int): Start time for goal satisfaction (t_g in PDF)
            t_constraints_start (int): Start time for safety/bounds constraints.
                Default=1 skips t=0 (the known initial state), which sits on the
                workspace boundary (x=0.0 == x_min), so including t=0 would make
                p_bounds = 0.5 and cap P_sat at ~0.45 regardless of plan quality.

        Returns:
            STL_Formula: The combined specification
        """
        preds = self.get_predicates()
        specs = []

        # 1. Goal Specification (Liveness)
        if preds["goal"]:
            specs.append(Eventually(preds["goal"], interval=[t_goal_start, T]))

        # 2. Visit Regions (Liveness)
        for visit_pred in preds["visit"]:
            specs.append(Eventually(visit_pred, interval=[0, T]))

        # 3. Obstacle Specification (Safety)
        if preds["obstacles"]:
            obs_preds = preds["obstacles"]
            current_safe_formula = obs_preds[0]
            for i in range(1, len(obs_preds)):
                current_safe_formula = And(current_safe_formula, obs_preds[i])
            phi_safety = Always(current_safe_formula, interval=[t_constraints_start, T])
            specs.append(phi_safety)

        # 4. Workspace Boundary (Always stay inside)
        if self.bounds is not None:
            bounds_pred = RectangularGoalPredicate(self.bounds)
            specs.append(Always(bounds_pred, interval=[t_constraints_start, T]))

        if not specs:
            raise ValueError("No constraints defined in environment.")

        # 4. Combined Specification
        combined_spec = specs[0]
        for i in range(1, len(specs)):
            combined_spec = And(combined_spec, specs[i])

        return combined_spec


# =============================================================================
# PROBABILISTIC PREDICATES (PDF Section 5)
# =============================================================================


def normal_cdf(value, mean, var):
    """
    Computes P(X <= value) for X ~ N(mean, var).
    Standard Normal CDF Phi(z) where z = (value - mean) / sigma
    """
    std = torch.sqrt(var + 1e-6)  # Add epsilon for stability
    z = (value - mean) / std
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


class RectangularGoalPredicate(STL_Formula):
    """
    Implements PDF Eq (9):
    P_goal(t) = min( P(x >= x_min), P(x <= x_max), P(y >= y_min), P(y <= y_max) )
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        # We process the entire trajectory at once

        # 1. Extract Means and Variances

        mu, var = extract_trajectory_stats(belief_trajectory)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]

        # 2. Compute Probabilities for intervals (assuming independence)
        # P(x_min <= x <= x_max) = CDF(x_max) - CDF(x_min)
        p_x = normal_cdf(self.x_max, mu_x, var_x) - normal_cdf(self.x_min, mu_x, var_x)

        # P(y_min <= y <= y_max) = CDF(y_max) - CDF(y_min)
        p_y = normal_cdf(self.y_max, mu_y, var_y) - normal_cdf(self.y_min, mu_y, var_y)

        # 3. Combine using Product (Independence)
        # This is more accurate for a rectangular region than min()
        p_goal = torch.clamp(p_x * p_y, min=0.0, max=1.0)

        # 4. Format Output for Operators
        # Since we calculated exact probabilities (surrogates), Lower = Upper
        return torch.stack([p_goal, p_goal], dim=-1)


class RectangularObstaclePredicate(STL_Formula):
    """
    Implements PDF Eq (10):
    P_safe(t) = max( P(x <= x_min), P(x >= x_max), P(y <= y_min), P(y >= y_max) )
    (Safe if Left OR Right OR Below OR Above)
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]

        # 2. Compute Probabilities for being OUTSIDE
        # P(x <= x_min) (Left of Obs)
        p_left = normal_cdf(self.x_min, mu_x, var_x)

        # P(x >= x_max) (Right of Obs)
        p_right = 1.0 - normal_cdf(self.x_max, mu_x, var_x)

        # P(y <= y_min) (Below Obs)
        p_below = normal_cdf(self.y_min, mu_y, var_y)

        # P(y >= y_max) (Above Obs)
        p_above = 1.0 - normal_cdf(self.y_max, mu_y, var_y)

        # 3. Combine using Max (Union)
        # Safe if ANY of these are true
        stacked_probs = torch.stack([p_left, p_right, p_below, p_above], dim=0)
        p_safe, _ = torch.max(stacked_probs, dim=0)

        # 4. Format Output
        return torch.stack([p_safe, p_safe], dim=-1)


class CircularObstaclePredicate(STL_Formula):
    """
    Implements probabilistic safety for a circular obstacle.
    P_safe(t) = P( ||x(t) - center|| > radius )
    Approximated using projected variance along the radial vector.
    """

    def __init__(self, circle_def, device="cpu"):
        super().__init__()
        self.center = torch.tensor(
            circle_def["center"], device=device, dtype=torch.float32
        )
        self.radius = circle_def["radius"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, sigma_stack = extract_trajectory_stats(belief_trajectory, diagonal_only=False)

        # Distance vector from center
        diff = mu - self.center
        dist = torch.norm(diff, dim=-1)  # [Batch, Time]

        # Normalized direction vector
        dir_vec = diff / (dist.unsqueeze(-1) + 1e-6)  # [Batch, Time, Dim]

        # Projected Variance along the radial direction
        if sigma_stack.ndim == 3:  # Diagonal Covariance [Batch, Time, Dim]
            sigma_proj = torch.sum(dir_vec**2 * sigma_stack, dim=-1)
        else:  # Full Covariance [Batch, Time, Dim, Dim]
            # v^T * Sigma * v
            sigma_proj = torch.einsum("bti,btij,btj->bt", dir_vec, sigma_stack, dir_vec)

        # P(safe) = P(actual_dist > radius) ~= 1 - CDF(radius | N(dist, sigma_proj))
        p_safe = 1.0 - normal_cdf(self.radius, dist, sigma_proj)

        return torch.stack([p_safe, p_safe], dim=-1)


class MovingRectangularObstaclePredicate(STL_Formula):
    """
    Probabilistic safety for a moving rectangular obstacle.
    """

    def __init__(self, obs_def, device="cpu"):
        super().__init__()
        # Trajectories are expected to be tensors of shape [T+1]
        self.x_traj = torch.as_tensor(
            obs_def["x_traj"], device=device, dtype=torch.float32
        )
        self.y_traj = torch.as_tensor(
            obs_def["y_traj"], device=device, dtype=torch.float32
        )
        self.width = obs_def["width"]
        self.height = obs_def["height"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]

        # Expand obstacle bounds over time
        x_min = self.x_traj - self.width / 2.0
        x_max = self.x_traj + self.width / 2.0
        y_min = self.y_traj - self.height / 2.0
        y_max = self.y_traj + self.height / 2.0

        # Compute Probabilities (Safe if Outside)
        p_left = normal_cdf(x_min, mu_x, var_x)
        p_right = 1.0 - normal_cdf(x_max, mu_x, var_x)
        p_below = normal_cdf(y_min, mu_y, var_y)
        p_above = 1.0 - normal_cdf(y_max, mu_y, var_y)

        stacked_probs = torch.stack([p_left, p_right, p_below, p_above], dim=0)
        p_safe, _ = torch.max(stacked_probs, dim=0)

        return torch.stack([p_safe, p_safe], dim=-1)
