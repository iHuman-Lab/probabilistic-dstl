import torch
from pdstl.operators import STL_Formula, Always, Eventually, And


class Environment:
    """
    Defines the workspace, obstacles, and goal regions.
    Generates the probabilistic STL specification based on the optimization problem PDF.
    """

    def __init__(self, device="cpu"):
        self.obstacles = []
        self.circle_obstacles = []
        self.visit_regions = []
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

    def get_predicates(self):
        """
        get_predicates Function
        Returns:
        """
        """
        Returns a dictionary of predicates for custom formula construction.
        """
        preds = {"obstacles": [], "visit": [], "goal": None}

        if self.goal:
            preds["goal"] = RectangularGoalPredicate(self.goal)

        for region in self.visit_regions:
            preds["visit"].append(RectangularGoalPredicate(region))

        if self.obstacles or self.circle_obstacles:
            obs_preds = [RectangularObstaclePredicate(obs) for obs in self.obstacles]
            obs_preds.extend(
                [CircularObstaclePredicate(obs, device=self.device) for obs in self.circle_obstacles]
            )
            preds["obstacles"] = obs_preds

        return preds

    def get_specification(self, T, t_goal_start=0):
        """
        Generates the STL formula: phi = (Always Safe) & (Eventually Goal)

        Args:
            T (int): Total time horizon
            t_goal_start (int): Start time for goal satisfaction (t_g in PDF)

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
            phi_safety = Always(current_safe_formula, interval=[0, T])
            specs.append(phi_safety)

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
    return 0.5 * (1 + torch.erf(z / 1.41421356))  # 1.414... is sqrt(2)


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
        # belief_trajectory.mean shape: [Batch, Time, Dim]
        # belief_trajectory.cov shape:  [Batch, Time, Dim, Dim] (or diagonal [Batch, Time, Dim])

        # 1. Extract Means and Variances
        # We iterate over beliefs to extract full traces first
        means = []
        vars_diag = []

        # Assuming belief_trajectory is a list of Belief objects or similar wrapper
        # We extract the underlying tensors
        for belief in belief_trajectory:
            means.append(belief.mean_full)
            # Extract diagonal variance (x and y are independent in axis-aligned checks)
            # If cov is full matrix [B, D, D], take diagonal. If [B, D], take as is.
            if belief.var_full.ndim > 2:
                # Taking diagonal: [B, D]
                diag = torch.diagonal(belief.var_full, dim1=-2, dim2=-1)
                vars_diag.append(diag)
            else:
                vars_diag.append(belief.var_full)

        # Stack to [Batch, Time, Dim]
        # Dim 0 = x, Dim 1 = y
        mu = torch.stack(means, dim=1)
        var = torch.stack(vars_diag, dim=1)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]

        # 2. Compute Probabilities for each edge
        # P(x >= x_min) = 1 - P(x <= x_min) = 1 - CDF(x_min)
        p_xmin = 1.0 - normal_cdf(self.x_min, mu_x, var_x)

        # P(x <= x_max) = CDF(x_max)
        p_xmax = normal_cdf(self.x_max, mu_x, var_x)

        # P(y >= y_min)
        p_ymin = 1.0 - normal_cdf(self.y_min, mu_y, var_y)

        # P(y <= y_max)
        p_ymax = normal_cdf(self.y_max, mu_y, var_y)

        # 3. Combine using Min (Intersection)
        # We stack them to find the element-wise min across the 4 conditions
        stacked_probs = torch.stack([p_xmin, p_xmax, p_ymin, p_ymax], dim=0)
        p_goal, _ = torch.min(stacked_probs, dim=0)  # [Batch, Time]

        # 4. Format Output for Operators
        # operators.py expects [Batch, Time, 2] (Lower, Upper)
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
        # 1. Extract Means and Variances (Same as Goal)
        means = []
        vars_diag = []

        for belief in belief_trajectory:
            means.append(belief.mean_full)
            if belief.var_full.ndim > 2:
                diag = torch.diagonal(belief.var_full, dim1=-2, dim2=-1)
                vars_diag.append(diag)
            else:
                vars_diag.append(belief.var_full)

        mu = torch.stack(means, dim=1)
        var = torch.stack(vars_diag, dim=1)

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
        means = []
        covs = []

        for belief in belief_trajectory:
            means.append(belief.mean_full)
            covs.append(belief.var_full)

        mu = torch.stack(means, dim=1)  # [Batch, Time, Dim]
        sigma_stack = torch.stack(
            covs, dim=1
        )  # [Batch, Time, Dim] or [Batch, Time, Dim, Dim]

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
