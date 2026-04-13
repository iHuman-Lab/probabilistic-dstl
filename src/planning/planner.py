import logging

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pdstl.base import BeliefTrajectory, Belief

logger = logging.getLogger(__name__)


class TorchGaussianBelief(Belief):
    """Wrapper to allow STL operators to access the tensor trace directly."""

    def __init__(self, mean_full, var_full):
        self.mean_full = mean_full  # [Batch, Dim]
        self.var_full = var_full    # [Batch, Dim] or [Batch, Dim, Dim]

    def value(self):
        return self.mean_full

    def probability_of(self, residual):
        raise NotImplementedError(
            "TorchGaussianBelief is designed for custom predicates that access mean/var directly."
        )


class ProbabilisticSTLPlanner:
    """Gradient-based motion planner for Probabilistic STL specifications."""

    def __init__(self, dynamics, environment, T, config=None):
        self.dyn = dynamics
        self.env = environment
        self.T = T
        self.device = dynamics.device

        self.cfg = {
            "w_u": 0.1,        # Control effort weight
            "w_du": 0.1,       # Smoothness weight
            "w_phi": 10.0,     # STL satisfaction weight
            "lr": 0.05,        # Adam learning rate
            "max_iters": 500,  # Maximum iterations
            "alpha": 0.95,     # Satisfaction threshold for early stopping
            "w_dist": 5.0,     # Goal guidance heuristic weight
            "w_obs": 5.0,      # Obstacle repulsion heuristic weight
            "w_visit": 5.0,    # Visit region heuristic weight
            "loss_tol": 1e-4,  # Loss convergence tolerance
        }
        if config:
            self.cfg.update(config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_controls(self, init_guess):
        """Initialise unconstrained control parameters V (pre-tanh space)."""
        if init_guess is not None:
            u_norm = torch.clamp(init_guess / (self.dyn.u_max + 1e-6), -0.99, 0.99)
            v_init = 0.5 * torch.log((1 + u_norm) / (1 - u_norm))
            return nn.Parameter(v_init.to(self.device), requires_grad=True)
        return nn.Parameter(
            torch.randn(self.T, 2, device=self.device) * 0.1
            + torch.tensor([0.5, 0.0], device=self.device),
            requires_grad=True,
        )

    def _goal_dist_loss(self, mean_trace):
        """Squared distance from final position to goal centre."""
        if self.env.goal is None:
            return torch.tensor(0.0, device=self.device)
        gx = sum(self.env.goal["x"]) / 2.0
        gy = sum(self.env.goal["y"]) / 2.0
        goal_center = torch.tensor([[gx, gy]], device=self.device)
        return torch.sum((mean_trace[:, -1, :2] - goal_center) ** 2)

    def _obs_repulsion_loss(self, mean_trace):
        """Penalise trajectory points that are too close to obstacle centres."""
        loss = torch.tensor(0.0, device=self.device)
        margin = 0.75

        for obs in self.env.obstacles:
            cx = (obs["x"][0] + obs["x"][1]) / 2.0
            cy = (obs["y"][0] + obs["y"][1]) / 2.0
            center = torch.tensor([[cx, cy]], device=self.device)
            radius = max(obs["x"][1] - obs["x"][0], obs["y"][1] - obs["y"][0]) / 2.0 + margin
            dists = torch.norm(mean_trace[:, :, :2] - center, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)

        for obs in self.env.circle_obstacles:
            center = torch.tensor([obs["center"]], device=self.device)
            radius = obs["radius"] + margin
            dists = torch.norm(mean_trace[:, :, :2] - center, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)

        for obs in self.env.moving_obstacles:
            ox = torch.as_tensor(obs["x_traj"], device=self.device)
            oy = torch.as_tensor(obs["y_traj"], device=self.device)
            centers = torch.stack([ox, oy], dim=1).unsqueeze(0)  # [1, T+1, 2]
            radius = max(obs["width"], obs["height"]) / 2.0 + margin
            dists = torch.norm(mean_trace[:, :, :2] - centers, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)

        return loss

    def _visit_loss(self, mean_trace):
        """Pull trajectory towards visit regions (Eventually semantics)."""
        loss = torch.tensor(0.0, device=self.device)
        for region in self.env.visit_regions:
            vx = (region["x"][0] + region["x"][1]) / 2.0
            vy = (region["y"][0] + region["y"][1]) / 2.0
            v_center = torch.tensor([[vx, vy]], device=self.device)
            dists_sq = torch.sum((mean_trace[:, :, :2] - v_center) ** 2, dim=2)
            min_dist_sq, _ = torch.min(dists_sq, dim=1)
            loss = loss + torch.sum(min_dist_sq)
        return loss

    def _compute_loss(self, mean_trace, u_seq, p_all, loss_fn):
        """Compute the total weighted objective J."""
        loss_u = torch.sum(u_seq ** 2)
        u_diff = u_seq[1:] - u_seq[:-1]
        loss_du = torch.sum(u_diff ** 2) + torch.sum(u_seq[0] ** 2)
        loss_phi = loss_fn(p_all) if loss_fn is not None else -torch.log(p_all + 1e-4)

        return (
            self.cfg["w_u"]    * loss_u
            + self.cfg["w_du"]   * loss_du
            + self.cfg["w_phi"]  * loss_phi
            + self.cfg["w_dist"] * self._goal_dist_loss(mean_trace)
            + self.cfg["w_obs"]  * self._obs_repulsion_loss(mean_trace)
            + self.cfg["w_visit"]* self._visit_loss(mean_trace)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, x0_mean, x0_cov, render=False, verbose=True, spec=None, init_guess=None, loss_fn=None):
        """Find optimal controls via gradient descent on the STL objective.

        Parameters
        ----------
        x0_mean, x0_cov : Tensors
            Initial belief state.
        render : bool
            Show a live trajectory plot during optimisation.
        verbose : bool
            Log progress every 50 iterations.
        spec : STL_Formula, optional
            Override the environment's default specification.
        init_guess : Tensor [T, 2], optional
            Warm-start control sequence.
        loss_fn : callable, optional
            Custom loss on p_all; defaults to -log(p + eps).
        """
        v_params = self._init_controls(init_guess)
        optimizer = optim.Adam([v_params], lr=self.cfg["lr"])
        phi = spec if spec is not None else self.env.get_specification(self.T)

        best_u, best_mean, best_cov = None, None, None
        best_p = -float("inf")
        history = []
        prev_loss = float("inf")
        converged_iters = 0

        if verbose:
            logger.info(f"Starting optimisation (max iters: {self.cfg['max_iters']})")

        if render:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-2, 12)
            ax.set_ylim(-2, 12)
            ax.grid(True)
            ax.set_aspect("equal")
            if self.env.goal:
                gx, gy = self.env.goal["x"], self.env.goal["y"]
                ax.add_patch(patches.Rectangle((gx[0], gy[0]), gx[1]-gx[0], gy[1]-gy[0], color="green", alpha=0.3))
            for obs in self.env.obstacles:
                ox, oy = obs["x"], obs["y"]
                ax.add_patch(patches.Rectangle((ox[0], oy[0]), ox[1]-ox[0], oy[1]-oy[0], color="red", alpha=0.5))
            (line,) = ax.plot([], [], "b.-", alpha=0.5)
            title = ax.set_title("Iteration 0")

        for k in range(self.cfg["max_iters"]):
            optimizer.zero_grad()

            mean_trace, cov_trace = self.dyn(v_params, x0_mean, x0_cov)
            u_seq = self.dyn.bound_control(v_params)

            beliefs = [TorchGaussianBelief(mean_trace[:, t, :], cov_trace[:, t]) for t in range(self.T + 1)]
            traj = BeliefTrajectory(beliefs)

            stl_trace = phi(traj)
            p_all = stl_trace[0, 0, 0]

            J = self._compute_loss(mean_trace, u_seq, p_all, loss_fn)
            J.backward()
            optimizer.step()

            current_p = p_all.item()
            history.append(J.item())

            if render and k % 10 == 0:
                path = mean_trace.detach().cpu().squeeze().numpy()
                line.set_data(path[:, 0], path[:, 1])
                title.set_text(f"Iteration {k} | P(Sat): {current_p:.4f}")
                plt.pause(0.01)

            if current_p > best_p:
                best_p = current_p
                best_u = u_seq.detach().clone()
                best_mean = mean_trace.detach().clone()
                best_cov = cov_trace.detach().clone()

            if loss_fn is None and current_p >= self.cfg["alpha"]:
                converged_iters += 1
                if converged_iters >= 50:
                    logger.info(f"Converged at iter {k}. P(Sat): {current_p:.4f}")
                    break
            else:
                converged_iters = 0

            if abs(prev_loss - J.item()) < self.cfg["loss_tol"] and k > 10:
                if verbose:
                    logger.info(f"Loss converged at iter {k}.")
                break
            prev_loss = J.item()

            if verbose and k % 50 == 0:
                logger.info(f"Iter {k:03d} | Loss: {J.item():.4f} | P(Sat): {current_p:.4f} | Best: {best_p:.4f}")

        if render:
            plt.ioff()
            plt.close(fig)

        return best_mean, best_cov, best_u, best_p, history
