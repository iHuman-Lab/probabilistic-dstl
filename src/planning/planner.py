import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pdstl.base import BeliefTrajectory, Belief


class TorchGaussianBelief(Belief):
    """
    Wrapper to allow STL operators to access the tensor trace directly.
    """

    def __init__(self, mean_full, var_full):
        self.mean_full = mean_full  # [Batch, Dim]
        self.var_full = var_full  # [Batch, Dim] or [Batch, Dim, Dim]

    def value(self):
        """Returns the mean trace for STL evaluation."""
        return self.mean_full

    def probability_of(self, residual):
        raise NotImplementedError(
            "TorchGaussianBelief is designed for custom predicates that access mean/var directly."
        )


class ProbabilisticSTLPlanner:
    """
    Implements the Gradient-Based Motion Planning Algorithm.
    """

    def __init__(self, dynamics, environment, T, config=None):
        self.dyn = dynamics
        self.env = environment
        self.T = T
        self.device = dynamics.device

        # Optimization Weights
        self.cfg = {
            "w_u": 0.1,  # Control effort weight
            "w_du": 0.1,  # Smoothness weight (delta u)
            "w_phi": 10.0,  # STL Satisfaction weight
            "lr": 0.05,  # Learning rate
            "max_iters": 500,  # K iterations
            "alpha": 0.95,  # Satisfaction threshold for early stop
            "w_dist": 5.0,  # Goal guidance heuristic weight
            "w_obs": 5.0,  # Obstacle repulsion heuristic weight
            "w_visit": 5.0,  # Visit region heuristic weight
            "loss_tol": 1e-4,  # Tolerance for loss convergence
        }
        if config:
            self.cfg.update(config)

    def solve(self, x0_mean, x0_cov, render=False, verbose=True, spec=None, init_guess=None, loss_fn=None):
        """
        Executes the optimization loop to find optimal controls V.
        init_guess: Optional tensor [T, 2] of control inputs to warm-start the optimization.
        """
        # 1. Initialize Control Parameters V
        
        if init_guess is not None:
            # Inverse tanh to get v from u
            # Clamp to avoid numerical instability at boundaries
            u_norm = init_guess / (self.dyn.u_max + 1e-6)
            u_norm = torch.clamp(u_norm, -0.99, 0.99)
            v_init = 0.5 * torch.log((1 + u_norm) / (1 - u_norm))
            v_params = nn.Parameter(v_init.to(self.device), requires_grad=True)
        else:
            # Initialize with small random noise to break symmetry
            # Bias slightly forward (x-direction) to encourage movement
            v_params = nn.Parameter(torch.randn(self.T, 2, device=self.device) * 0.1 + torch.tensor([0.5, 0.0], device=self.device), requires_grad=True)

        # Optimizer
        optimizer = optim.Adam([v_params], lr=self.cfg["lr"])

        # Get the STL formula from the environment
        if spec is not None:
            phi = spec
        else:
            phi = self.env.get_specification(self.T)

        best_u = None
        best_p = -float("inf")
        best_mean = None
        best_cov = None
        history = []
        prev_loss = float("inf")
        converged_iters = 0

        if verbose:
            print(f"Starting Optimization (Max Iters: {self.cfg['max_iters']})...")

        if render:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-2, 12)
            ax.set_ylim(-2, 12)
            ax.grid(True)
            ax.set_aspect("equal")
            # Draw static environment
            if self.env.goal:
                gx = self.env.goal["x"]
                gy = self.env.goal["y"]
                ax.add_patch(
                    patches.Rectangle(
                        (gx[0], gy[0]),
                        gx[1] - gx[0],
                        gy[1] - gy[0],
                        color="green",
                        alpha=0.3,
                    )
                )
            for obs in self.env.obstacles:
                ox = obs["x"]
                oy = obs["y"]
                ax.add_patch(
                    patches.Rectangle(
                        (ox[0], oy[0]),
                        ox[1] - ox[0],
                        oy[1] - oy[0],
                        color="red",
                        alpha=0.5,
                    )
                )
            (line,) = ax.plot([], [], "b.-", alpha=0.5)
            title = ax.set_title("Iteration 0")

        for k in range(self.cfg["max_iters"]):
            optimizer.zero_grad()

            # --- A. Rollout Belief Trajectory ---
            # Forward pass through dynamics
            mean_trace, cov_trace = self.dyn(v_params, x0_mean, x0_cov)
            u_seq = self.dyn.bound_control(v_params)

            # --- B. Wrap for STL Evaluation ---
            # We need to construct the BeliefTrajectory object for the operators
            beliefs = [
                TorchGaussianBelief(mean_trace[:, t, :], cov_trace[:, t])
                for t in range(self.T + 1)
            ]
            traj = BeliefTrajectory(beliefs)

            # --- C. Evaluate STL Satisfaction ---
            # robustness_trace returns [Batch, Time, 2]
            # We want P_lower at t=0
            stl_trace = phi(traj)
            p_all = stl_trace[0, 0, 0]

            # --- D. Compute Objective ---
            # 1. Control Effort: sum ||u_t||^2
            loss_u = torch.sum(u_seq**2)

            # 2. Smoothness: sum ||u_t - u_{t-1}||^2
            u_diff = u_seq[1:] - u_seq[:-1]
            loss_du = torch.sum(u_diff**2) + torch.sum(u_seq[0] ** 2)

            # 3. STL Satisfaction loss
            if loss_fn is not None:
                loss_phi = loss_fn(p_all)
            else:
                loss_phi = -torch.log(p_all + 0.0001)

            # 4. Goal Guidance Heuristic
            # Adds a gradient signal when the robot is far from the goal (P_all ~ 0)
            loss_dist = torch.tensor(0.0, device=self.device)
            if self.env.goal is not None:
                # Calculate Goal Center
                gx = sum(self.env.goal["x"]) / 2.0
                gy = sum(self.env.goal["y"]) / 2.0
                goal_center = torch.tensor([[gx, gy]], device=self.device)

                # Distance from final position to goal center
                loss_dist = torch.sum((mean_trace[:, -1, :2] - goal_center) ** 2)

            # 5. Obstacle Repulsion Heuristic
            # Penalize trajectory points that are too close to obstacle centers
            loss_obs = torch.tensor(0.0, device=self.device)
            
            # Static Rectangular Obstacles
            for obs in self.env.obstacles:
                ox = (obs["x"][0] + obs["x"][1]) / 2.0
                oy = (obs["y"][0] + obs["y"][1]) / 2.0
                obs_center = torch.tensor([[ox, oy]], device=self.device)

                # Approximate radius (half-width + margin)
                w = obs["x"][1] - obs["x"][0]
                h = obs["y"][1] - obs["y"][0]
                radius = max(w, h) / 2.0 + 0.75

                dists = torch.norm(mean_trace[:, :, :2] - obs_center, dim=2)
                loss_obs = loss_obs + torch.sum(torch.relu(radius - dists) ** 2)

            # Circle Obstacles
            for obs in self.env.circle_obstacles:
                center = torch.tensor([obs["center"]], device=self.device)
                radius = obs["radius"] + 0.75  # Margin
                dists = torch.norm(mean_trace[:, :, :2] - center, dim=2)
                loss_obs = loss_obs + torch.sum(torch.relu(radius - dists) ** 2)

            # Moving Obstacles
            for obs in self.env.moving_obstacles:
                # We construct a tensor of centers [1, T+1, 2]
                ox = torch.as_tensor(obs["x_traj"], device=self.device)
                oy = torch.as_tensor(obs["y_traj"], device=self.device)
                centers = torch.stack([ox, oy], dim=1).unsqueeze(0) # [1, T+1, 2]
                
                radius = max(obs["width"], obs["height"]) / 2.0 + 0.75
                dists = torch.norm(mean_trace[:, :, :2] - centers, dim=2)
                loss_obs = loss_obs + torch.sum(torch.relu(radius - dists) ** 2)

            # 6. Visit Region Heuristic
            # Pulls the trajectory towards visit regions (minimizing min_dist over time)
            loss_visit = torch.tensor(0.0, device=self.device)
            for region in self.env.visit_regions:
                vx = (region["x"][0] + region["x"][1]) / 2.0
                vy = (region["y"][0] + region["y"][1]) / 2.0
                v_center = torch.tensor([[vx, vy]], device=self.device)

                # Squared Euclidean distance at each time step
                dists_sq = torch.sum((mean_trace[:, :, :2] - v_center) ** 2, dim=2)

                # We satisfy "Eventually" by minimizing the distance at the closest time step
                min_dist_sq, _ = torch.min(dists_sq, dim=1)
                loss_visit = loss_visit + torch.sum(min_dist_sq)

            # Total Loss
            J = (
                self.cfg["w_u"] * loss_u
                + self.cfg["w_du"] * loss_du
                + self.cfg["w_phi"] * loss_phi
                + self.cfg["w_dist"] * loss_dist
                + self.cfg["w_obs"] * loss_obs
                + self.cfg["w_visit"] * loss_visit
            )

            # --- E. Update ---
            J.backward()
            optimizer.step()

            # Logging
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

            # Convergence Check (only meaningful for probabilistic mode where alpha is a probability)
            if loss_fn is None and current_p >= self.cfg["alpha"]:
                converged_iters += 1
                if converged_iters >= 50:
                    print(
                        f"Converged and held for {converged_iters} iterations. Final P(Sat): {current_p:.4f}. Stopping."
                    )
                    break
            else:
                converged_iters = 0  # Reset if satisfaction drops

            # Loss Convergence Check (Gradient is flat)
            if abs(prev_loss - J) < self.cfg.get("loss_tol", 1e-4):
                if verbose and k > 10:  # Ensure minimum iters
                    print(f"Loss converged at iter {k}. Stopping.")
                break
            prev_loss = J

            if verbose and k % 50 == 0:
                print(f"Iter {k:03d} | Loss: {J.item():.4f} | P(Sat): {current_p:.4f} | Best: {best_p:.4f}")

        if render:
            plt.ioff()
            plt.close(fig)

        return best_mean, best_cov, best_u, best_p, history
