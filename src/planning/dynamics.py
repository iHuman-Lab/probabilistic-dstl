import torch
import torch.nn as nn


class Dynamics(nn.Module):
    """
    Base class for system dynamics.
    Handles control bounding and common initialization.
    """

    def __init__(self, dt, u_max, device="cpu"):
        super().__init__()
        self.dt = dt
        self.u_max = u_max
        self.device = device

    def bound_control(self, v):
        """
        Applies smooth squashing to keep control within [-u_max, u_max].
        v: Unconstrained optimization variable (the 'knobs' for the optimizer)
        u: Physical control input applied to the robot
        """
        return self.u_max * torch.tanh(v)

    def forward(self, v_sequence, x0_mean, x0_cov):
        raise NotImplementedError


class SingleIntegrator(Dynamics):
    """
    Standard Position-Velocity model defined in the PDF.

    State:   [x, y]
    Control: [vx, vy] (Direct velocity control)

    Equations:
      mu_{t+1}   = mu_t + u_t * dt
      Sigma_{t+1} = Sigma_t + Q
    """

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.05, device="cpu"):
        super().__init__(dt, u_max, device)

        # Process Noise Covariance Q (Additive)
        # We assume diagonal noise for simplicity: Q = diag(q_std^2)
        self.Q = torch.eye(2, device=self.device) * q_std**2

    def forward(self, v_sequence, x0_mean, x0_cov):
        """
        Rolls out the trajectory from t=0 to T.

        Args:
            v_sequence: Tensor [T, 2] (Unconstrained controls)
            x0_mean:    Tensor [2]    (Initial position)
            x0_cov:     Tensor [2, 2] (Initial uncertainty)

        Returns:
            mean_stack: [1, T+1, 2]
            cov_stack:  [1, T+1, 2, 2]
        """
        T = v_sequence.shape[0]

        # Storage for the trajectory
        means = [x0_mean]
        covs = [x0_cov]

        curr_mu = x0_mean
        curr_sigma = x0_cov

        for t in range(T):
            # 1. Squash the optimization variable to get physical control
            #    u_t = u_max * tanh(v_t)
            u = self.bound_control(v_sequence[t])

            # 2. Update Mean (Differentiable)
            #    mu_{t+1} = mu_t + u_t * dt
            curr_mu = curr_mu + u * self.dt

            # 3. Update Covariance (Open Loop Uncertainty Growth)
            #    Sigma_{t+1} = Sigma_t + Q
            curr_sigma = curr_sigma + self.Q

            means.append(curr_mu)
            covs.append(curr_sigma)

        # Stack results into tensors
        # Output shape: [Batch=1, Time, Dim]
        mean_stack = torch.stack(means).unsqueeze(0)
        cov_stack = torch.stack(covs).unsqueeze(0)

        return mean_stack, cov_stack


class DoubleIntegrator(Dynamics):
    """
    Alternative Physics-based model (Acceleration control).

    State:   [px, py, vx, vy]
    Control: [ax, ay]

    Equations:
      mu_{t+1}    = A * mu_t + B * u_t
      Sigma_{t+1} = A * Sigma_t * A^T + Q
    """

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.02, device="cpu"):
        super().__init__(dt, u_max, device)

        # State Transition Matrix A
        self.A = torch.tensor(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )

        # Control Matrix B
        self.B = torch.tensor(
            [[0.5 * dt**2, 0.0], [0.0, 0.5 * dt**2], [dt, 0.0], [0.0, dt]],
            device=device,
        )

        # Process Noise Q
        self.Q = torch.eye(4, device=device) * q_std**2

    def forward(self, v_sequence, x0_mean, x0_cov):
        T = v_sequence.shape[0]
        means = [x0_mean]
        covs = [x0_cov]

        curr_mu = x0_mean
        curr_sigma = x0_cov

        for t in range(T):
            # 1. Bound Control
            u = self.bound_control(v_sequence[t])

            # 2. Update Mean
            curr_mu = self.A @ curr_mu + self.B @ u

            # 3. Update Covariance (Full Linear Update)
            #    Sigma_{t+1} = A * Sigma_t * A^T + Q
            curr_sigma = self.A @ curr_sigma @ self.A.t() + self.Q

            means.append(curr_mu)
            covs.append(curr_sigma)

        mean_stack = torch.stack(means).unsqueeze(0)
        cov_stack = torch.stack(covs).unsqueeze(0)

        return mean_stack, cov_stack
