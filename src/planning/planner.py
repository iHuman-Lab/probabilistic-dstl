import torch
import torch.nn as nn
import torch.optim as optim
from pdstl.base import BeliefTrajectory, Belief
from utils import load_config
from planning import log_utils


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


class Planner:
    """Gradient-based motion planner for Probabilistic STL specifications."""

    def __init__(self, dynamics, environment, T, config=None):
        self.dyn = dynamics
        self.env = environment
        self.T = T
        self.device = dynamics.device

        self.cfg = {**load_config("configs/planning.yaml"), **(config or {})}

    def _init_controls(self, init_guess):
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
        margin = self.cfg["obs_margin"]

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
            self.cfg["w_u"]     * loss_u
            + self.cfg["w_du"]  * loss_du
            + self.cfg["w_phi"] * loss_phi
            + self.cfg["w_dist"] * self._goal_dist_loss(mean_trace)
            + self.cfg["w_obs"]  * self._obs_repulsion_loss(mean_trace)
            + self.cfg["w_visit"] * self._visit_loss(mean_trace)
        )

    def _optimize_window(
        self, x0_mean, x0_cov, *, env=None, verbose=True,
        spec=None, init_guess=None, loss_fn=None,
    ):
        """Run gradient-descent optimisation for one planning window.

        Parameters
        ----------
        env : Environment, optional
            Override self.env for this window (used by MPC to pass env_t).
        """
        if env is not None:
            saved_env = self.env
            self.env = env

        v_params = self._init_controls(init_guess)
        optimizer = optim.Adam([v_params], lr=self.cfg["lr"])
        phi = spec if spec is not None else self.env.get_specification(self.T)

        best_u, best_mean, best_cov = None, None, None
        best_p = -float("inf")
        history = []
        prev_loss = float("inf")
        converged_iters = 0

        if verbose:
            log_utils._log.info(f"Starting optimisation (max iters: {self.cfg['max_iters']})")

        for k in range(self.cfg["max_iters"]):
            optimizer.zero_grad()

            mean_trace, cov_trace = self.dyn(v_params, x0_mean, x0_cov)
            u_seq = self.dyn.bound_control(v_params)

            beliefs = [
                TorchGaussianBelief(mean_trace[:, t, :], cov_trace[:, t])
                for t in range(self.T + 1)
            ]
            traj = BeliefTrajectory(beliefs)

            stl_trace = phi(traj)
            p_all = stl_trace[0, 0, 0]

            J = self._compute_loss(mean_trace, u_seq, p_all, loss_fn)
            J.backward()
            optimizer.step()

            current_p = p_all.item()
            history.append(J.item())

            if current_p > best_p:
                best_p = current_p
                best_u = u_seq.detach().clone()
                best_mean = mean_trace.detach().clone()
                best_cov = cov_trace.detach().clone()

            if loss_fn is None and current_p >= self.cfg["alpha"]:
                converged_iters += 1
                if converged_iters >= self.cfg["converge_patience"]:
                    log_utils._log.info(f"Converged at iter {k}. P(Sat): {current_p:.4f}")
                    break
            else:
                converged_iters = 0

            if abs(prev_loss - J.item()) < self.cfg["loss_tol"] and k > self.cfg["min_iters"]:
                if verbose:
                    log_utils._log.info(f"Loss converged at iter {k}.")
                break
            prev_loss = J.item()

            if verbose and k % 50 == 0:
                log_utils._log.info(
                    f"Iter {k:03d} | Loss: {J.item():.4f} | "
                    f"P(Sat): {current_p:.4f} | Best: {best_p:.4f}"
                )

        if env is not None:
            self.env = saved_env

        return best_mean, best_cov, best_u, best_p, history

    def _step_with_noise(self, curr_mean, curr_cov, u):
        pred_mean, next_cov = self.dyn.step(curr_mean, curr_cov, u)
        noise = torch.distributions.MultivariateNormal(
            torch.zeros_like(pred_mean), self.dyn.Q
        ).sample()
        return pred_mean + noise, next_cov

    def _empty_u_trace(self, x0_mean):
        return torch.empty(1, 0, 2, device=x0_mean.device, dtype=x0_mean.dtype)

    def _shift_controls(self, prev_u_sol):
        if prev_u_sol is None:
            return None
        return torch.cat([prev_u_sol[1:], prev_u_sol[-1:]], dim=0)

    def _pack_result(
        self,
        *,
        mean_trace,
        cov_trace,
        u_trace,
        p_sat_trace=None,
        loss_trace=None,
        all_plans=None,
        best_p=None,
        mode,
        stopped_reason=None,
    ):
        p_sat_trace = [] if p_sat_trace is None else p_sat_trace
        loss_trace = [] if loss_trace is None else loss_trace
        all_plans = [] if all_plans is None else all_plans
        if best_p is None and p_sat_trace:
            best_p = max(p_sat_trace)
        return {
            "mean_trace": mean_trace,
            "cov_trace": cov_trace,
            "u_trace": u_trace,
            "p_sat_trace": p_sat_trace,
            "loss_trace": loss_trace,
            "history": loss_trace,
            "all_plans": all_plans,
            "best_p": 0.0 if best_p is None else best_p,
            "mode": mode,
            "stopped_reason": stopped_reason,
        }

    def _goal_center(self, env):
        if env.goal is None:
            return None
        gx, gy = env.goal["x"], env.goal["y"]
        return torch.tensor(
            [(gx[0] + gx[1]) / 2, (gy[0] + gy[1]) / 2],
            device=self.device,
        )

    def _mpc_env_for_step(self, step, curr_mean):
        if self.cfg.get("mpc_mode") == "lane_change":
            return self.env.make_local_lane_change_window(step, curr_mean, self.cfg)
        return self.env

    def _log_lane_change_step(self, step, curr_mean, best_p):
        obs_pos = self.env.moving_obstacle_position(step)
        if obs_pos is None:
            return
        ego_pos = curr_mean.detach().cpu().numpy()
        dist = torch.linalg.norm(
            curr_mean[:2] - torch.as_tensor(obs_pos, device=self.device, dtype=curr_mean.dtype)
        ).item()
        if step % 5 == 0:
            log_utils.log_lane_step(step, ego_pos, obs_pos[0], dist, best_p)

    def _lane_change_success(self, curr_mean, success_counter):
        if self.env.success is None:
            return success_counter, False
        ego_y = curr_mean[1].item()
        if self.env.success["y_min"] <= ego_y <= self.env.success["y_max"]:
            success_counter += 1
        else:
            success_counter = 0
        done = success_counter >= self.env.success["consecutive_steps"]
        return success_counter, done

    def _run_mpc_fixed(self, x0_mean, x0_cov, *, verbose, step_callback=None):
        """Fixed-length MPC loop (T_SIM steps); lane-change builds a local env per step."""
        T_SIM = self.cfg["T_SIM"]
        curr_mean, curr_cov = x0_mean, x0_cov
        mean_trace_list = [curr_mean]
        cov_trace_list = [curr_cov]
        u_trace_list = []
        p_sat_trace = []
        loss_trace = []
        all_plans = []
        prev_u_sol = None
        success_counter = 0
        stopped_reason = "T_SIM"

        for t in range(T_SIM):
            env_t = self._mpc_env_for_step(t, curr_mean)

            win_guess = self._shift_controls(prev_u_sol)

            best_mean, best_cov, best_u, best_p, history = self._optimize_window(
                curr_mean, curr_cov,
                env=env_t,
                init_guess=win_guess,
                verbose=False,
            )

            prev_u_sol = best_u.detach()
            all_plans.append(best_mean)
            p_sat_trace.append(best_p)
            loss_trace.append(history[-1] if history else 0.0)

            u_curr = best_u[0]
            next_mean, next_cov = self._step_with_noise(curr_mean, curr_cov, u_curr)

            mean_trace_list.append(next_mean)
            cov_trace_list.append(next_cov)
            u_trace_list.append(u_curr)
            curr_mean = next_mean
            curr_cov = next_cov

            if step_callback is not None:
                step_callback(t, curr_mean, curr_cov, best_mean, best_p)

            if self.cfg.get("mpc_mode") == "lane_change":
                self._log_lane_change_step(t, curr_mean, best_p)
                success_counter, done = self._lane_change_success(curr_mean, success_counter)
                if done:
                    stopped_reason = "lane_change_success"
                    log_utils.log_lane_change_done(self.env.label, t)
                    break

        u_trace = (
            torch.stack(u_trace_list).unsqueeze(0)
            if u_trace_list
            else self._empty_u_trace(x0_mean)
        )
        return self._pack_result(
            mean_trace=torch.stack(mean_trace_list).unsqueeze(0),
            cov_trace=torch.stack(cov_trace_list).unsqueeze(0),
            u_trace=u_trace,
            p_sat_trace=p_sat_trace,
            loss_trace=loss_trace,
            all_plans=all_plans,
            mode="mpc_fixed",
            stopped_reason=stopped_reason,
        )

    def _run_mpc_goal(self, x0_mean, x0_cov, *, verbose, step_callback=None):
        """Goal-distance MPC loop; terminates when ego reaches goal or MAX_STEPS exceeded."""
        MAX_STEPS = self.cfg["MAX_STEPS"]
        curr_mean, curr_cov = x0_mean, x0_cov
        mean_trace_list = [curr_mean]
        cov_trace_list = [curr_cov]
        u_trace_list = []
        p_sat_trace = []
        loss_trace = []
        all_plans = []
        prev_u_sol = None
        stopped_reason = "MAX_STEPS"

        goal_center = self._goal_center(self.env)

        step = 0
        while step < MAX_STEPS:
            dist_to_goal = None
            if goal_center is not None:
                dist_to_goal = torch.norm(curr_mean[:2] - goal_center)
                if dist_to_goal < self.cfg.get("goal_reached_dist", 1.0):
                    stopped_reason = "goal_reached"
                    log_utils.log_goal_reached(step)
                    break

            win_guess = self._shift_controls(prev_u_sol)

            best_mean, best_cov, best_u, best_p, history = self._optimize_window(
                curr_mean, curr_cov,
                verbose=False,
                init_guess=win_guess,
            )

            prev_u_sol = best_u.detach()
            all_plans.append(best_mean)
            p_sat_trace.append(best_p)
            loss_trace.append(history[-1] if history else 0.0)

            u_curr = best_u[0]
            next_mean, next_cov = self._step_with_noise(curr_mean, curr_cov, u_curr)

            mean_trace_list.append(next_mean)
            cov_trace_list.append(next_cov)
            u_trace_list.append(u_curr)
            curr_mean = next_mean
            curr_cov = next_cov

            if step_callback is not None:
                step_callback(step, curr_mean, curr_cov, best_mean, best_p)

            dist_value = dist_to_goal.item() if dist_to_goal is not None else 0.0
            log_utils.log_mpc_step(step, curr_mean.cpu().numpy(), dist_value, best_p)

            step += 1

        u_trace = (
            torch.stack(u_trace_list).unsqueeze(0)
            if u_trace_list
            else self._empty_u_trace(x0_mean)
        )
        return self._pack_result(
            mean_trace=torch.stack(mean_trace_list).unsqueeze(0),
            cov_trace=torch.stack(cov_trace_list).unsqueeze(0),
            u_trace=u_trace,
            p_sat_trace=p_sat_trace,
            loss_trace=loss_trace,
            all_plans=all_plans,
            mode="mpc_goal",
            stopped_reason=stopped_reason,
        )

    def solve(self, x0_mean, x0_cov, *, verbose=True, step_callback=None):
        """Optimise controls; MPC mode triggered by 'T_SIM' or 'MAX_STEPS' in config."""
        if "T_SIM" in self.cfg:
            return self._run_mpc_fixed(x0_mean, x0_cov, verbose=verbose, step_callback=step_callback)
        elif "MAX_STEPS" in self.cfg:
            return self._run_mpc_goal(x0_mean, x0_cov, verbose=verbose, step_callback=step_callback)
        else:
            mean_trace, cov_trace, u_trace, best_p, history = self._optimize_window(
                x0_mean, x0_cov, verbose=verbose
            )
            return self._pack_result(
                mean_trace=mean_trace,
                cov_trace=cov_trace,
                u_trace=u_trace,
                p_sat_trace=[best_p],
                loss_trace=history,
                best_p=best_p,
                mode="single_shot",
                stopped_reason="optimized",
            )
