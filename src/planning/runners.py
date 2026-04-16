import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import get_device, load_config
from planning import log_utils
from planning.animation import animate_results
from planning.dynamics import DoubleIntegrator, SingleIntegrator
from planning.environment import Environment
from planning.planner import ProbabilisticSTLPlanner
from planning.visualization import (
    cov_ellipse_params,
    plot_covariance_ellipse,
    setup_lane_change_live_plot,
    setup_mpc_live_plot,
    visualize_lane_change,
    visualize_results,
)

RESULTS_DIR = "saved_data"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_scenario_config(cfg_path):
    """Load a scenario YAML merged with the default planning config.

    Returns
    -------
    cfg : dict
        Full scenario config.
    planner_cfg : dict
        Planning defaults overridden by any scenario-level planner keys.
    """
    cfg = load_config(cfg_path)
    planner_cfg = {**load_config("configs/planning.yaml"), **cfg.get("planner", {})}
    return cfg, planner_cfg


def build_environment(cfg, device):
    """Construct an Environment from a scenario config dict."""
    env = Environment(device=device)
    if "goal" in cfg:
        env.set_goal(**cfg["goal"])
    if "bounds" in cfg:
        env.set_bounds(**cfg["bounds"])
    for vr in cfg.get("visit_regions", []):
        env.add_visit_region(**vr)
    for obs in cfg.get("obstacles", []):
        if obs["type"] == "circle":
            env.add_circle_obstacle(center=obs["center"], radius=obs["radius"])
        else:
            env.add_obstacle(x_range=obs["x_range"], y_range=obs["y_range"])
    return env


def build_initial_belief(cfg, device):
    """Return (x0_mean, x0_cov) tensors from a scenario config dict."""
    x0_mean = torch.tensor(cfg["x0_mean"], device=device)
    x0_cov = torch.eye(len(cfg["x0_mean"]), device=device) * cfg["x0_cov_scale"]
    return x0_mean, x0_cov



def check_collision(mean_trace, env, r_robot=1.0, moving_obs_dist=2.25):
    """Check for collisions between the ego trajectory and environment obstacles."""
    traj = mean_trace.squeeze()  # [T, 2]
    if traj.ndim == 1:
        traj = traj.unsqueeze(0)
    T = traj.shape[0]

    is_safe = True
    min_sep = float("inf")

    for t in range(T):
        ego_pos = traj[t].cpu().numpy()

        # 1. Static Obstacles
        for obs in env.obstacles:
            x_min, x_max = obs["x"]
            y_min, y_max = obs["y"]
            if (x_min - r_robot <= ego_pos[0] <= x_max + r_robot) and (
                y_min - r_robot <= ego_pos[1] <= y_max + r_robot
            ):
                log_utils.log_collision_event(t, "Static obstacle", f"ego={ego_pos}")
                is_safe = False

        # 2. Moving Obstacles
        for obs in env.moving_obstacles:
            xt, yt = obs["x_traj"], obs["y_traj"]
            if t < len(xt):
                ox = xt[t].item() if isinstance(xt, torch.Tensor) else xt[t]
                oy = yt[t].item() if isinstance(yt, torch.Tensor) else yt[t]
                dist = np.linalg.norm(ego_pos[:2] - np.array([ox, oy]))
                if dist < min_sep:
                    min_sep = dist
                if dist < moving_obs_dist:
                    log_utils.log_collision_event(t, "Moving obstacle", f"dist={dist:.2f}")
                    is_safe = False

    log_utils.log_safety(is_safe, min_sep)


def run_single_shot(max_iterations=1000, load_from=None, force_run=False):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config("configs/scenarios/single_shot.yaml")

    T = cfg["T"]
    dt = cfg["dt"]

    if load_from is None:
        load_from = os.path.join(RESULTS_DIR, cfg["save_file"])

    env = build_environment(cfg, device)

    if not force_run and load_from and os.path.exists(load_from):
        log_utils.log_load(load_from)
        data = torch.load(load_from, map_location=device, weights_only=False)
        mean_trace = data["mean_trace"]
        cov_trace = data["cov_trace"]
        u_trace = data["u_trace"]
        history = data["history"]
        best_p = data.get("best_p", 0.0)
        log_utils._log.info(f"Loaded. Final P(Sat): {best_p:.4f}")
    else:
        # --- Setup Dynamics ---
        dynamics = SingleIntegrator(dt=dt, u_max=cfg["u_max"], q_std=cfg["q_std"], device=device)

        planner_cfg["max_iters"] = max_iterations
        planner = ProbabilisticSTLPlanner(dynamics, env, T, config=planner_cfg)

        # --- Initial Condition ---
        x0_mean, x0_cov = build_initial_belief(cfg, device)

        log_utils._log.info("Starting single-shot optimisation...")

        # Run optimization
        mean_trace, cov_trace, u_trace, best_p, history = planner.solve(
            x0_mean, x0_cov, render=True, init_guess=None
        )

        log_utils._log.info(f"Done. Final P(Sat): {best_p:.4f}")

        if load_from:
            torch.save(
                {
                    "mean_trace": mean_trace,
                    "cov_trace": cov_trace,
                    "u_trace": u_trace,
                    "history": history,
                    "best_p": best_p,
                },
                load_from,
            )
            log_utils.log_save(load_from)

    # --- Visualize ---
    visualize_results(mean_trace, cov_trace, u_trace, env, history)

    anim = cfg["animation"]
    animate_results(
        mean_trace,
        cov_trace,
        env,
        filename=anim["filename"],
        step=anim["step"],
        title=anim["title"],
        bounds=anim["bounds"],
    )


def run_mpc(load_from=None, force_run=False):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config("configs/scenarios/mpc.yaml")

    H = cfg["H"]
    MAX_STEPS = cfg["MAX_STEPS"]
    dt = cfg["dt"]

    if load_from is None:
        load_from = os.path.join(RESULTS_DIR, cfg["save_file"])

    env = build_environment(cfg, device)

    if not force_run and os.path.exists(load_from):
        log_utils.log_load(load_from)
        data = torch.load(load_from, map_location=device, weights_only=False)
        full_mean_trace = data["mean_trace"]
        full_cov_trace = data["cov_trace"]
        full_u_trace = data["u_trace"]
        loss_trace = data["loss_trace"]
        p_sat_trace = data["p_sat_trace"]
        all_plans = data["all_plans"]
    else:
        # --- Setup Dynamics ---
        dynamics = SingleIntegrator(dt=dt, u_max=cfg["u_max"], q_std=cfg["q_std"], device=device)

        # --- Initial Condition ---
        x0_mean, x0_cov = build_initial_belief(cfg, device)

        log_utils._log.info(f"Starting MPC execution (horizon={H})...")

        real_mean_trace = [x0_mean]
        real_cov_trace = [x0_cov]
        real_u_trace = []

        curr_mean = x0_mean
        curr_cov = x0_cov

        # Goal center for distance check (midpoint of goal range)
        gx, gy = cfg["goal"]["x_range"], cfg["goal"]["y_range"]
        goal_center = torch.tensor([(gx[0] + gx[1]) / 2, (gy[0] + gy[1]) / 2], device=device)

        fig, ax_map, ax_p, line_exec, line_plan, line_p = setup_mpc_live_plot(env)

        all_plans = []
        p_sat_trace = []
        loss_trace = []

        step = 0
        while step < MAX_STEPS:
            dist_to_goal = torch.norm(curr_mean - goal_center)
            if dist_to_goal < planner_cfg["goal_reached_dist"]:
                log_utils.log_goal_reached(step)
                break

            # Setup Planner for Sliding Window
            mpc_planner = ProbabilisticSTLPlanner(
                dynamics, env, T=H, config=planner_cfg
            )

            # Solve Optimization
            p_mean, _, p_u, p_val, history = mpc_planner.solve(
                curr_mean, curr_cov, render=False, verbose=False
            )

            # Store plan for animation
            all_plans.append(p_mean)
            p_sat_trace.append(p_val)
            # Store the final loss value of this optimization step
            loss_trace.append(history[-1] if history else 0.0)

            # Update Live Plot
            path_x = [m[0].item() for m in real_mean_trace]
            path_y = [m[1].item() for m in real_mean_trace]
            line_exec.set_data(path_x, path_y)

            plan_np = p_mean.detach().cpu().squeeze().numpy()
            line_plan.set_data(plan_np[:, 0], plan_np[:, 1])

            # Update P(Sat) Plot
            line_p.set_data(range(len(p_sat_trace)), p_sat_trace)
            if step > ax_p.get_xlim()[1]:
                ax_p.set_xlim(0, step + 50)

            plt.pause(0.01)  # Pause to render

            # Extract First Control Action (Receding Horizon)
            u_curr = p_u[0]  # [2]

            # Propagate Belief
            pred_mean, next_cov = dynamics.step(curr_mean, curr_cov, u_curr)

            # Simulate Reality (Sample from Process Noise)
            noise = torch.distributions.MultivariateNormal(
                torch.zeros_like(pred_mean), dynamics.Q
            ).sample()
            next_mean = pred_mean + noise

            # Store and Update
            real_mean_trace.append(next_mean)
            real_cov_trace.append(next_cov)
            real_u_trace.append(u_curr)

            curr_mean = next_mean
            curr_cov = next_cov

            log_utils.log_mpc_step(step, curr_mean.cpu().numpy(), dist_to_goal.item(), p_val)
            step += 1

        plt.ioff()
        plt.close(fig)

        # Stack results for visualization
        full_mean_trace = torch.stack(real_mean_trace).unsqueeze(0)  # [1, T, 2]
        full_cov_trace = torch.stack(real_cov_trace).unsqueeze(0)  # [1, T, 2, 2]
        full_u_trace = torch.stack(real_u_trace).unsqueeze(0)  # [1, T-1, 2]

        torch.save(
            {
                "mean_trace": full_mean_trace,
                "cov_trace": full_cov_trace,
                "u_trace": full_u_trace,
                "loss_trace": loss_trace,
                "p_sat_trace": p_sat_trace,
                "all_plans": all_plans,
            },
            load_from,
        )
        log_utils.log_save(load_from)

    # --- Visualize ---
    visualize_results(full_mean_trace, full_cov_trace, full_u_trace, env,
                      history=loss_trace, p_sat_trace=p_sat_trace)

    animate_results(
        full_mean_trace,
        full_cov_trace,
        env,
        filename="mpc_animation.gif",
        plan_traces=all_plans,
        step=cfg["animation"]["step"],
        title=cfg["animation"]["title"],
        bounds=([cfg["bounds"]["x_range"][0], cfg["bounds"]["x_range"][1]],
                [cfg["bounds"]["y_range"][0], cfg["bounds"]["y_range"][1]]),
    )


def _run_lane_change_scenario(cfg_path):
    """Shared implementation for all lane change scenarios.

    All scenario-specific values (dynamics, planner weights, obstacle config,
    road geometry, success thresholds) are read from the YAML file at cfg_path.
    """
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config(cfg_path)
    label = cfg.get("label", "")

    H = cfg["H"]
    T_SIM = cfg["T_SIM"]
    dt = cfg["dt"]

    log_utils.log_scenario_start(label)

    dynamics = DoubleIntegrator(dt=dt, u_max=cfg["u_max"], q_std=cfg["q_std"], device=device)

    # --- Global environment (road + full obstacle trajectory for collision check) ---
    road = cfg["road"]
    env_global = Environment(device=device)
    marking_x = cfg["road"]["marking_x_range"]
    env_global.add_lane_marking(x_range=marking_x, y_pos=road["lane_divider"], style="dashed")
    env_global.add_lane_marking(x_range=marking_x, y_pos=road["y_min"], style="solid")
    env_global.add_lane_marking(x_range=marking_x, y_pos=road["y_max"], style="solid")
    env_global.set_goal(**cfg["goal"])

    obs_cfg = cfg["obstacle"]
    total_points = T_SIM + H + 10
    times = np.arange(total_points) * dt
    obs_x_global = obs_cfg["x0"] + obs_cfg["speed"] * times
    obs_y_global = np.ones_like(times) * obs_cfg["y"]
    env_global.add_moving_obstacle(
        obs_x_global[: T_SIM + 1],
        obs_y_global[: T_SIM + 1],
        width=obs_cfg["width"],
        height=obs_cfg["height"],
    )

    # --- Initial state ---
    curr_mean, curr_cov = build_initial_belief(cfg, device)

    real_mean_trace = [curr_mean]
    real_cov_trace = [curr_cov]
    real_u_trace = []
    loss_trace = []
    p_sat_trace = []
    all_plans = []
    prev_u_sol = None
    success_counter = 0

    # --- Live Visualization ---
    success_cfg = cfg["success"]
    fig, ax, ego_dot, ego_trail, plan_line, ego_cov_patch, obs_rect = setup_lane_change_live_plot(
        road, obs_cfg, obs_x_global[0], obs_y_global[0], success_cfg, label
    )

    # --- MPC Loop ---
    robot_dims = tuple(cfg["robot_dims"])
    goal_lookahead = planner_cfg["mpc_goal_lookahead"]
    goal_window_width = planner_cfg["mpc_goal_window_width"]
    lane_margin = planner_cfg["lane_boundary_margin"]
    goal_y_inset = planner_cfg["goal_y_inset"]
    local_x_range = planner_cfg["mpc_local_x_range"]

    for t in range(T_SIM):
        ego_pos = curr_mean.cpu().numpy()
        curr_x = ego_pos[0]

        env_local = Environment(device=device)
        goal_x_lo = curr_x + goal_lookahead
        goal_x_hi = curr_x + goal_lookahead + goal_window_width
        env_local.set_goal(
            x_range=[goal_x_lo, goal_x_hi],
            y_range=[cfg["goal"]["y_range"][0] + goal_y_inset, cfg["goal"]["y_range"][1] - goal_y_inset],
        )

        y_min_bound = road["y_min"] + lane_margin
        if curr_mean[1] > road["lane_divider"] - lane_margin:
            y_min_bound = road["lane_divider"]
        env_local.set_bounds(x_range=local_x_range, y_range=[y_min_bound, road["y_max"]])

        idx_end = t + H + 1
        if idx_end <= len(obs_x_global):
            sl_x = obs_x_global[t:idx_end]
            sl_y = obs_y_global[t:idx_end]
        else:
            pad = idx_end - len(obs_x_global)
            sl_x = np.concatenate([obs_x_global[t:], np.full(pad, obs_x_global[-1])])
            sl_y = np.concatenate([obs_y_global[t:], np.full(pad, obs_y_global[-1])])
        env_local.add_moving_obstacle(sl_x, sl_y, width=obs_cfg["width"], height=obs_cfg["height"])

        init_guess = None
        if prev_u_sol is not None:
            init_guess = torch.cat([prev_u_sol[1:], prev_u_sol[-1:]], dim=0)

        planner = ProbabilisticSTLPlanner(dynamics, env_local, T=H, config=planner_cfg)
        p_mean, _, p_u, p_val, history = planner.solve(
            curr_mean, curr_cov, render=False, verbose=False, init_guess=init_guess
        )

        prev_u_sol = p_u.detach()
        all_plans.append(p_mean)
        loss_trace.append(history[-1] if history else 0.0)
        p_sat_trace.append(p_val)

        u_curr = p_u[0]
        pred_mean, next_cov = dynamics.step(curr_mean, curr_cov, u_curr)
        noise = torch.distributions.MultivariateNormal(
            torch.zeros_like(pred_mean), dynamics.Q
        ).sample()
        next_mean = pred_mean + noise

        real_mean_trace.append(next_mean)
        real_cov_trace.append(next_cov)
        real_u_trace.append(u_curr)

        curr_mean = next_mean
        curr_cov = next_cov

        obs_pos = np.array([obs_x_global[t], obs_y_global[t]])
        ego_pos = curr_mean.cpu().numpy()
        dist = np.linalg.norm(ego_pos[:2] - obs_pos)

        if t % 5 == 0:
            log_utils.log_lane_step(t, ego_pos, obs_pos[0], dist, p_val)

        ego_x, ego_y = ego_pos[0], ego_pos[1]
        ego_dot.set_data([ego_x], [ego_y])
        ego_trail.set_data(
            [m[0].item() for m in real_mean_trace],
            [m[1].item() for m in real_mean_trace],
        )

        theta, w_e, h_e = cov_ellipse_params(curr_cov[:2, :2].cpu().numpy())
        ego_cov_patch.set_center((ego_x, ego_y))
        ego_cov_patch.set_width(w_e)
        ego_cov_patch.set_height(h_e)
        ego_cov_patch.set_angle(theta)

        plan_np = p_mean.detach().cpu().squeeze().numpy()
        plan_line.set_data(plan_np[:, 0], plan_np[:, 1])
        obs_rect.set_xy((obs_pos[0] - obs_cfg["width"] / 2, obs_pos[1] - obs_cfg["height"] / 2))

        plt.draw()
        plt.pause(0.001)

        if success_cfg["y_min"] <= ego_pos[1] <= success_cfg["y_max"]:
            success_counter += 1
        else:
            success_counter = 0

        if success_counter >= success_cfg["consecutive_steps"]:
            log_utils.log_lane_change_done(label, t)
            break

    plt.ioff()
    plt.close(fig)

    actual_steps = len(real_mean_trace)
    for obs in env_global.moving_obstacles:
        obs["x_traj"] = obs["x_traj"][:actual_steps]
        obs["y_traj"] = obs["y_traj"][:actual_steps]

    full_mean_trace = torch.stack(real_mean_trace).unsqueeze(0)
    full_cov_trace = torch.stack(real_cov_trace).unsqueeze(0)
    full_u_trace = torch.stack(real_u_trace).unsqueeze(0)

    check_collision(
        full_mean_trace, env_global,
        r_robot=planner_cfg["r_robot"],
        moving_obs_dist=planner_cfg["moving_obs_dist"],
    )

    visualize_lane_change(
        full_mean_trace, full_cov_trace, full_u_trace, env_global,
        p_sat_trace=p_sat_trace, dt=dt, robot_dims=robot_dims, xlim=cfg["plot_xlim"],
    )

    animate_results(
        full_mean_trace, full_cov_trace, env_global,
        filename=f"lane_change_{label.lower()}.gif",
        plan_traces=all_plans, step=cfg["animation"]["step"],
        robot_dims=robot_dims, title=f"Lane Change: {label}", bounds=None,
    )


def run_lane_change():
    _run_lane_change_scenario("configs/scenarios/lane_change.yaml")


def run_lane_change_aggressive():
    _run_lane_change_scenario("configs/scenarios/lane_change_aggressive.yaml")

