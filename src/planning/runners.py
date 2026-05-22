import os

import numpy as np
import torch

from utils import get_device, load_config
from planning import log_utils
from planning.dynamics import DoubleIntegrator, SingleIntegrator
from planning.environment import Environment
from planning.planner import Planner
from visualization.animation import animate_results
from visualization.live_plots import make_mpc_live_callback, make_lane_change_live_callback
from visualization.planning import visualize_lane_change, visualize_results

RESULTS_DIR = "saved_data"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def load_scenario_config(cfg_path):
    cfg = load_config(cfg_path)
    planner_cfg = {**load_config("configs/planning.yaml"), **cfg.get("planner", {})}
    return cfg, planner_cfg


def build_environment(cfg, device):
    env = Environment(device=device)
    if "road" in cfg and "obstacle" in cfg:
        env.configure_lane_change(
            road=cfg["road"],
            obstacle=cfg["obstacle"],
            goal=cfg["goal"],
            success=cfg["success"],
            horizon=cfg["H"],
            total_steps=cfg["T_SIM"],
            dt=cfg["dt"],
            label=cfg.get("label", ""),
            plot_xlim=cfg.get("plot_xlim"),
            robot_dims=cfg.get("robot_dims"),
        )
        return env

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
    x0_mean = torch.tensor(cfg["x0_mean"], device=device)
    x0_cov = torch.eye(len(cfg["x0_mean"]), device=device) * cfg["x0_cov_scale"]
    return x0_mean, x0_cov


def build_dynamics(cfg, device):
    kind = cfg.get("dynamics", "single_integrator")
    if kind == "double_integrator":
        return DoubleIntegrator(
            dt=cfg["dt"],
            u_max=cfg["u_max"],
            q_std=cfg["q_std"],
            device=device,
        )
    return SingleIntegrator(dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"], device=device)


def check_collision(mean_trace, env, r_robot=1.0, moving_obs_dist=2.25):
    traj = mean_trace.squeeze()  # [T, 2]
    if traj.ndim == 1:
        traj = traj.unsqueeze(0)
    T = traj.shape[0]

    is_safe = True
    min_sep = float("inf")

    for t in range(T):
        ego_pos = traj[t].cpu().numpy()

        for obs in env.obstacles:
            x_min, x_max = obs["x"]
            y_min, y_max = obs["y"]
            if (x_min - r_robot <= ego_pos[0] <= x_max + r_robot) and (
                y_min - r_robot <= ego_pos[1] <= y_max + r_robot
            ):
                log_utils.log_collision_event(t, "Static obstacle", f"ego={ego_pos}")
                is_safe = False

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


def _scenario_result_path(cfg, load_from):
    if load_from is not None:
        return load_from
    if "save_file" not in cfg:
        return None
    return os.path.join(RESULTS_DIR, cfg["save_file"])


def _normalise_result(data):
    result = dict(data)
    if "loss_trace" not in result and "history" in result:
        result["loss_trace"] = result["history"]
    if "history" not in result and "loss_trace" in result:
        result["history"] = result["loss_trace"]
    result.setdefault("p_sat_trace", [result.get("best_p", 0.0)])
    result.setdefault("all_plans", [])
    result.setdefault("best_p", max(result["p_sat_trace"]) if result["p_sat_trace"] else 0.0)
    result.setdefault("mode", "loaded")
    result.setdefault("stopped_reason", None)
    return result


def _load_or_solve(cfg, planner_cfg, env, *, horizon, load_from=None, force_run=False,
                   make_callback=None):
    result_path = _scenario_result_path(cfg, load_from)
    if not force_run and result_path and os.path.exists(result_path):
        log_utils.log_load(result_path)
        return _normalise_result(
            torch.load(result_path, map_location=env.device, weights_only=False)
        )

    step_callback = make_callback(env) if make_callback is not None else None
    dynamics = build_dynamics(cfg, env.device)
    planner = Planner(dynamics, env, horizon, config=planner_cfg)
    x0_mean, x0_cov = build_initial_belief(cfg, env.device)
    result = planner.solve(x0_mean, x0_cov, step_callback=step_callback)

    if result_path:
        torch.save(result, result_path)
        log_utils.log_save(result_path)
    return result


def _clip_env_to_result(env, result):
    if env.moving_obstacles and "mean_trace" in result:
        env.clip_moving_obstacles(result["mean_trace"].shape[1])



def run_single_shot(max_iterations=1000, load_from=None, force_run=False):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config("configs/scenarios/single_shot.yaml")

    T = cfg["T"]

    env = build_environment(cfg, device)
    planner_cfg["max_iters"] = max_iterations

    log_utils._log.info("Starting single-shot optimisation...")
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=T, load_from=load_from, force_run=force_run
    )
    log_utils._log.info(f"Done. Final P(Sat): {result['best_p']:.4f}")

    visualize_results(
        result["mean_trace"],
        result["cov_trace"],
        result["u_trace"],
        env,
        result["loss_trace"],
    )

    anim = cfg["animation"]
    animate_results(
        result["mean_trace"],
        result["cov_trace"],
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
    planner_cfg = {**planner_cfg, "MAX_STEPS": cfg["MAX_STEPS"]}

    env = build_environment(cfg, device)

    log_utils._log.info(f"Starting MPC execution (horizon={H})...")
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=H, load_from=load_from, force_run=force_run,
        make_callback=make_mpc_live_callback,
    )

    visualize_results(
        result["mean_trace"],
        result["cov_trace"],
        result["u_trace"],
        env,
        history=result["loss_trace"],
        p_sat_trace=result["p_sat_trace"],
    )

    anim = cfg["animation"]
    animate_results(
        result["mean_trace"], result["cov_trace"], env,
        filename=anim["filename"], plan_traces=result["all_plans"],
        step=anim["step"], title=anim["title"], bounds=anim.get("bounds"),
    )


def _run_lane_change_scenario(cfg_path):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config(cfg_path)
    label = cfg.get("label", "")

    H = cfg["H"]
    dt = cfg["dt"]

    log_utils.log_scenario_start(label)

    env = build_environment(cfg, device)
    planner_cfg = {**planner_cfg, "T_SIM": cfg["T_SIM"], "mpc_mode": "lane_change"}
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=H, load_from=None, force_run=True,
        make_callback=make_lane_change_live_callback,
    )
    _clip_env_to_result(env, result)

    check_collision(
        result["mean_trace"], env,
        r_robot=planner_cfg["r_robot"],
        moving_obs_dist=planner_cfg["moving_obs_dist"],
    )

    visualize_lane_change(
        result["mean_trace"],
        result["cov_trace"],
        result["u_trace"],
        env,
        p_sat_trace=result["p_sat_trace"],
        dt=dt,
        robot_dims=env.robot_dims,
        xlim=env.plot_xlim,
    )

    anim = cfg["animation"]
    animate_results(
        result["mean_trace"], result["cov_trace"], env,
        filename=anim["filename"], plan_traces=result["all_plans"],
        step=anim["step"], robot_dims=env.robot_dims,
        title=anim["title"], bounds=anim.get("bounds"),
    )


def run_lane_change():
    _run_lane_change_scenario("configs/scenarios/lane_change.yaml")


def run_lane_change_aggressive():
    _run_lane_change_scenario("configs/scenarios/lane_change_aggressive.yaml")
