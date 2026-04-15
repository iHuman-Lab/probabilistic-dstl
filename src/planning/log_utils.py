"""Formatted log helpers for the planning runners.

All functions write to the "planning" logger at INFO level.
Configure handlers in the application entry point (e.g. main.py) as needed.
"""
import logging

_log = logging.getLogger("planning")


def log_device(device):
    _log.info(f"Using device: {device}")


def log_scenario_start(label):
    _log.info(f"=== Running {label} Scenario ===")


def log_load(path):
    _log.info(f"Loading results from {path}...")


def log_save(path):
    _log.info(f"Results saved to {path}")


def log_mpc_step(step, pos, dist_to_goal, p_val):
    _log.info(
        f"Step {step:03d} | Pos: [{pos[0]:.2f}, {pos[1]:.2f}] "
        f"| Dist: {dist_to_goal:.2f} | P(Sat): {p_val:.4f}"
    )


def log_lane_step(t, ego_pos, obs_x, dist, p_val):
    _log.info(
        f"Step {t:03d} | Ego: [{ego_pos[0]:.2f}, {ego_pos[1]:.2f}] "
        f"vx={ego_pos[2]:.2f} vy={ego_pos[3]:.2f} "
        f"| Obs x={obs_x:.2f} | Dist: {dist:.2f} | P(φ)={p_val:.3f}"
    )


def log_goal_reached(step):
    _log.info(f"Goal reached at step {step}.")


def log_lane_change_done(label, step):
    _log.info(f"Lane change ({label}) completed at step {step}.")


def log_safety(is_safe, min_sep):
    if is_safe:
        if min_sep < float("inf"):
            _log.info(f"Safety check: SAFE  (min separation {min_sep:.2f} m)")
        else:
            _log.info("Safety check: SAFE")
    else:
        _log.info("Safety check: UNSAFE — collisions detected")
