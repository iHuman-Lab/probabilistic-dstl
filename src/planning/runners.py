import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import blended_transform_factory
from planning.environment import Environment
from planning.dynamics import SingleIntegrator, DoubleIntegrator
from planning.planner import ProbabilisticSTLPlanner
from planning.visualization import visualize_results, visualize_lane_change, PALETTE, plot_covariance_ellipse
from planning.animation import animate_results

RESULTS_DIR = "saved_data"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def check_collision(mean_trace, env):
    """
    Checks for collisions between the ego vehicle trajectory and environment obstacles.
    """
    print("\n" + "=" * 30)
    print("SAFETY VERIFICATION")
    print("=" * 30)

    traj = mean_trace.squeeze()  # [T, 2]
    if traj.ndim == 1:
        traj = traj.unsqueeze(0)
    T = traj.shape[0]

    is_safe = True
    min_sep = float("inf")

    r_robot = 1.0

    for t in range(T):
        ego_pos = traj[t].cpu().numpy()

        # 1. Static Obstacles
        for obs in env.obstacles:
            x_min, x_max = obs["x"]
            y_min, y_max = obs["y"]
            if (x_min - r_robot <= ego_pos[0] <= x_max + r_robot) and (
                y_min - r_robot <= ego_pos[1] <= y_max + r_robot
            ):
                print(f"[COLLISION] Static Obstacle at Step {t}: Ego={ego_pos}")
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
                if dist < 2.25:
                    print(f"[COLLISION] Moving Obstacle at Step {t}: Dist={dist:.2f}")
                    is_safe = False

    if is_safe:
        print(f"Result: SAFE. (Min Separation from Moving Obs: {min_sep:.2f})")
    else:
        print("Result: UNSAFE (Collisions Detected)")
    print("=" * 30 + "\n")


def run_single_shot(max_iterations=1000, load_from=None, force_run=False):
    # Detect device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if load_from is None:
        load_from = os.path.join(RESULTS_DIR, "single_shot.pt")

    T = 100  # Time horizon
    dt = 0.2  # Time step size

    # Define workspace, goal, and obstacles
    env = Environment(device=device)

    # Scenario: Reach (3.5, 10.5)
    env.set_goal(x_range=[8.0, 10.0], y_range=[2.0, 4.0])
    env.set_bounds(x_range=[0.0, 12.0], y_range=[0.0, 10.5])
    

   # Add a visit region (waypoint)
    env.add_visit_region(x_range=[6.0, 8.0], y_range=[7.0, 9.0])

    # Add obstacles 
    # 1. Rectangle obstacle
    env.add_obstacle(x_range=[3.0, 6.0], y_range=[0.0, 3.0])
    env.add_obstacle(x_range=[3.0, 6.0], y_range=[4.0, 7.0])
    env.add_obstacle(x_range=[3.0, 6.0], y_range=[7.5, 10.0])


    # 2. Circle obstacle
    #env.add_circle_obstacle(center=[5.0, 5.0], radius=1.9)

    if not force_run and load_from and os.path.exists(load_from):
        print(f"Loading results from {load_from}...")
        data = torch.load(load_from, map_location=device)
        mean_trace = data["mean_trace"]
        cov_trace = data["cov_trace"]
        u_trace = data["u_trace"]
        history = data["history"]
        best_p = data.get("best_p", 0.0)
        print(f"Loaded data. Final Satisfaction Probability: {best_p:.4f}")
    else:
        # --- Setup Dynamics ---
        dynamics = SingleIntegrator(dt=dt, u_max=1.0, q_std=0.03, device=device)
        
        # --- Planner Config ---
        # Configuration for the gradient descent
        planner_cfg = {
            "w_u": 0.9,  # Reduced weight to allow higher speeds
            "w_u": 0.5,  # Reduced weight to allow higher speeds
            "w_du": 0.01,  # Weight on smoothness
            "w_phi": 100.0,  # Weight on STL satisfaction
            "lr": 0.05,  # Learning rate
            "max_iters": max_iterations,  # Max iterations
            "alpha": 0.95,  # Success threshold
            "w_dist": 50.0,  # Goal guidance heuristic weight
            "w_obs": 3.0,  # Obstacle repulsion heuristic weight
            "w_visit": 50.0,  # Visit region heuristic weight
        }
        planner = ProbabilisticSTLPlanner(dynamics, env, T, config=planner_cfg)

        # --- Initial Condition ---
        # Start at (0,0) with small uncertainty
        x0_mean = torch.tensor([0.0, 6.0], device=device)
        x0_cov = torch.eye(2, device=device) * 0.01

        # --- Initialization ---
        # We pass init_guess=None to let the optimizer figure out the path
        print("Initializing Probabilistic STL Motion Planning...")

        # Run optimization
        mean_trace, cov_trace, u_trace, best_p, history = planner.solve(
            x0_mean, x0_cov, render=True, init_guess=None
        )

        print("\nOptimization Complete.")
        print(f"Final Satisfaction Probability: {best_p:.4f}")

        if load_from:
            torch.save({"mean_trace": mean_trace, "cov_trace": cov_trace, "u_trace": u_trace, "history": history, "best_p": best_p}, load_from)
            print(f"Results saved to {load_from}")

    # --- Visualize ---
    visualize_results(mean_trace, cov_trace, u_trace, env, history, save_prefix="single_shot")

    animate_results(
        mean_trace, cov_trace, env, 
        filename="single_shot_animation.gif", step=2,
        title="Single Shot Planning",
        bounds=([-1, 13], [-1, 11])
    )


def run_mpc(load_from=None, force_run=False):
    # --- 1. Configuration ---
    # Detect device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MPC Parameters
    H = 100  # Sliding Window Horizon (Lookahead)
    MAX_STEPS = 300  # Safety limit, but we will use while loop
    dt = 0.3  # Time step size

    # Define workspace, goal, and obstacles
    env = Environment(device=device)

    # Scenario: Reach (3.5, 10.5)
    env.set_goal(x_range=[9.0, 10.0], y_range=[9.0, 10.0])
    env.set_bounds(x_range=[-2.0, 12.0], y_range=[-2.0, 12.0])

    env.add_obstacle(x_range=[0.0, 2.0], y_range=[3.0, 8.0])
    env.add_obstacle(x_range=[2.0, 6.0], y_range=[-1.0, 1.0])
    env.add_obstacle(x_range=[8.0, 10.0], y_range=[3.0, 8.0])

    env.add_circle_obstacle(center=[5.0, 5.0], radius=1.9)
    
    if load_from is None:
        load_from = os.path.join(RESULTS_DIR, "mpc.pt")

    if not force_run and os.path.exists(load_from):
        print(f"Loading MPC results from {load_from}...")
        data = torch.load(load_from, map_location=device)
        full_mean_trace = data["mean_trace"]
        full_cov_trace = data["cov_trace"]
        full_u_trace = data["u_trace"]
        loss_trace = data["loss_trace"]
        p_sat_trace = data["p_sat_trace"]
        all_plans = data["all_plans"]
    else:
        # --- Setup Dynamics ---
        dynamics = SingleIntegrator(dt=dt, u_max=1.0, q_std=0.02, device=device)

        # --- Planner Config ---
        # Configuration for the gradient descent
        planner_cfg = {
            "w_u": 0.9,  # Weight on control effort
            "w_du": 0.01,  # Weight on smoothness
            "w_phi": 100.0,  # Weight on STL satisfaction
            "lr": 0.05,  # Learning rate
            "max_iters": 200,  # Fewer iters needed for MPC warm start (or short horizon)
            "alpha": 0.95,  # Success threshold
            "w_dist": 4.0,  # Goal guidance heuristic weight
            "w_obs": 2.0,  # Obstacle repulsion heuristic weight
        }

        # --- Initial Condition ---
        # Start at (0,0) with small uncertainty
        x0_mean = torch.tensor([0.0, 0.0], device=device)
        x0_cov = torch.eye(2, device=device) * 0.01

        print(f"Starting MPC Execution (Horizon={H})...")

        real_mean_trace = [x0_mean]
        real_cov_trace = [x0_cov]
        real_u_trace = []

        curr_mean = x0_mean
        curr_cov = x0_cov

        # Goal center for distance check
        goal_center = torch.tensor([3.5, 10.5], device=device)

        # --- Live Visualization Setup ---
        plt.ion()
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        ax_map = fig.add_subplot(gs[0])
        ax_p = fig.add_subplot(gs[1])

        # Setup Map
        ax_map.set_xlim(-2, 12)
        ax_map.set_ylim(-2, 12)
        ax_map.set_aspect("equal")
        ax_map.grid(True, alpha=0.3)
        ax_map.set_title("MPC Live Execution")

        # Draw Static Environment
        if env.goal:
            gx, gy = env.goal["x"], env.goal["y"]
            ax_map.add_patch(
                patches.Rectangle(
                    (gx[0], gy[0]), gx[1] - gx[0], gy[1] - gy[0], facecolor=PALETTE["goal"]["fill"], edgecolor=PALETTE["goal"]["stroke"], alpha=0.3
                )
            )
        for obs in env.obstacles:
            ox, oy = obs["x"], obs["y"]
            ax_map.add_patch(
                patches.Rectangle(
                    (ox[0], oy[0]), ox[1] - ox[0], oy[1] - oy[0], facecolor=PALETTE["obs_static"]["fill"], edgecolor=PALETTE["obs_static"]["stroke"], alpha=0.5
                )
            )
        for obs in env.circle_obstacles:
            c = patches.Circle(obs["center"], obs["radius"], facecolor=PALETTE["obs_static"]["fill"], edgecolor=PALETTE["obs_static"]["stroke"], alpha=0.5)
            ax_map.add_patch(c)

        (line_exec,) = ax_map.plot([], [], color=PALETTE["ego"]["stroke"], marker="o", label="Executed Path")
        (line_plan,) = ax_map.plot(
            [], [], color=PALETTE["plan"]["stroke"], linestyle="--", alpha=0.8, label="Planned Window"
        )
        ax_map.legend(loc="upper left")

        # Setup P(Sat) Plot
        ax_p.set_xlim(0, 100)
        ax_p.set_ylim(0, 1.1)
        ax_p.set_title("Window Satisfaction Prob")
        ax_p.set_xlabel("Step")
        ax_p.set_ylabel("P(Sat)")
        ax_p.grid(True)
        (line_p,) = ax_p.plot([], [], color=PALETTE["goal"]["stroke"], marker="o", markersize=3)

        all_plans = []  # Store sliding windows for final animation
        p_sat_trace = []  # Store satisfaction probability
        loss_trace = []  # Store final loss of each step

        step = 0
        while step < MAX_STEPS:
            dist_to_goal = torch.norm(curr_mean - goal_center)
            if dist_to_goal < 0.5:
                print(f"Goal Reached at step {step}!")
                break

            # Setup Planner for Sliding Window
            mpc_planner = ProbabilisticSTLPlanner(dynamics, env, T=H, config=planner_cfg)

            # Solve Optimization
            p_mean, p_cov, p_u, p_val, history = mpc_planner.solve(
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
            noise = torch.distributions.MultivariateNormal(torch.zeros_like(pred_mean), dynamics.Q).sample()
            next_mean = pred_mean + noise

            # Store and Update
            real_mean_trace.append(next_mean)
            real_cov_trace.append(next_cov)
            real_u_trace.append(u_curr)

            curr_mean = next_mean
            curr_cov = next_cov

            print(
                f"Step {step:03d} | Pos: [{curr_mean[0]:.2f}, {curr_mean[1]:.2f}] | Goal Dist: {dist_to_goal:.2f} | Window P(Sat): {p_val:.4f}"
            )
            step += 1

        plt.ioff()
        plt.close(fig)

        # Stack results for visualization
        full_mean_trace = torch.stack(real_mean_trace).unsqueeze(0)  # [1, T, 2]
        full_cov_trace = torch.stack(real_cov_trace).unsqueeze(0)  # [1, T, 2, 2]
        full_u_trace = torch.stack(real_u_trace).unsqueeze(0)  # [1, T-1, 2]

        torch.save({
            "mean_trace": full_mean_trace,
            "cov_trace": full_cov_trace,
            "u_trace": full_u_trace,
            "loss_trace": loss_trace,
            "p_sat_trace": p_sat_trace,
            "all_plans": all_plans
        }, load_from)
        print(f"Results saved to {load_from}")

    # --- Visualize ---
    # Pass results to the visualization module
    visualize_results(
        full_mean_trace,
        full_cov_trace,
        full_u_trace,
        env,
        history=loss_trace,
        p_sat_trace=p_sat_trace,
        save_prefix="mpc",
    )

    # --- Animate ---
    # Step=5 to speed up animation significantly (300 frames -> 60 frames)
    animate_results(
        full_mean_trace,
        full_cov_trace,
        env,
        filename="mpc_animation.gif",
        plan_traces=all_plans,
        step=5, 
        title="MPC Receding Horizon",
        bounds=([-2, 12], [-2, 12])
    )


def run_lane_change(load_from=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    H = 35        # Planning horizon (lookahead steps)
    T_SIM = 150   # Total simulation steps
    dt = 0.2      # Time step [s]

    # Double Integrator: state [x, y, vx, vy], control [ax, ay]
    dynamics = DoubleIntegrator(dt=dt, u_max=1.5, q_std=0.001, device=device)

    # Planner Configuration
    planner_cfg = {
        "w_u":     0.05,
        "w_du":    1.5,    
        "w_phi":   100.0,
        "lr":      0.05,
        "max_iters": 200,
        "alpha":   0.90,
        "w_dist":  5.0,    # Reduced as goal center is further away
        "w_obs":   20.0,   # Increased penalty to ensure safety
        "w_visit": 0.0,
        "loss_tol": 1e-5,
    }

    print("\n=== Running Normal Scenario ===")
    
    # Setup Environment for this scenario
    env_global = Environment(device=device)
    env_global.add_lane_marking(x_range=[-5, 120], y_pos= 2.0, style="dashed")
    env_global.add_lane_marking(x_range=[-5, 120], y_pos=-2.0, style="solid")
    env_global.add_lane_marking(x_range=[-5, 120], y_pos= 6.0, style="solid")
    env_global.set_goal(x_range=[0.0, 200.0], y_range=[2.0, 6.0])

    # Moving obstacle: slower vehicle in Lane 2
    total_points = T_SIM + H + 10
    times = np.arange(total_points) * dt
    obs_x_global = 0.0 + 0.8 * times
    obs_y_global = np.ones_like(times) * 4.0

    env_global.add_moving_obstacle(
        obs_x_global[:T_SIM + 1],
        obs_y_global[:T_SIM + 1],
        width=2.5,
        height=1.5,
    )
    
    # Initial state: Lane 1 centre, forward velocity
    curr_mean = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    curr_cov  = torch.eye(4, device=device) * 0.01

    real_mean_trace = [curr_mean]
    real_cov_trace  = [curr_cov]
    real_u_trace    = []
    loss_trace      = []
    p_sat_trace     = []
    all_plans       = []
    prev_u_sol      = None
    success_counter = 0

    # --- Live Visualization ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 4))
    # No equal aspect — road is wide, lanes are narrow; let x scroll freely
    ax.grid(True, alpha=0.3, zorder=3)
    ax.set_title(f"Lane Change MPC (Normal) — Live Execution")
    ax.set_ylabel("$y$ [m]")
    ax.set_xlabel("$x$ [m]")

    # Road background and lane structure
    ax.axhspan(-2, 6, color=PALETTE["road"]["fill"], zorder=0)
    ax.axhspan(3.0, 5.0, color=PALETTE["goal"]["fill"], alpha=0.15, zorder=1)  # target band
    ax.axhline(-2, color=PALETTE["lane"]["stroke"], linewidth=2.0, linestyle="-",  alpha=0.8, zorder=2)
    ax.axhline( 6, color=PALETTE["lane"]["stroke"], linewidth=2.0, linestyle="-",  alpha=0.8, zorder=2)
    ax.axhline( 2, color=PALETTE["lane"]["stroke"], linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)
    # Lane labels: x pinned to axes (so they stay on-screen as camera scrolls),
    # y in data coordinates (so they sit at the correct lane centre).
    _blend = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.02,  0.0, "Lane 1", transform=_blend, color=PALETTE["lane"]["stroke"], fontsize=8, va="center", ha="left")
    ax.text(0.02,  4.0, "Lane 2", transform=_blend, color=PALETTE["lane"]["stroke"], fontsize=8, va="center", ha="left")

    # Dynamic actors
    (ego_dot,)   = ax.plot([], [], color=PALETTE["ego"]["stroke"],  marker="o", markersize=8, label="Ego",    zorder=10)
    (ego_trail,) = ax.plot([], [], color=PALETTE["ego"]["stroke"],  alpha=0.4, linewidth=1.5,                  zorder=9)
    (plan_line,) = ax.plot([], [], color=PALETTE["plan"]["stroke"], linestyle="--", alpha=0.8, linewidth=1.5,
                        label="Plan", zorder=8)
    ego_cov_patch = patches.Ellipse(
        (0, 0), width=0, height=0, angle=0,
        facecolor=PALETTE["ego"]["fill"], edgecolor=PALETTE["ego"]["stroke"],
        alpha=0.2, label="Uncertainty", zorder=7,
    )
    ax.add_patch(ego_cov_patch)
    # Obstacle — initialise at its true t=0 position
    obs_rect = patches.Rectangle(
        (obs_x_global[0] - 1.25, obs_y_global[0] - 0.75), 2.5, 1.5,
        facecolor=PALETTE["obs_moving"]["fill"], edgecolor=PALETTE["obs_moving"]["stroke"],
        alpha=0.8, label="Other Car", zorder=9,
    )
    ax.add_patch(obs_rect)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-3, 20)
    ax.set_ylim(-3, 7)

    # --- MPC Loop ---
    for t in range(T_SIM):
        ego_pos = curr_mean.cpu().numpy()
        curr_x  = ego_pos[0]

        # Local environment per window.
        env_local = Environment(device=device)

        # Goal: Fill the entire lane (Lane 2: y in [2, 6])
        # We set a long x-range ahead to encourage forward motion via the distance heuristic,
        
        goal_dist = 10.0
        goal_x_lo = curr_x + goal_dist
        goal_x_hi = curr_x + goal_dist + 60.0
        env_local.set_goal(x_range=[goal_x_lo, goal_x_hi], y_range=[2.1, 5.9])

        # Bounds: If we have successfully merged (y > 2.5), lock the bottom bound
        # to prevent drifting back to Lane 1.
        y_min_bound = -1.5
        if curr_mean[1] > 2.5:
            y_min_bound = 2.0
        env_local.set_bounds(x_range=[-100.0, 200.0], y_range=[y_min_bound, 6.0])

        # Slice moving obstacle trajectory for this window
        idx_end = t + H + 1
        if idx_end <= len(obs_x_global):
            sl_x = obs_x_global[t:idx_end]
            sl_y = obs_y_global[t:idx_end]
        else:
            pad  = idx_end - len(obs_x_global)
            sl_x = np.concatenate([obs_x_global[t:], np.full(pad, obs_x_global[-1])])
            sl_y = np.concatenate([obs_y_global[t:], np.full(pad, obs_y_global[-1])])
        env_local.add_moving_obstacle(sl_x, sl_y, width=2.5, height=1.5)

        # Warm start: shift previous solution by one step
        init_guess = None
        if prev_u_sol is not None:
            init_guess = torch.cat([prev_u_sol[1:], prev_u_sol[-1:]], dim=0)

        planner = ProbabilisticSTLPlanner(dynamics, env_local, T=H, config=planner_cfg)
        p_mean, p_cov, p_u, p_val, history = planner.solve(
            curr_mean, curr_cov, render=False, verbose=False, init_guess=init_guess
        )

        prev_u_sol = p_u.detach()
        all_plans.append(p_mean)
        loss_trace.append(history[-1] if history else 0.0)
        p_sat_trace.append(p_val)

        # Execute first action (receding horizon)
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
        curr_cov  = next_cov

        obs_pos = np.array([obs_x_global[t], obs_y_global[t]])
        ego_pos = curr_mean.cpu().numpy()
        dist    = np.linalg.norm(ego_pos[:2] - obs_pos)

        if t % 5 == 0:
            print(
                f"Step {t:03d} | Ego: [{ego_pos[0]:.2f}, {ego_pos[1]:.2f}]"
                f" vx={ego_pos[2]:.2f} vy={ego_pos[3]:.2f} | "
                f"Obs x={obs_pos[0]:.2f} | Dist: {dist:.2f} | P(φ)={p_val:.3f}"
            )

        # Update live plot — camera scrolls with ego, keeping obs in view
        ego_x, ego_y = ego_pos[0], ego_pos[1]
        # view_center = (ego_x + obs_pos[0]) / 2.0
        # ax.set_xlim(view_center - 14, view_center + 14)
        ego_dot.set_data([ego_x], [ego_y])
        ego_trail.set_data(
            [m[0].item() for m in real_mean_trace],
            [m[1].item() for m in real_mean_trace],
        )

        pos_cov_np = curr_cov[:2, :2].cpu().numpy()
        vals, vecs = np.linalg.eigh(pos_cov_np)
        order = vals.argsort()[::-1]
        theta = np.degrees(np.arctan2(*vecs[:, order][:, 0][::-1]))
        w_e, h_e = 2 * 2.45 * np.sqrt(vals[order])
        ego_cov_patch.set_center((ego_x, ego_y))
        ego_cov_patch.set_width(w_e)
        ego_cov_patch.set_height(h_e)
        ego_cov_patch.set_angle(theta)

        plan_np = p_mean.detach().cpu().squeeze().numpy()
        plan_line.set_data(plan_np[:, 0], plan_np[:, 1])
        obs_rect.set_xy((obs_pos[0] - 1.25, obs_pos[1] - 0.75))

        plt.draw()
        plt.pause(0.001)

        # Success: stably inside Lane 2 for 15 consecutive steps
        if 3.0 <= ego_pos[1] <= 5.5:
            success_counter += 1
        else:
            success_counter = 0

        if success_counter >= 15:
            print(f"Lane change completed successfully at step {t}!")
            break

    plt.ioff()
    plt.close(fig)

    # Truncate obstacle trajectory to actual simulation length
    actual_steps = len(real_mean_trace)
    for obs in env_global.moving_obstacles:
        obs["x_traj"] = obs["x_traj"][:actual_steps]
        obs["y_traj"] = obs["y_traj"][:actual_steps]

    full_mean_trace = torch.stack(real_mean_trace).unsqueeze(0)
    full_cov_trace  = torch.stack(real_cov_trace).unsqueeze(0)
    full_u_trace    = torch.stack(real_u_trace).unsqueeze(0)
    
    check_collision(full_mean_trace, env_global)

    print(f"Generating visualization for Normal...")
    visualize_lane_change(
        full_mean_trace,
        full_cov_trace,
        full_u_trace,
        env_global,
        p_sat_trace=p_sat_trace,
        dt=dt,
        robot_dims=(2.0, 1.0),
        save_prefix=f"lane_change_normal",
        comparison_data=None,
        xlim=[-3, 20]
    )

    animate_results(
        full_mean_trace,
        full_cov_trace,
        env_global,
        filename=f"lane_change_normal.gif",
        plan_traces=all_plans,
        step=4,
        robot_dims=(2.0, 1.0),
        title=f"Lane Change: Normal",
        bounds=None,
    )


def run_lane_change_aggressive(load_from=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    H = 35        # Planning horizon (lookahead steps)
    T_SIM = 150   # Total simulation steps
    dt = 0.2      # Time step [s]

    # Double Integrator: state [x, y, vx, vy], control [ax, ay]
    dynamics = DoubleIntegrator(dt=dt, u_max=1.5, q_std=0.001, device=device)

    planner_cfg = {
        "w_u":     0.1,
        "w_du":    1.5,    # High smoothness to prevent oscillation
        "w_phi":   500.0,
        "lr":      0.05,
        "max_iters": 200,
        "alpha":   0.95,
        "w_dist":  10.0,   # Increased to encourage merging
        "w_obs":   100.0,   # Increased penalty to ensure safety and force merge
        "w_visit": 0.0,
        "loss_tol": 1e-5,
    }

    print("\n=== Running Aggressive Scenario ===")
    
    # Setup Environment for this scenario
    env_global = Environment(device=device)
    env_global.add_lane_marking(x_range=[-5, 120], y_pos= 2.0, style="dashed")
    env_global.add_lane_marking(x_range=[-5, 120], y_pos=-2.0, style="solid")
    env_global.add_lane_marking(x_range=[-5, 120], y_pos= 6.0, style="solid")
    env_global.set_goal(x_range=[0.0, 200.0], y_range=[2.0, 6.0])

    # Moving obstacle: slower vehicle in Lane 2
    total_points = T_SIM + H + 10
    times = np.arange(total_points) * dt
    obs_x_global = 0.0 + 0.8 * times
    obs_y_global = np.ones_like(times) * 4.0

    env_global.add_moving_obstacle(
        obs_x_global[:T_SIM + 1],
        obs_y_global[:T_SIM + 1],
        width=2.5,
        height=1.5,
    )
    
    # Aggressive: Add static block in Lane 1 to force merge
    bx = 15.0
    # Add a long static obstacle to block the lane from bx onwards.
    env_global.add_obstacle(x_range=[bx, 120.0], y_range=[-2.0, 2.0])

    # Initial state: Lane 1 centre, forward velocity
    curr_mean = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    curr_cov  = torch.eye(4, device=device) * 0.01

    real_mean_trace = [curr_mean]
    real_cov_trace  = [curr_cov]
    real_u_trace    = []
    loss_trace      = []
    p_sat_trace     = []
    all_plans       = []
    prev_u_sol      = None

    # --- MPC Loop ---
    for t in range(T_SIM):
        ego_pos = curr_mean.cpu().numpy()
        curr_x  = ego_pos[0]

        # Local environment per window.
        env_local = Environment(device=device)

        # Goal: Fill the entire lane (Lane 2: y in [2, 6])
        # We set a long x-range ahead to encourage forward motion via the distance heuristic,
        
        goal_dist = 10.0
        goal_x_lo = curr_x + goal_dist
        goal_x_hi = curr_x + goal_dist + 60.0
        env_local.set_goal(x_range=[goal_x_lo, goal_x_hi], y_range=[2.1, 5.9])

        # Bounds: If we have successfully merged (y > 2.5), lock the bottom bound
        # to prevent drifting back to Lane 1.
        y_min_bound = -1.5
        if curr_mean[1] > 2.5:
            y_min_bound = 2.0
        env_local.set_bounds(x_range=[-100.0, 200.0], y_range=[y_min_bound, 6.0])

        # Slice moving obstacle trajectory for this window
        idx_end = t + H + 1
        if idx_end <= len(obs_x_global):
            sl_x = obs_x_global[t:idx_end]
            sl_y = obs_y_global[t:idx_end]
        else:
            pad  = idx_end - len(obs_x_global)
            sl_x = np.concatenate([obs_x_global[t:], np.full(pad, obs_x_global[-1])])
            sl_y = np.concatenate([obs_y_global[t:], np.full(pad, obs_y_global[-1])])
        env_local.add_moving_obstacle(sl_x, sl_y, width=2.5, height=1.5)

        # Warm start: shift previous solution by one step
        init_guess = None
        if prev_u_sol is not None:
            init_guess = torch.cat([prev_u_sol[1:], prev_u_sol[-1:]], dim=0)

        planner = ProbabilisticSTLPlanner(dynamics, env_local, T=H, config=planner_cfg)
        p_mean, p_cov, p_u, p_val, history = planner.solve(
            curr_mean, curr_cov, render=False, verbose=False, init_guess=init_guess
        )

        prev_u_sol = p_u.detach()
        all_plans.append(p_mean)
        loss_trace.append(history[-1] if history else 0.0)
        p_sat_trace.append(p_val)

        # Execute first action (receding horizon)
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
        curr_cov  = next_cov

        obs_pos = np.array([obs_x_global[t], obs_y_global[t]])
        ego_pos = curr_mean.cpu().numpy()
        dist    = np.linalg.norm(ego_pos[:2] - obs_pos)

        if t % 5 == 0:
            print(
                f"Step {t:03d} | Ego: [{ego_pos[0]:.2f}, {ego_pos[1]:.2f}]"
                f" vx={ego_pos[2]:.2f} vy={ego_pos[3]:.2f} | "
                f"Obs x={obs_pos[0]:.2f} | Dist: {dist:.2f} | P(φ)={p_val:.3f}"
            )

        # Termination: Stop if we reach x=20.0 (as requested)
        if curr_x >= 20.0:
            print(f"Aggressive scenario reached x=20.0 cutoff. Stopping.")
            break

    # Truncate obstacle trajectory to actual simulation length
    actual_steps = len(real_mean_trace)
    for obs in env_global.moving_obstacles:
        obs["x_traj"] = obs["x_traj"][:actual_steps]
        obs["y_traj"] = obs["y_traj"][:actual_steps]

    full_mean_trace = torch.stack(real_mean_trace).unsqueeze(0)
    full_cov_trace  = torch.stack(real_cov_trace).unsqueeze(0)
    full_u_trace    = torch.stack(real_u_trace).unsqueeze(0)

    torch.save({
        "mean_trace": full_mean_trace,
        "cov_trace": full_cov_trace,
        "u_trace": full_u_trace,
        "env": env_global
    }, os.path.join(RESULTS_DIR, "lane_change_aggressive.pt"))
    
    check_collision(full_mean_trace, env_global)

    print(f"Generating visualization for Aggressive...")
    visualize_lane_change(
        full_mean_trace,
        full_cov_trace,
        full_u_trace,
        env_global,
        p_sat_trace=p_sat_trace,
        dt=dt,
        robot_dims=(2.0, 1.0),
        save_prefix=f"lane_change_aggressive",
        comparison_data=None,
        xlim=[-3, 20]
    )

    animate_results(
        full_mean_trace,
        full_cov_trace,
        env_global,
        filename=f"lane_change_aggressive.gif",
        plan_traces=all_plans,
        step=4,
        robot_dims=(2.0, 1.0),
        title=f"Lane Change: Aggressive",
        bounds=None,
    )

    # --- Comparison Plot ---
    normal_res_path = os.path.join(RESULTS_DIR, "lane_change_normal.pt")
    if os.path.exists(normal_res_path):
        print("Generating comparison plot...")
        normal_data = torch.load(normal_res_path, map_location=device)
        norm_mean = normal_data["mean_trace"].cpu().squeeze().numpy()
        aggr_mean = full_mean_trace.cpu().squeeze().numpy()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title("Lane Change Comparison: Normal vs Aggressive")
        
        # Environment
        ax.axhspan(-2, 6, color=PALETTE["road"]["fill"], zorder=0)
        ax.axhline(-2, color=PALETTE["lane"]["stroke"], linewidth=2)
        ax.axhline(6, color=PALETTE["lane"]["stroke"], linewidth=2)
        ax.axhline(2, color=PALETTE["lane"]["stroke"], linestyle="--", alpha=0.5)
        
        # Traces
        ax.plot(norm_mean[:, 0], norm_mean[:, 1], color="blue", linestyle="--", label="Normal", linewidth=2)
        ax.plot(aggr_mean[:, 0], aggr_mean[:, 1], color="red", label="Aggressive", linewidth=2)
        
        # Blockage visual
        rect = patches.Rectangle((15.0, -2.0), 105.0, 4.0, facecolor="red", alpha=0.15, hatch="//", label="Blockage (Aggressive)")
        ax.add_patch(rect)
        
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.set_xlim(-3, 20)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=12, framealpha=0.95)
        fig.subplots_adjust(bottom=0.2)
        plt.savefig("lane_change_comparison.pdf")
        print("Saved comparison plot to lane_change_comparison.pdf")
        plt.close(fig)


def _check_trace_success(trace, env):
    """
    Returns True if the single trace satisfies all constraints (Goal, Visit, Obstacles).
    trace: [T+1, 2] numpy array
    """
    # 1. Check Obstacles (Safety) - Must be safe at ALL times
    for t in range(len(trace)):
        pos = trace[t]
        # Rectangles
        for obs in env.obstacles:
            if (obs["x"][0] <= pos[0] <= obs["x"][1]) and (obs["y"][0] <= pos[1] <= obs["y"][1]):
                return False # Collision
        # Circles
        for obs in env.circle_obstacles:
            if np.linalg.norm(pos - obs["center"]) <= obs["radius"]:
                return False

    # 2. Check Visit Region (Liveness) - Must visit at LEAST once
    visited = False
    if not env.visit_regions:
        visited = True
    else:
        for t in range(len(trace)):
            pos = trace[t]
            for reg in env.visit_regions:
                if (reg["x"][0] <= pos[0] <= reg["x"][1]) and (reg["y"][0] <= pos[1] <= reg["y"][1]):
                    visited = True
                    break
            if visited: break
    if not visited: return False

    # 3. Check Goal (Liveness) - Must be in goal at LAST step (or eventually, depending on spec)
    # For this scenario, we usually require being in goal at the end.
    if env.goal:
        final_pos = trace[-1]
        if not ((env.goal["x"][0] <= final_pos[0] <= env.goal["x"][1]) and 
                (env.goal["y"][0] <= final_pos[1] <= env.goal["y"][1])):
            return False

    return True


def run_paper_comparison(load_from=None, force_run=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    T = 80
    T = 100
    dt = 0.2
    MC_SAMPLES = 100  # Number of Monte Carlo simulations

    # --- Setup Environment (Same as Single Shot) ---
    env = Environment(device=device)
    env.set_goal(x_range=[9.0, 11.0], y_range=[2.0, 4.0])
    env.set_bounds(x_range=[0.0, 12.0], y_range=[0.0, 10.5])
    env.add_visit_region(x_range=[7.0, 9.0], y_range=[7.0, 9.0])
    
    # Obstacles
    env.add_obstacle(x_range=[3.0, 6.0], y_range=[0.0, 3.0])
    env.add_obstacle(x_range=[3.0, 6.0], y_range=[4.0, 7.0])
    env.add_obstacle(x_range=[3.0, 6.0], y_range=[7.5, 10.0])

    x0_mean = torch.tensor([0.0, 5.0], device=device)

    if load_from is None:
        load_from = os.path.join(RESULTS_DIR, "paper_comparison.pt")

    if not force_run and os.path.exists(load_from):
        print(f"Loading Paper Comparison results from {load_from}...")
        data = torch.load(load_from, map_location=device)
        mu_det = data["mu_det"]
        u_det = data["u_det"]
        mu_prob = data["mu_prob"]
        cov_prob = data["cov_prob"]
        u_prob = data["u_prob"]
    else:
        print("Running Deterministic Baseline...")
        dyn_det = SingleIntegrator(dt=dt, u_max=1.0, q_std=0.0, device=device)
        # --- 1. Deterministic Baseline (Naive) ---
        # We use a very small q_std to mimic a deterministic planner that assumes perfect execution.
        print("\n--- Running Deterministic Baseline (Naive) ---")
        dyn_det = SingleIntegrator(dt=dt, u_max=1.0, q_std=0.001, device=device)
        planner_det = ProbabilisticSTLPlanner(dyn_det, env, T, config={
            "w_u": 0.1, "w_du": 0.1, "w_phi": 1000.0, "lr": 0.02, "max_iters": 500,
            "w_dist": 5.0, "w_obs": 1.0, "alpha": 0.99,
            "w_u": 0.5, "w_du": 0.01, "w_phi": 100.0, "lr": 0.05, "max_iters": 1000,
            "w_dist": 50.0, "w_obs": 3.0, "w_visit": 50.0, "alpha": 0.99,
        })
        x0_cov_det = torch.eye(2, device=device) * 1e-6
        mu_det, _, u_det, _, _ = planner_det.solve(x0_mean, x0_cov_det, render=False, verbose=True)
        x0_cov_det = torch.eye(2, device=device) * 0.0001
        mu_det, _, u_det, _, _ = planner_det.solve(x0_mean, x0_cov_det, render=False, verbose=True, init_guess=None)

        print("Running Probabilistic Framework...")
        dyn_prob = SingleIntegrator(dt=dt, u_max=1.0, q_std=0.08, device=device)
        # --- 2. Probabilistic Planner (Ours) ---
        # We use the TRUE noise (q_std=0.05) in the planning model.
        print("\n--- Running Probabilistic Planner (Ours) ---")
        dyn_prob = SingleIntegrator(dt=dt, u_max=1.0, q_std=0.05, device=device)
        planner_prob = ProbabilisticSTLPlanner(dyn_prob, env, T, config={
            "w_u": 0.1, "w_du": 0.1, "w_phi": 1000.0, "lr": 0.02, "max_iters": 500,
            "w_dist": 5.0, "w_obs": 1.0, "alpha": 0.95,
            "w_u": 0.5, "w_du": 0.01, "w_phi": 100.0, "lr": 0.05, "max_iters": 1000,
            "w_dist": 50.0, "w_obs": 3.0, "w_visit": 50.0, "alpha": 0.95,
        })
        x0_cov_prob = torch.eye(2, device=device) * 0.01
        mu_prob, cov_prob, u_prob, _, _ = planner_prob.solve(x0_mean, x0_cov_prob, render=False, verbose=True)
        mu_prob, cov_prob, u_prob, _, _ = planner_prob.solve(x0_mean, x0_cov_prob, render=False, verbose=True, init_guess=None)

        torch.save({
            "mu_det": mu_det,
            "u_det": u_det,
            "mu_prob": mu_prob,
            "cov_prob": cov_prob,
            "u_prob": u_prob
        }, load_from)
        print(f"Results saved to {load_from}")

    # --- 3. Monte Carlo Simulation ---
    print(f"\n--- Running Monte Carlo Validation (N={MC_SAMPLES}) ---")
    # True Dynamics
    true_q_std = 0.05
    
    def run_mc(u_seq, name):
        success_count = 0
        traces = []
        for i in range(MC_SAMPLES):
            # Rollout with noise
            # We can use the dynamics forward pass but we need to inject noise manually if forward doesn't do it per step
            # Actually SingleIntegrator.forward computes the distribution. We need sample paths.
            # Let's do a manual loop.
            curr_x = x0_mean.clone()
            trace = [curr_x.cpu().numpy()]
            
            for t in range(T):
                u = u_seq[t]
                # Step with noise
                # x_next = x + u*dt + noise
                noise = torch.randn(2, device=device) * true_q_std
                curr_x = curr_x + u * dt + noise
                trace.append(curr_x.cpu().numpy())
            
            trace_np = np.array(trace)
            traces.append(trace_np)
            if _check_trace_success(trace_np, env):
                success_count += 1
        
        print(f"{name} Success Rate: {success_count}/{MC_SAMPLES} ({success_count/MC_SAMPLES*100:.1f}%)")
        return traces, success_count

    det_traces, det_success = run_mc(u_det, "Deterministic Plan")
    prob_traces, prob_success = run_mc(u_prob, "Probabilistic Plan")

    mu_d  = mu_det.detach().cpu().squeeze().numpy()
    mu_p  = mu_prob.detach().cpu().squeeze().numpy()
    cov_p = cov_prob.detach().cpu().squeeze().numpy()
    u_d   = u_det.detach().cpu().squeeze().numpy()
    u_p   = u_prob.detach().cpu().squeeze().numpy()

    # --- Trajectory Comparison ---
    fig1, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=True)
    
    # Common plotting helper
    def plot_env_and_traces(ax, traces, nominal, title, color_nominal, x0):
        ax.set_xlim(-1, 13)
        ax.set_ylim(-1, 11)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Env
        for obs in env.obstacles:
            ax.add_patch(patches.Rectangle((obs["x"][0], obs["y"][0]), obs["x"][1]-obs["x"][0], obs["y"][1]-obs["y"][0], facecolor=PALETTE["obs_static"]["fill"], edgecolor=PALETTE["obs_static"]["stroke"], alpha=0.6, hatch="//"))
        if env.goal:
            gx, gy = env.goal["x"], env.goal["y"]
            ax.add_patch(patches.Rectangle((gx[0], gy[0]), gx[1]-gx[0], gy[1]-gy[0], facecolor=PALETTE["goal"]["fill"], edgecolor=PALETTE["goal"]["stroke"], alpha=0.4))
            ax.text((gx[0] + gx[1]) / 2, (gy[0] + gy[1]) / 2, "G", fontsize=24, fontweight='bold', ha='center', va='center', color=PALETTE["goal"]["stroke"], zorder=30)
        for region in env.visit_regions:
            vx, vy = region["x"], region["y"]
            ax.add_patch(patches.Rectangle((vx[0], vy[0]), vx[1]-vx[0], vy[1]-vy[0], facecolor=PALETTE["visit"]["fill"], edgecolor=PALETTE["visit"]["stroke"], alpha=0.4))
            ax.text((vx[0] + vx[1]) / 2, (vy[0] + vy[1]) / 2, "V", fontsize=24, fontweight='bold', ha='center', va='center', color=PALETTE["visit"]["stroke"], zorder=30)

        # Add 'S' for start
        start_pos = x0.cpu().numpy()
        ax.text(start_pos[0] - 0.5, start_pos[1], "S", fontsize=24, fontweight='bold', ha='center', va='center', color='black', zorder=30)

        # MC Traces
        for tr in traces:
            # Check if this specific trace failed? 
            # For visual clarity, just plot them all faintly
            ax.plot(tr[:, 0], tr[:, 1], color="gray", alpha=0.15, linewidth=0.8)
            
        # Nominal
        ax.plot(nominal[:, 0], nominal[:, 1], color=color_nominal, linewidth=3.5, label="Planned Path")

    # Plot Deterministic
    plot_env_and_traces(axes[0], det_traces, mu_d, f"Deterministic Baseline\nSuccess: {det_success}%", PALETTE["lane"]["stroke"], x0_mean)
    axes[0].set_xlabel("$x$ [m]", fontsize=18, fontweight='bold')
    axes[0].set_ylabel("$y$ [m]", fontsize=18, fontweight='bold')
    
    # Plot Probabilistic
    plot_env_and_traces(axes[1], prob_traces, mu_p, f"Probabilistic (Ours)\nSuccess: {prob_success}%", PALETTE["ego"]["stroke"], x0_mean)
    axes[1].set_xlabel("$x$ [m]", fontsize=18, fontweight='bold')
    # Add uncertainty ellipses to probabilistic plot
    for t in range(0, T + 1, 5):
        plot_covariance_ellipse(axes[1], mu_p[t], cov_p[t], facecolor=PALETTE["ego"]["fill"], edgecolor=PALETTE["ego"]["stroke"], alpha=0.35)

    plt.tight_layout()
    plt.savefig("paper_comparison_mc.pdf", bbox_inches="tight")
    plt.close(fig1)
    print("Saved comparison plot to paper_comparison_mc.pdf")

    # --- Control Input Comparison ---
    t_sec = np.arange(T) * dt
    fig2, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t_sec, u_d[:, 0], color=PALETTE["lane"]["stroke"], linestyle="--", linewidth=1.8, label="Deterministic")
    axes[0].plot(t_sec, u_p[:, 0], color=PALETTE["ego"]["stroke"], linewidth=1.8, label="Probabilistic (Ours)")
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("$u_x$ [m/s]", fontsize=13)
    axes[0].tick_params(labelsize=11)
    axes[0].grid(True, alpha=0.35)
    axes[1].plot(t_sec, u_d[:, 1], color=PALETTE["lane"]["stroke"], linestyle="--", linewidth=1.8)
    axes[1].plot(t_sec, u_p[:, 1], color=PALETTE["ego"]["stroke"], linewidth=1.8)
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[1].set_ylabel("$u_y$ [m/s]", fontsize=13)
    axes[1].set_xlabel("Time [s]", fontsize=13)
    axes[1].tick_params(labelsize=11)
    axes[1].grid(True, alpha=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=11, framealpha=0.95, edgecolor="#cccccc")
    fig2.subplots_adjust(bottom=0.18, hspace=0.08)
    plt.savefig("paper_comparison_ctrl.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig2)
