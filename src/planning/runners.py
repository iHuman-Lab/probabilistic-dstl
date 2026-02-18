import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from planning.environment import Environment
from planning.dynamics import SingleIntegrator, DoubleIntegrator
from planning.planner import ProbabilisticSTLPlanner
from planning.visualization import visualize_results, PALETTE
from planning.animation import animate_results


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


def run_single_shot(max_iterations=1000):
    # Detect device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    T = 120  # Time horizon
    dt = 0.2  # Time step size

    # Define workspace, goal, and obstacles
    env = Environment(device=device)

    # Scenario: Reach (3.5, 10.5)
    env.set_goal(x_range=[4.0, 6.0], y_range=[9.0, 11.0])

    # Add a visit region (waypoint)
    env.add_visit_region(x_range=[8.0, 10.0], y_range=[3.0, 8.0])

    # Add obstacles 
    # 1. Rectangle obstacle
    env.add_obstacle(x_range=[0.0, 2.0], y_range=[3.0, 8.0])
    env.add_obstacle(x_range=[2.0, 6.0], y_range=[-1.0, 1.0])

    # 2. Circle obstacle
    env.add_circle_obstacle(center=[5.0, 5.0], radius=2.5)

    # --- Setup Dynamics ---
    dynamics = SingleIntegrator(dt=dt, u_max=1.0, q_std=0.02, device=device)
      
    # --- Planner Config ---
    # Configuration for the gradient descent
    planner_cfg = {
        "w_u": 0.9,  # Weight on control effort
        "w_du": 0.05,  # Weight on smoothness
        "w_phi": 100.0,  # Weight on STL satisfaction
        "lr": 0.05,  # Learning rate
        "max_iters": max_iterations,  # Max iterations
        "alpha": 0.95,  # Success threshold
        "w_dist": 4.0,  # Goal guidance heuristic weight
        "w_obs": 2.0,  # Obstacle repulsion heuristic weight
        "w_visit": 4.0,  # Visit region heuristic weight
    }
    planner = ProbabilisticSTLPlanner(dynamics, env, T, config=planner_cfg)

    # --- Initial Condition ---
    # Start at (0,0) with small uncertainty
    x0_mean = torch.tensor([0.0, 0.0], device=device)
    x0_cov = torch.eye(2, device=device) * 0.01

    print("Initializing Probabilistic STL Motion Planning...")

    # Run optimization
    mean_trace, cov_trace, u_trace, best_p, history = planner.solve(
        x0_mean, x0_cov, render=True
    )

    print("\nOptimization Complete.")
    print(f"Final Satisfaction Probability: {best_p:.4f}")

    # --- Visualize ---
    visualize_results(mean_trace, cov_trace, u_trace, env, history)

    animate_results(
        mean_trace, cov_trace, env, filename="single_shot_animation.gif", step=2
    )


def run_mpc():
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
    

    env.add_obstacle(x_range=[0.0, 2.0], y_range=[3.0, 8.0])
    env.add_obstacle(x_range=[2.0, 6.0], y_range=[-1.0, 1.0])
    env.add_obstacle(x_range=[8.0, 10.0], y_range=[3.0, 8.0])

    env.add_circle_obstacle(center=[5.0, 5.0], radius=2.0)
    
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

    # --- Visualize ---
    # Pass results to the visualization module
    visualize_results(
        full_mean_trace,
        full_cov_trace,
        full_u_trace,
        env,
        history=loss_trace,
        p_sat_trace=p_sat_trace,
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
    )


def run_lane_change():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    H = 35        # Planning Horizon (Lookahead)
    T_SIM = 100   # Total Simulation Steps
    dt = 0.2      # Time step
    
    # We define the "True" world here
    env_global = Environment(device=device)
    
    # Road Setup
    # Lane 1: y=[-2, 2], Lane 2: y=[2, 6]
    env_global.add_lane_marking(x_range=[-5, 30], y_pos=2.0, style="dashed") # Divider
    env_global.add_lane_marking(x_range=[-5, 30], y_pos=-2.0, style="solid") # Bottom
    env_global.add_lane_marking(x_range=[-5, 30], y_pos=6.0, style="solid")  # Top

    # Goal: Entirety of Lane 2 (y=4).
    # We set a long x-range to represent the "lane" concept and provide a heuristic pull forward.
    env_global.set_goal(x_range=[0.0, 60.0], y_range=[3.0, 5.0])

    # Moving Obstacle Trajectory (Global Truth)
    # Car in Lane 2 moving at constant velocity.
    # Starts at x=0, y=4. Velocity = 0.8 m/s.
    # This creates a moving block in the target lane that the ego must negotiate.
    total_points = T_SIM + H + 10
    times = np.arange(total_points) * dt
    
    # Obstacle starts at x=0.0, moves forward
    obs_x_global = 0.0 + 0.8 * times
    obs_y_global = np.ones_like(times) * 4.0
    
    # Add to global env for final render
    env_global.add_moving_obstacle(
        obs_x_global[:T_SIM+1], 
        obs_y_global[:T_SIM+1], 
        width=2.5, 
        height=1.5
    )

    # --- Dynamics ---
    # Double Integrator: State [x, y, vx, vy], Control [ax, ay]
    dynamics = DoubleIntegrator(dt=dt, u_max=1.5, q_std=0.02, device=device)

    # --- Planner Config ---
    planner_cfg = {
        "w_u": 0.05,      # Low cost on velocity to encourage movement
        "w_du": 0.1,
        "w_phi": 100.0,   # Safety (High importance)
        "lr": 0.05,
        "max_iters": 100, # Sufficient iters
        "alpha": 0.90,
        "w_dist": 20.0,   # Pull to goal center
        "w_obs": 100.0,   # Strong repulsion from obstacles
    }

    # --- Simulation Loop ---
    print("Starting Lane Change MPC Simulation...")
    
    # Initial State
    curr_mean = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device) # [x, y, vx, vy], Start in Lane 1 with velocity
    curr_cov = torch.eye(4, device=device) * 0.01
    
    real_mean_trace = [curr_mean]
    real_cov_trace = [curr_cov]
    real_u_trace = [] # Track controls for visualization
    loss_trace = []   # Track optimization loss
    p_sat_trace = []  # Track satisfaction probability
    all_plans = [] # For animation

    # Warm Start Variable
    prev_u_sol = None

    # --- Live Visualization Setup ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-2, 28)
    ax.set_ylim(-4, 8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Live MPC Execution")

    # Draw Static Env
    for lane in env_global.lane_markings:
        ax.plot(lane["x"], [lane["y"], lane["y"]], color=PALETTE["lane"]["stroke"], linestyle="--", alpha=0.7)
    for obs in env_global.obstacles:
        ax.add_patch(patches.Rectangle((obs["x"][0], obs["y"][0]), obs["x"][1]-obs["x"][0], obs["y"][1]-obs["y"][0], facecolor=PALETTE["obs_static"]["fill"], edgecolor=PALETTE["obs_static"]["stroke"], alpha=0.6))
    gx, gy = env_global.goal["x"], env_global.goal["y"]
    ax.add_patch(patches.Rectangle((gx[0], gy[0]), gx[1]-gx[0], gy[1]-gy[0], facecolor=PALETTE["goal"]["fill"], edgecolor=PALETTE["goal"]["stroke"], alpha=0.3))

    # Dynamic Actors
    (ego_dot,) = ax.plot([], [], color=PALETTE["ego"]["stroke"], marker="o", markersize=8, label="Ego")
    (ego_trail,) = ax.plot([], [], color=PALETTE["ego"]["stroke"], alpha=0.3)
    (plan_line,) = ax.plot([], [], color=PALETTE["plan"]["stroke"], linestyle="--", alpha=0.8, label="Plan")
    # Uncertainty Ellipse
    ego_cov_patch = patches.Ellipse((0,0), width=0, height=0, angle=0, facecolor=PALETTE["ego"]["fill"], edgecolor=PALETTE["ego"]["stroke"], alpha=0.2, label="Uncertainty")
    ax.add_patch(ego_cov_patch)
    
    obs_rect = patches.Rectangle((0,0), 0, 0, facecolor=PALETTE["obs_moving"]["fill"], edgecolor=PALETTE["obs_moving"]["stroke"], alpha=0.8, label="Other Car")
    ax.add_patch(obs_rect)
    ax.legend(loc="upper left")

    for t in range(T_SIM): 
        # A. Construct Local Environment for Horizon H
        # The planner needs to know where obstacles are from t to t+H
        env_local = Environment(device=device)
        # Pass goal (Entirety of Lane 2)
        env_local.set_goal(x_range=[0.0, 60.0], y_range=[3.0, 5.0])
        
        # Slice Moving Obstacle Trajectory
        idx_start = t
        idx_end = t + H + 1
        
        if idx_end <= len(obs_x_global):
            sl_x = obs_x_global[idx_start:idx_end]
            sl_y = obs_y_global[idx_start:idx_end]
        else:
            # Pad if we run past the defined trajectory
            pad = idx_end - len(obs_x_global)
            sl_x = np.concatenate([obs_x_global[idx_start:], np.full(pad, obs_x_global[-1])])
            sl_y = np.concatenate([obs_y_global[idx_start:], np.full(pad, obs_y_global[-1])])

        env_local.add_moving_obstacle(sl_x, sl_y, width=2.5, height=1.5)

        # B. Plan
        # Prepare Warm Start: Shift previous solution left by 1
        init_guess = None
        if prev_u_sol is not None:
            # [T, 2] -> Shift: u[0] becomes u_prev[1], ..., u[T-1] = u_prev[T-1]
            init_guess = torch.cat([prev_u_sol[1:], prev_u_sol[-1:]], dim=0)

        planner = ProbabilisticSTLPlanner(dynamics, env_local, T=H, config=planner_cfg)
        
        p_mean, p_cov, p_u, p_val, history = planner.solve(
            curr_mean, curr_cov, render=False, verbose=False, init_guess=init_guess
        )
        
        prev_u_sol = p_u.detach() # Store for next step
        all_plans.append(p_mean)
        loss_trace.append(history[-1] if history else 0.0)
        p_sat_trace.append(p_val)

        # C. Execute (Receding Horizon)
        u_curr = p_u[0]
        
        # Simulate Dynamics
        # Propagate Belief
        pred_mean, next_cov = dynamics.step(curr_mean, curr_cov, u_curr)
        
        # Simulate Reality
        noise = torch.distributions.MultivariateNormal(torch.zeros_like(pred_mean), dynamics.Q).sample()
        next_mean = pred_mean + noise

        real_mean_trace.append(next_mean)
        real_cov_trace.append(next_cov)
        real_u_trace.append(u_curr)
        
        curr_mean = next_mean
        curr_cov = next_cov
        
        # Logging
        obs_pos = np.array([obs_x_global[t], obs_y_global[t]])
        ego_pos = curr_mean.cpu().numpy()
        dist = np.linalg.norm(ego_pos[:2] - obs_pos)
        
        if t % 5 == 0:
            print(f"Step {t:03d} | Ego: [{ego_pos[0]:.2f}, {ego_pos[1]:.2f}] | Obs: [{obs_pos[0]:.2f}, {obs_pos[1]:.2f}] | Dist: {dist:.2f}")

        # Update Live Plot
        ego_x, ego_y = ego_pos[0], ego_pos[1]
        ego_dot.set_data([ego_x], [ego_y])
        
        # Update Trail
        path_x = [m[0].item() for m in real_mean_trace]
        path_y = [m[1].item() for m in real_mean_trace]
        ego_trail.set_data(path_x, path_y)

        # Update Uncertainty Ellipse
        pos_cov_np = curr_cov[:2, :2].cpu().numpy()
        vals, vecs = np.linalg.eigh(pos_cov_np)
        order = vals.argsort()[::-1]
        theta = np.degrees(np.arctan2(*vecs[:, order][:, 0][::-1]))
        w, h = 2 * 2.45 * np.sqrt(vals[order])
        ego_cov_patch.set_center((ego_x, ego_y))
        ego_cov_patch.set_width(w); ego_cov_patch.set_height(h); ego_cov_patch.set_angle(theta)

        # Update Plan
        plan_np = p_mean.detach().cpu().squeeze().numpy()
        plan_line.set_data(plan_np[:, 0], plan_np[:, 1])

        # Update Obstacle
        obs_rect.set_xy((obs_pos[0]-1.25, obs_pos[1]-0.75))
        obs_rect.set_width(2.5)
        obs_rect.set_height(1.5)

        plt.draw()
        plt.pause(0.001)

        # Check Goal
        if ego_pos[0] > 28.0 and 3.0 < ego_pos[1] < 5.0:
            print("Goal Reached!")
            break

    plt.ioff()
    plt.close(fig)

    # --- Sync Environment for Visualization ---
    # Truncate moving obstacle trajectory to match the actual simulation length
    actual_steps = len(real_mean_trace)
    for obs in env_global.moving_obstacles:
        obs["x_traj"] = obs["x_traj"][:actual_steps]
        obs["y_traj"] = obs["y_traj"][:actual_steps]

    # --- 6. Visualization ---
    full_mean_trace = torch.stack(real_mean_trace).unsqueeze(0)
    full_cov_trace = torch.stack(real_cov_trace).unsqueeze(0)
    full_u_trace = torch.stack(real_u_trace).unsqueeze(0)

    # --- Safety Check ---
    check_collision(full_mean_trace, env_global)

    # Static Plot with Controls
    visualize_results(
        full_mean_trace, 
        full_cov_trace, 
        full_u_trace, 
        env_global,
        history=loss_trace,
        p_sat_trace=p_sat_trace,
        robot_dims=(2.0, 1.0),
        layout="snapshots"
    )
    
    # We use env_global to show the full path of the obstacle in the animation
    animate_results(
        full_mean_trace, 
        full_cov_trace, 
        env_global, 
        filename="lane_change_mpc.gif", 
        plan_traces=all_plans, 
        step=4,  # Show every 4th frame
        robot_dims=(2.0, 1.0) # Draw ego vehicle as rectangle (Length, Width)
    )
