import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pdstl.operators import Always, Eventually, And
from planning.environment import Environment
from planning.dynamics import SingleIntegrator
from planning.planner import ProbabilisticSTLPlanner
from planning.visualization import visualize_results
from planning.animation import animate_results


def run_single_shot(max_iterations=500):
    # --- 1. Configuration ---
    # Detect device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    T = 120  # Time horizon
    dt = 0.2  # Time step size

    # --- 2. Setup Environment ---
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

    # --- 3. Setup Dynamics ---
    # Using SingleIntegrator (Velocity Control) as per PDF
    dynamics = SingleIntegrator(dt=dt, u_max=1.0, q_std=0.02, device=device)
      

    # --- 4. Planner Config ---
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

    # --- 5. Initial Condition ---
    # Start at (0,0) with small uncertainty
    x0_mean = torch.tensor([0.0, 0.0], device=device)
    x0_cov = torch.eye(2, device=device) * 0.01

    # --- 6. Solve ---
    print("Initializing Probabilistic STL Motion Planning...")

    # Run optimization
    mean_trace, cov_trace, u_trace, best_p, history = planner.solve(
        x0_mean, x0_cov, render=True
    )

    print("\nOptimization Complete.")
    print(f"Final Satisfaction Probability: {best_p:.4f}")

    # --- 7. Visualize ---
    # Pass results to the visualization module
    visualize_results(mean_trace, cov_trace, u_trace, env, history)

    # --- 8. Animate ---
    
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

    # --- 2. Setup Environment ---
    # Define workspace, goal, and obstacles
    env = Environment(device=device)

    # Scenario: Reach (3.5, 10.5)
    env.set_goal(x_range=[9.0, 10.0], y_range=[9.0, 10.0])
    

    # Add an obstacle between start and goal
    env.add_obstacle(x_range=[0.0, 2.0], y_range=[3.0, 8.0])
    env.add_obstacle(x_range=[2.0, 6.0], y_range=[-1.0, 1.0])
    env.add_obstacle(x_range=[8.0, 10.0], y_range=[3.0, 8.0])


    env.add_circle_obstacle(center=[5.0, 5.0], radius=2.0)
    
    # --- 3. Setup Dynamics ---
    # Using SingleIntegrator (Velocity Control) as per PDF
    dynamics = SingleIntegrator(dt=dt, u_max=1.0, q_std=0.02, device=device)

    # --- 4. Planner Config ---
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

    # --- 5. Initial Condition ---
    # Start at (0,0) with small uncertainty
    x0_mean = torch.tensor([0.0, 0.0], device=device)
    x0_cov = torch.eye(2, device=device) * 0.01

    # --- 6. MPC Loop ---
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
                (gx[0], gy[0]), gx[1] - gx[0], gy[1] - gy[0], color="green", alpha=0.3
            )
        )
    for obs in env.obstacles:
        ox, oy = obs["x"], obs["y"]
        ax_map.add_patch(
            patches.Rectangle(
                (ox[0], oy[0]), ox[1] - ox[0], oy[1] - oy[0], color="red", alpha=0.5
            )
        )
    for obs in env.circle_obstacles:
        c = patches.Circle(obs["center"], obs["radius"], color="red", alpha=0.5)
        ax_map.add_patch(c)

    (line_exec,) = ax_map.plot([], [], "b-o", label="Executed Path")
    (line_plan,) = ax_map.plot(
        [], [], color="orange", linestyle="--", alpha=0.8, label="Planned Window"
    )
    ax_map.legend(loc="upper left")

    # Setup P(Sat) Plot
    ax_p.set_xlim(0, 100)  # Initial view, will expand
    ax_p.set_ylim(0, 1.1)
    ax_p.set_title("Window Satisfaction Prob")
    ax_p.set_xlabel("Step")
    ax_p.set_ylabel("P(Sat)")
    ax_p.grid(True)
    (line_p,) = ax_p.plot([], [], "g-o", markersize=3)

    all_plans = []  # Store sliding windows for final animation
    p_sat_trace = []  # Store satisfaction probability
    loss_trace = []  # Store final loss of each step

    step = 0
    while step < MAX_STEPS:
        # 1. Check Goal Reached
        dist_to_goal = torch.norm(curr_mean - goal_center)
        if dist_to_goal < 0.5:
            print(f"Goal Reached at step {step}!")
            break

        # 2. Setup Planner for Sliding Window
        # The environment spec is generated for length H (Always_[0,H] Safe)
        mpc_planner = ProbabilisticSTLPlanner(dynamics, env, T=H, config=planner_cfg)

        # 3. Solve Optimization for the Window
        # verbose=False to avoid spamming console
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

        # 4. Extract First Control Action (Receding Horizon)
        u_curr = p_u[0]  # [2]

        # 5. Execute Dynamics (Simulate one step)
        # Add random noise to test robustness
        noise = torch.randn_like(curr_mean) * 0.05
        next_mean = curr_mean + u_curr * dt + noise
        next_cov = curr_cov + dynamics.Q

        # 6. Store and Update
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

    # --- 7. Visualize ---
    # Pass results to the visualization module
    visualize_results(
        full_mean_trace,
        full_cov_trace,
        full_u_trace,
        env,
        history=loss_trace,
        p_sat_trace=p_sat_trace,
    )

    # --- 8. Animate ---
    # Step=5 to speed up animation significantly (300 frames -> 60 frames)
    animate_results(
        full_mean_trace,
        full_cov_trace,
        env,
        filename="mpc_animation.gif",
        plan_traces=all_plans,
        step=5,
    )
