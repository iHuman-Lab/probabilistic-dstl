"""Gaussian calibration and the separation of lane truth from beliefs."""

import matplotlib.pyplot as plt
import pytest
import torch

from experiments import lane
from models.dynamics import DoubleIntegrator
from models.estimation import observe_gaussian, sample_gaussian
from utils import load_config
from visualization.figures import plot_lane_merge
from visualization.live_plots import _beliefs, create_live_view


def problem():
    cfg = load_config("configs/scenarios/lane/lane_change.yaml")
    cfg["planner"]["max_iters"] = 1
    return lane.build(cfg, device="cpu")


def executor(s, *, observation_noise=None, process_noise=None):
    return lane._executor(
        s,
        s.cfg["traffic_q_std"] ** 2 * torch.eye(4),
        torch.zeros(1, 4) if process_noise is None else process_noise,
        torch.zeros(1, 3, 4),
        False,
        ego_observation_noise=(
            torch.zeros(1, 4)
            if observation_noise is None
            else observation_noise
        ),
        traffic_observation_noise=torch.zeros(1, 3, 4),
    )


def test_kalman_update_matches_hand_calculation():
    mean, cov = observe_gaussian(
        torch.tensor([1.0, 2.0]),
        torch.diag(torch.tensor([1.0, 3.0])),
        torch.tensor([3.0, 6.0]),
        torch.eye(2),
    )
    torch.testing.assert_close(mean, torch.tensor([2.0, 5.0]))
    torch.testing.assert_close(cov, torch.diag(torch.tensor([0.5, 0.75])))


def test_sampling_preserves_singular_covariance_without_artificial_noise():
    mean = torch.tensor([1.0, 2.0, 3.0]).expand(20_000, -1)
    direction = torch.tensor([1.0, 2.0, 0.0])
    covariance = torch.outer(direction, direction)
    samples = sample_gaussian(
        mean, covariance, torch.Generator().manual_seed(5)
    )
    torch.testing.assert_close(samples[:, 2], mean[:, 2], atol=0, rtol=0)
    torch.testing.assert_close(
        torch.cov(samples.T), covariance, atol=0.05, rtol=0
    )
    deterministic = sample_gaussian(mean, torch.zeros(3, 3))
    torch.testing.assert_close(deterministic, mean, atol=0, rtol=0)


def test_filter_error_covariance_matches_monte_carlo_under_feedback():
    """20k independent worlds: estimation error matches the reported posterior."""
    generator = torch.Generator().manual_seed(41)
    dyn = DoubleIntegrator(q_std=0.03)
    mean = torch.zeros(20_000, 4)
    covariance = torch.diag(torch.tensor([0.1, 0.2, 0.03, 0.04]))
    truth = sample_gaussian(mean, covariance, generator)
    measurement_cov = torch.diag(torch.tensor([0.02, 0.03, 0.01, 0.04]))
    for _ in range(8):
        # The same control, determined from each world's estimate, acts on
        # its truth and prediction. It does not reveal hidden-state errors.
        controls = -0.2 * mean[:, :2] - 0.3 * mean[:, 2:]
        mean = mean @ dyn.A.T + controls @ dyn.B.T
        truth = sample_gaussian(
            truth @ dyn.A.T + controls @ dyn.B.T, dyn.Q, generator
        )
        covariance = dyn.A @ covariance @ dyn.A.T + dyn.Q
        observation = sample_gaussian(truth, measurement_cov, generator)
        mean, covariance = observe_gaussian(
            mean, covariance, observation, measurement_cov
        )
    errors = truth - mean
    torch.testing.assert_close(
        errors.mean(0), torch.zeros(4), atol=0.002, rtol=0
    )
    torch.testing.assert_close(
        torch.cov(errors.T), covariance, atol=0.0002, rtol=0.06
    )
    coverage = (
        (errors.abs() <= 1.96 * covariance.diag().sqrt()).float().mean(0)
    )
    assert ((coverage > 0.94) & (coverage < 0.96)).all()


def test_lane_initial_truth_is_sampled_from_each_configured_prior_in_lane():
    s = problem()
    generator = torch.Generator().manual_seed(8)
    states = [
        lane.initial_state_of(s, generator=generator) for _ in range(3000)
    ]
    ego = torch.stack([state.ego_true for state in states])
    traffic = torch.stack([state.traffic_true for state in states])
    torch.testing.assert_close(ego.mean(0), s.state[0], atol=0.006, rtol=0)
    torch.testing.assert_close(
        torch.cov(ego.T), s.state[1], atol=0.001, rtol=0
    )
    torch.testing.assert_close(
        traffic.mean(0), states[0].traffic_mean, atol=0.007, rtol=0
    )
    # Traffic keeps its lane: x, y, vx follow the prior, vy is exactly zero.
    for car in range(3):
        torch.testing.assert_close(
            torch.cov(traffic[:, car, :3].T),
            states[0].traffic_cov[car, :3, :3],
            atol=0.001,
            rtol=0,
        )
    assert (traffic[..., 3] == 0).all()
    torch.testing.assert_close(states[-1].ego_mean, s.state[0])
    torch.testing.assert_close(states[-1].ego_cov, s.state[1])


def test_execution_predicts_then_conditions_without_replacing_truth_by_mean():
    s = problem()
    state = lane.initial_state_of(
        s, generator=torch.Generator().manual_seed(4)
    )
    control = torch.tensor([0.3, 0.4])
    predicted_mean, predicted_cov = s.dyn.step(
        state.ego_mean, state.ego_cov, control
    )
    expected_true = s.dyn.A @ state.ego_true + s.dyn.B @ control
    updated = executor(s)(state, control, 0)
    expected_mean, expected_cov = observe_gaussian(
        predicted_mean, predicted_cov, expected_true, s.ego_observation_cov
    )
    torch.testing.assert_close(updated.ego_true, expected_true)
    torch.testing.assert_close(updated.ego_mean, expected_mean)
    torch.testing.assert_close(updated.ego_cov, expected_cov)
    assert not torch.equal(updated.ego_true, updated.ego_mean)
    assert torch.linalg.eigvalsh(predicted_cov - updated.ego_cov).min() > 0
    torch.testing.assert_close(
        updated.traffic_true, state.traffic_true @ s.dyn.A.T
    )
    traffic_mean, traffic_cov = observe_gaussian(
        state.traffic_mean @ s.dyn.A.T,
        s.dyn.A @ state.traffic_cov @ s.dyn.A.T
        + s.cfg["traffic_q_std"] ** 2 * torch.eye(4),
        updated.traffic_true,
        s.traffic_observation_cov,
    )
    torch.testing.assert_close(updated.traffic_mean, traffic_mean)
    torch.testing.assert_close(updated.traffic_cov, traffic_cov)
    other = executor(s, observation_noise=torch.ones(1, 4))(state, control, 0)
    torch.testing.assert_close(other.ego_true, updated.ego_true)
    assert not torch.equal(other.ego_mean, updated.ego_mean)


def test_physical_outcomes_use_truth_not_estimated_states_or_progress():
    s = problem()
    state = lane.initial_state_of(s)
    traffic = state.traffic_true.clone()
    traffic[0] = state.ego_true
    assert (
        lane._outcome(s)(state._replace(traffic_true=traffic), 0)
        == "collision"
    )
    ego = state.ego_true.clone()
    ego[1] = -10
    assert (
        lane._outcome(s)(state._replace(ego_true=ego), 0) == "road_violation"
    )
    estimated = state.ego_mean.clone()
    estimated[1] = 3.6
    assert (
        lane._outcome(s)(state._replace(ego_mean=estimated, streak=6), 18)
        is None
    )
    ego = state.ego_true.clone()
    ego[1] = 3.6
    settle = s.env.metadata["task"]["settle_steps"]
    done = state._replace(ego_true=ego, true_streak=6, true_completed=18)
    assert lane._outcome(s)(done, 18) is None
    assert lane._outcome(s)(done, 18 + settle) == "success"
    estimated_only = state._replace(completed=18)
    assert lane._outcome(s)(estimated_only, 18 + settle) != "success"


def test_speed_noise_is_reported_instead_of_silently_clamped():
    s = problem()
    state = lane.initial_state_of(s)
    noise = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
    updated = executor(s, process_noise=noise)(state, torch.zeros(2), 0)
    assert updated.ego_true[2] > 25
    assert lane._outcome(s)(updated, 1) == "speed_violation"


def test_estimated_completion_does_not_create_a_negative_dwell_window():
    s = problem()
    state = lane.initial_state_of(s)._replace(streak=12)
    spec = lane._specification(s, None, state, 24)
    assert spec is not None


def test_trials_are_seeded_locally_and_plans_start_from_beliefs(tmp_path):
    s = problem()
    rng = torch.random.get_rng_state().clone()
    first = lane.trial(s, max_steps=1)
    second = lane.trial(s, max_steps=1)
    assert first.stopped_reason == "step_limit"
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    for a, b in zip(first.states, second.states):
        for x, y in zip(a, b):
            if torch.is_tensor(x):
                torch.testing.assert_close(x, y)
            else:
                assert x == y
    plan = first.window_plans[0]
    torch.testing.assert_close(
        plan.rollout.aux["mean_trace"][0, 0], first.states[0].ego_mean
    )
    assert not torch.equal(first.states[0].ego_mean, first.states[0].ego_true)
    path = tmp_path / "lane.pt"
    torch.save(first, path)
    restored = torch.load(path, weights_only=False)
    torch.testing.assert_close(
        restored.states[-1].ego_true, first.states[-1].ego_true
    )
    torch.testing.assert_close(
        restored.states[-1].ego_cov, first.states[-1].ego_cov
    )


def test_executed_plots_use_truth_and_ellipses_use_beliefs():
    s = problem()
    result = lane.trial(s, max_steps=1)
    figures = plot_lane_merge(result, s.env, dt=s.cfg["dt"])
    axis = figures["combined"][1][0]
    executed = next(
        line for line in axis.lines if line.get_label() == "Executed"
    )
    assert list(executed.get_ydata()) == pytest.approx(
        [x.ego_true[1].item() for x in result.states]
    )
    mean, covariance, _ = _beliefs(result.states[-1])[0]
    torch.testing.assert_close(mean, result.states[-1].ego_mean)
    torch.testing.assert_close(covariance, result.states[-1].ego_cov)
    _, axes, observe = create_live_view(
        s.env, result.states[0], dt=s.cfg["dt"]
    )
    observe(0, result.states[1], result.window_plans[0])
    executed = next(
        line for line in axes[0].lines if line.get_label() == "Executed"
    )
    assert list(executed.get_ydata()) == pytest.approx(
        [x.ego_true[1].item() for x in result.states]
    )
    plt.close("all")


@pytest.mark.parametrize("std", [0, -1, float("nan"), [0.1, 0.2]])
def test_invalid_observation_noise_is_rejected(std):
    cfg = load_config("configs/scenarios/lane/lane_change.yaml")
    cfg["observation"]["ego_std"] = std
    with pytest.raises(ValueError, match="observation.ego_std"):
        lane.build(cfg, device="cpu")
