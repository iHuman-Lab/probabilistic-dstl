"""Controlled dynamics and the shared Gaussian belief boundary."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from models.beliefs import GaussianBelief
from models.dynamics import DoubleIntegrator, SingleIntegrator
from models.rollouts import (
    create_gaussian_belief_trajectory,
    gaussian_rollout,
    sample_trajectories,
)
from pdstl.base import Belief, BeliefTrajectory
from pdstl.predicates import GreaterThan, LessThan
from planning.environment import build_reach_avoid_environment
from planning.planner import Planner


def test_belief_contract_is_abstract_and_accepts_concrete_models():
    with pytest.raises(TypeError):
        Belief()
    belief = GaussianBelief(torch.zeros(1, 1), torch.eye(1).unsqueeze(0))
    assert isinstance(BeliefTrajectory([belief])[0], Belief)


@pytest.mark.parametrize(
    "model_type, dimension", [(SingleIntegrator, 2), (DoubleIntegrator, 4)]
)
def test_rollout_matches_step_and_linear_prediction(model_type, dimension):
    model = model_type(dt=0.2, u_max=2.0, q_std=0.1)
    initial_mean = torch.arange(dimension, dtype=torch.float32)
    initial_covariance = torch.eye(dimension) * 0.3
    parameters = torch.tensor([[0.2, -0.1], [-0.1, 0.3]], requires_grad=True)
    controls = model.bound_control(parameters)
    mean, covariance = model(parameters, initial_mean, initial_covariance)
    transition = torch.eye(dimension)
    if dimension == 2:
        control_matrix = torch.eye(2) * 0.2
    else:
        transition[0, 2] = transition[1, 3] = 0.2
        control_matrix = torch.tensor(
            [[0.02, 0], [0, 0.02], [0.2, 0], [0, 0.2]]
        )
    expected_mean, expected_covariance = initial_mean, initial_covariance
    torch.testing.assert_close(mean[0, 0], initial_mean)
    torch.testing.assert_close(covariance[0, 0], initial_covariance)
    for k, control in enumerate(controls):
        next_mean, next_covariance = model.step(
            expected_mean, expected_covariance, control
        )
        expected_mean = transition @ expected_mean + control_matrix @ control
        expected_covariance = (
            transition @ expected_covariance @ transition.T
            + torch.eye(dimension) * 0.01
        )
        torch.testing.assert_close(next_mean, expected_mean)
        torch.testing.assert_close(next_covariance, expected_covariance)
        torch.testing.assert_close(mean[0, k + 1], expected_mean)
        torch.testing.assert_close(covariance[0, k + 1], expected_covariance)
    mean[0, -1].sum().backward()
    assert torch.isfinite(parameters.grad).all()
    assert parameters.grad.abs().sum() > 0


def test_single_integrator_defaults_to_the_planar_model():
    model = SingleIntegrator(dt=0.2, q_std=0.05)
    torch.testing.assert_close(model.A, torch.eye(2))
    torch.testing.assert_close(model.B, 0.2 * torch.eye(2))
    torch.testing.assert_close(model.Q, 0.05**2 * torch.eye(2))


def test_scalar_single_integrator_propagates_mean_and_variance():
    dt, q_std = 0.2, 0.08
    model = SingleIntegrator(dt=dt, u_max=1.0, q_std=q_std, state_dim=1)
    v = torch.tensor([[0.3], [-0.5], [1.2]])
    mean, cov = model(v, torch.tensor([52.0]), torch.tensor([[0.25]]))

    u = model.bound_control(v)[:, 0]
    expected_mean = 52.0 + dt * torch.cat([torch.zeros(1), torch.cumsum(u, 0)])
    expected_var = 0.25 + q_std**2 * torch.arange(4.0)
    assert mean.shape == (1, 4, 1) and cov.shape == (1, 4, 1, 1)
    torch.testing.assert_close(mean[0, :, 0], expected_mean)
    torch.testing.assert_close(cov[0, :, 0, 0], expected_var)


def test_sample_step_is_the_predicted_step_plus_process_noise():
    model = SingleIntegrator(dt=0.2, u_max=1.0, q_std=0.1)
    x, P, u = (
        torch.tensor([1.0, -2.0]),
        torch.eye(2) * 0.3,
        torch.tensor([0.5, -0.25]),
    )
    predicted_mean, predicted_cov = model.step(x, P, u)

    torch.manual_seed(3)
    sampled, cov = model.sample_step(x, P, u)
    torch.manual_seed(3)
    noise = torch.distributions.MultivariateNormal(
        torch.zeros(2), model.Q
    ).sample()
    torch.testing.assert_close(sampled, predicted_mean + noise)
    torch.testing.assert_close(cov, predicted_cov)

    torch.manual_seed(4)
    draws = torch.stack([model.sample_step(x, P, u)[0] for _ in range(20_000)])
    torch.testing.assert_close(
        draws.mean(0), predicted_mean, atol=5e-3, rtol=0
    )
    torch.testing.assert_close(torch.cov(draws.T), model.Q, atol=5e-4, rtol=0)


def test_planner_accepts_the_shared_belief_in_a_small_window():
    environment = build_reach_avoid_environment(
        {
            "bounds": {"x_range": [-5.0, 5.0], "y_range": [-5.0, 5.0]},
            "goal": {"x_range": [0.5, 1.5], "y_range": [-0.5, 0.5]},
        }
    )
    planner = Planner(SingleIntegrator(), 3, config={"max_iters": 1})
    best = planner.optimize_window(
        gaussian_rollout(
            planner.dyn, torch.tensor([0.0, 0.0]), torch.eye(2) * 0.1
        ),
        init_guess=torch.zeros(3, 2),
        spec=environment.get_specification(3),
    )
    assert best.rollout.aux["mean_trace"].shape == (1, 4, 2)
    assert best.rollout.aux["cov_trace"].shape == (1, 4, 2, 2)
    assert best.controls.shape == (3, 2)
    assert 0 <= best.hard_interval[0] <= 1
    assert len(best.loss_history) == 1
    assert torch.isfinite(torch.tensor(best.loss_history)).all()


def test_gaussian_trajectory_factory_preserves_supported_shapes_and_types():
    scalar = create_gaussian_belief_trajectory(
        np.array([1.0, 2.0], dtype=np.float64), np.array([0.25, 1.0])
    )
    assert len(scalar) == 2 and scalar[0].mean.shape == (1, 1)
    assert scalar[0].mean.dtype == torch.float64

    vector = create_gaussian_belief_trajectory(
        np.zeros((3, 2)), np.eye(2)[None].repeat(3, 0)
    )
    assert len(vector) == 3 and vector[0].covariance.shape == (1, 2, 2)

    batched = create_gaussian_belief_trajectory(
        torch.zeros(4, 3, 2), torch.ones(4, 3, 2)
    )
    assert len(batched) == 3 and batched[0].mean.shape == (4, 2)
    assert GreaterThan(0.0)(batched).shape == (4, 3, 2)

    mean = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    trajectory = create_gaussian_belief_trajectory(mean, torch.ones(3, 2))
    assert len(trajectory) == 3 and trajectory[0].mean.shape == (1, 2)
    assert trajectory[0].mean.device == mean.device
    assert trajectory[0].covariance.dtype == torch.float64
    GreaterThan(0.0)(trajectory).sum().backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()


@pytest.mark.parametrize(
    "mean,covariance,message",
    [
        (
            np.zeros((2, 2, 2, 2)),
            np.zeros((2, 2, 2)),
            r"\[T\], \[T,D\] or \[B,T,D\]",
        ),
        (np.zeros((3, 2)), np.zeros((2, 2)), "same batch and number of steps"),
        (
            np.zeros((2, 3, 2)),
            np.zeros((2, 2, 2)),
            "same batch and number of steps",
        ),
    ],
)
def test_gaussian_trajectory_factory_rejects_mismatched_shapes(
    mean, covariance, message
):
    with pytest.raises(ValueError, match=message):
        create_gaussian_belief_trajectory(mean, covariance)


# --- Precise Gaussian semantics ----------------------------------------


@pytest.mark.parametrize("predicate", [GreaterThan(50.0), LessThan(50.0)])
def test_precise_gaussian_returns_the_exact_marginal_as_equal_bounds(
    predicate,
):
    mean, variance = 51.0, 4.0
    belief = GaussianBelief(
        torch.tensor([[mean]], dtype=torch.float64),
        torch.tensor([[variance]], dtype=torch.float64),
    )

    got = belief.probability_bounds(predicate)[0]

    z = (mean - 50.0) / variance**0.5
    expected = norm.cdf(z) if predicate.sense == ">=" else norm.cdf(-z)
    assert got[0].item() == pytest.approx(expected, abs=1e-12)
    assert got[0].item() == got[1].item()


@pytest.mark.parametrize(
    "mean, expect_ge, expect_le",
    [(51.0, 1.0, 0.0), (49.0, 0.0, 1.0), (50.0, 1.0, 1.0)],
)
def test_precise_gaussian_zero_variance_is_an_inclusive_comparison(
    mean, expect_ge, expect_le
):
    belief = GaussianBelief(torch.tensor([[mean]]), torch.zeros(1, 1))

    assert belief.probability_bounds(GreaterThan(50.0))[0].tolist() == [
        expect_ge,
        expect_ge,
    ]
    assert belief.probability_bounds(LessThan(50.0))[0].tolist() == [
        expect_le,
        expect_le,
    ]


# --- Input validation --------------------------------------------------


@pytest.mark.parametrize(
    "mean,covariance,message",
    [
        ([[float("nan"), 0.0]], [[1.0, 1.0]], "mean must be finite"),
        ([[0.0, 0.0]], [[float("-inf"), 1.0]], "covariance must be finite"),
    ],
)
def test_gaussian_belief_rejects_nonfinite_inputs(mean, covariance, message):
    mean, covariance = (
        torch.as_tensor(v, dtype=torch.float64) for v in (mean, covariance)
    )
    with pytest.raises(ValueError, match=message):
        GaussianBelief(mean, covariance)


def test_gaussian_belief_rejects_an_asymmetric_full_covariance():
    covariance = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]])

    with pytest.raises(ValueError, match="must be symmetric"):
        GaussianBelief(torch.zeros(1, 2), covariance)


def test_gaussian_belief_rejects_a_symmetric_non_positive_semidefinite_covariance():
    covariance = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])  # eigenvalues -1, 3

    with pytest.raises(ValueError, match="positive semi-definite"):
        GaussianBelief(torch.zeros(1, 2), covariance)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_large_deterministic_trajectory_avoids_eigensolver(
    device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    def unexpected_eigensolver(*args, **kwargs):
        pytest.fail("diagonal covariance must not require an eigensolver")

    monkeypatch.setattr(torch.linalg, "eigvalsh", unexpected_eigensolver)
    mean = torch.zeros(5000, 41, 2, device=device)
    mean[::2, :, 0] = 1.0
    trajectory = create_gaussian_belief_trajectory(
        mean, torch.zeros_like(mean)
    )
    bounds = trajectory.probability_bounds(GreaterThan(0.5))
    expected = (mean[..., 0] >= 0.5).to(mean.dtype)
    torch.testing.assert_close(bounds, torch.stack((expected, expected), -1))
    assert trajectory.trace.covariance.shape == (5000, 41, 2, 2)
    assert trajectory.trace.covariance.device == mean.device


@pytest.mark.parametrize("variance", [-1.0, -2e-6])
def test_gaussian_belief_rejects_negative_diagonal_variances(variance):
    with pytest.raises(ValueError, match="positive semi-definite"):
        GaussianBelief(torch.zeros(1, 2), torch.tensor([[variance, 1.0]]))


def test_gaussian_belief_accepts_diagonal_roundoff():
    GaussianBelief(torch.zeros(1, 2), torch.tensor([[-5e-7, 1.0]]))


def test_gaussian_belief_accepts_a_covariance_actually_propagated_by_rollout():
    """A regression guard: the new symmetry/PSD checks must not reject a
    covariance that came from real (float-roundoff-bearing) propagation."""
    model = DoubleIntegrator(dt=0.2, u_max=1.0, q_std=0.1)
    x, P = torch.zeros(4), torch.eye(4) * 0.3
    for _ in range(10):
        x, P = model.step(x, P, torch.tensor([0.3, -0.1]))

    GaussianBelief(x.unsqueeze(0), P.unsqueeze(0))  # must not raise


@pytest.mark.parametrize("z", [-7.3, -10.0])
def test_float32_gaussian_tail_keeps_a_usable_probability_and_gradient(z):
    mean = torch.tensor([[z]], requires_grad=True)
    belief = GaussianBelief(mean, torch.ones(1, 1))

    p = belief.probability_bounds(GreaterThan(0.0))[0, 0]
    p.backward()

    assert p.item() > 0
    assert p.item() == pytest.approx(norm.cdf(z), rel=1e-3, abs=0)
    assert torch.isfinite(mean.grad).all() and mean.grad.item() > 0


def test_sampled_trajectories_match_the_predicted_belief():
    model = SingleIntegrator(dt=0.2, u_max=1.0, q_std=0.1)
    mean0, cov0 = torch.tensor([1.0, -1.0]), torch.eye(2) * 0.04
    controls = torch.full((5, 2), 0.5)
    generator = torch.Generator().manual_seed(0)
    paths = sample_trajectories(
        model, mean0, cov0, controls, 20_000, generator
    )
    mean, covariance = model(torch.atanh(controls), mean0, cov0)
    assert paths.shape == (20_000, 6, 2)
    torch.testing.assert_close(paths.mean(0), mean[0], atol=5e-3, rtol=0)
    torch.testing.assert_close(
        torch.cov(paths[:, -1].T), covariance[0, -1], atol=5e-3, rtol=0
    )


def test_covariance_check_runs_in_chunks_and_still_finds_a_bad_matrix(
    monkeypatch,
):
    from models import beliefs

    monkeypatch.setattr(beliefs, "EIGEN_CHUNK", 4)
    mean = torch.zeros(3, 5, 2)
    covariance = torch.eye(2).expand(3, 5, 2, 2).clone()
    GaussianBelief(mean, covariance)  # 15 matrices, 4 chunks: all fine
    covariance[2, 4] = torch.tensor([[1.0, 2.0], [2.0, 1.0]])  # eigenvalue -1
    with pytest.raises(ValueError, match="positive semi-definite"):
        GaussianBelief(mean, covariance)  # the bad one is in the last chunk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_large_covariance_batch_is_checked_on_cuda():
    mean = torch.zeros(4096, 31, 4, device="cuda")
    covariance = torch.eye(4, device="cuda").expand(4096, 31, 4, 4).clone()
    GaussianBelief(mean, covariance)
