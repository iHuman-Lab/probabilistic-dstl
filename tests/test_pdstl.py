"""pdSTL semantics: atoms, rectangles, Boolean and temporal bounds, smoothing."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from models.beliefs import GaussianBelief, ProbabilityBelief
from models.rollouts import create_probability_belief_trajectory
from pdstl.base import BeliefTrajectory
from pdstl.operators import (
    END,
    Always,
    And,
    Eventually,
    Implies,
    Maxish,
    Minish,
    Negation,
    Or,
    Predicate,
    Until,
    _conjunction,
    _pair,
    _reduce_bounds,
)
from pdstl.recurrent import RecurrentTemporal
from pdstl.predicates import (
    AxisInterval,
    GreaterThan,
    HalfSpace,
    InsideRectangle,
    LessThan,
    OutsideRectangle,
)


def supplied(*rows):
    """Trajectory of supplied bounds; each row is {event: (lower, upper)}."""
    return BeliefTrajectory(
        [
            ProbabilityBelief(
                {k: torch.tensor([[lo, hi]]) for k, (lo, hi) in r.items()}
            )
            for r in rows
        ]
    )


def scalar(values, event="p"):
    """Trajectory of exact supplied probabilities for one event."""
    return supplied(*[{event: (v, v)} for v in values])


def gaussian(mean, var):
    """Trajectory of precise Gaussian steps from [T, D] mean and [T, D] or
    [T, D, D] var."""
    mean = torch.as_tensor(mean, dtype=torch.float64)
    var = torch.as_tensor(var, dtype=torch.float64)
    return BeliefTrajectory(
        [
            GaussianBelief(mean[t : t + 1], var[t : t + 1])
            for t in range(len(mean))
        ]
    )


# --- 1. Gaussian predicate -------------------------------------------------


def test_gaussian_matches_the_analytic_probability():
    mean, var = [[45.0], [55.0], [50.0]], [[4.0], [4.0], [9.0]]
    traj = gaussian(mean, var)

    got = GreaterThan(50.0)(traj)[0].numpy()
    exact = 1.0 - norm.cdf(
        (50.0 - np.array(mean)[:, 0]) / np.sqrt(np.array(var)[:, 0])
    )

    np.testing.assert_allclose(got[:, 0], exact, atol=1e-9)
    np.testing.assert_allclose(got[:, 1], exact, atol=1e-9)  # exact -> [p, p]


def test_less_than_is_the_complement():
    traj = gaussian([[45.0], [55.0]], [[4.0], [4.0]])

    gt = GreaterThan(50.0)(traj)[0].numpy()
    lt = LessThan(50.0)(traj)[0].numpy()

    np.testing.assert_allclose(lt[:, 0], 1.0 - gt[:, 0], atol=1e-9)


@pytest.mark.parametrize(
    "m, c, expect_ge",
    [
        (3.0, 1.0, 1.0),
        (1.0, 3.0, 0.0),
        (2.0, 2.0, 1.0),
    ],  # equality is inclusive
)
def test_zero_variance_is_an_inclusive_deterministic_comparison(
    m, c, expect_ge
):
    traj = gaussian([[m]], [[0.0]])

    assert GreaterThan(c)(traj)[0, 0, 0].item() == expect_ge
    assert LessThan(c)(traj)[0, 0, 0].item() == (1.0 if m <= c else 0.0)


def test_negative_variance_is_rejected():
    with pytest.raises(ValueError, match="positive semi-definite"):
        gaussian([[1.0]], [[-2.0]])


def test_full_covariance_selects_the_requested_marginal():
    mean = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    covariance = torch.tensor([[[4.0, 3.0], [3.0, 9.0]]], dtype=torch.float64)
    trajectory = BeliefTrajectory([GaussianBelief(mean, covariance)])

    got = GreaterThan(0.0, dim=1)(trajectory)[0, 0, 0].item()

    assert got == pytest.approx(norm.cdf(2.0 / 3.0), abs=1e-9)


@pytest.mark.parametrize(
    "mean,covariance,message",
    [
        (torch.zeros(2), torch.ones(2), "mean must have shape"),
        (torch.zeros(2, 2), torch.ones(3, 2), "covariance must be"),
        (torch.zeros(2, 2), torch.ones(2, 2, 3), "covariance must be"),
        (torch.zeros(2, 2), torch.ones(2, 3, 3), "covariance must be"),
    ],
)
def test_gaussian_belief_rejects_invalid_shapes(mean, covariance, message):
    with pytest.raises(ValueError, match=message):
        GaussianBelief(mean, covariance)


def test_gaussian_belief_rejects_an_invalid_predicate_dimension():
    trajectory = BeliefTrajectory(
        [GaussianBelief(torch.zeros(1, 2), torch.ones(1, 2))]
    )

    with pytest.raises(ValueError, match="dimension 2 is outside"):
        GreaterThan(0.0, dim=2)(trajectory)


def test_unsupported_event_fails_clearly():
    with pytest.raises(ValueError, match="cannot evaluate"):
        Predicate("safe")(gaussian([[1.0]], [[1.0]]))


# --- 2. Trace assembly and validation --------------------------------------


def test_supplied_bounds_assemble_into_a_trace():
    traj = scalar([0.2, 0.5, 0.9])

    trace = Predicate("p")(traj)

    assert trace.shape == (1, 3, 2)
    np.testing.assert_allclose(trace[0, :, 0].numpy(), [0.2, 0.5, 0.9])


def test_events_are_served_by_name():
    traj = supplied({"safe": (0.9, 0.97), "reach": (0.1, 0.42)})

    np.testing.assert_allclose(
        Predicate("safe")(traj)[0, 0].numpy(), [0.9, 0.97]
    )
    with pytest.raises(
        ValueError, match="no supplied bounds for event 'other'"
    ):
        Predicate("other")(traj)


@pytest.mark.parametrize(
    "bad", [(0.8, 0.3), (float("nan"), 0.5), (-0.2, 0.5), (0.5, 1.4)]
)
def test_malformed_bounds_are_rejected_naming_the_step(bad):
    traj = supplied({"p": (0.4, 0.6)}, {"p": bad})

    with pytest.raises(ValueError, match="at step 1"):
        Predicate("p")(traj)


def test_validation_can_be_disabled():
    traj = supplied({"p": (0.8, 0.3)})

    np.testing.assert_allclose(
        Predicate("p")(traj, validate=False)[0, 0].numpy(), [0.8, 0.3]
    )


# --- 3. Boolean bounds -----------------------------------------------------


def test_and_uses_frechet_not_the_product():
    # P(A)=0.9, P(B)=0.1 admits P(A and B)=0, so the product 0.09 is unsound.
    traj = supplied({"a": (0.9, 0.9), "b": (0.1, 0.1)})

    lower, upper = And(Predicate("a"), Predicate("b"))(traj)[0, 0].tolist()

    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(0.1)


def test_boolean_bounds_on_hand_calculated_values():
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    np.testing.assert_allclose(
        And(a, b)(traj)[0, 0].numpy(), [0.7 + 0.6 - 1, 0.8], atol=1e-6
    )
    np.testing.assert_allclose(
        Negation(a)(traj)[0, 0].numpy(), [1 - 0.8, 1 - 0.7], atol=1e-6
    )

    # Or, chosen so the upper endpoint does not saturate
    small = supplied({"a": (0.2, 0.3), "b": (0.1, 0.4)})
    np.testing.assert_allclose(
        Or(a, b)(small)[0, 0].numpy(), [0.2, 0.3 + 0.4], atol=1e-6
    )
    # and a case where it does
    assert Or(a, b)(traj)[0, 0, 1].item() == pytest.approx(1.0)


def test_nested_pointwise_boolean_composes_frechet_at_each_level():
    """And(And(a,b), Or(b,c)), evaluated standalone (not merely as a side
    effect of being wrapped in a temporal operator): is_pointwise propagates
    correctly through two levels of nesting, and the numeric result matches
    hand-computed nested Frechet bounds."""
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9), "c": (0.3, 0.5)})
    a, b, c = Predicate("a"), Predicate("b"), Predicate("c")

    inner_and = And(a, b)
    inner_or = Or(b, c)
    nested = And(inner_and, inner_or)

    assert (
        inner_and.is_pointwise
        and inner_or.is_pointwise
        and nested.is_pointwise
    )

    # inner_and = [max(0, 0.7+0.6-1), min(0.8,0.9)] = [0.3, 0.8]
    # inner_or  = [max(0.6,0.3), min(1, 0.9+0.5)]   = [0.6, 1.0]
    # nested    = [max(0, 0.3+0.6-1), min(0.8,1.0)] = [0.0, 0.8]
    np.testing.assert_allclose(
        inner_and(traj)[0, 0].numpy(), [0.3, 0.8], atol=1e-6
    )
    np.testing.assert_allclose(
        inner_or(traj)[0, 0].numpy(), [0.6, 1.0], atol=1e-6
    )
    np.testing.assert_allclose(
        nested(traj)[0, 0].numpy(), [0.0, 0.8], atol=1e-6
    )


def test_implies_is_negation_then_or():
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    torch.testing.assert_close(Implies(a, b)(traj), Or(Negation(a), b)(traj))


def test_both_conjunction_endpoints_are_smoothed():
    """Negation swaps the endpoints, so a hard upper would leave negated conjunctions
    -- every OutsideRectangle -- only partly differentiable."""
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    exact = And(a, b)(traj, beta=None)[0, 0]
    smooth = And(a, b)(traj, beta=5.0)[0, 0]

    torch.testing.assert_close(exact, torch.tensor([0.3, 0.8]))
    # A positive conjunction keeps its exact lower in smooth mode.
    torch.testing.assert_close(smooth[0], exact[0])
    assert smooth[1] != exact[1], "the upper endpoint must be a smooth min"
    assert smooth[1] >= exact[1], "the smooth upper must stay sound"


def test_disjunction_and_implication_are_smoothed():
    """Implies is built from Or, so a hard Or would make implications partly differentiable."""
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    for spec in (Or(a, b), Implies(a, b)):
        exact, smooth = spec(traj, beta=None)[0, 0], spec(traj, beta=5.0)[0, 0]
        assert not torch.allclose(exact, smooth), f"{spec} is not smoothed"


def test_negation_is_exact_in_both_modes():
    """Negation only swaps and complements, so it introduces no smoothing of its own."""
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    spec = Negation(Predicate("a"))
    torch.testing.assert_close(spec(traj, beta=5.0), spec(traj, beta=None))


def test_smoothed_operators_converge_to_the_exact_values_as_beta_grows():
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    for spec in (And(a, b), Or(a, b), Implies(a, b)):
        exact = spec(traj, beta=None)[0, 0]
        errors = [
            float((spec(traj, beta=beta)[0, 0] - exact).abs().max())
            for beta in (5.0, 50.0, 500.0)
        ]
        assert errors == sorted(errors, reverse=True), (
            f"{spec} does not converge"
        )
        assert errors[-1] < 1e-2


def test_smooth_and_escapes_the_exact_zero_clamp():
    lower = torch.tensor([[0.3], [0.4]], requires_grad=True)  # L1 + L2 - 1 < 0
    upper = torch.ones(2, 1)

    def conjunction_lower(beta):
        traj = BeliefTrajectory(
            [
                ProbabilityBelief(
                    {
                        "a": torch.stack([lower[0], upper[0]], dim=-1),
                        "b": torch.stack([lower[1], upper[1]], dim=-1),
                    }
                )
            ]
        )
        return And(Predicate("a"), Predicate("b"))(traj, beta=beta)[0, 0, 0]

    exact = conjunction_lower(None)
    (exact_grad,) = torch.autograd.grad(exact, lower)
    smooth = conjunction_lower(10.0)
    (smooth_grad,) = torch.autograd.grad(smooth, lower)

    assert exact.item() == 0.0 and exact_grad.abs().sum() == 0
    assert (
        smooth.item() <= exact.item()
        and torch.isfinite(smooth_grad).all()
        and (smooth_grad > 0).all()
    )


def test_smooth_min_and_max_converge_to_the_exact_values_as_beta_grows():
    x = torch.tensor([[0.2, 0.9, 0.5, 0.7]])
    errors = []
    for beta in (1.0, 10.0, 100.0, 1000.0):
        smooth_min, smooth_max = (
            Minish()(x, beta)[0, 0],
            Maxish()(x, beta)[0, 0],
        )
        assert (
            x.min() <= smooth_min <= x.mean() <= smooth_max <= x.max()
        )  # normalized
        errors.append(max(smooth_min - x.min(), x.max() - smooth_max).item())
        assert errors[-1] <= np.log(4) / beta + 1e-6
    assert errors == sorted(errors, reverse=True)
    assert (
        Minish()(x, None)[0, 0] == x.min()
        and Maxish()(x, None)[0, 0] == x.max()
    )


def _random_bounds(generator, steps):
    """[steps, 2] ordered probability bounds."""
    pair = torch.rand(steps, 2, generator=generator, dtype=torch.float64)
    return pair.sort(dim=-1).values


def _sound_formulas():
    a, b = Predicate("a"), Predicate("b")
    return {
        "always": Always(a, interval=[1, 6]),
        "eventually": Eventually(a, interval=[0, 5]),
        "suffix_always": Always(a),
        "delayed_eventually": Eventually(b, interval=[2, np.inf]),
        "nested": Eventually(Always(a, interval=[0, 3]), interval=[0, 4]),
        "outside": Negation(And(a, b)),
        "negated_always": Negation(Always(Or(a, b), interval=[0, 5])),
        "reach_avoid": Always(Negation(And(a, b)), interval=[1, 8])
        & Eventually(b, interval=[1, 8]),
        "either_or": Eventually(Always(a, interval=[0, 2]), interval=[0, 6])
        | Eventually(Always(b, interval=[0, 2]), interval=[0, 6]),
        "until": Until(a, b, interval=[0, 5]),
    }


@pytest.mark.parametrize("name", sorted(_sound_formulas()))
@pytest.mark.parametrize("beta", [1.0, 10.0, 200.0])
def test_smooth_pair_is_an_outer_interval_of_the_exact_one(name, beta):
    formula = _sound_formulas()[name]
    generator = torch.Generator().manual_seed(7)
    for _ in range(20):
        a, b = _random_bounds(generator, 12), _random_bounds(generator, 12)
        trajectory = supplied(
            *[
                {"a": tuple(a[t].tolist()), "b": tuple(b[t].tolist())}
                for t in range(12)
            ]
        )
        exact = formula(trajectory)
        smooth = formula(trajectory, beta=beta)
        # float32 ties: smooth and exact agree to round-off at the extreme.
        assert (smooth[..., 0] <= exact[..., 0] + 1e-6).all()
        assert (smooth[..., 1] >= exact[..., 1] - 1e-6).all()


def test_smooth_lower_approaches_a_positive_exact_lower_as_beta_grows():
    """Where the exact bound is positive, the smooth one closes on it.

    (A clamped zero conjunction keeps its affine excess at every beta.)
    """
    formula = _sound_formulas()["reach_avoid"]
    generator = torch.Generator().manual_seed(3)
    a = 0.1 * _random_bounds(generator, 12)  # rarely in the obstacle
    b = 0.8 + 0.2 * _random_bounds(generator, 12)  # usually in the goal
    trajectory = supplied(
        *[
            {"a": tuple(a[t].tolist()), "b": tuple(b[t].tolist())}
            for t in range(12)
        ]
    )
    exact = formula.probability_interval(trajectory)[0]
    assert exact > 0
    gaps = [
        (exact - formula.smooth_lower(trajectory, beta)).item()
        for beta in (10.0, 100.0, 1000.0, 10000.0)
    ]
    assert all(gap >= -1e-6 for gap in gaps)
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-2


def test_pointwise_flag_tracks_the_formula_kind():
    a, b = Predicate("a"), Predicate("b")
    windowed = Always(a, interval=[0, 1])

    assert a.is_pointwise
    assert Negation(a).is_pointwise
    assert And(a, b).is_pointwise
    assert Implies(a, b).is_pointwise

    assert not windowed.is_pointwise
    assert not Until(a, b).is_pointwise
    assert not And(windowed, b).is_pointwise
    assert not Negation(windowed).is_pointwise
    assert not Implies(a, windowed).is_pointwise


A_TRACE = [(0.9, 0.95), (0.7, 0.8), (0.85, 0.9), (0.6, 0.75), (0.95, 1.0)]
B_TRACE = [(0.2, 0.4), (0.5, 0.6), (0.3, 0.35), (0.8, 0.9), (0.1, 0.2)]


def ab_trajectory():
    return supplied(*[{"a": a, "b": b} for a, b in zip(A_TRACE, B_TRACE)])


def window_ref(trace, a, b, reduce):
    return [
        [
            reduce(x[0] for x in trace[t + a : t + b + 1]),
            reduce(x[1] for x in trace[t + a : t + b + 1]),
        ]
        for t in range(len(trace) - b)
    ]


def and_ref(x, y):
    return [max(0.0, x[0] + y[0] - 1.0), min(x[1], y[1])]


def or_ref(x, y):
    return [max(x[0], y[0]), min(1.0, x[1] + y[1])]


def not_ref(x):
    return [1.0 - x[1], 1.0 - x[0]]


def test_boolean_combines_temporal_children_at_the_same_origins():
    a, b = Predicate("a"), Predicate("b")
    always_a = Always(a, interval=[0, 2])  # 3 origins
    eventually_b = Eventually(b, interval=[0, 1])  # 4 origins
    alw = window_ref(A_TRACE, 0, 2, min)
    ev = window_ref(B_TRACE, 0, 1, max)

    got_and = And(always_a, eventually_b)(ab_trajectory())[0]
    got_or = Or(always_a, eventually_b)(ab_trajectory())[0]

    assert got_and.shape == got_or.shape == (3, 2)
    np.testing.assert_allclose(
        got_and.tolist(), [and_ref(alw[t], ev[t]) for t in range(3)], atol=1e-6
    )
    np.testing.assert_allclose(
        got_or.tolist(), [or_ref(alw[t], ev[t]) for t in range(3)], atol=1e-6
    )


def test_boolean_combines_temporal_and_atomic_children():
    a, b = Predicate("a"), Predicate("b")
    ev = window_ref(A_TRACE, 0, 1, max)

    got = And(Eventually(a, interval=[0, 1]), b)(ab_trajectory())[0]

    np.testing.assert_allclose(
        got.tolist(), [and_ref(ev[t], B_TRACE[t]) for t in range(4)], atol=1e-6
    )


def test_negation_of_a_temporal_formula_is_its_dual():
    a = Predicate("a")
    traj = ab_trajectory()

    got = Negation(Always(a, interval=[0, 2]))(traj)

    np.testing.assert_allclose(
        got[0].tolist(),
        [not_ref(x) for x in window_ref(A_TRACE, 0, 2, min)],
        atol=1e-6,
    )
    torch.testing.assert_close(
        got, Eventually(Negation(a), interval=[0, 2])(traj)
    )


def test_implies_over_temporal_children_is_negation_then_or():
    a, b = Predicate("a"), Predicate("b")
    left, right = Always(a, interval=[0, 1]), Eventually(b, interval=[0, 2])
    traj = ab_trajectory()

    torch.testing.assert_close(
        Implies(left, right)(traj), Or(Negation(left), right)(traj)
    )


def test_nested_boolean_and_temporal_composition_matches_the_reference():
    a, b = Predicate("a"), Predicate("b")
    inner = Or(
        Negation(Always(a, interval=[0, 2])),
        And(Eventually(b, interval=[0, 1]), a),
    )
    formula = Eventually(inner, interval=[0, 1])

    alw = window_ref(A_TRACE, 0, 2, min)
    ev = window_ref(B_TRACE, 0, 1, max)
    inner_ref = [
        or_ref(not_ref(alw[t]), and_ref(ev[t], A_TRACE[t])) for t in range(3)
    ]

    got = formula(ab_trajectory())[0]

    np.testing.assert_allclose(
        got.tolist(), window_ref(inner_ref, 0, 1, max), atol=1e-6
    )


# --- 4. Always / Eventually ------------------------------------------------


@pytest.mark.parametrize("interval", [[0, 1], [1, 2], [2, 3]])
def test_always_and_eventually_match_direct_window_reductions(interval):
    # Distinct lower and upper values, so each endpoint is reduced on its own.
    rng = np.random.default_rng(0)
    lows = rng.uniform(0.0, 0.5, size=9).tolist()
    highs = rng.uniform(0.5, 1.0, size=9).tolist()
    traj = supplied(*[{"p": (lo, hi)} for lo, hi in zip(lows, highs)])
    a, b = interval
    n = len(lows) - b

    got_min = Always(Predicate("p"), interval=interval)(traj)[0].tolist()
    got_max = Eventually(Predicate("p"), interval=interval)(traj)[0].tolist()

    assert len(got_min) == n and len(got_max) == n
    np.testing.assert_allclose(
        got_min,
        [
            [min(lows[t + a : t + b + 1]), min(highs[t + a : t + b + 1])]
            for t in range(n)
        ],
    )
    np.testing.assert_allclose(
        got_max,
        [
            [max(lows[t + a : t + b + 1]), max(highs[t + a : t + b + 1])]
            for t in range(n)
        ],
    )


@pytest.mark.parametrize("beta", [None, 0.5, 3.0, 40.0])
@pytest.mark.parametrize(
    "interval",
    [None, [0, np.inf], [3, np.inf], [0, 0], [0, 4], [2, 7], [5, 5]],
)
@pytest.mark.parametrize("operator", [Always, Eventually])
def test_temporal_windows_are_the_window_reductions(operator, interval, beta):
    torch.manual_seed(0)
    lower = torch.rand(2, 12) - 0.3  # smooth lowers can leave [0, 1]
    upper = lower + torch.rand(2, 12)
    trace = _pair(lower, upper).requires_grad_(True)

    class Supplied(Predicate):
        def robustness_trace(self, belief_trajectory, **kwargs):
            return trace

    operation = Minish() if operator is Always else Maxish()
    new = operator(Supplied(), interval)(None, beta=beta)
    # Eq. (9) with the sound smooth reductions: one reduction per origin
    # over its window (to the trace end for [a, inf)).
    a, b = [0, np.inf] if interval is None else interval
    steps = trace.shape[1]
    last = steps - 1 if np.isinf(b) else None
    origins = steps - (a if np.isinf(b) else b)
    direct = torch.stack(
        [
            _reduce_bounds(
                operation,
                trace[:, k + a : (k + b if last is None else last) + 1],
                beta,
                keepdim=False,
            )
            for k in range(origins)
        ],
        dim=1,
    )
    torch.testing.assert_close(new, direct, atol=1e-5, rtol=1e-5)
    (new_grad,) = torch.autograd.grad(new.sum(), trace)
    (direct_grad,) = torch.autograd.grad(direct.sum(), trace)
    torch.testing.assert_close(new_grad, direct_grad, atol=1e-5, rtol=1e-5)
    # Bounded windows, and every exact window, also equal the step-by-step
    # register recursion used before (Eq. (12)).
    if not np.isinf(b) or beta is None:
        old = RecurrentTemporal(Supplied(), interval, operation)(
            None, beta=beta
        )
        torch.testing.assert_close(new, old, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("beta", [None, 0.5, 3.0, 40.0])
@pytest.mark.parametrize("interval", [[0, 0], [0, 4], [2, 7], [5, 5]])
def test_vectorized_until_matches_the_step_by_step_loop(interval, beta):
    torch.manual_seed(1)

    def supplied_trace():
        lower = torch.rand(2, 12) - 0.3
        return _pair(lower, lower + torch.rand(2, 12)).requires_grad_(True)

    phi, psi = supplied_trace(), supplied_trace()

    class Supplied(Predicate):
        def __init__(self, trace):
            super().__init__()
            self.trace = trace

        def robustness_trace(self, belief_trajectory, **kwargs):
            return self.trace

    until = Until(Supplied(phi), Supplied(psi), interval)
    new = until(None, beta=beta)
    # The per-origin loop Until used before (it still runs for [a, inf)).
    a, b = interval
    n_valid = phi.shape[1] - b
    old = []
    for t in range(n_valid):
        candidates = [
            _conjunction(
                _reduce_bounds(
                    Minish(), phi[:, t : tau + 1], beta, keepdim=False
                ),
                psi[:, tau],
                beta,
            )
            for tau in range(t + a, t + b + 1)
        ]
        old.append(
            _reduce_bounds(
                Maxish(), torch.stack(candidates, dim=1), beta, keepdim=False
            )
        )
    old = torch.stack(old, dim=1)
    torch.testing.assert_close(new, old, atol=1e-5, rtol=1e-5)
    (new_grad,) = torch.autograd.grad(new.sum(), phi)
    (old_grad,) = torch.autograd.grad(old.sum(), phi)
    torch.testing.assert_close(new_grad, old_grad, atol=1e-5, rtol=1e-5)


def test_end_window_is_the_rest_of_the_trace_and_prints_as_end():
    values = [0.9, 0.2, 0.7, 0.4, 0.6]
    traj = scalar(values)
    spec = Always(Predicate("p"), [1, END])
    got = spec(traj)[0, :, 0].tolist()
    # At origin k the window is k + 1 .. last step: finite, length 4 - k.
    np.testing.assert_allclose(
        got, [min(values[k + 1 :]) for k in range(len(values) - 1)]
    )
    assert str(spec) == "□_[1, end](p)"
    assert "U_[0, end]" in str(Until(Predicate("p"), Predicate("q"), [0, END]))


def test_smooth_temporal_output_is_differentiable():
    p = torch.tensor([[0.2, 0.8]], requires_grad=True)
    values = [p, p * 0.5, p * 0.75, p * 0.9]
    traj = BeliefTrajectory([ProbabilityBelief({"p": v}) for v in values])

    Always(Predicate("p"), interval=[0, 2])(traj, beta=3.0)[0, 0, 0].backward()

    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


def _sound_min(window, beta):
    """Sound soft-min of [n, 2] bounds: un-normalized lower, normalized upper."""
    unnormalized = -torch.logsumexp(-window * beta, dim=0) / beta
    shift = window.new_tensor([0.0, np.log(len(window)) / beta])
    return unnormalized + shift


def _sound_max(window, beta):
    """Sound soft-max of [n, 2] bounds: normalized lower, un-normalized upper."""
    unnormalized = torch.logsumexp(window * beta, dim=0) / beta
    shift = window.new_tensor([np.log(len(window)) / beta, 0.0])
    return unnormalized - shift


@pytest.mark.parametrize(
    "operator,smooth_reduce", [(Always, _sound_min), (Eventually, _sound_max)]
)
def test_smooth_always_and_eventually_match_their_endpointwise_surrogates(
    operator, smooth_reduce
):
    rows = [
        {"p": (0.2, 0.7)},
        {"p": (0.5, 0.8)},
        {"p": (0.1, 0.9)},
        {"p": (0.6, 0.95)},
    ]
    trajectory = supplied(*rows)
    pointwise = Predicate("p")(trajectory)[0]
    beta = 3.0

    got = operator(Predicate("p"), interval=[0, 2])(trajectory, beta=beta)[0]
    expected = torch.stack(
        [smooth_reduce(pointwise[t : t + 3], beta) for t in range(2)]
    )

    torch.testing.assert_close(got, expected)


def test_nested_temporal_lookahead_composes():
    values = [0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.3, 0.5]
    traj = scalar(values)

    spec = Always(Eventually(Predicate("p"), interval=[0, 2]), interval=[0, 3])
    got = spec(traj)[0, :, 0].tolist()

    n = len(values) - 2 - 3
    inner = [max(values[t : t + 3]) for t in range(len(values) - 2)]
    assert len(got) == n
    np.testing.assert_allclose(got, [min(inner[t : t + 4]) for t in range(n)])


def test_nested_eventually_of_always_composes():
    values = [0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.3, 0.5]
    traj = scalar(values)

    spec = Eventually(Always(Predicate("p"), interval=[0, 1]), interval=[0, 2])
    trace = spec(traj)

    # lookahead adds: the inner window costs 1 step, the outer 2 more
    n = len(values) - 1 - 2
    assert trace.shape == (1, n, 2)
    inner = [min(values[t : t + 2]) for t in range(len(values) - 1)]
    np.testing.assert_allclose(
        trace[0, :, 0].tolist(), [max(inner[t : t + 3]) for t in range(n)]
    )


def test_suffix_windows_cover_whatever_trace_remains():
    values = [0.9, 0.2, 0.7, 0.4]
    traj = scalar(values)
    ref = [min(values[t:]) for t in range(len(values))]

    for spec in (
        Always(Predicate("p")),
        Always(
            Predicate("p"), interval=[0, np.inf]
        ),  # used to raise OverflowError
    ):
        got = spec(traj)[0, :, 0].tolist()
        assert len(got) == len(values)
        np.testing.assert_allclose(got, ref)


def test_unbounded_start_offsets_the_suffix():
    values = [0.9, 0.2, 0.7, 0.4, 0.6]
    got = Always(Predicate("p"), interval=[2, np.inf])(scalar(values))[
        0, :, 0
    ].tolist()

    assert len(got) == len(values) - 2
    np.testing.assert_allclose(
        got, [min(values[t + 2 :]) for t in range(len(values) - 2)]
    )


# --- 5. Until --------------------------------------------------------------


def until_reference(left, right, a, b):
    """[lower, upper] per origin, straight from the Until equations."""
    T = len(left)
    out = []
    for t in range(T - b):
        lowers, uppers = [], []
        for tau in range(t + a, t + b + 1):
            prefix_lower = min(lo for lo, _ in left[t : tau + 1])
            prefix_upper = min(hi for _, hi in left[t : tau + 1])
            lowers.append(max(0.0, prefix_lower + right[tau][0] - 1.0))
            uppers.append(min(prefix_upper, right[tau][1]))
        out.append([max(lowers), max(uppers)])
    return out


UNTIL_LEFT = [
    (0.9, 0.95),
    (0.8, 0.9),
    (0.4, 0.7),
    (0.7, 0.8),
    (0.95, 1.0),
    (0.6, 0.75),
]
UNTIL_RIGHT = [
    (0.1, 0.2),
    (0.3, 0.6),
    (0.9, 0.95),
    (0.2, 0.4),
    (0.5, 0.7),
    (0.85, 0.9),
]


def until_trajectory():
    return supplied(
        *[{"l": lo, "r": hi} for lo, hi in zip(UNTIL_LEFT, UNTIL_RIGHT)]
    )


@pytest.mark.parametrize("interval", [[0, 0], [0, 1], [1, 2], [0, 3]])
def test_until_matches_the_reference_equations(interval):
    got = Until(Predicate("l"), Predicate("r"), interval=interval)(
        until_trajectory()
    )

    np.testing.assert_allclose(
        got[0].tolist(),
        until_reference(UNTIL_LEFT, UNTIL_RIGHT, *interval),
        atol=1e-6,
    )


def test_until_lower_bound_is_a_frechet_conjunction_of_prefix_and_witness():
    traj = supplied({"l": (0.6, 0.9), "r": (0.6, 0.8)})

    lower, upper = Until(Predicate("l"), Predicate("r"), interval=[0, 0])(
        traj
    )[0, 0]

    assert lower.item() == pytest.approx(0.6 + 0.6 - 1.0)  # 0.2, not min = 0.6
    assert upper.item() == pytest.approx(0.8)


def test_until_smooth_approaches_the_exact_reduction():
    spec = Until(Predicate("l"), Predicate("r"), interval=[0, 2])
    traj = until_trajectory()

    exact = spec(traj, beta=None)
    smooth = spec(traj, beta=400.0)

    torch.testing.assert_close(smooth, exact, atol=5e-2, rtol=0)
    assert not torch.equal(smooth, exact)  # a surrogate, not the same tensor


def test_until_smooth_matches_its_reference():
    trajectory = supplied(
        {"l": (0.7, 0.9), "r": (0.1, 0.3)},
        {"l": (0.4, 0.8), "r": (0.6, 0.75)},
        {"l": (0.8, 0.95), "r": (0.5, 0.85)},
    )
    formula = Until(Predicate("l"), Predicate("r"), interval=[0, 2])

    exact = formula(trajectory, beta=None)[0, 0]
    # witnesses: [max(0, .7+.1-1), min(.9,.3)], [max(0, .4+.6-1), min(.8,.75)],
    #            [max(0, .4+.5-1), min(.8,.85)]
    torch.testing.assert_close(exact, torch.tensor([0.0, 0.8]))

    left = Predicate("l")(trajectory)[0]
    right = Predicate("r")(trajectory)[0]
    beta = 4.0
    candidates = []
    for witness in range(3):
        prefix = _sound_min(left[: witness + 1], beta)
        candidates.append(
            torch.stack(
                (
                    prefix[0] + right[witness, 0] - 1.0,
                    -(
                        torch.logsumexp(
                            -torch.stack((prefix[1], right[witness, 1]))
                            * beta,
                            dim=0,
                        )
                        - np.log(2)
                    )
                    / beta,
                )
            )
        )
    expected_smooth = _sound_max(torch.stack(candidates), beta)

    torch.testing.assert_close(
        formula(trajectory, beta=beta)[0, 0], expected_smooth
    )


def test_until_at_zero_zero_combines_both_operands_now():
    traj = supplied({"l": (0.9, 0.9), "r": (0.4, 0.4)})

    lower, upper = Until(Predicate("l"), Predicate("r"), interval=[0, 0])(
        traj
    )[0, 0]

    assert lower.item() == pytest.approx(0.9 + 0.4 - 1.0)
    assert upper.item() == pytest.approx(0.4)


# --- 6. Finite trace -------------------------------------------------------


def test_insufficient_prediction_length_is_rejected():
    traj = scalar([0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="needs 5 steps"):
        Always(Predicate("p"), interval=[0, 4])(traj)
    with pytest.raises(ValueError, match="needs 4 steps"):
        Until(Predicate("p"), Predicate("p"), interval=[0, 3])(traj)


def test_temporal_operator_accepts_a_nested_pointwise_boolean():
    traj = supplied(
        {"a": (0.7, 0.8), "b": (0.6, 0.9)},
        {"a": (0.9, 0.95), "b": (0.8, 0.85)},
        {"a": (0.5, 0.7), "b": (0.7, 0.9)},
    )
    pointwise = And(Predicate("a"), Or(Predicate("a"), Predicate("b")))

    trace = Always(pointwise, interval=[0, 1])(traj)

    assert trace.shape == (1, 2, 2)
    torch.testing.assert_close(trace[0, 0], torch.tensor([0.4, 0.8]))


# --- 7. dtype, device and gradients ----------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved_through_a_temporal_formula(dtype):
    mean = torch.tensor([[45.0], [55.0], [60.0]], dtype=dtype)
    var = torch.full((3, 1), 4.0, dtype=dtype)
    traj = BeliefTrajectory(
        [GaussianBelief(mean[t : t + 1], var[t : t + 1]) for t in range(3)]
    )

    assert Always(GreaterThan(50.0), interval=[0, 1])(traj).dtype == dtype


def test_gradients_reach_the_mean_through_a_composed_formula():
    mean = torch.tensor([[45.0], [55.0], [60.0], [48.0]], requires_grad=True)
    var = torch.full((4, 1), 4.0)
    traj = BeliefTrajectory(
        [GaussianBelief(mean[t : t + 1], var[t : t + 1]) for t in range(4)]
    )
    spec = Always(And(GreaterThan(50.0), LessThan(70.0)), interval=[0, 1])

    spec(traj)[0, 0, 0].backward()

    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()


def test_smooth_formula_passes_a_finite_difference_gradcheck():
    def f(mean):
        var = torch.full_like(mean, 4.0)
        traj = BeliefTrajectory(
            [
                GaussianBelief(mean[t : t + 1], var[t : t + 1])
                for t in range(len(mean))
            ]
        )
        spec = Always(GreaterThan(50.0), interval=[0, 1])
        return spec(traj, beta=2.0)[0, 0, 0]

    # well separated from ties, so min/logsumexp is smooth and nonsingular
    mean = torch.tensor(
        [[47.0], [53.0], [58.0]], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(f, (mean,), eps=1e-6, atol=1e-4)


def test_supplied_bounds_keep_their_graph():
    p = torch.tensor([[0.3, 0.7]], requires_grad=True)
    traj = BeliefTrajectory([ProbabilityBelief({"p": p})] * 3)

    Always(Predicate("p"), interval=[0, 1])(traj)[0, 0, 0].backward()

    assert p.grad is not None and p.grad.abs().sum() > 0


# --- 8. Differentiable control feeding a formula ---------------------------


def test_control_to_mean_to_formula_is_differentiable():
    u = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
    mean = torch.cumsum(u, dim=0).unsqueeze(-1)  # [T, 1]
    var = torch.full_like(mean, 0.25)
    traj = BeliefTrajectory(
        [
            GaussianBelief(mean[t : t + 1], var[t : t + 1])
            for t in range(len(u))
        ]
    )
    spec = Eventually(GreaterThan(2.5), interval=[0, 2])

    spec(traj)[0, 0, 0].backward()

    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    assert u.grad.abs().sum() > 0


# --- 9. Supplied probability-bound trajectories -----------------------------


def test_supplied_bounds_factory_keys_every_step_by_the_predicate_event():
    predicate = GreaterThan(50.0)
    bounds = [[0.2, 0.4], [0.6, 0.8], [0.5, 0.9]]

    traj = create_probability_belief_trajectory(predicate, bounds)

    assert len(traj) == 3
    assert set(traj[0].bounds) == {predicate.name}
    torch.testing.assert_close(
        predicate(traj)[0], torch.tensor(bounds, dtype=torch.float32)
    )


@pytest.mark.parametrize(
    "bounds,message",
    [
        ([0.2, 0.4], r"\[T, 2\]"),
        ([[0.2, 0.4, 0.6]], r"\[T, 2\]"),
        (np.zeros((0, 2)), "at least one step"),
        ([[float("nan"), 0.4]], "malformed"),
        ([[0.5, 1.5]], "malformed"),
        ([[0.8, 0.3]], "malformed"),
        ([[-0.2, 0.4]], "malformed"),
    ],
)
def test_supplied_bounds_factory_rejects_malformed_input(bounds, message):
    with pytest.raises(ValueError, match=message):
        create_probability_belief_trajectory(GreaterThan(50.0), bounds)


def test_supplied_bounds_factory_preserves_dtype_device_and_gradients():
    predicate = GreaterThan(50.0)
    bounds = torch.tensor(
        [[0.2, 0.4], [0.6, 0.8], [0.5, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )

    traj = create_probability_belief_trajectory(predicate, bounds)
    temporal = Always(predicate, interval=[0, 1])(traj, beta=None)

    assert temporal.dtype == torch.float64
    assert temporal.device == bounds.device
    temporal.sum().backward()
    assert bounds.grad is not None
    assert torch.isfinite(bounds.grad).all()
    assert bounds.grad.abs().sum() > 0


RECT = ([4.0, 6.0], [3.0, 5.0])


def evaluate(rectangle, mean, covariance, beta=None):
    """Probability interval [B, 2] of a rectangle event under one belief."""
    belief = GaussianBelief(mean, covariance)
    return rectangle(BeliefTrajectory([belief]), beta=beta)[:, 0]


# --- Hard semantics ---------------------------------------------------------


@pytest.mark.parametrize("point", [[5.0, 4.0], [4.0, 3.0], [6.0, 5.0]])
def test_deterministic_belief_inside_the_rectangle_is_certain(point):
    """Zero variance, mean inside (corners included: the rectangle is closed)."""
    bounds = evaluate(
        InsideRectangle(*RECT), torch.tensor([point]), torch.zeros(1, 2)
    )
    assert bounds.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("point", [[0.0, 0.0], [5.0, 5.01], [3.99, 4.0]])
def test_deterministic_belief_outside_the_rectangle_is_impossible(point):
    bounds = evaluate(
        InsideRectangle(*RECT), torch.tensor([point]), torch.zeros(1, 2)
    )
    assert bounds.tolist() == [[0.0, 0.0]]


def test_outside_is_exactly_the_complement_of_inside():
    torch.manual_seed(0)
    mean = torch.randn(32, 2) * 2 + torch.tensor([5.0, 4.0])
    covariance = torch.diag_embed(torch.rand(32, 2) + 0.05)

    inside = evaluate(InsideRectangle(*RECT), mean, covariance)
    outside = evaluate(OutsideRectangle(*RECT), mean, covariance)

    torch.testing.assert_close(outside[:, 0], 1.0 - inside[:, 1])
    torch.testing.assert_close(outside[:, 1], 1.0 - inside[:, 0])


def test_bounds_stay_valid_under_strong_correlation():
    """Frechet needs no independence, so a full covariance must not break the enclosure."""
    mean = torch.tensor([[5.0, 4.0]], dtype=torch.float64)
    for rho in (-0.95, -0.5, 0.0, 0.5, 0.95):
        covariance = torch.tensor(
            [[[0.6, rho * 0.6], [rho * 0.6, 0.6]]], dtype=torch.float64
        )
        for rectangle in (InsideRectangle(*RECT), OutsideRectangle(*RECT)):
            lower, upper = evaluate(rectangle, mean, covariance)[0].tolist()
            assert 0.0 <= lower <= upper <= 1.0, f"rho={rho} {rectangle}"


def test_monte_carlo_probability_lies_inside_the_analytical_interval():
    mean = torch.tensor([[5.1, 3.9]], dtype=torch.float64)
    covariance = torch.tensor(
        [[[0.5, -0.35], [-0.35, 0.45]]], dtype=torch.float64
    )
    lower, upper = evaluate(InsideRectangle(*RECT), mean, covariance)[
        0
    ].tolist()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        samples = torch.distributions.MultivariateNormal(
            mean[0], covariance[0]
        ).sample((200_000,))
    (x_lo, x_hi), (y_lo, y_hi) = RECT
    hit = (
        (
            (samples[:, 0] >= x_lo)
            & (samples[:, 0] <= x_hi)
            & (samples[:, 1] >= y_lo)
            & (samples[:, 1] <= y_hi)
        )
        .double()
        .mean()
        .item()
    )

    tolerance = 3.0 * (hit * (1 - hit) / 200_000) ** 0.5
    assert lower - tolerance <= hit <= upper + tolerance, (
        f"{hit} outside [{lower}, {upper}]"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_range": [6.0, 4.0], "y_range": [3.0, 5.0]},  # reversed
        {"x_range": [4.0, 4.0], "y_range": [3.0, 5.0]},  # degenerate
        {"x_range": [4.0, 6.0], "y_range": [3.0, 5.0], "dims": (1, 1)},
        {"x_range": [4.0, 6.0], "y_range": [3.0, 5.0], "dims": (0, -1)},
    ],
)
def test_invalid_geometry_raises_clearly(kwargs):
    with pytest.raises(ValueError):
        InsideRectangle(**kwargs)


@pytest.mark.parametrize("args", [(6.0, 4.0), (4.0, 4.0)])
def test_axis_interval_rejects_invalid_bounds(args):
    with pytest.raises(ValueError, match="min < max"):
        AxisInterval(*args, dim=0)


def test_batch_shape_and_dtype_are_preserved():
    mean = torch.randn(9, 2, dtype=torch.float64) + torch.tensor(
        [5.0, 4.0], dtype=torch.float64
    )
    covariance = torch.full((9, 2), 0.3, dtype=torch.float64)
    bounds = evaluate(InsideRectangle(*RECT), mean, covariance)
    assert bounds.shape == (9, 2) and bounds.dtype is torch.float64


def test_rectangle_over_non_default_state_dimensions():
    """A 4-state double integrator measures position on dims (0, 1) only."""
    mean = torch.tensor([[5.0, 4.0, 9.9, -9.9]])
    bounds = evaluate(
        InsideRectangle(*RECT, dims=(0, 1)), mean, torch.zeros(1, 4)
    )
    assert bounds.tolist() == [[1.0, 1.0]]


# --- Gradients --------------------------------------------------------------


def _goal_gradient(
    mean_value, beta, goal=([8.5, 9.5], [8.5, 9.5]), sigma_sq=0.04
):
    mean = torch.tensor([[mean_value, mean_value]], requires_grad=True)
    lower = evaluate(
        InsideRectangle(*goal),
        mean,
        torch.tensor([[sigma_sq, sigma_sq]]),
        beta,
    )[0, 0]
    lower.backward()
    return lower.item(), mean.grad.abs().sum().item()


def test_smooth_inside_goal_score_has_a_finite_gradient():
    value, gradient = _goal_gradient(8.4, beta=1.0)
    assert torch.isfinite(torch.tensor([value, gradient])).all()
    assert gradient > 0.0


def test_smooth_goal_gradient_is_an_approach_phase_fix():
    """Softplus revives the gradient near the goal -- and only near it.

    Within roughly 3 sigma the hard clamp is what kills the gradient, and smoothing fixes it.
    Further out the Gaussian CDF itself underflows to exactly zero in float32, so no operator
    smoothing can help; the goal-directed initialisation is what covers that regime. This test
    documents both halves so nobody mistakes the far-field zero for a regression.
    """
    _, near_hard = _goal_gradient(8.4, beta=None)
    _, near_smooth = _goal_gradient(8.4, beta=1.0)
    assert near_hard == 0.0, "hard clamp gives no gradient on approach"
    assert near_smooth > 1e-3, "softplus must revive it"

    _, far_smooth = _goal_gradient(1.0, beta=1.0)
    assert far_smooth == 0.0, (
        "far field is lost to CDF underflow, not to the clamp"
    )


def test_outside_rectangle_gradient_pushes_away_from_the_block():
    """Safety gradient must point out of the obstacle along the escaping axis."""
    block = ([4.0, 6.0], [0.0, 4.0])
    for mean_value, expect_x in [([6.2, 2.0], True), ([5.0, 4.3], False)]:
        mean = torch.tensor([mean_value], requires_grad=True)
        lower = evaluate(
            OutsideRectangle(*block), mean, torch.tensor([[0.09, 0.09]]), 20.0
        )[0, 0]
        lower.backward()
        gx, gy = mean.grad[0].tolist()
        assert torch.isfinite(mean.grad).all()
        if expect_x:
            # Beside the block: moving further out in x raises the safety probability.
            assert gx > 0.1, f"no lateral escape gradient at {mean_value}"
        else:
            # Centred in x: p_x sits on its plateau, so only y can help. Not a defect.
            assert abs(gx) < 1e-6 and gy > 0.1


def test_hard_evaluation_carries_no_gradient_into_the_clamped_region():
    mean = torch.tensor([[1.0, 1.0]], requires_grad=True)
    lower = evaluate(
        InsideRectangle([8.5, 9.5], [8.5, 9.5]),
        mean,
        torch.tensor([[0.04, 0.04]]),
    )[0, 0]
    assert lower.item() == 0.0
    lower.backward()
    assert mean.grad.abs().sum().item() == 0.0


@pytest.mark.parametrize("full", [False, True])
def test_half_space_uses_projected_gaussian_variance(full):
    mean = torch.tensor([[0.2, -0.4, 0.6]], dtype=torch.float64)
    covariance = torch.tensor(
        [[0.4, 0.1, -0.05], [0.1, 0.6, 0.2], [-0.05, 0.2, 0.8]],
        dtype=torch.float64,
    )
    cov = covariance.unsqueeze(0) if full else covariance.diag().unsqueeze(0)
    normal, cutoff = np.array([1.0, -2.0, 0.5]), 0.3
    projected = covariance.numpy() if full else np.diag(covariance.diag())
    expected = norm.cdf(
        (cutoff - mean.numpy() @ normal) / np.sqrt(normal @ projected @ normal)
    )
    bounds = GaussianBelief(mean, cov).probability_bounds(
        HalfSpace(normal, cutoff)
    )
    np.testing.assert_allclose(bounds.numpy(), [[expected[0], expected[0]]])


def test_half_space_gradient_reaches_gaussian_parameters():
    mean = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    factor = torch.tensor(
        [[[0.8, 0.0], [0.2, 0.6]]], dtype=torch.float64, requires_grad=True
    )
    event = HalfSpace([1, -2], 0.7)
    bounds = GaussianBelief(mean, factor @ factor.transpose(-1, -2))
    bounds.probability_bounds(event).sum().backward()
    assert mean.grad.abs().sum() > 0 and factor.grad.abs().sum() > 0
