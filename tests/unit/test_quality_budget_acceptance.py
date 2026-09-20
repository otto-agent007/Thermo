"""Guard simultaneous acceptance against optimistic thresholds and budget selection."""

import itertools

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import binom

from thermo_lab.quality_budget_acceptance import (
    binomial_intervals,
    budget_bracket,
    classify_bounds,
    classify_cell,
    classify_method,
    compare_budgets,
    occupancy_loss_bounds,
)


@pytest.mark.parametrize("n", [1, 2, 8, 32, 32768])
def test_intervals_match_independent_binomial_tail_inversion(n):
    alpha = 0.05 / 1560
    counts = sorted({0, 1, n // 2, n - 1, n})
    intervals = binomial_intervals(counts, n)
    for x, (lo, hi) in zip(counts, intervals, strict=True):
        lower = (
            0
            if x == 0
            else brentq(lambda p, x=x, n=n: binom.sf(x - 1, n, p) - alpha / 2, 0, 1, xtol=1e-15)
        )
        upper = (
            1
            if x == n
            else brentq(lambda p, x=x, n=n: binom.cdf(x, n, p) - alpha / 2, 0, 1, xtol=1e-15)
        )
        np.testing.assert_allclose((lo, hi), (lower, upper), atol=2e-12, rtol=0)
        assert 0 <= lo <= x / n <= hi <= 1


@pytest.mark.parametrize(
    "counts,n",
    [
        ([True], 8),
        ([1.0], 8),
        ([-1], 8),
        ([9], 8),
        ([1], 0),
        ([1], True),
        ([1], 8.0),
        ([], 8),
        ([[1]], 8),
        ([float("nan")], 8),
    ],
)
def test_malformed_counts_are_integrity_errors(counts, n):
    with pytest.raises(ValueError):
        binomial_intervals(counts, n)


def test_rectangle_bounds_cover_every_corner_and_interior_minimum():
    intervals = [[0.1, 0.3], [0.4, 0.8], [0.0, 0.2]]
    target = [0.2, 0.1, 0.9]
    lo, hi = occupancy_loss_bounds(intervals, target)
    assert lo == pytest.approx(0.58)
    assert hi == pytest.approx(1.31)
    for point in itertools.product(*intervals):
        value = sum((a - b) ** 2 for a, b in zip(point, target, strict=True))
        assert lo <= value <= hi
    assert sum((a - b) ** 2 for a, b in zip([0.2, 0.4, 0.2], target, strict=True)) == lo


@pytest.mark.parametrize(
    "intervals,target",
    [
        ([[0.2, 0.1]], [0.1]),
        ([[0.0, float("nan")]], [0.1]),
        ([[0.0, 1.0]], [1.1]),
        ([[0.0, 1.0]], [0.1, 0.2]),
        ([], []),
        ([[-0.1, 0.3]], [0.2]),
    ],
)
def test_rectangle_rejects_invalid_bounds(intervals, target):
    with pytest.raises(ValueError):
        occupancy_loss_bounds(intervals, target)


def test_literal_thresholds_and_unresolved_intervals():
    args = dict(
        loss=(0.0, 0.0625), leakage=(0.0, 0.05), survival=0.95, hop_mae=0.01, asymmetry_mae=0.01
    )
    assert classify_bounds(**args) == "pass"
    for key in ("survival", "hop_mae", "asymmetry_mae"):
        direction = 0 if key == "survival" else np.inf
        assert classify_bounds(**{**args, key: float(np.nextafter(args[key], direction))}) == "fail"
    assert classify_bounds(**{**args, "loss": (0.0, np.nextafter(0.0625, np.inf))}) == "unresolved"
    assert classify_bounds(**{**args, "leakage": (0.0, np.nextafter(0.05, np.inf))}) == "unresolved"
    assert classify_bounds(**{**args, "loss": (np.nextafter(0.0625, np.inf), 0.2)}) == "fail"
    # Exact survival's leakage implication must not override the sampled gate.
    assert classify_bounds(**{**args, "survival": 1.0, "leakage": (0.0, 0.06)}) == "unresolved"
    with pytest.raises(ValueError):
        classify_bounds(**{**args, "survival": float("nan")})


def test_numpy_scalars_are_compared_as_float64_without_threshold_rounding():
    args = dict(
        loss=(0.0, 0.0625), leakage=(0.0, 0.05), survival=0.95, hop_mae=0.01, asymmetry_mae=0.01
    )
    assert float(np.float32(0.95)) < 0.95
    assert classify_bounds(**{**args, "survival": np.float32(0.95)}) == "fail"
    assert float(np.float32(0.05)) > 0.05
    assert classify_bounds(**{**args, "leakage": (0.0, np.float32(0.05))}) == "unresolved"
    with pytest.raises(ValueError):
        classify_bounds(**{**args, "leakage": (0.050000001, np.float32(0.05))})


def test_cell_counts_are_checked_and_derived_not_trusted():
    target = [1.0] + [0.0] * 24
    counts = [32768] + [0] * 24
    histogram = [0, 32768] + [0] * 24
    args = dict(
        occupancy_counts=counts,
        particle_histogram=histogram,
        target=target,
        survival=1.0,
        hop_mae=0.0,
        asymmetry_mae=0.0,
    )
    result = classify_cell(**args)
    assert result["status"] == "pass" and result["leakage_count"] == 0
    with pytest.raises(ValueError):
        classify_cell(**{**args, "occupancy_counts": [1] + [0] * 24})
    with pytest.raises(ValueError):
        classify_cell(**{**args, "particle_histogram": [0, 32767] + [0] * 24})
    # Total ones agree but cannot be realized: two full columns vs one row with 2 ones.
    impossible = [32767, 0, 1] + [0] * 23
    with pytest.raises(ValueError):
        classify_cell(
            **{**args, "occupancy_counts": [2] + [0] * 24, "particle_histogram": impossible}
        )


def test_method_requires_exactly_three_seed_identities_and_all_to_pass():
    assert classify_method({0: "pass", 1: "pass", 2: "unresolved"}) == "unresolved"
    assert classify_method({0: "pass", 1: "unresolved", 2: "fail"}) == "fail"
    assert classify_method({0: "pass", 1: "pass", 2: "pass"}) == "pass"
    for invalid in (
        {0: "pass", 1: "pass"},
        {0: "pass", 1: "pass", 2: "unknown"},
        {False: "pass", 1: "pass", 2: "pass"},
    ):
        with pytest.raises(ValueError):
            classify_method(invalid)


def test_bracket_retains_cheaper_uncertainty_and_nonmonotone_cells():
    result = budget_bracket(["fail", "unresolved", "fail", "pass", "fail", "pass"])
    assert result == {"possible": 2, "certified": 8, "status": "bracketed"}
    assert budget_bracket(["fail"] * 6)["status"] == "quality_failure"
    assert budget_bracket(["unresolved"] * 6) == {
        "possible": 1,
        "certified": None,
        "status": "inconclusive_quality",
    }
    assert budget_bracket(["fail", "pass", "fail", "fail", "fail", "fail"]) == {
        "possible": 2,
        "certified": 2,
        "status": "resolved",
    }
    with pytest.raises(ValueError):
        budget_bracket(["pass"])


def test_savings_need_both_certified_and_strict_disjoint_brackets():
    finite = ["fail", "unresolved", "pass", "pass", "fail", "fail"]
    eq = ["fail", "fail", "fail", "unresolved", "pass", "fail"]
    result = compare_budgets(finite, eq)
    assert result["decision"] == "lower_budget_demonstrated"
    assert result["sweep_ratio_bounds"] == [2.0, 8.0]
    assert result["resolved_ratio"] is None
    eq[2] = "unresolved"
    overlapping = compare_budgets(finite, eq)
    assert overlapping["decision"] == "inconclusive"
    assert overlapping["sweep_ratio_bounds"] == [1.0, 8.0]
    for other in (["fail"] * 6, ["unresolved"] * 6):
        assert compare_budgets(finite, other)["sweep_ratio_bounds"] is None
    resolved = ["fail", "pass", "fail", "fail", "fail", "fail"]
    result = compare_budgets(resolved, resolved)
    assert result["decision"] == "equal_resolved_minima"
    assert result["resolved_ratio"] == 1.0
    assert result["sweep_ratio_bounds"] == [1.0, 1.0]


def test_all_nonmonotone_patterns_bracket_all_compatible_minima():
    horizons = (1, 2, 4, 8, 16, 30)
    for statuses in itertools.product(("pass", "fail", "unresolved"), repeat=6):
        bracket = budget_bracket(statuses)
        unknown = [i for i, s in enumerate(statuses) if s == "unresolved"]
        for outcomes in itertools.product((False, True), repeat=len(unknown)):
            truth = [s == "pass" for s in statuses]
            for index, value in zip(unknown, outcomes, strict=True):
                truth[index] = value
            actual = next((k for k, passes in zip(horizons, truth, strict=True) if passes), np.inf)
            assert (bracket["possible"] or np.inf) <= actual <= (bracket["certified"] or np.inf)
