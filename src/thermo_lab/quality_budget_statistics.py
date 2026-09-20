"""Deterministic numerical validation of the predeclared M4G acceptance policy."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

from thermo_lab.hashing import canonical_sha256
from thermo_lab.quality_budget_acceptance import (
    binomial_intervals,
    budget_bracket,
    classify_bounds,
    classify_cell,
    classify_method,
    compare_budgets,
    occupancy_loss_bounds,
)
from thermo_lab.quality_budget_protocol import HORIZONS


def validation_contract():
    return {
        "identity_version": "quality_budget_statistics.v1",
        "n_grid": [1, 2, 8, 32, 32768],
        "p_grid": [0.0, 0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 1.0],
        "alpha": 0.05 / 1560,
        "coverage_numeric_atol": 1e-12,
        "inversion_numeric_atol": 2e-12,
        "independent_inversion": "brentq binomial sf/cdf, xtol=1e-15",
        "joint_fixture": "all 4096 ordered size-four batches from eight three-bit states",
        "joint_atom_probabilities": [0.05, 0.1, 0.15, 0.2, 0.1, 0.1, 0.1, 0.2],
        "joint_target": [0.2, 0.4, 0.7],
        "correlated_family": "1560 identical Bernoulli counts; no independence factorization",
        "brackets": "729 ternary patterns; every compatible truth; all distinct bracket pairs",
        "scientific_threshold_epsilon": 0.0,
    }


def _require(condition, message):
    if not condition:
        raise ValueError(f"M4G statistical preflight failed: {message}")


def _coverage_checks():
    contract = validation_contract()
    alpha = contract["alpha"]
    rows, inversions, intervals_by_n = [], [], []
    for n in contract["n_grid"]:
        counts = np.arange(n + 1)
        intervals = binomial_intervals(counts, n)
        lower, upper = intervals.T
        _require(np.all(np.diff(lower) >= 0) and np.all(np.diff(upper) >= 0), "count monotonicity")
        # Nesting against independently inverted ordinary 95% intervals.
        probes = {0, 1, n // 2, n - 1, n}
        # Both sides of the production leakage pass/fail boundary.
        for bound in (lower, upper):
            crossing = int(np.searchsorted(bound, 0.05))
            probes.update(x for x in (crossing - 1, crossing, crossing + 1) if 0 <= x <= n)
        for x in sorted(probes):
            lo = (
                0.0
                if x == 0
                else brentq(lambda p, x=x, n=n: binom.sf(x - 1, n, p) - alpha / 2, 0, 1, xtol=1e-15)
            )
            hi = (
                1.0
                if x == n
                else brentq(lambda p, x=x, n=n: binom.cdf(x, n, p) - alpha / 2, 0, 1, xtol=1e-15)
            )
            error = float(np.max(np.abs(intervals[x] - (lo, hi))))
            _require(error <= contract["inversion_numeric_atol"], "binomial tail inversion")
            nominal_lo = (
                0.0
                if x == 0
                else brentq(lambda p, x=x, n=n: binom.sf(x - 1, n, p) - 0.025, 0, 1, xtol=1e-15)
            )
            nominal_hi = (
                1.0
                if x == n
                else brentq(lambda p, x=x, n=n: binom.cdf(x, n, p) - 0.025, 0, 1, xtol=1e-15)
            )
            _require(
                intervals[x, 0] <= nominal_lo and intervals[x, 1] >= nominal_hi,
                "confidence-level nesting",
            )
            inversions.append(
                {
                    "n": n,
                    "count": x,
                    "interval": intervals[x].tolist(),
                    "independent": [float(lo), float(hi)],
                    "maximum_error": error,
                }
            )
        for p in contract["p_grid"]:
            covered = (lower <= p) & (p <= upper)
            coverage = float(np.sum(binom.pmf(counts, n, p)[covered]))
            _require(coverage >= 1 - alpha - contract["coverage_numeric_atol"], "marginal coverage")
            rows.append(
                {"n": n, "p": p, "coverage": coverage, "covered_count_values": int(covered.sum())}
            )
        intervals_by_n.append({"n": n, "interval_table_digest": canonical_sha256(intervals)})
    return rows, inversions, intervals_by_n


def _joint_fixture():
    atoms = np.asarray(list(itertools.product((0, 1), repeat=3)))
    weights = np.asarray(validation_contract()["joint_atom_probabilities"])
    population = weights @ atoms
    target = np.asarray(validation_contract()["joint_target"])
    truth = float(np.sum((population - target) ** 2))
    intervals = binomial_intervals(list(range(5)), 4)
    coverage, loss_coverage = 0.0, 0.0
    for indices in itertools.product(range(8), repeat=4):
        counts = atoms[list(indices)].sum(axis=0)
        rectangle = intervals[counts]
        lo, hi = occupancy_loss_bounds(rectangle, target)
        probability = float(np.prod(weights[list(indices)]))
        covered = np.all((rectangle[:, 0] <= population) & (population <= rectangle[:, 1]))
        if covered:
            _require(lo <= truth <= hi, "rectangle bound with dependent site observations")
            coverage += probability
        if lo <= truth <= hi:
            loss_coverage += probability
    _require(coverage >= 0.95 - 1e-12, "enumerated joint fixture coverage")
    return {
        "ordered_batches": 4096,
        "population": population.tolist(),
        "true_loss": truth,
        "rectangle_coverage": coverage,
        "loss_bound_coverage": loss_coverage,
    }


def _decision_checks():
    representatives = {}
    for statuses in itertools.product(("pass", "fail", "unresolved"), repeat=6):
        bracket = budget_bracket(statuses)
        representatives[(bracket["possible"], bracket["certified"])] = statuses
        unknown = [i for i, s in enumerate(statuses) if s == "unresolved"]
        for values in itertools.product((False, True), repeat=len(unknown)):
            truth = [s == "pass" for s in statuses]
            for index, value in zip(unknown, values, strict=True):
                truth[index] = value
            actual = next((k for k, ok in zip(HORIZONS, truth, strict=True) if ok), np.inf)
            _require(
                (bracket["possible"] or np.inf) <= actual <= (bracket["certified"] or np.inf),
                "nonmonotone budget bracket",
            )
    for finite, eq in itertools.product(representatives.values(), repeat=2):
        result = compare_budgets(finite, eq)
        f, e = result["finite"], result["equilibrium"]
        expected = (
            f["certified"] is not None
            and e["certified"] is not None
            and f["certified"] < e["possible"]
        )
        _require(
            (result["decision"] == "lower_budget_demonstrated") == expected,
            "lower-budget claim condition",
        )
        if f["certified"] is not None and e["certified"] is not None:
            _require(
                result["sweep_ratio_bounds"]
                == [e["possible"] / f["certified"], e["certified"] / f["possible"]],
                "ratio bounds",
            )
    for statuses in itertools.product(("pass", "fail", "unresolved"), repeat=3):
        expected = (
            "fail"
            if "fail" in statuses
            else ("pass" if all(s == "pass" for s in statuses) else "unresolved")
        )
        _require(classify_method(dict(enumerate(statuses))) == expected, "three-seed rule")
    base = dict(
        loss=(0.0, 0.0625), leakage=(0.0, 0.05), survival=0.95, hop_mae=0.01, asymmetry_mae=0.01
    )
    _require(classify_bounds(**base) == "pass", "inclusive equality")
    for key in ("survival", "hop_mae", "asymmetry_mae"):
        value = float(np.nextafter(base[key], 0 if key == "survival" else np.inf))
        _require(classify_bounds(**{**base, key: value}) == "fail", "exact nextafter threshold")
    for key, threshold in (("loss", 0.0625), ("leakage", 0.05)):
        upper = float(np.nextafter(threshold, np.inf))
        _require(
            classify_bounds(**{**base, key: (0.0, upper)}) == "unresolved", "sample upper bound"
        )
        _require(classify_bounds(**{**base, key: (upper, upper)}) == "fail", "sample lower bound")
    return len(representatives) ** 2


def statistical_preflight():
    coverage, inversions, tables = _coverage_checks()
    joint = _joint_fixture()
    bracket_pairs = _decision_checks()
    # Identical cells have exactly the marginal coverage, not its 1560th power.
    # Feed complete synthetic binary margins through the production decision path.
    correlated_cells = []
    for count, expected in ((32768, "pass"), (27000, "fail"), (31129, "unresolved")):
        cell = classify_cell(
            occupancy_counts=[count] + [0] * 24,
            particle_histogram=[32768 - count, count] + [0] * 24,
            target=[1.0] + [0.0] * 24,
            survival=0.95,
            hop_mae=0.0,
            asymmetry_mae=0.0,
        )
        status = cell["status"]
        _require(status == expected, "correlated synthetic count-to-decision fixture")
        _require(
            classify_method({0: status, 1: status, 2: status}) == status,
            "perfectly correlated cells",
        )
        correlated_cells.append({"count": count, "decision": cell, "method_status": status})
    return {
        "coverage": coverage,
        "independent_inversions": inversions,
        "interval_tables": tables,
        "joint_fixture": joint,
        "budget_patterns": 729,
        "distinct_bracket_pairs": bracket_pairs,
        "perfectly_correlated_family": {
            "intervals": 1560,
            "minimum_joint_coverage": min(r["coverage"] for r in coverage),
            "all_three_seed_rule_checked": True,
            "synthetic_count_fixtures": correlated_cells,
        },
        "coverage_proof": "Each CP marginal covers with probability >=1-0.05/1560. "
        "The union bound covers all 1560 intervals with probability >=0.95, "
        "without cross-cell independence. On that event the rectangle encloses "
        "the population loss. Coverage is conditional on frozen fits and binomial "
        "within-cell sampling. A finite grid is a numerical check, not a uniform proof.",
    }
