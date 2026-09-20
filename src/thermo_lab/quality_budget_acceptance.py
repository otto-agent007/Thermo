"""Conservative M4G quality decisions; malformed evidence never becomes a failure."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.special import betaincinv

from thermo_lab.quality_budget_protocol import HORIZONS, QualityBudgetProtocol

STATUSES = ("pass", "fail", "unresolved")
_PROTOCOL = QualityBudgetProtocol()


def _counts(values, n):
    if type(n) is not int or not 1 <= n <= 32768:
        raise ValueError("trial count must be an integer in [1,32768]")
    try:
        raw = list(values)
    except TypeError as error:
        raise ValueError("counts must be a nonempty integer vector") from error
    if not raw or any(
        not isinstance(x, Integral) or isinstance(x, (bool, np.bool_)) or not 0 <= x <= n
        for x in raw
    ):
        raise ValueError("counts must be integers in [0,N], not booleans")
    return np.asarray(raw, dtype=np.int64)


def binomial_intervals(counts, n: int) -> np.ndarray:
    """Fixed two-sided CP intervals, including oracle cells in the Bonferroni family."""
    values = _counts(counts, n)
    alpha = _PROTOCOL.family_alpha / _PROTOCOL.interval_count
    lower, upper = np.zeros(len(values)), np.ones(len(values))
    positive, below_n = values > 0, values < n
    lower[positive] = betaincinv(values[positive], n - values[positive] + 1, alpha / 2)
    upper[below_n] = betaincinv(values[below_n] + 1, n - values[below_n], 1 - alpha / 2)
    result = np.column_stack((lower, upper))
    if (
        not np.all(np.isfinite(result))
        or np.any(lower > values / n)
        or np.any(upper < values / n)
        or np.any(lower < 0)
        or np.any(upper > 1)
    ):
        raise ValueError("binomial interval inversion failed")
    return result


def occupancy_loss_bounds(intervals, target) -> tuple[float, float]:
    rectangle = np.asarray(intervals, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)
    if (
        q.ndim != 1
        or not 1 <= len(q) <= 25
        or rectangle.shape != (len(q), 2)
        or not np.all(np.isfinite(rectangle))
        or not np.all(np.isfinite(q))
        or np.any(q < 0)
        or np.any(q > 1)
        or np.any(rectangle < 0)
        or np.any(rectangle > 1)
        or np.any(rectangle[:, 0] > rectangle[:, 1])
    ):
        raise ValueError("occupancy rectangle and target must be finite probabilities")
    lower, upper = rectangle.T
    distance = np.maximum(np.maximum(lower - q, q - upper), 0.0)
    return (
        float(np.sum(distance**2)),
        float(np.sum(np.maximum((lower - q) ** 2, (upper - q) ** 2))),
    )


def _finite_scalar(value, maximum, name):
    if (
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite in [0,{maximum}]")


def classify_bounds(*, loss, leakage, survival, hop_mae, asymmetry_mae) -> str:
    for name, bounds, maximum in (("loss", loss, 25), ("leakage", leakage, 1)):
        if len(bounds) != 2:
            raise ValueError(f"{name} requires two bounds")
        for value in bounds:
            _finite_scalar(value, maximum, name)
        if float(bounds[0]) > float(bounds[1]):
            raise ValueError(f"{name} bounds are reversed")
    for name, value, maximum in (
        ("survival", survival, 1),
        ("hop MAE", hop_mae, 1),
        ("asymmetry MAE", asymmetry_mae, 2),
    ):
        _finite_scalar(value, maximum, name)
    # NumPy weak scalar promotion can round a Python threshold down to float32.
    # Compare the actual supplied values as float64, never the rounded threshold.
    loss, leakage = tuple(map(float, loss)), tuple(map(float, leakage))
    survival, hop_mae, asymmetry_mae = map(float, (survival, hop_mae, asymmetry_mae))
    exact_pass = (
        survival >= _PROTOCOL.survival_min
        and hop_mae <= _PROTOCOL.hop_mae_max
        and asymmetry_mae <= _PROTOCOL.asymmetry_mae_max
    )
    if not exact_pass or loss[0] > _PROTOCOL.loss_max or leakage[0] > _PROTOCOL.leakage_max:
        return "fail"
    if loss[1] <= _PROTOCOL.loss_max and leakage[1] <= _PROTOCOL.leakage_max:
        return "pass"
    return "unresolved"


def classify_cell(
    *, occupancy_counts, particle_histogram, target, survival, hop_mae, asymmetry_mae
) -> dict:
    """Derive sampled bounds from production counts; exact metrics require upstream replay.

    This decision utility is not an experiment-record validator. A future full
    runner must reconstruct exact metrics and target from authenticated fits.
    """
    n = _PROTOCOL.batch_size
    counts, hist = _counts(occupancy_counts, n), _counts(particle_histogram, n)
    if len(counts) != 25 or len(hist) != 26 or int(hist.sum()) != n:
        raise ValueError("M4G requires 25 site counts and a complete 26-bin histogram")
    if int(counts.sum()) != sum(i * int(c) for i, c in enumerate(hist)):
        raise ValueError("occupancy and particle totals disagree")
    # Gale-Ryser inequalities: a binary matrix with these row/column margins exists.
    descending = sorted((int(c) for c in counts), reverse=True)
    for k in range(1, 26):
        if sum(descending[:k]) > sum(min(k, i) * int(c) for i, c in enumerate(hist)):
            raise ValueError("terminal margins cannot describe binary trajectories")
    if np.asarray(target).shape != (25,):
        raise ValueError("M4G requires a 25-site target")
    leakage_count = n - int(hist[1])
    intervals = binomial_intervals([*map(int, counts), leakage_count], n)
    loss = occupancy_loss_bounds(intervals[:25], target)
    leakage = tuple(float(x) for x in intervals[25])
    status = classify_bounds(
        loss=loss, leakage=leakage, survival=survival, hop_mae=hop_mae, asymmetry_mae=asymmetry_mae
    )
    return {
        "occupancy_intervals": intervals[:25].tolist(),
        "loss_bounds": list(loss),
        "leakage_count": leakage_count,
        "leakage_interval": list(leakage),
        "status": status,
    }


def classify_method(seed_statuses: dict[int, str]) -> str:
    if (
        set(seed_statuses) != {0, 1, 2}
        or any(type(k) is not int for k in seed_statuses)
        or any(s not in STATUSES for s in seed_statuses.values())
    ):
        raise ValueError("method classification requires exactly seeds 0,1,2 and valid statuses")
    if "fail" in seed_statuses.values():
        return "fail"
    return "pass" if all(s == "pass" for s in seed_statuses.values()) else "unresolved"


def budget_bracket(statuses) -> dict:
    statuses = tuple(statuses)
    if len(statuses) != 6 or any(s not in STATUSES for s in statuses):
        raise ValueError("budget bracket requires all six statuses in ascending K order")
    possible = next((k for k, s in zip(HORIZONS, statuses, strict=True) if s != "fail"), None)
    certified = next((k for k, s in zip(HORIZONS, statuses, strict=True) if s == "pass"), None)
    status = (
        "quality_failure"
        if possible is None
        else "inconclusive_quality"
        if certified is None
        else "resolved"
        if possible == certified
        else "bracketed"
    )
    # Null means +infinity; never serialize nonstandard Infinity in evidence JSON.
    return {"possible": possible, "certified": certified, "status": status}


def compare_budgets(finite_statuses, equilibrium_statuses) -> dict:
    finite, eq = budget_bracket(finite_statuses), budget_bracket(equilibrium_statuses)
    result = {
        "finite": finite,
        "equilibrium": eq,
        "decision": "inconclusive",
        "sweep_ratio_bounds": None,
        "resolved_ratio": None,
    }
    if finite["certified"] is None or eq["certified"] is None:
        if finite["status"] == eq["status"] == "quality_failure":
            result["decision"] = "both_quality_failure"
        return result
    resolved = finite["status"] == eq["status"] == "resolved"
    result["sweep_ratio_bounds"] = [
        eq["possible"] / finite["certified"],
        eq["certified"] / finite["possible"],
    ]
    if finite["certified"] < eq["possible"]:
        result["decision"] = "lower_budget_demonstrated"
    elif resolved:
        result["decision"] = (
            "equal_resolved_minima" if finite["possible"] == eq["possible"] else "no_lower_budget"
        )
    if resolved:
        result["resolved_ratio"] = eq["possible"] / finite["possible"]
    return result
