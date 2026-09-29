"""Exact, study-local inner dynamics for M5b; no sampling or fitting."""

import math
import warnings
from numbers import Integral

import numpy as np
from scipy.linalg import LinAlgWarning, solve
from scipy.special import logsumexp

from thermo_lab import meta_ebm_cap_baseline as m5a

TOLERANCE = {
    "local_absolute": 1e-12,
    "local_log": 1e-8,
    "large_k_contraction": 1e-14,
    "large_k_sweep": 2e-12,
    "chain_absolute": 1e-9,
    "chain_relative": 1e-8,
    "stationary": 1e-10,
}


class NumericalIntegrityError(ValueError):
    """The prescribed float64 method cannot certify this calculation."""


def inner_kernel(parameters, structure):
    """Enumerate positive escape rates for a hidden-first/output-second sweep.

    Hidden spins are independent given x,y. Log sums keep rare transitions
    separate from the complementary probabilities which may round to one.
    """
    vector = np.asarray(parameters, dtype=np.float64)
    if vector.shape != (m5a.parameter_count(structure),) or not np.isfinite(vector).all():
        raise ValueError("invalid parameter vector")
    inputs = m5a.blanket_inputs(structure)
    J, h, A, b, beta = m5a.unpack(vector, structure)
    nh = len(beta)
    if nh > 12:
        raise ValueError("hidden enumeration exceeds the bounded M5b family")
    hidden = ((np.arange(1 << nh)[:, None] >> np.arange(nh)) & 1) * 2.0 - 1.0
    hidden_drive = hidden @ beta
    logs = np.empty((len(inputs), 2), dtype=np.float64)
    for start in range(0, len(inputs), 128):
        x = inputs[start : start + 128]
        fields = x @ A.T + b
        output = (x @ J + h)[:, None] + hidden_drive
        for column, old_y in enumerate((-1, 1)):
            conditioned = fields + old_y * beta
            log_hidden = (
                conditioned @ hidden.T
                - np.logaddexp(conditioned, -conditioned).sum(axis=1)[:, None]
            )
            # Old y=-1 transitions to +1; old y=+1 transitions to -1.
            log_output = -np.logaddexp(0.0, 2 * old_y * output)
            logs[start : start + len(x), column] = logsumexp(log_hidden + log_output, axis=1)
    log_a, log_b = logs.T
    log_rate = np.logaddexp(log_a, log_b)
    a, b = np.exp(log_a), np.exp(log_b)
    rate = np.exp(log_rate)
    if (
        not np.isfinite(logs).all()
        or np.any(a <= 0)
        or np.any(b <= 0)
        or np.any(rate <= 0)
        or np.any(rate > 1 + TOLERANCE["local_absolute"])
    ):
        raise NumericalIntegrityError("inner transition is unresolved in float64")
    # A rate above one only within roundoff corresponds to lambda=0.
    rate = np.minimum(rate, 1.0)
    with np.errstate(divide="ignore"):
        log_lambda = np.log1p(-rate)
    log_plus, log_minus = log_a - log_rate, log_b - log_rate
    q_plus, q_minus = np.exp(log_plus), np.exp(log_minus)
    theta = m5a.kernel_logit(vector, structure, inputs)
    expected_plus = -np.logaddexp(0.0, -2 * theta)
    expected_minus = -np.logaddexp(0.0, 2 * theta)
    checks = {
        "row_sum_error": float(np.max(np.abs(q_plus + q_minus - 1))),
        "marginal_error": float(
            max(
                np.max(np.abs(q_plus - np.exp(expected_plus))),
                np.max(np.abs(q_minus - np.exp(expected_minus))),
            )
        ),
        "marginal_log_error": float(
            max(
                np.max(np.abs(log_plus - expected_plus)),
                np.max(np.abs(log_minus - expected_minus)),
            )
        ),
        "balance_error": float(
            np.max(np.abs(np.exp(expected_minus) * a - np.exp(expected_plus) * b))
        ),
        "balance_log_error": float(np.max(np.abs(expected_minus + log_a - expected_plus - log_b))),
    }
    for name, error in checks.items():
        tolerance = TOLERANCE["local_log" if "log" in name else "local_absolute"]
        if not math.isfinite(error) or error > tolerance:
            raise NumericalIntegrityError(f"local invariant failed: {name}={error}")
    return {
        "a": a,
        "b": b,
        "log_a": log_a,
        "log_b": log_b,
        "q_plus": q_plus,
        "q_minus": q_minus,
        "rate": rate,
        "lambda": 1 - rate,
        "log_lambda": log_lambda,
        "checks": checks,
    }


def powered_rates(kernel, k):
    """Two positive escape probabilities after K inner sweeps (None = limit)."""
    if k is not None and (isinstance(k, bool) or not isinstance(k, Integral) or k < 1):
        raise ValueError("K must be a positive integer")
    escape = 1.0 if k is None else -np.expm1(float(k) * kernel["log_lambda"])
    return np.column_stack((kernel["q_plus"] * escape, kernel["q_minus"] * escape))


def _required_k(log_lambda, tolerance, amplitude=1.0):
    """Conservative integer bound, including powers too slow for float lambda."""
    amplitude = np.broadcast_to(amplitude, log_lambda.shape)
    valid = np.isfinite(log_lambda) & (amplitude > tolerance)
    values = np.ones(log_lambda.shape)
    values[valid] = np.log(tolerance / amplitude[valid]) / log_lambda[valid]
    if not np.isfinite(values).all():
        raise NumericalIntegrityError("K bound is not representable in float64")
    k = max(1, math.ceil(float(values.max())))
    # For huge integers, increasing by one can round to the same float.
    for _ in range(4):
        achieved = float(np.max(amplitude * np.exp(float(k) * log_lambda)))
        if achieved <= tolerance:
            return k, achieved
        k = max(k + 1, math.ceil(float(k) * (1 + 4 * np.finfo(float).eps)))
    raise NumericalIntegrityError("integer K failed its contraction bound")


def mixing_summary(kernel):
    """Uniform-input per-site lambda distribution and local worst-start bounds."""
    lam = kernel["lambda"]
    amplitude = np.maximum(kernel["q_plus"], kernel["q_minus"])
    k_star, contraction = _required_k(kernel["log_lambda"], TOLERANCE["large_k_contraction"])
    return {
        "lambda_min": float(lam.min()),
        "lambda_median": float(np.quantile(lam, 0.5, method="linear")),
        "lambda_p95": float(np.quantile(lam, 0.95, method="linear")),
        "lambda_max": float(lam.max()),
        "maximizing_input": int(np.argmin(kernel["rate"])),
        "min_escape_rate": float(kernel["rate"].min()),
        "k_tv_1e-3": _required_k(kernel["log_lambda"], 1e-3, amplitude)[0],
        "k_tv_1e-6": _required_k(kernel["log_lambda"], 1e-6, amplitude)[0],
        "k_star": k_star,
        "k_star_contraction": contraction,
        "integrity": kernel["checks"],
    }


def round_parameters(vectors, cap, bits):
    """Apply the declared 2**bits-1-level codebook, recording coefficient errors."""
    if not math.isfinite(cap) or cap <= 0 or bits not in (4, 8, 12):
        raise ValueError("invalid cap or precision")
    levels = 2 ** (bits - 1) - 1
    step = cap / levels
    rounded, errors = [], []
    for vector in vectors:
        p = np.asarray(vector, dtype=np.float64)
        if p.ndim != 1 or not p.size or not np.isfinite(p).all() or np.any(np.abs(p) > cap):
            raise ValueError("parameter is nonfinite, empty or outside cap")
        q = step * np.clip(np.rint(p / step), -levels, levels)
        rounded.append(q)
        errors.append(np.abs(q - p))

    def summarize(error):
        return {
            "max_absolute": float(error.max()),
            "mean_absolute": float(error.mean()),
            "max_cap_normalized": float(error.max() / cap),
            "mean_cap_normalized": float(error.mean() / cap),
        }

    return rounded, {
        **summarize(np.concatenate(errors)),
        "sites": [summarize(error) for error in errors],
    }


def sweep(rows, rates):
    """Propagate each incoming output separately, in site-bit order."""
    rows = np.array(rows, dtype=np.float64, copy=True)
    size = 1 << len(rates)
    if rows.ndim != 2 or rows.shape[1] != size:
        raise ValueError("state table does not match site count")
    index = np.arange(size)
    for site, transition in enumerate(rates):
        transition = np.asarray(transition)
        if (
            transition.shape != (size, 2)
            or not np.isfinite(transition).all()
            or np.any(transition < 0)
            or np.any(transition > 1)
        ):
            raise NumericalIntegrityError("invalid site escape probabilities")
        low = index[(index >> site) & 1 == 0]
        high = low | (1 << site)
        a, b = transition[low, 0], transition[high, 1]
        old_low, old_high = rows[:, low], rows[:, high]
        rows[:, low] = old_low * (1 - a) + old_high * b
        rows[:, high] = old_low * a + old_high * (1 - b)
    return rows


def stationary_law(matrix):
    """Solve without clipping or regularization; reject unresolved uniqueness."""
    matrix = np.asarray(matrix, dtype=np.float64)
    size = len(matrix)
    if (
        matrix.shape != (size, size)
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or np.max(np.abs(matrix.sum(axis=1) - 1)) > TOLERANCE["local_absolute"]
    ):
        raise NumericalIntegrityError("invalid stochastic matrix")
    system = matrix.T.copy()
    system.flat[:: size + 1] -= 1
    system[-1] = 1.0
    rhs = np.zeros(size)
    rhs[-1] = 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            law = solve(
                system, rhs, overwrite_a=True, overwrite_b=True, check_finite=False, assume_a="gen"
            )
    except (np.linalg.LinAlgError, LinAlgWarning) as exc:
        raise NumericalIntegrityError("stationary solve cannot resolve a unique law") from exc
    if not np.isfinite(law).all() or law.min() < -TOLERANCE["local_absolute"]:
        raise NumericalIntegrityError("stationary solve produced an invalid law")
    # Stochastic propagation removes solve roundoff without clipping or damping.
    law = law / law.sum()
    for _ in range(20):
        law = law @ matrix
    if not np.isfinite(law).all() or law.min() < 0 or law.sum() <= 0:
        raise NumericalIntegrityError("stationary propagation produced an invalid law")
    law /= law.sum()
    if np.abs(law @ matrix - law).sum() > TOLERANCE["stationary"]:
        raise NumericalIntegrityError("stationary residual exceeds tolerance")
    return law


def max_row_tv(left, right):
    return max(
        float(0.5 * np.abs(left[i : i + 128] - right[i : i + 128]).sum(axis=1).max())
        for i in range(0, len(left), 128)
    )


def trajectory(matrix, horizon=30):
    path = [np.full(len(matrix), 1.0 / len(matrix))]
    for _ in range(horizon):
        path.append(path[-1] @ matrix)
    return np.array(path)


def chain_metrics(
    matrix,
    reference,
    ideal,
    target,
    spins,
    horizon=30,
    *,
    reference_law=None,
    reference_path=None,
    ideal_path=None,
):
    """Exact stationary and finite-horizon metrics, without sample surrogates."""
    law = stationary_law(matrix)
    if reference_law is None:
        reference_law = stationary_law(reference)
    if reference_path is None:
        reference_path = trajectory(reference, horizon)
    if ideal_path is None:
        ideal_path = trajectory(ideal, horizon)
    path = trajectory(matrix, horizon)
    errors = np.abs(spins.T @ (law - target))
    return {
        "bias": float(0.5 * np.abs(law - target).sum()),
        "stationary_shift": float(0.5 * np.abs(law - reference_law).sum()),
        "stationary_residual": float(np.abs(law @ matrix - law).sum()),
        "site_error_mean": float(errors.mean()),
        "site_error_max": float(errors.max()),
        "sweep_tv_ideal": max_row_tv(matrix, ideal),
        "sweep_tv_m5a": max_row_tv(matrix, reference),
        "tv_ideal": (0.5 * np.abs(path - ideal_path).sum(axis=1)).tolist(),
        "tv_m5a": (0.5 * np.abs(path - reference_path).sum(axis=1)).tolist(),
        "tv_target": (0.5 * np.abs(path - target).sum(axis=1)).tolist(),
        "tv_own_stationary": (0.5 * np.abs(path - law).sum(axis=1)).tolist(),
    }


def integrity_controls():
    """Independent tiny joint Gibbs enumeration and a compatible visible joint."""
    states = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])  # y,w
    y, w = states.T
    joint = np.exp(0.17 * y - 0.3 * w + 0.7 * y * w)
    joint /= joint.sum()
    transition = np.empty((4, 4))
    for i, (old_y, _) in enumerate(states):
        for j, (_, new_w) in enumerate(states):
            wm, ym = w == new_w, y == old_y
            transition[i, j] = joint[wm & ym].sum() / joint[ym].sum() * joint[j] / joint[wm].sum()
    kernel = inner_kernel([0.17, -0.3, 0.7], {"blanket": [], "triples": [((), 0)]})
    error = 0.0
    for k in (1, 2, 4):
        exact = np.linalg.matrix_power(transition, k)
        rates = powered_rates(kernel, k)[0]
        for start in range(4):
            up = rates[0] if y[start] == -1 else 1 - rates[1]
            error = max(error, abs(up - exact[start, y == 1].sum()))
    spins = ((np.arange(4)[:, None] >> np.arange(2)) & 1) * 2 - 1
    law = np.exp(0.7 * spins.prod(axis=1) + spins @ [0.2, -0.3])
    law /= law.sum()
    rates = []
    for site, field in enumerate((0.2, -0.3)):
        p = 1 / (1 + np.exp(-2 * (field + 0.7 * spins[:, 1 - site])))
        rates.append(np.column_stack((0.4 * p, 0.4 * (1 - p))))
    matrix = sweep(np.eye(4), rates)
    residual = float(np.abs(law @ matrix - law).sum())
    # Incoming-state dependence is observable even on the same fixed blanket.
    dependency = float(1 - powered_rates(kernel, 1).sum())
    if error > TOLERANCE["local_absolute"] or residual > TOLERANCE["stationary"] or dependency <= 0:
        raise NumericalIntegrityError("independent small-model control failed")
    return {
        "passed": True,
        "joint_enumeration_error": float(error),
        "compatible_joint_residual": residual,
        "incoming_output_dependence": dependency,
    }
