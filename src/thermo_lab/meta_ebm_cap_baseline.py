"""Predeclared exact-reference meta-EBM cap baseline (M5a).

Follows docs/experiments/meta-ebm-cap-baseline.md: d=12 three-body Ising
targets under two energy readings, single-site Gibbs kernels compiled at nine
coupling caps by a constructive recipe and by a variational fit, and every
chain metric computed by full enumeration of the 4,096 states.
"""

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from thermo_lab.hashing import canonical_json, canonical_sha256

D = 12
STATES = 1 << D
PAIR_COUNT, TRIPLE_COUNT = 18, 20
WEIGHT_SCALE = 0.6
SEEDS = (0, 1, 2, 3, 4)
# Reading -> (pair scale, triple scale) applied to the drawn weights.
READINGS = {"A": (1.0, 1.0), "B": (0.5, 1.0 / 6.0)}
CAPS = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0, 10.0)
METHODS = ("constructive", "variational")
HORIZON = 30
START_SEED_BASE = 5
RANDOM_STARTS = 7
OPTIMIZER = {"maxiter": 5000, "maxfun": 50000, "ftol": 1e-15, "gtol": 1e-12}
SETTLE_FRACTION = 0.1
TOLERANCE = {
    "stationary": 1e-12,
    "detailed_balance": 1e-13,
    "brute_force": 1e-12,
    "bound": 1e-12,
    "replay_relative": 1e-9,
    "replay_absolute": 1e-10,
}
SPINS = ((np.arange(STATES)[:, None] >> np.arange(D)) & 1) * 2.0 - 1.0


def study_request():
    package = Path(__file__).parent
    return {
        "schema": "meta_ebm_cap_baseline.v1",
        "protocol": "docs/experiments/meta-ebm-cap-baseline.md",
        "evidence_class": "exact_reference",
        "dtype": "float64",
        "sites": D,
        "state_encoding": "site n is bit n of the state index; bit set means x_n = +1",
        "target_recipe": {
            "generator": "numpy.random.default_rng(seed)",
            "draw_order": ["pairs", "triples", "fields", "pair_weights", "triple_weights"],
            "pairs": PAIR_COUNT,
            "triples": TRIPLE_COUNT,
            "weight_standard_deviation": WEIGHT_SCALE,
        },
        "seeds": list(SEEDS),
        "readings": {name: list(scale) for name, scale in READINGS.items()},
        "primary_reading": "B",
        "sweep_order": list(range(D)),
        "caps": list(CAPS),
        "methods": list(METHODS),
        "variational": {
            "objective": "mean over uniform blanket inputs of KL(target || kernel)",
            "method": "L-BFGS-B",
            "options": OPTIMIZER,
            "starts": ["constructive", *(f"uniform_{i}" for i in range(RANDOM_STARTS))],
            "uniform_range": "[-min(cap, 1), min(cap, 1)]",
            "seed": f"default_rng([{START_SEED_BASE}, reading_index, seed, site, cap_index])",
            "selection": "lowest final objective; ties by start order",
        },
        "horizon": HORIZON,
        "initial_distribution": "uniform",
        "settle_fraction": SETTLE_FRACTION,
        "tolerance": TOLERANCE,
        "sample_count": 0,
        "implementation_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in ("meta_ebm_cap_baseline.py", "hashing.py")
        },
    }


# ---------------------------------------------------------------- targets


def make_target(seed, reading):
    """Draw one instance exactly as the protocol's recipe orders the draws."""
    rng = np.random.default_rng(seed)
    pairs = rng.choice(list(combinations(range(D), 2)), PAIR_COUNT, replace=False)
    triples = rng.choice(list(combinations(range(D), 3)), TRIPLE_COUNT, replace=False)
    fields = rng.normal(0.0, WEIGHT_SCALE, D)
    pair_weights = rng.normal(0.0, WEIGHT_SCALE, PAIR_COUNT)
    triple_weights = rng.normal(0.0, WEIGHT_SCALE, TRIPLE_COUNT)
    pair_scale, triple_scale = READINGS[reading]
    return {
        "seed": seed,
        "reading": reading,
        "pairs": pairs.tolist(),
        "triples": triples.tolist(),
        "fields": fields.tolist(),
        "pair_weights": (pair_weights * pair_scale).tolist(),
        "triple_weights": (triple_weights * triple_scale).tolist(),
    }


def target_distribution(target):
    energy = -(SPINS @ np.asarray(target["fields"]))
    for (m, n), w in zip(target["pairs"], target["pair_weights"], strict=True):
        energy -= w * SPINS[:, m] * SPINS[:, n]
    for (a, b, c), w in zip(target["triples"], target["triple_weights"], strict=True):
        energy -= w * SPINS[:, a] * SPINS[:, b] * SPINS[:, c]
    weights = np.exp(-(energy - energy.min()))
    return weights / weights.sum()


def site_structure(target, n):
    """Blanket, pair terms and triple terms of site n's exact conditional."""
    pairs = [
        (b if a == n else a, w)
        for (a, b), w in zip(target["pairs"], target["pair_weights"], strict=True)
        if n in (a, b)
    ]
    triples = [
        (tuple(s for s in tri if s != n), w)
        for tri, w in zip(target["triples"], target["triple_weights"], strict=True)
        if n in tri
    ]
    blanket = sorted({m for m, _ in pairs} | {m for others, _ in triples for m in others})
    return {"site": n, "blanket": blanket, "pairs": pairs, "triples": triples}


def blanket_inputs(structure):
    k = len(structure["blanket"])
    return ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1) * 2.0 - 1.0


def blanket_codes(structure):
    """Row of the blanket-input table seen by each of the 4,096 states."""
    codes = np.zeros(STATES, dtype=np.int64)
    for j, m in enumerate(structure["blanket"]):
        codes |= ((np.arange(STATES) >> m) & 1) << j
    return codes


def exact_logit(structure, inputs):
    """Half log-odds of the exact conditional on blanket-input rows."""
    column = {m: j for j, m in enumerate(structure["blanket"])}
    fields = structure.get("field")
    theta = np.full(len(inputs), fields if fields is not None else 0.0)
    for m, w in structure["pairs"]:
        theta += w * inputs[:, column[m]]
    for (m, m2), w in structure["triples"]:
        theta += w * inputs[:, column[m]] * inputs[:, column[m2]]
    return theta


def structures(target):
    result = []
    for n in range(D):
        structure = site_structure(target, n)
        structure["field"] = target["fields"][n]
        result.append(structure)
    return result


# ---------------------------------------------------------------- kernels


def parameter_count(structure):
    k, nh = len(structure["blanket"]), len(structure["triples"])
    return k + 1 + nh * k + 2 * nh


def unpack(parameters, structure):
    k, nh = len(structure["blanket"]), len(structure["triples"])
    p = np.asarray(parameters, dtype=np.float64)
    J, h = p[:k], p[k]
    A = p[k + 1 : k + 1 + nh * k].reshape(nh, k)
    b = p[k + 1 + nh * k : k + 1 + nh * k + nh]
    beta = p[k + 1 + nh * k + nh :]
    return J, h, A, b, beta


def kernel_logit(parameters, structure, inputs):
    """Closed-form half log-odds of E = -(J.x+h)y - sum_a (A_a.x+b_a) w_a - sum_a beta_a w_a y."""
    J, h, A, b, beta = unpack(parameters, structure)
    s = inputs @ A.T + b
    features = 0.5 * (np.logaddexp(s + beta, -(s + beta)) - np.logaddexp(s - beta, -(s - beta)))
    return inputs @ J + h + features.sum(axis=1)


def brute_force_probability(parameters, structure, inputs):
    """P(y=+1 | x) by explicit enumeration of every hidden configuration."""
    J, h, A, b, beta = unpack(parameters, structure)
    hidden = np.array(list(product([-1.0, 1.0], repeat=len(beta)))).reshape(-1, len(beta))
    drive = (inputs @ J + h)[:, None]
    coupling = (inputs @ A.T + b) @ hidden.T
    plus = logsumexp(drive + coupling + (hidden @ beta)[None, :], axis=1)
    minus = logsumexp(-drive + coupling - (hidden @ beta)[None, :], axis=1)
    return expit(plus - minus)


def constructive_parameters(structure, cap):
    """The appendix recipe, fully specified, with hidden biases and clipping."""
    blanket, k = structure["blanket"], len(structure["blanket"])
    column = {m: j for j, m in enumerate(blanket)}
    nh = len(structure["triples"])
    J, h = np.zeros(k), float(structure["field"])
    A, b, beta = np.zeros((nh, k)), np.full(nh, cap), np.zeros(nh)
    for m, w in structure["pairs"]:
        J[column[m]] += w
    for a, ((m, m2), u) in enumerate(structure["triples"]):
        A[a, column[m]] = A[a, column[m2]] = cap / 2.0
        beta[a] = -4.0 * u
        J[column[m]] += u
        J[column[m2]] += u
        h += 3.0 * u
    vector = np.concatenate([J, [h], A.ravel(), b, beta])
    return np.clip(vector, -cap, cap)


def objective(parameters, structure, inputs, target_logit):
    """Mean KL(target || kernel) over uniform inputs and its exact gradient."""
    J, h, A, b, beta = unpack(parameters, structure)
    s = inputs @ A.T + b
    plus, minus = s + beta, s - beta
    theta = (
        inputs @ J
        + h
        + (0.5 * (np.logaddexp(plus, -plus) - np.logaddexp(minus, -minus))).sum(axis=1)
    )
    t = expit(2.0 * target_logit)
    log_t, log_not_t = -np.logaddexp(0.0, -2.0 * target_logit), -np.logaddexp(0.0, 2 * target_logit)
    log_p, log_not_p = -np.logaddexp(0.0, -2.0 * theta), -np.logaddexp(0.0, 2.0 * theta)
    kl = t * (log_t - log_p) + (1.0 - t) * (log_not_t - log_not_p)
    rows = len(inputs)
    d_theta = 2.0 * (expit(2.0 * theta) - t) / rows
    tp, tm = np.tanh(plus), np.tanh(minus)
    d_s = 0.5 * (tp - tm) * d_theta[:, None]
    gradient = np.concatenate(
        [
            inputs.T @ d_theta,
            [d_theta.sum()],
            (d_s.T @ inputs).ravel(),
            d_s.sum(axis=0),
            (0.5 * (tp + tm) * d_theta[:, None]).sum(axis=0),
        ]
    )
    return float(kl.mean()), gradient


def _starts(structure, cap, reading, seed, site):
    rng = np.random.default_rng(
        [START_SEED_BASE, list(READINGS).index(reading), seed, site, CAPS.index(cap)]
    )
    width = min(cap, 1.0)
    count = parameter_count(structure)
    uniform = [rng.uniform(-width, width, count) for _ in range(RANDOM_STARTS)]
    return [constructive_parameters(structure, cap), *uniform]


def fit_site(job):
    reading, seed, cap, site = job
    structure = structures(make_target(seed, reading))[site]
    inputs = blanket_inputs(structure)
    target_logit = exact_logit(structure, inputs)
    attempts = []
    for start in _starts(structure, cap, reading, seed, site):
        result = minimize(
            objective,
            start,
            args=(structure, inputs, target_logit),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-cap, cap)] * len(start),
            options=OPTIMIZER,
        )
        endpoint = np.clip(np.asarray(result.x, dtype=np.float64), -cap, cap)
        attempts.append(
            {
                "objective": objective(endpoint, structure, inputs, target_logit)[0],
                "iterations": int(result.nit),
                "success": bool(result.success),
                "termination": str(result.message),
                "parameters": endpoint.tolist(),
            }
        )
    selected = min(range(len(attempts)), key=lambda i: (attempts[i]["objective"], i))
    return {
        "selected": selected,
        "parameters": attempts[selected]["parameters"],
        "attempts": [
            {key: value for key, value in attempt.items() if key != "parameters"}
            for attempt in attempts
        ],
    }


# ---------------------------------------------------------------- chains


def sweep(rows, conditionals):
    """Apply one systematic sweep to each row distribution (rows x 4,096)."""
    rows = np.array(rows, dtype=np.float64)
    index = np.arange(STATES)
    for n in range(D):
        low = index[(index >> n) & 1 == 0]
        high = low | (1 << n)
        mass = rows[:, low] + rows[:, high]
        p_up = conditionals[n][low]
        rows[:, low], rows[:, high] = mass * (1.0 - p_up), mass * p_up
    return rows


def ideal_conditionals(target):
    result = []
    for structure in structures(target):
        inputs = blanket_inputs(structure)
        result.append(expit(2.0 * exact_logit(structure, inputs))[blanket_codes(structure)])
    return result


def dobrushin(matrix, block=32, chunk=512):
    """Exact Dobrushin coefficient: largest total variation between two rows."""
    worst = 0.0
    for i in range(0, len(matrix), block):
        a = matrix[i : i + block]
        for j in range(i, len(matrix), chunk):
            b = matrix[j : j + chunk]
            worst = max(worst, float(0.5 * np.abs(a[:, None, :] - b[None]).sum(-1).max()))
    return worst


def mixing(job):
    reading, seed = job
    P = sweep(np.eye(STATES), ideal_conditionals(make_target(seed, reading)))
    moduli = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return {"dobrushin": dobrushin(P), "slem": float(moduli[1])}


def ideal_checks(target):
    """Gate 1: the ideal chain and every site kernel are exact for pi."""
    pi = target_distribution(target)
    conditionals = ideal_conditionals(target)
    P = sweep(np.eye(STATES), conditionals)
    index = np.arange(STATES)
    balance = 0.0
    for n in range(D):
        low = index[(index >> n) & 1 == 0]
        high = low | (1 << n)
        p_up = conditionals[n][low]
        balance = max(balance, float(np.max(np.abs(pi[low] * p_up - pi[high] * (1 - p_up)))))
    stationary = float(np.abs(pi @ P - pi).sum())
    return {"stationary_residual": stationary, "detailed_balance_residual": balance}


def stationary_law(matrix):
    system = matrix.T - np.eye(STATES)
    system[-1] = 1.0
    rhs = np.zeros(STATES)
    rhs[-1] = 1.0
    law = np.clip(np.linalg.solve(system, rhs), 0.0, None)
    law /= law.sum()
    for _ in range(20):
        law = law @ matrix
    return law / law.sum()


def evaluate_chain(job):
    """Every exact metric of one compiled chain from its stored parameters."""
    reading, seed, cap, method, parameters, mix = job
    target = make_target(seed, reading)
    pi = target_distribution(target)
    ideal = ideal_conditionals(target)
    P = sweep(np.eye(STATES), ideal)
    compiled, sites = [], []
    for structure, vector in zip(structures(target), parameters, strict=True):
        inputs = blanket_inputs(structure)
        if len(vector) != parameter_count(structure):
            raise ValueError("stored parameter vector has the wrong length")
        exact = expit(2.0 * exact_logit(structure, inputs))
        closed = expit(2.0 * kernel_logit(vector, structure, inputs))
        brute = brute_force_probability(vector, structure, inputs)
        compiled.append(closed[blanket_codes(structure)])
        sites.append(
            {
                "epsilon": float(np.max(np.abs(closed - exact))),
                "kl": objective(vector, structure, inputs, exact_logit(structure, inputs))[0],
                "brute_force_error": float(np.max(np.abs(closed - brute))),
                "cap_excess": float(max(0.0, np.max(np.abs(vector)) - cap)),
            }
        )
    compiled_P = sweep(np.eye(STATES), compiled)
    eta = float(0.5 * np.abs(compiled_P - P).sum(axis=1).max())
    q = np.full(STATES, 1.0 / STATES)
    q_model, delta = q.copy(), [0.0]
    for _ in range(HORIZON):
        q, q_model = q @ P, q_model @ compiled_P
        delta.append(float(0.5 * np.abs(q - q_model).sum()))
    law = stationary_law(compiled_P)
    rho = mix["dobrushin"]
    bounds = [eta * sum(rho**k for k in range(t)) for t in range(HORIZON + 1)]
    bias = float(0.5 * np.abs(law - pi).sum())
    floor_d = eta / (1.0 - rho) if rho < 1.0 else None
    epsilon_bar = max(site["epsilon"] for site in sites)
    floor_paper = epsilon_bar / (1.0 - mix["slem"])
    plateau = delta[-1]
    settle = next(
        t for t, value in enumerate(delta) if abs(value - plateau) <= SETTLE_FRACTION * plateau
    )
    site_error = np.abs(SPINS.T @ law - SPINS.T @ pi)
    return {
        "reading": reading,
        "seed": seed,
        "cap": cap,
        "method": method,
        "sites": sites,
        "epsilon_bar": epsilon_bar,
        "eta_sweep": eta,
        "delta": delta,
        "bias": bias,
        "floor_dobrushin": floor_d,
        "floor_paper": floor_paper,
        "bias_over_floor_paper": bias / floor_paper if floor_paper > 0 else None,
        "settle": settle,
        "site_error_mean": float(site_error.mean()),
        "site_error_max": float(site_error.max()),
        "stationary_residual": float(np.abs(law @ compiled_P - law).sum()),
        "bound_slack_min": float(min(b - d for b, d in zip(bounds, delta, strict=True))),
    }


# ---------------------------------------------------------------- study


def _pool(workers):
    """Single-threaded BLAS per worker; spawned so the limit applies at import."""
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    return ProcessPoolExecutor(
        max_workers=workers or os.cpu_count(), mp_context=multiprocessing.get_context("spawn")
    )


def _targets():
    return [(reading, seed) for reading in READINGS for seed in SEEDS]


def _chain_jobs(targets, mixing_results, variational):
    jobs = []
    for (reading, seed), mix in zip(targets, mixing_results, strict=True):
        sites = structures(make_target(seed, reading))
        for cap in CAPS:
            constructive = [constructive_parameters(s, cap).tolist() for s in sites]
            jobs.append((reading, seed, cap, "constructive", constructive, mix))
            jobs.append((reading, seed, cap, "variational", variational[(reading, seed, cap)], mix))
    return jobs


def integrity(record):
    """Gates 1-4 of the frozen protocol; any failure blocks completion."""
    tol = TOLERANCE
    failures = []
    for target in record["targets"]:
        checks = target["ideal_checks"]
        if checks["stationary_residual"] > tol["stationary"]:
            failures.append(f"ideal chain not stationary for {target['reading']}{target['seed']}")
        if checks["detailed_balance_residual"] > tol["detailed_balance"]:
            failures.append(f"detailed balance fails for {target['reading']}{target['seed']}")
    for chain in record["chains"]:
        label = f"{chain['reading']}{chain['seed']} cap {chain['cap']} {chain['method']}"
        if any(site["brute_force_error"] > tol["brute_force"] for site in chain["sites"]):
            failures.append(f"closed form disagrees with enumeration: {label}")
        if any(site["cap_excess"] > 0 for site in chain["sites"]):
            failures.append(f"parameter outside the cap: {label}")
        if chain["stationary_residual"] > tol["stationary"]:
            failures.append(f"compiled stationary solve inexact: {label}")
        if chain["bound_slack_min"] < -tol["bound"]:
            failures.append(f"finite-sweep bound violated: {label}")
        if chain["floor_dobrushin"] is not None and (
            chain["bias"] > chain["floor_dobrushin"] + tol["bound"]
        ):
            failures.append(f"stationary bound violated: {label}")
    return {"passed": not failures, "failures": failures}


def build_study(workers=None, fits=None, mixing_results=None):
    """Generate the record; ``fits`` and ``mixing_results`` let replay reuse stored values."""
    targets = _targets()
    with _pool(workers) as pool:
        if mixing_results is None:
            mixing_results = list(pool.map(mixing, targets))
        if fits is None:
            jobs = [
                (reading, seed, cap, site)
                for reading, seed in targets
                for cap in CAPS
                for site in range(D)
            ]
            results = list(pool.map(fit_site, jobs, chunksize=4))
            fits = {}
            for (reading, seed, cap, site), result in zip(jobs, results, strict=True):
                fits.setdefault(f"{reading}|{seed}|{cap}", [None] * D)[site] = result
        variational = {
            (reading, seed, cap): [fit["parameters"] for fit in fits[f"{reading}|{seed}|{cap}"]]
            for reading, seed in targets
            for cap in CAPS
        }
        chains = list(pool.map(evaluate_chain, _chain_jobs(targets, mixing_results, variational)))
        checks = list(pool.map(ideal_checks, [make_target(s, r) for r, s in targets]))
    record = {
        "request": study_request(),
        "targets": [
            {**make_target(seed, reading), "mixing": mix, "ideal_checks": check}
            for (reading, seed), mix, check in zip(targets, mixing_results, checks, strict=True)
        ],
        "fits": fits,
        "chains": chains,
    }
    record["request_digest"] = canonical_sha256(record["request"])
    record["integrity"] = integrity(record)
    record["result_digest"] = canonical_sha256(
        {key: record[key] for key in ("targets", "fits", "chains", "integrity")}
    )
    return json.loads(canonical_json(record))


def _close(stored, rebuilt, path="record"):
    if isinstance(stored, dict):
        if not isinstance(rebuilt, dict) or stored.keys() != rebuilt.keys():
            raise ValueError(f"replay structure differs at {path}")
        for key in stored:
            _close(stored[key], rebuilt[key], f"{path}.{key}")
    elif isinstance(stored, list):
        if not isinstance(rebuilt, list) or len(stored) != len(rebuilt):
            raise ValueError(f"replay structure differs at {path}")
        for i, (a, b) in enumerate(zip(stored, rebuilt, strict=True)):
            _close(a, b, f"{path}[{i}]")
    elif isinstance(stored, float) and isinstance(rebuilt, float):
        if not math.isclose(
            stored,
            rebuilt,
            rel_tol=TOLERANCE["replay_relative"],
            abs_tol=TOLERANCE["replay_absolute"],
        ):
            raise ValueError(f"replay value differs at {path}: {stored} vs {rebuilt}")
    elif type(stored) is not type(rebuilt) or stored != rebuilt:
        raise ValueError(f"replay value differs at {path}: {stored!r} vs {rebuilt!r}")


def validate_study(record, workers=None, mixing_replay=None):
    """Replay from stored parameters without refitting.

    ``mixing_replay`` lists the (reading, seed) targets whose Dobrushin and
    spectral scans are recomputed; the default recomputes all ten.
    """
    if canonical_json(record.get("request")) != canonical_json(study_request()):
        raise ValueError("study request differs from the frozen protocol or implementation")
    if record["request_digest"] != canonical_sha256(record["request"]):
        raise ValueError("request digest mismatch")
    targets = _targets()
    stored_mixing = [target["mixing"] for target in record["targets"]]
    replay_set = set(targets if mixing_replay is None else mixing_replay)
    with _pool(workers) as pool:
        chosen = [t for t in targets if t in replay_set]
        recomputed = dict(zip(chosen, pool.map(mixing, chosen), strict=True))
    for target, mix in zip(targets, stored_mixing, strict=True):
        if target in recomputed:
            _close(mix, recomputed[target], f"mixing[{target}]")
    for key, sites in record["fits"].items():
        reading, seed, cap = key.split("|")
        structures_ = structures(make_target(int(seed), reading))
        for structure, fit in zip(structures_, sites, strict=True):
            inputs = blanket_inputs(structure)
            value = objective(fit["parameters"], structure, inputs, exact_logit(structure, inputs))
            _close(fit["attempts"][fit["selected"]]["objective"], value[0], f"fits[{key}]")
    body = ("targets", "fits", "chains", "integrity")
    if record["result_digest"] != canonical_sha256({key: record[key] for key in body}):
        raise ValueError("result digest does not match the stored record")
    rebuilt = build_study(workers, fits=record["fits"], mixing_results=stored_mixing)
    _close(
        {key: record[key] for key in (*body, "request", "request_digest")},
        {key: rebuilt[key] for key in (*body, "request", "request_digest")},
    )
    if not record["integrity"]["passed"]:
        raise ValueError(
            "integrity requirements failed: " + "; ".join(record["integrity"]["failures"])
        )


# ---------------------------------------------------------------- report


def _fmt(value, digits=3):
    if value is None:
        return "vacuous"
    if value == 0:
        return "0"
    return f"{value:.{digits}g}" if abs(value) >= 1e-3 else f"{value:.{digits - 1}e}"


def render_report(record, *, replayed=False, seconds=None):
    chains = record["chains"]
    lines = [
        "# M5a meta-EBM cap baseline",
        "",
        "Exact-reference study following "
        "[the frozen protocol](../../experiments/meta-ebm-cap-baseline.md). "
        f"{len(record['targets'])} targets, {len(CAPS)} caps, {len(METHODS)} compile methods, "
        f"{len(chains)} compiled chains, zero samples. Every metric is computed by full "
        "enumeration of the 4,096 states of the declared compiled models.",
        "",
        f"Integrity: **{'passed' if record['integrity']['passed'] else 'FAILED'}**"
        + (" (replayed from stored parameters before reporting)" if replayed else "")
        + ".",
        "",
        "## Mixing of the ideal sweep",
        "",
        "| Reading | Seed | Dobrushin ρ_D | SLEM |",
        "|---|---|---|---|",
    ]
    for target in record["targets"]:
        mix = target["mixing"]
        lines.append(
            f"| {target['reading']} | {target['seed']} | {mix['dobrushin']:.6f} | "
            f"{mix['slem']:.4f} |"
        )
    for reading in READINGS:
        for method in METHODS:
            lines += [
                "",
                f"## Reading {reading}, {method}: median over seeds [min, max]",
                "",
                "| Cap | ε̄ | η_sweep | Bias | floor_D | floor_paper | Bias/floor_paper | Settle |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for cap in CAPS:
                group = [
                    c
                    for c in chains
                    if c["reading"] == reading and c["method"] == method and c["cap"] == cap
                ]

                def cell(key, group=group):
                    values = [c[key] for c in group]
                    if any(v is None for v in values):
                        finite = [v for v in values if v is not None]
                        note = f" ({len(values) - len(finite)} vacuous)"
                        return (_fmt(float(np.median(finite))) if finite else "vacuous") + note
                    return (
                        f"{_fmt(float(np.median(values)))} "
                        f"[{_fmt(min(values))}, {_fmt(max(values))}]"
                    )

                lines.append(
                    f"| {cap} | {cell('epsilon_bar')} | {cell('eta_sweep')} | {cell('bias')} | "
                    f"{cell('floor_dobrushin')} | {cell('floor_paper')} | "
                    f"{cell('bias_over_floor_paper')} | {cell('settle')} |"
                )
    lines += [
        "",
        "## Descriptive comparisons (non-gating)",
        "",
    ]
    for reading in READINGS:
        for method in METHODS:
            nonmonotone = []
            for seed in SEEDS:
                biases = [
                    next(
                        c["bias"]
                        for c in chains
                        if (c["reading"], c["seed"], c["cap"], c["method"])
                        == (reading, seed, cap, method)
                    )
                    for cap in CAPS
                ]
                breaks = [
                    CAPS[i + 1] for i in range(len(CAPS) - 1) if biases[i + 1] > biases[i] + 1e-12
                ]
                if breaks:
                    nonmonotone.append(f"seed {seed} at caps {breaks}")
            lines.append(
                f"- Reading {reading}, {method}: bias falls monotonically as the cap loosens "
                + (
                    "for every seed."
                    if not nonmonotone
                    else "except " + "; ".join(nonmonotone) + "."
                )
            )
    paired = [
        (c, v)
        for c in chains
        if c["method"] == "constructive"
        for v in chains
        if v["method"] == "variational"
        and (v["reading"], v["seed"], v["cap"]) == (c["reading"], c["seed"], c["cap"])
    ]
    better = sum(v["bias"] < c["bias"] for c, v in paired)
    lines += [
        f"- Variational bias is lower than constructive in {better} of {len(paired)} "
        "matched chains.",
        "- The paper reports bias 0.46 at J_max = 0.3 and 0.024 at J_max = 10, settling "
        "within about 3 sweeps, and bias ≈ 0.6 × its plotted floor. Its instance is "
        "unpublished, so only trends and orders of magnitude are comparable.",
        "",
        "floor_D = η_sweep/(1 − ρ_D) is the only proven bound. floor_paper = ε̄/(1 − SLEM) "
        "reproduces the paper's plotted comparison and is not a guaranteed bound.",
        "",
        "## Scope",
        "",
        "Fully connected kernels with the cap as the only substrate constraint. No Z1 "
        "topology, embedding, finite thermalization, latency, energy or hardware claim. "
        "Hidden-spin and parameter counts are logical, not device operations.",
    ]
    if seconds is not None:
        lines += ["", f"Generation and replay took {seconds / 60:.1f} CPU-pool minutes."]
    return "\n".join(lines) + "\n"


def run_study(output_dir, *, replay=True, workers=None):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = build_study(workers)
    if not record["integrity"]["passed"]:
        (destination / "study.json.gz").write_bytes(
            gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
        )
        raise ValueError(
            "integrity requirements failed: " + "; ".join(record["integrity"]["failures"])
        )
    generated = time.monotonic()
    (destination / "study.json.gz").write_bytes(
        gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    )
    persisted = json.loads(gzip.decompress((destination / "study.json.gz").read_bytes()))
    if replay:
        validate_study(persisted, workers)
    replayed = time.monotonic()
    (destination / "summary.md").write_text(
        render_report(persisted, replayed=replay, seconds=replayed - started)
    )
    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "workers": workers or os.cpu_count(),
        "generation_seconds": generated - started,
        "replay_seconds": replayed - generated,
        "timing_scope": "CPU exact reference including host overhead; no hardware claims",
    }
    (destination / "provenance.json").write_text(canonical_json(provenance) + "\n")
    (destination / "completion.json").write_text(
        canonical_json(
            {
                "status": "meta_ebm_cap_baseline_complete",
                "targets": len(persisted["targets"]),
                "caps": len(CAPS),
                "methods": len(METHODS),
                "chains": len(persisted["chains"]),
                "samples": 0,
                "integrity": persisted["integrity"]["passed"],
                "replayed": replay,
                "request_digest": persisted["request_digest"],
                "result_digest": persisted["result_digest"],
                "provenance_digest": canonical_sha256(provenance),
            }
        )
        + "\n"
    )
    return persisted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_study(args.output_dir, workers=args.workers)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
