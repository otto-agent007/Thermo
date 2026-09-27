"""Predeclared exact-reference meta-EBM cap baseline (M5a).

Follows docs/experiments/meta-ebm-cap-baseline.md: d=12 three-body Ising
targets under two energy readings, single-site Gibbs kernels compiled at nine
coupling caps by a constructive recipe and by a variational fit, and every
chain metric computed by full enumeration of the 4,096 states.
"""

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import platform
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import combinations, product
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text

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
_ACTIVE_RUN = None


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


def _progress(message):
    """Emit timestamped progress that survives redirection to a run log."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{stamp} M5a {message}"
    if _ACTIVE_RUN is not None:
        log_path = _ACTIVE_RUN / "run.log"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        status_path = _ACTIVE_RUN / "run-status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        for marker, phase in (
            ("mixing scans", "mixing scans"),
            ("site fits", "site fits"),
            ("chain evaluations", "chain evaluations"),
            ("ideal checks", "ideal checks"),
            ("full persisted replay", "full persisted replay"),
            ("persisting archive", "persisting archive"),
        ):
            if marker in message:
                status["phase"] = phase
                break
        status.update(last_progress=message, updated_at=stamp)
        status["resources"] = _resource_snapshot()
        status.setdefault("schema", "meta_ebm_cap_baseline.run_status.v1")
        atomic_write_text(status_path, canonical_json(status) + "\n")
    print(line, flush=True)


def _utc_now():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resource_snapshot():
    """Read best-effort Linux cgroup v2 CPU and memory counters for run diagnostics."""
    root = Path("/sys/fs/cgroup")
    snapshot = {}
    for filename, key in (
        ("memory.current", "memory_current_bytes"),
        ("memory.peak", "memory_peak_bytes"),
        ("memory.max", "memory_limit_bytes"),
    ):
        path = root / filename
        if path.exists():
            value = path.read_text().strip()
            snapshot[key] = value if value == "max" else int(value)
    events = root / "memory.events"
    if events.exists():
        snapshot["memory_events"] = {
            key: int(value)
            for key, value in (line.split() for line in events.read_text().splitlines())
        }
    cpu_stat = root / "cpu.stat"
    if cpu_stat.exists():
        snapshot["cpu_stat_usec"] = {
            key: int(value)
            for key, value in (line.split() for line in cpu_stat.read_text().splitlines())
            if key.endswith("_usec")
        }
    cpu_max = root / "cpu.max"
    if cpu_max.exists():
        snapshot["cpu_max"] = cpu_max.read_text().strip()
    return snapshot


def _update_run_status(destination, **updates):
    path = Path(destination) / "run-status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.setdefault("schema", "meta_ebm_cap_baseline.run_status.v1")
    status.update(updates, updated_at=_utc_now())
    atomic_write_text(path, canonical_json(status) + "\n")


def _acquire_run_lock(destination):
    destination = Path(destination).resolve(strict=False)
    lock_path = destination.parent / f".{destination.name}.run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+", encoding="utf-8")
    locked = False
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())
    except BlockingIOError as exc:
        if locked:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
        if locked:
            raise
        raise RuntimeError(f"another M5a run is active for {destination}") from exc
    except BaseException:
        if locked:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
        raise
    return stream


def _release_run_lock(stream):
    if stream is not None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _target_key(reading, seed):
    return f"{reading}|{seed}"


def _fit_result_key(job):
    reading, seed, cap, site = job
    return f"{reading}|{seed}|{cap}|{site}"


def _chain_result_key(job):
    reading, seed, cap, method, *_ = job
    return f"{reading}|{seed}|{cap}|{method}"


def _empty_checkpoint_state():
    return {
        "mixing_results": {},
        "fit_results": {},
        "chain_results": {},
        "ideal_checks": {},
    }


def _new_runtime_attempt(runtime, attempt):
    return {
        "attempt": attempt,
        "started_at": _utc_now(),
        "generation_seconds": 0.0,
        "cumulative_generation_seconds": 0.0,
        "runtime": runtime,
    }


def _write_execution_checkpoint(
    destination,
    request,
    phase,
    state,
    elapsed_generation_seconds=0.0,
    runtime_attempts=None,
):
    resources = _resource_snapshot()
    payload = {
        "schema": "meta_ebm_cap_baseline.checkpoint.v1",
        "request": request,
        "request_digest": canonical_sha256(request),
        "phase": phase,
        "saved_at": _utc_now(),
        "elapsed_generation_seconds": elapsed_generation_seconds,
        "runtime_attempts": runtime_attempts or [],
        "resources": resources,
        **state,
    }
    payload["checkpoint_digest"] = canonical_sha256(payload)
    archive = gzip.compress((canonical_json(payload) + "\n").encode(), mtime=0)
    _atomic_write_archive(Path(destination) / "execution-checkpoint.json.gz", archive)
    _update_run_status(
        destination,
        phase=phase,
        checkpoint={
            "mixing_targets": len(state["mixing_results"]),
            "fit_jobs": len(state["fit_results"]),
            "chains": len(state["chain_results"]),
            "ideal_checks": len(state["ideal_checks"]),
            "digest": payload["checkpoint_digest"],
        },
        resources=resources,
    )


def _load_execution_checkpoint(path, request, runtime):
    try:
        payload = json.loads(gzip.decompress(Path(path).read_bytes()))
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"execution checkpoint is unreadable: {exc}") from exc
    digest = payload.pop("checkpoint_digest", None)
    if digest != canonical_sha256(payload):
        raise ValueError("execution checkpoint digest is invalid")
    if payload.get("schema") != "meta_ebm_cap_baseline.checkpoint.v1":
        raise ValueError("unsupported execution checkpoint schema")
    expected_request_digest = canonical_sha256(request)
    if (
        payload.get("request_digest") != expected_request_digest
        or payload.get("request") != request
    ):
        raise ValueError("execution checkpoint request or source hash differs from this run")
    elapsed = payload.get("elapsed_generation_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("execution checkpoint elapsed time is invalid")
    runtime_attempts = payload.get("runtime_attempts")
    if not isinstance(runtime_attempts, list) or not runtime_attempts:
        raise ValueError("execution checkpoint runtime history is invalid")
    for entry in runtime_attempts:
        if not isinstance(entry, dict) or not isinstance(entry.get("runtime"), dict):
            raise ValueError("execution checkpoint runtime history is invalid")
        prior_runtime = entry["runtime"]
        if any(prior_runtime.get(key) != runtime.get(key) for key in ("python", "numpy", "scipy")):
            raise ValueError(
                "resume runtime Python, NumPy, or SciPy version differs from checkpoint"
            )
        for field in ("generation_seconds", "cumulative_generation_seconds"):
            value = entry.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("execution checkpoint runtime history is invalid")
    state = {
        name: payload.get(name)
        for name in ("mixing_results", "fit_results", "chain_results", "ideal_checks")
    }
    if any(not isinstance(values, dict) for values in state.values()):
        raise ValueError("execution checkpoint result maps are invalid")
    return payload["phase"], state, elapsed, runtime_attempts


def _progress_interval(total):
    return max(1, total // 10)


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


def build_study(
    workers=None,
    fits=None,
    mixing_results=None,
    fit_workers=None,
    checkpoint_state=None,
    checkpoint_callback=None,
):
    """Generate the record, periodically checkpointing expensive completed work."""
    targets = _targets()
    if checkpoint_state is None:
        state = _empty_checkpoint_state()
        if mixing_results is not None:
            state["mixing_results"] = {
                _target_key(reading, seed): result
                for (reading, seed), result in zip(targets, mixing_results, strict=True)
            }
        if fits is not None:
            state["fit_results"] = {
                _fit_result_key((reading, seed, cap, site)): fits[f"{reading}|{seed}|{cap}"][site]
                for reading, seed in targets
                for cap in CAPS
                for site in range(D)
            }
    else:
        state = checkpoint_state
    mixing_by_key = dict(state["mixing_results"])
    fit_results_by_key = dict(state["fit_results"])
    chains_by_key = dict(state["chain_results"])
    checks_by_key = dict(state["ideal_checks"])

    def save(phase):
        if checkpoint_callback is not None:
            checkpoint_callback(
                phase,
                {
                    "mixing_results": mixing_by_key,
                    "fit_results": fit_results_by_key,
                    "chain_results": chains_by_key,
                    "ideal_checks": checks_by_key,
                },
            )

    target_keys = [_target_key(reading, seed) for reading, seed in targets]
    missing_targets = [
        (target, key)
        for target, key in zip(targets, target_keys, strict=True)
        if key not in mixing_by_key
    ]
    save("mixing scans")
    if missing_targets:
        mixing_workers = workers or os.cpu_count()
        _progress(
            f"mixing scans started: {len(missing_targets)} remaining of {len(targets)} targets; "
            f"workers={mixing_workers}"
        )
        with _pool(workers) as pool:
            futures = {pool.submit(mixing, target): key for target, key in missing_targets}
            for future in as_completed(futures):
                mixing_by_key[futures[future]] = future.result()
                save("mixing scans")
                _progress(f"mixing scans complete: {len(mixing_by_key)}/{len(targets)} targets")
    else:
        _progress(f"mixing scans reused: {len(mixing_by_key)} stored targets")
    ordered_mixing_results = [mixing_by_key[key] for key in target_keys]

    jobs = [
        (reading, seed, cap, site) for reading, seed in targets for cap in CAPS for site in range(D)
    ]
    missing_fit_jobs = [job for job in jobs if _fit_result_key(job) not in fit_results_by_key]
    save("site fits")
    if missing_fit_jobs:
        fit_worker_count = fit_workers if fit_workers is not None else workers
        _progress(
            f"site fits started: {len(missing_fit_jobs)} remaining of {len(jobs)} jobs; "
            f"workers={fit_worker_count or os.cpu_count()}"
        )
        interval = _progress_interval(len(jobs))
        save_interval = min(24, len(missing_fit_jobs))
        with _pool(fit_worker_count) as fit_pool:
            futures = {
                fit_pool.submit(fit_site, job): _fit_result_key(job) for job in missing_fit_jobs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                fit_results_by_key[futures[future]] = future.result()
                completed = len(fit_results_by_key)
                if index % save_interval == 0 or index == len(missing_fit_jobs):
                    save("site fits")
                if completed % interval == 0 or index == len(missing_fit_jobs):
                    _progress(f"site fits complete: {completed}/{len(jobs)} jobs")
        fits = {}
        for job in jobs:
            reading, seed, cap, site = job
            fits.setdefault(f"{reading}|{seed}|{cap}", [None] * D)[site] = fit_results_by_key[
                _fit_result_key(job)
            ]
    else:
        _progress(f"site fits reused: {len(fit_results_by_key)} stored fit jobs; no refitting")
        fits = {}
        for job in jobs:
            reading, seed, cap, site = job
            fits.setdefault(f"{reading}|{seed}|{cap}", [None] * D)[site] = fit_results_by_key[
                _fit_result_key(job)
            ]

    variational = {
        (reading, seed, cap): [fit["parameters"] for fit in fits[f"{reading}|{seed}|{cap}"]]
        for reading, seed in targets
        for cap in CAPS
    }
    chain_jobs = _chain_jobs(targets, ordered_mixing_results, variational)
    missing_chain_jobs = [job for job in chain_jobs if _chain_result_key(job) not in chains_by_key]
    missing_checks = [
        (target, key)
        for target, key in zip(targets, target_keys, strict=True)
        if key not in checks_by_key
    ]
    save("chain evaluations")
    if missing_chain_jobs:
        _progress(
            f"chain evaluations started: {len(missing_chain_jobs)} remaining of "
            f"{len(chain_jobs)} chains; workers={workers or os.cpu_count()}"
        )
    if missing_chain_jobs or missing_checks:
        with _pool(workers) as pool:
            interval = _progress_interval(len(chain_jobs))
            chain_save_interval = min(4, len(missing_chain_jobs)) if missing_chain_jobs else 1
            futures = {
                pool.submit(evaluate_chain, job): _chain_result_key(job)
                for job in missing_chain_jobs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                chains_by_key[futures[future]] = future.result()
                if index % chain_save_interval == 0 or index == len(missing_chain_jobs):
                    save("chain evaluations")
                completed = len(chains_by_key)
                if completed % interval == 0 or index == len(missing_chain_jobs):
                    _progress(f"chain evaluations complete: {completed}/{len(chain_jobs)} chains")
            if missing_checks:
                _progress(
                    f"ideal checks started: {len(missing_checks)} remaining of "
                    f"{len(targets)} targets"
                )
                save("ideal checks")
            futures = {
                pool.submit(ideal_checks, make_target(seed, reading)): key
                for (reading, seed), key in missing_checks
            }
            for future in as_completed(futures):
                checks_by_key[futures[future]] = future.result()
                save("ideal checks")
    if missing_chain_jobs:
        chains = [chains_by_key[_chain_result_key(job)] for job in chain_jobs]
    else:
        _progress(f"chain evaluations reused: {len(chains_by_key)} stored chains")
        chains = [chains_by_key[_chain_result_key(job)] for job in chain_jobs]
    if missing_checks:
        _progress(f"ideal checks complete: {len(checks_by_key)}/{len(targets)} targets")
    else:
        _progress(f"ideal checks reused: {len(checks_by_key)} stored targets")
    checks = [checks_by_key[key] for key in target_keys]
    record = {
        "request": study_request(),
        "targets": [
            {**make_target(seed, reading), "mixing": mix, "ideal_checks": check}
            for (reading, seed), mix, check in zip(
                targets, ordered_mixing_results, checks, strict=True
            )
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


def _validate_fit_records(fits):
    """Check the frozen fitting ledger without repeating optimization."""
    expected = {
        f"{reading}|{seed}|{cap}": structures(make_target(seed, reading))
        for reading, seed in _targets()
        for cap in CAPS
    }
    if not isinstance(fits, dict) or fits.keys() != expected.keys():
        raise ValueError("fit grid differs from the frozen protocol")
    attempt_fields = {"objective", "iterations", "success", "termination"}
    for key, structures_ in expected.items():
        sites = fits[key]
        if not isinstance(sites, list) or len(sites) != D:
            raise ValueError(f"fit site count differs at {key}")
        for site, (structure, fit) in enumerate(zip(structures_, sites, strict=True)):
            label = f"fit[{key}][{site}]"
            if not isinstance(fit, dict) or fit.keys() != {"selected", "parameters", "attempts"}:
                raise ValueError(f"invalid fit structure at {label}")
            parameters, attempts, selected = fit["parameters"], fit["attempts"], fit["selected"]
            if (
                not isinstance(parameters, list)
                or len(parameters) != parameter_count(structure)
                or any(type(p) not in (int, float) or not math.isfinite(p) for p in parameters)
            ):
                raise ValueError(f"invalid fit parameters at {label}")
            if not isinstance(attempts, list) or len(attempts) != RANDOM_STARTS + 1:
                raise ValueError(f"fit attempt count differs at {label}")
            for attempt in attempts:
                if (
                    not isinstance(attempt, dict)
                    or attempt.keys() != attempt_fields
                    or type(attempt["objective"]) not in (int, float)
                    or not math.isfinite(attempt["objective"])
                    or type(attempt["iterations"]) is not int
                    or attempt["iterations"] < 0
                    or type(attempt["success"]) is not bool
                    or not isinstance(attempt["termination"], str)
                ):
                    raise ValueError(f"invalid fit attempt metadata at {label}")
            if type(selected) is not int or not 0 <= selected < len(attempts):
                raise ValueError(f"invalid fit selection at {label}")
            lowest = min(range(len(attempts)), key=lambda i: (attempts[i]["objective"], i))
            if selected != lowest:
                raise ValueError(f"fit selection is not the lowest objective at {label}")


def _validate_record_identity(record):
    if canonical_json(record.get("request")) != canonical_json(study_request()):
        raise ValueError("study request differs from the frozen protocol or implementation")
    if record["request_digest"] != canonical_sha256(record["request"]):
        raise ValueError("request digest mismatch")
    body = ("targets", "fits", "chains", "integrity")
    if record["result_digest"] != canonical_sha256({key: record[key] for key in body}):
        raise ValueError("result digest does not match the stored record")
    _validate_fit_records(record["fits"])


def validate_study(record, workers=None, mixing_replay=None):
    """Replay from stored parameters without refitting.

    ``mixing_replay`` lists the (reading, seed) targets whose Dobrushin and
    spectral scans are recomputed; the default recomputes all ten.
    """
    _validate_record_identity(record)
    targets = _targets()
    stored_mixing = [target["mixing"] for target in record["targets"]]
    replay_set = set(targets if mixing_replay is None else mixing_replay)
    chosen = [t for t in targets if t in replay_set]
    _progress(f"replay mixing scans started: {len(chosen)} targets")
    recomputed = {}
    with _pool(workers) as pool:
        for index, (target, mix) in enumerate(
            zip(chosen, pool.map(mixing, chosen), strict=True), start=1
        ):
            recomputed[target] = mix
            _progress(f"replay mixing scans complete: {index}/{len(chosen)} targets")
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


def _runtime(workers, fit_workers=None):
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "workers": workers or os.cpu_count(),
    }
    if fit_workers is not None:
        runtime["fit_workers"] = fit_workers or os.cpu_count()
    return runtime


def _atomic_write_archive(path, data):
    """Publish a complete gzip stream with a single atomic rename."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _finish_replay(destination, record, archive, generation, workers):
    replay_started = time.monotonic()
    runtime = _runtime(workers)
    _progress("full persisted replay started")
    validate_study(record, workers)
    replay_seconds = time.monotonic() - replay_started
    _progress(f"full persisted replay passed in {replay_seconds:.1f} seconds")
    provenance = {
        "schema": "meta_ebm_cap_baseline.provenance.v2",
        "generation": generation,
        "replay": {
            "runtime": runtime,
            "replay_seconds": replay_seconds,
            "source_archive_sha256": hashlib.sha256(archive).hexdigest(),
            "request_digest": record["request_digest"],
            "result_digest": record["result_digest"],
        },
        "timing_scope": "CPU exact reference including host overhead; no hardware claims",
    }
    report = render_report(record, replayed=True)
    if generation["status"] == "available":
        minutes = generation["generation_seconds"] / 60
        report += f"\nOriginal generation took {minutes:.1f} wall-clock minutes.\n"
    else:
        report += "\nOriginal generation runtime provenance is unavailable.\n"
    report += f"This full replay took {replay_seconds / 60:.1f} wall-clock minutes.\n"
    atomic_write_text(destination / "summary.md", report)
    atomic_write_text(destination / "provenance.json", canonical_json(provenance) + "\n")
    atomic_write_text(
        destination / "completion.json",
        canonical_json(
            {
                "status": "meta_ebm_cap_baseline_complete",
                "targets": len(record["targets"]),
                "caps": len(CAPS),
                "methods": len(METHODS),
                "chains": len(record["chains"]),
                "samples": 0,
                "integrity": record["integrity"]["passed"],
                "replayed": True,
                "request_digest": record["request_digest"],
                "result_digest": record["result_digest"],
                "provenance_digest": canonical_sha256(provenance),
            }
        )
        + "\n",
    )
    _progress(f"study complete: {destination}")
    return record


def _load_completed_archive(destination, archive=None):
    if archive is None:
        archive = (Path(destination) / "study.json.gz").read_bytes()
    record = json.loads(gzip.decompress(archive))
    _validate_record_identity(record)
    generation = _original_generation(Path(destination), record, archive)
    completion = json.loads((Path(destination) / "completion.json").read_text())
    provenance = json.loads((Path(destination) / "provenance.json").read_text())
    replay = provenance.get("replay")
    archive_sha256 = hashlib.sha256(archive).hexdigest()
    if (
        completion.get("status") != "meta_ebm_cap_baseline_complete"
        or completion.get("replayed") is not True
        or completion.get("integrity") is not True
        or record["integrity"]["passed"] is not True
        or completion.get("request_digest") != record["request_digest"]
        or completion.get("result_digest") != record["result_digest"]
        or completion.get("targets") != len(record["targets"])
        or completion.get("chains") != len(record["chains"])
        or completion.get("samples") != 0
        or completion.get("provenance_digest") != canonical_sha256(provenance)
        or provenance.get("schema") != "meta_ebm_cap_baseline.provenance.v2"
        or provenance.get("generation") != generation
        or not isinstance(replay, dict)
        or replay.get("source_archive_sha256") != archive_sha256
        or replay.get("request_digest") != record["request_digest"]
        or replay.get("result_digest") != record["result_digest"]
    ):
        raise ValueError("completion marker, archive, and provenance do not agree")
    return record, generation


def _begin_run_status(destination, *, resume):
    status_path = Path(destination) / "run-status.json"
    previous = json.loads(status_path.read_text()) if resume and status_path.exists() else None
    if not resume:
        (Path(destination) / "run.log").write_text("", encoding="utf-8")
    previous_attempt = None
    if previous is not None:
        previous_attempt = {
            "status": previous.get("status", "unknown"),
            "phase": previous.get("phase"),
            "last_progress": previous.get("last_progress"),
            "checkpoint": previous.get("checkpoint"),
            "resources": previous.get("resources"),
            "error": previous.get("error"),
        }
        if previous.get("status") == "running":
            previous_attempt["status"] = "interrupted"
            previous_attempt["reason"] = (
                "previous process ended before it could write a terminal status; "
                "the exact stop cause is unavailable"
            )
            previous_attempt["resources_on_resume"] = _resource_snapshot()
    status = {
        "schema": "meta_ebm_cap_baseline.run_status.v1",
        "attempt": (previous.get("attempt", 0) + 1) if previous is not None else 1,
        "status": "running",
        "phase": "initializing",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    if previous_attempt is not None:
        status["previous_attempt"] = previous_attempt
    atomic_write_text(status_path, canonical_json(status) + "\n")
    return status


def _record_run_failure(destination, exc, *, error_updates=None, **status_updates):
    terminal_status = (
        "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
    )
    error = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    if error_updates:
        error.update(error_updates)
    with (Path(destination) / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(error["traceback"] + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _update_run_status(
        destination,
        status=terminal_status,
        stopped_at=_utc_now(),
        error=error,
        **status_updates,
    )
    _progress(f"run {terminal_status}: {error['type']}: {error['message']}")


def run_study(output_dir, *, replay=True, workers=None, fit_workers=None, resume=False):
    if replay is not True:
        raise ValueError("full replay is required before study completion")
    global _ACTIVE_RUN
    destination = Path(output_dir)
    run_lock = _acquire_run_lock(destination)
    try:
        checkpoint_path = destination / "execution-checkpoint.json.gz"
        archive_path = destination / "study.json.gz"
        if resume:
            if not destination.is_dir():
                raise FileNotFoundError(f"cannot resume missing study directory: {destination}")
        else:
            destination.mkdir(parents=True, exist_ok=False)
        status = _begin_run_status(destination, resume=resume)
    except BaseException:
        _release_run_lock(run_lock)
        raise
    _ACTIVE_RUN = destination
    status_path = destination / "run-status.json"

    started = None
    elapsed_generation_seconds = 0.0
    runtime_attempts = []
    latest_checkpoint_state = None
    generation_finalized = False
    try:
        request = study_request()
        started = time.monotonic()
        effective_fit_workers = fit_workers or workers or os.cpu_count()
        runtime = _runtime(workers, effective_fit_workers)
        _update_run_status(
            destination,
            workers=runtime["workers"],
            fit_workers=runtime["fit_workers"],
            resources=_resource_snapshot(),
        )

        if (destination / "completion.json").exists():
            _progress("resume found a completion marker; verifying its archive and provenance")
            result, generation = _load_completed_archive(destination)
        elif archive_path.exists():
            _progress("resume found a complete generated archive; retrying persisted replay")
            archive = archive_path.read_bytes()
            persisted = json.loads(gzip.decompress(archive))
            _validate_record_identity(persisted)
            generation = _original_generation(destination, persisted, archive)
            result = _finish_replay(destination, persisted, archive, generation, workers)
        else:
            if resume:
                if not checkpoint_path.exists():
                    raise ValueError(
                        "resume requires a valid execution checkpoint or generated archive"
                    )
                phase, checkpoint_state, elapsed_generation_seconds, runtime_attempts = (
                    _load_execution_checkpoint(checkpoint_path, request, runtime)
                )
                latest_checkpoint_state = checkpoint_state
                runtime_attempts.append(_new_runtime_attempt(runtime, status["attempt"]))
                _progress(f"resume loaded checkpoint: phase={phase}")
            else:
                checkpoint_state = _empty_checkpoint_state()
                latest_checkpoint_state = checkpoint_state
                runtime_attempts = [_new_runtime_attempt(runtime, status["attempt"])]
                _write_execution_checkpoint(
                    destination,
                    request,
                    "starting",
                    checkpoint_state,
                    elapsed_generation_seconds,
                    runtime_attempts,
                )
                _progress(
                    f"generation started: workers={runtime['workers']}; "
                    f"fit_workers={runtime['fit_workers']}"
                )

            def checkpoint(phase, state):
                nonlocal latest_checkpoint_state
                latest_checkpoint_state = state
                elapsed = elapsed_generation_seconds + time.monotonic() - started
                runtime_attempts[-1]["generation_seconds"] = time.monotonic() - started
                runtime_attempts[-1]["cumulative_generation_seconds"] = elapsed
                _write_execution_checkpoint(
                    destination, request, phase, state, elapsed, runtime_attempts
                )
                _update_run_status(destination, generation_elapsed_seconds=elapsed)

            record = build_study(
                workers,
                fit_workers=effective_fit_workers,
                checkpoint_state=checkpoint_state,
                checkpoint_callback=checkpoint,
            )
            elapsed_generation_seconds += time.monotonic() - started
            runtime_attempts[-1]["generation_seconds"] = time.monotonic() - started
            runtime_attempts[-1]["cumulative_generation_seconds"] = elapsed_generation_seconds
            generation_finalized = True
            _progress("generation finished; persisting archive")
            archive = gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
            generation = {
                "status": "available",
                "runtime": runtime,
                "runtime_attempts": runtime_attempts,
                "generation_seconds": elapsed_generation_seconds,
                "request_digest": record["request_digest"],
                "result_digest": record["result_digest"],
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
            }
            # Persist provenance first so an archive can survive interrupted replay.
            atomic_write_text(
                destination / "generation-provenance.json", canonical_json(generation) + "\n"
            )
            _atomic_write_archive(archive_path, archive)
            if not record["integrity"]["passed"]:
                raise ValueError(
                    "integrity requirements failed: " + "; ".join(record["integrity"]["failures"])
                )
            persisted = json.loads(gzip.decompress(archive_path.read_bytes()))
            result = _finish_replay(destination, persisted, archive, generation, workers)
        checkpoint_path.unlink(missing_ok=True)
        _update_run_status(
            destination,
            status="complete",
            phase="complete",
            completed_at=_utc_now(),
            request_digest=result["request_digest"],
            result_digest=result["result_digest"],
            generation_elapsed_seconds=(
                generation.get("generation_seconds", elapsed_generation_seconds)
            ),
        )
        return result
    except BaseException as exc:
        error_updates = {}
        if started is not None and runtime_attempts and not generation_finalized:
            active_seconds = time.monotonic() - started
            elapsed_generation_seconds += active_seconds
            runtime_attempts[-1]["generation_seconds"] = active_seconds
            runtime_attempts[-1]["cumulative_generation_seconds"] = elapsed_generation_seconds
            if latest_checkpoint_state is not None and not archive_path.exists():
                try:
                    current_status = json.loads(status_path.read_text())
                    _write_execution_checkpoint(
                        destination,
                        study_request(),
                        current_status.get("phase", "interrupted"),
                        latest_checkpoint_state,
                        elapsed_generation_seconds,
                        runtime_attempts,
                    )
                except OSError as checkpoint_error:
                    error_updates["checkpoint_write_error"] = str(checkpoint_error)
        _record_run_failure(
            destination,
            exc,
            error_updates=error_updates,
            generation_elapsed_seconds=elapsed_generation_seconds,
        )
        raise
    finally:
        _ACTIVE_RUN = None
        _release_run_lock(run_lock)


def _original_generation(source, record, archive):
    path = source / "generation-provenance.json"
    if path.exists():
        generation = json.loads(path.read_text())
    elif (source / "provenance.json").exists():
        previous = json.loads((source / "provenance.json").read_text())
        if "generation" in previous:
            generation = previous["generation"]
        else:
            # Legacy files describe both phases together. Retain those bytes' data,
            # but do not pretend they supply independently bound generation metadata.
            return {
                "status": "unavailable",
                "reason": "legacy provenance lacks a separately bound generation record",
                "legacy_provenance": previous,
            }
    else:
        return {"status": "unavailable", "reason": "source has no generation provenance"}
    if generation.get("status") == "unavailable":
        return generation
    expected = {
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
    }
    if generation.get("status") != "available" or any(
        generation.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("generation provenance does not match the source archive")
    if not isinstance(generation.get("runtime"), dict) or not isinstance(
        generation.get("generation_seconds"), (int, float)
    ):
        raise ValueError("generation provenance is incomplete")
    canonical_json(generation)  # Reject non-finite metadata before preserving it.
    return generation


def replay_study(source_dir: Path, output_dir: Path, *, workers=None, resume=False):
    """Fully replay an existing generated archive into a fresh directory, never refitting.

    Source archives must match this implementation. Missing historical generation
    metadata stays explicitly unavailable; this invocation records replay only.
    """
    global _ACTIVE_RUN
    source, destination = Path(source_dir), Path(output_dir)
    if not resume and destination.exists():
        raise FileExistsError(f"replay destination already exists: {destination}")

    if not resume:
        # Validate the source before creating the destination, so a bad archive
        # cannot leave behind something that looks like a resumable replay.
        archive = (source / "study.json.gz").read_bytes()
        record = json.loads(gzip.decompress(archive))
        _validate_record_identity(record)
        generation = _original_generation(source, record, archive)
    run_lock = _acquire_run_lock(destination)
    try:
        if resume:
            if not destination.is_dir():
                raise FileNotFoundError(f"cannot resume missing replay directory: {destination}")
        else:
            destination.mkdir(parents=True, exist_ok=False)
        _begin_run_status(destination, resume=resume)
    except BaseException:
        _release_run_lock(run_lock)
        raise
    _ACTIVE_RUN = destination
    try:
        if not resume:
            atomic_write_text(
                destination / "generation-provenance.json", canonical_json(generation) + "\n"
            )
            _atomic_write_archive(destination / "study.json.gz", archive)

        if resume:
            requested_archive = (source / "study.json.gz").read_bytes()
            persisted_archive = (destination / "study.json.gz").read_bytes()
            if (
                hashlib.sha256(requested_archive).digest()
                != hashlib.sha256(persisted_archive).digest()
            ):
                raise ValueError(
                    "requested source archive differs from the persisted replay archive"
                )
            archive = persisted_archive

        if (destination / "completion.json").exists():
            result, _ = _load_completed_archive(destination, archive=archive)
            if (destination / "study.json.gz").read_bytes() != archive:
                raise ValueError("persisted replay archive changed during resume")
            _update_run_status(
                destination,
                status="complete",
                phase="complete",
                completed_at=_utc_now(),
                request_digest=result["request_digest"],
                result_digest=result["result_digest"],
            )
            _progress("replay completion marker verified and status reconciled")
            return result

        if resume:
            record = json.loads(gzip.decompress(archive))
            _validate_record_identity(record)
            generation = _original_generation(destination, record, archive)

        persisted = json.loads(gzip.decompress(archive))
        result = _finish_replay(destination, persisted, archive, generation, workers)
        if resume and (destination / "study.json.gz").read_bytes() != archive:
            raise ValueError("persisted replay archive changed during resume")
        _update_run_status(
            destination,
            status="complete",
            phase="complete",
            completed_at=_utc_now(),
            request_digest=result["request_digest"],
            result_digest=result["result_digest"],
        )
        return result
    except BaseException as exc:
        _record_run_failure(destination, exc)
        raise
    finally:
        _ACTIVE_RUN = None
        _release_run_lock(run_lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--fit-workers",
        type=int,
        default=None,
        help="worker count for independent fit jobs (defaults to --workers)",
    )
    parser.add_argument(
        "--replay-from", type=Path, help="replay an existing archive without fitting"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume a checkpointed run or retry persisted replay"
    )
    args = parser.parse_args()
    if args.replay_from is not None:
        if args.fit_workers is not None:
            parser.error("--fit-workers applies only when generating a study")
        replay_study(args.replay_from, args.output_dir, workers=args.workers, resume=args.resume)
    else:
        run_study(
            args.output_dir,
            workers=args.workers,
            fit_workers=args.fit_workers,
            resume=args.resume,
        )
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
