"""Bounded exploratory comparison of cold-target sampling allocations on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("long", "independent", "tempering")
LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)
SOURCES = (
    "src/thermo_lab/fixed_budget_sampling.py",
    "src/thermo_lab/hashing.py",
    "src/thermo_lab/persistence.py",
    "docs/experiments/fixed-budget-sampling.md",
    "uv.lock",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    atomic_write_text(path, canonical_json(value) + "\n")


def make_request():
    targets = []
    for n in (12, 16):
        for seed in (100, 101, 102):
            rng = np.random.default_rng(seed)
            ring = {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
            for _ in range(10_000):
                order = rng.permutation(n)
                matching = {tuple(sorted(map(int, order[i : i + 2]))) for i in range(0, n, 2)}
                if not ring & matching:
                    break
            else:
                raise RuntimeError("matching generation failed")
            edges = sorted(ring | matching)
            weights = rng.integers(1, 6, len(edges))
            weak = np.random.default_rng(np.random.SeedSequence([20261003, n, seed]))
            weak_fields = weak.choice([-0.15, 0.15], n).astype(np.float32)
            for variant in ("zero", "weak"):
                targets.append(
                    {
                        "id": f"n{n}-g{seed}-{variant}",
                        "n": n,
                        "seed": seed,
                        "variant": variant,
                        "edges": edges,
                        "couplings": (-weights.astype(np.float32) / 5).tolist(),
                        "fields": (np.zeros(n) if variant == "zero" else weak_fields).tolist(),
                    }
                )
    return {
        "schema": "fixed_budget_sampling.v1",
        "status": "exploratory",
        "targets": targets,
        "trials": 16,
        "root_seed": 20261003,
        "budgets_replica_sweeps": [64, 256, 1024],
        "ladder": list(LADDER),
        "methods": list(METHODS),
        "burn_in_fraction": 0.25,
        "numeric_dtype": "float32 parameters and sampling; float64 exact reference",
        "primary_metric": "TV of joint distribution of spins 0,1,2,3",
        "sample": "initial state then each full systematic Gibbs sweep; after swap for PT",
        "budget": "5*T*n spin redraws per trial including burn-in; PT additionally 2*T swaps",
        "replication": "independent trials; budgets and field variants are paired",
        "fixture_sweeps": 4096,
        "sources": {name: sha(ROOT / name) for name in SOURCES},
    }


def model(target):
    fields = np.asarray(target["fields"], dtype=np.float32)
    matrix = np.zeros((len(fields), len(fields)), dtype=np.float32)
    for (i, j), coupling in zip(target["edges"], target["couplings"], strict=True):
        matrix[i, j] = matrix[j, i] = coupling
    return fields, matrix


def enumerate_target(fields, matrix, beta=4.0):
    n = len(fields)
    if not 1 <= n <= 16:
        raise ValueError("exact enumeration is bounded to 1..16 spins")
    states = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(bool)
    spins = 2 * states.astype(np.float64) - 1
    q = spins @ np.asarray(fields, dtype=np.float64)
    q += 0.5 * np.sum((spins @ np.asarray(matrix, dtype=np.float64)) * spins, axis=1)
    probability = np.exp(beta * (q - q.max()))
    probability /= probability.sum()
    return states, probability, q


def swap_log_acceptance(beta_i, beta_j, q_i, q_j):
    """Log of the Metropolis exchange acceptance, including the sign convention."""
    return jnp.minimum(0.0, (beta_i - beta_j) * (q_j - q_i))


def key_inputs(index, n, trials, root_seed=20261003):
    root = jax.random.fold_in(jax.random.key(root_seed), index)
    pairs = jax.vmap(lambda i: jax.random.split(jax.random.fold_in(root, i)))(jnp.arange(trials))
    initial = jax.vmap(lambda k: jax.random.bernoulli(k, shape=(5, n)))(pairs[:, 0])
    return initial, pairs[:, 1]


def compile_sampler(method, n, horizon):
    if method not in METHODS:
        raise ValueError("unknown sampling method")
    replicas = 1 if method == "long" else 5
    steps = 5 * horizon if method == "long" else horizon
    beta = jnp.asarray(LADDER if method == "tempering" else [4.0] * replicas, jnp.float32)

    def single(key, initial, fields, matrix):
        start = initial[-1:] if method == "long" else initial

        def step(carry, t):
            state, key = carry
            key, sweep_key, exchange_key = jax.random.split(key, 3)

            def update_site(i, state):
                spins = 2 * state.astype(jnp.float32) - 1
                local = fields[i] + spins @ matrix[:, i]
                p = jax.nn.sigmoid(2 * beta * local)
                draw = jax.random.bernoulli(jax.random.fold_in(sweep_key, i), p)
                return state.at[:, i].set(draw)

            state = jax.lax.fori_loop(0, n, update_site, state)
            accepted = jnp.zeros(2, dtype=jnp.bool_)
            if method == "tempering":
                left = jnp.asarray([0, 2]) + t % 2
                right = left + 1
                spins = 2 * state.astype(jnp.float32) - 1
                q = spins @ fields + 0.5 * jnp.sum((spins @ matrix) * spins, axis=1)
                log_a = swap_log_acceptance(beta[left], beta[right], q[left], q[right])
                accepted = jnp.log(jax.random.uniform(exchange_key, shape=(2,))) < log_a
                old_left, old_right = state[left], state[right]
                state = state.at[left].set(jnp.where(accepted[:, None], old_right, old_left))
                state = state.at[right].set(jnp.where(accepted[:, None], old_left, old_right))
            return (state, key), (state, accepted)

        _, (states, accepted) = jax.lax.scan(step, (start, key), jnp.arange(steps))
        return jnp.concatenate([start[None], states], axis=0), accepted

    return jax.jit(jax.vmap(single, in_axes=(0, 0, None, None)))


def fixture_model():
    fields = np.asarray([0.2, -0.3, 0.1], np.float32)
    matrix = np.asarray([[0, 0.25, 0], [0.25, 0, -0.2], [0, -0.2, 0]], np.float32)
    return fields, matrix


def exact_checks():
    fields, matrix = fixture_model()
    states, p, q = enumerate_target(fields, matrix)
    spins = 2 * states.astype(float) - 1
    kernel = np.eye(len(states))
    for i in range(3):
        single = np.zeros_like(kernel)
        for row in range(len(states)):
            prob = expit(8 * (float(fields[i]) + spins[row] @ matrix[:, i].astype(float)))
            single[row, row | (1 << i)] = prob
            single[row, row & ~(1 << i)] = 1 - prob
        kernel = kernel @ single
    residuals, wrong = [], []
    for hot, cold in zip(LADDER[:-1], LADDER[1:], strict=True):
        ph = np.exp(hot * (q - q.max()))
        pc = np.exp(cold * (q - q.max()))
        joint = np.outer(ph / ph.sum(), pc / pc.sum())
        log_ratio = (hot - cold) * (q[None, :] - q[:, None])
        flow = joint * np.exp(np.minimum(0, log_ratio))
        bad_flow = joint * np.exp(np.minimum(0, -log_ratio))
        residuals.append(np.max(np.abs(flow - flow.T)))
        wrong.append(np.max(np.abs(bad_flow - bad_flow.T)))
    checks = {
        "evidence_class": "exact_reference",
        "gibbs_stationarity_residual": float(np.max(np.abs(p @ kernel - p))),
        "exchange_balance_residual": float(max(residuals)),
        "reversed_sign_residual": float(max(wrong)),
    }
    if checks["gibbs_stationarity_residual"] > 1e-12:
        raise ValueError("Gibbs stationarity check failed")
    if checks["exchange_balance_residual"] > 1e-12 or checks["reversed_sign_residual"] < 0.001:
        raise ValueError("exchange detailed balance check failed")
    return checks


def fixture_metrics(traces, horizon=4096):
    fields, matrix = fixture_model()
    _, probability, _ = enumerate_target(fields, matrix)
    result = exact_checks()
    result["empirical_tv"] = {}
    for method in METHODS:
        states = traces[f"fixture__{method}"]
        length = 5 * horizon if method == "long" else horizon
        kept = states[:, length // 4 + 1 : length + 1]
        if method == "tempering":
            kept = kept[:, :, -1:]
        codes = kept.reshape(-1, 3).astype(int) @ (1 << np.arange(3))
        observed = np.bincount(codes, minlength=8) / len(codes)
        tv = float(0.5 * np.abs(observed - probability).sum())
        result["empirical_tv"][method] = tv
        if tv >= 0.04:
            raise ValueError(f"empirical fixture failed: {method} TV={tv}")
    return result


def validate_fixture():
    exact_checks()
    fields, matrix = fixture_model()
    initial, keys = key_inputs(999, 3, 16)
    traces = {}
    for index, method in enumerate(METHODS):
        fn = compile_sampler(method, 3, 4096)
        sample_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, index)
        states, _ = fn(sample_keys, initial, jnp.asarray(fields), jnp.asarray(matrix))
        traces[f"fixture__{method}"] = np.asarray(states)
    return fixture_metrics(traces), traces


def score(target, method, states, accepted, budget):
    fields, matrix = model(target)
    exact_states, probability, _ = enumerate_target(fields, matrix)
    n = len(fields)
    length = 5 * budget if method == "long" else budget
    kept = states[:, length // 4 + 1 : length + 1]
    if method == "tempering":
        kept = kept[:, :, -1:]
    kept = kept.reshape(len(states), -1, n)
    codes_exact = exact_states[:, :4].astype(int) @ (1 << np.arange(4))
    joint = np.bincount(codes_exact, weights=probability, minlength=16)
    marginal = probability @ exact_states
    spin_exact = 2 * exact_states.astype(float) - 1
    edges = np.asarray(target["edges"])
    correlation = probability @ (spin_exact[:, edges[:, 0]] * spin_exact[:, edges[:, 1]])
    per_trial = {"joint_tv": [], "marginal_mae": [], "edge_mae": []}
    for trial in kept:
        codes = trial[:, :4].astype(int) @ (1 << np.arange(4))
        empirical = np.bincount(codes, minlength=16) / len(codes)
        spins = 2 * trial.astype(float) - 1
        pairs = (spins[:, edges[:, 0]] * spins[:, edges[:, 1]]).mean(axis=0)
        per_trial["joint_tv"].append(float(0.5 * np.abs(empirical - joint).sum()))
        per_trial["marginal_mae"].append(float(np.abs(trial.mean(axis=0) - marginal).mean()))
        per_trial["edge_mae"].append(float(np.abs(pairs - correlation).mean()))
    swap_rates = []
    if method == "tempering":
        for pair in range(4):
            swap_rates.append(float(accepted[:, pair % 2 : budget : 2, pair // 2].mean()))
    return {
        "target": target["id"],
        "variant": target["variant"],
        "method": method,
        "budget": budget,
        "spin_updates_per_trial": 5 * budget * n,
        "swap_attempts_per_trial": 2 * budget if method == "tempering" else 0,
        "retained_cold_states_per_trial": kept.shape[1],
        "per_trial": per_trial,
        "means": {name: float(np.mean(values)) for name, values in per_trial.items()},
        "exchange_acceptance_by_pair": swap_rates,
        "exact_reference": {
            "evidence_class": "exact_reference",
            "joint_4_spins": joint.tolist(),
            "marginals": marginal.tolist(),
            "edge_correlations": correlation.tolist(),
        },
    }


def run_study(out, requested=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("use a fresh output directory")
    requested = make_request() if requested is None else requested
    if requested["ladder"] != list(LADDER) or requested["methods"] != list(METHODS):
        raise ValueError("this runner implements the frozen ladder and three methods")
    if requested["fixture_sweeps"] != 4096 or requested["burn_in_fraction"] != 0.25:
        raise ValueError("this runner implements the frozen fixture and burn-in")
    out.mkdir(parents=True)
    write(out / "request.json", requested)
    for name, digest in requested["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    started = time.perf_counter()
    fixture, traces = validate_fixture()
    horizon = max(requested["budgets_replica_sweeps"])
    records, timing, executables, compilation = [], [], {}, {}
    for index, target in enumerate(requested["targets"]):
        n = target["n"]
        fields, matrix = (jnp.asarray(a) for a in model(target))
        initial, keys = key_inputs(index, n, requested["trials"], requested["root_seed"])
        for method_index, method in enumerate(METHODS):
            sample_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, method_index)
            args = (sample_keys, initial, fields, matrix)
            cache_key = (n, method)
            if cache_key not in executables:
                before = time.perf_counter()
                executables[cache_key] = compile_sampler(method, n, horizon).lower(*args).compile()
                compilation[f"n{n}-{method}"] = time.perf_counter() - before
            fn = executables[cache_key]
            jax.block_until_ready(fn(*args))
            durations = []
            for _ in range(3):
                before = time.perf_counter()
                result = jax.block_until_ready(fn(*args))
                durations.append(time.perf_counter() - before)
            states, accepted = map(np.asarray, result)
            name = f"{target['id']}__{method}"
            traces[name] = states
            traces[name + "__accepted"] = accepted
            timing.append({"target": target["id"], "method": method, "seconds": durations})
            records.extend(
                score(target, method, states, accepted, budget)
                for budget in requested["budgets_replica_sweeps"]
            )
        print(f"Sampled and scored {target['id']}", flush=True)
    packed = {
        name: (
            value if name.endswith("__accepted") else np.packbits(value, axis=-1, bitorder="little")
        )
        for name, value in traces.items()
    }
    np.savez_compressed(out / "traces.npz", **packed)
    result = {
        "request_digest": canonical_sha256(requested),
        "evidence_class": "software_simulation",
        "fixture": fixture,
        "cells": records,
        "trace_sha256": sha(out / "traces.npz"),
        "compile_seconds": compilation,
        "batch_timings": timing,
        "batch_timing_scope": (
            "16 (or requested) trials at full horizon; synchronized; excludes compilation, "
            "initialization, host transfer, scoring; includes Gibbs, exchanges, "
            "device trace collection"
        ),
        "generation_seconds": time.perf_counter() - started,
        "generation_timing_scope": (
            "fixture, enumeration, compilation, initialization, warm-up, 3 timed launches, "
            "scoring, trace write; excludes interpreter/setup and final JSON write"
        ),
        "provenance": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "numpy", "scipy", "thrml")
            },
            "devices": [str(device) for device in jax.devices()],
            "jax_x64": bool(jax.config.jax_enable_x64),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "memory_limit_bytes": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        },
    }
    write(out / "results.json", result)
    return result


def replay(out):
    out = Path(out)
    started = time.perf_counter()
    requested = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    if canonical_sha256(requested) != result["request_digest"]:
        raise ValueError("request digest mismatch")
    if sha(out / "traces.npz") != result["trace_sha256"]:
        raise ValueError("trace hash mismatch")
    for name, digest in requested["sources"].items():
        if sha(out / "source" / name) != digest:
            raise ValueError(f"archived source mismatch: {name}")
        if name.endswith(".py") and sha(ROOT / name) != digest:
            raise ValueError(f"loaded evaluator mismatch: {name}")
    with np.load(out / "traces.npz", allow_pickle=False) as archive:
        fixture_traces = {
            f"fixture__{m}": np.unpackbits(
                archive[f"fixture__{m}"], axis=-1, count=3, bitorder="little"
            ).astype(bool)
            for m in METHODS
        }
        if canonical_json(fixture_metrics(fixture_traces)) != canonical_json(result["fixture"]):
            raise ValueError("fixture metrics differ")
        rows = []
        horizon = max(requested["budgets_replica_sweeps"])
        for index, target in enumerate(requested["targets"]):
            n = target["n"]
            initial, _ = key_inputs(index, n, requested["trials"], requested["root_seed"])
            for method in METHODS:
                name = f"{target['id']}__{method}"
                states = np.unpackbits(archive[name], axis=-1, count=n, bitorder="little").astype(
                    bool
                )
                accepted = archive[name + "__accepted"]
                steps = 5 * horizon if method == "long" else horizon
                replicas = 1 if method == "long" else 5
                if states.shape != (requested["trials"], steps + 1, replicas, n):
                    raise ValueError("trace shape mismatch")
                if accepted.shape != (requested["trials"], steps, 2):
                    raise ValueError("exchange trace shape mismatch")
                expected_initial = np.asarray(initial[:, -1:] if method == "long" else initial)
                if not np.array_equal(states[:, 0], expected_initial):
                    raise ValueError("initialization mismatch")
                rows.extend(
                    score(target, method, states, accepted, budget)
                    for budget in requested["budgets_replica_sweeps"]
                )
    if canonical_json(rows) != canonical_json(result["cells"]):
        raise ValueError("replayed metrics differ")
    completion = {
        "status": "exploratory_fixed_budget_complete",
        "request_digest": result["request_digest"],
        "results_sha256": sha(out / "results.json"),
        "trace_sha256": result["trace_sha256"],
        "cells_replayed": len(rows),
        "replay_seconds": time.perf_counter() - started,
        "replay_scope": (
            "source/request/trace hashes, exact fixture invariance and negative control, "
            "saved fixture distributions, initialization, trace dimensions, every exact "
            "reference/accuracy metric/work count/exchange rate; "
            "no resampling or timing reproduction"
        ),
    }
    write(out / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if not args.replay:
        run_study(args.output_dir)
    print(canonical_json(replay(args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
