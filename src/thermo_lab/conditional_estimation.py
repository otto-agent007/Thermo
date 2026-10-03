"""Paired empirical and conditional estimates on identical retained-state traces."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from thermo_lab import changing_evidence as stream
from thermo_lab import fixed_budget_sampling as base
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.provenance import collect_runtime_provenance

ROOT = base.ROOT
ESTIMATORS = ("empirical", "conditional")
ALGORITHMS = ("gibbs", "tempering")
SOURCES = (
    "src/thermo_lab/conditional_estimation.py",
    "docs/experiments/conditional-estimation.md",
    *stream.SOURCES,
)


def make_request():
    return {
        "schema": "conditional_estimation.v1",
        "evidence_class": "software_simulation",
        "reference_class": "exact_reference",
        "seeds": list(range(900, 916)),
        "conditions": stream.conditions(),
        "budgets": [4, 16, 64],
        "algorithms": list(ALGORITHMS),
        "estimators": list(ESTIMATORS),
        "queries": 25,
        "n": 12,
        "sampling_root": 20261008,
        "initialization_root": 20261009,
        "bootstrap_root": 20261010,
        "bootstrap_repeats": 10000,
        "timing_repeats": 5,
        "primary_budget": 16,
        "failure_threshold": 0.05,
        "ladder": list(base.LADDER),
        "dtype": "float32 sampling; float64 estimators and exact references",
        "replay_atol": 2e-10,
        "sources": {name: base.sha(ROOT / name) for name in SOURCES},
    }


def validate_request(request):
    if request["n"] != 12 or request["queries"] != 25:
        raise ValueError("frozen model/schedule dimensions")
    if request["algorithms"] != list(ALGORITHMS) or request["estimators"] != list(ESTIMATORS):
        raise ValueError("frozen algorithms and estimators")
    if request["ladder"] != list(base.LADDER) or request["replay_atol"] != 2e-10:
        raise ValueError("frozen ladder and replay tolerance")
    for name in ("seeds", "budgets"):
        values = request[name]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"empty or duplicate {name}")
    if any(t <= 0 or t % 4 for t in request["budgets"]):
        raise ValueError("positive budgets divisible by four required")
    if request["timing_repeats"] < 1:
        raise ValueError("timing repeats required")
    if request["failure_threshold"] != 0.05:
        raise ValueError("frozen failure threshold")
    conditions = request["conditions"]
    if not conditions or len({c["index"] for c in conditions}) != len(conditions):
        raise ValueError("empty or duplicate conditions")
    if any(c not in stream.conditions() for c in conditions):
        raise ValueError("unknown condition")


def conditional_probabilities(kept, fields, matrices):
    """Cold beta=4 conditionals on the archived quarter-scaled model."""
    spins = 2 * np.asarray(kept, dtype=float) - 1
    local = np.einsum("ski,sij->skj", spins, np.asarray(matrices, dtype=float))
    return expit(8 * (np.asarray(fields, dtype=float)[:, None, :] + local))


def estimate(kept, fields, matrices, edges, kind, alarm_threshold=8):
    spins = 2 * np.asarray(kept, dtype=float) - 1
    edge = np.asarray(edges)
    counts = kept.sum(axis=-1)
    if kind == "empirical":
        return {
            "marginal": kept.mean(axis=1),
            "edge": (spins[:, :, edge[:, 0]] * spins[:, :, edge[:, 1]]).mean(axis=1),
            "alarm": (counts >= alarm_threshold).mean(axis=1),
        }
    if kind != "conditional":
        raise ValueError(f"unknown estimator: {kind}")
    probability = conditional_probabilities(kept, fields, matrices)
    mean_spin = 2 * probability - 1
    other_count = counts[:, :, None] - kept.astype(int)
    return {
        "marginal": probability.mean(axis=1),
        "edge": (
            (
                spins[:, :, edge[:, 0]] * mean_spin[:, :, edge[:, 1]]
                + spins[:, :, edge[:, 1]] * mean_spin[:, :, edge[:, 0]]
            )
            / 2
        ).mean(axis=1),
        "alarm": (
            (other_count >= alarm_threshold) + (other_count == alarm_threshold - 1) * probability
        ).mean(axis=(1, 2)),
    }


def estimate_trace(states, algorithm, budget, fields, matrices, edges, kind):
    return estimate(stream.retained(states, algorithm, budget), fields, matrices, edges, kind)


def run_sequence(request, condition, inputs, algorithm, budget, kind, fn, keys):
    before = time.perf_counter()
    matrices = jax.block_until_ready(jnp.asarray(inputs["matrices"]))
    install_seconds = time.perf_counter() - before
    resident = None
    all_states, all_accepted, estimates, timings, estimator_timings = [], [], [], [], []
    for query in range(request["queries"]):
        before = time.perf_counter()
        if query == 0:
            resident = jnp.asarray(stream.initial_states(request, condition, request["seeds"], 0))
        device_states, device_accepted, resident = jax.block_until_ready(
            fn(keys[query], resident, jnp.asarray(inputs["fields"][query]), matrices)
        )
        states, accepted = np.asarray(device_states), np.asarray(device_accepted)
        estimator_start = time.perf_counter()
        estimated = estimate_trace(
            states,
            algorithm,
            budget,
            inputs["fields"][query],
            inputs["matrices"],
            inputs["edges"],
            kind,
        )
        estimator_timings.append(time.perf_counter() - estimator_start)
        timings.append(time.perf_counter() - before)
        all_states.append(states)
        all_accepted.append(accepted)
        estimates.append(estimated)
    return (
        np.array(all_states),
        np.array(all_accepted),
        estimates,
        {
            "query_seconds": timings,
            "estimator_seconds": estimator_timings,
            "model_install_seconds": install_seconds,
        },
    )


def cell_key(condition, algorithm, budget):
    return f"c{condition['index']}__{algorithm}__T{budget}"


def score_cell(request, condition, inputs, algorithm, budget, states, accepted, exact, timing):
    queries, seeds, n = request["queries"], request["seeds"], request["n"]
    if states.shape != (queries, len(seeds), budget + 1, 5, n):
        raise ValueError("state trace shape mismatch")
    if accepted.shape != (queries, len(seeds), budget, 2):
        raise ValueError("exchange trace shape mismatch")
    if algorithm == "gibbs" and accepted.any():
        raise ValueError("Gibbs cannot exchange")
    if not np.array_equal(states[0, :, 0], stream.initial_states(request, condition, seeds, 0)):
        raise ValueError("initialization mismatch")
    if not np.array_equal(states[1:, :, 0], states[:-1, :, -1]):
        raise ValueError("retained-state continuity mismatch")
    arms = {}
    for kind in ESTIMATORS:
        rows = [
            estimate_trace(
                states[q],
                algorithm,
                budget,
                inputs["fields"][q],
                inputs["matrices"],
                inputs["edges"],
                kind,
            )
            for q in range(queries)
        ]
        estimated = {name: np.array([row[name] for row in rows]) for name in rows[0]}
        error = estimated["marginal"] - exact["marginal"]
        mae = np.abs(error).mean(axis=-1)
        metrics = {
            "marginal_mae": mae,
            "marginal_mse": (error**2).mean(axis=-1),
            "failure": mae > request["failure_threshold"],
            "edge_mae": np.abs(estimated["edge"] - exact["edge"]).mean(axis=-1),
            "alarm_error": np.abs(estimated["alarm"] - exact["alarm"]),
            "regret": stream.regret(estimated["alarm"], exact["alarm"]),
        }
        observed = timing[kind]
        times = np.array([t["query_seconds"] for t in observed])
        estimate_times = np.array([t["estimator_seconds"] for t in observed])
        install_times = np.array([t["model_install_seconds"] for t in observed])
        if (
            times.shape != (request["timing_repeats"], queries)
            or estimate_times.shape != times.shape
        ):
            raise ValueError("timing shape mismatch")
        if (
            not np.isfinite(times).all()
            or not np.isfinite(estimate_times).all()
            or not np.isfinite(install_times).all()
            or np.any(times <= 0)
            or np.any(estimate_times <= 0)
            or np.any(estimate_times > times)
            or np.any(install_times <= 0)
        ):
            raise ValueError("invalid timing observation")
        arms[kind] = {
            "estimates": {name: value.tolist() for name, value in estimated.items()},
            "metrics": {name: value.tolist() for name, value in metrics.items()},
            "hysteresis_marginal_mae": (
                np.abs(estimated["marginal"][:12] - estimated["marginal"][24:12:-1])
                .mean(axis=(0, 2))
                .tolist()
                if condition["schedule"] == "roundtrip"
                else []
            ),
            "timing": observed,
            "warm_batch_seconds": float(np.median(times.sum(axis=1))),
            "estimator_batch_seconds": float(np.median(estimate_times.sum(axis=1))),
        }
    cold_count = (3 * budget // 4) * (5 if algorithm == "gibbs" else 1) * queries
    return {
        "condition": condition,
        "algorithm": algorithm,
        "budget": budget,
        "seeds": seeds,
        "arms": arms,
        "work_per_stream": {
            "spin_redraws": queries * 5 * budget * n,
            "swap_attempts": queries * 2 * budget if algorithm == "tempering" else 0,
            "initialized_spins": 5 * n,
            "trace_bits_returned": queries * (budget + 1) * 5 * n,
            "cold_observations": cold_count,
            "conditional_site_evaluations": {"empirical": 0, "conditional": cold_count * n},
        },
        "exchange_acceptance": float(accepted.mean()) if algorithm == "tempering" else None,
    }


def run_study(out, requested=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("use a fresh output directory")
    request = make_request() if requested is None else requested
    validate_request(request)
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise ValueError("use CPU and JAX_ENABLE_X64=false")
    out.mkdir(parents=True)
    base.write(out / "request.json", request)
    for name, digest in request["sources"].items():
        if base.sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    started = time.perf_counter()
    cells, packed, references, exact_timing, setup, compilation, cache = [], {}, {}, {}, {}, {}, {}
    for condition in request["conditions"]:
        before = time.perf_counter()
        inputs = stream.make_inputs(condition, request["seeds"])
        keys = {a: stream.planned_keys(request, condition, request["seeds"], a) for a in ALGORITHMS}
        jax.block_until_ready(keys)
        label = f"c{condition['index']}"
        setup[label] = time.perf_counter() - before
        exact, _, measured_exact = stream.references(inputs, request["timing_repeats"])
        references[label] = {k: v.tolist() for k, v in exact.items()}
        exact_timing[label] = measured_exact
        for bi, budget in enumerate(request["budgets"]):
            for ai, algorithm in enumerate(ALGORITHMS):
                cache_key = (algorithm, budget)
                if cache_key not in cache:
                    before = time.perf_counter()
                    cache[cache_key] = (
                        stream.make_sampler(algorithm, request["n"], budget)
                        .lower(
                            keys[algorithm][0],
                            jnp.asarray(
                                stream.initial_states(request, condition, request["seeds"], 0)
                            ),
                            jnp.asarray(inputs["fields"][0]),
                            jnp.asarray(inputs["matrices"]),
                        )
                        .compile()
                    )
                    compilation[str(cache_key)] = time.perf_counter() - before
                warm = {}
                for kind in ESTIMATORS:
                    warm[kind] = run_sequence(
                        request,
                        condition,
                        inputs,
                        algorithm,
                        budget,
                        kind,
                        cache[cache_key],
                        keys[algorithm],
                    )
                for index in (0, 1):
                    if not np.array_equal(warm["empirical"][index], warm["conditional"][index]):
                        raise ValueError("estimators changed the sampler trajectory")
                timing = {kind: [] for kind in ESTIMATORS}
                for repeat in range(request["timing_repeats"]):
                    order = (
                        ESTIMATORS
                        if (condition["index"] + bi + ai + repeat) % 2 == 0
                        else ESTIMATORS[::-1]
                    )
                    for kind in order:
                        current = run_sequence(
                            request,
                            condition,
                            inputs,
                            algorithm,
                            budget,
                            kind,
                            cache[cache_key],
                            keys[algorithm],
                        )
                        for index in (0, 1):
                            if not np.array_equal(current[index], warm[kind][index]):
                                raise ValueError("repeated trajectory differs")
                        for row, original in zip(current[2], warm[kind][2], strict=True):
                            if any(not np.array_equal(row[k], original[k]) for k in row):
                                raise ValueError("repeated estimator differs")
                        timing[kind].append(current[3])
                states, accepted = warm["empirical"][:2]
                key = cell_key(condition, algorithm, budget)
                packed[key] = np.packbits(states, axis=-1, bitorder="little")
                packed[key + "__accepted"] = np.packbits(accepted, axis=-1, bitorder="little")
                cells.append(
                    score_cell(
                        request,
                        condition,
                        inputs,
                        algorithm,
                        budget,
                        states,
                        accepted,
                        exact,
                        timing,
                    )
                )
        print(
            f"Measured condition {condition['index']}: {condition['strength']} "
            f"{condition['direction']} {condition['schedule']}",
            flush=True,
        )
    np.savez_compressed(out / "traces.npz", **packed)
    result = {
        "request_digest": canonical_sha256(request),
        "trace_sha256": base.sha(out / "traces.npz"),
        "cells": cells,
        "exact_references": references,
        "exact_timing": exact_timing,
        "setup_seconds": setup,
        "compile_seconds": compilation,
        "generation_seconds": time.perf_counter() - started,
        "provenance": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        "resources": {
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
            "threads": {
                name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
            },
        },
    }
    base.write(out / "results.json", result)
    return result


def replay(out):
    out = Path(out)
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    validate_request(request)
    if canonical_sha256(request) != result["request_digest"]:
        raise ValueError("request hash mismatch")
    if base.sha(out / "traces.npz") != result["trace_sha256"]:
        raise ValueError("trace hash mismatch")
    for name, digest in request["sources"].items():
        if base.sha(out / "source" / name) != digest:
            raise ValueError(f"archived source changed: {name}")
        if name.endswith(".py") and base.sha(ROOT / name) != digest:
            raise ValueError(f"loaded source changed: {name}")
    saved = {cell_key(c["condition"], c["algorithm"], c["budget"]): c for c in result["cells"]}
    expected_keys = {
        cell_key(c, a, b)
        for c in request["conditions"]
        for b in request["budgets"]
        for a in ALGORITHMS
    }
    if set(saved) != expected_keys or len(saved) != len(result["cells"]):
        raise ValueError("incomplete or duplicate grid")
    cells, references = [], {}
    with np.load(out / "traces.npz", allow_pickle=False) as archive:
        if set(archive.files) != expected_keys | {k + "__accepted" for k in expected_keys}:
            raise ValueError("trace grid mismatch")
        for condition in request["conditions"]:
            inputs = stream.make_inputs(condition, request["seeds"])
            exact, _, _ = stream.references(inputs, 0)
            label = f"c{condition['index']}"
            references[label] = {k: v.tolist() for k, v in exact.items()}
            observed = result["exact_timing"][label]
            times = np.asarray(observed["query_seconds"])
            if (
                times.shape != (request["timing_repeats"], request["queries"])
                or not np.isfinite(times).all()
                or np.any(times <= 0)
                or not np.isfinite(observed["setup_seconds"])
                or observed["setup_seconds"] <= 0
            ):
                raise ValueError("invalid exact timing")
            for budget in request["budgets"]:
                for algorithm in ALGORITHMS:
                    key = cell_key(condition, algorithm, budget)
                    states = np.unpackbits(
                        archive[key], axis=-1, count=request["n"], bitorder="little"
                    ).astype(bool)
                    accepted = np.unpackbits(
                        archive[key + "__accepted"], axis=-1, count=2, bitorder="little"
                    ).astype(bool)
                    timing = {kind: saved[key]["arms"][kind]["timing"] for kind in ESTIMATORS}
                    cells.append(
                        score_cell(
                            request,
                            condition,
                            inputs,
                            algorithm,
                            budget,
                            states,
                            accepted,
                            exact,
                            timing,
                        )
                    )
    stream.check_equal(references, result["exact_references"], "references")
    stream.check_equal(cells, result["cells"], "cells")
    completion = {
        "status": "conditional_estimation_complete",
        "request_digest": result["request_digest"],
        "results_sha256": base.sha(out / "results.json"),
        "trace_sha256": result["trace_sha256"],
        "trajectory_cells_replayed": len(cells),
        "estimator_cells_replayed": 2 * len(cells),
        "query_estimates_replayed": 2 * len(cells) * request["queries"] * len(request["seeds"]),
        "replay_seconds": time.perf_counter() - started,
        "scope": "authenticated sources/artifacts; complete grid, initialization, continuity, "
        "references, both estimates, errors, decisions, counts, timing medians; "
        "no historical timing or trajectory regeneration",
    }
    base.write(out / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if not args.replay:
        run_study(args.output_dir)
    print(canonical_json(replay(args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
