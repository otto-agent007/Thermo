"""Exploratory warm time-to-accuracy and posterior transfer with a frozen sampler."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from thermo_lab import fixed_budget_sampling as base
from thermo_lab.hashing import canonical_json, canonical_sha256

ROOT = base.ROOT
SOURCES = (
    "src/thermo_lab/sampling_time_to_accuracy.py",
    "docs/experiments/sampling-time-to-accuracy.md",
    *base.SOURCES,
)


def graph_targets():
    result = []
    for n in (12, 16):
        for seed in (200, 201, 202):
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
            couplings = -rng.integers(1, 6, len(edges)).astype(np.float32) / 5
            weak = np.random.default_rng(np.random.SeedSequence([20261004, n, seed]))
            fields = weak.choice([-0.15, 0.15], n).astype(np.float32)
            for variant in ("zero", "weak"):
                result.append(
                    {
                        "id": f"n{n}-g{seed}-{variant}",
                        "family": "graph",
                        "variant": variant,
                        "n": n,
                        "seed": seed,
                        "edges": edges,
                        "couplings": couplings.tolist(),
                        "fields": (np.zeros(n) if variant == "zero" else fields).tolist(),
                    }
                )
    return result


def denoising_targets():
    edges = [
        (i, j)
        for i in range(16)
        for j in range(i + 1, 16)
        if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1
    ]
    matrix = np.zeros((16, 16), np.float32)
    for i, j in edges:
        matrix[i, j] = matrix[j, i] = np.float32(0.4)
    states, probability, _ = base.enumerate_target(np.zeros(16), matrix, beta=1.0)
    result = []
    for seed in (100, 101, 102):
        rng = np.random.default_rng(np.random.SeedSequence([20261004, seed, 41]))
        clean = states[rng.choice(len(states), p=probability)]
        uniforms = rng.random(16)
        for noise in (0.2, 0.35):
            observed = clean ^ (uniforms < noise)
            fields = ((2 * observed.astype(int) - 1) * 0.5 * np.log((1 - noise) / noise)).astype(
                np.float32
            )
            result.append(
                {
                    "id": f"denoise-g{seed}-noise{noise:g}",
                    "family": "denoising",
                    "variant": f"noise{noise:g}",
                    "n": 16,
                    "seed": seed,
                    "noise": noise,
                    "clean": clean.astype(int).tolist(),
                    "observed": observed.astype(int).tolist(),
                    "edges": edges,
                    "couplings": [float(np.float32(0.4) / np.float32(4))] * len(edges),
                    "fields": (fields / np.float32(4)).tolist(),
                }
            )
    return result


def methods(target):
    return list(base.METHODS) + (
        ["long-flip", "independent-flip"] if target["variant"] == "zero" else []
    )


def make_request():
    return {
        "schema": "sampling_time_to_accuracy.v1",
        "status": "exploratory",
        "targets": graph_targets() + denoising_targets(),
        "budgets": [16, 64, 256, 1024, 4096],
        "trials": 16,
        "root_seed": 20261004,
        "timing_repeats": 5,
        "threshold": 0.05,
        "ladder": list(base.LADDER),
        "numeric_dtype": "float32 sampler; float64 exact enumeration",
        "sample": "initial plus each complete systematic Gibbs sweep; after exchange for tempering",
        "burn_in_fraction": 0.25,
        "replication_unit": (
            "independent trial; prefixes, paired variants and timing repeats are not extra trials"
        ),
        "timing_scope": (
            "warm 16-trial batch: sampling, exchanges, device trace collection, synchronization, "
            "host transfer, joint/marginal/edge estimation; excludes initialization, compilation, "
            "exact enumeration, oracle scoring, persistence"
        ),
        "qualification": "mean joint TV and edge MAE <= threshold, sustained at all later budgets",
        "replay_atol": 2e-12,
        "sources": {name: base.sha(ROOT / name) for name in SOURCES},
    }


def reference(target):
    states, p, _ = base.enumerate_target(*base.model(target))
    spins = 2 * states.astype(float) - 1
    edges = np.asarray(target["edges"])
    codes = states[:, :4].astype(int) @ (1 << np.arange(4))
    return {
        "evidence_class": "exact_reference",
        "joint": np.bincount(codes, weights=p, minlength=16).tolist(),
        "marginal": (p @ states).tolist(),
        "edge": (p @ (spins[:, edges[:, 0]] * spins[:, edges[:, 1]])).tolist(),
    }


def estimates(target, method, states, budget):
    if method.endswith("-flip") and np.any(np.asarray(target["fields"]) != 0):
        raise ValueError("global-flip augmentation requires zero fields")
    underlying = method.removesuffix("-flip")
    length = 5 * budget if underlying == "long" else budget
    kept = states[:, length // 4 + 1 : length + 1]
    if underlying == "tempering":
        kept = kept[:, :, -1:]
    kept = kept.reshape(len(states), -1, target["n"])
    edges = np.asarray(target["edges"])
    joint, marginal, correlation = [], [], []
    for trial in kept:
        codes = trial[:, :4].astype(int) @ (1 << np.arange(4))
        histogram = np.bincount(codes, minlength=16) / len(codes)
        spin_mean = trial.mean(axis=0)
        spins = 2 * trial.astype(float) - 1
        if method.endswith("-flip"):
            histogram = 0.5 * (histogram + histogram[::-1])
            spin_mean = np.full(target["n"], 0.5)
        joint.append(histogram)
        marginal.append(spin_mean)
        correlation.append((spins[:, edges[:, 0]] * spins[:, edges[:, 1]]).mean(axis=0))
    return {
        "joint": np.asarray(joint),
        "marginal": np.asarray(marginal),
        "edge": np.asarray(correlation),
    }


def accuracy(estimate, exact):
    per_trial = {}
    for key, name in (("joint", "joint_tv"), ("marginal", "marginal_mae"), ("edge", "edge_mae")):
        error = np.abs(np.asarray(estimate[key]) - np.asarray(exact[key]))
        value = error.sum(axis=1) / 2 if key == "joint" else error.mean(axis=1)
        per_trial[name] = value.tolist()
    return {
        "per_trial": per_trial,
        "means": {key: float(np.mean(value)) for key, value in per_trial.items()},
    }


def qualification(cells, threshold):
    cells = sorted(cells, key=lambda x: x["budget"])
    passes = [max(row["means"]["joint_tv"], row["means"]["edge_mae"]) <= threshold for row in cells]
    for i, row in enumerate(cells):
        if all(passes[i:]):
            return {
                "status": "reached",
                "budget": row["budget"],
                "warm_batch_seconds": row["pipeline_median_seconds"],
            }
    return {"status": "not_reached", "largest_tested_budget": cells[-1]["budget"]}


def decisions(request, cells):
    return [
        {
            "target": target["id"],
            "family": target["family"],
            "variant": target["variant"],
            "method": method,
            **qualification(
                [row for row in cells if row["target"] == target["id"] and row["method"] == method],
                request["threshold"],
            ),
        }
        for target in request["targets"]
        for method in methods(target)
    ]


def check_equal(actual, expected, path="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"keys differ: {path}")
        for name in expected:
            check_equal(actual[name], expected[name], f"{path}/{name}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"length differs: {path}")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            check_equal(a, b, f"{path}/{i}")
    elif isinstance(expected, float):
        if (
            not isinstance(actual, (float, int))
            or not np.isfinite(actual)
            or abs(actual - expected) > 2e-12
        ):
            raise ValueError(f"numeric value differs: {path}")
    elif actual != expected:
        raise ValueError(f"value differs: {path}")


def make_cell(target, method, states, accepted, budget, exact, timings, threshold):
    measured = accuracy(estimates(target, method, states, budget), exact)
    passed = (np.asarray(measured["per_trial"]["joint_tv"]) <= threshold) & (
        np.asarray(measured["per_trial"]["edge_mae"]) <= threshold
    )
    for timing in timings:
        if not 0 < timing["sample_transfer_seconds"] <= timing["pipeline_seconds"]:
            raise ValueError("invalid timing observation")
    return {
        "target": target["id"],
        "family": target["family"],
        "variant": target["variant"],
        "method": method,
        "budget": budget,
        "spin_updates_per_trial": 5 * budget * target["n"],
        "swap_attempts_per_trial": 2 * budget if method == "tempering" else 0,
        "retained_cold_states_per_trial": (3 * budget // 4) * (1 if method == "tempering" else 5),
        "augmentation": "analytic global flip" if method.endswith("-flip") else "none",
        **measured,
        "individual_trial_pass_fraction": float(passed.mean()),
        "exchange_acceptance_by_pair": (
            [float(accepted[:, pair % 2 : budget : 2, pair // 2].mean()) for pair in range(4)]
            if method == "tempering"
            else []
        ),
        "timings": timings,
        "pipeline_median_seconds": float(np.median([t["pipeline_seconds"] for t in timings])),
        "sample_transfer_median_seconds": float(
            np.median([t["sample_transfer_seconds"] for t in timings])
        ),
    }


def pipeline(fn, args, target, method, budget):
    started = time.perf_counter()
    states, accepted = map(np.asarray, jax.block_until_ready(fn(*args)))
    sampled = time.perf_counter()
    estimate = estimates(target, method, states, budget)
    ended = time.perf_counter()
    return (
        states,
        accepted,
        estimate,
        {
            "pipeline_seconds": ended - started,
            "sample_transfer_seconds": sampled - started,
        },
    )


def run_study(out, requested=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("use a fresh output directory")
    request = make_request() if requested is None else requested
    if any(b <= 0 or b % 4 for b in request["budgets"]):
        raise ValueError("budgets must be positive multiples of four")
    if request["ladder"] != list(base.LADDER) or request["burn_in_fraction"] != 0.25:
        raise ValueError("the reused sampler has a frozen ladder and burn-in")
    if request["replay_atol"] != 2e-12:
        raise ValueError("replay tolerance is frozen")
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise ValueError("use CPU with JAX_ENABLE_X64=false")
    out.mkdir(parents=True)
    base.write(out / "request.json", request)
    for name, digest in request["sources"].items():
        if base.sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    started = time.perf_counter()
    fixture, fixture_traces = base.validate_fixture()
    packed = {
        name: np.packbits(value, axis=-1, bitorder="little")
        for name, value in fixture_traces.items()
    }
    cells, references, compile_times, init_times = [], {}, {}, {}
    executables = {}
    horizon = max(request["budgets"])
    for index, target in enumerate(request["targets"]):
        n = target["n"]
        references[target["id"]] = reference(target)
        before = time.perf_counter()
        fields, matrix = (jnp.asarray(x) for x in base.model(target))
        initial, keys = base.key_inputs(index, n, request["trials"], request["root_seed"])
        arguments = {
            method: (
                jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, method_index),
                initial,
                fields,
                matrix,
            )
            for method_index, method in enumerate(base.METHODS)
        }
        jax.block_until_ready(arguments)
        init_times[target["id"]] = time.perf_counter() - before
        for budget in request["budgets"]:
            for method in base.METHODS:
                cache_key = (n, method, budget)
                if cache_key not in executables:
                    before = time.perf_counter()
                    executables[cache_key] = (
                        base.compile_sampler(method, n, budget).lower(*arguments[method]).compile()
                    )
                    compile_times[f"n{n}-{method}-T{budget}"] = time.perf_counter() - before
        longest = {}
        for method in base.METHODS:
            fn = executables[n, method, horizon]
            states, accepted = map(np.asarray, jax.block_until_ready(fn(*arguments[method])))
            longest[method] = (states, accepted)
            name = f"{target['id']}__{method}"
            packed[name] = np.packbits(states, axis=-1, bitorder="little")
            packed[name + "__accepted"] = accepted
        arms = methods(target)
        for budget in request["budgets"]:
            expected_estimates = {}
            timings = {method: [] for method in arms}
            for method in arms:
                underlying = method.removesuffix("-flip")
                fn = executables[n, underlying, budget]
                states, accepted, estimate, _ = pipeline(
                    fn, arguments[underlying], target, method, budget
                )
                long_states, long_accepted = longest[underlying]
                if not np.array_equal(states, long_states[:, : states.shape[1]]):
                    raise ValueError("horizon states are not a common prefix")
                if not np.array_equal(accepted, long_accepted[:, : accepted.shape[1]]):
                    raise ValueError("exchange flags are not a common prefix")
                expected_estimates[method] = estimate
            for repeat in range(request["timing_repeats"]):
                rotation = repeat % len(arms)
                for method in arms[rotation:] + arms[:rotation]:
                    underlying = method.removesuffix("-flip")
                    fn = executables[n, underlying, budget]
                    _, _, estimate, timing = pipeline(
                        fn, arguments[underlying], target, method, budget
                    )
                    if any(
                        not np.array_equal(estimate[k], expected_estimates[method][k])
                        for k in estimate
                    ):
                        raise ValueError("timing repetitions changed the estimates")
                    timings[method].append(timing)
            for method in arms:
                states, accepted = longest[method.removesuffix("-flip")]
                cells.append(
                    make_cell(
                        target,
                        method,
                        states,
                        accepted,
                        budget,
                        references[target["id"]],
                        timings[method],
                        request["threshold"],
                    )
                )
        print(f"Measured {index + 1}/{len(request['targets'])}: {target['id']}", flush=True)
    np.savez_compressed(out / "traces.npz", **packed)
    result = {
        "request_digest": canonical_sha256(request),
        "evidence_class": "software_simulation",
        "trace_sha256": base.sha(out / "traces.npz"),
        "fixture": fixture,
        "exact_references": references,
        "cells": cells,
        "decisions": decisions(request, cells),
        "compile_seconds": compile_times,
        "initialization_seconds": init_times,
        "generation_seconds": time.perf_counter() - started,
        "generation_scope": (
            "fixture, references, initialization, compilation, longest traces, warmups, "
            "five timed repeats, comparisons, scoring and trace write; excludes request "
            "construction, interpreter/setup, final JSON write and replay"
        ),
        "prefix_and_repeat_checks": (
            "all horizons share maximum trace prefixes; all timing estimates identical"
        ),
        "provenance": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "numpy", "scipy")
            },
            "devices": [str(device) for device in jax.devices()],
            "jax_x64": bool(jax.config.jax_enable_x64),
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "memory_limit_bytes": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
            "thread_settings": {
                name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
            },
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
            ),
        },
    }
    base.write(out / "results.json", result)
    return result


def replay(out):
    out = Path(out)
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    if canonical_sha256(request) != result["request_digest"]:
        raise ValueError("request digest mismatch")
    if base.sha(out / "traces.npz") != result["trace_sha256"]:
        raise ValueError("trace hash mismatch")
    for name, digest in request["sources"].items():
        if base.sha(out / "source" / name) != digest:
            raise ValueError(f"archived source mismatch: {name}")
        if name.endswith(".py") and base.sha(ROOT / name) != digest:
            raise ValueError(f"loaded evaluator mismatch: {name}")
    cells, references = [], {}
    timing_by_cell = {
        (row["target"], row["method"], row["budget"]): row["timings"] for row in result["cells"]
    }
    if len(timing_by_cell) != len(result["cells"]):
        raise ValueError("duplicate result cells")
    with np.load(out / "traces.npz", allow_pickle=False) as archive:
        fixture = {
            f"fixture__{method}": np.unpackbits(
                archive[f"fixture__{method}"], axis=-1, count=3, bitorder="little"
            ).astype(bool)
            for method in base.METHODS
        }
        check_equal(base.fixture_metrics(fixture), result["fixture"])
        horizon = max(request["budgets"])
        for index, target in enumerate(request["targets"]):
            n = target["n"]
            references[target["id"]] = reference(target)
            initial, _ = base.key_inputs(index, n, request["trials"], request["root_seed"])
            traces = {}
            for method in base.METHODS:
                name = f"{target['id']}__{method}"
                states = np.unpackbits(archive[name], axis=-1, count=n, bitorder="little").astype(
                    bool
                )
                accepted = archive[name + "__accepted"]
                steps = 5 * horizon if method == "long" else horizon
                replicas = 1 if method == "long" else 5
                if states.shape != (request["trials"], steps + 1, replicas, n):
                    raise ValueError("trace shape mismatch")
                if accepted.shape != (request["trials"], steps, 2):
                    raise ValueError("exchange shape mismatch")
                expected_initial = np.asarray(initial[:, -1:] if method == "long" else initial)
                if not np.array_equal(states[:, 0], expected_initial):
                    raise ValueError("initialization mismatch")
                traces[method] = (states, accepted)
            for budget in request["budgets"]:
                for method in methods(target):
                    states, accepted = traces[method.removesuffix("-flip")]
                    timings = timing_by_cell[target["id"], method, budget]
                    if len(timings) != request["timing_repeats"]:
                        raise ValueError("timing repeat count mismatch")
                    cells.append(
                        make_cell(
                            target,
                            method,
                            states,
                            accepted,
                            budget,
                            references[target["id"]],
                            timings,
                            request["threshold"],
                        )
                    )
    check_equal(references, result["exact_references"])
    check_equal(cells, result["cells"])
    check_equal(decisions(request, cells), result["decisions"])
    completion = {
        "status": "exploratory_sampling_time_complete",
        "request_digest": result["request_digest"],
        "results_sha256": base.sha(out / "results.json"),
        "trace_sha256": result["trace_sha256"],
        "cells_replayed": len(cells),
        "decisions_replayed": len(result["decisions"]),
        "replay_seconds": time.perf_counter() - started,
        "scope": (
            "source/request/trace authentication, exact and saved empirical fixtures, "
            "initialization, exact references, estimates, accuracy, work, exchange rates, "
            "timing medians and sustained decisions; no historical timing reproduction "
            "or trajectory regeneration; generation-only prefix/repeat checks are not rerun"
        ),
    }
    base.write(out / "completion.json", completion)
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
