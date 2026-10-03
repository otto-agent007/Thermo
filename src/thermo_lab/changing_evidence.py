"""Causal changing-evidence experiment using the immutable CPU Gibbs sampler."""

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
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from thermo_lab import fixed_budget_sampling as base
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.provenance import collect_runtime_provenance

ROOT = base.ROOT
METHODS = ("retain-gibbs", "restart-gibbs", "retain-tempering", "restart-tempering")
SOURCES = (
    "src/thermo_lab/changing_evidence.py",
    "docs/experiments/changing-evidence.md",
    "src/thermo_lab/provenance.py",
    "src/thermo_lab/records.py",
    *base.SOURCES,
)


def conditions():
    return [
        {"index": i, "strength": strength, "direction": direction, "schedule": schedule}
        for i, (strength, direction, schedule) in enumerate(
            (s, d, p)
            for s in (0.25, 0.65)
            for d in ("coherent", "checkerboard")
            for p in ("stationary", "gradual", "abrupt", "roundtrip")
        )
    ]


def make_request():
    return {
        "schema": "changing_evidence.v1",
        "evidence_class": "software_simulation",
        "reference_class": "exact_reference",
        "conditions": conditions(),
        "development_seeds": list(range(700, 708)),
        "heldout_seeds": list(range(800, 808)),
        "budgets": [4, 16, 64],
        "policy_budget": 16,
        "timing_repeats": 3,
        "sampling_root": 20261006,
        "initialization_root": 20261007,
        "queries": 25,
        "n": 12,
        "ladder": list(base.LADDER),
        "dtype": "float32 sampling, float64 exact references",
        "failure_threshold": 0.05,
        "policy_threshold": 0.5,
        "predictor_l2": 0.01,
        "replay_atol": 2e-10,
        "sources": {name: base.sha(ROOT / name) for name in SOURCES},
    }


def make_inputs(condition, seeds):
    n = 12
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1
    ]
    forward = np.linspace(0.35, -0.35, 13)
    amplitude = {
        "stationary": np.full(25, 0.35),
        "gradual": np.linspace(0.35, -0.35, 25),
        "abrupt": np.r_[np.full(12, 0.35), np.full(13, -0.35)],
        "roundtrip": np.r_[forward, forward[-2::-1]],
    }[condition["schedule"]]
    direction = (
        np.ones(n)
        if condition["direction"] == "coherent"
        else np.array([(-1) ** (i // 4 + i % 4) for i in range(n)])
    )
    matrices, fields = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        weights = (condition["strength"] * rng.uniform(0.9, 1.1, len(edges))).astype(np.float32)
        offset = rng.uniform(-0.03, 0.03, n)
        matrix = np.zeros((n, n), np.float32)
        for (i, j), weight in zip(edges, weights, strict=True):
            matrix[i, j] = matrix[j, i] = weight / np.float32(4)
        matrices.append(matrix)
        fields.append((offset + amplitude[:, None] * direction).astype(np.float32) / np.float32(4))
    return {
        "fields": np.stack(fields, axis=1),
        "matrices": np.asarray(matrices),
        "edges": edges,
        "strengths": 4
        * np.asarray(matrices)[:, tuple(np.array(edges).T)[0], tuple(np.array(edges).T)[1]].mean(
            axis=1
        ),
    }


def make_sampler(algorithm, n, budget):
    original = base.compile_sampler("independent" if algorithm == "gibbs" else algorithm, n, budget)

    def single(key, initial, fields, matrix):
        states, accepted = original(key[None], initial[None], fields, matrix)
        return states[0], accepted[0], states[0, -1]

    return jax.jit(jax.vmap(single))


def observable_matrix(states, edges):
    spins = 2 * states.astype(float) - 1
    edge = np.array(edges)
    return np.column_stack(
        [
            states.astype(float),
            spins[:, edge[:, 0]] * spins[:, edge[:, 1]],
            (states.sum(axis=1) >= 8).astype(float),
        ]
    )


def references(inputs, repeats):
    start = time.perf_counter()
    n = inputs["matrices"].shape[-1]
    states = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(bool)
    spins = 2 * states.astype(float) - 1
    prior = 2 * np.einsum("ki,sij,kj->sk", spins, inputs["matrices"].astype(float), spins)
    observables = observable_matrix(states, inputs["edges"])
    setup_seconds = time.perf_counter() - start
    timings, result, probabilities = [], None, None
    # The first complete pass warms the exact-query pipeline; only subsequent
    # passes are timed evidence. Reference setup remains separately visible.
    for repeat in range(repeats + 1):
        output, all_p, observed = [], [], []
        for fields in inputs["fields"]:
            before = time.perf_counter()
            score = 4 * fields.astype(float) @ spins.T + prior
            p = np.exp(score - score.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            moments = p @ observables
            estimates = {
                "marginal": moments[:, :n],
                "edge": moments[:, n:-1],
                "alarm": moments[:, -1],
            }
            observed.append(time.perf_counter() - before)
            output.append(estimates)
            all_p.append(p)
        current = {name: np.array([row[name] for row in output]) for name in output[0]}
        if result is not None:
            for name in result:
                if not np.array_equal(current[name], result[name]):
                    raise ValueError("exact-query repeats differ")
        result, probabilities = current, np.array(all_p)
        if repeat:
            timings.append(observed)
    result["target_tv"] = np.r_[
        np.zeros((1, len(prior))), 0.5 * np.abs(np.diff(probabilities, axis=0)).sum(axis=-1)
    ]
    native = 4 * inputs["fields"].astype(float)
    result["input_rms"] = np.r_[
        np.zeros((1, len(prior))), np.sqrt(np.mean(np.diff(native, axis=0) ** 2, axis=-1))
    ]
    return result, probabilities, {"setup_seconds": setup_seconds, "query_seconds": timings}


def retained(states, algorithm, budget):
    kept = states[:, budget // 4 + 1 : budget + 1]
    if algorithm == "tempering":
        kept = kept[:, :, -1:]
    return kept.reshape(len(states), -1, states.shape[-1])


def estimates(states, algorithm, budget, edges):
    kept = retained(states, algorithm, budget)
    edge = np.asarray(edges)
    spins = 2 * kept.astype(float) - 1
    return {
        "marginal": kept.mean(axis=1),
        "edge": (spins[:, :, edge[:, 0]] * spins[:, :, edge[:, 1]]).mean(axis=1),
        "alarm": (kept.sum(axis=-1) >= 8).mean(axis=1),
    }


def features(fields, previous_fields, previous_probability, strengths, query):
    delta = np.asarray(fields) - previous_fields
    return np.column_stack(
        [
            np.sqrt(np.mean(delta**2, axis=1)),
            -np.mean(delta * (2 * previous_probability - 1), axis=1),
            np.mean(4 * previous_probability * (1 - previous_probability), axis=1),
            strengths,
            np.full(len(delta), 1 / (query + 1)),
        ]
    )


def regret(estimated, exact):
    q = np.asarray(exact)
    risk = np.where(np.asarray(estimated) >= 0.2, 1 - q, 4 * q)
    return risk - np.minimum(1 - q, 4 * q)


def fit_predictor(x, y, columns):
    x, y = np.asarray(x)[:, columns], np.asarray(y, float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-12, 1, scale)
    model = {
        "columns": columns,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "prevalence": float(y.mean()),
    }
    if np.all(y == y[0]):
        return {**model, "constant": float(y[0])}
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])

    def objective(coef):
        logits = design @ coef
        loss = np.mean(np.logaddexp(0, logits) - y * logits) + 0.005 * np.sum(coef[1:] ** 2)
        gradient = design.T @ (expit(logits) - y) / len(y)
        gradient[1:] += 0.01 * coef[1:]
        return loss, gradient

    fit = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not fit.success:
        raise ValueError(f"predictor fit failed: {fit.message}")
    return {**model, "coef": fit.x.tolist(), "iterations": int(fit.nit)}


def predict(model, x):
    if "constant" in model:
        return np.full(len(x), model["constant"])
    values = (np.asarray(x)[:, model["columns"]] - model["mean"]) / model["scale"]
    return expit(np.column_stack([np.ones(len(values)), values]) @ model["coef"])


def prediction_metrics(y, probability):
    y, probability = np.asarray(y, bool), np.asarray(probability)
    positive, negative = int(y.sum()), int((~y).sum())
    decision = probability >= 0.5
    return {
        "count": len(y),
        "prevalence": float(y.mean()),
        "auc": (
            float(
                (rankdata(probability)[y].sum() - positive * (positive + 1) / 2)
                / (positive * negative)
            )
            if positive and negative
            else None
        ),
        "brier": float(np.mean((probability - y) ** 2)),
        "precision": float(y[decision].mean()) if decision.any() else None,
        "recall": float(decision[y].mean()) if positive else None,
        "alarm_fraction": float(decision.mean()),
    }


def initial_states(request, condition, seeds, query):
    return np.array(
        [
            np.random.default_rng(
                np.random.SeedSequence(
                    [request["initialization_root"], seed, condition["index"], query]
                )
            )
            .integers(0, 2, (5, request["n"]))
            .astype(bool)
            for seed in seeds
        ]
    )


def planned_keys(request, condition, seeds, algorithm):
    keys = []
    for query in range(request["queries"]):
        row = []
        for seed in seeds:
            key = jax.random.key(request["sampling_root"])
            for value in (seed, condition["index"], query, 1 if algorithm == "gibbs" else 2):
                key = jax.random.fold_in(key, value)
            row.append(key)
        keys.append(jnp.stack(row))
    return keys


def reset_mask(method, query, feature, policy, threshold):
    if query == 0 or method.startswith("restart-"):
        return np.ones(len(feature), bool)
    if method.startswith("policy-"):
        return predict(policy, feature) >= threshold
    return np.zeros(len(feature), bool)


def run_sequence(request, condition, seeds, inputs, method, budget, fn, keys, policy=None):
    algorithm = method.split("-")[-1]
    matrix_device = jnp.asarray(inputs["matrices"])
    jax.block_until_ready(matrix_device)
    resident = None
    previous = np.full((len(seeds), request["n"]), 0.5)
    native = 4 * inputs["fields"].astype(float)
    all_states, all_accepted, timings = [], [], []
    for query in range(request["queries"]):
        start = time.perf_counter()
        feature = features(
            native[query], native[max(0, query - 1)], previous, inputs["strengths"], query
        )
        mask = reset_mask(method, query, feature, policy, request["policy_threshold"])
        if mask.any():
            fresh = jnp.asarray(initial_states(request, condition, seeds, query))
            resident = (
                fresh
                if resident is None
                else jnp.where(jnp.asarray(mask)[:, None, None], fresh, resident)
            )
        device_states, device_accepted, resident = jax.block_until_ready(
            fn(keys[query], resident, jnp.asarray(inputs["fields"][query]), matrix_device)
        )
        states, accepted = np.asarray(device_states), np.asarray(device_accepted)
        estimate = estimates(states, algorithm, budget, inputs["edges"])
        previous = estimate["marginal"]
        timings.append(time.perf_counter() - start)
        all_states.append(states)
        all_accepted.append(accepted)
    return np.array(all_states), np.array(all_accepted), timings


def score_cell(
    request,
    condition,
    seeds,
    split,
    method,
    budget,
    states,
    accepted,
    inputs,
    exact,
    probabilities,
    timings,
    policy=None,
):
    algorithm = method.split("-")[-1]
    q_count, streams, n = request["queries"], len(seeds), request["n"]
    if states.shape != (q_count, streams, budget + 1, 5, n):
        raise ValueError("state trace shape mismatch")
    if accepted.shape != (q_count, streams, budget, 2):
        raise ValueError("exchange trace shape mismatch")
    if algorithm == "gibbs" and accepted.any():
        raise ValueError("Gibbs trace cannot contain exchanges")
    previous = np.full((streams, n), 0.5)
    native = 4 * inputs["fields"].astype(float)
    estimates_by_query, all_features, masks = [], [], []
    metrics = {
        name: []
        for name in (
            "marginal_mae",
            "full_joint_tv",
            "edge_mae",
            "alarm_error",
            "regret",
            "confident_wrong",
        )
    }
    for query in range(q_count):
        feature = features(
            native[query], native[max(0, query - 1)], previous, inputs["strengths"], query
        )
        mask = reset_mask(method, query, feature, policy, request["policy_threshold"])
        initial = initial_states(request, condition, seeds, query)
        if query:
            initial = np.where(mask[:, None, None], initial, states[query - 1, :, -1])
        if not np.array_equal(states[query, :, 0], initial):
            raise ValueError("retention/reset initialization mismatch")
        estimate = estimates(states[query], algorithm, budget, inputs["edges"])
        previous = estimate["marginal"]
        estimates_by_query.append(estimate)
        all_features.append(feature)
        masks.append(mask)
        for name, key in (("marginal_mae", "marginal"), ("edge_mae", "edge")):
            metrics[name].append(np.abs(estimate[key] - exact[key][query]).mean(axis=1))
        metrics["alarm_error"].append(np.abs(estimate["alarm"] - exact["alarm"][query]))
        metrics["regret"].append(regret(estimate["alarm"], exact["alarm"][query]))
        metrics["confident_wrong"].append(
            ((estimate["alarm"] <= 0.05) & (exact["alarm"][query] >= 0.95))
            | ((estimate["alarm"] >= 0.95) & (exact["alarm"][query] <= 0.05))
        )
        kept = retained(states[query], algorithm, budget)
        codes = kept.astype(int) @ (1 << np.arange(n))
        full_tv = []
        for index, observed in enumerate(codes):
            unique, count = np.unique(observed, return_counts=True)
            full_tv.append(
                1 - np.minimum(count / len(observed), probabilities[query, index, unique]).sum()
            )
        metrics["full_joint_tv"].append(full_tv)
    timing = np.asarray(timings)
    if timing.shape != (request["timing_repeats"], q_count) or not np.all(timing > 0):
        raise ValueError("invalid timings")
    marginal = np.array([x["marginal"] for x in estimates_by_query])
    mae = np.array(metrics["marginal_mae"])
    recovery = []
    if condition["schedule"] == "abrupt":
        for stream in range(streams):
            eligible = [
                t - 12
                for t in range(12, q_count)
                if np.all(mae[t:, stream] <= request["failure_threshold"])
            ]
            recovery.append(eligible[0] if eligible else None)
    return {
        "split": split,
        "condition": condition,
        "seeds": seeds,
        "method": method,
        "budget": budget,
        "metrics": {name: np.asarray(value).tolist() for name, value in metrics.items()},
        "marginal": marginal.tolist(),
        "alarm": np.array([x["alarm"] for x in estimates_by_query]).tolist(),
        "features": np.asarray(all_features).tolist(),
        "resets": np.asarray(masks).tolist(),
        "recovery_queries": recovery,
        "hysteresis_marginal_mae": (
            np.abs(marginal[:12] - marginal[24:12:-1]).mean(axis=(0, 2)).tolist()
            if condition["schedule"] == "roundtrip"
            else []
        ),
        "work_per_stream": {
            "spin_redraws": 5 * budget * n * q_count,
            "swap_attempts": 2 * budget * q_count if algorithm == "tempering" else 0,
            "initialized_spins": (np.sum(masks, axis=0) * 5 * n).tolist(),
            "changed_fields": (n + np.count_nonzero(np.diff(native, axis=0), axis=(0, 2))).tolist(),
            "trace_bits_returned": q_count * (budget + 1) * 5 * n,
            "cold_observations": q_count
            * (3 * budget // 4)
            * (1 if algorithm == "tempering" else 5),
        },
        "exchange_acceptance": float(accepted.mean()) if algorithm == "tempering" else None,
        "query_seconds": timings,
        "warm_batch_seconds": float(np.median(timing.sum(axis=1))),
    }


def fit_policies(cells, policy_budget):
    result = {}
    for algorithm in ("gibbs", "tempering"):
        chosen = [
            c
            for c in cells
            if c["split"] == "development"
            and c["method"] == f"retain-{algorithm}"
            and c["budget"] == policy_budget
        ]
        x = np.concatenate([np.array(c["features"]).reshape(-1, 5) for c in chosen])
        y = np.concatenate([np.array(c["metrics"]["marginal_mae"]).ravel() > 0.05 for c in chosen])
        result[algorithm] = {
            "state_aware": fit_predictor(x, y, [0, 1, 2, 3, 4]),
            "input_only": fit_predictor(x, y, [0, 3, 4]),
        }
    return result


def evaluate_predictions(cells, policies, policy_budget):
    result = {}
    for algorithm, models in policies.items():
        chosen = [
            c
            for c in cells
            if c["split"] == "heldout"
            and c["method"] == f"retain-{algorithm}"
            and c["budget"] == policy_budget
        ]
        x = np.concatenate([np.array(c["features"]).reshape(-1, 5) for c in chosen])
        y = np.concatenate([np.array(c["metrics"]["marginal_mae"]).ravel() > 0.05 for c in chosen])
        result[algorithm] = {}
        for name, model in models.items():
            per_stream = []
            for stream in range(len(chosen[0]["seeds"])):
                sx = np.concatenate([np.array(c["features"])[:, stream] for c in chosen])
                sy = np.concatenate(
                    [np.array(c["metrics"]["marginal_mae"])[:, stream] > 0.05 for c in chosen]
                )
                per_stream.append(prediction_metrics(sy, predict(model, sx)))
            result[algorithm][name] = {
                "pooled_descriptive": prediction_metrics(y, predict(model, x)),
                "independent_streams": per_stream,
            }
    return result


def check_equal(actual, expected, path="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"keys differ: {path}")
        for key in expected:
            check_equal(actual[key], expected[key], path + "/" + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"length differs: {path}")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            check_equal(a, b, path + f"/{index}")
    elif isinstance(expected, float):
        if (
            not isinstance(actual, (float, int))
            or not np.isfinite(actual)
            or abs(actual - expected) > 2e-10
        ):
            raise ValueError(f"value differs: {path}")
    elif actual != expected:
        raise ValueError(f"value differs: {path}")


def cell_key(split, condition, method, budget):
    return f"{split}__c{condition['index']}__{method}__T{budget}"


def validate_request(request):
    if request["queries"] != 25 or request["n"] != 12:
        raise ValueError("model and schedule dimensions are frozen")
    if request["ladder"] != list(base.LADDER) or request["failure_threshold"] != 0.05:
        raise ValueError("ladder and failure criterion are frozen")
    if request["policy_threshold"] != 0.5 or request["predictor_l2"] != 0.01:
        raise ValueError("policy fit and threshold are frozen")
    if set(request["development_seeds"]) & set(request["heldout_seeds"]):
        raise ValueError("development and held-out seeds overlap")
    if request["policy_budget"] not in request["budgets"]:
        raise ValueError("policy budget must have matched baselines")
    if any(t <= 0 or t % 4 for t in request["budgets"]):
        raise ValueError("positive budgets divisible by four required")
    if request["replay_atol"] != 2e-10:
        raise ValueError("replay tolerance is frozen")


def run_study(out, requested=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("use a fresh output directory")
    request = make_request() if requested is None else requested
    validate_request(request)
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise ValueError("use CPU with JAX_ENABLE_X64=false")
    out.mkdir(parents=True)
    base.write(out / "request.json", request)
    for name, digest in request["sources"].items():
        if base.sha(ROOT / name) != digest:
            raise ValueError(f"changed source: {name}")
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    started = time.perf_counter()
    fixture, fixture_traces = base.validate_fixture()
    packed = {
        name: np.packbits(x, axis=-1, bitorder="little") for name, x in fixture_traces.items()
    }
    cells, exact_results, exact_timing, setup, compilation = [], {}, {}, {}, {}
    cache, policies = {}, None
    for split in ("development", "heldout"):
        seeds = request[split + "_seeds"]
        for condition in request["conditions"]:
            before = time.perf_counter()
            inputs = make_inputs(condition, seeds)
            keys = {a: planned_keys(request, condition, seeds, a) for a in ("gibbs", "tempering")}
            jax.block_until_ready(keys)
            condition_key = f"{split}__c{condition['index']}"
            setup[condition_key] = time.perf_counter() - before
            exact, probability, measured_exact = references(inputs, request["timing_repeats"])
            exact_results[condition_key] = {name: x.tolist() for name, x in exact.items()}
            exact_timing[condition_key] = measured_exact
            for budget in request["budgets"]:
                methods = list(METHODS)
                if split == "heldout" and budget == request["policy_budget"]:
                    methods += ["policy-gibbs", "policy-tempering"]
                warm, timings = {}, {m: [] for m in methods}
                for method in methods:
                    algorithm = method.split("-")[-1]
                    cache_key = (algorithm, budget, len(seeds))
                    if cache_key not in cache:
                        before = time.perf_counter()
                        cache[cache_key] = (
                            make_sampler(algorithm, request["n"], budget)
                            .lower(
                                keys[algorithm][0],
                                jnp.asarray(initial_states(request, condition, seeds, 0)),
                                jnp.asarray(inputs["fields"][0]),
                                jnp.asarray(inputs["matrices"]),
                            )
                            .compile()
                        )
                        compilation[str(cache_key)] = time.perf_counter() - before
                    policy = policies[algorithm]["state_aware"] if policies else None
                    states, accepted, _ = run_sequence(
                        request,
                        condition,
                        seeds,
                        inputs,
                        method,
                        budget,
                        cache[cache_key],
                        keys[algorithm],
                        policy,
                    )
                    warm[method] = states, accepted
                for repeat in range(request["timing_repeats"]):
                    rotation = repeat % len(methods)
                    for method in methods[rotation:] + methods[:rotation]:
                        algorithm = method.split("-")[-1]
                        policy = policies[algorithm]["state_aware"] if policies else None
                        states, accepted, measured = run_sequence(
                            request,
                            condition,
                            seeds,
                            inputs,
                            method,
                            budget,
                            cache[algorithm, budget, len(seeds)],
                            keys[algorithm],
                            policy,
                        )
                        if not np.array_equal(states, warm[method][0]) or not np.array_equal(
                            accepted, warm[method][1]
                        ):
                            raise ValueError("warm/timed repeated trajectories differ")
                        timings[method].append(measured)
                for method in methods:
                    algorithm = method.split("-")[-1]
                    policy = policies[algorithm]["state_aware"] if policies else None
                    states, accepted = warm[method]
                    key = cell_key(split, condition, method, budget)
                    packed[key] = np.packbits(states, axis=-1, bitorder="little")
                    packed[key + "__accepted"] = np.packbits(accepted, axis=-1, bitorder="little")
                    cells.append(
                        score_cell(
                            request,
                            condition,
                            seeds,
                            split,
                            method,
                            budget,
                            states,
                            accepted,
                            inputs,
                            exact,
                            probability,
                            timings[method],
                            policy,
                        )
                    )
                for algorithm in ("gibbs", "tempering"):
                    if not np.array_equal(
                        warm["retain-" + algorithm][0][0], warm["restart-" + algorithm][0][0]
                    ):
                        raise ValueError("first-query retention/restart trajectories differ")
            print(
                f"Measured {split} condition {condition['index']}: {condition['strength']} "
                f"{condition['direction']} {condition['schedule']}",
                flush=True,
            )
        if split == "development":
            base.write(out / "development.json", {"cells": cells})
            policies = fit_policies(cells, request["policy_budget"])
            base.write(out / "policy.json", policies)
            print("Development policy fitted and frozen before held-out execution", flush=True)
    np.savez_compressed(out / "traces.npz", **packed)
    result = {
        "request_digest": canonical_sha256(request),
        "trace_sha256": base.sha(out / "traces.npz"),
        "development_sha256": base.sha(out / "development.json"),
        "policy_sha256": base.sha(out / "policy.json"),
        "fixture": fixture,
        "cells": cells,
        "exact_references": exact_results,
        "exact_timing": exact_timing,
        "setup_seconds": setup,
        "compile_seconds": compilation,
        "prediction": evaluate_predictions(cells, policies, request["policy_budget"]),
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
    start = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    validate_request(request)
    if canonical_sha256(request) != result["request_digest"]:
        raise ValueError("request hash mismatch")
    for artifact, key in (
        ("traces.npz", "trace_sha256"),
        ("development.json", "development_sha256"),
        ("policy.json", "policy_sha256"),
    ):
        if base.sha(out / artifact) != result[key]:
            raise ValueError(f"artifact hash mismatch: {artifact}")
    for name, digest in request["sources"].items():
        if base.sha(out / "source" / name) != digest:
            raise ValueError(f"archived source changed: {name}")
        if name.endswith(".py") and base.sha(ROOT / name) != digest:
            raise ValueError(f"loaded source changed: {name}")
    development = json.loads((out / "development.json").read_text())["cells"]
    policies = json.loads((out / "policy.json").read_text())
    times = {
        cell_key(c["split"], c["condition"], c["method"], c["budget"]): c["query_seconds"]
        for c in result["cells"]
    }
    if len(times) != len(result["cells"]):
        raise ValueError("duplicate cells")
    cells, references_replayed = [], {}
    with np.load(out / "traces.npz", allow_pickle=False) as archive:
        fixture = {
            f"fixture__{m}": np.unpackbits(
                archive[f"fixture__{m}"], axis=-1, count=3, bitorder="little"
            ).astype(bool)
            for m in base.METHODS
        }
        check_equal(base.fixture_metrics(fixture), result["fixture"], "fixture")
        for split in ("development", "heldout"):
            seeds = request[split + "_seeds"]
            for condition in request["conditions"]:
                inputs = make_inputs(condition, seeds)
                exact, probability, _ = references(inputs, repeats=0)
                references_replayed[f"{split}__c{condition['index']}"] = {
                    k: v.tolist() for k, v in exact.items()
                }
                for budget in request["budgets"]:
                    methods = list(METHODS)
                    if split == "heldout" and budget == request["policy_budget"]:
                        methods += ["policy-gibbs", "policy-tempering"]
                    for method in methods:
                        key = cell_key(split, condition, method, budget)
                        states = np.unpackbits(
                            archive[key], axis=-1, count=request["n"], bitorder="little"
                        ).astype(bool)
                        accepted = np.unpackbits(
                            archive[key + "__accepted"], axis=-1, count=2, bitorder="little"
                        ).astype(bool)
                        policy = (
                            policies[method.split("-")[-1]]["state_aware"]
                            if split == "heldout"
                            else None
                        )
                        cells.append(
                            score_cell(
                                request,
                                condition,
                                seeds,
                                split,
                                method,
                                budget,
                                states,
                                accepted,
                                inputs,
                                exact,
                                probability,
                                times[key],
                                policy,
                            )
                        )
            if split == "development":
                check_equal(cells, development, "development")
                check_equal(fit_policies(cells, request["policy_budget"]), policies, "policy")
    check_equal(references_replayed, result["exact_references"], "references")
    check_equal(cells, result["cells"], "cells")
    check_equal(
        evaluate_predictions(cells, policies, request["policy_budget"]),
        result["prediction"],
        "prediction",
    )
    completion = {
        "status": "changing_evidence_complete",
        "request_digest": result["request_digest"],
        "results_sha256": base.sha(out / "results.json"),
        "trace_sha256": result["trace_sha256"],
        "cells_replayed": len(cells),
        "query_estimates_replayed": sum(request["queries"] * len(c["seeds"]) for c in cells),
        "replay_seconds": time.perf_counter() - start,
        "scope": (
            "authenticated sources/artifacts, exact and saved empirical fixtures, initialization, "
            "continuity/resets, references, every error/feature/action/count, timing medians, "
            "development-only policy refits, held-out predictions; "
            "no historical timing or trajectory regeneration"
        ),
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
