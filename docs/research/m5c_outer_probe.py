"""Bounded exact outer-chain comparison of the M5c degree repairs.

Run: OPENBLAS_NUM_THREADS=1 uv run python docs/research/m5c_outer_probe.py --workers 3
JSON lines on stdout: scope, each completed seed/arm, then descriptive summaries.
Exploration only; no fits, sampling, new study gate, or hardware cost model.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from m5c_chain_probe import (
    PROBE_ARMS,
    chained_finite,
    controls,
    input_copies,
    prune_output,
    split_output,
)
from m5c_degree_probe import ARCHIVE_SHA256, arm_name, load_archive, references
from scipy.special import expit

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization_core as core

OUTER_KS = (4, 32, None)  # declared before seeing the outer results; None = equilibrium


def metrics(rates, target):
    """Use M5b's exact sweep, stationary solver, uniform start, and t=30."""
    matrix = core.sweep(np.eye(m5a.STATES), rates)
    law = core.stationary_law(matrix)
    path = core.trajectory(matrix, horizon=30)
    return {
        "bias": float(0.5 * np.abs(law - target).sum()),
        "tv_target_30": float(0.5 * np.abs(path[-1] - target).sum()),
        "tv_own_stationary_30": float(0.5 * np.abs(path[-1] - law).sum()),
        "stationary_residual": float(np.abs(law @ matrix - law).sum()),
        "row_sum_error": float(np.max(np.abs(matrix.sum(axis=1) - 1))),
        "tv_target": (0.5 * np.abs(path - target).sum(axis=1)).tolist(),
    }


def site_rates(p, s, cap):
    """Keep the 49 legal kernels identical; replace only an over-degree site."""
    J, _, _, _, beta = m5a.unpack(p, s)
    repair = np.count_nonzero(J) + np.count_nonzero(beta) > 16
    kernel = core.inner_kernel(p, s)
    original = {k: core.powered_rates(kernel, k) for k in OUTER_KS}
    vectors = {"original": p, "chain": p, "prune": p}
    rates = {"original": original, "chain": original, "prune": original}
    moved = np.zeros(len(beta), bool)
    integrity = {}
    if repair:
        moved = split_output(p, s)
        vectors["prune"], _ = prune_output(p, s)
        pruned = core.inner_kernel(vectors["prune"], s)
        rates["prune"] = {k: core.powered_rates(pruned, k) for k in OUTER_KS}
        x = m5a.blanket_inputs(s)
        _, integrity, chain = chained_finite(
            p,
            s,
            moved,
            cap,
            expit(2 * m5a.kernel_logit(p, s, x)),
            expit(2 * m5a.exact_logit(s, x)),
            include_rates=True,
        )
        rates["chain"] = {k: chain[k] for k in OUTER_KS}
    else:
        assert rates["chain"] is original and rates["prune"] is original
    cost = {}
    for method, vector in vectors.items():
        free = len(beta) + 1 + int(method == "chain" and repair)
        cost[method] = {
            "redraws_per_k": free,
            "clamp_writes": input_copies(vector, s, moved if method == "chain" else None),
            "free_reset_writes": free,
            "readout_bits": 1,
        }
    codes = m5a.blanket_codes(s)
    return (
        {method: {k: r[codes] for k, r in by_k.items()} for method, by_k in rates.items()},
        cost,
        integrity,
    )


def evaluate_base(job):
    arm, ref, finite_baselines = job
    started = time.monotonic()
    reading, _, cap = arm
    seed = ref["cell"]["seed"]
    target = m5a.make_target(seed, reading)
    structures = m5a.structures(target)
    distribution = m5a.target_distribution(target)
    rates = {method: {k: [] for k in OUTER_KS} for method in ("original", "chain", "prune")}
    cost = {
        method: dict.fromkeys(
            ("redraws_per_k", "clamp_writes", "free_reset_writes", "readout_bits"), 0
        )
        for method in rates
    }
    local_residual = 0.0
    for p, s in zip(ref["parameters"], structures, strict=True):
        local, counts, integrity = site_rates(np.asarray(p, dtype=np.float64), s, cap)
        local_residual = max(local_residual, max(integrity.values(), default=0.0))
        for method in rates:
            for k in OUTER_KS:
                rates[method][k].append(local[method][k])
            for key, value in counts[method].items():
                cost[method][key] += value
    rows = []
    baseline_error = 0.0
    for k in OUTER_KS:
        for method in rates:
            result = metrics(rates[method][k], distribution)
            if method == "original":
                archived = (ref if k is None else finite_baselines[k])["metrics"]
                for key, expected in (
                    ("bias", archived["bias"]),
                    ("tv_target", archived["tv_target"]),
                    ("tv_own_stationary_30", archived["tv_own_stationary"][-1]),
                ):
                    m5a._close(result[key], expected, f"M5b outer replay: {arm}/{seed}/{k}/{key}")
                    baseline_error = max(
                        baseline_error, float(np.max(np.abs(np.asarray(result[key]) - expected)))
                    )
            work = dict(cost[method])
            work["spin_redraws_per_outer_sweep"] = None if k is None else k * work["redraws_per_k"]
            work["spin_redraws_through_t30"] = (
                None if k is None else 30 * work["spin_redraws_per_outer_sweep"]
            )
            rows.append({"method": method, "k": k, "metrics": result, "work": work})
    return {
        "arm": arm_name(arm),
        "seed": seed,
        "rows": rows,
        "max_m5b_replay_error": baseline_error,
        "max_local_residual": local_residual,
        "elapsed_seconds": time.monotonic() - started,
    }


def summaries(bases):
    rows = []
    for name in sorted({base["arm"] for base in bases}):
        matching = [base for base in bases if base["arm"] == name]
        for k in OUTER_KS:
            for method in ("original", "chain", "prune"):
                cells = [
                    next(row for row in base["rows"] if row["method"] == method and row["k"] == k)
                    for base in matching
                ]
                summary = {"arm": name, "k": k, "method": method, "seeds": len(cells)}
                for metric in ("bias", "tv_target_30", "tv_own_stationary_30"):
                    summary[metric] = dict(
                        zip(
                            ("min", "median", "max"),
                            np.quantile(
                                [row["metrics"][metric] for row in cells], [0, 0.5, 1]
                            ).tolist(),
                            strict=True,
                        )
                    )
                summary["spin_redraws_per_outer_sweep"] = [
                    row["work"]["spin_redraws_per_outer_sweep"] for row in cells
                ]
                rows.append(summary)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--first-only", action="store_true", help="only B/cap1/seed0, for runtime measurement"
    )
    args = parser.parse_args()
    record = load_archive()
    finite = {
        (
            r["cell"]["reading"],
            r["cell"]["method"],
            r["cell"]["cap"],
            r["cell"]["seed"],
            r["cell"]["value"],
        ): r
        for r in record["results"].values()
        if r["cell"]["arm"] == "finite_k"
    }
    jobs = [
        (arm, ref, {k: finite[(*arm, ref["cell"]["seed"], k)] for k in OUTER_KS if k is not None})
        for arm in PROBE_ARMS
        for ref in references(record, arm)
    ]
    if args.first_only:
        jobs = jobs[:1]
    print(
        json.dumps(
            {
                "status": "exploration",
                "semantics": "exact_reference",
                "archive_sha256": ARCHIVE_SHA256,
                "scope": "first_only" if args.first_only else "five_arms_five_seeds",
                "outer_k": OUTER_KS,
                "outer_states": m5a.STATES,
                "start": "uniform",
                "horizon": 30,
                "controls": controls(),
            }
        ),
        flush=True,
    )
    started = time.monotonic()
    bases = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for future in as_completed([pool.submit(evaluate_base, job) for job in jobs]):
            result = future.result()
            bases.append(result)
            print(json.dumps({"base": result}, allow_nan=False), flush=True)
            print(
                f"{len(bases)}/{len(jobs)} {result['arm']} seed {result['seed']}: "
                f"{result['elapsed_seconds']:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    bases.sort(key=lambda b: (b["arm"], b["seed"]))
    print(
        json.dumps(
            {"summaries": summaries(bases), "elapsed_seconds": time.monotonic() - started},
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
