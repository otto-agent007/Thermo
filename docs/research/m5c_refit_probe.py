"""Refit only the 11 B/variational/cap-1 outputs above degree 16.

Exploration, exact_reference; no new objective, archive, study runner, or gate.
Run: OPENBLAS_NUM_THREADS=1 uv run python docs/research/m5c_refit_probe.py --workers 3
Budget sensitivity: add --maxiter 20000 (maxfun scales as 10 x maxiter, as in M5a).
JSON lines: fixed scope, each completed seed, then descriptive summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from m5c_chain_probe import controls, input_copies
from m5c_degree_probe import ARCHIVE_SHA256, ROOT, load_archive, references
from m5c_outer_probe import OUTER_KS, evaluate_base, metrics, summaries
from scipy.optimize import minimize
from scipy.special import expit

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization_core as core

ARM = ("B", "variational", 1.0)
METHODS = ("original", "chain", "prune", "j_prune", "refit")
PRIMARY_K = 4
USEFUL = 8.3e-3
SMALL_COST = 5e-4
DEFAULT_MAXITER = m5a.OPTIMIZER["maxiter"]


def j_mask(p, s):
    """Freeze the smallest archived nonzero J edges; never mask beta."""
    J, _, _, _, beta = m5a.unpack(p, s)
    count = max(0, int(np.count_nonzero(J) + np.count_nonzero(beta)) - 16)
    candidates = sorted((abs(value), i) for i, value in enumerate(J) if value != 0)
    assert count <= len(candidates)
    return [i for _, i in candidates[:count]]


def fit_masked(p, s, seed, mask, maxiter):
    x = m5a.blanket_inputs(s)
    logit = m5a.exact_logit(s, x)
    starts = m5a._starts(s, ARM[2], ARM[0], seed, s["site"])
    starts.append(p.copy())
    assert len(starts) == 9
    bounds = [(0.0, 0.0) if i in mask else (-1.0, 1.0) for i in range(len(p))]
    options = optimizer_options(maxiter)
    attempts, endpoints = [], []
    for start in starts:
        start[mask] = 0.0
        result = minimize(
            m5a.objective,
            start,
            args=(s, x, logit),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
        endpoint = np.clip(np.asarray(result.x, dtype=np.float64), -1.0, 1.0)
        assert np.isfinite(endpoint).all() and np.all(endpoint[mask] == 0.0)
        attempts.append(
            {
                "initial_objective": m5a.objective(start, s, x, logit)[0],
                "objective": m5a.objective(endpoint, s, x, logit)[0],
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "success": bool(result.success),
                "termination": str(result.message),
            }
        )
        endpoints.append(endpoint)
    selected = min(range(len(attempts)), key=lambda i: (attempts[i]["objective"], i))
    return endpoints[selected], {"selected": selected, "attempts": attempts}


def optimizer_options(maxiter):
    """M5a's options, with only the iteration and evaluation limits scaled."""
    assert m5a.OPTIMIZER["maxfun"] == 10 * m5a.OPTIMIZER["maxiter"]
    return {**m5a.OPTIMIZER, "maxiter": maxiter, "maxfun": 10 * maxiter}


def local_diagnostics(p, s):
    x = m5a.blanket_inputs(s)
    kernel = core.inner_kernel(p, s)
    probability = expit(2 * m5a.kernel_logit(p, s, x))
    brute_error = float(np.max(np.abs(probability - m5a.brute_force_probability(p, s, x))))
    assert brute_error < 2e-12
    equilibrium = core.powered_rates(kernel, None)
    J, _, _, _, beta = m5a.unpack(p, s)
    return {
        "objective": m5a.objective(p, s, x, m5a.exact_logit(s, x))[0],
        "output_degree": int(np.count_nonzero(J) + np.count_nonzero(beta)),
        "hidden_count": len(beta),
        "zero_beta_count": int(np.count_nonzero(beta == 0)),
        "brute_force_error": brute_error,
        "mixing": core.mixing_summary(kernel),
        "max_finite_vs_equilibrium": {
            str(k): float(np.max(np.abs(core.powered_rates(kernel, k) - equilibrium)))
            for k in OUTER_KS
            if k is not None
        },
    }


def evaluate_refit(job, maxiter=DEFAULT_MAXITER):
    started = time.monotonic()
    _, ref, _ = job
    seed = ref["cell"]["seed"]
    target = m5a.make_target(seed, ARM[0])
    structures = m5a.structures(target)
    original = [np.asarray(p, dtype=np.float64) for p in ref["parameters"]]
    vectors = {method: [p.copy() for p in original] for method in ("j_prune", "refit")}
    fits = []
    for p, s in zip(original, structures, strict=True):
        mask = j_mask(p, s)
        if not mask:
            continue
        site = s["site"]
        pruned = vectors["j_prune"][site]
        pruned[mask] = 0.0
        # A mask affects only J, leaving every other archived coordinate identical.
        assert np.array_equal(pruned[len(s["blanket"]) :], p[len(s["blanket"]) :])
        fitted, attempts = fit_masked(p, s, seed, mask, maxiter)
        vectors["refit"][site] = fitted
        diagnostics = {
            method: local_diagnostics(q, s)
            for method, q in (("original", p), ("j_prune", pruned), ("refit", fitted))
        }
        assert all(diagnostics[method]["output_degree"] <= 16 for method in vectors)
        fits.append(
            {
                "site": site,
                "masked_j_indices": mask,
                "masked_input_sites": [s["blanket"][i] for i in mask],
                "parameters": fitted.tolist(),
                **attempts,
                "diagnostics": diagnostics,
            }
        )
        print(f"seed {seed} site {site}: nine starts complete", file=sys.stderr, flush=True)
    changed = {fit["site"] for fit in fits}
    for method in vectors:
        for site, p in enumerate(original):
            if site not in changed:
                assert vectors[method][site].tobytes() == p.tobytes()
    fit_seconds = time.monotonic() - started
    print(f"seed {seed}: exact outer evaluation", file=sys.stderr, flush=True)
    base = evaluate_base(job)  # original / legacy prune / chain, including archive replay
    distribution = m5a.target_distribution(target)
    for method, parameters in vectors.items():
        local_rates = {k: [] for k in OUTER_KS}
        work = dict.fromkeys(
            ("redraws_per_k", "clamp_writes", "free_reset_writes", "readout_bits"), 0
        )
        for p, s in zip(parameters, structures, strict=True):
            kernel = core.inner_kernel(p, s)
            codes = m5a.blanket_codes(s)
            for k in OUTER_KS:
                local_rates[k].append(core.powered_rates(kernel, k)[codes])
            free = len(s["triples"]) + 1
            work["redraws_per_k"] += free
            work["free_reset_writes"] += free
            work["clamp_writes"] += input_copies(p, s)
            work["readout_bits"] += 1
        for k in OUTER_KS:
            counts = dict(work)
            counts["spin_redraws_per_outer_sweep"] = (
                None if k is None else k * work["redraws_per_k"]
            )
            counts["spin_redraws_through_t30"] = (
                None if k is None else 30 * counts["spin_redraws_per_outer_sweep"]
            )
            base["rows"].append(
                {
                    "method": method,
                    "k": k,
                    "metrics": metrics(local_rates[k], distribution),
                    "work": counts,
                }
            )
    base.update(
        fits=fits,
        unchanged_kernels=12 - len(fits),
        fit_seconds=fit_seconds,
        elapsed_seconds=time.monotonic() - started,
    )
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    args = parser.parse_args()
    record = load_archive()
    for name, expected in record["request"]["implementation_sha256"].items():
        actual = hashlib.sha256((ROOT / "src/thermo_lab" / name).read_bytes()).hexdigest()
        assert actual == expected, f"archive-bound source changed: {name}"
    finite = {
        (r["cell"]["seed"], r["cell"]["value"]): r
        for r in record["results"].values()
        if r["cell"]["arm"] == "finite_k"
        and (r["cell"]["reading"], r["cell"]["method"], r["cell"]["cap"]) == ARM
    }
    jobs = [
        (ARM, ref, {k: finite[ref["cell"]["seed"], k] for k in OUTER_KS if k is not None})
        for ref in references(record, ARM)
    ]
    print(
        json.dumps(
            {
                "status": "exploration",
                "semantics": "exact_reference",
                "dtype": "float64",
                "archive_sha256": ARCHIVE_SHA256,
                "arm": ARM,
                "fits": 11,
                "starts_per_fit": 9,
                "mask": "smallest archived nonzero J only; tie by J index",
                "primary_k": PRIMARY_K,
                "primary_metrics": ["bias", "tv_target_30"],
                "aggregation": "five-seed medians; both metrics strictly below threshold",
                "useful_threshold": USEFUL,
                "small_cost_threshold": SMALL_COST,
                "methods": METHODS,
                "outer_k": OUTER_KS,
                "outer_states": m5a.STATES,
                "start": "uniform",
                "horizon": 30,
                "optimizer": optimizer_options(args.maxiter),
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
        for future in as_completed(
            [pool.submit(evaluate_refit, job, args.maxiter) for job in jobs]
        ):
            base = future.result()
            bases.append(base)
            print(json.dumps({"base": base}, allow_nan=False), flush=True)
            print(f"{len(bases)}/5 seeds complete", file=sys.stderr, flush=True)
    bases.sort(key=lambda base: base["seed"])
    assert sum(len(base["fits"]) for base in bases) == 11
    assert sum(base["unchanged_kernels"] for base in bases) == 49
    summary = summaries(bases, methods=METHODS)
    primary = next(row for row in summary if row["method"] == "refit" and row["k"] == PRIMARY_K)
    values = [primary[key]["median"] for key in ("bias", "tv_target_30")]
    print(
        json.dumps(
            {
                "summaries": summary,
                "primary_useful": max(values) < USEFUL,
                "primary_small_cost": max(values) < SMALL_COST,
                "elapsed_seconds": time.monotonic() - started,
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
