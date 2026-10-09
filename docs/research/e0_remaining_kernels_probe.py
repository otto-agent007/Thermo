"""Probe for queue row `e0-remaining-kernels` (proposal P-0003). Exploration only.

Not recorded evidence. No archive, no replay, no gate. It answers two sizing
questions before a protocol exists:

1. Exact side, all 60 compiled M5a kernels (reading B, variational, cap 1,
   seeds 0-4, sites 0-11): under stage B's fixed N = 65,536 and its 0.999
   output tolerance, which of stage B's four wrong-reference controls can
   separate from the right law at K in {1, 2, 4}? Stage B's runner stops the
   whole study if any control fails to separate, so this decides whether its
   contract can be applied unchanged to every kernel.
2. Cost: exact-side time per kernel (joint laws and output tolerances), the
   joint-tolerance draw time on the largest kernel, and THRML execution time
   per input for a spread of kernel shapes at stage B's N, so the production
   run can be sized.

The exact side imports stage B's study-local helpers and the hash-bound M5
modules; nothing here edits them. THRML timings are `software_simulation`
wall-clock on CPU (JAX synchronized, compile separated); none of it is a
device operation or hardware evidence.

Run:
    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run python docs/research/e0_remaining_kernels_probe.py \
      --output docs/research/2026-10-09-e0-remaining-kernels-probe.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization_core as core
from thermo_lab import meta_ebm_topology as m5c
from thermo_lab import thrml_m5a_kernel_inner_sweep as sb

SWEEPS = sb.SWEEPS
INCOMING = sb.INCOMING
CHAINS = 65_536
# probe-only tolerance seed, distinct from stage B's 20261004
PROBE_NUMPY_SEED = 9_900
TIMING_KERNELS = ((4, 6), (3, 4), (0, 1), (1, 9), (2, 5))  # (seed, site)
TIMING_BATCHES = 2  # batches of 16 inputs per timed cell


def exact_side(seed: int, site: int, sources: dict) -> dict:
    structure = m5a.structures(m5a.make_target(seed, "B"))[site]
    vector = np.asarray(sources[seed]["parameters"][site], dtype=np.float64)
    inputs = m5a.blanket_inputs(structure)
    nh = len(structure["triples"])
    inner = core.inner_kernel(vector, structure)
    mix = core.mixing_summary(inner)
    t0 = time.perf_counter()
    laws = {}
    for name, order, sign in (
        ("hidden_first", "hidden_first", 1.0),
        ("output_first", "output_first", 1.0),
        ("negated_inputs", "hidden_first", -1.0),
    ):
        laws[name], _ = sb._joint_laws(vector, structure, inputs, order, sign, max(SWEEPS) + 1)
    joint_seconds = time.perf_counter() - t0
    y_plus = sb.joint_states(nh)[:, nh] > 0
    spec = {"quantile": 0.999, "draws_output": 4000, "numpy_seed": PROBE_NUMPY_SEED}
    cells, tol_seconds, agreement = [], 0.0, 0.0
    for index, (y0, k) in enumerate((y0, k) for y0 in INCOMING for k in SWEEPS):
        out = sb.output_law(inner, k, y0)
        joint = laws["hidden_first"][(y0, k)]
        agreement = max(agreement, float(np.max(np.abs(joint[:, y_plus].sum(1) - out))))
        t1 = time.perf_counter()
        tol = sb._output_tolerance(out, CHAINS, spec, index)
        tol_seconds += time.perf_counter() - t1
        wrong = {
            "off_by_one": laws["hidden_first"][(y0, k + 1)][:, y_plus].sum(1),
            "output_first": laws["output_first"][(y0, k)][:, y_plus].sum(1),
            "negated_inputs": laws["negated_inputs"][(y0, k)][:, y_plus].sum(1),
            "marginal_limit": sb.output_law(inner, None, y0),
        }
        sep = {name: float(np.max(np.abs(w - out))) for name, w in wrong.items()}
        cells.append(
            {
                "incoming": y0,
                "sweeps": k,
                "tolerance_output": tol,
                "separation": sep,
                "separates": {name: s > tol for name, s in sep.items()},
            }
        )
    return {
        "seed": seed,
        "site": site,
        "blanket": len(structure["blanket"]),
        "inputs": len(inputs),
        "n_hidden": nh,
        "joint_states": 1 << (nh + 1),
        "lambda_max": mix["lambda_max"],
        "joint_vs_archived_output": agreement,
        "joint_law_seconds": joint_seconds,
        "output_tolerance_seconds": tol_seconds,
        "cells": cells,
    }


def joint_tolerance_seconds(seed: int, site: int, sources: dict) -> dict:
    """Time stage B's joint-tolerance draws for one cell (K=4, y0=-1)."""
    structure = m5a.structures(m5a.make_target(seed, "B"))[site]
    vector = np.asarray(sources[seed]["parameters"][site], dtype=np.float64)
    inputs = m5a.blanket_inputs(structure)
    laws, _ = sb._joint_laws(vector, structure, inputs, "hidden_first", 1.0, 4)
    spec = {"quantile": 0.999, "draws_joint": 1000, "numpy_seed": PROBE_NUMPY_SEED}
    t0 = time.perf_counter()
    tol = sb._joint_tolerance(laws[(-1, 4)], CHAINS, spec, 0)
    return {"seed": seed, "site": site, "seconds_one_cell": time.perf_counter() - t0, "tol": tol}


def thrml_timing(seed: int, site: int) -> list[dict]:
    """Time stage B's run_cell on the first TIMING_BATCHES batches of inputs."""
    request = sb.study_request(CHAINS)
    request["kernel"]["seed"], request["kernel"]["site"] = seed, site
    request["jax_root_seed"] = 9_901  # probe-only
    loaded = sb.load_kernel(request)
    structure = loaded["structure"]
    full = m5a.blanket_inputs(structure)
    batch = request["inputs_per_batch"]
    limited = dict(structure)
    original = m5a.blanket_inputs
    out = []
    try:
        m5a.blanket_inputs = lambda s, _rows=full[: batch * TIMING_BATCHES]: _rows  # probe-only
        for index, k in enumerate(SWEEPS):
            cell = {"incoming": -1, "sweeps": k, "index": index}
            res = sb.run_cell(request, cell, {"structure": limited, "vector": loaded["vector"]})
            hist = np.asarray(res["histograms"])
            out.append(
                {
                    "seed": seed,
                    "site": site,
                    "sweeps": k,
                    "inputs_timed": batch * TIMING_BATCHES,
                    "inputs_total": len(full),
                    "compile_seconds": res["compile_seconds"],
                    "execute_seconds_per_input": res["execute_seconds"] / (batch * TIMING_BATCHES),
                    "counts_ok": bool((hist.sum(1) == CHAINS).all()),
                }
            )
    finally:
        m5a.blanket_inputs = original
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    sources = m5c.references(m5c.load_source())
    kernels = []
    for s in m5c.SEEDS:
        for i in range(12):
            kernels.append(exact_side(s, i, sources))
            print(f"exact {s}/{i} {time.perf_counter() - started:.0f}s", flush=True)
    largest = max(kernels, key=lambda r: r["inputs"] * r["joint_states"])
    joint_tol = joint_tolerance_seconds(largest["seed"], largest["site"], sources)
    print(f"joint tolerance timed {time.perf_counter() - started:.0f}s", flush=True)
    timing = []
    for s, i in TIMING_KERNELS:
        timing += thrml_timing(s, i)
        print(f"thrml {s}/{i} {time.perf_counter() - started:.0f}s", flush=True)
    result = {
        "label": "exploration, not evidence (probe for P-0003)",
        "chains_per_input": CHAINS,
        "kernels": kernels,
        "joint_tolerance_timing": joint_tol,
        "thrml_timing": timing,
        "probe_seconds": time.perf_counter() - started,
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1)
        stream.write("\n")


if __name__ == "__main__":
    main()
