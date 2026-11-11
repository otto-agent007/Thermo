"""Exploration probe: does domain-wall encoding mix under an annealed penalty?

Exploration only, not a recorded study and not evidence. Sampling is THRML 0.1.4
block Gibbs on CPU in float32 (``software_simulation``); equilibrium recalls are
float64 enumeration (``exact_reference``). No hardware claim.

Question. Associative memory stage A (docs/experiment-reports/2026-10-07-am-binary-emulation/)
found that a domain-wall chain reproduces the categorical hidden unit's
equilibrium but does not mix from a random start at a fixed penalty. The
codon-optimization paper (docs/knowledge/sources/extropic-codon-optimization.md)
reports domain-wall Ising runs that work, on chains of at most 5 spins, with the
chain penalty ramped from low to high during the run. Does a ramped penalty
(and, as a second schedule, a ramped beta) close stage A's gap, and do short
chains (P = 6, five chain spins) behave differently?

Fixed choices, set before any result (this file is committed before it runs):

- Model, patterns, cues, targets, chains, start state, held-out and development
  pattern sets, budgets and spin form: stage A's, imported unedited from
  ``thermo_lab.am_binary_emulation`` (its SHA-256 is checked against the archive).
- Cells: stage A's P in {8, 32, 128} x cue in {12, 8}, plus P = 6 with both cues.
- Ramp arm, four configurations: beta_final in {8, 16} x schedule in
  {"penalty": beta fixed at beta_final; "penalty_beta": beta log-spaced from
  beta_final / 8 to beta_final}. In both, the penalty fraction (of a_max = beta sqrt N,
  stage A's convention) is log-spaced from 0.1 to 1.6, stage A's own grid ends.
- A budget of K sweeps runs its own schedule: S = min(K, 8) stages of K / S
  sweeps, stage s at the s-th schedule point. One sample is one chain's missing
  bits after exactly K sweeps, as in stage A.
- Selection: per cell and budget, the development-best ramp configuration by
  mean development recall (ties by id), evaluated on the 12 held-out sets.
- Comparison: held-out recall paired by pattern set against stage A's
  development-best fixed-penalty domain-wall, bias and categorical arms at the
  same K (archived per-set values; for P = 6, stage A's own pipeline is run
  here with ps = [6]). Bootstrap 95% interval over sets, 2,000 draws, and stage
  A's verdict rule with margin 0.02.
- Diagnostics: domain walls per chain at K and the fraction of chains with
  exactly one wall (a valid label). Exact equilibrium recall of each ramp
  endpoint (beta_final, fraction 1.6) on the held-out sets.
- Controls, which must pass before results are read: (1) stage A's
  ``run_unit`` reproduces the archived held unit
  held/domainwall/b8/f0.1/P8/c12/s7100 exactly; (2) the staged runner with a
  constant schedule reproduces ``sample`` at K = 256 bitwise on that unit's
  key; (3) archived held-out exact recalls for the ramp endpoints match the
  recomputed ones to 1e-12.

Run from the repository root:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      uv run python docs/research/am_domainwall_anneal_probe/probe.py
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "probe.json"
ARCHIVE = REPO / "docs/experiment-reports/2026-10-07-am-binary-emulation/study.json.gz"
ARCHIVE_SHA256 = "c83ee9dcd33c0a698ed03f60913f869ffa9088355902c7d1f07bde1fbd31b26f"
SOURCE_SHA256 = "004055add3b00166f08d3d15a7b6fa43072725e298885225722c00448a5ac211"
PROBE_ROOT_SEED = 20261111
NEW_P = 6
CELLS = [(p, c) for p in (NEW_P, 8, 32, 128) for c in (12, 8)]
RAMPS = [
    {"id": f"ramp/{sched}/b{bf}", "schedule": sched, "beta_final": float(bf)}
    for bf in (8, 16)
    for sched in ("penalty", "penalty_beta")
]
FRAC_START, FRAC_END = 0.1, 1.6
BETA_START_RATIO = 8.0
MAX_STAGES = 8
WORKERS = 4


def _a():
    from thermo_lab import am_binary_emulation as a

    return a


def cell_id(p: int, c: int) -> str:
    return f"P{p}/c{c}"


def schedule(cfg: dict, k: int) -> list[tuple[float, float, int]]:
    """(beta, penalty fraction, sweeps) per stage for a budget of k sweeps."""
    s = min(k, MAX_STAGES)
    fr = np.geomspace(FRAC_START, FRAC_END, s)
    bf = cfg["beta_final"]
    if cfg["schedule"] == "penalty":
        be = np.full(s, bf)
    else:
        be = np.geomspace(bf / BETA_START_RATIO, bf, s)
    return [(float(b), float(f), k // s) for b, f in zip(be, fr, strict=True)]


def staged_sample(stages, xi, cue, k, chains, key, targets):
    """Stage A's sampler, run in stages that swap weights; returns the final free state."""
    import jax
    import jax.numpy as jnp

    a = _a()
    st = a.structure("domainwall", xi.shape[0], xi.shape[1], cue)
    total = targets * chains
    k_init, k_run = jax.random.split(key)
    state = st.init_state(k_init, (total,))
    chain_keys = jax.random.split(k_run, total)
    sweep_keys = jax.vmap(lambda kk: jax.random.split(kk, k))(chain_keys)
    clamp = jnp.asarray(np.repeat(xi[:targets, :cue] > 0, chains, axis=0))
    done = 0
    for beta, frac, n in stages:
        program = st.program(beta, frac, xi)
        states = st.runner()(program, sweep_keys[:, done : done + n], state, clamp)
        state = [s[:, -1] for s in states]
        done += n
    assert done == k
    return [np.asarray(s) for s in state]


def final_summary(final, xi, cue, chains, targets) -> dict:
    a = _a()
    st = a.structure("domainwall", xi.shape[0], xi.shape[1], cue)
    truth = np.repeat(xi[:targets, cue:] > 0, chains, axis=0)
    right = (final[0] == truth).sum(axis=1).reshape(targets, chains).sum(axis=1)
    z = np.zeros((final[0].shape[0], len(st.hid)), dtype=bool)
    for block_state, block in zip(final[1:], st.free[1:], strict=True):
        z[:, [st.hid.index(node) for node in block.nodes]] = block_state
    zs = np.where(z, 1, -1)
    full = np.concatenate([np.ones((zs.shape[0], 1), int), zs, -np.ones((zs.shape[0], 1), int)], 1)
    walls = (full[:, 1:] != full[:, :-1]).sum(axis=1)
    return {
        "correct": right.astype(int).tolist(),
        "bits": int(chains * (xi.shape[1] - cue)),
        "walls": int(walls.sum()),
        "one_wall": int((walls == 1).sum()),
        "chains": int(walls.size),
    }


def _stable(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def ramp_unit(args) -> tuple[str, dict]:
    phase, cfg, p, c, seed = args
    import jax

    a = _a()
    xi = a.patterns(seed, p)
    uid = f"{phase}/{cfg['id']}/{cell_id(p, c)}/s{seed}"
    out = {}
    for k in a.BUDGETS:
        key = jax.random.fold_in(jax.random.key(PROBE_ROOT_SEED), _stable(f"{uid}/K{k}"))
        final = staged_sample(schedule(cfg, k), xi, c, k, a.CHAINS, key, a.TARGETS)
        out[f"K{k}"] = final_summary(final, xi, c, a.CHAINS, a.TARGETS)
    if phase == "held":
        out["exact_endpoint"] = [
            a.exact_recall("domainwall", cfg["beta_final"], FRAC_END, xi, t, c)
            for t in range(a.TARGETS)
        ]
    return uid, out


def stage_a_unit(args) -> tuple[str, dict]:
    request, phase, cfg, cell, seed = args
    a = _a()
    return a.unit_id(phase, cfg, cell, seed), a.run_unit(request, phase, cfg, cell, seed, True)


def recall(summary: dict) -> float:
    return float(np.sum(summary["correct"]) / (len(summary["correct"]) * summary["bits"]))


def controls(archive: dict) -> dict:
    import jax

    a = _a()
    request = archive["request"]
    by_id = {c["id"]: c for c in a.configs(request)}
    cfg = by_id["domainwall/b8/f0.1"]
    cell = {"id": "P8/c12", "p": 8, "cue": 12}
    uid = a.unit_id("held", cfg, cell, 7100)
    got = a.run_unit(request, "held", cfg, cell, 7100, True)
    want = archive["held"][uid]
    c1 = got["sampled"] == want["sampled"]
    xi = a.patterns(7100, 8)
    key = jax.random.fold_in(jax.random.key(request["root_seed"]), a._stable_int(uid))
    ref = a.sample("domainwall", 8.0, 0.1, xi, 12, 256, request["chains"], key, request["targets"])
    staged = staged_sample([(8.0, 0.1, 64)] * 4, xi, 12, 256, request["chains"], key, 3)
    c2 = all(np.array_equal(r[:, -1], s) for r, s in zip(ref, staged, strict=True))
    return {"archived_unit_reproduced": bool(c1), "staged_runner_bitwise": bool(c2)}


def bootstrap(diffs: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(2000, len(diffs)))
    m = diffs[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def verdict(lo: float, hi: float, margin: float = 0.02) -> str:
    if lo >= -margin and hi <= margin:
        return "matches"
    if hi < -margin:
        return "falls_short"
    if lo > margin:
        return "exceeds"
    return "inconclusive"


def main() -> None:
    t0 = time.time()
    blob = ARCHIVE.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == ARCHIVE_SHA256, "stage A archive hash mismatch"
    src = (REPO / "src/thermo_lab/am_binary_emulation.py").read_bytes()
    assert hashlib.sha256(src).hexdigest() == SOURCE_SHA256, "stage A source hash mismatch"
    archive = json.loads(gzip.decompress(blob))
    assert archive["source_sha256"] == SOURCE_SHA256
    a = _a()

    ctl = controls(archive)
    print("controls", ctl, flush=True)
    if not all(ctl.values()):
        sys.exit("control failed; results not read")

    # Stage A's own pipeline on the new short-chain cells (all arms, so evaluate() runs as is).
    req6 = a.study_request()
    req6["ps"] = [NEW_P]
    cells6 = a.cells(req6)
    jobs6 = [
        (req6, phase, cfg, cell, seed)
        for cfg in a.configs(req6)
        for cell in cells6
        for phase, seeds in (("dev", req6["dev_seeds"]), ("held", req6["held_seeds"]))
        for seed in seeds
    ]
    jobs = [
        (phase, cfg, p, c, seed)
        for cfg in RAMPS
        for p, c in CELLS
        for phase, seeds in (("dev", a.DEV_SEEDS), ("held", a.HELD_SEEDS))
        for seed in seeds
    ]
    ctx = multiprocessing.get_context("spawn")
    stage6, ramp = {}, {}
    with concurrent.futures.ProcessPoolExecutor(WORKERS, mp_context=ctx) as pool:
        for i, (uid, val) in enumerate(pool.map(stage_a_unit, jobs6, chunksize=4)):
            stage6[uid] = val
            if i % 200 == 0:
                print(f"P6 stage A units {i}/{len(jobs6)} {time.time() - t0:.0f}s", flush=True)
        for i, (uid, val) in enumerate(pool.map(ramp_unit, jobs, chunksize=1)):
            ramp[uid] = val
            if i % 32 == 0:
                print(f"ramp units {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"sampling done {time.time() - t0:.0f}s", flush=True)

    dev6 = {u: v for u, v in stage6.items() if u.startswith("dev/")}
    held6 = {u: v for u, v in stage6.items() if u.startswith("held/")}
    eval6 = a.evaluate(req6, dev6, held6, held6)
    archived = archive["evaluation"]

    # Control 3: recomputed endpoint exact recalls equal the archived held-out exact recalls.
    worst = 0.0
    for cfg in RAMPS:
        for p, c in CELLS:
            if p == NEW_P:
                continue
            for seed in a.HELD_SEEDS:
                key = f"held/domainwall/b{cfg['beta_final']:g}/f1.6/{cell_id(p, c)}/s{seed}"
                got = ramp[f"held/{cfg['id']}/{cell_id(p, c)}/s{seed}"]["exact_endpoint"]
                worst = max(
                    worst,
                    float(np.max(np.abs(np.array(got) - archive["held_exact"][key]["exact"]))),
                )
    ctl["endpoint_exact_matches_archive"] = worst <= 1e-12
    ctl["endpoint_exact_max_abs_diff"] = worst

    rows = {}
    for p, c in CELLS:
        cid = cell_id(p, c)
        base = eval6 if p == NEW_P else archived
        for k in a.BUDGETS:
            best = max(
                RAMPS,
                key=lambda cfg: (
                    np.mean(
                        [recall(ramp[f"dev/{cfg['id']}/{cid}/s{s}"][f"K{k}"]) for s in a.DEV_SEEDS]
                    ),
                    cfg["id"],
                ),
            )
            units = [ramp[f"held/{best['id']}/{cid}/s{s}"] for s in a.HELD_SEEDS]
            vals = np.array([recall(u[f"K{k}"]) for u in units])
            chains = sum(u[f"K{k}"]["chains"] for u in units)
            row = {
                "config": best["id"],
                "recall": float(vals.mean()),
                "per_set": vals.tolist(),
                "walls_per_chain": sum(u[f"K{k}"]["walls"] for u in units) / chains,
                "one_wall_fraction": sum(u[f"K{k}"]["one_wall"] for u in units) / chains,
                "endpoint_exact": float(np.mean([np.mean(u["exact_endpoint"]) for u in units])),
                "versus": {},
            }
            for arm in ("domainwall", "bias", "categorical"):
                ref = base["budget"][f"{cid}/{arm}/K{k}"]
                d = vals - np.array(ref["per_set"])
                lo, hi = bootstrap(d, _stable(f"{cid}/{arm}/K{k}/probe"))
                row["versus"][arm] = {
                    "config": ref["config"],
                    "recall": ref["recall"],
                    "diff": float(d.mean()),
                    "ci": [lo, hi],
                    "verdict": verdict(lo, hi),
                }
            fixed_diag = base["diagnostics"].get(f"{cid}/domainwall/K{k}", {})
            row["fixed_domainwall_walls_per_chain"] = fixed_diag.get("walls_per_chain")
            rows[f"{cid}/K{k}"] = row
        rows[f"{cid}/equilibrium"] = {
            arm: base["equilibrium"][f"{cid}/{arm}"]
            for arm in ("domainwall", "bias", "categorical")
        }

    out = {
        "schema": "am_domainwall_anneal_probe.v1",
        "evidence": "exploration; sampled recall software_simulation (THRML 0.1.4, CPU, float32); "
        "equilibrium recall exact_reference; not hardware",
        "fixed_choices": {
            "cells": [cell_id(p, c) for p, c in CELLS],
            "ramps": RAMPS,
            "frac": [FRAC_START, FRAC_END],
            "beta_start_ratio": BETA_START_RATIO,
            "max_stages": MAX_STAGES,
            "probe_root_seed": PROBE_ROOT_SEED,
            "stage_a_archive_sha256": ARCHIVE_SHA256,
            "stage_a_source_sha256": SOURCE_SHA256,
        },
        "controls": ctl,
        "rows": rows,
        "p6_stage_a": {
            "chosen": eval6["chosen"],
            "budget": {
                k: {kk: vv for kk, vv in v.items() if kk != "per_set"}
                for k, v in eval6["budget"].items()
            },
            "equilibrium": {
                k: {kk: vv for kk, vv in v.items() if kk != "per_set"}
                for k, v in eval6["equilibrium"].items()
            },
            "diagnostics": eval6["diagnostics"],
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "workers": WORKERS,
        "cpu_count": os.cpu_count(),
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
