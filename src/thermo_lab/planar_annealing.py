"""Annealing, restarts and tempering against exact ground states on frustrated grids.

The planar Ising scaling study and the October 8 probe showed that mixed-sign
grids at beta = 4 are out of reach for equilibrium samplers at this lab's
budgets, while an annealed chain reaches the exact energy per spin within a
percent. This study asks the optimization question at equal elapsed sweeps:
how close to the exact ground-state energy do annealing (to beta 8 and 16),
four parallel restarts, a cold chain and the nine-replica tempering ladder get
on 64-, 256- and 576-spin frustrated grids, and what does each cost in the Z1
model?

References are exact: a max-plus transfer matrix over 2^L window states gives
the ground-state energy, and a log-sum-exp transfer matrix gives the thermal
energy at beta 8 and 16 (bounded to L <= 24). The kernel is the scaling
study's two-colour sweep with a per-sweep beta schedule; its sampler, keys and
estimator are imported unchanged for the tempering arm.

Sweeps are software_simulation; references exact_reference; energies and
sweep times calibrated_projection. Nothing here ran on hardware.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from thermo_lab import fixed_budget_sampling as base
from thermo_lab import planar_ising_scaling as scaling
from thermo_lab.exchange_cost_projection import CONVENTIONS
from thermo_lab.hardware.z1 import Z1HardwareProfile, Z1OperationCounts, project_z1_operations
from thermo_lab.hashing import canonical_sha256
from thermo_lab.provenance import collect_runtime_provenance

ROOT = base.ROOT
SOURCES = (
    "src/thermo_lab/planar_annealing.py",
    "src/thermo_lab/planar_ising_scaling.py",
    "src/thermo_lab/exchange_cost_projection.py",
    "src/thermo_lab/hardware/z1.py",
    "src/thermo_lab/hashing.py",
    "src/thermo_lab/persistence.py",
    "docs/experiments/planar-annealing.md",
    "uv.lock",
)
SIZES = (8, 16, 24)
SEEDS = (400, 401, 402)
BUDGETS = (256, 1024, 4096, 16384, 65536)
THERMAL_BETAS = (8.0, 16.0)
BETA_START = 0.5
RAMP_FRACTION = 0.75
ARMS = {
    # name: (kind, endpoint beta, chains)
    "anneal8": ("anneal", 8.0, 1),
    "anneal16": ("anneal", 16.0, 1),
    "restart8x4": ("anneal", 8.0, 4),
    "restart16x4": ("anneal", 16.0, 4),
    "cold16": ("cold", 16.0, 1),
    "tempering9": ("tempering", 4.0, 9),
}
TOLERANCES = (1e-3, 1e-4)


# ----------------------------------------------------------------------------
# Targets and exact references
# ----------------------------------------------------------------------------


def grid_targets():
    result = []
    for size in SIZES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000 + size)
            edges = scaling.grid_edges(size)
            weights = rng.integers(1, 6, len(edges)).astype(np.float32) / 5
            signs = rng.choice([-1.0, 1.0], len(edges)).astype(np.float32)
            result.append(
                {
                    "id": f"L{size}-mixed-s{seed}",
                    "size": size,
                    "n": size * size,
                    "variant": "mixed",
                    "seed": seed,
                    "couplings": (signs * weights).tolist(),
                }
            )
    return result


def _float64_arrays(size, values):
    """Horizontal (L, L-1) and vertical (L-1, L) arrays in float64, no float32 cast."""
    values = np.asarray(values, dtype=np.float64)
    return (
        values[: size * (size - 1)].reshape(size, size - 1),
        values[size * (size - 1) :].reshape(size - 1, size),
    )


def _sweep(size, interactions, combine, init):
    """Row transfer in raster order over 2^L window states.

    Window bit L-1 is the newest spin (left neighbour of the next site), bit 0
    the oldest (its up neighbour). Adding a spin maps source state k to target
    (k >> 1) | (s_new << (L-1)), so sources 2j and 2j+1 merge into j; ``combine``
    reduces that pair (log-sum-exp for Z, max for the ground state).
    """
    if not 1 <= size <= 24:
        raise ValueError("the transfer matrix is bounded to L <= 24")
    horizontal, vertical = _float64_arrays(size, interactions)
    count = 1 << size
    half = count >> 1
    states = np.arange(count)
    oldest = (2 * (states & 1) - 1).astype(float)
    newest = (2 * ((states >> (size - 1)) & 1) - 1).astype(float)
    v = np.full(count, init)
    for i in range(size * size):
        r, c = divmod(i, size)
        field = np.zeros(count)
        if r > 0:
            field += vertical[r - 1, c] * oldest
        if c > 0:
            field += horizontal[r, c - 1] * newest
        v = np.concatenate(
            [combine((v - field).reshape(half, 2)), combine((v + field).reshape(half, 2))]
        )
    return v


def transfer_log_z(size, interactions):
    v = _sweep(size, interactions, lambda a: np.logaddexp(a[:, 0], a[:, 1]), 0.0)
    return float(np.logaddexp.reduce(v) - size * np.log(2))


def ground_state_energy(size, couplings):
    """Exact max over configurations of sum_e J_e s_i s_j, per spin."""
    v = _sweep(size, couplings, lambda a: a.max(axis=1), 0.0)
    return float(v.max()) / (size * size)


def thermal_energy(size, couplings, beta):
    """q(beta) = d ln Z / d beta / N exactly, in one sweep.

    Carries the derivative of every window state's log partial sum alongside
    it: merging two log values a, b with derivatives da, db gives
    logaddexp(a, b) and the softmax-weighted (da, db). Each site's field
    contribution beta * J * s has derivative J * s.
    """
    if not 1 <= size <= 24:
        raise ValueError("the transfer matrix is bounded to L <= 24")
    couplings = np.asarray(couplings, dtype=float)
    horizontal, vertical = _float64_arrays(size, couplings)
    count = 1 << size
    half = count >> 1
    states = np.arange(count)
    oldest = (2 * (states & 1) - 1).astype(float)
    newest = (2 * ((states >> (size - 1)) & 1) - 1).astype(float)
    v = np.zeros(count)
    dv = np.zeros(count)
    for i in range(size * size):
        r, c = divmod(i, size)
        field = np.zeros(count)  # J * s, so the field is beta * this
        if r > 0:
            field += vertical[r - 1, c] * oldest
        if c > 0:
            field += horizontal[r, c - 1] * newest
        parts, dparts = [], []
        for sign in (-1.0, 1.0):
            a = (v + sign * beta * field).reshape(half, 2)
            da = (dv + sign * field).reshape(half, 2)
            merged = np.logaddexp(a[:, 0], a[:, 1])
            weight = np.exp(a[:, 0] - merged)
            parts.append(merged)
            dparts.append(weight * da[:, 0] + (1 - weight) * da[:, 1])
        v = np.concatenate(parts)
        dv = np.concatenate(dparts)
    weights = np.exp(v - v.max())
    return float((weights @ dv) / weights.sum()) / (size * size)


def brute_force_ground_state(size, couplings):
    if size * size > 16:
        raise ValueError("brute force is bounded to 16 spins")
    edges = scaling.grid_edges(size)
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=size * size)))
    products = states[:, [e[0] for e in edges]] * states[:, [e[1] for e in edges]]
    return float((products @ np.asarray(couplings, dtype=float)).max()) / (size * size)


def reference(target, thermal=True):
    """Exact ground state, and thermal energies where the protocol asks for them."""
    couplings = np.asarray(target["couplings"], dtype=float)
    result = {
        "evidence_class": "exact_reference",
        "method": "max_plus_transfer_matrix",
        "ground_state_per_spin": ground_state_energy(target["size"], couplings),
    }
    if thermal:
        betas = THERMAL_BETAS if target["size"] <= 16 else (16.0,)
        result["thermal_per_spin"] = {
            str(beta): thermal_energy(target["size"], couplings, beta) for beta in betas
        }
        for beta, value in result["thermal_per_spin"].items():
            if value > result["ground_state_per_spin"] + 1e-9:
                raise ValueError(f"thermal energy at beta {beta} exceeds the ground state")
    return result


def reference_checks():
    """Brute-force and cross-implementation checks of the transfer matrices."""
    rng = np.random.default_rng(2026)
    worst_ground, worst_log_z, worst_cross, worst_thermal = 0.0, 0.0, 0.0, 0.0
    for size in (2, 3, 4):
        for variant in ("ferro", "mixed"):
            m = len(scaling.grid_edges(size))
            weights = rng.integers(1, 6, m) / 5
            signs = rng.choice([-1.0, 1.0], m)
            couplings = -weights if variant == "ferro" else signs * weights
            worst_ground = max(
                worst_ground,
                abs(
                    ground_state_energy(size, couplings) - brute_force_ground_state(size, couplings)
                ),
            )
            for beta in (4.0, 16.0):
                exact_log_z, exact_edge = scaling.brute_force(size, beta * couplings)
                worst_log_z = max(
                    worst_log_z, abs(transfer_log_z(size, beta * couplings) - exact_log_z)
                )
                worst_thermal = max(
                    worst_thermal,
                    abs(
                        thermal_energy(size, couplings, beta)
                        - float(couplings @ exact_edge) / (size * size)
                    ),
                )
    for target in grid_targets():
        if target["size"] <= 16:
            interactions = 4.0 * np.asarray(target["couplings"], dtype=float)
            mine = transfer_log_z(target["size"], interactions)
            theirs = scaling.transfer_matrix_log_z(target["size"], interactions)
            worst_cross = max(worst_cross, abs(mine - theirs) / abs(theirs))
    if worst_ground > 1e-12 or worst_log_z > 1e-9 or worst_cross > 1e-12 or worst_thermal > 1e-9:
        raise ValueError("transfer-matrix references failed their checks")
    return {
        "evidence_class": "exact_reference",
        "brute_force_sizes": [2, 3, 4],
        "max_ground_state_error": worst_ground,
        "max_log_z_error": worst_log_z,
        "max_thermal_energy_error": worst_thermal,
        "max_cross_implementation_relative_error": worst_cross,
    }


# ----------------------------------------------------------------------------
# Samplers
# ----------------------------------------------------------------------------


def schedule(kind, endpoint, horizon):
    """Per-sweep inverse temperatures for an annealed or cold chain."""
    if kind == "cold":
        return np.full(horizon, endpoint, dtype=np.float32)
    ramp = int(RAMP_FRACTION * horizon)
    return np.concatenate(
        [np.geomspace(BETA_START, endpoint, ramp), np.full(horizon - ramp, endpoint)]
    ).astype(np.float32)


def compile_scheduled_sampler(size, chains, betas):
    """Independent chains following one beta schedule.

    Returns the exact int32 sum of every edge product over the hold phase per
    chain, (trials, chains, edges); the caller forms the energy in float64 so
    the gap to the exact ground state carries no float32 accumulation error.
    """
    betas = jnp.asarray(betas, jnp.float32)
    steps = len(betas)
    hold_start = steps - steps // 4
    checker = (jnp.arange(size)[:, None] + jnp.arange(size)[None, :]) % 2 == 0
    edges = 2 * size * (size - 1)

    def single(key, initial, horizontal, vertical):
        state = initial[:chains]

        def half_sweep(state, key, mask, beta):
            field = scaling.local_fields(state, horizontal, vertical)
            draw = jax.random.bernoulli(key, jax.nn.sigmoid(2 * beta * field))
            return jnp.where(mask, draw, state)

        def step(carry, t):
            state, key, sums = carry
            key, key_a, key_b = jax.random.split(key, 3)
            state = half_sweep(state, key_a, checker, betas[t])
            state = half_sweep(state, key_b, ~checker, betas[t])
            products = scaling.edge_products(state, size)
            sums = sums + jnp.where(t >= hold_start, products, 0)
            return (state, key, sums), None

        (_, _, sums), _ = jax.lax.scan(
            step, (state, key, jnp.zeros((chains, edges), jnp.int32)), jnp.arange(steps)
        )
        return sums

    return jax.jit(jax.vmap(single, in_axes=(0, 0, None, None)))


def hold_energy(sums, target, horizon):
    """Energy per spin over the hold phase from exact integer edge sums, in float64."""
    hold = horizon // 4
    couplings = np.asarray(target["couplings"], dtype=np.float64)
    return np.asarray(sums, dtype=np.float64) @ couplings / (hold * target["n"])


# ----------------------------------------------------------------------------
# Cells, pricing, decisions
# ----------------------------------------------------------------------------


def price(target, chains, budget, exchanges_per_trial):
    """Z1 projection per trial under both conventions; annealing has no I/O."""
    n = target["n"]
    pair_tries = exchanges_per_trial["pair_tries"]
    energies = {}
    for name, rule in CONVENTIONS.items():
        values = []
        for accepts in exchanges_per_trial["accepted"]:
            counts = Z1OperationCounts.constant_participation(
                logical_pbits=n * chains,
                physical_pbits_used=n * chains,
                participating_free_pbits=n * chains,
                elapsed_complete_sweeps=budget,
                node_reads=int(pair_tries * rule["reads_per_attempt_per_pair"] * n),
                node_full_sram_writes=int(accepts * rule["writes_per_accept_per_pair"] * n),
                host_round_trips=int(exchanges_per_trial["round_trips"]),
            )
            values.append(project_z1_operations(counts).modeled_total_energy_j)
        energies[name] = float(np.mean(values))
    return {
        "elapsed_complete_sweeps": budget,
        "sweep_time_at_assumed_max_clock_s": budget
        / Z1HardwareProfile().cost_model_max_complete_sweep_rate.value_hz,
        "physical_pbits_used": n * chains,
        "mean_total_energy_j": energies,
        "evidence_class": "calibrated_projection",
    }


def make_cell(target, arm, budget, energies, exact, accepted=None):
    """``energies`` is (trials, chains) hold-phase energy per spin."""
    kind, endpoint, chains = ARMS[arm]
    energies = np.asarray(energies, dtype=np.float64)
    reported = energies.max(axis=1)  # best chain per trial (one chain: itself)
    gap = exact["ground_state_per_spin"] - reported
    if np.any(gap < -1e-9):
        raise ValueError("a chain reports an energy above the exact ground state")
    if accepted is not None:
        flags = np.asarray(accepted[:, :budget, :], dtype=bool)
        exchanges = {
            "pair_tries": 4 * budget,
            "round_trips": budget,
            "accepted": flags.sum(axis=(1, 2)).astype(int).tolist(),
        }
        acceptance = [float(flags[:, :, slot].mean()) for slot in range(flags.shape[2])]
    else:
        exchanges = {"pair_tries": 0, "round_trips": 0, "accepted": [0] * len(reported)}
        acceptance = []
    return {
        "target": target["id"],
        "size": target["size"],
        "arm": arm,
        "kind": kind,
        "endpoint_beta": endpoint,
        "chains": chains,
        "budget": budget,
        "per_trial": {"energy_per_spin": reported.tolist(), "gap": gap.tolist()},
        "means": {
            "gap": float(gap.mean()),
            "best_gap": float(gap.min()),
            "worst_gap": float(gap.max()),
        },
        "fraction_within": {str(tol): float((gap <= tol).mean()) for tol in TOLERANCES},
        "fraction_at_ground_state": float((gap <= 1e-9).mean()),
        "exchange_acceptance_by_slot": acceptance,
        "z1": price(target, chains, budget, exchanges),
    }


def decide(request, cells):
    """Per target and arm: the smallest budget whose mean gap stays within each tolerance."""
    decisions = []
    for target in request["targets"]:
        for arm in request["arms"]:
            rows = sorted(
                (c for c in cells if c["target"] == target["id"] and c["arm"] == arm),
                key=lambda c: c["budget"],
            )
            entry = {"target": target["id"], "size": target["size"], "arm": arm}
            for tol in TOLERANCES:
                passes = [c["means"]["gap"] <= tol for c in rows]
                reached = next((c for i, c in enumerate(rows) if all(passes[i:])), None)
                entry[f"within_{tol}"] = (
                    {
                        "status": "reached",
                        "budget": reached["budget"],
                        "sweep_time_s": reached["z1"]["sweep_time_at_assumed_max_clock_s"],
                        "energy_j": reached["z1"]["mean_total_energy_j"],
                        "physical_pbits": reached["z1"]["physical_pbits_used"],
                    }
                    if reached
                    else {"status": "not_reached", "largest_tested_budget": rows[-1]["budget"]}
                )
            entry["gap_at_largest_budget"] = rows[-1]["means"]["gap"]
            decisions.append(entry)
    return decisions


def summarize(decisions):
    """Qualification counts per arm and size, and best arm per target per tolerance."""
    counts = {}
    best = []
    for d in decisions:
        for tol in TOLERANCES:
            key = f"within_{tol}"
            counts.setdefault(d["arm"], {}).setdefault(f"L{d['size']}", {}).setdefault(key, [0, 0])
            cell = counts[d["arm"]][f"L{d['size']}"][key]
            cell[1] += 1
            if d[key]["status"] == "reached":
                cell[0] += 1
    for target in sorted({d["target"] for d in decisions}):
        rows = [d for d in decisions if d["target"] == target]
        entry = {"target": target}
        for tol in TOLERANCES:
            key = f"within_{tol}"
            reached = [d for d in rows if d[key]["status"] == "reached"]
            entry[f"fastest_{tol}"] = (
                min(reached, key=lambda d: (d[key]["budget"], d["arm"]))["arm"] if reached else None
            )
            entry[f"cheapest_published_{tol}"] = (
                min(reached, key=lambda d: (d[key]["energy_j"]["published"], d["arm"]))["arm"]
                if reached
                else None
            )
        entry["smallest_gap_at_largest_budget"] = min(
            rows, key=lambda d: (d["gap_at_largest_budget"], d["arm"])
        )["arm"]
        best.append(entry)
    return {"qualifying_counts": counts, "best_per_target": best}


# ----------------------------------------------------------------------------
# Study
# ----------------------------------------------------------------------------


def make_request():
    return {
        "schema": "planar_annealing.v1",
        "status": "exploratory",
        "arms": list(ARMS),
        "arm_definitions": {
            name: {"kind": kind, "endpoint_beta": beta, "chains": chains}
            for name, (kind, beta, chains) in ARMS.items()
        },
        "schedule": {
            "shape": "geometric ramp then hold",
            "beta_start": BETA_START,
            "ramp_fraction": RAMP_FRACTION,
            "measure": "hold phase, final quarter of the horizon",
        },
        "tempering_ladder": list(scaling.LADDERS[9]),
        "thermal_betas": list(THERMAL_BETAS),
        "targets": grid_targets(),
        "budgets": list(BUDGETS),
        "trials": 16,
        "root_seed": 20261011,
        "tolerances": list(TOLERANCES),
        "metric": "ground-state energy per spin minus the chain's hold-phase energy per spin",
        "reported_energy": "best chain per trial for multi-chain arms",
        "replication_unit": "independent trial; chains within a restart arm are one trial",
        "profile_id": Z1HardwareProfile().profile_id,
        "profile_hash": Z1HardwareProfile().profile_hash,
        "conventions": CONVENTIONS,
        "numeric_dtype": "float32 sampler; float64 transfer matrices",
        "replay_atol": 2e-12,
        "replay_scope": (
            "replay authenticates sources, archived references and per-trial energies, "
            "recomputes references for L <= 16 and all gaps, fractions, pricing, decisions and "
            "summaries; it verifies L = 24 references by digest and does not regenerate sweeps"
        ),
        "sources": {name: base.sha(ROOT / name) for name in SOURCES},
    }


def _validate_request(request):
    if request["arms"] != list(ARMS):
        raise ValueError("the study arms are frozen")
    if any(b % 4 for b in request["budgets"]):
        raise ValueError("budgets must be multiples of four")
    if request["schedule"]["beta_start"] != BETA_START:
        raise ValueError("the schedule start is frozen")
    if request["replay_atol"] != 2e-12:
        raise ValueError("replay tolerance is frozen")
    if request["profile_hash"] != Z1HardwareProfile().profile_hash:
        raise ValueError("Z1 profile changed")
    for target in request["targets"]:
        if target["size"] > 24:
            raise ValueError("targets are bounded to L <= 24")


def _target_cells(request, target, exact):
    """Sample every arm and budget for one target; returns cells and packed flags."""
    size = target["size"]
    horizontal, vertical = (jnp.asarray(x) for x in scaling.coupling_arrays(target))
    index = request["targets"].index(target)
    initial, keys = scaling.key_inputs(index, size, 9, request["trials"], request["root_seed"])
    cells, flags = [], {}
    for arm in request["arms"]:
        kind, endpoint, chains = ARMS[arm]
        arm_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, list(ARMS).index(arm))
        if kind == "tempering":
            horizon = max(request["budgets"])
            plan = scaling.snapshot_plan("tempering", request["budgets"])
            fn = scaling.compile_sampler("tempering", chains, 1, size, horizon, plan)
            snaps, accepted = map(
                np.asarray, jax.block_until_ready(fn(arm_keys, initial, horizontal, vertical))
            )
            couplings = np.asarray(target["couplings"], dtype=np.float64)
            for budget in request["budgets"]:
                estimate = scaling.window_estimate(
                    scaling.window_sums(snaps, plan, "tempering", budget), "tempering", 1, budget
                )
                energies = (estimate @ couplings / target["n"])[:, None]
                cells.append(make_cell(target, arm, budget, energies, exact, accepted))
            flags[f"{target['id']}__{arm}__accepted"] = np.packbits(accepted, axis=1)
            continue
        for budget in request["budgets"]:
            fn = compile_scheduled_sampler(size, chains, schedule(kind, endpoint, budget))
            sums = np.asarray(jax.block_until_ready(fn(arm_keys, initial, horizontal, vertical)))
            cells.append(make_cell(target, arm, budget, hold_energy(sums, target, budget), exact))
    return cells, flags


def run_study(out, requested=None, resume=False):
    out = Path(out)
    request = make_request() if requested is None else requested
    _validate_request(request)
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise ValueError("use CPU with JAX_ENABLE_X64=false")
    partial = out / "partial"
    if out.exists() and not resume:
        raise FileExistsError("use a fresh output directory or --resume")
    if resume:
        if not (out / "request.json").exists():
            raise FileNotFoundError("nothing to resume")
        if json.loads((out / "request.json").read_text()) != request:
            raise ValueError("resume request differs from the saved request")
    else:
        out.mkdir(parents=True)
        partial.mkdir()
        base.write(out / "request.json", request)
        for name, digest in request["sources"].items():
            if base.sha(ROOT / name) != digest:
                raise ValueError(f"source changed: {name}")
            destination = out / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    started = time.perf_counter()
    checks_path = partial / "checks.json"
    if checks_path.exists():
        checks = json.loads(checks_path.read_text())
    else:
        checks = {"reference": reference_checks(), "fixture": scaling.fixture_checks()}
        base.write(checks_path, checks)
        print("Exact reference and kernel checks passed", flush=True)
    references, cells, flags = {}, [], {}
    for target in request["targets"]:
        unit = partial / f"{target['id']}.json"
        unit_flags = partial / f"{target['id']}.npz"
        if unit.exists():
            saved = json.loads(unit.read_text())
            references[target["id"]] = saved["reference"]
            cells.extend(saved["cells"])
            flags.update(dict(np.load(unit_flags)))
            print(f"Resumed {target['id']}", flush=True)
            continue
        before = time.perf_counter()
        exact = reference(target)
        reference_seconds = time.perf_counter() - before
        before = time.perf_counter()
        target_cells, target_flags = _target_cells(request, target, exact)
        sampling_seconds = time.perf_counter() - before
        np.savez_compressed(unit_flags, **target_flags)
        base.write(
            unit,
            {
                "reference": exact,
                "cells": target_cells,
                "reference_seconds": reference_seconds,
                "sampling_seconds": sampling_seconds,
            },
        )
        references[target["id"]] = exact
        cells.extend(target_cells)
        flags.update(target_flags)
        print(
            f"Completed {target['id']} (reference {reference_seconds:.0f} s, "
            f"sampling {sampling_seconds:.0f} s)",
            flush=True,
        )
    np.savez_compressed(out / "flags.npz", **flags)
    decisions = decide(request, cells)
    result = {
        "request_digest": canonical_sha256(request),
        "evidence_class": {
            "sweeps": "software_simulation",
            "references": "exact_reference",
            "z1_pricing": "calibrated_projection",
        },
        "flags_sha256": base.sha(out / "flags.npz"),
        "checks": checks,
        "exact_references": references,
        "cells": cells,
        "decisions": decisions,
        "summary": summarize(decisions),
        "generation_seconds": time.perf_counter() - started,
        "provenance": collect_runtime_provenance(ROOT).model_dump(mode="json"),
    }
    base.write(out / "results.json", result)
    return result


def replay(out, light=False):
    """Recompute everything from the persisted per-trial energies and flags.

    ``light`` skips recomputing the L = 24 references (about 45 minutes each);
    they are then verified by digest against the archived values only.
    """
    out = Path(out)
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    _validate_request(request)
    if result["request_digest"] != canonical_sha256(request):
        raise ValueError("request digest mismatch")
    for name, digest in request["sources"].items():
        for path in (ROOT / name, out / "source" / name):
            if base.sha(path) != digest:
                raise ValueError(f"source changed: {name}")
    if base.sha(out / "flags.npz") != result["flags_sha256"]:
        raise ValueError("flags digest mismatch")
    scaling.check_equal(
        {"reference": reference_checks(), "fixture": scaling.fixture_checks()}, result["checks"]
    )
    flags = dict(np.load(out / "flags.npz"))
    recomputed = 0
    cells = []
    for target in request["targets"]:
        archived = result["exact_references"][target["id"]]
        if target["size"] <= 16 or not light:
            scaling.check_equal(reference(target), archived)
            recomputed += 1
        exact = archived
        for arm in request["arms"]:
            kind, _, _ = ARMS[arm]
            accepted = None
            if kind == "tempering":
                packed = flags[f"{target['id']}__{arm}__accepted"]
                accepted = np.unpackbits(packed, axis=1, count=max(request["budgets"])).astype(bool)
            for budget in request["budgets"]:
                recorded = next(
                    c
                    for c in result["cells"]
                    if c["target"] == target["id"] and c["arm"] == arm and c["budget"] == budget
                )
                energies = np.asarray(recorded["per_trial"]["energy_per_spin"])[:, None]
                cells.append(make_cell(target, arm, budget, energies, exact, accepted))
    scaling.check_equal(cells, result["cells"])
    decisions = decide(request, cells)
    scaling.check_equal(decisions, result["decisions"])
    scaling.check_equal(summarize(decisions), result["summary"])
    completion = {
        "status": "planar_annealing_complete",
        "request_digest": result["request_digest"],
        "targets": len(request["targets"]),
        "cells_replayed": len(cells),
        "decisions_replayed": len(decisions),
        "references_recomputed": recomputed,
        "references_verified_by_digest": len(request["targets"]) - recomputed,
        "reference_checks_passed": True,
        "fixture_stationarity_passed": True,
        "replay_seconds": time.perf_counter() - started,
        "evidence_class": result["evidence_class"],
    }
    base.write(out / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--light", action="store_true", help="replay: skip L = 24 references")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.output_dir, light=args.light), indent=2))
    else:
        run_study(args.output_dir, resume=args.resume)
        print(json.dumps(replay(args.output_dir, light=True), indent=2))


if __name__ == "__main__":
    main()
