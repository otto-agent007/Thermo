"""Price replica exchange in the Z1 Appendix-B cost model.

Stage A re-scores the archived fresh-seed symmetry-plus-tempering study
(October 2) under the sealed Z1 projection without drawing a sample. Stage B
samples the same six zero-field graphs with fresh seeds, varying the number
of sweeps between exchange attempts, and reports the (sweep-time, energy,
accuracy) frontier per target.

The archived sampler, estimator, exact references and qualification rule are
imported unchanged. The study-local sampler below reproduces the archived one
bit for bit at interval 1 (see tests) and only adds the exchange interval.

Everything here is a calibrated projection over software-simulation traces:
no sweep ran on hardware, and the Appendix-B model excludes host latency, so
"time" means elapsed complete sweeps at the assumed maximum clock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from thermo_lab import fixed_budget_sampling as base
from thermo_lab import sampling_time_to_accuracy as prior
from thermo_lab import symmetry_tempering as archived
from thermo_lab.hardware.z1 import Z1HardwareProfile, Z1OperationCounts, project_z1_operations
from thermo_lab.hashing import canonical_sha256
from thermo_lab.provenance import collect_runtime_provenance

ROOT = base.ROOT
SOURCES = (
    "src/thermo_lab/exchange_cost_projection.py",
    "src/thermo_lab/hardware/z1.py",
    "docs/experiments/exchange-cost-projection.md",
    *archived.SOURCES,
)
ARCHIVE_REPORT = "docs/experiment-reports/2026-10-02-symmetry-tempering"
INTERVALS = (1, 4, 16, 64, 256)
CONVENTIONS = {
    "published": {
        "description": (
            "an exchange attempt reads every p-bit of both replicas in the pair so the host "
            "can form their energies; an accepted exchange writes both replicas' states back "
            "(full SRAM write per p-bit). Nothing in this convention is outside the sealed "
            "Appendix-B profile."
        ),
        "reads_per_attempt_per_pair": 2,
        "writes_per_accept_per_pair": 2,
    },
    "beta_knob": {
        "description": (
            "hypothetical: the same reads, but an accepted exchange swaps the two replicas' "
            "inverse temperatures through a per-replica control at zero node-write cost. "
            "The published profile has no such control; this bounds how much a future one "
            "could help."
        ),
        "reads_per_attempt_per_pair": 2,
        "writes_per_accept_per_pair": 0,
    },
}


def arms():
    return ["long-flip", "independent-flip", *(f"tempering-flip-k{k}" for k in INTERVALS)]


def split_arm(arm):
    """Return (underlying sampler, exchange interval) for a stage-B arm name."""
    if arm in ("long-flip", "independent-flip"):
        return arm.removesuffix("-flip"), 0
    prefix = "tempering-flip-k"
    if not arm.startswith(prefix):
        raise ValueError(f"unknown arm {arm}")
    interval = int(arm.removeprefix(prefix))
    if interval not in INTERVALS:
        raise ValueError(f"unknown exchange interval {interval}")
    return "tempering", interval


def compile_interval_sampler(method, n, horizon, interval):
    """The archived sampler with an exchange attempted every ``interval`` sweeps.

    At ``interval == 1`` the random stream, update order and exchange schedule are
    identical to ``fixed_budget_sampling.compile_sampler``; the unit tests pin
    that bit for bit. Pairs (0,1)/(2,3) and (1,2)/(3,4) still alternate, now by
    exchange index rather than sweep index. Every sweep still splits an exchange
    key so the sampling stream does not depend on the interval.
    """
    if method not in base.METHODS:
        raise ValueError("unknown sampling method")
    if method != "tempering" and interval != 0:
        raise ValueError("only tempering has an exchange interval")
    if method == "tempering" and interval not in INTERVALS:
        raise ValueError("unsupported exchange interval")
    replicas = 1 if method == "long" else 5
    steps = 5 * horizon if method == "long" else horizon
    beta = jnp.asarray(base.LADDER if method == "tempering" else [4.0] * replicas, jnp.float32)

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
                due = (t + 1) % interval == 0
                exchange_index = (t + 1) // interval - 1
                left = jnp.asarray([0, 2]) + exchange_index % 2
                right = left + 1
                spins = 2 * state.astype(jnp.float32) - 1
                q = spins @ fields + 0.5 * jnp.sum((spins @ matrix) * spins, axis=1)
                log_a = base.swap_log_acceptance(beta[left], beta[right], q[left], q[right])
                accepted = (jnp.log(jax.random.uniform(exchange_key, shape=(2,))) < log_a) & due
                old_left, old_right = state[left], state[right]
                state = state.at[left].set(jnp.where(accepted[:, None], old_right, old_left))
                state = state.at[right].set(jnp.where(accepted[:, None], old_left, old_right))
            return (state, key), (state, accepted)

        _, (states, accepted) = jax.lax.scan(step, (start, key), jnp.arange(steps))
        return jnp.concatenate([start[None], states], axis=0), accepted

    return jax.jit(jax.vmap(single, in_axes=(0, 0, None, None)))


def make_request():
    return {
        "schema": "exchange_cost_projection.v1",
        "status": "exploratory",
        "stage_a_archive": ARCHIVE_REPORT,
        "stage_a_archive_sha256": json.loads((ROOT / ARCHIVE_REPORT / "manifest.json").read_text())[
            "sha256"
        ],
        "profile_id": Z1HardwareProfile().profile_id,
        "profile_hash": Z1HardwareProfile().profile_hash,
        "conventions": CONVENTIONS,
        "sweep_convention": (
            "one systematic sequential Gibbs sweep over n sites is counted as one elapsed "
            "complete sweep with n node updates; the Appendix-B sweep is two-colour, so this "
            "is an update-count equivalence, not a schedule equivalence"
        ),
        "placement": "idealised one logical p-bit per physical p-bit; no embedding overhead",
        "arms": arms(),
        "intervals": list(INTERVALS),
        "targets": archived.graph_targets(),
        "budgets": [16, 64, 256, 1024, 4096],
        "trials": 16,
        "root_seed": 20261008,
        "threshold": 0.05,
        "ladder": list(base.LADDER),
        "numeric_dtype": "float32 sampler; float64 exact enumeration",
        "sample": "initial plus each complete systematic Gibbs sweep; after exchange for tempering",
        "burn_in_fraction": 0.25,
        "replication_unit": (
            "independent trial; prefixes and symmetry variants are not extra trials"
        ),
        "qualification": "mean joint TV and edge MAE <= threshold, sustained at all later budgets",
        "replay_atol": 2e-12,
        "sources": {name: base.sha(ROOT / name) for name in SOURCES},
    }


def price_trial(n, method, budget, accepts_per_pair_set, attempts, convention):
    """Z1 operation counts and projection for one trial of one cell."""
    rule = CONVENTIONS[convention]
    replicas = 1 if method == "long" else 5
    sweeps = 5 * budget if method == "long" else budget
    reads = attempts * rule["reads_per_attempt_per_pair"] * n
    writes = accepts_per_pair_set * rule["writes_per_accept_per_pair"] * n
    counts = Z1OperationCounts.constant_participation(
        logical_pbits=n * replicas,
        physical_pbits_used=n * replicas,
        participating_free_pbits=n * replicas,
        elapsed_complete_sweeps=sweeps,
        node_reads=int(reads),
        node_full_sram_writes=int(writes),
        host_round_trips=int(attempts // 2),
    )
    return project_z1_operations(counts)


def price_cell(target, method, interval, budget, accepted):
    """Price every trial of a cell; ``accepted`` has shape (trials, steps, 2)."""
    n = target["n"]
    if method == "tempering":
        flags = np.asarray(accepted[:, :budget, :], dtype=bool)
        attempts = 2 * (budget // interval)
        accepts = flags.sum(axis=(1, 2)).astype(int)
        if np.any(flags.sum(axis=(1, 2)) > attempts):
            raise ValueError("more accepted exchanges than attempts")
    else:
        attempts = 0
        accepts = np.zeros(len(accepted), dtype=int)
    priced = {}
    for convention in CONVENTIONS:
        projections = [
            price_trial(n, method, budget, int(a), attempts, convention) for a in accepts
        ]
        energies = np.asarray([p.modeled_total_energy_j for p in projections])
        priced[convention] = {
            "mean_total_energy_j": float(energies.mean()),
            "min_total_energy_j": float(energies.min()),
            "max_total_energy_j": float(energies.max()),
            "mean_sampling_energy_j": float(np.mean([p.sampling_energy_j for p in projections])),
            "mean_read_energy_j": float(np.mean([p.read_energy_j for p in projections])),
            "mean_write_energy_j": float(np.mean([p.write_energy_j for p in projections])),
        }
    first = price_trial(n, method, budget, int(accepts[0]), attempts, "published")
    return {
        "elapsed_complete_sweeps": first.operation_counts.elapsed_complete_sweeps,
        "sweep_time_at_assumed_max_clock_s": first.sampling_time_at_assumed_max_clock_s,
        "physical_pbits_used": first.operation_counts.physical_pbits_used,
        "gibbs_node_updates": first.operation_counts.gibbs_node_updates,
        "exchange_attempts_per_trial": int(attempts),
        "host_round_trips_per_trial": int(attempts // 2),
        "mean_accepted_exchanges_per_trial": float(accepts.mean()),
        "energy": priced,
        "evidence_class": str(first.evidence_class),
        "excluded_costs": list(first.excluded_costs),
    }


def qualify(cells, threshold):
    cells = sorted(cells, key=lambda row: row["budget"])
    passes = [max(row["means"]["joint_tv"], row["means"]["edge_mae"]) <= threshold for row in cells]
    for i, row in enumerate(cells):
        if all(passes[i:]):
            return {
                "status": "reached",
                "budget": row["budget"],
                "sweep_time_s": row["z1"]["sweep_time_at_assumed_max_clock_s"],
                "host_round_trips": row["z1"]["host_round_trips_per_trial"],
                "energy_j": {
                    name: row["z1"]["energy"][name]["mean_total_energy_j"] for name in CONVENTIONS
                },
            }
    return {"status": "not_reached", "largest_tested_budget": cells[-1]["budget"]}


def comparisons(decisions):
    """Per target: cheapest qualifying ordinary baseline and tempering ratios."""
    result = []
    for target in sorted({d["target"] for d in decisions}):
        rows = {d["arm"]: d for d in decisions if d["target"] == target}
        ordinary = [
            rows[a] for a in ("long-flip", "independent-flip") if rows[a]["status"] == "reached"
        ]
        entry = {"target": target, "ordinary_qualifying": [d["arm"] for d in ordinary]}
        for convention in CONVENTIONS:
            cheapest = min(ordinary, key=lambda d: d["energy_j"][convention], default=None)
            fastest = min(ordinary, key=lambda d: d["sweep_time_s"], default=None)
            block = {
                "cheapest_ordinary": cheapest["arm"] if cheapest else None,
                "fastest_ordinary": fastest["arm"] if fastest else None,
                "tempering": {},
            }
            for arm, d in rows.items():
                if not arm.startswith("tempering"):
                    continue
                if d["status"] != "reached":
                    block["tempering"][arm] = {"status": "not_reached"}
                    continue
                block["tempering"][arm] = {
                    "status": "reached",
                    "budget": d["budget"],
                    "energy_ratio_vs_cheapest_ordinary": (
                        d["energy_j"][convention] / cheapest["energy_j"][convention]
                        if cheapest
                        else None
                    ),
                    "sweep_time_ratio_vs_fastest_ordinary": (
                        d["sweep_time_s"] / fastest["sweep_time_s"] if fastest else None
                    ),
                }
            entry[convention] = block
        result.append(entry)
    return result


def pareto(decisions):
    """Arms not dominated on (sweep time, published energy) among qualifying arms."""
    result = []
    for target in sorted({d["target"] for d in decisions}):
        reached = [d for d in decisions if d["target"] == target and d["status"] == "reached"]
        front = []
        for d in reached:
            dominated = any(
                o is not d
                and o["sweep_time_s"] <= d["sweep_time_s"]
                and o["energy_j"]["published"] <= d["energy_j"]["published"]
                and (
                    o["sweep_time_s"] < d["sweep_time_s"]
                    or o["energy_j"]["published"] < d["energy_j"]["published"]
                )
                for o in reached
            )
            if not dominated:
                front.append(d["arm"])
        result.append({"target": target, "frontier": sorted(front)})
    return result


# ----------------------------------------------------------------------------
# Stage A: price the archived study
# ----------------------------------------------------------------------------


def _archived_arm_name(method):
    return {"tempering": "tempering-k1", "tempering-flip": "tempering-flip-k1"}.get(method, method)


def load_archive(report=None):
    """Extract and authenticate the archived symmetry-tempering evidence."""
    report = ROOT / (report or ARCHIVE_REPORT)
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("archived evidence digest mismatch")
    scratch = Path(tempfile.mkdtemp(prefix="exchange-cost-stage-a-"))
    with tarfile.open(archive_path) as archive:
        archive.extractall(scratch, filter="data")
    output = scratch / "symmetry-tempering"
    for name, digest in manifest["members"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"archived member changed: {name}")
    request = json.loads((output / "request.json").read_text())
    results = json.loads((output / "results.json").read_text())
    traces = dict(np.load(output / "traces.npz"))
    shutil.rmtree(scratch)
    return manifest, request, results, traces


def stage_a(manifest, request, results, traces):
    targets = {t["id"]: t for t in request["targets"]}
    cells = []
    for row in results["cells"]:
        target = targets[row["target"]]
        underlying = row["method"].removesuffix("-flip")
        accepted = traces[f"{target['id']}__{underlying}__accepted"]
        priced = price_cell(target, underlying, 1, row["budget"], accepted)
        if (
            underlying == "tempering"
            and priced["exchange_attempts_per_trial"] != (row["swap_attempts_per_trial"])
        ):
            raise ValueError("archived swap attempts disagree with the priced attempts")
        if priced["gibbs_node_updates"] != row["spin_updates_per_trial"]:
            raise ValueError("archived spin updates disagree with the priced node updates")
        cells.append(
            {
                "target": row["target"],
                "arm": _archived_arm_name(row["method"]),
                "budget": row["budget"],
                "means": row["means"],
                "z1": priced,
            }
        )
    decisions = []
    for d in results["decisions"]:
        arm = _archived_arm_name(d["method"])
        mine = qualify(
            [c for c in cells if c["target"] == d["target"] and c["arm"] == arm],
            request["threshold"],
        )
        if mine["status"] != d["status"] or mine.get("budget") != d.get("budget"):
            raise ValueError("archived qualification decision not reproduced")
        decisions.append({"target": d["target"], "arm": arm, **mine})
    return {
        "archive_sha256": manifest["sha256"],
        "cells": cells,
        "decisions": decisions,
        "comparisons": comparisons(decisions),
        "pareto": pareto(decisions),
    }


# ----------------------------------------------------------------------------
# Stage B: fresh-seed exchange-interval sweep
# ----------------------------------------------------------------------------


def make_cell(target, arm, states, accepted, budget, exact, threshold):
    method, interval = split_arm(arm)
    estimator = "tempering-flip" if method == "tempering" else arm
    measured = prior.accuracy(prior.estimates(target, estimator, states, budget), exact)
    passed = (np.asarray(measured["per_trial"]["joint_tv"]) <= threshold) & (
        np.asarray(measured["per_trial"]["edge_mae"]) <= threshold
    )
    return {
        "target": target["id"],
        "arm": arm,
        "method": method,
        "exchange_interval": interval,
        "budget": budget,
        "augmentation": "analytic global flip",
        **measured,
        "individual_trial_pass_fraction": float(passed.mean()),
        "z1": price_cell(target, method, interval, budget, accepted),
    }


def _validate_request(request):
    if request["arms"] != arms() or request["intervals"] != list(INTERVALS):
        raise ValueError("the study arms are frozen")
    if any(b <= 0 or b % 4 for b in request["budgets"]):
        raise ValueError("budgets must be positive multiples of four")
    if max(request["budgets"]) % max(INTERVALS):
        raise ValueError("the horizon must be a multiple of the largest exchange interval")
    if request["ladder"] != list(base.LADDER) or request["burn_in_fraction"] != 0.25:
        raise ValueError("the reused sampler has a frozen ladder and burn-in")
    if request["replay_atol"] != 2e-12:
        raise ValueError("replay tolerance is frozen")
    if request["profile_hash"] != Z1HardwareProfile().profile_hash:
        raise ValueError("Z1 profile changed")
    for target in request["targets"]:
        if np.any(np.asarray(target["fields"]) != 0):
            raise ValueError("global-flip augmentation requires zero fields")


def run_study(out, requested=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("use a fresh output directory")
    request = make_request() if requested is None else requested
    _validate_request(request)
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

    manifest, archive_request, archive_results, archive_traces = load_archive(
        request["stage_a_archive"]
    )
    if manifest["sha256"] != request["stage_a_archive_sha256"]:
        raise ValueError("stage A archive is not the requested one")
    stage_a_result = stage_a(manifest, archive_request, archive_results, archive_traces)
    print(f"Stage A priced {len(stage_a_result['cells'])} archived cells", flush=True)

    fixture, _ = base.validate_fixture()
    packed, cells, references, compile_times = {}, [], {}, {}
    horizon = max(request["budgets"])
    for index, target in enumerate(request["targets"]):
        n = target["n"]
        references[target["id"]] = prior.reference(target)
        fields, matrix = (jnp.asarray(x) for x in base.model(target))
        initial, keys = base.key_inputs(index, n, request["trials"], request["root_seed"])
        for arm in request["arms"]:
            method, interval = split_arm(arm)
            method_index = base.METHODS.index(method)
            sample_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, method_index)
            before = time.perf_counter()
            fn = compile_interval_sampler(method, n, horizon, interval)
            fn = fn.lower(sample_keys, initial, fields, matrix).compile()
            compile_times[f"n{n}-{arm}"] = time.perf_counter() - before
            states, accepted = map(
                np.asarray, jax.block_until_ready(fn(sample_keys, initial, fields, matrix))
            )
            # Prefix property: a shorter horizon is the same random stream truncated.
            short = compile_interval_sampler(method, n, request["budgets"][0], interval)
            short_states, short_accepted = map(
                np.asarray, jax.block_until_ready(short(sample_keys, initial, fields, matrix))
            )
            if not np.array_equal(short_states, states[:, : short_states.shape[1]]):
                raise ValueError("horizon states are not a common prefix")
            if not np.array_equal(short_accepted, accepted[:, : short_accepted.shape[1]]):
                raise ValueError("exchange flags are not a common prefix")
            kept = states[:, :, -1:] if method == "tempering" else states
            packed[f"{target['id']}__{arm}"] = np.packbits(kept, axis=-1, bitorder="little")
            packed[f"{target['id']}__{arm}__accepted"] = accepted
            for budget in request["budgets"]:
                cells.append(
                    make_cell(
                        target,
                        arm,
                        kept,
                        accepted,
                        budget,
                        references[target["id"]],
                        request["threshold"],
                    )
                )
        print(f"Sampled {index + 1}/{len(request['targets'])}: {target['id']}", flush=True)
    np.savez_compressed(out / "traces.npz", **packed)
    decisions = [
        {
            "target": target["id"],
            "arm": arm,
            **qualify(
                [c for c in cells if c["target"] == target["id"] and c["arm"] == arm],
                request["threshold"],
            ),
        }
        for target in request["targets"]
        for arm in request["arms"]
    ]
    result = {
        "request_digest": canonical_sha256(request),
        "evidence_class": {
            "traces": "software_simulation",
            "references": "exact_reference",
            "z1_pricing": "calibrated_projection",
        },
        "trace_sha256": base.sha(out / "traces.npz"),
        "fixture": fixture,
        "stage_a": stage_a_result,
        "exact_references": references,
        "cells": cells,
        "decisions": decisions,
        "comparisons": comparisons(decisions),
        "pareto": pareto(decisions),
        "compile_seconds": compile_times,
        "generation_seconds": time.perf_counter() - started,
        "provenance": collect_runtime_provenance(ROOT).model_dump(mode="json"),
    }
    base.write(out / "results.json", result)
    return result


def replay(out):
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
    if base.sha(out / "traces.npz") != result["trace_sha256"]:
        raise ValueError("trace digest mismatch")
    manifest, archive_request, archive_results, archive_traces = load_archive(
        request["stage_a_archive"]
    )
    if manifest["sha256"] != request["stage_a_archive_sha256"]:
        raise ValueError("stage A archive is not the requested one")
    prior.check_equal(
        stage_a(manifest, archive_request, archive_results, archive_traces), result["stage_a"]
    )
    traces = dict(np.load(out / "traces.npz"))
    cells = []
    for target in request["targets"]:
        n = target["n"]
        exact = prior.reference(target)
        prior.check_equal(exact, result["exact_references"][target["id"]])
        for arm in request["arms"]:
            method, _ = split_arm(arm)
            replicas = 5 if method == "independent" else 1  # tempering stores its cold replica
            packed = traces[f"{target['id']}__{arm}"]
            states = np.unpackbits(packed, axis=-1, bitorder="little", count=n).astype(bool)
            if states.shape[2] != replicas:
                raise ValueError("unexpected replica count in traces")
            accepted = traces[f"{target['id']}__{arm}__accepted"]
            for budget in request["budgets"]:
                cells.append(
                    make_cell(target, arm, states, accepted, budget, exact, request["threshold"])
                )
    prior.check_equal(cells, result["cells"])
    decisions = [
        {
            "target": target["id"],
            "arm": arm,
            **qualify(
                [c for c in cells if c["target"] == target["id"] and c["arm"] == arm],
                request["threshold"],
            ),
        }
        for target in request["targets"]
        for arm in request["arms"]
    ]
    prior.check_equal(decisions, result["decisions"])
    prior.check_equal(comparisons(decisions), result["comparisons"])
    prior.check_equal(pareto(decisions), result["pareto"])
    completion = {
        "status": "exchange_cost_projection_complete",
        "request_digest": result["request_digest"],
        "archived_cells_priced": len(result["stage_a"]["cells"]),
        "archived_decisions_priced": len(result["stage_a"]["decisions"]),
        "cells_replayed": len(cells),
        "decisions_replayed": len(decisions),
        "profile_id": request["profile_id"],
        "replay_seconds": time.perf_counter() - started,
        "evidence_class": result["evidence_class"],
    }
    base.write(out / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.output_dir), indent=2))
    else:
        run_study(args.output_dir)
        print(json.dumps(replay(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
