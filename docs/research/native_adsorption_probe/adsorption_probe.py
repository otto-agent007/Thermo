"""Bounded, disposable adsorption feasibility probe; not a production benchmark.

Run with the Thermo clone's frozen uv environment.  --self-test does not run
the probe; --run requires a fresh JSON --output path.  No draw traces persist.
"""

# Numerical libraries must see thread limits before import.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import adsorption_reference as reference
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

from thermo_lab.constraints.gibbs import exact_chromatic_kernels
from thermo_lab.constraints.jitter import static_gains
from thermo_lab.constraints.lattice import check
from thermo_lab.constraints.model import PairwiseModel
from thermo_lab.constraints.quantize import quantize_array
from thermo_lab.hardware.z1 import Z1OperationCounts, project_z1_operations
from thermo_lab.hashing import canonical_sha256

SHAPE = (3, 3, 16)
TARGETS = (0.2, 0.5, 0.8)
ARMS = (("ideal", 0, 0.0), ("8bit_jitter_0.1", 8, 0.1), ("6bit_jitter_0.1", 6, 0.1))
SEEDS = (0, 1, 2)
SWEEPS, BURN, STRIDE = 4096, 1024, 48
T95_DF2 = 4.302652729911275


def requested_inputs():
    return {
        "schema": "exploratory_adsorption_v1",
        "purpose": "bounded feasibility probe, not a final advantage claim",
        "model": {
            "shape": list(SHAPE),
            "epsilon": 1.0,
            "beta": 1.0,
            "lateral_wall_phi": 3.0,
            "open_ends": True,
            "numeric_dtype": "float64",
            "site_order": "z,y,x",
        },
        "chemical_potentials": {
            "targets": list(TARGETS),
            "bracket": [-10, 10],
            "method": "brentq_exact_full_precision",
            "xtol": 1e-12,
        },
        "quantization": {
            "exact_bits": [0, 6, 8],
            "coupling_cap": 0.25,
            "field_cap": "max_abs_bias_over_all_three_full_precision_points",
            "policy": "manual_frozen_family_split_quantize_array",
        },
        "arms": [{"name": name, "bits": bits, "jitter": jitter} for name, bits, jitter in ARMS],
        "sampling": {
            "seeds": list(SEEDS),
            "starts": ["empty", "full"],
            "sweeps": SWEEPS,
            "burn_sweeps": BURN,
            "read_every_sweeps": STRIDE,
            "reads": 64,
            "block_order": [0, 1],
            "exchange": False,
            "gain_seed_words": "[20261020,seed,0]",
            "gain_reuse": "same field across mu,bits,starts within seed",
            "stream_words": "[20261020,seed,arm_index,mu_index,start_index,1]",
            "batch_means": "8 contiguous batches of 8 read observables",
            "confidence_unit": (
                "independent seed; per-start intervals and pooled average of both starts"
            ),
            "confidence": "three-seed Student t95, df=2",
        },
        "embedding": "(u,v)=(x+4*z,y+z), bounding rectangle 63x18",
        "feasibility_tolerances": {
            "loading": 0.01,
            "pair_occupation": 0.02,
            "working_capacity": 0.01,
            "paired_start_difference": "t95 interval within the same observable tolerance",
            "gate": "upper_absolute_t95_error_vs_full_precision_for_pooled_and_both_starts",
        },
        "limits": {
            "transverse_spins": 12,
            "enumeration_sites": 16,
            "cpu_seconds": 1790,
        },
        "exact_timing": {
            "repeats_per_mu": 5,
            "includes_parameter_setup": True,
            "observables": "logZ,aggregate_loading,aggregate_pair_occupation",
            "algorithm": "factorized_forward_expectation_semiring",
            "energy": "not_measured",
        },
        "projection": {
            "active_sites": 144,
            "bounding_sites": 1134,
            "reads_only_active": True,
            "writes": "one full-node program plus one initialize per participating site",
            "inactive_bounding_sites": "free, updated and fully written",
            "clamp_changes": 0,
        },
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    repo = Path(os.environ.get("THERMO_REPO", Path(__file__).resolve().parents[3]))

    def revision(ref):
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", ref], text=True
        ).strip()

    versions = {}
    for package in ("numpy", "scipy", "thermo-lab", "jax", "jaxlib", "thrml"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    choices = os.environ.get(
        "ADSORPTION_FIXED_CHOICES",
        str(repo / "docs/research/native_adsorption_probe/fixed-choices.md"),
    )
    return {
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "main_sha": revision("refs/remotes/origin/main"),
        "head_sha": revision("HEAD"),
        "script_sha256": sha256(__file__),
        "helper_sha256": sha256(reference.__file__),
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines(),
        "fixed_choices_sha256": sha256(choices) if choices and Path(choices).is_file() else None,
        "fixed_choices_path": choices,
        "script_path": str(Path(__file__).resolve()),
        "helper_path": str(Path(reference.__file__).resolve()),
        "clone_path": str(repo.resolve()),
        "cwd": os.getcwd(),
        "invocation": sys.argv,
        "relevant_environment": {
            name: os.environ.get(name)
            for name in (
                "THERMO_REPO",
                "ADSORPTION_FIXED_CHOICES",
                "VIRTUAL_ENV",
                "JAX_ENABLE_X64",
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "thread_environment": {
            name: os.environ[name]
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def placed_model(weights, bias, shape=SHAPE):
    coords, edges, _ = reference.geometry(*shape)
    sites = tuple((int(x + 4 * z), int(y + z)) for x, y, z in coords)
    model = PairwiseModel(
        sites=sites,
        edges=tuple(map(tuple, edges)),
        weights=weights,
        bias=bias,
        free=tuple(range(len(coords))),
        clamped=(),
        clamped_input=(),
        first_colour=0,
    )
    report = check(model)
    if not report.ok or model.free_colour_violations():
        raise AssertionError("invalid direct lattice embedding")
    return model, {
        **asdict(report),
        "ok": report.ok,
        "bounding_shape": [max(u for u, v in sites) + 1, max(v for u, v in sites) + 1],
    }


def quantized(weights, bias, bits, field_cap):
    if bits == 0:
        return weights.copy(), bias.copy()
    return (quantize_array(weights, 0.25, bits), quantize_array(bias, field_cap, bits))


def neighbor_table(model):
    degree = model.degrees()
    neighbors = np.full((model.nodes, int(degree.max(initial=0))), model.nodes, dtype=int)
    couplings = np.zeros(neighbors.shape)
    filled = np.zeros(model.nodes, dtype=int)
    for (u, v), weight in zip(model.edges, model.weights, strict=True):
        for a, b in ((u, v), (v, u)):
            slot = filled[a]
            neighbors[a, slot], couplings[a, slot] = b, weight
            filled[a] += 1
    return neighbors, couplings


def conditional_probabilities(spins, nodes, bias, neighbors, couplings, gains):
    local_fields = bias[nodes] + np.sum(couplings[nodes] * spins[neighbors[nodes]], axis=1)
    return expit(2 * gains[nodes] * local_fields)


def small_kernel_check():
    """Enumerate actual NumPy block-conditionals against the layer's finite law."""
    model, _ = placed_model(
        np.array([0.23, -0.17, 0.41, 0.09]),
        np.array([-0.3, 0.2, 0.1, -0.15]),
        (2, 2, 1),
    )
    gains = np.array([0.91, 1.07, 0.95, 1.03])
    kernels, states = exact_chromatic_kernels(model, np.empty((1, 0)), gains)
    neighbors, couplings = neighbor_table(model)
    blocks = [np.flatnonzero(np.asarray(model.node_colours) == c) for c in (0, 1)]
    enumerated = np.empty((len(states), len(states)))
    for old_index, old in enumerate(states):
        p0 = conditional_probabilities(
            np.r_[old, 0], blocks[0], model.bias, neighbors, couplings, gains
        )
        for new_index, new in enumerate(states):
            hybrid = old.copy()
            hybrid[blocks[0]] = new[blocks[0]]
            p1 = conditional_probabilities(
                np.r_[hybrid, 0], blocks[1], model.bias, neighbors, couplings, gains
            )
            enumerated[old_index, new_index] = np.prod(
                np.where(new[blocks[0]] > 0, p0, 1 - p0)
            ) * np.prod(np.where(new[blocks[1]] > 0, p1, 1 - p1))
    error = float(np.max(np.abs(enumerated - kernels[0])))
    np.testing.assert_allclose(enumerated.sum(1), 1, rtol=0, atol=1e-12)
    assert error < 1e-12
    return error


def chain_summary(model, gains, words, start, sweeps=SWEEPS, burn=BURN, stride=STRIDE):
    neighbors, couplings = neighbor_table(model)
    colours = np.asarray(model.node_colours)
    blocks = [np.flatnonzero(colours == colour) for colour in (0, 1)]
    spins = np.full(model.nodes + 1, -1 if start == "empty" else 1, dtype=np.int8)
    spins[-1] = 0  # absent-neighbor sentinel
    edges = np.asarray(model.edges)
    rng = np.random.default_rng(words)
    observations = []
    for sweep in range(1, sweeps + 1):
        for nodes in blocks:
            probability = conditional_probabilities(
                spins, nodes, model.bias, neighbors, couplings, gains
            )
            spins[nodes] = 2 * (rng.random(len(nodes)) < probability) - 1
        if sweep > burn and (sweep - burn) % stride == 0:
            occupancy = (spins[:-1].astype(float) + 1) / 2
            observations.append(
                (
                    occupancy.mean(),
                    (occupancy[edges[:, 0]] * occupancy[edges[:, 1]]).mean(),
                )
            )
    observations = np.asarray(observations)
    if len(observations) % 8:
        raise AssertionError("reads must split into eight contiguous batches")
    out = {
        "reads": len(observations),
        "final_state_sha256": hashlib.sha256(spins[:-1].tobytes()).hexdigest(),
    }
    for column, name in enumerate(("loading", "pair_occupation")):
        values = observations[:, column]
        x, y = values[:-1], values[1:]
        lag1 = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
        out[name] = {
            "mean": float(values.mean()),
            "batch_means": values.reshape(8, -1).mean(axis=1).tolist(),
            "lag1": lag1,
        }
    return out


def interval(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,):
        raise ValueError("intervals require exactly three independent seed values")
    mean = float(values.mean())
    half = float(T95_DF2 * float(values.std(ddof=1)) / np.sqrt(3))
    return {
        "seed_values": values.tolist(),
        "mean": mean,
        "t95": [mean - half, mean + half],
        "df": 2,
    }


def summarize(chains, exact):
    summary, capacities = [], []
    for name, bits, jitter in ARMS:
        seed_arrays = {}
        for mu_index in range(3):
            record = {
                "arm": name,
                "mu_index": mu_index,
                "equilibrium_reference": (
                    "same_quantized_symmetric_model"
                    if not jitter
                    else "zero_jitter_quantized_control_only"
                ),
            }
            for observable, exact_key in (
                ("loading", "mean_occupancy"),
                ("pair_occupation", "pair_occupancy"),
            ):
                by_seed, starts = [], []
                per_start = {"empty": [], "full": []}
                for seed in SEEDS:
                    rows = [
                        row
                        for row in chains
                        if row["arm"] == name
                        and row["mu_index"] == mu_index
                        and row["seed"] == seed
                    ]
                    rows.sort(key=lambda row: row["start_index"])
                    values = [row[observable]["mean"] for row in rows]
                    for start_name, value in zip(("empty", "full"), values, strict=True):
                        per_start[start_name].append(value)
                    by_seed.append(float(np.mean(values)))
                    starts.append(values[1] - values[0])
                seed_arrays[mu_index, observable] = np.asarray(by_seed)
                full = exact[mu_index]["bits"]["0"][exact_key]
                control = exact[mu_index]["bits"][str(bits)][exact_key]
                record[observable] = {
                    **interval(by_seed),
                    "error_vs_full_precision": interval(np.asarray(by_seed) - full),
                    "error_vs_zero_jitter_control": interval(np.asarray(by_seed) - control),
                    "full_minus_empty_by_seed": starts,
                    "full_minus_empty_mean": float(np.mean(starts)),
                    "full_minus_empty_t95": interval(starts),
                    "start_differences_are_descriptive": True,
                }
                start_intervals = {}
                for start_name, values in per_start.items():
                    seed_arrays[mu_index, observable, start_name] = np.asarray(values)
                    start_intervals[start_name] = {
                        **interval(values),
                        "error_vs_full_precision": interval(np.asarray(values) - full),
                        "error_vs_zero_jitter_control": interval(np.asarray(values) - control),
                    }
                record[observable]["per_start_three_seed_intervals"] = start_intervals
                record[observable]["pooled_and_both_start_max_abs_t95_error_vs_full"] = max(
                    abs(endpoint)
                    for estimate in [record[observable], *start_intervals.values()]
                    for endpoint in estimate["error_vs_full_precision"]["t95"]
                )
            summary.append(record)
        full_capacity = (
            exact[2]["bits"]["0"]["mean_occupancy"] - exact[0]["bits"]["0"]["mean_occupancy"]
        )
        capacity = seed_arrays[2, "loading"] - seed_arrays[0, "loading"]
        capacity_record = {
            "arm": name,
            "working_capacity_high_minus_low": interval(capacity),
            "error_vs_full_precision": interval(capacity - full_capacity),
            "per_start_three_seed_intervals": {},
        }
        for start_name in ("empty", "full"):
            values = seed_arrays[2, "loading", start_name] - seed_arrays[0, "loading", start_name]
            capacity_record["per_start_three_seed_intervals"][start_name] = {
                **interval(values),
                "error_vs_full_precision": interval(values - full_capacity),
            }
        capacity_record["pooled_and_both_start_max_abs_t95_error_vs_full"] = max(
            abs(endpoint)
            for estimate in [
                capacity_record,
                *capacity_record["per_start_three_seed_intervals"].values(),
            ]
            for endpoint in estimate["error_vs_full_precision"]["t95"]
        )
        capacity_record["full_minus_empty_t95"] = interval(
            seed_arrays[2, "loading", "full"]
            - seed_arrays[0, "loading", "full"]
            - seed_arrays[2, "loading", "empty"]
            + seed_arrays[0, "loading", "empty"]
        )
        capacities.append(capacity_record)
    gates = []
    for name, _, _ in ARMS:
        point_records = [row for row in summary if row["arm"] == name]
        loading = all(
            row["loading"]["pooled_and_both_start_max_abs_t95_error_vs_full"] <= 0.01
            for row in point_records
        )
        pair = all(
            row["pair_occupation"]["pooled_and_both_start_max_abs_t95_error_vs_full"] <= 0.02
            for row in point_records
        )
        capacity = (
            next(row for row in capacities if row["arm"] == name)[
                "pooled_and_both_start_max_abs_t95_error_vs_full"
            ]
            <= 0.01
        )
        start_agreement = all(
            max(abs(v) for v in row[observable]["full_minus_empty_t95"]["t95"]) <= tolerance
            for row in point_records
            for observable, tolerance in (("loading", 0.01), ("pair_occupation", 0.02))
        ) and all(
            max(abs(v) for v in row["full_minus_empty_t95"]["t95"]) <= 0.01
            for row in capacities
            if row["arm"] == name
        )
        gates.append(
            {
                "arm": name,
                "loading_within_0.01": loading,
                "pair_within_0.02": pair,
                "capacity_within_0.01": capacity,
                "start_agreement": start_agreement,
                "feasibility_gate": loading and pair and capacity and start_agreement,
                "criterion_includes_pooled_and_both_starts": True,
                "physical_device_reliability_claim": False,
            }
        )
    return {
        "points": summary,
        "software_feasibility_gates": gates,
        "paired_seed_working_capacity": capacities,
        "exact_working_capacity": {
            str(bits): exact[2]["bits"][str(bits)]["mean_occupancy"]
            - exact[0]["bits"][str(bits)]["mean_occupancy"]
            for bits in (0, 6, 8)
        },
    }


def projection_records():
    output = {}
    for name, sites in (
        ("active_only", 144),
        ("bounding_rectangle_free_inactive", 1134),
    ):
        counts = Z1OperationCounts.constant_participation(
            logical_pbits=144,
            physical_pbits_used=sites,
            participating_free_pbits=sites,
            elapsed_complete_sweeps=SWEEPS,
            node_reads=144 * 64,
            node_full_sram_writes=sites * 2,
            clamp_state_changes=0,
        )
        output[name] = asdict(project_z1_operations(counts))
    active = output["active_only"]["operation_counts"]
    assert (
        active["gibbs_node_updates"],
        active["node_reads"],
        active["node_full_sram_writes"],
        active["clamp_state_changes"],
    ) == (589824, 9216, 288, 0)
    return {
        "unit": "one trajectory",
        "scenarios": output,
        "physical_hardware_measurement": False,
        "software_updates_are_not_measured_device_operations": True,
    }


def preflight():
    # Derived comparisons must stay Python booleans for strict JSON output.
    json.dumps({"gate": max(map(abs, interval([0.1, 0.2, 0.3])["t95"])) <= 0.01})
    transcript = io.StringIO()
    with contextlib.redirect_stdout(transcript):
        reference.self_test()
    shape = (2, 2, 2)
    coords, edges, _ = reference.geometry(*shape)
    rng = np.random.default_rng(492001)
    weights, bias = rng.normal(size=len(edges)), rng.normal(size=len(coords))
    model, placement = placed_model(weights, bias, shape)
    neighbors, couplings = neighbor_table(model)
    states = np.asarray(list(itertools.product((-1, 1), repeat=len(coords))))
    energies = states @ bias + (states[:, edges[:, 0]] * states[:, edges[:, 1]]) @ weights
    probabilities = np.exp(energies - logsumexp(energies))
    layer = reference.exact_stats(shape, weights, bias)
    layer_error = float(
        np.max(np.abs(probabilities @ ((states + 1) / 2) - layer["site_occupancies"]))
    )
    conditional_error = 0.0
    for state in states:
        padded = np.r_[state, 0]
        predictions = conditional_probabilities(
            padded,
            np.arange(len(coords)),
            bias,
            neighbors,
            couplings,
            np.ones(len(coords)),
        )
        for site in range(len(coords)):
            minus, plus = state.copy(), state.copy()
            minus[site], plus[site] = -1, 1

            def score(s):
                return s @ bias + (s[edges[:, 0]] * s[edges[:, 1]]) @ weights

            expected = expit(score(plus) - score(minus))
            conditional_error = max(conditional_error, abs(float(predictions[site]) - expected))
    assert max(layer_error, conditional_error) < 1e-12
    small = chain_summary(model, np.ones(8), [20261020, 0, 99], "empty", 16, 0, 1)
    assert small == chain_summary(model, np.ones(8), [20261020, 0, 99], "empty", 16, 0, 1)
    for shape, enumeration in (((13, 1, 1), False), ((1, 1, 17), True)):
        try:
            if enumeration:
                reference._brute_force(shape, 0.25, 0.0)
            else:
                reference.exact_stats(shape, 0.25, 0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("reference bound did not reject oversized request")
    p = reference.adsorption_parameters(0)
    _, full_placement = placed_model(p["weights"], p["bias"])
    assert full_placement["bounding_shape"] == [63, 18]
    projection_records()
    return {
        "status": "passed",
        "heterogeneous_gain_one_sweep_kernel_max_error": small_kernel_check(),
        "reference_self_test": transcript.getvalue().splitlines(),
        "conditional_max_error": conditional_error,
        "small_layer_max_error": layer_error,
        "small_placement": placement,
        "full_placement": full_placement,
        "sampler_repeatability": True,
        "reference_bounds": True,
    }


def run_probe():
    started = time.perf_counter()
    requested = requested_inputs()
    checks = preflight()
    points = []
    for target in TARGETS:

        def residual(mu, target=target):
            p = reference.adsorption_parameters(mu)
            return reference.exact_stats(SHAPE, p["weights"], p["bias"])["mean_occupancy"] - target

        mu = brentq(residual, -10, 10, xtol=1e-12)
        points.append((mu, reference.adsorption_parameters(mu)))
    field_cap = max(float(np.max(np.abs(p["bias"]))) for _, p in points)
    exact, timings, rounded_models, gain_inputs = [], [], [], []
    for mu_index, (mu, parameters) in enumerate(points):
        record = {
            "mu_index": mu_index,
            "target_loading": TARGETS[mu_index],
            "mu": mu,
            "bits": {},
        }
        for bits in (0, 6, 8):
            weights, bias = quantized(parameters["weights"], parameters["bias"], bits, field_cap)
            rounded_models.append(
                {
                    "mu_index": mu_index,
                    "bits": bits,
                    "weights": weights.tolist(),
                    "bias": bias.tolist(),
                }
            )
            stats = reference.exact_stats(SHAPE, weights, bias)
            record["bits"][str(bits)] = {
                "evidence_class": "exact_reference",
                "logZ_spin": stats["logZ"],
                "mean_occupancy": stats["mean_occupancy"],
                "pair_occupancy": stats["pair_occupancy"],
                "max_abs_bias_rounding_error": float(np.max(np.abs(bias - parameters["bias"]))),
                "max_abs_weight_rounding_error": float(
                    np.max(np.abs(weights - parameters["weights"]))
                ),
            }
            if bits == 0:
                record["bits"]["0"]["logZ_lattice_gas"] = stats["logZ"] + parameters["logZ_offset"]
        durations = []
        for _ in range(5):
            tick = time.perf_counter()
            p = reference.adsorption_parameters(mu)
            reference.exact_aggregate(SHAPE, p["weights"], p["bias"])
            durations.append(time.perf_counter() - tick)
        timings.append(
            {
                "mu_index": mu_index,
                "wall_seconds": durations,
                "median_wall_seconds": float(np.median(durations)),
            }
        )
        exact.append(record)
    chains = []
    for seed in SEEDS:
        gains = static_gains([20261020, seed, 0], [144], 0.1)[0]
        gain_inputs.append({"seed": seed, "words": [20261020, seed, 0], "gains": gains.tolist()})
        gain_hash = hashlib.sha256(gains.tobytes()).hexdigest()
        for arm_index, (name, bits, jitter) in enumerate(ARMS):
            for mu_index, (_mu, parameters) in enumerate(points):
                weights, bias = quantized(
                    parameters["weights"], parameters["bias"], bits, field_cap
                )
                model, _ = placed_model(weights, bias)
                for start_index, start in enumerate(("empty", "full")):
                    words = [20261020, seed, arm_index, mu_index, start_index, 1]
                    result = chain_summary(model, gains if jitter else np.ones(144), words, start)
                    chains.append(
                        {
                            "evidence_class": "software_simulation",
                            "seed": seed,
                            "arm": name,
                            "arm_index": arm_index,
                            "mu_index": mu_index,
                            "start": start,
                            "start_index": start_index,
                            "gain_sha256": gain_hash if jitter else None,
                            "sampler_words": words,
                            **result,
                        }
                    )
    assert len(chains) == 54 and all(row["reads"] == 64 for row in chains)
    derived_inputs = {
        "chemical_potentials": [mu for mu, _ in points],
        "shape": list(SHAPE),
        "coupling_cap": 0.25,
        "field_cap": field_cap,
        "models": rounded_models,
        "gain_groups": gain_inputs,
    }
    return {
        "requested_inputs": requested,
        "requested_inputs_sha256": canonical_sha256(requested),
        "derived_numerical_inputs": derived_inputs,
        "derived_numerical_inputs_sha256": canonical_sha256(derived_inputs),
        "runtime": {
            "provenance": provenance(),
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time(),
            "preflight": checks,
        },
        "family_quantization": {"field_cap": field_cap, "coupling_cap": 0.25},
        "exact_references": exact,
        "chains": chains,
        "summaries": summarize(chains, exact),
        "exact_baseline_timing": {
            "evidence_class": "software_wall_timing",
            "setup_included": True,
            "energy_measured": False,
            "algorithm": "factorized_forward_expectation_semiring",
            "operation_count": "unknown_not_MAC_equivalent",
            "points": timings,
        },
        "hardware_projection": projection_records(),
        "limitations": [
            "exploratory feasibility, not final advantage",
            "no physical hardware measurements",
            "three independent seed groups only",
            "jittered conditionals need not share a symmetric equilibrium Hamiltonian",
            "lag and start diagnostics do not certify mixing",
            "no measured baseline energy or invented MAC count",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.run and args.output is None:
        parser.error("--run requires --output")
    if args.output is not None and args.output.exists():
        parser.error("output already exists; use a fresh path")
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    limit = 1790 if hard == resource.RLIM_INFINITY else min(1790, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    result = preflight() if args.self_test else run_probe()
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x") as stream:
            stream.write(encoded)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(args.output),
                    "sha256": sha256(args.output),
                },
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
