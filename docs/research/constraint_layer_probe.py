"""Exploratory probe for queue row `constraint-layer` (loop proposal P-0004).

EXPLORATION ONLY. Nothing this script prints or writes is recorded evidence and
none of it may be cited as a study result. Its job is to find which fixed
choice of a hardware-constraint layer bounds the row's metric ("bitwise
agreement with the M5c placement archive, the M5b rounding archive, the
am-coupling-bits quantized recall at 6 bits and the E0 stage B inner-K law
through the layer; one priced example per constraint") before any protocol or
module exists, and to size the work.

It holds a throwaway prototype of the layer (the `Layer*` helpers below): a
declared, hashed constraint request; a codebook quantizer under a declared cap
rule; static per-site gains; a checker that a pairwise graph lives on the Z1
16-offset lattice with a two-colour schedule; a generic exact chromatic
block-Gibbs kernel; a THRML lowering; and Z1 operation counts. The prototype
imports the hash-bound modules and archives read-only and edits none of them.

Parts
  P0  offsets: M5c, planar-16-offset-ferro and Z1HardwareProfile agree.
  P1  M5b rounding: layer codebook on every reading-B precision cell (120) ->
      archived `rounding` summary, bitwise; three cells' chain metrics, bitwise.
  P2  M5c placement: 60 archived layouts re-verified by the layer's own lattice
      check and by the hash-bound verify_layout (bitwise against archive); the
      fastest archived MILP placements re-solved and compared layout-for-layout.
      Planar-16-offset-ferro: the 18 archived graphs rebuilt and lattice-checked.
      AM bias design: degree against 16, and a chain lower bound.
  P3  E0 stage B: (a) layer exact chromatic kernel on the *placed* seed-0 site-1
      layout against the archived inner-K law; (b) THRML through the layer's
      unplaced lowering with E0's keys, one cell, histogram bitwise; (c) THRML on
      the placed layout (input copies), one cell, against E0's archived tolerance.
  P4  am-coupling-bits: layer quantizer (cap rule `coupling`, 6 bits) on all 96
      archived exact units, bitwise; two archived sampled runs (jitter 0 and
      0.1) bitwise through layer gains.
  P5  Z1 pricing: one priced example per constraint, relaxed flags in the
      hashed request.

Evidence classes if this were recorded: P1/P2/P3a/P4-exact are float64
enumerations (exact_reference); P3b/P3c/P4-sampled are THRML 0.1.4 on CPU in
float32 (software_simulation); P5 is a calibrated projection of algorithmic
counts. No hardware claim anywhere.

Run (CPU only):
  JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run python docs/research/constraint_layer_probe.py [parts...]
"""

from __future__ import annotations

import gzip
import io
import json
import math
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization as m5b
from thermo_lab import meta_ebm_thermalization_core as core
from thermo_lab import meta_ebm_topology as m5c
from thermo_lab.hardware.z1 import Z1HardwareProfile, Z1OperationCounts, project_z1_operations
from thermo_lab.hashing import canonical_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "2026-10-10-constraint-layer-probe.json"
REPORTS = ROOT / "docs" / "experiment-reports"
M5C_ARCHIVE = REPORTS / "2026-09-30-meta-ebm-topology" / "study.json.gz"
E0B_ARCHIVE = REPORTS / "2026-10-03-thrml-m5a-kernel-inner-sweep" / "study.json.gz"
AM_ARCHIVE = REPORTS / "2026-10-08-am-coupling-bits" / "study.json.gz"
PLANAR_TAR = REPORTS / "2026-10-08-planar-16-offset-ferro" / "evidence.tar.gz"
PROFILE = Z1HardwareProfile()
Z1_OFFSETS = PROFILE.interior_offsets()


# --------------------------------------------------------------------------- prototype layer


def layer_request(**fields) -> dict:
    """A declared constraint request; every relaxation is a hashed field."""
    request = {
        "schema": "constraints.request.probe",
        "lattice": {"rules": [list(r) for r in PROFILE.connection_rules], "degree": 16},
        "codebook": "L = 2^(b-1) - 1; step = cap / L; q = step * clip(rint(x / step), -L, L)",
        "schedule": "chromatic two-colour block Gibbs, colour (u+v) mod 2",
        "replica_exchange": False,
        "relaxed": [],
    }
    request.update(fields)
    request["relaxed"] = sorted(request["relaxed"])
    return request


def layer_codebook(x, cap: float, bits: int) -> np.ndarray:
    """The M5b rounding rule with clipping and any width (same arithmetic)."""
    levels = 2 ** (bits - 1) - 1
    step = cap / levels
    return step * np.clip(np.rint(np.asarray(x, dtype=np.float64) / step), -levels, levels)


def layer_rounding_summary(vectors, cap: float, bits: int) -> dict:
    """Same summary fields and reduction order as M5b's round_parameters."""
    errors = [np.abs(layer_codebook(v, cap, bits) - np.asarray(v, np.float64)) for v in vectors]

    def summarize(error):
        return {
            "max_absolute": float(error.max()),
            "mean_absolute": float(error.mean()),
            "max_cap_normalized": float(error.max() / cap),
            "mean_cap_normalized": float(error.mean() / cap),
        }

    return {**summarize(np.concatenate(errors)), "sites": [summarize(e) for e in errors]}


def layer_gains(words, sizes, jitter: float):
    """Static per-site gains: one default_rng(words), uniform(1-j, 1+j) per group in order."""
    if jitter == 0:
        return [np.ones(n) for n in sizes]
    rng = np.random.default_rng(list(words))
    return [rng.uniform(1 - jitter, 1 + jitter, n) for n in sizes]


def colour(u) -> int:
    return (u[0] + u[1]) % 2


def lattice_check(sites, edges) -> dict:
    """Every coupled pair is a Z1 offset apart, opposite colours; degree <= 16."""
    degree: dict = {}
    bad_offset = bad_colour = 0
    for u, v in edges:
        a, b = sites[u], sites[v]
        if (b[0] - a[0], b[1] - a[1]) not in Z1_OFFSETS:
            bad_offset += 1
        if colour(a) == colour(b):
            bad_colour += 1
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    return {
        "edges": len(edges),
        "offset_violations": bad_offset,
        "colour_violations": bad_colour,
        "max_degree": max(degree.values(), default=0),
        "degree_violations": sum(d > PROFILE.interior_degree for d in degree.values()),
        "distinct_sites": len(set(map(tuple, sites))) == len(sites),
    }


def lower_placed_kernel(layout, vector, structure):
    """Placed M5c layout -> pairwise model with free (y, hidden) and clamped copy nodes.

    Independent of verify_layout: any adjacent copy of an input carries it.
    """
    J, h, A, b, beta = m5a.unpack(vector, structure)
    nodes = [((e[0], e[1]), e[2], e[3]) for e in layout]
    y = next(i for i, n in enumerate(nodes) if n[1] == "y")
    hidden = {n[2]: i for i, n in enumerate(nodes) if n[1] == "h"}
    copies = [(i, n[2]) for i, n in enumerate(nodes) if n[1] == "copy"]
    sites = [n[0] for n in nodes]

    def adjacent_copy(i, inp):
        for c, k in copies:
            d = (sites[c][0] - sites[i][0], sites[c][1] - sites[i][1])
            if k == inp and d in Z1_OFFSETS:
                return c
        raise ValueError("no adjacent copy")

    edges, weights = [], []
    for j in np.flatnonzero(J):
        edges.append((y, adjacent_copy(y, int(j))))
        weights.append(J[j])
    for a in range(len(beta)):
        for j in np.flatnonzero(A[a]):
            edges.append((hidden[a], adjacent_copy(hidden[a], int(j))))
            weights.append(A[a, j])
        edges.append((hidden[a], y))
        weights.append(beta[a])
    bias = np.zeros(len(nodes))
    bias[y] = h
    for a, i in hidden.items():
        bias[i] = b[a]
    free = [hidden[a] for a in range(len(beta))] + [y]  # E0 joint order: w..., y
    return {
        "sites": sites,
        "edges": edges,
        "weights": np.asarray(weights, np.float64),
        "bias": bias,
        "free": free,
        "clamped": [c for c, _ in copies],
        "clamped_input": [k for _, k in copies],
        "first_colour": colour(sites[hidden[0]]),
    }


def exact_chromatic_kernels(model, x_batch):
    """Exact one-sweep kernel on the free nodes for each clamped input row (float64)."""
    free, nf = model["free"], len(model["free"])
    slot = {n: k for k, n in enumerate(free)}
    states = ((np.arange(1 << nf)[:, None] >> np.arange(nf)) & 1) * 2.0 - 1.0
    pair = np.zeros((nf, nf))
    ext = np.zeros((len(x_batch), nf))
    ext += model["bias"][free][None]
    clamp_pos = {c: k for k, c in enumerate(model["clamped"])}
    for (u, v), w in zip(model["edges"], model["weights"], strict=True):
        if u in slot and v in slot:
            pair[slot[u], slot[v]] += w
            pair[slot[v], slot[u]] += w
        else:
            f, c = (u, v) if u in slot else (v, u)
            ext[:, slot[f]] += w * x_batch[:, model["clamped_input"][clamp_pos[c]]]
    colours = np.array([colour(model["sites"][n]) for n in free])
    out = []
    for e in ext:
        kernel = np.eye(1 << nf)
        for c in (model["first_colour"], 1 - model["first_colour"]):
            block = colours == c
            field = e[None, :] + states @ pair  # local field of each node, per state
            p_up = expit(2.0 * field)  # THRML spin convention
            prob = np.where(states[None, :, :] > 0, p_up[:, None, :], 1 - p_up[:, None, :])
            keep = np.all(states[:, None, ~block] == states[None, :, ~block], axis=2)
            t = np.where(keep, np.prod(prob[:, :, block], axis=2), 0.0)
            kernel = kernel @ t
        out.append(kernel)
    return np.stack(out), states


def thrml_lowering(model, dtype="float32"):
    """Pairwise model -> THRML IsingEBM with free colour blocks and one clamped block."""
    import jax.numpy as jnp
    from thrml import Block, SpinNode
    from thrml.models import IsingEBM, IsingSamplingProgram

    nodes = [SpinNode() for _ in model["sites"]]
    ebm = IsingEBM(
        nodes,
        [(nodes[u], nodes[v]) for u, v in model["edges"]],
        jnp.asarray(model["bias"], dtype=dtype),
        jnp.asarray(model["weights"], dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
    )
    free = model["free"]
    by_colour = [
        Block([nodes[n] for n in free if colour(model["sites"][n]) == c])
        for c in (model["first_colour"], 1 - model["first_colour"])
    ]
    clamped = [Block([nodes[c] for c in model["clamped"]])]
    return ebm, IsingSamplingProgram(ebm, by_colour, clamped_blocks=clamped), by_colour


def z1_counts(exact_updates=None, **kw) -> dict:
    counts = Z1OperationCounts.constant_participation(**kw)
    if exact_updates is not None:  # sequential kernels: only the active kernel updates
        fields = {k: getattr(counts, k) for k in counts.__dataclass_fields__}
        counts = Z1OperationCounts(**{**fields, "gibbs_node_updates": exact_updates})
    proj = project_z1_operations(counts)
    return {
        "counts": {k: getattr(counts, k) for k in counts.__dataclass_fields__},
        "sampling_energy_j": proj.sampling_energy_j,
        "read_energy_j": proj.read_energy_j,
        "write_energy_j": proj.write_energy_j,
        "modeled_total_energy_j": proj.modeled_total_energy_j,
        "io_share": (proj.read_energy_j + proj.write_energy_j) / proj.modeled_total_energy_j,
        "evidence_class": str(proj.evidence_class),
    }


# --------------------------------------------------------------------------- parts


def p0_offsets() -> dict:
    from thermo_lab import planar_16_offset_ferro as planar

    return {
        "m5c_equals_z1": set(m5c.OFFSETS) == set(Z1_OFFSETS),
        "planar_equals_z1": set(planar.OFFSETS) == set(Z1_OFFSETS),
        "count": len(Z1_OFFSETS),
        "all_odd_parity": all((dx + dy) % 2 for dx, dy in Z1_OFFSETS),
    }


def p1_rounding(metric_cells=3) -> dict:
    results = m5c.load_source()["results"]  # M5b archive (authenticated)
    source = m5b.load_source()  # M5a archive holding the variational fits
    checked = equal_summary = equal_arrays = 0
    mismatches = []
    for key, value in results.items():
        cell = value["cell"]
        if cell["arm"] != "precision" or cell["reading"] != "B":
            continue
        vectors = m5b.parameters(source, "B", cell["seed"], cell["cap"], cell["method"])
        mine = layer_rounding_summary(vectors, cell["cap"], cell["value"])
        theirs, _ = core.round_parameters(vectors, cell["cap"], cell["value"])
        checked += 1
        equal_arrays += all(
            np.array_equal(layer_codebook(v, cell["cap"], cell["value"]), t)
            for v, t in zip(vectors, theirs, strict=True)
        )
        if mine == value["rounding"]:
            equal_summary += 1
        else:
            mismatches.append(key)
    # Chain metrics through layer-rounded vectors, replicating M5b's precision arm.
    metric_equal, metric_keys, seconds = 0, [], []
    for bits in (4, 8, 12)[:metric_cells]:
        key = f"B|0|1.0|variational|precision|{bits}"
        job = {"arm": "precision", "reading": "B", "seed": 0, "cap": 1.0, "method": "variational"}
        t0 = time.process_time()
        context, _ = m5b._context({**job, "value": bits})
        compiled = []
        for p, s in zip(context["vectors"], context["structures"], strict=True):
            q = layer_codebook(p, 1.0, bits)
            theta = m5a.kernel_logit(q, s, m5a.blanket_inputs(s))
            compiled.append(expit(2 * theta)[m5a.blanket_codes(s)])
        metrics = m5b._metrics(m5a.sweep(np.eye(m5a.STATES), compiled), context)
        seconds.append(time.process_time() - t0)
        archived = results[key]["metrics"]
        metrics = json.loads(json.dumps(metrics))
        same = metrics == archived
        metric_equal += same
        diffs = {
            k: float(np.max(np.abs(np.asarray(metrics[k]) - np.asarray(archived[k]))))
            for k in archived
        }
        metric_keys.append(
            {
                "cell": key,
                "bitwise": same,
                "max_abs_diff": max(diffs.values()),
                "fields_bitwise": sorted(k for k, d in diffs.items() if d == 0.0),
            }
        )
    return {
        "precision_cells_reading_b": checked,
        "rounded_arrays_bitwise_vs_round_parameters": equal_arrays,
        "rounding_summary_bitwise_vs_archive": equal_summary,
        "mismatches": mismatches[:10],
        "metric_cells": metric_keys,
        "metric_cells_bitwise": metric_equal,
        "metric_cpu_seconds_per_cell": seconds,
        "note": "round_parameters accepts bits in {4,8,12} and raises outside the cap; "
        "the layer codebook clips and takes any width, and equals it bitwise on that domain",
    }


def _m5c_vectors():
    archive = json.loads(gzip.open(M5C_ARCHIVE).read())
    source = m5c.load_source()
    refs = m5c.references(source, m5c.SEEDS)
    vectors = {seed: m5c._vectors(refs, archive["refits"], seed) for seed in m5c.SEEDS}
    return archive, vectors


def p2_placement(resolve_limit_seconds=5.0, max_resolves=12) -> dict:
    archive, vectors = _m5c_vectors()
    rows, verify_equal, lattice_ok = [], 0, 0
    fields = ("conditional_error", "j_copies", "a_copies", "copies", "physical_pbits")
    for key, placed in archive["placements"].items():
        seed, site = map(int, key.split("|"))
        vecs, structs = vectors[seed]
        p, s = vecs["refit"][site], structs[site]
        model = lower_placed_kernel(placed["layout"], p, s)
        check = lattice_check(model["sites"], model["edges"])
        ok = (
            check["offset_violations"] == 0
            and check["colour_violations"] == 0
            and check["degree_violations"] == 0
            and check["distinct_sites"]
        )
        lattice_ok += ok
        again = m5c.verify_layout(placed["layout"], p, s)
        verify_equal += all(again[f] == placed[f] for f in fields)
        rows.append({"key": key, "lattice_ok": ok, "max_degree": check["max_degree"]})
    # Re-solve the fastest archived placements (MILP, HiGHS) and compare layouts.
    prov = json.loads((M5C_ARCHIVE.parent / "provenance.json").read_text())
    order = list(archive["placements"])  # generation order: seed-major, site 0..11
    jobs = [f"{s}|{i}" for s in m5c.SEEDS for i in range(m5a.D)]
    times = dict(zip(jobs, (t["seconds"] for t in prov["unit_seconds"]["placement"]), strict=True))
    fast = sorted((k for k in order if times[k] <= resolve_limit_seconds), key=times.get)
    resolves = []
    for key in fast[:max_resolves]:
        seed, site = map(int, key.split("|"))
        vecs, structs = vectors[seed]
        t0 = time.perf_counter()
        layout, solver = m5c.place_kernel(vecs["refit"][site], structs[site])
        resolves.append(
            {
                "key": key,
                "archived_seconds": times[key],
                "seconds": time.perf_counter() - t0,
                "layout_bitwise": layout == archive["placements"][key]["layout"],
                "copies_equal": sum(e[2] == "copy" for e in layout)
                == archive["placements"][key]["copies"],
                "optimal": solver["optimal"],
            }
        )
    return {
        "placements": len(rows),
        "layer_lattice_check_passed": lattice_ok,
        "verify_layout_fields_bitwise": verify_equal,
        "max_degree_seen": max(r["max_degree"] for r in rows),
        "resolved": resolves,
        "placement_cpu_seconds_archived_total": sum(times.values()),
    }


def p2_planar() -> dict:
    from thermo_lab import planar_16_offset_ferro as planar

    with tarfile.open(PLANAR_TAR) as tar:
        member = tar.extractfile("planar-16-offset-ferro/request.json")
        request = json.load(io.TextIOWrapper(member))
    rows = []
    for target in request["targets"]:
        rebuilt = planar.build_target(target["size"], target["graph"], target["seed"])
        positions = planar.target_positions(target)
        sites = [tuple(map(int, p)) for p in positions]
        edges = [tuple(e) for e in target["edges"]]
        check = lattice_check(sites, edges)
        rows.append(
            {
                "id": target["id"],
                "rebuilt_bitwise": rebuilt["edges"] == target["edges"]
                and rebuilt["couplings"] == target["couplings"],
                "lattice_ok": check["offset_violations"] == 0 and check["colour_violations"] == 0,
                "max_degree": check["max_degree"],
            }
        )
    return {
        "targets": len(rows),
        "rebuilt_bitwise": sum(r["rebuilt_bitwise"] for r in rows),
        "lattice_ok": sum(r["lattice_ok"] for r in rows),
        "max_degree": max(r["max_degree"] for r in rows),
    }


def p2_am_degree() -> dict:
    """The bias design is complete bipartite: hidden degree N, visible degree P."""
    out = {}
    n = 24
    for p in (8, 32, 128):
        for cue in (12, 8):
            # clamped cue bits are inputs; copies can repeat them (M5c), free spins cannot.
            free_vis_degree, hidden_degree = p, n
            chain = {d: 1 if d <= 16 else math.ceil((d - 2) / 14) for d in (p, n)}
            out[f"P{p}/c{cue}"] = {
                "hidden_degree": hidden_degree,
                "free_visible_degree": free_vis_degree,
                "over_degree": max(hidden_degree, free_vis_degree) > 16,
                "chain_nodes_lower_bound": (n - cue) * chain[p] + p * chain[n],
                "logical_free": (n - cue) + p,
                "note": "lower bound on p-bits if every free spin of degree d became a "
                "ferromagnetic chain of ceil((d-2)/14) nodes; not a placement, changes the law",
            }
    return out


def p3_e0b(run_thrml=True, unplaced_cell_ids=("y0-/K1", "y0+/K1", "y0-/K2")) -> dict:
    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0

    record = json.loads(gzip.open(E0B_ARCHIVE).read())
    request = record["request"]
    loaded = e0.load_kernel(request)
    structure, vector = loaded["structure"], loaded["vector"]
    archive, vectors = _m5c_vectors()
    placed = archive["placements"]["0|1"]
    refit = vectors[0][0]["refit"][1]
    out = {
        "e0_kernel_output_degree": m5c.output_degree(vector, structure),
        "e0_kernel_j_mask": m5c.j_mask(vector, structure),
        "placed_vector_is_m5c_refit": True,
    }
    model = lower_placed_kernel(placed["layout"], refit, structure)
    inner = core.inner_kernel(refit, structure)  # the placed kernel's own exact law
    inner_e0 = core.inner_kernel(vector, structure)  # the archived E0 / M5b law
    inputs = m5a.blanket_inputs(structure)
    nh = len(structure["triples"])
    nf = nh + 1
    y_plus = None
    out.update(placed_physical_pbits=len(model["sites"]), free=nf)
    t0 = time.process_time()
    max_dev = {k: 0.0 for k in request["sweeps"]}
    gap = {k: 0.0 for k in request["sweeps"]}
    joint_k1 = {}
    for start in range(0, len(inputs), 64):
        xb = inputs[start : start + 64]
        kernels, states = exact_chromatic_kernels(model, xb)
        y_plus = states[:, nh] > 0
        for y0 in (-1, 1):
            p0 = np.zeros(1 << nf)
            p0[e0.joint_index(np.array([[-1.0] * nh + [float(y0)]]))[0]] = 1.0
            law = np.broadcast_to(p0, (len(xb), 1 << nf))
            for k in range(1, max(request["sweeps"]) + 1):
                law = np.einsum("is,ist->it", law, kernels)
                if k in request["sweeps"]:
                    mine = law[:, y_plus].sum(axis=1)
                    ref = e0.output_law(inner, k, y0)[start : start + 64]
                    max_dev[k] = max(max_dev[k], float(np.max(np.abs(mine - ref))))
                    arch = e0.output_law(inner_e0, k, y0)[start : start + 64]
                    gap[k] = max(gap[k], float(np.max(np.abs(mine - arch))))
                if k == 1 and y0 == -1:
                    joint_k1[start] = law.copy()
    # one-input cross-check against E0's own joint enumeration
    ref_kernel = e0.joint_sweep_kernel(refit, structure, inputs[5], "hidden_first")
    mine, _ = exact_chromatic_kernels(model, inputs[5:6])
    out["placed_exact_vs_refit_inner_k_law_max_abs"] = {str(k): v for k, v in max_dev.items()}
    out["placed_exact_vs_archived_e0_law_max_abs"] = {str(k): v for k, v in gap.items()}
    out["exact_joint_kernel_vs_e0_one_input_max_abs"] = float(np.max(np.abs(mine[0] - ref_kernel)))
    out["exact_cpu_seconds"] = time.process_time() - t0
    if not run_thrml:
        return out
    import jax
    import jax.numpy as jnp
    from thrml import SamplingSchedule, sample_states

    cell = next(c for c in e0.cells(request) if c["id"] == "y0-/K1")
    # (b) unplaced lowering: the layer emits E0's node and edge order -> same program.
    # Placement relaxed: sites only carry the colour (inputs/output colour 0, hidden 1).
    J, h, A, b, beta = m5a.unpack(vector, structure)
    k_in = len(J)
    weights, edges = [], []
    for j in range(k_in):
        edges.append((j, k_in + nh))
        weights.append(J[j])
    for a in range(nh):
        for j in range(k_in):
            edges.append((j, k_in + a))
            weights.append(A[a, j])
        edges.append((k_in + a, k_in + nh))
        weights.append(beta[a])
    bias = np.zeros(k_in + nh + 1)
    bias[k_in : k_in + nh] = b
    bias[-1] = h
    unplaced = {
        "sites": [(0, 0)] * k_in + [(1, 0)] * nh + [(0, 0)],
        "edges": edges,
        "weights": np.asarray(weights),
        "bias": bias,
        "free": list(range(k_in, k_in + nh + 1)),
        "clamped": list(range(k_in)),
        "clamped_input": list(range(k_in)),
        "first_colour": 1,
    }

    def run(model_, chains, cell):
        _, program, blocks = thrml_lowering(model_)
        n_hid = len(blocks[0].nodes)
        schedule = SamplingSchedule(n_warmup=cell["sweeps"], n_samples=1, steps_per_sample=1)
        clamp_cols = np.asarray(model_["clamped_input"])

        def one_chain(key, x):
            _, sample_key = jax.random.split(key, 2)
            state = [jnp.zeros((n_hid,), dtype=jnp.bool_), jnp.asarray([cell["incoming"] == 1])]
            observed = sample_states(sample_key, program, schedule, state, [x], blocks)
            return jnp.concatenate([observed[0][0], observed[1][0]])

        root = jax.random.fold_in(jax.random.key(request["jax_root_seed"]), cell["index"])
        fn = jax.jit(jax.vmap(one_chain))
        hist = np.zeros((len(inputs), 1 << nf), dtype=np.int64)
        batch = request["inputs_per_batch"]
        for start in range(0, len(inputs), batch):
            rows = inputs[start : start + batch]
            keys = jax.random.split(jax.random.fold_in(root, start), len(rows) * chains)
            x = jnp.asarray(np.repeat(rows[:, clamp_cols] > 0, chains, axis=0))
            res = np.asarray(fn(keys, x).block_until_ready())
            idx = e0.joint_index(res).reshape(len(rows), chains)
            for r in range(len(rows)):
                hist[start + r] = np.bincount(idx[r], minlength=hist.shape[1])
        return hist

    chains = request["chains_per_input"]
    unplaced_cells = {}
    for cid in unplaced_cell_ids:
        c = next(x for x in e0.cells(request) if x["id"] == cid)
        t0 = time.perf_counter()
        h_ = run(unplaced, chains, c)
        unplaced_cells[cid] = {
            "histogram_bitwise": bool(np.array_equal(h_, np.asarray(record["histograms"][cid]))),
            "seconds": time.perf_counter() - t0,
        }
    out["thrml_unplaced_cells"] = unplaced_cells
    archived = np.asarray(record["histograms"][cell["id"]])
    # (c) placed lowering: input copies as clamped nodes
    t0 = time.perf_counter()
    hist_p = run(model, chains, cell)
    out["thrml_placed_seconds"] = time.perf_counter() - t0
    law = np.concatenate([joint_k1[s] for s in sorted(joint_k1)])
    emp = hist_p / chains
    tol = record["references"]["cells"][cell["id"]]
    out_dev = float(np.max(np.abs(emp[:, y_plus].sum(axis=1) - law[:, y_plus].sum(axis=1))))
    joint_tv = float(np.max(0.5 * np.abs(emp - law).sum(axis=1)))
    out["thrml_placed"] = {
        "cell": cell["id"],
        "output_max_dev": out_dev,
        "tolerance_output": tol["tolerance_output"],
        "joint_max_tv": joint_tv,
        "tolerance_joint": tol["tolerance_joint"],
        "pass": out_dev <= tol["tolerance_output"] and joint_tv <= tol["tolerance_joint"],
        "histogram_bitwise_vs_archive": bool(np.array_equal(hist_p, archived)),
        "max_input_tv_vs_archived_e0_histogram": float(
            np.max(0.5 * np.abs(emp - archived / chains).sum(axis=1))
        ),
        "note": "placed kernel is the M5c refit (J[4] masked), not the E0 vector; "
        "compared with its own exact law using E0's archived tolerance for this cell",
    }
    return out


def p4_am(run_sampled=True) -> dict:
    from thermo_lab import am_binary_emulation as stage_a
    from thermo_lab import am_coupling_bits as am

    record = json.loads(gzip.open(AM_ARCHIVE).read())
    request, configs = record["request"], record["configs"]
    t0 = time.process_time()
    equal = total = 0
    worst = []
    for uid, unit in record["exact"].items():
        _, pc, cc, sc = uid.split("/")
        p, cue, seed = int(pc[1:]), int(cc[1:]), int(sc[1:])
        beta, theta = am.parse_config(configs[f"{pc}/{cc}"])
        xi = stage_a.patterns(seed, p)
        z = am.nominal_terms(beta, theta, xi)
        cap = 31 * float(np.abs(z["j_vh"]).max())  # cap rule `coupling` at 6 bits: step = J
        zq = {k: layer_codebook(v, cap, 6) for k, v in z.items()}
        mine = [am.exact_recall_terms(zq, xi, t, cue) for t in range(request["targets"])]
        total += 1
        equal += mine == unit["models"]["coupling/6"]
        worst.append(
            max(abs(a - b) for a, b in zip(mine, unit["models"]["coupling/6"], strict=True))
        )
    out = {
        "exact_units": total,
        "exact_bitwise": equal,
        "exact_max_abs_diff": max(worst),
        "exact_cpu_seconds": time.process_time() - t0,
    }
    if not run_sampled:
        return out
    seed, p, cue = 7400, 8, 12
    beta, theta = am.parse_config(configs["P8/c12"])
    xi = stage_a.patterns(seed, p)
    z = am.nominal_terms(beta, theta, xi)
    zq = {k: layer_codebook(v, 31 * float(np.abs(z["j_vh"]).max()), 6) for k, v in z.items()}
    st = am.gain_structure(p, request["n"], cue)
    key = am.unit_key(request, seed, p, cue)
    max_k = max(request["budgets"])
    sampled = {}
    t0 = time.perf_counter()
    for jitter in (0.0, 0.1):
        gv, gh = layer_gains([seed, p, cue, am.GAIN_STREAM, 0], (request["n"], p), jitter)
        states = am.drive(
            st,
            st.gain_program(zq, gv, gh),
            xi,
            cue,
            max_k,
            request["chains"],
            key,
            request["targets"],
            max_k,
        )
        counts = am.correct_counts(
            states[0], xi, cue, request["budgets"], request["chains"], request["targets"]
        )
        rid = f"coupling/6/j{jitter:g}/d0"
        sampled[rid] = counts == record["sampled"][f"sampled/P8/c12/s{seed}"]["runs"][rid]
    out["sampled_runs_bitwise"] = sampled
    out["sampled_seconds"] = time.perf_counter() - t0
    return out


def p5_pricing() -> dict:
    """One priced example per constraint; counts are algorithmic, not device operations."""
    archive = json.loads(gzip.open(M5C_ARCHIVE).read())
    patch = archive["patches"]["0"]
    copies0 = sum(archive["placements"][f"0|{s}"]["copies"] for s in range(m5a.D))
    examples = {}
    # placement: one M5c outer sweep (seed 0) at K = 4 on the packed patch
    req = layer_request(model="m5c seed 0 refit, packed patch", bits=None, jitter=0.0)
    req["relaxed"] = ["precision (float64 parameters)", "beta jitter"]
    examples["placement"] = {
        "request_digest": canonical_sha256(req),
        "relaxed": req["relaxed"],
        "unit": "one outer sweep: every site kernel K = 4 inner sweeps, clamps rewritten",
        **z1_counts(
            exact_updates=4 * 72,
            logical_pbits=72,
            physical_pbits_used=patch["sites"],
            participating_free_pbits=72,
            elapsed_complete_sweeps=4 * m5a.D,
            node_reads=12,
            node_full_sram_writes=copies0 + 72,
            clamp_state_changes=copies0,
        ),
        "note": "kernels run one site at a time (M5b schedule): 12 x 4 complete sweeps "
        "elapsed, 4 x 72 updates (each free p-bit of the active kernel once per sweep); "
        "clamp writes are the placed copies (272), resets the 72 free p-bits",
    }
    # quantization / jitter: am-coupling-bits P8/c12 at 6 bits, step = J; one chain
    for name, jitter, relaxed in (
        ("quantization", 0.0, ["placement (complete bipartite, degree 24 and 8)", "beta jitter"]),
        ("beta_jitter", 0.1, ["placement (complete bipartite, degree 24 and 8)"]),
    ):
        req = layer_request(model="am bias P8/c12", bits=6, cap_rule="coupling", jitter=jitter)
        req["relaxed"] = relaxed
        examples[name] = {
            "request_digest": canonical_sha256(req),
            "relaxed": req["relaxed"],
            "unit": "one recalled pattern: cue written once, 256 sweeps, missing bits read once",
            **z1_counts(
                logical_pbits=32,
                physical_pbits_used=32,
                participating_free_pbits=20,
                elapsed_complete_sweeps=256,
                node_reads=12,
                node_full_sram_writes=12 + 8 + 12,
                clamp_state_changes=12,
            ),
        }
    # chromatic schedule / no exchange: planar 16-offset ferro L32 single chain, 4096 sweeps
    req = layer_request(model="planar-16-offset-ferro L32 greedy-long", bits=None, jitter=0.0)
    req["relaxed"] = ["precision (float32 couplings)", "beta jitter"]
    examples["chromatic_no_exchange"] = {
        "request_digest": canonical_sha256(req),
        "relaxed": req["relaxed"],
        "unit": "one chain, 4096 sweeps, all spins read every 4 sweeps after 25% burn-in",
        **z1_counts(
            logical_pbits=1024,
            physical_pbits_used=1024,
            participating_free_pbits=1024,
            elapsed_complete_sweeps=4096,
            node_reads=1024 * (3072 // 4),
            node_full_sram_writes=1024,
        ),
    }
    flipped = layer_request(model="planar-16-offset-ferro L32 greedy-long", bits=None, jitter=0.0)
    flipped["relaxed"] = ["precision (float32 couplings)"]
    examples["relaxed_flag_changes_digest"] = (
        canonical_sha256(flipped) != examples["chromatic_no_exchange"]["request_digest"]
    )
    return examples


PARTS = {
    "p0": p0_offsets,
    "p1": p1_rounding,
    "p2": p2_placement,
    "p2_planar": p2_planar,
    "p2_am": p2_am_degree,
    "p3": p3_e0b,
    "p4": p4_am,
    "p5": p5_pricing,
}


def main() -> None:
    wanted = sys.argv[1:] or list(PARTS)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results["label"] = "exploration, not evidence (loop P-0004 probe)"
    for name in wanted:
        t0, c0 = time.perf_counter(), time.process_time()
        results[name] = PARTS[name]()
        results[name]["_wall_seconds"] = time.perf_counter() - t0
        results[name]["_cpu_seconds_main"] = time.process_time() - c0
        print(name, json.dumps(results[name], default=str)[:600], flush=True)
        OUT.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
