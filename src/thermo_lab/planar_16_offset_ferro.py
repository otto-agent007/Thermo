"""Planar Ising scaling allocations on a planar subgraph of the Z1 16-offset lattice.

Protocol: ``docs/experiments/planar-16-offset-ferro.md`` (proposal P-0001).
The planar Ising scaling study (PR #92) ran on open square grids. The Z1
interior rule links a site to 16 offsets, (1,0), (2,1), (2,3), (4,1) and their
rotations; that graph is not planar. This study takes the ``greedy-long``
maximal straight-line planar subgraph of the rule on an L x L patch (all
candidate edges longest first, ties in an order seeded by the target's coupling
seed, each kept if its segment crosses no kept segment) and compares the six
#92 arms on it against a paired open grid with the same coupling seeds.

Every offset has odd parity, so every subgraph is bipartite under the (x + y)
checkerboard and the two-colour block-Gibbs sweep stays exact. The exact
reference is the Kac-Ward determinant on an explicit straight-line embedding
(study-local, checked bitwise against ``planar_ising_scaling.kac_ward`` on
grids and against brute force on small 16-offset patches). The sampler is the
#92 sampler over a padded neighbour list, checked bitwise against
``planar_ising_scaling.compile_sampler`` on a grid. Pricing, qualification,
snapshot plans and window sums are imported from #92 unchanged.

Long run (about 1 to 2 h): every (target, arm) unit is written atomically to
``units/`` with a digest in ``units/index.json``; ``--resume`` re-authenticates
the request and sources, verifies each unit by digest, and runs only the
missing ones.

Sweeps are software_simulation (CPU, JAX float32); references are
exact_reference (float64); Z1 energies and sweep times are
calibrated_projection. Nothing here ran on hardware, and the subgraph is a
synthetic patch of the published local rule, not the physical chip map.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import itertools
import json
import multiprocessing
import os
import shutil
import time
import traceback
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from thermo_lab import planar_ising_scaling as study
from thermo_lab.exchange_cost_projection import CONVENTIONS
from thermo_lab.hardware.z1 import Z1HardwareProfile
from thermo_lab.hashing import canonical_sha256
from thermo_lab.provenance import collect_runtime_provenance

base = study.base
ROOT = study.ROOT
SOURCES = (
    "src/thermo_lab/planar_16_offset_ferro.py",
    "src/thermo_lab/planar_ising_scaling.py",
    "src/thermo_lab/fixed_budget_sampling.py",
    "src/thermo_lab/exchange_cost_projection.py",
    "src/thermo_lab/hardware/z1.py",
    "src/thermo_lab/hashing.py",
    "src/thermo_lab/persistence.py",
    "docs/experiments/planar-16-offset-ferro.md",
    "uv.lock",
)
SIZES = (8, 16, 32)
GRAPHS = ("grid", "greedy-long")
SEEDS = (4510, 4511, 4512)
BUDGETS = (64, 256, 1024, 4096, 16384)
TRIALS = 16
ROOT_SEED = 20261013
THRESHOLD = 0.05
ARMS = study.ARMS
COLD_BETA = study.COLD_BETA
OFFSETS = tuple(sorted(Z1HardwareProfile().interior_offsets()))
PRIMARY_SIZES = (16, 32)
NINE = ("tempering9-k1", "tempering9-k4")
FIVE = ("tempering5-k1", "tempering5-k4")
ORDINARY = ("long", "independent")
VERDICTS = {
    "holds": "#92 holds on the 16-offset subgraph",
    "holds_thin_ladder_slip": "#92 holds with a thin-ladder slip",
    "does_not_hold": "#92 does not hold",
}
DEFAULT_WORKERS = 2

# ----------------------------------------------------------------------------
# Graphs and targets
# ----------------------------------------------------------------------------


def positions_for(rows, cols):
    """Site positions (x, y) = (col, row); site index row * cols + col."""
    return np.array([(c, r) for r in range(rows) for c in range(cols)], dtype=np.int64)


def _side(p, q, a):
    """Sign of the turn p -> q -> a for every row of ``a``."""
    return np.sign((q[0] - p[0]) * (a[:, 1] - p[1]) - (q[1] - p[1]) * (a[:, 0] - p[0]))


def planar_subgraph(rows, cols, family, seed):
    """Edges (i, j), i < j, sorted, and positions for one planar family.

    ``grid`` returns ``planar_ising_scaling.grid_edges`` order on square
    patches (horizontal rows, then vertical), so couplings align with #92.
    ``greedy-long`` and ``greedy-random`` insert 16-offset candidate edges in
    a seeded order (longest first for ``greedy-long``, stable within a
    length) and keep each one whose segment crosses no kept segment.
    """
    positions = positions_for(rows, cols)
    if family == "grid":
        edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
        edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
        return edges, positions
    if family not in ("greedy-long", "greedy-random"):
        raise ValueError(f"unknown graph family: {family}")
    candidates = set()
    for r in range(rows):
        for c in range(cols):
            for dx, dy in OFFSETS:
                rr, cc = r + dy, c + dx
                if 0 <= rr < rows and 0 <= cc < cols:
                    a, b = r * cols + c, rr * cols + cc
                    candidates.add((min(a, b), max(a, b)))
    candidates = sorted(candidates)
    order = np.random.default_rng(seed).permutation(len(candidates))
    candidates = [candidates[i] for i in order]
    if family == "greedy-long":
        length = {e: int(np.sum((positions[e[0]] - positions[e[1]]) ** 2)) for e in candidates}
        candidates.sort(key=lambda e: -length[e])  # stable: seeded order within a length
    kept = []
    a_pts = np.zeros((len(candidates), 2), np.int64)
    b_pts = np.zeros((len(candidates), 2), np.int64)
    for e in candidates:
        p, q = positions[e[0]], positions[e[1]]
        k = len(kept)
        if k:
            a, b = a_pts[:k], b_pts[:k]
            # Offsets are primitive vectors, so no lattice site lies inside a
            # segment and collinear overlap cannot occur between kept edges
            # that share no endpoint; a strict sign test finds every crossing.
            d1, d2 = _side(p, q, a), _side(p, q, b)
            d3 = np.sign(
                (b[:, 0] - a[:, 0]) * (p[1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (p[0] - a[:, 0])
            )
            d4 = np.sign(
                (b[:, 0] - a[:, 0]) * (q[1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (q[0] - a[:, 0])
            )
            if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
                continue
        kept.append(e)
        a_pts[k], b_pts[k] = p, q
    kept.sort()
    return kept, positions


def graph_stats(edges, positions):
    n = len(positions)
    degree = np.bincount(np.asarray(edges).ravel(), minlength=n)
    classes = {}
    for i, j in edges:
        d = tuple(sorted(np.abs(positions[j] - positions[i]).tolist()))
        classes[f"{d[0]},{d[1]}"] = classes.get(f"{d[0]},{d[1]}", 0) + 1
    return {
        "sites": int(n),
        "edges": len(edges),
        "planar_bipartite_bound_2n_minus_4": int(2 * n - 4),
        "mean_degree": float(degree.mean()),
        "max_degree": int(degree.max()),
        "isolated_sites": int((degree == 0).sum()),
        "edges_by_offset_class": dict(sorted(classes.items())),
        "bipartite_checkerboard": bool(
            all((positions[i].sum() + positions[j].sum()) % 2 == 1 for i, j in edges)
        ),
    }


def build_target(size, graph, seed):
    edges, _ = planar_subgraph(size, size, graph, seed)
    rng = np.random.default_rng(seed * 1000 + size)
    weights = rng.integers(1, 6, len(edges)).astype(np.float32) / 5
    return {
        "id": f"L{size}-{graph}-s{seed}",
        "size": size,
        "n": size * size,
        "graph": graph,
        "seed": seed,
        "edges": [list(e) for e in edges],
        # Archived sign convention; ferromagnetic by gauge on a bipartite graph.
        "couplings": (-weights).tolist(),
    }


def study_targets():
    return [build_target(size, graph, seed) for size in SIZES for graph in GRAPHS for seed in SEEDS]


def edge_tuples(target):
    return [tuple(e) for e in target["edges"]]


def target_positions(target):
    return positions_for(target["size"], target["size"])


# ----------------------------------------------------------------------------
# Exact reference: Kac-Ward on a straight-line planar embedding
# ----------------------------------------------------------------------------


def kac_ward_matrix(positions, edges, interactions):
    """``planar_ising_scaling.kac_ward_matrix`` with the site positions as an argument."""
    m = len(edges)
    position = np.asarray(positions, dtype=float)
    x = np.tanh(np.asarray(interactions, dtype=float))
    tail = np.array([e[0] for e in edges] + [e[1] for e in edges])
    head = np.array([e[1] for e in edges] + [e[0] for e in edges])
    weight = np.concatenate([x, x])
    delta = position[head] - position[tail]
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    follows = head[:, None] == tail[None, :]
    reverse = (np.arange(2 * m)[:, None] + m) % (2 * m) == np.arange(2 * m)[None, :]
    rows, cols = np.nonzero(follows & ~reverse)
    turn = (angle[cols] - angle[rows] + np.pi) % (2 * np.pi) - np.pi
    lam = np.zeros((2 * m, 2 * m), dtype=complex)
    lam[rows, cols] = np.exp(1j * turn / 2) * weight[cols]
    return np.eye(2 * m) - lam, lam, x


def kac_ward_log_z(positions, edges, interactions):
    matrix, _, _ = kac_ward_matrix(positions, edges, interactions)
    sign, logdet = np.linalg.slogdet(matrix)
    if abs(sign - 1.0) > 1e-6:
        raise ValueError("Kac-Ward determinant is not positive real")
    n = len(positions)
    return float(n * np.log(2) + np.sum(np.log(np.cosh(interactions))) + 0.5 * logdet.real)


def kac_ward(positions, edges, interactions):
    """ln Z and <s_i s_j> per edge; same algebra as ``planar_ising_scaling.kac_ward``."""
    matrix, _, x = kac_ward_matrix(positions, edges, interactions)
    log_z = kac_ward_log_z(positions, edges, interactions)
    m = len(x)
    diagonal = np.diag(np.linalg.inv(matrix)).real - 1.0
    factor = (1 - x**2) / x
    correlation = x - 0.5 * factor * (diagonal[:m] + diagonal[m:])
    if np.any(np.abs(correlation) > 1 + 1e-6):
        raise ValueError("edge correlation outside [-1, 1]")
    return log_z, np.clip(correlation, -1, 1)


def brute_force(n, edges, interactions):
    if n > 16:
        raise ValueError("brute force is bounded to 16 spins")
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    products = states[:, [e[0] for e in edges]] * states[:, [e[1] for e in edges]]
    energy = products @ np.asarray(interactions, dtype=float)
    weights = np.exp(energy - energy.max())
    return float(np.log(weights.sum()) + energy.max()), (weights @ products) / weights.sum()


def reference(target):
    """Exact edge correlations, finite-difference precision and (small grids) transfer matrix."""
    edges, positions = edge_tuples(target), target_positions(target)
    interactions = COLD_BETA * np.asarray(target["couplings"], dtype=float)
    log_z, correlation = kac_ward(positions, edges, interactions)
    checked = np.argsort(np.abs(correlation))[:2]
    worst = 0.0
    for e in checked:
        plus, minus = interactions.copy(), interactions.copy()
        plus[e] += 1e-4
        minus[e] -= 1e-4
        difference = kac_ward_log_z(positions, edges, plus) - kac_ward_log_z(
            positions, edges, minus
        )
        worst = max(worst, abs(difference / 2e-4 - correlation[e]))
    if worst > 1e-4:
        raise ValueError(f"Kac-Ward edge correlations lost precision on {target['id']}")
    result = {
        "evidence_class": "exact_reference",
        "method": "kac_ward_determinant_explicit_embedding",
        "log_z": log_z,
        "edge": correlation.tolist(),
        "q_per_spin": float(np.asarray(target["couplings"]) @ correlation / target["n"]),
        "min_abs_edge_correlation": float(np.abs(correlation).min()),
        "finite_difference_checked_edges": [int(e) for e in checked],
        "finite_difference_max_abs_error": worst,
    }
    if target["graph"] == "grid" and target["size"] <= 16:
        tm = study.transfer_matrix_log_z(target["size"], interactions)
        result["transfer_matrix_log_z"] = tm
        if abs(tm - log_z) > 1e-8 * max(1.0, abs(log_z)):
            raise ValueError("Kac-Ward and transfer-matrix ln Z disagree")
    return result


def reference_checks():
    """Study-local Kac-Ward against the imported grid version and against brute force."""
    rng = np.random.default_rng(2027)
    grid_rows = []
    for size in (3, 8):
        positions = positions_for(size, size)
        edges = study.grid_edges(size)
        k = COLD_BETA * -(rng.integers(1, 6, len(edges)) / 5)
        mine_matrix = kac_ward_matrix(positions, edges, k)[0]
        theirs_matrix = study.kac_ward_matrix(size, k)[0]
        mine, theirs = kac_ward(positions, edges, k), study.kac_ward(size, k)
        grid_rows.append(
            {
                "size": size,
                "matrix_bitwise_equal": bool(np.array_equal(mine_matrix, theirs_matrix)),
                "log_z_bitwise_equal": bool(mine[0] == theirs[0]),
                "edge_bitwise_equal": bool(np.array_equal(mine[1], theirs[1])),
            }
        )
    patches = []
    for rows, cols in ((4, 4), (3, 5), (2, 8)):
        for family in ("greedy-long", "greedy-random"):
            for seed in (1, 2):
                edges, positions = planar_subgraph(rows, cols, family, seed)
                for beta, sign in ((COLD_BETA, "ferro"), (1.0, "mixed")):
                    w = rng.integers(1, 6, len(edges)) / 5
                    s = (
                        -np.ones(len(edges))
                        if sign == "ferro"
                        else rng.choice([-1.0, 1.0], len(edges))
                    )
                    k = beta * s * w
                    log_z, corr = kac_ward(positions, edges, k)
                    exact_log_z, exact_corr = brute_force(rows * cols, edges, k)
                    patches.append(
                        {
                            "patch": [rows, cols],
                            "family": family,
                            "graph_seed": seed,
                            "edges": len(edges),
                            "non_grid_edges": int(
                                sum(np.abs(positions[j] - positions[i]).sum() > 1 for i, j in edges)
                            ),
                            "couplings": sign,
                            "beta": beta,
                            "log_z_abs_error": abs(log_z - exact_log_z),
                            "edge_max_abs_error": float(np.max(np.abs(corr - exact_corr))),
                        }
                    )
    result = {
        "evidence_class": "exact_reference",
        "grid_vs_imported": grid_rows,
        "brute_force_patches": patches,
        "max_log_z_error": max(p["log_z_abs_error"] for p in patches),
        "max_edge_error": max(p["edge_max_abs_error"] for p in patches),
        "tolerance": 1e-9,
    }
    result["passed"] = bool(
        all(all(v for k, v in row.items() if k != "size") for row in grid_rows)
        and len(patches) >= 12
        and all(p["non_grid_edges"] > 0 for p in patches)
        and result["max_log_z_error"] <= 1e-9
        and result["max_edge_error"] <= 1e-9
    )
    return result


# ----------------------------------------------------------------------------
# Sampler: the #92 two-colour block Gibbs over a padded neighbour list
# ----------------------------------------------------------------------------


def neighbour_arrays(n, edges, couplings):
    """Neighbours of each site in edge-list order, padded with weight 0."""
    adjacency = [[] for _ in range(n)]
    for (i, j), c in zip(edges, couplings, strict=True):
        adjacency[i].append((j, c))
        adjacency[j].append((i, c))
    width = max(1, max(len(a) for a in adjacency))
    nbr = np.zeros((n, width), np.int32)
    wts = np.zeros((n, width), np.float32)
    for i, row in enumerate(adjacency):
        for k, (j, c) in enumerate(row):
            nbr[i, k], wts[i, k] = j, c
    return nbr, wts


def compile_sampler(method, replicas, interval, target, horizon, snapshot_steps):
    """``planar_ising_scaling.compile_sampler`` on an arbitrary checkerboard-bipartite graph.

    Inputs per trial: a key and the (9, n) initial state. Returns prefix sums of
    the retained replicas' edge products at ``snapshot_steps`` and the
    exchange flags per sweep, exactly as the #92 sampler. The local field is
    accumulated neighbour by neighbour in edge-list order, which on a grid is
    the #92 order (left, right, up, down), so the float32 fields agree bitwise.
    """
    if method not in base.METHODS:
        raise ValueError("unknown sampling method")
    n = target["n"]
    edges = edge_tuples(target)
    couplings = np.asarray(target["couplings"], np.float32)
    nbr_np, wts_np = neighbour_arrays(n, edges, couplings)
    nbr, wts = jnp.asarray(nbr_np), jnp.asarray(wts_np)
    width = nbr_np.shape[1]
    ei = jnp.asarray([e[0] for e in edges], jnp.int32)
    ej = jnp.asarray([e[1] for e in edges], jnp.int32)
    jvec = jnp.asarray(couplings)
    steps = 5 * horizon if method == "long" else horizon
    ladder = study.LADDERS[replicas] if method == "tempering" else (COLD_BETA,) * replicas
    beta = jnp.asarray(ladder, jnp.float32)
    pairs = max(1, (replicas - 1) // 2)
    position = target_positions(target)
    checker = jnp.asarray((position.sum(axis=1) % 2) == 0)
    snapshot_steps = jnp.asarray(snapshot_steps, jnp.int32)
    m = len(edges)

    def products(state):
        spins = 2 * state.astype(jnp.int32) - 1
        return spins[..., ei] * spins[..., ej]

    def single(key, initial):
        start = initial[:replicas]

        def half_sweep(state, key, mask):
            spins = 2 * state.astype(jnp.float32) - 1
            field = jnp.zeros_like(spins)
            for k in range(width):
                field = field + wts[:, k] * spins[:, nbr[:, k]]
            draw = jax.random.bernoulli(key, jax.nn.sigmoid(2 * beta[:, None] * field))
            return jnp.where(mask, draw, state)

        def step(carry, t):
            state, key, sums, snaps = carry
            key, key_a, key_b, exchange_key = jax.random.split(key, 4)
            state = half_sweep(state, key_a, checker)
            state = half_sweep(state, key_b, ~checker)
            accepted = jnp.zeros(pairs, dtype=jnp.bool_)
            if method == "tempering":
                due = (t + 1) % interval == 0
                exchange_index = (t + 1) // interval - 1
                left = 2 * jnp.arange(pairs) + exchange_index % 2
                right = left + 1
                valid = right < replicas
                left = jnp.where(valid, left, 0)
                right = jnp.where(valid, right, 1)
                q = jnp.sum(products(state).astype(jnp.float32) * jvec, axis=-1)
                log_a = base.swap_log_acceptance(beta[left], beta[right], q[left], q[right])
                accepted = (
                    (jnp.log(jax.random.uniform(exchange_key, shape=(pairs,))) < log_a)
                    & due
                    & valid
                )
                old_left, old_right = state[left], state[right]
                state = state.at[left].set(jnp.where(accepted[:, None], old_right, old_left))
                state = state.at[right].set(jnp.where(accepted[:, None], old_left, old_right))
            retained = state[-1:] if method == "tempering" else state
            sums = sums + jnp.sum(products(retained), axis=0)
            hit = snapshot_steps == t + 1
            snaps = jnp.where(hit[:, None], sums[None, :], snaps)
            return (state, key, sums, snaps), accepted

        init = (
            start,
            key,
            jnp.zeros((m,), jnp.int32),
            jnp.zeros((len(snapshot_steps), m), jnp.int32),
        )
        (_, _, _, snaps), accepted = jax.lax.scan(step, init, jnp.arange(steps))
        return snaps, accepted

    return jax.jit(jax.vmap(single))


def unit_keys(index, target, method, trials, root_seed):
    """#92 key layout: one initialization per trial shared by all arms, keys folded by method."""
    initial, keys = study.key_inputs(index, target["size"], 9, trials, root_seed)
    keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, base.METHODS.index(method))
    return initial.reshape(trials, 9, target["n"]), keys


def retained_count(method, replicas):
    return 1 if method in ("tempering", "long") else replicas


def run_unit(request, index, arm):
    """Sample one (target, arm) unit; return uint16 counts and timings."""
    target = request["targets"][index]
    method, replicas, interval = ARMS[arm]
    budgets = list(request["budgets"])
    horizon = max(budgets)
    initial, keys = unit_keys(index, target, method, request["trials"], request["root_seed"])
    plan = study.snapshot_plan(method, budgets)
    before = time.perf_counter()
    fn = compile_sampler(method, replicas, interval, target, horizon, plan)
    fn = fn.lower(keys, initial).compile()
    compile_seconds = time.perf_counter() - before
    before = time.perf_counter()
    snaps, accepted = map(np.asarray, jax.block_until_ready(fn(keys, initial)))
    run_seconds = time.perf_counter() - before
    retained = retained_count(method, replicas)
    plus, acc_prefix, acc_window = [], [], []
    for b in budgets:
        sums = study.window_sums(snaps, plan, method, b)
        length = study.window_length(method, retained, b)
        doubled = sums + length
        if np.any(doubled % 2) or np.any(doubled < 0) or np.any(doubled > 2 * length):
            raise ValueError("window sums are inconsistent with the window length")
        if length > np.iinfo(np.uint16).max:
            raise ValueError("window exceeds uint16 counts")
        plus.append((doubled // 2).astype(np.uint16))
        acc_prefix.append(accepted[:, :b, :].sum(axis=1).astype(np.uint16))
        acc_window.append(accepted[:, b // 4 : b, :].sum(axis=1).astype(np.uint16))
    arrays = {
        "plus_counts": np.stack(plus),
        "accepted_prefix": np.stack(acc_prefix),
        "accepted_window": np.stack(acc_window),
    }
    return arrays, {"compile_seconds": compile_seconds, "run_seconds": run_seconds}


# ----------------------------------------------------------------------------
# Kernel checks
# ----------------------------------------------------------------------------


def stationarity_check(rows, cols, family, graph_seed, beta=1.0):
    """Exact stationarity of one colour-A-then-colour-B sweep on an enumerable patch."""
    edges, positions = planar_subgraph(rows, cols, family, graph_seed)
    n = rows * cols
    rng = np.random.default_rng(5)
    couplings = rng.choice([-1.0, 1.0], len(edges)) * rng.integers(1, 6, len(edges)) / 5
    states = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int64)
    spins = 2.0 * states - 1
    ei = np.array([e[0] for e in edges])
    ej = np.array([e[1] for e in edges])
    q = (spins[:, ei] * spins[:, ej]) @ couplings
    probability = np.exp(beta * (q - q.max()))
    probability /= probability.sum()
    adjacency = np.zeros((n, n))
    adjacency[ei, ej] = couplings
    adjacency[ej, ei] = couplings
    field = spins @ adjacency
    up = expit(2 * beta * field)
    colour = positions.sum(axis=1) % 2
    kernel = np.eye(len(states))
    for c in (0, 1):
        sites = colour == c
        same_other = np.all(states[:, None, ~sites] == states[None, :, ~sites], axis=2)
        factors = np.where(states[None, :, sites] == 1, up[:, None, sites], 1 - up[:, None, sites])
        block = np.prod(factors, axis=2) * same_other
        kernel = kernel @ block
    residual = float(np.max(np.abs(probability @ kernel - probability)))
    row_sum_error = float(np.max(np.abs(kernel.sum(axis=1) - 1)))
    return {
        "patch": [rows, cols],
        "family": family,
        "graph_seed": graph_seed,
        "edges": len(edges),
        "non_grid_edges": int(sum(np.abs(positions[j] - positions[i]).sum() > 1 for i, j in edges)),
        "bipartite_checkerboard": bool(np.all(colour[ei] != colour[ej])),
        "beta": beta,
        "gibbs_stationarity_residual": residual,
        "kernel_row_sum_error": row_sum_error,
    }


def grid_equivalence_check(root_seed, size=8, horizon=64, trials=4):
    """Neighbour-list sampler against ``planar_ising_scaling.compile_sampler``, all six arms."""
    rng = np.random.default_rng(4500 * 1000 + size)
    edges = study.grid_edges(size)
    couplings = (-(rng.integers(1, 6, len(edges)).astype(np.float32) / 5)).tolist()
    target = {
        "id": f"L{size}-grid-check",
        "size": size,
        "n": size * size,
        "graph": "grid",
        "edges": [list(e) for e in edges],
        "couplings": couplings,
    }
    horizontal, vertical = (jnp.asarray(x) for x in study.coupling_arrays(target))
    initial, keys = study.key_inputs(0, size, 9, trials, root_seed)
    rows = {}
    for arm, (method, replicas, interval) in ARMS.items():
        k = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, base.METHODS.index(method))
        plan = study.snapshot_plan(method, [horizon // 4, horizon])
        a, fa = study.compile_sampler(method, replicas, interval, size, horizon, plan)(
            k, initial, horizontal, vertical
        )
        b, fb = compile_sampler(method, replicas, interval, target, horizon, plan)(
            k, initial.reshape(trials, 9, size * size)
        )
        rows[arm] = {
            "snapshot_sums_identical": bool(np.array_equal(np.asarray(a), np.asarray(b))),
            "exchange_flags_identical": bool(np.array_equal(np.asarray(fa), np.asarray(fb))),
        }
    return {"size": size, "horizon": horizon, "trials": trials, "arms": rows}


def empirical_check(root_seed, trials=TRIALS, horizon=65536):
    """tempering5-k1 on a 4 x 4 greedy-long patch against brute force at the cold beta."""
    edges, positions = planar_subgraph(4, 4, "greedy-long", 3)
    rng = np.random.default_rng(45)
    couplings = (-(rng.integers(1, 6, len(edges)) / 5)).astype(np.float32)
    target = {
        "id": "fixture-L4-greedy-long",
        "size": 4,
        "n": 16,
        "graph": "greedy-long",
        "edges": [list(e) for e in edges],
        "couplings": couplings.tolist(),
    }
    _, exact = brute_force(16, edges, COLD_BETA * couplings.astype(float))
    initial, keys = study.key_inputs(999, 4, 9, trials, root_seed)
    initial = initial.reshape(trials, 9, 16)
    result = {
        "evidence_class": "software_simulation",
        "patch": [4, 4],
        "edges": len(edges),
        "non_grid_edges": int(sum(np.abs(positions[j] - positions[i]).sum() > 1 for i, j in edges)),
        "horizon": horizon,
        "edge_mae": {},
    }
    for arm in ("independent", "tempering5-k1"):
        method, replicas, interval = ARMS[arm]
        k = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, base.METHODS.index(method))
        plan = study.snapshot_plan(method, [horizon])
        snaps, _ = compile_sampler(method, replicas, interval, target, horizon, plan)(k, initial)
        retained = retained_count(method, replicas)
        estimate = study.window_estimate(
            study.window_sums(np.asarray(snaps), plan, method, horizon), method, retained, horizon
        )
        result["edge_mae"][arm] = float(np.abs(estimate - exact[None, :]).mean())
    result["tolerance"] = 0.01
    return result


def kernel_checks(root_seed, empirical_horizon=65536):
    stationarity = [
        stationarity_check(2, 5, "greedy-long", 1),
        stationarity_check(3, 3, "greedy-long", 2),
    ]
    equivalence = grid_equivalence_check(root_seed)
    empirical = empirical_check(root_seed, horizon=empirical_horizon)
    passed = bool(
        all(
            s["gibbs_stationarity_residual"] <= 1e-12
            and s["non_grid_edges"] > 0
            and s["bipartite_checkerboard"]
            for s in stationarity
        )
        and all(all(row.values()) for row in equivalence["arms"].values())
        and empirical["edge_mae"]["tempering5-k1"] <= 0.01
    )
    return {
        "stationarity": stationarity,
        "grid_equivalence": equivalence,
        "empirical": empirical,
        "passed": passed,
    }


# ----------------------------------------------------------------------------
# Cells, decisions and the primary rule
# ----------------------------------------------------------------------------


def estimates_from_counts(plus_counts, method, retained, budget):
    length = study.window_length(method, retained, budget)
    sums = 2 * np.asarray(plus_counts, dtype=np.int64) - length
    return study.window_estimate(sums, method, retained, budget)


def make_cell(target, arm, estimate, exact, acc_prefix, acc_window, budget, threshold):
    method, replicas, interval = ARMS[arm]
    exact_edge = np.asarray(exact["edge"])
    couplings = np.asarray(target["couplings"], dtype=float)
    error = np.abs(estimate - exact_edge[None, :])
    edge_mae = error.mean(axis=1)
    edge_max = error.max(axis=1)
    q_error = np.abs((estimate - exact_edge[None, :]) @ couplings) / target["n"]
    trials = len(estimate)
    if method == "tempering":
        prefix = np.asarray(acc_prefix, dtype=np.int64)
        window = np.asarray(acc_window, dtype=np.int64)
        accepted_counts = prefix.sum(axis=1)
        acceptance = [
            float(prefix[:, p].sum() / (trials * budget) * interval) for p in range(prefix.shape[1])
        ]
        window_attempts = (budget - budget // 4) // interval
        acceptance_window = [
            float(window[:, p].sum() / (trials * window_attempts)) for p in range(window.shape[1])
        ]
    else:
        accepted_counts = np.zeros(trials, dtype=int)
        acceptance, acceptance_window = [], []
    return {
        "target": target["id"],
        "size": target["size"],
        "graph": target["graph"],
        "seed": target["seed"],
        "arm": arm,
        "method": method,
        "replicas": replicas,
        "exchange_interval": interval,
        "budget": budget,
        "per_trial": {
            "edge_mae": edge_mae.tolist(),
            "edge_max": edge_max.tolist(),
            "q_per_spin_error": q_error.tolist(),
        },
        "means": {
            "edge_mae": float(edge_mae.mean()),
            "edge_max": float(edge_max.mean()),
            "q_per_spin_error": float(q_error.mean()),
        },
        "individual_trial_pass_fraction": float((edge_mae <= threshold).mean()),
        "exchange_acceptance_by_pair_slot": acceptance,
        "exchange_acceptance_by_pair_slot_in_window": acceptance_window,
        "z1": study.price(target, method, replicas, interval, budget, accepted_counts),
    }


def cells_for_unit(request, target, arm, arrays, exact):
    method, replicas, _ = ARMS[arm]
    retained = retained_count(method, replicas)
    return [
        make_cell(
            target,
            arm,
            estimates_from_counts(arrays["plus_counts"][i], method, retained, budget),
            exact,
            arrays["accepted_prefix"][i],
            arrays["accepted_window"][i],
            budget,
            request["threshold"],
        )
        for i, budget in enumerate(request["budgets"])
    ]


def decide(request, cells):
    return [
        {
            "target": target["id"],
            "size": target["size"],
            "graph": target["graph"],
            "seed": target["seed"],
            "arm": arm,
            **study.qualify(
                [c for c in cells if c["target"] == target["id"] and c["arm"] == arm],
                request["threshold"],
            ),
        }
        for target in request["targets"]
        for arm in request["arms"]
    ]


def budget_step(decision, budgets):
    """Index of the qualifying budget; a censored decision is one step past the largest."""
    if decision["status"] == "reached":
        return list(budgets).index(decision["budget"])
    return len(budgets)


def classify(differences):
    """holds / slips / improves / mixed from per-seed step differences (greedy-long minus grid)."""
    if sum(d == 0 for d in differences) >= 2 and all(abs(d) <= 1 for d in differences):
        return "holds"
    if sum(d >= 1 for d in differences) >= 2:
        return "slips"
    if sum(d <= -1 for d in differences) >= 2:
        return "improves"
    return "mixed"


def comparisons(request, decisions):
    by_key = {(d["size"], d["graph"], d["seed"], d["arm"]): d for d in decisions}
    seeds = sorted({t["seed"] for t in request["targets"]})
    sizes = sorted({t["size"] for t in request["targets"]})
    rows = []
    for arm in request["arms"]:
        for size in sizes:
            per_seed = []
            for seed in seeds:
                grid = by_key.get((size, "grid", seed, arm))
                long = by_key.get((size, "greedy-long", seed, arm))
                if grid is None or long is None:
                    continue
                g = budget_step(grid, request["budgets"])
                h = budget_step(long, request["budgets"])
                per_seed.append(
                    {
                        "seed": seed,
                        "grid_budget": grid.get("budget"),
                        "greedy_long_budget": long.get("budget"),
                        "grid_step": g,
                        "greedy_long_step": h,
                        "difference_steps": h - g,
                    }
                )
            if not per_seed:
                continue
            rows.append(
                {
                    "arm": arm,
                    "size": size,
                    "gating": size in PRIMARY_SIZES,
                    "per_seed": per_seed,
                    "classification": classify([s["difference_steps"] for s in per_seed]),
                }
            )
    return rows


def row_verdict(request, decisions, rows):
    """The pre-registered row verdict; needs the full primary design to be defined."""
    sizes = {t["size"] for t in request["targets"]}
    seeds = sorted({t["seed"] for t in request["targets"]})
    complete = (
        set(PRIMARY_SIZES) <= sizes and len(seeds) == 3 and list(request["arms"]) == list(ARMS)
    )
    if not complete:
        return {"status": "not_applicable_partial_request"}
    long_targets = [
        d for d in decisions if d["graph"] == "greedy-long" and d["size"] in PRIMARY_SIZES
    ]
    ordinary_early = [
        {"target": d["target"], "arm": d["arm"], "budget": d["budget"]}
        for d in long_targets
        if d["arm"] in ORDINARY and d["status"] == "reached" and d["budget"] <= 4096
    ]
    tempering_missing = [
        {"target": d["target"], "arm": d["arm"]}
        for d in long_targets
        if d["arm"] not in ORDINARY and d["status"] != "reached"
    ]
    clause_a = not ordinary_early and not tempering_missing
    by = {(r["arm"], r["size"]): r["classification"] for r in rows}
    nine = {f"{arm}/L{size}": by[(arm, size)] for arm in NINE for size in PRIMARY_SIZES}
    clause_b = all(v == "holds" for v in nine.values())
    five = {f"{arm}/L{size}": by[(arm, size)] for arm in FIVE for size in PRIMARY_SIZES}
    slip = any(v == "slips" for v in five.values())
    if clause_a and clause_b:
        verdict = "holds_thin_ladder_slip" if slip else "holds"
    else:
        verdict = "does_not_hold"
    failing = []
    if ordinary_early:
        failing.append("a: long or independent qualifies within 4096 sweeps on greedy-long")
    if tempering_missing:
        failing.append("a: a tempering arm misses 16384 sweeps on a greedy-long target")
    if not clause_b:
        failing.append("b: a nine-replica arm is not 'holds' at L = 16 or 32")
    return {
        "status": verdict,
        "sentence": VERDICTS[verdict],
        "clause_a": clause_a,
        "clause_b": clause_b,
        "ordinary_qualifying_within_4096": ordinary_early,
        "tempering_not_reached": tempering_missing,
        "nine_replica_classifications": nine,
        "five_replica_classifications": five,
        "failing_clauses": failing,
    }


def summarize(request, decisions, cells, rows):
    counts = {}
    for d in decisions:
        key = f"L{d['size']}-{d['graph']}"
        counts.setdefault(d["arm"], {}).setdefault(key, [0, 0])
        counts[d["arm"]][key][1] += 1
        if d["status"] == "reached":
            counts[d["arm"]][key][0] += 1
    horizon = max(request["budgets"])
    acceptance = {}
    for c in cells:
        if c["budget"] == horizon and c["method"] == "tempering":
            acceptance.setdefault(f"L{c['size']}-{c['graph']}", {}).setdefault(c["arm"], []).append(
                c["exchange_acceptance_by_pair_slot"]
            )
    return {
        "qualifying_counts": counts,
        "acceptance_at_horizon_by_seed": acceptance,
        "comparisons": rows,
        "row_verdict": row_verdict(request, decisions, rows),
    }


# ----------------------------------------------------------------------------
# Request, persistence, autosave
# ----------------------------------------------------------------------------


def make_request():
    return {
        "schema": "planar_16_offset_ferro.v1",
        "status": "exploratory",
        "protocol": "docs/experiments/planar-16-offset-ferro.md",
        "proposal": "P-0001",
        "arms": list(ARMS),
        "arm_definitions": {
            name: {"method": m, "replicas": r, "exchange_interval": k}
            for name, (m, r, k) in ARMS.items()
        },
        "ladders": {str(k): list(v) for k, v in study.LADDERS.items()},
        "cold_beta": COLD_BETA,
        "offsets": [list(o) for o in OFFSETS],
        "graph_rule": (
            "greedy-long: all 16-offset candidate edges inside the L x L patch, sorted by "
            "squared length descending, ties in the order of default_rng(coupling seed)."
            "permutation over the sorted candidate list; keep an edge when its straight segment "
            "crosses no kept segment. grid: planar_ising_scaling.grid_edges order"
        ),
        "coupling_rule": "-default_rng(seed * 1000 + L).integers(1, 6, E) / 5 in edge-list order",
        "targets": study_targets(),
        "budgets": list(BUDGETS),
        "trials": TRIALS,
        "root_seed": ROOT_SEED,
        "threshold": THRESHOLD,
        "sweep": (
            "two-colour (checkerboard) block Gibbs; one complete sweep updates every site once; "
            "exchange attempted after the sweep when due"
        ),
        "sample": "edge products of the retained replicas after each complete sweep",
        "burn_in_fraction": 0.25,
        "qualification": (
            "mean trial edge-correlation MAE <= threshold, sustained at all later budgets"
        ),
        "replication_unit": "independent trial; prefixes are not extra trials",
        "persisted_counts": (
            "per (target, arm): plus_counts uint16 (budgets x trials x edges) = number of +1 "
            "edge products in the window (b/4, b] (x5 for long); accepted_prefix and "
            "accepted_window uint16 (budgets x trials x pairs) = accepted exchanges per pair "
            "slot in sweeps (0, b] (the span #92 prices) and in the window (b/4, b]"
        ),
        "profile_id": Z1HardwareProfile().profile_id,
        "profile_hash": Z1HardwareProfile().profile_hash,
        "conventions": CONVENTIONS,
        "numeric_dtype": "float32 sampler, int32 prefix sums, uint16 counts; float64 references",
        "empirical_check_horizon": 65536,
        "replay_atol": 2e-12,
        "replay_scope": (
            "replay authenticates sources, request and counts, rebuilds every graph from its "
            "seed, recomputes every reference and check, then every estimate, error, price, "
            "decision, comparison and verdict; it does not regenerate study sweeps"
        ),
        "sources": {name: base.sha(ROOT / name) for name in SOURCES},
    }


def validate_request(request):
    if request["schema"] != "planar_16_offset_ferro.v1":
        raise ValueError("unknown request schema")
    if request["arms"] != list(ARMS):
        raise ValueError("the study arms are frozen")
    if any(b <= 0 or b % 4 for b in request["budgets"]):
        raise ValueError("budgets must be positive multiples of four")
    if request["budgets"] != sorted(request["budgets"]):
        raise ValueError("budgets must be increasing")
    if request["cold_beta"] != COLD_BETA or request["burn_in_fraction"] != 0.25:
        raise ValueError("cold temperature and burn-in are frozen")
    if request["replay_atol"] != 2e-12:
        raise ValueError("replay tolerance is frozen")
    if request["profile_hash"] != Z1HardwareProfile().profile_hash:
        raise ValueError("Z1 profile changed")
    for target in request["targets"]:
        size, edges = target["size"], edge_tuples(target)
        if len(target["couplings"]) != len(edges):
            raise ValueError("coupling count does not match the edge list")
        if target["graph"] == "grid" and len(edges) != 2 * size * (size - 1):
            raise ValueError("grid edge count")
        if len(edges) > 2 * target["n"] - 4:
            raise ValueError("more edges than any planar bipartite graph has")
        stats = graph_stats(edges, target_positions(target))
        if not stats["bipartite_checkerboard"]:
            raise ValueError("graph is not checkerboard-bipartite")


def write_npz(path, arrays):
    """Deterministic npz: fixed member timestamps, sorted members, deflate."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]))
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.getvalue())
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def read_npz(path):
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def unit_name(target, arm):
    return f"{target['id']}__{arm}"


class Runner:
    """Operational state: log, status, unit index. Not part of the scientific record."""

    def __init__(self, out):
        self.out = Path(out)
        self.units = self.out / "units"

    def log(self, message):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with (self.out / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
            handle.flush()
        print(f"{stamp} {message}", flush=True)

    def status(self, **fields):
        path = self.out / "run-status.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(fields, updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        base.write(path, current)

    def index(self):
        path = self.units / "index.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_unit(self, name, arrays, timing):
        path = self.units / f"{name}.npz"
        write_npz(path, arrays)
        index = self.index()
        index[name] = {"sha256": base.sha(path), **timing}
        base.write(self.units / "index.json", index)

    def verified_units(self):
        """Units whose file matches its recorded digest; mismatches are dropped for rerun."""
        index, good = self.index(), {}
        for name, entry in index.items():
            path = self.units / f"{name}.npz"
            if path.exists() and base.sha(path) == entry["sha256"]:
                good[name] = entry
            else:
                self.log(f"unit {name} failed digest verification; it will be rerun")
        return good


def _pool_unit(request, index, arm):
    arrays, timing = run_unit(request, index, arm)
    return index, arm, arrays, timing


def _copy_sources(out, request):
    for name, digest in request["sources"].items():
        if base.sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and base.sha(destination) != digest:
            raise ValueError(f"archived source copy changed: {name}")
        shutil.copyfile(ROOT / name, destination)


def require_single_thread_blas():
    """The Kac-Ward inverse differs at 1e-5 under other BLAS thread counts; replay is 2e-12."""
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"set {name}=1 before running or replaying this study")


def run_study(out, requested=None, resume=False, workers=DEFAULT_WORKERS, stop_after=None):
    """Generate (or resume) the study, assemble results, replay, write completion last."""
    out = Path(out)
    require_single_thread_blas()
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise ValueError("use CPU with JAX_ENABLE_X64=false")
    runner = Runner(out)
    if resume:
        if not (out / "request.json").exists():
            raise FileNotFoundError("nothing to resume: request.json is missing")
        if (out / "completion.json").exists():
            raise FileExistsError("study already complete")
        request = json.loads((out / "request.json").read_text())
        guard = json.loads((out / "checkpoint-guard.json").read_text())
        if guard["request_digest"] != canonical_sha256(request):
            raise ValueError("checkpoint guard mismatch: request changed")
        if requested is None and request != make_request():
            raise ValueError("stored request differs from this runner's request")
        if requested is not None and request != requested:
            raise ValueError("stored request differs from the requested one")
        validate_request(request)
        _copy_sources(out, request)
        attempt = json.loads((out / "run-status.json").read_text()).get("attempt", 0) + 1
        runner.log(f"resume attempt {attempt}: request and sources re-authenticated")
    else:
        if out.exists():
            raise FileExistsError("use a fresh output directory, or --resume")
        request = make_request() if requested is None else requested
        validate_request(request)
        out.mkdir(parents=True)
        runner.units.mkdir()
        base.write(out / "request.json", request)
        base.write(out / "checkpoint-guard.json", {"request_digest": canonical_sha256(request)})
        _copy_sources(out, request)
        attempt = 1
        runner.log("fresh run started")
    runner.status(phase="checks", attempt=attempt, workers=workers)
    attempts_path = out / "attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempts.append(
        {
            "attempt": attempt,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "workers": workers,
            "provenance": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        }
    )
    base.write(attempts_path, attempts)
    started = time.perf_counter()
    try:
        checks_path = out / "checks.json"
        if checks_path.exists():
            checks = json.loads(checks_path.read_text())
        else:
            checks = {
                "reference": reference_checks(),
                "kernel": kernel_checks(request["root_seed"], request["empirical_check_horizon"]),
            }
            base.write(checks_path, checks)
        if not (checks["reference"]["passed"] and checks["kernel"]["passed"]):
            runner.log("STOP: an exact or kernel check failed; see checks.json")
            raise ValueError("exact or kernel check failed; stopped before sampling")
        runner.log("exact reference and kernel checks passed")
        references_dir = out / "references"
        references_dir.mkdir(exist_ok=True)
        runner.status(phase="references")
        for target in request["targets"]:
            path = references_dir / f"{target['id']}.json"
            if path.exists():
                continue
            before = time.perf_counter()
            exact = reference(target)
            base.write(path, exact)
            runner.log(
                f"reference {target['id']}: {len(target['edges'])} edges, "
                f"fd err {exact['finite_difference_max_abs_error']:.1e}, "
                f"{time.perf_counter() - before:.1f} s"
            )
        runner.status(phase="sampling")
        done = runner.verified_units()
        pending = [
            (index, arm)
            for index, target in enumerate(request["targets"])
            for arm in request["arms"]
            if unit_name(target, arm) not in done
        ]
        # Largest units first so the pool drains evenly.
        pending.sort(key=lambda u: (-request["targets"][u[0]]["n"], u[0], u[1]))
        runner.log(f"units: {len(done)} verified, {len(pending)} to run, {workers} worker(s)")
        completed = 0

        def record(index, arm, arrays, timing):
            nonlocal completed
            name = unit_name(request["targets"][index], arm)
            runner.save_unit(name, arrays, timing)
            completed += 1
            runner.status(last_unit=name, units_done=len(done) + completed)
            runner.log(
                f"unit {name}: compile {timing['compile_seconds']:.1f} s, "
                f"run {timing['run_seconds']:.1f} s ({len(done) + completed}/"
                f"{len(done) + len(pending)})"
            )
            if stop_after is not None and completed >= stop_after:
                raise KeyboardInterrupt("stop_after reached (interruption test)")

        if workers <= 1:
            for index, arm in pending:
                record(index, arm, *run_unit(request, index, arm))
        elif pending:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(workers, mp_context=context) as pool:
                futures = [pool.submit(_pool_unit, request, i, a) for i, a in pending]
                try:
                    for future in concurrent.futures.as_completed(futures):
                        record(*future.result())
                except BaseException:
                    pool.shutdown(cancel_futures=True)
                    raise
        runner.status(phase="assembling")
        assemble(out, request, runner, time.perf_counter() - started)
    except BaseException as error:
        runner.status(
            phase="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        runner.log(f"stopped: {error!r}")
        raise
    runner.status(phase="replaying")
    completion = replay(out)
    runner.status(phase="complete", error=None, traceback=None)
    runner.log(f"completion written after replay ({completion['replay_seconds']:.1f} s replay)")
    return completion


def assemble(out, request, runner, seconds):
    index = runner.verified_units()
    store, timings, cells, references = {}, {}, [], {}
    for target in request["targets"]:
        exact = json.loads((out / "references" / f"{target['id']}.json").read_text())
        references[target["id"]] = exact
        for arm in request["arms"]:
            name = unit_name(target, arm)
            if name not in index:
                raise ValueError(f"unit missing at assembly: {name}")
            arrays = read_npz(runner.units / f"{name}.npz")
            for key, value in arrays.items():
                store[f"{name}__{key}"] = value
            timings[name] = {k: v for k, v in index[name].items() if k != "sha256"}
            cells.extend(cells_for_unit(request, target, arm, arrays, exact))
    write_npz(out / "counts.npz", store)
    decisions = decide(request, cells)
    rows = comparisons(request, decisions)
    checks = json.loads((out / "checks.json").read_text())
    result = {
        "request_digest": canonical_sha256(request),
        "evidence_class": {
            "sweeps": "software_simulation",
            "references": "exact_reference",
            "z1_pricing": "calibrated_projection",
        },
        "hardware_claim": False,
        "exact_physical_graph_available": False,
        "counts_sha256": base.sha(out / "counts.npz"),
        "checks": checks,
        "graphs": {
            t["id"]: graph_stats(edge_tuples(t), target_positions(t)) for t in request["targets"]
        },
        "exact_references": references,
        "cells": cells,
        "decisions": decisions,
        "summary": summarize(request, decisions, cells, rows),
        "unit_timings_seconds": timings,
        "generation_seconds_this_attempt": seconds,
    }
    base.write(out / "results.json", result)
    attempts = json.loads((out / "attempts.json").read_text())
    base.write(
        out / "provenance.json",
        {
            "schema": "planar_16_offset_ferro.provenance.v1",
            "request_digest": result["request_digest"],
            "attempts": attempts,
            "runtime": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        },
    )
    runner.log("results assembled; counts.npz and results.json written")


# ----------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------


def replay(out):
    out = Path(out)
    require_single_thread_blas()
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    validate_request(request)
    if result["request_digest"] != canonical_sha256(request):
        raise ValueError("request digest mismatch")
    for name, digest in request["sources"].items():
        for path in (ROOT / name, out / "source" / name):
            if base.sha(path) != digest:
                raise ValueError(f"source changed: {name}")
    if base.sha(out / "counts.npz") != result["counts_sha256"]:
        raise ValueError("counts digest mismatch")
    for target in request["targets"]:
        edges, _ = planar_subgraph(target["size"], target["size"], target["graph"], target["seed"])
        if [list(e) for e in edges] != target["edges"]:
            raise ValueError(f"graph does not rebuild from its seed: {target['id']}")
        rebuilt = build_target(target["size"], target["graph"], target["seed"])
        if rebuilt != target:
            raise ValueError(f"target does not rebuild from its seed: {target['id']}")
    checks = {
        "reference": reference_checks(),
        "kernel": kernel_checks(request["root_seed"], request["empirical_check_horizon"]),
    }
    study.check_equal(checks, result["checks"])
    if not (checks["reference"]["passed"] and checks["kernel"]["passed"]):
        raise ValueError("exact or kernel check failed on replay")
    store = read_npz(out / "counts.npz")
    cells, references = [], 0
    for target in request["targets"]:
        exact = reference(target)
        study.check_equal(exact, result["exact_references"][target["id"]])
        references += 1
        for arm in request["arms"]:
            method, replicas, _ = ARMS[arm]
            name = unit_name(target, arm)
            keys = ("plus_counts", "accepted_prefix", "accepted_window")
            arrays = {k: store[f"{name}__{k}"] for k in keys}
            if any(a.dtype != np.uint16 for a in arrays.values()):
                raise ValueError(f"counts must be uint16: {name}")
            shape = (len(request["budgets"]), request["trials"])
            if arrays["plus_counts"].shape != (*shape, len(target["edges"])):
                raise ValueError(f"unexpected count shape: {name}")
            pairs = max(1, (replicas - 1) // 2)
            for key in keys[1:]:
                if arrays[key].shape != (*shape, pairs):
                    raise ValueError(f"unexpected exchange count shape: {name}")
            cells.extend(cells_for_unit(request, target, arm, arrays, exact))
    study.check_equal(cells, result["cells"])
    decisions = decide(request, cells)
    study.check_equal(decisions, result["decisions"])
    rows = comparisons(request, decisions)
    summary = summarize(request, decisions, cells, rows)
    study.check_equal(summary, result["summary"])
    completion = {
        "status": "planar_16_offset_ferro_complete",
        "request_digest": result["request_digest"],
        "counts_sha256": result["counts_sha256"],
        "targets": len(request["targets"]),
        "cells_replayed": len(cells),
        "decisions_replayed": len(decisions),
        "comparisons_replayed": len(rows),
        "references_recomputed": references,
        "reference_checks_passed": True,
        "kernel_checks_passed": True,
        "graph_rebuild_passed": True,
        "row_verdict": summary["row_verdict"].get("status"),
        "replay_seconds": time.perf_counter() - started,
        "evidence_class": result["evidence_class"],
        "hardware_claim": False,
    }
    base.write(out / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    parser.add_argument("--replay", action="store_true", help="replay a finished run only")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--request", help="request JSON (tests only; production uses the default)")
    parser.add_argument("--stop-after", type=int, help="interrupt after N new units (tests only)")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.output_dir), indent=2))
        return
    requested = json.loads(Path(args.request).read_text()) if args.request else None
    try:
        completion = run_study(
            args.output_dir,
            requested,
            resume=args.resume,
            workers=args.workers,
            stop_after=args.stop_after,
        )
    except KeyboardInterrupt as error:
        print(f"interrupted: {error}", flush=True)
        raise SystemExit(130) from None
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
