"""Sampling allocations on planar zero-field Ising grids with an exact reference.

Every recorded sampling study so far fits inside exact enumeration (at most 16
free spins). This study leaves that regime: open L x L grids at L = 8, 16 and
32 (64 to 1024 spins) with an exact Kac-Ward reference for ln Z and every edge
correlation, which is valid for any planar graph and is checked here against
brute force (L <= 4) and an independent transfer matrix (L <= 16).

Arms run at equal elapsed sweeps (the parallel-hardware comparison): five
independent cold chains, the archived five-replica tempering ladder, a denser
nine-replica ladder, each with exchanges every sweep or every four sweeps, and
a single long chain. The sweep is the two-colour block-Gibbs schedule the Z1
Appendix-B model prices, which the bipartite grid makes exact. Every cell is
also priced in that model under the two exchange conventions of the exchange
cost projection study.

Traces are software_simulation; references are exact_reference; energies and
sweep times are calibrated_projection. Nothing here ran on hardware.
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
from scipy.special import expit

from thermo_lab import fixed_budget_sampling as base
from thermo_lab.exchange_cost_projection import CONVENTIONS
from thermo_lab.hardware.z1 import Z1HardwareProfile, Z1OperationCounts, project_z1_operations
from thermo_lab.hashing import canonical_sha256
from thermo_lab.provenance import collect_runtime_provenance

ROOT = base.ROOT
SOURCES = (
    "src/thermo_lab/planar_ising_scaling.py",
    "src/thermo_lab/exchange_cost_projection.py",
    "src/thermo_lab/hardware/z1.py",
    "src/thermo_lab/hashing.py",
    "src/thermo_lab/persistence.py",
    "docs/experiments/planar-ising-scaling.md",
    "uv.lock",
)
SIZES = (8, 16, 32)
VARIANTS = ("ferro", "mixed")
SEEDS = (400, 401, 402)
COLD_BETA = 4.0
LADDERS = {
    5: tuple(base.LADDER),
    9: tuple(float(COLD_BETA * 2.0 ** (-(8 - i) / 2)) for i in range(9)),
}
ARMS = {
    "long": ("long", 1, 0),
    "independent": ("independent", 5, 0),
    "tempering5-k1": ("tempering", 5, 1),
    "tempering5-k4": ("tempering", 5, 4),
    "tempering9-k1": ("tempering", 9, 1),
    "tempering9-k4": ("tempering", 9, 4),
}
BUDGETS = (16, 64, 256, 1024, 4096)


# ----------------------------------------------------------------------------
# Targets and exact references
# ----------------------------------------------------------------------------


def grid_edges(size):
    """Edges (i, j) with i < j: all horizontal rows first, then all vertical."""

    def idx(r, c):
        return r * size + c

    edges = [(idx(r, c), idx(r, c + 1)) for r in range(size) for c in range(size - 1)]
    edges += [(idx(r, c), idx(r + 1, c)) for r in range(size - 1) for c in range(size)]
    return edges


def grid_targets():
    result = []
    for size in SIZES:
        for variant in VARIANTS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 1000 + size)
                edges = grid_edges(size)
                weights = rng.integers(1, 6, len(edges)).astype(np.float32) / 5
                signs = rng.choice([-1.0, 1.0], len(edges)).astype(np.float32)
                couplings = -weights if variant == "ferro" else signs * weights
                result.append(
                    {
                        "id": f"L{size}-{variant}-s{seed}",
                        "size": size,
                        "n": size * size,
                        "variant": variant,
                        "seed": seed,
                        "couplings": couplings.tolist(),
                    }
                )
    return result


def coupling_arrays(target):
    size = target["size"]
    couplings = np.asarray(target["couplings"], dtype=np.float32)
    horizontal = couplings[: size * (size - 1)].reshape(size, size - 1)
    vertical = couplings[size * (size - 1) :].reshape(size - 1, size)
    return horizontal, vertical


def kac_ward_matrix(size, interactions):
    """Kac-Ward matrix I - Lambda over directed edges, and tanh of the interactions."""
    edges = grid_edges(size)
    count, m = size * size, len(edges)
    position = np.array([(i % size, i // size) for i in range(count)], dtype=float)
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


def kac_ward_log_z(size, interactions):
    """ln Z alone; one determinant, no inverse."""
    matrix, _, _ = kac_ward_matrix(size, interactions)
    sign, logdet = np.linalg.slogdet(matrix)
    # The determinant is positive real for a planar graph; the LU phase carries
    # rounding of order 1e-9 at 4k x 4k, so the check is loose and the real part is used.
    if abs(sign - 1.0) > 1e-6:
        raise ValueError("Kac-Ward determinant is not positive real")
    return float(
        size * size * np.log(2) + np.sum(np.log(np.cosh(interactions))) + 0.5 * logdet.real
    )


def kac_ward(size, interactions):
    """ln Z and <s_i s_j> per edge for the zero-field Ising model on an open grid.

    ``interactions[e] = beta * J_e`` in ``grid_edges`` order. Kac-Ward for a
    planar graph: Z = 2^N prod_e cosh(K_e) sqrt(det(I - Lambda)) with Lambda
    indexed by directed edges, Lambda[d, d'] = exp(i*turn/2) tanh(K_{e(d')})
    when head(d) = tail(d') and d' is not the reverse of d. Edge correlations
    are d ln Z / dK_e, which scales the two columns of Lambda belonging to e,
    so they need the inverse's diagonal: tr(M^-1 dLambda) = ((M^-1)_cc - 1)
    summed over the two columns. The inverse is ill-conditioned at low
    temperature on frustrated graphs; ``reference`` measures its error against
    finite differences of ln Z, which needs no inverse.
    """
    matrix, _, x = kac_ward_matrix(size, interactions)
    log_z = kac_ward_log_z(size, interactions)
    m = len(x)
    diagonal = np.diag(np.linalg.inv(matrix)).real - 1.0
    factor = (1 - x**2) / x
    correlation = x - 0.5 * factor * (diagonal[:m] + diagonal[m:])
    if np.any(np.abs(correlation) > 1 + 1e-6):
        raise ValueError("edge correlation outside [-1, 1]")
    return log_z, np.clip(correlation, -1, 1)


def brute_force(size, interactions):
    if size * size > 16:
        raise ValueError("brute force is bounded to 16 spins")
    edges = grid_edges(size)
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=size * size)))
    products = states[:, [e[0] for e in edges]] * states[:, [e[1] for e in edges]]
    energy = products @ np.asarray(interactions, dtype=float)
    weights = np.exp(energy - energy.max())
    return float(np.log(weights.sum()) + energy.max()), (weights @ products) / weights.sum()


def transfer_matrix_log_z(size, interactions):
    """Independent ln Z by adding spins in raster order (bounded to L <= 16)."""
    if size > 16:
        raise ValueError("transfer matrix is bounded to L <= 16")
    edges = grid_edges(size)
    interaction = {edge: k for edge, k in zip(edges, interactions, strict=True)}
    states = np.arange(1 << size)
    oldest = 2 * (states & 1) - 1
    newest = 2 * ((states >> (size - 1)) & 1) - 1
    log_v = np.zeros(1 << size)
    for i in range(size * size):
        r, c = divmod(i, size)
        shift = log_v.max()
        values = np.zeros_like(log_v)
        for s_new in (0, 1):
            s = 2 * s_new - 1
            w = np.zeros_like(log_v)
            if r > 0:
                w = w + interaction[i - size, i] * oldest * s
            if c > 0:
                w = w + interaction[i - 1, i] * newest * s
            target = (states >> 1) | (s_new << (size - 1))
            np.add.at(values, target, np.exp(log_v + w - shift))
        log_v = np.log(values) + shift
    return float(np.logaddexp.reduce(log_v) - size * np.log(2))


def reference(target):
    interactions = COLD_BETA * np.asarray(target["couplings"], dtype=float)
    log_z, correlation = kac_ward(target["size"], interactions)
    # Precision of the inverse-based correlations, measured on the two edges with the
    # weakest correlation (the hardest) by central differences of ln Z, h = 1e-4:
    # truncation about 1e-8, roundoff about 1e-9, no inverse involved.
    checked = np.argsort(np.abs(correlation))[:2]
    worst = 0.0
    for e in checked:
        plus, minus = interactions.copy(), interactions.copy()
        plus[e] += 1e-4
        minus[e] -= 1e-4
        difference = kac_ward_log_z(target["size"], plus) - kac_ward_log_z(target["size"], minus)
        worst = max(worst, abs(difference / 2e-4 - correlation[e]))
    if worst > 1e-4:
        raise ValueError("Kac-Ward edge correlations lost precision")
    result = {
        "evidence_class": "exact_reference",
        "method": "kac_ward_determinant",
        "log_z": log_z,
        "edge": correlation.tolist(),
        "q_per_spin": float(np.asarray(target["couplings"]) @ correlation / target["n"]),
        "finite_difference_checked_edges": [int(e) for e in checked],
        "finite_difference_max_abs_error": worst,
    }
    if target["size"] <= 16:
        result["transfer_matrix_log_z"] = transfer_matrix_log_z(target["size"], interactions)
        if abs(result["transfer_matrix_log_z"] - log_z) > 1e-8 * max(1.0, abs(log_z)):
            raise ValueError("Kac-Ward and transfer-matrix ln Z disagree")
    return result


def reference_checks():
    """Exact cross-checks of the Kac-Ward reference against brute force."""
    rng = np.random.default_rng(2026)
    worst_log_z, worst_edge = 0.0, 0.0
    for size in (2, 3, 4):
        for variant in VARIANTS:
            m = len(grid_edges(size))
            weights = rng.integers(1, 6, m) / 5
            signs = rng.choice([-1.0, 1.0], m)
            interactions = COLD_BETA * (-weights if variant == "ferro" else signs * weights)
            log_z, edge = kac_ward(size, interactions)
            exact_log_z, exact_edge = brute_force(size, interactions)
            worst_log_z = max(worst_log_z, abs(log_z - exact_log_z))
            worst_edge = max(worst_edge, float(np.max(np.abs(edge - exact_edge))))
    if worst_log_z > 1e-9 or worst_edge > 1e-9:
        raise ValueError("Kac-Ward reference failed its brute-force check")
    return {
        "evidence_class": "exact_reference",
        "sizes": [2, 3, 4],
        "max_log_z_error": worst_log_z,
        "max_edge_error": worst_edge,
    }


# ----------------------------------------------------------------------------
# Sampler: two-colour block Gibbs on the grid, replica exchange on the host
# ----------------------------------------------------------------------------


def local_fields(state, horizontal, vertical):
    """Sum of J * neighbour spin at every site; ``state`` is (..., L, L) spins."""
    spins = 2 * state.astype(jnp.float32) - 1
    field = jnp.zeros_like(spins)
    field = field.at[..., :, 1:].add(horizontal * spins[..., :, :-1])
    field = field.at[..., :, :-1].add(horizontal * spins[..., :, 1:])
    field = field.at[..., 1:, :].add(vertical * spins[..., :-1, :])
    field = field.at[..., :-1, :].add(vertical * spins[..., 1:, :])
    return field


def edge_products(state, size):
    spins = 2 * state.astype(jnp.int32) - 1
    horizontal = (spins[..., :, :-1] * spins[..., :, 1:]).reshape(*state.shape[:-2], -1)
    vertical = (spins[..., :-1, :] * spins[..., 1:, :]).reshape(*state.shape[:-2], -1)
    return jnp.concatenate([horizontal, vertical], axis=-1)


def key_inputs(index, size, replicas, trials, root_seed):
    root = jax.random.fold_in(jax.random.key(root_seed), index)
    pairs = jax.vmap(lambda i: jax.random.split(jax.random.fold_in(root, i)))(jnp.arange(trials))
    initial = jax.vmap(lambda k: jax.random.bernoulli(k, shape=(replicas, size, size)))(pairs[:, 0])
    return initial, pairs[:, 1]


def compile_sampler(method, replicas, interval, size, horizon, snapshot_steps):
    """Return a jitted sampler over trials.

    Per trial it returns prefix sums of the retained replicas' edge products at
    ``snapshot_steps`` (shape (snapshots, edges), summed over retained replicas),
    and the exchange flags per sweep (shape (steps, pairs)). Retained replicas
    are the cold one for tempering, all five for independent, the one for long.
    """
    if method not in base.METHODS:
        raise ValueError("unknown sampling method")
    steps = 5 * horizon if method == "long" else horizon
    ladder = LADDERS[replicas] if method == "tempering" else (COLD_BETA,) * replicas
    beta = jnp.asarray(ladder, jnp.float32)
    pairs = max(1, (replicas - 1) // 2)
    checker = (jnp.arange(size)[:, None] + jnp.arange(size)[None, :]) % 2 == 0
    snapshot_steps = jnp.asarray(snapshot_steps, jnp.int32)
    m = 2 * size * (size - 1)

    def single(key, initial, horizontal, vertical):
        start = initial[:replicas]

        def half_sweep(state, key, color_mask):
            field = local_fields(state, horizontal, vertical)
            p = jax.nn.sigmoid(2 * beta[:, None, None] * field)
            draw = jax.random.bernoulli(key, p)
            return jnp.where(color_mask, draw, state)

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
                q = jnp.sum(
                    edge_products(state, size).astype(jnp.float32)
                    * jnp.concatenate([horizontal.ravel(), vertical.ravel()]),
                    axis=-1,
                )
                log_a = base.swap_log_acceptance(beta[left], beta[right], q[left], q[right])
                accepted = (
                    (jnp.log(jax.random.uniform(exchange_key, shape=(pairs,))) < log_a)
                    & due
                    & valid
                )
                old_left, old_right = state[left], state[right]
                state = state.at[left].set(jnp.where(accepted[:, None, None], old_right, old_left))
                state = state.at[right].set(jnp.where(accepted[:, None, None], old_left, old_right))
            retained = state[-1:] if method == "tempering" else state
            sums = sums + jnp.sum(edge_products(retained, size), axis=0)
            hit = snapshot_steps == t + 1
            snaps = jnp.where(hit[:, None], sums[None, :], snaps)
            return (state, key, sums, snaps), accepted

        sums = jnp.zeros((m,), jnp.int32)
        snaps = jnp.zeros((len(snapshot_steps), m), jnp.int32)
        (_, _, _, snaps), accepted = jax.lax.scan(
            step, (start, key, sums, snaps), jnp.arange(steps)
        )
        return snaps, accepted

    return jax.jit(jax.vmap(single, in_axes=(0, 0, None, None)))


def snapshot_plan(method, budgets):
    """Steps at which prefix sums are needed: window (b/4, b] per budget (x5 for long)."""
    scale = 5 if method == "long" else 1
    steps = sorted({scale * b // 4 for b in budgets} | {scale * b for b in budgets})
    return steps


def window_length(method, retained, budget):
    scale = 5 if method == "long" else 1
    return (scale * budget - scale * budget // 4) * retained


def window_sums(snaps, snapshot_steps, method, budget):
    """Exact integer sum of retained edge products over the budget's window."""
    scale = 5 if method == "long" else 1
    start, end = scale * budget // 4, scale * budget
    total = snaps[..., snapshot_steps.index(end), :].astype(np.int64)
    if start > 0:
        total = total - snaps[..., snapshot_steps.index(start), :]
    return total


def window_estimate(sums, method, retained, budget):
    """Mean edge product over the window, from its exact integer sums."""
    return np.asarray(sums, dtype=np.int64).astype(np.float64) / window_length(
        method, retained, budget
    )


# ----------------------------------------------------------------------------
# Exact fixture: the two-colour kernel preserves the Boltzmann distribution
# ----------------------------------------------------------------------------


def fixture_checks(beta=1.0):
    """Exact stationarity of one colour-A-then-colour-B sweep on a 2 x 3 open grid."""
    rows, cols = 2, 3
    count = rows * cols
    rng = np.random.default_rng(5)
    horizontal = (
        rng.choice([-1.0, 1.0], (rows, cols - 1)) * rng.integers(1, 6, (rows, cols - 1)) / 5
    )
    vertical = rng.choice([-1.0, 1.0], (rows - 1, cols)) * rng.integers(1, 6, (rows - 1, cols)) / 5
    states = ((np.arange(1 << count)[:, None] >> np.arange(count)) & 1).astype(bool)
    spins = 2 * states.astype(float) - 1
    grids = spins.reshape(-1, rows, cols)
    q = np.sum(horizontal * grids[:, :, :-1] * grids[:, :, 1:], axis=(1, 2))
    q += np.sum(vertical * grids[:, :-1, :] * grids[:, 1:, :], axis=(1, 2))
    probability = np.exp(beta * (q - q.max()))
    probability /= probability.sum()
    kernel = np.eye(len(states))
    for color in (0, 1):
        block = np.zeros_like(kernel)
        sites = [(r, c) for r in range(rows) for c in range(cols) if (r + c) % 2 == color]
        for row_index, grid in enumerate(grids):
            field = np.zeros((rows, cols))
            field[:, 1:] += horizontal * grid[:, :-1]
            field[:, :-1] += horizontal * grid[:, 1:]
            field[1:, :] += vertical * grid[:-1, :]
            field[:-1, :] += vertical * grid[1:, :]
            up = {site: expit(2 * beta * field[site]) for site in sites}
            for assignment in itertools.product([0, 1], repeat=len(sites)):
                new = grid.copy()
                weight = 1.0
                for site, value in zip(sites, assignment, strict=True):
                    new[site] = 2 * value - 1
                    weight *= up[site] if value else 1 - up[site]
                code = int(np.sum(((new.ravel() > 0).astype(int)) << np.arange(count)))
                block[row_index, code] += weight
        kernel = kernel @ block
    residual = float(np.max(np.abs(probability @ kernel - probability)))
    if residual > 1e-12:
        raise ValueError("two-colour Gibbs sweep is not stationary")
    return {
        "evidence_class": "exact_reference",
        "grid": [rows, cols],
        "beta": beta,
        "gibbs_stationarity_residual": residual,
    }


def empirical_checks(trials=16, horizon=65536):
    """Sampler against brute force on a 4 x 4 mixed-sign grid at the cold beta.

    Software-simulation evidence that the two-colour kernel and exchange step as
    compiled reach the exact edge correlations: the five-replica tempering arm
    must come within 0.01 mean absolute edge error at the horizon (it reaches
    about 0.005; the error falls roughly as the inverse square root of sweeps).
    """
    size = 4
    rng = np.random.default_rng(44)
    m = len(grid_edges(size))
    couplings = (rng.choice([-1.0, 1.0], m) * rng.integers(1, 6, m) / 5).astype(np.float32)
    target = {"id": "fixture-L4-mixed", "size": size, "n": 16, "couplings": couplings.tolist()}
    _, exact_edge = brute_force(size, COLD_BETA * couplings.astype(float))
    horizontal, vertical = (jnp.asarray(x) for x in coupling_arrays(target))
    initial, keys = key_inputs(999, size, 9, trials, 20261009)
    result = {"evidence_class": "software_simulation", "grid": [size, size], "edge_mae": {}}
    for arm in ("independent", "tempering5-k1"):
        method, replicas, interval = ARMS[arm]
        plan = snapshot_plan(method, [horizon])
        fn = compile_sampler(method, replicas, interval, size, horizon, plan)
        snaps, _ = map(np.asarray, jax.block_until_ready(fn(keys, initial, horizontal, vertical)))
        retained = 1 if method == "tempering" else replicas
        estimate = window_estimate(
            window_sums(snaps, plan, method, horizon), method, retained, horizon
        )
        result["edge_mae"][arm] = float(np.abs(estimate - exact_edge[None, :]).mean())
    if result["edge_mae"]["tempering5-k1"] > 0.01:
        raise ValueError("empirical fixture failed: tempering does not reach the exact reference")
    return result


# ----------------------------------------------------------------------------
# Z1 pricing and decisions
# ----------------------------------------------------------------------------


def price(target, method, replicas, interval, budget, accepted_counts):
    """Per-trial Z1 projections for one cell; ``accepted_counts`` is per trial."""
    n = target["n"]
    sweeps = 5 * budget if method == "long" else budget
    pairs_per_attempt = max(1, (replicas - 1) // 2)
    # Each exchange step tries ``pairs_per_attempt`` pairs (two for five replicas, as
    # in the exchange cost projection study; four for nine).
    pair_tries = (budget // interval) * pairs_per_attempt if method == "tempering" else 0
    energies = {name: [] for name in CONVENTIONS}
    for accepts in accepted_counts:
        for name, rule in CONVENTIONS.items():
            counts = Z1OperationCounts.constant_participation(
                logical_pbits=n * replicas,
                physical_pbits_used=n * replicas,
                participating_free_pbits=n * replicas,
                elapsed_complete_sweeps=sweeps,
                node_reads=int(pair_tries * rule["reads_per_attempt_per_pair"] * n),
                node_full_sram_writes=int(accepts * rule["writes_per_accept_per_pair"] * n),
                host_round_trips=budget // interval if method == "tempering" else 0,
            )
            energies[name].append(project_z1_operations(counts).modeled_total_energy_j)
    sweep_time = sweeps / Z1HardwareProfile().cost_model_max_complete_sweep_rate.value_hz
    return {
        "elapsed_complete_sweeps": sweeps,
        "sweep_time_at_assumed_max_clock_s": sweep_time,
        "physical_pbits_used": n * replicas,
        "pair_tries_per_trial": int(pair_tries),
        "host_round_trips_per_trial": budget // interval if method == "tempering" else 0,
        "mean_accepted_exchanges_per_trial": float(np.mean(accepted_counts)),
        "energy": {
            name: {
                "mean_total_energy_j": float(np.mean(values)),
                "min_total_energy_j": float(np.min(values)),
                "max_total_energy_j": float(np.max(values)),
            }
            for name, values in energies.items()
        },
        "evidence_class": "calibrated_projection",
    }


def make_cell(target, arm, estimate, exact, accepted, budget, threshold):
    method, replicas, interval = ARMS[arm]
    exact_edge = np.asarray(exact["edge"])
    couplings = np.asarray(target["couplings"], dtype=float)
    error = np.abs(estimate - exact_edge[None, :])
    edge_mae = error.mean(axis=1)
    edge_max = error.max(axis=1)
    q_error = np.abs((estimate - exact_edge[None, :]) @ couplings) / target["n"]
    if method == "tempering":
        flags = np.asarray(accepted[:, :budget, :], dtype=bool)
        accepted_counts = flags.sum(axis=(1, 2)).astype(int)
        acceptance = [float(flags[:, :, pair].mean() * interval) for pair in range(flags.shape[2])]
    else:
        accepted_counts = np.zeros(len(estimate), dtype=int)
        acceptance = []
    return {
        "target": target["id"],
        "size": target["size"],
        "variant": target["variant"],
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
        "z1": price(target, method, replicas, interval, budget, accepted_counts),
    }


def qualify(cells, threshold):
    cells = sorted(cells, key=lambda row: row["budget"])
    passes = [row["means"]["edge_mae"] <= threshold for row in cells]
    for i, row in enumerate(cells):
        if all(passes[i:]):
            return {
                "status": "reached",
                "budget": row["budget"],
                "sweep_time_s": row["z1"]["sweep_time_at_assumed_max_clock_s"],
                "energy_j": {
                    name: row["z1"]["energy"][name]["mean_total_energy_j"] for name in CONVENTIONS
                },
            }
    return {"status": "not_reached", "largest_tested_budget": cells[-1]["budget"]}


def decide(request, cells):
    return [
        {
            "target": target["id"],
            "size": target["size"],
            "variant": target["variant"],
            "arm": arm,
            **qualify(
                [c for c in cells if c["target"] == target["id"] and c["arm"] == arm],
                request["threshold"],
            ),
        }
        for target in request["targets"]
        for arm in request["arms"]
    ]


def summarize(decisions):
    """Qualification counts per arm, size and variant, plus best arm per target."""
    counts = {}
    for d in decisions:
        key = f"L{d['size']}-{d['variant']}"
        counts.setdefault(d["arm"], {}).setdefault(key, [0, 0])
        counts[d["arm"]][key][1] += 1
        if d["status"] == "reached":
            counts[d["arm"]][key][0] += 1
    best = []
    for target in sorted({d["target"] for d in decisions}):
        reached = [d for d in decisions if d["target"] == target and d["status"] == "reached"]
        if not reached:
            best.append({"target": target, "fastest": None, "cheapest_published": None})
            continue
        fastest = min(reached, key=lambda d: (d["sweep_time_s"], d["arm"]))
        cheapest = min(reached, key=lambda d: (d["energy_j"]["published"], d["arm"]))
        best.append(
            {"target": target, "fastest": fastest["arm"], "cheapest_published": cheapest["arm"]}
        )
    return {"qualifying_counts": counts, "best_per_target": best}


# ----------------------------------------------------------------------------
# Study
# ----------------------------------------------------------------------------


def make_request():
    return {
        "schema": "planar_ising_scaling.v1",
        "status": "exploratory",
        "arms": list(ARMS),
        "arm_definitions": {
            name: {"method": m, "replicas": r, "exchange_interval": k}
            for name, (m, r, k) in ARMS.items()
        },
        "ladders": {str(k): list(v) for k, v in LADDERS.items()},
        "cold_beta": COLD_BETA,
        "targets": grid_targets(),
        "budgets": list(BUDGETS),
        "trials": 16,
        "root_seed": 20261009,
        "threshold": 0.05,
        "sweep": (
            "two-colour (checkerboard) block Gibbs; one complete sweep updates every site once; "
            "exchange attempted after the sweep when due"
        ),
        "sample": "edge products of the retained replicas after each complete sweep",
        "burn_in_fraction": 0.25,
        "qualification": (
            "mean trial edge-correlation MAE <= threshold, sustained at all later budgets"
        ),
        "metrics_note": (
            "zero-field edge correlations are invariant under the global flip, so the analytic "
            "symmetry estimator of the archived studies is the identity here and is not an arm"
        ),
        "replication_unit": "independent trial; prefixes are not extra trials",
        "profile_id": Z1HardwareProfile().profile_id,
        "profile_hash": Z1HardwareProfile().profile_hash,
        "conventions": CONVENTIONS,
        "numeric_dtype": "float32 sampler, int32 edge-product sums; float64 exact references",
        "replay_atol": 2e-12,
        "replay_scope": (
            "replay authenticates sources, exact integer window sums and exchange flags, then "
            "recomputes references, kernel checks, estimates, errors, pricing, decisions and "
            "summaries; it does not regenerate sweeps"
        ),
        "sources": {name: base.sha(ROOT / name) for name in SOURCES},
    }


def _validate_request(request):
    if request["arms"] != list(ARMS):
        raise ValueError("the study arms are frozen")
    if any(b <= 0 or b % 4 for b in request["budgets"]):
        raise ValueError("budgets must be positive multiples of four")
    if request["cold_beta"] != COLD_BETA or request["burn_in_fraction"] != 0.25:
        raise ValueError("cold temperature and burn-in are frozen")
    if request["replay_atol"] != 2e-12:
        raise ValueError("replay tolerance is frozen")
    if request["profile_hash"] != Z1HardwareProfile().profile_hash:
        raise ValueError("Z1 profile changed")
    for target in request["targets"]:
        if len(target["couplings"]) != 2 * target["size"] * (target["size"] - 1):
            raise ValueError("coupling count does not match the grid")


def check_equal(actual, expected, path="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"keys differ: {path}")
        for name in expected:
            check_equal(actual[name], expected[name], f"{path}/{name}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"length differs: {path}")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            check_equal(a, b, f"{path}/{i}")
    elif isinstance(expected, float):
        if (
            not isinstance(actual, (float, int))
            or not np.isfinite(actual)
            or abs(actual - expected) > 2e-12 * max(1.0, abs(expected))
        ):
            raise ValueError(f"numeric value differs: {path}")
    elif actual != expected:
        raise ValueError(f"value differs: {path}")


def _cells_for(request, target, arm, estimates, accepted, exact):
    return [
        make_cell(
            target, arm, estimates[str(budget)], exact, accepted, budget, request["threshold"]
        )
        for budget in request["budgets"]
    ]


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
    checks = {
        "reference": reference_checks(),
        "fixture": fixture_checks(),
        "empirical": empirical_checks(),
    }
    print("Exact reference, kernel and empirical checks passed", flush=True)
    references, cells, compile_times, estimates_store, flags = {}, [], {}, {}, {}
    horizon = max(request["budgets"])
    for index, target in enumerate(request["targets"]):
        before = time.perf_counter()
        exact = reference(target)
        references[target["id"]] = exact
        reference_seconds = time.perf_counter() - before
        horizontal, vertical = (jnp.asarray(x) for x in coupling_arrays(target))
        for arm in request["arms"]:
            method, replicas, interval = ARMS[arm]
            initial, keys = key_inputs(
                index,
                target["size"],
                max(r for _, r, _ in ARMS.values()),
                request["trials"],
                request["root_seed"],
            )
            keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, base.METHODS.index(method))
            plan = snapshot_plan(method, request["budgets"])
            before = time.perf_counter()
            fn = compile_sampler(method, replicas, interval, target["size"], horizon, plan)
            fn = fn.lower(keys, initial, horizontal, vertical).compile()
            compile_times[f"{target['id']}-{arm}"] = time.perf_counter() - before
            snaps, accepted = map(
                np.asarray, jax.block_until_ready(fn(keys, initial, horizontal, vertical))
            )
            retained = 1 if method in ("tempering", "long") else replicas
            sums = np.stack([window_sums(snaps, plan, method, b) for b in request["budgets"]])
            if np.any(np.abs(sums) > np.iinfo(np.int16).max):
                raise ValueError("window sums exceed int16")
            sums = sums.astype(np.int16)
            per_budget = {
                str(budget): window_estimate(sums[i], method, retained, budget)
                for i, budget in enumerate(request["budgets"])
            }
            estimates_store[f"{target['id']}__{arm}"] = sums
            flags[f"{target['id']}__{arm}__accepted"] = np.packbits(accepted, axis=1)
            cells.extend(_cells_for(request, target, arm, per_budget, accepted, exact))
        print(
            f"Sampled {index + 1}/{len(request['targets'])}: {target['id']} "
            f"(reference {reference_seconds:.1f} s)",
            flush=True,
        )
    np.savez_compressed(out / "window-sums.npz", **estimates_store, **flags)
    decisions = decide(request, cells)
    result = {
        "request_digest": canonical_sha256(request),
        "evidence_class": {
            "sweeps": "software_simulation",
            "references": "exact_reference",
            "z1_pricing": "calibrated_projection",
        },
        "window_sums_sha256": base.sha(out / "window-sums.npz"),
        "checks": checks,
        "exact_references": references,
        "cells": cells,
        "decisions": decisions,
        "summary": summarize(decisions),
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
    if base.sha(out / "window-sums.npz") != result["window_sums_sha256"]:
        raise ValueError("window sums digest mismatch")
    check_equal(
        {
            "reference": reference_checks(),
            "fixture": fixture_checks(),
            "empirical": empirical_checks(),
        },
        result["checks"],
    )
    store = dict(np.load(out / "window-sums.npz"))
    cells = []
    for target in request["targets"]:
        exact = reference(target)
        check_equal(exact, result["exact_references"][target["id"]])
        for arm in request["arms"]:
            method, replicas, _ = ARMS[arm]
            pairs = max(1, (replicas - 1) // 2)
            retained = 1 if method in ("tempering", "long") else replicas
            sums = store[f"{target['id']}__{arm}"]
            per_budget = {
                str(b): window_estimate(sums[i], method, retained, b)
                for i, b in enumerate(request["budgets"])
            }
            packed = store[f"{target['id']}__{arm}__accepted"]
            steps = 5 * max(request["budgets"]) if arm == "long" else max(request["budgets"])
            accepted = np.unpackbits(packed, axis=1, count=steps).astype(bool)
            if accepted.shape[2] != pairs:
                raise ValueError("unexpected exchange pair count")
            cells.extend(_cells_for(request, target, arm, per_budget, accepted, exact))
    check_equal(cells, result["cells"])
    decisions = decide(request, cells)
    check_equal(decisions, result["decisions"])
    check_equal(summarize(decisions), result["summary"])
    completion = {
        "status": "planar_ising_scaling_complete",
        "request_digest": result["request_digest"],
        "targets": len(request["targets"]),
        "cells_replayed": len(cells),
        "decisions_replayed": len(decisions),
        "references_recomputed": len(request["targets"]),
        "reference_checks_passed": True,
        "fixture_stationarity_passed": True,
        "empirical_check_passed": True,
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
