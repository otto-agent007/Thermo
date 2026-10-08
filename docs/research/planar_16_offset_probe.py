"""Exploration: planar Ising scaling allocations on planar subgraphs of the 16-offset lattice.

Probe for loop proposal P-0001 (queue row ``planar-16-offset-ferro``). The
recorded planar Ising scaling study (PR #92) ran on open square grids. The Z1
interior connection rule links a site to 16 offsets, (1,0), (2,1), (2,3), (4,1)
and their rotations. That graph is not planar, so "the planar subgraph" has to
be chosen. All 16 offsets have odd parity, so every subgraph is bipartite under
the (x + y) checkerboard and the two-colour sweep stays exact; a planar
bipartite graph has at most 2N - 4 edges, so any planar subgraph has mean
degree below 4, which the nearest-neighbour grid already almost reaches.

Graph families (all on an L x L patch of the lattice, straight-line embedding):

- ``grid``: the (1,0) offsets only, the PR #92 graph, run through this
  script's general code as a control.
- ``greedy-random``: all 16-offset edges inside the patch in a seeded random
  order, each kept if its segment crosses no kept segment. A maximal planar
  straight-line subgraph that uses every offset class.
- ``greedy-long``: the same, longest offsets first, then a seeded order within
  each length.

Exact reference: the Kac-Ward determinant as in ``planar_ising_scaling``,
generalised from grid positions to an arbitrary straight-line planar embedding
(study-local code here, checked equal to the imported function on grids and
against brute force on small patches of each family). Sampler: the study's
two-colour block Gibbs and replica exchange, rewritten over a padded
neighbour list.

Exploration only: no archive, no replay, no gate; one coupling seed per
family and size. Output: a JSON file next to this script.

Run:  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=false uv run python docs/research/planar_16_offset_probe.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from thermo_lab import planar_ising_scaling as study
from thermo_lab.hardware.z1 import Z1HardwareProfile

HERE = Path(__file__).resolve().parent
OUT = HERE / "2026-10-08-planar-16-offset-ferro-probe.json"
TRIALS = 16
ROOT_SEED = 20261012  # fresh: not used by any archived study or earlier probe
COUPLING_SEED = 4500  # fresh: archived planar targets use 400-402
GRAPH_SEED = 7
OFFSETS = tuple(sorted(Z1HardwareProfile().interior_offsets()))
FAMILIES = ("grid", "greedy-random", "greedy-long")
THRESHOLD = 0.05


# ----------------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------------


def _orient(p, q, r):
    return np.sign((q[0] - p[0]) * (r[..., 1] - p[1]) - (q[1] - p[1]) * (r[..., 0] - p[0]))


def planar_subgraph(rows, cols, family, seed=GRAPH_SEED):
    """Edges (i, j), i < j, and site positions (x, y) = (col, row), index row * cols + col."""
    positions = np.array([(c, r) for r in range(rows) for c in range(cols)], dtype=np.int64)
    if family == "grid":
        edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
        edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
        return edges, positions
    candidates = set()
    for r in range(rows):
        for c in range(cols):
            for dx, dy in OFFSETS:
                rr, cc = r + dy, c + dx
                if 0 <= rr < rows and 0 <= cc < cols:
                    a, b = r * cols + c, rr * cols + cc
                    candidates.add((min(a, b), max(a, b)))
    candidates = sorted(candidates)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in order]
    if family == "greedy-long":
        length = {e: int(np.sum((positions[e[0]] - positions[e[1]]) ** 2)) for e in candidates}
        candidates.sort(key=lambda e: -length[e])  # stable: seeded order within a length
    elif family != "greedy-random":
        raise ValueError(family)
    kept, a_pts, b_pts = [], np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64)
    for e in candidates:
        p, q = positions[e[0]], positions[e[1]]
        if len(kept):
            # Offsets are primitive, so no lattice point lies inside a segment and
            # collinear overlap is impossible: a strict sign test finds every crossing.
            d1, d2 = _orient(p, q, a_pts), _orient(p, q, b_pts)
            d3 = np.sign(
                (b_pts[:, 0] - a_pts[:, 0]) * (p[1] - a_pts[:, 1])
                - (b_pts[:, 1] - a_pts[:, 1]) * (p[0] - a_pts[:, 0])
            )
            d4 = np.sign(
                (b_pts[:, 0] - a_pts[:, 0]) * (q[1] - a_pts[:, 1])
                - (b_pts[:, 1] - a_pts[:, 1]) * (q[0] - a_pts[:, 0])
            )
            if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
                continue
        kept.append(e)
        a_pts = np.vstack([a_pts, p[None]])
        b_pts = np.vstack([b_pts, q[None]])
    kept.sort()
    return kept, positions


def graph_stats(edges, positions):
    n = len(positions)
    degree = np.bincount(np.array(edges).ravel(), minlength=n)
    lengths = {}
    for i, j in edges:
        d = tuple(sorted(np.abs(positions[j] - positions[i]).tolist()))
        lengths[str(d)] = lengths.get(str(d), 0) + 1
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    components = len({find(i) for i in range(n) if degree[i] > 0})
    return {
        "sites": n,
        "edges": len(edges),
        "planar_bipartite_bound_2n_minus_4": 2 * n - 4,
        "mean_degree": float(degree.mean()),
        "degree_histogram": np.bincount(degree).tolist(),
        "isolated_sites": int((degree == 0).sum()),
        "components_with_edges": components,
        "edges_by_offset_class": lengths,
        "bipartite_checkerboard": bool(
            all((positions[i].sum() + positions[j].sum()) % 2 == 1 for i, j in edges)
        ),
    }


def make_target(size, family, seed=COUPLING_SEED):
    edges, positions = planar_subgraph(size, size, family)
    rng = np.random.default_rng(seed * 1000 + size)
    weights = rng.integers(1, 6, len(edges)).astype(np.float32) / 5
    return {
        "id": f"L{size}-{family}-s{seed}",
        "size": size,
        "n": size * size,
        "family": family,
        "edges": edges,
        "positions": positions,
        "couplings": (
            -weights
        ).tolist(),  # archived sign convention; ferro by gauge on a bipartite graph
    }


# ----------------------------------------------------------------------------
# Exact reference: Kac-Ward on a straight-line planar embedding
# ----------------------------------------------------------------------------


def kac_ward_matrix(positions, edges, interactions):
    """Same construction as ``planar_ising_scaling.kac_ward_matrix`` with explicit positions."""
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
    return np.eye(2 * m) - lam, x


def kac_ward_log_z(positions, edges, interactions):
    matrix, _ = kac_ward_matrix(positions, edges, interactions)
    sign, logdet = np.linalg.slogdet(matrix)
    if abs(sign - 1.0) > 1e-6:
        raise ValueError("Kac-Ward determinant is not positive real")
    n = len(positions)
    return float(n * np.log(2) + np.sum(np.log(np.cosh(interactions))) + 0.5 * logdet.real)


def kac_ward(positions, edges, interactions):
    matrix, x = kac_ward_matrix(positions, edges, interactions)
    log_z = kac_ward_log_z(positions, edges, interactions)
    m = len(x)
    diagonal = np.diag(np.linalg.inv(matrix)).real - 1.0
    correlation = x - 0.5 * ((1 - x**2) / x) * (diagonal[:m] + diagonal[m:])
    if np.any(np.abs(correlation) > 1 + 1e-6):
        raise ValueError("edge correlation outside [-1, 1]")
    return log_z, np.clip(correlation, -1, 1)


def brute_force(n, edges, interactions):
    if n > 16:
        raise ValueError("brute force is bounded to 16 spins")
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    products = states[:, [e[0] for e in edges]] * states[:, [e[1] for e in edges]]
    energy = products @ np.asarray(interactions, dtype=float)
    w = np.exp(energy - energy.max())
    return float(np.log(w.sum()) + energy.max()), (w @ products) / w.sum()


def reference(target):
    interactions = study.COLD_BETA * np.asarray(target["couplings"], dtype=float)
    log_z, corr = kac_ward(target["positions"], target["edges"], interactions)
    checked = np.argsort(np.abs(corr))[:2]
    worst = 0.0
    for e in checked:
        plus, minus = interactions.copy(), interactions.copy()
        plus[e] += 1e-4
        minus[e] -= 1e-4
        diff = kac_ward_log_z(target["positions"], target["edges"], plus) - kac_ward_log_z(
            target["positions"], target["edges"], minus
        )
        worst = max(worst, abs(diff / 2e-4 - corr[e]))
    return {
        "log_z": log_z,
        "edge": corr,
        "q_per_spin": float(np.asarray(target["couplings"]) @ corr / target["n"]),
        "min_abs_edge_correlation": float(np.abs(corr).min()),
        "finite_difference_max_abs_error": worst,
    }


def reference_checks():
    """General Kac-Ward against the imported grid version and against brute force."""
    out = {"grid_vs_imported": [], "brute_force": []}
    rng = np.random.default_rng(2027)
    for size in (3, 8):
        _, positions = planar_subgraph(size, size, "grid")
        # The study orders grid edges horizontal-then-vertical; use its order here.
        edges = study.grid_edges(size)
        k = study.COLD_BETA * -(rng.integers(1, 6, len(edges)) / 5)
        mine = kac_ward(positions, edges, k)
        theirs = study.kac_ward(size, k)
        out["grid_vs_imported"].append(
            {
                "size": size,
                "log_z_abs_diff": abs(mine[0] - theirs[0]),
                "edge_max_abs_diff": float(np.max(np.abs(mine[1] - theirs[1]))),
            }
        )
    for rows, cols in ((4, 4), (3, 5), (2, 8)):
        for family in ("greedy-random", "greedy-long"):
            for seed in (1, 2):
                edges, positions = planar_subgraph(rows, cols, family, seed)
                for beta, sign in ((study.COLD_BETA, "ferro"), (1.0, "mixed")):
                    w = rng.integers(1, 6, len(edges)) / 5
                    s = (
                        -np.ones(len(edges))
                        if sign == "ferro"
                        else rng.choice([-1.0, 1.0], len(edges))
                    )
                    k = beta * s * w
                    log_z, corr = kac_ward(positions, edges, k)
                    exact_log_z, exact_corr = brute_force(rows * cols, edges, k)
                    out["brute_force"].append(
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
                            "log_z_abs_err": abs(log_z - exact_log_z),
                            "edge_max_abs_err": float(np.max(np.abs(corr - exact_corr))),
                        }
                    )
    out["max_brute_force_log_z_err"] = max(r["log_z_abs_err"] for r in out["brute_force"])
    out["max_brute_force_edge_err"] = max(r["edge_max_abs_err"] for r in out["brute_force"])
    return out


# ----------------------------------------------------------------------------
# Sampler: two-colour block Gibbs over a padded neighbour list
# ----------------------------------------------------------------------------


def neighbour_arrays(n, edges, couplings):
    adj = [[] for _ in range(n)]
    for (i, j), c in zip(edges, couplings, strict=True):
        adj[i].append((j, c))
        adj[j].append((i, c))
    width = max(1, max(len(a) for a in adj))
    nbr = np.zeros((n, width), np.int32)
    wts = np.zeros((n, width), np.float32)
    for i, a in enumerate(adj):
        for k, (j, c) in enumerate(a):
            nbr[i, k], wts[i, k] = j, c
    return jnp.asarray(nbr), jnp.asarray(wts)


def compile_sampler(method, replicas, interval, target, horizon, snapshot_steps):
    """Mirror of ``planar_ising_scaling.compile_sampler`` on an arbitrary bipartite graph."""
    n = target["n"]
    edges = target["edges"]
    couplings = np.asarray(target["couplings"], np.float32)
    nbr, wts = neighbour_arrays(n, edges, couplings)
    ei = jnp.asarray([e[0] for e in edges], jnp.int32)
    ej = jnp.asarray([e[1] for e in edges], jnp.int32)
    jvec = jnp.asarray(couplings)
    steps = 5 * horizon if method == "long" else horizon
    ladder = study.LADDERS[replicas] if method == "tempering" else (study.COLD_BETA,) * replicas
    beta = jnp.asarray(ladder, jnp.float32)
    pairs = max(1, (replicas - 1) // 2)
    pos = np.asarray(target["positions"])
    checker = jnp.asarray((pos.sum(axis=1) % 2) == 0)
    snapshot_steps = jnp.asarray(snapshot_steps, jnp.int32)
    m = len(edges)

    def products(state):
        spins = 2 * state.astype(jnp.int32) - 1
        return spins[..., ei] * spins[..., ej]

    def single(key, initial):
        start = initial[:replicas]

        def half_sweep(state, key, mask):
            spins = 2 * state.astype(jnp.float32) - 1
            field = jnp.sum(wts * spins[:, nbr], axis=-1)
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
                log_a = study.base.swap_log_acceptance(beta[left], beta[right], q[left], q[right])
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


def run_target(index, target, exact, budgets):
    horizon = max(budgets)
    initial, keys = study.key_inputs(index, target["size"], 9, TRIALS, ROOT_SEED)
    initial = initial.reshape(TRIALS, 9, target["n"])
    rows = {}
    for arm, (method, replicas, interval) in study.ARMS.items():
        k = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, study.base.METHODS.index(method))
        plan = study.snapshot_plan(method, budgets)
        before = time.perf_counter()
        fn = compile_sampler(method, replicas, interval, target, horizon, plan)
        snaps, accepted = map(np.asarray, jax.block_until_ready(fn(k, initial)))
        seconds = time.perf_counter() - before
        retained = 1 if method in ("tempering", "long") else replicas
        per_budget = {}
        for b in budgets:
            est = study.window_estimate(
                study.window_sums(snaps, plan, method, b), method, retained, b
            )
            err = np.abs(est - exact["edge"][None, :])
            per_budget[str(b)] = {
                "edge_mae": float(err.mean()),
                "edge_mae_trial_sd": float(err.mean(axis=1).std()),
                "edge_max": float(err.max(axis=1).mean()),
            }
        maes = [per_budget[str(b)]["edge_mae"] for b in budgets]
        qual = next(
            (b for i, b in enumerate(budgets) if all(x <= THRESHOLD for x in maes[i:])), None
        )
        acc = accepted.mean(axis=(0, 1)) * (interval or 1) if method == "tempering" else []
        rows[arm] = {
            "per_budget": per_budget,
            "qualifying_budget": qual,
            "acceptance_by_slot_at_horizon": [float(a) for a in acc],
            "seconds_incl_compile": seconds,
        }
        print(f"  {arm:14s} mae@{horizon}={maes[-1]:.4f} qual={qual} ({seconds:.0f}s)", flush=True)
    return rows


def grid_equivalence(size=8, horizon=256):
    """Do the general sampler and the study sampler give the same draws on a grid?"""
    target = make_target(size, "grid")
    target["edges"] = study.grid_edges(size)  # study order
    stargets = {"size": size, "couplings": target["couplings"]}
    initial, keys = study.key_inputs(0, size, 9, 4, ROOT_SEED)
    plan = study.snapshot_plan("tempering", [horizon])
    a, fa = study.compile_sampler("tempering", 5, 1, size, horizon, plan)(
        keys, initial, *(jnp.asarray(x) for x in study.coupling_arrays(stargets))
    )
    b, fb = compile_sampler("tempering", 5, 1, target, horizon, plan)(
        keys, initial.reshape(4, 9, size * size)
    )
    return {
        "size": size,
        "horizon": horizon,
        "snapshot_sums_identical": bool(np.array_equal(np.asarray(a), np.asarray(b))),
        "exchange_flags_identical": bool(np.array_equal(np.asarray(fa), np.asarray(fb))),
    }


def empirical_check(horizon=65536):
    """Tempering5-k1 on a 4 x 4 greedy-random patch against brute force at beta 4."""
    edges, positions = planar_subgraph(4, 4, "greedy-random", 3)
    rng = np.random.default_rng(45)
    couplings = (-(rng.integers(1, 6, len(edges)) / 5)).astype(np.float32)
    target = {
        "n": 16,
        "size": 4,
        "edges": edges,
        "positions": positions,
        "couplings": couplings.tolist(),
    }
    _, exact = brute_force(16, edges, study.COLD_BETA * couplings.astype(float))
    initial, keys = study.key_inputs(999, 4, 9, TRIALS, ROOT_SEED)
    initial = initial.reshape(TRIALS, 9, 16)
    out = {"edges": len(edges)}
    for arm in ("independent", "tempering5-k1"):
        method, replicas, interval = study.ARMS[arm]
        plan = study.snapshot_plan(method, [horizon])
        snaps, _ = compile_sampler(method, replicas, interval, target, horizon, plan)(keys, initial)
        retained = 1 if method == "tempering" else replicas
        est = study.window_estimate(
            study.window_sums(np.asarray(snaps), plan, method, horizon), method, retained, horizon
        )
        out[arm] = float(np.abs(est - exact[None, :]).mean())
    return out


def main(sizes):
    started = time.perf_counter()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results.update(
        {
            "label": "exploration, not evidence",
            "root_seed": ROOT_SEED,
            "coupling_seed": COUPLING_SEED,
            "graph_seed": GRAPH_SEED,
            "trials": TRIALS,
            "threshold": THRESHOLD,
            "cold_beta": study.COLD_BETA,
            "budgets": list(study.BUDGETS),
            "offsets": [list(o) for o in OFFSETS],
        }
    )
    if "checks" not in results:
        results["checks"] = {
            "reference": reference_checks(),
            "grid_equivalence": grid_equivalence(),
            "empirical": empirical_check(),
        }
        print(
            "checks",
            json.dumps(results["checks"]["grid_equivalence"]),
            results["checks"]["empirical"],
        )
        OUT.write_text(json.dumps(results, indent=2))
    results.setdefault("graphs", {})
    results.setdefault("targets", {})
    for size in sizes:
        for family in FAMILIES:
            target = make_target(size, family)
            results["graphs"][target["id"]] = graph_stats(target["edges"], target["positions"])
            if target["id"] in results["targets"]:
                continue
            before = time.perf_counter()
            exact = reference(target)
            ref_seconds = time.perf_counter() - before
            fd_error = exact["finite_difference_max_abs_error"]
            print(
                f"{target['id']}: {len(target['edges'])} edges, reference {ref_seconds:.0f}s, "
                f"q/spin {exact['q_per_spin']:.4f}, fd err {fd_error:.1e}",
                flush=True,
            )
            index = 100 * size + FAMILIES.index(family)
            rows = run_target(index, target, exact, list(study.BUDGETS))
            results["targets"][target["id"]] = {
                "reference": {k: v for k, v in exact.items() if k != "edge"},
                "reference_seconds": ref_seconds,
                "arms": rows,
            }
            OUT.write_text(json.dumps(results, indent=2))
    results["wall_seconds_last_invocation"] = time.perf_counter() - started
    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT, f"{time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [8, 16, 32])
