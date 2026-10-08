"""Exploration: scaled tempering ladders and mixed-grid reachability at 1024 spins.

Two questions the planar Ising scaling study left open, asked on one ferro and
one mixed 32 x 32 target with the study's exact Kac-Ward reference and its
two-colour sampler, unchanged:

A. How many replicas does a geometric ladder from 0.25 to 4 need before the
   cold-pair exchange acceptance recovers, and does the cold-replica edge error
   follow? Ladders are monkeypatched into the study module's table; the
   sampler, keys and estimator are the recorded ones.

B. Does anything reach the 0.05 edge-MAE threshold on the mixed grid: the
   nine-replica ladder or a single chain at 100k sweeps, five cold chains at
   100k sweeps, or an annealed single chain that cools from beta 0.5 to 4 and
   then measures?

Exploration only: no archive, no replay, no gate. Output is a JSON file next
to this script for the dated note to quote.

Run:  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=false uv run python docs/research/ladder_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from thermo_lab import planar_ising_scaling as study

HERE = Path(__file__).resolve().parent
OUT = HERE / "2026-10-08-ladder-probe.json"
TRIALS = 16
ROOT_SEED = 20261010  # fresh; not the study's seed


def target(name):
    return next(t for t in study.grid_targets() if t["id"] == name)


def geometric_ladder(replicas, hot=0.25, cold=study.COLD_BETA):
    return tuple(float(x) for x in np.geomspace(hot, cold, replicas))


def run_arm(tgt, method, replicas, interval, horizon, ladder=None):
    """Return cold edge MAE at the final window and per-slot acceptance."""
    if ladder is not None:
        study.LADDERS[replicas] = ladder  # exploration-only monkeypatch
    horizontal, vertical = (jnp.asarray(x) for x in study.coupling_arrays(tgt))
    initial, keys = study.key_inputs(0, tgt["size"], max(replicas, 9), TRIALS, ROOT_SEED)
    keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, study.base.METHODS.index(method))
    plan = study.snapshot_plan(method, [horizon])
    fn = study.compile_sampler(method, replicas, interval, tgt["size"], horizon, plan)
    started = time.perf_counter()
    snaps, accepted = map(
        np.asarray, jax.block_until_ready(fn(keys, initial, horizontal, vertical))
    )
    seconds = time.perf_counter() - started
    retained = 1 if method in ("tempering", "long") else replicas
    estimate = study.window_estimate(
        study.window_sums(snaps, plan, method, horizon), method, retained, horizon
    )
    exact = np.asarray(REFERENCES[tgt["id"]]["edge"])
    error = np.abs(estimate - exact[None, :])
    acceptance = accepted.mean(axis=(0, 1)) * (interval or 1)
    couplings = np.asarray(tgt["couplings"])
    return {
        "q_per_spin_estimate": float((estimate @ couplings).mean() / tgt["n"]),
        "q_per_spin_exact": REFERENCES[tgt["id"]]["q_per_spin"],
        "edge_mae": float(error.mean()),
        "edge_mae_per_trial_sd": float(error.mean(axis=1).std()),
        "edge_max": float(error.max(axis=1).mean()),
        "acceptance_by_slot": acceptance.tolist(),
        "min_pair_acceptance": float(acceptance.min()) if len(acceptance) else None,
        "median_pair_acceptance": float(np.median(acceptance)) if len(acceptance) else None,
        "seconds": seconds,
    }


def annealed_sampler(size, horizon, beta_hot=0.5, beta_cold=study.COLD_BETA, anneal_fraction=0.75):
    """Single chain; beta rises geometrically over the first fraction, then holds.

    Measures edge products over the final quarter only (the held part), so the
    estimate is 'simulated annealing then equilibrate' rather than equilibrium
    sampling. Exploration-only; mirrors the study kernel otherwise.
    """
    steps = horizon
    ramp = int(anneal_fraction * steps)
    schedule = np.concatenate(
        [np.geomspace(beta_hot, beta_cold, ramp), np.full(steps - ramp, beta_cold)]
    ).astype(np.float32)
    schedule = jnp.asarray(schedule)
    checker = (jnp.arange(size)[:, None] + jnp.arange(size)[None, :]) % 2 == 0
    start_measure = steps - steps // 4

    def single(key, initial, horizontal, vertical):
        state = initial[:1]

        def half_sweep(state, key, color_mask, beta):
            field = study.local_fields(state, horizontal, vertical)
            draw = jax.random.bernoulli(key, jax.nn.sigmoid(2 * beta * field))
            return jnp.where(color_mask, draw, state)

        def step(carry, t):
            state, key, sums = carry
            key, key_a, key_b = jax.random.split(key, 3)
            beta = schedule[t]
            state = half_sweep(state, key_a, checker, beta)
            state = half_sweep(state, key_b, ~checker, beta)
            products = jnp.sum(study.edge_products(state, size), axis=0)
            sums = sums + jnp.where(t >= start_measure, products, 0)
            return (state, key, sums), None

        m = 2 * size * (size - 1)
        (_, _, sums), _ = jax.lax.scan(
            step, (state, key, jnp.zeros((m,), jnp.int32)), jnp.arange(steps)
        )
        return sums

    return jax.jit(jax.vmap(single, in_axes=(0, 0, None, None))), steps - start_measure


def run_annealed(tgt, horizon):
    horizontal, vertical = (jnp.asarray(x) for x in study.coupling_arrays(tgt))
    initial, keys = study.key_inputs(0, tgt["size"], 9, TRIALS, ROOT_SEED)
    fn, window = annealed_sampler(tgt["size"], horizon)
    started = time.perf_counter()
    sums = np.asarray(jax.block_until_ready(fn(keys, initial, horizontal, vertical)))
    seconds = time.perf_counter() - started
    estimate = sums.astype(np.float64) / window
    exact = np.asarray(REFERENCES[tgt["id"]]["edge"])
    error = np.abs(estimate - exact[None, :])
    couplings = np.asarray(tgt["couplings"])
    return {
        "edge_mae": float(error.mean()),
        "edge_mae_per_trial_sd": float(error.mean(axis=1).std()),
        "q_per_spin_estimate": float((estimate @ couplings).mean() / tgt["n"]),
        "q_per_spin_exact": REFERENCES[tgt["id"]]["q_per_spin"],
        "seconds": seconds,
    }


def addendum():
    """Rerun only probe B's two sampled arms to record their energy per spin."""
    global REFERENCES
    tgt = target("L32-mixed-s400")
    REFERENCES = {tgt["id"]: study.reference(tgt)}
    results = json.loads(OUT.read_text())
    for label, kwargs in (
        ("tempering9-k1-100k", dict(method="tempering", replicas=9, interval=1, horizon=100_000)),
        ("independent-100k", dict(method="independent", replicas=5, interval=0, horizon=100_000)),
    ):
        results["probe_b"][label] = run_arm(tgt, **kwargs)
        print(label, {k: results["probe_b"][label][k] for k in ("edge_mae", "q_per_spin_estimate")})
        OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__" and __import__("sys").argv[1:] == ["--addendum"]:
    addendum()
    raise SystemExit

if __name__ == "__main__":
    targets = {name: target(name) for name in ("L32-ferro-s400", "L32-mixed-s400")}
    REFERENCES = {}
    for name, tgt in targets.items():
        started = time.perf_counter()
        REFERENCES[name] = study.reference(tgt)
        print(f"reference {name} {time.perf_counter() - started:.0f}s", flush=True)

    results = {"root_seed": ROOT_SEED, "trials": TRIALS, "probe_a": {}, "probe_b": {}}

    # Probe A: geometric ladders of increasing density, exchange every sweep, T=4096.
    for name, tgt in targets.items():
        results["probe_a"][name] = {}
        for replicas in (5, 9, 13, 17, 25, 33):
            ladder = geometric_ladder(replicas)
            row = run_arm(tgt, "tempering", replicas, 1, 4096, ladder)
            row["ladder_ratio"] = float(ladder[1] / ladder[0])
            row["physical_pbits"] = replicas * tgt["n"]
            results["probe_a"][name][str(replicas)] = row
            print(
                f"A {name} R={replicas:2d} ratio={row['ladder_ratio']:.3f} "
                f"min/median acc={row['min_pair_acceptance']:.3f}"
                f"/{row['median_pair_acceptance']:.3f} "
                f"edge_mae={row['edge_mae']:.4f} ({row['seconds']:.0f}s)",
                flush=True,
            )
            OUT.write_text(json.dumps(results, indent=2))

    # Probe B: reachability on the mixed grid at 100k sweeps.
    tgt = targets["L32-mixed-s400"]
    results["probe_b"]["threshold"] = 0.05
    for label, kwargs in (
        ("tempering9-k1-100k", dict(method="tempering", replicas=9, interval=1, horizon=100_000)),
        ("long-100k", dict(method="long", replicas=1, interval=0, horizon=20_000)),
        ("independent-100k", dict(method="independent", replicas=5, interval=0, horizon=100_000)),
    ):
        row = run_arm(tgt, **kwargs)
        results["probe_b"][label] = row
        print(f"B {label}: edge_mae={row['edge_mae']:.4f} ({row['seconds']:.0f}s)", flush=True)
        OUT.write_text(json.dumps(results, indent=2))
    row = run_annealed(tgt, 100_000)
    results["probe_b"]["annealed-100k"] = row
    print(
        f"B annealed-100k: edge_mae={row['edge_mae']:.4f} q/spin {row['q_per_spin_estimate']:.4f} "
        f"vs exact {row['q_per_spin_exact']:.4f} ({row['seconds']:.0f}s)",
        flush=True,
    )
    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT)
