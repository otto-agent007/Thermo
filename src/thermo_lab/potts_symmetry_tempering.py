"""Potts stage B: label symmetry and tempering on fresh antiferromagnetic Potts targets.

Study-local code. Nothing here is a hash-bound evaluator and nothing here edits
a committed archive. Sampling is THRML 0.1.4 categorical block Gibbs on CPU in
float32 (``software_simulation``); references are float64 enumeration of all
3^12 states (``exact_reference``). No result in this module is hardware
evidence, and label redraws are algorithmic counts, not device operations.

Question (frozen in ``docs/experiments/potts-symmetry-tempering.md``). On
zero-field antiferromagnetic three-state Potts targets, does tempering combined
with the analytic label-permutation estimator reach the October accuracy rule
at a smaller matched budget than the best symmetry-aware ordinary Gibbs
baseline, as it did for Ising? Six fresh n = 12 weighted cubic graphs at cold
beta 8 and 16; samplers ``long`` (one chain, 5T sweeps), ``independent`` (five
cold chains, T sweeps) and ``tempering`` (five replicas on a ladder, T sweeps,
two exchange attempts per sweep, cold replica retained). Tempering is one
THRML program over five disjoint replica copies of the graph.

One retained sample is the full label vector of one chain after one complete
block sweep (after the exchange step for tempering), outside the first quarter
of the run. Independent trials are the replication unit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block
from thrml.block_sampling import BlockGibbsSpec, sample_blocks
from thrml.factor import FactorSamplingProgram
from thrml.models import CategoricalGibbsConditional
from thrml.models.discrete_ebm import SquareCategoricalEBMFactor
from thrml.pgm import CategoricalNode

from thermo_lab import thrml_potts_contract as stage_a
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import collect_runtime_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_SEED = 20261006
Q = 3
N_SITES = 12
GRAPH_SEEDS = (400, 401, 402, 403, 404, 405)
BETAS = (8.0, 16.0)
LADDER = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
BUDGETS = (64, 256, 1024, 4096, 16384)
TRIALS = 16
SAMPLERS = ("long", "independent", "tempering")
ESTIMATORS = ("plain", "sym")
THRESHOLD = 0.05
JOINT_SITES = 4
TIMING_REPEATS = 3
FLOOR_DRAWS = 400
PREFLIGHT_CHAINS = 100_000
PERMUTATIONS = tuple(itertools.permutations(range(Q)))


# --- targets -----------------------------------------------------------------


def cubic_graph(n: int, seed: int) -> tuple[list[tuple[int, int]], list[int]]:
    """Ring plus a disjoint perfect matching and integer weights 1..5 (October generator)."""
    rng = np.random.default_rng(seed)
    ring = {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
    for _ in range(10_000):
        order = rng.permutation(n)
        matching = {tuple(sorted(map(int, order[i : i + 2]))) for i in range(0, n, 2)}
        if not ring & matching:
            break
    else:
        raise RuntimeError("matching generation failed")
    edges = sorted(ring | matching)
    weights = rng.integers(1, 6, len(edges))
    return [tuple(map(int, e)) for e in edges], [int(w) for w in weights]


def three_colouring(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Deterministic backtracking; colours tried in order 0, 1, 2; empty blocks dropped."""
    adjacent = {i: set() for i in range(n)}
    for a, b in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    colour: dict[int, int] = {}

    def assign(i: int) -> bool:
        if i == n:
            return True
        for c in range(Q):
            if all(colour.get(j) != c for j in adjacent[i]):
                colour[i] = c
                if assign(i + 1):
                    return True
                del colour[i]
        return False

    if not assign(0):
        raise ValueError("graph is not 3-colourable")
    blocks = [[i for i in range(n) if colour[i] == c] for c in range(Q)]
    return [b for b in blocks if b]


def study_request() -> dict:
    targets = []
    for seed in GRAPH_SEEDS:
        edges, weights = cubic_graph(N_SITES, seed)
        for beta in BETAS:
            targets.append(
                {
                    "id": f"n{N_SITES}-g{seed}-b{beta:g}",
                    "seed": seed,
                    "beta": beta,
                    "edges": [list(e) for e in edges],
                    "weights": weights,
                    "couplings": [-w / 5 for w in weights],
                    "blocks": three_colouring(N_SITES, edges),
                }
            )
    return {
        "schema": "potts_symmetry_tempering.request.v1",
        "protocol": "docs/experiments/potts-symmetry-tempering.md",
        "q": Q,
        "n": N_SITES,
        "targets": targets,
        "model": (
            "score(c) = sum_e K_e delta(c_a, c_b), K_e = -w_e / 5, zero field, "
            "P(c) proportional to exp(beta * score)"
        ),
        "numeric_dtype": "float32 THRML weights beta_r * K_e * I_q; float64 exact reference",
        "samplers": list(SAMPLERS),
        "estimators": list(ESTIMATORS),
        "ladder": list(LADDER),
        "budgets": list(BUDGETS),
        "trials": TRIALS,
        "burn_in": "first floor(L/4) of L recorded sweeps discarded",
        "sweeps": {"long": "5T, one chain", "independent": "T, five cold chains", "tempering": "T"},
        "exchange": (
            "after each sweep: pairs (0,1),(2,3) on even sweeps, (1,2),(3,4) on odd; "
            "accept with min(1, exp((beta_i - beta_j) * (score_j - score_i)))"
        ),
        "joint_sites": JOINT_SITES,
        "metrics": "per-trial TV of the sites 0-3 joint (81 bins); MAE of P(c_a = c_b) over edges",
        "qualification": (
            f"mean trial joint TV <= {THRESHOLD} and mean trial edge MAE <= {THRESHOLD} at the "
            "selected budget and every larger tested budget"
        ),
        "threshold": THRESHOLD,
        "root_seed": ROOT_SEED,
        "keys": (
            "fold_in(fold_in(key(root), target), trial) split into init and sampling keys; "
            "sampling key folded with the sampler index"
        ),
        "timing": {
            "repeats": TIMING_REPEATS,
            "scope": "warm CPU: sweeps, exchanges, transfer and histogram estimation",
        },
        "noise_floor": {"draws": FLOOR_DRAWS, "numpy_seed": ROOT_SEED},
        "preflight": {
            "chains": PREFLIGHT_CHAINS,
            "graph": "stage A six-site patch with delta couplings",
            "betas": [16.0 * r for r in LADDER],
        },
        "sample_definition": (
            "full label vector of one chain after one block sweep (after exchanges for "
            "tempering), outside the burn-in; tempering retains the cold replica only"
        ),
    }


# --- exact side -----------------------------------------------------------------


def all_states(n: int = N_SITES) -> np.ndarray:
    return np.array(list(itertools.product(range(Q), repeat=n)), dtype=np.int8)


def joint_codes(states: np.ndarray) -> np.ndarray:
    """Code of sites 0..3, site i weighted by 3^i."""
    return states[:, :JOINT_SITES].astype(np.int64) @ (Q ** np.arange(JOINT_SITES))


def symmetry_matrix() -> np.ndarray:
    """Right-multiplying a joint law by S averages it over all label permutations."""
    codes = np.array(list(itertools.product(range(Q), repeat=JOINT_SITES)))[:, ::-1]
    base = Q ** np.arange(JOINT_SITES)
    if not np.array_equal(codes @ base, np.arange(Q**JOINT_SITES)):
        raise ValueError("joint code layout mismatch")
    s = np.zeros((Q**JOINT_SITES, Q**JOINT_SITES))
    for perm in PERMUTATIONS:
        s[np.arange(Q**JOINT_SITES), np.array(perm)[codes] @ base] += 1.0 / len(PERMUTATIONS)
    return s


def exact_references(request: dict) -> dict:
    states = all_states(request["n"])
    codes = joint_codes(states)
    out = {}
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for target in request["targets"]:
        if target["seed"] not in cache:
            score = np.zeros(len(states))
            equal = []
            for (a, b), k in zip(target["edges"], target["couplings"], strict=True):
                eq = states[:, a] == states[:, b]
                score += k * eq
                equal.append(eq)
            cache[target["seed"]] = (score, np.array(equal))
        score, equal = cache[target["seed"]]
        lw = target["beta"] * score
        p = np.exp(lw - lw.max())
        p /= p.sum()
        out[target["id"]] = {
            "joint": np.bincount(codes, weights=p, minlength=Q**JOINT_SITES).tolist(),
            "edge_agreement": (equal.astype(np.float64) @ p).tolist(),
            "ground_state_mass": float(p[score == score.max()].sum()),
        }
    return out


def retained(sampler: str, budget: int) -> tuple[int, int]:
    """(recorded sweeps L, retained states per trial)."""
    length = 5 * budget if sampler == "long" else budget
    per_chain = length - length // 4
    return length, per_chain * (5 if sampler == "independent" else 1)


def noise_floors(request: dict, references: dict) -> dict:
    s = symmetry_matrix()
    out = {}
    for t_index, target in enumerate(request["targets"]):
        joint = np.array(references[target["id"]]["joint"])
        for s_index, sampler in enumerate(request["samplers"]):
            for b_index, budget in enumerate(request["budgets"]):
                n = retained(sampler, budget)[1]
                seed = [request["noise_floor"]["numpy_seed"], t_index, s_index, b_index]
                rng = np.random.default_rng(seed)
                draws = rng.multinomial(n, joint, size=request["noise_floor"]["draws"]) / n
                plain = 0.5 * np.abs(draws - joint).sum(axis=1)
                sym = 0.5 * np.abs(draws @ s - joint).sum(axis=1)
                out[f"{target['id']}/{sampler}/T{budget}"] = {
                    "retained": n,
                    "plain": float(plain.mean()),
                    "sym": float(sym.mean()),
                }
    return out


# --- THRML sampler ---------------------------------------------------------------


def build_program(n: int, edges, couplings, blocks, betas):
    """One THRML program over len(betas) disjoint copies of the graph."""
    r = len(betas)
    nodes = [CategoricalNode() for _ in range(r * n)]
    heads = [nodes[k * n + a] for k in range(r) for a, _ in edges]
    tails = [nodes[k * n + b] for k in range(r) for _, b in edges]
    weights = np.concatenate(
        [
            beta * np.asarray(couplings, dtype=np.float64)[:, None, None] * np.eye(Q)
            for beta in betas
        ]
    )
    factor = SquareCategoricalEBMFactor(
        [Block(heads), Block(tails)], jnp.asarray(weights, dtype=jnp.float32)
    )
    layout = [np.array([k * n + i for k in range(r) for i in block]) for block in blocks]
    free = [Block([nodes[j] for j in idx]) for idx in layout]
    program = FactorSamplingProgram(
        BlockGibbsSpec(free, []), [CategoricalGibbsConditional(Q) for _ in free], [factor], []
    )
    return program, layout


def layout_maps(layout: list[np.ndarray], r: int, n: int):
    flat_index = np.concatenate(layout)
    splits = np.cumsum([len(x) for x in layout])[:-1]

    def to_labels(state):
        flat = jnp.concatenate(state)
        return jnp.zeros(r * n, jnp.uint8).at[flat_index].set(flat).reshape(r, n)

    def from_labels(labels):
        return list(jnp.split(labels.reshape(-1)[flat_index], splits))

    return to_labels, from_labels


def exchange_log_acceptance(beta_i, beta_j, score_i, score_j):
    """Log Metropolis acceptance of swapping replicas i, j under P ~ exp(beta * score)."""
    return jnp.minimum(0.0, (beta_i - beta_j) * (score_j - score_i))


def sampler_betas(sampler: str, beta: float) -> np.ndarray:
    if sampler == "tempering":
        return np.asarray(LADDER) * beta
    return np.full(1 if sampler == "long" else 5, beta)


def compile_sampler(target: dict, sampler: str, steps: int):
    n = N_SITES
    edges = [tuple(e) for e in target["edges"]]
    betas = sampler_betas(sampler, target["beta"])
    r = len(betas)
    program, layout = build_program(n, edges, target["couplings"], target["blocks"], betas)
    to_labels, from_labels = layout_maps(layout, r, n)
    a_idx = np.array([a for a, _ in edges])
    b_idx = np.array([b for _, b in edges])
    k_vec = jnp.asarray(target["couplings"], jnp.float32)
    beta_vec = jnp.asarray(betas, jnp.float32)

    def single(key, init):
        sampler_states = [s.init() for s in program.samplers]

        def step(carry, t):
            state, key = carry
            key, sweep_key, exchange_key = jax.random.split(key, 3)
            state, _ = sample_blocks(sweep_key, state, [], program, sampler_states)
            labels = to_labels(state)
            accepted = jnp.zeros(2, jnp.bool_)
            if sampler == "tempering":
                score = jnp.sum(k_vec * (labels[:, a_idx] == labels[:, b_idx]), axis=1)
                left = jnp.asarray([0, 2]) + t % 2
                right = left + 1
                log_a = exchange_log_acceptance(
                    beta_vec[left], beta_vec[right], score[left], score[right]
                )
                accepted = jnp.log(jax.random.uniform(exchange_key, (2,))) < log_a
                old_left, old_right = labels[left], labels[right]
                labels = labels.at[left].set(jnp.where(accepted[:, None], old_right, old_left))
                labels = labels.at[right].set(jnp.where(accepted[:, None], old_left, old_right))
                state = from_labels(labels)
            kept = labels[-1:] if sampler == "tempering" else labels
            return (state, key), (kept, accepted)

        _, (kept, accepted) = jax.lax.scan(step, (from_labels(init), key), jnp.arange(steps))
        return kept, accepted

    return jax.jit(jax.vmap(single)), r


def trial_keys(request: dict, t_index: int, sampler: str):
    root = jax.random.fold_in(jax.random.key(request["root_seed"]), t_index)
    pairs = jax.vmap(lambda i: jax.random.split(jax.random.fold_in(root, i)))(
        jnp.arange(request["trials"])
    )
    s_index = request["samplers"].index(sampler)
    sample_keys = jax.vmap(lambda k: jax.random.fold_in(k, s_index))(pairs[:, 1])
    return pairs[:, 0], sample_keys


def initial_labels(init_keys, r: int):
    """Uniform labels; replica k of every sampler starts from the same draw."""
    full = jax.vmap(lambda k: jax.random.randint(k, (5, N_SITES), 0, Q))(init_keys)
    return full[:, -r:].astype(jnp.uint8) if r == 1 else full[:, :r].astype(jnp.uint8)


def counts(kept: np.ndarray, edges, length: int) -> dict:
    """Per-trial joint counts and edge-agreement counts over the retained window."""
    window = kept[:, length // 4 : length].reshape(kept.shape[0], -1, N_SITES).astype(np.int64)
    a = np.array([a for a, _ in edges])
    b = np.array([b for _, b in edges])
    joint, agree = [], []
    for trial in window:
        joint.append(np.bincount(joint_codes(trial), minlength=Q**JOINT_SITES).tolist())
        agree.append((trial[:, a] == trial[:, b]).sum(axis=0).tolist())
    return {"retained": int(window.shape[1]), "joint": joint, "agree": agree}


def exchange_counts(accepted: np.ndarray, budget: int) -> dict:
    """Accepted and attempted exchanges per left replica index over the first T sweeps."""
    acc, att = [0, 0, 0, 0], [0, 0, 0, 0]
    for t in range(budget):
        for slot in range(2):
            left = 2 * slot + t % 2
            att[left] += accepted.shape[0]
            acc[left] += int(accepted[:, t, slot].sum())
    return {"accepted": acc, "attempted": att}


def run_target_sampler(request: dict, t_index: int, sampler: str) -> dict:
    """Timed runs at every budget; the T_max run supplies every budget's prefix counts."""
    target = request["targets"][t_index]
    edges = [tuple(e) for e in target["edges"]]
    init_keys, sample_keys = trial_keys(request, t_index, sampler)
    budgets = request["budgets"]
    out = {"budgets": {}, "timing": {}}
    max_kept = max_accepted = None
    r = None
    for budget in budgets:
        length, _ = retained(sampler, budget)
        t0 = time.perf_counter()
        fn, r = compile_sampler(target, sampler, length)
        init = initial_labels(init_keys, r)
        executable = fn.lower(sample_keys, init).compile()
        compile_seconds = time.perf_counter() - t0

        def pipeline(executable=executable, init=init, length=length):
            kept, accepted = executable(sample_keys, init)
            kept = np.asarray(kept.block_until_ready())
            return kept, np.asarray(accepted), counts(kept, edges, length)

        pipeline()  # warm-up
        seconds = []
        for _ in range(request["timing"]["repeats"]):
            t0 = time.perf_counter()
            kept, accepted, estimated = pipeline()
            seconds.append(time.perf_counter() - t0)
        out["timing"][f"T{budget}"] = {"compile_seconds": compile_seconds, "warm_seconds": seconds}
        out["budgets"][f"T{budget}"] = {"timed_counts": estimated}
        if budget == budgets[-1]:
            max_kept, max_accepted = kept, accepted
    prefix_ok = True
    for budget in budgets:
        length, _ = retained(sampler, budget)
        prefix = counts(max_kept[:, :length], edges, length)
        entry = out["budgets"][f"T{budget}"]
        if prefix != entry.pop("timed_counts"):
            prefix_ok = False
        entry.update(prefix)
        if sampler == "tempering":
            entry["exchanges"] = exchange_counts(max_accepted, budget)
    out["prefix_ok"] = prefix_ok
    out["replicas"] = r
    return out


# --- preflight ---------------------------------------------------------------------


def preflight(request: dict) -> dict:
    """Layout round trip and one-sweep law per replica against stage A's exact kernel."""
    spec = request["preflight"]
    stage_request = stage_a.study_request(chains=spec["chains"])
    edges = [tuple(e) for e in stage_request["potts"]["edges"]]
    rng = np.random.default_rng(ROOT_SEED)
    couplings = (-rng.integers(1, 6, len(edges)) / 5).tolist()
    blocks = [list(b) for b in stage_request["blocks"]]
    betas = spec["betas"]
    n = stage_request["potts"]["sites"]
    program, layout = build_program(n, edges, couplings, blocks, betas)
    to_labels, from_labels = layout_maps(layout, len(betas), n)
    probe = jax.random.randint(jax.random.key(3), (len(betas), n), 0, Q).astype(jnp.uint8)
    round_trip = bool(np.array_equal(np.asarray(to_labels(from_labels(probe))), np.asarray(probe)))

    def one(key):
        sampler_states = [s.init() for s in program.samplers]
        init = from_labels(jnp.zeros((len(betas), n), jnp.uint8))
        state, _ = sample_blocks(key, init, [], program, sampler_states)
        return to_labels(state)

    keys = jax.random.split(jax.random.key(ROOT_SEED + 1), spec["chains"])
    labels = np.asarray(jax.jit(jax.vmap(one))(keys), dtype=np.int64)
    states = stage_a.potts_states(Q, n)
    replicas = []
    for k, beta in enumerate(betas):
        params = {
            "fields": np.zeros((n, Q)),
            "couplings": np.asarray(couplings)[:, None, None] * np.eye(Q)[None],
            "edges": edges,
            "beta": beta,
            "q": Q,
        }
        law = stage_a.point_mass(states, [0] * n, Q) @ stage_a.sweep_kernel(params, states, blocks)
        hist = np.bincount(stage_a.potts_index(labels[:, k]), minlength=len(states))
        tol = stage_a.tolerance(law, spec["chains"], stage_request["tolerance"], k)
        tv = stage_a.total_variation(hist / spec["chains"], law)
        replicas.append({"beta": beta, "tv": tv, "tolerance": tol, "pass": tv <= tol})
    return {
        "round_trip": round_trip,
        "replicas": replicas,
        "couplings": couplings,
        "passed": round_trip and all(x["pass"] for x in replicas),
    }


# --- evaluation ----------------------------------------------------------------------


def evaluate(request: dict, references: dict, floors: dict, results: dict) -> dict:
    s = symmetry_matrix()
    cells = {}
    for target in request["targets"]:
        ref = references[target["id"]]
        joint = np.array(ref["joint"])
        agree = np.array(ref["edge_agreement"])
        for sampler in request["samplers"]:
            for budget in request["budgets"]:
                entry = results[target["id"]][sampler]["budgets"][f"T{budget}"]
                n = entry["retained"]
                hist = np.array(entry["joint"], dtype=np.float64) / n
                edge_mae = np.abs(np.array(entry["agree"]) / n - agree).mean(axis=1)
                for estimator in request["estimators"]:
                    est = hist @ s if estimator == "sym" else hist
                    tv = 0.5 * np.abs(est - joint).sum(axis=1)
                    key = f"{target['id']}/{sampler}+{estimator}/T{budget}"
                    floor = floors[f"{target['id']}/{sampler}/T{budget}"][estimator]
                    cells[key] = {
                        "joint_tv_mean": float(tv.mean()),
                        "edge_mae_mean": float(edge_mae.mean()),
                        "joint_tv_per_trial": tv.tolist(),
                        "edge_mae_per_trial": edge_mae.tolist(),
                        "trials_passing": int(((tv <= THRESHOLD) & (edge_mae <= THRESHOLD)).sum()),
                        "iid_floor_joint_tv": floor,
                        "retained_per_trial": n,
                    }
    qualification = {}
    for target in request["targets"]:
        for sampler in request["samplers"]:
            for estimator in request["estimators"]:
                arm = f"{sampler}+{estimator}"
                ok = [
                    cells[f"{target['id']}/{arm}/T{b}"]["joint_tv_mean"] <= THRESHOLD
                    and cells[f"{target['id']}/{arm}/T{b}"]["edge_mae_mean"] <= THRESHOLD
                    for b in request["budgets"]
                ]
                selected = None
                for i, budget in enumerate(request["budgets"]):
                    if all(ok[i:]):
                        selected = budget
                        break
                qualification[f"{target['id']}/{arm}"] = selected
    decisions = {}
    for target in request["targets"]:
        tid = target["id"]
        combo = qualification[f"{tid}/tempering+sym"]
        baselines = {arm: qualification[f"{tid}/{arm}"] for arm in ("long+sym", "independent+sym")}
        qualifying = {arm: b for arm, b in baselines.items() if b is not None}
        best_arm = min(qualifying, key=lambda a: (qualifying[a], a)) if qualifying else None
        best = qualifying.get(best_arm) if best_arm else None
        if combo is None and best is None:
            outcome, ratio = "neither_qualifies", None
        elif best is None:
            outcome, ratio = "only_tempering_sym_qualifies", None
        elif combo is None:
            outcome, ratio = "only_baseline_qualifies", None
        else:
            ratio = best / combo
            outcome = (
                "tempering_sym_smaller_budget"
                if combo < best
                else "baseline_smaller_budget"
                if best < combo
                else "equal_budget"
            )
        decisions[tid] = {
            "beta": target["beta"],
            "tempering_sym_budget": combo,
            "baseline_budgets": baselines,
            "best_baseline": best_arm,
            "outcome": outcome,
            "budget_ratio_baseline_over_tempering_sym": ratio,
        }
    exchange = {}
    for target in request["targets"]:
        ex = results[target["id"]]["tempering"]["budgets"][f"T{request['budgets'][-1]}"]
        exchange[target["id"]] = [
            a / b if b else None
            for a, b in zip(ex["exchanges"]["accepted"], ex["exchanges"]["attempted"], strict=True)
        ]
    return {
        "cells": cells,
        "qualification": qualification,
        "decisions": decisions,
        "exchange_acceptance": exchange,
    }


def timing_summary(request: dict, results: dict, evaluation: dict) -> dict:
    """Median warm seconds per sampler and budget, and time ratios at selected budgets."""
    medians = {}
    for target in request["targets"]:
        for sampler in request["samplers"]:
            for budget in request["budgets"]:
                warm = results[target["id"]][sampler]["timing"][f"T{budget}"]["warm_seconds"]
                medians[f"{target['id']}/{sampler}/T{budget}"] = float(np.median(warm))
    ratios = {}
    for tid, d in evaluation["decisions"].items():
        if d["budget_ratio_baseline_over_tempering_sym"] is None:
            ratios[tid] = None
            continue
        base_sampler = d["best_baseline"].split("+")[0]
        base = medians[f"{tid}/{base_sampler}/T{d['baseline_budgets'][d['best_baseline']]}"]
        combo = medians[f"{tid}/tempering/T{d['tempering_sym_budget']}"]
        ratios[tid] = base / combo
    return {"median_warm_seconds": medians, "time_ratio_baseline_over_tempering_sym": ratios}


# --- persistence, replay, report -------------------------------------------------------


def hashed_results(results: dict) -> dict:
    """Counts only; timings are not hashed."""
    return {
        tid: {
            sampler: {
                "budgets": r["budgets"],
                "prefix_ok": r["prefix_ok"],
                "replicas": r["replicas"],
            }
            for sampler, r in by_sampler.items()
        }
        for tid, by_sampler in results.items()
    }


def assemble(request, check, references, floors, results) -> dict:
    counted = hashed_results(results)
    evaluation = evaluate(request, references, floors, counted)
    record = {
        "schema": "potts_symmetry_tempering.record.v1",
        "evidence": {
            "exact_references": "exact_reference (float64 enumeration of 3^12 states)",
            "sampled_cells": "software_simulation (THRML 0.1.4, CPU, float32); not hardware",
            "timings": "warm CPU wall seconds; descriptive, not hashed, not a hardware claim",
        },
        "request": request,
        "request_digest": canonical_sha256(request),
        "preflight": check,
        "references": references,
        "noise_floors": floors,
        "results": counted,
        "evaluation": evaluation,
        "timings": {
            tid: {s: r["timing"] for s, r in by_sampler.items()}
            for tid, by_sampler in results.items()
        },
    }
    record["timing_summary"] = timing_summary(
        request, _with_timings(counted, record["timings"]), evaluation
    )
    record["result_digest"] = canonical_sha256(
        {"preflight": check, "results": counted, "evaluation": evaluation}
    )
    return record


def _with_timings(counted: dict, timings: dict) -> dict:
    return {
        tid: {s: {**r, "timing": timings[tid][s]} for s, r in by_sampler.items()}
        for tid, by_sampler in counted.items()
    }


EXACT_REPLAY_ATOL = 1e-12
FLOOR_REPLAY_RTOL = 1e-3
"""Noise floors are redrawn from the archived exact joint; the slack covers libm last bits."""


def _close(new, old, what: str) -> None:
    if not np.allclose(new, old, atol=EXACT_REPLAY_ATOL, rtol=0):
        raise ValueError(f"{what} drifted")


def replay(record: dict, request: dict) -> None:
    if record["request"] != request or record["request_digest"] != canonical_sha256(request):
        raise ValueError("archived request does not match this code's request")
    references = exact_references(request)
    for tid, ref in references.items():
        stored = record["references"][tid]
        for key in ("joint", "edge_agreement", "ground_state_mass"):
            _close(ref[key], stored[key], f"exact {key} for {tid}")
    floors = noise_floors(request, record["references"])
    for key, value in floors.items():
        stored = record["noise_floors"][key]
        if value["retained"] != stored["retained"]:
            raise ValueError(f"retained count drifted for {key}")
        for est in ("plain", "sym"):
            if not np.isclose(value[est], stored[est], rtol=FLOOR_REPLAY_RTOL, atol=0):
                raise ValueError(f"noise floor {est} drifted for {key}")
    for tid, by_sampler in record["results"].items():
        for sampler, r in by_sampler.items():
            if not r["prefix_ok"]:
                raise ValueError(f"prefix check failed for {tid}/{sampler}")
            for budget in request["budgets"]:
                entry = r["budgets"][f"T{budget}"]
                if entry["retained"] != retained(sampler, budget)[1]:
                    raise ValueError(f"retained count drifted for {tid}/{sampler}/T{budget}")
                for row in entry["joint"]:
                    if sum(row) != entry["retained"]:
                        raise ValueError(f"joint counts do not sum for {tid}/{sampler}/T{budget}")
    evaluation = evaluate(request, record["references"], record["noise_floors"], record["results"])
    digest = canonical_sha256(
        {"preflight": record["preflight"], "results": record["results"], "evaluation": evaluation}
    )
    if digest != record["result_digest"]:
        raise ValueError("result digest does not replay")
    summary = timing_summary(
        request, _with_timings(record["results"], record["timings"]), evaluation
    )
    if canonical_json(summary) != canonical_json(record["timing_summary"]):
        raise ValueError("timing summary does not replay")


def _fmt_budget(b) -> str:
    return "NR" if b is None else str(b)


def render_report(record: dict) -> str:
    req, ev = record["request"], record["evaluation"]
    tim = record["timing_summary"]
    lines = [
        "# Potts stage B: label symmetry and tempering",
        "",
        "Exact references: float64 enumeration of all 3^12 states (`exact_reference`). Sampled",
        "cells: THRML 0.1.4 categorical block Gibbs on CPU, float32, 16 independent trials",
        "(`software_simulation`). Timings are warm CPU seconds and are descriptive. No hardware",
        "evidence anywhere in this report.",
        "",
        f"Request digest `{record['request_digest']}`; result digest `{record['result_digest']}`.",
        "",
        "## Selected budgets (smallest T passing at every larger T; NR = not reached by "
        f"T = {req['budgets'][-1]})",
        "",
        "| target | ground-state mass | long+plain | long+sym | independent+plain "
        "| independent+sym | tempering+plain | tempering+sym | outcome | budget ratio "
        "| time ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for target in req["targets"]:
        tid = target["id"]
        q = ev["qualification"]
        d = ev["decisions"][tid]
        ratio = d["budget_ratio_baseline_over_tempering_sym"]
        tr = tim["time_ratio_baseline_over_tempering_sym"][tid]
        arms = " | ".join(
            _fmt_budget(q[f"{tid}/{s}+{e}"]) for s in req["samplers"] for e in req["estimators"]
        )
        lines.append(
            f"| {tid} | {record['references'][tid]['ground_state_mass']:.3f} | {arms} "
            f"| {d['outcome']} | {'n/a' if ratio is None else f'{ratio:.2f}'} "
            f"| {'n/a' if tr is None else f'{tr:.2f}'} |"
        )
    lines += [
        "",
        "Budget ratio: best qualifying symmetry-aware ordinary baseline over tempering+sym,",
        "in T at matched label redraws (tempering also pays 2T exchange attempts). Time ratio:",
        "the same comparison in median warm CPU seconds at the selected budgets.",
        "",
        "## Mean errors per budget",
        "",
        "Each entry is mean trial joint TV / iid floor at the same retained count; edge MAE in",
        "the last column is the largest over arms at that budget.",
        "",
    ]
    for target in req["targets"]:
        tid = target["id"]
        lines += [
            f"### {tid}",
            "",
            "| T | "
            + " | ".join(f"{s}+{e}" for s in req["samplers"] for e in req["estimators"])
            + " | max edge MAE |",
            "| --- |" + " --- |" * (len(req["samplers"]) * len(req["estimators"]) + 1),
        ]
        for budget in req["budgets"]:
            row, edge = [], 0.0
            for s in req["samplers"]:
                for e in req["estimators"]:
                    c = ev["cells"][f"{tid}/{s}+{e}/T{budget}"]
                    row.append(f"{c['joint_tv_mean']:.3f} / {c['iid_floor_joint_tv']:.3f}")
                    edge = max(edge, c["edge_mae_mean"])
            lines.append(f"| {budget} | " + " | ".join(row) + f" | {edge:.4f} |")
        acc = ev["exchange_acceptance"][tid]
        lines += [
            "",
            "Exchange acceptance by ladder pair (coldest last): "
            + ", ".join("n/a" if a is None else f"{a:.2f}" for a in acc),
            "",
        ]
    pre = record["preflight"]
    lines += [
        "## Preflight",
        "",
        f"Layout round trip: {pre['round_trip']}. One-sweep law per replica against the stage A",
        "exact kernel (beta, TV, tolerance): "
        + "; ".join(
            f"{r['beta']:g}: {r['tv']:.4f} <= {r['tolerance']:.4f}" for r in pre["replicas"]
        ),
    ]
    return "\n".join(lines) + "\n"


def completion(record: dict, provenance: dict, archive_bytes: bytes) -> dict:
    ev = record["evaluation"]
    req = record["request"]
    outcomes: dict[str, int] = {}
    for d in ev["decisions"].values():
        key = f"beta{d['beta']:g}/{d['outcome']}"
        outcomes[key] = outcomes.get(key, 0) + 1
    qualifying = {
        arm: sum(ev["qualification"][f"{t['id']}/{arm}"] is not None for t in req["targets"])
        for arm in (f"{s}+{e}" for s in req["samplers"] for e in req["estimators"])
    }
    return {
        "status": "potts_symmetry_tempering_complete",
        "targets": len(req["targets"]),
        "sampler_cells": len(req["targets"]) * len(req["samplers"]) * len(req["budgets"]),
        "estimator_cells": len(ev["cells"]),
        "primary_decisions": len(ev["decisions"]),
        "targets_qualifying_by_arm": qualifying,
        "decision_outcomes": dict(sorted(outcomes.items())),
        "preflight_passed": record["preflight"]["passed"],
        "prefix_checks_passed": all(
            r["prefix_ok"] for by in record["results"].values() for r in by.values()
        ),
        "trials": req["trials"],
        "replayed": True,
        "integrity": True,
        "autosave": "none; the study runs in about 15 minutes",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def run_study(output_dir: str | Path) -> dict:
    request = study_request()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    log_path = destination / "run.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} Potts-B {message}\n"
            )

    log("preflight")
    check = preflight(request)
    if not check["passed"]:
        atomic_write_text(destination / "preflight.json", canonical_json(check) + "\n")
        log("stopped: preflight failed; no production sampling")
        raise SystemExit("preflight failed; see preflight.json")
    log("preflight passed; exact references and noise floors")
    t0 = time.monotonic()
    references = exact_references(request)
    floors = noise_floors(request, references)
    exact_seconds = time.monotonic() - t0
    log(f"exact side {exact_seconds:.1f}s")
    results: dict = {}
    for t_index, target in enumerate(request["targets"]):
        results[target["id"]] = {}
        for sampler in request["samplers"]:
            t0 = time.monotonic()
            out = run_target_sampler(request, t_index, sampler)
            results[target["id"]][sampler] = out
            seconds = time.monotonic() - t0
            log(f"{target['id']}/{sampler} {seconds:.1f}s prefix_ok={out['prefix_ok']}")
    record = assemble(request, check, references, floors, results)
    archive_path = destination / "study.json.gz"
    archive_bytes = gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    archive_path.write_bytes(archive_bytes)
    log("archive written; replaying")
    t0 = time.monotonic()
    persisted = json.loads(gzip.decompress(archive_path.read_bytes()))
    replay(persisted, study_request())
    replay_seconds = time.monotonic() - t0
    runtime = collect_runtime_provenance(REPO_ROOT)
    provenance = {
        "schema": "potts_symmetry_tempering.provenance.v1",
        "request_digest": persisted["request_digest"],
        "result_digest": persisted["result_digest"],
        "runtime": json.loads(canonical_json(runtime.model_dump())),
        "exact_side_seconds": exact_seconds,
        "replay_seconds": replay_seconds,
        "total_seconds": time.monotonic() - started,
    }
    atomic_write_text(destination / "summary.md", render_report(persisted))
    atomic_write_text(destination / "provenance.json", canonical_json(provenance) + "\n")
    atomic_write_text(
        destination / "completion.json",
        canonical_json(completion(persisted, provenance, archive_bytes)) + "\n",
    )
    log(f"completion published after {time.monotonic() - started:.1f}s")
    return persisted


def replay_archive(archive: str | Path, output_dir: str | Path) -> dict:
    archive_bytes = Path(archive).read_bytes()
    record = json.loads(gzip.decompress(archive_bytes))
    replay(record, study_request())
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": "potts_symmetry_tempering.provenance.v1",
        "mode": "replay_only",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
    }
    atomic_write_text(destination / "summary.md", render_report(record))
    atomic_write_text(
        destination / "completion.json",
        canonical_json(completion(record, provenance, archive_bytes)) + "\n",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay", help="replay this study.json.gz instead of sampling")
    args = parser.parse_args()
    if args.replay:
        replay_archive(args.replay, args.output_dir)
    else:
        run_study(args.output_dir)


if __name__ == "__main__":
    main()
