"""Exploratory probe: THRML Potts tempering and label symmetry on weighted cubic graphs.

EXPLORATION ONLY. Output of this script is not recorded evidence and must not
be cited as a study result. It checks, before a Potts stage B protocol is
frozen, (1) that replica-exchange tempering runs inside THRML as one program
over disjoint replica copies with exchanges between sweeps, and how fast; and
(2) which cold inverse temperature makes zero-field antiferromagnetic
three-state Potts targets hard but reachable, so the protocol's fixed choices
do not cap or trivialize the accuracy metric. It uses graph seeds 900-901,
which no protocol may reuse.

Run: JAX_PLATFORMS=cpu uv run python docs/research/potts_tempering_probe.py
"""

from __future__ import annotations

import itertools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block
from thrml.block_sampling import BlockGibbsSpec, sample_blocks
from thrml.factor import FactorSamplingProgram
from thrml.models import CategoricalGibbsConditional
from thrml.models.discrete_ebm import SquareCategoricalEBMFactor
from thrml.pgm import CategoricalNode

Q = 3
LADDER = np.array(
    [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
)  # times the cold beta, as in the Ising studies
TRIALS = 16
PERMS = list(itertools.permutations(range(Q)))


def cubic_graph(n: int, seed: int):
    rng = np.random.default_rng(seed)
    ring = {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
    for _ in range(10_000):
        order = rng.permutation(n)
        matching = {tuple(sorted(map(int, order[i : i + 2]))) for i in range(0, n, 2)}
        if not ring & matching:
            break
    edges = sorted(ring | matching)
    weights = rng.integers(1, 6, len(edges))
    return edges, -weights / 5.0  # K_e < 0: equal neighbouring labels are penalized


def greedy_colouring(n: int, edges) -> list[list[int]]:
    colour = {}
    for i in range(n):
        used = {
            colour[j] for a, b in edges for j in (a, b) if i in (a, b) and j != i and j in colour
        }
        colour[i] = min(c for c in range(n) if c not in used)
    return [[i for i in range(n) if colour[i] == c] for c in sorted(set(colour.values()))]


def exact(n, edges, couplings, beta):
    states = np.array(list(itertools.product(range(Q), repeat=n)), dtype=np.int8)
    score = np.zeros(len(states))
    for (a, b), k in zip(edges, couplings, strict=True):
        score += k * (states[:, a] == states[:, b])
    p = np.exp(beta * (score - score.max()))
    p /= p.sum()
    codes = states[:, :4].astype(np.int64) @ (Q ** np.arange(4))
    joint = np.bincount(codes, weights=p, minlength=Q**4)
    agree = np.array([p @ (states[:, a] == states[:, b]) for a, b in edges])
    return joint, agree


def build(n, edges, couplings, blocks, betas):
    r = len(betas)
    nodes = [CategoricalNode() for _ in range(r * n)]
    heads = [nodes[k * n + a] for k in range(r) for a, _ in edges]
    tails = [nodes[k * n + b] for k in range(r) for _, b in edges]
    weights = np.concatenate([b * np.asarray(couplings)[:, None, None] * np.eye(Q) for b in betas])
    factor = SquareCategoricalEBMFactor(
        [Block(heads), Block(tails)], jnp.asarray(weights, jnp.float32)
    )
    layout = [[k * n + i for k in range(r) for i in block] for block in blocks]
    free = [Block([nodes[j] for j in idx]) for idx in layout]
    program = FactorSamplingProgram(
        BlockGibbsSpec(free, []), [CategoricalGibbsConditional(Q) for _ in free], [factor], []
    )
    return program, [np.array(idx) for idx in layout]


def compile_run(n, edges, couplings, blocks, method, beta, steps):
    betas = LADDER * beta if method == "tempering" else np.full(1 if method == "long" else 5, beta)
    program, layout = build(n, edges, couplings, blocks, betas)
    r = len(betas)
    flat_index = np.concatenate(layout)  # positions of every node in concatenated block state
    a_idx = np.array([a for a, _ in edges])
    b_idx = np.array([b for _, b in edges])
    k_vec = jnp.asarray(couplings, jnp.float32)
    beta_vec = jnp.asarray(betas, jnp.float32)
    sizes = [len(x) for x in layout]
    splits = np.cumsum(sizes)[:-1]

    def to_labels(state):
        flat = jnp.concatenate(state)
        return jnp.zeros(r * n, jnp.uint8).at[flat_index].set(flat).reshape(r, n)

    def from_labels(labels):
        flat = labels.reshape(-1)[flat_index]
        return list(jnp.split(flat, splits))

    def single(key, init):
        sampler_states = [s.init() for s in program.samplers]
        state = from_labels(init)

        def step(carry, t):
            state, key = carry
            key, sweep_key, ex_key = jax.random.split(key, 3)
            state, _ = sample_blocks(sweep_key, state, [], program, sampler_states)
            labels = to_labels(state)
            accepted = jnp.zeros(2, jnp.bool_)
            if method == "tempering":
                score = jnp.sum(k_vec * (labels[:, a_idx] == labels[:, b_idx]), axis=1)
                left = jnp.asarray([0, 2]) + t % 2
                right = left + 1
                log_a = jnp.minimum(
                    0.0, (beta_vec[left] - beta_vec[right]) * (score[right] - score[left])
                )
                accepted = jnp.log(jax.random.uniform(ex_key, (2,))) < log_a
                old_l, old_r = labels[left], labels[right]
                labels = labels.at[left].set(jnp.where(accepted[:, None], old_r, old_l))
                labels = labels.at[right].set(jnp.where(accepted[:, None], old_l, old_r))
                state = from_labels(labels)
            keep = labels[-1:] if method == "tempering" else labels
            return (state, key), (keep, accepted)

        _, (kept, accepted) = jax.lax.scan(step, (state, key), jnp.arange(steps))
        return kept, accepted

    return jax.jit(jax.vmap(single)), r


def symmetrized(hist):
    out = np.zeros_like(hist)
    codes = np.array(list(itertools.product(range(Q), repeat=4)))[:, ::-1]  # digit i = site i
    base = Q ** np.arange(4)
    for perm in PERMS:
        out += hist[(np.array(perm)[codes] @ base)]
    return out / len(PERMS)


def score(kept, edges, joint, agree, sym):
    a = np.array([a for a, _ in edges])
    b = np.array([b for _, b in edges])
    tvs, maes = [], []
    for trial in kept:
        states = trial.reshape(-1, trial.shape[-1]).astype(np.int64)
        hist = np.bincount(states[:, :4] @ (Q ** np.arange(4)), minlength=Q**4) / len(states)
        if sym:
            hist = symmetrized(hist)
        tvs.append(0.5 * np.abs(hist - joint).sum())
        maes.append(np.abs((states[:, a] == states[:, b]).mean(0) - agree).mean())
    return float(np.mean(tvs)), float(np.mean(maes))


def main() -> None:
    sizes = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [10]
    betas = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4.0]
    budgets = (64, 256, 1024)
    for n in sizes:
        for seed in (900, 901):
            edges, couplings = cubic_graph(n, seed)
            blocks = greedy_colouring(n, edges)
            for beta in betas:
                joint, agree = exact(n, edges, couplings, beta)
                print(f"n={n} seed={seed} beta={beta} colours={len(blocks)}")
                for method in ("long", "independent", "tempering"):
                    horizon = max(budgets)
                    steps = 5 * horizon if method == "long" else horizon
                    fn, r = compile_run(n, edges, couplings, blocks, method, beta, steps)
                    keys = jax.random.split(jax.random.key(seed * 10 + n), TRIALS)
                    init = jax.random.randint(jax.random.key(1), (TRIALS, r, n), 0, Q).astype(
                        jnp.uint8
                    )
                    t0 = time.perf_counter()
                    kept, accepted = fn(keys, init)
                    kept = np.asarray(kept.block_until_ready())
                    seconds = time.perf_counter() - t0
                    row = []
                    for t in budgets:
                        length = 5 * t if method == "long" else t
                        window = kept[:, length // 4 : length]
                        plain = score(window, edges, joint, agree, sym=False)
                        sym = score(window, edges, joint, agree, sym=True)
                        row.append(f"T{t}: {plain[0]:.3f}/{sym[0]:.3f} e{plain[1]:.3f}")
                    rate = float(np.asarray(accepted).mean()) if method == "tempering" else 0.0
                    print(f"  {method:11s} {seconds:6.1f}s swap {rate:.2f}  " + "  ".join(row))


if __name__ == "__main__":
    main()
