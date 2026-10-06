"""Exploratory probe: three stochastic associative-memory constructions in THRML.

EXPLORATION ONLY. Output of this script is not recorded evidence and must not
be cited as a study result. It checks, before any associative-memory protocol
is drafted:

1. Exact recall curves. N visible +-1 spins store P random patterns. A cue
   clamps the first half of one stored pattern; the memory's exact
   conditional law over the missing half (enumerated) gives the expected
   fraction of missing bits recalled correctly and the probability of exact
   completion. Three energies, all functions of the overlaps O_mu = xi_mu . s:
   - hopfield: log w = beta / (2N) sum_mu O_mu^2 (Hebbian pairwise couplings);
   - rbm: one binary hidden spin per pattern, couplings beta xi / sqrt(N);
     marginal log w = sum_mu log cosh(beta O_mu / sqrt(N));
   - dense: one categorical hidden node with P labels, couplings
     beta xi / sqrt(N); marginal log w = log sum_mu exp(beta O_mu / sqrt(N)).
2. THRML conventions for each construction, including the mixed
   spin-categorical factor that Potts stage A did not cover: sampled
   conditional marginals and law at a long sweep budget against the exact
   conditional, at N = 16.

Pattern seeds 9000+ are probe-only; no protocol may reuse them.

Run: JAX_PLATFORMS=cpu uv run python docs/research/associative_memory_probe.py
"""

from __future__ import annotations

import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models import CategoricalGibbsConditional, SpinGibbsConditional
from thrml.models.discrete_ebm import DiscreteEBMFactor, SpinEBMFactor
from thrml.pgm import CategoricalNode

MODELS = ("hopfield", "rbm", "dense")


def patterns(p: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1, 1], size=(p, n)).astype(np.float64)


def log_weight(model: str, overlaps: np.ndarray, n: int, beta: float) -> np.ndarray:
    if model == "hopfield":
        return beta / (2 * n) * (overlaps**2).sum(axis=1)
    x = beta * overlaps / np.sqrt(n)
    if model == "rbm":
        return np.logaddexp(x, -x).sum(axis=1)  # log 2cosh, up to a constant
    return logsumexp(x, axis=1)


def missing_states(m: int) -> np.ndarray:
    return np.array(list(itertools.product((-1, 1), repeat=m)), dtype=np.float64)


def exact_conditional(model, xi, target, observed, beta):
    n = xi.shape[1]
    miss = missing_states(n - observed)
    full = np.concatenate([np.repeat(xi[target, :observed][None], len(miss), 0), miss], axis=1)
    lw = log_weight(model, full @ xi.T, n, beta)
    p = np.exp(lw - lw.max())
    return miss, p / p.sum()


def recall(model, xi, target, observed, beta):
    miss, p = exact_conditional(model, xi, target, observed, beta)
    truth = xi[target, observed:]
    bit_acc = float(p @ (miss == truth).mean(axis=1))
    exact = float(p[np.all(miss == truth, axis=1)].sum())
    return bit_acc, exact


def part_one() -> None:
    n, observed = 32, 16
    print(f"Part 1: exact recall, N={n}, cue = first {observed} bits, 5 pattern draws")
    print("model     beta  " + "  ".join(f"P={p:<3d}" for p in (2, 4, 8, 16, 32, 64)))
    for model in MODELS:
        for beta in (1.0, 2.0, 4.0, 8.0):
            row = []
            for p in (2, 4, 8, 16, 32, 64):
                accs = []
                for draw in range(5):
                    xi = patterns(p, n, 9000 + 100 * p + draw)
                    accs.append(
                        np.mean(
                            [recall(model, xi, mu, observed, beta)[0] for mu in range(min(p, 8))]
                        )
                    )
                row.append(f"{np.mean(accs):.3f}")
            print(f"{model:8s} {beta:5.1f}  " + "  ".join(row))


# --- THRML builds -------------------------------------------------------------


def build(model, xi, beta, observed):
    p, n = xi.shape
    vis = [SpinNode() for _ in range(n)]
    obs, free = vis[:observed], vis[observed:]
    factors, blocks, samplers = [], [Block(free)], [SpinGibbsConditional()]
    if model == "hopfield":
        pairs = list(itertools.combinations(range(n), 2))
        j = xi.T @ xi / n
        w = np.array([beta * j[a, b] for a, b in pairs])
        factors.append(
            SpinEBMFactor(
                [Block([vis[a] for a, _ in pairs]), Block([vis[b] for _, b in pairs])],
                jnp.asarray(w, jnp.float32),
            )
        )
        # fully connected free spins: one block per spin
        blocks = [Block([v]) for v in free]
        samplers = [SpinGibbsConditional() for _ in free]
    elif model == "rbm":
        hid = [SpinNode() for _ in range(p)]
        pairs = [(i, mu) for i in range(n) for mu in range(p)]
        w = np.array([beta * xi[mu, i] / np.sqrt(n) for i, mu in pairs])
        factors.append(
            SpinEBMFactor(
                [Block([vis[i] for i, _ in pairs]), Block([hid[mu] for _, mu in pairs])],
                jnp.asarray(w, jnp.float32),
            )
        )
        blocks.append(Block(hid))
        samplers.append(SpinGibbsConditional())
    else:
        hid = CategoricalNode()
        for i in range(n):
            factors.append(
                DiscreteEBMFactor(
                    [Block([vis[i]])],
                    [Block([hid])],
                    jnp.asarray(beta * xi[:, i][None] / np.sqrt(n), jnp.float32),
                )
            )
        blocks.append(Block([hid]))
        samplers.append(CategoricalGibbsConditional(p))
    spec = BlockGibbsSpec(blocks, [Block(obs)])
    return FactorSamplingProgram(spec, samplers, factors, []), blocks, obs, free


def sample_conditional(model, xi, beta, observed, target, sweeps, chains, seed):
    program, blocks, obs, free = build(model, xi, beta, observed)
    clamp = [jnp.asarray(xi[target, :observed] > 0)]
    schedule = SamplingSchedule(n_warmup=sweeps, n_samples=1, steps_per_sample=1)

    def init_block(key, block):
        node = block.nodes[0]
        if isinstance(node, CategoricalNode):
            return jax.random.randint(key, (len(block.nodes),), 0, xi.shape[0]).astype(jnp.uint8)
        return jax.random.bernoulli(key, 0.5, (len(block.nodes),))

    def one(key):
        k_init, k_sample = jax.random.split(key)
        keys = jax.random.split(k_init, len(blocks))
        state = [init_block(k, b) for k, b in zip(keys, blocks, strict=True)]
        return sample_states(k_sample, program, schedule, state, clamp, [Block(free)])[0][0]

    out = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(seed), chains))
    return 2 * np.asarray(out, dtype=np.int64) - 1


def part_two() -> None:
    n, observed, chains = 16, 8, 40_000
    print(
        f"\nPart 2: THRML against exact conditionals, N={n}, cue {observed} bits, {chains} chains"
    )
    miss = missing_states(n - observed)
    codes = (miss > 0).astype(np.int64) @ (1 << np.arange(n - observed)[::-1])
    assert np.array_equal(codes, np.arange(len(miss)))
    for model in MODELS:
        for p, beta in ((3, 2.0), (6, 4.0)):
            xi = patterns(p, n, 9500 + p)
            _, law = exact_conditional(model, xi, 0, observed, beta)
            for sweeps in (1, 64):
                t0 = time.perf_counter()
                s = sample_conditional(model, xi, beta, observed, 0, sweeps, chains, 7 + sweeps)
                seconds = time.perf_counter() - t0
                idx = (s > 0).astype(np.int64) @ (1 << np.arange(n - observed)[::-1])
                hist = np.bincount(idx, minlength=len(miss)) / chains
                tv = 0.5 * np.abs(hist - law).sum()
                marg = np.abs((s > 0).mean(0) - law @ (miss > 0)).max()
                draws = np.random.default_rng(1).multinomial(chains, law, size=400) / chains
                tol = np.quantile(0.5 * np.abs(draws - law).sum(1), 0.999)
                print(
                    f"  {model:8s} P={p} beta={beta} K={sweeps:3d}: TV {tv:.4f} "
                    f"(iid 0.999 tol {tol:.4f}) max marginal err {marg:.4f}  ({seconds:.1f}s)"
                )


def part_three() -> None:
    """Best recall over a wider beta grid, and the slow rbm case at longer budgets."""
    n, observed = 32, 16
    betas = (1, 2, 4, 8, 16, 32, 64)
    print("\nPart 3: best recall over beta (exact): bit accuracy / P(exact completion) @ beta")
    for model in MODELS:
        row = []
        for p in (4, 8, 16, 32, 64, 128):
            best = None
            for beta in betas:
                accs, exacts = [], []
                for draw in range(3):
                    xi = patterns(p, n, 9000 + 100 * p + draw)
                    r = [recall(model, xi, mu, observed, beta) for mu in range(min(p, 6))]
                    accs.append(np.mean([x[0] for x in r]))
                    exacts.append(np.mean([x[1] for x in r]))
                cand = (np.mean(accs), np.mean(exacts), beta)
                best = cand if best is None or cand[0] > best[0] else best
            row.append(f"P={p}: {best[0]:.3f}/{best[1]:.2f}@{best[2]}")
        print(f"{model:8s} " + "  ".join(row))
    xi = patterns(6, 16, 9506)
    _, law = exact_conditional("rbm", xi, 0, 8, 4.0)
    for k in (256, 1024):
        s = sample_conditional("rbm", xi, 4.0, 8, 0, k, 20000, 11 + k)
        idx = (s > 0).astype(np.int64) @ (1 << np.arange(8)[::-1])
        hist = np.bincount(idx, minlength=256) / 20000
        print(f"rbm P=6 beta=4 K={k}: TV {0.5 * np.abs(hist - law).sum():.4f}")


if __name__ == "__main__":
    part_one()
    part_two()
    part_three()
