"""Exploratory probe: THRML 0.1.4 categorical (Potts) Gibbs conventions.

EXPLORATION ONLY. Output of this script is not recorded evidence and must not
be cited as a study result. It exists to check, before a protocol is frozen,
that a categorical finite-sweep contract is testable at all: how THRML's
CategoricalEBMFactor/CategoricalGibbsConditional read weights, whether a
three-colour block schedule matches an exact sweep kernel, and which wrong
references separate at what chain count.

Run: JAX_PLATFORMS=cpu uv run python docs/research/potts_contract_probe.py
"""

from __future__ import annotations

import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, sample_states
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models import CategoricalGibbsConditional
from thrml.models.discrete_ebm import CategoricalEBMFactor
from thrml.pgm import CategoricalNode

Q = 3
N_SITES = 6
# 2x3 patch with two parallel diagonals; chromatic number 3.
EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5), (0, 4), (1, 5)]
BLOCKS = [[0, 5], [1, 3], [2, 4]]
BETA = 0.8
CHAINS = 200_000


def parameters(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fields = rng.normal(0.0, 0.8, size=(N_SITES, Q))
    couplings = rng.normal(
        0.0, 0.8, size=(len(EDGES), Q, Q)
    )  # deliberately not symmetric in (c_a, c_b)
    return fields, couplings


STATES = np.array(list(itertools.product(range(Q), repeat=N_SITES)), dtype=np.int64)
POWERS = Q ** np.arange(N_SITES - 1, -1, -1)


def index(states: np.ndarray) -> np.ndarray:
    return states @ POWERS


def log_weight(fields, couplings, beta, states=STATES) -> np.ndarray:
    lw = sum(fields[i, states[:, i]] for i in range(N_SITES))
    for e, (a, b) in enumerate(EDGES):
        lw = lw + couplings[e, states[:, a], states[:, b]]
    return beta * lw


def block_kernel(fields, couplings, beta, block) -> np.ndarray:
    """Exact kernel of one block update: product of single-site softmax conditionals."""
    n = len(STATES)
    kernel = np.zeros((n, n))
    for s_idx, state in enumerate(STATES):
        conditionals = []
        for i in block:
            trial = np.repeat(state[None, :], Q, axis=0)
            trial[:, i] = np.arange(Q)
            lw = log_weight(fields, couplings, beta, trial)
            p = np.exp(lw - lw.max())
            conditionals.append(p / p.sum())
        for values in itertools.product(range(Q), repeat=len(block)):
            new = state.copy()
            new[block] = values
            kernel[s_idx, index(new[None, :])[0]] = np.prod(
                [c[v] for c, v in zip(conditionals, values, strict=True)]
            )
    return kernel


def sweep_kernel(fields, couplings, beta, order) -> np.ndarray:
    t = np.eye(len(STATES))
    for block in order:
        t = t @ block_kernel(fields, couplings, beta, block)
    return t


def tv(p, q) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def multinomial_tolerance(p, chains, quantile=0.999, draws=2000, seed=1) -> float:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(chains, p, size=draws) / chains
    return float(np.quantile(0.5 * np.abs(counts - p).sum(axis=1), quantile))


def thrml_histogram(fields, couplings, beta, order, sweeps, chains, seed) -> np.ndarray:
    nodes = [CategoricalNode() for _ in range(N_SITES)]
    factors = [
        CategoricalEBMFactor([Block(nodes)], jnp.asarray(beta * fields, dtype=jnp.float32)),
        CategoricalEBMFactor(
            [Block([nodes[a] for a, _ in EDGES]), Block([nodes[b] for _, b in EDGES])],
            jnp.asarray(beta * couplings, dtype=jnp.float32),
        ),
    ]
    free = [Block([nodes[i] for i in block]) for block in order]
    spec = BlockGibbsSpec(free, [])
    program = FactorSamplingProgram(
        spec, [CategoricalGibbsConditional(Q) for _ in free], factors, []
    )
    schedule = SamplingSchedule(n_warmup=sweeps, n_samples=1, steps_per_sample=1)

    def one_chain(key):
        _, sample_key = jax.random.split(key)
        state = [jnp.zeros((len(b),), dtype=jnp.uint8) for b in order]
        return sample_states(sample_key, program, schedule, state, [], [Block(nodes)])[0][0]

    keys = jax.random.split(jax.random.key(seed), chains)
    out = np.asarray(jax.jit(jax.vmap(one_chain))(keys), dtype=np.int64)
    return np.bincount(index(out), minlength=len(STATES)) / chains


def single_site_check(chains: int) -> None:
    field = np.array([0.7, -0.4, 0.2])
    node = CategoricalNode()
    factor = CategoricalEBMFactor([Block([node])], jnp.asarray([BETA * field], dtype=jnp.float32))
    spec = BlockGibbsSpec([Block([node])], [])
    program = FactorSamplingProgram(spec, [CategoricalGibbsConditional(Q)], [factor], [])
    schedule = SamplingSchedule(n_warmup=1, n_samples=1, steps_per_sample=1)

    def one(key):
        return sample_states(
            key, program, schedule, [jnp.zeros((1,), jnp.uint8)], [], [Block([node])]
        )[0][0, 0]

    out = np.asarray(
        jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(7), chains)), dtype=np.int64
    )
    hist = np.bincount(out, minlength=Q) / chains

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    print("single site, P(c) after one update:", np.round(hist, 4))
    for name, law in [
        ("softmax(+beta h)", softmax(BETA * field)),
        ("softmax(2 beta h)", softmax(2 * BETA * field)),
        ("softmax(-beta h)", softmax(-BETA * field)),
    ]:
        print(f"  TV to {name:18s} {tv(hist, law):.4f}")


def main() -> None:
    single_site_check(CHAINS)
    fields, couplings = parameters()
    p0 = np.zeros(len(STATES))
    p0[0] = 1.0  # all labels 0
    t0 = time.perf_counter()
    kernels = {
        "forward": sweep_kernel(fields, couplings, BETA, BLOCKS),
        "reversed": sweep_kernel(fields, couplings, BETA, BLOCKS[::-1]),
        "transposed_w": sweep_kernel(fields, couplings.transpose(0, 2, 1), BETA, BLOCKS),
    }
    stationary = np.exp(log_weight(fields, couplings, BETA))
    stationary /= stationary.sum()
    pi_check = tv(stationary @ kernels["forward"], stationary)
    print(f"exact side {time.perf_counter() - t0:.1f}s; stationary invariance TV {pi_check:.2e}")
    for sweeps in (1, 2, 3, 4, 8):
        t0 = time.perf_counter()
        hist = thrml_histogram(fields, couplings, BETA, BLOCKS, sweeps, CHAINS, seed=100 + sweeps)
        seconds = time.perf_counter() - t0
        right = p0 @ np.linalg.matrix_power(kernels["forward"], sweeps)
        tol = multinomial_tolerance(right, CHAINS)
        row = {
            "right": tv(hist, right),
            "K-1": tv(hist, p0 @ np.linalg.matrix_power(kernels["forward"], sweeps - 1)),
            "K+1": tv(hist, p0 @ np.linalg.matrix_power(kernels["forward"], sweeps + 1)),
            "reversed": tv(hist, p0 @ np.linalg.matrix_power(kernels["reversed"], sweeps)),
            "transposed_w": tv(hist, p0 @ np.linalg.matrix_power(kernels["transposed_w"], sweeps)),
            "stationary": tv(hist, stationary),
        }
        cells = "  ".join(f"{k} {v:.4f}" for k, v in row.items())
        print(f"K={sweeps} tol {tol:.4f}  {cells}  ({seconds:.1f}s)")


if __name__ == "__main__":
    main()
