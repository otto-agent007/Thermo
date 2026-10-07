"""Associative memory stage A: can pairwise binary hidden units stand in for a categorical one?

Study-local code. Nothing here is a hash-bound evaluator and nothing here edits
a committed archive. Sampling is THRML 0.1.4 block Gibbs on CPU in float32
(``software_simulation``); references are float64 enumeration of every
completion of the missing bits with the hidden layer summed out exactly
(``exact_reference``). No result here is hardware evidence, and sweeps are
algorithmic counts, not device operations.

Question (frozen in ``docs/experiments/am-binary-emulation.md``). A dense
associative memory whose hidden layer is one categorical unit (log-sum-exp
marginal) recalls far better than Hebbian Hopfield. On pairwise binary units,
how much of that recall do one-hot inhibition, a domain-wall chain or
negative-bias binary units recover, at equilibrium and within K Gibbs sweeps,
and at what cost in units, couplings, coupling range and sequential steps?

One recorded sample is one chain's missing visible bits after exactly K
sweeps. Pattern sets are the replication unit. Parameters are chosen on
development pattern sets and compared on held-out sets. Long runs save each
work unit atomically and resume (experiment-runner autosave contract).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from thrml import Block
from thrml.block_sampling import BlockGibbsSpec, sample_blocks
from thrml.factor import FactorSamplingProgram
from thrml.models import CategoricalGibbsConditional, SpinGibbsConditional
from thrml.models.discrete_ebm import DiscreteEBMFactor, SpinEBMFactor
from thrml.pgm import CategoricalNode, SpinNode

from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.potts_symmetry_tempering import _match as match_numeric
from thermo_lab.provenance import collect_runtime_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
N = 24
PS = (8, 32, 128)
CUES = (12, 8)
BETAS = (4.0, 8.0, 16.0)
GRIDS = {
    "categorical": (None,),
    "onehot": (0.1, 0.4, 1.6),
    "domainwall": (0.1, 0.4, 1.6),
    "bias": (0.0, 0.2, 0.4, 0.6, 0.8),
    "hopfield": (None,),
}
ARMS = tuple(GRIDS)
BUDGETS = (4, 16, 64, 256)
DEV_SEEDS = (7000, 7001, 7002, 7003)
HELD_SEEDS = tuple(range(7100, 7112))
TARGETS = 3
CHAINS = 256
ROOT_SEED = 20261008
RANGE_GRID = (2, 4, 8, 16, 32, 64, 128, None)
MATCH_MARGIN = 0.02
BOOTSTRAP = 2000
PREFIX_CHECK_BUDGET = 64
EXACT_WORKERS = 6


def study_request(chains: int = CHAINS) -> dict:
    return {
        "schema": "am_binary_emulation.request.v1",
        "protocol": "docs/experiments/am-binary-emulation.md",
        "n": N,
        "ps": list(PS),
        "cues": list(CUES),
        "betas": list(BETAS),
        "grids": {arm: list(grid) for arm, grid in GRIDS.items()},
        "budgets": list(BUDGETS),
        "dev_seeds": list(DEV_SEEDS),
        "held_seeds": list(HELD_SEEDS),
        "targets": TARGETS,
        "chains": chains,
        "root_seed": ROOT_SEED,
        "range_grid": [r for r in RANGE_GRID],
        "match_margin": MATCH_MARGIN,
        "bootstrap": BOOTSTRAP,
        "patterns": "default_rng([seed, P]).choice([-1, 1], (P, N)); targets: patterns 0..2",
        "fields": "a_mu = beta xi_mu . sigma / sqrt(N); a_max = beta sqrt(N); penalty = frac a_max",
        "sample": "missing visible bits of one chain after exactly K block sweeps",
        "start": "missing bits uniform; onehot/bias hidden off; chain and label uniform",
        "selection": "per arm, cell and budget on development sets; held-out sets for comparison",
    }


def configs(request: dict) -> list[dict]:
    out = []
    for arm, grid in request["grids"].items():
        for beta in request["betas"]:
            for frac in grid:
                cid = f"{arm}/b{beta:g}" + ("" if frac is None else f"/f{frac:g}")
                out.append({"id": cid, "arm": arm, "beta": beta, "frac": frac})
    return out


def cells(request: dict) -> list[dict]:
    return [{"id": f"P{p}/c{c}", "p": p, "cue": c} for p in request["ps"] for c in request["cues"]]


def patterns(seed: int, p: int, n: int = N) -> np.ndarray:
    return np.random.default_rng([seed, p]).choice([-1.0, 1.0], size=(p, n))


# --- exact side ---------------------------------------------------------------------------


def completions(xi: np.ndarray, target: int, cue: int) -> tuple[np.ndarray, np.ndarray]:
    n = xi.shape[1]
    miss = np.array(list(itertools.product((-1.0, 1.0), repeat=n - cue)))
    full = np.concatenate([np.repeat(xi[target, :cue][None], len(miss), 0), miss], axis=1)
    return miss, full


def log_onehot(a: np.ndarray, lam: float) -> np.ndarray:
    """log sum_k e_k(exp a) exp(-lam (k-1)^2), elementary symmetric polynomials by a scaled DP."""
    shift = a.max(axis=1, keepdims=True)
    w = np.exp(a - shift)
    s, p = w.shape
    e = np.zeros((s, p + 1))
    e[:, 0] = 1.0
    for mu in range(p):
        e[:, 1 : mu + 2] = e[:, 1 : mu + 2] + w[:, mu : mu + 1] * e[:, 0 : mu + 1]
    k = np.arange(p + 1)
    with np.errstate(divide="ignore"):
        terms = np.log(e) + k[None] * shift - lam * (k[None] - 1.0) ** 2
    return logsumexp(terms, axis=1)


def log_domainwall(a: np.ndarray, j: float) -> np.ndarray:
    """Exact sum over chain z_1..z_{P-1} with z_0 = +1, z_P = -1, by a transfer recursion."""
    s, p = a.shape
    const = 0.5 * (a[:, 0] + a[:, p - 1])
    if p == 1:
        return const
    field = 0.5 * (a[:, 1:] - a[:, :-1])
    up = j + field[:, 0]  # log weight with z_1 = +1, including the bond to z_0 = +1
    down = -j - field[:, 0]
    for k in range(1, p - 1):
        new_up = np.logaddexp(up + j, down - j) + field[:, k]
        down = np.logaddexp(up - j, down + j) - field[:, k]
        up = new_up
    return np.logaddexp(up - j, down + j) + const  # bond to z_P = -1


def log_marginal(arm: str, beta: float, frac, overlaps: np.ndarray, n: int) -> np.ndarray:
    a = beta * overlaps / math.sqrt(n)
    a_max = beta * math.sqrt(n)
    if arm == "categorical":
        return logsumexp(a, axis=1)
    if arm == "hopfield":
        return beta / (2 * n) * (overlaps**2).sum(axis=1)
    if arm == "bias":
        return np.logaddexp(0.0, a - frac * a_max).sum(axis=1)
    if arm == "onehot":
        return log_onehot(a, frac * a_max)
    if arm == "domainwall":
        return log_domainwall(a, frac * a_max)
    raise ValueError(arm)


def exact_recall(arm: str, beta: float, frac, xi: np.ndarray, target: int, cue: int) -> float:
    miss, full = completions(xi, target, cue)
    lw = log_marginal(arm, beta, frac, full @ xi.T, xi.shape[1])
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return float(w @ (miss == xi[target, cue:]).mean(axis=1))


# --- spin-form terms (what THRML samples) ---------------------------------------------------


def spin_terms(arm: str, beta: float, frac, xi: np.ndarray) -> dict:
    """Log-weight terms in +-1 spin form: sum J s_i s_j + sum b s_i (+ constant).

    {0,1} hidden units use h = (s + 1) / 2. Keys: j_vv (n x n, upper), j_vh (H x n),
    j_hh (H x H, upper), b_v (n), b_h (H). The categorical arm has no spin form."""
    p, n = xi.shape
    scale = beta / math.sqrt(n)
    a_max = beta * math.sqrt(n)
    z = {"j_vv": np.zeros((n, n)), "b_v": np.zeros(n)}
    if arm == "hopfield":
        c = xi.T @ xi
        z["j_vv"] = np.triu(beta * c / n, 1)
        z.update(j_vh=np.zeros((0, n)), j_hh=np.zeros((0, 0)), b_h=np.zeros(0))
        return z
    if arm in ("bias", "onehot"):
        e = scale * xi  # cross term e_{mu i} h_mu s_i
        c = np.full(p, -frac * a_max) if arm == "bias" else np.full(p, frac * a_max)
        d = np.zeros((p, p)) if arm == "bias" else np.triu(np.full((p, p), -2.0 * frac * a_max), 1)
        z["j_vh"] = e / 2
        z["b_v"] = e.sum(axis=0) / 2
        z["j_hh"] = d / 4
        dsym = d + d.T
        z["b_h"] = c / 2 + dsym.sum(axis=1) / 4
        return z
    if arm == "domainwall":
        j = frac * a_max
        h = p - 1
        z["j_vh"] = scale * (xi[1:] - xi[:-1]) / 2  # chain spin k (1..P-1) row k-1
        z["b_v"] = scale * (xi[0] + xi[p - 1]) / 2
        jhh = np.zeros((h, h))
        for k in range(h - 1):
            jhh[k, k + 1] = j
        z["j_hh"] = jhh
        bh = np.zeros(h)
        if h >= 1:
            bh[0] += j
            bh[h - 1] -= j
        z["b_h"] = bh
        return z
    raise ValueError(arm)


def dynamic_range(arm: str, beta: float, frac, xi: np.ndarray) -> float:
    """Largest |coupling or field| over the smallest nonzero |two-body coupling|, spin form."""
    if arm == "categorical":
        return 1.0
    z = spin_terms(arm, beta, frac, xi)
    two = np.concatenate([z["j_vv"].ravel(), z["j_vh"].ravel(), z["j_hh"].ravel()])
    nonzero = np.abs(two[np.abs(two) > 1e-12])
    allv = np.abs(np.concatenate([two, z["b_v"], z["b_h"]]))
    return float(allv.max() / nonzero.min())


def joint_log_weight(
    arm: str, beta: float, frac, xi: np.ndarray, sigma: np.ndarray, hidden
) -> float:
    """Original-variable log weight of one joint state (tests and preflight)."""
    p, n = xi.shape
    a = beta * (xi @ sigma) / math.sqrt(n)
    a_max = beta * math.sqrt(n)
    if arm == "hopfield":
        return float(beta / (2 * n) * ((xi @ sigma) ** 2).sum())
    if arm == "categorical":
        return float(a[int(hidden)])
    h = np.asarray(hidden, dtype=float)
    if arm == "bias":
        return float(h @ (a - frac * a_max))
    if arm == "onehot":
        return float(h @ a - frac * a_max * (h.sum() - 1.0) ** 2)
    if arm == "domainwall":
        zfull = np.concatenate([[1.0], h, [-1.0]])
        delta = 0.5 * (zfull[:-1] - zfull[1:])
        return float(delta @ a + frac * a_max * (zfull[:-1] * zfull[1:]).sum())
    raise ValueError(arm)


# --- THRML programs ----------------------------------------------------------------------


class Structure:
    """Nodes, blocks and factor layout for one (arm, P, n, cue); weights are swapped per run."""

    def __init__(self, arm: str, p: int, n: int, cue: int):
        self.arm, self.p, self.n, self.cue = arm, p, n, cue
        self.vis = [SpinNode() for _ in range(n)]
        self.missing = Block(self.vis[cue:])
        self.cue_block = Block(self.vis[:cue])
        if arm == "categorical":
            self.hid = [CategoricalNode()]
        elif arm == "hopfield":
            self.hid = []
        else:
            self.hid = [SpinNode() for _ in range(p - 1 if arm == "domainwall" else p)]
        if arm == "hopfield":
            self.free = [Block([v]) for v in self.vis[cue:]]
        elif arm in ("categorical", "bias"):
            self.free = [self.missing, Block(self.hid)]
        elif arm == "onehot":
            self.free = [self.missing] + [Block([h]) for h in self.hid]
        else:
            odd = [self.hid[k] for k in range(0, len(self.hid), 2)]
            even = [self.hid[k] for k in range(1, len(self.hid), 2)]
            self.free = [self.missing] + [Block(b) for b in (odd, even) if b]
        self.spec = BlockGibbsSpec(self.free, [self.cue_block])
        samp = CategoricalGibbsConditional(p) if arm == "categorical" else SpinGibbsConditional()
        self.samplers = [
            samp if (arm == "categorical" and i > 0) else SpinGibbsConditional()
            for i in range(len(self.free))
        ]
        self.vv_pairs = (
            [(i, j) for i in range(n) for j in range(i + 1, n)] if arm == "hopfield" else []
        )
        hcount = len(self.hid) if arm != "categorical" else 0
        self.vh_pairs = [(k, i) for k in range(hcount) for i in range(n)]
        self.hh_pairs = [(k, m) for k in range(hcount) for m in range(k + 1, hcount)]
        self._run = None

    def program(self, beta: float, frac, xi: np.ndarray):
        f32 = jnp.float32
        if self.arm == "categorical":
            factors = [
                DiscreteEBMFactor(
                    [Block([self.vis[i]])],
                    [Block([self.hid[0]])],
                    jnp.asarray(beta * xi[:, i][None] / math.sqrt(self.n), f32),
                )
                for i in range(self.n)
            ]
            return FactorSamplingProgram(self.spec, self.samplers, factors, [])
        z = spin_terms(self.arm, beta, frac, xi)
        factors = []
        if self.vv_pairs:
            w = np.array([z["j_vv"][i, j] for i, j in self.vv_pairs])
            factors.append(
                SpinEBMFactor(
                    [
                        Block([self.vis[i] for i, _ in self.vv_pairs]),
                        Block([self.vis[j] for _, j in self.vv_pairs]),
                    ],
                    jnp.asarray(w, f32),
                )
            )
        if self.vh_pairs:
            w = np.array([z["j_vh"][k, i] for k, i in self.vh_pairs])
            factors.append(
                SpinEBMFactor(
                    [
                        Block([self.hid[k] for k, _ in self.vh_pairs]),
                        Block([self.vis[i] for _, i in self.vh_pairs]),
                    ],
                    jnp.asarray(w, f32),
                )
            )
        if self.hh_pairs:
            w = np.array([z["j_hh"][k, m] for k, m in self.hh_pairs])
            factors.append(
                SpinEBMFactor(
                    [
                        Block([self.hid[k] for k, _ in self.hh_pairs]),
                        Block([self.hid[m] for _, m in self.hh_pairs]),
                    ],
                    jnp.asarray(w, f32),
                )
            )
        factors.append(SpinEBMFactor([Block(self.vis)], jnp.asarray(z["b_v"], f32)))
        if self.hid:
            factors.append(SpinEBMFactor([Block(self.hid)], jnp.asarray(z["b_h"], f32)))
        return FactorSamplingProgram(self.spec, self.samplers, factors, [])

    def init_state(self, key, chains_shape):
        """Free-block initial states for a batch: missing uniform, hidden per the protocol."""
        keys = jax.random.split(key, len(self.free))
        out = []
        for k, block in zip(keys, self.free, strict=True):
            size = len(block.nodes)
            node = block.nodes[0]
            if isinstance(node, CategoricalNode):
                out.append(
                    jax.random.randint(k, (*chains_shape, size), 0, self.p).astype(jnp.uint8)
                )
            elif node in self.hid and self.arm in ("onehot", "bias"):
                out.append(jnp.zeros((*chains_shape, size), dtype=jnp.bool_))
            else:
                out.append(jax.random.bernoulli(k, 0.5, (*chains_shape, size)))
        return out

    def runner(self):
        """jit-compiled (program, keys, init, clamp) -> per-sweep free states; compiled once."""
        if self._run is None:

            def one(program, sweep_keys, init, clamp):
                sampler_states = [s.init() for s in program.samplers]

                def step(state, key):
                    state, _ = sample_blocks(key, state, [clamp], program, sampler_states)
                    return state, state

                _, states = jax.lax.scan(step, init, sweep_keys)
                return states

            batched = eqx.filter_vmap(one, in_axes=(None, 0, 0, 0))
            self._run = eqx.filter_jit(batched)
        return self._run


_STRUCTURES: dict = {}


def structure(arm: str, p: int, n: int, cue: int) -> Structure:
    key = (arm, p, n, cue)
    if key not in _STRUCTURES:
        _STRUCTURES[key] = Structure(arm, p, n, cue)
    return _STRUCTURES[key]


def sample(arm, beta, frac, xi, cue, sweeps, chains, key, targets=TARGETS):
    """Run `chains` chains per target for `sweeps` sweeps; return per-sweep free-block states."""
    st = structure(arm, xi.shape[0], xi.shape[1], cue)
    program = st.program(beta, frac, xi)
    total = targets * chains
    k_init, k_run = jax.random.split(key)
    init = st.init_state(k_init, (total,))
    chain_keys = jax.random.split(k_run, total)
    sweep_keys = jax.vmap(lambda k: jax.random.split(k, sweeps))(chain_keys)
    clamp = jnp.asarray(np.repeat(xi[:targets, :cue] > 0, chains, axis=0))
    states = st.runner()(program, sweep_keys, init, clamp)
    return [np.asarray(s) for s in states]


def summarize(
    arm: str, states: list, xi: np.ndarray, cue: int, budgets, chains: int, targets=TARGETS
) -> dict:
    """Recall counts per target and budget plus diagnostics, from per-sweep free-block states."""
    st = structure(arm, xi.shape[0], xi.shape[1], cue)
    if arm == "hopfield":
        missing = np.concatenate([s for s in states], axis=-1)  # each block one spin
    else:
        missing = states[0]
    truth = np.repeat(xi[:targets, cue:] > 0, chains, axis=0)  # (chains_total, m)
    out = {"correct": [], "bits": int(chains * (xi.shape[1] - cue))}
    for k in budgets:
        right = (missing[:, k - 1, :] == truth).sum(axis=1).reshape(targets, chains).sum(axis=1)
        out["correct"].append(right.astype(int).tolist())
    if arm == "onehot":
        hidden = np.concatenate(states[1:], axis=-1)  # (chains, sweeps, P)
        on = hidden.sum(axis=-1)
        label = np.where(on == 1, hidden.argmax(axis=-1), -1)
        switches = (label[:, 1:] != label[:, :-1]).cumsum(axis=1)
        out["onehot_valid"] = [int((on[:, k - 1] == 1).sum()) for k in budgets]
        out["label_switches"] = [int(switches[:, k - 2].sum()) if k >= 2 else 0 for k in budgets]
    if arm == "domainwall" and len(st.hid) > 0:
        z = np.zeros((missing.shape[0], missing.shape[1], len(st.hid)), dtype=bool)
        for block_state, block in zip(states[1:], st.free[1:], strict=True):
            idx = [st.hid.index(node) for node in block.nodes]
            z[:, :, idx] = block_state
        zs = np.where(z, 1, -1)
        full = np.concatenate(
            [np.ones(zs.shape[:2] + (1,), int), zs, -np.ones(zs.shape[:2] + (1,), int)], axis=-1
        )
        walls = (full[..., 1:] != full[..., :-1]).sum(axis=-1)
        out["walls"] = [int(walls[:, k - 1].sum()) for k in budgets]
    return out


# --- preflight: one THRML sweep against the exact one-sweep law ------------------------


PREFLIGHT = {
    "n": 6,
    "p": 3,
    "cue": 2,
    "beta": 2.0,
    "frac": 0.4,
    "chains": 100_000,
    "sweeps": (1, 2),
    "seed": 6,
}


def free_layout(arm: str, st: Structure) -> list[tuple]:
    """(block index, position) of each free variable: missing bits first, then hidden units."""
    order = []
    for node in list(st.missing.nodes) + list(st.hid):
        for bi, block in enumerate(st.free):
            if node in block.nodes:
                order.append((bi, block.nodes.index(node)))
    return order


def exact_sweep_law(
    arm: str, beta: float, frac, xi: np.ndarray, cue: int, sweeps: int, start: np.ndarray
) -> tuple:
    """Exact law of the free state after `sweeps` sweeps from a point mass, by brute force."""
    st = structure(arm, xi.shape[0], xi.shape[1], cue)
    m = xi.shape[1] - cue
    hvals = [range(xi.shape[0])] if arm == "categorical" else [(-1, 1)] * len(st.hid)
    space = [list(v) for v in itertools.product(*([(-1, 1)] * m + hvals))]
    index = {tuple(s): i for i, s in enumerate(space)}
    layout = free_layout(arm, st)

    def logw(state):
        sigma = np.concatenate([xi[0, :cue], np.array(state[:m], float)])
        hid = state[m:]
        if arm == "categorical":
            hidden = hid[0]
        elif arm in ("onehot", "bias"):
            hidden = (np.array(hid, float) + 1) / 2
        else:
            hidden = np.array(hid, float)
        return (
            joint_log_weight(arm, beta, frac, xi, sigma, hidden)
            if arm != "hopfield"
            else joint_log_weight(arm, beta, frac, xi, sigma, None)
        )

    lw = np.array([logw(s) for s in space])
    t = np.eye(len(space))
    for bi, _block in enumerate(st.free):
        pos = [v for v, (b, _) in enumerate(layout) if b == bi]
        k = np.zeros((len(space), len(space)))
        for x, s in enumerate(space):
            opts = []
            for vals in itertools.product(
                *[
                    (range(xi.shape[0]) if (arm == "categorical" and v >= m) else (-1, 1))
                    for v in pos
                ]
            ):
                y = list(s)
                for v, val in zip(pos, vals, strict=True):
                    y[v] = val
                opts.append(index[tuple(y)])
            w = np.exp(lw[opts] - lw[opts].max())
            k[x, opts] += w / w.sum()
        t = t @ k
    p0 = np.zeros(len(space))
    p0[index[tuple(start)]] = 1.0
    return space, p0 @ np.linalg.matrix_power(t, sweeps)


def preflight() -> dict:
    spec = PREFLIGHT
    xi = patterns(ROOT_SEED, spec["p"], spec["n"])
    out = {"spec": spec, "arms": {}}
    for arm in ARMS:
        frac = None if GRIDS[arm][0] is None else spec["frac"]
        st = structure(arm, spec["p"], spec["n"], spec["cue"])
        m = spec["n"] - spec["cue"]
        if arm == "categorical":
            start_hidden = [0]
        elif arm == "domainwall":
            start_hidden = [-1] * len(st.hid)
        else:
            start_hidden = [-1] * len(st.hid)
        start = [-1] * m + start_hidden
        res = {}
        for sweeps in spec["sweeps"]:
            space, law = exact_sweep_law(arm, spec["beta"], frac, xi, spec["cue"], sweeps, start)
            program = st.program(spec["beta"], frac, xi)
            layout = free_layout(arm, st)
            chains = spec["chains"]
            init = []
            for block in st.free:
                vals = []
                for node in block.nodes:
                    v = (
                        start[list(st.missing.nodes).index(node)]
                        if node in st.missing.nodes
                        else start[m + st.hid.index(node)]
                    )
                    vals.append(v)
                arr = np.array(vals)
                if isinstance(block.nodes[0], CategoricalNode):
                    init.append(jnp.asarray(np.repeat(arr[None], chains, 0), jnp.uint8))
                else:
                    init.append(jnp.asarray(np.repeat((arr > 0)[None], chains, 0)))
            key = jax.random.key(spec["seed"] * 100 + sweeps)
            sweep_keys = jax.vmap(lambda k, s=sweeps: jax.random.split(k, s))(
                jax.random.split(key, chains)
            )
            clamp = jnp.asarray(np.repeat((xi[0, : spec["cue"]] > 0)[None], chains, 0))
            states = st.runner()(program, sweep_keys, init, clamp)
            final = [np.asarray(s)[:, -1] for s in states]
            cols = []
            for bi, posn in layout:
                col = final[bi][:, posn]
                cols.append(
                    col.astype(int) if arm == "categorical" and bi > 0 else np.where(col, 1, -1)
                )
            obs = np.stack(cols, axis=1)
            index = {tuple(s): i for i, s in enumerate(space)}
            hist = (
                np.bincount([index[tuple(r)] for r in obs.tolist()], minlength=len(space)) / chains
            )
            tv = 0.5 * float(np.abs(hist - law).sum())
            draws = np.random.default_rng(spec["seed"]).multinomial(chains, law, size=1000) / chains
            tol = float(np.quantile(0.5 * np.abs(draws - law).sum(axis=1), 0.999))
            res[f"K{sweeps}"] = {
                "tv": tv,
                "tolerance": tol,
                "pass": tv <= tol,
                "states": len(space),
            }
        out["arms"][arm] = res
    out["passed"] = all(r["pass"] for a in out["arms"].values() for r in a.values())
    return out


# --- work units, autosave and resume ---------------------------------------------------------


def unit_id(phase: str, config: dict, cell: dict, seed: int) -> str:
    return f"{phase}/{config['id']}/{cell['id']}/s{seed}"


def run_unit(request: dict, phase: str, config: dict, cell: dict, seed: int, sampled: bool) -> dict:
    xi = patterns(seed, cell["p"])
    arm, beta, frac = config["arm"], config["beta"], config["frac"]
    out = {
        "exact": [
            exact_recall(arm, beta, frac, xi, t, cell["cue"]) for t in range(request["targets"])
        ],
        "range": dynamic_range(arm, beta, frac, xi),
    }
    if sampled:
        key = jax.random.fold_in(
            jax.random.key(request["root_seed"]), _stable_int(unit_id(phase, config, cell, seed))
        )
        states = sample(
            arm,
            beta,
            frac,
            xi,
            cell["cue"],
            max(request["budgets"]),
            request["chains"],
            key,
            request["targets"],
        )
        out["sampled"] = summarize(
            arm, states, xi, cell["cue"], request["budgets"], request["chains"], request["targets"]
        )
    return out


def parse_unit(uid: str, by_id: dict, cell_by_id: dict) -> tuple[dict, dict, int]:
    """Inverse of unit_id: phase/<config id>/<cell id>/s<seed>."""
    phase = uid.split("/", 1)[0]
    cell_id = next(cid for cid in cell_by_id if f"/{cid}/s" in uid)
    cfg_id = uid[len(phase) + 1 : uid.index(f"/{cell_id}/s")]
    return by_id[cfg_id], cell_by_id[cell_id], int(uid.rsplit("/s", 1)[1])


def heldexact_id(config: dict, cell: dict, seed: int) -> str:
    return "heldexact/" + unit_id("held", config, cell, seed).split("/", 1)[1]


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def source_hash() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


class UnitStore:
    """One atomically written file per completed unit, guarded by request digest and source hash."""

    def __init__(self, root: Path, request: dict):
        self.root = root / "units"
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard = {"request_digest": canonical_sha256(request), "source_sha256": source_hash()}
        guard_path = root / "checkpoint-guard.json"
        if guard_path.exists():
            stored = json.loads(guard_path.read_text())
            if stored != self.guard:
                raise SystemExit(
                    "checkpoint guard mismatch: request or runner changed; use a fresh directory"
                )
        else:
            atomic_write_text(guard_path, canonical_json(self.guard) + "\n")

    def path(self, uid: str) -> Path:
        return self.root / (uid.replace("/", "__") + ".json.gz")

    def get(self, uid: str):
        p = self.path(uid)
        if not p.exists():
            return None
        return json.loads(gzip.decompress(p.read_bytes()))

    def put(self, uid: str, value: dict) -> None:
        tmp = self.path(uid).with_suffix(".tmp")
        tmp.write_bytes(
            gzip.compress(
                canonical_json({"uid": uid, **self.guard, "value": value}).encode(), mtime=0
            )
        )
        tmp.replace(self.path(uid))


# --- selection and evaluation -----------------------------------------------------------------


def unit_recall(value: dict, k_index: int) -> float:
    s = value["sampled"]
    return float(np.sum(s["correct"][k_index]) / (len(s["correct"][k_index]) * s["bits"]))


def select(request: dict, dev: dict, range_limit=None) -> dict:
    """Per arm, cell and budget (and 'eq'), the development-best configuration id."""
    chosen = {}
    for cell in cells(request):
        for arm in ARMS:
            cands = [c for c in configs(request) if c["arm"] == arm]
            if range_limit is not None:
                cands = [
                    c
                    for c in cands
                    if np.mean(
                        [dev[unit_id("dev", c, cell, s)]["range"] for s in request["dev_seeds"]]
                    )
                    <= range_limit
                ]
            if not cands:
                continue
            eq = max(
                cands,
                key=lambda c: (
                    np.mean(
                        [
                            np.mean(dev[unit_id("dev", c, cell, s)]["exact"])
                            for s in request["dev_seeds"]
                        ]
                    ),
                    c["id"],
                ),
            )
            chosen[f"{cell['id']}/{arm}/eq"] = eq["id"]
            for ki, k in enumerate(request["budgets"]):
                if "sampled" not in dev[unit_id("dev", cands[0], cell, request["dev_seeds"][0])]:
                    continue
                best = max(
                    cands,
                    key=lambda c: (
                        np.mean(
                            [
                                unit_recall(dev[unit_id("dev", c, cell, s)], ki)
                                for s in request["dev_seeds"]
                            ]
                        ),
                        c["id"],
                    ),
                )
                chosen[f"{cell['id']}/{arm}/K{k}"] = best["id"]
    return chosen


def bootstrap_ci(diffs: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(draws, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(request: dict, dev: dict, held: dict, held_exact: dict) -> dict:
    by_id = {c["id"]: c for c in configs(request)}
    chosen = select(request, dev)
    results = {
        "chosen": chosen,
        "equilibrium": {},
        "budget": {},
        "cost": {},
        "curve_eq": {},
        "curve_budget": {},
        "diagnostics": {},
    }
    for cell in cells(request):
        cid = cell["id"]
        for arm in ARMS:
            eq_cfg = by_id[chosen[f"{cid}/{arm}/eq"]]
            vals = [
                np.mean(held_exact[unit_id("held", eq_cfg, cell, s)]["exact"])
                for s in request["held_seeds"]
            ]
            results["equilibrium"][f"{cid}/{arm}"] = {
                "config": eq_cfg["id"],
                "recall": float(np.mean(vals)),
                "per_set": [float(v) for v in vals],
            }
        for ki, k in enumerate(request["budgets"]):
            ref_cfg = by_id[chosen[f"{cid}/categorical/K{k}"]]
            ref = np.array(
                [
                    unit_recall(held[unit_id("held", ref_cfg, cell, s)], ki)
                    for s in request["held_seeds"]
                ]
            )
            for arm in ARMS:
                cfg = by_id[chosen[f"{cid}/{arm}/K{k}"]]
                vals = np.array(
                    [
                        unit_recall(held[unit_id("held", cfg, cell, s)], ki)
                        for s in request["held_seeds"]
                    ]
                )
                lo, hi = bootstrap_ci(
                    vals - ref, request["bootstrap"], _stable_int(f"{cid}/{arm}/K{k}")
                )
                margin = request["match_margin"]
                verdict = (
                    "reference"
                    if arm == "categorical"
                    else (
                        "matches"
                        if lo >= -margin and hi <= margin
                        else (
                            "falls_short"
                            if hi < -margin
                            else ("exceeds" if lo > margin else "inconclusive")
                        )
                    )
                )
                results["budget"][f"{cid}/{arm}/K{k}"] = {
                    "config": cfg["id"],
                    "recall": float(vals.mean()),
                    "diff_vs_categorical": float((vals - ref).mean()),
                    "ci": [lo, hi],
                    "verdict": verdict,
                    "per_set": vals.tolist(),
                }
                units = [
                    held[unit_id("held", cfg, cell, s)]["sampled"] for s in request["held_seeds"]
                ]
                diag = {}
                if arm == "onehot":
                    tot = len(request["held_seeds"]) * request["targets"] * request["chains"]
                    diag["onehot_valid_fraction"] = float(
                        sum(u["onehot_valid"][ki] for u in units) / tot
                    )
                    diag["label_switches_per_chain"] = float(
                        sum(u["label_switches"][ki] for u in units) / tot
                    )
                if arm == "domainwall":
                    tot = len(request["held_seeds"]) * request["targets"] * request["chains"]
                    diag["walls_per_chain"] = float(sum(u["walls"][ki] for u in units) / tot)
                if diag:
                    results["diagnostics"][f"{cid}/{arm}/K{k}"] = diag
        for arm in ARMS:
            cfg = by_id[chosen[f"{cid}/{arm}/K{request['budgets'][-1]}"]]
            st = structure(arm, cell["p"], request["n"], cell["cue"])
            hidden = 1 if arm == "categorical" else len(st.hid)
            couplings = (
                len(st.vv_pairs)
                + len(st.vh_pairs)
                + len(st.hh_pairs)
                + (request["n"] * cell["p"] if arm == "categorical" else 0)
            )
            results["cost"][f"{cid}/{arm}"] = {
                "config": cfg["id"],
                "hidden_units": hidden,
                "couplings": couplings,
                "dynamic_range": float(
                    np.mean(
                        [
                            held_exact[unit_id("held", cfg, cell, s)]["range"]
                            for s in request["held_seeds"]
                        ]
                    )
                ),
                "blocks_per_sweep": len(st.free),
                "updates_per_sweep": (request["n"] - cell["cue"]) + hidden,
            }
    for r in request["range_grid"]:
        chosen_r = select(request, dev, r)
        tag = "inf" if r is None else str(r)
        for cell in cells(request):
            for arm in ARMS:
                key = f"{cell['id']}/{arm}/eq"
                if key not in chosen_r:
                    results["curve_eq"][f"R{tag}/{cell['id']}/{arm}"] = None
                    continue
                cfg = by_id[chosen_r[key]]
                results["curve_eq"][f"R{tag}/{cell['id']}/{arm}"] = float(
                    np.mean(
                        [
                            np.mean(held_exact[unit_id("held", cfg, cell, s)]["exact"])
                            for s in request["held_seeds"]
                        ]
                    )
                )
                for ki, k in enumerate(request["budgets"]):
                    kk = f"{cell['id']}/{arm}/K{k}"
                    if kk in chosen_r:
                        cfgk = by_id[chosen_r[kk]]
                        results["curve_budget"][f"R{tag}/{kk}"] = float(
                            np.mean(
                                [
                                    unit_recall(dev[unit_id("dev", cfgk, cell, s)], ki)
                                    for s in request["dev_seeds"]
                                ]
                            )
                        )
    return results


# --- study driver ------------------------------------------------------------------------------


def plan(request: dict) -> list[tuple]:
    """Development units (sampled) and held-out exact-only units, for every configuration."""
    units = []
    for cell in cells(request):
        for cfg in configs(request):
            for s in request["dev_seeds"]:
                units.append(("dev", cfg, cell, s, True))
            for s in request["held_seeds"]:
                units.append(("heldexact", cfg, cell, s, False))
    return units


def run_study(
    output_dir: str | Path, request: dict | None = None, stop_after: int | None = None
) -> dict:
    request = request or study_request()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    store = UnitStore(root, request)
    log_path = root / "run.log"

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} AM-A {msg}\n")
            fh.flush()

    def status(phase: str, last: str | None) -> None:
        atomic_write_text(
            root / "run-status.json",
            canonical_json({"phase": phase, "last_unit": last, "time": time.time()}) + "\n",
        )

    started = time.monotonic()
    pre_path = root / "preflight.json"
    if pre_path.exists():
        check = json.loads(pre_path.read_text())
    else:
        log("preflight")
        check = preflight()
        atomic_write_text(pre_path, canonical_json(check) + "\n")
    if not check["passed"]:
        log("stopped: preflight failed")
        raise SystemExit("preflight failed; see preflight.json")
    done = 0

    def do(phase, cfg, cell, seed, sampled):
        nonlocal done
        uid = unit_id("dev" if phase == "dev" else "held", cfg, cell, seed)
        uid = uid if phase != "heldexact" else "heldexact/" + uid.split("/", 1)[1]
        if store.get(uid) is None:
            t0 = time.monotonic()
            store.put(
                uid,
                run_unit(request, "dev" if phase == "dev" else "held", cfg, cell, seed, sampled),
            )
            log(f"{uid} {time.monotonic() - t0:.1f}s")
            done += 1
            status(phase, uid)
            if stop_after is not None and done >= stop_after:
                raise KeyboardInterrupt("stop_after reached (test interruption)")
        return store.get(uid)["value"]

    pending = [
        (cfg, cell, s)
        for phase, cfg, cell, s, _ in plan(request)
        if phase == "heldexact" and store.get(heldexact_id(cfg, cell, s)) is None
    ]
    if pending:
        log(f"exact-only held-out units: {len(pending)} in a process pool")
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(EXACT_WORKERS, mp_context=ctx) as pool:
            futures = {
                pool.submit(run_unit, request, "held", c, cl, s, False): (c, cl, s)
                for c, cl, s in pending
            }
            for fut in concurrent.futures.as_completed(futures):
                c, cl, s = futures[fut]
                store.put(heldexact_id(c, cl, s), fut.result())
                done += 1
                status("heldexact", heldexact_id(c, cl, s))
                if stop_after is not None and done >= stop_after:
                    pool.shutdown(cancel_futures=True)
                    raise KeyboardInterrupt("stop_after reached (test interruption)")
        log("exact-only held-out units done")
    dev, held_exact = {}, {}
    for phase, cfg, cell, seed, sampled in plan(request):
        value = do(phase, cfg, cell, seed, sampled)
        (dev if phase == "dev" else held_exact)[
            unit_id("dev" if phase == "dev" else "held", cfg, cell, seed)
        ] = value
    chosen = select(request, dev)
    by_id = {c["id"]: c for c in configs(request)}
    needed = sorted(
        {
            (cell["id"], v)
            for k, v in chosen.items()
            for cell in cells(request)
            if k.startswith(cell["id"] + "/") and not k.endswith("/eq")
        }
    )
    held = {}
    cell_by_id = {c["id"]: c for c in cells(request)}
    for cell_id, cfg_id in needed:
        cell, cfg = cell_by_id[cell_id], by_id[cfg_id]
        for s in request["held_seeds"]:
            held[unit_id("held", cfg, cell, s)] = do("held", cfg, cell, s, True)
    # prefix check: re-run one held unit per arm at a shorter budget and compare its sampled recall
    prefix_ok = prefix_check(request, held, by_id, cell_by_id)
    evaluation = evaluate(request, dev, held, held_exact)
    record = {
        "schema": "am_binary_emulation.record.v1",
        "evidence": {
            "exact_references": "exact_reference (float64 enumeration, hidden layers exact)",
            "sampled_cells": "software_simulation (THRML 0.1.4, CPU, float32); not hardware",
        },
        "request": request,
        "request_digest": canonical_sha256(request),
        "source_sha256": source_hash(),
        "preflight": check,
        "dev": dev,
        "held": held,
        "held_exact": held_exact,
        "prefix_ok": prefix_ok,
        "evaluation": evaluation,
    }
    record["result_digest"] = canonical_sha256(
        {"dev": dev, "held": held, "held_exact": held_exact, "evaluation": evaluation}
    )
    archive_bytes = gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    (root / "study.json.gz").write_bytes(archive_bytes)
    log("archive written; replaying")
    persisted = json.loads(gzip.decompress((root / "study.json.gz").read_bytes()))
    replay(persisted, request, full=True)
    provenance = {
        "schema": "am_binary_emulation.provenance.v1",
        "request_digest": persisted["request_digest"],
        "result_digest": persisted["result_digest"],
        "runtime": json.loads(canonical_json(collect_runtime_provenance(REPO_ROOT).model_dump())),
        "total_seconds": time.monotonic() - started,
    }
    atomic_write_text(root / "summary.md", render_report(persisted))
    atomic_write_text(root / "provenance.json", canonical_json(provenance) + "\n")
    atomic_write_text(
        root / "completion.json",
        canonical_json(completion(persisted, provenance, archive_bytes)) + "\n",
    )
    status("complete", None)
    log(f"completion published after {time.monotonic() - started:.1f}s")
    return persisted


def prefix_check(request, held, by_id, cell_by_id) -> bool:
    """Re-run one held unit per arm with a shorter sweep count; its recall must equal the prefix."""
    k_short = min(PREFIX_CHECK_BUDGET, max(request["budgets"]))
    budgets_short = [k for k in request["budgets"] if k <= k_short]
    ok = True
    for arm in ARMS:
        uid = next(u for u in sorted(held) if u.split("/")[1] == arm)
        cfg, cell, seed = parse_unit(uid, by_id, cell_by_id)
        xi = patterns(seed, cell["p"])
        key = jax.random.fold_in(
            jax.random.key(request["root_seed"]), _stable_int(unit_id("held", cfg, cell, seed))
        )
        # same per-sweep keys as the full run, truncated
        st = structure(cfg["arm"], cell["p"], request["n"], cell["cue"])
        program = st.program(cfg["beta"], cfg["frac"], xi)
        total = request["targets"] * request["chains"]
        k_init, k_run = jax.random.split(key)
        init = st.init_state(k_init, (total,))
        chain_keys = jax.random.split(k_run, total)
        sweep_keys = jax.vmap(lambda k: jax.random.split(k, max(request["budgets"]))[:k_short])(
            chain_keys
        )
        clamp = jnp.asarray(
            np.repeat(xi[: request["targets"], : cell["cue"]] > 0, request["chains"], axis=0)
        )
        states = [np.asarray(s) for s in st.runner()(program, sweep_keys, init, clamp)]
        short = summarize(
            cfg["arm"],
            states,
            xi,
            cell["cue"],
            budgets_short,
            request["chains"],
            request["targets"],
        )
        full = held[uid]["sampled"]
        ok &= all(short["correct"][i] == full["correct"][i] for i in range(len(budgets_short)))
    return bool(ok)


# --- replay, report, completion ------------------------------------------------------------------


def replay(record: dict, request: dict, full: bool = False) -> None:
    """Recompute evaluation from archived units and compare numerically; `full` also recomputes
    every exact reference and range (slow), otherwise a deterministic subset (cue 12 units)."""
    if record["request"] != request or record["request_digest"] != canonical_sha256(request):
        raise ValueError("archived request does not match this code's request")
    if not record["preflight"]["passed"] or not record["prefix_ok"]:
        raise ValueError("preflight or prefix check did not pass")
    by_id = {c["id"]: c for c in configs(request)}
    cell_by_id = {c["id"]: c for c in cells(request)}
    for table in ("dev", "held_exact"):
        for uid, value in record[table].items():
            cfg, cell, seed = parse_unit(uid, by_id, cell_by_id)
            if not full and cell["cue"] != max(request["cues"]):
                continue
            xi = patterns(seed, cell["p"])
            exact = [
                exact_recall(cfg["arm"], cfg["beta"], cfg["frac"], xi, t, cell["cue"])
                for t in range(request["targets"])
            ]
            match_numeric(exact, value["exact"], f"{uid} exact")
            match_numeric(
                dynamic_range(cfg["arm"], cfg["beta"], cfg["frac"], xi),
                value["range"],
                f"{uid} range",
            )
    evaluation = evaluate(request, record["dev"], record["held"], record["held_exact"])
    match_numeric(evaluation, record["evaluation"], "evaluation")
    digest = canonical_sha256(
        {
            "dev": record["dev"],
            "held": record["held"],
            "held_exact": record["held_exact"],
            "evaluation": record["evaluation"],
        }
    )
    if digest != record["result_digest"]:
        raise ValueError("result digest does not replay")


def render_report(record: dict) -> str:
    req, ev = record["request"], record["evaluation"]
    lines = [
        "# Associative memory stage A: binary emulation of a categorical hidden unit",
        "",
        "Exact references: float64 enumeration, hidden layers summed exactly (`exact_reference`).",
        "Sampled cells: THRML 0.1.4 on CPU, float32 (`software_simulation`). Sweeps are",
        "algorithmic counts. No hardware claim.",
        "",
        f"Request digest `{record['request_digest']}`; result digest `{record['result_digest']}`.",
        "",
        "## Held-out recall after K sweeps (difference vs categorical, 95% bootstrap interval)",
        "",
    ]
    for cell in cells(req):
        lines += [
            f"### {cell['id']}",
            "",
            "| arm | " + " | ".join(f"K={k}" for k in req["budgets"]) + " | equilibrium (exact) |",
            "| --- |" + " --- |" * (len(req["budgets"]) + 1),
        ]
        for arm in ARMS:
            row = []
            for k in req["budgets"]:
                r = ev["budget"][f"{cell['id']}/{arm}/K{k}"]
                if arm == "categorical":
                    row.append(f"{r['recall']:.3f}")
                else:
                    row.append(
                        f"{r['recall']:.3f} ({r['diff_vs_categorical']:+.3f} "
                        f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] {r['verdict']})"
                    )
            eq = ev["equilibrium"][f"{cell['id']}/{arm}"]
            lines.append(
                f"| {arm} | " + " | ".join(row) + f" | {eq['recall']:.3f} ({eq['config']}) |"
            )
        lines.append("")
    lines += [
        "## Cost of the configuration selected at the largest budget",
        "",
        "| cell | arm | config | hidden units | couplings | dynamic range "
        "| blocks per sweep | updates per sweep |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cells(req):
        for arm in ARMS:
            c = ev["cost"][f"{cell['id']}/{arm}"]
            lines.append(
                f"| {cell['id']} | {arm} | {c['config']} | {c['hidden_units']} | {c['couplings']} "
                f"| {c['dynamic_range']:.1f} | {c['blocks_per_sweep']} | {c['updates_per_sweep']} |"
            )
    lines += [
        "",
        "## Range-sensitivity curve (held-out exact recall, best development config with D <= R)",
        "",
    ]
    tags = ["inf" if r is None else str(r) for r in req["range_grid"]]
    for cell in cells(req):
        lines += [
            f"### {cell['id']}",
            "",
            "| arm | " + " | ".join(f"R={t}" for t in tags) + " |",
            "| --- |" + " --- |" * len(tags),
        ]
        for arm in ARMS:
            vals = [ev["curve_eq"].get(f"R{t}/{cell['id']}/{arm}") for t in tags]
            lines.append(
                f"| {arm} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def completion(record: dict, provenance: dict, archive_bytes: bytes) -> dict:
    ev = record["evaluation"]
    verdicts: dict[str, int] = {}
    for key, r in ev["budget"].items():
        arm = key.split("/")[2]
        if arm != "categorical":
            verdicts[f"{arm}/{r['verdict']}"] = verdicts.get(f"{arm}/{r['verdict']}", 0) + 1
    return {
        "status": "am_binary_emulation_complete",
        "cells": len(cells(record["request"])),
        "dev_units": len(record["dev"]),
        "held_units": len(record["held"]),
        "held_exact_units": len(record["held_exact"]),
        "preflight_passed": record["preflight"]["passed"],
        "prefix_checks_passed": record["prefix_ok"],
        "verdict_counts": dict(sorted(verdicts.items())),
        "replayed": True,
        "integrity": True,
        "autosave": "per work unit under units/, guarded by request digest and runner source hash",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def replay_archive(archive: str | Path, output_dir: str | Path, full: bool = False) -> dict:
    archive_bytes = Path(archive).read_bytes()
    record = json.loads(gzip.decompress(archive_bytes))
    replay(record, study_request(record["request"]["chains"]), full=full)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": "am_binary_emulation.provenance.v1",
        "mode": "replay_only",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
    }
    atomic_write_text(dest / "summary.md", render_report(record))
    atomic_write_text(
        dest / "completion.json",
        canonical_json(completion(record, provenance, archive_bytes)) + "\n",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay", help="replay this study.json.gz")
    parser.add_argument(
        "--full", action="store_true", help="with --replay: recompute every exact reference"
    )
    args = parser.parse_args()
    if args.replay:
        replay_archive(args.replay, args.output_dir, full=args.full)
    else:
        run_study(args.output_dir)


if __name__ == "__main__":
    main()
