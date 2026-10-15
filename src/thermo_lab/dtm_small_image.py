"""DTM small-image denoiser under the constraint layer (proposal P-0006).

Protocol: ``docs/experiments/dtm-small-image.md`` (approved at commit 711ed4b,
merged by PR #111). A chained denoising model with T = 1 or 2 steps is trained
in software (float64 NumPy, contrastive, Adam) on a four-component mixture of
nearest-neighbour Ising grids (L = 8, J = 0.55, h0 = 0.2, flip noise q = 0.2).
Each layer sits on the Z1 16-offset lattice: pixel, clamp, latent A and latent
B per 2 x 2 cell. Each trained model is quantized under M5b's codebook (split
cap) at 3, 4, 6 and 8 bits, with 0 or 10 percent static beta jitter, and run
through THRML (float64, CPU) under the host-per-step I/O path with
continuation: each layer region carries its free-node state from one image of
a stream to the next.

The data model, layout, trainer, NumPy sampler and logistic baseline are the
probe's (``docs/research/dtm_small_image_probe.py`` and ``..._r1.py``), copied
here so the import package does not depend on ``docs/``; the unit tests check
them bitwise against the probe. THRML lowering, quantization, jitter, placement
and pricing are imported from ``thermo_lab.constraints`` unchanged.

Long run (about 1.5 to 2 h): every work unit (exact reference, trained model,
logistic baseline, inference cell) is written atomically to ``units/`` with a
digest in ``units/index.json``; ``--resume`` re-authenticates the request and
sources and runs only the missing units.

Evidence classes: the mixture likelihood, exact samples, posterior marginals
and Bayes MPM error are ``exact_reference`` (float64). THRML and NumPy
denoiser runs and the logistic baseline are ``software_simulation`` (CPU).
Joules are ``calibrated_projection`` of algorithmic counts under the sealed
Z1 profile. Nothing here ran on hardware.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
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
from scipy.special import expit, logsumexp
from thrml import SamplingSchedule
from thrml import sample_states as thrml_sample_states

from thermo_lab.constraints import (
    ConstraintRequest,
    PairwiseModel,
    check,
    count,
    price,
    quantize,
    require_placement,
)
from thermo_lab.constraints.jitter import request_gains
from thermo_lab.constraints.lattice import NEIGHBOURS
from thermo_lab.constraints.thrml_lowering import colour_blocks, lower
from thermo_lab.hardware.z1 import Z1HardwareProfile
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.planar_ising_scaling import check_equal
from thermo_lab.provenance import collect_runtime_provenance

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/thermo_lab/dtm_small_image.py",
    "src/thermo_lab/constraints/__init__.py",
    "src/thermo_lab/constraints/accounting.py",
    "src/thermo_lab/constraints/gibbs.py",
    "src/thermo_lab/constraints/jitter.py",
    "src/thermo_lab/constraints/lattice.py",
    "src/thermo_lab/constraints/model.py",
    "src/thermo_lab/constraints/quantize.py",
    "src/thermo_lab/constraints/request.py",
    "src/thermo_lab/constraints/thrml_lowering.py",
    "src/thermo_lab/am_coupling_bits.py",
    "src/thermo_lab/meta_ebm_thermalization_core.py",
    "src/thermo_lab/hardware/z1.py",
    "src/thermo_lab/hashing.py",
    "src/thermo_lab/persistence.py",
    "src/thermo_lab/planar_ising_scaling.py",
    "docs/experiments/dtm-small-image.md",
    "uv.lock",
)
PROTOCOL_COMMIT = "711ed4ba628d24cb92a9550bdd15055817a07336"
PROBE_SEED = 20261014
WRITE_J = Z1HardwareProfile().node_full_sram_write_energy_j
READ_J = Z1HardwareProfile().node_read_energy_j
DEFAULT_WORKERS = 2
TOL = 0.01


def sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    atomic_write_text(Path(path), canonical_json(value) + "\n")


# ----------------------------------------------------------------------------
# Data model (exact reference): copied from the probe, checked bitwise in tests
# ----------------------------------------------------------------------------


def templates(L: int) -> np.ndarray:
    """Four +-1 shape templates (disk, vertical bar, horizontal bar, ring)."""
    r, c = np.mgrid[0:L, 0:L] + 0.5
    m = L / 2
    d = np.hypot(r - m, c - m)
    disk = d < L / 3
    vbar = np.abs(c - m) < L / 6
    hbar = np.abs(r - m) < L / 6
    ring = (d > L / 4) & (d < 0.45 * L)
    return np.stack([disk, vbar, hbar, ring]).astype(float) * 2 - 1


class GridIsing:
    """Nearest-neighbour Ising on an L x L open grid, raster order, exact by transfer.

    p(x) proportional to exp(J sum_<ij> x_i x_j + sum_i h_i x_i). The state is a
    window of the last L spins; cost O(L^2 2^L) per field configuration.
    """

    def __init__(self, L: int, J: float):
        self.L, self.J, self.N = L, J, L * L
        self.M = 1 << (L - 1)
        rest = np.arange(self.M)
        self.s_lat = (rest & 1) * 2.0 - 1.0

    def _ju_jl(self, k: int) -> tuple[float, float]:
        r, c = divmod(k, self.L)
        return (self.J if r > 0 else 0.0), (self.J if c > 0 else 0.0)

    def forward(self, h: np.ndarray, store: bool = False):
        B = h.shape[0]
        a = np.zeros((B, 2 * self.M))
        a[:, -1] = 1.0
        logz = np.zeros(B)
        alphas = np.empty((self.N, B, 2 * self.M)) if store else None
        for k in range(self.N):
            ju, jl = self._ju_jl(k)
            a2 = a.reshape(B, 2, self.M)
            new = np.empty((B, self.M, 2))
            for bit, s in ((1, 1.0), (0, -1.0)):
                up = a2[:, 0] * np.exp(-ju * s) + a2[:, 1] * np.exp(ju * s)
                new[:, :, bit] = up * np.exp(h[:, k : k + 1] * s + jl * s * self.s_lat[None])
            a = new.reshape(B, 2 * self.M)
            z = a.sum(1)
            a /= z[:, None]
            logz += np.log(z)
            if store:
                alphas[k] = a
        return logz, alphas

    def marginals(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        B = h.shape[0]
        logz, alphas = self.forward(h, store=True)
        beta = np.ones((B, 2 * self.M))
        p_up = np.empty((B, self.N))
        for k in range(self.N - 1, -1, -1):
            a = alphas[k].reshape(B, self.M, 2)
            bk = beta.reshape(B, self.M, 2)
            w = (a * bk).sum(1)
            p_up[:, k] = w[:, 1] / w.sum(1)
            ju, jl = self._ju_jl(k)
            nb = np.zeros((B, 2, self.M))
            for bit, s in ((1, 1.0), (0, -1.0)):
                base = bk[:, :, bit] * np.exp(h[:, k : k + 1] * s + jl * s * self.s_lat[None])
                nb[:, 0] += base * np.exp(-ju * s)
                nb[:, 1] += base * np.exp(ju * s)
            beta = nb.reshape(B, 2 * self.M)
            beta /= beta.max(1, keepdims=True)
        return p_up, logz

    def sample(self, h: np.ndarray, rng) -> np.ndarray:
        B = h.shape[0]
        _, alphas = self.forward(h, store=True)
        cum = alphas[-1].cumsum(1)
        w = (rng.random((B, 1)) < cum).argmax(1)
        x = np.empty((B, self.N))
        for k in range(self.N - 1, -1, -1):
            x[:, k] = (w & 1) * 2.0 - 1.0
            if k == 0:
                break
            ju, _ = self._ju_jl(k)
            rest = w >> 1
            prev = alphas[k - 1].reshape(B, 2, self.M)[np.arange(B), :, rest]
            s_new = x[:, k]
            wt = prev * np.exp(ju * s_new[:, None] * np.array([-1.0, 1.0])[None])
            old = (rng.random(B) < wt[:, 1] / wt.sum(1)).astype(np.int64)
            w = old * self.M + rest
        return x

    def neg_energy(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        g = x.reshape(-1, self.L, self.L)
        nn = (g[:, 1:, :] * g[:, :-1, :]).sum((1, 2)) + (g[:, :, 1:] * g[:, :, :-1]).sum((1, 2))
        return self.J * nn + (h * x).sum(1)


class Mixture:
    """Uniform mixture of GridIsing components sharing J, fields h0 * template_c."""

    def __init__(self, L: int, J: float, h0: float):
        self.grid = GridIsing(L, J)
        self.h = h0 * templates(L).reshape(4, -1)
        self.logz = self.grid.forward(self.h)[0]

    def sample(self, n: int, rng) -> np.ndarray:
        comp = rng.integers(0, len(self.h), n)
        return self.grid.sample(self.h[comp], rng)

    def loglik(self, x: np.ndarray) -> np.ndarray:
        terms = [
            self.grid.neg_energy(x, hc[None]) - lz for hc, lz in zip(self.h, self.logz, strict=True)
        ]
        return logsumexp(np.stack(terms), axis=0) - np.log(len(self.h))

    def posterior(self, y: np.ndarray, q: float) -> np.ndarray:
        g = 0.5 * np.log((1 - q) / q)
        lw, pm = [], []
        for hc, lz in zip(self.h, self.logz, strict=True):
            p, lzy = self.grid.marginals(hc[None] + g * y)
            lw.append(lzy - lz)
            pm.append(p)
        w = np.exp(np.stack(lw) - logsumexp(np.stack(lw), axis=0))
        return np.einsum("cb,cbn->bn", w, np.stack(pm))


def flip(x: np.ndarray, p: float, rng) -> np.ndarray:
    return np.where(rng.random(x.shape) < p, -x, x)


def brute_force_grid(L, J, h):
    """log Z and P(x_k = +1) of one L x L grid by enumeration (L <= 4)."""
    if L > 4:
        raise ValueError("brute force is bounded to 4 x 4")
    grid = GridIsing(L, J)
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=L * L)))
    energy = grid.neg_energy(states, h[None])
    top = energy.max()
    weights = np.exp(energy - top)
    z = weights.sum()
    return float(np.log(z) + top), (weights @ (states > 0)) / z


# ----------------------------------------------------------------------------
# Lattice denoiser: layout, NumPy chromatic Gibbs, trainer (probe, copied)
# ----------------------------------------------------------------------------


def layout(L: int):
    """Pixel (i, j) at site (2i, 2j), clamp at (2i+1, 2j), A at (2i, 2j+1), B at (2i+1, 2j+1)."""
    role, sites, index = [], [], {}
    roles = (("pix", (0, 0)), ("clamp", (1, 0)), ("A", (0, 1)), ("B", (1, 1)))
    for i in range(L):
        for j in range(L):
            for kind, (du, dv) in roles:
                index[(2 * i + du, 2 * j + dv)] = len(sites)
                sites.append((2 * i + du, 2 * j + dv))
                role.append(kind)
    free_edges = set()
    for k, s in enumerate(sites):
        if role[k] not in ("pix", "B"):
            continue
        for off in NEIGHBOURS:
            t = (s[0] + off[0], s[1] + off[1])
            if t in index and role[index[t]] == "A":
                free_edges.add((min(k, index[t]), max(k, index[t])))
    pix = [k for k, r in enumerate(role) if r == "pix"]
    clamps = [k for k, r in enumerate(role) if r == "clamp"]
    clamp_edges = [(p, c) for p, c in zip(pix, clamps, strict=True)]
    return role, sites, sorted(free_edges), clamp_edges, pix, clamps


class Layer:
    """One DTM step P(x^{t-1} | x^t) on the lattice layout."""

    def __init__(self, L: int, gamma: float, rng):
        self.role, self.sites, self.fe, self.ce, self.pix, self.clamps = layout(L)
        self.n = len(self.sites)
        self.free = [k for k, r in enumerate(self.role) if r != "clamp"]
        self.lat = [k for k, r in enumerate(self.role) if r in ("A", "B")]
        self.colour = np.array([(u + v) % 2 for u, v in self.sites])
        self.gamma = gamma
        self.w = rng.normal(0, 0.01, len(self.fe))
        self.b = np.zeros(self.n)
        self.fe_u = np.array([e[0] for e in self.fe])
        self.fe_v = np.array([e[1] for e in self.fe])

    def model(self, w=None, b=None, wc=None) -> PairwiseModel:
        w = self.w if w is None else w
        wc = np.full(len(self.ce), self.gamma) if wc is None else wc
        b = self.b if b is None else b
        return PairwiseModel(
            sites=tuple(self.sites),
            edges=tuple(self.fe) + tuple(self.ce),
            weights=np.concatenate([w, wc]),
            bias=b,
            free=tuple(self.free),
            clamped=tuple(self.clamps),
            clamped_input=tuple(range(len(self.pix))),
            first_colour=0,
        )

    @staticmethod
    def dense(model: PairwiseModel) -> np.ndarray:
        W = np.zeros((model.nodes, model.nodes))
        for (u, v), x in zip(model.edges, model.weights, strict=True):
            W[u, v] += x
            W[v, u] += x
        return W

    def run(self, model, x_t, sweeps, rng, fixed=None, gains=None, stats_from=None):
        """Probe sampler: uniform start, clamps = x_t (training phases)."""
        W, b = self.dense(model), model.bias
        B = x_t.shape[0]
        S = rng.integers(0, 2, (B, self.n)) * 2.0 - 1.0
        S[:, self.clamps] = x_t
        upd = list(self.free) if fixed is None else list(self.lat)
        if fixed is not None:
            S[:, self.pix] = fixed
        g = np.ones(self.n) if gains is None else gains
        blocks = [np.array([k for k in upd if self.colour[k] == c]) for c in (0, 1)]
        Wb = [W[:, blk] for blk in blocks]
        ee = np.zeros(len(self.fe))
        ss = np.zeros(self.n)
        m = 0
        for t in range(sweeps):
            for blk, Wk in zip(blocks, Wb, strict=True):
                f = g[blk] * (S @ Wk + b[blk])
                S[:, blk] = np.where(rng.random((B, len(blk))) < expit(2 * f), 1.0, -1.0)
            if stats_from is not None and t >= stats_from:
                ee += (S[:, self.fe_u] * S[:, self.fe_v]).mean(0)
                ss += S.mean(0)
                m += 1
        if stats_from is None:
            return S
        return S, ee / m, ss / m


def schedule(q: float, T: int) -> float:
    return 0.5 * (1 - (1 - 2 * q) ** (1 / T))


def cumulative(p: float, t: int) -> float:
    return 0.5 * (1 - (1 - 2 * p) ** t)


def train_layer(L, T, t, q, data, steps, batch, K, rng, lr=0.03):
    """Train layer t (1-based) of a T-step chain; the probe's inner loop, one rng."""
    p = schedule(q, T)
    gamma = 0.5 * np.log((1 - p) / p)
    lay = Layer(L, gamma, rng)
    mw, vw = np.zeros_like(lay.w), np.zeros_like(lay.w)
    mb, vb = np.zeros(lay.n), np.zeros(lay.n)
    for it in range(1, steps + 1):
        x0 = data[rng.integers(0, len(data), batch)]
        x_prev = flip(x0, cumulative(p, t - 1), rng) if t > 1 else x0
        x_t = flip(x_prev, p, rng)
        model = lay.model()
        _, e_pos, s_pos = lay.run(model, x_t, K, rng, fixed=x_prev, stats_from=K // 2)
        _, e_neg, s_neg = lay.run(model, x_t, K, rng, stats_from=K // 2)
        gw, gb = e_pos - e_neg, s_pos - s_neg
        gb[lay.clamps] = 0.0
        for g_, m_, v_, par in ((gw, mw, vw, lay.w), (gb, mb, vb, lay.b)):
            m_ *= 0.9
            m_ += 0.1 * g_
            v_ *= 0.999
            v_ += 0.001 * g_ * g_
            par += lr * (m_ / (1 - 0.9**it)) / (np.sqrt(v_ / (1 - 0.999**it)) + 1e-8)
    return lay


def layer_from(L, gamma, w, b):
    lay = Layer(L, gamma, np.random.default_rng(0))
    lay.w = np.array(w, dtype=np.float64)
    lay.b = np.array(b, dtype=np.float64)
    return lay


def sweep_from(lay: Layer, model, x_t, S, sweeps, rng, gains=None):
    """Chromatic Gibbs from the given full state S (rows = chains), clamps = x_t (r1 probe)."""
    W, b = Layer.dense(model), model.bias
    S = S.copy()
    S[:, lay.clamps] = x_t
    g = np.ones(lay.n) if gains is None else gains
    blocks = [np.array([k for k in lay.free if lay.colour[k] == c]) for c in (0, 1)]
    Wb = [W[:, blk] for blk in blocks]
    for _ in range(sweeps):
        for blk, Wk in zip(blocks, Wb, strict=True):
            f = g[blk] * (S @ Wk + b[blk])
            S[:, blk] = np.where(rng.random((S.shape[0], len(blk))) < expit(2 * f), 1.0, -1.0)
    return S


def numpy_streams(layers, y, K, rng, start, draws, streams, per_stream):
    """The r1 probe's ``denoise_streams`` with the stream geometry as arguments.

    Returns xs (draws, images, pixels). ``start='continue'`` carries each layer
    region's state through a stream; ``'random'`` redraws it uniform every step.
    """
    rows = draws * streams
    L2 = len(layers[0].pix)
    state = [rng.integers(0, 2, (rows, lay.n)) * 2.0 - 1.0 for lay in layers]
    xs = np.empty((draws, len(y), L2))
    for j in range(per_stream):
        idx = np.arange(streams) * per_stream + j
        x = np.tile(y[idx], (draws, 1))
        for li in range(len(layers) - 1, -1, -1):
            lay = layers[li]
            S0 = state[li]
            if start == "random":
                S0 = rng.integers(0, 2, (rows, lay.n)) * 2.0 - 1.0
            S = sweep_from(lay, lay.model(), x, S0, K, rng)
            state[li] = S
            x = S[:, lay.pix]
        xs[:, idx] = x.reshape(draws, streams, -1)
    return xs


# ----------------------------------------------------------------------------
# Conventional baseline (probe, copied)
# ----------------------------------------------------------------------------


def patch_features(y: np.ndarray, L: int, r: int = 2) -> np.ndarray:
    g = np.pad(y.reshape(-1, L, L), ((0, 0), (r, r), (r, r)))
    cols = [g[:, i : i + L, j : j + L] for i in range(2 * r + 1) for j in range(2 * r + 1)]
    f = np.stack(cols, -1).reshape(y.shape[0] * L * L, -1)
    pos = np.tile(np.eye(L * L), (y.shape[0], 1))
    return np.concatenate([f, pos, np.ones((f.shape[0], 1))], 1)


def train_logistic(L, q, data, rng, steps=400, lr=0.05, r=2):
    w = np.zeros((2 * r + 1) ** 2 + L * L + 1)
    m, v = np.zeros_like(w), np.zeros_like(w)
    for it in range(1, steps + 1):
        x = data[rng.integers(0, len(data), 64)]
        F = patch_features(flip(x, q, rng), L, r)
        t = (x.ravel() + 1) / 2
        g = F.T @ (t - expit(F @ w)) / len(t)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w += lr * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-8)
    return w


# ----------------------------------------------------------------------------
# THRML: lowering through the constraint layer, per-chain carried state
# ----------------------------------------------------------------------------


def make_thrml_runner(program, blocks, sweeps):
    """``thrml_lowering.final_states`` with one initial state per chain.

    Each chain key is split into (init, sample) and the init half is unused,
    exactly as ``final_states``; the tests check bitwise equality when every
    chain starts from the same state. Call inside ``jax.enable_x64(True)``.
    """
    sched = SamplingSchedule(n_warmup=sweeps, n_samples=1, steps_per_sample=1)

    def one_chain(key, x, *init):
        _, sample_key = jax.random.split(key, 2)
        observed = thrml_sample_states(sample_key, program, sched, list(init), [x], blocks)
        return jnp.concatenate([o[0] for o in observed])

    return jax.jit(jax.vmap(one_chain))


def arm_request(request, s, T, li, arm):
    spec = request["arm_definitions"][arm]
    relaxed, reasons = [], {}
    if spec["bits"] is None:
        relaxed.append("quantization")
        reasons["quantization"] = request["relaxed_reasons"]["quantization"]
    if spec["jitter"] == 0.0:
        relaxed.append("beta_jitter")
        reasons["beta_jitter"] = request["relaxed_reasons"]["beta_jitter"]
    return ConstraintRequest(
        model_id=f"dtm-small-image s{s} T{T} layer{li} {arm}",
        bits=spec["bits"],
        cap_rule="none" if spec["bits"] is None else request["cap_rule"],
        jitter=spec["jitter"],
        jitter_words=(*request["seeds"]["jitter_words"], s, T, li) if spec["jitter"] else None,
        update_rule="active_kernel",
        dtype="float64",
        relaxed=tuple(relaxed),
        reasons=reasons,
    )


def arm_model(lay: Layer, creq: ConstraintRequest):
    """(model, per-node gains or None, rounding summary or None) for one layer and arm."""
    model = lay.model()
    summary = None
    if creq.applies("quantization"):
        model, summary = quantize(model, creq)
    gains = None
    if creq.jitter > 0:
        (free_gains,) = request_gains(creq, (len(lay.free),))
        gains = np.ones(lay.n)
        gains[lay.free] = free_gains
    return model, gains, summary


class LoweredLayer:
    def __init__(self, lay, creq):
        self.model, self.gains, self.rounding = arm_model(lay, creq)
        require_placement(self.model, creq)
        with jax.enable_x64(True):
            self.ebm, self.program, self.blocks = lower(self.model, creq, gains=self.gains)
        self.groups = colour_blocks(self.model)
        order = [n for g in self.groups for n in g]
        position = {n: k for k, n in enumerate(order)}
        self.pix_cols = np.array([position[p] for p in lay.pix])
        self.sizes = [len(g) for g in self.groups]


def chain_keys(request, words, chains):
    key = jax.random.key(request["seeds"]["jax_root"])
    for w in words:
        key = jax.random.fold_in(key, int(w))
    return jax.random.split(key, chains)


# ----------------------------------------------------------------------------
# Request
# ----------------------------------------------------------------------------

BITS = (3, 4, 6, 8)
JITTERS = (0.0, 0.1)


def arm_names(bits=BITS, jitters=JITTERS):
    return ["float"] + [f"b{b}_j{j}" for b in bits for j in jitters]


def make_request(**overrides):
    """The frozen production request; tests pass smaller geometry through ``overrides``."""
    bits = overrides.pop("bits", BITS)
    arms = arm_names(bits)
    request = {
        "schema": "dtm_small_image.v1",
        "status": "exploratory",
        "protocol": "docs/experiments/dtm-small-image.md",
        "protocol_commit": PROTOCOL_COMMIT,
        "proposal": "P-0006",
        "family": {"L": 8, "J": 0.55, "h0": 0.2, "q": 0.2, "components": 4},
        "n_train": 4000,
        "n_test": 1000,
        "draws": 16,
        "streams": 50,
        "images_per_stream": 20,
        "training_seeds": [0, 1, 2],
        "steps": [1, 2],
        "training": {
            "rule": "contrastive, Adam (0.9, 0.999, 1e-8), uniform-start chains, half burn-in",
            "lr": 0.03,
            "batch": 64,
            "K_train": 64,
            "steps_per_layer": 1600,
            "init": "w ~ normal(0, 0.01), b = 0; clamp couplings fixed at gamma = atanh(1 - 2p)",
            "rng": "one default_rng([20261103, s, T, layer]) per layer draws init, batches, "
            "noise and chains in the probe's order",
        },
        "logistic": {"radius": 3, "steps": 2000, "lr": 0.05, "batch": 64},
        "arms": arms,
        "arm_definitions": {
            a: {"bits": None, "jitter": 0.0}
            if a == "float"
            else {"bits": int(a[1 : a.index("_")]), "jitter": float(a[a.index("_j") + 2 :])}
            for a in arms
        },
        "cap_rule": "split",
        "relaxed_reasons": {
            "quantization": "float arm: the reference the 0.01 tolerance is measured against",
            "beta_jitter": "noise-free arm: the jitter control",
        },
        "jitter_rule": "one gain per free node in free order, request_gains(request, (192,))",
        "sweeps": [16, 64, 256],
        "reference_sweeps": 1024,
        "numpy_controls": ["continue", "random"],
        "seeds": {
            "training_data": [20261101],
            "test": 20261102,
            "training_chains": [20261103],
            "jax_root": 20261104,
            "jitter_words": [20261105],
            "logistic": [20261106],
            "stream_states": [20261107],
            "bootstrap": [20261108],
            "chain_key_words": "fold_in(key(20261104), s, T, cell_code, image_j, layer)",
            "stream_state_rule": "default_rng([20261107, s, T, cell_code]) draws one uniform "
            "full-layer state per layer (layer 0 first); NumPy cells use the same rng for "
            "chains afterwards",
        },
        "bootstrap": {
            "resamples": 2000,
            "unit": "stream",
            "interval": "95 percent percentile, paired, within seed",
            "indices": "default_rng([20261108, s]).integers(0, streams, (2000, streams)), "
            "shared by every comparison of seed s",
        },
        "tolerance": TOL,
        "sample": (
            "one chain's pixel state after T steps of K sweeps from the noisy image; a "
            "recorded estimate is the pixelwise majority of 16 draws (ties to +1)"
        ),
        "persisted_per_image": (
            "majority wrong-pixel count (uint8), wrong-pixel count summed over draws "
            "(uint16), Jeffreys-smoothed 16-draw conditional log-loss (float32)"
        ),
        "update_rule": "active_kernel",
        "dtype": "float64",
        "io_path": "host per step, continuation; random-start control counts 3 L^2 T writes",
        "profile_id": Z1HardwareProfile().profile_id,
        "profile_hash": Z1HardwareProfile().profile_hash,
        "preflight": {
            "probe_seed": PROBE_SEED,
            "steps_per_layer": 1600,
            "n_train": 4000,
            "images": 10,
            "chains": 6400,
            "sweeps": 64,
            "z_max": 5.0,
            "transfer_tolerance": 1e-12,
        },
        "cpu_stop_hours": 6.0,
        "replay_rtol": 1e-12,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(request.get(key), dict):
            request[key] = {**request[key], **value}
        else:
            request[key] = value
    request["constraint_requests"] = {
        f"s{s}_T{T}_l{li}_{arm}": {
            "digest": (cr := arm_request(request, s, T, li, arm)).digest(),
            "canonical": cr.to_canonical(),
        }
        for s in request["training_seeds"]
        for T in request["steps"]
        for li in range(T)
        for arm in request["arms"]
    }
    request["sources"] = {name: sha(ROOT / name) for name in SOURCES}
    validate_request(request)
    return request


def validate_request(request):
    if request["schema"] != "dtm_small_image.v1":
        raise ValueError("unknown request schema")
    if request["n_test"] != request["streams"] * request["images_per_stream"]:
        raise ValueError("n_test must equal streams x images_per_stream")
    if request["arms"][0] != "float":
        raise ValueError("the float arm comes first")
    if request["profile_hash"] != Z1HardwareProfile().profile_hash:
        raise ValueError("Z1 profile changed")
    if request["tolerance"] != TOL or request["replay_rtol"] != 1e-12:
        raise ValueError("tolerance and replay tolerance are frozen")


def cells(request):
    """Every inference cell: (s, T, kind, arm, K) in a fixed order with a cell code."""
    out = []
    for s in request["training_seeds"]:
        for T in request["steps"]:
            code = 0
            for arm in request["arms"]:
                ks = list(request["sweeps"])
                if arm == "float":
                    ks.append(request["reference_sweeps"])
                for K in ks:
                    out.append({"s": s, "T": T, "kind": "thrml", "arm": arm, "K": K, "code": code})
                    code += 1
            for kind in request["numpy_controls"]:
                for K in request["sweeps"]:
                    out.append(
                        {"s": s, "T": T, "kind": f"numpy_{kind}", "arm": "float", "K": K}
                        | {"code": code}
                    )
                    code += 1
    return out


def cell_name(c):
    return f"cell__s{c['s']}_T{c['T']}_{c['kind']}_{c['arm']}_K{c['K']}"


# ----------------------------------------------------------------------------
# Exact reference
# ----------------------------------------------------------------------------


def family(request):
    f = request["family"]
    return Mixture(f["L"], f["J"], f["h0"]), f["q"]


def training_data(request, s):
    mix, _ = family(request)
    return mix.sample(
        request["n_train"], np.random.default_rng([*request["seeds"]["training_data"], s])
    )


def test_set(request):
    mix, q = family(request)
    rng = np.random.default_rng(request["seeds"]["test"])
    x = mix.sample(request["n_test"], rng)
    y = flip(x, q, rng)
    return x, y


def exact_reference(request):
    mix, q = family(request)
    x, y = test_set(request)
    t0 = time.process_time()
    post = mix.posterior(y, q)
    cpu_post = time.process_time() - t0
    pb = np.clip(post, 1e-300, 1.0)
    pm = np.clip(1 - post, 1e-300, 1.0)
    ll = -np.log(np.where(x > 0, pb, pm)).mean(1)
    mpm = (np.where(post > 0.5, 1.0, -1.0) != x).sum(1)
    out = {
        "evidence_class": "exact_reference",
        "test_x_sha256": _array_sha(x),
        "test_y_sha256": _array_sha(y),
        "training_data_sha256": {
            str(s): _array_sha(training_data(request, s)) for s in request["training_seeds"]
        },
        "component_log_z": mix.logz.tolist(),
        "test_loglik_per_pixel": float(mix.loglik(x).mean() / x.shape[1]),
        "noisy_wrong": (y != x).sum(1).astype(int).tolist(),
        "bayes_mpm_wrong": mpm.astype(int).tolist(),
        "bayes_sample_error": float(np.where(x > 0, 1 - post, post).mean()),
        "bayes_logloss": ll.tolist(),
    }
    return out, {"exact_posterior_cpu_s_per_image": cpu_post / len(y)}


def _array_sha(a):
    import hashlib

    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()
    ).hexdigest()


# ----------------------------------------------------------------------------
# Work units
# ----------------------------------------------------------------------------


def run_training(request, s, T):
    data = training_data(request, s)
    tr = request["training"]
    q = request["family"]["q"]
    layers, timing = [], {}
    for li in range(T):
        rng = np.random.default_rng([*request["seeds"]["training_chains"], s, T, li])
        t0 = time.process_time()
        lay = train_layer(
            request["family"]["L"],
            T,
            li + 1,
            q,
            data,
            tr["steps_per_layer"],
            tr["batch"],
            tr["K_train"],
            rng,
            tr["lr"],
        )
        timing[f"layer{li}_cpu_s"] = time.process_time() - t0
        layers.append(lay)
    arrays = {}
    for li, lay in enumerate(layers):
        arrays[f"w{li}"] = lay.w
        arrays[f"b{li}"] = lay.b
        arrays[f"gamma{li}"] = np.array([lay.gamma])
    return arrays, timing


def run_logistic(request, s):
    data = training_data(request, s)
    lg = request["logistic"]
    L, q = request["family"]["L"], request["family"]["q"]
    rng = np.random.default_rng([*request["seeds"]["logistic"], s])
    t0 = time.process_time()
    w = train_logistic(L, q, data, rng, steps=lg["steps"], lr=lg["lr"], r=lg["radius"])
    cpu_train = time.process_time() - t0
    x, y = test_set(request)
    t0 = time.process_time()
    pr = expit(patch_features(y, L, lg["radius"]) @ w).reshape(len(y), -1)
    cpu_inf = (time.process_time() - t0) / len(y)
    prc = np.clip(pr, 1e-12, 1 - 1e-12)
    arrays = {
        "w": w,
        "mpm_wrong": (np.where(pr > 0.5, 1.0, -1.0) != x).sum(1).astype(np.uint8),
        "logloss": (-np.log(np.where(x > 0, prc, 1 - prc))).mean(1),
    }
    return arrays, {"train_cpu_s": cpu_train, "inference_cpu_s_per_image": cpu_inf}


def load_layers(request, arrays, T):
    L = request["family"]["L"]
    return [
        layer_from(L, float(arrays[f"gamma{li}"][0]), arrays[f"w{li}"], arrays[f"b{li}"])
        for li in range(T)
    ]


def image_metrics(xs, truth):
    """Per-image (majority wrong, wrong summed over draws, Jeffreys log-loss)."""
    draws = xs.shape[0]
    maj = np.where(xs.mean(0) >= 0, 1.0, -1.0)
    maj_wrong = (maj != truth).sum(1).astype(np.uint8)
    draw_wrong = (xs != truth[None]).sum((0, 2)).astype(np.uint16)
    p_up = ((xs > 0).sum(0) + 0.5) / (draws + 1.0)
    ll = (-np.log(np.where(truth > 0, p_up, 1 - p_up))).mean(1).astype(np.float32)
    return {"maj_wrong": maj_wrong, "draw_wrong": draw_wrong, "logloss": ll}


def thrml_xs(request, s, T, code, layers, arm, K, y):
    """Denoise every test image through THRML under continuation; returns xs and timings."""
    draws, streams, per = request["draws"], request["streams"], request["images_per_stream"]
    rows = draws * streams
    L2 = request["family"]["L"] ** 2
    rng = np.random.default_rng([*request["seeds"]["stream_states"], s, T, code])
    full = [rng.integers(0, 2, (rows, lay.n)) * 2.0 - 1.0 for lay in layers]
    lowered = [
        LoweredLayer(lay, arm_request(request, s, T, li, arm)) for li, lay in enumerate(layers)
    ]
    state = [
        [jnp.asarray(full[li][:, g] > 0) for g in low.groups] for li, low in enumerate(lowered)
    ]
    xs = np.empty((draws, len(y), L2))
    compile_cpu = 0.0
    run_cpu = 0.0
    with jax.enable_x64(True):
        fns = []
        for li, low in enumerate(lowered):
            fn = make_thrml_runner(low.program, low.blocks, K)
            t0 = time.process_time()
            keys = chain_keys(request, (s, T, code, 0, li), rows)
            x0 = jnp.zeros((rows, L2), dtype=bool)
            fn = fn.lower(keys, x0, *state[li]).compile()
            compile_cpu += time.process_time() - t0
            fns.append(fn)
        t0 = time.process_time()
        for j in range(per):
            idx = np.arange(streams) * per + j
            x = np.tile(y[idx], (draws, 1))
            for li in range(len(layers) - 1, -1, -1):
                keys = chain_keys(request, (s, T, code, j, li), rows)
                out = np.asarray(fns[li](keys, jnp.asarray(x > 0), *state[li]).block_until_ready())
                low = lowered[li]
                split = np.cumsum(low.sizes)[:-1]
                state[li] = [jnp.asarray(part) for part in np.split(out, split, axis=1)]
                x = np.where(out[:, low.pix_cols], 1.0, -1.0)
            xs[:, idx] = x.reshape(draws, streams, -1)
        run_cpu = time.process_time() - t0
    return xs, {"compile_cpu_s": compile_cpu, "steady_cpu_s": run_cpu}


def run_cell(request, cell, model_arrays):
    s, T, K, code = cell["s"], cell["T"], cell["K"], cell["code"]
    x, y = test_set(request)
    layers = load_layers(request, model_arrays, T)
    if cell["kind"] == "thrml":
        xs, timing = thrml_xs(request, s, T, code, layers, cell["arm"], K, y)
    else:
        rng = np.random.default_rng([*request["seeds"]["stream_states"], s, T, code])
        start = cell["kind"].removeprefix("numpy_")
        t0 = time.process_time()
        xs = numpy_streams(
            layers,
            y,
            K,
            rng,
            start,
            request["draws"],
            request["streams"],
            request["images_per_stream"],
        )
        timing = {"compile_cpu_s": 0.0, "steady_cpu_s": time.process_time() - t0}
    return image_metrics(xs, x), timing


# ----------------------------------------------------------------------------
# Preflight and calibration (probe seeds)
# ----------------------------------------------------------------------------


def transfer_check(tolerance):
    rng = np.random.default_rng([PROBE_SEED, 301])
    rows = []
    for L in (3, 4):
        for J in (0.35, 0.55, -0.4):
            h = rng.normal(0, 0.5, L * L)
            p, lz = GridIsing(L, J).marginals(h[None])
            blz, bp = brute_force_grid(L, J, h)
            rows.append(
                {
                    "L": L,
                    "J": J,
                    "log_z_rel_error": abs(lz[0] - blz) / abs(blz),
                    "marginal_max_abs_error": float(np.abs(p[0] - bp).max()),
                }
            )
    passed = all(
        r["log_z_rel_error"] <= tolerance and r["marginal_max_abs_error"] <= tolerance for r in rows
    )
    return {"rows": rows, "tolerance": tolerance, "passed": bool(passed)}


def probe_model(request):
    pf = request["preflight"]
    mix, q = family(request)
    data = mix.sample(pf["n_train"], np.random.default_rng([PROBE_SEED, 201]))
    t0 = time.process_time()
    lay = train_layer(
        request["family"]["L"],
        1,
        1,
        q,
        data,
        pf["steps_per_layer"],
        request["training"]["batch"],
        request["training"]["K_train"],
        np.random.default_rng([PROBE_SEED, 202]),
        request["training"]["lr"],
    )
    return lay, time.process_time() - t0


def _z(a, b, n):
    p = (a + b) / 2
    var = p * (1 - p) * 2 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, (a - b) / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)
    return z


def sampler_check(request, lay):
    pf = request["preflight"]
    mix, q = family(request)
    rng = np.random.default_rng([PROBE_SEED, 203])
    x = mix.sample(pf["images"], rng)
    y = flip(x, q, rng)
    fixed = np.random.default_rng([PROBE_SEED, 204]).integers(0, 2, lay.n) * 2.0 - 1.0
    creq = ConstraintRequest(
        model_id="dtm-small-image preflight probe layer float",
        update_rule="active_kernel",
        dtype="float64",
        relaxed=("beta_jitter", "quantization"),
        reasons=dict(request["relaxed_reasons"]),
    )
    low = LoweredLayer(lay, creq)
    chains = pf["chains"]
    out = {"fixed_state": fixed.astype(int).tolist(), "starts": {}}
    with jax.enable_x64(True):
        fn = make_thrml_runner(low.program, low.blocks, pf["sweeps"])
        for start in ("uniform", "fixed"):
            zs = []
            for i in range(pf["images"]):
                srng = np.random.default_rng([PROBE_SEED, 205, i, start == "fixed"])
                if start == "uniform":
                    S0 = srng.integers(0, 2, (chains, lay.n)) * 2.0 - 1.0
                else:
                    S0 = np.tile(fixed, (chains, 1))
                xt = np.tile(y[i], (chains, 1))
                init = [jnp.asarray(S0[:, g] > 0) for g in low.groups]
                keys = chain_keys(
                    {"seeds": {"jax_root": PROBE_SEED}}, (205, i, start == "fixed"), chains
                )
                th = np.asarray(fn(keys, jnp.asarray(xt > 0), *init))[:, low.pix_cols]
                npx = sweep_from(lay, lay.model(), xt, S0, pf["sweeps"], srng)[:, lay.pix] > 0
                zs.append(_z(th.mean(0), npx.mean(0), chains))
            zmax = float(np.abs(np.concatenate(zs)).max())
            out["starts"][start] = {"max_abs_z": zmax}
    out["z_max"] = pf["z_max"]
    out["passed"] = bool(all(v["max_abs_z"] < pf["z_max"] for v in out["starts"].values()))
    return out


def lowering_check(request, lay):
    rows = {}
    for arm in request["arms"]:
        creq = arm_request(request, 0, 1, 0, arm)
        creq = ConstraintRequest(**{**_fields(creq), "model_id": f"preflight probe layer {arm}"})
        low = LoweredLayer(lay, creq)
        rows[arm] = {
            "weights_equal": bool(np.array_equal(np.asarray(low.ebm.weights), low.model.weights)),
            "bias_equal": bool(np.array_equal(np.asarray(low.ebm.biases), low.model.bias)),
            "dtype": str(np.asarray(low.ebm.weights).dtype),
            "edges": len(low.model.edges),
            "placement_ok": bool(check(low.model).ok),
            "max_degree": int(check(low.model).max_degree),
        }
    passed = all(
        r["weights_equal"] and r["bias_equal"] and r["dtype"] == "float64" and r["placement_ok"]
        for r in rows.values()
    )
    return {"arms": rows, "passed": bool(passed)}


def _fields(creq):
    return {
        "model_id": creq.model_id,
        "bits": creq.bits,
        "cap_rule": creq.cap_rule,
        "cap": creq.cap,
        "jitter": creq.jitter,
        "jitter_words": creq.jitter_words,
        "update_rule": creq.update_rule,
        "dtype": creq.dtype,
        "relaxed": creq.relaxed,
        "reasons": creq.reasons,
    }


def price_sample(L, T, K, start="continue"):
    """Projected joule per denoised sample, host path, with the clamp-load band."""
    n_pix, free, phys = L * L, 3 * L * L, 4 * L * L
    free_writes = free * T if start == "random" else 0
    writes = n_pix * T + free_writes
    creq = ConstraintRequest(
        model_id=f"dtm-small-image pricing L{L} T{T} K{K} {start}",
        bits=6,
        cap_rule="split",
        update_rule="active_kernel",
    )
    c = count(
        creq,
        logical_pbits=phys * T,
        physical_pbits_used=phys * T,
        free_pbits=free * T,
        sweeps=K * T,
        node_reads=n_pix * T,
        node_full_sram_writes=writes,
        clamp_state_changes=0,
        active_updates=free * K * T,
    )
    p = price(creq, c)
    clamp_loads = n_pix * T
    low = p["modeled_total_energy_j"] - clamp_loads * WRITE_J + clamp_loads * READ_J
    return {
        "reads": c.node_reads,
        "writes": c.node_full_sram_writes,
        "free_node_writes": free_writes,
        "clamp_state_changes": c.clamp_state_changes,
        "updates": c.gibbs_node_updates,
        "energy_j": p["modeled_total_energy_j"],
        "io_share": p["io_share"],
        "energy_j_clamp_loads_at_read_constant_outside_profile": low,
        "evidence_class": p["evidence_class"],
    }


def free_copy_bound(L, T, K):
    """On-chip chaining with free inter-layer handoff (outside the profile)."""
    n_pix, free, phys = L * L, 3 * L * L, 4 * L * L
    creq = ConstraintRequest(
        model_id=f"dtm-small-image free-copy bound L{L} T{T} K{K}",
        bits=6,
        cap_rule="split",
        update_rule="active_kernel",
    )
    c = count(
        creq,
        logical_pbits=(3 * T + 1) * n_pix,
        physical_pbits_used=(3 * T + 1) * n_pix,
        free_pbits=free * T,
        sweeps=K * T,
        node_reads=n_pix,
        node_full_sram_writes=n_pix,
        clamp_state_changes=0,
        active_updates=free * K * T,
    )
    del phys
    p = price(creq, c)
    return {
        "energy_j": p["modeled_total_energy_j"],
        "label": "outside the profile: hypothetical free inter-layer copy",
        "evidence_class": p["evidence_class"],
    }


def pricing_check(request):
    L = request["family"]["L"]
    rows = []
    ok = True
    for T in (1, 2):
        for K in (16, 64, 256, 1024):
            for start in ("continue", "random"):
                p = price_sample(L, T, K, start)
                n = L * L
                want_writes = n * T + (3 * n * T if start == "random" else 0)
                expected = 3 * n * K * T * 7.09e-15 + n * T * READ_J + want_writes * WRITE_J
                low = expected - n * T * WRITE_J + n * T * READ_J
                good = (
                    p["reads"] == n * T
                    and p["writes"] == want_writes
                    and p["clamp_state_changes"] == 0
                    and p["updates"] == 3 * n * K * T
                    and abs(p["energy_j"] - expected) <= 1e-12 * expected
                    and abs(p["energy_j_clamp_loads_at_read_constant_outside_profile"] - low)
                    <= 1e-12 * expected
                )
                ok = ok and good
                rows.append({"T": T, "K": K, "start": start, "passed": bool(good), **p})
    return {"rows": rows, "passed": bool(ok)}


def calibrate(request, lay):
    """Steady CPU per 800-chain THRML call by K, NumPy per call, and the full projection."""
    rows = request["draws"] * request["streams"]
    creq = ConstraintRequest(
        model_id="dtm-small-image calibration",
        update_rule="active_kernel",
        dtype="float64",
        relaxed=("beta_jitter", "quantization"),
        reasons=dict(request["relaxed_reasons"]),
    )
    low = LoweredLayer(lay, creq)
    rng = np.random.default_rng([PROBE_SEED, 206])
    S0 = rng.integers(0, 2, (rows, lay.n)) * 2.0 - 1.0
    x = rng.integers(0, 2, (rows, request["family"]["L"] ** 2)) > 0
    per_call, compile_cpu = {}, {}
    ks = sorted({*request["sweeps"], request["reference_sweeps"]})
    with jax.enable_x64(True):
        init = [jnp.asarray(S0[:, g] > 0) for g in low.groups]
        keys = chain_keys({"seeds": {"jax_root": PROBE_SEED}}, (206,), rows)
        for K in ks:
            fn = make_thrml_runner(low.program, low.blocks, K)
            t0 = time.process_time()
            fn = fn.lower(keys, jnp.asarray(x), *init).compile()
            compile_cpu[K] = time.process_time() - t0
            fn(keys, jnp.asarray(x), *init).block_until_ready()
            t0 = time.process_time()
            reps = 2
            for _ in range(reps):
                fn(keys, jnp.asarray(x), *init).block_until_ready()
            per_call[K] = (time.process_time() - t0) / reps
    t0 = time.process_time()
    sweep_from(lay, lay.model(), np.where(x, 1.0, -1.0), S0, 16, rng)
    numpy_per_sweep = (time.process_time() - t0) / 16
    per = request["images_per_stream"]
    seeds = len(request["training_seeds"])
    layer_calls = sum(request["steps"]) * per * seeds
    n_arms = len(request["arms"])
    thrml = layer_calls * (
        n_arms * sum(per_call[K] for K in request["sweeps"]) + per_call[request["reference_sweeps"]]
    )
    compiles = (
        sum(request["steps"])
        * seeds
        * (
            n_arms * sum(compile_cpu[K] for K in request["sweeps"])
            + compile_cpu[request["reference_sweeps"]]
        )
    )
    numpy_cpu = (
        layer_calls * len(request["numpy_controls"]) * numpy_per_sweep * sum(request["sweeps"])
    )
    return {
        "thrml_cpu_s_per_800_chain_call": {str(k): v for k, v in per_call.items()},
        "thrml_compile_cpu_s": {str(k): v for k, v in compile_cpu.items()},
        "numpy_cpu_s_per_sweep_800_chains": numpy_per_sweep,
        "projected_cpu_s": {"thrml": thrml, "compile": compiles, "numpy": numpy_cpu},
    }


def preflight(request):
    lay, train_cpu = probe_model(request)
    result = {
        "seeds": "probe seeds (20261014, ...) only",
        "transfer_matrix": transfer_check(request["preflight"]["transfer_tolerance"]),
        "placement": {"probe_layer_ok": bool(check(lay.model()).ok)},
        "samplers": sampler_check(request, lay),
        "lowering": lowering_check(request, lay),
        "pricing": pricing_check(request),
        "probe_layer": {"w": lay.w.tolist(), "b": lay.b.tolist(), "gamma": lay.gamma},
    }
    try:
        require_placement(lay.model(), arm_request(request, 0, 1, 0, "float"))
        result["placement"]["passed"] = result["placement"]["probe_layer_ok"]
    except ValueError:
        result["placement"]["passed"] = False
    result["passed"] = bool(
        all(result[k]["passed"] for k in ("transfer_matrix", "placement", "samplers", "lowering"))
        and result["pricing"]["passed"]
    )
    cal = calibrate(request, lay)
    layers_total = sum(request["steps"]) * len(request["training_seeds"])
    cal["probe_layer_train_cpu_s"] = train_cpu
    cal["projected_cpu_s"]["training"] = (
        train_cpu
        * layers_total
        * request["training"]["steps_per_layer"]
        / max(1, request["preflight"]["steps_per_layer"])
    )
    total = sum(cal["projected_cpu_s"].values())
    cal["projected_cpu_hours"] = total / 3600
    cal["stop_hours"] = request["cpu_stop_hours"]
    cal["within_budget"] = bool(cal["projected_cpu_hours"] <= request["cpu_stop_hours"])
    return result, cal


# ----------------------------------------------------------------------------
# Evaluation: bootstraps, decisions, pricing
# ----------------------------------------------------------------------------


def boot_indices(request, s):
    rng = np.random.default_rng([*request["seeds"]["bootstrap"], s])
    n = request["streams"]
    return rng.integers(0, n, (request["bootstrap"]["resamples"], n))


def interval(diff_img, idx, streams, per):
    d = np.asarray(diff_img, dtype=np.float64).reshape(streams, per).mean(1)
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(d.mean()), "lower": float(lo), "upper": float(hi)}


def cell_summary(request, arrays, exact):
    L2 = request["family"]["L"] ** 2
    draws, streams, per = request["draws"], request["streams"], request["images_per_stream"]
    maj = np.asarray(arrays["maj_wrong"], dtype=np.float64) / L2
    first = np.zeros(len(maj), bool)
    first[np.arange(streams) * per] = True
    e = maj.reshape(streams, per) - maj.mean()
    denom = (e * e).mean()
    lag1 = float((e[:, 1:] * e[:, :-1]).mean() / denom) if denom > 0 else 0.0
    ll = np.asarray(arrays["logloss"], dtype=np.float64)
    return {
        "majority_error": float(maj.mean()),
        "majority_error_se": float(maj.std(ddof=1) / np.sqrt(len(maj))),
        "sample_error": float(
            np.asarray(arrays["draw_wrong"], np.float64).sum() / (draws * L2 * len(maj))
        ),
        "cond_logloss_nats": float(ll.mean()),
        "bayes_logloss_nats": float(np.mean(exact["bayes_logloss"])),
        "stream_first_majority_error": float(maj[first].mean()),
        "rest_majority_error": float(maj[~first].mean()),
        "lag1_autocorr": lag1,
    }


def evaluate(request, exact, logistic, store, timings):
    """Pricing, intervals, the six decisions and the operating point from archived counts."""
    L = request["family"]["L"]
    L2 = L * L
    streams, per = request["streams"], request["images_per_stream"]
    seeds, steps = request["training_seeds"], request["steps"]
    ks, kref = list(request["sweeps"]), request["reference_sweeps"]
    bits = sorted({d["bits"] for d in request["arm_definitions"].values() if d["bits"]})
    jits = sorted({d["jitter"] for d in request["arm_definitions"].values()})
    noisy = np.asarray(exact["noisy_wrong"], np.float64) / L2
    bayes = np.asarray(exact["bayes_mpm_wrong"], np.float64) / L2
    idx = {s: boot_indices(request, s) for s in seeds}

    def err(s, T, kind, arm, K):
        a = store[cell_name({"s": s, "T": T, "kind": kind, "arm": arm, "K": K})]
        return np.asarray(a["maj_wrong"], np.float64) / L2

    def arm_of(b, j):
        return f"b{b}_j{j}"

    def iv(a, b, s):
        return interval(a - b, idx[s], streams, per)

    summaries = {}
    for c in cells(request):
        summaries[cell_name(c)] = cell_summary(request, store[cell_name(c)], exact)

    # 1. K* per T
    kstar = {}
    for T in steps:
        rows = {
            str(K): {
                str(s): iv(err(s, T, "thrml", "float", K), err(s, T, "thrml", "float", kref), s)
                for s in seeds
            }
            for K in ks
        }
        chosen = next(
            (K for K in ks if all(rows[str(K)][str(s)]["upper"] <= TOL for s in seeds)), None
        )
        kstar[str(T)] = {
            "intervals_vs_reference": rows,
            "k_star": chosen if chosen is not None else f"> {max(ks)}",
            "read_at": chosen if chosen is not None else max(ks),
            "flag_above_range": chosen is None,
        }

    # 2. b* per jitter and T at K*
    bstar = {}
    for T in steps:
        K = kstar[str(T)]["read_at"]
        for j in jits:
            rows = {
                str(b): {
                    str(s): iv(
                        err(s, T, "thrml", arm_of(b, j), K), err(s, T, "thrml", "float", K), s
                    )
                    for s in seeds
                }
                for b in bits
            }
            within = {b: all(rows[str(b)][str(s)]["upper"] <= TOL for s in seeds) for b in bits}
            chosen = next((b for b in bits if all(within[c] for c in bits if c >= b)), None)
            bstar[f"T{T}_j{j}"] = {
                "K": K,
                "intervals_vs_float": rows,
                "b_star": chosen if chosen is not None else f"> {max(bits)}",
                "read_at": chosen if chosen is not None else max(bits),
            }

    # 3. Jitter
    jitter = {}
    if len(jits) == 2:
        j0, j1 = jits
        for T in steps:
            K = kstar[str(T)]["read_at"]
            rows = {
                str(b): {
                    str(s): iv(
                        err(s, T, "thrml", arm_of(b, j1), K),
                        err(s, T, "thrml", arm_of(b, j0), K),
                        s,
                    )
                    for s in seeds
                }
                for b in bits
            }
            flat = [v for r in rows.values() for v in r.values()]
            if all(-TOL <= v["lower"] and v["upper"] <= TOL for v in flat):
                verdict = "negligible"
            elif any(v["lower"] > TOL for v in flat):
                verdict = "matters"
            else:
                verdict = "inconclusive"
            jitter[str(T)] = {"K": K, "intervals": rows, "verdict": verdict}

    # 4. Steps
    step = {}
    if 1 in steps and 2 in steps:
        k1, k2 = kstar["1"]["read_at"], kstar["2"]["read_at"]
        rows = {
            str(s): iv(err(s, 2, "thrml", "float", k2), err(s, 1, "thrml", "float", k1), s)
            for s in seeds
        }
        if all(v["upper"] < 0 for v in rows.values()):
            verdict = "helps"
        elif all(v["lower"] >= -TOL for v in rows.values()):
            verdict = "does_not_help"
        else:
            verdict = "inconclusive"
        step = {"K_T1": k1, "K_T2": k2, "intervals_T2_minus_T1": rows, "verdict": verdict}
        op_T = 2 if verdict == "helps" else 1
    else:
        op_T = steps[0]

    # 5. Baseline (operating point: jittered arm; jitter 0 is the control)
    jop = max(jits)
    baseline = {}
    for T in steps:
        for j in jits:
            K = kstar[str(T)]["read_at"]
            b = bstar[f"T{T}_j{j}"]["read_at"]
            rows = {
                str(s): iv(
                    err(s, T, "thrml", arm_of(b, j), K),
                    np.asarray(logistic[str(s)]["mpm_wrong"], np.float64) / L2,
                    s,
                )
                for s in seeds
            }
            if all(-TOL <= v["lower"] and v["upper"] <= TOL for v in rows.values()):
                verdict = "matches"
            elif all(v["lower"] > TOL for v in rows.values()):
                verdict = "worse"
            else:
                verdict = "inconclusive"
            baseline[f"T{T}_j{j}"] = {
                "bits": b,
                "K": K,
                "intervals_dtm_minus_logistic": rows,
                "verdict": verdict,
            }

    # 3-bit positive control at K*, jitter 0
    positive = {}
    if 3 in bits:
        for T in steps:
            K = kstar[str(T)]["read_at"]
            rows = {
                str(s): iv(
                    err(s, T, "thrml", arm_of(3, min(jits)), K), err(s, T, "thrml", "float", K), s
                )
                for s in seeds
            }
            positive[str(T)] = {
                "K": K,
                "intervals": rows,
                "detectable_loss": bool(all(v["lower"] > 0 for v in rows.values())),
                "tolerance_resolves": bool(all(v["lower"] > 0 for v in rows.values())),
            }

    # 6. Operating point
    K_op = kstar[str(op_T)]["read_at"]
    b_op = bstar[f"T{op_T}_j{jop}"]["read_at"]
    arm_op = arm_of(b_op, jop)
    per_sample = price_sample(L, op_T, K_op)
    draws = request["draws"]
    n_img = request["n_test"]

    def cpu_per_estimate(kind, arm, K):
        vals = [
            timings[cell_name({"s": s, "T": op_T, "kind": kind, "arm": arm, "K": K})][
                "steady_cpu_s"
            ]
            for s in seeds
        ]
        return float(np.mean(vals) / n_img)

    op_err = {str(s): float(err(s, op_T, "thrml", arm_op, K_op).mean()) for s in seeds}
    operating = {
        "T": op_T,
        "K": K_op,
        "bits": b_op,
        "jitter": jop,
        "arm": arm_op,
        "projected_j_per_sample": per_sample["energy_j"],
        "projected_j_per_sample_clamp_loads_at_read_constant_outside_profile": per_sample[
            "energy_j_clamp_loads_at_read_constant_outside_profile"
        ],
        "projected_j_per_estimate": draws * per_sample["energy_j"],
        "projected_j_per_estimate_clamp_loads_at_read_constant_outside_profile": draws
        * per_sample["energy_j_clamp_loads_at_read_constant_outside_profile"],
        "io_share": per_sample["io_share"],
        "free_copy_bound_j_per_sample_outside_profile": free_copy_bound(L, op_T, K_op)["energy_j"],
        "evidence_class_energy": "calibrated_projection",
        "majority_error_by_seed": op_err,
        "majority_error_mean": float(np.mean(list(op_err.values()))),
        "noisy_error": float(noisy.mean()),
        "bayes_mpm_error": float(bayes.mean()),
        "gap_to_bayes_by_seed": {
            str(s): iv(err(s, op_T, "thrml", arm_op, K_op), bayes, s) for s in seeds
        },
        "below_noisy_by_seed": {
            str(s): iv(err(s, op_T, "thrml", arm_op, K_op), noisy, s) for s in seeds
        },
        "logistic_error_by_seed": {
            str(s): float(np.mean(logistic[str(s)]["mpm_wrong"]) / L2) for s in seeds
        },
    }
    if timings:
        operating["cpu_s_per_estimate"] = {
            "thrml_steady": cpu_per_estimate("thrml", arm_op, K_op),
            "numpy_gibbs_float_continue": cpu_per_estimate(
                "numpy_continue", "float", K_op if K_op in ks else max(ks)
            ),
            "logistic_inference": float(
                np.mean([logistic[str(s)]["timing"]["inference_cpu_s_per_image"] for s in seeds])
            ),
            "exact_posterior": exact.get("timing", {}).get("exact_posterior_cpu_s_per_image"),
            "label": "software CPU seconds on this box; not hardware, not in the result digest",
        }

    pricing = {
        f"T{T}_K{K}_{st}": price_sample(L, T, K, st)
        for T in steps
        for K in [*ks, kref]
        for st in ("continue", "random")
    }
    float_reference = {
        f"T{T}": {
            str(s): {
                "majority_error_K_ref": float(err(s, T, "thrml", "float", kref).mean()),
                "noisy_error": float(noisy.mean()),
                "below_noisy": bool(err(s, T, "thrml", "float", kref).mean() < noisy.mean()),
            }
            for s in seeds
        }
        for T in steps
    }
    random_control = {
        f"T{T}_K{K}": {
            str(s): iv(
                err(s, T, "numpy_random", "float", K), err(s, T, "numpy_continue", "float", K), s
            )
            for s in seeds
        }
        for T in steps
        for K in ks
    }
    cross = {
        f"T{T}_K{K}": {
            str(s): iv(err(s, T, "thrml", "float", K), err(s, T, "numpy_continue", "float", K), s)
            for s in seeds
        }
        for T in steps
        for K in ks
    }
    return {
        "cell_summaries": summaries,
        "float_reference": float_reference,
        "decisions": {
            "k_star": kstar,
            "b_star": bstar,
            "jitter": jitter,
            "steps": step,
            "baseline": baseline,
            "positive_control_3bit": positive,
        },
        "operating_point": operating,
        "random_start_minus_continue": random_control,
        "thrml_minus_numpy_continue": cross,
        "pricing": pricing,
        "exact": {
            "noisy_error": float(noisy.mean()),
            "bayes_mpm_error": float(bayes.mean()),
            "bayes_sample_error": exact["bayes_sample_error"],
            "bayes_logloss_nats": float(np.mean(exact["bayes_logloss"])),
            "test_loglik_per_pixel": exact["test_loglik_per_pixel"],
        },
    }


def float_reference_stop(request, store):
    """Stop rule: float K_ref not below noisy input in two seeds (for any T)."""
    L2 = request["family"]["L"] ** 2
    x, y = test_set(request)
    noisy = float((x != y).mean())
    failures = {}
    for T in request["steps"]:
        bad = []
        for s in request["training_seeds"]:
            c = {"s": s, "T": T, "kind": "thrml", "arm": "float", "K": request["reference_sweeps"]}
            e = float(np.mean(store[cell_name(c)]["maj_wrong"]) / L2)
            if not e < noisy:
                bad.append(s)
        failures[str(T)] = bad
    return any(len(b) >= 2 for b in failures.values()), failures


# ----------------------------------------------------------------------------
# Persistence and autosave
# ----------------------------------------------------------------------------


def write_npz(path, arrays):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]))
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.getvalue())
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(path)


def read_npz(path):
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def write_gz_json(path, value):
    raw = (canonical_json(value) + "\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=9) as handle:
        handle.write(raw)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(path)


def read_gz_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


class Runner:
    def __init__(self, out, request_digest, source_digest):
        self.out = Path(out)
        self.units = self.out / "units"
        self.request_digest = request_digest
        self.source_digest = source_digest

    def log(self, message):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with (self.out / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
        print(f"{stamp} {message}", flush=True)

    def status(self, **fields):
        path = self.out / "run-status.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(fields, updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write(path, current)

    def index(self):
        path = self.units / "index.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save(self, name, arrays, timing):
        path = self.units / f"{name}.npz"
        write_npz(path, arrays)
        index = self.index()
        index[name] = {
            "sha256": sha(path),
            "request_digest": self.request_digest,
            "runner_source_sha256": self.source_digest,
            "timing": timing,
        }
        write(self.units / "index.json", index)

    def verified(self):
        good = {}
        for name, entry in self.index().items():
            path = self.units / f"{name}.npz"
            if (
                path.exists()
                and sha(path) == entry["sha256"]
                and entry["request_digest"] == self.request_digest
                and entry["runner_source_sha256"] == self.source_digest
            ):
                good[name] = entry
            else:
                self.log(f"unit {name} failed verification; it will be rerun")
        return good

    def load(self, name):
        return read_npz(self.units / f"{name}.npz")


def _pool_unit(request, kind, key, model_arrays):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if kind == "exact":
        result, timing = exact_reference(request)
        arrays = {"json": np.frombuffer(canonical_json(result).encode(), dtype=np.uint8)}
    elif kind == "train":
        arrays, timing = run_training(request, *key)
    elif kind == "logistic":
        arrays, timing = run_logistic(request, key)
    else:
        arrays, timing = run_cell(request, key, model_arrays)
    return kind, key, arrays, timing


def unit_name(kind, key):
    if kind == "exact":
        return "exact"
    if kind == "train":
        return f"train__s{key[0]}_T{key[1]}"
    if kind == "logistic":
        return f"logistic__s{key}"
    return cell_name(key)


def require_single_thread_blas():
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"set {name}=1 before running or replaying this study")


def _copy_sources(out, request):
    for name, digest in request["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)


def _run_units(runner, request, jobs, workers, stop_after, counter):
    """Run (kind, key, model_arrays) jobs, saving each unit as it finishes."""

    def record(kind, key, arrays, timing):
        name = unit_name(kind, key)
        runner.save(name, arrays, timing)
        counter["done"] += 1
        runner.status(last_unit=name, units_done=len(runner.index()))
        cpu = timing.get("steady_cpu_s", sum(v for v in timing.values() if isinstance(v, float)))
        runner.log(f"unit {name}: {cpu:.1f} CPU-s ({len(runner.index())} units saved)")
        if stop_after is not None and counter["done"] >= stop_after:
            raise KeyboardInterrupt("stop_after reached (interruption test)")

    if workers <= 1:
        for kind, key, model in jobs:
            record(*_pool_unit(request, kind, key, model))
        return
    if not jobs:
        return
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        futures = [pool.submit(_pool_unit, request, k, key, m) for k, key, m in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):
                record(*future.result())
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def run_study(out, requested=None, resume=False, workers=DEFAULT_WORKERS, stop_after=None):
    out = Path(out)
    require_single_thread_blas()
    if jax.default_backend() != "cpu":
        raise ValueError("CPU only")
    if resume:
        if not (out / "request.json").exists():
            raise FileNotFoundError("nothing to resume: request.json is missing")
        if (out / "completion.json").exists():
            raise FileExistsError("study already complete")
        request = json.loads((out / "request.json").read_text())
        current = make_request() if requested is None else requested
        if request != current:
            raise ValueError("stored request differs from this runner's request")
    else:
        if out.exists():
            raise FileExistsError("use a fresh output directory, or --resume")
        request = make_request() if requested is None else requested
        out.mkdir(parents=True)
        (out / "units").mkdir()
        write(out / "request.json", request)
    digest = canonical_sha256(request)
    runner = Runner(out, digest, request["sources"]["src/thermo_lab/dtm_small_image.py"])
    _copy_sources(out, request)
    attempts_path = out / "attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempt = len(attempts) + 1
    attempts.append(
        {
            "attempt": attempt,
            "resume": resume,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "workers": workers,
            "provenance": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        }
    )
    write(attempts_path, attempts)
    runner.log(f"attempt {attempt} ({'resume' if resume else 'fresh'}), {workers} worker(s)")
    started = time.perf_counter()
    counter = {"done": 0}
    try:
        runner.status(phase="preflight", attempt=attempt)
        pre_path = out / "preflight.json"
        if pre_path.exists():
            pre = json.loads(pre_path.read_text())
        else:
            result, cal = preflight(request)
            pre = {"preflight": result, "calibration": cal}
            write(pre_path, pre)
        if not pre["preflight"]["passed"]:
            runner.log("STOP: preflight failed; see preflight.json")
            raise ValueError("preflight failed; stopped before production")
        runner.log(
            "preflight passed; calibration projects "
            f"{pre['calibration']['projected_cpu_hours']:.2f} CPU-hours"
        )
        if not pre["calibration"]["within_budget"]:
            runner.log("STOP: calibration projects over the CPU stop budget")
            raise ValueError("calibration projects over the CPU budget")

        runner.status(phase="training")
        done = runner.verified()
        jobs = [("exact", None, None)] if "exact" not in done else []
        jobs += [
            ("train", (s, T), None)
            for s in request["training_seeds"]
            for T in request["steps"]
            if unit_name("train", (s, T)) not in done
        ]
        jobs += [
            ("logistic", s, None)
            for s in request["training_seeds"]
            if unit_name("logistic", s) not in done
        ]
        jobs.sort(key=lambda j: (j[0] != "train", -(j[1][1] if j[0] == "train" else 0)))
        _run_units(runner, request, jobs, workers, stop_after, counter)

        def models():
            return {
                (s, T): runner.load(unit_name("train", (s, T)))
                for s in request["training_seeds"]
                for T in request["steps"]
            }

        model_arrays = models()
        all_cells = cells(request)
        refs = [
            c for c in all_cells if c["kind"] == "thrml" and c["K"] == request["reference_sweeps"]
        ]
        rest = [c for c in all_cells if c not in refs]

        runner.status(phase="reference_cells")
        done = runner.verified()
        jobs = [
            ("cell", c, model_arrays[(c["s"], c["T"])]) for c in refs if cell_name(c) not in done
        ]
        _run_units(runner, request, jobs, workers, stop_after, counter)
        store = {cell_name(c): runner.load(cell_name(c)) for c in refs}
        stop, failures = float_reference_stop(request, store)
        write(out / "float-reference-check.json", {"stop": stop, "failures": failures})
        if stop:
            runner.log(f"STOP: float K_ref not below the noisy input in two seeds: {failures}")
            raise ValueError("trained float model is not a denoiser in two seeds")
        runner.log("float reference cells are below the noisy input")

        runner.status(phase="cells")
        done = runner.verified()
        jobs = [
            ("cell", c, model_arrays[(c["s"], c["T"])]) for c in rest if cell_name(c) not in done
        ]
        jobs.sort(key=lambda j: (-j[1]["K"] * j[1]["T"] * (4 if j[1]["kind"] != "thrml" else 1),))
        runner.log(f"cells: {len(rest) - len(jobs)} verified, {len(jobs)} to run")
        _run_units(runner, request, jobs, workers, stop_after, counter)

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
    completion = replay(out, light=False)
    runner.status(phase="complete", error=None, traceback=None)
    runner.log(f"completion written after replay ({completion['replay_seconds']:.1f} s)")
    return completion


def _archive_arrays(a):
    return {
        "maj_wrong": np.asarray(a["maj_wrong"]).astype(int).tolist(),
        "draw_wrong": np.asarray(a["draw_wrong"]).astype(int).tolist(),
        "logloss": [float(str(v)) for v in np.asarray(a["logloss"], dtype=np.float32)],
    }


def assemble(out, request, runner, seconds):
    index = runner.verified()
    exact = json.loads(bytes(runner.load("exact")["json"]).decode())
    exact_timing = index["exact"]["timing"]
    trained = {}
    for s in request["training_seeds"]:
        for T in request["steps"]:
            a = runner.load(unit_name("train", (s, T)))
            trained[f"s{s}_T{T}"] = {
                f"layer{li}": {
                    "gamma": float(a[f"gamma{li}"][0]),
                    "w": a[f"w{li}"].tolist(),
                    "b": a[f"b{li}"].tolist(),
                }
                for li in range(T)
            }
    logistic = {}
    for s in request["training_seeds"]:
        a = runner.load(unit_name("logistic", s))
        logistic[str(s)] = {
            "w": a["w"].tolist(),
            "mpm_wrong": a["mpm_wrong"].astype(int).tolist(),
            "logloss": a["logloss"].tolist(),
        }
    store, timings = {}, {}
    for c in cells(request):
        name = cell_name(c)
        store[name] = _archive_arrays(runner.load(name))
        timings[name] = index[name]["timing"]
    pre = json.loads((out / "preflight.json").read_text())
    rounding = rounding_record(request, trained)
    timing_block = {
        "label": "CPU seconds (time.process_time) on this box; excluded from the result digest",
        "exact": exact_timing,
        "training": {k: v["timing"] for k, v in index.items() if k.startswith("train__")},
        "logistic": {k: v["timing"] for k, v in index.items() if k.startswith("logistic__")},
        "cells": timings,
        "calibration": pre["calibration"],
        "generation_seconds_this_attempt": seconds,
    }
    for s in request["training_seeds"]:
        logistic[str(s)]["timing"] = index[unit_name("logistic", s)]["timing"]
    evaluation = evaluate(request, exact | {"timing": exact_timing}, logistic, store, timings)
    for s in request["training_seeds"]:
        logistic[str(s)].pop("timing")
    result = {
        "request_digest": canonical_sha256(request),
        "evidence_class": {
            "exact": "exact_reference",
            "thrml_cells": "software_simulation",
            "numpy_cells": "software_simulation",
            "logistic": "software_simulation",
            "training": "software_simulation",
            "joules": "calibrated_projection",
            "cpu_seconds": "software CPU measurement on this box (not evidence of hardware)",
        },
        "hardware_claim": False,
        "preflight": pre["preflight"],
        "exact_reference": exact,
        "trained_weights": trained,
        "rounding": rounding,
        "logistic": logistic,
        "cells": store,
    }
    result["result_digest"] = canonical_sha256(result)
    study = {**result, "evaluation": evaluation, "timings": timing_block}
    write_gz_json(out / "study.json.gz", study)
    attempts = json.loads((out / "attempts.json").read_text())
    write(
        out / "provenance.json",
        {
            "schema": "dtm_small_image.provenance.v1",
            "request_digest": result["request_digest"],
            "attempts": attempts,
            "runtime": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        },
    )
    runner.log("assembled study.json.gz and provenance.json")


def rounding_record(request, trained):
    """Caps and rounding summaries per (seed, T, layer, quantized arm); re-verifies placement."""
    out = {}
    L = request["family"]["L"]
    for key, layers in trained.items():
        s, T = int(key[1 : key.index("_")]), int(key[key.index("T") + 1 :])
        for li in range(T):
            d = layers[f"layer{li}"]
            lay = layer_from(L, d["gamma"], d["w"], d["b"])
            for arm in request["arms"]:
                creq = arm_request(request, s, T, li, arm)
                model, _, summary = arm_model(lay, creq)
                require_placement(model, creq)
                if summary is not None:
                    out[f"{key}_l{li}_{arm}"] = {
                        "coupling_cap": summary["coupling_cap"],
                        "field_cap": summary["field_cap"],
                        "coupling_max_cap_normalized": summary["couplings"]["max_cap_normalized"],
                        "field_max_cap_normalized": summary["fields"]["max_cap_normalized"],
                    }
    return out


# ----------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------


def _close(actual, expected, rtol, what):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != e.shape or not np.all(np.abs(a - e) <= rtol * np.maximum(1.0, np.abs(e))):
        raise ValueError(f"replay mismatch: {what}")


def replay(out, light=False):
    out = Path(out)
    require_single_thread_blas()
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    study = read_gz_json(out / "study.json.gz")
    validate_request(request)
    digest = canonical_sha256(request)
    if study["request_digest"] != digest:
        raise ValueError("request digest mismatch")
    for name, d in request["sources"].items():
        for path in (ROOT / name, out / "source" / name):
            if sha(path) != d:
                raise ValueError(f"source changed: {name}")
    for key, value in request["constraint_requests"].items():
        s, T, li, arm = _parse_cr_key(key)
        cr = arm_request(request, s, T, li, arm)
        if cr.digest() != value["digest"] or cr.to_canonical() != value["canonical"]:
            raise ValueError(f"constraint request does not rebuild: {key}")
    core = {k: study[k] for k in study if k not in ("evaluation", "timings", "result_digest")}
    if canonical_sha256(core) != study["result_digest"]:
        raise ValueError("result digest mismatch")
    # exact references
    exact, _ = exact_reference(request)
    stored = study["exact_reference"]
    rtol = request["replay_rtol"]
    for key in (
        "test_x_sha256",
        "test_y_sha256",
        "training_data_sha256",
        "noisy_wrong",
        "bayes_mpm_wrong",
    ):
        if exact[key] != stored[key]:
            raise ValueError(f"exact reference mismatch: {key}")
    for key in ("component_log_z", "test_loglik_per_pixel", "bayes_sample_error", "bayes_logloss"):
        _close(exact[key], stored[key], rtol, key)
    transfer = transfer_check(request["preflight"]["transfer_tolerance"])
    if not transfer["passed"]:
        raise ValueError("transfer-matrix check failed on replay")
    # placement and rounding
    check_equal(rounding_record(request, study["trained_weights"]), study["rounding"])
    # pricing check
    if not pricing_check(request)["passed"]:
        raise ValueError("pricing check failed on replay")
    # cells: shapes and ranges
    n, L2, draws = request["n_test"], request["family"]["L"] ** 2, request["draws"]
    names = [cell_name(c) for c in cells(request)]
    if sorted(study["cells"]) != sorted(names):
        raise ValueError("cell set differs from the request")
    for name in names:
        a = study["cells"][name]
        if len(a["maj_wrong"]) != n or not all(0 <= v <= L2 for v in a["maj_wrong"]):
            raise ValueError(f"bad majority counts: {name}")
        if len(a["draw_wrong"]) != n or not all(0 <= v <= draws * L2 for v in a["draw_wrong"]):
            raise ValueError(f"bad draw counts: {name}")
        if len(a["logloss"]) != n:
            raise ValueError(f"bad log-loss length: {name}")
    # evaluation recomputed from archived counts
    logistic = {
        s: v | {"timing": study["timings"]["logistic"][f"logistic__s{s}"]}
        for s, v in study["logistic"].items()
    }
    evaluation = evaluate(
        request,
        study["exact_reference"] | {"timing": study["timings"]["exact"]},
        logistic,
        study["cells"],
        study["timings"]["cells"],
    )
    check_equal(evaluation, study["evaluation"])
    # prefix check
    prefix = {"run": not light}
    if not light:
        for T in request["steps"]:
            s = request["training_seeds"][0]
            K = max(request["sweeps"])
            c = next(
                c
                for c in cells(request)
                if c["s"] == s
                and c["T"] == T
                and c["kind"] == "thrml"
                and c["arm"] == "float"
                and c["K"] == K
            )
            layers = study["trained_weights"][f"s{s}_T{T}"]
            arrays = {}
            for li in range(T):
                d = layers[f"layer{li}"]
                arrays[f"w{li}"] = np.asarray(d["w"])
                arrays[f"b{li}"] = np.asarray(d["b"])
                arrays[f"gamma{li}"] = np.asarray([d["gamma"]])
            got, _ = run_cell(request, c, arrays)
            same = _archive_arrays(got) == study["cells"][cell_name(c)]
            prefix[cell_name(c)] = bool(same)
            if not same:
                raise ValueError(f"prefix check failed: {cell_name(c)}")
        prefix["passed"] = True
    dec = evaluation["decisions"]
    thrml_cells = sum(1 for c in cells(request) if c["kind"] == "thrml")
    completion = {
        "status": "dtm_small_image_complete",
        "request_digest": digest,
        "result_digest": study["result_digest"],
        "seeds": len(request["training_seeds"]),
        "trained_models": len(request["training_seeds"]) * len(request["steps"]),
        "thrml_cells": thrml_cells,
        "numpy_cells": len(names) - thrml_cells,
        "test_images": n,
        "preflight_passed": bool(study["preflight"]["passed"]),
        "placement_passed": True,
        "references_recomputed": True,
        "prefix_check_passed": bool(prefix.get("passed", False)),
        "light_replay": bool(light),
        "replayed": True,
        "k_star": {T: v["k_star"] for T, v in dec["k_star"].items()},
        "b_star": {k: v["b_star"] for k, v in dec["b_star"].items()},
        "jitter_verdict": {T: v["verdict"] for T, v in dec["jitter"].items()},
        "steps_verdict": dec["steps"].get("verdict"),
        "baseline_verdict": {k: v["verdict"] for k, v in dec["baseline"].items()},
        "positive_control_detectable": {
            T: v["detectable_loss"] for T, v in dec["positive_control_3bit"].items()
        },
        "operating_point": {
            k: evaluation["operating_point"][k]
            for k in ("T", "K", "bits", "jitter", "projected_j_per_sample", "majority_error_mean")
        },
        "evidence_class": study["evidence_class"],
        "hardware_claim": False,
        "replay_seconds": time.perf_counter() - started,
    }
    if not light:
        write(out / "completion.json", completion)
    return completion


def _parse_cr_key(key):
    s_part, t_part, l_part, arm = key.split("_", 3)
    return int(s_part[1:]), int(t_part[1:]), int(l_part[1:]), arm


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    parser.add_argument("--replay", action="store_true", help="replay a finished run only")
    parser.add_argument("--light", action="store_true", help="replay without the prefix check")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--request", help="request JSON (tests only)")
    parser.add_argument("--stop-after", type=int, help="interrupt after N new units (tests only)")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.output_dir, light=args.light), indent=2))
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
