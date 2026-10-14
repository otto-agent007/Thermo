"""dtm-small-image probe (P-0006). Exploration only, never evidence.

Settles the row's fixed choices before a protocol exists:

1. The I/O path per denoised sample, priced through ``thermo_lab.constraints``.
2. Which graph the exact reference covers: a synthetic data set drawn exactly
   from a mixture of nearest-neighbour Ising grids (width L <= 24), whose
   likelihood, exact samples and exact flip-noise posterior come from a
   site-by-site transfer matrix. The denoiser's own lattice graph has long
   offsets and latents, so its likelihood is not tractable.
3. Synthetic first: no binarized image set is on disk, and fetching one is a
   download (owner decision); no new dependency is allowed.

It also trains a small lattice-native chained denoiser (T = 1, 2, 4 steps) on
the 16-offset lattice, quantizes it at 4, 6 and 8 bits under the `split` and
`absolute` cap rules with 0 and 10 percent static jitter, and times CPU Gibbs,
the exact posterior and a small logistic denoiser, to size the production run.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
    uv run python docs/research/dtm_small_image_probe.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import expit, logsumexp  # noqa: E402

from thermo_lab.constraints import (  # noqa: E402
    ConstraintRequest,
    PairwiseModel,
    check,
    count,
    price,
    quantize,
    require_placement,
)
from thermo_lab.constraints.jitter import static_gains  # noqa: E402
from thermo_lab.constraints.lattice import NEIGHBOURS  # noqa: E402

OUT = Path(__file__).with_name("2026-10-14-dtm-small-image-probe.json")
SEED = 20261014  # probe seed; a protocol must use fresh seeds

# ----------------------------------------------------------------- data model


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
    window of the last L spins (bit L-1 oldest = up neighbour, bit 0 latest =
    left neighbour); cost O(L^2 2^L) per field configuration.
    """

    def __init__(self, L: int, J: float):
        self.L, self.J, self.N = L, J, L * L
        self.M = 1 << (L - 1)
        rest = np.arange(self.M)
        self.s_lat = (rest & 1) * 2.0 - 1.0  # latest spin of each "rest" index

    def _ju_jl(self, k: int) -> tuple[float, float]:
        r, c = divmod(k, self.L)
        return (self.J if r > 0 else 0.0), (self.J if c > 0 else 0.0)

    def forward(self, h: np.ndarray, store: bool = False):
        """h: (B, N). Returns log Z (B,) and optionally normalized alphas (N, B, 2^L)."""
        B = h.shape[0]
        a = np.zeros((B, 2 * self.M))
        a[:, -1] = 1.0  # dummy window, all +1, uncoupled
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
        """Exact P(x_k = +1) for every site, and log Z. h: (B, N)."""
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
        """One exact sample per row of h (FFBS). Returns (B, N) +-1."""
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
            prev = alphas[k - 1].reshape(B, 2, self.M)[np.arange(B), :, rest]  # (B, 2) old
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
        """Exact P(x_i = +1 | y) under symmetric flip noise of rate q."""
        g = 0.5 * np.log((1 - q) / q)
        lw, pm = [], []
        for hc, lz in zip(self.h, self.logz, strict=True):
            p, lzy = self.grid.marginals(hc[None] + g * y)
            lw.append(lzy - lz)
            pm.append(p)
        w = np.exp(np.stack(lw) - logsumexp(np.stack(lw), axis=0))  # (C, B)
        return np.einsum("cb,cbn->bn", w, np.stack(pm))


def flip(x: np.ndarray, p: float, rng) -> np.ndarray:
    return np.where(rng.random(x.shape) < p, -x, x)


# --------------------------------------------------------- lattice denoiser


def layout(L: int):
    """Lattice-native layer: pixel (i, j) at site (2i, 2j), its clamp at (2i+1, 2j),
    latent A at (2i, 2j+1), latent B at (2i+1, 2j+1). Edges: every Z1 offset between
    pixel-A and A-B pairs inside the region, plus pixel-own clamp. Returns roles,
    sites and edge list (free-free edges first, then clamp edges)."""
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
    """One DTM step P(x^{t-1} | x^t) on the lattice layout; numpy chromatic Gibbs."""

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
        """Chromatic Gibbs from a uniform start (no write), clamps = x_t.

        ``fixed`` (pixel values) clamps the pixels too (positive phase). Returns the
        final state and, if ``stats_from`` is set, time-averaged s_u s_v over the free
        edges and s over nodes from that sweep on."""
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
    """Per-step flip rate p with T composed steps giving cumulative rate q."""
    return 0.5 * (1 - (1 - 2 * q) ** (1 / T))


def cumulative(p: float, t: int) -> float:
    return 0.5 * (1 - (1 - 2 * p) ** t)


def train_dtm(L, T, q, data, steps, batch, K, rng, lr=0.03):
    p = schedule(q, T)
    gamma = 0.5 * np.log((1 - p) / p)
    layers = []
    for t in range(1, T + 1):
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
        layers.append(lay)
    return layers, p, gamma


def denoise(layers, y, K, rng, models=None, gains=None):
    """Run the chain from x^T = y down to x^0 (host path: read pixels, rewrite clamps)."""
    x = y
    for idx in range(len(layers) - 1, -1, -1):
        lay = layers[idx]
        model = lay.model() if models is None else models[idx]
        g = None if gains is None else gains[idx]
        S = lay.run(model, x, K, rng, gains=g)
        x = S[:, lay.pix]
    return x


# ------------------------------------------------------- conventional baseline


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


# -------------------------------------------------------------------- pricing


def price_paths(L: int, T: int, K: int) -> dict:
    """Projected joule per denoised sample under three declared I/O paths."""
    n_pix, free, phys = L * L, 3 * L * L, 4 * L * L
    out = {}
    req = ConstraintRequest(model_id=f"dtm L{L} T{T}", bits=6, cap_rule="split")
    paths = {
        # host: per step write x^t into the clamps, sample K sweeps, read the pixels
        "host_per_step": dict(
            physical=phys * T,
            free=free * T,
            sweeps=K * T,
            reads=n_pix * T,
            writes=n_pix * T,
            clamps=0,
            active=free * K * T,
        ),
        # on chip, clamp-state changes priced as writes (sealed profile): write x^T once,
        # freeze then unfreeze each intermediate pixel layer, read x^0 once
        "chained_clamp_writes": dict(
            physical=(3 * T + 1) * n_pix,
            free=free * T,
            sweeps=K * T,
            reads=n_pix,
            writes=n_pix + 2 * (T - 1) * n_pix,
            clamps=2 * (T - 1) * n_pix,
            active=free * K * T,
        ),
        # on chip, layers gated by the schedule with no clamp change (outside the profile)
        "chained_gated_free": dict(
            physical=(3 * T + 1) * n_pix,
            free=free * T,
            sweeps=K * T,
            reads=n_pix,
            writes=n_pix,
            clamps=0,
            active=free * K * T,
        ),
    }
    for name, d in paths.items():
        r = ConstraintRequest(
            model_id=f"{req.model_id} {name}",
            bits=6,
            cap_rule="split",
            update_rule="active_kernel",
        )
        c = count(
            r,
            logical_pbits=d["physical"],
            physical_pbits_used=d["physical"],
            free_pbits=d["free"],
            sweeps=d["sweeps"],
            node_reads=d["reads"],
            node_full_sram_writes=d["writes"],
            clamp_state_changes=d["clamps"],
            active_updates=d["active"],
        )
        pr = price(r, c)
        out[name] = {
            "updates": c.gibbs_node_updates,
            "reads": c.node_reads,
            "writes": c.node_full_sram_writes,
            "clamp_changes": c.clamp_state_changes,
            "energy_j": pr["modeled_total_energy_j"],
            "io_share": pr["io_share"],
            "evidence_class": pr["evidence_class"],
        }
    return out


# ------------------------------------------------------------------- stage 2


def metrics(xs: np.ndarray, truth: np.ndarray) -> dict:
    """xs: (draws, images, pixels). Per-image means give the standard errors."""
    n = len(truth)
    err_img = (xs != truth[None]).mean((0, 2))
    p_up = ((xs > 0).sum(0) + 0.5) / (xs.shape[0] + 1.0)  # Jeffreys-smoothed marginal
    ll_img = -np.log(np.where(truth > 0, p_up, 1 - p_up)).mean(1)
    maj = np.where(xs.mean(0) >= 0, 1.0, -1.0)
    maj_img = (maj != truth).mean(1)
    return {
        "sample_error": float(err_img.mean()),
        "sample_error_se": float(err_img.std(ddof=1) / np.sqrt(n)),
        "majority_error": float(maj_img.mean()),
        "majority_error_se": float(maj_img.std(ddof=1) / np.sqrt(n)),
        "cond_logloss_nats": float(ll_img.mean()),
        "draws": int(xs.shape[0]),
    }


def stage2(mix, train, test, y, post, q, rng) -> dict:
    L = mix.grid.L
    out: dict = {}
    pb = np.clip(post, 1e-12, 1 - 1e-12)
    ll = -np.log(np.where(test > 0, pb, 1 - pb)).mean(1)
    se = np.where(test > 0, 1 - post, post).mean(1)
    mpm = (np.where(post > 0.5, 1.0, -1.0) != test).mean(1)
    out["bayes"] = {
        "sample_error": float(se.mean()),
        "sample_error_se": float(se.std(ddof=1) / np.sqrt(len(se))),
        "mpm_error": float(mpm.mean()),
        "mpm_error_se": float(mpm.std(ddof=1) / np.sqrt(len(mpm))),
        "cond_logloss_nats": float(ll.mean()),
    }
    # conventional baseline: 7x7 patch logistic plus position bias, 2000 steps
    t0 = time.process_time()
    w = train_logistic(L, q, train, rng, steps=2000, r=3)
    t_tr = time.process_time() - t0
    t0 = time.process_time()
    pr = expit(patch_features(y, L, 3) @ w).reshape(len(y), -1)
    t_inf = (time.process_time() - t0) / len(y)
    prc = np.clip(pr, 1e-12, 1 - 1e-12)
    lse = np.where(test > 0, 1 - pr, pr).mean(1)
    lmp = (np.where(pr > 0.5, 1.0, -1.0) != test).mean(1)
    out["logistic7"] = {
        "params": int(w.size),
        "sample_error": float(lse.mean()),
        "sample_error_se": float(lse.std(ddof=1) / np.sqrt(len(lse))),
        "mpm_error": float(lmp.mean()),
        "mpm_error_se": float(lmp.std(ddof=1) / np.sqrt(len(lmp))),
        "cond_logloss_nats": float(-np.log(np.where(test > 0, prc, 1 - prc)).mean()),
        "cpu_s_train": t_tr,
        "cpu_s_per_image": t_inf,
    }
    print(json.dumps(out["logistic7"]), flush=True)
    out["runs"] = {}
    best = None
    for T, K_train, steps in ((1, 16, 1600), (1, 64, 1600), (1, 16, 4800), (2, 16, 1600)):
        t0 = time.process_time()
        layers, _, _ = train_dtm(L, T, q, train, steps, 64, K_train, rng)
        t_train = time.process_time() - t0
        row = {"T": T, "K_train": K_train, "steps_per_layer": steps, "cpu_s_train": t_train}
        for K in (16, 64, 256):
            t0 = time.process_time()
            xs = np.stack([denoise(layers, y, K, rng) for _ in range(16)])
            row[f"K{K}"] = metrics(xs, test) | {
                "cpu_s_per_sample": (time.process_time() - t0) / (16 * len(y))
            }
        key = f"T{T}_Ktr{K_train}_steps{steps}"
        out["runs"][key] = row
        print(key, json.dumps(row), flush=True)
        if best is None or row["K64"]["sample_error"] < best[1]["K64"]["sample_error"]:
            best = (key, row, layers)
    # quantization and jitter on the best model, 16 draws, K = 64
    key, _, layers = best
    qrows: dict[str, object] = {"model": key}
    for rule, cap in (("split", None), ("absolute", 2.5)):
        for bits in (3, 4, 6, 8):
            for jit in (0.0, 0.1):
                models, gains = [], []
                for li, lay in enumerate(layers):
                    r = ConstraintRequest(
                        model_id=f"probe2 {key} layer {li}",
                        bits=bits,
                        cap_rule=rule,
                        cap=cap,
                        jitter=jit,
                        jitter_words=(SEED, 2, li) if jit else None,
                        relaxed=() if jit else ("beta_jitter",),
                        reasons={} if jit else {"beta_jitter": "noise-free arm"},
                    )
                    mq, _ = quantize(lay.model(), r)
                    models.append(mq)
                    g = np.ones(lay.n)
                    g[lay.free] = static_gains(r.jitter_words, [len(lay.free)], jit)[0]
                    gains.append(g)
                xs = np.stack([denoise(layers, y, 64, rng, models, gains) for _ in range(16)])
                qrows[f"{rule}_b{bits}_j{jit}"] = metrics(xs, test)
    out["quantized_K64"] = qrows
    print(json.dumps(qrows), flush=True)
    return out


def headroom(rng) -> dict:
    """Noisy-input error against the exact Bayes MPM error over a small grid of the
    data family's (J, h0) and the noise rate q, L = 8, 200 exact test images each."""
    out = {}
    for J in (0.35, 0.45, 0.55):
        for h0 in (0.2, 0.35, 0.5):
            mix = Mixture(8, J, h0)
            x = mix.sample(200, rng)
            for q in (0.1, 0.2, 0.3):
                y = flip(x, q, rng)
                post = mix.posterior(y, q)
                mpm = (np.where(post > 0.5, 1.0, -1.0) != x).mean(1)
                noisy = (y != x).mean(1)
                gap = noisy - mpm
                out[f"J{J}_h{h0}_q{q}"] = {
                    "noisy_error": float(noisy.mean()),
                    "bayes_mpm_error": float(mpm.mean()),
                    "headroom": float(gap.mean()),
                    "headroom_se": float(gap.std(ddof=1) / np.sqrt(len(gap))),
                    "loglik_per_pixel": float(mix.loglik(x).mean() / 64),
                    "pixel_mean_abs": float(np.abs(x.mean(0)).mean()),
                }
    print(json.dumps(out), flush=True)
    return out


# ----------------------------------------------------------------------- main


def main() -> None:
    t_start = time.process_time()
    rng = np.random.default_rng(SEED)
    res: dict = {
        "label": "exploration, not evidence",
        "seed": SEED,
        "python": sys.version.split()[0],
    }

    # 1. pricing (instant)
    res["pricing"] = {
        f"L{L}_T{T}_K{K}": price_paths(L, T, K)
        for L in (8, 16)
        for T in (1, 2, 4)
        for K in (16, 64, 256, 1024, 4096)
    }

    # 2. exact reference: data family, likelihood, Bayes floors
    L, J, h0, q = 8, 0.35, 0.35, 0.2
    mix = Mixture(L, J, h0)
    t0 = time.process_time()
    train = mix.sample(2000, rng)
    test = mix.sample(200, rng)
    t_sample = time.process_time() - t0
    y = flip(test, q, rng)
    t0 = time.process_time()
    post = mix.posterior(y, q)
    t_post = (time.process_time() - t0) / len(y)
    mpm = np.where(post > 0.5, 1.0, -1.0)
    exact = {
        "L": L,
        "J": J,
        "h0": h0,
        "q": q,
        "components": 4,
        "test_loglik_per_pixel": float(mix.loglik(test).mean() / (L * L)),
        "pixel_mean_abs": float(np.abs(test.mean(0)).mean()),
        "noisy_error": float((y != test).mean()),
        "bayes_mpm_error": float((mpm != test).mean()),
        "bayes_sample_error": float(np.where(test > 0, 1 - post, post).mean()),
        "cpu_s_exact_posterior_per_image": t_post,
        "cpu_s_exact_sample_2200_images": t_sample,
    }
    # L = 16 cost of the exact reference (one image, one component)
    g16 = GridIsing(16, J)
    h16 = h0 * templates(16).reshape(4, -1)[:1]
    t0 = time.process_time()
    g16.marginals(h16)
    exact["cpu_s_exact_marginals_L16_one_component"] = time.process_time() - t0
    res["exact"] = exact
    print(json.dumps(exact, indent=1), flush=True)

    # 3. lattice check
    lay = Layer(L, 1.0, rng)
    model = lay.model()
    req = ConstraintRequest(model_id="probe layer", bits=6, cap_rule="split")
    require_placement(model, req)
    rep = check(model)
    deg = model.degrees()
    res["layout"] = {
        "L": L,
        "nodes": model.nodes,
        "free": len(lay.free),
        "clamped": len(lay.clamps),
        "edges": rep.edges,
        "ok": rep.ok,
        "max_degree": rep.max_degree,
        "degree_by_role": {
            r: [
                int(deg[[k for k, x in enumerate(lay.role) if x == r]].min()),
                int(deg[[k for k, x in enumerate(lay.role) if x == r]].max()),
            ]
            for r in ("pix", "clamp", "A", "B")
        },
    }
    print(json.dumps(res["layout"]), flush=True)

    # 4. train DTMs and evaluate (software_simulation, float64 numpy)
    res["dtm"] = {}
    K_train, steps = 16, 400
    for T in (1, 2, 4):
        t0 = time.process_time()
        layers, p, gamma = train_dtm(L, T, q, train, steps, 64, K_train, rng)
        t_train = time.process_time() - t0
        row = {
            "p_step": p,
            "gamma": gamma,
            "cpu_s_train": t_train,
            "K_train": K_train,
            "steps_per_layer": steps,
        }
        for K in (4, 16, 64):
            t0 = time.process_time()
            xs = np.stack([denoise(layers, y, K, rng) for _ in range(8)])
            dt = (time.process_time() - t0) / (8 * len(y))
            row[f"K{K}"] = {
                "sample_error": float((xs != test[None]).mean()),
                "majority8_error": float((np.where(xs.mean(0) >= 0, 1.0, -1.0) != test).mean()),
                "cpu_s_per_sample": dt,
            }
        # quantization and jitter at K = K_train
        qrows = {}
        for rule, cap in (("split", None), ("absolute", 2.0)):
            for bits in (4, 6, 8):
                for jit in (0.0, 0.1):
                    models, gains = [], []
                    summ = None
                    for li, lay in enumerate(layers):
                        relaxed = () if jit else ("beta_jitter",)
                        r = ConstraintRequest(
                            model_id=f"probe T{T} layer {li}",
                            bits=bits,
                            cap_rule=rule,
                            cap=cap,
                            jitter=jit,
                            jitter_words=(SEED, T, li) if jit else None,
                            relaxed=relaxed,
                            reasons={"beta_jitter": "noise-free arm"} if not jit else {},
                        )
                        mq, summ = quantize(lay.model(), r)
                        models.append(mq)
                        g = np.ones(lay.n)
                        g[lay.free] = static_gains(r.jitter_words, [len(lay.free)], jit)[0]
                        gains.append(g)
                    xs = np.stack(
                        [denoise(layers, y, K_train, rng, models, gains) for _ in range(4)]
                    )
                    qrows[f"{rule}_b{bits}_j{jit}"] = {
                        "sample_error": float((xs != test[None]).mean()),
                        "coupling_cap": summ["coupling_cap"],
                        "field_cap": summ["field_cap"],
                    }
        row["quantized_K16"] = qrows
        row["max_abs_w"] = float(max(np.abs(lay.w).max() for lay in layers))
        row["max_abs_b"] = float(max(np.abs(lay.b).max() for lay in layers))
        res["dtm"][f"T{T}"] = row
        print(f"T{T}", json.dumps(row), flush=True)

    # 5. conventional baseline: logistic regression on a 5x5 patch plus position bias
    t0 = time.process_time()
    w = train_logistic(L, q, train, rng)
    t_tr = time.process_time() - t0
    t0 = time.process_time()
    pr = expit(patch_features(y, L) @ w).reshape(len(y), -1)
    t_inf = (time.process_time() - t0) / len(y)
    res["logistic"] = {
        "params": int(w.size),
        "flops_per_image": int(2 * w.size * L * L),
        "mpm_error": float((np.where(pr > 0.5, 1.0, -1.0) != test).mean()),
        "sample_error": float(np.where(test > 0, 1 - pr, pr).mean()),
        "cpu_s_train": t_tr,
        "cpu_s_per_image": t_inf,
    }
    print(json.dumps(res["logistic"]), flush=True)

    # 6. L = 16 sizing: one layer, training step and inference sweep timings
    lay16 = Layer(16, 1.0, rng)
    x16 = rng.integers(0, 2, (64, 256)) * 2.0 - 1.0
    t0 = time.process_time()
    lay16.run(lay16.model(), x16, 16, rng, fixed=x16, stats_from=8)
    lay16.run(lay16.model(), x16, 16, rng, stats_from=8)
    res["sizing_L16"] = {
        "cpu_s_per_train_step_batch64_K16": time.process_time() - t0,
        "nodes": lay16.n,
        "free": len(lay16.free),
    }
    # 7. stage 2: what bounds quality? training budget and K_train, with SEs
    res["stage2"] = stage2(mix, train, test, y, post, q, rng)
    # 8. task headroom of the exact data family (separate generator; stages above unchanged)
    res["headroom"] = headroom(np.random.default_rng([SEED, 8]))
    res["cpu_s_total"] = time.process_time() - t_start
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("total cpu s", res["cpu_s_total"])


if __name__ == "__main__":
    main()
