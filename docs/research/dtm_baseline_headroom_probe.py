"""Exploration probe for loop proposal P-0007 (`dtm-baseline-headroom`).

Exact scan, no EBM. For each candidate denoising family (a mixture of
nearest-neighbour Ising grids with per-component fields, flipped with rate q)
compute the exact Bayes MPM pixel error by transfer matrix, then fit a declared
digital ladder (window logistics and small convolutional nets) and record each
model's error, parameter count and multiply-accumulates per image. The
question is whether any family leaves a gap of at least 0.01 between the Bayes
floor and every cheap ladder model.

Evidence classes: the data model, test set and Bayes posterior are
``exact_reference`` (float64 NumPy). Ladder errors are ``software_simulation``
(float32 JAX on CPU). Joules are ``calibrated_projection``: the DTM's from the
sealed Z1 profile via ``thermo_lab.dtm_small_image.price_sample``, the
baseline's from MACs at a stated pJ per MAC. Nothing here ran on hardware.
This is exploration, not evidence.

Run (CPU, single-thread BLAS):
    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run python docs/research/dtm_baseline_headroom_probe.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from thermo_lab.dtm_small_image import GridIsing, flip, price_sample, templates

PROBE_SEED = 20261016  # never reused by a protocol
DRAWS = 16  # P-0006 estimate = majority of 16 samples
PJ_PER_MAC = (0.1, 1.0, 10.0)


# ---------------------------------------------------------------------------
# Families: exact mixture of grids with arbitrary per-component fields
# ---------------------------------------------------------------------------


class FieldMixture:
    """Uniform mixture of GridIsing(L, J) components with fields ``h`` (C, L*L)."""

    def __init__(self, L: int, J: float, h: np.ndarray):
        self.L, self.grid, self.h = L, GridIsing(L, J), np.asarray(h, dtype=float)
        self.logz = self.grid.forward(self.h)[0]

    def sample(self, n: int, rng) -> np.ndarray:
        comp = rng.integers(0, len(self.h), n)
        out = np.empty((n, self.L * self.L))
        for c in range(len(self.h)):
            idx = np.flatnonzero(comp == c)
            if len(idx):
                out[idx] = self.grid.sample(np.repeat(self.h[c : c + 1], len(idx), 0), rng)
        return out

    def posterior(self, y: np.ndarray, q: float, chunk: int = 256) -> np.ndarray:
        g = 0.5 * np.log((1 - q) / q)
        res = []
        for i in range(0, len(y), chunk):
            yy = y[i : i + chunk]
            lw, pm = [], []
            for hc, lz in zip(self.h, self.logz, strict=True):
                p, lzy = self.grid.marginals(hc[None] + g * yy)
                lw.append(lzy - lz)
                pm.append(p)
            lw = np.stack(lw)
            w = np.exp(lw - logsumexp(lw, axis=0))
            res.append(np.einsum("cb,cbn->bn", w, np.stack(pm)))
        return np.concatenate(res)


def smooth_templates(L: int, n: int, rng, width: float) -> np.ndarray:
    """``n`` +-1 templates: sign of a Gaussian-smoothed white field (fixed rng)."""
    r, c = np.mgrid[0:L, 0:L]
    out = []
    for _ in range(n):
        z = rng.normal(size=(L, L))
        f = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                f += z[i, j] * np.exp(-((r - i) ** 2 + (c - j) ** 2) / (2 * width**2))
        out.append(np.where(f >= 0, 1.0, -1.0))
    return np.stack(out).reshape(n, -1)


def parity_fields(L: int) -> np.ndarray:
    """Four components (a, b): field a on the top-left quadrant, b on bottom-right,
    a*b on the other two. A pixel's template sign depends on two distant quadrants."""
    m = L // 2
    out = []
    for a in (1.0, -1.0):
        for b in (1.0, -1.0):
            f = np.empty((L, L))
            f[:m, :m], f[m:, m:] = a, b
            f[:m, m:] = f[m:, :m] = a * b
            out.append(f.ravel())
    return np.stack(out)


def corner_key_fields(L: int, n: int, rng) -> np.ndarray:
    """``n`` components share a random smooth body sign pattern up to a global
    sign chosen by a 2 x 2 key in one corner: the key decides the body's sign."""
    body = smooth_templates(L, 1, rng, L / 4)[0].reshape(L, L)
    out = []
    for k in range(n):
        s = 1.0 if k % 2 == 0 else -1.0
        f = s * body.copy()
        key = 1.0 if (k // 2) % 2 == 0 else -1.0
        f[:2, :2] = key * s  # corner encodes the body sign (with a second bit)
        out.append(f.ravel())
    return np.stack(out)


def family_specs(L: int):
    rng = np.random.default_rng([PROBE_SEED, L, 1])
    t4 = templates(L).reshape(4, -1)
    specs = [
        ("anchor_p0006", 0.55, 0.2, 0.2, t4),
        ("anchor_q030", 0.55, 0.2, 0.30, t4),
        ("anchor_q035", 0.55, 0.2, 0.35, t4),
        ("anchor_q040", 0.55, 0.2, 0.40, t4),
        ("t4_h05_q035", 0.35, 0.5, 0.35, t4),
        ("smooth8_h05_q035", 0.35, 0.5, 0.35, smooth_templates(L, 8, rng, L / 6)),
        ("smooth16_h05_q035", 0.35, 0.5, 0.35, smooth_templates(L, 16, rng, L / 6)),
        ("parity_J03_h06_q035", 0.30, 0.6, 0.35, parity_fields(L)),
        ("parity_J055_h03_q030", 0.55, 0.3, 0.30, parity_fields(L)),
        ("cornerkey_J03_h06_q035", 0.30, 0.6, 0.35, corner_key_fields(L, 4, rng)),
        ("zerofield_J060_q035", 0.60, 0.0, 0.35, np.ones((1, L * L))),
    ]
    return [
        {"name": n, "J": J, "h0": h0, "q": q, "fields": h0 * f, "components": len(f)}
        for n, J, h0, q, f in specs
    ]


# ---------------------------------------------------------------------------
# Digital ladder (JAX, float32, CPU)
# ---------------------------------------------------------------------------


def ladder(L: int):
    """(name, hidden widths, kernel per layer). Output layer is one channel."""
    full = 2 * L - 1
    return [
        ("logistic_7x7", (), (7,)),
        ("logistic_11x11", (), (11,)),
        (f"logistic_full_{full}x{full}", (), (full,)),
        ("cnn1_w4_k5", (4,), (5, 5)),
        ("cnn1_w16_k5", (16,), (5, 5)),
        ("cnn2_w8_k5", (8, 8), (5, 5, 5)),
        ("cnn2_w32_k5", (32, 32), (5, 5, 5)),
        # Sensitivity rung outside the row's declared ladder: the same 8-wide
        # net with each hidden layer's channel means broadcast back as extra
        # inputs (global pooling), the cheapest digital way to see the whole image.
        ("cnn2_w8_k5_gpool", (8, 8), (5, 5, 5)),
    ]


def in_chans(widths, gpool):
    chans = (1, *widths, 1)
    return [c * 2 if (gpool and i > 0) else c for i, c in enumerate(chans[:-1])], chans[1:]


def init_params(L, widths, ks, key, gpool=False):
    cin, cout = in_chans(widths, gpool)
    params = []
    for i, k in enumerate(ks):
        key, sub = jax.random.split(key)
        fan = k * k * cin[i]
        w = jax.random.normal(sub, (k, k, cin[i], cout[i])) * (1.0 / np.sqrt(fan))
        if i == len(ks) - 1 and not widths:
            w = w * 0.0
        params.append({"w": w, "b": jnp.zeros(cout[i])})
    return {"layers": params, "pos": jnp.zeros((L, L))}


def forward(params, y, L, gpool=False):
    x = y.reshape(-1, L, L, 1)
    n = len(params["layers"])
    for i, p in enumerate(params["layers"]):
        if gpool and i > 0:
            x = jnp.concatenate([x, jnp.broadcast_to(x.mean((1, 2), keepdims=True), x.shape)], -1)
        x = jax.lax.conv_general_dilated(
            x, p["w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        x = x + p["b"]
        if i < n - 1:
            x = jax.nn.relu(x)
    return x[..., 0] + params["pos"]


def counts(L, widths, ks, gpool=False):
    """Parameters (with the per-pixel bias map) and MACs per image (conv only;
    pooled channels counted as full conv inputs, an upper bound)."""
    cin, cout = in_chans(widths, gpool)
    macs = sum(L * L * k * k * cin[i] * cout[i] for i, k in enumerate(ks))
    params = sum(k * k * cin[i] * cout[i] + cout[i] for i, k in enumerate(ks))
    return int(params + L * L), int(macs)


def train_model(L, widths, ks, q, pool, steps, batch, lr, seed, gpool=False):
    key = jax.random.PRNGKey(seed)
    params = init_params(L, widths, ks, key, gpool)
    rng = np.random.default_rng([PROBE_SEED, seed, L])

    def loss(p, y, t):
        z = forward(p, y, L, gpool)
        return jnp.mean(jax.nn.softplus(z) - t * z)

    @jax.jit
    def step(p, m, v, it, y, t):
        g = jax.grad(loss)(p, y, t)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        c1, c2 = 1 - 0.9**it, 1 - 0.999**it
        p = jax.tree_util.tree_map(
            lambda a, mm, vv: a - lr * (mm / c1) / (jnp.sqrt(vv / c2) + 1e-8), p, m, v
        )
        return p, m, v

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    for it in range(1, steps + 1):
        x = pool[rng.integers(0, len(pool), batch)]
        y = flip(x, q, rng)
        t = (x.reshape(-1, L, L) + 1) / 2
        params, m, v = step(
            params, m, v, float(it), jnp.asarray(y, jnp.float32), jnp.asarray(t, jnp.float32)
        )
    return params


def errors(params, L, x, y, gpool=False):
    f = jax.jit(lambda p, a: forward(p, a, L, gpool))
    z = np.asarray(f(params, jnp.asarray(y, jnp.float32)))
    pred = np.where(z.reshape(len(y), -1) > 0, 1.0, -1.0)
    per_img = (pred != x).mean(1)
    return per_img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(L, n_test, n_pool, steps, batch, only=None, ladder_only=None, seed=1):
    out = {"L": L, "n_test": n_test, "n_pool": n_pool, "steps": steps, "batch": batch}
    out["ladder_seed"] = seed
    dtm = price_sample(L, 2, 64)
    out["dtm_operating_point"] = {
        "point": "P-0006 T=2, 4 bits split, jitter 0.1, K=64, host per step",
        "per_sample": dtm,
        "per_estimate_j_write_constant": DRAWS * dtm["energy_j"],
        "per_estimate_j_clamp_at_read_constant": DRAWS
        * dtm["energy_j_clamp_loads_at_read_constant_outside_profile"],
        "evidence_class": "calibrated_projection",
    }
    hi = out["dtm_operating_point"]["per_estimate_j_write_constant"]
    lo = out["dtm_operating_point"]["per_estimate_j_clamp_at_read_constant"]
    out["break_even_macs_per_image"] = {
        f"{pj}pJ": {"write_constant": hi / (pj * 1e-12), "clamp_at_read": lo / (pj * 1e-12)}
        for pj in PJ_PER_MAC
    }
    fams = []
    for fi, spec in enumerate(family_specs(L)):
        if only and spec["name"] not in only:
            continue
        t0 = time.process_time()
        mix = FieldMixture(L, spec["J"], spec["fields"])
        rng = np.random.default_rng([PROBE_SEED, L, 1000 + fi])
        x = mix.sample(n_test, rng)
        y = flip(x, spec["q"], rng)
        post = mix.posterior(y, spec["q"])
        bayes_img = (np.where(post > 0.5, 1.0, -1.0) != x).mean(1)
        noisy_img = (y != x).mean(1)
        pool = mix.sample(n_pool, np.random.default_rng([PROBE_SEED, L, 99]))
        exact_cpu = time.process_time() - t0
        rec = {
            "name": spec["name"],
            "J": spec["J"],
            "h0": spec["h0"],
            "q": spec["q"],
            "components": spec["components"],
            "noisy": float(noisy_img.mean()),
            "bayes_mpm": float(bayes_img.mean()),
            "exact_cpu_s": exact_cpu,
            "ladder": [],
        }
        for name, widths, ks in ladder(L):
            if ladder_only and name not in ladder_only:
                continue
            t1 = time.process_time()
            lr = 0.05 if not widths else 0.01
            gp = name.endswith("_gpool")
            params = train_model(L, widths, ks, spec["q"], pool, steps, batch, lr, seed, gp)
            e = errors(params, L, x, y, gp)
            d = e - bayes_img
            n_params, macs = counts(L, widths, ks, gp)
            rec["ladder"].append(
                {
                    "model": name,
                    "declared_ladder": not gp,
                    "error": float(e.mean()),
                    "gap_to_bayes": float(d.mean()),
                    "gap_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                    "params": n_params,
                    "macs_per_image": macs,
                    "joule_per_image": {f"{pj}pJ": macs * pj * 1e-12 for pj in PJ_PER_MAC},
                    "cpu_s": time.process_time() - t1,
                }
            )
            print(
                spec["name"], name, round(float(e.mean()), 4), round(float(d.mean()), 4), flush=True
            )
        within = [m for m in rec["ladder"] if m["declared_ladder"] and m["gap_to_bayes"] <= 0.01]
        rec["smallest_within_0p01"] = (
            min(within, key=lambda m: m["macs_per_image"])["model"] if within else None
        )
        rec["smallest_within_0p01_macs"] = (
            min(m["macs_per_image"] for m in within) if within else None
        )
        rec["best_gap_declared"] = min(
            (m["gap_to_bayes"] for m in rec["ladder"] if m["declared_ladder"]), default=None
        )
        rec["best_gap_any"] = min(m["gap_to_bayes"] for m in rec["ladder"])
        print(
            spec["name"],
            "noisy",
            rec["noisy"],
            "bayes",
            rec["bayes_mpm"],
            "smallest",
            rec["smallest_within_0p01"],
            flush=True,
        )
        fams.append(rec)
    out["families"] = fams
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-pool", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--ladder-only", nargs="*")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    t0 = time.process_time()
    res = run(a.L, a.n_test, a.n_pool, a.steps, a.batch, a.only, a.ladder_only, a.seed)
    res["probe_seed"] = PROBE_SEED
    res["label"] = "exploration, not evidence"
    res["evidence_classes"] = {
        "bayes_mpm": "exact_reference",
        "ladder_errors": "software_simulation (float32 JAX, CPU)",
        "joules": "calibrated_projection",
    }
    res["total_cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
