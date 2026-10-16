"""DTM baseline headroom (proposal P-0007, queue row ``dtm-baseline-headroom``).

Protocol: ``docs/experiments/dtm-baseline-headroom.md`` (approved at commit
1c91073). Six binary denoising families at L = 8 with an exact transfer-matrix
posterior. Stage 1 fits a nine-rung digital ladder (float32 JAX, CPU) on every
family, with the learning rate chosen by validation at ladder seed 0 and then
three seeds at that rate. A family is a candidate when no rung below X MACs per
image reaches the Bayes MPM error within 0.01. Stage 2 trains P-0006's lattice
denoiser (T = 2, imported unchanged from ``thermo_lab.dtm_small_image``) on at
most three candidates, three seeds each, and runs it at P-0006's operating
point (4 bits under ``split``, 10 percent jitter, K = 64, host-per-step,
continuation) under the constraint layer, with a float K = 1024 reference, a
random-start control and a continuation control.

The family generators, the ladder trainer and its error function are the
probe's (``docs/research/dtm_baseline_headroom_probe.py``), copied here so the
import package does not depend on ``docs/``; the unit tests check them bitwise
against the probe. The data model, DTM trainer, NumPy sampler, THRML runner,
lowering, quantization, jitter, placement and pricing are imported from
``thermo_lab.dtm_small_image`` and ``thermo_lab.constraints`` unchanged.

Long run (about 3 h on 3 workers): every work unit (pool, exact reference,
ladder fit, trained DTM, inference cell) is written atomically to ``units/``
with a digest in ``units/index.json``; ``--resume`` re-authenticates the request
and sources and runs only the missing units. Each worker process is pinned to
one CPU, which makes the float32 ladder fits bitwise reproducible on this box.

Evidence classes: mixture samples, posterior marginals and Bayes MPM errors are
``exact_reference`` (float64). The ladder (float32 JAX), THRML and NumPy DTM runs
and DTM training are ``software_simulation`` (CPU). Every joule, on both sides,
is a ``calibrated_projection``. Nothing here ran on hardware.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import multiprocessing
import os
import shutil
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from thermo_lab import dtm_small_image as dtm
from thermo_lab.constraints import ConstraintRequest, check, require_placement
from thermo_lab.dtm_small_image import (
    READ_J,
    WRITE_J,
    GridIsing,
    LoweredLayer,
    brute_force_grid,
    flip,
    make_thrml_runner,
    price_sample,
    read_gz_json,
    sweep_from,
    templates,
    write_gz_json,
)
from thermo_lab.hardware.z1 import Z1HardwareProfile
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.planar_ising_scaling import check_equal
from thermo_lab.provenance import collect_runtime_provenance

ROOT = Path(__file__).resolve().parents[2]
MODULE = "src/thermo_lab/dtm_baseline_headroom.py"
SOURCES = (
    MODULE,
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
    "docs/research/dtm_baseline_headroom_probe.py",
    "docs/experiments/dtm-baseline-headroom.md",
    "uv.lock",
)
PROTOCOL_COMMIT = "1c91073b9fefc0944f1d0ac8adcfacdec07a02a0"
PROBE_SEED = 20261016
DEFAULT_WORKERS = 3
TOL = 0.01
PJ_PER_MAC = (0.1, 1.0, 10.0)
X_MACS_ROUNDED = 320826
FAMILY_TABLE = (
    ("anchor_p0006", 0.55, 0.2, 0.2),
    ("t4_h05_q035", 0.35, 0.5, 0.35),
    ("smooth8_h05_q035", 0.35, 0.5, 0.35),
    ("smooth16_h05_q035", 0.35, 0.5, 0.35),
    ("parity_J03_h06_q035", 0.30, 0.6, 0.35),
    ("cornerkey_J03_h06_q035", 0.30, 0.6, 0.35),
)
ANCHOR = "anchor_p0006"
CELL_KINDS = ("op", "float_ref", "numpy_random", "numpy_continue")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()
    ).hexdigest()


def write(path, value):
    atomic_write_text(Path(path), canonical_json(value) + "\n")


# ----------------------------------------------------------------------------
# Families (exact reference): the probe's generators, copied, checked bitwise
# ----------------------------------------------------------------------------


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
    body = smooth_templates(L, 1, rng, L / 4)[0].reshape(L, L)
    out = []
    for k in range(n):
        s = 1.0 if k % 2 == 0 else -1.0
        f = s * body.copy()
        key = 1.0 if (k // 2) % 2 == 0 else -1.0
        f[:2, :2] = key * s
        out.append(f.ravel())
    return np.stack(out)


def family_patterns(L: int) -> dict[str, np.ndarray]:
    """Unit field patterns per family name, drawn in the probe's rng order."""
    rng = np.random.default_rng([PROBE_SEED, L, 1])
    t4 = templates(L).reshape(4, -1)
    s8 = smooth_templates(L, 8, rng, L / 6)
    s16 = smooth_templates(L, 16, rng, L / 6)
    ck = corner_key_fields(L, 4, rng)
    return {
        "anchor_p0006": t4,
        "t4_h05_q035": t4,
        "smooth8_h05_q035": s8,
        "smooth16_h05_q035": s16,
        "parity_J03_h06_q035": parity_fields(L),
        "cornerkey_J03_h06_q035": ck,
    }


def family_fields(L, table=FAMILY_TABLE):
    patterns = family_patterns(L)
    out = []
    for i, (name, J, h0, q) in enumerate(table):
        fields = h0 * patterns[name]
        out.append(
            {
                "index": i,
                "name": name,
                "J": J,
                "h0": h0,
                "q": q,
                "components": int(len(fields)),
                "fields": fields.tolist(),
                "fields_sha256": array_sha(fields),
            }
        )
    return out


def mixture(request, f):
    fam = request["families"][f]
    return FieldMixture(request["L"], fam["J"], np.asarray(fam["fields"], dtype=np.float64))


# ----------------------------------------------------------------------------
# Digital ladder (float32 JAX on CPU): the probe's model, copied, checked bitwise
# ----------------------------------------------------------------------------


def rung_table(L: int):
    full = 2 * L - 1
    return [
        ("logistic_7x7", (), (7,), False),
        ("logistic_11x11", (), (11,), False),
        (f"logistic_full_{full}x{full}", (), (full,), False),
        ("cnn1_w4_k5", (4,), (5, 5), False),
        ("cnn1_w16_k5", (16,), (5, 5), False),
        ("cnn2_w8_k5", (8, 8), (5, 5, 5), False),
        ("cnn2_w32_k5", (32, 32), (5, 5, 5), False),
        ("cnn2_w8_k5_gpool", (8, 8), (5, 5, 5), True),
        ("cnn2_w32_k5_gpool", (32, 32), (5, 5, 5), True),
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
    """(parameters with the per-pixel bias map, conv MACs per image) as the probe."""
    cin, cout = in_chans(widths, gpool)
    macs = sum(L * L * k * k * cin[i] * cout[i] for i, k in enumerate(ks))
    params = sum(k * k * cin[i] * cout[i] + cout[i] for i, k in enumerate(ks))
    return int(params + L * L), int(macs)


def hand_counts(L, widths, ks, gpool):
    """Independent hand count: per conv layer, (cin, cout) written out explicitly."""
    if not widths:
        layers = [(1, 1)]
    elif len(widths) == 1:
        layers = [(1, widths[0]), (widths[0], 1)]
    else:
        a, b = widths
        layers = [(1, a), (2 * a if gpool else a, b), (2 * b if gpool else b, 1)]
    macs = 0
    params = L * L
    for (cin, cout), k in zip(layers, ks, strict=True):
        macs += L * L * k * k * cin * cout
        params += k * k * cin * cout + cout
    return params, macs


def train_model(L, widths, ks, q, pool, steps, batch, lr, key, rng, gpool=False):
    """The probe's ``train_model`` with the init key and batch rng as arguments."""
    params = init_params(L, widths, ks, key, gpool)

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


def predict(params, L, y, gpool=False):
    f = jax.jit(lambda p, a: forward(p, a, L, gpool))
    z = np.asarray(f(params, jnp.asarray(y, jnp.float32)))
    return np.where(z.reshape(len(y), -1) > 0, 1.0, -1.0)


def errors(params, L, x, y, gpool=False):
    """Per-image fraction of wrong pixels (the probe's ``errors``)."""
    return (predict(params, L, y, gpool) != x).mean(1)


# ----------------------------------------------------------------------------
# Request
# ----------------------------------------------------------------------------


def cell_codes():
    return {k: i for i, k in enumerate(CELL_KINDS)}


def make_request(**overrides):
    """The frozen production request; tests pass smaller geometry through ``overrides``."""
    L = overrides.pop("L", 8)
    table = overrides.pop("family_table", FAMILY_TABLE)
    rung_names = overrides.pop("rungs", None)
    rungs = []
    for i, (name, widths, ks, gpool) in enumerate(rung_table(L)):
        if rung_names is not None and name not in rung_names:
            continue
        n_params, macs = counts(L, widths, ks, gpool)
        rungs.append(
            {
                "index": i,
                "name": name,
                "widths": list(widths),
                "kernels": list(ks),
                "global_pooling": gpool,
                "params": n_params,
                "macs_per_image": macs,
            }
        )
    request = {
        "schema": "dtm_baseline_headroom.v1",
        "status": "exploratory",
        "protocol": "docs/experiments/dtm-baseline-headroom.md",
        "protocol_commit": PROTOCOL_COMMIT,
        "proposal": "P-0007",
        "L": L,
        "families": family_fields(L, table),
        "anchor": ANCHOR,
        "n_pool": 20000,
        "n_val": 1000,
        "n_test": 1000,
        "ladder": {
            "rungs": rungs,
            "rates": [0.003, 0.01, 0.03],
            "seeds": [0, 1, 2],
            "steps": 6000,
            "batch": 128,
            "optimizer": "Adam (0.9, 0.999, 1e-8), logistic loss, fresh flip noise every batch",
            "selection": "rate by validation error at seed 0 (ties to the first rate); then "
            "seeds 1 and 2 at that rate; the rung's error is the test error of the seed with "
            "the lowest validation error (ties to the lowest seed)",
            "failed_to_train": "best-seed validation error above the noisy input's",
            "macs": "conv MACs per image; pooled channels counted as full conv inputs; "
            "activations, bias map and sigmoid not counted",
            "dtype": "float32",
        },
        "dtm": {
            "seeds": [0, 1, 2],
            "T": 2,
            "n_train": 4000,
            "training_data": "the first n_train images of the seed-s pool",
            "lr": 0.03,
            "batch": 64,
            "K_train": 64,
            "steps_per_layer": 1600,
            "operating_point": {"bits": 4, "cap_rule": "split", "jitter": 0.1, "K": 64},
            "reference_sweeps": 1024,
            "draws": 16,
            "streams": 50,
            "images_per_stream": 20,
            "max_candidates": 3,
            "cells": list(CELL_KINDS),
            "update_rule": "active_kernel",
            "dtype": "float64",
            "io_path": "host per step, continuation; random-start control counts 3 L^2 T writes",
        },
        "relaxed_reasons": {
            "quantization": "float K = 1024 arm: the convergence reference",
            "beta_jitter": "float K = 1024 arm: the convergence reference",
        },
        "seeds": {
            "pool": [20261201],
            "validation": [20261202],
            "test": [20261203],
            "ladder_batches": [20261204],
            "ladder_init": 20261205,
            "dtm_training": [20261206],
            "thrml_root": 20261207,
            "jitter_words": [20261208],
            "stream_states": [20261209],
            "bootstrap": [20261210],
            "rules": {
                "pool": "default_rng([20261201, f, s]).sample(n_pool) per family f and seed s; "
                "ladder seed s draws its batches from pool (f, s); DTM seed s trains on the "
                "first n_train images of pool (f, s)",
                "ladder_batches": "default_rng([20261204, f, rung, seed, rate_index])",
                "ladder_init": "PRNGKey(20261205) folded with f, rung, seed in that order",
                "dtm_training": "default_rng([20261206, f, s, layer]) per layer",
                "thrml_keys": "fold_in(key(20261207), f, s, cell, image_position, layer)",
                "jitter_words": "(20261208, f, s, layer)",
                "stream_states": "default_rng([20261209, f, s, cell]); THRML draws one "
                "uniform full-layer state per layer (layer 0 first); NumPy cells pass it to "
                "dtm_small_image.numpy_streams",
                "bootstrap": "default_rng([20261210, f, s]) streams for DTM comparisons; "
                "default_rng([20261210, f]) images for ladder-only comparisons",
                "cell_codes": cell_codes(),
            },
        },
        "bootstrap": {"resamples": 2000, "interval": "95 percent percentile, paired"},
        "tolerance": TOL,
        "pj_per_mac": list(PJ_PER_MAC),
        "x_rule": "X = DTM joule per 16-draw estimate at the profile write constant / 1 pJ",
        "x_macs_rounded": X_MACS_ROUNDED,
        "sample": (
            "DTM: one chain's pixel state after two steps of 64 sweeps from the noisy image, "
            "continuing each layer region's state from the previous image of its stream; an "
            "estimate is the pixelwise majority of 16 samples (ties to +1). Ladder: one forward "
            "pass thresholded at 0.5. Bayes: the exact marginal posterior mode per pixel"
        ),
        "profile_id": Z1HardwareProfile().profile_id,
        "profile_hash": Z1HardwareProfile().profile_hash,
        "preflight": {
            "probe_seed": PROBE_SEED,
            "transfer_tolerance": 1e-12,
            "sampler_family": "parity_J03_h06_q035",
            "steps_per_layer": 1600,
            "n_train": 4000,
            "images": 10,
            "chains": 6400,
            "sweeps": 64,
            "z_max": 5.0,
        },
        "cpu_stop_hours": 10.0,
        "replay_rtol": 1e-12,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(request.get(key), dict):
            request[key] = {**request[key], **value}
        else:
            request[key] = value
    request["constraint_requests"] = {
        cr_key(f, s, li, arm): {"digest": (c := arm_request(request, f, s, li, arm)).digest()}
        | {"canonical": c.to_canonical()}
        for f in range(len(request["families"]))
        if request["families"][f]["name"] != request["anchor"]
        for s in request["dtm"]["seeds"]
        for li in range(request["dtm"]["T"])
        for arm in ("op", "float_ref")
    }
    request["sources"] = {name: sha(ROOT / name) for name in SOURCES}
    validate_request(request)
    return request


def cr_key(f, s, li, arm):
    return f"f{f}_s{s}_l{li}_{arm}"


def validate_request(request):
    if request["schema"] != "dtm_baseline_headroom.v1":
        raise ValueError("unknown request schema")
    d = request["dtm"]
    if request["n_test"] != d["streams"] * d["images_per_stream"]:
        raise ValueError("n_test must equal streams x images_per_stream")
    if d["n_train"] > request["n_pool"]:
        raise ValueError("DTM training data is a prefix of the pool")
    if request["profile_hash"] != Z1HardwareProfile().profile_hash:
        raise ValueError("Z1 profile changed")
    if request["tolerance"] != TOL or request["replay_rtol"] != 1e-12:
        raise ValueError("tolerance and replay tolerance are frozen")
    if sum(f["name"] == request["anchor"] for f in request["families"]) != 1:
        raise ValueError("exactly one anchor family")
    if 0 not in request["ladder"]["seeds"]:
        raise ValueError("ladder seed 0 selects the rate")


def arm_request(request, f, s, li, arm):
    op = request["dtm"]["operating_point"]
    if arm == "op":
        return ConstraintRequest(
            model_id=f"dtm-baseline-headroom f{f} s{s} layer{li} op",
            bits=op["bits"],
            cap_rule=op["cap_rule"],
            jitter=op["jitter"],
            jitter_words=(*request["seeds"]["jitter_words"], f, s, li),
            update_rule=request["dtm"]["update_rule"],
            dtype=request["dtm"]["dtype"],
        )
    if arm == "float_ref":
        return ConstraintRequest(
            model_id=f"dtm-baseline-headroom f{f} s{s} layer{li} float_ref",
            update_rule=request["dtm"]["update_rule"],
            dtype=request["dtm"]["dtype"],
            relaxed=("beta_jitter", "quantization"),
            reasons=dict(request["relaxed_reasons"]),
        )
    raise ValueError(f"no constraint request for arm {arm}")


def x_threshold(request):
    """(X in MACs per image, DTM pricing block) from the sealed profile."""
    L, d = request["L"], request["dtm"]
    p = price_sample(L, d["T"], d["operating_point"]["K"])
    draws = d["draws"]
    hi = draws * p["energy_j"]
    lo = draws * p["energy_j_clamp_loads_at_read_constant_outside_profile"]
    x = hi / 1e-12
    return x, {
        "per_sample": p,
        "per_estimate_j_write_constant": hi,
        "per_estimate_j_clamp_loads_at_read_constant_outside_profile": lo,
        "x_macs_per_image": x,
        "break_even_macs_per_image": {
            f"{pj}pJ": {"write_constant": hi / (pj * 1e-12), "clamp_at_read": lo / (pj * 1e-12)}
            for pj in request["pj_per_mac"]
        },
        "random_start_per_sample": price_sample(L, d["T"], d["operating_point"]["K"], "random"),
        "evidence_class": "calibrated_projection",
    }


# ----------------------------------------------------------------------------
# Data and exact references
# ----------------------------------------------------------------------------


def pool(request, f, s):
    rng = np.random.default_rng([*request["seeds"]["pool"], f, s])
    return mixture(request, f).sample(request["n_pool"], rng)


def held_set(request, f, which):
    mix = mixture(request, f)
    rng = np.random.default_rng([*request["seeds"][which], f])
    n = request["n_val"] if which == "validation" else request["n_test"]
    x = mix.sample(n, rng)
    y = flip(x, request["families"][f]["q"], rng)
    return x, y


def exact_reference(request, f):
    mix = mixture(request, f)
    q = request["families"][f]["q"]
    x, y = held_set(request, f, "test")
    xv, yv = held_set(request, f, "validation")
    t0 = time.process_time()
    post = mix.posterior(y, q)
    cpu = time.process_time() - t0
    pb = np.clip(post, 1e-300, 1.0)
    pm = np.clip(1 - post, 1e-300, 1.0)
    return {
        "evidence_class": "exact_reference",
        "test_x_sha256": array_sha(x),
        "test_y_sha256": array_sha(y),
        "validation_x_sha256": array_sha(xv),
        "validation_y_sha256": array_sha(yv),
        "component_log_z": mix.logz.tolist(),
        "noisy_wrong": (y != x).sum(1).astype(int).tolist(),
        "validation_noisy_wrong_total": int((yv != xv).sum()),
        "bayes_mpm_wrong": (np.where(post > 0.5, 1.0, -1.0) != x).sum(1).astype(int).tolist(),
        "bayes_sample_error": float(np.where(x > 0, 1 - post, post).mean()),
        "bayes_logloss": (-np.log(np.where(x > 0, pb, pm))).mean(1).tolist(),
    }, {"exact_posterior_cpu_s_per_image": cpu / len(y)}


# ----------------------------------------------------------------------------
# Work units
# ----------------------------------------------------------------------------


def ladder_key(request, f, r, s):
    key = jax.random.PRNGKey(request["seeds"]["ladder_init"])
    for w in (f, r, s):
        key = jax.random.fold_in(key, w)
    return key


def run_fit(request, f, rung, s, ri, pool_x):
    """One ladder fit: (test wrong-pixel counts, validation wrong total), timings."""
    lad = request["ladder"]
    L = request["L"]
    widths, ks, gp = tuple(rung["widths"]), tuple(rung["kernels"]), rung["global_pooling"]
    rng = np.random.default_rng([*request["seeds"]["ladder_batches"], f, rung["index"], s, ri])
    t0 = time.process_time()
    params = train_model(
        L,
        widths,
        ks,
        request["families"][f]["q"],
        pool_x,
        lad["steps"],
        lad["batch"],
        lad["rates"][ri],
        ladder_key(request, f, rung["index"], s),
        rng,
        gp,
    )
    jax.block_until_ready(params)
    train_cpu = time.process_time() - t0
    xv, yv = held_set(request, f, "validation")
    x, y = held_set(request, f, "test")
    fwd = jax.jit(lambda p, a: forward(p, a, L, gp))

    def hard(a):
        z = np.asarray(fwd(params, jnp.asarray(a, jnp.float32)))
        return np.where(z.reshape(len(a), -1) > 0, 1.0, -1.0)

    val_wrong = int((hard(yv) != xv).sum())
    hard(y)
    t0 = time.process_time()
    pred = hard(y)
    inf_cpu = (time.process_time() - t0) / len(y)
    arrays = {
        "test_wrong": (pred != x).sum(1).astype(np.uint8),
        "val_wrong": np.array([val_wrong], dtype=np.int64),
    }
    return arrays, {"train_cpu_s": train_cpu, "inference_cpu_s_per_image": inf_cpu}


def run_training(request, f, s, pool_x):
    d = request["dtm"]
    data = pool_x[: d["n_train"]]
    arrays, timing = {}, {}
    for li in range(d["T"]):
        rng = np.random.default_rng([*request["seeds"]["dtm_training"], f, s, li])
        t0 = time.process_time()
        lay = dtm.train_layer(
            request["L"],
            d["T"],
            li + 1,
            request["families"][f]["q"],
            data,
            d["steps_per_layer"],
            d["batch"],
            d["K_train"],
            rng,
            d["lr"],
        )
        timing[f"layer{li}_cpu_s"] = time.process_time() - t0
        arrays[f"w{li}"] = lay.w
        arrays[f"b{li}"] = lay.b
        arrays[f"gamma{li}"] = np.array([lay.gamma])
    return arrays, timing


def load_layers(request, arrays):
    return [
        dtm.layer_from(
            request["L"], float(arrays[f"gamma{li}"][0]), arrays[f"w{li}"], arrays[f"b{li}"]
        )
        for li in range(request["dtm"]["T"])
    ]


def thrml_stream_xs(lowered, y, K, draws, streams, per, state_rng, key_root, words):
    """``dtm_small_image.thrml_xs`` with the stream-state rng and chain-key words as
    arguments (the tests check it bitwise against ``thrml_xs``). ``words(j, li)``."""
    rows = draws * streams
    L2 = y.shape[1]
    full = [state_rng.integers(0, 2, (rows, low.n)) * 2.0 - 1.0 for low in lowered]
    state = [
        [jnp.asarray(full[li][:, g] > 0) for g in low.groups] for li, low in enumerate(lowered)
    ]
    root = {"seeds": {"jax_root": key_root}}
    xs = np.empty((draws, len(y), L2))
    compile_cpu = 0.0
    with jax.enable_x64(True):
        fns = []
        for li, low in enumerate(lowered):
            fn = make_thrml_runner(low.program, low.blocks, K)
            t0 = time.process_time()
            keys = dtm.chain_keys(root, words(0, li), rows)
            x0 = jnp.zeros((rows, L2), dtype=bool)
            fn = fn.lower(keys, x0, *state[li]).compile()
            compile_cpu += time.process_time() - t0
            fns.append(fn)
        t0 = time.process_time()
        for j in range(per):
            idx = np.arange(streams) * per + j
            x = np.tile(y[idx], (draws, 1))
            for li in range(len(lowered) - 1, -1, -1):
                keys = dtm.chain_keys(root, words(j, li), rows)
                out = np.asarray(fns[li](keys, jnp.asarray(x > 0), *state[li]).block_until_ready())
                low = lowered[li]
                split = np.cumsum(low.sizes)[:-1]
                state[li] = [jnp.asarray(part) for part in np.split(out, split, axis=1)]
                x = np.where(out[:, low.pix_cols], 1.0, -1.0)
            xs[:, idx] = x.reshape(draws, streams, -1)
        run_cpu = time.process_time() - t0
    return xs, {"compile_cpu_s": compile_cpu, "steady_cpu_s": run_cpu}


class _Lowered(LoweredLayer):
    """``LoweredLayer`` plus the node count the stream state needs."""

    def __init__(self, lay, creq):
        super().__init__(lay, creq)
        self.n = lay.n


def run_cell(request, cell, model_arrays):
    f, s, kind = cell["f"], cell["s"], cell["kind"]
    d = request["dtm"]
    code = cell_codes()[kind]
    x, y = held_set(request, f, "test")
    layers = load_layers(request, model_arrays)
    rng = np.random.default_rng([*request["seeds"]["stream_states"], f, s, code])
    if kind in ("op", "float_ref"):
        lowered = [
            _Lowered(lay, arm_request(request, f, s, li, kind)) for li, lay in enumerate(layers)
        ]
        K = d["operating_point"]["K"] if kind == "op" else d["reference_sweeps"]
        xs, timing = thrml_stream_xs(
            lowered,
            y,
            K,
            d["draws"],
            d["streams"],
            d["images_per_stream"],
            rng,
            request["seeds"]["thrml_root"],
            lambda j, li: (f, s, code, j, li),
        )
    else:
        t0 = time.process_time()
        xs = dtm.numpy_streams(
            layers,
            y,
            d["operating_point"]["K"],
            rng,
            kind.removeprefix("numpy_"),
            d["draws"],
            d["streams"],
            d["images_per_stream"],
        )
        timing = {"compile_cpu_s": 0.0, "steady_cpu_s": time.process_time() - t0}
    return dtm.image_metrics(xs, x), timing


# ----------------------------------------------------------------------------
# Preflight and calibration (probe seeds)
# ----------------------------------------------------------------------------


def brute_mixture_posterior(L, J, fields, y, q):
    grid = GridIsing(L, J)
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=L * L)))
    g = 0.5 * np.log((1 - q) / q)
    terms = []
    for h in fields:
        e = grid.neg_energy(states, h[None])
        terms.append(e - logsumexp(e))
    logp = logsumexp(np.stack(terms), axis=0) - np.log(len(fields))
    lw = logp + g * (states @ y)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return w @ (states > 0)


def transfer_check(request):
    tol = request["preflight"]["transfer_tolerance"]
    rows = []
    for L in (3, 4):
        table = [(fam["name"], fam["J"], fam["h0"], fam["q"]) for fam in request["families"]]
        for fam in family_fields(L, table):
            fields = np.asarray(fam["fields"])
            worst_z, worst_m = 0.0, 0.0
            for h in fields:
                p, lz = GridIsing(L, fam["J"]).marginals(h[None])
                blz, bp = brute_force_grid(L, fam["J"], h)
                worst_z = max(worst_z, abs(lz[0] - blz) / abs(blz))
                worst_m = max(worst_m, float(np.abs(p[0] - bp).max()))
            rng = np.random.default_rng([PROBE_SEED, 301, L, fam["index"]])
            y = rng.integers(0, 2, (3, L * L)) * 2.0 - 1.0
            post = FieldMixture(L, fam["J"], fields).posterior(y, fam["q"])
            brute = np.stack(
                [brute_mixture_posterior(L, fam["J"], fields, yy, fam["q"]) for yy in y]
            )
            rows.append(
                {
                    "L": L,
                    "family": fam["name"],
                    "components": fam["components"],
                    "log_z_rel_error": worst_z,
                    "marginal_max_abs_error": worst_m,
                    "mixture_posterior_max_abs_error": float(np.abs(post - brute).max()),
                }
            )
    passed = all(
        r["log_z_rel_error"] <= tol
        and r["marginal_max_abs_error"] <= tol
        and r["mixture_posterior_max_abs_error"] <= tol
        for r in rows
    )
    return {"rows": rows, "tolerance": tol, "passed": bool(passed)}


def counts_check(request):
    rows = []
    for rung in request["ladder"]["rungs"]:
        args = (request["L"], tuple(rung["widths"]), tuple(rung["kernels"]), rung["global_pooling"])
        hand = hand_counts(*args)
        code = counts(*args)
        good = hand == code == (rung["params"], rung["macs_per_image"])
        rows.append({"rung": rung["name"], "params": hand[0], "macs": hand[1], "passed": good})
    return {"rows": rows, "passed": bool(all(r["passed"] for r in rows))}


def pricing_check(request):
    L, d = request["L"], request["dtm"]
    T, K = d["T"], d["operating_point"]["K"]
    p = price_sample(L, T, K)
    n = L * L
    expected = 3 * n * K * T * 7.09e-15 + n * T * READ_J + n * T * WRITE_J
    low = expected - n * T * WRITE_J + n * T * READ_J
    x, block = x_threshold(request)
    good = (
        p["reads"] == n * T
        and p["writes"] == n * T
        and p["clamp_state_changes"] == 0
        and p["updates"] == 3 * n * K * T
        and abs(p["energy_j"] - expected) <= 1e-12 * expected
        and abs(p["energy_j_clamp_loads_at_read_constant_outside_profile"] - low)
        <= 1e-12 * expected
    )
    if L == 8 and T == 2 and K == 64 and d["draws"] == 16:
        good = good and (p["reads"], p["writes"], p["updates"]) == (128, 128, 24576)
        good = good and round(x) == request["x_macs_rounded"]
    return {
        "reads": p["reads"],
        "writes": p["writes"],
        "updates": p["updates"],
        "x_macs_per_image": x,
        "x_rounded": round(x),
        "per_estimate_j": [
            block["per_estimate_j_clamp_loads_at_read_constant_outside_profile"],
            block["per_estimate_j_write_constant"],
        ],
        "passed": bool(good),
    }


def _probe_layer(request):
    pf = request["preflight"]
    f = next(i for i, fam in enumerate(request["families"]) if fam["name"] == pf["sampler_family"])
    mix = mixture(request, f)
    q = request["families"][f]["q"]
    data = mix.sample(pf["n_train"], np.random.default_rng([PROBE_SEED, 201]))
    t0 = time.process_time()
    d = request["dtm"]
    lay = dtm.train_layer(
        request["L"],
        d["T"],
        1,
        q,
        data,
        pf["steps_per_layer"],
        d["batch"],
        d["K_train"],
        np.random.default_rng([PROBE_SEED, 202]),
        d["lr"],
    )
    return lay, mix, q, time.process_time() - t0


def sampler_check(request, lay, mix, q):
    """THRML against NumPy on the probe layer, uniform and carried start (as P-0006)."""
    pf = request["preflight"]
    rng = np.random.default_rng([PROBE_SEED, 203])
    x = mix.sample(pf["images"], rng)
    y = flip(x, q, rng)
    fixed = np.random.default_rng([PROBE_SEED, 204]).integers(0, 2, lay.n) * 2.0 - 1.0
    creq = ConstraintRequest(
        model_id="dtm-baseline-headroom preflight probe layer float",
        update_rule="active_kernel",
        dtype="float64",
        relaxed=("beta_jitter", "quantization"),
        reasons=dict(request["relaxed_reasons"]),
    )
    low = LoweredLayer(lay, creq)
    chains = pf["chains"]
    out = {"starts": {}}
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
                keys = dtm.chain_keys(
                    {"seeds": {"jax_root": PROBE_SEED}}, (205, i, start == "fixed"), chains
                )
                th = np.asarray(fn(keys, jnp.asarray(xt > 0), *init))[:, low.pix_cols]
                npx = sweep_from(lay, lay.model(), xt, S0, pf["sweeps"], srng)[:, lay.pix] > 0
                zs.append(dtm._z(th.mean(0), npx.mean(0), chains))
            out["starts"][start] = {"max_abs_z": float(np.abs(np.concatenate(zs)).max())}
    out["z_max"] = pf["z_max"]
    out["passed"] = bool(all(v["max_abs_z"] < pf["z_max"] for v in out["starts"].values()))
    return out


def calibrate(request, lay, mix, q):
    """CPU seconds per unit kind on probe seeds; projects the whole study (3 candidates)."""
    L = request["L"]
    lad, d = request["ladder"], request["dtm"]
    pool_x = mix.sample(2000, np.random.default_rng([PROBE_SEED, 207]))
    t0 = time.process_time()
    mix.sample(2000, np.random.default_rng([PROBE_SEED, 208]))
    pool_cpu = (time.process_time() - t0) * request["n_pool"] / 2000
    biggest = max(request["families"], key=lambda fam: fam["components"])
    big = FieldMixture(L, biggest["J"], np.asarray(biggest["fields"]))
    yy = flip(
        big.sample(100, np.random.default_rng([PROBE_SEED, 209])),
        biggest["q"],
        np.random.default_rng([PROBE_SEED, 211]),
    )
    t0 = time.process_time()
    big.posterior(yy, biggest["q"])
    exact_cpu = (time.process_time() - t0) * request["n_test"] / 100
    fit = {}
    for rung in lad["rungs"]:
        args = (L, tuple(rung["widths"]), tuple(rung["kernels"]), q, pool_x)
        per = {}
        for steps in (20, 220):
            t0 = time.process_time()
            p = train_model(
                *args,
                steps,
                lad["batch"],
                0.01,
                jax.random.PRNGKey(PROBE_SEED),
                np.random.default_rng([PROBE_SEED, 210]),
                rung["global_pooling"],
            )
            jax.block_until_ready(p)
            per[steps] = time.process_time() - t0
        step_cpu = (per[220] - per[20]) / 200
        fit[rung["name"]] = {"overhead_cpu_s": per[20] - 20 * step_cpu, "step_cpu_s": step_cpu}
    n_fam = len(request["families"])
    fits_per = len(lad["rates"]) + len(lad["seeds"]) - 1
    ladder = (
        n_fam
        * fits_per
        * sum(v["overhead_cpu_s"] + lad["steps"] * v["step_cpu_s"] for v in fit.values())
    )
    # DTM: training per step, THRML per 800-chain call, NumPy per sweep
    data = pool_x
    t0 = time.process_time()
    dtm.train_layer(
        L, d["T"], 1, q, data, 20, d["batch"], d["K_train"], np.random.default_rng(1), d["lr"]
    )
    train_step = (time.process_time() - t0) / 20
    rows = d["draws"] * d["streams"]
    creq = arm_request(request, 1, 0, 0, "float_ref")
    low = LoweredLayer(lay, creq)
    rng = np.random.default_rng([PROBE_SEED, 206])
    S0 = rng.integers(0, 2, (rows, lay.n)) * 2.0 - 1.0
    xin = rng.integers(0, 2, (rows, L * L)) > 0
    call, comp = {}, {}
    with jax.enable_x64(True):
        init = [jnp.asarray(S0[:, g] > 0) for g in low.groups]
        keys = dtm.chain_keys({"seeds": {"jax_root": PROBE_SEED}}, (206,), rows)
        for K in (d["operating_point"]["K"], d["reference_sweeps"]):
            fn = make_thrml_runner(low.program, low.blocks, K)
            t0 = time.process_time()
            fn = fn.lower(keys, jnp.asarray(xin), *init).compile()
            comp[K] = time.process_time() - t0
            fn(keys, jnp.asarray(xin), *init).block_until_ready()
            t0 = time.process_time()
            fn(keys, jnp.asarray(xin), *init).block_until_ready()
            call[K] = time.process_time() - t0
    t0 = time.process_time()
    sweep_from(lay, lay.model(), np.where(xin, 1.0, -1.0), S0, 8, rng)
    np_sweep = (time.process_time() - t0) / 8
    calls = d["images_per_stream"] * d["T"]
    K = d["operating_point"]["K"]
    per_model = (
        train_step * d["steps_per_layer"] * d["T"]
        + calls * (call[K] + call[d["reference_sweeps"]])
        + d["T"] * (comp[K] + comp[d["reference_sweeps"]])
        + 2 * calls * K * np_sweep
    )
    n_models = d["max_candidates"] * len(d["seeds"])
    projected = {
        "pools": pool_cpu * n_fam * len(lad["seeds"]),
        "exact": exact_cpu * n_fam,
        "ladder": ladder,
        "dtm_at_max_candidates": per_model * n_models,
    }
    total = sum(projected.values()) / 3600
    return {
        "ladder_fit_cpu_s": fit,
        "dtm_train_cpu_s_per_step": train_step,
        "thrml_cpu_s_per_800_chain_call": {str(k): v for k, v in call.items()},
        "thrml_compile_cpu_s": {str(k): v for k, v in comp.items()},
        "numpy_cpu_s_per_sweep_800_chains": np_sweep,
        "projected_cpu_s": projected,
        "projected_cpu_hours": total,
        "stop_hours": request["cpu_stop_hours"],
        "within_budget": bool(total <= request["cpu_stop_hours"]),
    }


def preflight(request):
    lay, mix, q, train_cpu = _probe_layer(request)
    placement = {"probe_layer_ok": bool(check(lay.model()).ok)}
    try:
        for arm in ("op", "float_ref"):
            require_placement(lay.model(), arm_request(request, 1, 0, 0, arm))
        placement["passed"] = placement["probe_layer_ok"]
    except ValueError:
        placement["passed"] = False
    result = {
        "seeds": f"probe seeds ({PROBE_SEED}, ...) only",
        "transfer_matrix": transfer_check(request),
        "placement": placement,
        "samplers": sampler_check(request, lay, mix, q),
        "counts": counts_check(request),
        "pricing": pricing_check(request),
    }
    result["passed"] = bool(all(v["passed"] for k, v in result.items() if k != "seeds"))
    cal = calibrate(request, lay, mix, q)
    cal["probe_layer_train_cpu_s"] = train_cpu
    return result, cal


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


def fit_name(f, r, s, ri):
    return f"fit__f{f}_r{r}_s{s}_lr{ri}"


def cell_name(c):
    return f"cell__f{c['f']}_s{c['s']}_{c['kind']}"


def image_interval(diff, idx):
    d = np.asarray(diff, dtype=np.float64)
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(d.mean()), "lower": float(lo), "upper": float(hi)}


def stage1(request, exact, fits):
    """Rate choice, best seed, failed_to_train, gaps and candidacy from archived counts."""
    L2 = request["L"] ** 2
    lad = request["ladder"]
    seeds, n_rates = lad["seeds"], len(lad["rates"])
    x, _ = x_threshold(request)
    nval = request["n_val"] * L2
    families = {}
    for f, fam in enumerate(request["families"]):
        ex = exact[str(f)]
        bayes = np.asarray(ex["bayes_mpm_wrong"], np.float64) / L2
        noisy = np.asarray(ex["noisy_wrong"], np.float64) / L2
        val_noisy = ex["validation_noisy_wrong_total"] / nval
        idx = np.random.default_rng([*request["seeds"]["bootstrap"], f]).integers(
            0, len(bayes), (request["bootstrap"]["resamples"], len(bayes))
        )
        rungs = {}
        for rung in lad["rungs"]:
            r = rung["index"]

            def val(s, ri, r=r, f=f):
                return fits[fit_name(f, r, s, ri)]["val_wrong"] / nval

            rate_val = [val(0, ri) for ri in range(n_rates)]
            ri = int(np.argmin(rate_val))
            seed_val = {s: val(s, ri) for s in seeds}
            best = min(seeds, key=lambda s: (seed_val[s], s))
            test = np.asarray(fits[fit_name(f, r, best, ri)]["test_wrong"], np.float64) / L2
            failed = bool(seed_val[best] > val_noisy)
            gap = image_interval(test - bayes, idx)
            rungs[rung["name"]] = {
                "name": rung["name"],
                "index": r,
                "macs_per_image": rung["macs_per_image"],
                "params": rung["params"],
                "below_x": bool(rung["macs_per_image"] < x),
                "validation_by_rate": rate_val,
                "rate_index": ri,
                "rate": lad["rates"][ri],
                "validation_by_seed": {str(s): v for s, v in seed_val.items()},
                "best_seed": best,
                "test_error": float(test.mean()),
                "test_error_by_seed": {
                    str(s): float(np.mean(fits[fit_name(f, r, s, ri)]["test_wrong"]) / L2)
                    for s in seeds
                },
                "failed_to_train": failed,
                "gap_to_bayes": gap,
                "reaches_bayes": bool(not failed and test.mean() <= bayes.mean() + TOL),
            }
        below = [v for v in rungs.values() if v["below_x"] and not v["failed_to_train"]]
        best_below = min(below, key=lambda v: (v["test_error"], v["macs_per_image"]), default=None)
        reaching = [v for v in rungs.values() if v["reaches_bayes"]]
        smallest = min(reaching, key=lambda v: v["macs_per_image"], default=None)
        families[fam["name"]] = {
            "index": f,
            "noisy_error": float(noisy.mean()),
            "bayes_mpm_error": float(bayes.mean()),
            "bayes_logloss_nats": float(np.mean(ex["bayes_logloss"])),
            "validation_noisy_error": val_noisy,
            "rungs": rungs,
            "smallest_reaching_bayes": None
            if smallest is None
            else {"rung": smallest["name"], "macs_per_image": smallest["macs_per_image"]},
            "best_rung_below_x": None
            if best_below is None
            else {
                "rung": best_below["name"],
                "test_error": best_below["test_error"],
                "gap": best_below["test_error"] - float(bayes.mean()),
            },
            "candidate": bool(
                fam["name"] != request["anchor"] and not any(v["reaches_bayes"] for v in below)
            ),
        }
    anchor = families[request["anchor"]]
    anchor_ok = any(v["reaches_bayes"] for v in anchor["rungs"].values())
    cands = [n for n, v in families.items() if v["candidate"]]

    def gap_of(n):
        b = families[n]["best_rung_below_x"]
        return float("inf") if b is None else b["gap"]

    chosen = sorted(cands, key=lambda n: (-gap_of(n), families[n]["index"]))
    chosen = chosen[: request["dtm"]["max_candidates"]] if anchor_ok else []
    selected = sorted(chosen, key=lambda n: families[n]["index"])
    return {
        "x_macs_per_image": x,
        "families": families,
        "anchor_control_passed": bool(anchor_ok),
        "candidate_families": cands,
        "stage2_families": selected,
        "stage2_indices": [families[n]["index"] for n in selected],
    }


def dtm_cells(request, stage2_indices):
    return [
        {"f": f, "s": s, "kind": kind}
        for f in stage2_indices
        for s in request["dtm"]["seeds"]
        for kind in CELL_KINDS
    ]


def boot_indices(request, f, s):
    rng = np.random.default_rng([*request["seeds"]["bootstrap"], f, s])
    n = request["dtm"]["streams"]
    return rng.integers(0, n, (request["bootstrap"]["resamples"], n))


def stage2(request, exact, s1, cells_store, dead, timings=None):
    """Seed verdicts, family verdicts and break-evens from archived DTM counts."""
    L2 = request["L"] ** 2
    d = request["dtm"]
    streams, per = d["streams"], d["images_per_stream"]
    _, price = x_threshold(request)
    x = s1["x_macs_per_image"]
    hi = price["per_estimate_j_write_constant"]
    lo = price["per_estimate_j_clamp_loads_at_read_constant_outside_profile"]
    out = {}
    largest = max(request["ladder"]["rungs"], key=lambda r: r["macs_per_image"])
    for name in s1["stage2_families"]:
        fam1 = s1["families"][name]
        f = fam1["index"]
        ex = exact[str(f)]
        bayes = np.asarray(ex["bayes_mpm_wrong"], np.float64) / L2
        noisy = np.asarray(ex["noisy_wrong"], np.float64) / L2
        if name in dead:
            out[name] = {"verdict": "not_a_denoiser", "dead_seeds": dead[name]}
            continue

        def err(s, kind, f=f):
            return (
                np.asarray(
                    cells_store[cell_name({"f": f, "s": s, "kind": kind})]["maj_wrong"], np.float64
                )
                / L2
            )

        seeds = {}
        for s in d["seeds"]:
            idx = boot_indices(request, f, s)

            def iv(a, b, idx=idx):
                return dtm.interval(a - b, idx, streams, per)

            op = err(s, "op")
            e_s = float(op.mean())
            reaching = [
                (v["macs_per_image"], k)
                for k, v in fam1["rungs"].items()
                if not v["failed_to_train"] and v["test_error"] <= e_s + TOL
            ]
            base = min(reaching)[1] if reaching else None
            base_macs = fam1["rungs"][base]["macs_per_image"] if base else None
            favours = base is None or base_macs >= x
            op_ref = iv(op, err(s, "float_ref"))
            rand_cont = iv(err(s, "numpy_random"), err(s, "numpy_continue"))
            resolved = bool(
                op_ref["upper"] < TOL and -TOL <= rand_cont["lower"] and rand_cont["upper"] <= TOL
            )
            denom = base_macs if base else largest["macs_per_image"]
            rung_err = fam1["rungs"][base]["test_error"] if base else None
            summary = {
                kind: dtm.cell_summary(
                    {"family": {"L": request["L"]}, "draws": d["draws"], "streams": streams}
                    | {"images_per_stream": per},
                    cells_store[cell_name({"f": f, "s": s, "kind": kind})],
                    ex,
                )
                for kind in CELL_KINDS
            }
            seeds[str(s)] = {
                "dtm_error": e_s,
                "float_ref_error": float(err(s, "float_ref").mean()),
                "gap_to_bayes": iv(op, bayes),
                "below_noisy": iv(op, noisy),
                "op_minus_float_ref": op_ref,
                "random_minus_continue": rand_cont,
                "thrml_op_minus_numpy_continue": iv(op, err(s, "numpy_continue")),
                "baseline": base if base else "none_in_ladder",
                "baseline_macs_per_image": base_macs,
                "baseline_test_error": rung_err,
                "baseline_cost_j": {
                    f"{pj}pJ": None if base is None else base_macs * pj * 1e-12
                    for pj in request["pj_per_mac"]
                },
                "favours": "dtm" if favours else "ladder",
                "operating_point_resolved": resolved,
                "verdict": ("dtm" if favours else "ladder")
                + ("" if resolved else " (operating_point_unresolved)"),
                "break_even_pj_per_mac": {
                    "write_constant": hi / denom / 1e-12,
                    "clamp_at_read": lo / denom / 1e-12,
                    "kind": "estimate" if base else "upper_bound_vs_largest_rung",
                },
                "cell_summaries": summary,
            }
        fav = [v["favours"] for v in seeds.values()]
        if all(v == "dtm" for v in fav):
            verdict = "qualifies"
        elif all(v == "ladder" for v in fav):
            verdict = "ladder_wins"
        else:
            verdict = "inconclusive"
        out[name] = {
            "index": f,
            "verdict": verdict,
            "all_seeds_resolved": bool(all(v["operating_point_resolved"] for v in seeds.values())),
            "seeds": seeds,
        }
        if timings:
            out[name]["cpu_s_per_estimate"] = {
                kind: float(
                    np.mean(
                        [
                            timings[cell_name({"f": f, "s": s, "kind": kind})]["steady_cpu_s"]
                            for s in d["seeds"]
                        ]
                    )
                    / request["n_test"]
                )
                for kind in CELL_KINDS
            }
    return out


def evaluate(request, exact, fits, cells_store, trained, dead, timings=None):
    s1 = stage1(request, exact, fits)
    _, price = x_threshold(request)
    s2 = stage2(request, exact, s1, cells_store, dead, timings)
    return {
        "pricing": price,
        "stage1": s1,
        "stage2": s2,
        "verdicts": {n: v["verdict"] for n, v in s2.items()},
    }


def dead_models(request, f, store):
    """Stop rule: float K = 1024 not below the noisy input in two seeds."""
    L2 = request["L"] ** 2
    x, y = held_set(request, f, "test")
    noisy = float((x != y).mean())
    bad = []
    for s in request["dtm"]["seeds"]:
        e = float(
            np.mean(store[cell_name({"f": f, "s": s, "kind": "float_ref"})]["maj_wrong"]) / L2
        )
        if not e < noisy:
            bad.append(s)
    return bad


# ----------------------------------------------------------------------------
# Persistence, workers and autosave
# ----------------------------------------------------------------------------


class Runner(dtm.Runner):
    pass


_WORKER_SLOT = None


def _init_worker(counter, cpus):
    """Pin this worker to one CPU (bitwise-reproducible float32 ladder fits)."""
    global _WORKER_SLOT
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with counter.get_lock():
        slot = counter.value
        counter.value += 1
    _WORKER_SLOT = slot
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpus[slot % len(cpus)]})


def _pool_unit(request, kind, key, payload):
    if kind == "pool":
        arrays = {"x": pool(request, *key).astype(np.int8)}
        timing = {}
    elif kind == "exact":
        result, timing = exact_reference(request, key)
        arrays = {"json": np.frombuffer(canonical_json(result).encode(), dtype=np.uint8)}
    elif kind == "fit":
        f, r, s, ri = key
        rung = next(x for x in request["ladder"]["rungs"] if x["index"] == r)
        arrays, timing = run_fit(request, f, rung, s, ri, payload.astype(np.float64))
    elif kind == "train":
        arrays, timing = run_training(request, *key, payload.astype(np.float64))
    elif kind == "preflight":
        result, cal = preflight(request)
        arrays = {
            "json": np.frombuffer(
                canonical_json({"preflight": result, "calibration": cal}).encode(), dtype=np.uint8
            )
        }
        timing = {}
    else:
        arrays, timing = run_cell(request, key, payload)
    return kind, key, arrays, timing


def unit_name(kind, key):
    if kind == "pool":
        return f"pool__f{key[0]}_s{key[1]}"
    if kind == "exact":
        return f"exact__f{key}"
    if kind == "fit":
        return fit_name(*key)
    if kind == "train":
        return f"train__f{key[0]}_s{key[1]}"
    if kind == "preflight":
        return "preflight"
    return cell_name(key)


def require_single_thread_blas():
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"set {name}=1 before running or replaying this study")


def _cpus():
    if not hasattr(os, "sched_getaffinity"):
        return []
    cpus = sorted(os.sched_getaffinity(0))
    return list(reversed(cpus))


def _run_units(runner, request, jobs, workers, stop_after, counter):
    """Run (kind, key, payload) jobs in pinned worker processes, saving each unit."""

    def record(kind, key, arrays, timing):
        name = unit_name(kind, key)
        runner.save(name, arrays, timing)
        counter["done"] += 1
        runner.status(last_unit=name, units_done=len(runner.index()))
        cpu = timing.get("steady_cpu_s", sum(v for v in timing.values() if isinstance(v, float)))
        runner.log(f"unit {name}: {cpu:.1f} CPU-s ({len(runner.index())} units saved)")
        if stop_after is not None and counter["done"] >= stop_after:
            raise KeyboardInterrupt("stop_after reached (interruption test)")

    if not jobs:
        return
    ctx = multiprocessing.get_context("spawn")
    slot = ctx.Value("i", 0)
    with concurrent.futures.ProcessPoolExecutor(
        max(1, workers), mp_context=ctx, initializer=_init_worker, initargs=(slot, _cpus())
    ) as pool_exec:
        futures = [pool_exec.submit(_pool_unit, request, k, key, p) for k, key, p in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):
                record(*future.result())
        except BaseException:
            pool_exec.shutdown(cancel_futures=True)
            raise


def _copy_sources(out, request):
    for name, digest in request["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)


def _json_unit(runner, name):
    return json.loads(bytes(runner.load(name)["json"]).decode())


def _fit_record(arrays):
    return {
        "test_wrong": np.asarray(arrays["test_wrong"]).astype(int).tolist(),
        "val_wrong": int(np.asarray(arrays["val_wrong"])[0]),
    }


def _cell_record(a):
    return {
        "maj_wrong": np.asarray(a["maj_wrong"]).astype(int).tolist(),
        "draw_wrong": np.asarray(a["draw_wrong"]).astype(int).tolist(),
        "logloss": [float(str(v)) for v in np.asarray(a["logloss"], dtype=np.float32)],
    }


def _fit_cost(request, key):
    rung = next(x for x in request["ladder"]["rungs"] if x["index"] == key[1])
    return rung["macs_per_image"] * (2 if rung["global_pooling"] else 1)


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
    runner = Runner(out, digest, request["sources"][MODULE])
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
            "cpus": _cpus(),
            "provenance": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        }
    )
    write(attempts_path, attempts)
    runner.log(f"attempt {attempt} ({'resume' if resume else 'fresh'}), {workers} worker(s)")
    started = time.perf_counter()
    counter = {"done": 0}
    F = range(len(request["families"]))
    lad = request["ladder"]
    try:
        runner.status(phase="preflight", attempt=attempt)
        if "preflight" not in runner.verified():
            _run_units(runner, request, [("preflight", None, None)], 1, stop_after, counter)
        pre = _json_unit(runner, "preflight")
        if not pre["preflight"]["passed"]:
            runner.log("STOP: preflight failed; see units/preflight")
            raise ValueError("preflight failed; stopped before production")
        hours = pre["calibration"]["projected_cpu_hours"]
        runner.log(f"preflight passed; calibration projects {hours:.2f} CPU-hours")
        if not pre["calibration"]["within_budget"]:
            runner.log("STOP: calibration projects over the CPU stop budget")
            raise ValueError("calibration projects over the CPU budget")

        runner.status(phase="pools_and_exact")
        done = runner.verified()
        jobs = [("pool", (f, s), None) for f in F for s in lad["seeds"]]
        jobs += [("exact", f, None) for f in F]
        jobs = [j for j in jobs if unit_name(j[0], j[1]) not in done]
        _run_units(runner, request, jobs, workers, stop_after, counter)

        def pool_of(f, s):
            return runner.load(unit_name("pool", (f, s)))["x"]

        runner.status(phase="ladder_rates")
        done = runner.verified()
        keys = [
            (f, r["index"], 0, ri)
            for f in F
            for r in lad["rungs"]
            for ri in range(len(lad["rates"]))
        ]
        jobs = [("fit", k, pool_of(k[0], 0)) for k in keys if fit_name(*k) not in done]
        jobs.sort(key=lambda j: -_fit_cost(request, j[1]))
        runner.log(f"ladder rate fits: {len(keys) - len(jobs)} verified, {len(jobs)} to run")
        _run_units(runner, request, jobs, workers, stop_after, counter)

        runner.status(phase="ladder_seeds")
        done = runner.verified()
        keys = []
        for f in F:
            for r in lad["rungs"]:
                vals = [
                    int(runner.load(fit_name(f, r["index"], 0, ri))["val_wrong"][0])
                    for ri in range(len(lad["rates"]))
                ]
                ri = int(np.argmin(vals))
                keys += [(f, r["index"], s, ri) for s in lad["seeds"] if s != 0]
        jobs = [("fit", k, pool_of(k[0], k[2])) for k in keys if fit_name(*k) not in done]
        jobs.sort(key=lambda j: -_fit_cost(request, j[1]))
        runner.log(f"ladder seed fits: {len(keys) - len(jobs)} verified, {len(jobs)} to run")
        _run_units(runner, request, jobs, workers, stop_after, counter)

        runner.status(phase="candidates")
        exact = {str(f): _json_unit(runner, unit_name("exact", f)) for f in F}
        fits = {n: _fit_record(runner.load(n)) for n in runner.verified() if n.startswith("fit__")}
        s1 = stage1(request, exact, fits)
        cand = {
            "candidate_families": s1["candidate_families"],
            "stage2_families": s1["stage2_families"],
            "stage2_indices": s1["stage2_indices"],
            "anchor_control_passed": s1["anchor_control_passed"],
            "x_macs_per_image": s1["x_macs_per_image"],
        }
        cand["digest"] = canonical_sha256(cand)
        cpath = out / "candidates.json"
        if cpath.exists() and json.loads(cpath.read_text()) != cand:
            raise ValueError("stage-1 candidates changed on resume")
        write(cpath, cand)
        runner.log(
            f"stage 1 complete: candidates {s1['candidate_families']}; stage 2 on "
            f"{s1['stage2_families']}; anchor control {s1['anchor_control_passed']}"
        )
        if not s1["anchor_control_passed"]:
            runner.log("STOP: the ladder does not reach Bayes + 0.01 on the anchor; no stage 2")

        idx2 = s1["stage2_indices"]
        d = request["dtm"]
        runner.status(phase="dtm_training")
        done = runner.verified()
        jobs = [
            ("train", (f, s), pool_of(f, s)[: d["n_train"]])
            for f in idx2
            for s in d["seeds"]
            if unit_name("train", (f, s)) not in done
        ]
        _run_units(runner, request, jobs, workers, stop_after, counter)
        models = {(f, s): runner.load(unit_name("train", (f, s))) for f in idx2 for s in d["seeds"]}

        all_cells = dtm_cells(request, idx2)
        runner.status(phase="reference_cells")
        done = runner.verified()
        refs = [c for c in all_cells if c["kind"] == "float_ref"]
        jobs = [("cell", c, models[(c["f"], c["s"])]) for c in refs if cell_name(c) not in done]
        _run_units(runner, request, jobs, workers, stop_after, counter)
        store = {cell_name(c): runner.load(cell_name(c)) for c in refs}
        dead = {}
        for f in idx2:
            bad = dead_models(request, f, store)
            if len(bad) >= 2:
                dead[request["families"][f]["name"]] = bad
                runner.log(f"STOP family {f}: float K_ref not below the noisy input in seeds {bad}")
        write(out / "float-reference-check.json", {"dead": dead})

        runner.status(phase="cells")
        done = runner.verified()
        dead_idx = {request["families"][f]["name"]: f for f in idx2}
        skip = {dead_idx[n] for n in dead}
        rest = [c for c in all_cells if c["kind"] != "float_ref" and c["f"] not in skip]
        jobs = [("cell", c, models[(c["f"], c["s"])]) for c in rest if cell_name(c) not in done]
        jobs.sort(key=lambda j: j[1]["kind"] != "op")
        runner.log(f"cells: {len(rest) - len(jobs)} verified, {len(jobs)} to run")
        _run_units(runner, request, jobs, workers, stop_after, counter)

        runner.status(phase="assembling")
        assemble(out, request, runner, time.perf_counter() - started, dead)
    except BaseException as error:
        runner.status(
            phase="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        runner.log(f"stopped: {error!r}")
        raise
    runner.status(phase="replaying")
    completion = replay(out, light=False, workers=workers)
    runner.status(phase="complete", error=None, traceback=None)
    runner.log(f"completion written after replay ({completion['replay_seconds']:.1f} s)")
    return completion


def assemble(out, request, runner, seconds, dead):
    index = runner.verified()
    F = range(len(request["families"]))
    exact = {str(f): _json_unit(runner, unit_name("exact", f)) for f in F}
    pools = {
        f"f{f}_s{s}": array_sha(runner.load(unit_name("pool", (f, s)))["x"])
        for f in F
        for s in request["ladder"]["seeds"]
    }
    fits = {n: _fit_record(runner.load(n)) for n in sorted(index) if n.startswith("fit__")}
    cand = json.loads((out / "candidates.json").read_text())
    idx2 = cand["stage2_indices"]
    trained = {}
    for f in idx2:
        for s in request["dtm"]["seeds"]:
            a = runner.load(unit_name("train", (f, s)))
            trained[f"f{f}_s{s}"] = {
                f"layer{li}": {
                    "gamma": float(a[f"gamma{li}"][0]),
                    "w": a[f"w{li}"].tolist(),
                    "b": a[f"b{li}"].tolist(),
                }
                for li in range(request["dtm"]["T"])
            }
    store, cell_timing = {}, {}
    for c in dtm_cells(request, idx2):
        name = cell_name(c)
        if name in index:
            store[name] = _cell_record(runner.load(name))
            cell_timing[name] = index[name]["timing"]
    pre = _json_unit(runner, "preflight")
    timing_block = {
        "label": "CPU seconds (time.process_time) on this box; excluded from the result digest",
        "exact": {k: v["timing"] for k, v in index.items() if k.startswith("exact__")},
        "fits": {k: v["timing"] for k, v in index.items() if k.startswith("fit__")},
        "training": {k: v["timing"] for k, v in index.items() if k.startswith("train__")},
        "cells": cell_timing,
        "calibration": pre["calibration"],
        "generation_seconds_this_attempt": seconds,
    }
    result = {
        "request_digest": canonical_sha256(request),
        "evidence_class": {
            "exact": "exact_reference",
            "ladder": "software_simulation (float32 JAX, CPU)",
            "dtm_training": "software_simulation",
            "thrml_cells": "software_simulation",
            "numpy_cells": "software_simulation",
            "joules": "calibrated_projection",
            "cpu_seconds": "software CPU measurement on this box (not evidence of hardware)",
        },
        "hardware_claim": False,
        "preflight": pre["preflight"],
        "exact_reference": exact,
        "pool_sha256": pools,
        "ladder_fits": fits,
        "candidates": cand,
        "dead_models": dead,
        "trained_weights": trained,
        "rounding": rounding_record(request, trained),
        "cells": store,
    }
    result["result_digest"] = canonical_sha256(result)
    evaluation = evaluate(request, exact, fits, store, trained, dead, cell_timing)
    evaluation["cpu_s_per_estimate_ladder"] = _ladder_cpu(request, evaluation, timing_block)
    study = {**result, "evaluation": evaluation, "timings": timing_block}
    write_gz_json(out / "study.json.gz", study)
    attempts = json.loads((out / "attempts.json").read_text())
    write(
        out / "provenance.json",
        {
            "schema": "dtm_baseline_headroom.provenance.v1",
            "request_digest": result["request_digest"],
            "attempts": attempts,
            "runtime": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        },
    )
    runner.log("assembled study.json.gz and provenance.json")


def _ladder_cpu(request, evaluation, timing_block):
    """CPU seconds per image of each family's best-seed fit per rung (not in the digest)."""
    out = {}
    for name, fam in evaluation["stage1"]["families"].items():
        f = fam["index"]
        out[name] = {
            k: timing_block["fits"]
            .get(fit_name(f, v["index"], v["best_seed"], v["rate_index"]), {})
            .get("inference_cpu_s_per_image")
            for k, v in fam["rungs"].items()
        }
    return {"label": "software CPU seconds on this box; not hardware", "per_image": out}


def rounding_record(request, trained):
    """Caps per (family, seed, layer) for the operating-point arm; re-verifies placement."""
    out = {}
    for key, layers in trained.items():
        f, s = (int(p[1:]) for p in key.split("_"))
        for li in range(request["dtm"]["T"]):
            dd = layers[f"layer{li}"]
            lay = dtm.layer_from(request["L"], dd["gamma"], dd["w"], dd["b"])
            for arm in ("op", "float_ref"):
                creq = arm_request(request, f, s, li, arm)
                model, _, summary = dtm.arm_model(lay, creq)
                require_placement(model, creq)
                if summary is not None:
                    out[f"{key}_l{li}_{arm}"] = {
                        "coupling_cap": summary["coupling_cap"],
                        "field_cap": summary["field_cap"],
                        "coupling_max_cap_normalized": summary["couplings"]["max_cap_normalized"],
                        "field_max_cap_normalized": summary["fields"]["max_cap_normalized"],
                        "max_degree": int(check(model).max_degree),
                    }
    return out


# ----------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------


def _close(actual, expected, rtol, what):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != e.shape or not np.all(np.abs(a - e) <= rtol * np.maximum(1.0, np.abs(e))):
        raise ValueError(f"replay mismatch: {what}")


def _prefix_jobs(request, study):
    """One ladder fit (anchor, first rung, seed 0, rate 0) and one DTM operating-point cell."""
    anchor = next(
        f for f, fam in enumerate(request["families"]) if fam["name"] == request["anchor"]
    )
    r = request["ladder"]["rungs"][0]["index"]
    fit_key = (anchor, r, 0, 0)
    jobs = [("fit", fit_key, pool(request, anchor, 0).astype(np.int8))]
    idx2 = study["candidates"]["stage2_indices"]
    alive = [f for f in idx2 if request["families"][f]["name"] not in study["dead_models"]]
    if alive:
        f, s = alive[0], request["dtm"]["seeds"][0]
        layers = study["trained_weights"][f"f{f}_s{s}"]
        arrays = {}
        for li in range(request["dtm"]["T"]):
            dd = layers[f"layer{li}"]
            arrays[f"w{li}"] = np.asarray(dd["w"])
            arrays[f"b{li}"] = np.asarray(dd["b"])
            arrays[f"gamma{li}"] = np.asarray([dd["gamma"]])
        jobs.append(("cell", {"f": f, "s": s, "kind": "op"}, arrays))
    return jobs


def replay(out, light=False, workers=1):
    out = Path(out)
    require_single_thread_blas()
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    study = read_gz_json(out / "study.json.gz")
    validate_request(request)
    digest = canonical_sha256(request)
    if study["request_digest"] != digest:
        raise ValueError("request digest mismatch")
    for name, dg in request["sources"].items():
        for path in (ROOT / name, out / "source" / name):
            if sha(path) != dg:
                raise ValueError(f"source changed: {name}")
    for key, value in request["constraint_requests"].items():
        f, s, li, arm = key.split("_", 3)
        cr = arm_request(request, int(f[1:]), int(s[1:]), int(li[1:]), arm)
        if cr.digest() != value["digest"] or cr.to_canonical() != value["canonical"]:
            raise ValueError(f"constraint request does not rebuild: {key}")
    core = {k: study[k] for k in study if k not in ("evaluation", "timings", "result_digest")}
    if canonical_sha256(core) != study["result_digest"]:
        raise ValueError("result digest mismatch")
    # family fields rebuild from the probe's code
    table = [(fam["name"], fam["J"], fam["h0"], fam["q"]) for fam in request["families"]]
    if family_fields(request["L"], table) != request["families"]:
        raise ValueError("family fields do not rebuild")
    # exact references
    rtol = request["replay_rtol"]
    F = range(len(request["families"]))
    for f in F:
        ex, _ = exact_reference(request, f)
        st = study["exact_reference"][str(f)]
        for key in (
            "test_x_sha256",
            "test_y_sha256",
            "validation_x_sha256",
            "validation_y_sha256",
            "noisy_wrong",
            "validation_noisy_wrong_total",
            "bayes_mpm_wrong",
        ):
            if ex[key] != st[key]:
                raise ValueError(f"exact reference mismatch: family {f} {key}")
        for key in ("component_log_z", "bayes_sample_error", "bayes_logloss"):
            _close(ex[key], st[key], rtol, f"family {f} {key}")
        if not light:
            for s in request["ladder"]["seeds"]:
                if (
                    array_sha(pool(request, f, s).astype(np.int8))
                    != study["pool_sha256"][f"f{f}_s{s}"]
                ):
                    raise ValueError(f"pool mismatch: f{f} s{s}")
    if not transfer_check(request)["passed"]:
        raise ValueError("transfer-matrix check failed on replay")
    if not counts_check(request)["passed"]:
        raise ValueError("ladder counts check failed on replay")
    if not pricing_check(request)["passed"]:
        raise ValueError("pricing check failed on replay")
    # ladder fit set and ranges
    L2, n = request["L"] ** 2, request["n_test"]
    for name, a in study["ladder_fits"].items():
        if len(a["test_wrong"]) != n or not all(0 <= v <= L2 for v in a["test_wrong"]):
            raise ValueError(f"bad ladder counts: {name}")
    # candidates, placement, cells and evaluation recomputed from archived counts
    s1 = stage1(request, study["exact_reference"], study["ladder_fits"])
    for key in ("candidate_families", "stage2_families", "stage2_indices", "anchor_control_passed"):
        if s1[key] != study["candidates"][key]:
            raise ValueError(f"candidate list does not recompute: {key}")
    cand = {k: v for k, v in study["candidates"].items() if k != "digest"}
    if canonical_sha256(cand) != study["candidates"]["digest"]:
        raise ValueError("candidate digest mismatch")
    check_equal(rounding_record(request, study["trained_weights"]), study["rounding"])
    idx2 = s1["stage2_indices"]
    dead_idx = {request["families"][f]["name"] for f in idx2} & set(study["dead_models"])
    expected = [
        cell_name(c)
        for c in dtm_cells(request, idx2)
        if request["families"][c["f"]]["name"] not in dead_idx or c["kind"] == "float_ref"
    ]
    if sorted(study["cells"]) != sorted(expected):
        raise ValueError("cell set differs from the request")
    draws = request["dtm"]["draws"]
    for name in expected:
        a = study["cells"][name]
        if len(a["maj_wrong"]) != n or not all(0 <= v <= L2 for v in a["maj_wrong"]):
            raise ValueError(f"bad majority counts: {name}")
        if len(a["draw_wrong"]) != n or not all(0 <= v <= draws * L2 for v in a["draw_wrong"]):
            raise ValueError(f"bad draw counts: {name}")
    evaluation = evaluate(
        request,
        study["exact_reference"],
        study["ladder_fits"],
        study["cells"],
        study["trained_weights"],
        study["dead_models"],
        study["timings"]["cells"],
    )
    evaluation["cpu_s_per_estimate_ladder"] = _ladder_cpu(request, evaluation, study["timings"])
    check_equal(evaluation, study["evaluation"])
    # prefix check: refit one ladder rung and rerun one DTM cell in a pinned worker
    prefix = {"run": not light}
    if not light:
        jobs = _prefix_jobs(request, study)
        got = {}
        ctx = multiprocessing.get_context("spawn")
        slot = ctx.Value("i", 0)
        with concurrent.futures.ProcessPoolExecutor(
            1, mp_context=ctx, initializer=_init_worker, initargs=(slot, _cpus())
        ) as ex_pool:
            for kind, key, arrays, _ in ex_pool.map(
                _pool_unit, *zip(*[(request, *j) for j in jobs], strict=True)
            ):
                got[unit_name(kind, key)] = arrays
        for name, arrays in got.items():
            if name.startswith("fit__"):
                same = _fit_record(arrays) == study["ladder_fits"][name]
            else:
                same = _cell_record(arrays) == study["cells"][name]
            prefix[name] = bool(same)
            if not same:
                raise ValueError(f"prefix check failed: {name}")
        prefix["passed"] = True
    s2 = evaluation["stage2"]
    completion = {
        "status": "dtm_baseline_headroom_complete",
        "request_digest": digest,
        "result_digest": study["result_digest"],
        "families": len(request["families"]),
        "ladder_fits": len(study["ladder_fits"]),
        "test_images_per_family": n,
        "candidates": len(idx2),
        "candidate_families_before_cap": s1["candidate_families"],
        "candidate_list": s1["stage2_families"],
        "dtm_cells": len(study["cells"]),
        "preflight_passed": bool(study["preflight"]["passed"]),
        "placement_passed": True,
        "references_recomputed": True,
        "anchor_control_passed": s1["anchor_control_passed"],
        "prefix_check_passed": bool(prefix.get("passed", False)),
        "light_replay": bool(light),
        "replayed": True,
        "x_macs_per_image": s1["x_macs_per_image"],
        "x_macs_rounded": round(s1["x_macs_per_image"]),
        "family_verdicts": {k: v["verdict"] for k, v in s2.items()},
        "seed_verdicts": {
            k: {s: x["verdict"] for s, x in v.get("seeds", {}).items()} for k, v in s2.items()
        },
        "break_even_pj_per_mac": {
            k: {s: x["break_even_pj_per_mac"] for s, x in v.get("seeds", {}).items()}
            for k, v in s2.items()
        },
        "evidence_class": study["evidence_class"],
        "hardware_claim": False,
        "replay_seconds": time.perf_counter() - started,
    }
    if not light:
        write(out / "completion.json", completion)
    return completion


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
