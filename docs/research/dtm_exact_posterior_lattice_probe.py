"""Exploration probe for loop proposal P-0008 (`dtm-exact-posterior-lattice`).

Unplaced model only: exact enumeration over hidden configurations times the
transfer matrix, and NumPy chromatic Gibbs. No THRML, no training.

The model is built from the mixture posterior of P-0007's families
(`t4_h05_q035`, `smooth8_h05_q035`, fields imported from the archived runner):

    p(x, c | y) ~ (1/C) exp(J sum_<ij> x_i x_j + h_c . x - log Z_c + g y . x)

with g = atanh(1 - 2q)/... = 0.5 log((1 - q)/q). The template latent c becomes
binary hidden units z_c in {0, 1}, one per template, with pixel couplings
d_c = h_c - m (m is the pixel bias; ``centered``: m = template mean, ``raw``:
m = 0) and hidden bias B_c = b0 - log Z_c.

- ``bias`` design (stage A): z_c independent, b0 a hand-set scalar. Spurious
  states (no unit on, several on) are what it gets wrong.
- ``onehot`` design: pairwise inhibition lambda (z form: -lambda (sum z - 1)^2),
  exact as lambda -> infinity, range-hungry at finite bits.

Every model is converted to spin form (s = 2 z - 1, the lattice's +-1 nodes,
conditionals sigmoid(2 f)) and, for quantized arms, rounded through the
constraint layer's codebook under ``split`` (couplings and fields capped
separately at their own max). Exact equilibrium: sum over 2^C hidden spin
states of exp(hidden terms) * Z_grid(J, pixel field), using the P-0006
``GridIsing`` transfer matrix imported unchanged.

Sampled: chromatic Gibbs on the unplaced spin model (blocks: even pixels,
odd pixels, hidden units), 16 chains per image, 16-draw pixelwise majority
(ties to +1), streams of 20 images, starts continuation / random / reset
every N images, K sweeps per image.

Evidence classes: the mixture data, Bayes posterior, model equilibria and
16-draw floors are ``exact_reference`` (float64). Gibbs errors are
``software_simulation`` (NumPy, CPU). Joules are ``calibrated_projection``
(sealed Z1 profile constants). Nothing here ran on hardware. Exploration,
not evidence.

Run (CPU, single-thread BLAS):
    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
    uv run python docs/research/dtm_exact_posterior_lattice_probe.py --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import binom

from thermo_lab.constraints.quantize import cap_for, quantize_array
from thermo_lab.dtm_baseline_headroom import FieldMixture, family_patterns
from thermo_lab.dtm_small_image import GridIsing, flip
from thermo_lab.hardware.z1 import Z1HardwareProfile

PROBE_SEED = 20261017  # fresh: not in any archive, probe or protocol
L = 8
N = L * L
DRAWS = 16
TOL = 0.01
FAMILIES = {  # name: (J, h0, q); archived Bayes and ladder from P-0007 findings
    "t4_h05_q035": (0.35, 0.5, 0.35),
    "smooth8_h05_q035": (0.35, 0.5, 0.35),
}
ARCHIVED = {
    "t4_h05_q035": {"bayes": 0.216, "rung": "cnn2_w32_k5", "macs": 1_740_000},
    "smooth8_h05_q035": {"bayes": 0.219, "rung": "cnn2_w32_k5_gpool", "macs": 3_430_000},
}
PROFILE = Z1HardwareProfile()
WRITE_J = PROFILE.node_full_sram_write_energy_j
READ_J = PROFILE.node_read_energy_j
UPDATE_J = PROFILE.gibbs_node_update_energy_j


# ---------------------------------------------------------------------------
# Data and exact Bayes
# ---------------------------------------------------------------------------


def family(name):
    J, h0, q = FAMILIES[name]
    h = h0 * family_patterns(L)[name]
    return J, h, q


def test_set(name, n, salt):
    J, h, q = family(name)
    mix = FieldMixture(L, J, h)
    rng = np.random.default_rng([PROBE_SEED, salt, list(FAMILIES).index(name)])
    x = mix.sample(n, rng)
    y = flip(x, q, rng)
    return mix, x, y


def majority_wrong_prob(p_up, x):
    """Expected 16-draw majority error (ties to +1) from i.i.d. draws with marginals p_up."""
    p_pred_up = binom.sf(DRAWS // 2 - 1, DRAWS, p_up)  # P(#up >= 8)
    return np.where(x > 0, 1.0 - p_pred_up, p_pred_up)


def component_posterior(mix, y, q):
    g = 0.5 * np.log((1 - q) / q)
    lw = np.stack(
        [mix.grid.forward(hc[None] + g * y)[0] - lz for hc, lz in zip(mix.h, mix.logz, strict=True)]
    )
    return np.exp(lw - logsumexp(lw, axis=0)).T  # (images, C)


# ---------------------------------------------------------------------------
# Hand-built model in spin form
# ---------------------------------------------------------------------------


def build(J, h, q, design, centering, b0=0.0, lam=0.0):
    """Spin-form parameters of the unplaced model (no quantization)."""
    C = len(h)
    grid = GridIsing(L, J)
    logz = grid.forward(h)[0]
    m = h.mean(0) if centering == "centered" else np.zeros(N)
    d = h - m  # (C, N)
    if design == "bias":
        B = b0 - (logz - logz.mean())
        whh = 0.0
        extra_bh = 0.0
    elif design == "onehot":
        B = -(logz - logz.mean())
        whh = -lam / 2.0
        extra_bh = -lam * (C - 2) / 2.0
    else:
        raise ValueError(design)
    return {
        "J": J,
        "g": 0.5 * np.log((1 - q) / q),
        "W": d.T / 2.0,  # (N, C) pixel-hidden coupling
        "bp": m + d.sum(0) / 2.0,  # pixel bias
        "bh": B / 2.0 + extra_bh,  # hidden bias
        "whh": whh,  # uniform hidden-hidden coupling (one-hot only)
        "C": C,
    }


def quantize(p, bits):
    """``split`` codebook through the constraint layer: one cap for couplings, one for fields."""
    if bits is None:
        return dict(p)
    C = p["C"]
    ncoup = [np.full(2 * L * (L - 1), p["J"]), np.full(N, p["g"]), p["W"].ravel()]
    if p["whh"] != 0.0:
        ncoup.append(np.full(C * (C - 1) // 2, p["whh"]))
    couplings = np.concatenate(ncoup)
    fields = np.concatenate([p["bp"], p["bh"]])
    cc, cf = cap_for("split", couplings, fields, bits)
    qc = quantize_array(couplings, cc, bits)
    qf = quantize_array(fields, cf, bits)
    k = 2 * L * (L - 1)
    out = dict(p)
    out["J"] = float(qc[0])
    out["g"] = float(qc[k])
    out["W"] = qc[k + N : k + N + N * C].reshape(N, C)
    if p["whh"] != 0.0:
        out["whh"] = float(qc[-1])
    out["bp"], out["bh"] = qf[:N], qf[N:]
    out["caps"] = {"coupling": cc, "field": cf}
    return out


def hidden_states(C):
    return np.array(list(itertools.product((-1.0, 1.0), repeat=C)))


def exact_equilibrium(p, y, prune=1e-9, chunk=128):
    """Model's equilibrium pixel marginals P(x_i = +1 | y) and hidden-state weights."""
    C = p["C"]
    S = hidden_states(C)  # (2^C, C)
    hid = S @ p["bh"] + p["whh"] * ((S.sum(1) ** 2 - C) / 2.0)
    grid = GridIsing(L, p["J"])
    base = p["bp"][None] + S @ p["W"].T  # (2^C, N)
    out = np.empty((len(y), N))
    wts = np.empty((len(y), len(S)))
    for i in range(0, len(y), chunk):
        yy = y[i : i + chunk]
        fields = base[:, None, :] + p["g"] * yy[None]  # (2^C, B, N)
        lz = grid.forward(fields.reshape(-1, N))[0].reshape(len(S), len(yy))
        lw = hid[:, None] + lz
        w = np.exp(lw - logsumexp(lw, axis=0))  # (2^C, B)
        wts[i : i + chunk] = w.T
        acc = np.zeros((len(yy), N))
        for si in range(len(S)):
            keep = np.flatnonzero(w[si] > prune)
            if len(keep):
                pm, _ = grid.marginals(fields[si, keep])
                acc[keep] += w[si, keep, None] * pm
        out[i : i + chunk] = acc / wts[i : i + chunk].sum(1, keepdims=True).clip(1e-300)
    return out, wts, S


def summarize(p_up, x, bayes_img):
    mpm = (np.where(p_up > 0.5, 1.0, -1.0) != x).mean(1)
    maj = majority_wrong_prob(p_up, x).mean(1)
    return {
        "mpm_error": float(mpm.mean()),
        "mpm_gap_to_bayes": float((mpm - bayes_img).mean()),
        "mpm_gap_se": float((mpm - bayes_img).std(ddof=1) / np.sqrt(len(x))),
        "maj16_error": float(maj.mean()),
        "maj16_gap_to_bayes": float((maj - bayes_img).mean()),
    }


# ---------------------------------------------------------------------------
# NumPy chromatic Gibbs on the unplaced spin model
# ---------------------------------------------------------------------------


def gibbs_streams(p, y, K, start, rng, streams, per, reset_every=None, jitter=0.0):
    """Returns 16-draw majority predictions (images, N) and mean hidden switch count.

    Rows are DRAWS x streams chains. ``continue`` carries every free node from
    the previous image of the stream; ``random`` redraws them uniform for each
    image; ``reset`` redraws every ``reset_every`` images and carries otherwise.
    """
    C = p["C"]
    rows = DRAWS * streams
    r, c = np.divmod(np.arange(N), L)
    colour = (r + c) % 2
    blocks = [np.flatnonzero(colour == 0), np.flatnonzero(colour == 1)]
    gain_x = np.ones(N)
    gain_h = np.ones(C)
    if jitter:
        jr = np.random.default_rng([PROBE_SEED, 77])
        gain_x = 1.0 + jitter * (2 * jr.random(N) - 1)
        gain_h = 1.0 + jitter * (2 * jr.random(C) - 1)
    X = rng.integers(0, 2, (rows, N)) * 2.0 - 1.0
    H = rng.integers(0, 2, (rows, C)) * 2.0 - 1.0
    pred = np.empty((len(y), N))
    hid_last = np.empty((len(y), DRAWS, C))
    Whh = p["whh"] * (np.ones((C, C)) - np.eye(C))
    for j in range(per):
        idx = np.arange(streams) * per + j
        yy = np.repeat(y[idx], DRAWS, axis=0)  # row = stream * DRAWS + draw
        fresh = start == "random" or (start == "reset" and j % reset_every == 0)
        if fresh:
            X = rng.integers(0, 2, (rows, N)) * 2.0 - 1.0
            H = rng.integers(0, 2, (rows, C)) * 2.0 - 1.0
        ext = p["bp"][None] + p["g"] * yy
        for _ in range(K):
            for blk in blocks:
                Xg = X.reshape(rows, L, L)
                nb = np.zeros((rows, L, L))
                nb[:, 1:] += Xg[:, :-1]
                nb[:, :-1] += Xg[:, 1:]
                nb[:, :, 1:] += Xg[:, :, :-1]
                nb[:, :, :-1] += Xg[:, :, 1:]
                f = p["J"] * nb.reshape(rows, N)[:, blk] + ext[:, blk] + H @ p["W"][blk].T
                X[:, blk] = np.where(
                    rng.random((rows, len(blk))) < expit(2 * gain_x[blk] * f), 1.0, -1.0
                )
            fh = X @ p["W"] + p["bh"][None] + H @ Whh
            H = np.where(rng.random((rows, C)) < expit(2 * gain_h * fh), 1.0, -1.0)
        up = (X.reshape(streams, DRAWS, N) > 0).sum(1)
        pred[idx] = np.where(up >= DRAWS // 2, 1.0, -1.0)
        hid_last[idx] = H.reshape(streams, DRAWS, C)
    return pred, hid_last


def price(free_nodes, K, start, reset_every=None):
    """Per-sample reads, writes, free-node writes, updates and the joule band (host per step)."""
    if start == "continue":
        fw = 0.0
    elif start == "random":
        fw = float(free_nodes)
    else:
        fw = free_nodes / reset_every
    writes = N + fw  # clamp loads (N full SRAM writes) plus free-node writes
    updates = free_nodes * K
    e_hi = N * READ_J + writes * WRITE_J + updates * UPDATE_J
    e_lo = e_hi - N * WRITE_J + N * READ_J
    return {
        "reads": N,
        "writes": writes,
        "free_node_writes": fw,
        "updates": updates,
        "joule_per_sample_write_constant": e_hi,
        "joule_per_sample_clamp_at_read": e_lo,
        "joule_per_16_draw_estimate_write_constant": DRAWS * e_hi,
        "joule_per_16_draw_estimate_clamp_at_read": DRAWS * e_lo,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def exact_stage(name, n_dev, n_test, b0_grid, lam_grid, bits_list):
    t0 = time.process_time()
    J, h, q = family(name)
    mix, x, y = test_set(name, n_test, 1)
    post = mix.posterior(y, q)
    bayes_img = (np.where(post > 0.5, 1.0, -1.0) != x).mean(1)
    comp = component_posterior(mix, y, q)
    rec = {
        "name": name,
        "C": len(h),
        "n_test": n_test,
        "noisy": float((y != x).mean()),
        "bayes_mpm": float(bayes_img.mean()),
        "bayes_maj16_floor": float(majority_wrong_prob(post, x).mean()),
        "bayes_maj16_floor_gap": float(majority_wrong_prob(post, x).mean() - bayes_img.mean()),
        "component_posterior_max_mean": float(comp.max(1).mean()),
        "component_posterior_max_quantiles": np.quantile(comp.max(1), [0.1, 0.25, 0.5]).tolist(),
        "archived": ARCHIVED[name],
    }
    # Oracle check: one-hot restricted enumeration reproduces Bayes.
    p_ref = build(J, h, q, "onehot", "centered", lam=0.0)
    S = hidden_states(len(h))
    onehot_rows = np.flatnonzero(S.sum(1) == 2 - len(h))
    grid = GridIsing(L, J)
    lw, pm = [], []
    for si in onehot_rows:
        f = p_ref["bp"][None] + S[si] @ p_ref["W"].T + p_ref["g"] * y[:200]
        pmi, lz = grid.marginals(f)
        lw.append(lz + S[si] @ p_ref["bh"])
        pm.append(pmi)
    w = np.exp(np.stack(lw) - logsumexp(np.stack(lw), axis=0))
    oracle = np.einsum("cb,cbn->bn", w, np.stack(pm))
    rec["oracle_onehot_max_abs_diff_vs_bayes"] = float(np.abs(oracle - post[:200]).max())

    # Development set (fresh salt) to choose b0 / lambda; then score on the test set.
    mix_d, xd, yd = test_set(name, n_dev, 2)
    post_d = mix_d.posterior(yd, q)
    bayes_d = (np.where(post_d > 0.5, 1.0, -1.0) != xd).mean(1)
    scans = {}
    # 2^C hidden states: on C = 8 the raw centering (lost by 0.04 on t4) and the
    # outer b0 values are skipped to stay inside the 30 CPU-minute probe budget.
    centerings = ("centered", "raw") if len(h) <= 4 else ("centered",)
    if len(h) > 4:
        b0_grid = [v for v in b0_grid if -10.0 <= v <= 0.0]
        lam_grid = [v for v in lam_grid if v <= 32.0]
    for centering in centerings:
        rows = []
        for b0 in b0_grid:
            pu, _, _ = exact_equilibrium(build(J, h, q, "bias", centering, b0=b0), yd)
            rows.append({"b0": b0, **summarize(pu, xd, bayes_d)})
        scans[f"bias_{centering}"] = rows
    rows = []
    for lam in lam_grid:
        pu, _, _ = exact_equilibrium(build(J, h, q, "onehot", "centered", lam=lam), yd)
        rows.append({"lambda": lam, **summarize(pu, xd, bayes_d)})
    scans["onehot_centered"] = rows
    rec["dev_scans"] = scans

    best = {}
    for key, rows in scans.items():
        r = min(rows, key=lambda t: t["mpm_error"])
        best[key] = r
    rec["dev_best"] = best

    arms = []
    for key in scans:
        design, centering = key.split("_")
        par = best[key]
        for bits in bits_list:
            kw = {"b0": par["b0"]} if design == "bias" else {"lam": par["lambda"]}
            p = quantize(build(J, h, q, design, centering, **kw), bits)
            pu, wts, S = exact_equilibrium(p, y)
            on = (S > 0).sum(1)
            arm = {
                "design": design,
                "centering": centering,
                **kw,
                "bits": bits,
                **summarize(pu, x, bayes_img),
                "hidden_mass_none_on": float(wts[:, on == 0].sum(1).mean()),
                "hidden_mass_one_on": float(wts[:, on == 1].sum(1).mean()),
                "hidden_mass_multi_on": float(wts[:, on >= 2].sum(1).mean()),
                "caps": p.get("caps"),
            }
            if bits is not None:
                arm["quantized_J"] = p["J"]
                arm["quantized_g"] = p["g"]
                arm["distinct_bp_levels"] = int(len(np.unique(p["bp"])))
            arms.append(arm)
            print(name, key, bits, round(arm["mpm_gap_to_bayes"], 4), flush=True)
    rec["exact_arms"] = arms
    rec["exact_cpu_s"] = time.process_time() - t0
    return rec, (J, h, q, x, y, bayes_img, post)


def gibbs_stage(name, data, design, centering, kw, bits, Ks, starts, streams, per, jitter):
    J, h, q, x, y, bayes_img, post = data
    p = quantize(build(J, h, q, design, centering, **kw), bits)
    n = streams * per
    x, y, b_img = x[:n], y[:n], bayes_img[:n]
    pu, _, _ = exact_equilibrium(p, y)
    model_maj = majority_wrong_prob(pu, x).mean(1)
    out = {
        "design": design,
        "centering": centering,
        **kw,
        "bits": bits,
        "jitter": jitter,
        "images": n,
        "bayes_mpm": float(b_img.mean()),
        "bayes_maj16_floor": float(majority_wrong_prob(post[:n], x).mean()),
        "model_equilibrium_maj16": float(model_maj.mean()),
        "cells": [],
    }
    for start, N_reset in starts:
        for K in Ks:
            t0 = time.process_time()
            code = {"continue": 0, "random": 1, "reset": 2}[start]
            rng = np.random.default_rng(
                [PROBE_SEED, 9, K, code, N_reset or 0, bits or 0, int(jitter * 100)]
            )
            pred, hid = gibbs_streams(p, y, K, start, rng, streams, per, N_reset, jitter)
            err = (pred != x).mean(1)
            d = (err - b_img).reshape(streams, per).mean(1)
            # stream-bootstrap 95% interval
            br = np.random.default_rng([PROBE_SEED, 10])
            bs = d[br.integers(0, streams, (2000, streams))].mean(1)
            cell = {
                "start": start,
                "reset_every": N_reset,
                "K": K,
                "error": float(err.mean()),
                "gap_to_bayes": float(d.mean()),
                "gap_ci95": np.quantile(bs, [0.025, 0.975]).tolist(),
                "gap_to_model_maj16": float((err - model_maj).mean()),
                "hidden_on_fraction": float((hid > 0).mean()),
                "cpu_s": time.process_time() - t0,
                "price_unplaced": price(N + p["C"], K, start, N_reset),
            }
            out["cells"].append(cell)
            print(name, design, bits, start, N_reset, K, round(cell["gap_to_bayes"], 4), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--streams", type=int, default=25)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--no-gibbs", action="store_true")
    a = ap.parse_args()
    t0 = time.process_time()
    res = {
        "probe_seed": PROBE_SEED,
        "label": "exploration, not evidence",
        "evidence_classes": {
            "bayes_and_model_equilibria": "exact_reference",
            "gibbs_errors": "software_simulation (NumPy, CPU)",
            "joules": "calibrated_projection",
        },
        "profile_constants": {"write_j": WRITE_J, "read_j": READ_J, "update_j": UPDATE_J},
        "families": [],
    }
    b0_grid = [float(v) for v in np.arange(-12.0, 12.1, 1.0)]
    lam_grid = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    bits_list = [None, 8, 6, 4]
    for name in a.families:
        # Budget: 2^8 hidden states make C = 8 sixteen times dearer; its test set
        # is the 500 images the sampled stage scores anyway.
        n_test = a.n_test if len(family(name)[1]) <= 4 else a.streams * a.per
        rec, data = exact_stage(name, a.n_dev, n_test, b0_grid, lam_grid, bits_list)
        if not a.no_gibbs:
            # Sampled arm: stage A's bias design, centered, at its development b0.
            kw = {"b0": rec["dev_best"]["bias_centered"]["b0"]}
            g = []
            starts = [("continue", None), ("random", None), ("reset", 5)]
            for bits in (None, 8):
                g.append(
                    gibbs_stage(
                        name,
                        data,
                        "bias",
                        "centered",
                        kw,
                        bits,
                        (64, 256, 1024),
                        starts,
                        a.streams,
                        a.per,
                        0.0,
                    )
                )
            for bits, jit in ((6, 0.0), (8, 0.1)):
                g.append(
                    gibbs_stage(
                        name,
                        data,
                        "bias",
                        "centered",
                        kw,
                        bits,
                        (64, 256, 1024),
                        [("random", None)],
                        a.streams,
                        a.per,
                        jit,
                    )
                )
            # One-hot control at 8 bits (range-hungry, AM A): random start only.
            g.append(
                gibbs_stage(
                    name,
                    data,
                    "onehot",
                    "centered",
                    {"lam": rec["dev_best"]["onehot_centered"]["lambda"]},
                    8,
                    (1024,),
                    [("random", None)],
                    a.streams,
                    a.per,
                    0.0,
                )
            )
            rec["gibbs"] = g
        res["families"].append(rec)
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    res["total_cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")


if __name__ == "__main__":
    main()
