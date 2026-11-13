"""Exploratory probe for queue row ``aggregator-read-gain`` (loop P-0012).

Exploration, not a recorded study and not evidence. Every number is an
``exact_reference`` equilibrium quantity of the frozen adsorption pore, or of a
one-way listener attached to it, computed by transfer matrix on a CPU. No
sampling, no seeds, no training, no hardware. Run from the repository root:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
      uv run python docs/research/aggregator_read_gain_probe/probe.py \\
      --output docs/research/aggregator_read_gain_probe/probe.json

``--self-test`` runs the small-system enumeration checks only.

Fixed before any result was computed (committed with this script, before it ran)
------------------------------------------------------------------------------

* Model. The frozen ``native-adsorption-eligibility`` pore: open 3x3x16 lattice
  gas, beta = epsilon = 1, wall field phi = 3, the archived spin map
  (``adsorption_parameters``) and its three frozen chemical potentials. The
  archived transfer helper and probe JSON are loaded through the hash-checked
  loaders of ``thermo_lab.adsorption_precision_requirement`` (imported, never
  edited). The model is float: the row asks about the read path, so the pore's
  own rounding (P-0009) is not applied; both arms read the same model.
* Aggregator. A ~ Bernoulli(sigmoid(2 g (w S_B + b))), S_B the spin sum of a
  block B of k = 16 sites, couplings one-way (model -> aggregator), so the pore's
  law is unchanged (relaxed constraint, rule 8). w in {0.25, 0.5, 1}. g is the
  aggregator's static gain: 1, or 0.9 and 1.1 for the 10 percent jitter arms.
  Jitter is applied to the aggregator only. Per-site jitter on the pore has no
  exact stationary law (archived probe note), so it is out of scope here.
* One read. One node read of one aggregator (aggregator arm) or of one site
  (baseline), taken at an instant thinned far enough in time that successive
  reads are independent. That is the most favourable case for the baseline;
  spatial spacing beyond the correlation length is implied by it.
* Metric. Fisher information about loading L per read, in the one-parameter
  family indexed by the chemical potential mu: I_L = (dp/dmu)^2 / (p (1 - p)) /
  (dL/dmu)^2. Reads to reach loading error 0.01 = 1 / (I_L 0.01^2), i.e. one
  asymptotic standard error of the efficient (maximum-likelihood) estimator.
  The read gain is a ratio of reads, so the 1-sigma convention cancels.
* Primary baseline ``B_rr``: round-robin model-based reads of all 144 sites,
  I = mean_i I_i. Also reported: a uniformly random site read (Bernoulli(L),
  1 / (L(1-L)), model-free), the stratified sample mean (model-free) and the
  best single site (``B_best``).
* Primary aggregator arm ``A_rr``: round-robin reads of one aggregator per
  block over the 9 blocks that tile the pore. Also reported: the best block.
* Primary block geometry ``columns``: the 9 (x, y) channels, 16 sites each along
  z. Secondary ``slabs``: 16 consecutive sites in site order (z, y, x), the most
  compact tiling, reported as a sensitivity.
* Primary threshold rule ``half``: the host adapts b until the aggregator fires
  at rate 0.5 on the device (bisection; ``bits`` writes per aggregator per
  chemical potential, plus one programming write per aggregator). Secondary
  rules: ``opt`` (oracle Fisher-optimal b, an upper bound, not implementable),
  ``fixed_mid`` (b fixed at the middle point's half-rate value, no adaptation),
  ``bank3`` (three aggregators per block at the three points' half-rate values,
  read round-robin; ``bank3_select`` reads the best member).
* Precision. Float; then 4 and 6 bits on the constraint layer's ``codebook``.
  Primary caps ``own``: coupling cap = w (exact), bias cap = the largest float
  half-rate |b| of that aggregator over the three points. Secondary caps
  ``pore``: the archived family split caps (coupling 0.25, field 1.9904...);
  w above 0.25 then clips, and b clips at the field cap. With a code grid the
  host picks the code whose actual firing rate is nearest 0.5.
* Jitter bias. The host inverts with g = 1. The loading bias is
  L(mu_hat) - L(mu), mu_hat solving p_nominal(mu_hat) = p_actual(mu). A bias
  above 0.01 cannot be removed by reads.
* Prediction (row): gain A_rr / B_rr >= 3 at all three chemical potentials.
  Falsifier: a gain under 3 with correlations included. The verdict is read for
  each w on the primary geometry, rule and caps, at float, 4 and 6 bits and g in
  {0.9, 1, 1.1}. If even the oracle rule gives a gain under 3 at some chemical
  potential for every w, the falsifier fires regardless of the rule.
* Counterfactual: the same block with independent sites at the exact marginals
  (Poisson-binomial), to isolate the within-block correlation penalty.
* Stop case reported as a number, not a variant: with symmetric couplings the
  aggregator back-acts (adds log 2cosh(w S_B + b) to the energy); the loading
  shift it causes is computed.
* Placement: the largest number of pore sites within the 16 Z1 offsets of any
  free slot of the archived map (x + 4z, y + z), against k = 16.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit

from thermo_lab import adsorption_precision_requirement as apr
from thermo_lab.constraints.lattice import OFFSETS
from thermo_lab.constraints.quantize import codebook

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
SHAPE = (3, 3, 16)
NX, NY, NZ = SHAPE
WIDTH = NX * NY
NSITES = NX * NY * NZ
K = 16
WS = (0.25, 0.5, 1.0)
BITS = (4, 6)
GAINS = (0.9, 1.0, 1.1)
TOL = 0.01
TARGET_GAIN = 3.0
FD = 1e-4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- exact count distributions -------------------------------------------------------------


def _bits_table(width: int) -> np.ndarray:
    return ((np.arange(1 << width)[:, None] >> np.arange(width)[None, :]) & 1).astype(int)


def _transfer_linear(vec: np.ndarray, couplings: np.ndarray) -> np.ndarray:
    """Apply the product of 2x2 factors exp(J s t), one per bit, on the last axis."""
    out = np.array(vec, copy=True)
    lead = out.shape[:-1]
    n = out.shape[-1]
    for bit, j in enumerate(couplings):
        view = out.reshape(*lead, n // (2 << bit), 2, 1 << bit)
        lo = view[..., 0, :].copy()
        hi = view[..., 1, :].copy()
        ep, em = np.exp(j), np.exp(-j)
        view[..., 0, :] = lo * ep + hi * em
        view[..., 1, :] = lo * em + hi * ep
    return out


def count_distribution(shape, weights, bias, masks) -> np.ndarray:
    """Exact joint law of the occupied-site counts inside one or two site masks.

    Linear-domain transfer with per-slice renormalization. Returns an array of
    shape (|m1|+1,) or (|m1|+1, |m2|+1) summing to one.
    """
    ref = apr.load_reference()
    nx, ny, nz = shape
    width = nx * ny
    coords, edges, _ = ref.geometry(nx, ny, nz)
    weights = np.broadcast_to(np.asarray(weights, float), (len(edges),))
    bias = np.broadcast_to(np.asarray(bias, float), (len(coords),))
    occ = _bits_table(width)
    spins = 2 * occ - 1
    local = bias.reshape(nz, width) @ spins.T
    vertical = np.zeros((max(nz - 1, 0), width))
    for e, (u, v) in enumerate(edges):
        z, a = divmod(int(u), width)
        zz, b = divmod(int(v), width)
        if z == zz:
            local[z] += weights[e] * spins[:, a] * spins[:, b]
        else:
            vertical[z, a] = weights[e]
    masks = [np.asarray(m, bool) for m in masks]
    if len(masks) == 1:
        masks.append(np.zeros(len(coords), bool))
        single = True
    else:
        single = False
    c1, c2 = int(masks[0].sum()) + 1, int(masks[1].sum()) + 1
    forward = np.zeros((c1, c2, 1 << width))
    forward[0, 0, :] = 1.0
    for z in range(nz):
        if z > 0:
            forward = _transfer_linear(forward, vertical[z - 1])
        forward = forward * np.exp(local[z] - local[z].max())[None, None, :]
        d1 = occ[:, masks[0].reshape(nz, width)[z]].sum(1)
        d2 = occ[:, masks[1].reshape(nz, width)[z]].sum(1)
        shifted = np.zeros_like(forward)
        for a, b in set(zip(d1.tolist(), d2.tolist(), strict=True)):
            idx = np.flatnonzero((d1 == a) & (d2 == b))
            shifted[a:, b:, idx] += forward[: c1 - a, : c2 - b, idx]
        forward = shifted / shifted.sum()
    law = forward.sum(-1)
    law = law / law.sum()
    return law[:, 0] if single else law


def brute_count(shape, weights, bias, masks) -> np.ndarray:
    """Independent enumeration reference (no transfer), capped at 16 sites."""
    ref = apr.load_reference()
    coords, edges, _ = ref.geometry(*shape)
    if len(coords) > 16:
        raise ValueError("enumeration capped at 16 sites")
    s = np.array(list(itertools.product((-1, 1), repeat=len(coords))))
    logw = s @ np.broadcast_to(bias, (len(coords),)) + (s[:, edges[:, 0]] * s[:, edges[:, 1]]) @ (
        np.broadcast_to(weights, (len(edges),))
    )
    prob = np.exp(logw - logw.max())
    prob /= prob.sum()
    n = (s + 1) // 2
    counts = [n[:, np.asarray(m, bool)].sum(1) for m in masks]
    shape_out = tuple(int(np.asarray(m, bool).sum()) + 1 for m in masks)
    out = np.zeros(shape_out)
    np.add.at(out, tuple(counts), prob)
    return out


def self_test() -> float:
    ref = apr.load_reference()
    rng = np.random.default_rng(20261113)
    worst = 0.0
    for shape in ((2, 2, 3), (1, 3, 4), (2, 1, 5), (3, 1, 4)):
        _, edges, _ = ref.geometry(*shape)
        n = int(np.prod(shape))
        for _ in range(3):
            w = rng.normal(size=len(edges))
            h = rng.normal(size=n) * 1.5
            m1 = rng.random(n) < 0.5
            m2 = np.ones(n, bool)
            for masks in ([m1], [m1, m2]):
                a = count_distribution(shape, w, h, masks)
                b = brute_count(shape, w, h, masks)
                worst = max(worst, float(np.max(np.abs(a - b))))
    if worst > 1e-12:
        raise AssertionError(f"count distribution mismatch {worst}")
    return worst


# --- model quantities ----------------------------------------------------------------------


def params(mu: float) -> dict:
    return apr.load_reference().adsorption_parameters(mu, shape=SHAPE)


def site_stats(mu: float) -> np.ndarray:
    p = params(mu)
    return np.asarray(
        apr.load_reference().exact_stats(SHAPE, p["weights"], p["bias"])["site_occupancies"]
    )


def loading(mu: float) -> float:
    p = params(mu)
    return float(
        apr.load_reference().exact_aggregate(SHAPE, p["weights"], p["bias"])["mean_occupancy"]
    )


def block_law(mu: float, mask: np.ndarray) -> np.ndarray:
    p = params(mu)
    return count_distribution(SHAPE, p["weights"], p["bias"], [mask])


def poisson_binomial(theta: np.ndarray) -> np.ndarray:
    law = np.array([1.0])
    for t in theta:
        law = np.convolve(law, [1 - t, t])
    return law


def blocks(geometry: str) -> list[np.ndarray]:
    out = []
    for j in range(9):
        m = np.zeros(NSITES, bool)
        if geometry == "columns":
            m[[z * WIDTH + j for z in range(NZ)]] = True
        elif geometry == "slabs":
            m[16 * j : 16 * j + 16] = True
        else:
            raise ValueError(geometry)
        out.append(m)
    return out


# --- aggregator ----------------------------------------------------------------------------

SPIN_SUM = 2 * np.arange(K + 1) - K


def fire(law: np.ndarray, w: float, b: float, g: float = 1.0) -> float:
    return float(law @ expit(2 * g * (w * SPIN_SUM + b)))


def fisher_l(law, dlaw, dl, w, b, g=1.0) -> float:
    s = expit(2 * g * (w * SPIN_SUM + b))
    p = float(law @ s)
    dp = float(dlaw @ s)
    if p <= 0 or p >= 1:
        return 0.0
    return dp * dp / (p * (1 - p)) / (dl * dl)


def b_half(law, w, g=1.0) -> float:
    lim = w * K + 40.0
    return float(brentq(lambda b: fire(law, w, b, g) - 0.5, -lim, lim, xtol=1e-13))


def b_opt(law, dlaw, dl, w, g=1.0) -> tuple[float, float]:
    grid = np.linspace(-w * K - 6, w * K + 6, 4001)
    vals = np.array([fisher_l(law, dlaw, dl, w, b, g) for b in grid])
    i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(
        lambda b: -fisher_l(law, dlaw, dl, w, b, g),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(res.x), float(-res.fun)


def code_half(law, wq, cap_b, bits, g) -> float:
    """The bias code whose actual firing rate is nearest 0.5."""
    levels = 2 ** (bits - 1) - 1
    codes = codebook(np.arange(-levels, levels + 1) * (cap_b / levels), cap_b, bits)
    rates = np.array([fire(law, wq, c, g) for c in codes])
    return float(codes[int(np.argmin(np.abs(rates - 0.5)))])


# --- placement ----------------------------------------------------------------------------


def placement_check() -> dict:
    offsets = list(OFFSETS)  # the Z1 profile's 16 interior offsets
    ref = apr.load_reference()
    coords, _, _ = ref.geometry(*SHAPE)
    slots = {(int(x + 4 * z), int(y + z)): i for i, (x, y, z) in enumerate(coords)}
    xs = [s[0] for s in slots]
    ys = [s[1] for s in slots]
    best, best_slot = 0, None
    for sx in range(min(xs) - 5, max(xs) + 6):
        for sy in range(min(ys) - 5, max(ys) + 6):
            if (sx, sy) in slots:
                continue
            seen = sum((sx + dx, sy + dy) in slots for dx, dy in offsets)
            if seen > best:
                best, best_slot = seen, (sx, sy)
    return {
        "offsets": len(offsets),
        "max_pore_sites_seen_by_one_free_slot": best,
        "slot": best_slot,
        "k_required": K,
        "k16_one_hop_placeable": best >= K,
    }


# --- main ----------------------------------------------------------------------------------


def correlation_length(mu: float) -> dict:
    p = params(mu)
    a = 4 * WIDTH + 4  # centre channel (1, 1), z = 4
    out = {}
    for d in range(1, 9):
        m1 = np.zeros(NSITES, bool)
        m2 = np.zeros(NSITES, bool)
        m1[a] = True
        m2[a + d * WIDTH] = True
        j = count_distribution(SHAPE, p["weights"], p["bias"], [m1, m2])
        t1, t2 = j[1, :].sum(), j[:, 1].sum()
        cov = j[1, 1] - t1 * t2
        out[d] = float(cov / np.sqrt(t1 * (1 - t1) * t2 * (1 - t2)))
    r = np.array([out[d] for d in range(1, 9)])
    good = r > 1e-12
    ds = np.arange(1, 9)[good]
    slope = np.polyfit(ds, np.log(r[good]), 1)[0] if good.sum() >= 2 else float("nan")
    return {"centre_channel_r_of_dz": out, "xi_sites": float(-1 / slope)}


def symmetric_shift(mu: float, mask: np.ndarray, w: float, b: float, l0: float) -> float:
    p = params(mu)
    joint = count_distribution(SHAPE, p["weights"], p["bias"], [mask, np.ones(NSITES, bool)])
    tilt = np.cosh(w * SPIN_SUM + b)[:, None]
    n = np.arange(NSITES + 1)[None, :]
    lsym = float((joint * tilt * n).sum() / (joint * tilt).sum() / NSITES)
    return lsym - l0


def run(output: Path) -> None:
    t0 = time.process_time()
    wall0 = time.perf_counter()
    archive = apr.load_archive()
    mus = [float(m) for m in archive["derived_numerical_inputs"]["chemical_potentials"]]
    pore_caps = archive["family_quantization"]
    st = self_test()
    print(f"self-test max error {st:.2e}", flush=True)

    # Point-level quantities.
    points = []
    for mu in mus:
        th = site_stats(mu)
        dth = (site_stats(mu + FD) - site_stats(mu - FD)) / (2 * FD)
        l0 = loading(mu)
        dl = (loading(mu + FD) - loading(mu - FD)) / (2 * FD)
        p = params(mu)
        nlaw = count_distribution(SHAPE, p["weights"], p["bias"], [np.ones(NSITES, bool)])
        n = np.arange(NSITES + 1)
        var_n = float(nlaw @ n**2 - (nlaw @ n) ** 2)
        i_sites = dth**2 / (th * (1 - th)) / dl**2
        base = {
            "B_rr": float(i_sites.mean()),
            "B_best": float(i_sites.max()),
            "B_random_site": float(1 / (l0 * (1 - l0))),
            "B_stratified_mean": float(1 / np.mean(th * (1 - th))),
        }
        points.append(
            {
                "mu": mu,
                "loading": l0,
                "dL_dmu_fd": float(dl),
                "dL_dmu_varN": var_n / NSITES,
                "var_N": var_n,
                "var_N_if_independent": float(np.sum(th * (1 - th))),
                "fisher_per_read": base,
                "reads_to_0p01": {k: float(1 / (v * TOL**2)) for k, v in base.items()},
                "correlation": correlation_length(mu),
                "_th": th,
                "_dth": dth,
                "_dl": dl,
            }
        )
        print(f"mu={mu:+.4f} L={l0:.6f} baseline {base}", flush=True)

    results = {}
    for geometry in ("columns", "slabs"):
        geo = {"blocks": []}
        laws = []
        for bi, mask in enumerate(blocks(geometry)):
            per_point = []
            for pt in points:
                mu = pt["mu"]
                law = block_law(mu, mask)
                dlaw = (block_law(mu + FD, mask) - block_law(mu - FD, mask)) / (2 * FD)
                th, dth = pt["_th"][mask], pt["_dth"][mask]
                iid = poisson_binomial(th)
                diid = (poisson_binomial(th + FD * dth) - poisson_binomial(th - FD * dth)) / (
                    2 * FD
                )
                m = np.arange(K + 1)
                var_m = float(law @ m**2 - (law @ m) ** 2)
                per_point.append(
                    {
                        "law": law,
                        "dlaw": dlaw,
                        "iid": iid,
                        "diid": diid,
                        "var_M": var_m,
                        "var_M_if_independent": float(np.sum(th * (1 - th))),
                    }
                )
            laws.append(per_point)
            geo["blocks"].append(
                {
                    "block": bi,
                    "block_loading": [float(pt["_th"][mask].mean()) for pt in points],
                    "var_M_over_independent": [
                        pp["var_M"] / pp["var_M_if_independent"] for pp in per_point
                    ],
                }
            )
        cells = []
        for w in WS:
            # Float half-rate biases per block and point, g = 1 (also sets the own bias cap).
            bh = [[b_half(laws[bi][pi]["law"], w) for pi in range(3)] for bi in range(9)]
            for caps in ("own", "pore"):
                for prec in ("float", 4, 6):
                    if prec == "float" and caps == "pore":
                        continue
                    for g in GAINS:
                        cell = {"w": w, "caps": caps, "precision": prec, "gain_g": g, "points": []}
                        if prec == "float":
                            wq = w
                        else:
                            cap_w = w if caps == "own" else pore_caps["coupling_cap"]
                            wq = float(codebook(np.array([w]), cap_w, prec)[0])
                        cell["w_programmed"] = wq
                        for pi, pt in enumerate(points):
                            dl = pt["_dl"]
                            per_rule = {
                                r: []
                                for r in (
                                    "half",
                                    "opt",
                                    "fixed_mid",
                                    "bank3",
                                    "bank3_select",
                                    "half_iid",
                                )
                            }
                            biases = []
                            for bi in range(9):
                                L = laws[bi][pi]
                                if prec == "float":
                                    bsel = b_half(L["law"], wq, g)
                                    bmid = bh[bi][1]
                                    bank = bh[bi]
                                else:
                                    cap_b = (
                                        max(abs(x) for x in bh[bi])
                                        if caps == "own"
                                        else pore_caps["field_cap"]
                                    )
                                    bsel = code_half(L["law"], wq, cap_b, prec, g)
                                    bmid = float(codebook(np.array([bh[bi][1]]), cap_b, prec)[0])
                                    bank = list(codebook(np.array(bh[bi]), cap_b, prec))
                                biases.append(bsel)

                                def f(b, L=L, dl=dl, wq=wq, g=g):
                                    return fisher_l(L["law"], L["dlaw"], dl, wq, b, g)

                                per_rule["half"].append(f(bsel))
                                per_rule["fixed_mid"].append(f(bmid))
                                bank_i = [f(b) for b in bank]
                                per_rule["bank3"].append(float(np.mean(bank_i)))
                                per_rule["bank3_select"].append(float(np.max(bank_i)))
                                if prec == "float":
                                    per_rule["opt"].append(b_opt(L["law"], L["dlaw"], dl, wq, g)[1])
                                    bi_iid = b_half(L["iid"], wq, g)
                                    per_rule["half_iid"].append(
                                        fisher_l(L["iid"], L["diid"], dl, wq, bi_iid, g)
                                    )
                            b_rr = pt["fisher_per_read"]["B_rr"]
                            b_best = pt["fisher_per_read"]["B_best"]
                            summary = {}
                            for rule, vals in per_rule.items():
                                if not vals:
                                    continue
                                summary[rule] = {
                                    "fisher_rr": float(np.mean(vals)),
                                    "fisher_best_block": float(np.max(vals)),
                                    "gain_rr": float(np.mean(vals) / b_rr),
                                    "gain_best_vs_best_site": float(np.max(vals) / b_best),
                                    "reads_rr_to_0p01": float(1 / (np.mean(vals) * TOL**2)),
                                }
                            cell["points"].append(
                                {
                                    "mu": pt["mu"],
                                    "rules": summary,
                                    "half_biases": [float(b) for b in biases],
                                    "per_block_half_fisher": [float(v) for v in per_rule["half"]],
                                }
                            )
                        cell["min_gain_half_rr"] = min(
                            p["rules"]["half"]["gain_rr"] for p in cell["points"]
                        )
                        if prec == "float":
                            cell["min_gain_opt_rr"] = min(
                                p["rules"]["opt"]["gain_rr"] for p in cell["points"]
                            )
                        cells.append(cell)
                        print(
                            f"{geometry} w={w} {caps} {prec} g={g}: half gains "
                            + " ".join(
                                f"{p['rules']['half']['gain_rr']:.3f}" for p in cell["points"]
                            ),
                            flush=True,
                        )
        geo["cells"] = cells
        # Jitter inversion bias, primary rule, both cap rules, columns and slabs.
        bias_rows = []
        for w in WS:
            bh = [[b_half(laws[bi][pi]["law"], w) for pi in range(3)] for bi in range(9)]
            for caps in ("own", "pore"):
                for prec in ("float", 4, 6):
                    if prec == "float" and caps == "pore":
                        continue
                    if prec == "float":
                        wq = w
                    else:
                        cap_w = w if caps == "own" else pore_caps["coupling_cap"]
                        wq = float(codebook(np.array([w]), cap_w, prec)[0])
                    for g in (0.9, 1.1):
                        worst = 0.0
                        for pi, pt in enumerate(points):
                            for bi, mask in enumerate(blocks(geometry)):
                                L = laws[bi][pi]
                                if prec == "float":
                                    bsel = b_half(L["law"], wq, g)
                                else:
                                    cap_b = (
                                        max(abs(x) for x in bh[bi])
                                        if caps == "own"
                                        else pore_caps["field_cap"]
                                    )
                                    bsel = code_half(L["law"], wq, cap_b, prec, g)
                                target = fire(L["law"], wq, bsel, g)
                                mu = pt["mu"]

                                def resid(m, mask=mask, bsel=bsel, target=target, wq=wq):
                                    return fire(block_law(m, mask), wq, bsel, 1.0) - target

                                lo, hi = mu - 1.5, mu + 1.5
                                if resid(lo) * resid(hi) > 0:
                                    worst = float("inf")
                                    continue
                                mh = brentq(resid, lo, hi, xtol=1e-9)
                                worst = max(worst, abs(loading(mh) - pt["loading"]))
                        bias_rows.append(
                            {
                                "w": w,
                                "caps": caps,
                                "precision": prec,
                                "gain_g": g,
                                "max_abs_loading_bias": worst,
                            }
                        )
                        print(f"{geometry} bias w={w} {caps} {prec} g={g}: {worst:.5f}", flush=True)
        geo["jitter_inversion_bias"] = bias_rows
        # Symmetric back-action (stop case), float half-rate bias.
        sym = []
        for w in WS:
            worst = 0.0
            for pi, pt in enumerate(points):
                for bi, mask in enumerate(blocks(geometry)):
                    b = b_half(laws[bi][pi]["law"], w)
                    worst = max(worst, abs(symmetric_shift(pt["mu"], mask, w, b, pt["loading"])))
            sym.append({"w": w, "max_abs_loading_shift_one_aggregator": worst})
        geo["symmetric_backaction"] = sym
        results[geometry] = geo

    for pt in points:
        for k in ("_th", "_dth", "_dl"):
            pt.pop(k)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out = {
        "label": "exploration, not evidence; exact_reference quantities; CPU only; no sampling",
        "proposal": "P-0012",
        "row": "aggregator-read-gain",
        "script_sha256": sha256_file(HERE),
        "imported_runner": "src/thermo_lab/adsorption_precision_requirement.py",
        "imported_runner_sha256": sha256_file(
            ROOT / "src/thermo_lab/adsorption_precision_requirement.py"
        ),
        "helper": apr.HELPER,
        "helper_sha256": apr.HELPER_SHA256,
        "archive_json": apr.ARCHIVE_JSON,
        "archive_json_sha256": apr.ARCHIVE_JSON_SHA256,
        "frozen_mu": mus,
        "pore_caps": pore_caps,
        "fixed": {
            "k": K,
            "ws": WS,
            "bits": BITS,
            "gains": GAINS,
            "tol": TOL,
            "target_gain": TARGET_GAIN,
            "fd_step": FD,
            "reads_definition": "1 / (I_L * tol^2), one asymptotic standard error",
        },
        "self_test_max_error": st,
        "placement": placement_check(),
        "points": points,
        "results": results,
        "cpu_seconds": time.process_time() - t0,
        "wall_seconds": time.perf_counter() - wall0,
        "max_rss_mib": usage.ru_maxrss / 1024,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    output.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print(f"cpu {out['cpu_seconds']:.1f}s wall {out['wall_seconds']:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        print(f"PASS self-test, max error {self_test():.2e}")
        return
    if not args.output:
        ap.error("--output is required")
    path = Path(args.output)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    run(path)


if __name__ == "__main__":
    main()
