"""P-0015 round-2 probe: pooled gain-spread test, scan attribution margin, scan temperature.

Exploration, not evidence. CPU only:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run python docs/research/z1_acceptance_kit_probe_r2.py --out <json>

Answers the round-1 review on PR #174:

G. Static beta variation as the row defines it: every site's gain drawn
   uniform in [1 - j, 1 + j]. Primary test: one bias on all isolated sites,
   pooled per read; Pearson homogeneity statistic over sites (df N - 1).
   Read counts from the noncentral chi-square approximation on exact per-site
   laws, checked by simulation. Secondary: one outlying site at +-delta,
   sign-agnostic, one bias, exact two-sided binomial (alpha/2 per tail).
A. Attribution margin for the Bonferroni missing-coupling scan at 0.05 / E:
   coherent bias kappa and heterogeneous rms rho, per edge in units of that
   edge's standard error at the scan length.
B. Temperature on the scan statistic: per-edge exact c0, c1 (all edges, 64 and
   256 spins), per-edge scatter ratios from calibration chains (one seed set)
   and a second independent chain set, at a fixed window of 4,096 reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import z1_acceptance_kit_review_probe as rv  # noqa: E402

from thermo_lab import planar_16_offset_ferro as p16  # noqa: E402

ALPHA, MISS, LIMIT = 0.05, 0.05, 10_000
ZB = float(stats.norm.isf(MISS))
SIZES = (64, 256, 1024)
BURN, L_CAL, CHAINS = 512, 4096, 128
BETAS = (0.5, 0.6, 0.75, 1.0)
SEED_CAL, SEED_CHK = 20261119, 20261120


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------
# G. gain spread
# ----------------------------------------------------------------------------

GL = np.polynomial.legendre.leggauss(64)


def spread_moments(x, j):
    """Mean and variance over g ~ U(1-j, 1+j) of p(g) = sigmoid(2 x g)."""
    g = 1.0 + j * GL[0]
    p = sig(2 * x * g)
    m = float(np.sum(GL[1] * p) / 2)
    v = float(np.sum(GL[1] * (p - m) ** 2) / 2)
    return m, v


def pooled_power(n, sites, x, j):
    m, v = spread_moments(x, j)
    df = sites - 1
    lam = n * sites * v / (m * (1 - m))
    crit = stats.chi2.isf(ALPHA, df)
    return float(stats.ncx2.sf(crit, df, lam))


def pooled_reads(sites, x, j):
    lo, hi = 1, 2
    while pooled_power(hi, sites, x, j) < 1 - MISS:
        hi *= 2
        if hi > 10**7:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if pooled_power(mid, sites, x, j) < 1 - MISS else (lo, mid)
    return hi


def pooled_sim(n, sites, x, j, reps, rng):
    df = sites - 1
    crit = stats.chi2.isf(ALPHA, df)
    out = {}
    for label, jj in (("ideal", 0.0), ("spread", j)):
        rej = 0
        for _ in range(reps):
            g = rng.uniform(1 - jj, 1 + jj, sites)
            k = rng.binomial(n, sig(2 * x * g))
            ph = k.sum() / (n * sites)
            x2 = float(np.sum((k - n * ph) ** 2) / (n * ph * (1 - ph)))
            rej += x2 > crit
        out[label] = rej / reps
    return out


def two_sided_reads(x, delta, alpha, nmax=40_000):
    """Smallest n (searched exhaustively to nmax) with two-sided exact binomial at alpha/2
    per tail and power >= 0.95 at both gains 1 + delta and 1 - delta, and all larger n up
    to n + 50 also passing (sawtooth guard)."""
    p0, pp, pm = sig(2 * x), sig(2 * x * (1 + delta)), sig(2 * x * (1 - delta))
    z = stats.norm.isf(alpha / 2)
    sd = np.sqrt(p0 * (1 - p0))
    guess = int(((z + ZB) * sd / min(pp - p0, p0 - pm)) ** 2)

    def passes(n):
        a = int(stats.binom.ppf(alpha / 2, n, p0))
        if stats.binom.cdf(a, n, p0) > alpha / 2:
            a -= 1
        b = int(stats.binom.isf(alpha / 2, n, p0)) + 1
        while b > 0 and stats.binom.sf(b - 2, n, p0) <= alpha / 2:
            b -= 1
        while stats.binom.sf(b - 1, n, p0) > alpha / 2:
            b += 1
        low = (lambda p: stats.binom.cdf(a, n, p)) if a >= 0 else (lambda p: 0.0)
        pw = [stats.binom.sf(b - 1, n, p) + low(p) for p in (pp, pm)]
        return min(pw) >= 1 - MISS

    lo, hi = max(1, int(guess * 0.5)), max(2, int(guess * 2))
    while not passes(hi):
        hi *= 2
        if hi > nmax:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if not passes(mid) else (lo, mid)
    n = hi
    while not all(passes(m) for m in range(n, n + 51)):  # sawtooth guard
        n += 1
    return n


def part_g():
    rng = np.random.default_rng(SEED_CAL + 1)
    xs = np.round(np.arange(0.4, 2.01, 0.05), 2)
    rows = []
    for j in (0.1, 0.3):
        for s in SIZES:
            best = None
            for x in xs:
                n = pooled_reads(s, float(x), j)
                if n and (best is None or n < best["reads"]):
                    best = {"beta_h": float(x), "reads": n}
            best["at_beta_h_1"] = pooled_reads(s, 1.0, j)
            best["sim_at_reads"] = pooled_sim(best["reads"], s, best["beta_h"], j, 2000, rng)
            rows.append({"spread_j": j, "sites": s, **best})
            print("G", j, s, best, flush=True)
    smallest = {}
    for s in SIZES:
        lo, hi = 1e-3, 0.3
        for _ in range(30):
            mid = (lo + hi) / 2
            n = min(pooled_reads(s, x, mid) or 10**9 for x in (1.0, 1.2, 1.4))
            lo, hi = (lo, mid) if n <= LIMIT else (mid, hi)
        smallest[f"sites_{s}"] = hi
    sec = []
    for delta in (0.1, 0.3):
        for scope, alpha in [("known_site", ALPHA)] + [(f"scan_{s}", ALPHA / s) for s in SIZES]:
            best = None
            grid = np.round(np.arange(0.8, 1.41, 0.05), 2)
            for x in grid:
                n = two_sided_reads(float(x), delta, alpha)
                if n and (best is None or n < best["reads"]):
                    best = {"beta_h": float(x), "reads": n}
            sec.append({"delta_abs": delta, "scope": scope, **best})
            print("G2", delta, scope, best, flush=True)
    return {
        "evidence_class": "exact_reference (per-site laws); noncentral chi-square approximation "
        "(pooled counts); software_simulation (NumPy check)",
        "pooled_rows": rows,
        "pooled_smallest_spread_within_limit": smallest,
        "single_outlier_two_sided": sec,
    }


# ----------------------------------------------------------------------------
# A. attribution margin arithmetic
# ----------------------------------------------------------------------------


def fwer(e, kappa, rho):
    z = stats.norm.isf(ALPHA / e)
    return float(min(1.0, e * stats.norm.sf((z - kappa) / np.sqrt(1 + rho**2))))


def part_a(edge_counts):
    rows = []
    for e in edge_counts:
        z = float(stats.norm.isf(ALPHA / e))
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fwer(e, mid, 0.0) <= 0.06 else (lo, mid)
        k06 = lo
        rows.append(
            {
                "edges": e,
                "z_alpha": z,
                "kappa_for_fwer_0p06_rho0": k06,
                "kappa_as_fraction_of_shift_at_own_nstar": k06 / (z + ZB),
                "fwer_at_5pct_shift": fwer(e, 0.05 * (z + ZB), 0.0),
                "fwer_at_1pct_shift": fwer(e, 0.01 * (z + ZB), 0.0),
                "fwer_rho_0p1": fwer(e, 0.0, 0.1),
                "fwer_rho_0p2": fwer(e, 0.0, 0.2),
                "fwer_kappa_0p02_rho_0p1": fwer(e, 0.02, 0.1),
            }
        )
    return {"evidence_class": "exact arithmetic (union bound, normal scan)", "rows": rows}


# ----------------------------------------------------------------------------
# B. temperature on the scan statistic
# ----------------------------------------------------------------------------


def gibbs_edge_means(target, beta, chains, burn, window, seed):
    n = target["n"]
    edges = p16.edge_tuples(target)
    j0 = np.asarray(target["couplings"], dtype=float)
    nbr, wts = p16.neighbour_arrays(n, edges, j0)
    wts = wts.astype(float)
    colour = (p16.target_positions(target).sum(axis=1) % 2) == 0
    ei = np.array([e[0] for e in edges])
    ej = np.array([e[1] for e in edges])
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(chains, n))
    acc = np.zeros((chains, len(edges)))
    idxs = (np.flatnonzero(colour), np.flatnonzero(~colour))
    for it in range(burn + window):
        for idx in idxs:
            field = np.einsum("ckw,kw->ck", s[:, nbr[idx]], wts[idx])
            p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * field))
            s[:, idx] = np.where(rng.random(field.shape) < p_up, 1.0, -1.0)
        if it >= burn:
            acc += s[:, ei] * s[:, ej]
    return acc / window


def scan_stats(means, c0, window):
    var = 1 - c0**2
    r = means.var(axis=0, ddof=1) * window / var
    se_rep = np.sqrt(var / window)
    zb = (means - c0) / se_rep  # standardized chain-level biases, independent-read units
    zbar_chain = zb.mean(axis=1)
    return r, {
        "r_median": float(np.median(r)),
        "r_max": float(r.max()),
        "edge_bias_mean_z": float(zb.mean() / (zbar_chain.std(ddof=1) / np.sqrt(zb.shape[0]))),
        "sd_of_chain_mean_standardized_bias": float(zbar_chain.std(ddof=1)),
    }


def part_b():
    rows = []
    for gi, target in enumerate(rv.greedy_long()):
        n = target["n"]
        if n > 256:
            continue
        j0 = np.asarray(target["couplings"], dtype=float)
        pos, edges = p16.target_positions(target), p16.edge_tuples(target)
        e_cnt = len(j0)
        za = float(stats.norm.isf(ALPHA / e_cnt))
        for beta in BETAS:
            t0 = time.time()
            k0 = beta * j0
            _, c0 = rv.kac_ward(target, k0)
            c1 = np.empty(e_cnt)
            for e in range(e_cnt):
                plus, minus = k0.copy(), k0.copy()
                plus[e], minus[e] = 1e-4, -1e-4
                c1[e] = (
                    p16.kac_ward_log_z(pos, edges, plus) - p16.kac_ward_log_z(pos, edges, minus)
                ) / 2e-4
            sign = np.sign(j0)
            shift = sign * (c0 - c1)
            n_ind = ((za * np.sqrt(1 - c0**2) + ZB * np.sqrt(1 - c1**2)) / shift) ** 2
            sets = {}
            for label, seed in (("calibration", SEED_CAL), ("check", SEED_CHK)):
                m = gibbs_edge_means(
                    target, beta, CHAINS, BURN, L_CAL, seed + 1000 * gi + int(beta * 100)
                )
                r, summ = scan_stats(m, c0, L_CAL)
                nstar = np.ceil(n_ind * r)
                summ["scan_reads_nstar"] = int(nstar.max())
                summ["scan_sweeps"] = int(BURN + nstar.max())
                sets[label] = (r, summ)
            r_cal, r_chk = sets["calibration"][0], sets["check"][0]
            # the scan read count fixed from calibration, applied to the check set's ratios
            n_scan = int(np.ceil(n_ind * r_cal).max())
            rows.append(
                {
                    "target": target["id"],
                    "n": n,
                    "edges": e_cnt,
                    "beta": beta,
                    "scan_independent_reads": int(np.ceil(n_ind.max())),
                    "edge_independent_reads_median": float(np.median(n_ind)),
                    "calibration": sets["calibration"][1],
                    "check": sets["check"][1],
                    "n_scan_from_calibration": n_scan,
                    "check_ratio_max_over_cal": float(np.max(r_chk / r_cal)),
                    "check_ratio_rms_log": float(np.sqrt(np.mean(np.log(r_chk / r_cal) ** 2))),
                    "check_nstar_over_cal_nstar": float(np.ceil(n_ind * r_chk).max() / n_scan),
                    "seconds": time.time() - t0,
                }
            )
            print(
                f"B {target['id']} b={beta} ind={int(np.ceil(n_ind.max()))} "
                f"n*cal={n_scan} n*chk={sets['check'][1]['scan_reads_nstar']} "
                f"rmax={sets['calibration'][1]['r_max']:.2f} {time.time() - t0:.0f}s",
                flush=True,
            )
    best = {}
    for r in rows:
        best[r["target"]] = min(best.get(r["target"], np.inf), r["calibration"]["scan_sweeps"])
    sel = []
    for beta in BETAS:
        cand = [r for r in rows if r["beta"] == beta]
        sel.append(
            {
                "beta": beta,
                "graphs": len(cand),
                "max_regret": max(
                    r["calibration"]["scan_sweeps"] / best[r["target"]] for r in cand
                ),
                "max_abs_edge_bias_mean_z": max(
                    abs(r["calibration"]["edge_bias_mean_z"]) for r in cand
                ),
                "nstar_exceeds_window": any(r["n_scan_from_calibration"] > L_CAL for r in cand),
            }
        )
    return {
        "evidence_class": "exact_reference (c0, c1, independent reads); software_simulation "
        "(scatter ratios, biases)",
        "window_reads": L_CAL,
        "burn_in_sweeps": BURN,
        "chains_per_set": CHAINS,
        "rows": rows,
        "selection_64_256_only": sel,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="g,a,b")
    args = ap.parse_args()
    out = {"probe": "P-0015 round 2", "evidence": "exploration, not evidence"}
    t0 = time.process_time()
    parts = args.parts.split(",")
    if "g" in parts:
        out["gain_spread"] = part_g()
    if "a" in parts:
        counts = sorted({len(t["couplings"]) for t in rv.greedy_long()})
        out["attribution"] = part_a(counts)
    if "b" in parts:
        out["scan_temperature"] = part_b()
    out["cpu_seconds"] = time.process_time() - t0
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
