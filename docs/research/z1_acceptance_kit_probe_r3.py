"""P-0015 round-3 probe: Student-t scan calibration, attribution limit, draw-averaged gain test.

Exploration, not evidence. CPU only:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run python docs/research/z1_acceptance_kit_probe_r3.py --out <json>

Answers the round-2 review on PR #174:

T. Student-t(127) quantile at 0.05 / E against the normal quantile: thresholds,
   scan-length factor, and the coherent bias admitted by the attribution rule
   at limits 0.06 and 0.065 (union bound with a t(127) tail).
R. Reproduces the round-2 beta 0.5 calibration chains on the six 64- and
   256-spin graphs (same seeds, 4,096-read window) and recomputes n_scan with
   the t quantile.
F. Family-wise false alarm of the whole scan under an ideal sampler, on the six
   graphs at beta 0.5, protocol procedure: 128 calibration chains, 512 burn-in,
   8,192-read window, fresh tested chains at n_scan, t and normal thresholds.
G. Pooled gain-spread test with the count averaged over gain draws (a fresh
   uniform draw per replicate) and the threshold taken from the null Monte
   Carlo at each candidate count; spread of power over individual draws.
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
import z1_acceptance_kit_probe_r2 as r2  # noqa: E402
import z1_acceptance_kit_review_probe as rv  # noqa: E402

from thermo_lab import planar_16_offset_ferro as p16  # noqa: E402

ALPHA, MISS, DF = 0.05, 0.05, 127
ZB = float(stats.norm.isf(MISS))
BETA = 0.5
BURN, CHAINS = 512, 128
SEED_F_CAL, SEED_F_TEST, SEED_G = 20261123, 20261124, 20261125


def q_norm(e):
    return float(stats.norm.isf(ALPHA / e))


def q_t(e):
    return float(stats.t.isf(ALPHA / e, DF))


def fwer_t(e, kappa, rho):
    return float(min(1.0, e * stats.t.sf((q_t(e) - kappa) / np.sqrt(1 + rho**2), DF)))


def part_t(edge_counts):
    rows = []
    for e in edge_counts:
        zn, zt = q_norm(e), q_t(e)
        row = {
            "edges": e,
            "z_normal": zn,
            "t127": zt,
            "scan_length_factor_equal_variance": ((zt + ZB) / (zn + ZB)) ** 2,
        }
        for lim in (0.06, 0.065):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if fwer_t(e, mid, 0.0) <= lim else (lo, mid)
            key = str(lim).replace(".", "p")
            row[f"kappa_admitted_{key}_rho0"] = lo
            row[f"kappa_fraction_of_shift_{key}"] = lo / (zt + ZB)
        row["fwer_rho_0p1"] = fwer_t(e, 0.0, 0.1)
        row["fwer_rho_0p2"] = fwer_t(e, 0.0, 0.2)
        rows.append(row)
    return {"evidence_class": "exact arithmetic (union bound, t(127) tail)", "rows": rows}


def exact_tables(target, beta):
    j0 = np.asarray(target["couplings"], dtype=float)
    pos, edges = p16.target_positions(target), p16.edge_tuples(target)
    k0 = beta * j0
    _, c0 = rv.kac_ward(target, k0)
    c1 = np.empty(len(j0))
    for e in range(len(j0)):
        plus, minus = k0.copy(), k0.copy()
        plus[e], minus[e] = 1e-4, -1e-4
        c1[e] = (
            p16.kac_ward_log_z(pos, edges, plus) - p16.kac_ward_log_z(pos, edges, minus)
        ) / 2e-4
    sign = np.sign(j0)
    return c0, c1, sign


def n_ind(c0, c1, sign, q):
    shift = sign * (c0 - c1)
    return ((q * np.sqrt(1 - c0**2) + ZB * np.sqrt(1 - c1**2)) / shift) ** 2


def part_r(archived_rows):
    rows = []
    for gi, target in enumerate(rv.greedy_long()):
        if target["n"] > 256:
            continue
        e = len(target["couplings"])
        c0, c1, sign = exact_tables(target, BETA)
        m = r2.gibbs_edge_means(target, BETA, CHAINS, BURN, 4096, r2.SEED_CAL + 1000 * gi + 50)
        r, _ = r2.scan_stats(m, c0, 4096)
        n_norm = int(np.ceil(n_ind(c0, c1, sign, q_norm(e)) * r).max())
        n_t = int(np.ceil(n_ind(c0, c1, sign, q_t(e)) * r).max())
        arch = archived_rows[target["id"]]
        rows.append(
            {
                "target": target["id"],
                "n": target["n"],
                "edges": e,
                "n_scan_normal": n_norm,
                "n_scan_normal_archived_r2": arch,
                "reproduced": n_norm == arch,
                "n_scan_t127": n_t,
                "factor": n_t / n_norm,
                "independent_reads_t127": int(np.ceil(n_ind(c0, c1, sign, q_t(e)).max())),
            }
        )
        print("R", rows[-1], flush=True)
    return {"window_reads": 4096, "beta": BETA, "rows": rows}


def part_f(reps_by_size):
    rows = []
    for gi, target in enumerate(rv.greedy_long()):
        n = target["n"]
        if n > 256:
            continue
        t0 = time.time()
        e = len(target["couplings"])
        c0, c1, sign = exact_tables(target, BETA)
        var = 1 - c0**2
        cal = r2.gibbs_edge_means(target, BETA, CHAINS, BURN, 8192, SEED_F_CAL + gi)
        r, _ = r2.scan_stats(cal, c0, 8192)
        out = {"target": target["id"], "n": n, "edges": e}
        n_scan = {}
        for label, q in (("normal", q_norm(e)), ("t127", q_t(e))):
            n_scan[label] = int(np.ceil(n_ind(c0, c1, sign, q) * r).max())
        reps = reps_by_size[n]
        # one tested chain set at the longer (t) length; the normal scan uses its prefix
        nt, nn = n_scan["t127"], n_scan["normal"]
        edges = p16.edge_tuples(target)
        ei = np.array([a for a, _ in edges])
        ej = np.array([b for _, b in edges])
        j0 = np.asarray(target["couplings"], dtype=float)
        nbr, wts = p16.neighbour_arrays(n, edges, j0)
        wts = wts.astype(float)
        colour = (p16.target_positions(target).sum(axis=1) % 2) == 0
        idxs = (np.flatnonzero(colour), np.flatnonzero(~colour))
        rng = np.random.default_rng(SEED_F_TEST + gi)
        s = rng.choice([-1.0, 1.0], size=(reps, n))
        acc = np.zeros((reps, e))
        acc_n = None
        for it in range(BURN + nt):
            for idx in idxs:
                field = np.einsum("ckw,kw->ck", s[:, nbr[idx]], wts[idx])
                p_up = 1.0 / (1.0 + np.exp(-2.0 * BETA * field))
                s[:, idx] = np.where(rng.random(field.shape) < p_up, 1.0, -1.0)
            if it >= BURN:
                acc += s[:, ei] * s[:, ej]
                if it - BURN + 1 == nn:
                    acc_n = acc.copy()
        for label, q, length, a in (
            ("normal", q_norm(e), nn, acc_n),
            ("t127", q_t(e), nt, acc),
        ):
            se = np.sqrt(r * var / length)
            # towards "missing": sign * (mean - c0) negative, i.e. |c| falls
            zstat = -sign * (a / length - c0) / se
            rej = (zstat > q).any(axis=1)
            out[label] = {
                "n_scan": length,
                "family_wise_false_alarm": float(rej.mean()),
                "binomial_se": float(np.sqrt(rej.mean() * (1 - rej.mean()) / reps)),
            }
        out["replicates"] = reps
        out["seconds"] = time.time() - t0
        rows.append(out)
        print("F", out, flush=True)
    return {
        "evidence_class": "software_simulation (NumPy float64) against exact Kac-Ward moments",
        "beta": BETA,
        "calibration_chains": CHAINS,
        "calibration_window": 8192,
        "rows": rows,
    }


def x2_stat(k, n):
    ph = k.sum(axis=1) / (n * k.shape[1])
    ph = np.clip(ph, 1e-12, 1 - 1e-12)
    return np.sum((k - n * ph[:, None]) ** 2, axis=1) / (n * ph * (1 - ph))


def mc_power(n, sites, x, j, reps, rng, chunk=2000):
    """Null threshold (empirical, false alarm <= 0.05) and draw-averaged power at n."""
    p0 = r2.sig(2 * x)
    null = np.concatenate(
        [x2_stat(rng.binomial(n, p0, size=(chunk, sites)), n) for _ in range(reps // chunk)]
    )
    null.sort()
    crit = float(null[int(np.ceil((1 - ALPHA) * reps)) - 1])
    alt = []
    for _ in range(reps // chunk):
        g = rng.uniform(1 - j, 1 + j, size=(chunk, sites))
        alt.append(x2_stat(rng.binomial(n, r2.sig(2 * x * g)), n))
    alt = np.concatenate(alt)
    return crit, float((null > crit).mean()), float((alt > crit).mean())


def part_g(pooled_rows, reps):
    rng = np.random.default_rng(SEED_G)
    rows = []
    for row in pooled_rows:
        sites, j, x, approx = row["sites"], row["spread_j"], row["beta_h"], row["reads"]
        cache = {}

        def ok(n, cache=cache, sites=sites, x=x, j=j):
            if n not in cache:
                cache[n] = mc_power(n, sites, x, j, reps, rng)
            return cache[n][2] >= 1 - MISS

        lo, hi = max(1, int(approx * 0.7)), int(approx * 1.5) + 2
        while ok(lo):
            lo = max(1, lo // 2)
            if lo == 1:
                break
        while not ok(hi):
            hi *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        crit, fa, pw = cache[hi]
        chi2_crit = float(stats.chi2.isf(ALPHA, sites - 1))
        # power spread over individual draws at the chosen count and Monte Carlo threshold
        per_draw = []
        p0 = r2.sig(2 * x)
        for _ in range(200):
            g = rng.uniform(1 - j, 1 + j, sites)
            k = rng.binomial(hi, r2.sig(2 * x * g), size=(1000, sites))
            per_draw.append(float((x2_stat(k, hi) > crit).mean()))
        declared = np.random.default_rng({64: 4520, 256: 4521, 1024: 4522}[sites]).uniform(
            1 - j, 1 + j, sites
        )
        kd = rng.binomial(hi, r2.sig(2 * x * declared), size=(reps, sites))
        kn = rng.binomial(hi, p0, size=(reps, sites))
        out = {
            "spread_j": j,
            "sites": sites,
            "beta_h": x,
            "approx_reads_r2": approx,
            "mc_reads_draw_averaged": hi,
            "mc_threshold": crit,
            "chi2_threshold": chi2_crit,
            "mc_false_alarm": fa,
            "mc_power_draw_averaged": pw,
            "chi2_threshold_false_alarm_at_mc_reads": float((x2_stat(kn, hi) > chi2_crit).mean()),
            "per_draw_power_min": float(np.min(per_draw)),
            "per_draw_power_p05": float(np.quantile(per_draw, 0.05)),
            "per_draw_power_median": float(np.median(per_draw)),
            "declared_draw_power_at_mc_reads": float((x2_stat(kd, hi) > crit).mean()),
        }
        rows.append(out)
        print("G", out, flush=True)
    return {
        "evidence_class": "Monte Carlo on exact per-site binomial laws (software_simulation)",
        "replicates": reps,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="t,r,f,g")
    ap.add_argument("--reps64", type=int, default=4000)
    ap.add_argument("--reps256", type=int, default=2000)
    ap.add_argument("--g-reps", type=int, default=20000)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    arch = json.loads((root / "2026-11-17-z1-acceptance-kit-probe-r2.json").read_text())
    out: dict[str, object] = {"probe": "P-0015 round 3", "evidence": "exploration, not evidence"}
    t0 = time.process_time()
    parts = args.parts.split(",")
    if "t" in parts:
        counts = sorted({len(t["couplings"]) for t in rv.greedy_long()})
        out["t_calibration"] = part_t(counts)
    if "r" in parts:
        rows = {
            r["target"]: r["n_scan_from_calibration"]
            for r in arch["scan_temperature"]["rows"]
            if r["beta"] == BETA
        }
        out["scan_length_t"] = part_r(rows)
    if "f" in parts:
        out["scan_false_alarm"] = part_f({64: args.reps64, 256: args.reps256})
    if "g" in parts:
        out["pooled_gain_mc"] = part_g(arch["gain_spread"]["pooled_rows"], args.g_reps)
    out["cpu_seconds"] = time.process_time() - t0
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
