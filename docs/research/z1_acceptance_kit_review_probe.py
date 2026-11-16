"""Review probe for the z1-acceptance-kit row (exploration only, CPU, about 40 minutes).

Written while reviewing the draft protocol in PR #174 (P-0015). It asks two things the draft's
own probe left open, on the nine greedy-long P-0001 graphs:

  slow   At beta 4 from the ordered start, are consecutive one-sweep reads close to independent
         on every graph? The draft measured one graph per size with a short-window estimator.
         Here the test is direct: the variance, across independent chains, of the mean of a
         window of reads, times the window length, over the exact per-read variance. Independent
         reads give 1. The short-window estimate is reported next to it on the same samples.
  bits7  Reads to detect 7-bit rounding at 1,024 spins and beta 4 (the draft computed 4, 6, 8).
  beta   The same read counts at warmer temperatures, with ordinary two-colour Gibbs started
         from a random state: its bias against the exact energy and the same scatter ratio.

Read counts are for the log-likelihood-ratio statistic at 5 percent false alarm and 5 percent
miss, from exact Kac-Ward means and exact per-read variances (central second differences of
log Z), with the normal approximation the draft uses. They count independent reads.

Evidence: exact_reference for means and variances; software_simulation (NumPy float64) for
everything sampled. One seed per cell unless stated. Nothing here is a hardware claim, a
recorded study or a recommendation of an operating point.

Usage: uv run python docs/research/z1_acceptance_kit_review_probe.py --out <file.json>
       (--parts slow,bits7,beta selects parts; an existing output file is updated in place)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from thermo_lab import planar_16_offset_ferro as p16
from thermo_lab.constraints.quantize import quantize_array

Z = float(stats.norm.isf(0.05))
BETA0 = 4.0
OUT = Path(__file__).with_name("2026-11-16-z1-acceptance-kit-review-probe.json")


def log_z(target, k):
    return p16.kac_ward_log_z(p16.target_positions(target), p16.edge_tuples(target), k)


def kac_ward(target, k):
    return p16.kac_ward(p16.target_positions(target), p16.edge_tuples(target), k)


def variance_along(target, k, direction, lz):
    """Exact Var(direction . S) at interactions k, S the edge products."""
    h = 1e-3 / float(np.max(np.abs(direction)))
    return (log_z(target, k + h * direction) - 2 * lz + log_z(target, k - h * direction)) / h**2


def rounded(j0, bits):
    return quantize_array(j0, float(np.max(np.abs(j0))), bits)


def llr_reads(target, j0, j1, beta):
    """Independent reads for the LLR test between couplings j0 and j1, and its ideal variance."""
    k0, k1 = beta * j0, beta * j1
    dk = k1 - k0
    lz0, c0 = kac_ward(target, k0)
    lz1, c1 = kac_ward(target, k1)
    v0 = variance_along(target, k0, dk, lz0)
    v1 = variance_along(target, k1, dk, lz1)
    shift = float(dk @ (c1 - c0))
    reads = int(
        np.ceil(((Z * np.sqrt(max(v0, 0.0)) + Z * np.sqrt(max(v1, 0.0))) / abs(shift)) ** 2)
    )
    return reads, float(v0)


def gibbs(target, beta, start, chains, sweeps, seed, directions):
    """Two-colour heat bath on the ideal couplings; one read per sweep of each direction . S."""
    n = target["n"]
    edges = p16.edge_tuples(target)
    j0 = np.asarray(target["couplings"], dtype=float)
    nbr, wts = p16.neighbour_arrays(n, edges, j0)
    wts = wts.astype(float)
    pos = p16.target_positions(target)
    colour = (pos.sum(axis=1) % 2) == 0
    ei = np.array([e[0] for e in edges])
    ej = np.array([e[1] for e in edges])
    rng = np.random.default_rng(seed)
    if start == "random":
        s = rng.choice([-1.0, 1.0], size=(chains, n))
    else:  # a ground state, with a fair coin per chain for which of the two
        s = rng.choice([-1.0, 1.0], size=(chains, 1)) * np.where(colour, 1.0, -1.0)[None, :]
    out = np.empty((len(directions), sweeps, chains))
    for it in range(sweeps):
        for idx in (np.flatnonzero(colour), np.flatnonzero(~colour)):
            field = np.einsum("ckw,kw->ck", s[:, nbr[idx]], wts[idx])
            p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * field))
            s[:, idx] = np.where(rng.random(field.shape) < p_up, 1.0, -1.0)
        products = s[:, ei] * s[:, ej]
        for d, direction in enumerate(directions):
            out[d, it] = products @ direction
    return out


def short_window_tau(x):
    """Integrated autocorrelation with a self-consistent window of 5 tau (the draft's estimator)."""
    x = x - x.mean(axis=0, keepdims=True)
    var = float(np.mean(x * x))
    if var == 0:
        return 1.0
    tau = 1.0
    for lag in range(1, x.shape[0] // 4):
        tau += 2 * float(np.mean(x[:-lag] * x[lag:])) / var
        if lag >= 5 * tau:
            break
    return tau


def window_row(x, a, b, exact_mean, exact_var):
    """Bias and scatter of the chain means of reads a..b against the exact per-read moments."""
    means = x[a:b].mean(axis=0)
    chains = means.size
    se = float(means.std(ddof=1) / np.sqrt(chains))
    return {
        "sweeps": [a, b],
        "bias": float(means.mean() - exact_mean),
        "bias_se": se,
        "scatter_ratio": float(means.var(ddof=1) * (b - a) / exact_var),
        "scatter_ratio_rel_se": float(np.sqrt(2.0 / (chains - 1))),  # if chain means were normal
        "within_chain_var_over_exact": float(x[a:b].var(axis=0, ddof=1).mean() / exact_var),
        "short_window_tau": short_window_tau(x[a:b]),
    }


def greedy_long(n=None):
    return [
        t for t in p16.study_targets() if t["graph"] == "greedy-long" and (n is None or t["n"] == n)
    ]


def part_slow():
    plan = {64: (1024, 16384, (4242,)), 256: (512, 8256, (4242, 777)), 1024: (256, 4256, (4242,))}
    edges = (0, 256, 1256, 2256, 4256, 8256, 16384)
    rows = []
    for target in greedy_long():
        n = target["n"]
        j0 = np.asarray(target["couplings"], dtype=float)
        k0 = BETA0 * j0
        lz0, c0 = kac_ward(target, k0)
        mean = float(j0 @ c0 / n)
        var = float(variance_along(target, k0, k0, lz0) / (BETA0 * n) ** 2)
        chains, sweeps, seeds = plan[n]
        for seed in seeds:
            t0 = time.time()
            u = gibbs(target, BETA0, "ordered", chains, sweeps, seed, [j0 / n])[0]
            windows = [
                window_row(u, a, b, mean, var)
                for a, b in zip(edges[:-1], edges[1:], strict=True)
                if b <= sweeps
            ]
            rows.append(
                {
                    "target": target["id"],
                    "n": n,
                    "seed": seed,
                    "chains": chains,
                    "sweeps": sweeps,
                    "exact_energy_per_spin": mean,
                    "exact_sd_per_read": float(np.sqrt(var)),
                    "windows": windows,
                    "seconds": time.time() - t0,
                }
            )
            ratios = [round(w["scatter_ratio"], 2) for w in windows[1:]]
            print(f"slow {target['id']} seed {seed}: scatter ratios {ratios}", flush=True)
    return {
        "beta": BETA0,
        "start": "ordered, fair coin per chain",
        "statistic": "nominal energy per spin, one read per sweep",
        "rows": rows,
    }


def part_bits7():
    rows = []
    for target in greedy_long(1024):
        j0 = np.asarray(target["couplings"], dtype=float)
        row = {"target": target["id"], "n": 1024}
        for bits in (6, 7):
            row[f"llr_reads_{bits}bit"] = llr_reads(target, j0, rounded(j0, bits), BETA0)[0]
        rows.append(row)
        print("bits7", row, flush=True)
    return {"beta": BETA0, "rows": rows}


def part_beta():
    plan = {64: (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0), 256: (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)}
    plan[1024] = (0.5, 0.75)
    chains, sweeps = 128, 4096
    rows = []
    for target in greedy_long():
        n = target["n"]
        j0 = np.asarray(target["couplings"], dtype=float)
        for beta in plan[n]:
            t0 = time.time()
            k0 = beta * j0
            lz0, c0 = kac_ward(target, k0)
            mean_u = float(j0 @ c0 / n)
            var_u = float(variance_along(target, k0, k0, lz0) / (beta * n) ** 2)
            reads = {}
            for bits in (6, 8):
                reads[bits], var_llr = llr_reads(target, j0, rounded(j0, bits), beta)
            dk8 = beta * (rounded(j0, 8) - j0)
            mean_llr = float(dk8 @ c0)
            x = gibbs(target, beta, "random", chains, sweeps, 5, [j0 / n, dk8])
            half = sweeps // 2
            energy = window_row(x[0], half, sweeps, mean_u, var_u)
            llr = window_row(x[1], half, sweeps, mean_llr, var_llr)
            rows.append(
                {
                    "target": target["id"],
                    "n": n,
                    "beta": beta,
                    "llr_reads_6bit": reads[6],
                    "llr_reads_8bit": reads[8],
                    "median_abs_edge_correlation": float(np.median(np.abs(c0))),
                    "random_start": {"chains": chains, "sweeps": sweeps, "window": [half, sweeps]},
                    "energy": energy,
                    "llr_8bit": llr,
                    "seconds": time.time() - t0,
                }
            )
            print(
                f"beta {target['id']} {beta}: 6-bit {reads[6]} 8-bit {reads[8]} "
                f"bias z {energy['bias'] / energy['bias_se']:+.1f} "
                f"scatter energy {energy['scatter_ratio']:.1f} llr {llr['scatter_ratio']:.1f}",
                flush=True,
            )
    return {"start": "random", "statistic_for_scatter": "energy per spin; 8-bit LLR", "rows": rows}


PARTS = {"slow": part_slow, "bits7": part_bits7, "beta": part_beta}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--parts", default="slow,bits7,beta")
    args = parser.parse_args()
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    result.update(
        label="exploration",
        row="z1-acceptance-kit",
        reviews="PR #174 (P-0015)",
        false_alarm=0.05,
        miss=0.05,
        hardware_claim=False,
        evidence_class="exact_reference (means, variances); software_simulation (sampled)",
    )
    for name in args.parts.split(","):
        result[name] = PARTS[name]()
        args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
