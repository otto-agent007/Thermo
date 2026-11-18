"""P-0015 round-1 probe: simple tests, temperature choice and pass-rule arithmetic.

Exploration, not evidence. CPU only:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run python docs/research/z1_acceptance_kit_probe_r1.py --out <json>

Answers the 2026-11-16 amendments of queue row `z1-acceptance-kit`:

S. Isolated pairs and single sites (amendment of D-0012): closed-form exact
   read and programming-write counts for all four declared defects, with an
   enumerated check that pair and site reads are exactly independent under the
   two-colour sweep (`exact_reference`).
T. Temperature (amendment items 1, 2): on all nine greedy-long graphs, exact
   rounding-LLR read counts at 6 and 8 bits, and ordinary two-colour Gibbs from
   a random start with 512 burn-in sweeps. The read correlation is the scatter
   of window means across independent chains at the test length (capped at
   the 10,000-read limit), over the exact per-read variance. Declared rule:
   among betas evaluated on all nine graphs whose random-start bias passes
   (|z| <= 3.5 for energy and both LLRs on every graph), choose the beta that
   minimises the maximum over (graph, test) of
   (512 + reads * scatter) / (the graph's best value over its own grid).
U. At the chosen beta: a second-seed slow-state check (two consecutive test
   windows, chain-level outliers), rounding at every computed bit depth
   (item 4) and missing couplings (all edges at 64 spins, a stride at 256).
P. Pass-rule arithmetic with the number of checks in view (item 5).

L32 at beta 0.5 and 0.75 reuse the exact LLR read counts of the review probe
(`2026-11-16-z1-acceptance-kit-review-probe.json`); 0.75 is not resampled at
1,024 spins (the review measured scatter 3.5 to 9.8 there).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

sys.path.insert(0, str(Path(__file__).resolve().parent))
import z1_acceptance_kit_review_probe as rv  # noqa: E402

from thermo_lab import planar_16_offset_ferro as p16  # noqa: E402
from thermo_lab.constraints.quantize import quantize_array  # noqa: E402

ALPHA = MISS = 0.05
LIMIT = 10_000
BURN = 512
SEED = 20261117
REVIEW = Path(__file__).with_name("2026-11-16-z1-acceptance-kit-review-probe.json")
GRID = {64: (0.5, 0.6, 0.75, 1.0, 1.5, 2.0), 256: (0.5, 0.6, 0.75, 1.0), 1024: (0.5, 0.6)}
CANDIDATES = (0.5, 0.6)  # evaluated on all nine graphs
CHAINS = {64: 256, 256: 192, 1024: 128}
WEIGHTS = (0.2, 0.4, 0.6, 0.8, 1.0)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------
# exact binomial read counts (one-sided count test)
# ----------------------------------------------------------------------------


def _ok(trials, q0, q1, alpha, miss):
    t = stats.binom.isf(alpha, trials, q0) + 1
    while t > 0 and stats.binom.sf(t - 2, trials, q0) <= alpha:
        t -= 1
    return stats.binom.cdf(t - 1, trials, q1) <= miss


def reads_needed(q0, q1, per_read=1, alpha=ALPHA, miss=MISS, max_trials=10**10):
    """Smallest n reads (each giving per_read iid Bernoulli trials) for the count test."""
    if q0 == q1:
        return None
    if q1 < q0:
        q0, q1 = 1 - q0, 1 - q1
    hi = 1
    while not _ok(hi * per_read, q0, q1, alpha, miss):
        hi *= 2
        if hi * per_read > max_trials:
            return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _ok(mid * per_read, q0, q1, alpha, miss):
            hi = mid
        else:
            lo = mid
    return int(hi)


# ----------------------------------------------------------------------------
# S. isolated pairs and single sites
# ----------------------------------------------------------------------------


def enumerated_pair_check(k, gain_a, gain_b):
    """Two-colour sweep on one pair: site a (colour A) then site b (colour B).

    Returns the max deviation of the stationary joint law of consecutive edge
    products from the product of marginals, and P(e = sign K) against
    sigmoid(2 gain_b |K|).
    """
    states = list(itertools.product([-1, 1], repeat=2))
    t = np.zeros((4, 4))
    for i, (_sa, sb) in enumerate(states):
        for j, (na, nb) in enumerate(states):
            pa = sig(2 * gain_a * k * sb)
            pa = pa if na == 1 else 1 - pa
            pb = sig(2 * gain_b * k * na)
            pb = pb if nb == 1 else 1 - pb
            t[i, j] = pa * pb
    a = t.T - np.eye(4)
    a[-1] = 1
    pi = np.linalg.solve(a, np.array([0, 0, 0, 1.0]))
    e = np.array([sa * sb for sa, sb in states])
    joint = np.array(
        [
            [
                sum(pi[i] * t[i, j] for i in range(4) for j in range(4) if e[i] == x and e[j] == y)
                for y in (-1, 1)
            ]
            for x in (-1, 1)
        ]
    )
    marg = joint.sum(axis=1)
    agree = float(pi[e == np.sign(k)].sum())
    return {
        "k": k,
        "gain_a": gain_a,
        "gain_b": gain_b,
        "consecutive_independence_max": float(np.max(np.abs(joint - np.outer(marg, marg)))),
        "p_agree": agree,
        "p_agree_minus_sigmoid_gain_b": abs(agree - sig(2 * gain_b * abs(k))),
    }


def graph_facts(target):
    n = target["n"]
    edges = p16.edge_tuples(target)
    pos = p16.target_positions(target)
    colour = (pos.sum(axis=1) % 2) == 0
    deg = np.bincount(np.array(edges).ravel(), minlength=n)
    a_idx = {s: r for r, s in enumerate(np.flatnonzero(colour))}
    b_idx = {s: r for r, s in enumerate(np.flatnonzero(~colour))}
    rows, cols = [], []
    for i, j in edges:
        a, b = (i, j) if colour[i] else (j, i)
        rows.append(a_idx[a])
        cols.append(b_idx[b])
    bi = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(a_idx), len(b_idx)))
    match = maximum_bipartite_matching(bi, perm_type="column")
    return {
        "target": target["id"],
        "n": n,
        "edges": len(edges),
        "bipartite": bool(all(colour[i] != colour[j] for i, j in edges)),
        "max_degree": int(deg.max()),
        "matchings_to_cover_edges": int(deg.max()),  # Konig: bipartite edge chromatic number
        "max_matching_pairs": int(np.sum(match >= 0)),
    }


def part_s(beta_sel):
    out = {"evidence_class": "exact_reference (closed form; enumerated two-site check)"}
    out["enumerated_checks"] = [
        enumerated_pair_check(k, ga, gb)
        for k, ga, gb in ((-1.0, 1.0, 1.0), (-0.3, 0.9, 1.1), (-2.0, 1.3, 0.7), (0.5, 1.0, 1.3))
    ]
    facts = [graph_facts(t) for t in rv.greedy_long()]
    out["graphs"] = facts
    sizes = sorted({f["n"] for f in facts})
    pairs = {s: min(f["max_matching_pairs"] for f in facts if f["n"] == s) for s in sizes}
    nedges = {s: max(f["edges"] for f in facts if f["n"] == s) for s in sizes}
    deg = {s: max(f["max_degree"] for f in facts if f["n"] == s) for s in sizes}

    # missing coupling: the coupler reads 0 whatever is programmed (q1 = 1/2)
    miss_rows = []
    for label, beta, w in (
        ("cap_beta4", 4.0, 1.0),
        ("beta_j_1", 1.0, 1.0),
        ("cap_beta_sel", beta_sel, 1.0),
        ("weakest_archived_beta_sel", beta_sel, 0.2),
    ):
        q0 = float(sig(2 * beta * w))
        row = {"program": label, "beta": beta, "abs_j": w, "p_agree_ideal": q0}
        row["reads_known_coupler"] = reads_needed(q0, 0.5)
        for s in sizes:
            per = reads_needed(q0, 0.5, alpha=ALPHA / nedges[s])
            row[f"scan_n{s}"] = {
                "programming_writes": deg[s],
                "reads_per_matching": per,
                "reads_total": per * deg[s] if per else None,
            }
        miss_rows.append(row)
    out["missing_coupling"] = miss_rows

    # rounding: all pairs of a matching programmed at the worst-rounded weight
    betas = np.round(np.arange(0.5, 4.0001, 0.1), 2)
    rnd = []
    for bits in range(4, 13):
        err = {w: float(quantize_array(np.array([-w]), 1.0, bits)[0] + w) for w in WEIGHTS}
        row = {"bits": bits, "coupling_error": err}
        if max(abs(v) for v in err.values()) == 0:
            row["defect"] = "none (exact on this codebook)"
            rnd.append(row)
            continue
        for s in sizes:
            best = None
            for w in WEIGHTS:
                if err[w] == 0:
                    continue
                for b in betas:
                    q0, q1 = float(sig(2 * b * w)), float(sig(2 * b * (w - err[w])))
                    n_r = reads_needed(q0, q1, per_read=pairs[s])
                    if n_r and (best is None or n_r < best["reads"]):
                        best = {"weight": w, "beta": float(b), "reads": n_r}
            at_sel = None
            for w in WEIGHTS:
                if err[w] == 0:
                    continue
                q0, q1 = float(sig(2 * beta_sel * w)), float(sig(2 * beta_sel * (w - err[w])))
                n_r = reads_needed(q0, q1, per_read=pairs[s])
                if n_r and (at_sel is None or n_r < at_sel["reads"]):
                    at_sel = {"weight": w, "beta": beta_sel, "reads": n_r}
            row[f"n{s}"] = {
                "pairs_per_read": pairs[s],
                "programming_writes": 1,
                "best_beta_and_weight": best,
                "at_beta_sel": at_sel,
            }
        rnd.append(row)
    out["rounding"] = rnd

    # static beta variation: one isolated site with bias, gain 1 + delta
    xs = np.round(np.arange(0.05, 3.0001, 0.01), 2)  # x = beta h
    bv = []
    for delta in (0.1, -0.1, 0.3, -0.3):
        row = {"delta": delta}
        for label, xgrid in (("two_beta_h_1", [0.5]), ("best_x", xs)):
            best = None
            for x in xgrid:
                q0, q1 = float(sig(2 * x)), float(sig(2 * x * (1 + delta)))
                n_r = reads_needed(q0, q1)
                if best is None or n_r < best["reads_known_site"]:
                    best = {"beta_h": float(x), "reads_known_site": n_r}
            x = best["beta_h"]
            q0, q1 = float(sig(2 * x)), float(sig(2 * x * (1 + delta)))
            best["reads_scan_sites"] = {
                f"n{s}": reads_needed(q0, q1, alpha=ALPHA / s) for s in sizes
            }
            row[label] = best
        bv.append(row)
    # smallest |delta| detectable within LIMIT reads at the best x, known site and scan
    smallest = {}
    for scope, alpha in [("known_site", ALPHA)] + [(f"scan_n{s}", ALPHA / s) for s in sizes]:
        lo, hi = 1e-4, 0.3
        for _ in range(40):
            mid = (lo + hi) / 2
            ok = any(
                (r := reads_needed(float(sig(2 * x)), float(sig(2 * x * (1 + mid))), alpha=alpha))
                and r <= LIMIT
                for x in (1.0, 1.2, 1.4)
            )
            lo, hi = (lo, mid) if ok else (mid, hi)
        smallest[scope] = hi
    out["beta_variation"] = {"rows": bv, "smallest_positive_delta_within_limit": smallest}

    # stuck p-bit: isolated unbiased site, fresh fair coin every sweep
    out["stuck_pbit"] = {
        "reads_known_site_known_value": int(np.ceil(np.log2(1 / ALPHA))),
        "reads_known_site": int(np.ceil(1 + np.log2(1 / ALPHA))),
        **{f"reads_scan_n{s}": int(np.ceil(1 + np.log2(s / ALPHA))) for s in sizes},
        "miss_probability": 0.0,
        "programming_writes": 1,
        "state_writes": 0,
        "covers": ["ignores writes (reads a constant)", "accepts a write and never updates"],
    }
    return out


# ----------------------------------------------------------------------------
# T. temperature choice
# ----------------------------------------------------------------------------


def llr_exact(target, j0, beta, bits, lz0, c0, need_reads=True):
    k0 = beta * j0
    dk = beta * (rv.rounded(j0, bits) - j0)
    v0 = rv.variance_along(target, k0, dk, lz0)
    row = {"bits": bits, "mean0": float(dk @ c0), "var0": float(v0), "dk": dk}
    if need_reads:
        lz1, c1 = rv.kac_ward(target, beta * rv.rounded(j0, bits))
        v1 = rv.variance_along(target, beta * rv.rounded(j0, bits), dk, lz1)
        shift = float(dk @ (c1 - c0))
        z = float(stats.norm.isf(ALPHA))
        row["shift"] = shift
        row["var1"] = float(v1)
        row["reads"] = int(np.ceil(((z * np.sqrt(v0) + z * np.sqrt(max(v1, 0))) / abs(shift)) ** 2))
    return row


def scatter(x, a, w, mean, var):
    means = x[a : a + w].mean(axis=0)
    chains = means.size
    se = float(means.std(ddof=1) / np.sqrt(chains))
    zc = (means - mean) / np.sqrt(var / w)
    r = float(means.var(ddof=1) * w / var)
    return {
        "window": [a, a + w],
        "bias": float(means.mean() - mean),
        "bias_z": float((means.mean() - mean) / se),
        "scatter_ratio": r,
        "scatter_ratio_rel_se": float(np.sqrt(2.0 / (chains - 1))),
        "max_abs_chain_z_over_sqrt_r": float(np.max(np.abs(zc)) / np.sqrt(max(r, 1e-12))),
        "short_window_tau": rv.short_window_tau(x[a : a + w]),
    }


def review_reads():
    d = json.loads(REVIEW.read_text())
    return {(r["target"], r["beta"]): r for r in d["beta"]["rows"]}


def part_t():
    rev = review_reads()
    rows = []
    for gi, target in enumerate(rv.greedy_long()):
        n = target["n"]
        j0 = np.asarray(target["couplings"], dtype=float)
        for beta in GRID[n]:
            t0 = time.time()
            k0 = beta * j0
            lz0, c0 = rv.kac_ward(target, k0)
            mu = float(j0 @ c0 / n)
            var_u = float(rv.variance_along(target, k0, k0, lz0) / (beta * n) ** 2)
            reuse = n == 1024 and (target["id"], beta) in rev
            tests = {}
            for bits in (6, 8):
                tests[bits] = llr_exact(target, j0, beta, bits, lz0, c0, need_reads=not reuse)
                if reuse:
                    tests[bits]["reads"] = rev[(target["id"], beta)][f"llr_reads_{bits}bit"]
                    tests[bits]["reads_source"] = "review probe"
            w = {b: min(tests[b]["reads"], LIMIT) for b in tests}
            sweeps = BURN + max(w.values())
            x = rv.gibbs(
                target,
                beta,
                "random",
                CHAINS[n],
                sweeps,
                SEED + 100 * gi + int(beta * 100),
                [j0 / n, tests[6]["dk"], tests[8]["dk"]],
            )
            res = {
                "energy": scatter(x[0], BURN, w[8], mu, var_u),
                "llr6": scatter(x[1], BURN, w[6], tests[6]["mean0"], tests[6]["var0"]),
                "llr8": scatter(x[2], BURN, w[8], tests[8]["mean0"], tests[8]["var0"]),
            }
            for b in (6, 8):
                tests[b].pop("dk")
                r = res[f"llr{b}"]["scatter_ratio"]
                tests[b]["sweeps_per_test"] = BURN + tests[b]["reads"] * r
            rows.append(
                {
                    "target": target["id"],
                    "n": n,
                    "beta": beta,
                    "chains": CHAINS[n],
                    "sweeps": sweeps,
                    "exact_energy_per_spin": mu,
                    "llr": {str(b): tests[b] for b in tests},
                    "sampled": res,
                    "seconds": time.time() - t0,
                }
            )
            print(
                f"T {target['id']} b={beta} reads6={tests[6]['reads']} reads8={tests[8]['reads']} "
                f"r6={res['llr6']['scatter_ratio']:.2f} r8={res['llr8']['scatter_ratio']:.2f} "
                f"zE={res['energy']['bias_z']:+.1f} {time.time() - t0:.0f}s",
                flush=True,
            )
    # selection
    best = {}
    for r in rows:
        for b in ("6", "8"):
            key = (r["target"], b)
            best[key] = min(best.get(key, np.inf), r["llr"][b]["sweeps_per_test"])
    sel = []
    for beta in CANDIDATES:
        cand = [r for r in rows if r["beta"] == beta]
        bias_ok = all(
            abs(r["sampled"][s]["bias_z"]) <= 3.5 for r in cand for s in ("energy", "llr6", "llr8")
        )
        regret = max(
            r["llr"][b]["sweeps_per_test"] / best[(r["target"], b)]
            for r in cand
            for b in ("6", "8")
        )
        sel.append({"beta": beta, "graphs": len(cand), "bias_ok": bias_ok, "max_regret": regret})
    ok = [s for s in sel if s["bias_ok"] and s["graphs"] == 9]
    chosen = min(ok, key=lambda s: s["max_regret"])["beta"] if ok else None
    return {
        "evidence_class": "exact_reference (reads); software_simulation (scatter, bias)",
        "burn_in_sweeps": BURN,
        "rows": rows,
        "selection": sel,
        "chosen_beta": chosen,
    }


# ----------------------------------------------------------------------------
# U. at the chosen beta
# ----------------------------------------------------------------------------


def part_u(beta, t_rows):
    out = {"beta": beta, "slow": [], "rounding": [], "missing": []}
    bits_by_n = {64: (4, 6, 7, 8, 10, 11, 12), 256: (4, 6, 7, 8, 10, 11, 12), 1024: (7, 10)}
    for gi, target in enumerate(rv.greedy_long()):
        n = target["n"]
        j0 = np.asarray(target["couplings"], dtype=float)
        k0 = beta * j0
        t0 = time.time()
        lz0, c0 = rv.kac_ward(target, k0)
        mu = float(j0 @ c0 / n)
        var_u = float(rv.variance_along(target, k0, k0, lz0) / (beta * n) ** 2)
        trow = next(r for r in t_rows if r["target"] == target["id"] and r["beta"] == beta)
        t8 = llr_exact(target, j0, beta, 8, lz0, c0, need_reads=False)
        w8 = min(trow["llr"]["8"]["reads"], LIMIT)
        chains = 256 if n < 1024 else 128
        x = rv.gibbs(
            target, beta, "random", chains, BURN + 2 * w8, SEED + 7777 + gi, [j0 / n, t8["dk"]]
        )
        out["slow"].append(
            {
                "target": target["id"],
                "chains": chains,
                "energy": [scatter(x[0], BURN + i * w8, w8, mu, var_u) for i in (0, 1)],
                "llr8": [scatter(x[1], BURN + i * w8, w8, t8["mean0"], t8["var0"]) for i in (0, 1)],
            }
        )
        # rounding at every computed depth (5 and 9 bits are exact on this codebook)
        rr = {"target": target["id"], "rows": []}
        for bits in bits_by_n[n]:
            if bits in (6, 8) and str(bits) in trow["llr"]:
                rr["rows"].append({"bits": bits, "reads": trow["llr"][str(bits)]["reads"]})
                continue
            e = llr_exact(target, j0, beta, bits, lz0, c0)
            e.pop("dk")
            e["max_coupling_error"] = float(np.max(np.abs(rv.rounded(j0, bits) - j0)))
            rr["rows"].append(e)
        out["rounding"].append(rr)
        # missing couplings: own-edge statistic, binomial at independent reads
        if n <= 256:
            idx = range(len(j0)) if n == 64 else range(0, len(j0), max(1, len(j0) // 48))
            mrows = []
            pos, edges = p16.target_positions(target), p16.edge_tuples(target)
            for e in idx:
                keep = [i for i in range(len(j0)) if i != e]
                plus, minus = k0.copy(), k0.copy()
                plus[e], minus[e] = 1e-4, -1e-4
                c1 = (
                    p16.kac_ward_log_z(pos, edges, plus) - p16.kac_ward_log_z(pos, edges, minus)
                ) / 2e-4
                del keep
                q0, q1 = (1 - c0[e]) / 2, (1 - c1) / 2
                mrows.append(
                    {
                        "edge": int(e),
                        "abs_j": float(abs(j0[e])),
                        "c0": float(c0[e]),
                        "c1": float(c1),
                        "reads_known": reads_needed(float(q0), float(q1)),
                        "reads_scan": reads_needed(float(q0), float(q1), alpha=ALPHA / len(j0)),
                    }
                )
            out["missing"].append({"target": target["id"], "rows": mrows})
        print(f"U {target['id']} {time.time() - t0:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------------
# P. pass-rule arithmetic
# ----------------------------------------------------------------------------


def part_p():
    rows = []
    for reps in (400, 1000):
        for m in (1, 10, 25, 50, 100):
            a = ALPHA / (2 * m)
            k_fa = int(stats.binom.isf(a, reps, 0.05))  # fail if ideal rejections > k_fa
            k_det = int(stats.binom.ppf(a, reps, 0.95))  # fail if defective rejections < k_det
            while stats.binom.cdf(k_det - 1, reps, 0.95) > a:
                k_det -= 1
            rows.append(
                {
                    "replicates": reps,
                    "checks": m,
                    "max_ideal_rejections": k_fa,
                    "min_defect_rejections": k_det,
                    "familywise_false_failure_bound": float(
                        m
                        * (
                            stats.binom.sf(k_fa, reps, 0.05)
                            + stats.binom.cdf(k_det - 1, reps, 0.95)
                        )
                    ),
                    "power_true_fa_0p08": float(stats.binom.sf(k_fa, reps, 0.08)),
                    "power_true_fa_0p10": float(stats.binom.sf(k_fa, reps, 0.10)),
                    "power_true_detect_0p90": float(stats.binom.cdf(k_det - 1, reps, 0.90)),
                }
            )
    # draft rule for comparison: <= 31 of 400 and >= 369 of 400, about 54 checks
    single = stats.binom.sf(31, 400, 0.05) + stats.binom.cdf(368, 400, 0.95)
    return {
        "rows": rows,
        "draft_rule_single_check_false_failure": float(single),
        "draft_rule_54_checks_any_false_failure": float(1 - (1 - single) ** 54),
        "bias_margin_note": (
            "a mixing bias b in a statistic whose ideal-vs-defect shift is d, tested at n* "
            "(d = 3.29 sd/sqrt(n*)), moves false alarm from 5% to P(Z > 1.645 - 3.29 b/d)"
        ),
        "false_alarm_at_bias_fraction": {
            str(f): float(stats.norm.sf(1.645 - 3.29 * f)) for f in (0.02, 0.05, 0.1, 0.2)
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    res = {
        "proposal": "P-0015",
        "round": 1,
        "row": "z1-acceptance-kit",
        "label": "exploration, not evidence",
        "hardware_claim": False,
        "false_alarm": ALPHA,
        "miss": MISS,
        "read_limit": LIMIT,
        "read_definition": "one full-array snapshot after one complete two-colour sweep",
    }
    res["P_pass_rule"] = part_p()
    res["T_temperature"] = part_t()
    args.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    beta = res["T_temperature"]["chosen_beta"]
    print(f"T done {time.time() - t0:.0f}s chosen beta {beta}", flush=True)
    res["S_simple_tests"] = part_s(beta if beta else 0.5)
    args.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"S done {time.time() - t0:.0f}s", flush=True)
    if beta:
        res["U_chosen_beta"] = part_u(beta, res["T_temperature"]["rows"])
    res["seconds_total"] = time.time() - t0
    args.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"done {res['seconds_total']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
