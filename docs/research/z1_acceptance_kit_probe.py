"""P-0015 probe: exact-answer acceptance tests on the P-0001 planar 16-offset graphs.

Exploration, not evidence. CPU only:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        uv run python docs/research/z1_acceptance_kit_probe.py --out <json>

Parts (each labelled in the JSON):

A. Reproduce the 18 archived Kac-Ward references of planar-16-offset-ferro
   (P-0001) from the committed evidence archive (`exact_reference`).
B. Defect admission on small patches by exhaustive enumeration
   (`exact_reference` on the patch only): static per-site beta variation
   under the two-colour sweep (is the stationary law any pairwise Ising law?)
   and one pinned spin (stationary conditional law; even and odd statistics).
C. Exact detection read counts for the admitted defects on the nine
   greedy-long targets at beta = 4 (coupling rounding at 4 to 12 bits under
   the `full` cap, one missing coupling, one stuck p-bit), for independent
   equilibrium reads. Means and per-read variances come from Kac-Ward ln Z
   and edge correlations (variances by a central second difference of ln Z;
   the FD error is checked on a brute-force patch). Read counts for the
   energy and log-likelihood-ratio statistics use the normal approximation on
   exact moments; single-edge and stuck-spin counts use exact binomial laws.
D. Ordinary two-colour Gibbs (no exchange) on L8 and L32 greedy-long s4510
   at beta = 4 from the archived random start and from a flip-symmetric
   ordered start: burn-in bias, integrated autocorrelation and the sampled
   shift under rounding (`software_simulation`, NumPy float64).
"""

from __future__ import annotations

import argparse
import itertools
import json
import tarfile
import time
from pathlib import Path

import numpy as np
from scipy import stats

from thermo_lab import planar_16_offset_ferro as p16
from thermo_lab.constraints.quantize import quantize_array

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/experiment-reports/2026-10-08-planar-16-offset-ferro/evidence.tar.gz"
BETA = 4.0
ALPHA = 0.05
MISS = 0.05
Z = float(stats.norm.isf(ALPHA))
READ_LIMIT = 10_000
SEED = 20261115


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def log_z(target, k):
    return p16.kac_ward_log_z(p16.target_positions(target), p16.edge_tuples(target), k)


def kw(target, k):
    """Kac-Ward ln Z and edge correlations; zero couplings handled exactly.

    The Kac-Ward edge formula divides by tanh(K_e), so an edge with K_e = 0
    is dropped from the graph (still planar) and its correlation is taken as
    d ln Z / d K_e at 0 by a central difference (h = 1e-4), as the archived
    reference checks do for weak edges.
    """
    positions, edges = p16.target_positions(target), p16.edge_tuples(target)
    k = np.asarray(k, dtype=float)
    zero = np.flatnonzero(k == 0)
    if zero.size == 0:
        return p16.kac_ward(positions, edges, k)
    keep = np.flatnonzero(k != 0)
    lz, c_keep = p16.kac_ward(positions, [edges[i] for i in keep], k[keep])
    corr = np.empty(len(k))
    corr[keep] = c_keep
    for e in zero:
        plus, minus = k.copy(), k.copy()
        plus[e], minus[e] = 1e-4, -1e-4
        corr[e] = (
            p16.kac_ward_log_z(positions, edges, plus) - p16.kac_ward_log_z(positions, edges, minus)
        ) / 2e-4
    return lz, corr


def second_derivative(target, k, direction, center):
    """Var(direction . S) under exp(k . S) = d^2/dh^2 ln Z(k + h direction) at 0."""
    h = 1e-3 / float(np.max(np.abs(direction)))
    plus = log_z(target, k + h * direction)
    minus = log_z(target, k - h * direction)
    return (plus - 2 * center + minus) / h**2


def normal_reads(mu0, mu1, var0, var1):
    delta = abs(mu1 - mu0)
    if delta == 0:
        return None
    return int(np.ceil(((Z * np.sqrt(max(var0, 0)) + Z * np.sqrt(max(var1, 0))) / delta) ** 2))


def binomial_reads(q0, q1, alpha=ALPHA, miss=MISS, limit=10**9):
    """Smallest n for a one-sided count test with exact binomial error rates.

    q0, q1: probability that one read shows the tested outcome under the
    ideal and the defective model. Returns None if q0 == q1. Bracketing then
    bisection on n; exact rates are re-checked at the returned n.
    """
    if q0 == q1:
        return None
    q0, q1 = min(max(q0, 0.0), 1.0), min(max(q1, 0.0), 1.0)
    if q0 == q1:
        return None
    if q1 < q0:
        q0, q1 = 1 - q0, 1 - q1

    def ok(n):
        t = stats.binom.isf(alpha, n, q0) + 1  # reject when X >= t
        while t > 0 and stats.binom.sf(t - 2, n, q0) <= alpha:
            t -= 1
        return stats.binom.cdf(t - 1, n, q1) <= miss

    hi = 1
    while not ok(hi):
        hi *= 2
        if hi > limit:
            return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return int(hi)


def archived_references():
    with tarfile.open(ARCHIVE) as tar:
        member = tar.extractfile("planar-16-offset-ferro/results.json")
        return json.load(member)["exact_references"]


# ----------------------------------------------------------------------------
# A. reproduce the 18 references
# ----------------------------------------------------------------------------


def part_a():
    archived = archived_references()
    rows, refs = [], {}
    for target in p16.study_targets():
        t0 = time.time()
        ref = p16.reference(target)
        old = archived[target["id"]]
        refs[target["id"]] = ref
        rows.append(
            {
                "target": target["id"],
                "edges": len(target["edges"]),
                "log_z_abs_diff": abs(ref["log_z"] - old["log_z"]),
                "edge_max_abs_diff": float(
                    np.max(np.abs(np.asarray(ref["edge"]) - np.asarray(old["edge"])))
                ),
                "q_per_spin_abs_diff": abs(ref["q_per_spin"] - old["q_per_spin"]),
                "bitwise_equal": bool(ref["log_z"] == old["log_z"] and ref["edge"] == old["edge"]),
                "min_abs_edge_correlation": ref["min_abs_edge_correlation"],
                "seconds": time.time() - t0,
            }
        )
    worst = max(max(r["log_z_abs_diff"], r["edge_max_abs_diff"]) for r in rows)
    return {
        "evidence_class": "exact_reference",
        "targets": len(rows),
        "bitwise_equal_all": all(r["bitwise_equal"] for r in rows),
        "max_abs_diff": worst,
        "reproduced": bool(len(rows) == 18 and worst <= 1e-9),
        "rows": rows,
    }, refs


# ----------------------------------------------------------------------------
# B. defect admission by enumeration on small patches
# ----------------------------------------------------------------------------


def patch_model(rows, cols, seed):
    edges, positions = p16.planar_subgraph(rows, cols, "greedy-long", seed)
    rng = np.random.default_rng(seed * 1000 + rows * cols)
    weights = rng.integers(1, 6, len(edges)) / 5
    return edges, positions, -weights


def sweep_matrix(n, edges, couplings, beta_site, colour, pinned=None):
    """Exact transition matrix of one colour-A-then-colour-B parallel heat-bath sweep.

    Site i uses P(s_i = +1 | rest) = sigmoid(2 beta_i h_i), h_i = sum_j J_ij s_j
    (the archived sampler's convention with a per-site gain). A pinned site
    keeps its value.
    """
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    coupling = np.zeros((n, n))
    for (i, j), c in zip(edges, couplings, strict=True):
        coupling[i, j] = coupling[j, i] = c
    index = ((states > 0).astype(int) * (2 ** np.arange(n - 1, -1, -1))).sum(axis=1)
    assert np.array_equal(index, np.arange(len(states)))

    def half(mask):
        field = states @ coupling  # (S, n)
        up = 1 / (1 + np.exp(-2 * beta_site[None, :] * field))
        free = [i for i in range(n) if mask[i] and not (pinned is not None and i == pinned)]
        prob = np.zeros((len(states), len(states)))
        weights = 2 ** np.arange(n - 1, -1, -1)
        rows = np.arange(len(states))
        for new in itertools.product([0, 1], repeat=len(free)):
            p = np.ones(len(states))
            target = (states > 0).astype(int)
            for i, v in zip(free, new, strict=True):
                p = p * (up[:, i] if v else 1 - up[:, i])
                target[:, i] = v
            prob[rows, target @ weights] += p
        return prob

    return half(colour) @ half(~colour), states


def stationary(matrix):
    """Stationary vector by a direct solve of pi (P - I) = 0 with sum(pi) = 1."""
    a = matrix.T - np.eye(len(matrix))
    a[-1, :] = 1.0
    b = np.zeros(len(matrix))
    b[-1] = 1.0
    v = np.linalg.solve(a, b)
    return v / v.sum()


def pairwise_fit_residual(states, law, support=None):
    """Least-squares residual of ln(law) on {1, s_i, s_i s_j (all pairs)}."""
    n = states.shape[1]
    feats = [np.ones(len(states))] + [states[:, i] for i in range(n)]
    feats += [states[:, i] * states[:, j] for i in range(n) for j in range(i + 1, n)]
    x = np.stack(feats, axis=1)
    y = np.log(law)
    if support is not None:
        x, y = x[support], y[support]
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    return float(np.sqrt(np.mean(resid**2))), float(np.max(np.abs(resid)))


def part_b():
    out = {"evidence_class": "exact_reference (enumerated patches only)", "beta_variation": []}
    rows, cols, seed = 3, 4, 7
    edges, positions, couplings = patch_model(rows, cols, seed)
    n = rows * cols
    colour = (positions.sum(axis=1) % 2) == 0
    products_idx = (np.array([e[0] for e in edges]), np.array([e[1] for e in edges]))
    for beta in (1.0, BETA):
        for jitter in (0.0, 0.1, 0.3):
            rng = np.random.default_rng([SEED, int(jitter * 100), int(beta)])
            gains = rng.uniform(1 - jitter, 1 + jitter, n) if jitter else np.ones(n)
            matrix, states = sweep_matrix(n, edges, couplings, beta * gains, colour)
            law = stationary(matrix)
            support = law > 1e-9 * law.max()
            rms, mx = pairwise_fit_residual(states, law, support)
            prods = states[:, products_idx[0]] * states[:, products_idx[1]]
            boltz = np.exp(beta * prods @ couplings)
            boltz /= boltz.sum()
            corr_sweep = law @ prods
            corr_boltz = boltz @ prods
            # Detailed balance of the sweep chain itself (it is not reversible even
            # when ideal; reported for completeness).
            flux = law[:, None] * matrix
            out["beta_variation"].append(
                {
                    "patch": [rows, cols],
                    "graph_seed": seed,
                    "edges": len(edges),
                    "beta": beta,
                    "jitter": jitter,
                    "gain_range": [float(gains.min()), float(gains.max())],
                    "support_states": int(support.sum()),
                    "pairwise_family_fit_rms": rms,
                    "pairwise_family_fit_max": mx,
                    "nominal_boltzmann_max_edge_shift": float(
                        np.max(np.abs(corr_sweep - corr_boltz))
                    ),
                    "nominal_energy_per_spin_shift": float(
                        couplings @ (corr_sweep - corr_boltz) / n
                    ),
                    "flip_symmetric": bool(np.allclose(law, law[::-1], atol=1e-12)),
                    "sweep_detailed_balance_max": float(np.max(np.abs(flux - flux.T))),
                }
            )

    # One pinned spin: stationary law of the kernel with site k frozen at +1.
    k = 5
    matrix, states = sweep_matrix(n, edges, couplings, BETA * np.ones(n), colour, pinned=k)
    support = states[:, k] > 0
    sub = matrix[np.ix_(support, support)]
    law_sub = stationary(sub)
    prods = states[:, products_idx[0]] * states[:, products_idx[1]]
    boltz = np.exp(BETA * prods @ couplings)
    boltz /= boltz.sum()
    cond = boltz * support
    cond /= cond.sum()
    neighbours = sorted({j for i, j in edges if i == k} | {i for i, j in edges if j == k})
    edge_with_k = [e for e, (i, j) in enumerate(edges) if k in (i, j)]
    out["stuck_spin"] = {
        "patch": [rows, cols],
        "pinned_site": k,
        "pinned_value": 1,
        "stationary_equals_conditional_max": float(np.max(np.abs(law_sub - cond[support]))),
        "even_edge_max_shift_vs_unpinned": float(
            np.max(np.abs(law_sub @ prods[support] - boltz @ prods))
        ),
        "odd_neighbour_means": [float(law_sub @ states[support][:, j]) for j in neighbours],
        "ideal_edge_correlations_to_pinned": [float(boltz @ prods[:, e]) for e in edge_with_k],
        "odd_identity_max": float(
            max(
                abs(law_sub @ states[support][:, j] - boltz @ (states[:, j] * states[:, k]))
                for j in neighbours
            )
        ),
    }
    # FD second-derivative variance against brute force on a 4x4 patch.
    edges4, pos4, coup4 = patch_model(4, 4, 3)
    k4 = BETA * coup4
    st = np.array(list(itertools.product([-1.0, 1.0], repeat=16)))
    pr = st[:, [e[0] for e in edges4]] * st[:, [e[1] for e in edges4]]
    w = np.exp(pr @ k4 - (pr @ k4).max())
    w /= w.sum()
    direction = BETA * (quantize_array(coup4, 1.0, 4) - coup4)

    center = p16.kac_ward_log_z(pos4, edges4, k4)
    h = 1e-3 / np.max(np.abs(direction))
    fd = (
        p16.kac_ward_log_z(pos4, edges4, k4 + h * direction)
        - 2 * center
        + p16.kac_ward_log_z(pos4, edges4, k4 - h * direction)
    ) / h**2
    stat = pr @ direction
    brute = float(w @ stat**2 - (w @ stat) ** 2)
    out["fd_variance_check"] = {
        "patch": [4, 4],
        "fd": fd,
        "brute": brute,
        "rel_err": abs(fd - brute) / brute,
    }

    return out


# ----------------------------------------------------------------------------
# C. exact detection read counts
# ----------------------------------------------------------------------------


def detection_for_defect(target, k0, lz0, c0, j0, k1, label, var_u0, edge_scan=True):
    """Statistics for one defect with exact means and per-read variances."""
    n = target["n"]
    lz1, c1 = kw(target, k1)
    m = len(c0)
    # energy per spin u = J0 . S / N (nominal couplings)
    mu0, mu1 = j0 @ c0 / n, j0 @ c1 / n
    var_u1 = second_derivative(target, k1, BETA * j0, lz1) / (BETA * n) ** 2
    energy_reads = normal_reads(mu0, mu1, var_u0, var_u1)
    # log-likelihood ratio, the most powerful test for independent reads
    dk = k1 - k0
    llr = None
    if np.any(dk != 0):
        v0 = second_derivative(target, k0, dk, lz0)
        v1 = second_derivative(target, k1, dk, lz1)
        llr = {
            "mean_shift": float(dk @ (c1 - c0)),
            "var0": v0,
            "var1": v1,
            "kl_1_0": float(lz0 - lz1 + dk @ c1),
            "reads": normal_reads(dk @ c0, dk @ c1, v0, v1),
        }
    # best single edge, location known, and the scan over all edges (Bonferroni)
    q0, q1 = (1 - c0) / 2, (1 - c1) / 2
    order = np.argsort(-np.abs(c1 - c0))[:5]
    singles = []
    for e in order:
        singles.append(
            {
                "edge": int(e),
                "c0": float(c0[e]),
                "c1": float(c1[e]),
                "reads_known": binomial_reads(float(q0[e]), float(q1[e])),
                "reads_scan": binomial_reads(float(q0[e]), float(q1[e]), alpha=ALPHA / m)
                if edge_scan
                else None,
            }
        )
    best = min(
        (s for s in singles if s["reads_known"]), key=lambda s: s["reads_known"], default=None
    )
    return {
        "defect": label,
        "log_z_shift": float(lz1 - lz0),
        "energy_per_spin": {
            "mu0": float(mu0),
            "mu1": float(mu1),
            "shift": float(mu1 - mu0),
            "sd0": float(np.sqrt(var_u0)),
            "sd1": float(np.sqrt(var_u1)),
            "reads": energy_reads,
        },
        "llr": llr,
        "max_abs_edge_shift": float(np.max(np.abs(c1 - c0))),
        "best_single_edges": singles,
        "best_single_edge_reads_known": best["reads_known"] if best else None,
        "best_single_edge_reads_scan": best["reads_scan"] if best else None,
    }


def part_c(refs, bits_by_size, missing_by_size):
    out = {
        "evidence_class": (
            "exact_reference (means, per-read variances); "
            "normal approximation for energy and LLR read counts"
        ),
        "targets": [],
    }
    for target in p16.study_targets():
        if target["graph"] != "greedy-long":
            continue
        t0 = time.time()
        size, n = target["size"], target["n"]
        ref = refs[target["id"]]
        j0 = np.asarray(target["couplings"], dtype=float)
        k0 = BETA * j0
        lz0, c0 = ref["log_z"], np.asarray(ref["edge"])
        var_u0 = second_derivative(target, k0, BETA * j0, lz0) / (BETA * n) ** 2
        row = {"target": target["id"], "n": n, "edges": len(j0), "rounding": [], "missing": []}
        cap = float(np.max(np.abs(j0)))
        for bits in bits_by_size[size]:
            j1 = quantize_array(j0, cap, bits)
            res = detection_for_defect(target, k0, lz0, c0, j0, BETA * j1, f"round{bits}", var_u0)
            res["bits"] = bits
            res["max_abs_coupling_error"] = float(np.max(np.abs(j1 - j0)))
            res["cap"] = cap
            row["rounding"].append(res)
        # missing couplings: all edges at small sizes, a declared few at L32
        lengths = np.array(
            [
                np.sum((p16.target_positions(target)[i] - p16.target_positions(target)[j]) ** 2)
                for i, j in p16.edge_tuples(target)
            ]
        )
        if missing_by_size[size] == "all":
            chosen = list(range(len(j0)))
        elif missing_by_size[size] == "stride":
            chosen = list(range(0, len(j0), max(1, len(j0) // 48)))
        else:
            # declared few at L32: the weakest correlation and the middle (1,4) edge
            weakest_c = int(np.argmin(np.abs(c0)))
            long_mid = int(np.flatnonzero(lengths == 17)[len(np.flatnonzero(lengths == 17)) // 2])
            chosen = sorted({weakest_c, long_mid})
        miss_rows = []
        for e in chosen:
            j1 = j0.copy()
            j1[e] = 0.0
            if missing_by_size[size] == "few":
                res = detection_for_defect(
                    target, k0, lz0, c0, j0, BETA * j1, f"missing{e}", var_u0
                )
            else:
                # cheap path: own-edge statistic only (no variance second differences)
                _, c1 = kw(target, BETA * j1)
                q0, q1 = (1 - c0[e]) / 2, (1 - c1[e]) / 2
                res = {
                    "defect": f"missing{e}",
                    "own_edge_c0": float(c0[e]),
                    "own_edge_c1": float(c1[e]),
                    "own_edge_reads_known": binomial_reads(float(q0), float(q1)),
                    "own_edge_reads_scan": binomial_reads(
                        float(q0), float(q1), alpha=ALPHA / len(j0)
                    ),
                    "energy_shift": float(j0 @ (c1 - c0) / n),
                    "max_abs_edge_shift": float(np.max(np.abs(c1 - c0))),
                }
            res["edge"] = int(e)
            res["coupling"] = float(j0[e])
            res["length2"] = int(lengths[e])
            miss_rows.append(res)
        row["missing"] = miss_rows
        # stuck p-bit: exact binomial counts from flip symmetry (any finite time,
        # independent chains with a flip-symmetric start; see note)
        row["stuck"] = {
            "reads_known_site_known_value": int(np.ceil(np.log2(1 / ALPHA))),
            "reads_known_site": int(np.ceil(1 + np.log2(1 / ALPHA))),
            "reads_scan_all_sites": int(np.ceil(1 + np.log2(n / ALPHA))),
            "miss_probability": 0.0,
        }
        row["var_u0"] = var_u0
        row["seconds"] = time.time() - t0
        out["targets"].append(row)
        print(f"C {target['id']} {row['seconds']:.1f}s", flush=True)
    return out


# ----------------------------------------------------------------------------
# D. ordinary two-colour Gibbs (software_simulation)
# ----------------------------------------------------------------------------


def gibbs_run(target, couplings_run, j_nominal, start, chains, sweeps, seed, dk=None):
    n = target["n"]
    edges = p16.edge_tuples(target)
    nbr, wts = p16.neighbour_arrays(n, edges, couplings_run)
    wts = wts.astype(np.float64)
    pos = p16.target_positions(target)
    colour = (pos.sum(axis=1) % 2) == 0
    ei = np.array([e[0] for e in edges])
    ej = np.array([e[1] for e in edges])
    rng = np.random.default_rng(seed)
    if start == "random":
        s = rng.choice([-1.0, 1.0], size=(chains, n))
    else:
        neel = np.where(colour, 1.0, -1.0)
        sign = np.where(np.arange(chains) % 2 == 0, 1.0, -1.0)[:, None]
        # couplings are negative (antiferro in this sign convention): Neel is a ground state
        s = sign * neel[None, :]
    a_idx, b_idx = np.flatnonzero(colour), np.flatnonzero(~colour)
    u = np.empty((sweeps, chains))
    llr = np.empty((sweeps, chains)) if dk is not None else None
    edge_sum = np.zeros((chains, len(edges)))
    for t in range(sweeps):
        for idx in (a_idx, b_idx):
            field = np.einsum("ckw,kw->ck", s[:, nbr[idx]], wts[idx])
            up = 1 / (1 + np.exp(-2 * BETA * field))
            s[:, idx] = np.where(rng.random(up.shape) < up, 1.0, -1.0)
        prod = s[:, ei] * s[:, ej]
        u[t] = prod @ j_nominal / n
        if dk is not None:
            llr[t] = prod @ dk
        if t >= sweeps // 2:
            edge_sum += prod
    return u, llr, edge_sum / (sweeps - sweeps // 2)


def tau_int(x):
    """Integrated autocorrelation time per chain series, averaged (Sokal window c=5)."""
    x = x - x.mean(axis=0, keepdims=True)
    var = np.mean(x * x)
    if var == 0:
        return 1.0
    n = x.shape[0]
    tau = 1.0
    for lag in range(1, n // 4):
        rho = np.mean(x[:-lag] * x[lag:]) / var
        tau += 2 * rho
        if lag >= 5 * tau:
            break
    return float(tau)


def part_d(refs, plan):
    out = {"evidence_class": "software_simulation (NumPy float64, CPU)", "runs": []}
    for size, chains, sweeps in plan:
        target = p16.build_target(size, "greedy-long", 4510)
        ref = refs[target["id"]]
        n = target["n"]
        j0 = np.asarray(target["couplings"], dtype=float)
        c0 = np.asarray(ref["edge"])
        mu0 = j0 @ c0 / n
        cap = float(np.max(np.abs(j0)))
        cases = [("ideal", "random", j0), ("ideal", "ordered", j0)]
        for bits in (4, 8):
            cases.append((f"round{bits}", "ordered", quantize_array(j0, cap, bits)))
        for label, start, jrun in cases:
            t0 = time.time()
            dk = BETA * (quantize_array(j0, cap, 8) - j0)
            u, llr, edges_mean = gibbs_run(target, jrun, j0, start, chains, sweeps, SEED + size, dk)
            _, c_run = (None, c0) if label == "ideal" else kw(target, BETA * jrun)
            mu_run = j0 @ c_run / n
            windows = []
            for budget in (64, 256, 1024, 4096, 16384):
                if budget > sweeps:
                    continue
                w = u[budget // 4 : budget].mean(axis=0)
                windows.append(
                    {
                        "budget": budget,
                        "u_bias_vs_model_exact": float(w.mean() - mu_run),
                        "u_bias_se": float(w.std(ddof=1) / np.sqrt(chains)),
                    }
                )
            second = u[sweeps // 2 :]
            run = {
                "size": size,
                "case": label,
                "start": start,
                "chains": chains,
                "sweeps": sweeps,
                "exact_u_ideal": float(mu0),
                "exact_u_model": float(mu_run),
                "exact_u_shift": float(mu_run - mu0),
                "windows": windows,
                "second_half_u_bias": float(second.mean() - mu_run),
                "second_half_u_bias_se_chains": float(
                    second.mean(axis=0).std(ddof=1) / np.sqrt(chains)
                ),
                "second_half_edge_mae": float(np.mean(np.abs(edges_mean.mean(axis=0) - c_run))),
                "second_half_edge_max_err": float(np.max(np.abs(edges_mean.mean(axis=0) - c_run))),
                "frac_chains_with_frozen_wall_edge": float(
                    np.mean(np.any(edges_mean * np.sign(c_run)[None, :] < 0, axis=1))
                ),
                "tau_int_u_second_half": tau_int(second),
                "tau_int_llr8_second_half": tau_int(llr[sweeps // 2 :]),
                "sd_u_per_read": float(second.std()),
                "seconds": time.time() - t0,
            }
            out["runs"].append(run)
            print(
                f"D L{size} {label} {start} {run['seconds']:.1f}s",
                flush=True,
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    result = {
        "proposal": "P-0015",
        "row": "z1-acceptance-kit",
        "label": "exploration, not evidence",
        "beta": BETA,
        "false_alarm": ALPHA,
        "miss": MISS,
        "read_limit": READ_LIMIT,
        "read_definition": "one full-array snapshot after one complete two-colour sweep",
        "hardware_claim": False,
    }
    a, refs = part_a()
    result["A_reproduce_references"] = a
    print(f"A done {time.time() - t0:.0f}s reproduced={a['reproduced']}", flush=True)
    result["B_defect_admission"] = part_b()
    print(f"B done {time.time() - t0:.0f}s", flush=True)
    bits = {8: (4, 5, 6, 7, 8, 9, 10, 12), 16: (4, 5, 6, 7, 8, 9, 10, 12), 32: (4, 6, 8)}
    missing = {8: "all", 16: "stride", 32: "few"}
    result["C_exact_detection"] = part_c(refs, bits, missing)
    print(f"C done {time.time() - t0:.0f}s", flush=True)
    plan = [(8, 64, 16384), (16, 64, 16384), (32, 64, 16384)] if not args.quick else [(8, 16, 1024)]
    result["D_ordinary_gibbs"] = part_d(refs, plan)
    result["seconds_total"] = time.time() - t0
    Path(args.out).write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(f"done {result['seconds_total']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
