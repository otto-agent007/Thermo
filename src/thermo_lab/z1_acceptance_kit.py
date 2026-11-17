"""Z1 acceptance kit: exact-answer tests for a sampled array (proposal P-0015).

Protocol: ``docs/experiments/z1-acceptance-kit.md`` (frozen at commit dc32142).

For four declared defects (coupling rounding, a missing coupling, a stuck p-bit,
static per-site gain variation) the study computes the reads and programming
writes of the cheapest exact test at 5 percent false alarm and 5 percent miss,
and checks every retained test's operating characteristic by sampling.

- Simple tests (isolated pairs and sites) have closed-form laws; their counts
  are ``exact_reference``. The pooled gain-spread count and threshold are a
  Monte Carlo evaluation of exact per-site binomial laws, averaged over gain
  draws ("Monte Carlo on exact laws").
- The collective test is the Bonferroni scan for a missing coupling on the
  nine P-0001 ``greedy-long`` graphs at zero field. Its exact means come from
  the Kac-Ward determinant imported unchanged from ``planar_16_offset_ferro``;
  the correlation of a missing edge is d ln Z / dK_e at K_e = 0, for every
  edge by a rank-2 update of the Kac-Ward inverse, checked against a direct
  evaluation on the rebuilt matrix (1e-9) and against the central difference
  (the finite-difference check, 1e-4, the P-0001 reference tolerance).
- Scatter ratios, sampled means and operating characteristics are
  ``software_simulation`` (NumPy float64, CPU). Nothing runs on hardware and no
  result here is a claim about any chip.

One read is the full array after one complete two-colour sweep (colour A,
then colour B). Replicates, not reads, are the replication unit. Every chain
and replicate draws from its own generator, folded from a declared root with
arm, target, beta, model and replicate, so a replicate is reproducible alone.

Long run (about 1 to 1.5 h with three workers): every work unit is written
atomically to ``units/`` with a digest index; ``--resume`` re-authenticates
the request and sources, verifies each unit and runs only the missing ones.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from thermo_lab import fixed_budget_sampling as base
from thermo_lab import planar_16_offset_ferro as p16
from thermo_lab import planar_ising_scaling as pis
from thermo_lab.constraints.quantize import quantize_array
from thermo_lab.hashing import canonical_sha256
from thermo_lab.provenance import collect_runtime_provenance

ROOT = p16.ROOT
PROTOCOL = "docs/experiments/z1-acceptance-kit.md"
P0001_ARCHIVE = "docs/experiment-reports/2026-10-08-planar-16-offset-ferro/evidence.tar.gz"
SOURCES = (
    "src/thermo_lab/z1_acceptance_kit.py",
    "src/thermo_lab/planar_16_offset_ferro.py",
    "src/thermo_lab/planar_ising_scaling.py",
    "src/thermo_lab/fixed_budget_sampling.py",
    "src/thermo_lab/constraints/quantize.py",
    "src/thermo_lab/hashing.py",
    "src/thermo_lab/persistence.py",
    PROTOCOL,
    P0001_ARCHIVE,
    "uv.lock",
)
STATUS = "z1_acceptance_kit_complete"
ALPHA = MISS = 0.05
ZB = float(stats.norm.isf(MISS))
READ_LIMIT = 10_000
DF = 127
BURN = 512
SIZES = (8, 16, 32)
SEEDS = (4510, 4511, 4512)
CANDIDATES = (0.5, 0.6, 0.75)
REFERENCE_BETA = 4.0
DRAW_ROOTS = {64: 4520, 256: 4521, 1024: 4522}
SEED_CAL, SEED_MC, SEED_TEST = 20261121, 20261122, 20261118
PROBE_SEEDS = (20261115, 20261117, 20261119, 20261120, 20261123, 20261124, 20261125)
ARM_CAL, ARM_COLLECTIVE, ARM_SIMPLE, ARM_BOOT = 1, 3, 4, 5
WEIGHTS = (0.2, 0.4, 0.6, 0.8, 1.0)
ROUNDING_BITS = (4, 5, 6, 7, 8, 9, 10, 11, 12)
CHECK_BITS = (4, 6, 7, 8)
GAIN_SPREADS = (0.1, 0.3)
OUTLIER_DELTAS = (0.1, 0.3)
SITE_BIAS = 0.5  # site tests program bias 0.5 and beta = 2 x, so beta h = x
UNIFORM_BLOCK = 16  # sweeps of uniforms drawn per replicate at a time (no effect on values)
DEFAULT_WORKERS = 3
EXACT_WORKERS = 2  # dense 1,024-spin Kac-Ward work peaks near 1.2 GiB per process
REPLAY_TOLERANCE = 2e-12
ARCHIVE_HOST = os.environ.get("THERMO_ARCHIVE_HOST") == "1"


class StudyStop(RuntimeError):
    """A protocol stop rule fired; the run stops before the next stage."""


# ----------------------------------------------------------------------------
# Small numerical helpers
# ----------------------------------------------------------------------------


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def t_quantile(edges):
    return float(stats.t.isf(ALPHA / edges, DF))


def target_code(target):
    return int(target["size"]) * 100_000 + int(target["seed"])


def beta_code(beta):
    return int(round(beta * 1000))


def greedy_targets():
    return [p16.build_target(size, "greedy-long", seed) for size in SIZES for seed in SEEDS]


def target_by_id(request, tid):
    for target in request["targets"]:
        if target["id"] == tid:
            return target
    raise KeyError(tid)


# ----------------------------------------------------------------------------
# Exact binomial count tests (closed form)
# ----------------------------------------------------------------------------


def upper_threshold(trials, q0, alpha):
    """Smallest t with P0(X >= t) <= alpha (the round-1 probe's construction)."""
    t = int(stats.binom.isf(alpha, trials, q0)) + 1
    while t > 0 and stats.binom.sf(t - 2, trials, q0) <= alpha:
        t -= 1
    return t


def _ok(trials, q0, q1, alpha, miss):
    t = upper_threshold(trials, q0, alpha)
    return stats.binom.cdf(t - 1, trials, q1) <= miss


def reads_needed(q0, q1, per_read=1, alpha=ALPHA, miss=MISS, max_trials=10**10):
    """Smallest n reads (each giving per_read iid trials) for the one-sided count test."""
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


def count_test(q0, q1, trials, alpha=ALPHA):
    """Rejection rule for a count of 'agree' outcomes: direction and threshold."""
    if q1 < q0:  # count disagreements
        return {"count": "disagree", "reject_at_or_above": upper_threshold(trials, 1 - q0, alpha)}
    return {"count": "agree", "reject_at_or_above": upper_threshold(trials, q0, alpha)}


def two_sided_bounds(n, p0, alpha):
    """Two-sided exact binomial at alpha/2 per tail: reject if K <= a or K >= b."""
    a = int(stats.binom.ppf(alpha / 2, n, p0))
    if stats.binom.cdf(a, n, p0) > alpha / 2:
        a -= 1
    b = int(stats.binom.isf(alpha / 2, n, p0)) + 1
    while b > 0 and stats.binom.sf(b - 2, n, p0) <= alpha / 2:
        b -= 1
    while stats.binom.sf(b - 1, n, p0) > alpha / 2:
        b += 1
    return a, b


def two_sided_reads(x, delta, alpha, nmax=40_000):
    """Round-2 probe count: two-sided test, power >= 0.95 at gains 1 + delta and 1 - delta."""
    p0, pp, pm = sig(2 * x), sig(2 * x * (1 + delta)), sig(2 * x * (1 - delta))
    z = stats.norm.isf(alpha / 2)
    sd = np.sqrt(p0 * (1 - p0))
    guess = int(((z + ZB) * sd / min(pp - p0, p0 - pm)) ** 2)

    def passes(n):
        a, b = two_sided_bounds(n, p0, alpha)
        low = (lambda p: stats.binom.cdf(a, n, p)) if a >= 0 else (lambda p: 0.0)
        power = [stats.binom.sf(b - 1, n, p) + low(p) for p in (pp, pm)]
        return min(power) >= 1 - MISS

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


# ----------------------------------------------------------------------------
# Pooled gain-spread test (approximation and draw-averaged Monte Carlo)
# ----------------------------------------------------------------------------

GL = np.polynomial.legendre.leggauss(64)


def spread_moments(x, j):
    g = 1.0 + j * GL[0]
    p = sig(2 * x * g)
    m = float(np.sum(GL[1] * p) / 2)
    v = float(np.sum(GL[1] * (p - m) ** 2) / 2)
    return m, v


def pooled_power(n, sites, x, j):
    m, v = spread_moments(x, j)
    lam = n * sites * v / (m * (1 - m))
    return float(stats.ncx2.sf(stats.chi2.isf(ALPHA, sites - 1), sites - 1, lam))


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


def x2_stat(k, n):
    """Pearson homogeneity statistic over sites, per row of up counts k."""
    ph = k.sum(axis=1) / (n * k.shape[1])
    ph = np.clip(ph, 1e-12, 1 - 1e-12)
    return np.sum((k - n * ph[:, None]) ** 2, axis=1) / (n * ph * (1 - ph))


def gain_draws(sites, j, first, count):
    """Gain arrays from the declared draw root, folded with the replicate index."""
    root = DRAW_ROOTS[sites]
    return np.stack(
        [
            np.random.default_rng([root, r]).uniform(1 - j, 1 + j, sites)
            for r in range(first, first + count)
        ]
    )


def mc_threshold_power(n, sites, x, j, gains, chunk=2000):
    """Null threshold (95th-percentile order statistic) and draw-averaged power at n reads."""
    reps = gains.shape[0]
    rng = np.random.default_rng([SEED_MC, sites, beta_code(j), int(round(x * 1000)), n])
    p0 = sig(2 * x)
    null = np.concatenate(
        [
            x2_stat(rng.binomial(n, p0, size=(min(chunk, reps - i), sites)), n)
            for i in range(0, reps, chunk)
        ]
    )
    ordered = np.sort(null)
    crit = float(ordered[int(np.ceil((1 - ALPHA) * reps)) - 1])
    alt = np.concatenate(
        [
            x2_stat(rng.binomial(n, sig(2 * x * gains[i : i + chunk])), n)
            for i in range(0, reps, chunk)
        ]
    )
    return crit, float((null > crit).mean()), float((alt > crit).mean())


def gain_spread_row(sites, j, reps, xs):
    """Draw-averaged Monte Carlo count and threshold at the approximation's best beta h."""
    best = None
    for x in xs:
        n = pooled_reads(sites, float(x), j)
        if n and (best is None or n < best[1]):
            best = (float(x), n)
    x, approx = best
    gains = gain_draws(sites, j, 0, reps)
    cache = {}

    def ok(n):
        if n not in cache:
            cache[n] = mc_threshold_power(n, sites, x, j, gains)
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
    crit, fa, power = cache[hi]
    declared = np.random.default_rng(DRAW_ROOTS[sites]).uniform(1 - j, 1 + j, sites)
    rng = np.random.default_rng([SEED_MC, sites, beta_code(j), 7, hi])
    kd = rng.binomial(hi, sig(2 * x * declared), size=(reps, sites))
    return {
        "spread_j": j,
        "sites": sites,
        "beta_h": x,
        "approx_reads": approx,
        "reads": hi,
        "threshold": crit,
        "chi2_threshold": float(stats.chi2.isf(ALPHA, sites - 1)),
        "mc_false_alarm": fa,
        "mc_power_draw_averaged": power,
        "declared_draw_power_draw_specific": float((x2_stat(kd, hi) > crit).mean()),
        "mc_replicates": reps,
        "programming_writes": 1,
        "evidence_class": "monte_carlo_on_exact_laws",
    }


def smallest_spread(sites, reps, xs, steps):
    """Smallest uniform spread j with draw-averaged power >= 0.95 at 10,000 reads."""
    n = READ_LIMIT
    base_u = gain_draws(sites, 0.5, 0, reps)  # uniform(0.5, 1.5) = 1 + (u - 0.5) * 1
    lo, hi = 1e-3, 0.3
    for _ in range(steps):
        mid = (lo + hi) / 2
        x = min(xs, key=lambda v: pooled_reads(sites, float(v), mid) or 10**9)
        gains = 1.0 + (base_u - 1.0) * (2 * mid)
        _, _, power = mc_threshold_power(n, sites, float(x), mid, gains)
        lo, hi = (lo, mid) if power >= 1 - MISS else (mid, hi)
    return {"sites": sites, "reads": n, "smallest_spread_j": hi, "bisection_steps": steps}


# ----------------------------------------------------------------------------
# Graph facts and matchings
# ----------------------------------------------------------------------------


def colours(target):
    return (p16.target_positions(target).sum(axis=1) % 2) == 0


def matching(target):
    """A maximum matching of the target's graph (pairs (a, b), a in colour A), sorted."""
    n = target["n"]
    edges = p16.edge_tuples(target)
    colour = colours(target)
    a_sites, b_sites = np.flatnonzero(colour), np.flatnonzero(~colour)
    a_idx = {int(s): r for r, s in enumerate(a_sites)}
    b_idx = {int(s): r for r, s in enumerate(b_sites)}
    rows, cols = [], []
    for i, j in edges:
        a, b = (i, j) if colour[i] else (j, i)
        rows.append(a_idx[a])
        cols.append(b_idx[b])
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(a_sites), len(b_sites)))
    match = maximum_bipartite_matching(graph, perm_type="column")
    pairs = sorted((int(a_sites[r]), int(b_sites[c])) for r, c in enumerate(match) if c >= 0)
    degree = np.bincount(np.asarray(edges).ravel(), minlength=n)
    return pairs, int(degree.max())


def graph_facts(target):
    pairs, max_degree = matching(target)
    edges = set(p16.edge_tuples(target))
    colour = colours(target)
    used = [s for pair in pairs for s in pair]
    return {
        "target": target["id"],
        "n": target["n"],
        "edges": len(target["edges"]),
        "bipartite": bool(all(colour[i] != colour[j] for i, j in edges)),
        "max_degree": max_degree,
        "matchings_to_cover_edges": max_degree,
        "max_matching_pairs": len(pairs),
        "matching_valid": bool(
            len(set(used)) == len(used)
            and all(tuple(sorted(p)) in edges for p in pairs)
            and all(colour[a] and not colour[b] for a, b in pairs)
        ),
    }


# ----------------------------------------------------------------------------
# Exact collective tables: Kac-Ward c0 and the missing-edge correlation c1
# ----------------------------------------------------------------------------


def kac_ward_pattern(positions, edges):
    """Nonzeros (rows, cols, phase) of the Kac-Ward Lambda with unit edge weights.

    Same algebra as ``planar_16_offset_ferro.kac_ward_matrix``: Lambda[r, c] =
    phase * tanh(K) of directed edge c.
    """
    m = len(edges)
    position = np.asarray(positions, dtype=float)
    tail = np.array([e[0] for e in edges] + [e[1] for e in edges])
    head = np.array([e[1] for e in edges] + [e[0] for e in edges])
    delta = position[head] - position[tail]
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    rows, cols = [], []
    by_tail = {}
    for c, t in enumerate(tail):
        by_tail.setdefault(int(t), []).append(c)
    for r in range(2 * m):
        for c in by_tail.get(int(head[r]), []):
            if c != (r + m) % (2 * m):
                rows.append(r)
                cols.append(c)
    rows, cols = np.asarray(rows), np.asarray(cols)
    turn = (angle[cols] - angle[rows] + np.pi) % (2 * np.pi) - np.pi
    return rows, cols, np.exp(1j * turn / 2)


def missing_edge_correlations(positions, edges, interactions):
    """c1_e = d ln Z / dK_e at K_e = 0 for every edge, by a rank-2 update.

    With G = M^{-1} and h_e the 2 x 2 block of G P on directed edges (e, e + m),
    zeroing x_e = tanh K_e adds x_e P[:, (e, e+m)] to M, and
    c1_e = -1/2 Re tr(h_e (I + x_e h_e)^{-1}). Also returns the ideal
    c0_e = x_e - 1/2 (1 - x_e^2) Re tr(h_e) as a cross-check of the algebra.
    """
    m = len(edges)
    matrix, _, x = p16.kac_ward_matrix(positions, edges, interactions)
    g = np.linalg.inv(matrix)
    del matrix
    rows, cols, phase = kac_ward_pattern(positions, edges)
    edge_of, block = cols % m, (cols >= m).astype(int)
    h = np.zeros((m, 2, 2), complex)
    for a in (0, 1):
        np.add.at(h, (edge_of, a, block), g[edge_of + a * m, rows] * phase)
    del g
    eye = np.eye(2)
    inv = np.linalg.inv(eye[None] + x[:, None, None] * h)
    c1 = -0.5 * np.einsum("eab,eba->e", h, inv).real
    c0 = x - 0.5 * (1 - x**2) * np.einsum("eaa->e", h).real
    return c1, c0


def missing_edge_direct(positions, edges, interactions, e):
    """d ln Z / dK_e at K_e = 0 evaluated directly on the rebuilt matrix (LU, two rows)."""
    m = len(edges)
    k = np.array(interactions, dtype=float)
    k[e] = 0.0
    matrix, _, _ = p16.kac_ward_matrix(positions, edges, k)
    rows, cols, phase = kac_ward_pattern(positions, edges)
    lu = sla.lu_factor(matrix)
    total = 0.0
    for kk in (e, e + m):
        rhs = np.zeros(2 * m)
        rhs[kk] = 1.0
        row = sla.lu_solve(lu, rhs, trans=1)
        sel = cols == kk
        total += np.sum(row[rows[sel]] * phase[sel])
    return float(-0.5 * total.real)


def missing_edge_central(positions, edges, interactions, e, h=1e-4):
    plus, minus = np.array(interactions, dtype=float), np.array(interactions, dtype=float)
    plus[e], minus[e] = h, -h
    return (
        p16.kac_ward_log_z(positions, edges, plus) - p16.kac_ward_log_z(positions, edges, minus)
    ) / (2 * h)


def declared_edges(edge_count, count):
    return sorted({int(v) for v in np.linspace(0, edge_count - 1, count).round()})


def independent_reads(c0, c1, sign, q):
    shift = sign * (c0 - c1)
    return ((q * np.sqrt(1 - c0**2) + ZB * np.sqrt(1 - c1**2)) / shift) ** 2


def exact_unit(request, tid, beta):
    """Exact tables for one (target, beta): c0, c1 on every edge, direct and FD checks."""
    target = target_by_id(request, tid)
    positions, edges = p16.target_positions(target), p16.edge_tuples(target)
    j0 = np.asarray(target["couplings"], dtype=float)
    k0 = beta * j0
    _, c0 = p16.kac_ward(positions, edges, k0)
    c1, c0_alt = missing_edge_correlations(positions, edges, k0)
    picks = declared_edges(len(edges), request["direct_check_edges"])
    direct = np.array([missing_edge_direct(positions, edges, k0, e) for e in picks])
    central = np.array([missing_edge_central(positions, edges, k0, e) for e in picks])
    return {
        "c0": c0,
        "c1": c1,
        "c0_rank2_algebra": c0_alt,
        "checked_edges": np.asarray(picks, dtype=np.int64),
        "c1_direct": direct,
        "c1_central_difference": central,
    }


def exact_summary(request, tid, beta, arrays):
    target = target_by_id(request, tid)
    c0, c1 = arrays["c0"], arrays["c1"]
    picks = arrays["checked_edges"]
    sign = np.sign(np.asarray(target["couplings"], dtype=float))
    direct_err = float(np.max(np.abs(arrays["c1_direct"] - c1[picks])))
    fd_err = float(np.max(np.abs(arrays["c1_central_difference"] - c1[picks])))
    algebra_err = float(np.max(np.abs(arrays["c0_rank2_algebra"] - c0)))
    shift_positive = bool(np.all(sign * (c0 - c1) > 0))
    return {
        "target": tid,
        "beta": beta,
        "edges": int(len(c0)),
        "checked_edges": len(picks),
        "rank2_vs_direct_max_abs": direct_err,
        "central_difference_vs_rank2_max_abs": fd_err,
        "c0_algebra_max_abs": algebra_err,
        "shift_towards_zero_all_edges": shift_positive,
        "rank2_check_passed": bool(direct_err <= 1e-9 and algebra_err <= 1e-9 and shift_positive),
        "finite_difference_check_passed": bool(fd_err <= 1e-4),
    }


# ----------------------------------------------------------------------------
# Exact and enumeration checks (before any sampling)
# ----------------------------------------------------------------------------


def archived_references():
    with tarfile.open(ROOT / P0001_ARCHIVE) as tar:
        return json.load(tar.extractfile("planar-16-offset-ferro/results.json"))["exact_references"]


def reference_reproduction(ids):
    archived = archived_references()
    rows = []
    for target in p16.study_targets():
        if target["id"] not in ids:
            continue
        ref = p16.reference(target)
        old = archived[target["id"]]
        diff = max(
            abs(ref["log_z"] - old["log_z"]),
            float(np.max(np.abs(np.asarray(ref["edge"]) - np.asarray(old["edge"])))),
        )
        rows.append(
            {
                "target": target["id"],
                "max_abs_diff": diff,
                "bitwise_equal": bool(ref["log_z"] == old["log_z"] and ref["edge"] == old["edge"]),
                "finite_difference_max_abs_error": ref["finite_difference_max_abs_error"],
            }
        )
    return {
        "evidence_class": "exact_reference",
        "beta": REFERENCE_BETA,
        "rows": rows,
        "reproduced": sum(r["max_abs_diff"] <= 1e-9 for r in rows),
        "bitwise_equal": sum(r["bitwise_equal"] for r in rows),
    }


def patch_checks(betas):
    """Kac-Ward c0 and rank-2 c1 against brute force on small 16-offset patches."""
    rng = np.random.default_rng(2028)
    rows = []
    for r, c in ((4, 4), (3, 5), (2, 8)):
        for seed in (1, 2):
            edges, positions = p16.planar_subgraph(r, c, "greedy-long", seed)
            w = -(rng.integers(1, 6, len(edges)) / 5)
            for beta in betas:
                k = beta * w
                _, c0 = p16.kac_ward(positions, edges, k)
                c1, _ = missing_edge_correlations(positions, edges, k)
                _, exact0 = p16.brute_force(r * c, edges, k)
                exact1 = np.empty(len(edges))
                for e in range(len(edges)):
                    kk = k.copy()
                    kk[e] = 0.0
                    exact1[e] = p16.brute_force(r * c, edges, kk)[1][e]
                rows.append(
                    {
                        "patch": [r, c],
                        "graph_seed": seed,
                        "beta": beta,
                        "edges": len(edges),
                        "c0_max_abs_error": float(np.max(np.abs(c0 - exact0))),
                        "c1_max_abs_error": float(np.max(np.abs(c1 - exact1))),
                    }
                )
    worst = max(max(row["c0_max_abs_error"], row["c1_max_abs_error"]) for row in rows)
    facts = [graph_facts(t) for t in greedy_targets()]
    return {
        "evidence_class": "exact_reference (enumerated patches)",
        "brute_force_patches": rows,
        "max_abs_error": worst,
        "tolerance": 1e-9,
        "graphs": facts,
        "passed": bool(
            worst <= 1e-9 and all(f["bipartite"] and f["matching_valid"] for f in facts)
        ),
    }


def enumerated_pair(k, gain_a, gain_b):
    """Two-colour sweep on one pair: consecutive-read independence and memorylessness."""
    states = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    t = np.zeros((4, 4))
    for i, (_, sb) in enumerate(states):
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
    after_one = t[:, e == np.sign(k)].sum(axis=1)  # P(agree after one sweep | start)
    return {
        "k": k,
        "gain_a": gain_a,
        "gain_b": gain_b,
        "consecutive_independence_max": float(np.max(np.abs(joint - np.outer(marg, marg)))),
        "p_agree_minus_sigmoid_gain_b": abs(agree - sig(2 * gain_b * abs(k))),
        "one_sweep_from_any_start_max": float(np.max(np.abs(after_one - agree))),
    }


def enumeration_checks():
    pairs = [
        enumerated_pair(k, ga, gb)
        for k, ga, gb in (
            (-1.0, 1.0, 1.0),
            (-0.3, 0.9, 1.1),
            (-2.0, 1.3, 0.7),
            (0.5, 1.0, 1.3),
            (-4.0, 1.0, 1.0),
        )
    ]
    worst = max(
        max(
            r["consecutive_independence_max"],
            r["p_agree_minus_sigmoid_gain_b"],
            r["one_sweep_from_any_start_max"],
        )
        for r in pairs
    )
    return {
        "evidence_class": "exact_reference (enumerated two-site chain)",
        "pairs": pairs,
        "max_abs": worst,
        "tolerance": 1e-12,
        "passed": bool(worst <= 1e-12),
    }


def run_checks(request):
    reference = reference_reproduction(request["reference_ids"])
    p16_checks = p16.reference_checks()
    return {
        "references": reference,
        "planar_16_offset_reference_checks": {
            "passed": p16_checks["passed"],
            "max_log_z_error": p16_checks["max_log_z_error"],
            "max_edge_error": p16_checks["max_edge_error"],
        },
        "patches": patch_checks(request["candidate_betas"]),
        "enumeration": enumeration_checks(),
    }


def checks_passed(checks, request):
    return {
        "reference_checks_passed": bool(
            checks["references"]["reproduced"] == len(request["reference_ids"])
            and checks["planar_16_offset_reference_checks"]["passed"]
        ),
        "patch_checks_passed": bool(checks["patches"]["passed"]),
        "enumeration_checks_passed": bool(checks["enumeration"]["passed"]),
    }


# ----------------------------------------------------------------------------
# Simple-test tables (closed form, plus the draw-averaged Monte Carlo counts)
# ----------------------------------------------------------------------------


def simple_tables(request):
    facts = [graph_facts(t) for t in request["simple_targets"]]
    sizes = sorted({f["n"] for f in facts})
    pairs = {s: min(f["max_matching_pairs"] for f in facts if f["n"] == s) for s in sizes}
    nedges = {s: max(f["edges"] for f in facts if f["n"] == s) for s in sizes}
    degree = {s: max(f["max_degree"] for f in facts if f["n"] == s) for s in sizes}
    degree_range = {
        s: [min(f["max_degree"] for f in facts if f["n"] == s), degree[s]] for s in sizes
    }

    stuck = {
        "reads_known_site": int(np.ceil(1 + np.log2(1 / ALPHA))),
        **{f"reads_scan_{s}": int(np.ceil(1 + np.log2(s / ALPHA))) for s in sizes},
        "programming_writes": 1,
        "state_writes": 0,
        "miss_probability": 0.0,
        "covers": ["ignores writes (reads a constant)", "accepts a write and never updates"],
    }

    q0 = float(sig(2 * REFERENCE_BETA * 1.0))
    missing = {
        "beta": REFERENCE_BETA,
        "abs_j": 1.0,
        "p_agree_ideal": q0,
        "reads_known_coupler": reads_needed(q0, 0.5),
    }
    for s in sizes:
        per = reads_needed(q0, 0.5, alpha=ALPHA / nedges[s])
        missing[f"scan_{s}"] = {
            "edges_bonferroni": nedges[s],
            "reads_per_matching": per,
            "matchings": degree_range[s],
            "reads_total": [per * degree_range[s][0], per * degree_range[s][1]],
            "programming_writes": degree_range[s],
        }

    betas = np.round(np.arange(0.5, 4.0001, 0.1), 2)
    rounding = []
    for bits in ROUNDING_BITS:
        err = {w: float(quantize_array(np.array([-w]), 1.0, bits)[0] + w) for w in WEIGHTS}
        row = {"bits": bits, "coupling_error": {str(w): err[w] for w in WEIGHTS}}
        if max(abs(v) for v in err.values()) == 0:
            row["defect"] = "none (exact on this codebook)"
            rounding.append(row)
            continue
        for s in sizes:
            best = None
            for w in WEIGHTS:
                if err[w] == 0:
                    continue
                for b in betas:
                    q0r, q1r = float(sig(2 * b * w)), float(sig(2 * b * (w - err[w])))
                    n_r = reads_needed(q0r, q1r, per_read=pairs[s])
                    if n_r and (best is None or n_r < best["reads"]):
                        best = {"weight": w, "beta": float(b), "reads": n_r}
            row[f"n{s}"] = {"pairs_per_read": pairs[s], "programming_writes": 1, **best}
        rounding.append(row)
    finest = {}
    for s in sizes:
        ok = [r["bits"] for r in rounding if f"n{s}" in r and r[f"n{s}"]["reads"] <= READ_LIMIT]
        finest[f"n{s}"] = max(ok) if ok else None

    xs_outlier = np.round(np.arange(0.8, 1.41, 0.05), 2)
    outlier = []
    for delta in OUTLIER_DELTAS:
        for scope, alpha in [("known_site", ALPHA)] + [(f"scan_{s}", ALPHA / s) for s in sizes]:
            best = None
            for x in xs_outlier:
                n = two_sided_reads(float(x), delta, alpha)
                if n and (best is None or n < best["reads"]):
                    best = {"beta_h": float(x), "reads": n}
            outlier.append({"delta_abs": delta, "scope": scope, "alpha": alpha, **best})
    smallest_outlier = {}
    for scope, alpha in [("known_site", ALPHA)] + [(f"scan_{s}", ALPHA / s) for s in sizes]:
        lo, hi = 1e-3, 0.3
        for _ in range(request["bisection_steps"]):
            mid = (lo + hi) / 2
            n = min(two_sided_reads(x, mid, alpha) or 10**9 for x in (1.0, 1.2, 1.4))
            lo, hi = (lo, mid) if n <= READ_LIMIT else (mid, hi)
        smallest_outlier[scope] = hi

    xs_pooled = np.round(np.arange(0.4, 2.01, 0.05), 2)
    pooled = [
        gain_spread_row(s, j, request["gain_mc_replicates"], xs_pooled)
        for j in GAIN_SPREADS
        for s in sizes
    ]
    smallest_pooled = [
        smallest_spread(
            s, request["gain_mc_replicates"], (1.0, 1.2, 1.4), request["bisection_steps"]
        )
        for s in sizes
    ]
    return {
        "evidence_class": {
            "counts": "exact_reference (closed-form binomial laws)",
            "pooled_gain_spread": "monte_carlo_on_exact_laws",
        },
        "graphs": facts,
        "stuck_pbit": stuck,
        "missing_coupling": missing,
        "rounding": rounding,
        "rounding_finest_bits_within_limit": finest,
        "outlier_two_sided": outlier,
        "outlier_smallest_delta_within_limit": smallest_outlier,
        "gain_spread_pooled": pooled,
        "gain_spread_smallest_within_limit": smallest_pooled,
    }


# ----------------------------------------------------------------------------
# The two-colour heat-bath kernel (NumPy float64, CPU)
# ----------------------------------------------------------------------------


def neighbour_arrays(n, edges, couplings):
    """``planar_16_offset_ferro.neighbour_arrays`` with float64 weights (same order).

    The archived couplings are float32 values, so on the collective targets the
    weights equal the imported float32 arrays cast to float64; float64 keeps the
    simple tests' programmed couplings (0.2, quantized values) exact.
    """
    adjacency = [[] for _ in range(n)]
    for (i, j), c in zip(edges, couplings, strict=True):
        adjacency[i].append((j, float(c)))
        adjacency[j].append((i, float(c)))
    width = max(1, max((len(a) for a in adjacency), default=1))
    nbr = np.zeros((n, width), np.int32)
    wts = np.zeros((n, width))
    for i, row in enumerate(adjacency):
        for k, (j, c) in enumerate(row):
            nbr[i, k], wts[i, k] = j, c
    return nbr, wts


def make_array(n, positions, edges, couplings, gains=None, bias=None, beta=1.0):
    """Arrays for the kernel: padded neighbours (edge-list order), colours, coefficients."""
    nbr, wts = neighbour_arrays(n, edges, couplings)
    colour = (np.asarray(positions).sum(axis=1) % 2) == 0
    beta_site = beta * (np.ones(n) if gains is None else np.asarray(gains, dtype=float))
    return {
        "n": n,
        "nbr": nbr,
        "wts": wts,
        "a": np.flatnonzero(colour),
        "b": np.flatnonzero(~colour),
        "coef": -2.0 * beta_site,
        "bias": None if bias is None else np.asarray(bias, dtype=float),
        "ei": np.array([e[0] for e in edges], dtype=np.int64),
        "ej": np.array([e[1] for e in edges], dtype=np.int64),
    }


def half_sweep(state, array, idx, uniforms, coef=None):
    """Heat-bath update of one colour: P(s_i = +1) = sigmoid(2 beta_i (field_i + bias_i))."""
    field = np.einsum("ckw,kw->ck", state[:, array["nbr"][idx]], array["wts"][idx])
    if array["bias"] is not None:
        field = field + array["bias"][idx]
    c = array["coef"][idx] if coef is None else coef[:, idx]
    p_up = 1.0 / (1.0 + np.exp(c * field))
    state[:, idx] = np.where(uniforms < p_up, 1.0, -1.0)


def simulate(array, rngs, burn, reads, frozen=None, coef=None):
    """Chains from a random start; one generator per chain, n uniforms per sweep.

    Returns per-chain sums over the ``reads`` reads after ``burn`` sweeps of
    every edge product (``edge_sums``) and of every site value (``site_sums``),
    as float64 holding exact integers. ``frozen`` sites keep their start value.
    ``coef`` (chains, n) overrides the per-site -2 beta_i (gain draws per chain).
    """
    n = array["n"]
    chains = len(rngs)
    state = np.where(np.stack([rng.random(n) for rng in rngs]) < 0.5, 1.0, -1.0)
    edge_sums = np.zeros((chains, len(array["ei"])))
    site_sums = np.zeros((chains, n))
    keep = None if frozen is None else state[:, frozen].copy()
    total = burn + reads
    for start in range(0, total, UNIFORM_BLOCK):
        block = min(UNIFORM_BLOCK, total - start)
        u = np.stack([rng.random((block, n)) for rng in rngs])
        for t in range(block):
            for idx in (array["a"], array["b"]):
                if idx.size:
                    half_sweep(state, array, idx, u[:, t, idx], coef)
            if keep is not None:
                state[:, frozen] = keep
            if start + t >= burn:
                if edge_sums.shape[1]:
                    edge_sums += state[:, array["ei"]] * state[:, array["ej"]]
                site_sums += state
    return edge_sums, site_sums


# ----------------------------------------------------------------------------
# Calibration chains and the temperature rule
# ----------------------------------------------------------------------------


def calibration_unit(request, tid, beta, window):
    target = target_by_id(request, tid)
    edges = p16.edge_tuples(target)
    array = make_array(
        target["n"], p16.target_positions(target), edges, target["couplings"], beta=beta
    )
    rngs = [
        np.random.default_rng([SEED_CAL, ARM_CAL, target_code(target), beta_code(beta), window, c])
        for c in range(request["calibration_chains"])
    ]
    edge_sums, _ = simulate(array, rngs, request["burn_in"], window)
    return {"edge_sums": edge_sums.astype(np.int32)}


def calibration_summary(target, beta, window, edge_sums, c0, c1):
    """Scatter ratios, bias z-scores and scan lengths for one calibration set."""
    j0 = np.asarray(target["couplings"], dtype=float)
    sums = edge_sums.astype(float)
    means = sums / window
    chains = means.shape[0]
    var = 1 - c0**2
    r = means.var(axis=0, ddof=1) * window / var
    zb = (means - c0) / np.sqrt(var / window)
    zbar = zb.mean(axis=1)
    energy = means @ j0 / target["n"]
    exact_energy = float(j0 @ c0 / target["n"])
    q = t_quantile(len(j0))
    n_ind = independent_reads(c0, c1, np.sign(j0), q)
    nstar = np.ceil(n_ind * r)
    return {
        "r": r,
        "n_ind": n_ind,
        "nstar": nstar,
        "summary": {
            "target": target["id"],
            "beta": beta,
            "window": window,
            "chains": chains,
            "t_quantile": q,
            "scan_independent_reads": int(np.ceil(n_ind.max())),
            "n_scan": int(nstar.max()),
            "scan_sweeps": int(BURN + nstar.max()),
            "r_median": float(np.median(r)),
            "r_max": float(r.max()),
            "energy_bias_z": float(
                (energy.mean() - exact_energy) / (energy.std(ddof=1) / np.sqrt(chains))
            ),
            "edge_bias_mean_z": float(zb.mean() / (zbar.std(ddof=1) / np.sqrt(chains))),
            "sd_of_chain_mean_standardized_bias": float(zbar.std(ddof=1)),
        },
    }


def select_beta(request, summaries):
    """The protocol's rule: admitted candidates, then the smallest largest regret."""
    best = {}
    for s in summaries:
        best[s["target"]] = min(best.get(s["target"], np.inf), s["scan_sweeps"])
    rows = []
    for beta in request["candidate_betas"]:
        cand = [s for s in summaries if s["beta"] == beta]
        rows.append(
            {
                "beta": beta,
                "targets": len(cand),
                "admitted": bool(
                    len(cand) == len(request["targets"])
                    and all(
                        abs(s["energy_bias_z"]) <= request["bias_z_limit"]
                        and abs(s["edge_bias_mean_z"]) <= request["bias_z_limit"]
                        for s in cand
                    )
                ),
                "max_abs_bias_z": max(
                    max(abs(s["energy_bias_z"]), abs(s["edge_bias_mean_z"])) for s in cand
                ),
                "max_regret": max(s["scan_sweeps"] / best[s["target"]] for s in cand),
            }
        )
    admitted = [r for r in rows if r["admitted"]]
    chosen = min(admitted, key=lambda r: (r["max_regret"], r["beta"]))["beta"] if admitted else None
    return {"rows": rows, "chosen_beta": chosen}


# ----------------------------------------------------------------------------
# Collective sampled checks
# ----------------------------------------------------------------------------


def collective_models(nstar):
    """Declared edges: cheapest, median and dearest by n*_e (stable order by edge index)."""
    order = np.argsort(nstar, kind="stable")
    return {
        "cheapest": int(order[0]),
        "median": int(order[len(order) // 2]),
        "dearest": int(order[-1]),
    }


def collective_unit(request, tid, beta, model_edge, n_scan):
    target = target_by_id(request, tid)
    edges = p16.edge_tuples(target)
    couplings = np.asarray(target["couplings"], dtype=np.float32).copy()
    model_code = 0
    if model_edge is not None:
        couplings[model_edge] = 0.0  # the coupler reads 0 whatever is programmed
        model_code = model_edge + 1
    array = make_array(target["n"], p16.target_positions(target), edges, couplings, beta=beta)
    rngs = [
        np.random.default_rng(
            [SEED_TEST, ARM_COLLECTIVE, target_code(target), beta_code(beta), model_code, r]
        )
        for r in range(request["replicates"])
    ]
    edge_sums, _ = simulate(array, rngs, request["burn_in"], n_scan)
    return {"edge_sums": edge_sums.astype(np.int32)}


def scan_statistics(edge_sums, c0, r, sign, n_scan):
    se = np.sqrt(r * (1 - c0**2) / n_scan)
    return sign * (c0 - edge_sums.astype(float) / n_scan) / se


def attribution(request, target, z, t_alpha):
    """Coherent and heterogeneous bias of the ideal scan statistic, with bootstrap bounds."""
    reps, edges = z.shape

    def parts(weights):
        b = weights @ z / reps
        second = weights @ (z * z) / reps
        var_b = (second - b**2) * reps / (reps - 1) / reps
        kappa = b.mean(axis=-1)
        rho2 = ((b - kappa[..., None]) ** 2).mean(axis=-1) - var_b.mean(axis=-1)
        return kappa, rho2

    kappa, rho2 = parts(np.ones((1, reps)))
    rng = np.random.default_rng([SEED_TEST, ARM_BOOT, target_code(target)])
    weights = np.stack(
        [
            np.bincount(rng.integers(0, reps, reps), minlength=reps)
            for _ in range(request["bootstrap_resamples"])
        ]
    ).astype(float)
    kb, rb = parts(weights)
    level = 1 - request["attribution_level"]
    kappa_u = float(np.quantile(kb, level))
    rho_u = float(np.sqrt(max(float(np.quantile(rb, level)), 0.0)))
    rho_hat = float(np.sqrt(max(float(rho2[0]), 0.0)))

    def bound(k, rho):
        return float(min(1.0, edges * stats.t.sf((t_alpha - k) / np.sqrt(1 + rho**2), DF)))

    bound_upper = bound(kappa_u, rho_u)
    bound_point = bound(float(kappa[0]), rho_hat)
    limit = request["attribution_limit"]
    attributable = bound_upper <= limit
    return {
        "kappa_hat": float(kappa[0]),
        "rho2_hat": float(rho2[0]),
        "kappa_upper": kappa_u,
        "rho_upper": rho_u,
        "union_bound_at_upper": bound_upper,
        "union_bound_at_point": bound_point,
        "limit": limit,
        "attributable": bool(attributable),
        "failure": None if attributable else ("bias" if bound_point > limit else "precision"),
        "bootstrap_resamples": request["bootstrap_resamples"],
        "level": request["attribution_level"],
    }


# ----------------------------------------------------------------------------
# Simple sampled checks
# ----------------------------------------------------------------------------


def simple_check_plan(request, tables):
    """The 14 simple checks: arrays, counts, thresholds and defect models."""
    big = target_by_id({"targets": request["simple_targets"]}, request["simple_array_target"])
    sites = big["n"]
    pairs_all, _ = matching(big)
    size_key = f"n{sites}"
    stuck = tables["stuck_pbit"]
    plans = []
    plans.append(
        {
            "name": "stuck_known_site",
            "kind": "stuck",
            "sites": 1,
            "reads": stuck["reads_known_site"],
            "alpha": ALPHA,
            "stuck_site": 0,
        }
    )
    plans.append(
        {
            "name": f"stuck_scan_{sites}",
            "kind": "stuck",
            "sites": sites,
            "reads": stuck[f"reads_scan_{sites}"],
            "alpha": ALPHA / sites,
            "stuck_site": sites // 2 + 5,
        }
    )
    for row in tables["gain_spread_pooled"]:
        if row["sites"] == sites:
            plans.append(
                {
                    "name": f"gain_spread_{row['spread_j']}_{sites}",
                    "kind": "spread",
                    "sites": sites,
                    "reads": row["reads"],
                    "beta_h": row["beta_h"],
                    "spread_j": row["spread_j"],
                    "threshold": row["threshold"],
                }
            )
    for row in tables["outlier_two_sided"]:
        if row["scope"] != "known_site":
            continue
        for sgn in (1, -1):
            delta = sgn * row["delta_abs"]
            plans.append(
                {
                    "name": f"outlier_known_{delta:+.1f}",
                    "kind": "outlier",
                    "sites": 1,
                    "reads": row["reads"],
                    "beta_h": row["beta_h"],
                    "gain": 1 + delta,
                    "bounds": list(
                        two_sided_bounds(row["reads"], float(sig(2 * row["beta_h"])), ALPHA)
                    ),
                }
            )
    miss = tables["missing_coupling"]
    q0 = miss["p_agree_ideal"]
    plans.append(
        {
            "name": "missing_known_coupler",
            "kind": "pairs",
            "pairs": [pairs_all[len(pairs_all) // 2]],
            "reads": miss["reads_known_coupler"],
            "beta": REFERENCE_BETA,
            "ideal_abs_j": 1.0,
            "defect": {"type": "missing", "pair": 0},
            "pooled": False,
            **count_test(q0, 0.5, miss["reads_known_coupler"]),
        }
    )
    scan = miss[f"scan_{sites}"]
    plans.append(
        {
            "name": f"missing_scan_matching_{sites}",
            "kind": "pairs",
            "pairs": pairs_all,
            "reads": scan["reads_per_matching"],
            "beta": REFERENCE_BETA,
            "ideal_abs_j": 1.0,
            "defect": {"type": "missing", "pair": len(pairs_all) // 2},
            "pooled": False,
            **count_test(q0, 0.5, scan["reads_per_matching"], ALPHA / scan["edges_bonferroni"]),
        }
    )
    for row in tables["rounding"]:
        if row["bits"] not in CHECK_BITS:
            continue
        best = row[size_key]
        w, b, n = best["weight"], best["beta"], best["reads"]
        jq = float(quantize_array(np.array([-w]), 1.0, row["bits"])[0])
        npairs = best["pairs_per_read"]
        plans.append(
            {
                "name": f"rounding_{row['bits']}bit_{sites}",
                "kind": "pairs",
                "pairs": pairs_all[:npairs],
                "reads": n,
                "beta": b,
                "ideal_abs_j": w,
                "defect": {"type": "rounded", "coupling": jq},
                "pooled": True,
                **count_test(float(sig(2 * b * w)), float(sig(2 * b * abs(jq))), n * npairs),
            }
        )
    for i, plan in enumerate(plans):
        plan["index"] = i
    return plans


def _site_positions(count):
    """Isolated sites on a row of a lattice patch; colours alternate (all are free sites)."""
    return np.array([(i, 0) for i in range(count)], dtype=np.int64)


def simple_unit(request, plan, model):
    reps = request["replicates"]
    rngs = [
        np.random.default_rng([SEED_TEST, ARM_SIMPLE, plan["index"], model, r]) for r in range(reps)
    ]
    defect = model == 1
    kind = plan["kind"]
    if kind == "stuck":
        n = plan["sites"]
        array = make_array(n, _site_positions(n), [], [])
        _, site_sums = simulate(
            array, rngs, 0, plan["reads"], frozen=[plan["stuck_site"]] if defect else None
        )
        return {"site_sums": site_sums.astype(np.int32)}
    if kind == "outlier":
        x = plan["beta_h"]
        array = make_array(
            1,
            _site_positions(1),
            [],
            [],
            gains=[plan["gain"] if defect else 1.0],
            bias=[SITE_BIAS],
            beta=2 * x,
        )
        _, site_sums = simulate(array, rngs, 0, plan["reads"])
        return {"site_sums": site_sums.astype(np.int32)}
    if kind == "spread":
        n, x = plan["sites"], plan["beta_h"]
        array = make_array(n, _site_positions(n), [], [], bias=np.full(n, SITE_BIAS), beta=2 * x)
        coef = None
        if defect:
            gains = gain_draws(n, plan["spread_j"], request["gain_mc_replicates"], reps)
            coef = -2.0 * (2 * x * gains)
        _, site_sums = simulate(array, rngs, 0, plan["reads"], coef=coef)
        return {"site_sums": site_sums.astype(np.int32)}
    # isolated pairs of a matching, all other couplers off
    sites = sorted({s for pair in plan["pairs"] for s in pair})
    local = {s: i for i, s in enumerate(sites)}
    big = target_by_id({"targets": request["simple_targets"]}, request["simple_array_target"])
    positions = p16.target_positions(big)[sites]
    edges = [tuple(sorted((local[a], local[b]))) for a, b in plan["pairs"]]
    couplings = np.full(len(edges), -plan["ideal_abs_j"])
    if defect:
        if plan["defect"]["type"] == "missing":
            couplings[plan["defect"]["pair"]] = 0.0
        else:
            couplings[:] = plan["defect"]["coupling"]
    array = make_array(len(sites), positions, edges, couplings, beta=plan["beta"])
    edge_sums, _ = simulate(array, rngs, 0, plan["reads"])
    return {"edge_sums": edge_sums.astype(np.int32)}


def simple_rejections(plan, arrays):
    """Per-replicate rejection flags and the tested variable's scatter ratio."""
    n = plan["reads"]
    kind = plan["kind"]
    if kind == "stuck":
        sums = arrays["site_sums"].astype(float)
        reject = np.any(np.abs(sums) == n, axis=1)
        values = sums
    elif kind == "outlier":
        k = (arrays["site_sums"][:, 0].astype(float) + n) / 2
        a, b = plan["bounds"]
        reject = (k <= a) | (k >= b)
        values = arrays["site_sums"].astype(float)
    elif kind == "spread":
        k = (arrays["site_sums"].astype(float) + n) / 2
        reject = x2_stat(k, n) > plan["threshold"]
        values = arrays["site_sums"].astype(float)
    else:
        agree = (n - arrays["edge_sums"].astype(float)) / 2  # couplings are negative
        counts = agree if plan["count"] == "agree" else n - agree
        if plan["pooled"]:
            reject = counts.sum(axis=1) >= plan["reject_at_or_above"]
        else:
            reject = np.any(counts >= plan["reject_at_or_above"], axis=1)
        values = arrays["edge_sums"].astype(float)
    pooled = values.sum(axis=1) / (n * values.shape[1])
    mean = float(pooled.mean())
    scatter = float(pooled.var(ddof=1) * n * values.shape[1] / max(1 - mean**2, 1e-300))
    return reject, scatter


# ----------------------------------------------------------------------------
# Request
# ----------------------------------------------------------------------------


def pass_thresholds(replicates, checks):
    a = ALPHA / (2 * checks)
    max_ideal = int(stats.binom.isf(a, replicates, 0.05))
    min_defect = int(stats.binom.ppf(a, replicates, 0.95))
    while stats.binom.cdf(min_defect - 1, replicates, 0.95) > a:
        min_defect -= 1
    return max_ideal, min_defect


def make_request(**overrides):
    request = {
        "schema": "z1_acceptance_kit.request.v1",
        "protocol": PROTOCOL,
        "protocol_commit": "dc32142a1bed288adaedab3f84ca0071dd1bee4b",
        "targets": greedy_targets(),
        "simple_targets": greedy_targets(),
        "simple_array_target": "L32-greedy-long-s4510",
        "reference_ids": [t["id"] for t in p16.study_targets()],
        "candidate_betas": list(CANDIDATES),
        "burn_in": BURN,
        "calibration_chains": 128,
        "calibration_window": 8192,
        "recalibration_window": 16384,
        "bias_z_limit": 3.5,
        "direct_check_edges": 20,
        "replicates": 1000,
        "simple_checks": 14,
        "collective_checks_per_target": 3,
        "attribution_limit": 0.065,
        "attribution_level": 0.025 / 9,
        "bootstrap_resamples": 2000,
        "gain_mc_replicates": 20000,
        "bisection_steps": 20,
        "read_limit": READ_LIMIT,
        "false_alarm": ALPHA,
        "miss": MISS,
        "t_degrees_of_freedom": DF,
        "seeds": {
            "calibration_root": SEED_CAL,
            "gain_monte_carlo_root": SEED_MC,
            "tested_root": SEED_TEST,
            "draw_roots": {str(k): v for k, v in DRAW_ROOTS.items()},
            "probe_roots_excluded": list(PROBE_SEEDS),
        },
        "uniform_block": UNIFORM_BLOCK,
    }
    request.update(overrides)
    m = request["simple_checks"] + request["collective_checks_per_target"] * len(request["targets"])
    request["sampled_checks"] = m
    request["max_ideal_rejections"], request["min_defect_rejections"] = pass_thresholds(
        request["replicates"], m
    )
    request["sources"] = {name: base.sha(ROOT / name) for name in SOURCES}
    return request


def validate_request(request):
    for target in request["targets"] + request["simple_targets"]:
        if p16.build_target(target["size"], target["graph"], target["seed"]) != target:
            raise ValueError(f"target does not rebuild from its seed: {target['id']}")
    roots = {
        request["seeds"]["calibration_root"],
        request["seeds"]["gain_monte_carlo_root"],
        request["seeds"]["tested_root"],
    }
    if len(roots) != 3 or roots & set(request["seeds"]["probe_roots_excluded"]):
        raise ValueError("seed roots must be distinct and fresh")
    m = request["simple_checks"] + request["collective_checks_per_target"] * len(request["targets"])
    if request["sampled_checks"] != m:
        raise ValueError("sampled check count mismatch")
    if (request["max_ideal_rejections"], request["min_defect_rejections"]) != pass_thresholds(
        request["replicates"], m
    ):
        raise ValueError("pass thresholds do not follow from replicates and checks")


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


def _job(kind, request, args):
    started = time.perf_counter()
    if kind == "exact":
        arrays = exact_unit(request, *args)
    elif kind == "calibration":
        arrays = calibration_unit(request, *args)
    elif kind == "collective":
        arrays = collective_unit(request, *args)
    elif kind == "simple":
        arrays = simple_unit(request, *args)
    else:
        raise ValueError(kind)
    return arrays, {"run_seconds": time.perf_counter() - started}


def _pool_job(name, kind, request, args):
    arrays, timing = _job(kind, request, args)
    return name, arrays, timing


def run_jobs(runner, request, jobs, workers, budget):
    """Run (name, kind, args) jobs not yet verified; save each unit as it completes."""
    done = runner.verified_units()
    pending = [j for j in jobs if j[0] not in done]
    if not pending:
        return
    runner.log(
        f"{len(pending)} unit(s) to run ({len(jobs) - len(pending)} verified), {workers} worker(s)"
    )

    def record(name, arrays, timing):
        runner.save_unit(name, arrays, timing)
        budget["completed"] += 1
        runner.status(last_unit=name, units_done=len(runner.index()))
        runner.log(f"unit {name}: {timing['run_seconds']:.1f} s")
        if budget["stop_after"] is not None and budget["completed"] >= budget["stop_after"]:
            raise KeyboardInterrupt("stop_after reached (interruption test)")

    if workers <= 1:
        for name, kind, args in pending:
            record(name, *_job(kind, request, args))
        return
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=context) as pool:
        futures = [
            pool.submit(_pool_job, name, kind, request, args) for name, kind, args in pending
        ]
        try:
            for future in concurrent.futures.as_completed(futures):
                record(*future.result())
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def exact_name(tid, beta):
    return f"exact__{tid}__b{beta_code(beta)}"


def calibration_name(tid, beta, window):
    return f"calibration__{tid}__b{beta_code(beta)}__w{window}"


def collective_name(tid, label):
    return f"collective__{tid}__{label}"


def simple_name(plan, model):
    return f"simple__{plan['index']:02d}_{plan['name']}__{'defect' if model else 'ideal'}"


def load(runner, name):
    return p16.read_npz(runner.units / f"{name}.npz")


def _copy_sources(out, request):
    p16._copy_sources(out, request)


def run_study(out, requested=None, resume=False, workers=DEFAULT_WORKERS, stop_after=None):
    out = Path(out)
    p16.require_single_thread_blas()
    runner = p16.Runner(out)
    if resume:
        if not (out / "request.json").exists():
            raise FileNotFoundError("nothing to resume: request.json is missing")
        if (out / "completion.json").exists():
            raise FileExistsError("study already complete")
        request = json.loads((out / "request.json").read_text())
        guard = json.loads((out / "checkpoint-guard.json").read_text())
        if guard["request_digest"] != canonical_sha256(request):
            raise ValueError("checkpoint guard mismatch: request changed")
        expected = make_request() if requested is None else requested
        if request != expected:
            raise ValueError("stored request differs from this runner's request")
        validate_request(request)
        _copy_sources(out, request)
        attempt = json.loads((out / "run-status.json").read_text()).get("attempt", 0) + 1
        runner.log(f"resume attempt {attempt}: request and sources re-authenticated")
    else:
        if out.exists():
            raise FileExistsError("use a fresh output directory, or --resume")
        request = make_request() if requested is None else requested
        validate_request(request)
        out.mkdir(parents=True)
        runner.units.mkdir()
        base.write(out / "request.json", request)
        base.write(out / "checkpoint-guard.json", {"request_digest": canonical_sha256(request)})
        _copy_sources(out, request)
        attempt = 1
        runner.log("fresh run started")
    runner.status(phase="checks", attempt=attempt, workers=workers)
    attempts_path = out / "attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempts.append(
        {
            "attempt": attempt,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "workers": workers,
            "provenance": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        }
    )
    base.write(attempts_path, attempts)
    started = time.perf_counter()
    budget = {"completed": 0, "stop_after": stop_after}
    try:
        stage_checks(out, request, runner)
        runner.status(phase="exact")
        jobs = [
            (exact_name(t["id"], b), "exact", (t["id"], b))
            for t in request["targets"]
            for b in request["candidate_betas"]
        ]
        jobs.sort(key=lambda j: -target_by_id(request, j[2][0])["n"])
        run_jobs(runner, request, jobs, min(workers, EXACT_WORKERS), budget)
        exact_ok = [exact_summary(request, j[2][0], j[2][1], load(runner, j[0])) for j in jobs]
        if not all(
            e["rank2_check_passed"] and e["finite_difference_check_passed"] for e in exact_ok
        ):
            runner.log("STOP: a rank-2 or finite-difference check failed")
            raise StudyStop("rank-2 or finite-difference check failed; stopped before sampling")
        runner.log("exact tables complete; rank-2 and finite-difference checks passed")
        runner.status(phase="simple_tables")
        tables_path = out / "simple-tables.json"
        if not tables_path.exists():
            before = time.perf_counter()
            base.write(tables_path, simple_tables(request))
            runner.log(f"simple-test tables written ({time.perf_counter() - before:.0f} s)")
        tables = json.loads(tables_path.read_text())
        runner.status(phase="calibration")
        window = request["calibration_window"]
        jobs = [
            (calibration_name(t["id"], b, window), "calibration", (t["id"], b, window))
            for t in request["targets"]
            for b in request["candidate_betas"]
        ]
        jobs.sort(key=lambda j: -target_by_id(request, j[2][0])["n"])
        run_jobs(runner, request, jobs, workers, budget)
        plan = scan_plan(request, runner)
        recal = [
            (
                calibration_name(tid, plan["chosen_beta"], request["recalibration_window"]),
                "calibration",
                (tid, plan["chosen_beta"], request["recalibration_window"]),
            )
            for tid in plan["recalibrate"]
        ]
        if recal:
            runner.log(
                f"recalibrating {len(recal)} target(s) at {request['recalibration_window']} reads"
            )
            run_jobs(runner, request, recal, workers, budget)
            plan = scan_plan(request, runner)
        runner.log(
            f"chosen beta {plan['chosen_beta']}; n_scan "
            + ", ".join(f"{k}={v['n_scan']}" for k, v in plan["targets"].items())
        )
        runner.status(phase="sampling", chosen_beta=plan["chosen_beta"])
        jobs = []
        for tid, entry in plan["targets"].items():
            jobs.append(
                (
                    collective_name(tid, "ideal"),
                    "collective",
                    (tid, plan["chosen_beta"], None, entry["n_scan"]),
                )
            )
            for label, edge in entry["edges"].items():
                jobs.append(
                    (
                        collective_name(tid, label),
                        "collective",
                        (tid, plan["chosen_beta"], edge, entry["n_scan"]),
                    )
                )
        jobs.sort(key=lambda j: -target_by_id(request, j[2][0])["n"] * j[2][3])
        for p in simple_check_plan(request, tables):
            for model in (0, 1):
                jobs.append((simple_name(p, model), "simple", (p, model)))
        run_jobs(runner, request, jobs, workers, budget)
        runner.status(phase="assembling")
        assemble(out, request, runner, time.perf_counter() - started)
    except BaseException as error:
        runner.status(
            phase="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        runner.log(f"stopped: {error!r}")
        raise
    runner.status(phase="replaying")
    completion = replay(out)
    runner.status(phase="complete", error=None, traceback=None)
    runner.log(f"completion written after replay ({completion['replay_seconds']:.1f} s replay)")
    return completion


def stage_checks(out, request, runner):
    path = out / "checks.json"
    if not path.exists():
        before = time.perf_counter()
        base.write(path, run_checks(request))
        runner.log(f"exact and enumeration checks computed ({time.perf_counter() - before:.0f} s)")
    checks = json.loads(path.read_text())
    flags = checks_passed(checks, request)
    if not all(flags.values()):
        runner.log(f"STOP: a check failed: {flags}")
        raise StudyStop("reference, patch or enumeration check failed; stopped before sampling")
    runner.log("reference, patch and enumeration checks passed")


def scan_plan(request, runner, loader=None):
    """Calibration summaries, the chosen beta, recalibration and the declared edges."""
    loader = loader or (lambda name: load(runner, name))
    window = request["calibration_window"]
    summaries, details = [], {}
    for t in request["targets"]:
        for b in request["candidate_betas"]:
            ex = loader(exact_name(t["id"], b))
            cal = calibration_summary(
                t,
                b,
                window,
                loader(calibration_name(t["id"], b, window))["edge_sums"],
                ex["c0"],
                ex["c1"],
            )
            summaries.append(cal["summary"])
            details[(t["id"], b)] = cal
    selection = select_beta(request, summaries)
    beta = selection["chosen_beta"]
    if beta is None:
        raise StudyStop("no candidate beta passed the calibration bias check")
    plan = {
        "selection": selection,
        "calibration": summaries,
        "chosen_beta": beta,
        "recalibrate": [],
        "recalibration": [],
        "targets": {},
    }
    for t in request["targets"]:
        cal = details[(t["id"], beta)]
        if cal["summary"]["n_scan"] > window:
            plan["recalibrate"].append(t["id"])
            name = calibration_name(t["id"], beta, request["recalibration_window"])
            try:
                arrays = loader(name)
            except (FileNotFoundError, KeyError):
                arrays = None
            if arrays is None:
                continue
            ex = loader(exact_name(t["id"], beta))
            cal = calibration_summary(
                t, beta, request["recalibration_window"], arrays["edge_sums"], ex["c0"], ex["c1"]
            )
            plan["recalibration"].append(cal["summary"])
        n_scan = int(cal["nstar"].max())
        plan["targets"][t["id"]] = {
            "n_scan": n_scan,
            "window": cal["summary"]["window"],
            "fits_read_limit": bool(n_scan <= request["read_limit"]),
            "edges": collective_models(cal["nstar"]),
            "r": cal["r"],
            "nstar": cal["nstar"],
        }
    if any(t not in [s["target"] for s in plan["recalibration"]] for t in plan["recalibrate"]):
        plan["recalibration_pending"] = True
    return plan


def collective_results(request, plan, loader):
    beta = plan["chosen_beta"]
    rows, stats_store = [], {}
    for t in request["targets"]:
        tid = t["id"]
        entry = plan["targets"][tid]
        ex = loader(exact_name(tid, beta))
        c0, c1 = ex["c0"], ex["c1"]
        j0 = np.asarray(t["couplings"], dtype=float)
        sign = np.sign(j0)
        edges = len(j0)
        t_alpha = t_quantile(edges)
        n_scan = entry["n_scan"]
        z_ideal = scan_statistics(
            loader(collective_name(tid, "ideal"))["edge_sums"], c0, entry["r"], sign, n_scan
        )
        ideal_reject = z_ideal.max(axis=1) > t_alpha
        stats_store[f"{tid}__ideal__max_z"] = z_ideal.max(axis=1)
        stats_store[f"{tid}__ideal__argmax_edge"] = z_ideal.argmax(axis=1).astype(np.int32)
        stats_store[f"{tid}__ideal__kappa_per_replicate"] = z_ideal.mean(axis=1)
        attr = attribution(request, t, z_ideal, t_alpha)
        ideal_means = loader(collective_name(tid, "ideal"))["edge_sums"].astype(float) / n_scan
        checks = []
        for label, edge in entry["edges"].items():
            z = scan_statistics(
                loader(collective_name(tid, label))["edge_sums"], c0, entry["r"], sign, n_scan
            )
            reject = z.max(axis=1) > t_alpha
            own = z[:, edge] > t_alpha
            stats_store[f"{tid}__{label}__max_z"] = z.max(axis=1)
            stats_store[f"{tid}__{label}__own_z"] = z[:, edge]
            stats_store[f"{tid}__{label}__ideal_own_z"] = z_ideal[:, edge]
            mix = ideal_means[:, edge]
            checks.append(
                {
                    "check": f"collective_{tid}_{label}",
                    "target": tid,
                    "edge_label": label,
                    "edge": edge,
                    "coupling": float(j0[edge]),
                    "n_star": int(entry["nstar"][edge]),
                    "n_scan": n_scan,
                    "c0": float(c0[edge]),
                    "c1": float(c1[edge]),
                    "device_error_exact_shift": float(c1[edge] - c0[edge]),
                    "mixing_error_ideal_mean_minus_exact": float(mix.mean() - c0[edge]),
                    "mixing_error_se": float(mix.std(ddof=1) / np.sqrt(len(mix))),
                    "ideal_rejections": int(ideal_reject.sum()),
                    "defect_rejections": int(reject.sum()),
                    "defect_rejections_own_edge": int(own.sum()),
                    "replicates": int(len(reject)),
                }
            )
        rows.append(
            {
                "target": tid,
                "n": t["n"],
                "edges": edges,
                "t_quantile": t_alpha,
                "n_scan": n_scan,
                "fits_read_limit": entry["fits_read_limit"],
                "calibration_window": entry["window"],
                "scan_independent_reads": int(
                    np.ceil(independent_reads(c0, c1, sign, t_alpha).max())
                ),
                "r_median": float(np.median(entry["r"])),
                "r_max": float(entry["r"].max()),
                "family_wise_false_alarm": float(ideal_reject.mean()),
                "attribution": attr,
                "checks": checks,
            }
        )
    return rows, stats_store


def simple_results(request, tables, loader):
    rows, store = [], {}
    for plan in simple_check_plan(request, tables):
        ideal, s_ideal = simple_rejections(plan, loader(simple_name(plan, 0)))
        defect, s_defect = simple_rejections(plan, loader(simple_name(plan, 1)))
        store[f"simple__{plan['index']:02d}__ideal_reject"] = ideal.astype(np.uint8)
        store[f"simple__{plan['index']:02d}__defect_reject"] = defect.astype(np.uint8)
        rows.append(
            {
                "check": plan["name"],
                "kind": plan["kind"],
                "reads": plan["reads"],
                "ideal_rejections": int(ideal.sum()),
                "defect_rejections": int(defect.sum()),
                "replicates": int(len(ideal)),
                "ideal_scatter_ratio": s_ideal,
                "defect_scatter_ratio": s_defect,
                "programming_writes": 1,
                "state_writes": 0,
            }
        )
    return rows, store


def decide(request, simple_rows, collective_rows):
    lo, hi = request["max_ideal_rejections"], request["min_defect_rejections"]
    checks = []
    for row in simple_rows:
        checks.append(
            {
                "check": row["check"],
                "family": "simple",
                "ideal_rejections": row["ideal_rejections"],
                "defect_rejections": row["defect_rejections"],
                "passed": bool(row["ideal_rejections"] <= lo and row["defect_rejections"] >= hi),
                "counts": True,
            }
        )
    unattributable_sizes = set()
    for target in collective_rows:
        attributable = target["attribution"]["attributable"]
        if not attributable:
            unattributable_sizes.add(target["n"])
        for row in target["checks"]:
            checks.append(
                {
                    "check": row["check"],
                    "family": "collective",
                    "ideal_rejections": row["ideal_rejections"],
                    "defect_rejections": row["defect_rejections"],
                    "passed": bool(
                        row["ideal_rejections"] <= lo and row["defect_rejections"] >= hi
                    ),
                    "counts": bool(attributable),
                }
            )
    failed = [c for c in checks if c["counts"] and not c["passed"]]
    if unattributable_sizes:
        sizes = sorted(unattributable_sizes)
        verdict = {
            "status": "collective_not_attributable",
            "sentence": "The collective test is not attributable at size "
            + ", ".join(f"{int(np.sqrt(s))} ({s} spins)" for s in sizes)
            + ". The simple tests are unaffected.",
            "sizes": sizes,
        }
    elif failed:
        verdict = {
            "status": "holds_except_named",
            "sentence": "The kit holds except for named checks: "
            + "; ".join(
                f"{c['check']} (ideal {c['ideal_rejections']}, "
                f"defect {c['defect_rejections']} of {request['replicates']})"
                for c in failed
            )
            + ".",
        }
    else:
        verdict = {"status": "holds", "sentence": "The kit holds."}
    verdict["failed_checks"] = [c["check"] for c in failed]
    if unattributable_sizes and failed:
        verdict["also_failed_counting_checks"] = [c["check"] for c in failed]
    return checks, verdict


def assemble(out, request, runner, seconds):
    def loader(name):
        return load(runner, name)

    checks = json.loads((out / "checks.json").read_text())
    tables = json.loads((out / "simple-tables.json").read_text())
    record = build_record(request, checks, tables, loader)
    exact_store, cal_store = {}, {}
    for t in request["targets"]:
        for b in request["candidate_betas"]:
            ex = loader(exact_name(t["id"], b))
            exact_store[f"{t['id']}__b{beta_code(b)}__c0"] = ex["c0"]
            exact_store[f"{t['id']}__b{beta_code(b)}__c1"] = ex["c1"]
    plan = scan_plan(request, runner)
    for tid, entry in plan["targets"].items():
        cal_store[f"{tid}__r"] = entry["r"]
    p16.write_npz(out / "exact.npz", exact_store)
    p16.write_npz(out / "calibration.npz", cal_store)
    p16.write_npz(out / "statistics.npz", record.pop("_statistics"))
    record["exact_sha256"] = base.sha(out / "exact.npz")
    record["calibration_sha256"] = base.sha(out / "calibration.npz")
    record["statistics_sha256"] = base.sha(out / "statistics.npz")
    record["unit_timings_seconds"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "sha256"} for k, v in runner.index().items()
    }
    record["generation_seconds_this_attempt"] = seconds
    base.write(out / "results.json", record)
    base.write(
        out / "provenance.json",
        {
            "schema": "z1_acceptance_kit.provenance.v1",
            "request_digest": record["request_digest"],
            "attempts": json.loads((out / "attempts.json").read_text()),
            "runtime": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        },
    )
    runner.log("results assembled")


def build_record(request, checks, tables, loader):
    exact = [
        exact_summary(request, t["id"], b, loader(exact_name(t["id"], b)))
        for t in request["targets"]
        for b in request["candidate_betas"]
    ]
    plan = scan_plan(request, None, loader)
    collective, cstore = collective_results(request, plan, loader)
    simple, sstore = simple_results(request, tables, loader)
    decisions, verdict = decide(request, simple, collective)
    flags = checks_passed(checks, request)
    record = {
        "request_digest": canonical_sha256(request),
        "evidence_class": {
            "simple_test_counts": "exact_reference",
            "collective_means_shifts_variances_independent_reads": "exact_reference",
            "pooled_gain_spread_counts": "monte_carlo_on_exact_laws",
            "scatter_ratios_sampled_means_operating_characteristics": "software_simulation",
            "references_p0001": "exact_reference",
        },
        "hardware_claim": False,
        "checks": checks,
        "check_flags": flags,
        "exact_tables": exact,
        "missing_edges_exact": int(
            sum(len(t["edges"]) for t in request["targets"]) * len(request["candidate_betas"])
        ),
        "simple_tables": tables,
        "temperature": {
            "selection": plan["selection"],
            "calibration": plan["calibration"],
            "recalibrated_targets": plan["recalibrate"],
            "recalibration": plan["recalibration"],
            "chosen_beta": plan["chosen_beta"],
        },
        "collective": collective,
        "simple_checks": simple,
        "decisions": decisions,
        "pass_rule": {
            "replicates": request["replicates"],
            "checks": request["sampled_checks"],
            "max_ideal_rejections": request["max_ideal_rejections"],
            "min_defect_rejections": request["min_defect_rejections"],
        },
        "row_verdict": verdict,
        "_statistics": {**cstore, **sstore},
    }
    return record


# ----------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------


def compare(actual, expected, path="root"):
    """Bitwise with THERMO_ARCHIVE_HOST=1 (or same host); else floats at 2e-12 relative."""
    if ARCHIVE_HOST:
        if actual != expected:
            raise ValueError(f"value differs: {path}")
        return
    pis.check_equal(actual, expected, path)


def replay(out):
    """Full replay from the persisted units: no sweeps; exact tables recomputed."""
    out = Path(out)
    p16.require_single_thread_blas()
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    validate_request(request)
    if result["request_digest"] != canonical_sha256(request):
        raise ValueError("request digest mismatch")
    for name, digest in request["sources"].items():
        for path in (ROOT / name, out / "source" / name):
            if base.sha(path) != digest:
                raise ValueError(f"source changed: {name}")
    runner = p16.Runner(out)
    index = runner.index()
    for name, entry in index.items():
        if base.sha(runner.units / f"{name}.npz") != entry["sha256"]:
            raise ValueError(f"unit digest mismatch: {name}")
    for key, file in (
        ("exact_sha256", "exact.npz"),
        ("calibration_sha256", "calibration.npz"),
        ("statistics_sha256", "statistics.npz"),
    ):
        if base.sha(out / file) != result[key]:
            raise ValueError(f"{file} digest mismatch")
    checks = run_checks(request)
    compare(checks, result["checks"], "checks")
    recomputed = {}
    for t in request["targets"]:
        positions, edges = p16.target_positions(t), p16.edge_tuples(t)
        for b in request["candidate_betas"]:
            name = exact_name(t["id"], b)
            stored = load(runner, name)
            k0 = b * np.asarray(t["couplings"], dtype=float)
            _, c0 = p16.kac_ward(positions, edges, k0)
            c1, c0_alt = missing_edge_correlations(positions, edges, k0)
            for key, value in (("c0", c0), ("c1", c1), ("c0_rank2_algebra", c0_alt)):
                compare(value.tolist(), stored[key].tolist(), f"{name}/{key}")
            recomputed[name] = stored
    tables = simple_tables(request)
    compare(tables, result["simple_tables"], "simple_tables")

    def loader(name):
        return recomputed[name] if name in recomputed else load(runner, name)

    record = build_record(request, checks, tables, loader)
    statistics = record.pop("_statistics")
    stored_stats = p16.read_npz(out / "statistics.npz")
    if sorted(statistics) != sorted(stored_stats):
        raise ValueError("statistics members differ")
    for key, value in statistics.items():
        compare(value.tolist(), stored_stats[key].tolist(), f"statistics/{key}")
    for key in record:
        compare(record[key], result[key], key)
    completion = completion_record(request, result, time.perf_counter() - started)
    base.write(out / "completion.json", completion)
    return completion


def completion_record(request, result, seconds):
    flags = result["check_flags"]
    collective = result["collective"]
    exact_ok = all(
        e["rank2_check_passed"] and e["finite_difference_check_passed"]
        for e in result["exact_tables"]
    )
    return {
        "status": STATUS,
        "request_digest": result["request_digest"],
        "statistics_sha256": result["statistics_sha256"],
        "targets": len(request["targets"]),
        "references_reproduced": result["checks"]["references"]["reproduced"],
        "references_bitwise_equal": result["checks"]["references"]["bitwise_equal"],
        "chosen_beta": result["temperature"]["chosen_beta"],
        "candidate_betas": request["candidate_betas"],
        "scan_quantile": "student_t_127",
        "attribution_limit": request["attribution_limit"],
        "gain_spread_count": "draw_averaged_monte_carlo",
        "calibration_seed_root_disjoint": bool(
            request["seeds"]["calibration_root"]
            not in (request["seeds"]["tested_root"], request["seeds"]["gain_monte_carlo_root"])
        ),
        "simple_checks": len(result["simple_checks"]),
        "collective_checks": sum(len(t["checks"]) for t in collective),
        "missing_edges_exact": result["missing_edges_exact"],
        "n_scan": {t["target"]: t["n_scan"] for t in collective},
        "scans_exceeding_read_limit": [t["target"] for t in collective if not t["fits_read_limit"]],
        "attribution_checks_passed": bool(
            all(t["attribution"]["attributable"] for t in collective)
        ),
        "reference_checks_passed": flags["reference_checks_passed"],
        "rank2_checks_passed": bool(exact_ok),
        "patch_checks_passed": flags["patch_checks_passed"],
        "enumeration_checks_passed": flags["enumeration_checks_passed"],
        "checks_passed": sum(d["passed"] for d in result["decisions"]),
        "checks_failed": [d["check"] for d in result["decisions"] if not d["passed"]],
        "row_verdict": result["row_verdict"]["status"],
        "replayed": True,
        "replay_seconds": seconds,
        "evidence_class": result["evidence_class"],
        "hardware_claim": False,
    }


# ----------------------------------------------------------------------------
# Bounded archive and its light replay (CI)
# ----------------------------------------------------------------------------

ARCHIVE_MEMBERS = (
    "request.json",
    "checks.json",
    "simple-tables.json",
    "results.json",
    "exact.npz",
    "calibration.npz",
    "statistics.npz",
    "provenance.json",
    "completion.json",
)


def write_archive(out, destination):
    """Deterministic gzipped tar of the bounded record (no unit arrays)."""
    out = Path(out)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in ARCHIVE_MEMBERS:
            data = (out / name).read_bytes()
            info = tarfile.TarInfo(f"z1-acceptance-kit/{name}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))

    Path(destination).write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    return Path(destination).stat().st_size


def replay_archive(path):
    """Light replay of the bounded archive: digests, decisions, scan lengths, attribution."""
    started = time.perf_counter()
    members = {}
    with tarfile.open(path) as tar:
        for name in ARCHIVE_MEMBERS:
            members[name] = tar.extractfile(f"z1-acceptance-kit/{name}").read()
    request = json.loads(members["request.json"])
    result = json.loads(members["results.json"])
    completion = json.loads(members["completion.json"])
    if canonical_sha256(request) != result["request_digest"]:
        raise ValueError("request digest mismatch")

    for key, name in (
        ("exact_sha256", "exact.npz"),
        ("calibration_sha256", "calibration.npz"),
        ("statistics_sha256", "statistics.npz"),
    ):
        if hashlib.sha256(members[name]).hexdigest() != result[key]:
            raise ValueError(f"{name} digest mismatch")
    exact = dict(np.load(io.BytesIO(members["exact.npz"])))
    cal = dict(np.load(io.BytesIO(members["calibration.npz"])))
    st = dict(np.load(io.BytesIO(members["statistics.npz"])))
    beta = result["temperature"]["chosen_beta"]
    simple_rows = []
    plans = simple_check_plan(request, result["simple_tables"])
    for plan, row in zip(plans, result["simple_checks"], strict=True):
        ideal = int(st[f"simple__{plan['index']:02d}__ideal_reject"].sum())
        defect = int(st[f"simple__{plan['index']:02d}__defect_reject"].sum())
        if (ideal, defect) != (row["ideal_rejections"], row["defect_rejections"]):
            raise ValueError(f"simple rejections differ: {plan['name']}")
        simple_rows.append(row)
    for trow in result["collective"]:
        t = target_by_id(request, trow["target"])
        j0 = np.asarray(t["couplings"], dtype=float)
        c0 = exact[f"{t['id']}__b{beta_code(beta)}__c0"]
        c1 = exact[f"{t['id']}__b{beta_code(beta)}__c1"]
        r = cal[f"{t['id']}__r"]
        t_alpha = t_quantile(len(j0))
        nstar = np.ceil(independent_reads(c0, c1, np.sign(j0), t_alpha) * r)
        if int(nstar.max()) != trow["n_scan"]:
            raise ValueError(f"n_scan differs: {t['id']}")
        if (
            int((st[f"{t['id']}__ideal__max_z"] > t_alpha).sum())
            != trow["checks"][0]["ideal_rejections"]
        ):
            raise ValueError(f"ideal rejections differ: {t['id']}")
        for row in trow["checks"]:
            if (
                int((st[f"{t['id']}__{row['edge_label']}__max_z"] > t_alpha).sum())
                != row["defect_rejections"]
            ):
                raise ValueError(f"defect rejections differ: {row['check']}")
        a = trow["attribution"]
        bound = min(
            1.0,
            len(j0)
            * stats.t.sf((t_alpha - a["kappa_upper"]) / np.sqrt(1 + a["rho_upper"] ** 2), DF),
        )
        if (
            abs(bound - a["union_bound_at_upper"]) > 1e-12
            or (bound <= a["limit"]) != a["attributable"]
        ):
            raise ValueError(f"attribution differs: {t['id']}")
    decisions, verdict = decide(request, simple_rows, result["collective"])
    pis.check_equal(decisions, result["decisions"])
    pis.check_equal(verdict, result["row_verdict"])
    if completion["status"] != STATUS or completion["row_verdict"] != verdict["status"]:
        raise ValueError("completion record does not match")
    return {
        "archive_replayed": True,
        "decisions": len(decisions),
        "row_verdict": verdict["status"],
        "seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    parser.add_argument("--replay", action="store_true", help="full replay of a finished run")
    parser.add_argument("--replay-archive", help="light replay of a bounded archive (CI)")
    parser.add_argument("--write-archive", help="write the bounded archive of a finished run here")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--request", help="request JSON (tests only; production uses the default)")
    parser.add_argument("--stop-after", type=int, help="interrupt after N new units (tests only)")
    args = parser.parse_args()
    if args.replay_archive:
        print(json.dumps(replay_archive(args.replay_archive), indent=2))
        return
    if not args.output_dir:
        parser.error("--output-dir is required")
    if args.write_archive:
        print(write_archive(args.output_dir, args.write_archive))
        return
    if args.replay:
        print(json.dumps(replay(args.output_dir), indent=2))
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
    except StudyStop as error:
        print(f"stopped by protocol rule: {error}", flush=True)
        raise SystemExit(3) from None
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
