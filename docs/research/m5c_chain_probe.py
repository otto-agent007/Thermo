"""Exact conditional exploration of degree repair by chaining or pruning.

Run: OPENBLAS_NUM_THREADS=1 uv run python docs/research/m5c_chain_probe.py > results.json
This is a bounded exploration, not an M5c protocol, runner, or release gate.
All inputs are enumerated; no sampling, refitting, or outer-chain study.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from m5c_degree_probe import ARCHIVE_SHA256, ARMS, arm_name, load_archive, references
from scipy.special import expit, logsumexp

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization_core as core

KS = (0, 1, 2, 4, 8, 16, 32)
PROBE_ARMS = (*ARMS, ("B", "variational", 3.0), ("B", "variational", 10.0))


def spins(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) * 2.0 - 1.0


def split_output(p, s):
    """Keep largest beta on y1; hidden-index tie break. y2 has zero bias."""
    J, _, _, _, beta = m5a.unpack(p, s)
    order = sorted(np.flatnonzero(beta), key=lambda a: (-abs(beta[a]), a))
    slots = 15 - np.count_nonzero(J)  # one y1 slot reserved for chain
    moved = np.zeros(len(beta), dtype=bool)
    moved[order[slots:]] = True
    assert np.count_nonzero(J) + np.count_nonzero(beta[~moved]) + 1 <= 16
    assert np.count_nonzero(beta[moved]) + 1 <= 16
    return moved


def prune_output(p, s):
    """Drop degree-16 smallest |J|/|beta|; ties: J before beta, then index."""
    q = np.array(p, dtype=np.float64, copy=True)
    J, _, _, _, beta = m5a.unpack(q, s)
    edges = [(abs(v), 0, i) for i, v in enumerate(J) if v != 0]
    edges += [(abs(v), 1, a) for a, v in enumerate(beta) if v != 0]
    removed = []
    for magnitude, kind, index in sorted(edges)[: max(0, len(edges) - 16)]:
        (J if kind == 0 else beta)[index] = 0
        removed.append(
            {"kind": "J" if kind == 0 else "beta", "index": int(index), "abs": magnitude}
        )
    assert np.count_nonzero(J) + np.count_nonzero(beta) <= 16
    return q, removed


def input_copies(p, s, moved=None):
    """One clamped copy per used input/color; a parity lower bound on placement."""
    J, _, A, _, beta = m5a.unpack(p, s)
    if moved is None:
        moved = np.zeros(len(beta), dtype=bool)
    # y1 is color 1, retained hidden color 0; y2 color 0, moved hidden color 1.
    color0 = (J != 0) | np.any(A[moved] != 0, axis=0)
    color1 = np.any(A[~moved] != 0, axis=0)
    return int(color0.sum() + color1.sum())


def pair_probabilities(p, s, x, moved, coupling):
    """Sum independent hidden spins, leaving the four (y1,y2) configurations."""
    J, h, A, b, beta = m5a.unpack(p, s)
    pair = spins(2)
    attachment = np.where(moved, 1, 0)
    field = x @ A.T + b
    z = field[:, None, :] + pair[None, :, attachment] * beta
    lw = (x @ J + h)[:, None] * pair[:, 0] + coupling * pair[:, 0] * pair[:, 1]
    lw += np.logaddexp(z, -z).sum(axis=2)
    return np.exp(lw - logsumexp(lw, axis=1)[:, None])


def pair_equilibrium(p, s, x, moved, coupling):
    pair = spins(2)
    prob = pair_probabilities(p, s, x, moved, coupling)
    return prob[:, pair[:, 0] == 1].sum(axis=1), prob[:, pair[:, 0] != pair[:, 1]].sum(axis=1)


def reduced_chain(p, s, x, moved, coupling):
    """Exact block Gibbs on B=(y1, moved hidden), at complete-sweep boundaries.

    First update A=(y2, retained hidden) given B; then update B given A.
    Marginalizing A makes B Markov (at most 16 states in the archived repairs).
    Initial B is y1=incoming y and moved hidden=-1. Initial y2 is the same y,
    but the first A block redraw overwrites it before any other spin uses it.
    """
    J, h, A, b, beta = m5a.unpack(p, s)
    field, drive = x @ A.T + b, x @ J + h
    kept = ~moved
    block = spins(1 + int(moved.sum()))
    y, w = block[:, 0], block[:, 1:]
    retained = spins(int(kept.sum()))
    # g[x, old y1 bit, new y2 bit, new y1 bit] averages over retained hidden.
    # Sum both outcomes positively; 1-g_plus loses rare high-cap escapes.
    g = np.empty((len(x), 2, 2, 2))
    for old_bit, old_y in enumerate((-1, 1)):
        f = field[:, kept] + old_y * beta[kept]
        ph = np.exp(f @ retained.T - np.logaddexp(f, -f).sum(axis=1)[:, None])
        for new_bit, new_y2 in enumerate((-1, 1)):
            output_field = drive[:, None] + new_y2 * coupling + retained @ beta[kept]
            for output_bit, new_y1 in enumerate((-1, 1)):
                g[:, old_bit, new_bit, output_bit] = (ph * expit(2 * new_y1 * output_field)).sum(
                    axis=1
                )
    transition = np.zeros((len(x), len(block), len(block)))
    for bit, y2 in enumerate((-1, 1)):
        pa = expit(2 * y2 * (coupling * y + w @ beta[moved]))
        f = field[:, moved] + y2 * beta[moved]
        pw = np.exp(f @ w.T - np.logaddexp(f, -f).sum(axis=1)[:, None])
        y_bit = (y == 1).astype(int)
        py = g[:, y_bit[:, None], bit, y_bit[None, :]]
        transition += pa[None, :, None] * py * pw[:, None, :]
    # Sum the A block analytically to obtain stationary mass on B.
    f = field[:, kept, None] + beta[kept, None] * y
    c = coupling * y + w @ beta[moved]
    lw = drive[:, None] * y + field[:, moved] @ w.T
    lw += np.logaddexp(c, -c) + np.logaddexp(f, -f).sum(axis=1)
    stationary = np.exp(lw - logsumexp(lw, axis=1)[:, None])
    return block, transition, stationary


def direct_chain(p, s, x, moved, coupling):
    """Independent full-state enumeration for the small integrity controls."""
    J, h, A, b, beta = m5a.unpack(p, s)
    states = spins(len(beta) + 2)  # y1, y2, all hidden
    adjacency = np.zeros((len(beta) + 2, len(beta) + 2))
    adjacency[0, 1] = adjacency[1, 0] = coupling
    for a, v in enumerate(beta):
        output = 1 if moved[a] else 0
        adjacency[output, a + 2] = adjacency[a + 2, output] = v
    fields = np.r_[x @ J + h, 0.0, x @ A.T + b]
    lw = states @ fields + 0.5 * np.sum((states @ adjacency) * states, axis=1)
    equilibrium = np.exp(lw - logsumexp(lw))
    groups = ([1, *(np.flatnonzero(~moved) + 2)], [0, *(np.flatnonzero(moved) + 2)])
    transition = np.eye(len(states))
    for group in groups:
        other = [i for i in range(states.shape[1]) if i not in group]
        local = fields[group] + states @ adjacency[:, group]
        lw = -np.logaddexp(0.0, -2 * local[:, None, :] * states[None, :, group]).sum(axis=2)
        same = np.all(states[:, None, other] == states[None, :, other], axis=2)
        transition = transition @ (np.exp(lw) * same)
    return states, transition, equilibrium


def controls():
    # Deliberately unequal/negative couplings and all input rows; test both empty
    # hidden groups too. This does not depend on a fitted archive's near-zeros.
    s = {"blanket": [0, 1], "triples": [None] * 3}
    p = np.array(
        [0.17, -0.23, 0.31, 0.2, -0.3, 0.4, 0.1, -0.5, 0.6, 0.13, -0.2, 0.4, 0.7, -0.8, 0.9]
    )
    x = spins(2)
    worst = 0.0
    for moved in (np.array([False, True, True]), np.zeros(3, bool), np.ones(3, bool)):
        for coupling in (0.0, 1.0, 3.0):
            block, transition, stationary = reduced_chain(p, s, x, moved, coupling)
            q, _ = pair_equilibrium(p, s, x, moved, coupling)
            for i, row in enumerate(x):
                states, full, eq = direct_chain(p, s, row, moved, coupling)
                worst = max(worst, abs(eq[states[:, 0] == 1].sum() - q[i]))
                columns = [0, *(np.flatnonzero(moved) + 2)]
                codes = ((states[:, columns] > 0) * (1 << np.arange(len(columns)))).sum(axis=1)
                projection = np.eye(len(block))[codes]
                worst = max(worst, float(np.max(np.abs(full @ projection - transition[i, codes]))))
                for bit in (0, 1):
                    start = bit + 2 * bit  # both outputs incoming y, all hidden -1
                    d = np.eye(len(states))[start]
                    r = np.eye(len(block))[bit]
                    for _ in range(4):
                        worst = max(
                            worst, abs(d[states[:, 0] == 1].sum() - r[block[:, 0] == 1].sum())
                        )
                        d, r = d @ full, r @ transition[i]
                worst = max(worst, float(np.max(np.abs(eq @ full - eq))))
            worst = max(worst, float(np.max(np.abs(stationary @ (block[:, 0] == 1) - q))))
    # Infinite chain-strength limit recovers original conditional (diagnostic only).
    moved = np.array([False, True, True])
    q, _ = pair_equilibrium(p, s, x, moved, 40.0)
    worst = max(worst, float(np.max(np.abs(q - expit(2 * m5a.kernel_logit(p, s, x))))))
    if worst > 2e-12:
        raise ValueError(f"full-state control failed: {worst}")
    # The outer stationary solve must retain escapes below machine epsilon.
    strong = p.copy()
    strong[2] = 25.0
    q = expit(2 * m5a.kernel_logit(strong, s, x))
    _, _, rates = chained_finite(strong, s, moved, 10.0, q, q, include_rates=True)
    log_error = 0.0
    for i, row in enumerate(x):
        states, full, _ = direct_chain(strong, s, row, moved, 10.0)
        expected = full[3, states[:, 0] == -1].sum()
        actual = rates[1][i, 1]
        if not 0 < actual < np.finfo(float).eps:
            raise ValueError("rare positive readout escape was lost")
        log_error = max(log_error, abs(float(np.log(actual) - np.log(expected))))
    if log_error > 1e-8:
        raise ValueError("rare escape disagrees with independent full enumeration")
    return {"max_full_state_error": worst, "rare_escape_log_error": log_error}


def error_summary(prob, reference):
    error = np.abs(prob - reference)
    return {"max": float(error.max()), "mean": float(error.mean())}


def standard_finite(p, s, compiled, target):
    kernel = core.inner_kernel(p, s)
    equilibrium = expit(2 * m5a.kernel_logit(p, s, m5a.blanket_inputs(s)))
    finite = {}
    for k in KS:
        if k == 0:
            probability = np.broadcast_to([0.0, 1.0], (len(compiled), 2))
        else:
            rates = core.powered_rates(kernel, k)
            probability = np.column_stack((rates[:, 0], 1 - rates[:, 1]))
        finite[str(k)] = {
            "vs_original": error_summary(probability, compiled[:, None]),
            "vs_target": error_summary(probability, target[:, None]),
            "readout_mixing": error_summary(probability, equilibrium[:, None]),
        }
    return finite


def chained_finite(p, s, moved, coupling, compiled, target, *, include_rates=False):
    x = m5a.blanket_inputs(s)
    pieces = {str(k): [] for k in KS}
    row_error = stationary_error = marginal_error = 0.0
    for start in range(0, len(x), 128):
        rows = x[start : start + 128]
        block, transition, stationary = reduced_chain(p, s, rows, moved, coupling)
        q, _ = pair_equilibrium(p, s, rows, moved, coupling)
        row_error = max(row_error, float(np.max(np.abs(transition.sum(axis=2) - 1))))
        stationary_error = max(
            stationary_error,
            float(np.max(np.abs(np.einsum("xi,xij->xj", stationary, transition) - stationary))),
        )
        marginal_error = max(
            marginal_error, float(np.max(np.abs(stationary @ (block[:, 0] == 1) - q)))
        )
        power = np.broadcast_to(np.eye(len(block)), transition.shape).copy()
        for k in KS:
            if k == 1:
                power = transition.copy()
            elif k > 1:
                power = power @ power  # KS after 1 consists of successive powers of two
            plus = power[:, :2] @ (block[:, 0] == 1)
            minus = power[:, :2] @ (block[:, 0] == -1)
            mass = plus + minus
            row_error = max(row_error, float(np.max(np.abs(mass - 1))))
            # Remove only accumulated summation roundoff, retaining both tails.
            probability = plus / mass
            # Separate positive escape sums for the two incoming logical spins.
            rates = np.column_stack((plus[:, 0] / mass[:, 0], minus[:, 1] / mass[:, 1]))
            # Retain the full reduced-state mixing diagnostic, over ALL B starts.
            block_tv = 0.5 * np.abs(power - stationary[:, None, :]).sum(axis=2).max(axis=1)
            pieces[str(k)].append((probability, q, block_tv, rates))
    checks = {"row_sum": row_error, "stationarity": stationary_error, "marginal": marginal_error}
    if max(checks.values()) > 2e-12:
        raise ValueError(f"chained invariant failed: {checks}")
    finite, rates_by_k = {}, {}
    for k, chunks in pieces.items():
        probability = np.concatenate([c[0] for c in chunks])
        q = np.concatenate([c[1] for c in chunks])
        block_tv = np.concatenate([c[2] for c in chunks])
        rates_by_k[int(k)] = np.concatenate([c[3] for c in chunks])
        if k == "0" and not np.array_equal(probability, np.broadcast_to([0, 1], probability.shape)):
            raise ValueError("K=0 must be identity")
        finite[k] = {
            "vs_original": error_summary(probability, compiled[:, None]),
            "vs_target": error_summary(probability, target[:, None]),
            "readout_mixing": error_summary(probability, q[:, None]),
            "worst_start_reduced_state_tv": float(block_tv.max()),
        }
    if include_rates:
        prob = pair_probabilities(p, s, x, moved, coupling)
        pair = spins(2)
        rates_by_k[None] = np.column_stack(
            (prob[:, pair[:, 0] == 1].sum(axis=1), prob[:, pair[:, 0] == -1].sum(axis=1))
        )
        rates_by_k[None] /= rates_by_k[None].sum(axis=1)[:, None]
        return finite, checks, rates_by_k
    return finite, checks


def summarize(sites):
    changed = [s for s in sites if s["chained"]]
    unchanged = [s for s in sites if not s["chained"]]
    assert (len(changed), len(unchanged)) == (11, 49)
    return {
        "changed_equilibrium": {
            key: {
                "max": max(s["equilibrium"][key]["max"] for s in changed),
                "uniform_site_mean": float(
                    np.mean([s["equilibrium"][key]["mean"] for s in changed])
                ),
            }
            for key in changed[0]["equilibrium"]
        },
        "prune_beats_chain_site_count": {
            metric: sum(
                s["equilibrium"]["prune_vs_original"][metric]
                < s["equilibrium"]["chain_vs_original"][metric]
                for s in changed
            )
            for metric in ("max", "mean")
        },
        "unchanged_equilibrium_target_max": max(
            s["equilibrium"]["original_vs_target"]["max"] for s in unchanged
        ),
        "changed_finite_max": {
            method: {
                str(k): {
                    metric: max(s["finite"][method][str(k)][metric]["max"] for s in changed)
                    for metric in ("vs_original", "vs_target", "readout_mixing")
                }
                for k in KS
            }
            for method in ("original", "chain", "prune")
        },
        "unchanged_finite_max": {
            str(k): {
                metric: max(s["finite"]["original"][str(k)][metric]["max"] for s in unchanged)
                for metric in ("vs_original", "vs_target", "readout_mixing")
            }
            for k in KS
        },
    }


def main():
    checks = controls()
    record = load_archive()
    archived_finite = {
        (
            r["cell"]["reading"],
            r["cell"]["method"],
            r["cell"]["cap"],
            r["cell"]["seed"],
            r["cell"]["value"],
        ): r
        for r in record["results"].values()
        if r["cell"]["arm"] == "finite_k"
    }
    output = {
        "status": "exploration",
        "semantics": "exact_reference",
        "archive_sha256": ARCHIVE_SHA256,
        "controls": checks,
        "arms": {},
    }
    for arm in PROBE_ARMS:
        reading, _, cap = arm
        sites, work = [], []
        # Complete equilibrium comparisons for this arm before any finite K.
        pending = []
        for ref in references(record, arm):
            seed = ref["cell"]["seed"]
            structures = m5a.structures(m5a.make_target(seed, reading))
            cost = {
                method: {
                    "redraws_per_k": 0,
                    "clamp_writes": 0,
                    "free_reset_writes": 0,
                    "readout_bits": 12,
                }
                for method in ("original", "chain", "prune")
            }
            for raw, s in zip(ref["parameters"], structures, strict=True):
                p = np.array(raw, dtype=np.float64)
                J, _, _, _, beta = m5a.unpack(p, s)
                degree = int(np.count_nonzero(J) + np.count_nonzero(beta))
                repair = degree > 16
                moved = split_output(p, s) if repair else np.zeros(len(beta), bool)
                pruned, removed = prune_output(p, s)
                x = m5a.blanket_inputs(s)
                original = expit(2 * m5a.kernel_logit(p, s, x))
                target = expit(2 * m5a.exact_logit(s, x))
                q, breaks = (
                    pair_equilibrium(p, s, x, moved, cap)
                    if repair
                    else (original, np.zeros(len(x)))
                )
                qp = expit(2 * m5a.kernel_logit(pruned, s, x))
                item = {
                    "seed": seed,
                    "site": s["site"],
                    "output_degree": degree,
                    "chained": repair,
                    "moved_hidden_indices": np.flatnonzero(moved).tolist(),
                    "pruned_edges": removed,
                    "equilibrium": {
                        "original_vs_target": error_summary(original, target),
                        "chain_vs_original": error_summary(q, original),
                        "chain_vs_target": error_summary(q, target),
                        "prune_vs_original": error_summary(qp, original),
                        "prune_vs_target": error_summary(qp, target),
                        "chain_disagreement": {
                            "max": float(breaks.max()),
                            "mean": float(breaks.mean()),
                        },
                    },
                }
                if not repair:
                    assert np.array_equal(pruned, p) and np.array_equal(q, original)
                sites.append(item)
                pending.append((item, p, pruned, s, moved, original, target))
                for method, vector, split, extra in (
                    ("original", p, None, 0),
                    ("chain", p, moved, int(repair)),
                    ("prune", pruned, None, 0),
                ):
                    cost[method]["redraws_per_k"] += len(beta) + 1 + extra
                    cost[method]["free_reset_writes"] += len(beta) + 1 + extra
                    cost[method]["clamp_writes"] += input_copies(vector, s, split)
            work.append({"seed": seed, "methods": cost})
        print(
            f"{arm_name(arm)}: equilibrium complete; evaluating finite K",
            file=sys.stderr,
            flush=True,
        )
        for item, p, pruned, s, moved, original, target in pending:
            baseline = standard_finite(p, s, original, target)
            for k in KS[1:]:
                archived = archived_finite[(*arm, item["seed"], k)]["local"]["sites"][s["site"]]
                for new, old in (("vs_original", "vs_m5a"), ("vs_target", "vs_target")):
                    if abs(baseline[str(k)][new]["max"] - archived[old]) > 2e-12:
                        raise ValueError("unchanged finite kernel disagrees with M5b archive")
            item["finite"] = {"original": baseline}
            if item["chained"]:
                chain, integrity = chained_finite(p, s, moved, cap, original, target)
                item["finite"].update(
                    chain=chain, prune=standard_finite(pruned, s, original, target)
                )
                item["integrity"] = integrity
            # Other 49 retain identical M5b kernels, schedule, and finite rates.
        output["arms"][arm_name(arm)] = {
            "chain_coupling": cap,
            "sites": sites,
            "work_per_outer_sweep": work,
            "summary": summarize(sites),
        }
        print(f"{arm_name(arm)}: complete", file=sys.stderr, flush=True)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
