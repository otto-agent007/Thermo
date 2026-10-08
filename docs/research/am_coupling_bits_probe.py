"""Exploratory probe: associative-memory recall against coupling bits and per-site beta jitter.

EXPLORATION ONLY. Output of this script is not recorded evidence and must not
be cited as a study result. It exists to find which fixed choice bounds the
`am-coupling-bits` metric before a protocol is frozen, and to size that run.

Design under test: stage A's bias-binary associative memory
(`thermo_lab.am_binary_emulation`, arm `bias`), N = 24 visible spins, P binary
hidden units with negative bias theta * a_max and no inhibition. In the +-1
spin form THRML samples, its parameters are

    J   = beta / (2 sqrt N)              every visible-hidden coupling, times xi_mu_i
    b_h = -theta beta sqrt N / 2         every hidden field
    b_v_i = J * sum_mu xi_mu_i           visible fields (an integer multiple of J)

so the whole model is three kinds of numbers. Quantization uses the M5b
codebook (`meta_ebm_thermalization_core.round_parameters`): L = 2^(b-1) - 1,
step = cap / L, q(x) = step * clip(rint(x / step), -L, L). The cap is the
free choice; three schemes are probed:

    full      one cap = largest |parameter| of the instance (no clipping)
    split     couplings and fields each get their own full-scale codebook
              (couplings are then exact, because they share one magnitude)
    coupling  step = |J| (cap = L |J|): couplings exact, fields rounded to
              multiples of J and clipped at L |J|

Part 1 (exact): equilibrium recall of the quantized model by enumerating every
completion of the missing bits with the hidden layer summed out exactly.
Part 2 (exact, P = 8): the finite-K law of the two-block Gibbs chain (visible
block first, hidden start all off, as stage A) by explicit transition
matrices, with static per-site gains g_i: P(s_i = +1) = sigmoid(2 g_i gamma_i).
Per-site gains make the effective couplings asymmetric, so the chain has no
Boltzmann stationary law; its stationary law is found by power iteration.
Part 3 (sampled, NumPy, common random numbers): the same chain at P = 32 and
128, recall after K sweeps, nominal against jittered and quantized arms.

Pattern seeds 9800+ are probe-only; no protocol may reuse them.
Run: uv run python docs/research/am_coupling_bits_probe.py
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

N = 24
PS = (8, 32, 128)
CUES = (12, 8)
BETAS = (4.0, 8.0, 16.0)
THETAS = (0.0, 0.2, 0.4, 0.6, 0.8)
BITS = (3, 4, 5, 6, 7, 8, 10, 12)
SCHEMES = ("full", "split", "coupling")
PROBE_SEEDS = (9800, 9801, 9802, 9803)
TARGETS = 3
JITTERS = (0.0, 0.05, 0.1, 0.2, 0.3)
GAIN_DRAWS = 4
BUDGETS = (4, 16, 64, 256)
CHAINS = 256
OUT = Path(__file__).with_name("2026-10-08-am-coupling-bits-probe.json")


def patterns(seed: int, p: int) -> np.ndarray:
    """Same generator as stage A (`default_rng([seed, P])`), probe-only seeds."""
    return np.random.default_rng([seed, p]).choice([-1.0, 1.0], size=(p, N))


def completions(xi, target, cue):
    miss = np.array(list(itertools.product((-1.0, 1.0), repeat=N - cue)))
    full = np.concatenate([np.repeat(xi[target, :cue][None], len(miss), 0), miss], axis=1)
    return miss, full


def nominal(beta, theta, xi):
    j = beta / (2 * math.sqrt(N))
    bh = -theta * beta * math.sqrt(N) / 2
    bv = j * xi.sum(axis=0)
    return j, bh, bv


def codebook(x, cap, bits):
    levels = 2 ** (bits - 1) - 1
    step = cap / levels
    return step * np.clip(np.rint(np.asarray(x) / step), -levels, levels)


def quantize(j, bh, bv, bits, scheme):
    """Return quantized (J, b_h, b_v) and the codebook caps used."""
    if bits is None:
        return j, bh, bv
    fields = np.concatenate([[bh], bv])
    fmax = float(np.abs(fields).max())
    if scheme == "full":
        cap = max(abs(j), fmax)
        return (
            float(codebook(j, cap, bits)),
            float(codebook(bh, cap, bits)),
            codebook(bv, cap, bits),
        )
    if scheme == "split":
        cap_f = fmax if fmax > 0 else 1.0
        return (
            float(codebook(j, abs(j), bits)),
            float(codebook(bh, cap_f, bits)),
            codebook(bv, cap_f, bits),
        )
    if scheme == "coupling":
        cap = (2 ** (bits - 1) - 1) * abs(j)
        return (
            float(codebook(j, cap, bits)),
            float(codebook(bh, cap, bits)),
            codebook(bv, cap, bits),
        )
    raise ValueError(scheme)


def log_weight(full, overlaps, j, bh, bv):
    z = j * overlaps + bh
    return full @ bv + np.logaddexp(z, -z).sum(axis=1)


def recall_from(lw, match):
    w = np.exp(lw - lw.max())
    return float(w @ match / w.sum())


# --- part 1: exact equilibrium recall under quantization ------------------------------------


def part1_unit(args):
    p, cue, seed, target = args
    xi = patterns(seed, p)
    miss, full = completions(xi, target, cue)
    match = (miss == xi[target, cue:]).mean(axis=1)
    overlaps = full @ xi.T
    out = {"p": p, "cue": cue, "seed": seed, "target": target, "rows": []}
    cat = []
    for beta in BETAS:
        cat.append(recall_from(logsumexp(beta * overlaps / math.sqrt(N), axis=1), match))
    out["categorical"] = dict(zip([str(b) for b in BETAS], cat, strict=True))
    for beta, theta in itertools.product(BETAS, THETAS):
        j, bh, bv = nominal(beta, theta, xi)
        fields = np.concatenate([[bh], bv])
        drange = float(max(abs(j), np.abs(fields).max()) / abs(j))
        row = {"beta": beta, "theta": theta, "range": drange, "recall": {}}
        row["recall"]["exact"] = recall_from(log_weight(full, overlaps, j, bh, bv), match)
        for scheme, bits in itertools.product(SCHEMES, BITS):
            jq, bhq, bvq = quantize(j, bh, bv, bits, scheme)
            row["recall"][f"{scheme}/{bits}"] = recall_from(
                log_weight(full, overlaps, jq, bhq, bvq), match
            )
        out["rows"].append(row)
    return out


def summarize_part1(units):
    keys = ["exact"] + [f"{s}/{b}" for s in SCHEMES for b in BITS]
    summary = {}
    for p, cue in itertools.product(PS, CUES):
        us = [u for u in units if u["p"] == p and u["cue"] == cue]
        cat = {b: float(np.mean([u["categorical"][b] for u in us])) for b in us[0]["categorical"]}
        configs = [(r["beta"], r["theta"]) for r in us[0]["rows"]]
        table = {}
        for idx, (beta, theta) in enumerate(configs):
            table[f"b{beta:g}/t{theta:g}"] = {
                "range": float(np.mean([u["rows"][idx]["range"] for u in us])),
                **{k: float(np.mean([u["rows"][idx]["recall"][k] for u in us])) for k in keys},
            }
        best = max(table, key=lambda c: table[c]["exact"])
        frozen = {k: table[best][k] for k in keys}
        reselected = {k: max(table[c][k] for c in table) for k in keys}
        reselected_cfg = {k: max(table, key=lambda c, k=k: table[c][k]) for k in keys}
        summary[f"P{p}/c{cue}"] = {
            "categorical_best": max(cat.values()),
            "bias_best_config": best,
            "bias_best_range": table[best]["range"],
            "frozen_config": frozen,
            "reselected": reselected,
            "reselected_config": reselected_cfg,
            "table": table,
        }
    return summary


# --- parts 2 and 3: two-block Gibbs chain with per-site gains --------------------------------


def chain_params(xi, beta, theta, bits, scheme):
    j, bh, bv = nominal(beta, theta, xi)
    jq, bhq, bvq = quantize(j, bh, bv, bits, scheme)
    return jq * xi, np.full(xi.shape[0], bhq), np.asarray(bvq, float)  # W (P x N)


def exact_finite_k(xi, target, cue, w, bh, bv, gv, gh, budgets):
    """P = 8 only: exact visible law after K sweeps and the stationary law."""
    miss, full = completions(xi, target, cue)
    match = (miss == xi[target, cue:]).mean(axis=1)
    p = xi.shape[0]
    hid = np.array(list(itertools.product((-1.0, 1.0), repeat=p)))  # (2^P, P)
    m = N - cue

    def vis_given_h(hstates):  # (H, 2^m) law of missing bits given hidden states
        gamma = bv[cue:][None] + hstates @ w[:, cue:]  # (H, m)
        pu = expit(2 * gv[cue:][None] * gamma)
        pr = np.ones((len(hstates), len(miss)))
        for i in range(m):
            pr *= np.where(miss[None, :, i] > 0, pu[:, i : i + 1], 1 - pu[:, i : i + 1])
        return pr

    gamma_h = bh[None] + full @ w.T  # (S, P)
    ph = expit(2 * gh[None] * gamma_h)
    a = np.ones((len(full), len(hid)))
    for mu in range(p):
        a *= np.where(hid[None, :, mu] > 0, ph[:, mu : mu + 1], 1 - ph[:, mu : mu + 1])
    b = vis_given_h(hid)
    pi = vis_given_h(-np.ones((1, p)))[0]  # first sweep from hidden all off
    out, k = {}, 1
    for target_k in budgets:
        while k < target_k:
            pi = (pi @ a) @ b
            k += 1
        out[target_k] = float(pi @ match)
    stat = pi
    for _ in range(4000):
        new = (stat @ a) @ b
        if np.abs(new - stat).sum() < 1e-13:
            stat = new
            break
        stat = new
    out["stationary"] = float(stat @ match)
    return out


def sample_chain(xi, target_rows, cue, w, bh, bv, gv, gh, budgets, chains, seed):
    """NumPy two-block Gibbs, common random numbers via `seed`. Returns recall per budget."""
    rng = np.random.default_rng(seed)
    t = len(target_rows)
    total = t * chains
    cue_bits = np.repeat(xi[target_rows, :cue], chains, axis=0)
    truth = np.repeat(xi[target_rows, cue:], chains, axis=0)
    vis = np.concatenate([cue_bits, rng.choice([-1.0, 1.0], size=(total, N - cue))], axis=1)
    hidden = -np.ones((total, xi.shape[0]))
    out, k = {}, 0
    for kmax in budgets:
        while k < kmax:
            u_v = rng.random((total, N - cue))
            u_h = rng.random((total, xi.shape[0]))
            gamma_v = bv[cue:][None] + hidden @ w[:, cue:]
            vis[:, cue:] = np.where(u_v < expit(2 * gv[cue:][None] * gamma_v), 1.0, -1.0)
            gamma_h = bh[None] + vis @ w.T
            hidden = np.where(u_h < expit(2 * gh[None] * gamma_h), 1.0, -1.0)
            k += 1
        out[kmax] = float((vis[:, cue:] == truth).mean())
    return out


def gains(rng, size, jitter):
    return 1.0 + rng.uniform(-jitter, jitter, size=size) if jitter > 0 else np.ones(size)


def chain_unit(args):
    """One (P, cue, seed, beta, theta, quantization, jitter, gain draw) chain unit."""
    p, cue, seed, beta, theta, bits, scheme, jitter, draw, exact = args
    xi = patterns(seed, p)
    w, bh, bv = chain_params(xi, beta, theta, bits, scheme)
    grng = np.random.default_rng([seed, p, 31, draw])
    gv, gh = gains(grng, N, jitter), gains(grng, p, jitter)
    res = {
        "p": p,
        "cue": cue,
        "seed": seed,
        "beta": beta,
        "theta": theta,
        "bits": bits,
        "scheme": scheme,
        "jitter": jitter,
        "draw": draw,
    }
    if exact:
        per_t = [exact_finite_k(xi, t, cue, w, bh, bv, gv, gh, BUDGETS) for t in range(TARGETS)]
        res["exact"] = {
            str(k): float(np.mean([r[k] for r in per_t])) for k in [*BUDGETS, "stationary"]
        }
    crn = int(np.random.SeedSequence([seed, p, cue, 77]).generate_state(1)[0])
    sampled = sample_chain(xi, list(range(TARGETS)), cue, w, bh, bv, gv, gh, BUDGETS, CHAINS, crn)
    res["sampled"] = {str(k): v for k, v in sampled.items()}
    return res


def main() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    workers = int(os.environ.get("PROBE_WORKERS", "4"))
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        jobs1 = [
            (p, c, s, t) for p in PS for c in CUES for s in PROBE_SEEDS for t in range(TARGETS)
        ]
        # heaviest first
        jobs1.sort(key=lambda j: -(j[0] * 2 ** (N - j[1])))
        units1 = list(pool.map(part1_unit, jobs1))
        t1 = time.time()
        part1 = summarize_part1(units1)
        print(f"part 1 done in {t1 - t0:.0f} s", flush=True)

        jobs2 = []
        for p, c in itertools.product(PS, CUES):
            best = part1[f"P{p}/c{c}"]["bias_best_config"]
            beta = float(best.split("/")[0][1:])
            theta = float(best.split("/")[1][1:])
            exact = p == 8 and c == 12
            quants = [(None, "none")] + [(b, s) for s in ("full", "split") for b in (4, 6, 8)]
            for s in PROBE_SEEDS:
                for bits, scheme in quants:
                    jits = JITTERS if bits is None else (0.0, 0.1)
                    for jit in jits:
                        draws = range(GAIN_DRAWS) if jit > 0 else range(1)
                        for d in draws:
                            jobs2.append((p, c, s, beta, theta, bits, scheme, jit, d, exact))
        units2 = list(pool.map(chain_unit, jobs2, chunksize=4))
    t2 = time.time()
    print(f"chains done in {t2 - t1:.0f} s ({len(jobs2)} units)", flush=True)

    chains_summary = {}
    for p, c in itertools.product(PS, CUES):
        cell = [u for u in units2 if u["p"] == p and u["cue"] == c]
        groups = {}
        for u in cell:
            key = f"{u['scheme']}/{u['bits']}/j{u['jitter']:g}"
            groups.setdefault(key, []).append(u)
        nominal_by_seed = {u["seed"]: u for u in cell if u["bits"] is None and u["jitter"] == 0.0}
        entry = {}
        for key, us in groups.items():
            row: dict = {"units": len(us)}
            for kind in ("sampled", "exact"):
                if kind not in us[0]:
                    continue
                for k in us[0][kind]:
                    vals = [u[kind][k] for u in us]
                    diffs = [u[kind][k] - nominal_by_seed[u["seed"]][kind][k] for u in us]
                    row[f"{kind}/K{k}"] = float(np.mean(vals))
                    row[f"{kind}/K{k}/diff"] = float(np.mean(diffs))
                    row[f"{kind}/K{k}/diff_min"] = float(np.min(diffs))
            entry[key] = row
        chains_summary[f"P{p}/c{c}"] = entry

    record = {
        "label": "exploration only; not recorded evidence",
        "probe": "docs/research/am_coupling_bits_probe.py",
        "settings": {
            "n": N,
            "ps": PS,
            "cues": CUES,
            "betas": BETAS,
            "thetas": THETAS,
            "bits": BITS,
            "schemes": SCHEMES,
            "probe_seeds": PROBE_SEEDS,
            "targets": TARGETS,
            "jitters": JITTERS,
            "gain_draws": GAIN_DRAWS,
            "budgets": BUDGETS,
            "chains": CHAINS,
            "gain_model": "static per-site gain g ~ U[1-j, 1+j]; P(s=+1) = sigmoid(2 g gamma)",
            "sweep": "missing visible block, then hidden block; hidden start all off",
        },
        "timing_s": {"part1": t1 - t0, "chains": t2 - t1, "workers": workers},
        "equilibrium": part1,
        "chains": chains_summary,
    }
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
