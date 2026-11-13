"""Margin-budget probe for queue row `margin-budget-copies` (P-0011).

Exploration, not a recorded study. Exact (`exact_reference`) conditionals of the
five-spin PAsymSwap kernel family of `docs/research/cap_probe.py`, computed on a
CPU. No sampling, no training, no hardware.

Run from a Thermo checkout:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      uv run python docs/research/margin_budget_copies_probe/probe.py

Fixed choices, declared before the run:

- Family: cap_probe's nine parameters (hidden field hh, output fields h0 h1,
  input->output couplings J[k, i], hidden->output couplings jh0 jh1), plus two
  input->hidden couplings k0 k1 that are zero except in the hidden arm.
  Spins s = 2b - 1; word order 00, 01, 10, 11; no output-output coupling.
- Copies arm (c, r), r in {1, 2, 4}, c in {1, 2, 3, 4}: each input bit k has r
  clamped copies, copy m coupled to output i by J[k, m, i] in [-c, c]. Every
  conditional (equilibrium and finite K) depends on the copies only through
  sum_m J[k, m, i], whose range is exactly [-r c, r c]. The arm is therefore
  searched as the nine-parameter family with J bounded by r c and every other
  parameter bounded by c. This reduction is exact, not an approximation.
- Hidden arm (c), r = 1: one input->hidden coupling k1 (input bit 1) in [-c, c];
  k0 = 0. Two constraints are reported: both occupied inputs (as cap_probe),
  and the coupled pair (00, 01) alone.
- Horizons: equilibrium and K = 4 uniform-reset hidden-then-output sweeps
  (cap_probe's convention, cross-checked against thermo_lab).
- Leakage: L00 = P(out != 00 | 00), computed as the sum of the three other
  entries (not 1 - C00, which floors at ~1e-16); L01 = P(out in {00, 11} | 01);
  L10 likewise. Occupied leakage for the product is max(L01, L10).
- Product search: minimize log L00 subject to max occupied leakage <= eps
  (cap_probe's penalty objective), L-BFGS-B in the box, eps in
  {exp(-2 c r + 1), exp(-2 c r - 1)} (either side of the balanced point,
  eps < 0.5). Two independent search seeds (one for the hidden arm's
  both-pair constraint, which the uncoupled bit bounds), 6 starts each,
  alternating random corners and uniform points. (A first run at 16 starts
  crashed in post-processing after 1,146 CPU-s and saved nothing; 6 starts
  keep the probe under its 30 CPU-minute ceiling.)
  Best product = min over eps and seeds of L00 * max(L01, L10) at feasible
  points (occupied <= 1.01 eps). Resolution = |log10| ratio of the two seeds'
  best products. Local search gives upper bounds on the minimum.
- Survival: uniform visitation on the 5x5 torus schedule from the cap note,
  460 empty-edge and 40 occupied-edge operations (20 per occupied input),
  one kernel, hop fidelity ignored: S = (1-L00)^460 (1-L01)^20 (1-L10)^20.
  Maximized directly by the same multistart search.
- Analytic bound (derived in the note, checked here): at equilibrium with no
  input->hidden coupling, p(o | 01) is p(o | 00) tilted by exp(2 sum_i o_i
  Jsum[1, i]), independent of the hidden spin, which gives
  L00 * L01 >= (1 - L00)(1 - L01) exp(-4 max_i Jsum[1, i]+) >= (...) e^(-4cr).
  Same for (00, 10) with bit 0. Checked on random draws and on every search
  result; at K = 4 the same ratio is only checked numerically.
- Control: cap_probe.min_empty_leak run unedited (60 starts, seed = cap) at
  caps 2 to 4, equilibrium and K = 4, eps 1e-2 and 1e-3, against the archived
  table of docs/research/2026-09-23-cap-leakage-analysis.md.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import expit, logsumexp

HERE = Path(__file__).resolve().parent
CAP_PROBE = HERE.parent / "cap_probe.py"
S = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
H = np.array([-1.0, 1.0])
CAPS = (1, 2, 3, 4)
COPIES = (1, 2, 4)
HORIZONS = (None, 4)
STARTS = 6
SEEDS = (101, 202)
N_EMPTY, N_OCC = 460, 40
SURVIVAL_TARGET = 0.95
WRITE_PJ = 153.6  # sealed profile full write, calibrated_projection
ARCHIVED = {  # cap_leakage_analysis table: (cap, K) -> (eps 1e-2, eps 1e-3)
    (2, None): (3.32e-2, 3.37e-1),
    (2, 4): (3.42e-2, 4.05e-1),
    (3, None): (6.08e-4, 6.16e-3),
    (3, 4): (5.93e-4, 6.23e-3),
    (4, None): (1.11e-5, 1.12e-4),
    (4, 4): (1.12e-5, 1.12e-4),
}


def load_cap_probe():
    spec = importlib.util.spec_from_file_location("cap_probe", CAP_PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unpack(p: np.ndarray):
    hh, h0, h1, j00, j01, j10, j11, jh0, jh1, k0, k1 = p
    return (
        hh,
        np.array([h0, h1]),
        np.array([[j00, j01], [j10, j11]]),
        np.array([jh0, jh1]),
        np.array([k0, k1]),
    )


def log_conditional(p: np.ndarray, K: int | None) -> np.ndarray:
    """log C[x, o], input-major, word order 00 01 10 11."""
    hh, hf, J, jh, kh = unpack(p)
    a = hf + S @ J  # (x, i)
    hx = hh + S @ kh  # (x,) hidden field including input->hidden
    lo = a[:, None, :] + H[None, :, None] * jh  # (x, h, i)
    if K is None:
        lw = hx[:, None, None] * H[None, :, None] + (S[None, None] * lo[:, :, None, :]).sum(-1)
        lc = logsumexp(lw, axis=1)
        return lc - logsumexp(lc, axis=1, keepdims=True)
    ph = expit(2 * (hx[:, None] + (S @ jh)[None, :]))  # (x, o) p(h=+1 | o, x)
    Ph = np.stack([1 - ph, ph], -1)  # (x, o, h)
    pu, pd = expit(2 * lo), expit(-2 * lo)  # (x, h, i)
    Po = np.prod(np.where(S[None, None] > 0, pu[:, :, None, :], pd[:, :, None, :]), -1)  # (x,h,o')
    T = np.einsum("xoh,xhp->xohp", Ph, Po)  # (x, o, h', o')
    T = np.broadcast_to(T[:, None], (4, 2, 4, 2, 4)).reshape(4, 8, 8)
    d = np.full((4, 8), 1 / 8)
    for _ in range(K):
        d = np.einsum("xs,xst->xt", d, T)
    C = d.reshape(4, 2, 4).sum(1)
    with np.errstate(divide="ignore"):
        return np.log(C)


def log_leaks(p: np.ndarray, K: int | None) -> np.ndarray:
    lc = log_conditional(p, K)
    return np.array(
        [
            logsumexp(lc[0, 1:]),
            np.logaddexp(lc[1, 0], lc[1, 3]),
            np.logaddexp(lc[2, 0], lc[2, 3]),
        ]
    )


def bounds_for(arm: str, c: float, r: int) -> list[tuple[float, float]]:
    b = [(-c, c)] * 3 + [(-r * c, r * c)] * 4 + [(-c, c)] * 2
    return b + ([(0.0, 0.0), (-c, c)] if arm == "hidden" else [(0.0, 0.0), (0.0, 0.0)])


def starts(rng, bnds, n):
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])
    for i in range(n):
        if i % 2:
            yield np.where(rng.random(len(bnds)) < 0.5, lo, hi)
        else:
            yield rng.uniform(lo, hi)


def tilt_bound_ratio(p: np.ndarray, ll: np.ndarray, c: float, r: int) -> list[float]:
    """Product / analytic floor for pairs (00,01) and (00,10), floor at e^(-4cr)."""
    L = np.exp(ll)
    out = []
    for occ in (1, 2):
        floor = (1 - L[0]) * (1 - L[occ]) * math.exp(-4 * c * r)
        out.append(float(L[0] * L[occ] / floor) if floor > 0 else float("nan"))
    return out


def search_product(task):
    arm, c, r, K, eps, seed, pair = task
    bnds = bounds_for(arm, c, r)
    rng = np.random.default_rng(seed)
    occ = [1, 2] if pair == "both" else [1]
    best = None
    t0 = time.process_time()
    for x0 in starts(rng, bnds, STARTS):

        def f(p):
            ll = log_leaks(p, K)
            excess = max(0.0, ll[occ].max() - math.log(eps))
            return ll[0] + 1e3 * excess**2

        res = minimize(f, x0, method="L-BFGS-B", bounds=bnds)
        ll = log_leaks(res.x, K)
        if ll[occ].max() <= math.log(eps * 1.01) and ll[0] < math.log(0.5):
            lp = ll[0] + ll[occ].max()
            if best is None or lp < best["log_product"]:
                best = {
                    "log_product": float(lp),
                    "log_leaks": ll.tolist(),
                    "params": res.x.tolist(),
                }
    out = {"arm": arm, "cap": c, "r": r, "K": K, "eps": eps, "seed": seed, "pair": pair}
    out["cpu_s"] = time.process_time() - t0
    out["best"] = best
    if best is not None:
        out["tilt_ratio"] = tilt_bound_ratio(
            np.array(best["params"]), np.array(best["log_leaks"]), c, r
        )
    return out


def search_survival(task):
    arm, c, r, K, seed = task
    bnds = bounds_for(arm, c, r)
    rng = np.random.default_rng(seed)
    best = None
    t0 = time.process_time()

    def hazard(p):
        L = np.exp(log_leaks(p, K))
        return -(
            N_EMPTY * np.log1p(-min(L[0], 1 - 1e-15))
            + (N_OCC / 2) * (np.log1p(-min(L[1], 1 - 1e-15)) + np.log1p(-min(L[2], 1 - 1e-15)))
        )

    for x0 in starts(rng, bnds, STARTS):
        res = minimize(lambda p: math.log(hazard(p)), x0, method="L-BFGS-B", bounds=bnds)
        hz = hazard(res.x)
        if best is None or hz < best["hazard"]:
            best = {
                "hazard": float(hz),
                "log_leaks": log_leaks(res.x, K).tolist(),
                "params": res.x.tolist(),
            }
    best["survival"] = math.exp(-best["hazard"])
    return {
        "arm": arm,
        "cap": c,
        "r": r,
        "K": K,
        "seed": seed,
        "cpu_s": time.process_time() - t0,
        "best": best,
    }


def control_task(task):
    cap, K, eps = task
    cp = load_cap_probe()
    t0 = time.process_time()
    v = cp.min_empty_leak(cap, eps, K, starts=60, seed=cap)
    return {
        "cap": cap,
        "K": K,
        "eps": eps,
        "min_empty_leak": float(v),
        "cpu_s": time.process_time() - t0,
    }


def survival_bound(cr: float) -> float:
    """Max uniform-visitation survival allowed by the equilibrium tilt bound."""

    def neg_log_s(z):
        log_l00 = -np.logaddexp(0.0, -z)
        log_q00 = -np.logaddexp(0.0, z)  # log(1 - L00)
        log_q01 = log_l00 - np.logaddexp(log_l00, log_q00 - 4 * cr)  # log(1 - L01)
        return -(N_EMPTY * log_q00 + N_OCC * log_q01)

    res = minimize_scalar(neg_log_s, bounds=(-80, 5), method="bounded", options={"xatol": 1e-10})
    return math.exp(-res.fun)


def random_check(seed: int, n: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for c in CAPS:
        for r in COPIES:
            bnds = bounds_for("copies", c, r)
            lo = np.array([b[0] for b in bnds])
            hi = np.array([b[1] for b in bnds])
            for K in HORIZONS:
                mins = []
                for _ in range(n):
                    p = rng.uniform(lo, hi)
                    ll = log_leaks(p, K)
                    mins.append(min(tilt_bound_ratio(p, ll, c, r)))
                out[f"c{c}_r{r}_K{K}"] = {
                    "draws": n,
                    "min_ratio": float(np.nanmin(mins)),
                    "violations": int(np.sum(np.array(mins) < 1 - 1e-9)),
                }
    return out


def cross_checks() -> dict:
    cp = load_cap_probe()
    rng = np.random.default_rng(0)
    worst_cp, worst_l00 = 0.0, 0.0
    for _ in range(20):
        p9 = rng.uniform(-2, 2, 9)
        p = np.concatenate([p9, [0.0, 0.0]])
        for K in (None, 1, 4, 30):
            C = np.exp(log_conditional(p, K))
            worst_cp = max(worst_cp, float(np.abs(C - cp.conditional(p9, K)).max()))
            worst_l00 = max(worst_l00, abs(float(np.exp(log_leaks(p, K))[0]) - float(1 - C[0, 0])))
    # Hidden arm equilibrium vs brute-force enumeration of the five-spin Boltzmann law.
    worst_hid = 0.0
    for _ in range(20):
        p = rng.uniform(-2, 2, 11)
        hh, hf, J, jh, kh = unpack(p)
        for x in range(4):
            w = np.zeros(4)
            for o in range(4):
                for h in H:
                    e = hh * h + h * (S[x] @ kh) + S[o] @ (hf + S[x] @ J + h * jh)
                    w[o] += math.exp(e)
            worst_hid = max(
                worst_hid, float(np.abs(w / w.sum() - np.exp(log_conditional(p, None))[x]).max())
            )
    # Hidden arm finite K vs explicit 16-state (h, o) chain built by enumeration.
    worst_hid_k = 0.0
    for _ in range(10):
        p = rng.uniform(-2, 2, 11)
        hh, hf, J, jh, kh = unpack(p)
        for x in range(4):
            T = np.zeros((8, 8))
            for h0 in range(2):
                for o0 in range(4):
                    ph = expit(2 * (hh + S[x] @ kh + S[o0] @ jh))
                    for h1 in range(2):
                        pr_h = ph if H[h1] > 0 else 1 - ph
                        f = hf + S[x] @ J + H[h1] * jh
                        for o1 in range(4):
                            po = np.prod(expit(2 * S[o1] * f))
                            T[h0 * 4 + o0, h1 * 4 + o1] += pr_h * po
            d = np.full(8, 1 / 8)
            for _ in range(4):
                d = d @ T
            C = d.reshape(2, 4).sum(0)
            worst_hid_k = max(
                worst_hid_k, float(np.abs(C - np.exp(log_conditional(p, 4))[x]).max())
            )
    return {
        "thermo_lab_vs_cap_probe": float(cp.cross_check()),
        "probe_vs_cap_probe": worst_cp,
        "l00_sum_vs_one_minus": worst_l00,
        "hidden_eq_vs_enumeration": worst_hid,
        "hidden_k4_vs_explicit_chain": worst_hid_k,
    }


def eps_grid(c: float, r: int) -> list[float]:
    vals = []
    for e in (math.exp(-2 * c * r + 1), math.exp(-2 * c * r - 1)):
        if e < 0.5:
            vals.append(e)
    return vals


def main() -> None:
    t_wall = time.time()
    record = {
        "label": "exploration, exact_reference conditionals, not evidence",
        "starts": STARTS,
        "seeds": SEEDS,
    }
    record["cross_checks"] = cross_checks()
    print("cross-checks", record["cross_checks"], flush=True)

    ctrl = [(cap, K, eps) for cap in (2, 3, 4) for K in HORIZONS for eps in (1e-2, 1e-3)]
    prod = []
    for c in CAPS:
        for r in COPIES:
            for K in HORIZONS:
                for eps in eps_grid(c, r):
                    for s in SEEDS:
                        prod.append(("copies", c, r, K, eps, s, "both"))
        for K in HORIZONS:
            for eps in eps_grid(c, 1):
                for s in SEEDS:
                    prod.append(("hidden", c, 1, K, eps, s, "coupled"))
                prod.append(("hidden", c, 1, K, eps, SEEDS[0], "both"))
    surv = [("copies", c, r, K, SEEDS[0]) for c in CAPS for r in COPIES for K in HORIZONS]
    surv += [("hidden", c, 1, K, SEEDS[0]) for c in CAPS for K in HORIZONS]

    with Pool(4) as pool:
        record["control"] = pool.map(control_task, ctrl)
        print("control done", round(time.time() - t_wall), flush=True)
        record["product_search"] = pool.map(search_product, prod, chunksize=1)
        print("product done", round(time.time() - t_wall), flush=True)
        record["survival_search"] = pool.map(search_survival, surv, chunksize=1)
        print("survival done", round(time.time() - t_wall), flush=True)
    out = HERE / "probe.json"
    out.write_text(json.dumps(record, indent=1, default=lambda v: None) + "\n")  # partial
    for row in record["control"]:
        a = ARCHIVED[(row["cap"], row["K"])][0 if row["eps"] == 1e-2 else 1]
        row["archived"] = a
        row["ratio_to_archived"] = row["min_empty_leak"] / a
    t_main = time.process_time()
    record["random_tilt_check"] = random_check(7, 500)
    sb = {f"{cr:g}": survival_bound(cr) for cr in (1, 2, 3, 4, 6, 8, 12, 16)}
    record["survival_bound_by_cr"] = sb
    record["cr_star_95"] = brentq(lambda cr: survival_bound(cr) - SURVIVAL_TARGET, 2, 10, xtol=1e-6)
    record["write_pj_per_copy"] = WRITE_PJ
    cpu = sum(
        x["cpu_s"] for k in ("control", "product_search", "survival_search") for x in record[k]
    )
    record["cpu_s_workers"] = cpu
    record["cpu_s_main_postprocess"] = time.process_time() - t_main
    record["wall_s"] = time.time() - t_wall
    out.write_text(json.dumps(record, indent=1, default=lambda v: None) + "\n")
    print(
        "wrote", out, "worker cpu s", round(cpu), "wall s", round(record["wall_s"]), file=sys.stderr
    )


if __name__ == "__main__":
    main()
