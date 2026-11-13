"""Directed relay latent: exploratory probe for P-0013 (row ``directed-relay-latent``).

Exploration, not evidence. Every number is an exact stationary law of a small
Markov chain, computed on a CPU by evolving the full probability vector
(``exact_reference`` arithmetic of a ``software``-defined update process). No
sampling, no training, no THRML, no hardware.

Reduced instance (fixed before any result): P-0008's ``t4_h05_q035`` family
(J = 0.35, h0 = 0.5, q = 0.35, the four 8 x 8 templates of
``dtm_small_image.templates``) cut to a 2 x 3 pixel window and two templates.
The window and the pair are the ones with the most pixels where the two
templates differ (ties: first window in raster order, then first pair). The
reduced model is its own open-boundary grid Ising on the window:

    p(x, s | y) ~ exp(J sum_<ij> x_i x_j + sum_i (m_i + g y_i) x_i + s sum_i D_i x_i + bh s)

with s = +1 for the first template, m = (h_a + h_b) / 2, D = (h_a - h_b) / 2,
g = 0.5 ln((1 - q) / q) and bh = -(ln Z_a - ln Z_b) / 2. Every observation y
(2^6 = 64) is enumerated and weighted by its exact probability p(y).

Constructions (one global binary latent):

- directed: latent s; fan-in by aggregator p-bits reading one colour class of
  pixels each (blocks of 3, the reduced stand-in for the 16-site fan-in cap),
  one-way; fan-out by one-way copy p-bits: r = 1 (two leaf copies read s) or
  r = 2 (two level-1 copies read s, each leaf reads both). A pixel reads its
  class's leaf copy with coupling D_i, one-way. Two fan-in rules:
  ``linear`` (m aggregators per block, threshold 0, gain w_a, latent input
  weight v = lam / (m w_a)) and ``thermo`` (a thermometer bank: one aggregator
  per gap between attainable block sums, threshold at the midpoint, gain w_a,
  latent weight v = gap / 2, so the noiseless bank delivers the exact block
  sum). ``ideal`` relays use gain 50 (float only).
- tied (baseline, P-0008 style): two symmetric copies of the latent, one per
  pixel colour class, each coupled symmetrically to its class's pixels by D_i
  and tied to the other by w_t (``direct``) or through two relay copies
  (``path``); latent bias bh split equally over the copies; the latent
  estimate is the mean copy marginal. Reversible, so its stationary law is
  also enumerated statically as a control.

Update process: ``chromatic`` (the Z1 two-colour schedule: phase A then phase
B; a site reading a same-colour site sees its value from before the phase) or
``sequential`` (diagnostic: fan-in, latent, copies, then pixels in raster
order; with ideal relays this is exact systematic-scan Gibbs, control C1).
Colours: pixels by checkerboard; an aggregator is the opposite colour of the
pixels it reads; latent A; a leaf is the opposite colour of the pixels it
feeds; level-1 copies B. Tied: copy of class X is the opposite colour of X;
path relays alternate.

Quantization: float, or 8 bits under ``split`` (thermo_lab.constraints
cap_for / quantize_array), every coupling (directed ones included) on one
coupling codebook and every field on one field codebook. One-way couplings
are a relaxed constraint (charter amendment 1, rule 8): Z1 publishes no cost,
precision or placement rule for them; the probe assumes the symmetric
coupling codebook applies.

Metrics per configuration: E = sum_y p(y) |P_stat(s=+1|y) - P(s=+1|y)|
(primary), the maximum over y, the p(y)-weighted mean pixel-marginal error,
sweeps from a uniform-random start (and from a wrong-latent start for the
final cells) until E stays at or below 0.01, extra sites and programming
writes (one write per programmed non-pixel site).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from thermo_lab.constraints import cap_for, quantize_array
from thermo_lab.dtm_small_image import templates

OUT = Path(__file__).with_name("probe.json")
L8, J, H0, Q = 8, 0.35, 0.5, 0.35
G = 0.5 * np.log((1 - Q) / Q)
ROWS, COLS = 2, 3
NPIX = ROWS * COLS
TOL = 0.01
IDEAL = 50.0
CONV = 1e-9
TMAX = 400
LOGGED: dict = {}


# ----------------------------------------------------------------------------
# Reduced instance (exact)
# ----------------------------------------------------------------------------


def choose_instance():
    T = templates(L8)
    best = None
    for r0 in range(L8 - ROWS + 1):
        for c0 in range(L8 - COLS + 1):
            for a, b in itertools.combinations(range(4), 2):
                ta = T[a, r0 : r0 + ROWS, c0 : c0 + COLS].ravel()
                tb = T[b, r0 : r0 + ROWS, c0 : c0 + COLS].ravel()
                nd = int((ta != tb).sum())
                if best is None or nd > best[0]:
                    best = (nd, r0, c0, a, b, ta, tb)
    nd, r0, c0, a, b, ta, tb = best
    return {"differing": nd, "window": [r0, c0], "pair": [a, b], "ta": ta, "tb": tb}


def grid_edges():
    e = []
    for r in range(ROWS):
        for c in range(COLS):
            i = r * COLS + c
            if c + 1 < COLS:
                e.append((i, i + 1))
            if r + 1 < ROWS:
                e.append((i, i + COLS))
    return e


EDGES = grid_edges()
XS = np.array(list(itertools.product((-1.0, 1.0), repeat=NPIX)))  # all pixel states
YS = XS.copy()
# The reduced instance is symmetric under a global flip (m = 0, bh = 0, equal D),
# and so is every construction here, so chains run on the 32 observations with
# y_0 = -1 at doubled weight (YC); control C0 checks this against all 64.
YC = YS


def instance_params(inst):
    ha, hb = H0 * inst["ta"], H0 * inst["tb"]
    pair = sum(XS[:, i] * XS[:, j] for i, j in EDGES)
    lza = np.logaddexp.reduce(J * pair + XS @ ha)
    lzb = np.logaddexp.reduce(J * pair + XS @ hb)
    return {"m": (ha + hb) / 2, "D": (ha - hb) / 2, "bh": -(lza - lzb) / 2, "pair": pair}


def exact_posterior(par):
    """P(s=+1|y), pixel marginals and p(y), all y, by enumeration."""
    m, D, bh, pair = par["m"], par["D"], par["bh"], par["pair"]
    flip_ll = np.log(Q) * (YS[:, None, :] != XS[None]).sum(2) + np.log(1 - Q) * (
        YS[:, None, :] == XS[None]
    ).sum(2)  # log p(y|x), (y, x)
    out_s = np.empty(len(YS))
    out_x = np.empty((len(YS), NPIX))
    lw = {}
    for s in (1.0, -1.0):
        lw[s] = J * pair + XS @ (m + s * D) + s * bh  # log p(x, s) up to a shared constant
    lpx_s = {s: lw[s] - np.logaddexp.reduce(np.concatenate([lw[1.0], lw[-1.0]])) for s in lw}
    py = np.zeros(len(YS))
    for k in range(len(YS)):
        j1 = lpx_s[1.0] + flip_ll[k]
        j0 = lpx_s[-1.0] + flip_ll[k]
        tot = np.logaddexp(np.logaddexp.reduce(j1), np.logaddexp.reduce(j0))
        py[k] = np.exp(tot)
        out_s[k] = np.exp(np.logaddexp.reduce(j1) - tot)
        w = np.exp(j1 - tot) + np.exp(j0 - tot)
        out_x[k] = w @ (XS > 0)
    return out_s, out_x, py


# ----------------------------------------------------------------------------
# Constructions
# ----------------------------------------------------------------------------


def colour(i):
    r, c = divmod(i, COLS)
    return (r + c) % 2


class Net:
    def __init__(self):
        self.kind, self.col, self.bias, self.inputs, self.clamp = [], [], [], [], []

    def add(self, kind, col, bias=0.0, clamp=None):
        self.kind.append(kind)
        self.col.append(col)
        self.bias.append(bias)
        self.inputs.append([])
        self.clamp.append(clamp)
        return len(self.kind) - 1

    def link(self, src, dst, w, sym=False):
        self.inputs[dst].append([src, w])
        if sym:
            self.inputs[src].append([dst, w])

    @property
    def M(self):
        return len(self.kind)


def add_pixels(net, par):
    for i in range(NPIX):
        net.add("pixel", colour(i), par["m"][i], clamp=i)
    for i, j in EDGES:
        net.link(i, j, J, sym=True)


DEFAULT_COLS = {"agg0": 1, "agg1": 0, "latent": 0, "leaf0": 1, "leaf1": 0, "copy1": 1}


def directed(par, fanin, r, w_a, w_c, m=1, lam=1.0, cols=None):
    cols = {**DEFAULT_COLS, **(cols or {})}
    net = Net()
    add_pixels(net, par)
    D = par["D"]
    s = net.add("latent", cols["latent"])
    aggs, offset = [], 0.0
    for cls in (0, 1):
        pix = [i for i in range(NPIX) if colour(i) == cls and D[i] != 0]
        if not pix:
            continue
        if fanin == "linear":
            for _ in range(m):
                a = net.add("agg", cols[f"agg{cls}"], 0.0)
                for i in pix:
                    net.link(i, a, w_a * D[i])
                aggs.append((a, lam / (m * w_a)))
        else:
            sums = sorted(
                {
                    round(float(np.dot(D[pix], z)), 12)
                    for z in itertools.product((-1, 1), repeat=len(pix))
                }
            )
            gaps = np.diff(sums)
            if not np.allclose(gaps, gaps[0]):
                raise ValueError("thermometer needs equally spaced block sums")
            gap = float(gaps[0])
            for k in range(len(sums) - 1):
                theta = 0.5 * (sums[k] + sums[k + 1])
                a = net.add("agg", cols[f"agg{cls}"], -w_a * theta)
                for i in pix:
                    net.link(i, a, w_a * D[i])
                aggs.append((a, gap / 2))
            offset += sums[0] + (len(sums) - 1) * gap / 2
    net.bias[s] = par["bh"] + offset
    for a, v in aggs:
        net.link(a, s, v)
    if r == 1:
        parents = [s]
    else:
        parents = []
        for _ in range(r):
            u = net.add("copy1", cols["copy1"])
            net.link(s, u, w_c)
            parents.append(u)
    for cls in (0, 1):
        leaf = net.add("leaf", cols[f"leaf{cls}"])
        for p in parents:
            net.link(p, leaf, w_c)
        for i in range(NPIX):
            if colour(i) == cls and D[i] != 0:
                net.link(leaf, i, D[i])
    return net


def tied(par, w_t, path=False):
    net = Net()
    add_pixels(net, par)
    D = par["D"]
    ncop = 4 if path else 2
    copies = {}
    for cls in (0, 1):
        copies[cls] = net.add("tcopy", 1 - cls, par["bh"] / ncop)
        for i in range(NPIX):
            if colour(i) == cls and D[i] != 0:
                net.link(copies[cls], i, D[i], sym=True)
    chain = [copies[0]]
    if path:
        r1 = net.add("tcopy", 0, par["bh"] / ncop)
        r2 = net.add("tcopy", 1, par["bh"] / ncop)
        chain += [r1, r2]
    chain.append(copies[1])
    for a, b in zip(chain, chain[1:], strict=False):
        net.link(a, b, w_t, sym=True)
    return net


def quantize(net, bits):
    if bits is None:
        return net, None
    coup = [w for ins in net.inputs for _, w in ins] + [G]
    cc, cf = cap_for("split", coup, net.bias, bits)
    q = Net()
    q.kind, q.col, q.clamp = list(net.kind), list(net.col), list(net.clamp)
    q.bias = [float(v) for v in quantize_array(np.array(net.bias), cf, bits)]
    q.inputs = [
        [[s, float(quantize_array(np.array([w]), cc, bits)[0])] for s, w in ins]
        for ins in net.inputs
    ]
    gq = float(quantize_array(np.array([G]), cc, bits)[0])
    return q, {"coupling_cap": cc, "field_cap": cf, "g": gq}


# ----------------------------------------------------------------------------
# Exact evolution of the chain's law
# ----------------------------------------------------------------------------


def schedule(net, kind):
    M = net.M
    if kind == "sequential":
        rank = {"agg": 0, "latent": 1, "copy1": 2, "leaf": 3, "tcopy": 1, "pixel": 4}
        return [[j] for j in sorted(range(M), key=lambda j: (rank[net.kind[j]], j))]
    phases = []
    for c in (0, 1):
        sites = [j for j in range(M) if net.col[j] == c]
        reads = {j: {s for s, _ in net.inputs[j] if net.col[s] == c} for j in sites}
        order, done = [], set()
        # simultaneous semantics: a reader is updated before the same-colour site it reads
        while len(order) < len(sites):
            ready = [
                j for j in sites if j not in done and all(k in done for k in sites if j in reads[k])
            ]
            if not ready:
                raise ValueError("same-colour cycle")
            for j in ready:
                order.append(j)
                done.add(j)
        phases.append(order)
    return phases


class Chain:
    def __init__(self, net, gclamp, phase_kind):
        self.net, self.M = net, net.M
        M = self.M
        idx = np.arange(1 << M, dtype=np.int64)
        spins = ((idx[:, None] >> np.arange(M)) & 1).astype(np.float64) * 2 - 1
        self.order = [j for ph in schedule(net, phase_kind) for j in ph]
        self.p = {}
        for j in range(M):
            mask0 = ((idx >> j) & 1) == 0
            f = np.full(int(mask0.sum()), net.bias[j])
            sp = spins[mask0]
            for s, w in net.inputs[j]:
                f = f + w * sp[:, s]
            if net.clamp[j] is not None:
                f = f[None, :] + gclamp * YC[:, net.clamp[j]][:, None]
            else:
                f = f[None, :]
            self.p[j] = expit(2 * f).reshape(f.shape[0], 1 << (M - 1 - j), 1 << j)
        self.spins = spins

    def sweep(self, v):
        M = self.M
        for j in self.order:
            vr = v.reshape(len(YC), 1 << (M - 1 - j), 2, 1 << j)
            tot = vr[:, :, 0, :] + vr[:, :, 1, :]
            up = tot * self.p[j]
            vr[:, :, 1, :] = up
            vr[:, :, 0, :] = tot - up
        return v

    def marg(self, v, j):
        M = self.M
        return v.reshape(len(YC), 1 << (M - 1 - j), 2, 1 << j)[:, :, 1, :].sum((1, 2))


def estimate_sites(net):
    lat = [j for j in range(net.M) if net.kind[j] == "latent"]
    return lat if lat else [j for j in range(net.M) if net.kind[j] == "tcopy"]


def run_chain(net, gclamp, phase_kind, ex, start="random", tmax=TMAX):
    ps, px, py = ex
    ch = Chain(net, gclamp, phase_kind)
    M = ch.M
    est = estimate_sites(net)
    if start == "random":
        v = np.full((len(YC), 1 << M), 1.0 / (1 << M))
    else:  # wrong latent: latent and every relay opposite the posterior mode, pixels uniform
        relay = [j for j in range(M) if net.kind[j] != "pixel" and net.kind[j] != "agg"]
        bad = np.where(ps > 0.5, -1.0, 1.0)
        ok = np.all(ch.spins[None, :, relay] == bad[:, None, None], axis=2)
        v = ok / ok.sum(1, keepdims=True)
    errs = []
    t0 = time.process_time()
    diff = np.inf
    t = 0
    while t < tmax:
        old = v.copy()
        v = ch.sweep(v)
        t += 1
        pl = np.mean([ch.marg(v, j) for j in est], axis=0)
        errs.append(float(py @ np.abs(pl - ps)))
        diff = float(np.abs(v - old).sum(1).max())
        if diff < CONV:
            break
    pl = np.mean([ch.marg(v, j) for j in est], axis=0)
    pxs = np.stack([ch.marg(v, i) for i in range(NPIX)], 1)
    e = np.abs(pl - ps)
    errs = np.array(errs)
    above = np.flatnonzero(errs > TOL)
    reach = None if errs[-1] > TOL else int(above[-1] + 2 if len(above) else 1)
    return {
        "M": M,
        "error": float(py @ e),
        "error_max_y": float(e.max()),
        "pixel_error": float(py @ np.abs(pxs - px).mean(1)),
        "sweeps_run": t,
        "converged": diff < CONV,
        "last_diff": diff,
        "reach_sweeps": reach,
        "curve_head": [round(x, 6) for x in errs[:12]],
        "cpu_s": round(time.process_time() - t0, 2),
    }


def footprint(net):
    extra = sum(1 for k in net.kind if k not in ("pixel", "latent"))
    nonpix = sum(1 for k in net.kind if k != "pixel")
    if "latent" not in net.kind:
        extra -= 1  # one copy plays the latent
    return {"extra_sites": extra, "programming_writes": nonpix, "updates_per_sweep": net.M}


def tied_static(net, gq, ex):
    """Boltzmann law of the symmetric tied model by enumeration (control C2)."""
    ps, px, py = ex
    M = net.M
    idx = np.arange(1 << M, dtype=np.int64)
    sp = ((idx[:, None] >> np.arange(M)) & 1).astype(np.float64) * 2 - 1
    e = sp @ np.array(net.bias)
    for j in range(M):
        for s, w in net.inputs[j]:
            e = e + 0.5 * w * sp[:, j] * sp[:, s]  # each symmetric edge listed twice
    e = e[None] + gq * YC @ sp[:, :NPIX].T
    w = np.exp(e - e.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    est = estimate_sites(net)
    pl = np.mean([w @ (sp[:, j] > 0) for j in est], axis=0)
    return pl


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------


def cell(label, net, bits, phase, ex, start="random"):
    qn, caps = quantize(net, bits)
    gq = caps["g"] if caps else G
    key = (label, bits, phase, start)
    if key in LOGGED:
        # computed by run 1 of this script (run1-screen.log) before it was stopped
        # for the CPU ceiling; reused unchanged, not recomputed
        out = dict(LOGGED[key])
        out.update({"from_run1_log": True, **footprint(net)})
        return out
    out = run_chain(qn, gq, phase, ex, start)
    out.update({"label": label, "bits": bits, "schedule": phase, "start": start, **footprint(net)})
    if caps:
        out["caps"] = caps
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "label",
                    "bits",
                    "schedule",
                    "start",
                    "M",
                    "error",
                    "error_max_y",
                    "reach_sweeps",
                    "sweeps_run",
                    "cpu_s",
                )
            }
        ),
        flush=True,
    )
    return out


def main():
    global YC
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--run1-log", default=None)
    args = ap.parse_args()
    if args.run1_log:
        for line in Path(args.run1_log).read_text().splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                LOGGED[(rec["label"], rec["bits"], rec["schedule"], rec["start"])] = rec
    t0 = time.process_time()
    inst = choose_instance()
    par = instance_params(inst)
    ps_all, px_all, py_all = exact_posterior(par)
    neg = np.array([np.flatnonzero((YS == -y).all(1))[0] for y in YS])
    sym = {
        "posterior_antisymmetric": float(np.abs(ps_all - (1 - ps_all[neg])).max()),
        "py_symmetric": float(np.abs(py_all - py_all[neg]).max()),
    }
    half = YS[:, 0] < 0
    ex_full = (ps_all, px_all, py_all)
    ex = (ps_all[half], px_all[half], 2 * py_all[half])
    ps, px, py = ex
    YC = YS[half]
    res = {
        "label": "exploration, not evidence; exact stationary laws on CPU; no sampling",
        "instance": {
            "family": "t4_h05_q035",
            "window_rows_cols": [ROWS, COLS],
            "window_origin": inst["window"],
            "template_pair": inst["pair"],
            "differing_pixels": inst["differing"],
            "D": par["D"].tolist(),
            "m": par["m"].tolist(),
            "bh": float(par["bh"]),
            "g": float(G),
            "py_sum": float(py.sum()),
            "fraction_py_posterior_in_0.1_0.9": float(py @ ((ps > 0.1) & (ps < 0.9))),
            "mean_abs_posterior_minus_half": float(py @ np.abs(ps - 0.5)),
            "symmetry": sym,
        },
        "stages": {},
    }
    S = res["stages"]
    ideal = dict(w_a=IDEAL, w_c=IDEAL)

    # Controls
    c = [
        cell(
            "C1 ideal thermo r=1 sequential (exact Gibbs)",
            directed(par, "thermo", 1, **ideal),
            None,
            "sequential",
            ex,
        ),
        cell(
            "C1b ideal thermo r=2 sequential (exact Gibbs)",
            directed(par, "thermo", 2, **ideal),
            None,
            "sequential",
            ex,
        ),
    ]
    tn = tied(par, 2.0)
    dyn = cell("C2 tied direct w_t=2 chromatic", tn, None, "chromatic", ex)
    dyn["static_error"] = float(py @ np.abs(tied_static(tn, G, ex) - ps))
    c.append(dyn)
    S["controls"] = c

    # Stage B: two-colour assignments with ideal relays (staleness only), r = 1
    sb = []
    for code in itertools.product((0, 1), repeat=5):
        cols = dict(zip(("agg0", "agg1", "latent", "leaf0", "leaf1"), code, strict=True))
        try:
            schedule(directed(par, "thermo", 1, **ideal, cols=cols), "chromatic")
        except ValueError:
            # a same-colour loop pixel -> aggregator -> latent -> leaf -> pixel: every
            # site in it reads stale values, so this colouring is skipped as infeasible
            sb.append(
                {
                    "label": f"staleness r=1 cols={''.join(map(str, code))}",
                    "skipped": "same-colour cycle",
                    "cols": cols,
                }
            )
            continue
        out = cell(
            f"staleness r=1 cols={''.join(map(str, code))}",
            directed(par, "thermo", 1, **ideal, cols=cols),
            None,
            "chromatic",
            ex,
        )
        out["cols"] = cols
        sb.append(out)
    best_b = min((x for x in sb if "error" in x), key=lambda x: x["error"])
    bcols = best_b["cols"]
    r2 = []
    for c1 in (0, 1):
        try:
            schedule(directed(par, "thermo", 2, **ideal, cols={**bcols, "copy1": c1}), "chromatic")
        except ValueError:
            sb.append(
                {"label": f"staleness r=2 best cols copy1={c1}", "skipped": "same-colour cycle"}
            )
            continue
        out = cell(
            f"staleness r=2 best cols copy1={c1}",
            directed(par, "thermo", 2, **ideal, cols={**bcols, "copy1": c1}),
            None,
            "chromatic",
            ex,
        )
        out["cols"] = {**bcols, "copy1": c1}
        sb.append(out)
        r2.append(out)
    bcols2 = min(r2, key=lambda x: x["error"])["cols"]
    YC = YS
    chk = cell(
        "C0 best staleness cell on all 64 observations",
        directed(par, "thermo", 1, **ideal, cols=bcols),
        None,
        "chromatic",
        ex_full,
    )
    YC = YS[half]
    chk["half_set_error"] = best_b["error"]
    S["controls"].append(chk)
    S["staleness_colourings"] = sb
    S["best_colouring"] = bcols

    # Stage C: relay losses without staleness (sequential order), float
    sc = []
    for w_a in (1.0, 2.0, 4.0):
        sc.append(
            cell(
                f"fanin thermo w_a={w_a} (ideal fan-out)",
                directed(par, "thermo", 1, w_a, IDEAL),
                None,
                "sequential",
                ex,
            )
        )
    for m in (1, 3):
        for lam in (1.0, 2.0):
            sc.append(
                cell(
                    f"fanin linear m={m} w_a=0.5 lam={lam} (ideal fan-out)",
                    directed(par, "linear", 1, 0.5, IDEAL, m=m, lam=lam),
                    None,
                    "sequential",
                    ex,
                )
            )
    for r in (1, 2):
        for w_c in (1.0, 2.0, 3.0):
            sc.append(
                cell(
                    f"fanout r={r} w_c={w_c} (ideal fan-in)",
                    directed(par, "thermo", r, IDEAL, w_c),
                    None,
                    "sequential",
                    ex,
                )
            )
    S["relay_losses_sequential"] = sc

    # Stage D: combined finite relays, float and 8 bits
    sd = []
    for bits in (None, 8):
        for r in (1, 2):
            for w_a in (4.0,):
                for w_c in (2.0, 3.0):
                    sd.append(
                        cell(
                            f"combined thermo w_a={w_a} r={r} w_c={w_c}",
                            directed(par, "thermo", r, w_a, w_c),
                            bits,
                            "sequential",
                            ex,
                        )
                    )
        for r in (1, 2):
            sd.append(
                cell(
                    f"combined thermo w_a=4.0 r={r} w_c=3.0 best colouring",
                    directed(par, "thermo", r, 4.0, 3.0, cols=bcols if r == 1 else bcols2),
                    bits,
                    "chromatic",
                    ex,
                )
            )
    S["combined"] = sd

    # Stage E: tied baseline (P-0008 style), static exact law, then chromatic mixing
    se = []
    for path in (False, True):
        for bits in (None, 8):
            for w_t in (0.5, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0):
                net = tied(par, w_t, path)
                qn, caps = quantize(net, bits)
                st = tied_static(qn, caps["g"] if caps else G, ex)
                se.append(
                    {
                        "label": f"tied {'path' if path else 'direct'} w_t={w_t}",
                        "bits": bits,
                        "static_error": float(py @ np.abs(st - ps)),
                        "static_error_max_y": float(np.abs(st - ps).max()),
                        **footprint(net),
                    }
                )
    S["tied_static"] = se
    sm = []
    for bits in (None, 8):
        for w_t in (1.0, 2.0, 3.0, 4.0):
            out = cell(f"tied direct w_t={w_t}", tied(par, w_t), bits, "chromatic", ex)
            sm.append(out)
    S["tied_dynamic"] = sm

    # Stage F: wrong-latent start for the best 8-bit sequential cell and the best tied cell
    sf = []
    b8 = min(
        (x for x in sd if x["bits"] == 8 and x["schedule"] == "sequential"),
        key=lambda x: (x["error"], x["extra_sites"]),
    )
    toks = dict(t.split("=") for t in b8["label"].split() if "=" in t)
    sf.append(
        cell(
            b8["label"],
            directed(par, "thermo", int(toks["r"]), float(toks["w_a"]), float(toks["w_c"])),
            8,
            "sequential",
            ex,
            start="wrong",
        )
    )
    bt = min(sm, key=lambda x: x["error"])
    sf.append(
        cell(
            bt["label"],
            tied(par, float(bt["label"].split("=")[1])),
            bt["bits"],
            "chromatic",
            ex,
            start="wrong",
        )
    )
    S["wrong_start"] = sf

    res["cpu_s_total"] = round(time.process_time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("cpu_s_total", res["cpu_s_total"])


if __name__ == "__main__":
    main()
