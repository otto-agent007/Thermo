"""DTM exact-posterior lattice (proposal P-0008, queue row ``dtm-exact-posterior-lattice``).

Protocol: ``docs/experiments/dtm-exact-posterior-lattice.md`` (approved at commit
d4f8b2a). A hand-built model of the exact mixture posterior of P-0007's families
(``t4_h05_q035`` first, ``smooth8_h05_q035`` only if ``t4`` passes stops (a) and
(b)): one binary template unit per component (stage A's centered bias design),
spread over the Z1 16-offset lattice as tied copies joined by relay paths,
quantized under ``split`` and sampled by NumPy two-colour Gibbs. Nothing is
trained.

Stages per family, in order: E-dev (exact tie-limit model on both layouts at six
tie values, development images; admission and the carried layout), D (sampled
development cells at the admitted tie values; w*), E-test (exact rows at w* on
the test images; stop (a) and, on ``smooth8``, the stage E stop) and S (81
sampled cells, a float K = 4096 reference and the oracle pair per family,
three sampler seeds).

The model builder, the exact tie-limit equilibrium, the layout code and the
NumPy sampler are the probes' (``docs/research/dtm_exact_posterior_lattice_*.py``),
copied here so the import package does not depend on ``docs/``; the unit tests
check them bitwise against the probes. Families, the transfer-matrix posterior,
quantization, jitter, placement and pricing are imported unchanged from
``thermo_lab.dtm_baseline_headroom``, ``thermo_lab.dtm_small_image`` and
``thermo_lab.constraints``.

The sampler is NumPy two-colour Gibbs over the constraint layer's quantized,
jittered model. It is **not the lowered THRML program**; preflight 3 checks it
against the layer's exact chromatic kernels.

Long run (about 2.6 CPU-hours on ``t4``): every work unit is written atomically
to ``units/`` with a digest in ``units/index.json``; ``--resume`` re-authenticates
the request and sources and runs only missing units.

Evidence classes: mixture samples, Bayes marginals, every tie-limit and
conditioned equilibrium and every n-draw floor are ``exact_reference`` (float64).
Every Gibbs cell is ``software_simulation`` (NumPy, CPU). Every joule is a
``calibrated_projection``. Nothing here ran on hardware.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import multiprocessing
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit, logsumexp
from scipy.stats import binom

from thermo_lab import dtm_baseline_headroom as hr
from thermo_lab.constraints import (
    ConstraintRequest,
    PairwiseModel,
    cap_for,
    check,
    count,
    exact_chromatic_kernels,
    price,
    propagate,
    quantize_array,
    request_gains,
    require_placement,
)
from thermo_lab.constraints.lattice import OFFSETS
from thermo_lab.dtm_small_image import (
    READ_J,
    WRITE_J,
    GridIsing,
    Runner,
    flip,
    read_gz_json,
    write_gz_json,
)
from thermo_lab.hardware.z1 import Z1HardwareProfile
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.planar_ising_scaling import check_equal
from thermo_lab.provenance import collect_runtime_provenance

ROOT = Path(__file__).resolve().parents[2]
MODULE = "src/thermo_lab/dtm_exact_posterior_lattice.py"
SOURCES = (
    MODULE,
    "src/thermo_lab/dtm_baseline_headroom.py",
    "src/thermo_lab/dtm_small_image.py",
    "src/thermo_lab/constraints/__init__.py",
    "src/thermo_lab/constraints/accounting.py",
    "src/thermo_lab/constraints/gibbs.py",
    "src/thermo_lab/constraints/jitter.py",
    "src/thermo_lab/constraints/lattice.py",
    "src/thermo_lab/constraints/model.py",
    "src/thermo_lab/constraints/quantize.py",
    "src/thermo_lab/constraints/request.py",
    "src/thermo_lab/meta_ebm_thermalization_core.py",
    "src/thermo_lab/hardware/z1.py",
    "src/thermo_lab/hashing.py",
    "src/thermo_lab/persistence.py",
    "src/thermo_lab/planar_ising_scaling.py",
    "docs/research/dtm_exact_posterior_lattice_probe.py",
    "docs/research/dtm_exact_posterior_lattice_placed.py",
    "docs/research/dtm_exact_posterior_lattice_layout4.py",
    "docs/research/dtm_exact_posterior_lattice_oracle_ref.py",
    "docs/experiments/dtm-exact-posterior-lattice.md",
    "uv.lock",
)
PROTOCOL_COMMIT = "d4f8b2a7f498743ce15170d45a14d555bb1a21fd"
PROBE_SEED = 20261017
DEFAULT_WORKERS = 3
L = 8
N = L * L
PROFILE = Z1HardwareProfile()
UPDATE_J = PROFILE.gibbs_node_update_energy_j
BASES = {"index5": ((2, 1), (-1, 2)), "index4": ((-4, -1), (0, -1))}
MARGIN = 14
PAD = 10
ARMS = ("op", "jit0", "b6")
STARTS = ("continue", "random", "reset")


# ----------------------------------------------------------------------------
# Model (exact reference): the round-1 probe's builder, copied, checked bitwise
# ----------------------------------------------------------------------------


def build(J, h, q, b0):
    """Spin-form parameters of the unplaced bias-design model (centered), as the probe."""
    C = len(h)
    grid = GridIsing(L, J)
    logz = grid.forward(h)[0]
    m = h.mean(0)
    d = h - m
    B = b0 - (logz - logz.mean())
    return {
        "J": J,
        "g": 0.5 * np.log((1 - q) / q),
        "W": d.T / 2.0,
        "bp": m + d.sum(0) / 2.0,
        "bh": B / 2.0,
        "whh": 0.0,
        "C": C,
    }


def build_onehot(J, h, q):
    """One-hot design at lambda = 0 (the oracle check's model), as the probe."""
    C = len(h)
    logz = GridIsing(L, J).forward(h)[0]
    m = h.mean(0)
    d = h - m
    B = -(logz - logz.mean())
    return {
        "J": J,
        "g": 0.5 * np.log((1 - q) / q),
        "W": d.T / 2.0,
        "bp": m + d.sum(0) / 2.0,
        "bh": B / 2.0 - 0.0 * (C - 2) / 2.0,
        "whh": -0.0 / 2.0,
        "C": C,
    }


def quantize_unplaced(p, bits):
    """The probe's ``quantize``: ``split`` codebook, one cap for couplings, one for fields."""
    if bits is None:
        return dict(p)
    C = p["C"]
    ncoup = [np.full(2 * L * (L - 1), p["J"]), np.full(N, p["g"]), p["W"].ravel()]
    couplings = np.concatenate(ncoup)
    fields = np.concatenate([p["bp"], p["bh"]])
    cc, cf = cap_for("split", couplings, fields, bits)
    qc = quantize_array(couplings, cc, bits)
    qf = quantize_array(fields, cf, bits)
    k = 2 * L * (L - 1)
    out = dict(p)
    out["J"] = float(qc[0])
    out["g"] = float(qc[k])
    out["W"] = qc[k + N : k + N + N * C].reshape(N, C)
    out["bp"], out["bh"] = qf[:N], qf[N:]
    out["caps"] = {"coupling": cc, "field": cf}
    return out


def hidden_states(C):
    return np.array(list(itertools.product((-1.0, 1.0), repeat=C)))


def exact_equilibrium(p, y, prune=1e-9, chunk=128):
    """Equilibrium pixel marginals P(x_i = +1 | y) of the unplaced spin model (probe)."""
    C = p["C"]
    S = hidden_states(C)
    hid = S @ p["bh"] + p["whh"] * ((S.sum(1) ** 2 - C) / 2.0)
    grid = GridIsing(L, p["J"])
    base = p["bp"][None] + S @ p["W"].T
    out = np.empty((len(y), N))
    wts = np.empty((len(y), len(S)))
    for i in range(0, len(y), chunk):
        yy = y[i : i + chunk]
        fields = base[:, None, :] + p["g"] * yy[None]
        lz = grid.forward(fields.reshape(-1, N))[0].reshape(len(S), len(yy))
        lw = hid[:, None] + lz
        w = np.exp(lw - logsumexp(lw, axis=0))
        wts[i : i + chunk] = w.T
        acc = np.zeros((len(yy), N))
        for si in range(len(S)):
            keep = np.flatnonzero(w[si] > prune)
            if len(keep):
                pm, _ = grid.marginals(fields[si, keep])
                acc[keep] += w[si, keep, None] * pm
        out[i : i + chunk] = acc / wts[i : i + chunk].sum(1, keepdims=True).clip(1e-300)
    return out, wts, S


def maj_wrong(p_up, x, draws):
    """Per-image expected n-draw majority error (ties to +1) from i.i.d. marginals."""
    pp = binom.sf(draws // 2 - 1, draws, p_up)
    return np.where(x > 0, 1.0 - pp, pp).mean(1)


# ----------------------------------------------------------------------------
# Layout (round-3 probe), copied, checked bitwise
# ----------------------------------------------------------------------------


class Basis:
    def __init__(self, a, b):
        self.A, self.B = np.array(a), np.array(b)
        self.M = np.stack([self.A, self.B], 1)
        self.Mi = np.linalg.inv(self.M)
        self.index = abs(round(np.linalg.det(self.M)))

    def site(self, i, j, rep=(0, 0)):
        s = i * self.A + j * self.B + np.array(rep)
        return (int(s[0]), int(s[1]))

    def pix(self, site):
        c = self.Mi @ np.array(site, dtype=float)
        r = np.round(c).astype(int)
        return (int(r[0]), int(r[1])) if np.allclose(c, r) else None

    def cosets(self):
        reps, seen = [], set()
        for o in OFFSETS:
            if self.pix(o) is not None:
                continue
            key = self.canon(o)
            if key not in seen:
                seen.add(key)
                reps.append(o)
        return reps

    def canon(self, s):
        c = self.Mi @ np.array(s, dtype=float)
        f = c - np.floor(c + 1e-9)
        r = self.M @ f
        return (round(r[0], 6), round(r[1], 6))

    def touched(self, site):
        out = []
        for o in OFFSETS:
            p = self.pix((site[0] + o[0], site[1] + o[1]))
            if p is not None:
                out.append(p)
        return out


def hnf_sublattices(n):
    for a in range(1, n + 1):
        if n % a:
            continue
        d = n // a
        for b in range(a):
            yield np.array([[a, b], [0, d]])


def _residue(H):
    Hi = np.linalg.inv(H)

    def res(p):
        c = Hi @ np.array(p, dtype=float)
        f = c - np.floor(c + 1e-9)
        r = H @ f
        return (round(r[0], 6), round(r[1], 6))

    return res


def tilings(P, n):
    out = []
    for H in hnf_sublattices(n):
        res = _residue(H)
        if len({res(p) for p in P}) == n:
            allr = sorted({res((i, j)) for i in range(n) for j in range(n)})
            out.append((H, res, [frozenset([r]) for r in allr]))
    for H in hnf_sublattices(2 * n):
        res = _residue(H)
        allr = sorted({res((i, j)) for i in range(2 * n) for j in range(2 * n)})
        for v in allr[1:]:
            covered = [res((p[0] + a[0], p[1] + a[1])) for a in ((0, 0), v) for p in P]
            if len(set(covered)) != 2 * n:
                continue
            classes, used = [], set()
            for u in allr:
                pair = {u, res((u[0] + v[0], u[1] + v[1]))}
                if len(pair) == 2 and not pair & used:
                    classes.append(frozenset(pair))
                    used |= pair
            if len(classes) == n:
                out.append((H, res, classes))
    return out


def build_layout(basis, C, clamp_rep, unit_reps, order):
    occupied = {basis.site(i, j): ("pix", i * L + j) for i in range(L) for j in range(L)}
    for i, j in itertools.product(range(L), repeat=2):
        occupied[basis.site(i, j, clamp_rep)] = ("clamp", i * L + j)
    slots = []
    for rep in unit_reps:
        s0 = basis.site(0, 0, rep)
        P = [(p[0], p[1]) for p in basis.touched(s0)]
        rel = [(p[0], p[1]) for p in P]
        best = None
        for H, res, classes in tilings(rel, len(rel)):
            ties = 0
            for cls in classes[:1]:
                cs = [
                    basis.site(i, j, rep)
                    for i, j in itertools.product(range(-PAD, L + PAD), repeat=2)
                    if res((i, j)) in cls
                ]
                cset = set(cs)
                ties = sum((s[0] + o[0], s[1] + o[1]) in cset for s in cs for o in OFFSETS)
            if best is None or ties > best[0]:
                best = (ties, H, res, classes)
        if best is None:
            continue
        _, H, res, classes = best
        for cls in classes:
            slots.append((rep, res, cls, H))
    if len(slots) < C:
        raise ValueError(f"only {len(slots)} copy classes tile the grid; {C} units needed")
    slots = [slots[k] for k in order if k < len(slots)][:C]
    units = []
    for c, (rep, res, cls, _H) in enumerate(slots):
        copies = []
        for i, j in itertools.product(range(-PAD, L + PAD), repeat=2):
            if res((i, j)) not in cls:
                continue
            s = basis.site(i, j, rep)
            px = [p[0] * L + p[1] for p in basis.touched(s) if 0 <= p[0] < L and 0 <= p[1] < L]
            if px:
                copies.append((s, sorted(px)))
        for k, (s, _) in enumerate(copies):
            if s in occupied:
                raise ValueError(f"site clash at {s}")
            occupied[s] = ("copy", c, k)
        units.append({"copies": copies, "ties": [], "relay_nodes": [], "relay_edges": []})
    for u in units:
        idx = {s: k for k, (s, _) in enumerate(u["copies"])}
        for a, (s, _) in enumerate(u["copies"]):
            for o in OFFSETS:
                b = idx.get((s[0] + o[0], s[1] + o[1]))
                if b is not None and a < b:
                    u["ties"].append((a, b))
    xs = [s[0] for s in occupied]
    ys = [s[1] for s in occupied]
    box = (min(xs) - MARGIN, max(xs) + MARGIN, min(ys) - MARGIN, max(ys) + MARGIN)
    for u in units:
        route_relays(u, occupied, box)
    return units


def route_relays(u, occupied, box):
    n = len(u["copies"])
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in u["ties"]:
        parent[find(a)] = find(b)
    copy_at = {s: k for k, (s, _) in enumerate(u["copies"])}
    member_sites = {k: [u["copies"][k][0]] for k in range(n)}
    while len({find(k) for k in range(n)}) > 1:
        root = find(0)
        sources = [s for k in range(n) if find(k) == root for s in member_sites[k]]
        prev = {s: None for s in sources}
        q = deque(sources)
        hit = None
        while q and hit is None:
            s = q.popleft()
            for o in OFFSETS:
                t = (s[0] + o[0], s[1] + o[1])
                if t in prev:
                    continue
                k = copy_at.get(t)
                if k is not None and find(k) != root:
                    if s in copy_at:
                        continue
                    prev[t] = s
                    hit = t
                    break
                if t in occupied or not (box[0] <= t[0] <= box[1] and box[2] <= t[1] <= box[3]):
                    continue
                prev[t] = s
                q.append(t)
        if hit is None:
            raise ValueError("relay routing failed: no empty-site path between copy components")
        path = [hit]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        path = path[::-1]
        nodes = []
        for s in path:
            if s in copy_at:
                nodes.append(("c", copy_at[s]))
            elif s in occupied and occupied[s][0] == "relay":
                nodes.append(("r", occupied[s][1]))
            else:
                ridx = len(u["relay_nodes"])
                u["relay_nodes"].append(s)
                occupied[s] = ("relay", ridx)
                nodes.append(("r", ridx))
        for a, b in zip(nodes[:-1], nodes[1:], strict=True):
            u["relay_edges"].append((a, b))
        rroot = find(copy_at[hit])
        parent[rroot] = root
        new_sites = [s for s in path if s not in copy_at]
        member_sites[0] = member_sites[0] + new_sites


def to_pairwise(units, basis, clamp_rep, p, w):
    """Constraint-layer model: pixels, copies, relays free; clamps clamped (probe)."""
    sites, bias = [], []
    for i, j in itertools.product(range(L), repeat=2):
        sites.append(basis.site(i, j))
        bias.append(p["bp"][i * L + j])
    copy_id, relay_id = {}, {}
    for c, u in enumerate(units):
        for k, (s, _) in enumerate(u["copies"]):
            copy_id[c, k] = len(sites)
            sites.append(s)
            bias.append(p["bh"][c] / len(u["copies"]))
    for c, u in enumerate(units):
        for r, s in enumerate(u["relay_nodes"]):
            relay_id[c, r] = len(sites)
            sites.append(s)
            bias.append(0.0)
    nfree = len(sites)
    clamp0 = len(sites)
    for i, j in itertools.product(range(L), repeat=2):
        sites.append(basis.site(i, j, clamp_rep))
        bias.append(0.0)
    edges, weights, kinds = [], [], []
    for i, j in itertools.product(range(L), repeat=2):
        a = i * L + j
        if i + 1 < L:
            edges.append((a, a + L))
            weights.append(p["J"])
            kinds.append("grid")
        if j + 1 < L:
            edges.append((a, a + 1))
            weights.append(p["J"])
            kinds.append("grid")
    for q in range(N):
        edges.append((q, clamp0 + q))
        weights.append(p["g"])
        kinds.append("clamp")
    for c, u in enumerate(units):
        for k, (_, px) in enumerate(u["copies"]):
            for q in px:
                edges.append((q, copy_id[c, k]))
                weights.append(p["W"][q, c])
                kinds.append("copy")
        for a, b in u["ties"]:
            edges.append((copy_id[c, a], copy_id[c, b]))
            weights.append(w)
            kinds.append("tie")

        def nid(x, c=c):
            return copy_id[c, x[1]] if x[0] == "c" else relay_id[c, x[1]]

        for a, b in u["relay_edges"]:
            edges.append((nid(a), nid(b)))
            weights.append(w)
            kinds.append("tie")
    model = PairwiseModel(
        sites=tuple(sites),
        edges=tuple(edges),
        weights=np.array(weights),
        bias=np.array(bias),
        free=tuple(range(nfree)),
        clamped=tuple(range(clamp0, clamp0 + N)),
        clamped_input=tuple(range(N)),
        first_colour=0,
    )
    return model, nfree, kinds


def quantized_sampler(model, nfree, kinds, bits):
    """Quantize under ``split``; fold clamps into an external field (probe)."""
    wts = np.asarray(model.weights, dtype=float)
    fields = np.asarray(model.bias[:nfree], dtype=float)
    caps = None
    if bits is not None:
        cc, cf = cap_for("split", wts, fields, bits)
        wts = quantize_array(wts, cc, bits)
        fields = quantize_array(fields, cf, bits)
        caps = {"coupling": float(cc), "field": float(cf)}
    kinds = np.array(kinds)
    g = float(wts[kinds == "clamp"][0])
    rows, cols, vals = [], [], []
    for (a, b), v, k in zip(model.edges, wts, kinds, strict=True):
        if k == "clamp":
            continue
        rows += [a, b]
        cols += [b, a]
        vals += [v, v]
    Wm = sparse.csr_matrix((vals, (rows, cols)), shape=(nfree, nfree))
    colour = np.array(model.node_colours[:nfree])
    return {"W": Wm, "b": fields, "g": g, "colour": colour, "n": nfree, "caps": caps, "wq": wts}


def collapsed(p, units, model, nfree, kinds, bits):
    """Tie-limit model from the placed, quantized codebook (probe)."""
    m = quantized_sampler(model, nfree, kinds, bits)
    wq, kinds_a = m["wq"], np.array(kinds)
    out = dict(p)
    out["J"] = float(wq[kinds_a == "grid"][0])
    out["g"] = m["g"]
    W = np.zeros((N, p["C"]))
    seen = np.zeros((N, p["C"]), dtype=int)
    k = N
    cw = wq[kinds_a == "copy"]
    e = 0
    bh = np.zeros(p["C"])
    for c, u in enumerate(units):
        for _, px in u["copies"]:
            for q in px:
                W[q, c] = cw[e]
                seen[q, c] += 1
                e += 1
            bh[c] += m["b"][k]
            k += 1
    if not (seen == 1).all():
        raise ValueError("a pixel does not couple to exactly one copy of every unit")
    out["W"], out["bp"], out["bh"] = W, m["b"][:N], bh
    return out, m["caps"]


def unit_slices(units):
    out, k = [], N
    for u in units:
        out.append(list(range(k, k + len(u["copies"]))))
        k += len(u["copies"])
    for u in units:
        out.append(list(range(k, k + len(u["relay_nodes"]))))
        k += len(u["relay_nodes"])
    return out


def oracle_states(units, comp, nfree, n_img):
    C = len(units)
    sl = unit_slices(units)
    S = np.zeros((n_img, nfree))  # pixel columns are set to the noisy image by the sampler
    for c in range(C):
        val = np.where(comp == c, 1.0, -1.0)[:, None]
        S[:, sl[c]] = val
        S[:, sl[C + c]] = val
    return S


# ----------------------------------------------------------------------------
# Sampler (software_simulation): the round-3 probe's gibbs plus the protocol's
# reset and continuation starts, per-node gains and a copy-agreement diagnostic.
# With gains None, start random or oracle, it is the probe's loop draw for draw.
# ----------------------------------------------------------------------------


def gibbs(
    m,
    y,
    K,
    start,
    rng,
    streams,
    per,
    draws,
    oracle=None,
    gains=None,
    reset_every=None,
    copy_units=None,
    n_pix=N,
    return_states=False,
):
    """64-draw majority predictions (images, N) and last-sweep copy agreement per image."""
    n, rows = m["n"], draws * streams
    blocks = [np.flatnonzero(m["colour"] == c) for c in (0, 1)]
    WT = [m["W"][blk].T.tocsr() for blk in blocks]
    G = None if gains is None else [np.asarray(gains)[blk] for blk in blocks]
    S = rng.integers(0, 2, (rows, n)) * 2.0 - 1.0
    pred = np.empty((len(y), n_pix))
    agree = np.full(len(y), np.nan)
    for j in range(per):
        idx = np.arange(streams) * per + j
        if start == "random" or (start == "reset" and j % reset_every == 0):
            S = rng.integers(0, 2, (rows, n)) * 2.0 - 1.0
        elif start == "oracle":
            S = np.repeat(oracle[idx], draws, axis=0).astype(float)
            S[:, :n_pix] = np.repeat(y[idx], draws, axis=0)
        ext = np.tile(m["b"], (rows, 1))
        ext[:, :n_pix] += m["g"] * np.repeat(y[idx], draws, axis=0)
        for _ in range(K):
            for bi, (blk, wt) in enumerate(zip(blocks, WT, strict=True)):
                f = (wt.T @ S.T).T + ext[:, blk]
                if G is not None:
                    f = G[bi][None] * f
                S[:, blk] = np.where(rng.random((rows, len(blk))) < expit(2 * f), 1.0, -1.0)
        if return_states:
            return S
        up = (S[:, :n_pix].reshape(streams, draws, n_pix) > 0).sum(1)
        pred[idx] = np.where(up >= draws // 2, 1.0, -1.0)
        if copy_units is not None:
            frac = []
            for sl in copy_units:
                v = S[:, sl]
                maj = np.where(v.sum(1, keepdims=True) >= 0, 1.0, -1.0)
                frac.append((v == maj).mean(1))
            agree[idx] = np.mean(frac, axis=0).reshape(streams, draws).mean(1)
    return pred, agree


# ----------------------------------------------------------------------------
# Request
# ----------------------------------------------------------------------------


def layout_config(lname, C):
    """(basis, clamp coset, unit cosets, slot order) of a layout, as the round-3 probe."""
    basis = Basis(*BASES[lname])
    if lname == "index5":
        return basis, (0, 1), [(1, 0), (-1, 0), (0, -1)], list(range(9))
    clamp = (-3, 2)
    reps = basis.cosets()
    ureps = [r for r in reps if basis.canon(r) != basis.canon(clamp)]
    return basis, clamp, ureps, list(range(8))


def make_request(**overrides):
    fams_all = {f["name"]: f for f in hr.family_fields(L)}
    names = overrides.pop("families", ["t4_h05_q035", "smooth8_h05_q035"])
    b0 = overrides.pop("b0", {"t4_h05_q035": -5.0, "smooth8_h05_q035": -15.0})
    fams = []
    for i, name in enumerate(names):
        src = fams_all[name]
        fams.append(
            {
                "index": i,
                "name": name,
                "J": src["J"],
                "h0": src["h0"],
                "q": src["q"],
                "components": src["components"],
                "fields": src["fields"],
                "fields_sha256": src["fields_sha256"],
                "b0": b0[name],
                "admission_requires_maj": name.startswith("smooth8"),
                "stage_e_stop": name.startswith("smooth8"),
                "rung": {
                    "t4_h05_q035": {"name": "cnn2_w32_k5", "macs": 1740800},
                    "smooth8_h05_q035": {"name": "cnn2_w32_k5_gpool", "macs": 3430000},
                }.get(name),
            }
        )
    req = {
        "schema": "dtm_exact_posterior_lattice.request.v1",
        "protocol": "docs/experiments/dtm-exact-posterior-lattice.md",
        "protocol_commit": PROTOCOL_COMMIT,
        "L": L,
        "families": fams,
        "layouts": {k: {"basis": [list(v) for v in BASES[k]]} for k in ("index5", "index4")},
        "carried_layout_rule": (
            "fewest free nodes among layouts with at least one admitted tie value; index4 on "
            "equal counts; index4 if neither has one"
        ),
        "w_grid": [1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
        "bits_operating": 8,
        "jitter": 0.1,
        "tol": 0.01,
        "per_seed_bound": 0.015,
        "draws": 64,
        "n_dev": 300,
        "dev_streams": 15,
        "n_test_exact": 1000,
        "streams": 25,
        "images_per_stream": 20,
        "sampler_seeds": [0, 1, 2],
        "K_grid": [64, 256, 1024],
        "K_dev": 1024,
        "K_ref": 4096,
        "reset_every": 5,
        "oracle_agree_bound": 0.01,
        "bootstrap_resamples": 2000,
        "pj_per_mac": [0.1, 1.0, 10.0],
        "cpu_stop_hours": 10.0,
        "prune": 1e-9,
        "replay_rtol": 1e-12,
        "seeds": {
            "test": 20261301,
            "dev": 20261302,
            "dev_sampler": 20261303,
            "stage_s": 20261304,
            "float_ref": 20261305,
            "jitter_words": 20261306,
            "bootstrap": 20261307,
            "oracle_pair": 20261308,
        },
        "preflight": {
            "seed": PROBE_SEED,
            "kernel_chains": 400000,
            "kernel_sweeps": [1, 2, 4],
            "max_z": 5.0,
            "oracle_images": 200,
            "calibration_streams": 25,
            "calibration_K": 1024,
        },
        "sample_definition": (
            "one chain's 64 pixel states after K sweeps of two-colour Gibbs on the placed model "
            "with the noisy image clamped; an estimate is the pixelwise majority of 64 "
            "independent chains, ties to +1"
        ),
    }
    for k, v in overrides.items():
        if k not in req:
            raise KeyError(k)
        req[k] = v
    req["n_test_total"] = req["streams"] * req["images_per_stream"] * len(req["sampler_seeds"])
    req["layouts"] = {
        name: {**spec, **layout_digest_record(req, name)} for name, spec in req["layouts"].items()
    }
    req["constraint_requests"] = {
        key: {"canonical": cr.to_canonical(), "digest": cr.digest()}
        for key, cr in all_constraint_requests(req).items()
    }
    req["sources"] = {name: hr.sha(ROOT / name) for name in SOURCES}
    return req


def layout_digest_record(req, lname):
    out = {}
    for fam in req["families"]:
        basis, clamp, ureps, order = layout_config(lname, fam["components"])
        units = build_layout(basis, fam["components"], clamp, ureps, order)
        nfree = N + sum(len(u["copies"]) + len(u["relay_nodes"]) for u in units)
        out[fam["name"]] = {
            "clamp_coset": list(clamp),
            "unit_cosets": [list(r) for r in ureps],
            "slot_order": order[: fam["components"]],
            "free_nodes": nfree,
            "layout_sha256": canonical_sha256(_units_json(units)),
        }
    return {"families": out}


def _units_json(units):
    return [
        {
            "copies": [[list(s), list(px)] for s, px in u["copies"]],
            "ties": [list(t) for t in u["ties"]],
            "relay_nodes": [list(s) for s in u["relay_nodes"]],
            "relay_edges": [[list(a), list(b)] for a, b in u["relay_edges"]],
        }
        for u in units
    ]


_REASON_JIT = "secondary arm that locates the jitter or precision limit (protocol, rule 7)"
_REASON_FLOAT_Q = "float K = 4096 convergence reference for the placed model (protocol)"
_REASON_FLOAT_J = "float reference runs without jitter (protocol)"
_REASON_DEV = "stage D samples at 8 bits, jitter 0 (protocol)"


def arm_request(req, f, s, arm):
    """Constraint request of one sampled arm; words (20261306, f, s) on jittered arms."""
    fam = req["families"][f]["name"]
    mid = f"dtm-exact-posterior-lattice {fam} seed {s} {arm}"
    if arm in ("op", "pair"):
        return ConstraintRequest(
            model_id=mid,
            bits=req["bits_operating"],
            cap_rule="split",
            jitter=req["jitter"],
            jitter_words=(req["seeds"]["jitter_words"], f, s),
        )
    if arm in ("jit0", "dev"):
        return ConstraintRequest(
            model_id=mid,
            bits=req["bits_operating"],
            cap_rule="split",
            relaxed=("beta_jitter",),
            reasons={"beta_jitter": _REASON_DEV if arm == "dev" else _REASON_JIT},
        )
    if arm == "b6":
        return ConstraintRequest(
            model_id=mid,
            bits=6,
            cap_rule="split",
            relaxed=("beta_jitter",),
            reasons={"beta_jitter": _REASON_JIT},
        )
    if arm == "float":
        return ConstraintRequest(
            model_id=mid,
            relaxed=("quantization", "beta_jitter"),
            reasons={"quantization": _REASON_FLOAT_Q, "beta_jitter": _REASON_FLOAT_J},
        )
    raise ValueError(arm)


def all_constraint_requests(req):
    out = {}
    for f in range(len(req["families"])):
        out[f"f{f}_dev"] = arm_request(req, f, -1, "dev")
        for s in req["sampler_seeds"]:
            for arm in (*ARMS, "float", "pair"):
                out[f"f{f}_s{s}_{arm}"] = arm_request(req, f, s, arm)
    return out


def validate_request(req):
    rebuilt = make_request(
        families=[f["name"] for f in req["families"]],
        b0={f["name"]: f["b0"] for f in req["families"]},
        **{
            k: req[k]
            for k in req
            if k
            not in (
                "families",
                "layouts",
                "constraint_requests",
                "sources",
                "n_test_total",
                "schema",
            )
        },
    )
    if rebuilt != req:
        raise ValueError("request does not rebuild from this runner")


# ----------------------------------------------------------------------------
# Data and models (exact reference)
# ----------------------------------------------------------------------------

_CACHE: dict = {}


def sample_with_components(mix, n, rng):
    """``FieldMixture.sample`` draw for draw, also returning the generating component."""
    comp = rng.integers(0, len(mix.h), n)
    out = np.empty((n, mix.L * mix.L))
    for c in range(len(mix.h)):
        idx = np.flatnonzero(comp == c)
        if len(idx):
            out[idx] = mix.grid.sample(np.repeat(mix.h[c : c + 1], len(idx), 0), rng)
    return out, comp


def family_model(req, f):
    fam = req["families"][f]
    h = np.asarray(fam["fields"], dtype=np.float64)
    return fam, h, build(fam["J"], h, fam["q"], fam["b0"])


def images(req, f, which):
    key = (canonical_sha256({k: req[k] for k in ("families", "n_dev", "n_test_total", "seeds")}),)
    key += (f, which)
    if key not in _CACHE:
        fam = req["families"][f]
        mix = hr.FieldMixture(L, fam["J"], np.asarray(fam["fields"], dtype=np.float64))
        n = req["n_dev"] if which == "dev" else req["n_test_total"]
        rng = np.random.default_rng([req["seeds"][which], f])
        x, comp = sample_with_components(mix, n, rng)
        y = flip(x, fam["q"], rng)
        _CACHE[key] = (mix, x, y, comp)
    return _CACHE[key]


def layout_units(req, f, lname):
    key = ("layout", lname, req["families"][f]["components"])
    if key not in _CACHE:
        C = req["families"][f]["components"]
        basis, clamp, ureps, order = layout_config(lname, C)
        units = build_layout(basis, C, clamp, ureps, order)
        _CACHE[key] = (basis, clamp, units)
    return _CACHE[key]


def placed(req, f, lname, w):
    _, _, base = family_model(req, f)
    basis, clamp, units = layout_units(req, f, lname)
    model, nfree, kinds = to_pairwise(units, basis, clamp, base, w)
    return base, units, model, nfree, kinds


def bayes(mix, x, y, q):
    post = mix.posterior(y, q)
    wrong = (np.where(post > 0.5, 1.0, -1.0) != x).sum(1)
    return post, wrong


def exact_row(p, x, y, bayes_wrong, draws, prune):
    pe, _, _ = exact_equilibrium(p, y, prune=prune)
    mpm = (np.where(pe > 0.5, 1.0, -1.0) != x).sum(1)
    maj = maj_wrong(pe, x, draws)
    return {
        "mpm_gap": float(((mpm - bayes_wrong) / N).mean()),
        "maj_gap": float((maj - bayes_wrong / N).mean()),
        "mpm_wrong": mpm.astype(int).tolist(),
    }, pe


def stage_e_dev(req, f):
    t0 = time.process_time()
    fam = req["families"][f]
    mix, x, y, _ = images(req, f, "dev")
    _, bw = bayes(mix, x, y, fam["q"])
    rows = {}
    for lname in ("index5", "index4"):
        rows[lname] = {}
        for w in req["w_grid"]:
            base, units, model, nfree, kinds = placed(req, f, lname, w)
            pc, caps = collapsed(base, units, model, nfree, kinds, req["bits_operating"])
            row, _ = exact_row(pc, x, y, bw, req["draws"], req["prune"])
            rows[lname][str(w)] = {**row, "caps": caps, "free_nodes": nfree}
    return {
        "bayes_wrong": bw.astype(int).tolist(),
        "noisy_wrong": (y != x).sum(1).astype(int).tolist(),
        "rows": rows,
    }, {"cpu_s": time.process_time() - t0}


def seed_slice(req, s):
    n = req["streams"] * req["images_per_stream"]
    k = req["sampler_seeds"].index(s)
    return slice(k * n, (k + 1) * n)


def stage_e_test(req, f, lname, w):
    t0 = time.process_time()
    fam, h, base = family_model(req, f)
    mix, x, y, comp = images(req, f, "test")
    post, bw = bayes(mix, x, y, fam["q"])
    ne = req["n_test_exact"]
    draws, prune = req["draws"], req["prune"]
    rows = {}
    _, units, model, nfree, kinds = placed(req, f, lname, w)
    # 8-bit row: equilibrium on every test image (per-seed floors), scored on the exact images
    p8, caps8 = collapsed(base, units, model, nfree, kinds, 8)
    pe8, _, _ = exact_equilibrium(p8, y, prune=prune)
    maj8 = maj_wrong(pe8, x, draws)
    mpm8 = (np.where(pe8 > 0.5, 1.0, -1.0) != x).sum(1)
    rows["placed_8"] = {
        "mpm_gap": float(((mpm8[:ne] - bw[:ne]) / N).mean()),
        "maj_gap": float((maj8[:ne] - bw[:ne] / N).mean()),
        "mpm_wrong": mpm8[:ne].astype(int).tolist(),
        "caps": caps8,
    }
    for bits in (None, 6, 4):
        pc, caps = collapsed(base, units, model, nfree, kinds, bits)
        row, _ = exact_row(pc, x[:ne], y[:ne], bw[:ne], draws, prune)
        rows[f"placed_{bits}"] = {**row, "caps": caps}
    for bits in (None, 8, 6, 4):
        pu = quantize_unplaced(base, bits)
        row, _ = exact_row(pu, x[:ne], y[:ne], bw[:ne], draws, prune)
        rows[f"unplaced_{bits}"] = {**row, "caps": pu.get("caps")}
    # per-seed exact floors and conditioned targets (oracle-cell retention diagnostics)
    g = 0.5 * np.log((1 - fam["q"]) / fam["q"])
    S = np.where(np.arange(len(h))[None] == comp[:, None], 1.0, -1.0)
    pm_c, _ = GridIsing(L, p8["J"]).marginals(p8["bp"][None] + S @ p8["W"].T + p8["g"] * y)
    pt_c, _ = GridIsing(L, fam["J"]).marginals(h[comp] + g * y)
    true_maj = maj_wrong(post, x, draws)
    cond_tl = maj_wrong(pm_c, x, draws)
    cond_true = maj_wrong(pt_c, x, draws)
    per_seed = {}
    for s in req["sampler_seeds"]:
        sl = seed_slice(req, s)
        b = bw[sl] / N
        per_seed[str(s)] = {
            "true_posterior_maj_floor_gap": float((true_maj[sl] - b).mean()),
            "tie_limit_8bit_maj_gap": float((maj8[sl] - b).mean()),
            "conditioned_tie_limit_maj_gap": float((cond_tl[sl] - b).mean()),
            "conditioned_true_maj_gap": float((cond_true[sl] - b).mean()),
        }
    return {
        "layout": lname,
        "w": w,
        "bayes_wrong": bw.astype(int).tolist(),
        "noisy_wrong": (y != x).sum(1).astype(int).tolist(),
        "components": comp.astype(int).tolist(),
        "true_posterior_maj_floor_gap_exact_images": float((true_maj[:ne] - bw[:ne] / N).mean()),
        "rows": rows,
        "per_seed": per_seed,
    }, {"cpu_s": time.process_time() - t0}


# ----------------------------------------------------------------------------
# Sampled cells (software_simulation)
# ----------------------------------------------------------------------------


def stage_s_cells(req, f, lname, w):
    cells = []
    i = 0
    for s in req["sampler_seeds"]:
        i = 0
        for arm in ARMS:
            for start in STARTS:
                for K in req["K_grid"]:
                    cells.append(
                        {
                            "kind": "s",
                            "f": f,
                            "s": s,
                            "arm": arm,
                            "start": start,
                            "K": K,
                            "index": i,
                            "layout": lname,
                            "w": w,
                        }
                    )
                    i += 1
        common = {"f": f, "s": s, "K": req["K_ref"], "layout": lname, "w": w}
        cells.append({"kind": "float", "arm": "float", "start": "random", **common})
        for code, start in enumerate(("oracle", "random")):
            cells.append({"kind": "pair", "arm": "pair", "start": start, "code": code, **common})
    return cells


def dev_cells(req, f, lname, ws):
    return [
        {
            "kind": "dev",
            "f": f,
            "arm": "dev",
            "start": "random",
            "K": req["K_dev"],
            "layout": lname,
            "w": w,
            "w_index": req["w_grid"].index(w),
        }
        for w in ws
    ]


def cell_name(c):
    if c["kind"] == "dev":
        return f"dev__f{c['f']}_w{c['w']}"
    return f"cell__f{c['f']}_s{c['s']}_{c['arm']}_{c['start']}_K{c['K']}"


def cell_rng(req, c):
    sd = req["seeds"]
    if c["kind"] == "dev":
        return np.random.default_rng([sd["dev_sampler"], c["f"], c["w_index"]])
    if c["kind"] == "s":
        return np.random.default_rng([sd["stage_s"], c["f"], c["s"], c["index"]])
    if c["kind"] == "float":
        return np.random.default_rng([sd["float_ref"], c["f"], c["s"]])
    return np.random.default_rng([sd["oracle_pair"], c["f"], c["s"], c["code"]])


def run_cell(req, c):
    f = c["f"]
    _, units, model, nfree, kinds = placed(req, f, c["layout"], c["w"])
    key = f"f{f}_dev" if c["kind"] == "dev" else f"f{f}_s{c['s']}_{c['arm']}"
    creq = arm_request(req, f, c.get("s", -1), c["arm"])
    if creq.digest() != req["constraint_requests"][key]["digest"]:
        raise ValueError(f"constraint request mismatch: {key}")
    require_placement(model, creq)
    m = quantized_sampler(model, nfree, kinds, creq.bits)
    gains = request_gains(creq, [nfree])[0] if creq.jitter > 0 else None
    if c["kind"] == "dev":
        _, x, y, comp = images(req, f, "dev")
        streams = req["dev_streams"]
    else:
        _, x, y, comp = images(req, f, "test")
        sl = seed_slice(req, c["s"])
        x, y, comp = x[sl], y[sl], comp[sl]
        streams = req["streams"]
    per = req["images_per_stream"]
    orc = oracle_states(units, comp, nfree, len(x)) if c["start"] == "oracle" else None
    rng = cell_rng(req, c)
    t0 = time.process_time()
    pred, agree = gibbs(
        m,
        y,
        c["K"],
        c["start"],
        rng,
        streams,
        per,
        req["draws"],
        oracle=orc,
        gains=gains,
        reset_every=req["reset_every"],
        copy_units=unit_slices(units)[: len(units)],
    )
    cpu = time.process_time() - t0
    wrong = (pred != x).sum(1).astype(np.uint8)
    return {"maj_wrong": wrong, "agreement": agree.astype(np.float64)}, {
        "cpu_s": cpu,
        "image_sweeps": len(x) * c["K"],
        "chains": streams * req["draws"],
    }


# ----------------------------------------------------------------------------
# Pricing (calibrated_projection)
# ----------------------------------------------------------------------------


def price_sample(req, f, nfree_total, K, start):
    """Per-sample counts and joule band on the host-per-step path (sealed profile).

    ``count`` and ``price`` see an integer block of R samples (R = reset_every for the
    reset start, else 1) and the result is divided by R. Clamp loads are full SRAM
    writes at the top of the band and read-constant writes at the bottom.
    """
    free = int(nfree_total)
    R = req["reset_every"] if start == "reset" else 1
    fw_block = {"continue": 0, "random": free, "reset": free}[start]
    creq = arm_request(req, f, req["sampler_seeds"][0], "op")
    c = count(
        creq,
        logical_pbits=free + N,
        physical_pbits_used=free + N,
        free_pbits=free,
        sweeps=K * R,
        node_reads=N * R,
        node_full_sram_writes=N * R + fw_block,
    )
    p = price(creq, c)
    hi = p["modeled_total_energy_j"] / R
    lo = hi - N * WRITE_J + N * READ_J
    draws = req["draws"]
    return {
        "reads": c.node_reads / R,
        "writes": c.node_full_sram_writes / R,
        "clamp_loads": N,
        "free_node_writes": fw_block / R,
        "updates": c.gibbs_node_updates / R,
        "joule_per_sample_write_constant": hi,
        "joule_per_sample_clamp_at_read": lo,
        "joule_per_estimate_write_constant": draws * hi,
        "joule_per_estimate_clamp_at_read": draws * lo,
        "evidence_class": p["evidence_class"],
        "profile_hash": p["profile_hash"],
        "request_digest": p["request_digest"],
    }


def pricing_check(req):
    """``count``/``price`` against a hand count, every start, both band ends."""
    ok, rows = True, []
    for f in range(len(req["families"])):
        for nfree in (189, 332):
            for K in (*req["K_grid"], req["K_ref"]):
                for start in STARTS:
                    p = price_sample(req, f, nfree, K, start)
                    fw = {"continue": 0.0, "random": float(nfree), "reset": nfree / 5}[start]
                    hi = N * READ_J + (N + fw) * WRITE_J + nfree * K * UPDATE_J
                    lo = N * READ_J + N * READ_J + fw * WRITE_J + nfree * K * UPDATE_J
                    good = (
                        p["reads"] == N
                        and abs(p["writes"] - (N + fw)) <= 1e-12 * (N + fw)
                        and abs(p["free_node_writes"] - fw) <= 1e-12 * max(fw, 1.0)
                        and p["updates"] == nfree * K
                        and abs(p["joule_per_sample_write_constant"] - hi) <= 1e-12 * hi
                        and abs(p["joule_per_sample_clamp_at_read"] - lo) <= 1e-12 * hi
                    )
                    ok = ok and good
                    rows.append({"f": f, "free": nfree, "K": K, "start": start, "ok": bool(good)})
    return {"passed": bool(ok), "rows": len(rows)}


# ----------------------------------------------------------------------------
# Preflight (gating, probe seed) and calibration
# ----------------------------------------------------------------------------


def _layout_check(req, f, lname):
    fam = req["families"][f]
    basis, clamp, units = layout_units(req, f, lname)
    _, _, model, nfree, kinds = placed(req, f, lname, 2.0)
    op = arm_request(req, f, req["sampler_seeds"][0], "op")
    rep = check(model)
    try:
        require_placement(model, op)
        placed_ok = True
    except ValueError:
        placed_ok = False
    cover = all(sorted(q for _, px in u["copies"] for q in px) == list(range(N)) for u in units)
    sites = model.sites
    relay_ok = True
    for u in units:
        for a, b in u["relay_edges"]:

            def site(x, u=u):
                return u["copies"][x[1]][0] if x[0] == "c" else u["relay_nodes"][x[1]]

            d = (site(b)[0] - site(a)[0], site(b)[1] - site(a)[1])
            relay_ok = relay_ok and d in set(OFFSETS)
    digest_ok = (
        canonical_sha256(_units_json(units))
        == req["layouts"][lname]["families"][fam["name"]]["layout_sha256"]
    )
    return {
        "every_unit_covers_each_pixel_once": bool(cover),
        "distinct_sites": len(set(sites)) == len(sites),
        "relay_paths_on_z1_offsets": bool(relay_ok),
        "placement_passed_nothing_relaxed": bool(placed_ok),
        "request_relaxed": list(op.relaxed),
        "max_degree": rep.max_degree,
        "free_nodes": nfree,
        "layout_digest_matches_request": bool(digest_ok),
        "passed": bool(cover and placed_ok and relay_ok and digest_ok and rep.ok),
    }


def _tie_limit_check(req, f):
    """Summed quantized copy biases and common pixel couplings equal the collapsed model."""
    worst = 0.0
    for lname in ("index5", "index4"):
        for w in req["w_grid"]:
            base, units, model, nfree, kinds = placed(req, f, lname, w)
            pc, _ = collapsed(base, units, model, nfree, kinds, req["bits_operating"])
            m = quantized_sampler(model, nfree, kinds, req["bits_operating"])
            sl = unit_slices(units)
            for c in range(len(units)):
                worst = max(worst, abs(m["b"][sl[c]].sum() - pc["bh"][c]))
                cols = m["W"][:N][:, sl[c]].toarray()
                worst = max(worst, float(np.abs(cols.sum(1) - pc["W"][:, c]).max()))
                if not ((cols != 0).sum(1) <= 1).all():
                    worst = max(worst, 1.0)
            worst = max(worst, float(np.abs(m["b"][:N] - pc["bp"]).max()))
    return {"max_abs_diff": worst, "passed": bool(worst <= 1e-12)}


def kernel_submodel(req, f):
    """A 2 x 2 pixel patch, one unit's copies over it and one relay (<= 12 free nodes)."""
    _, _, base = family_model(req, f)
    basis, clamp, units = layout_units(req, f, "index4")
    u = units[0]
    patch = {0, 1, L, L + 1}
    cps = [k for k, (_, px) in enumerate(u["copies"]) if set(px) & patch]
    relay = None
    for a, b in u["relay_edges"]:
        for x, y_ in ((a, b), (b, a)):
            if x[0] == "r" and y_[0] == "c" and y_[1] in cps:
                relay = (x[1], y_[1])
                break
        if relay:
            break
    sites, bias = [], []
    pix = sorted(patch)
    for q in pix:
        sites.append(basis.site(q // L, q % L))
        bias.append(base["bp"][q])
    cid = {}
    for k in cps:
        cid[k] = len(sites)
        sites.append(u["copies"][k][0])
        bias.append(base["bh"][0] / len(u["copies"]))
    rid = None
    if relay is not None:
        rid = len(sites)
        sites.append(u["relay_nodes"][relay[0]])
        bias.append(0.0)
    nfree = len(sites)
    for q in pix:
        sites.append(basis.site(q // L, q % L, clamp))
        bias.append(0.0)
    edges, weights = [], []
    pos = {q: i for i, q in enumerate(pix)}
    for a in pix:
        for b in pix:
            if a < b and (b - a == 1 and a % L != L - 1 or b - a == L):
                edges.append((pos[a], pos[b]))
                weights.append(base["J"])
    for i, _ in enumerate(pix):
        edges.append((i, nfree + i))
        weights.append(base["g"])
    for k in cps:
        for q in u["copies"][k][1]:
            if q in pos:
                edges.append((pos[q], cid[k]))
                weights.append(base["W"][q, 0])
    w = 2.0
    for a, b in u["ties"]:
        if a in cid and b in cid:
            edges.append((cid[a], cid[b]))
            weights.append(w)
    if rid is not None:
        edges.append((rid, cid[relay[1]]))
        weights.append(w)
    model = PairwiseModel(
        sites=tuple(sites),
        edges=tuple(edges),
        weights=np.array(weights),
        bias=np.array(bias),
        free=tuple(range(nfree)),
        clamped=tuple(range(nfree, nfree + len(pix))),
        clamped_input=tuple(range(len(pix))),
        first_colour=0,
    )
    kinds = [
        "clamp" if (b >= nfree) else "x"  # only "clamp" matters to quantized_sampler
        for a, b in edges
    ]
    return model, nfree, kinds


def sampler_check(req):
    """NumPy engine against ``exact_chromatic_kernels`` on a <= 12 free-node sub-model."""
    pf = req["preflight"]
    f = 0
    model, nfree, kinds = kernel_submodel(req, f)
    if nfree > 12:
        raise ValueError("kernel sub-model exceeds 12 free nodes")
    creq = ConstraintRequest(
        model_id="dtm-exact-posterior-lattice preflight kernel",
        bits=8,
        cap_rule="split",
        jitter=0.1,
        jitter_words=(pf["seed"], 3),
    )
    require_placement(model, creq)
    m = quantized_sampler(model, nfree, kinds, 8)
    gains = request_gains(creq, [nfree])[0]
    qmodel = model.with_parameters(m["wq"], np.concatenate([m["b"], np.zeros(model.nodes - nfree)]))
    rng = np.random.default_rng([pf["seed"], 3, 1])
    xin = rng.integers(0, 2, (1, 4)) * 2.0 - 1.0
    kern, states = exact_chromatic_kernels(qmodel, xin, gains)
    chains = pf["kernel_chains"]
    worst = 0.0
    rows = []
    start_state = np.ones((1, nfree))
    start_state[0, :4] = xin[0]
    s0 = np.flatnonzero((states == start_state[0][None]).all(1))[0]
    for K in pf["kernel_sweeps"]:
        init = np.zeros(len(states))
        init[s0] = 1.0
        law = propagate(init, kern, K)[0]
        r = np.random.default_rng([pf["seed"], 3, 2, K])
        # the production engine, oracle-style fixed start (pixels at the input, rest +1)
        S = gibbs(
            m,
            xin,
            K,
            "oracle",
            r,
            1,
            1,
            chains,
            oracle=start_state,
            gains=gains,
            n_pix=4,
            return_states=True,
        )
        idx = ((S > 0).astype(np.int64) << np.arange(nfree)).sum(1)
        emp = np.bincount(idx, minlength=len(states)) / chains
        # states with expected count >= 5 are tested one by one; the rest form one pooled bin
        big = law * chains >= 5.0
        lw = np.append(law[big], law[~big].sum())
        ew = np.append(emp[big], emp[~big].sum())
        se = np.sqrt(np.maximum(lw * (1 - lw), 1e-300) / chains)
        z = np.abs(ew - lw) / se
        worst = max(worst, float(z.max()))
        rows.append({"K": K, "max_z": float(z.max()), "bins": int(big.sum()) + 1})
    return {"free_nodes": nfree, "rows": rows, "max_z": worst, "passed": bool(worst < pf["max_z"])}


def oracle_check(req):
    """Unplaced one-hot restriction reproduces the Bayes posterior (round-1 oracle)."""
    worst = 0.0
    n = req["preflight"]["oracle_images"]
    for f, fam in enumerate(req["families"]):
        h = np.asarray(fam["fields"], dtype=np.float64)
        mix = hr.FieldMixture(L, fam["J"], h)
        rng = np.random.default_rng([req["preflight"]["seed"], 5, f])
        x = mix.sample(n, rng)
        y = flip(x, fam["q"], rng)
        post = mix.posterior(y, fam["q"])
        p = build_onehot(fam["J"], h, fam["q"])
        S = hidden_states(len(h))
        rows = np.flatnonzero(S.sum(1) == 2 - len(h))
        grid = GridIsing(L, fam["J"])
        lw, pm = [], []
        for si in rows:
            fl = p["bp"][None] + S[si] @ p["W"].T + p["g"] * y
            pmi, lz = grid.marginals(fl)
            lw.append(lz + S[si] @ p["bh"])
            pm.append(pmi)
        w = np.exp(np.stack(lw) - logsumexp(np.stack(lw), axis=0))
        oracle = np.einsum("cb,cbn->bn", w, np.stack(pm))
        worst = max(worst, float(np.abs(oracle - post).max()))
    return {"max_abs_diff": worst, "passed": bool(worst <= 1e-12)}


def oracle_start_check(req):
    ok = True
    for f in range(len(req["families"])):
        _, _, _, comp = images(req, f, "test")
        for lname in ("index5", "index4"):
            _, units, _, nfree, _ = placed(req, f, lname, 2.0)
            S = oracle_states(units, comp, nfree, len(comp))
            sl = unit_slices(units)
            C = len(units)
            for c in range(C):
                want = np.where(comp == c, 1.0, -1.0)[:, None]
                ok = ok and bool((S[:, sl[c]] == want).all() and (S[:, sl[C + c]] == want).all())
            ok = ok and len(sum(sl, [])) == nfree - N
    return {"passed": bool(ok)}


def calibrate(req):
    """Full-width (25 streams x 64 chains), full-K cells on one image per stream."""
    pf = req["preflight"]
    out = {}
    for f, fam in enumerate(req["families"]):
        _, units, model, nfree, kinds = placed(req, f, "index4", 2.0)
        creq = arm_request(req, f, req["sampler_seeds"][0], "op")
        m = quantized_sampler(model, nfree, kinds, 8)
        gains = request_gains(creq, [nfree])[0]
        rng = np.random.default_rng([pf["seed"], 7, f])
        streams = pf["calibration_streams"]
        y = rng.integers(0, 2, (streams, N)) * 2.0 - 1.0
        t0 = time.process_time()
        gibbs(m, y, pf["calibration_K"], "random", rng, streams, 1, req["draws"], gains=gains)
        cpu = time.process_time() - t0
        per = cpu / (streams * pf["calibration_K"])  # CPU-s per image per sweep
        t0 = time.process_time()
        mix, x, yy, _ = images(req, f, "dev")
        base = family_model(req, f)[2]
        pc, _ = collapsed(base, units, model, nfree, kinds, 8)
        exact_equilibrium(pc, yy[:50], prune=req["prune"])
        ex = (time.process_time() - t0) / 50
        out[fam["name"]] = {"cpu_s_per_image_sweep": per, "exact_cpu_s_per_image_row": ex}
    n_img = req["streams"] * req["images_per_stream"]
    seeds = len(req["sampler_seeds"])
    sweeps_s = len(ARMS) * len(STARTS) * sum(req["K_grid"]) * n_img * seeds
    sweeps_ref = 3 * req["K_ref"] * n_img * seeds
    sweeps_dev = len(req["w_grid"]) * req["K_dev"] * req["n_dev"]
    rows = 12 * req["n_dev"] + 8 * req["n_test_exact"] + 2 * req["n_test_total"]
    proj = {}
    for name, c in out.items():
        proj[name] = (
            (sweeps_s + sweeps_ref + sweeps_dev) * c["cpu_s_per_image_sweep"]
            + rows * c["exact_cpu_s_per_image_row"]
        ) / 3600
    total = sum(proj.values())
    return {
        "per_family": out,
        "projected_cpu_hours_by_family": proj,
        "projected_cpu_hours": total,
        "within_budget": bool(total <= req["cpu_stop_hours"]),
        "label": "software CPU measurement on this box; full-family plan, both families",
    }


def preflight(req):
    t0 = time.process_time()
    layouts = {
        f"{fam['name']}/{ln}": _layout_check(req, f, ln)
        for f, fam in enumerate(req["families"])
        for ln in ("index5", "index4")
    }
    tie = {fam["name"]: _tie_limit_check(req, f) for f, fam in enumerate(req["families"])}
    res = {
        "layouts": layouts,
        "tie_limit": tie,
        "sampler": sampler_check(req),
        "pricing": pricing_check(req),
        "oracle": oracle_check(req),
        "oracle_start": oracle_start_check(req),
    }
    res["placement_passed"] = all(v["passed"] for v in layouts.values())
    res["passed"] = bool(
        res["placement_passed"]
        and all(v["passed"] for v in tie.values())
        and res["sampler"]["passed"]
        and res["pricing"]["passed"]
        and res["oracle"]["passed"]
        and res["oracle_start"]["passed"]
    )
    cal = calibrate(req)
    return res, cal, {"cpu_s": time.process_time() - t0}


# ----------------------------------------------------------------------------
# Decisions (recomputed on replay from archived counts and exact rows)
# ----------------------------------------------------------------------------


def admitted(req, f, edev):
    fam = req["families"][f]
    tol = req["tol"]
    out = {}
    for lname, rows in edev["rows"].items():
        out[lname] = [
            w
            for w in req["w_grid"]
            if rows[str(w)]["mpm_gap"] <= tol
            and (not fam["admission_requires_maj"] or rows[str(w)]["maj_gap"] <= tol)
        ]
    return out


def carried_layout(req, f, edev, adm):
    nfree = {ln: edev["rows"][ln][str(req["w_grid"][0])]["free_nodes"] for ln in adm}
    cand = [ln for ln in ("index4", "index5") if adm[ln]]
    if not cand:
        return "index4"
    return min(cand, key=lambda ln: (nfree[ln], ln != "index4"))


def stream_gaps(req, wrong, bw, streams):
    d = (np.asarray(wrong, dtype=np.float64) - np.asarray(bw, dtype=np.float64)) / N
    return d.reshape(streams, req["images_per_stream"]).mean(1)


def dev_gap(req, cell_wrong, bw):
    return float(stream_gaps(req, cell_wrong, bw, req["dev_streams"]).mean())


def choose_w(req, f, edev, adm, carried, dev):
    if adm[carried]:
        gaps = {w: dev[str(w)] for w in adm[carried]}
        w = min(gaps, key=lambda v: (gaps[v], v))
        return w, "stage_d_smallest_dev_gap"
    key = "maj_gap" if req["families"][f]["admission_requires_maj"] else "mpm_gap"
    rows = edev["rows"][carried]
    w = min(req["w_grid"], key=lambda v: (rows[str(v)][key], v))
    return w, f"no admitted value: smallest development tie-limit {key}"


def stage_e_verdicts(req, f, et):
    fam = req["families"][f]
    tol = req["tol"]
    p8, pf = et["rows"]["placed_8"], et["rows"]["placed_None"]
    a_fired = p8["mpm_gap"] > tol
    out = {
        "stop_a": {
            "fired": bool(a_fired),
            "placed_8bit_mpm_gap": p8["mpm_gap"],
            "placed_float_mpm_gap": pf["mpm_gap"],
            "mechanism": (
                None if not a_fired else ("precision" if pf["mpm_gap"] <= tol else "bias design")
            ),
        },
        "stage_e_stop": None,
    }
    if fam["stage_e_stop"]:
        fired = p8["maj_gap"] > tol
        out["stage_e_stop"] = {
            "fired": bool(fired),
            "placed_8bit_maj_gap": p8["maj_gap"],
            "mechanism": "bias-design representation" if fired else None,
        }
    return out


def _boot(req, f, s, d):
    rng = np.random.default_rng([req["seeds"]["bootstrap"], f, s])
    n = len(d)
    return d[rng.integers(0, n, (req["bootstrap_resamples"], n))].mean(1)


def cell_stats(req, f, et, wrong_by_seed):
    """Per-seed and pooled gaps with stream-bootstrap intervals; rule 3 reaching."""
    per, boots = {}, []
    for s in req["sampler_seeds"]:
        bw = np.asarray(et["bayes_wrong"])[seed_slice(req, s)]
        d = stream_gaps(req, wrong_by_seed[str(s)], bw, req["streams"])
        b = _boot(req, f, s, d)
        boots.append(b)
        per[str(s)] = {"gap": float(d.mean()), "ci95": np.quantile(b, [0.025, 0.975]).tolist()}
    pooled = float(np.mean([v["gap"] for v in per.values()]))
    pb = np.mean(boots, axis=0)
    reach = pooled <= req["tol"] and all(v["gap"] <= req["per_seed_bound"] for v in per.values())
    return {
        "per_seed": per,
        "pooled_gap": pooled,
        "pooled_ci95": np.quantile(pb, [0.025, 0.975]).tolist(),
        "rule3_reaches": bool(reach),
    }


def paired_upper(req, f, et, a_by_seed, b_by_seed):
    """Paired stream-bootstrap 97.5 percent upper bound of pooled (a - b), pooled over seeds."""
    boots = []
    for s in req["sampler_seeds"]:
        da = np.asarray(a_by_seed[str(s)], dtype=np.float64)
        db = np.asarray(b_by_seed[str(s)], dtype=np.float64)
        d = ((da - db) / N).reshape(req["streams"], req["images_per_stream"]).mean(1)
        boots.append(_boot(req, f, s, d))
    pb = np.mean(boots, axis=0)
    return float(np.quantile(pb, 0.975)), np.quantile(pb, [0.025, 0.975]).tolist()


def evaluate_stage_s(req, f, et, nfree, cells):
    """Rules 3 to 8 from archived per-image counts. ``cells`` maps cell name -> per-seed counts."""
    tol = req["tol"]
    fam = req["families"][f]
    out = {"cells": {}}

    def key(arm, start, K):
        return f"{arm}_{start}_K{K}"

    def wrong(arm, start, K):
        return {
            str(s): cells[f"cell__f{f}_s{s}_{arm}_{start}_K{K}"]["maj_wrong"]
            for s in req["sampler_seeds"]
        }

    for arm in ARMS:
        for start in STARTS:
            for K in req["K_grid"]:
                st = cell_stats(req, f, et, wrong(arm, start, K))
                st["agreement"] = float(
                    np.mean(
                        [
                            np.nanmean(cells[f"cell__f{f}_s{s}_{arm}_{start}_K{K}"]["agreement"])
                            for s in req["sampler_seeds"]
                        ]
                    )
                )
                if start != "random":
                    ub, ci = paired_upper(req, f, et, wrong(arm, start, K), wrong(arm, "random", K))
                    st["carried_minus_random_upper"] = ub
                    st["carried_minus_random_ci95"] = ci
                    st["carried_check_passed"] = bool(ub <= tol)
                    st["reaches"] = bool(st["rule3_reaches"] and ub <= tol)
                else:
                    st["reaches"] = st["rule3_reaches"]
                st["price"] = price_sample(req, f, nfree, K, start)
                out["cells"][key(arm, start, K)] = st
    K4 = req["K_ref"]
    for arm, start in (("float", "random"), ("pair", "oracle"), ("pair", "random")):
        st = cell_stats(req, f, et, wrong(arm, start, K4))
        st["agreement"] = float(
            np.mean(
                [
                    np.nanmean(cells[f"cell__f{f}_s{s}_{arm}_{start}_K{K4}"]["agreement"])
                    for s in req["sampler_seeds"]
                ]
            )
        )
        out["cells"][key(arm, start, K4)] = st
    C = out["cells"]
    # rule 6: operating point
    op_point = None
    for K in req["K_grid"]:
        reach = [st for st in STARTS if C[key("op", st, K)]["reaches"]]
        if reach:
            best = min(reach, key=lambda st: (C[key("op", st, K)]["price"]["writes"], st))
            op_point = {"K": K, "start": best, **C[key("op", best, K)]["price"]}
            op_point["pooled_gap"] = C[key("op", best, K)]["pooled_gap"]
            break
    out["operating_point"] = op_point
    out["stop_b_fired"] = op_point is None
    # rule 7: mechanism
    mech = None
    rule7 = {}
    if op_point is None:
        jit0_k = [
            K
            for K in req["K_grid"]
            if any(C[key("jit0", st, K)]["reaches"] for st in STARTS)
            and not any(C[key("op", st, K)]["reaches"] for st in STARTS)
        ]
        float_reaches = C[key("float", "random", K4)]["rule3_reaches"]
        pair_random_reaches = C[key("pair", "random", K4)]["rule3_reaches"]
        rule7["jit0_reaches_where_op_does_not_at_K"] = jit0_k
        rule7["float_ref_reaches"] = bool(float_reaches)
        rule7["pair_random_K4096_reaches"] = bool(pair_random_reaches)
        g_o = C[key("pair", "oracle", K4)]["pooled_gap"]
        g_r = C[key("pair", "random", K4)]["pooled_gap"]
        d = g_o - g_r
        _, dci = paired_upper(req, f, et, wrong("pair", "oracle", K4), wrong("pair", "random", K4))
        agrees = abs(d) <= req["oracle_agree_bound"]
        rule7["oracle_pair"] = {
            "g_o": g_o,
            "g_r": g_r,
            "d": d,
            "d_ci95": dci,
            "agrees": bool(agrees),
            "g_star": (g_o + g_r) / 2 if agrees else None,
        }
        if jit0_k or (float_reaches and not pair_random_reaches):
            mech = "precision or jitter"
        elif agrees:
            mech = "mixing within budget" if (g_o + g_r) / 2 <= tol else "finite-w representation"
        else:
            mech = "mixing beyond 4096, representation unresolved"
    else:
        g_o = C[key("pair", "oracle", K4)]["pooled_gap"]
        g_r = C[key("pair", "random", K4)]["pooled_gap"]
        rule7["oracle_pair"] = {
            "g_o": g_o,
            "g_r": g_r,
            "d": g_o - g_r,
            "agrees": bool(abs(g_o - g_r) <= req["oracle_agree_bound"]),
        }
    out["rule7"] = rule7
    out["mechanism"] = mech
    # oracle retention diagnostic (does not gate)
    ps = et["per_seed"]
    tgt = float(
        np.mean([ps[str(s)]["conditioned_tie_limit_maj_gap"] for s in req["sampler_seeds"]])
    )
    out["oracle_retention"] = {
        "oracle_cell_pooled_gap": C[key("pair", "oracle", K4)]["pooled_gap"],
        "conditioned_tie_limit_target": tgt,
        "conditioned_true_target": float(
            np.mean([ps[str(s)]["conditioned_true_maj_gap"] for s in req["sampler_seeds"]])
        ),
        "random_cell_pooled_gap": C[key("pair", "random", K4)]["pooled_gap"],
        "gates": False,
    }
    # rule 8: stop (c), smooth8 only
    if fam["stage_e_stop"] and op_point is not None:
        rung = fam["rung"]
        j_hi = op_point["joule_per_estimate_write_constant"]
        out["stop_c"] = {
            "fired": bool(j_hi >= rung["macs"] * 1e-12),
            "joule_per_estimate_write_constant": j_hi,
            "joule_per_estimate_clamp_at_read": op_point["joule_per_estimate_clamp_at_read"],
            "rung": rung,
            "rung_joule": {str(c): rung["macs"] * c * 1e-12 for c in req["pj_per_mac"]},
        }
    else:
        out["stop_c"] = None
    return out


def evaluate_family(req, f, units):
    """Every decision for family f from archived units (dicts already loaded)."""
    fam = req["families"][f]
    edev = units.get("edev")
    if edev is None:
        return {"ran": False}
    adm = admitted(req, f, edev)
    carried = carried_layout(req, f, edev, adm)
    dev = {}
    for w in adm[carried]:
        dev[str(w)] = dev_gap(req, units["dev"][str(w)], edev["bayes_wrong"])
    w_star, source = choose_w(req, f, edev, adm, carried, dev)
    out = {
        "ran": True,
        "family": fam["name"],
        "stage_e_dev": {
            ln: {w: {k: v for k, v in r.items() if k != "mpm_wrong"} for w, r in rows.items()}
            for ln, rows in edev["rows"].items()
        },
        "admitted": adm,
        "carried_layout": carried,
        "dev_gaps": dev,
        "w_star": w_star,
        "w_star_source": source,
        "free_nodes": edev["rows"][carried][str(w_star)]["free_nodes"],
    }
    et = units.get("etest")
    if et is None:
        return out
    out["stage_e_test"] = {
        k: {kk: vv for kk, vv in r.items() if kk != "mpm_wrong"} for k, r in et["rows"].items()
    }
    out["exact_floors"] = {
        "true_posterior_maj_floor_gap": et["true_posterior_maj_floor_gap_exact_images"],
        "per_seed": et["per_seed"],
    }
    n_bayes = req["n_test_total"]
    out["bayes_mpm_error"] = float(np.sum(et["bayes_wrong"]) / (n_bayes * N))
    out["noisy_error"] = float(np.sum(et["noisy_wrong"]) / (n_bayes * N))
    out.update(stage_e_verdicts(req, f, et))
    run_s = not out["stop_a"]["fired"] and not (
        out["stage_e_stop"] is not None and out["stage_e_stop"]["fired"]
    )
    out["stage_s_runs"] = bool(run_s)
    if not run_s:
        out["stop_b"] = {
            "fired": bool(out["stage_e_stop"] and out["stage_e_stop"]["fired"]),
            "at_stage_e": bool(out["stage_e_stop"] and out["stage_e_stop"]["fired"]),
        }
        out["mechanism"] = (
            out["stop_a"]["mechanism"]
            if out["stop_a"]["fired"]
            else out["stage_e_stop"]["mechanism"]
        )
        out["stop_c"] = None
        return out
    if "cells" not in units:
        return out
    s = evaluate_stage_s(req, f, et, out["free_nodes"], units["cells"])
    out["stage_s"] = s
    out["stop_b"] = {"fired": s["stop_b_fired"], "at_stage_e": False}
    out["mechanism"] = s["mechanism"]
    out["operating_point"] = s["operating_point"]
    out["stop_c"] = s["stop_c"]
    out["baseline_joule_per_estimate"] = {
        str(c): fam["rung"]["macs"] * c * 1e-12 for c in req["pj_per_mac"]
    }
    return out


def family_passes(ev):
    """t4 passes (a) and (b) -> smooth8 runs."""
    return bool(
        ev.get("ran")
        and ev.get("stage_s_runs")
        and "stop_b" in ev
        and not ev["stop_b"]["fired"]
        and not ev["stop_a"]["fired"]
    )


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


def _pool_unit(req, kind, key):
    if kind == "preflight":
        res, cal, timing = preflight(req)
        return kind, key, {"json": _js({"preflight": res, "calibration": cal})}, timing
    if kind == "edev":
        res, timing = stage_e_dev(req, key)
        return kind, key, {"json": _js(res)}, timing
    if kind == "etest":
        res, timing = stage_e_test(req, *key)
        return kind, key, {"json": _js(res)}, timing
    arrays, timing = run_cell(req, key)
    return kind, key, arrays, timing


def _js(value):
    return np.frombuffer(canonical_json(value).encode(), dtype=np.uint8)


def unit_name(kind, key):
    if kind == "preflight":
        return "preflight"
    if kind == "edev":
        return f"edev__f{key}"
    if kind == "etest":
        return f"etest__f{key[0]}"
    return cell_name(key)


def _cost(kind, key):
    if kind in ("edev", "etest", "preflight"):
        return 1e9
    return key["K"]


def _run_units(runner, req, jobs, workers, stop_after, counter):
    def record(kind, key, arrays, timing):
        name = unit_name(kind, key)
        runner.save(name, arrays, timing)
        counter["done"] += 1
        runner.status(last_unit=name, units_done=len(runner.index()))
        runner.log(f"unit {name}: {timing.get('cpu_s', 0.0):.1f} CPU-s")
        if stop_after is not None and counter["done"] >= stop_after:
            raise KeyboardInterrupt("stop_after reached (interruption test)")

    if not jobs:
        return
    jobs = sorted(jobs, key=lambda j: -_cost(*j))
    ctx = multiprocessing.get_context("spawn")
    slot = ctx.Value("i", 0)
    with concurrent.futures.ProcessPoolExecutor(
        max(1, min(workers, len(jobs))),
        mp_context=ctx,
        initializer=hr._init_worker,
        initargs=(slot, hr._cpus()),
    ) as ex:
        futures = [ex.submit(_pool_unit, req, k, key) for k, key in jobs]
        try:
            for fut in concurrent.futures.as_completed(futures):
                record(*fut.result())
        except BaseException:
            ex.shutdown(cancel_futures=True)
            raise


def _json_unit(runner, name):
    return json.loads(bytes(runner.load(name)["json"]).decode())


def _cell_arrays(a):
    return {
        "maj_wrong": np.asarray(a["maj_wrong"]).astype(int).tolist(),
        "agreement": [float(v) for v in np.asarray(a["agreement"], dtype=np.float64)],
    }


def _family_units(runner, req, f, verified):
    units = {}
    if f"edev__f{f}" not in verified:
        return units
    units["edev"] = _json_unit(runner, f"edev__f{f}")
    units["dev"] = {}
    for n in verified:
        if n.startswith(f"dev__f{f}_w"):
            w = n.split("_w", 1)[1]
            units["dev"][w] = _cell_arrays(runner.load(n))["maj_wrong"]
    if f"etest__f{f}" in verified:
        units["etest"] = _json_unit(runner, f"etest__f{f}")
    cells = {n: _cell_arrays(runner.load(n)) for n in verified if n.startswith(f"cell__f{f}_")}
    if cells:
        units["cells"] = cells
    return units


def _write_decision(out, name, value):
    value = {**value, "digest": canonical_sha256(value)}
    path = out / f"{name}.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"{name} changed on resume")
    hr.write(path, value)
    return value


def run_study(out, requested=None, resume=False, workers=DEFAULT_WORKERS, stop_after=None):
    out = Path(out)
    hr.require_single_thread_blas()
    if resume:
        if not (out / "request.json").exists():
            raise FileNotFoundError("nothing to resume: request.json is missing")
        if (out / "completion.json").exists():
            raise FileExistsError("study already complete")
        req = json.loads((out / "request.json").read_text())
        current = make_request() if requested is None else requested
        if req != current:
            raise ValueError("stored request differs from this runner's request")
    else:
        if out.exists():
            raise FileExistsError("use a fresh output directory, or --resume")
        req = make_request() if requested is None else requested
        out.mkdir(parents=True)
        (out / "units").mkdir()
        hr.write(out / "request.json", req)
    digest = canonical_sha256(req)
    runner = Runner(out, digest, req["sources"][MODULE])
    hr._copy_sources(out, req)
    ap = out / "attempts.json"
    attempts = json.loads(ap.read_text()) if ap.exists() else []
    attempt = len(attempts) + 1
    attempts.append(
        {
            "attempt": attempt,
            "resume": resume,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "workers": workers,
            "cpus": hr._cpus(),
            "provenance": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        }
    )
    hr.write(ap, attempts)
    runner.log(f"attempt {attempt} ({'resume' if resume else 'fresh'}), {workers} worker(s)")
    started = time.perf_counter()
    counter = {"done": 0}
    try:
        runner.status(phase="preflight", attempt=attempt)
        if "preflight" not in runner.verified():
            _run_units(runner, req, [("preflight", None)], 1, stop_after, counter)
        pre = _json_unit(runner, "preflight")
        if not pre["preflight"]["passed"]:
            runner.log("STOP: preflight failed; see units/preflight")
            raise ValueError("preflight failed; stopped before production")
        hours = pre["calibration"]["projected_cpu_hours"]
        runner.log(f"preflight passed; calibration projects {hours:.2f} CPU-hours")
        if not pre["calibration"]["within_budget"]:
            runner.log("STOP: calibration projects over the CPU stop budget")
            raise ValueError("calibration projects over the CPU budget")
        for f, fam in enumerate(req["families"]):
            if f > 0:
                prev = evaluate_family(
                    req, f - 1, _family_units(runner, req, f - 1, runner.verified())
                )
                if not family_passes(prev):
                    runner.log(
                        f"{fam['name']} does not run: {req['families'][f - 1]['name']} stopped"
                    )
                    break
            name = fam["name"]
            runner.status(phase=f"{name}: stage E-dev")
            if f"edev__f{f}" not in runner.verified():
                _run_units(runner, req, [("edev", f)], 1, stop_after, counter)
            edev = _json_unit(runner, f"edev__f{f}")
            adm = admitted(req, f, edev)
            carried = carried_layout(req, f, edev, adm)
            _write_decision(out, f"admission__f{f}", {"admitted": adm, "carried_layout": carried})
            runner.log(f"{name}: admitted {adm}; carried layout {carried}")
            runner.status(phase=f"{name}: stage D")
            done = runner.verified()
            jobs = [
                ("cell", c)
                for c in dev_cells(req, f, carried, adm[carried])
                if cell_name(c) not in done
            ]
            _run_units(runner, req, jobs, workers, stop_after, counter)
            units = _family_units(runner, req, f, runner.verified())
            dev = {
                str(w): dev_gap(req, units["dev"][str(w)], edev["bayes_wrong"])
                for w in adm[carried]
            }
            w_star, source = choose_w(req, f, edev, adm, carried, dev)
            _write_decision(
                out, f"w_star__f{f}", {"w_star": w_star, "source": source, "dev_gaps": dev}
            )
            runner.log(f"{name}: w* = {w_star} ({source}); dev gaps {dev}")
            runner.status(phase=f"{name}: stage E-test")
            if f"etest__f{f}" not in runner.verified():
                _run_units(runner, req, [("etest", (f, carried, w_star))], 1, stop_after, counter)
            ev = evaluate_family(req, f, _family_units(runner, req, f, runner.verified()))
            _write_decision(
                out,
                f"stage_e__f{f}",
                {k: ev[k] for k in ("stop_a", "stage_e_stop", "stage_s_runs")},
            )
            runner.log(
                f"{name}: stop (a) {ev['stop_a']}; stage E stop {ev['stage_e_stop']}; "
                f"stage S runs {ev['stage_s_runs']}"
            )
            if not ev["stage_s_runs"]:
                continue
            runner.status(phase=f"{name}: stage S")
            done = runner.verified()
            jobs = [
                ("cell", c)
                for c in stage_s_cells(req, f, carried, w_star)
                if cell_name(c) not in done
            ]
            runner.log(f"{name}: stage S, {len(jobs)} cells to run")
            _run_units(runner, req, jobs, workers, stop_after, counter)
        runner.status(phase="assembling")
        assemble(out, req, runner, time.perf_counter() - started)
    except BaseException as error:
        runner.status(
            phase="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        runner.log(f"stopped: {error!r}")
        raise
    runner.status(phase="replaying")
    completion = replay(out, light=False, workers=workers)
    runner.status(phase="complete", error=None, traceback=None)
    runner.log(f"completion written after replay ({completion['replay_seconds']:.1f} s)")
    return completion


def assemble(out, req, runner, seconds):
    verified = runner.verified()
    index = runner.index()
    pre = _json_unit(runner, "preflight")
    fams = {}
    for f in range(len(req["families"])):
        u = _family_units(runner, req, f, verified)
        if u:
            fams[str(f)] = u
    result = {
        "request_digest": canonical_sha256(req),
        "evidence_class": {
            "bayes_tie_limit_conditioned_floors": "exact_reference",
            "gibbs_cells": (
                "software_simulation (NumPy two-colour Gibbs, CPU; not the lowered THRML program)"
            ),
            "joules": "calibrated_projection",
            "cpu_seconds": "software CPU measurement on this box (not evidence of hardware)",
        },
        "hardware_claim": False,
        "preflight": pre["preflight"],
        "families": fams,
    }
    result["result_digest"] = canonical_sha256(result)
    evaluation = {
        req["families"][int(f)]["name"]: evaluate_family(req, int(f), u) for f, u in fams.items()
    }
    timings = {
        "label": "CPU seconds (time.process_time) on this box; excluded from the result digest",
        "units": {k: v["timing"] for k, v in index.items()},
        "calibration": pre["calibration"],
        "generation_seconds_this_attempt": seconds,
    }
    study = {**result, "evaluation": evaluation, "timings": timings}
    write_gz_json(out / "study.json.gz", study)
    attempts = json.loads((out / "attempts.json").read_text())
    hr.write(
        out / "provenance.json",
        {
            "schema": "dtm_exact_posterior_lattice.provenance.v1",
            "request_digest": result["request_digest"],
            "attempts": attempts,
            "runtime": collect_runtime_provenance(ROOT).model_dump(mode="json"),
        },
    )
    runner.log("assembled study.json.gz and provenance.json")


# ----------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------


def _close_tree(actual, expected, rtol, path="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"replay mismatch (keys): {path}")
        for k in expected:
            _close_tree(actual[k], expected[k], rtol, f"{path}/{k}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"replay mismatch (length): {path}")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            _close_tree(a, b, rtol, f"{path}/{i}")
    elif isinstance(expected, float):
        if abs(actual - expected) > rtol * max(1.0, abs(expected)):
            raise ValueError(f"replay mismatch: {path}")
    elif actual != expected:
        raise ValueError(f"replay mismatch: {path}")


def _exact_jobs(req, study):
    jobs = []
    for f, u in study["families"].items():
        f = int(f)
        jobs.append(("edev", f))
        if "etest" in u:
            jobs.append(("etest", (f, u["etest"]["layout"], u["etest"]["w"])))
    return jobs


def _prefix_jobs(req, study):
    jobs = []
    for f, u in study["families"].items():
        f = int(f)
        ev = study["evaluation"][req["families"][f]["name"]]
        if u.get("dev"):
            w = float(sorted(u["dev"], key=float)[0])
            c = dev_cells(req, f, ev["carried_layout"], [w])[0]
            jobs.append(("cell", c))
        if u.get("cells"):
            c = next(
                c
                for c in stage_s_cells(req, f, ev["carried_layout"], ev["w_star"])
                if c["kind"] == "s" and c["K"] == req["K_grid"][0]
            )
            jobs.append(("cell", c))
        if jobs:
            break
    return jobs


def _in_pool(req, jobs, workers):
    got = {}
    if not jobs:
        return got
    ctx = multiprocessing.get_context("spawn")
    slot = ctx.Value("i", 0)
    with concurrent.futures.ProcessPoolExecutor(
        max(1, min(workers, len(jobs))),
        mp_context=ctx,
        initializer=hr._init_worker,
        initargs=(slot, hr._cpus()),
    ) as ex:
        for kind, key, arrays, _ in ex.map(
            _pool_unit, *zip(*[(req, k, key) for k, key in jobs], strict=True)
        ):
            got[unit_name(kind, key)] = arrays
    return got


def replay(out, light=False, workers=1):
    out = Path(out)
    hr.require_single_thread_blas()
    started = time.perf_counter()
    req = json.loads((out / "request.json").read_text())
    study = read_gz_json(out / "study.json.gz")
    validate_request(req)
    digest = canonical_sha256(req)
    if study["request_digest"] != digest:
        raise ValueError("request digest mismatch")
    for name, dg in req["sources"].items():
        for path in (ROOT / name, out / "source" / name):
            if hr.sha(path) != dg:
                raise ValueError(f"source changed: {name}")
    for key, cr in all_constraint_requests(req).items():
        v = req["constraint_requests"][key]
        if cr.digest() != v["digest"] or cr.to_canonical() != v["canonical"]:
            raise ValueError(f"constraint request does not rebuild: {key}")
    core = {k: study[k] for k in study if k not in ("evaluation", "timings", "result_digest")}
    if canonical_sha256(core) != study["result_digest"]:
        raise ValueError("result digest mismatch")
    # placement of both layouts, every family, re-verified
    for f in range(len(req["families"])):
        for ln in ("index5", "index4"):
            if not _layout_check(req, f, ln)["passed"]:
                raise ValueError(f"placement fails on replay: f{f} {ln}")
    if not pricing_check(req)["passed"]:
        raise ValueError("pricing check failed on replay")
    # exact stages recomputed (Bayes errors, tie-limit rows, conditioned targets, floors)
    rtol = req["replay_rtol"]
    exact = _in_pool(req, _exact_jobs(req, study), workers)
    for name, arrays in exact.items():
        f = name.split("__f")[1]
        kind = "edev" if name.startswith("edev") else "etest"
        got = json.loads(bytes(arrays["json"]).decode())
        _close_tree(got, study["families"][f][kind], rtol, name)
    # cell counts in range and the expected cell set
    n_img = req["streams"] * req["images_per_stream"]
    for f, u in study["families"].items():
        for name, a in u.get("cells", {}).items():
            if len(a["maj_wrong"]) != n_img or not all(0 <= v <= N for v in a["maj_wrong"]):
                raise ValueError(f"bad counts: {name}")
        for w, a in u.get("dev", {}).items():
            if len(a) != req["n_dev"] or not all(0 <= v <= N for v in a):
                raise ValueError(f"bad dev counts: f{f} w{w}")
    evaluation = {
        req["families"][int(f)]["name"]: evaluate_family(req, int(f), u)
        for f, u in study["families"].items()
    }
    check_equal(evaluation, study["evaluation"])
    # the run order: smooth8 only after t4 passed; units only where the stages allowed
    expected_dev, expected_cells = 0, 0
    for f in range(len(req["families"])):
        name = req["families"][f]["name"]
        ev = evaluation.get(name)
        if f > 0 and not family_passes(evaluation[req["families"][f - 1]["name"]]):
            if ev is not None:
                raise ValueError(f"{name} ran although the previous family stopped")
            continue
        if ev is None:
            raise ValueError(f"{name} is missing")
        u = study["families"][str(f)]
        if sorted(u.get("dev", {}), key=float) != [
            str(w) for w in ev["admitted"][ev["carried_layout"]]
        ]:
            raise ValueError(f"{name}: development cells differ from admission")
        expected_dev += len(u.get("dev", {}))
        want = (
            sorted(cell_name(c) for c in stage_s_cells(req, f, ev["carried_layout"], ev["w_star"]))
            if ev["stage_s_runs"]
            else []
        )
        if sorted(u.get("cells", {})) != want:
            raise ValueError(f"{name}: stage S cell set differs")
        expected_cells += len(want)
        if u["etest"]["layout"] != ev["carried_layout"] or u["etest"]["w"] != ev["w_star"]:
            raise ValueError(f"{name}: stage E-test not at the carried layout and w*")
    prefix = {"run": not light}
    if not light:
        jobs = _prefix_jobs(req, study)
        got = _in_pool(req, jobs, 1)
        for kind, key in jobs:
            name = unit_name(kind, key)
            arrays = got[name]
            f = str(key["f"])
            u = study["families"][f]
            rec = _cell_arrays(arrays)
            if key["kind"] == "dev":
                same = rec["maj_wrong"] == u["dev"][str(key["w"])]
            else:
                same = rec == u["cells"][name]
            prefix[name] = bool(same)
            if not same:
                raise ValueError(f"prefix check failed: {name}")
        prefix["passed"] = bool(jobs)
    fams_run = len(study["families"])
    exact_rows = 0
    for u in study["families"].values():
        exact_rows += sum(len(rows) for rows in u["edev"]["rows"].values())
        exact_rows += len(u["etest"]["rows"]) if "etest" in u else 0
    split = {
        req["families"][int(f)]["name"]: {
            "development_cells": len(u.get("dev", {})),
            "stage_s_and_reference_cells": len(u.get("cells", {})),
        }
        for f, u in study["families"].items()
    }
    verdicts = {}
    for name, ev in evaluation.items():
        verdicts[name] = {
            "admitted": ev["admitted"],
            "carried_layout": ev["carried_layout"],
            "w_star": ev["w_star"],
            "w_star_source": ev["w_star_source"],
            "stop_a": ev["stop_a"]["fired"],
            "stop_b": ev["stop_b"]["fired"],
            "stop_b_at_stage_e": ev["stop_b"]["at_stage_e"],
            "stop_c": None if ev.get("stop_c") is None else ev["stop_c"]["fired"],
            "mechanism": ev.get("mechanism"),
            "oracle_pair": ev.get("stage_s", {}).get("rule7", {}).get("oracle_pair"),
            "operating_point": ev.get("operating_point"),
        }
    completion = {
        "status": "dtm_exact_posterior_lattice_complete",
        "request_digest": digest,
        "result_digest": study["result_digest"],
        "families_run": fams_run,
        "exact_rows": exact_rows,
        "sampled_cells": expected_dev + expected_cells,
        "sampled_cells_by_family": split,
        "preflight_passed": bool(study["preflight"]["passed"]),
        "placement_passed": bool(study["preflight"]["placement_passed"]),
        "references_recomputed": True,
        "prefix_check_passed": bool(prefix.get("passed", False)),
        "light_replay": bool(light),
        "replayed": True,
        "verdicts": verdicts,
        "evidence_class": study["evidence_class"],
        "hardware_claim": False,
        "replay_seconds": time.perf_counter() - started,
    }
    if not light:
        hr.write(out / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    parser.add_argument("--replay", action="store_true", help="replay a finished run only")
    parser.add_argument("--light", action="store_true", help="replay without the prefix check")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--request", help="request JSON (tests only)")
    parser.add_argument("--stop-after", type=int, help="interrupt after N new units (tests only)")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.output_dir, light=args.light, workers=args.workers), indent=2))
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
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
