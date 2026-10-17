"""Index-4 layout check and development scan for P-0008 round 3 (exploration).

Round 2 placed each template unit as tied copies on the index-5 pixel basis
(2, 1), (-1, 2) and joined parallel copy chains with direct ties that broke
placement. This probe:

1. builds copy layouts for a pixel basis generically: every non-pixel coset
   site touches a fixed set P of pixels; a copy class is a coset of a
   sublattice T of the pixel lattice with index |P| such that P + T tiles the
   grid, chosen to maximise direct Z1 ties between copies of one class;
2. joins each unit's copies into one connected component with **relay paths**
   routed by breadth-first search over empty sites on Z1 offsets (relay
   nodes: free, zero bias, tie w on every relay edge);
3. builds the constraint layer's ``PairwiseModel`` with sites and checks it
   with ``require_placement`` under a request that relaxes nothing (8 bits,
   ``split``, 10 percent jitter);
4. for every layout that places, computes the tie-limit (w -> infinity)
   placed codebook's exact 64-draw gap (stop (a)), and runs short NumPy
   two-colour Gibbs cells: random start and an oracle start (copies and relays
   at the generating template, pixels at the noisy image), K = 1024.

Both bases are run so the index-5 layout is also checked with real relays.
Evidence: exact gaps are ``exact_reference`` (float64); Gibbs errors are
``software_simulation`` (NumPy, CPU; not the lowered THRML program). Run:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
    uv run python docs/research/dtm_exact_posterior_lattice_layout4.py --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dtm_exact_posterior_lattice_placed as placed  # noqa: E402
import dtm_exact_posterior_lattice_probe as probe  # noqa: E402

from thermo_lab.constraints.lattice import OFFSETS, check, require_placement  # noqa: E402
from thermo_lab.constraints.model import PairwiseModel  # noqa: E402
from thermo_lab.constraints.quantize import cap_for, quantize_array  # noqa: E402
from thermo_lab.constraints.request import ConstraintRequest  # noqa: E402

L, N, DRAWS = probe.L, probe.N, 64
NB = set(OFFSETS)
BASES = {"index5": ((2, 1), (-1, 2)), "index4": ((-4, -1), (0, -1))}
MARGIN = 14  # relay search box: image site bounding box plus this margin
PAD = 10  # copy-anchor window beyond the image; tiles reach at most 7 pixels


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
        """Non-pixel coset representatives that are Z1 offsets, one per coset."""
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
        for b in range(a):  # b reduced modulo the first column (a, 0)
            yield np.array([[a, b], [0, d]])  # columns: (a, 0), (b, d)


def _residue(H):
    Hi = np.linalg.inv(H)

    def res(p):
        c = Hi @ np.array(p, dtype=float)
        f = c - np.floor(c + 1e-9)
        r = H @ f
        return (round(r[0], 6), round(r[1], 6))

    return res


def tilings(P, n):
    """Periodic tilings of the pixel lattice by translates of P.

    Returns (H, res, classes): a period sublattice H, its residue map, and a
    partition of the residues into n classes; each class is a set of anchor
    residues whose P-translates cover every pixel exactly once. Searched: lattice
    tilings (H of index n, one anchor residue per class) and two-anchor periodic
    tilings (H of index 2n, anchors {0, v}); the second catches tiles such as
    {(0, 0), (1, -3), (1, -5), (1, -7)} that tile Z^2 but not by a lattice.
    """
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
            for u in allr:  # greedy partition into translates {u, u + v}
                pair = {u, res((u[0] + v[0], u[1] + v[1]))}
                if len(pair) == 2 and not pair & used:
                    classes.append(frozenset(pair))
                    used |= pair
            if len(classes) == n:
                out.append((H, res, classes))
    return out


def build_layout(basis, C, clamp_rep, unit_reps, order):
    """Units placed as copy classes in ``unit_reps`` cosets; relays by BFS."""
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
            continue  # this coset's pixel tile admits no lattice or two-anchor tiling
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
    """Join the unit's copy components with shortest relay paths through empty sites."""
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
    # node ids: copies 0..n-1, relays n.. ; sites of component members act as BFS sources
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
                        continue  # copy-to-copy: a direct tie would already be in ``ties``
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
        path = path[::-1]  # source member ... hit copy
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
    """Constraint-layer model: pixels, copies, relays free; clamps clamped."""
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


def operating_request(name):
    return ConstraintRequest(
        model_id=name,
        bits=8,
        cap_rule="split",
        jitter=0.1,
        jitter_words=(probe.PROBE_SEED, 51),
    )


def quantized_sampler(model, nfree, kinds, bits):
    """Quantize couplings and fields under ``split``; fold clamps into an external field."""
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
    """Tie-limit model from the placed, quantized codebook (relays carry zero bias)."""
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


def gibbs(m, y, K, start, rng, streams, per, oracle=None):
    n, rows = m["n"], DRAWS * streams
    blocks = [np.flatnonzero(m["colour"] == c) for c in (0, 1)]
    WT = [m["W"][blk].T.tocsr() for blk in blocks]
    S = rng.integers(0, 2, (rows, n)) * 2.0 - 1.0
    pred = np.empty((len(y), N))
    for j in range(per):
        idx = np.arange(streams) * per + j
        if start == "random":
            S = rng.integers(0, 2, (rows, n)) * 2.0 - 1.0
        elif start == "oracle":
            S = np.repeat(oracle[idx], DRAWS, axis=0).astype(float)
            S[:, :N] = np.repeat(y[idx], DRAWS, axis=0)
        ext = np.tile(m["b"], (rows, 1))
        ext[:, :N] += m["g"] * np.repeat(y[idx], DRAWS, axis=0)
        for _ in range(K):
            for blk, wt in zip(blocks, WT, strict=True):
                f = (wt.T @ S.T).T + ext[:, blk]
                S[:, blk] = np.where(rng.random((rows, len(blk))) < expit(2 * f), 1.0, -1.0)
        up = (S[:, :N].reshape(streams, DRAWS, N) > 0).sum(1)
        pred[idx] = np.where(up >= DRAWS // 2, 1.0, -1.0)
    return pred


def unit_slices(units):
    out, k = [], N
    for u in units:
        out.append(list(range(k, k + len(u["copies"]))))
        k += len(u["copies"])
    for u in units:
        out_r = list(range(k, k + len(u["relay_nodes"])))
        k += len(u["relay_nodes"])
        out.append(out_r)
    return out


def oracle_states(units, comp, nfree, n_img):
    C = len(units)
    sl = unit_slices(units)
    S = np.empty((n_img, nfree))
    for c in range(C):
        val = np.where(comp == c, 1.0, -1.0)[:, None]
        S[:, sl[c]] = val
        S[:, sl[C + c]] = val
    return S


def layout_summary(units, model, nfree, req):
    rep = check(model)
    try:
        require_placement(model, req)
        ok, err = True, None
    except ValueError as exc:
        ok, err = False, str(exc)
    cover = [sorted(q for _, px in u["copies"] for q in px) == list(range(N)) for u in units]
    return {
        "every_unit_covers_each_pixel_once": all(cover),
        "placement_passed_nothing_relaxed": ok,
        "placement_error": err,
        "request_relaxed": list(req.relaxed),
        "edges": rep.edges,
        "offset_violations": rep.offset_violations,
        "colour_violations": rep.colour_violations,
        "max_degree": rep.max_degree,
        "pixel_degree": int(model.degrees()[:N].max()),
        "free_nodes": nfree,
        "copies_per_unit": [len(u["copies"]) for u in units],
        "direct_ties_per_unit": [len(u["ties"]) for u in units],
        "relay_nodes_per_unit": [len(u["relay_nodes"]) for u in units],
        "relay_edges_per_unit": [len(u["relay_edges"]) for u in units],
        "max_pixels_per_copy": max(len(px) for u in units for _, px in u["copies"]),
        "copy_free_nodes": int(sum(len(u["copies"]) for u in units)),
        "relay_free_nodes": int(sum(len(u["relay_nodes"]) for u in units)),
    }


def configs(basis_name, C):
    """(clamp coset rep, unit coset reps, slot order) candidates for the basis."""
    basis = Basis(*BASES[basis_name])
    reps = basis.cosets()
    out = []
    if basis_name == "index5":
        out.append(((0, 1), [(1, 0), (-1, 0), (0, -1)], list(range(9))))
    else:
        for clamp in reps:
            units = [r for r in reps if basis.canon(r) != basis.canon(clamp)]
            fill = list(range(8))  # coset 1 classes then coset 2 classes
            spread = [0, 4, 1, 5, 2, 6, 3, 7]
            out.append((clamp, units, fill))
            if C < 8:
                out.append((clamp, units, spread))
    return basis, out


def run(args):
    t_start = time.process_time()
    res = {
        "label": "exploration, not evidence",
        "draws_per_estimate": DRAWS,
        "evidence_classes": {
            "tie_limit_gaps": "exact_reference (float64)",
            "gibbs": (
                "software_simulation (NumPy two-colour Gibbs, CPU; not the lowered THRML program)"
            ),
        },
        "families": [],
    }
    fams = {"t4_h05_q035": -5.0, "smooth8_h05_q035": -15.0}
    fams = {k: v for k, v in fams.items() if k in args.families}
    for name, b0 in fams.items():
        J, h, q = probe.family(name)
        C = len(h)
        n_img = args.streams * args.per
        mix, x, y = probe.test_set(name, 1000 if C <= 4 else 500, 1)
        comp_all = np.random.default_rng([probe.PROBE_SEED, 1, list(probe.FAMILIES).index(name)])
        comp = comp_all.integers(0, C, len(x))[:n_img]
        x, y = x[:n_img], y[:n_img]
        post = mix.posterior(y, q)
        bayes = (np.where(post > 0.5, 1.0, -1.0) != x).mean(1)
        base = probe.build(J, h, q, "bias", "centered", b0=b0)
        frec = {
            "family": name,
            "b0": b0,
            "images": n_img,
            "bayes_mpm": float(bayes.mean()),
            "true_posterior_maj64_floor_gap": float(
                (placed.maj_wrong(post, x, DRAWS) - bayes).mean()
            ),
            "layouts": [],
        }
        pu, _, _ = probe.exact_equilibrium(probe.quantize(base, 8), y)
        frec["unplaced_8bit_maj64_gap"] = float((placed.maj_wrong(pu, x, DRAWS) - bayes).mean())
        cp = probe.component_posterior(mix, y, q)
        frec["oracle_component_check"] = {
            "mean_posterior_weight_of_generating_component": float(
                cp[np.arange(n_img), comp].mean()
            ),
            "argmax_agrees": float((cp.argmax(1) == comp).mean()),
        }
        built = []
        for bname in ("index5", "index4"):
            basis, cands = configs(bname, C)
            for clamp, ureps, order in cands:
                lrec = {"basis": bname, "basis_vectors": BASES[bname], "clamp_coset": clamp}
                lrec["unit_cosets"] = ureps
                lrec["slot_order"] = order[:C]
                try:
                    units = build_layout(basis, C, clamp, ureps, order)
                except ValueError as exc:
                    lrec["built"] = False
                    lrec["error"] = str(exc)
                    frec["layouts"].append(lrec)
                    print(name, bname, clamp, "not built:", exc, flush=True)
                    continue
                lrec["built"] = True
                model, nfree, kinds = to_pairwise(units, basis, clamp, base, 2.0)
                lrec.update(layout_summary(units, model, nfree, operating_request(name)))
                print(
                    name, bname, clamp, lrec["free_nodes"], lrec["placement_passed_nothing_relaxed"]
                )
                frec["layouts"].append(lrec)
                if (
                    lrec["placement_passed_nothing_relaxed"]
                    and lrec["every_unit_covers_each_pixel_once"]
                ):
                    built.append((lrec, basis, clamp, units))
        # scanned: the index-5 layout and the index-4 layout with the fewest free nodes
        i5 = [b for b in built if b[0]["basis"] == "index5"]
        i4 = sorted(
            [b for b in built if b[0]["basis"] == "index4"], key=lambda b: b[0]["free_nodes"]
        )
        for tag, (lrec, basis, clamp, units) in enumerate(i5[:1] + i4[:1]):
            bname = lrec["basis"]
            lrec["scanned"] = True
            if True:
                lrec["tie_limit"] = {}
                for w in args.ws:
                    model, nfree, kinds = to_pairwise(units, basis, clamp, base, w)
                    pc, caps = collapsed(base, units, model, nfree, kinds, 8)
                    pe, _, _ = probe.exact_equilibrium(pc, y)
                    mpm = (np.where(pe > 0.5, 1.0, -1.0) != x).mean(1) - bayes
                    lrec["tie_limit"][str(w)] = {
                        "caps": caps,
                        "mpm_gap": float(mpm.mean()),
                        "mpm_gap_se": float(mpm.std(ddof=1) / np.sqrt(n_img)),
                        "maj64_gap": float((placed.maj_wrong(pe, x, DRAWS) - bayes).mean()),
                    }
                    print(name, bname, "tie-limit", w, lrec["tie_limit"][str(w)], flush=True)
                lrec["cells"] = []
                plan = args.cells_t4 if C <= 4 else args.cells_smooth8
                if bname == "index5":
                    plan = [c for c in plan if c[0] == 2.0]
                for w, start in plan:
                    model, nfree, kinds = to_pairwise(units, basis, clamp, base, w)
                    m = quantized_sampler(model, nfree, kinds, 8)
                    orc = oracle_states(units, comp, nfree, n_img) if start == "oracle" else None
                    rng = np.random.default_rng(
                        [
                            probe.PROBE_SEED,
                            41,
                            C,
                            tag,
                            int(w * 100),
                            ["random", "oracle"].index(start),
                        ]
                    )
                    t0 = time.process_time()
                    pred = gibbs(m, y, args.K, start, rng, args.streams, args.per, orc)
                    d = ((pred != x).mean(1) - bayes).reshape(args.streams, args.per).mean(1)
                    br = np.random.default_rng([probe.PROBE_SEED, 42])
                    bs = d[br.integers(0, args.streams, (2000, args.streams))].mean(1)
                    cell = {
                        "tie_w": w,
                        "start": start,
                        "K": args.K,
                        "bits": 8,
                        "jitter": 0.0,
                        "caps": m["caps"],
                        "gap_to_bayes": float(d.mean()),
                        "gap_ci95": np.quantile(bs, [0.025, 0.975]).tolist(),
                        "cpu_s": time.process_time() - t0,
                    }
                    lrec["cells"].append(cell)
                    print(
                        name,
                        bname,
                        w,
                        start,
                        round(cell["gap_to_bayes"], 4),
                        round(cell["cpu_s"], 1),
                        flush=True,
                    )
                    Path(args.out).write_text(
                        json.dumps(
                            res | {"families": res["families"] + [frec]}, indent=1, default=float
                        )
                        + "\n"
                    )
        res["families"].append(frec)
        Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    res["total_cpu_s"] = time.process_time() - t_start
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout-only", action="store_true")
    ap.add_argument("--families", nargs="+", default=["t4_h05_q035", "smooth8_h05_q035"])
    a = ap.parse_args()
    a.ws = (1.5, 2.0, 2.5, 3.0)
    a.K, a.streams, a.per = 1024, 10, 20
    a.cells_t4 = [
        (1.5, "random"),
        (2.0, "random"),
        (2.5, "random"),
        (3.0, "random"),
        (2.0, "oracle"),
    ]
    a.cells_smooth8 = [(2.0, "random"), (2.0, "oracle")]
    if a.layout_only:
        out = {}
        for name, C in (("t4", 4), ("smooth8", 8)):
            base = probe.build(*probe.family(f"{name}_h05_q035"), "bias", "centered", b0=-5.0)
            for bname in ("index5", "index4"):
                basis, cands = configs(bname, C)
                for clamp, ureps, order in cands:
                    key = f"{name}/{bname}/clamp{clamp}/order{order[:C]}"
                    try:
                        units = build_layout(basis, C, clamp, ureps, order)
                    except ValueError as exc:
                        out[key] = {"built": False, "error": str(exc)}
                        continue
                    model, nfree, _ = to_pairwise(units, basis, clamp, base, 2.0)
                    out[key] = layout_summary(units, model, nfree, operating_request(name))
        print(json.dumps(out, indent=1))
        return
    run(a)


if __name__ == "__main__":
    main()
