"""Placed tied-copy check for P-0008 round 2 (exploration, not evidence).

Round 1 probed only the unplaced model. Here each template unit is spread over
the 16-offset lattice as tied copies and sampled with NumPy two-colour Gibbs.

Layout (checked by ``layout_report``): pixel (i, j) sits at site
i*(2, 1) + j*(-1, 2); the four grid offsets are Z1 offsets. The other four
cosets of that index-5 sublattice hold one site per pixel cell: coset (0, 1)
holds the clamps, and each of cosets (1, 0), (-1, 0), (0, -1) holds up to three
units. A site there touches exactly three pixels, and those triangles tile the
grid under an index-3 sublattice, so one copy class per unit couples to every
pixel exactly once. Copies of one unit are tied along one axis by a Z1 offset,
so a unit is three or four parallel chains. Joining the chains needs relay
paths outside the pixel region; this probe joins chain ends with one direct
tie instead (``placement`` relaxed for those 2-3 edges per unit, declared).

Each copy carries the unit's per-pixel couplings unchanged and 1/n of the unit
bias, so infinite ties reproduce the unplaced model. The tie coupling w is a
coupling, so under ``split`` it sets the coupling cap when w > 0.375.

Evidence: Gibbs errors are ``software_simulation`` (NumPy, CPU); the unplaced
64-draw equilibrium floors are ``exact_reference``. Run:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
    uv run python docs/research/dtm_exact_posterior_lattice_placed.py --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dtm_exact_posterior_lattice_probe as probe  # noqa: E402

from thermo_lab.constraints.lattice import OFFSETS  # noqa: E402
from thermo_lab.constraints.quantize import cap_for, quantize_array  # noqa: E402

L = probe.L
N = probe.N
DRAWS = 64  # owner's estimate size (P-0008 round 1 answer)
A_, B_ = np.array([2, 1]), np.array([-1, 2])
MI = np.linalg.inv(np.stack([A_, B_], 1))
# (coset rep, tie axis in pixel coords); tie axis = the pixel step of the tie offset.
COSETS = [((1, 0), (1, 0)), ((-1, 0), (1, 0)), ((0, -1), (0, 1))]


def pix_of(site):
    c = MI @ np.array(site, dtype=float)
    r = np.round(c).astype(int)
    return (int(r[0]), int(r[1])) if np.allclose(c, r) else None


def site_of(i, j, rep=(0, 0)):
    s = i * A_ + j * B_ + np.array(rep)
    return (int(s[0]), int(s[1]))


def layout(C):
    """Unit c -> list of copies (site, [pixel indices]); tie edges; relay edges."""
    classes = [(ci, k) for ci in range(3) for k in range(3)][:C]
    units = []
    for ci, k in classes:
        rep, axis = COSETS[ci]
        copies = []
        for i, j in itertools.product(range(-3, L + 3), repeat=2):
            if (j if axis == (1, 0) else i) % 3 != k:
                continue
            s = site_of(i, j, rep)
            px = []
            for o in OFFSETS:
                p = pix_of((s[0] + o[0], s[1] + o[1]))
                if p is not None and 0 <= p[0] < L and 0 <= p[1] < L:
                    px.append(p[0] * L + p[1])
            if px:
                copies.append(((i, j), s, px))
        ties, chains = [], {}
        for a, b in itertools.combinations(range(len(copies)), 2):
            d = (copies[b][1][0] - copies[a][1][0], copies[b][1][1] - copies[a][1][1])
            if d in OFFSETS:
                ties.append((a, b))
        for idx, ((i, j), _, _) in enumerate(copies):
            key = j if axis == (1, 0) else i
            chains.setdefault(key, []).append(idx)

        def col(t, copies=copies):
            s = copies[t][1]
            return (s[0] + s[1]) % 2

        def far(v, copies=copies):  # the two copies at the far end of a chain
            return sorted(v, key=lambda t: copies[t][0][0] + copies[t][0][1])[-2:]

        ordered = [v for _, v in sorted(chains.items())]
        relays = []
        for va, vb in zip(ordered[:-1], ordered[1:], strict=True):
            pairs = [(a, b) for a in far(va) for b in far(vb) if col(a) != col(b)]
            relays.append(
                min(
                    pairs,
                    key=lambda e: (
                        abs(copies[e[0]][1][0] - copies[e[1]][1][0])
                        + abs(copies[e[0]][1][1] - copies[e[1]][1][1])
                    ),
                )
            )
        units.append({"copies": copies, "ties": ties, "relays": relays, "chains": len(chains)})
    return units


def layout_report(units):
    allpix = []
    rep = []
    for u in units:
        pix = [p for _, _, px in u["copies"] for p in px]
        allpix.append(sorted(pix) == list(range(N)))
        cols = [(s[0] + s[1]) % 2 for _, s, _ in u["copies"]]
        relay_same_colour = sum(cols[a] == cols[b] for a, b in u["relays"])
        rep.append(
            {
                "copies": len(u["copies"]),
                "chains": u["chains"],
                "ties": len(u["ties"]),
                "relays_relaxed": len(u["relays"]),
                "relay_same_colour": int(relay_same_colour),
                "max_pixels_per_copy": max(len(px) for _, _, px in u["copies"]),
            }
        )
    sites = [s for u in units for _, s, _ in u["copies"]]
    sites += [site_of(i, j) for i in range(L) for j in range(L)]
    sites += [site_of(i, j, (0, 1)) for i in range(L) for j in range(L)]
    return {
        "units": rep,
        "every_unit_covers_each_pixel_once": all(allpix),
        "distinct_sites": len(set(sites)) == len(sites),
        "pixel_degree": 4 + 1 + len(units),
    }


def placed_model(p, units, w, bits):
    """Free nodes: 64 pixels then copies. Returns sparse coupling matrix, bias, colours, g."""
    rows, cols, vals = [], [], []
    grid = []
    for i, j in itertools.product(range(L), repeat=2):
        if i + 1 < L:
            grid.append((i * L + j, (i + 1) * L + j))
        if j + 1 < L:
            grid.append((i * L + j, i * L + j + 1))
    off = N
    copy_edges, copy_w, tie_edges, bias_c = [], [], [], []
    for c, u in enumerate(units):
        n = len(u["copies"])
        for k, (_, _, px) in enumerate(u["copies"]):
            for q in px:
                copy_edges.append((q, off + k))
                copy_w.append(p["W"][q, c])
            bias_c.append(p["bh"][c] / n)
        for a, b in u["ties"] + u["relays"]:
            tie_edges.append((off + a, off + b))
        off += n
    nfree = off
    coup = np.concatenate(
        [np.full(len(grid), p["J"]), [p["g"]], copy_w, np.full(len(tie_edges), w)]
    )
    fields = np.concatenate([p["bp"], bias_c])
    caps = None
    if bits is not None:
        cc, cf = cap_for("split", coup, fields, bits)
        coup = quantize_array(coup, cc, bits)
        fields = quantize_array(fields, cf, bits)
        caps = {"coupling": float(cc), "field": float(cf)}
    Jq, gq = coup[0], coup[len(grid)]
    wq = coup[len(grid) + 1 :]
    edges = grid + copy_edges + tie_edges
    ew = np.concatenate([np.full(len(grid), Jq), wq])
    for (a, b), v in zip(edges, ew, strict=True):
        rows += [a, b]
        cols += [b, a]
        vals += [v, v]
    Wm = sparse.csr_matrix((vals, (rows, cols)), shape=(nfree, nfree))
    colour = np.empty(nfree, dtype=int)
    colour[:N] = [(i + j) % 2 for i, j in itertools.product(range(L), repeat=2)]
    k = N
    for u in units:
        for _, s, _ in u["copies"]:
            colour[k] = (s[0] + s[1]) % 2
            k += 1
    return {"W": Wm, "b": fields, "g": float(gq), "colour": colour, "n": nfree, "caps": caps}


def gibbs(m, y, K, start, rng, streams, per):
    n, rows = m["n"], DRAWS * streams
    blocks = [np.flatnonzero(m["colour"] == c) for c in (0, 1)]
    WT = [m["W"][blk].T.tocsr() for blk in blocks]  # (n, |blk|)
    S = rng.integers(0, 2, (rows, n)) * 2.0 - 1.0
    pred = np.empty((len(y), N))
    for j in range(per):
        idx = np.arange(streams) * per + j
        if start == "random":
            S = rng.integers(0, 2, (rows, n)) * 2.0 - 1.0
        ext = np.tile(m["b"], (rows, 1))
        ext[:, :N] += m["g"] * np.repeat(y[idx], DRAWS, axis=0)
        for _ in range(K):
            for blk, wt in zip(blocks, WT, strict=True):
                f = (wt.T @ S.T).T + ext[:, blk]
                S[:, blk] = np.where(rng.random((rows, len(blk))) < expit(2 * f), 1.0, -1.0)
        up = (S[:, :N].reshape(streams, DRAWS, N) > 0).sum(1)
        pred[idx] = np.where(up >= DRAWS // 2, 1.0, -1.0)
    return pred


def maj_wrong(p_up, x, draws):
    pp = binom.sf(draws // 2 - 1, draws, p_up)
    return np.where(x > 0, 1.0 - pp, pp).mean(1)


def run_family(name, b0, ws, bits_list, Ks, starts, streams, per, seed_tag):
    J, h, q = probe.family(name)
    mix, x, y = probe.test_set(name, 1000 if len(h) <= 4 else 500, 1)  # round-1 test images
    n = streams * per
    x, y = x[:n], y[:n]
    post = mix.posterior(y, q)
    bayes = (np.where(post > 0.5, 1.0, -1.0) != x).mean(1)
    units = layout(len(h))
    rec = {
        "family": name,
        "b0": b0,
        "images": n,
        "layout": layout_report(units),
        "bayes_mpm": float(bayes.mean()),
        "bayes_maj64_floor_gap": float((maj_wrong(post, x, DRAWS) - bayes).mean()),
        "unplaced": {},
        "cells": [],
    }
    for bits in bits_list:
        pu_, _, _ = probe.exact_equilibrium(
            probe.quantize(probe.build(J, h, q, "bias", "centered", b0=b0), bits), y
        )
        rec["unplaced"][str(bits)] = float((maj_wrong(pu_, x, DRAWS) - bayes).mean())
    base = probe.build(J, h, q, "bias", "centered", b0=b0)
    for bits in bits_list:
        for w in ws:
            m = placed_model(base, units, w, bits)
            for start in starts:
                for K in Ks:
                    t0 = time.process_time()
                    rng = np.random.default_rng(
                        [
                            probe.PROBE_SEED,
                            31,
                            seed_tag,
                            int(w * 100),
                            bits or 0,
                            K,
                            starts.index(start),
                        ]
                    )
                    pred = gibbs(m, y, K, start, rng, streams, per)
                    d = ((pred != x).mean(1) - bayes).reshape(streams, per).mean(1)
                    br = np.random.default_rng([probe.PROBE_SEED, 32])
                    bs = d[br.integers(0, streams, (2000, streams))].mean(1)
                    cell = {
                        "bits": bits,
                        "tie_w": w,
                        "caps": m["caps"],
                        "start": start,
                        "K": K,
                        "gap_to_bayes": float(d.mean()),
                        "gap_ci95": np.quantile(bs, [0.025, 0.975]).tolist(),
                        "cpu_s": time.process_time() - t0,
                    }
                    rec["cells"].append(cell)
                    print(
                        name,
                        bits,
                        w,
                        start,
                        K,
                        round(cell["gap_to_bayes"], 4),
                        round(cell["cpu_s"], 1),
                        flush=True,
                    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout-only", action="store_true")
    a = ap.parse_args()
    if a.layout_only:
        print(json.dumps({c: layout_report(layout(c)) for c in (4, 8)}, indent=1))
        return
    t0 = time.process_time()
    res = {
        "label": "exploration, not evidence",
        "draws_per_estimate": DRAWS,
        "evidence_classes": {
            "unplaced_floors": "exact_reference",
            "gibbs": "software_simulation (NumPy, CPU)",
        },
        "relaxed": "placement, for 2-3 direct relay ties per unit joining parallel copy chains",
        "families": [],
    }
    res["families"].append(
        run_family(
            "t4_h05_q035",
            -5.0,
            (1.0, 2.0, 4.0),
            (8,),
            (256, 1024),
            ["random", "continue"],
            10,
            20,
            1,
        )
    )
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    res["families"].append(
        run_family(
            "smooth8_h05_q035", -15.0, (2.0, 4.0), (8,), (1024,), ["random", "continue"], 10, 20, 2
        )
    )
    res["total_cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")


if __name__ == "__main__":
    main()
