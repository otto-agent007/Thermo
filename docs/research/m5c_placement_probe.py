"""Place the primary-arm M5c kernels on a synthetic patch of the published offset lattice.

Exploration, exact_reference; no fits, samples, study runner, or gate.
Run: uv run python docs/research/m5c_placement_probe.py [--seeds 0,1,2,3,4]
JSON lines: fixed scope, one record per seed, then a summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
from m5c_chain_probe import input_copies
from m5c_degree_probe import ARCHIVE_SHA256, DEGREE_CAP, OFFSETS, load_archive, references
from m5c_refit_probe import ARM, j_mask
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import expit, logsumexp

from thermo_lab import meta_ebm_cap_baseline as m5a

TIME_LIMIT = 120.0
TOLERANCE = 1e-12
NEIGHBORS = frozenset(OFFSETS)
ORIGIN = (0, 0)


def add(u, v):
    return (u[0] + v[0], u[1] + v[1])


def rotate(u, turns):
    for _ in range(turns):
        u = (-u[1], u[0])
    return u


def adjacent(u, v):
    return (v[0] - u[0], v[1] - u[1]) in NEIGHBORS


def color(u):
    return (u[0] + u[1]) % 2


def masked_vector(p, s):
    """J-only mask on over-degree kernels; every other kernel is the archived vector."""
    p = np.asarray(p, dtype=np.float64).copy()
    p[j_mask(p, s)] = 0.0
    J, _, A, _, beta = m5a.unpack(p, s)
    assert np.count_nonzero(J) + np.count_nonzero(beta) <= DEGREE_CAP
    assert np.all(beta != 0)
    # Identical hidden supports make hidden spins interchangeable for placement.
    assert all(np.array_equal(row != 0, A[0] != 0) for row in A)
    return p


def place_kernel(p, s):
    """Exact minimum-copy placement with y at the origin of the unbounded lattice."""
    J, _, A, _, beta = m5a.unpack(p, s)
    nh = len(beta)
    needed = [int(i) for i in np.flatnonzero(A[0])]
    candidates = sorted({add(o, d) for o in OFFSETS for d in OFFSETS} - {ORIGIN})
    n_u, n_i = len(OFFSETS), len(needed)
    column = {
        (c, i): n_u + ci * n_i + ii
        for ci, c in enumerate(candidates)
        for ii, i in enumerate(needed)
    }
    n = n_u + len(candidates) * n_i
    cost = np.zeros(n)
    cost[n_u:] = 1.0
    rows, lower, upper = [], [], []

    def constrain(row, lo, hi):
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    row = np.zeros(n)
    row[:n_u] = 1.0
    constrain(row, nh, nh)
    for ci in range(len(candidates)):
        row = np.zeros(n)
        row[n_u + ci * n_i : n_u + (ci + 1) * n_i] = 1.0
        constrain(row, 0.0, 1.0)
    for oi, o in enumerate(OFFSETS):
        for i in needed:
            row = np.zeros(n)
            row[oi] = -1.0
            for d in OFFSETS:
                if add(o, d) != ORIGIN:
                    row[column[add(o, d), i]] = 1.0
            constrain(row, 0.0, np.inf)
    started = time.monotonic()
    result = milp(
        cost,
        integrality=np.ones(n),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(np.array(rows), lower, upper),
        options={"time_limit": TIME_LIMIT},
    )
    seconds = time.monotonic() - started
    assert result.x is not None, result.message
    chosen = np.round(result.x).astype(int)
    used = [o for oi, o in enumerate(OFFSETS) if chosen[oi]]
    free_slots = [o for o in OFFSETS if o not in used]
    j_inputs = [int(i) for i in np.flatnonzero(J)]
    assert len(used) == nh and len(j_inputs) <= len(free_slots)
    layout = {ORIGIN: ("y",)}
    layout.update({o: ("h", a) for a, o in enumerate(used)})
    layout.update({o: ("copy", i) for o, i in zip(free_slots, j_inputs, strict=False)})
    layout.update({c: ("copy", i) for (c, i), k in column.items() if chosen[k]})
    return layout, {
        "status": int(result.status),
        "message": str(result.message),
        "mip_gap": float(result.mip_gap),
        "dual_bound": float(result.mip_dual_bound),
        "seconds": seconds,
        "a_copies": int(chosen[n_u:].sum()),
        "a_copy_parity_bound": n_i,
        "j_copies": len(j_inputs),
        "hidden_offsets": used,
    }


def site_couplings(layout, p, s):
    """Rebuild the placed kernel as site couplings; each coefficient on one lattice edge."""
    J, _, A, _, beta = m5a.unpack(p, s)
    role = {site: r for site, r in layout.items()}
    assert len(role) == len(layout)
    (y,) = [site for site, r in role.items() if r == ("y",)]
    hidden = {r[1]: site for site, r in role.items() if r[0] == "h"}
    copies = sorted((site, r[1]) for site, r in role.items() if r[0] == "copy")
    edges = {}

    def couple(u, i, weight):
        options = [c for c, j in copies if j == i and adjacent(u, c)]
        assert options, "missing adjacent input copy"
        edges[u, options[0]] = weight

    for i in np.flatnonzero(J):
        couple(y, int(i), J[i])
    for a, w in hidden.items():
        assert adjacent(y, w)
        edges[y, w] = beta[a]
        for i in np.flatnonzero(A[a]):
            couple(w, int(i), A[a, i])
    assert all(adjacent(u, v) and color(u) != color(v) for u, v in edges)
    return y, hidden, copies, edges


def placed_conditional(layout, p, s):
    """Enumerate free p-bits of the site-level model for every blanket input row."""
    _, h, _, b, _ = m5a.unpack(p, s)
    y, hidden, copies, edges = site_couplings(layout, p, s)
    free = [y] + [hidden[a] for a in sorted(hidden)]
    slot = {site: f for f, site in enumerate(free)}
    x = m5a.blanket_inputs(s)
    fields = np.zeros((len(x), len(free)))
    fields[:, 0] = h
    fields[:, 1:] = b
    pair = np.zeros((len(free), len(free)))
    copy_input = dict(copies)
    for (u, v), w in edges.items():
        if v in copy_input:
            fields[:, slot[u]] += w * x[:, copy_input[v]]
        else:
            pair[slot[u], slot[v]] += w
    states = ((np.arange(1 << len(free))[:, None] >> np.arange(len(free))) & 1) * 2.0 - 1.0
    log_weight = fields @ states.T + np.einsum("si,ij,sj->s", states, pair, states)[None, :]
    up = states[:, 0] > 0
    return np.exp(logsumexp(log_weight[:, up], axis=1) - logsumexp(log_weight, axis=1))


def pack(footprints):
    """Greedy first-fit of rotated/translated footprints into the smallest square found."""
    total = sum(map(len, footprints))
    side = math.isqrt(total - 1) + 1
    order = sorted(range(len(footprints)), key=lambda n: (-len(footprints[n]), n))
    shapes = {}
    for n in order:
        for turns in range(4):
            points = [rotate(u, turns) for u in footprints[n]]
            x0, y0 = min(u[0] for u in points), min(u[1] for u in points)
            shapes[n, turns] = [(u[0] - x0, u[1] - y0) for u in points]
    while True:
        occupied, placements = set(), {}
        for n in order:
            placements[n] = next(
                (
                    (ax, ay, turns)
                    for ay in range(side)
                    for ax in range(side)
                    for turns in range(4)
                    if all(
                        u[0] + ax < side
                        and u[1] + ay < side
                        and (u[0] + ax, u[1] + ay) not in occupied
                        for u in shapes[n, turns]
                    )
                ),
                None,
            )
            if placements[n] is None:
                break
            ax, ay, turns = placements[n]
            occupied.update((u[0] + ax, u[1] + ay) for u in shapes[n, turns])
        else:
            return side, placements
        side += 1


def transform(layout, placement):
    ax, ay, turns = placement
    points = {rotate(u, turns): r for u, r in layout.items()}
    x0, y0 = min(u[0] for u in points), min(u[1] for u in points)
    return {(u[0] - x0 + ax, u[1] - y0 + ay): r for u, r in points.items()}


def bounding_box(sites):
    xs, ys = [u[0] for u in sites], [u[1] for u in sites]
    return [max(xs) - min(xs) + 1, max(ys) - min(ys) + 1]


def evaluate_seed(ref):
    seed = ref["cell"]["seed"]
    structures = m5a.structures(m5a.make_target(seed, ARM[0]))
    kernels, layouts, vectors = [], [], []
    for p, s in zip(ref["parameters"], structures, strict=True):
        original = np.asarray(p, dtype=np.float64)
        q = masked_vector(original, s)
        layout, solve = place_kernel(q, s)
        J, _, _, _, beta = m5a.unpack(q, s)
        exact = expit(2 * m5a.kernel_logit(q, s, m5a.blanket_inputs(s)))
        error = float(np.max(np.abs(placed_conditional(layout, q, s) - exact)))
        assert error < TOLERANCE, error
        parity = input_copies(q, s)
        assert parity == solve["j_copies"] + solve["a_copy_parity_bound"]
        copies = solve["j_copies"] + solve["a_copies"]
        kernels.append(
            {
                "site": s["site"],
                "masked": bool(j_mask(original, s)),
                "blanket": len(J),
                "hidden": len(beta),
                "output_degree": int(np.count_nonzero(J) + np.count_nonzero(beta)),
                **solve,
                "copies": copies,
                "copy_parity_bound": parity,
                "physical_pbits": 1 + len(beta) + copies,
                "parity_pbits": 1 + len(beta) + parity,
                "footprint": len(layout),
                "bounding_box": bounding_box(layout),
                "conditional_error": error,
            }
        )
        layouts.append(layout)
        vectors.append((q, s))
        print(f"seed {seed} site {s['site']}: placed", file=sys.stderr, flush=True)
    side, placements = pack([list(layout) for layout in layouts])
    occupied = {}
    patch_error = 0.0
    for n, (layout, (q, s)) in enumerate(zip(layouts, vectors, strict=True)):
        placed = transform(layout, placements[n])
        assert all(0 <= u[0] < side and 0 <= u[1] < side for u in placed)
        assert not occupied.keys() & placed.keys()
        occupied.update(dict.fromkeys(placed, n))
        exact = expit(2 * m5a.kernel_logit(q, s, m5a.blanket_inputs(s)))
        patch_error = max(
            patch_error, float(np.max(np.abs(placed_conditional(placed, q, s) - exact)))
        )
    assert patch_error < TOLERANCE
    free = sum(1 + k["hidden"] for k in kernels)
    return {
        "seed": seed,
        "kernels": kernels,
        "free_pbits": free,
        "copies": sum(k["copies"] for k in kernels),
        "copy_parity_bound": sum(k["copy_parity_bound"] for k in kernels),
        "physical_pbits": sum(k["physical_pbits"] for k in kernels),
        "parity_pbits": sum(k["parity_pbits"] for k in kernels),
        "patch_side": side,
        "patch_utilization": len(occupied) / side**2,
        "placements": {str(n): list(v) for n, v in placements.items()},
        "patch_conditional_error": patch_error,
        "multiplexed_region": max(
            (k["bounding_box"] for k in kernels), key=lambda box: box[0] * box[1]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    wanted = {int(seed) for seed in parser.parse_args().seeds.split(",")}
    assert len(NEIGHBORS) == 16 and all(rotate(o, 1) in NEIGHBORS for o in OFFSETS)
    record = load_archive()
    print(
        json.dumps(
            {
                "status": "exploration",
                "semantics": "exact_reference",
                "lattice": "synthetic; arXiv:2608.01615v2 section II.2.1 offsets",
                "archive_sha256": ARCHIVE_SHA256,
                "arm": ARM,
                "mask": "J-only on over-degree kernels, as in the refit probe",
                "input_model": "clamped copies, shared within a kernel only",
                "objective": "minimum clamped copies per kernel (HiGHS milp)",
                "time_limit_seconds": TIME_LIMIT,
                "packing": "greedy first-fit, resident, open square patch",
                "offsets": OFFSETS,
                "seeds": sorted(wanted),
            }
        ),
        flush=True,
    )
    started = time.monotonic()
    seeds = []
    for ref in references(record, ARM):
        if ref["cell"]["seed"] not in wanted:
            continue
        seeds.append(evaluate_seed(ref))
        print(json.dumps({"seed": seeds[-1]}), flush=True)
    kernels = [k for seed in seeds for k in seed["kernels"]]
    print(
        json.dumps(
            {
                "kernels": len(kernels),
                "optimal": sum(k["status"] == 0 for k in kernels),
                "max_mip_gap": max(k["mip_gap"] for k in kernels),
                "copies": [seed["copies"] for seed in seeds],
                "copy_parity_bound": [seed["copy_parity_bound"] for seed in seeds],
                "physical_pbits": [seed["physical_pbits"] for seed in seeds],
                "parity_pbits": [seed["parity_pbits"] for seed in seeds],
                "patch_side": [seed["patch_side"] for seed in seeds],
                "patch_utilization": [seed["patch_utilization"] for seed in seeds],
                "multiplexed_region": [seed["multiplexed_region"] for seed in seeds],
                "max_conditional_error": max(k["conditional_error"] for k in kernels),
                "max_patch_conditional_error": max(s["patch_conditional_error"] for s in seeds),
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
