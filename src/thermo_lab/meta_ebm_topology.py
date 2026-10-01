"""M5c: degree repair and placement on a synthetic offset lattice (exact reference).

Protocol: docs/experiments/meta-ebm-synthetic-topology.md. CPU float64, zero samples.
The lattice is synthetic: the published local offset rule on an open square.
"""

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.special import expit, logsumexp

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization_core as core
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text

ARM = ("B", "variational", 1.0)
SEEDS = (0, 1, 2, 3, 4)
K_VALUES = (4, 32, None)  # None is the exact-marginal limit
METHODS = ("original", "j_prune", "refit")
DEGREE_CAP = 16
# Thermalizing Stochastic Programs, arXiv:2608.01615v2 (2026-08-13), section II.2.1.
BASE_OFFSETS = ((1, 0), (2, 1), (2, 3), (4, 1))
OFFSETS = tuple(o for a, b in BASE_OFFSETS for o in ((a, b), (-b, a), (-a, -b), (b, -a)))
NEIGHBORS = frozenset(OFFSETS)
ORIGIN = (0, 0)
REFIT_OPTIONS = {**m5a.OPTIMIZER, "maxiter": 20_000, "maxfun": 200_000}
PLACEMENT_TIME_LIMIT = 600.0
REFERENCE_BARS = {"useful": 8.3e-3, "small_cost": 5e-4}
TOLERANCE = {"conditional": 1e-12, "m5b_replay": 1e-12, "stationary": 1e-10}
HORIZON = 30
ARCHIVE = Path(__file__).resolve().parents[2] / (
    "docs/experiment-reports/2026-09-28-meta-ebm-thermalization/study.json.gz"
)
SOURCE = {
    "archive_sha256": "4e5592572f078f1d36ff89928d145be62588ed5fb8c2adc842a403f278c9c674",
    "request_digest": "sha256:36836ed0c15023f2261c98c930783065ab5a662c760d617231928c6a6399bb8d",
    "result_digest": "sha256:bf4771cfa760e1b099e7726490cef240d144493458f7f2c69d59386d224d7c0b",
}
PINNED = (
    "hashing.py",
    "persistence.py",
    "meta_ebm_cap_baseline.py",
    "meta_ebm_thermalization.py",
    "meta_ebm_thermalization_core.py",
)
BODY = ("refits", "placements", "patches", "outer", "summaries", "integrity")


def _file_sha256(name):
    return hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()


def load_source():
    """Authenticate the M5b archive and the implementations it pins."""
    data = ARCHIVE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE["archive_sha256"]:
        raise ValueError("M5b archive hash mismatch")
    record = json.loads(gzip.decompress(data))
    for key in ("request_digest", "result_digest"):
        if record[key] != SOURCE[key]:
            raise ValueError(f"M5b {key} mismatch")
    for name, expected in record["request"]["implementation_sha256"].items():
        if _file_sha256(name) != expected:
            raise ValueError(f"archive-bound source changed: {name}")
    return record


def references(source, seeds=SEEDS):
    """Archived M5a parameters and M5b finite-K baselines for the primary arm."""
    out = {}
    for result in source["results"].values():
        cell = result["cell"]
        if (cell["reading"], cell["method"], cell["cap"]) != ARM or cell["seed"] not in seeds:
            continue
        entry = out.setdefault(cell["seed"], {"baselines": {}})
        if cell["arm"] == "reference":
            entry["parameters"] = result["parameters"]
            entry["baselines"]["limit"] = result["metrics"]
        elif cell["arm"] == "finite_k" and cell["value"] in K_VALUES:
            entry["baselines"][str(cell["value"])] = result["metrics"]
    if sorted(out) != sorted(seeds) or any(len(e["baselines"]) != 3 for e in out.values()):
        raise ValueError("expected one reference and K=4, 32 baselines per seed")
    return out


def study_request(mode="full"):
    if mode not in ("full", "benchmark"):
        raise ValueError("unknown study mode")
    return {
        "schema": "meta_ebm_topology.v1",
        "mode": mode,
        "protocol": "docs/experiments/meta-ebm-synthetic-topology.md",
        "evidence_class": "exact_reference",
        "dtype": "float64",
        "sample_count": 0,
        "source": SOURCE,
        "arm": list(ARM),
        "seeds": list(SEEDS if mode == "full" else SEEDS[:1]),
        "k_values": [k_label(k) for k in K_VALUES],
        "methods": list(METHODS),
        "mask": "zero degree-16 smallest-|J| nonzero J per over-degree kernel; ties by J index",
        "refit": {
            "objective": "M5a uniform-input KL with analytic gradient",
            "bounds": "(0,0) masked, (-1,1) otherwise",
            "options": REFIT_OPTIONS,
            "starts": "M5a eight starts masked, plus masked archived vector",
            "selection": "minimum endpoint objective, ties by start index",
        },
        "lattice": {
            "source": "arXiv:2608.01615v2 section II.2.1; synthetic open square",
            "offsets": [list(o) for o in OFFSETS],
        },
        "placement": {
            "free": "y and hidden spins one site each; no chains or routing nodes",
            "copies": "one input per clamped site; J copies on y neighbors in offset order; "
            "each nonzero A on exactly one adjacent copy; shared within a kernel only",
            "objective": "minimum copies, SciPy milp (HiGHS), y at origin",
            "time_limit_seconds": PLACEMENT_TIME_LIMIT,
            "packing": "greedy first-fit, largest footprint first, row-major anchors, "
            "rotations 0-3, side from ceil(sqrt(total sites))",
        },
        "outer": "M5b sweep, sites 0-11, uniform start over 4096 states",
        "horizon": HORIZON,
        "reference_bars": REFERENCE_BARS,
        "tolerance": TOLERANCE,
        "implementation_sha256": {
            name: _file_sha256(name) for name in (*PINNED, "meta_ebm_topology.py")
        },
    }


def k_label(k):
    return "limit" if k is None else str(k)


# ---------------------------------------------------------------- lattice


def adjacent(u, v):
    return (v[0] - u[0], v[1] - u[1]) in NEIGHBORS


def color(u):
    return (u[0] + u[1]) % 2


def rotate(u, turns):
    for _ in range(turns):
        u = (-u[1], u[0])
    return u


# ---------------------------------------------------------------- repair


def output_degree(p, s):
    J, _, _, _, beta = m5a.unpack(p, s)
    return int(np.count_nonzero(J) + np.count_nonzero(beta))


def copy_parity_bound(p, s):
    """One clamped copy per used input and color."""
    J, _, A, _, _ = m5a.unpack(p, s)
    return int(np.count_nonzero(J) + np.count_nonzero(np.any(A != 0, axis=0)))


def j_mask(p, s):
    """Smallest archived nonzero J edges beyond degree 16; beta is never masked."""
    J, _, _, _, _ = m5a.unpack(p, s)
    count = max(0, output_degree(p, s) - DEGREE_CAP)
    candidates = sorted((abs(value), i) for i, value in enumerate(J) if value != 0)
    if count > len(candidates):
        raise ValueError("degree cannot be repaired by masking J")
    return sorted(i for _, i in candidates[:count])


def evaluate_refit(job):
    """Fit one over-degree kernel under its fixed J-only mask."""
    started = time.monotonic()
    seed, site = job["seed"], job["site"]
    s = m5a.structures(m5a.make_target(seed, ARM[0]))[site]
    archived = np.asarray(job["archived"], dtype=np.float64)
    mask = j_mask(archived, s)
    if not mask:
        raise ValueError("refit requested for a legal kernel")
    x = m5a.blanket_inputs(s)
    logit = m5a.exact_logit(s, x)
    starts = [*m5a._starts(s, ARM[2], ARM[0], seed, site), archived.copy()]
    bounds = [(0.0, 0.0) if i in mask else (-ARM[2], ARM[2]) for i in range(len(archived))]
    attempts, endpoints = [], []
    for start in starts:
        start[mask] = 0.0
        result = minimize(
            m5a.objective,
            start,
            args=(s, x, logit),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=REFIT_OPTIONS,
        )
        endpoint = np.clip(np.asarray(result.x, dtype=np.float64), -ARM[2], ARM[2])
        attempts.append(
            {
                "initial_objective": float(m5a.objective(start, s, x, logit)[0]),
                "objective": float(m5a.objective(endpoint, s, x, logit)[0]),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "success": bool(result.success),
                "termination": str(result.message),
            }
        )
        endpoints.append(endpoint)
    selected = min(range(len(attempts)), key=lambda i: (attempts[i]["objective"], i))
    parameters = endpoints[selected]
    pruned = archived.copy()
    pruned[mask] = 0.0
    record = {
        "seed": seed,
        "site": site,
        "mask": mask,
        "masked_input_sites": [s["blanket"][i] for i in mask],
        "attempts": attempts,
        "selected": selected,
        "parameters": parameters.tolist(),
        **refit_checks(archived, parameters, s),
        "mixing": {
            name: core.mixing_summary(core.inner_kernel(vector, s))
            for name, vector in (("original", archived), ("j_prune", pruned), ("refit", parameters))
        },
    }
    return record, {"seconds": time.monotonic() - started}


def refit_checks(archived, parameters, s):
    """Mask, bounds, degree and conditional checks shared by generation and replay."""
    mask = j_mask(archived, s)
    J, _, _, _, beta = m5a.unpack(parameters, s)
    if np.any(parameters[mask] != 0.0) or np.any(np.abs(parameters) > ARM[2]):
        raise ValueError("refit violates its mask or cap")
    if output_degree(parameters, s) > DEGREE_CAP or np.any(beta == 0.0):
        raise ValueError("refit degree or retained beta check failed")
    x = m5a.blanket_inputs(s)
    exact = expit(2 * m5a.kernel_logit(parameters, s, x))
    error = float(np.max(np.abs(exact - m5a.brute_force_probability(parameters, s, x))))
    if error > TOLERANCE["conditional"]:
        raise ValueError(f"refit conditional differs from brute force: {error}")
    return {
        "objective": float(m5a.objective(parameters, s, x, m5a.exact_logit(s, x))[0]),
        "output_degree": output_degree(parameters, s),
        "hidden_count": len(beta),
        "brute_force_error": error,
    }


# ---------------------------------------------------------------- placement


def place_kernel(p, s):
    """Exact minimum-copy placement with y at the origin of the unbounded lattice."""
    J, _, A, _, beta = m5a.unpack(p, s)
    if output_degree(p, s) > DEGREE_CAP:
        raise ValueError("kernel exceeds the lattice degree")
    support = A != 0
    if not all(np.array_equal(row, support[0]) for row in support):
        raise ValueError("hidden spins do not share one input support")
    nh, needed = len(beta), [int(i) for i in np.flatnonzero(support[0])]
    candidates = sorted({(o[0] + d[0], o[1] + d[1]) for o in OFFSETS for d in OFFSETS} - {ORIGIN})
    n_u, n_i = len(OFFSETS), len(needed)
    column = {
        (c, i): n_u + ci * n_i + ii
        for ci, c in enumerate(candidates)
        for ii, i in enumerate(needed)
    }
    n = n_u + len(candidates) * n_i
    rows, lower, upper = [], [], []

    def constrain(row, lo, hi):
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    row = np.zeros(n)
    row[:n_u] = 1.0
    constrain(row, nh, nh)  # every hidden spin on a distinct neighbor of y
    for ci in range(len(candidates)):
        row = np.zeros(n)
        row[n_u + ci * n_i : n_u + (ci + 1) * n_i] = 1.0
        constrain(row, 0.0, 1.0)  # one input per copy site
    for oi, o in enumerate(OFFSETS):
        for i in needed:
            row = np.zeros(n)
            row[oi] = -1.0
            for d in OFFSETS:
                c = (o[0] + d[0], o[1] + d[1])
                if c != ORIGIN:
                    row[column[c, i]] = 1.0
            constrain(row, 0.0, np.inf)  # a used hidden offset sees a copy of input i
    cost = np.zeros(n)
    cost[n_u:] = 1.0
    result = milp(
        cost,
        integrality=np.ones(n),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(np.array(rows), lower, upper),
        options={"time_limit": PLACEMENT_TIME_LIMIT},
    )
    if result.x is None:
        raise ValueError(f"no placement found: {result.message}")
    chosen = np.round(result.x).astype(int)
    used = [o for oi, o in enumerate(OFFSETS) if chosen[oi]]
    free_slots = [o for o in OFFSETS if o not in used]
    j_inputs = [int(i) for i in np.flatnonzero(J)]
    if len(used) != nh or len(j_inputs) > len(free_slots):
        raise ValueError("placement does not seat every free p-bit")
    layout = [[*ORIGIN, "y", None]]
    layout += [[*o, "h", a] for a, o in enumerate(used)]
    layout += [[*o, "copy", i] for o, i in zip(free_slots, j_inputs, strict=False)]
    layout += [[*c, "copy", i] for (c, i), k in column.items() if chosen[k]]
    return sorted(layout, key=lambda e: (e[0], e[1])), {
        "status": int(result.status),
        "message": str(result.message),
        "optimal": int(result.status) == 0,
        "dual_bound": float(result.mip_dual_bound),
        "mip_gap": float(result.mip_gap),
    }


def verify_layout(layout, p, s, side=None):
    """Rebuild site couplings, check topology, and enumerate the placed conditional."""
    _, h, A, b, beta = m5a.unpack(p, s)
    J = m5a.unpack(p, s)[0]
    sites = [(e[0], e[1]) for e in layout]
    if len(set(sites)) != len(sites):
        raise ValueError("two nodes share a site")
    if side is not None and not all(0 <= u < side and 0 <= v < side for u, v in sites):
        raise ValueError("node outside the patch")
    roles = {(e[0], e[1]): (e[2], e[3]) for e in layout}
    ys = [u for u, r in roles.items() if r[0] == "y"]
    hidden = {r[1]: u for u, r in roles.items() if r[0] == "h"}
    copies = sorted((u, r[1]) for u, r in roles.items() if r[0] == "copy")
    if len(ys) != 1 or sorted(hidden) != list(range(len(beta))):
        raise ValueError("free p-bits are not seated exactly once")
    if len(roles) != 1 + len(hidden) + len(copies):
        raise ValueError("unknown node role")
    (y,) = ys
    edges = {}

    def couple(u, i, weight):
        options = [c for c, j in copies if j == i and adjacent(u, c)]
        if not options:
            raise ValueError("missing adjacent input copy")
        edges[u, options[0]] = weight

    for i in np.flatnonzero(J):
        couple(y, int(i), J[i])
    for a, w in hidden.items():
        if not adjacent(y, w):
            raise ValueError("beta edge is not a lattice edge")
        edges[y, w] = beta[a]
        for i in np.flatnonzero(A[a]):
            couple(w, int(i), A[a, i])
    if not all(adjacent(u, v) and color(u) != color(v) for u, v in edges):
        raise ValueError("coupled pair is not an opposite-color lattice edge")
    free = [y] + [hidden[a] for a in sorted(hidden)]
    slot = {u: f for f, u in enumerate(free)}
    x = m5a.blanket_inputs(s)
    fields = np.zeros((len(x), len(free)))
    fields[:, 0], fields[:, 1:] = h, b
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
    placed = np.exp(logsumexp(log_weight[:, up], axis=1) - logsumexp(log_weight, axis=1))
    error = float(np.max(np.abs(placed - expit(2 * m5a.kernel_logit(p, s, x)))))
    if error > TOLERANCE["conditional"]:
        raise ValueError(f"placed conditional differs: {error}")
    j_copies = sum(color(c) != color(y) for c, _ in copies)
    xs, ys_ = [u for u, _ in sites], [v for _, v in sites]
    return {
        "conditional_error": error,
        "j_copies": int(j_copies),
        "a_copies": len(copies) - int(j_copies),
        "copies": len(copies),
        "copy_parity_bound": copy_parity_bound(p, s),
        "physical_pbits": len(layout),
        "free_pbits": 1 + len(hidden),
        "bounding_box": [max(xs) - min(xs) + 1, max(ys_) - min(ys_) + 1],
        "topology_violations": 0,
    }


def evaluate_placement(job):
    started = time.monotonic()
    s = m5a.structures(m5a.make_target(job["seed"], ARM[0]))[job["site"]]
    p = np.asarray(job["parameters"], dtype=np.float64)
    layout, solver = place_kernel(p, s)
    seconds = time.monotonic() - started
    J, _, _, _, beta = m5a.unpack(p, s)
    return {
        "seed": job["seed"],
        "site": job["site"],
        "blanket": len(J),
        "hidden": len(beta),
        "output_degree": output_degree(p, s),
        "layout": layout,
        "solver": solver,
        **verify_layout(layout, p, s),
    }, {"seconds": seconds}


def _shapes(layout):
    points = [(e[0], e[1]) for e in layout]
    out = []
    for turns in range(4):
        rotated = [rotate(u, turns) for u in points]
        x0, y0 = min(u[0] for u in rotated), min(u[1] for u in rotated)
        out.append([(u[0] - x0, u[1] - y0) for u in rotated])
    return out


def pack(layouts):
    """Greedy first-fit of rotated/translated footprints into the smallest square found."""
    total = sum(map(len, layouts))
    side = math.isqrt(total - 1) + 1
    order = sorted(range(len(layouts)), key=lambda n: (-len(layouts[n]), n))
    shapes = {n: _shapes(layouts[n]) for n in order}
    while True:
        occupied, placements = set(), {}
        for n in order:
            spot = next(
                (
                    (ax, ay, turns)
                    for ay in range(side)
                    for ax in range(side)
                    for turns in range(4)
                    if all(
                        u + ax < side and v + ay < side and (u + ax, v + ay) not in occupied
                        for u, v in shapes[n][turns]
                    )
                ),
                None,
            )
            if spot is None:
                break
            placements[n] = list(spot)
            occupied.update((u + spot[0], v + spot[1]) for u, v in shapes[n][spot[2]])
        else:
            return side, [placements[n] for n in range(len(layouts))]
        side += 1


def transform(layout, placement):
    ax, ay, turns = placement
    rotated = [(*rotate((e[0], e[1]), turns), e[2], e[3]) for e in layout]
    x0, y0 = min(e[0] for e in rotated), min(e[1] for e in rotated)
    return sorted(
        ([e[0] - x0 + ax, e[1] - y0 + ay, e[2], e[3]] for e in rotated), key=lambda e: e[:2]
    )


def patch_record(layouts, vectors, structures):
    side, placements = pack(layouts)
    occupied, error = set(), 0.0
    for layout, placement, p, s in zip(layouts, placements, vectors, structures, strict=True):
        placed = transform(layout, placement)
        sites = {(e[0], e[1]) for e in placed}
        if occupied & sites:
            raise ValueError("packed kernels overlap")
        occupied |= sites
        error = max(error, verify_layout(placed, p, s, side)["conditional_error"])
    boxes = [verify_box(layout) for layout in layouts]
    return {
        "side": side,
        "placements": placements,
        "sites": len(occupied),
        "utilization": len(occupied) / side**2,
        "conditional_error": error,
        "multiplexed_region": max(boxes, key=lambda box: (box[0] * box[1], box)),
    }


def verify_box(layout):
    xs, ys = [e[0] for e in layout], [e[1] for e in layout]
    return [max(xs) - min(xs) + 1, max(ys) - min(ys) + 1]


# ---------------------------------------------------------------- outer chain


def outer_metrics(rates, target):
    """M5b's exact sweep and stationary solver; uniform start; t = 0..30."""
    matrix = core.sweep(np.eye(m5a.STATES), rates)
    law = core.stationary_law(matrix)
    path = core.trajectory(matrix, horizon=HORIZON)
    residual = float(np.abs(law @ matrix - law).sum())
    if residual > TOLERANCE["stationary"]:
        raise core.NumericalIntegrityError("stationary residual exceeds tolerance")
    return {
        "bias": float(0.5 * np.abs(law - target).sum()),
        "tv_target": (0.5 * np.abs(path - target).sum(axis=1)).tolist(),
        "tv_own_stationary": (0.5 * np.abs(path - law).sum(axis=1)).tolist(),
        "stationary_residual": residual,
        "row_sum_error": float(np.max(np.abs(matrix.sum(axis=1) - 1))),
    }


def evaluate_outer(job):
    """All methods and K values for one seed; the original replays M5b."""
    started = time.monotonic()
    seed = job["seed"]
    target = m5a.make_target(seed, ARM[0])
    structures = m5a.structures(target)
    distribution = m5a.target_distribution(target)
    cells, replay_error = {}, 0.0
    for method in METHODS:
        vectors = [np.asarray(v, dtype=np.float64) for v in job["vectors"][method]]
        kernels = [core.inner_kernel(v, s) for v, s in zip(vectors, structures, strict=True)]
        codes = [m5a.blanket_codes(s) for s in structures]
        parity = sum(copy_parity_bound(v, s) for v, s in zip(vectors, structures, strict=True))
        for k in K_VALUES:
            rates = [
                core.powered_rates(kernel, k)[c] for kernel, c in zip(kernels, codes, strict=True)
            ]
            metrics = outer_metrics(rates, distribution)
            if method == "original":
                archived = job["baselines"][k_label(k)]
                for key in ("bias", "tv_target", "tv_own_stationary"):
                    error = float(np.max(np.abs(np.asarray(metrics[key]) - archived[key])))
                    if error > TOLERANCE["m5b_replay"]:
                        raise ValueError(f"M5b replay differs: seed {seed} K {k} {key}")
                    replay_error = max(replay_error, error)
            cells[f"{seed}|{method}|{k_label(k)}"] = {
                "seed": seed,
                "method": method,
                "k": k_label(k),
                "metrics": metrics,
                "work": {
                    "spin_redraws_per_outer_sweep": None if k is None else k * 72,
                    "free_reset_writes": 72,
                    "readout_bits": 12,
                    "clamp_writes_parity_bound": parity,
                    "clamp_writes_placed": job["placed_copies"].get(method),
                },
            }
    return cells, {"seconds": time.monotonic() - started, "m5b_replay_error": replay_error}


# ---------------------------------------------------------------- assembly


def _plain(value):
    """The JSON form a persisted value takes, for exact type-aware comparison."""
    return json.loads(canonical_json(value))


def _spread(values):
    return dict(
        zip(("min", "median", "max"), np.quantile(values, [0, 0.5, 1]).tolist(), strict=True)
    )


def summaries(record):
    seeds = sorted(record["patches"], key=int)
    outer = []
    for method in METHODS:
        for k in K_VALUES:
            cells = [record["outer"][f"{seed}|{method}|{k_label(k)}"] for seed in seeds]
            outer.append(
                {
                    "method": method,
                    "k": k_label(k),
                    "bias": _spread([c["metrics"]["bias"] for c in cells]),
                    "tv_target_30": _spread([c["metrics"]["tv_target"][-1] for c in cells]),
                    "tv_own_stationary_30": _spread(
                        [c["metrics"]["tv_own_stationary"][-1] for c in cells]
                    ),
                }
            )
    primary = next(row for row in outer if row["method"] == "refit" and row["k"] == "4")
    medians = [primary[key]["median"] for key in ("bias", "tv_target_30")]
    per_seed = {}
    for seed in seeds:
        kernels = [v for v in record["placements"].values() if str(v["seed"]) == seed]
        per_seed[seed] = {
            key: sum(k[key] for k in kernels)
            for key in ("free_pbits", "copies", "copy_parity_bound", "physical_pbits")
        }
        per_seed[seed].update(
            side=record["patches"][seed]["side"],
            utilization=record["patches"][seed]["utilization"],
            multiplexed_region=record["patches"][seed]["multiplexed_region"],
        )
    placements = list(record["placements"].values())
    return {
        "outer": outer,
        "reference_bars_at_k4": {name: max(medians) < bar for name, bar in REFERENCE_BARS.items()},
        "placement_by_seed": per_seed,
        "placements_optimal": sum(p["solver"]["optimal"] for p in placements),
        "placements": len(placements),
        "refits_successful": sum(
            r["attempts"][r["selected"]]["success"] for r in record["refits"].values()
        ),
        "refits": len(record["refits"]),
    }


def _vectors(refs, refits, seed):
    original = [np.asarray(p, dtype=np.float64) for p in refs[seed]["parameters"]]
    structures = m5a.structures(m5a.make_target(seed, ARM[0]))
    pruned, fitted = [], []
    for site, (p, s) in enumerate(zip(original, structures, strict=True)):
        mask = j_mask(p, s)
        q = p.copy()
        q[mask] = 0.0
        pruned.append(q)
        key = f"{seed}|{site}"
        if mask:
            fitted.append(np.asarray(refits[key]["parameters"], dtype=np.float64))
        elif key in refits:
            raise ValueError("refit recorded for a legal kernel")
        else:
            fitted.append(p)
    return {"original": original, "j_prune": pruned, "refit": fitted}, structures


def _outer_jobs(refs, refits, placements, seeds):
    jobs = []
    for seed in seeds:
        vectors, _ = _vectors(refs, refits, seed)
        copies = sum(placements[f"{seed}|{site}"]["copies"] for site in range(m5a.D))
        jobs.append(
            {
                "seed": seed,
                "vectors": {m: [v.tolist() for v in vs] for m, vs in vectors.items()},
                "baselines": refs[seed]["baselines"],
                # The J-only prune and refit share one nonzero pattern; the original
                # has over-degree kernels and no placement.
                "placed_copies": {"original": None, "j_prune": copies, "refit": copies},
            }
        )
    return jobs


def _map(workers, function, jobs, log, label):
    results, timings = [], []
    with m5a._pool(workers) as pool:
        for job, (result, timing) in zip(jobs, pool.map(function, jobs), strict=True):
            results.append(result)
            timings.append(timing)
            log(f"{label} {job.get('seed')}|{job.get('site', '')} ({timing['seconds']:.1f}s)")
    return results, timings


def generate(request, source, workers, placement_workers, log):
    seeds = request["seeds"]
    refs = references(source, seeds)
    timings = {}
    refit_jobs = []
    for seed in seeds:
        structures = m5a.structures(m5a.make_target(seed, ARM[0]))
        for site, (p, s) in enumerate(zip(refs[seed]["parameters"], structures, strict=True)):
            if j_mask(np.asarray(p, dtype=np.float64), s):
                refit_jobs.append({"seed": seed, "site": site, "archived": p})
    results, timings["refit"] = _map(workers, evaluate_refit, refit_jobs, log, "refit")
    refits = {f"{r['seed']}|{r['site']}": r for r in results}
    placement_jobs = []
    for seed in seeds:
        vectors, _ = _vectors(refs, refits, seed)
        placement_jobs += [
            {"seed": seed, "site": site, "parameters": v.tolist()}
            for site, v in enumerate(vectors["refit"])
        ]
    results, timings["placement"] = _map(
        placement_workers, evaluate_placement, placement_jobs, log, "placement"
    )
    placements = {f"{r['seed']}|{r['site']}": r for r in results}
    patches = {}
    for seed in seeds:
        vectors, structures = _vectors(refs, refits, seed)
        layouts = [placements[f"{seed}|{site}"]["layout"] for site in range(m5a.D)]
        patches[str(seed)] = patch_record(layouts, vectors["refit"], structures)
        log(f"packed seed {seed} into side {patches[str(seed)]['side']}")
    jobs = _outer_jobs(refs, refits, placements, seeds)
    results, timings["outer"] = _map(workers, evaluate_outer, jobs, log, "outer")
    outer = {key: cell for cells in results for key, cell in cells.items()}
    body = {"refits": refits, "placements": placements, "patches": patches, "outer": outer}
    body["summaries"] = summaries(body)
    body["integrity"] = integrity(body, [t["m5b_replay_error"] for t in timings["outer"]])
    return body, timings


def integrity(body, replay_errors):
    return {
        "m5b_original_replays": 3 * len(body["patches"]),
        "max_m5b_replay_error": max(replay_errors),
        "max_refit_brute_force_error": max(
            (r["brute_force_error"] for r in body["refits"].values()), default=0.0
        ),
        "max_placed_conditional_error": max(
            p["conditional_error"] for p in body["placements"].values()
        ),
        "max_patch_conditional_error": max(
            p["conditional_error"] for p in body["patches"].values()
        ),
        "max_stationary_residual": max(
            c["metrics"]["stationary_residual"] for c in body["outer"].values()
        ),
        "max_row_sum_error": max(c["metrics"]["row_sum_error"] for c in body["outer"].values()),
        "unchanged_vectors": m5a.D * len(body["patches"]) - len(body["refits"]),
        "topology_violations": 0,
        "unresolved_placement_residual": 0,
        "passed": True,
    }


def assemble(request, body):
    return {
        "request": request,
        "request_digest": canonical_sha256(request),
        **body,
        "result_digest": canonical_sha256({key: body[key] for key in BODY}),
    }


def validate_record(record, request):
    if record.get("request") != request or record.get("request_digest") != canonical_sha256(
        request
    ):
        raise ValueError("archive request differs")
    if record.get("result_digest") != canonical_sha256({key: record[key] for key in BODY}):
        raise ValueError("archive result digest mismatch")
    if record["integrity"].get("passed") is not True:
        raise ValueError("archive integrity did not pass")


def replay(record, source, workers, log, seeds=None):
    """Recompute from stored parameters and layouts; never refit or re-solve."""
    request = record["request"]
    seeds = request["seeds"] if seeds is None else seeds
    refs = references(source, request["seeds"])
    refits = record["refits"]
    for key, stored in refits.items():
        seed, site = map(int, key.split("|"))
        if seed not in seeds:
            continue
        s = m5a.structures(m5a.make_target(seed, ARM[0]))[site]
        archived = np.asarray(refs[seed]["parameters"][site], dtype=np.float64)
        parameters = np.asarray(stored["parameters"], dtype=np.float64)
        objectives = [a["objective"] for a in stored["attempts"]]
        if stored["mask"] != j_mask(archived, s) or len(objectives) != 9:
            raise ValueError(f"refit ledger differs: {key}")
        if stored["selected"] != min(range(9), key=lambda i: (objectives[i], i)):
            raise ValueError(f"refit selection differs: {key}")
        rebuilt = refit_checks(archived, parameters, s)
        m5a._close({k: stored[k] for k in rebuilt}, _plain(rebuilt), f"refits.{key}")
        pruned = archived.copy()
        pruned[stored["mask"]] = 0.0
        mixing = {
            name: core.mixing_summary(core.inner_kernel(vector, s))
            for name, vector in (("original", archived), ("j_prune", pruned), ("refit", parameters))
        }
        m5a._close(stored["mixing"], _plain(mixing), f"refits.{key}.mixing")
        if rebuilt["objective"] != objectives[stored["selected"]]:
            raise ValueError(f"selected objective differs: {key}")
    for seed in seeds:
        vectors, structures = _vectors(refs, refits, seed)
        for site, (p, s) in enumerate(zip(vectors["refit"], structures, strict=True)):
            stored = record["placements"][f"{seed}|{site}"]
            rebuilt = _plain(verify_layout(stored["layout"], p, s))
            m5a._close({k: stored[k] for k in rebuilt}, rebuilt, f"placements.{seed}|{site}")
        layouts = [record["placements"][f"{seed}|{site}"]["layout"] for site in range(m5a.D)]
        m5a._close(
            record["patches"][str(seed)],
            _plain(patch_record(layouts, vectors["refit"], structures)),
            f"patches.{seed}",
        )
        log(f"replayed refits, placements and patch for seed {seed}")
    jobs = _outer_jobs(refs, refits, record["placements"], seeds)
    results, _ = _map(workers, evaluate_outer, jobs, log, "replay outer")
    for cells in results:
        for key, cell in cells.items():
            m5a._close(record["outer"][key], _plain(cell), f"outer.{key}")
    if seeds == request["seeds"]:
        m5a._close(record["summaries"], _plain(summaries(record)), "summaries")
        if record["integrity"]["unchanged_vectors"] != m5a.D * len(seeds) - len(refits):
            raise ValueError("unchanged-vector count differs")


# ---------------------------------------------------------------- publication


def render_report(record):
    request, summary = record["request"], record["summaries"]
    benchmark = request["mode"] == "benchmark"
    lines = [
        f"# M5c {'runtime benchmark (seed 0 only)' if benchmark else 'study'}",
        "",
        "Exact reference on CPU, float64, zero generated samples. The lattice is synthetic: "
        "the published offset rule (arXiv:2608.01615v2 §II.2.1) on an open square.",
        "",
    ]
    if benchmark:
        lines += ["**Runtime calibration only: this subset is not the M5c study.**", ""]
    lines += [
        "## Outer chain",
        "",
        "Median [min, max] across seeds.",
        "",
        "| Method | K | Stationary bias | TV to target at t=30 | TV to own law at t=30 |",
        "|---|---|---|---|---|",
    ]
    for row in summary["outer"]:

        def cell(name, row=row):
            s = row[name]
            return f"{s['median']:.6g} [{s['min']:.6g}, {s['max']:.6g}]"

        lines.append(
            f"| {row['method']} | {row['k']} | {cell('bias')} | {cell('tv_target_30')} | "
            f"{cell('tv_own_stationary_30')} |"
        )
    bars = summary["reference_bars_at_k4"]
    lines += [
        "",
        f"Refit at K=4 below the 8.3e-3 reference bar: {bars['useful']}; "
        f"below 5e-4: {bars['small_cost']}. Reference points, not gates.",
        "",
        "## Placement",
        "",
        "| Seed | Free p-bits | Copies (parity bound) | Physical p-bits | Patch side | "
        "Utilization | Largest kernel box |",
        "|---|---|---|---|---|---|---|",
    ]
    for seed, row in summary["placement_by_seed"].items():
        box = row["multiplexed_region"]
        lines.append(
            f"| {seed} | {row['free_pbits']} | {row['copies']} ({row['copy_parity_bound']}) | "
            f"{row['physical_pbits']} | {row['side']} | {row['utilization']:.1%} | "
            f"{box[0]} × {box[1]} |"
        )
    integrity = record["integrity"]
    lines += [
        "",
        f"Placements proven optimal: {summary['placements_optimal']}/{summary['placements']}. "
        f"Refits with successful termination: "
        f"{summary['refits_successful']}/{summary['refits']}.",
        "",
        "## Integrity",
        "",
        f"- M5b original replays: {integrity['m5b_original_replays']}, maximum error "
        f"{integrity['max_m5b_replay_error']:.3g}.",
        f"- Placed conditional error: {integrity['max_placed_conditional_error']:.3g} at the "
        f"origin, {integrity['max_patch_conditional_error']:.3g} in the patch.",
        f"- Refit brute-force error: {integrity['max_refit_brute_force_error']:.3g}.",
        f"- Stationary residual: {integrity['max_stationary_residual']:.3g}; "
        f"topology violations: {integrity['topology_violations']}.",
        "",
        "Copy, p-bit and clamp-write counts are algorithmic counts under the stated input "
        "model, not device operations. Energy, latency and reprogramming are excluded.",
    ]
    return "\n".join(lines) + "\n"


def completion(record, provenance, archive_bytes):
    benchmark = record["request"]["mode"] == "benchmark"
    return {
        "status": "m5c_benchmark_complete" if benchmark else "meta_ebm_topology_complete",
        "original_replays": record["integrity"]["m5b_original_replays"],
        "refits": len(record["refits"]),
        "new_outer_cells": sum(c["method"] != "original" for c in record["outer"].values()),
        "placements": len(record["placements"]),
        "patches": len(record["patches"]),
        "samples": 0,
        "topology_violations": 0,
        "replayed": True,
        "integrity": True,
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def run_study(output_dir, *, workers=3, placement_workers=6, mode="full"):
    """Generate, persist, replay from the archive, then publish completion last."""
    for value in (workers, placement_workers):
        if type(value) is not int or value < 1:
            raise ValueError("worker counts must be positive integers")
    request = study_request(mode)
    source = load_source()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    lock = m5a._acquire_run_lock(destination)
    started, started_at = time.monotonic(), m5a._utc_now()

    def log(message):
        with (destination / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{m5a._utc_now()} M5c {message}\n")

    try:
        log(f"started; {workers} workers, {placement_workers} placement workers")
        body, timings = generate(request, source, workers, placement_workers, log)
        if study_request(mode) != request:
            raise ValueError("implementation changed during generation")
        record = assemble(request, body)
        archive_path = destination / "study.json.gz"
        m5a._atomic_write_archive(
            archive_path, gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
        )
        generated = time.monotonic() - started
        log(f"generation complete in {generated:.1f}s; replaying from the archive")
        archive_bytes = archive_path.read_bytes()
        persisted = json.loads(gzip.decompress(archive_bytes))
        validate_record(persisted, request)
        replay(persisted, load_source(), workers, log)
        if study_request(mode) != request or archive_path.read_bytes() != archive_bytes:
            raise ValueError("implementation or archive changed during replay")
        provenance = {
            "schema": "m5c.provenance.v1",
            "request_digest": persisted["request_digest"],
            "result_digest": persisted["result_digest"],
            "runtime": m5a._runtime(workers),
            "placement_workers": placement_workers,
            "started_at": started_at,
            "completed_at": m5a._utc_now(),
            "generation_seconds": generated,
            "total_seconds": time.monotonic() - started,
            "unit_seconds": timings,
            "timing_scope": "CPU exact-reference wall seconds; no device claim.",
        }
        atomic_write_text(destination / "summary.md", render_report(persisted))
        atomic_write_text(destination / "provenance.json", canonical_json(provenance) + "\n")
        atomic_write_text(
            destination / "completion.json",
            canonical_json(completion(persisted, provenance, archive_bytes)) + "\n",
        )
        log(f"completion published after {time.monotonic() - started:.1f}s")
        return persisted
    except BaseException as exc:
        log(f"stopped: {type(exc).__name__}: {exc}")
        raise
    finally:
        m5a._release_run_lock(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--placement-workers", type=int, default=6)
    parser.add_argument("--benchmark", action="store_true", help="seed 0 only; not the study")
    args = parser.parse_args()
    run_study(
        args.output_dir,
        workers=args.workers,
        placement_workers=args.placement_workers,
        mode="benchmark" if args.benchmark else "full",
    )


if __name__ == "__main__":
    main()
