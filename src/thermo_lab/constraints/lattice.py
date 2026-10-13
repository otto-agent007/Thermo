"""The Z1 16-offset lattice: colouring, a placement checker and model lowerings.

Offsets come from the sealed ``Z1HardwareProfile``; M5c placement and the
planar-16-offset-ferro graphs are imported from their hash-bound modules and
never edited. Every offset has odd parity, so any lattice edge joins opposite
colours, and a two-colour (u + v) mod 2 schedule is a valid block-Gibbs sweep.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_topology as m5c
from thermo_lab import planar_16_offset_ferro as planar
from thermo_lab.constraints.model import PairwiseModel
from thermo_lab.constraints.request import ConstraintRequest
from thermo_lab.hardware.z1 import Z1HardwareProfile

PROFILE = Z1HardwareProfile()
OFFSETS = tuple(sorted(PROFILE.interior_offsets()))
NEIGHBOURS = frozenset(OFFSETS)
DEGREE = PROFILE.interior_degree


def colour(site: Sequence[int]) -> int:
    return (int(site[0]) + int(site[1])) % 2


@dataclass(frozen=True)
class LatticeReport:
    edges: int
    offset_violations: int
    colour_violations: int
    max_degree: int
    degree_violations: int
    distinct_sites: bool

    @property
    def ok(self) -> bool:
        return (
            self.offset_violations == 0
            and self.colour_violations == 0
            and self.degree_violations == 0
            and self.distinct_sites
        )


def check(model: PairwiseModel) -> LatticeReport:
    """Every coupled pair is a Z1 offset apart with opposite colours; degree <= 16."""
    if model.sites is None:
        raise ValueError("an unplaced model has no sites to check; relax placement instead")
    sites = model.sites
    bad_offset = bad_colour = 0
    for u, v in model.edges:
        a, b = sites[u], sites[v]
        if (b[0] - a[0], b[1] - a[1]) not in NEIGHBOURS:
            bad_offset += 1
        if colour(a) == colour(b):
            bad_colour += 1
    degree = model.degrees()
    return LatticeReport(
        edges=len(model.edges),
        offset_violations=bad_offset,
        colour_violations=bad_colour,
        max_degree=int(degree.max(initial=0)),
        degree_violations=int((degree > DEGREE).sum()),
        distinct_sites=len(set(sites)) == len(sites),
    )


def require_placement(model: PairwiseModel, request: ConstraintRequest) -> None:
    """Placement applies -> the model is placed and passes ``check``; else it is relaxed."""
    if request.applies("placement"):
        if not model.placed:
            raise ValueError("placement applies but the model is unplaced")
        report = check(model)
        if not report.ok:
            raise ValueError(f"model is not on the Z1 lattice: {report}")
    if model.free_colour_violations():
        raise ValueError("two coupled free nodes share a colour; a two-colour sweep is invalid")


# ---------------------------------------------------------------- M5c kernels


def solve_m5c_layout(vector, structure) -> tuple[list, dict]:
    """M5c's exact minimum-copy MILP placement (imported ``place_kernel``)."""
    return m5c.place_kernel(np.asarray(vector, dtype=np.float64), structure)


def lower_m5c_layout(layout, vector, structure) -> PairwiseModel:
    """An archived or solved M5c layout as a pairwise model with clamped copy nodes.

    Built independently of ``verify_layout``: each coupling to input i goes to the
    first copy of i (in layout order) a lattice offset away. Free order is the
    hidden spins in index order, then y (E0's joint order).
    """
    J, h, A, b, beta = m5a.unpack(np.asarray(vector, dtype=np.float64), structure)
    nodes = [((int(e[0]), int(e[1])), e[2], e[3]) for e in layout]
    y = next(i for i, n in enumerate(nodes) if n[1] == "y")
    hidden = {int(n[2]): i for i, n in enumerate(nodes) if n[1] == "h"}
    copies = [(i, int(n[2])) for i, n in enumerate(nodes) if n[1] == "copy"]
    if len(nodes) != 1 + len(hidden) + len(copies) or sorted(hidden) != list(range(len(beta))):
        raise ValueError("layout does not seat y and every hidden spin exactly once")
    sites = [n[0] for n in nodes]

    def adjacent_copy(i: int, inp: int) -> int:
        for c, k in copies:
            if k == inp and (sites[c][0] - sites[i][0], sites[c][1] - sites[i][1]) in NEIGHBOURS:
                return c
        raise ValueError("no adjacent copy of a coupled input")

    edges, weights = [], []
    for j in np.flatnonzero(J):
        edges.append((y, adjacent_copy(y, int(j))))
        weights.append(J[j])
    for a in range(len(beta)):
        for j in np.flatnonzero(A[a]):
            edges.append((hidden[a], adjacent_copy(hidden[a], int(j))))
            weights.append(A[a, j])
        edges.append((hidden[a], y))
        weights.append(beta[a])
    bias = np.zeros(len(nodes))
    bias[y] = h
    for a, i in hidden.items():
        bias[i] = b[a]
    return PairwiseModel(
        sites=tuple(sites),
        edges=tuple(edges),
        weights=np.asarray(weights, dtype=np.float64),
        bias=bias,
        free=tuple(hidden[a] for a in range(len(beta))) + (y,),
        clamped=tuple(c for c, _ in copies),
        clamped_input=tuple(k for _, k in copies),
        first_colour=colour(sites[hidden[0]]) if hidden else colour(sites[y]),
    )


def place_m5c(vector, structure) -> PairwiseModel:
    """Solve the M5c placement and lower it; raises for kernels over degree 16."""
    layout, _ = solve_m5c_layout(vector, structure)
    return lower_m5c_layout(layout, vector, structure)


# ---------------------------------------------------------------- planar graphs


def planar_target(target: Mapping) -> PairwiseModel:
    """A planar-16-offset-ferro target on its square patch (site = (col, row)).

    Weights are the archive's ``couplings`` as given (its own sign convention);
    zero field, every spin free, colour 0 first.
    """
    positions = planar.target_positions(target)
    return PairwiseModel(
        sites=tuple((int(p[0]), int(p[1])) for p in positions),
        edges=tuple(tuple(e) for e in target["edges"]),
        weights=np.asarray(target["couplings"], dtype=np.float64),
        bias=np.zeros(len(positions)),
        free=tuple(range(len(positions))),
        clamped=(),
        clamped_input=(),
        first_colour=0,
    )


def build_planar_target(size: int, graph: str, seed: int) -> dict:
    """Rebuild a planar target with the hash-bound ``build_target``."""
    return planar.build_target(size, graph, seed)


# ---------------------------------------------------------------- unplaced models


def unplaced(
    model_like: Mapping, colours: Sequence[int], request: ConstraintRequest
) -> PairwiseModel:
    """A model without sites, carrying only a colour per node; needs placement relaxed.

    ``model_like`` holds ``edges``, ``weights``, ``bias``, ``free``, ``clamped``,
    ``clamped_input`` and ``first_colour``.
    """
    if request.applies("placement"):
        raise ValueError("an unplaced model requires 'placement' in request.relaxed")
    return PairwiseModel(
        sites=None,
        colours=tuple(colours),
        edges=tuple(tuple(e) for e in model_like["edges"]),
        weights=model_like["weights"],
        bias=model_like["bias"],
        free=tuple(model_like["free"]),
        clamped=tuple(model_like["clamped"]),
        clamped_input=tuple(model_like["clamped_input"]),
        first_colour=int(model_like["first_colour"]),
    )


def unplaced_m5a_kernel(vector, structure, request: ConstraintRequest) -> PairwiseModel:
    """An M5a site kernel in E0 stage B's node and edge order, unplaced.

    Nodes: inputs x (clamped, colour 0), hidden w (colour 1), output y (colour 0);
    the hidden block is updated first (M5b's hidden-first sweep).
    """
    J, h, A, b, beta = m5a.unpack(np.asarray(vector, dtype=np.float64), structure)
    k, nh = len(J), len(beta)
    edges, weights = [], []
    for j in range(k):
        edges.append((j, k + nh))
        weights.append(J[j])
    for a in range(nh):
        for j in range(k):
            edges.append((j, k + a))
            weights.append(A[a, j])
        edges.append((k + a, k + nh))
        weights.append(beta[a])
    bias = np.zeros(k + nh + 1)
    bias[k : k + nh] = b
    bias[-1] = h
    model_like = {
        "edges": edges,
        "weights": np.asarray(weights, dtype=np.float64),
        "bias": bias,
        "free": list(range(k, k + nh + 1)),
        "clamped": list(range(k)),
        "clamped_input": list(range(k)),
        "first_colour": 1,
    }
    return unplaced(model_like, [0] * k + [1] * nh + [0], request)
