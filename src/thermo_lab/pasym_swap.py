"""Paper PAsymSwap channels and the immutable five-by-five torus fixture."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from thermo_lab.hashing import canonical_sha256

Coordinate = tuple[int, int]
OrientedEdge = tuple[Coordinate, Coordinate]
CoordinatePairClass = tuple[tuple[int, int], ...]

WORD_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
PAPER_SOURCE = "https://arxiv.org/abs/2608.01615v2"
COLOR_ORDER = ("H1", "H2", "H3", "V1", "V2", "V3")
COORDINATE_PAIR_CLASSES: Mapping[str, CoordinatePairClass] = MappingProxyType(
    {
        "H1": ((0, 1), (2, 3)),
        "H2": ((1, 2), (3, 4)),
        "H3": ((4, 0),),
        "V1": ((0, 1), (2, 3)),
        "V2": ((1, 2), (3, 4)),
        "V3": ((4, 0),),
    }
)

ConditionalTable = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class PAsymSwapTarget:
    """One canonical input-major PAsymSwap target channel."""

    p_ij: float
    p_ji: float
    conditional: ConditionalTable
    target_hash: str


@dataclass(frozen=True)
class GateOccurrence:
    """One application of a canonical target channel in the paper schedule."""

    macrostep: int
    layer: int
    color: str
    edge: OrientedEdge
    target_hash: str


@dataclass(frozen=True)
class PAsymSwapFixture:
    """The paper's finite colored-torus gate schedule and canonical targets."""

    side: int
    color_order: tuple[str, ...]
    color_classes: Mapping[str, tuple[OrientedEdge, ...]]
    targets: tuple[PAsymSwapTarget, ...]
    occurrences: tuple[GateOccurrence, ...]


def build_pasym_swap_conditional(p_ij: float, p_ji: float) -> ConditionalTable:
    """Build the input-major PAsymSwap conditional for an oriented edge."""
    if not 0.0 < p_ij < 1.0 or not 0.0 < p_ji < 1.0:
        raise ValueError("PAsymSwap hop probabilities must lie strictly between zero and one")
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, _unit_complement(p_ji), p_ji, 0.0),
        (0.0, p_ij, _unit_complement(p_ij), 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _unit_complement(probability: float) -> float:
    """Return a display-stable complement without sacrificing an exact row sum."""
    complement = 1.0 - probability
    rounded = round(complement, 15)
    return rounded if rounded + probability == 1.0 else complement


def paper_logit(x: int, y: int) -> float:
    """Return the paper's logit at a torus coordinate."""
    return 2.0 * math.sin(2.0 * math.pi * ((2 * x + y) / 5.0 + 0.2)) + 0.75 * math.cos(
        2.0 * math.pi * ((x - 2 * y) / 5.0 - 0.4)
    )


def hop_probability(source: Coordinate, target: Coordinate) -> float:
    """Return the directed paper hop probability for an oriented edge."""
    delta = paper_logit(*target) - paper_logit(*source)
    return 2.0 * (1.0 / (1.0 + math.exp(-delta))) * 0.05


def _build_color_classes(side: int) -> Mapping[str, tuple[OrientedEdge, ...]]:
    color_classes: dict[str, tuple[OrientedEdge, ...]] = {}
    for color in COLOR_ORDER:
        pairs = COORDINATE_PAIR_CLASSES[color]
        if color.startswith("H"):
            edges = tuple(
                ((source_x, y), (target_x, y)) for source_x, target_x in pairs for y in range(side)
            )
        else:
            edges = tuple(
                ((x, source_y), (x, target_y)) for source_y, target_y in pairs for x in range(side)
            )
        color_classes[color] = edges
    _verify_color_classes(color_classes, side)
    return MappingProxyType(color_classes)


def _undirected(edge: OrientedEdge) -> frozenset[Coordinate]:
    return frozenset(edge)


def _verify_color_classes(color_classes: Mapping[str, tuple[OrientedEdge, ...]], side: int) -> None:
    if tuple(color_classes) != COLOR_ORDER:
        raise ValueError("PAsymSwap color classes must follow the canonical color order")
    for color, edges in color_classes.items():
        vertices = [vertex for edge in edges for vertex in edge]
        if len(vertices) != len(set(vertices)):
            raise ValueError(f"PAsymSwap color class {color} is not a matching")

    macrostep_edges = {_undirected(edge) for color in COLOR_ORDER for edge in color_classes[color]}
    if len(macrostep_edges) != 2 * side * side:
        raise ValueError("PAsymSwap macrostep must cover each torus edge exactly once")


def _target_for_edge(edge: OrientedEdge) -> PAsymSwapTarget:
    source, target = edge
    p_ij = hop_probability(source, target)
    p_ji = hop_probability(target, source)
    conditional = build_pasym_swap_conditional(p_ij=p_ij, p_ji=p_ji)
    target_hash = canonical_sha256({"word_order": WORD_ORDER, "conditional": conditional})
    return PAsymSwapTarget(
        p_ij=p_ij,
        p_ji=p_ji,
        conditional=conditional,
        target_hash=target_hash,
    )


def build_paper_fixture(*, reverse_edge_enumeration: bool = False) -> PAsymSwapFixture:
    """Build the paper's immutable 500-occurrence colored-torus fixture.

    ``reverse_edge_enumeration`` exists only to make the canonical identity
    contract executable: it changes intermediate target discovery order, not
    the resulting sorted target collection or occurrence schedule.
    """
    side = 5
    color_classes = _build_color_classes(side)
    edges = tuple(edge for color in COLOR_ORDER for edge in color_classes[color])
    target_edges = tuple(reversed(edges)) if reverse_edge_enumeration else edges

    targets_by_hash: dict[str, PAsymSwapTarget] = {}
    for edge in target_edges:
        target = _target_for_edge(edge)
        targets_by_hash.setdefault(target.target_hash, target)
    targets = tuple(sorted(targets_by_hash.values(), key=lambda target: target.target_hash))

    edge_hashes = {edge: _target_for_edge(edge).target_hash for edge in edges}
    occurrences = tuple(
        GateOccurrence(
            macrostep=macrostep,
            layer=macrostep * len(COLOR_ORDER) + color_index,
            color=color,
            edge=edge,
            target_hash=edge_hashes[edge],
        )
        for macrostep in range(10)
        for color_index, color in enumerate(COLOR_ORDER)
        for edge in color_classes[color]
    )
    target_hashes = {target.target_hash for target in targets}
    if any(occurrence.target_hash not in target_hashes for occurrence in occurrences):
        raise ValueError("PAsymSwap occurrence references an unknown target hash")

    for macrostep in range(10):
        macrostep_edges = {
            _undirected(occurrence.edge)
            for occurrence in occurrences
            if occurrence.macrostep == macrostep
        }
        if len(macrostep_edges) != 2 * side * side:
            raise ValueError("PAsymSwap macrostep must cover 50 distinct undirected edges")

    return PAsymSwapFixture(
        side=side,
        color_order=COLOR_ORDER,
        color_classes=color_classes,
        targets=targets,
        occurrences=occurrences,
    )
