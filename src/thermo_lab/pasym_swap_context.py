"""Exact pre-gate target contexts for the paper PAsymSwap schedule."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any

from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import (
    PAPER_SOURCE,
    WORD_ORDER,
    Coordinate,
    OrientedEdge,
    PAsymSwapFixture,
)

ContextWeights = tuple[float, float, float, float]
OccupancyVector = tuple[float, ...]
SupportMask = tuple[bool, bool, bool, bool]
OCCUPANCY_ORDER = tuple((x, y) for x in range(5) for y in range(5))

_MASS_TOLERANCE = 1e-12
_INITIAL_STATE = "single_particle"
_CONTEXT_SOURCE = "exact_target_pre_gate"
_ZERO_SUPPORT_POLICY = "exact_unsmoothed"
_CONTEXT_REDUCTION = "equal_occurrence_mean_by_target_hash"


def _coordinate(value: object, *, field_name: str) -> Coordinate:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a two-item coordinate sequence")
    coordinate = tuple(value)
    if len(coordinate) != 2:
        raise TypeError(f"{field_name} must be a two-item coordinate sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in coordinate):
        raise TypeError(f"{field_name} must contain integer coordinates")
    return coordinate  # type: ignore[return-value]


def _probability(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number, not a boolean")
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"{field_name} must be finite")
    if probability < 0.0:
        raise ValueError(f"{field_name} must be nonnegative")
    return probability


def _occupancy_vector(value: object, *, field_name: str) -> OccupancyVector:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a probability sequence")
    values = tuple(value)
    if len(values) != len(OCCUPANCY_ORDER):
        raise ValueError(f"{field_name} must contain exactly 25 probabilities")
    occupancy = tuple(
        _probability(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(values)
    )
    if not math.isclose(math.fsum(occupancy), 1.0, abs_tol=_MASS_TOLERANCE, rel_tol=0.0):
        raise ValueError(f"{field_name} must sum to one within {_MASS_TOLERANCE}")
    return occupancy


def _context_weights(value: object) -> ContextWeights:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("context_weights must be a probability sequence")
    values = tuple(value)
    if len(values) != 4:
        raise ValueError("context_weights must contain exactly four probabilities")
    context = tuple(
        _probability(item, field_name=f"context_weights[{index}]")
        for index, item in enumerate(values)
    )
    if not math.isclose(math.fsum(context), 1.0, abs_tol=_MASS_TOLERANCE, rel_tol=0.0):
        raise ValueError(f"context_weights must sum to one within {_MASS_TOLERANCE}")
    return context  # type: ignore[return-value]


def _support_mask(value: object) -> SupportMask:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("support_mask must be a four-item boolean sequence")
    values = tuple(value)
    if len(values) != 4:
        raise ValueError("support_mask must contain exactly four booleans")
    if not all(isinstance(item, bool) for item in values):
        raise TypeError("support_mask must contain only booleans")
    return values  # type: ignore[return-value]


def _occurrence_indices(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("occurrence_indices must be an integer sequence")
    indices = tuple(value)
    if not indices:
        raise ValueError("occurrence_indices must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in indices):
        raise TypeError("occurrence_indices must contain only nonnegative integers")
    if any(item < 0 or item >= 500 for item in indices):
        raise ValueError("occurrence_indices must be between 0 and 499")
    if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
        raise ValueError("occurrence_indices must be strictly ascending")
    return indices  # type: ignore[return-value]


def _nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


@dataclass(frozen=True)
class TargetContextOccurrence:
    """The target input distribution immediately before one gate occurrence."""

    occurrence_index: int
    macrostep: int
    layer: int
    color: str
    edge: OrientedEdge
    target_hash: str
    context_weights: ContextWeights

    def __post_init__(self) -> None:
        if isinstance(self.occurrence_index, bool) or self.occurrence_index < 0:
            raise ValueError("occurrence_index must be a nonnegative integer")
        if isinstance(self.macrostep, bool) or self.macrostep < 0:
            raise ValueError("macrostep must be a nonnegative integer")
        if isinstance(self.layer, bool) or self.layer < 0:
            raise ValueError("layer must be a nonnegative integer")
        edge = (
            _coordinate(self.edge[0], field_name="edge source"),
            _coordinate(self.edge[1], field_name="edge target"),
        )
        if edge[0] == edge[1]:
            raise ValueError("edge endpoints must differ")
        object.__setattr__(self, "edge", edge)
        object.__setattr__(self, "context_weights", _context_weights(self.context_weights))


@dataclass(frozen=True)
class TargetContextTrace:
    """The immutable, canonical target-context evidence for one paper run."""

    source_reference: str
    word_order: tuple[Coordinate, ...]
    initial_state: str
    initial_particle_site: Coordinate
    initial_occupancy_order: tuple[Coordinate, ...]
    initial_occupancy: OccupancyVector
    context_source: str
    zero_support_policy: str
    occurrences: tuple[TargetContextOccurrence, ...]
    trace_hash: str = field(init=False)

    def __post_init__(self) -> None:
        word_order = tuple(
            _coordinate(item, field_name="word_order item") for item in self.word_order
        )
        occupancy_order = tuple(
            _coordinate(item, field_name="initial_occupancy_order item")
            for item in self.initial_occupancy_order
        )
        if len(word_order) != 4:
            raise ValueError("word_order must contain exactly four coordinates")
        if len(occupancy_order) != len(OCCUPANCY_ORDER) or len(set(occupancy_order)) != len(
            OCCUPANCY_ORDER
        ):
            raise ValueError("initial_occupancy_order must contain 25 distinct coordinates")
        occurrences = tuple(self.occurrences)
        if len(occurrences) != 500:
            raise ValueError("occurrences must contain exactly 500 entries")
        if not all(isinstance(item, TargetContextOccurrence) for item in occurrences):
            raise TypeError("occurrences must contain TargetContextOccurrence entries")
        if self.initial_state != _INITIAL_STATE:
            raise ValueError(f"initial_state must be {_INITIAL_STATE!r}")
        if self.context_source != _CONTEXT_SOURCE:
            raise ValueError(f"context_source must be {_CONTEXT_SOURCE!r}")
        if self.zero_support_policy != _ZERO_SUPPORT_POLICY:
            raise ValueError(f"zero_support_policy must be {_ZERO_SUPPORT_POLICY!r}")
        initial_site = _coordinate(self.initial_particle_site, field_name="initial_particle_site")
        initial_occupancy = _occupancy_vector(
            self.initial_occupancy, field_name="initial_occupancy"
        )
        _validate_checked_initial_state(
            initial_state=self.initial_state,
            initial_particle_site=initial_site,
            initial_occupancy=initial_occupancy,
        )
        object.__setattr__(self, "word_order", word_order)
        object.__setattr__(self, "initial_particle_site", initial_site)
        object.__setattr__(self, "initial_occupancy_order", occupancy_order)
        object.__setattr__(self, "initial_occupancy", initial_occupancy)
        object.__setattr__(self, "occurrences", occurrences)
        object.__setattr__(self, "trace_hash", canonical_sha256(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "identity_version": "target_context_trace.v1",
            "source_reference": self.source_reference,
            "word_order": self.word_order,
            "initial_state": self.initial_state,
            "initial_particle_site": self.initial_particle_site,
            "initial_occupancy_order": self.initial_occupancy_order,
            "initial_occupancy": self.initial_occupancy,
            "context_source": self.context_source,
            "zero_support_policy": self.zero_support_policy,
            "occurrences": tuple(asdict(item) for item in self.occurrences),
        }


@dataclass(frozen=True)
class PooledTargetContextProfile:
    """The exact equal-occurrence target context for one shared target channel."""

    trace_hash: str
    target_hash: str
    word_order: tuple[Coordinate, ...]
    context_reduction: str
    zero_support_policy: str
    occurrence_indices: tuple[int, ...]
    multiplicity: int
    context_weights: ContextWeights
    support_mask: SupportMask
    profile_hash: str = field(init=False)

    def __post_init__(self) -> None:
        trace_hash = _nonempty_string(self.trace_hash, field_name="trace_hash")
        target_hash = _nonempty_string(self.target_hash, field_name="target_hash")
        word_order = tuple(
            _coordinate(item, field_name="word_order item") for item in self.word_order
        )
        if len(word_order) != 4:
            raise ValueError("word_order must contain exactly four coordinates")
        if self.context_reduction != _CONTEXT_REDUCTION:
            raise ValueError(f"context_reduction must be {_CONTEXT_REDUCTION!r}")
        if self.zero_support_policy != _ZERO_SUPPORT_POLICY:
            raise ValueError(f"zero_support_policy must be {_ZERO_SUPPORT_POLICY!r}")
        occurrence_indices = _occurrence_indices(self.occurrence_indices)
        if isinstance(self.multiplicity, bool) or not isinstance(self.multiplicity, int):
            raise TypeError("multiplicity must be an integer")
        if self.multiplicity != len(occurrence_indices):
            raise ValueError("multiplicity must equal the number of occurrence_indices")
        context_weights = _context_weights(self.context_weights)
        support_mask = _support_mask(self.support_mask)
        expected_support = tuple(weight != 0.0 for weight in context_weights)
        if support_mask != expected_support:
            raise ValueError("support_mask must exactly match nonzero context_weights")
        object.__setattr__(self, "trace_hash", trace_hash)
        object.__setattr__(self, "target_hash", target_hash)
        object.__setattr__(self, "word_order", word_order)
        object.__setattr__(self, "occurrence_indices", occurrence_indices)
        object.__setattr__(self, "context_weights", context_weights)
        object.__setattr__(self, "support_mask", support_mask)
        object.__setattr__(self, "profile_hash", canonical_sha256(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "identity_version": "target_context_profile.v1",
            "trace_hash": self.trace_hash,
            "target_hash": self.target_hash,
            "word_order": self.word_order,
            "context_reduction": self.context_reduction,
            "zero_support_policy": self.zero_support_policy,
            "occurrence_indices": self.occurrence_indices,
            "multiplicity": self.multiplicity,
            "context_weights": self.context_weights,
            "support_mask": self.support_mask,
        }


def _component_mean(rows: tuple[ContextWeights, ...], index: int) -> float:
    return math.fsum(row[index] for row in rows) / len(rows)


def pool_target_context_profiles(
    trace: TargetContextTrace, *, context_reduction: str
) -> tuple[PooledTargetContextProfile, ...]:
    """Pool a canonical trace into exact equal-occurrence target profiles."""

    if not isinstance(trace, TargetContextTrace):
        raise TypeError("trace must be a TargetContextTrace")
    if context_reduction != _CONTEXT_REDUCTION:
        raise ValueError(f"context_reduction must be {_CONTEXT_REDUCTION!r}")

    occurrences_by_target: dict[str, list[TargetContextOccurrence]] = {}
    for occurrence in trace.occurrences:
        occurrences_by_target.setdefault(occurrence.target_hash, []).append(occurrence)

    profiles: list[PooledTargetContextProfile] = []
    for target_hash in sorted(occurrences_by_target):
        contributors = tuple(
            sorted(
                occurrences_by_target[target_hash],
                key=lambda occurrence: occurrence.occurrence_index,
            )
        )
        rows = tuple(occurrence.context_weights for occurrence in contributors)
        context_weights = tuple(_component_mean(rows, index) for index in range(4))
        profiles.append(
            PooledTargetContextProfile(
                trace_hash=trace.trace_hash,
                target_hash=target_hash,
                word_order=trace.word_order,
                context_reduction=context_reduction,
                zero_support_policy=trace.zero_support_policy,
                occurrence_indices=tuple(
                    occurrence.occurrence_index for occurrence in contributors
                ),
                multiplicity=len(contributors),
                context_weights=context_weights,  # type: ignore[arg-type]
                support_mask=tuple(weight != 0.0 for weight in context_weights),
            )
        )

    multiplicities = Counter(profile.multiplicity for profile in profiles)
    if len(profiles) != 37 or multiplicities != {10: 26, 20: 9, 30: 2}:
        raise ValueError("trace does not have the checked target-profile multiplicities")
    if math.fsum(profile.multiplicity for profile in profiles) != 500.0:
        raise ValueError("profile multiplicities must total 500")
    return tuple(profiles)


def _validate_checked_initial_state(
    *, initial_state: str, initial_particle_site: Coordinate, initial_occupancy: OccupancyVector
) -> None:
    if initial_state != _INITIAL_STATE:
        raise ValueError(f"initial_state must be {_INITIAL_STATE!r}")
    if initial_particle_site != (0, 0):
        raise ValueError("initial_particle_site must be (0, 0)")
    particle_index = OCCUPANCY_ORDER.index(initial_particle_site)
    if initial_occupancy[particle_index] != 1.0 or any(
        mass != 0.0 for index, mass in enumerate(initial_occupancy) if index != particle_index
    ):
        raise ValueError("initial_occupancy must encode the checked one-particle state")


def derive_target_context_trace(
    fixture: PAsymSwapFixture,
    *,
    initial_state: str,
    initial_particle_site: Coordinate,
    initial_occupancy: OccupancyVector,
    context_source: str,
    zero_support_policy: str,
) -> TargetContextTrace:
    """Derive exact, unsmoothed pre-gate target contexts in fixture order."""

    if context_source != _CONTEXT_SOURCE:
        raise ValueError(f"context_source must be {_CONTEXT_SOURCE!r}")
    if zero_support_policy != _ZERO_SUPPORT_POLICY:
        raise ValueError(f"zero_support_policy must be {_ZERO_SUPPORT_POLICY!r}")
    initial_site = _coordinate(initial_particle_site, field_name="initial_particle_site")
    occupancy = _occupancy_vector(initial_occupancy, field_name="initial_occupancy")
    _validate_checked_initial_state(
        initial_state=initial_state,
        initial_particle_site=initial_site,
        initial_occupancy=occupancy,
    )
    if fixture.side != 5 or len(fixture.occurrences) != 500:
        raise ValueError("fixture must be the 5 by 5 paper schedule")

    targets = {target.target_hash: target for target in fixture.targets}
    if len(targets) != len(fixture.targets):
        raise ValueError("fixture targets must have unique target hashes")
    contexts: list[TargetContextOccurrence] = []
    mutable_occupancy = dict(zip(OCCUPANCY_ORDER, occupancy, strict=True))
    for occurrence_index, occurrence in enumerate(fixture.occurrences):
        source, target_site = occurrence.edge
        target = targets.get(occurrence.target_hash)
        if target is None:
            raise ValueError("fixture occurrence references an unknown target hash")
        q_source = mutable_occupancy[source]
        q_target = mutable_occupancy[target_site]
        other_mass = math.fsum(
            mutable_occupancy[site]
            for site in OCCUPANCY_ORDER
            if site != source and site != target_site
        )
        context = (other_mass, q_target, q_source, 0.0)
        contexts.append(
            TargetContextOccurrence(
                occurrence_index=occurrence_index,
                macrostep=occurrence.macrostep,
                layer=occurrence.layer,
                color=occurrence.color,
                edge=occurrence.edge,
                target_hash=occurrence.target_hash,
                context_weights=context,
            )
        )
        next_source = math.fsum(((1.0 - target.p_ij) * q_source, target.p_ji * q_target))
        next_target = math.fsum((target.p_ij * q_source, (1.0 - target.p_ji) * q_target))
        mutable_occupancy[source] = next_source
        mutable_occupancy[target_site] = next_target
        if any(not math.isfinite(mass) or mass < 0.0 for mass in mutable_occupancy.values()):
            raise ValueError("target propagation produced invalid occupancy mass")
        if not math.isclose(
            math.fsum(mutable_occupancy[site] for site in OCCUPANCY_ORDER),
            1.0,
            abs_tol=_MASS_TOLERANCE,
            rel_tol=0.0,
        ):
            raise ValueError("target propagation did not conserve mass")

    return TargetContextTrace(
        source_reference=PAPER_SOURCE,
        word_order=WORD_ORDER,
        initial_state=initial_state,
        initial_particle_site=initial_site,
        initial_occupancy_order=OCCUPANCY_ORDER,
        initial_occupancy=occupancy,
        context_source=context_source,
        zero_support_policy=zero_support_policy,
        occurrences=tuple(contexts),
    )
