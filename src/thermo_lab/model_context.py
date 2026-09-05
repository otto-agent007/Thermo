"""Deterministic first-moment model contexts for PAsymSwap."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import PAsymSwapFixture
from thermo_lab.pasym_swap_context import (
    OCCUPANCY_ORDER,
    ContextWeights,
    OccupancyVector,
    SupportMask,
)

_CONTEXT_SOURCE = "mean_field_model_pre_gate"
_CONTEXT_REDUCTION = "equal_occurrence_mean_by_target_hash"
_TRACE_POLICY = "one_pass_first_moment_factorization"
_ZERO_SUPPORT_POLICY = "exact_unsmoothed"
_TOLERANCE = 1e-12

ConditionalTable = tuple[tuple[float, float, float, float], ...]


def _probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return result


def _context(value: Sequence[object], *, field: str) -> ContextWeights:
    if len(value) != 4:
        raise ValueError(f"{field} must have four entries")
    result = tuple(
        _probability(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )
    if not math.isclose(math.fsum(result), 1.0, abs_tol=_TOLERANCE, rel_tol=0.0):
        raise ValueError(f"{field} must sum to one")
    return result  # type: ignore[return-value]


def _conditional(value: object, *, field: str) -> ConditionalTable:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a four-row conditional table")
    if len(value) != 4:
        raise ValueError(f"{field} must have four rows")
    rows = tuple(_context(row, field=f"{field}[{index}]") for index, row in enumerate(value))
    return rows


def _occupancy(value: Sequence[object]) -> OccupancyVector:
    if len(value) != len(OCCUPANCY_ORDER):
        raise ValueError("initial_occupancy must have 25 entries")
    result = tuple(
        _probability(item, field=f"initial_occupancy[{index}]") for index, item in enumerate(value)
    )
    if not math.isclose(math.fsum(result), 1.0, abs_tol=_TOLERANCE, rel_tol=0.0):
        raise ValueError("initial_occupancy must sum to one")
    return result


@dataclass(frozen=True)
class ModelContextArtifact:
    """Frozen exact local conditional exposed by an upstream target artifact."""

    artifact_hash: str
    conditional: ConditionalTable

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_hash, str) or not self.artifact_hash:
            raise ValueError("artifact_hash must be nonempty")
        object.__setattr__(self, "conditional", _conditional(self.conditional, field="conditional"))


@dataclass(frozen=True)
class ModelContextOccurrence:
    occurrence_index: int
    target_hash: str
    edge: tuple[tuple[int, int], tuple[int, int]]
    upstream_artifact_hash: str
    context_weights: ContextWeights
    source_mean_before: float
    target_mean_before: float
    source_mean_after: float
    target_mean_after: float
    expected_occupancy_before: float
    expected_occupancy_after: float


@dataclass(frozen=True)
class ModelContextTrace:
    occurrences: tuple[ModelContextOccurrence, ...]
    initial_occupancy: OccupancyVector
    context_source: str = _CONTEXT_SOURCE
    model_trace_policy: str = _TRACE_POLICY
    zero_support_policy: str = _ZERO_SUPPORT_POLICY
    trace_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.context_source != _CONTEXT_SOURCE or self.model_trace_policy != _TRACE_POLICY:
            raise ValueError("unchecked model-context policy")
        object.__setattr__(self, "initial_occupancy", _occupancy(self.initial_occupancy))
        occurrences = tuple(self.occurrences)
        if len(occurrences) != 500:
            raise ValueError("model-context trace must have 500 occurrences")
        if tuple(item.occurrence_index for item in occurrences) != tuple(range(500)):
            raise ValueError("model-context occurrences must be in canonical order")
        object.__setattr__(self, "occurrences", occurrences)
        object.__setattr__(self, "trace_hash", canonical_sha256(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "identity_version": "model_context_trace.v1",
            "initial_occupancy": self.initial_occupancy,
            "context_source": self.context_source,
            "model_trace_policy": self.model_trace_policy,
            "zero_support_policy": self.zero_support_policy,
            "occurrences": tuple(asdict(item) for item in self.occurrences),
        }


@dataclass(frozen=True)
class PooledModelContextProfile:
    trace_hash: str
    target_hash: str
    occurrence_indices: tuple[int, ...]
    multiplicity: int
    context_weights: ContextWeights
    support_mask: SupportMask
    upstream_artifact_hash: str
    profile_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "context_weights", _context(self.context_weights, field="context_weights")
        )
        if self.multiplicity != len(self.occurrence_indices) or self.multiplicity <= 0:
            raise ValueError("profile multiplicity must match occurrence indices")
        if self.support_mask != tuple(weight != 0.0 for weight in self.context_weights):
            raise ValueError("support_mask must match exact context support")
        object.__setattr__(self, "profile_hash", canonical_sha256(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "identity_version": "model_context_profile.v1",
            "trace_hash": self.trace_hash,
            "target_hash": self.target_hash,
            "occurrence_indices": self.occurrence_indices,
            "multiplicity": self.multiplicity,
            "context_weights": self.context_weights,
            "support_mask": self.support_mask,
            "upstream_artifact_hash": self.upstream_artifact_hash,
        }


def derive_model_context_trace(
    fixture: PAsymSwapFixture,
    target_artifacts: Mapping[str, ModelContextArtifact],
    *,
    initial_occupancy: Sequence[object],
) -> ModelContextTrace:
    """Propagate one pass of factorized endpoint means through frozen conditionals."""

    if fixture.side != 5 or len(fixture.occurrences) != 500:
        raise ValueError("fixture must be the canonical 5 by 5 paper schedule")
    occupancy = dict(zip(OCCUPANCY_ORDER, _occupancy(initial_occupancy), strict=True))
    records: list[ModelContextOccurrence] = []
    for index, occurrence in enumerate(fixture.occurrences):
        artifact = target_artifacts.get(occurrence.target_hash)
        if artifact is None:
            raise ValueError("missing target-context artifact")
        source, target = occurrence.edge
        qi, qj = occupancy[source], occupancy[target]
        context = _context(
            ((1.0 - qi) * (1.0 - qj), (1.0 - qi) * qj, qi * (1.0 - qj), qi * qj),
            field="context_weights",
        )
        conditional = artifact.conditional
        next_i = math.fsum(
            context[row] * (conditional[row][2] + conditional[row][3]) for row in range(4)
        )
        next_j = math.fsum(
            context[row] * (conditional[row][1] + conditional[row][3]) for row in range(4)
        )
        next_i = _probability(next_i, field="source_mean_after")
        next_j = _probability(next_j, field="target_mean_after")
        before = math.fsum(occupancy.values())
        occupancy[source], occupancy[target] = next_i, next_j
        after = math.fsum(occupancy.values())
        records.append(
            ModelContextOccurrence(
                index,
                occurrence.target_hash,
                occurrence.edge,
                artifact.artifact_hash,
                context,
                qi,
                qj,
                next_i,
                next_j,
                before,
                after,
            )
        )
    return ModelContextTrace(tuple(records), _occupancy(initial_occupancy))


def pool_model_context_profiles(trace: ModelContextTrace) -> tuple[PooledModelContextProfile, ...]:
    grouped: dict[str, list[ModelContextOccurrence]] = {}
    for occurrence in trace.occurrences:
        grouped.setdefault(occurrence.target_hash, []).append(occurrence)
    profiles = []
    for target_hash in sorted(grouped):
        rows = tuple(sorted(grouped[target_hash], key=lambda item: item.occurrence_index))
        hashes = {row.upstream_artifact_hash for row in rows}
        if len(hashes) != 1:
            raise ValueError("one model profile must have one upstream artifact")
        weights = tuple(
            math.fsum(row.context_weights[index] for row in rows) / len(rows) for index in range(4)
        )
        profiles.append(
            PooledModelContextProfile(
                trace.trace_hash,
                target_hash,
                tuple(row.occurrence_index for row in rows),
                len(rows),
                _context(weights, field="pooled_context_weights"),
                tuple(value != 0.0 for value in weights),
                hashes.pop(),
            )
        )
    if len(profiles) != 37 or Counter(profile.multiplicity for profile in profiles) != {
        10: 26,
        20: 9,
        30: 2,
    }:
        raise ValueError("model trace does not have checked profile multiplicities")
    return tuple(profiles)
