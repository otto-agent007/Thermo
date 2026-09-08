"""Digest-bound scientific measurements reconstructed from checkpoint counts."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from thermo_lab.composed_pasym_swap import (
    CellTrajectorySource,
    CheckpointSource,
    ComposedSamplingSources,
)
from thermo_lab.composed_pasym_swap_artifacts import (
    ARTIFACT_FAMILIES,
    HORIZON_LABELS,
    ArtifactFamily,
    ComposedArtifactBundle,
    ExactTargetCheckpoint,
    HorizonLabel,
    validate_exact_target_checkpoints,
)
from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import build_paper_fixture

_SITE_COUNT = 25
_TARGET_MASS_TOLERANCE = 1e-12
_ROUND_OFF_FACTOR = 64.0

FiniteVector25: TypeAlias = tuple[StrictFloat, ...]
SectorProbabilities: TypeAlias = tuple[StrictFloat, StrictFloat, StrictFloat]
FiniteFloat: TypeAlias = StrictFloat
CountVector25: TypeAlias = tuple[StrictInt, ...]

_CHECKPOINT_OCCURRENCES = tuple(range(0, 501, 50))
_RELEASE_BATCH_SIZE = 32_768
_PAIR_LABELS = (
    "target_context_minus_independent",
    "model_context_minus_target_context",
    "model_context_minus_independent",
)
_DIFFERENCE_KEYS = (
    "occupancy_half_l1_error",
    "maximum_site_occupancy_error",
    "particle_number_leakage",
    "expected_particle_count",
    "signed_mass_drift",
    "particle_count_variance",
    "ever_left_sector_probability",
    "conditional_location_half_l1_error",
)


class _StrictFrozenMetricsModel(BaseModel):
    """Strict immutable base for persisted float64 checkpoint measurements."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)


def _strict_finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite strict float")
    return value


def _finite_tuple(value: object, *, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise ValueError(f"{name} must be a tuple of exactly {length} finite strict floats")
    return tuple(
        _strict_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )


def _json_array_to_tuple(value: object, info: object) -> object:
    """Restore immutable tuple vectors when strict persisted JSON is reloaded."""

    if getattr(info, "mode", None) == "json" and isinstance(value, list):
        return tuple(value)
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _target_digest(target: tuple[float, ...]) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_sampling_checkpoint_target.v1",
            "occupancy": target,
        }
    )


def _metrics_digest(
    *,
    source_digest: str,
    target_digest: str,
    occupancy: tuple[float, ...],
    occupancy_half_l1_error: float,
    maximum_site_occupancy_error: float,
    particle_number_leakage: float,
    sector_probabilities: tuple[float, float, float],
    expected_particle_count: float,
    signed_mass_drift: float,
    particle_count_variance: float,
    ever_left_sector_probability: float,
    conditional_location: tuple[float, ...] | None,
    conditional_location_half_l1_error: float | None,
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_sampling_checkpoint_metrics.v1",
            "source_digest": source_digest,
            "target_digest": target_digest,
            "occupancy": occupancy,
            "occupancy_half_l1_error": occupancy_half_l1_error,
            "maximum_site_occupancy_error": maximum_site_occupancy_error,
            "particle_number_leakage": particle_number_leakage,
            "sector_probabilities": sector_probabilities,
            "expected_particle_count": expected_particle_count,
            "signed_mass_drift": signed_mass_drift,
            "particle_count_variance": particle_count_variance,
            "ever_left_sector_probability": ever_left_sector_probability,
            "conditional_location": conditional_location,
            "conditional_location_half_l1_error": conditional_location_half_l1_error,
        }
    )


def _close(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= _ROUND_OFF_FACTOR * sys.float_info.epsilon * scale


def _checked_target(value: object) -> tuple[float, ...]:
    target = _finite_tuple(value, name="target", length=_SITE_COUNT)
    if any(probability < 0.0 or probability > 1.0 for probability in target):
        raise ValueError("target must contain probabilities in [0, 1]")
    if not math.isclose(math.fsum(target), 1.0, rel_tol=0.0, abs_tol=_TARGET_MASS_TOLERANCE):
        raise ValueError("target must conserve unit mass")
    return target


def _variance_roundoff_bound(second_moment: float, expected_mass: float) -> float:
    """Bound cancellation error in ``E[N²] - E[N]²`` at the observed scale."""

    scale = max(1.0, abs(second_moment), abs(expected_mass * expected_mass))
    return _ROUND_OFF_FACTOR * sys.float_info.epsilon * scale


class CheckpointMetrics(_StrictFrozenMetricsModel):
    """Finite float64 metrics, bound to one source and exact target occupancy."""

    source_digest: str
    target_digest: str
    occupancy: FiniteVector25
    occupancy_half_l1_error: StrictFloat
    maximum_site_occupancy_error: StrictFloat
    particle_number_leakage: StrictFloat
    sector_probabilities: SectorProbabilities
    expected_particle_count: StrictFloat
    signed_mass_drift: StrictFloat
    particle_count_variance: StrictFloat
    ever_left_sector_probability: StrictFloat
    conditional_location: FiniteVector25 | None
    conditional_location_half_l1_error: StrictFloat | None
    metrics_digest: str | None = None

    @field_validator("source_digest", "target_digest", mode="before")
    @classmethod
    def validate_input_digest(cls, value: object, info: object) -> str:
        return _digest(value, name=getattr(info, "field_name", "digest"))

    @field_validator("metrics_digest")
    @classmethod
    def validate_metrics_digest_shape(cls, value: str | None) -> str | None:
        return None if value is None else _digest(value, name="metrics_digest")

    @field_validator("occupancy", mode="before")
    @classmethod
    def validate_occupancy(cls, value: object, info: object) -> tuple[float, ...]:
        return _finite_tuple(
            _json_array_to_tuple(value, info), name="occupancy", length=_SITE_COUNT
        )

    @field_validator("sector_probabilities", mode="before")
    @classmethod
    def validate_sector_probabilities(
        cls, value: object, info: object
    ) -> tuple[float, float, float]:
        checked = _finite_tuple(
            _json_array_to_tuple(value, info), name="sector_probabilities", length=3
        )
        return checked[0], checked[1], checked[2]

    @field_validator("conditional_location", mode="before")
    @classmethod
    def validate_conditional_location(cls, value: object, info: object) -> tuple[float, ...] | None:
        return (
            None
            if value is None
            else _finite_tuple(
                _json_array_to_tuple(value, info), name="conditional_location", length=_SITE_COUNT
            )
        )

    @field_validator(
        "occupancy_half_l1_error",
        "maximum_site_occupancy_error",
        "particle_number_leakage",
        "expected_particle_count",
        "signed_mass_drift",
        "particle_count_variance",
        "ever_left_sector_probability",
        "conditional_location_half_l1_error",
        mode="before",
    )
    @classmethod
    def validate_metric_float(cls, value: object, info: object) -> float | None:
        if (
            value is None
            and getattr(info, "field_name", "") == "conditional_location_half_l1_error"
        ):
            return None
        return _strict_finite_float(value, name=getattr(info, "field_name", "metric"))

    @model_validator(mode="after")
    def validate_metrics(self) -> CheckpointMetrics:
        if any(probability < 0.0 or probability > 1.0 for probability in self.occupancy):
            raise ValueError("occupancy must contain probabilities in [0, 1]")
        if any(probability < 0.0 or probability > 1.0 for probability in self.sector_probabilities):
            raise ValueError("sector_probabilities must lie in [0, 1]")
        if not _close(math.fsum(self.sector_probabilities), 1.0):
            raise ValueError("sector_probabilities must sum to one")
        if not _close(
            self.particle_number_leakage,
            self.sector_probabilities[0] + self.sector_probabilities[2],
        ):
            raise ValueError("particle_number_leakage must match sector_probabilities")
        if not _close(self.expected_particle_count, math.fsum(self.occupancy)):
            raise ValueError("expected_particle_count must match occupancy")
        if not _close(self.signed_mass_drift, self.expected_particle_count - 1.0):
            raise ValueError("signed_mass_drift must match expected_particle_count")
        if self.particle_count_variance < 0.0:
            raise ValueError("particle_count_variance must be nonnegative")
        if not 0.0 <= self.ever_left_sector_probability <= 1.0:
            raise ValueError("ever_left_sector_probability must lie in [0, 1]")
        if self.conditional_location is None:
            if self.conditional_location_half_l1_error is not None:
                raise ValueError(
                    "conditional_location error must be absent with conditional_location"
                )
        else:
            if any(
                probability < 0.0 or probability > 1.0 for probability in self.conditional_location
            ):
                raise ValueError("conditional_location must contain probabilities in [0, 1]")
            if not _close(math.fsum(self.conditional_location), 1.0):
                raise ValueError("conditional_location must sum to one")
            if self.conditional_location_half_l1_error is None:
                raise ValueError("conditional_location error is required with conditional_location")
        expected = _metrics_digest(
            source_digest=self.source_digest,
            target_digest=self.target_digest,
            occupancy=tuple(self.occupancy),
            occupancy_half_l1_error=self.occupancy_half_l1_error,
            maximum_site_occupancy_error=self.maximum_site_occupancy_error,
            particle_number_leakage=self.particle_number_leakage,
            sector_probabilities=tuple(self.sector_probabilities),
            expected_particle_count=self.expected_particle_count,
            signed_mass_drift=self.signed_mass_drift,
            particle_count_variance=self.particle_count_variance,
            ever_left_sector_probability=self.ever_left_sector_probability,
            conditional_location=(
                tuple(self.conditional_location) if self.conditional_location is not None else None
            ),
            conditional_location_half_l1_error=self.conditional_location_half_l1_error,
        )
        if self.metrics_digest is None:
            object.__setattr__(self, "metrics_digest", expected)
        elif self.metrics_digest != expected:
            raise ValueError("metrics_digest does not bind the checkpoint metric payload")
        return self


def derive_checkpoint_metrics(
    source: CheckpointSource, target: tuple[float, ...]
) -> CheckpointMetrics:
    """Rebuild deterministic float64 checkpoint metrics from bounded integer counts."""

    if not isinstance(source, CheckpointSource):
        raise TypeError("source must be a CheckpointSource")
    checked_source = CheckpointSource.model_validate(source.model_dump())
    checked_target = _checked_target(target)
    sample_count = float(checked_source.sample_count)
    occupancy = tuple(float(count) / sample_count for count in checked_source.occupancy_counts)
    errors = tuple(
        abs(actual - expected) for actual, expected in zip(occupancy, checked_target, strict=True)
    )
    expected_mass = math.fsum((float(checked_source.particle_count_sum) / sample_count,))
    second_moment = math.fsum((float(checked_source.particle_count_sum_squares) / sample_count,))
    variance = math.fsum((second_moment, -(expected_mass * expected_mass)))
    if variance < -_variance_roundoff_bound(second_moment, expected_mass):
        raise ValueError("particle-count variance is materially negative")
    variance = max(0.0, variance)
    one_count = checked_source.sector_counts[1]
    conditional = (
        tuple(
            float(count) / float(one_count) for count in checked_source.one_particle_location_counts
        )
        if one_count
        else None
    )
    conditional_error = (
        0.5
        * math.fsum(
            abs(actual - expected)
            for actual, expected in zip(conditional, checked_target, strict=True)
        )
        if conditional is not None
        else None
    )
    return CheckpointMetrics(
        source_digest=checked_source.source_digest,
        target_digest=_target_digest(checked_target),
        occupancy=occupancy,
        occupancy_half_l1_error=0.5 * math.fsum(errors),
        maximum_site_occupancy_error=max(errors),
        particle_number_leakage=(
            float(checked_source.sector_counts[0] + checked_source.sector_counts[2]) / sample_count
        ),
        sector_probabilities=tuple(
            float(count) / sample_count for count in checked_source.sector_counts
        ),
        expected_particle_count=expected_mass,
        signed_mass_drift=expected_mass - 1.0,
        particle_count_variance=variance,
        ever_left_sector_probability=float(checked_source.ever_left_count) / sample_count,
        conditional_location=conditional,
        conditional_location_half_l1_error=conditional_error,
    )


def validate_checkpoint_metrics(
    value: object, *, source: CheckpointSource, target: tuple[float, ...]
) -> CheckpointMetrics:
    """Strictly reload and exactly regenerate one persisted checkpoint metric payload."""

    if isinstance(value, (str, bytes, bytearray)):
        observed = CheckpointMetrics.model_validate_json(value)
    else:
        observed = CheckpointMetrics.model_validate(value)
    expected = derive_checkpoint_metrics(source, target)
    if observed != expected:
        raise ValueError("persisted checkpoint metrics do not match regenerated metrics")
    return observed


# These aliases deliberately reuse the bounded integer source and its derived
# metric model.  A composed summary must not grow a second, weaker source or
# metric schema beside the release-gated contracts above.
CheckpointSourceResult = CheckpointSource
CheckpointMetricsResult = CheckpointMetrics


class CheckpointResult(_StrictFrozenMetricsModel):
    """One source, exact target, and fully regenerated metric result."""

    source: CheckpointSourceResult
    exact_target_occupancy: FiniteVector25
    metrics: CheckpointMetricsResult

    @field_validator("exact_target_occupancy", mode="before")
    @classmethod
    def validate_target_occupancy(cls, value: object, info: object) -> tuple[float, ...]:
        return _checked_target(_json_array_to_tuple(value, info))

    @model_validator(mode="after")
    def validate_checkpoint_result(self) -> CheckpointResult:
        # Do not rely on the nested metric digest: it is only an internal
        # integrity marker.  Rebuild it from the persisted integer source.
        validate_checkpoint_metrics(
            self.metrics, source=self.source, target=tuple(self.exact_target_occupancy)
        )
        return self


class PairedCheckpointComparison(_StrictFrozenMetricsModel):
    """First-minus-second scalar metric differences at one paired checkpoint."""

    label: Literal[
        "target_context_minus_independent",
        "model_context_minus_target_context",
        "model_context_minus_independent",
    ]
    horizon: HorizonLabel
    occurrence_count: StrictInt
    differences: Mapping[str, FiniteFloat | None]

    @field_validator("occurrence_count", mode="before")
    @classmethod
    def validate_comparison_occurrence(cls, value: object) -> int:
        if type(value) is not int or value not in _CHECKPOINT_OCCURRENCES:
            raise ValueError("occurrence_count must be a canonical checkpoint occurrence")
        return value

    @field_validator("differences", mode="before")
    @classmethod
    def validate_differences(cls, value: object) -> dict[str, float | None]:
        if not isinstance(value, Mapping) or set(value) != set(_DIFFERENCE_KEYS):
            raise ValueError("differences must use exactly the canonical paired metric keys")
        checked: dict[str, float | None] = {}
        for name in _DIFFERENCE_KEYS:
            item = value[name]
            if item is None:
                if name != "conditional_location_half_l1_error":
                    raise ValueError(f"{name} difference must be a finite strict float")
                checked[name] = None
            else:
                checked[name] = _strict_finite_float(item, name=name)
        return checked

    @model_validator(mode="after")
    def freeze_differences(self) -> PairedCheckpointComparison:
        object.__setattr__(
            self,
            "differences",
            MappingProxyType({name: self.differences[name] for name in _DIFFERENCE_KEYS}),
        )
        return self

    @field_serializer("differences")
    def serialize_differences(self, value: Mapping[str, float | None]) -> dict[str, float | None]:
        return {name: value[name] for name in _DIFFERENCE_KEYS}

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> PairedCheckpointComparison:
        """Revalidate public copy updates instead of exposing Pydantic's unchecked path."""

        del deep  # Every persisted member is immutable; validation rebuilds the mapping proxy.
        payload = self.model_dump()
        if update is not None:
            payload.update(update)
        return type(self).model_validate(payload)

    @property
    def occupancy_half_l1_difference(self) -> float:
        value = self.differences["occupancy_half_l1_error"]
        assert value is not None
        return value

    @property
    def particle_number_leakage_difference(self) -> float:
        value = self.differences["particle_number_leakage"]
        assert value is not None
        return value


def derive_paired_checkpoint_comparison(
    *,
    first: CheckpointMetrics,
    second: CheckpointMetrics,
    label: Literal[
        "target_context_minus_independent",
        "model_context_minus_target_context",
        "model_context_minus_independent",
    ],
    horizon: HorizonLabel = "equilibrium",
    occurrence_count: int = 0,
) -> PairedCheckpointComparison:
    """Derive canonical *first minus second* paired scalar differences."""

    if not isinstance(first, CheckpointMetrics) or not isinstance(second, CheckpointMetrics):
        raise TypeError("first and second must be CheckpointMetrics")
    differences = {
        name: (
            None
            if getattr(first, name) is None or getattr(second, name) is None
            else float(getattr(first, name) - getattr(second, name))
        )
        for name in _DIFFERENCE_KEYS
    }
    return PairedCheckpointComparison(
        label=label,
        horizon=horizon,
        occurrence_count=occurrence_count,
        differences=differences,
    )


def _cell_digest(
    *, family: ArtifactFamily, horizon: HorizonLabel, checkpoints: tuple[CheckpointResult, ...]
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_pasym_swap_cell_result.v1",
            "family": family,
            "horizon": horizon,
            "checkpoints": checkpoints,
        }
    )


class ComposedCellResult(_StrictFrozenMetricsModel):
    """All eleven exact-target comparisons for one family/horizon cell."""

    family: ArtifactFamily
    horizon: HorizonLabel
    checkpoints: tuple[CheckpointResult, ...]
    cell_digest: str

    @field_validator("checkpoints", mode="before")
    @classmethod
    def validate_checkpoint_tuple(cls, value: object, info: object) -> object:
        value = _json_array_to_tuple(value, info)
        if not isinstance(value, tuple):
            raise ValueError("checkpoints must be a tuple")
        return value

    @field_validator("cell_digest")
    @classmethod
    def validate_cell_digest_shape(cls, value: str) -> str:
        return _digest(value, name="cell_digest")

    @model_validator(mode="after")
    def validate_cell_result(self) -> ComposedCellResult:
        if (
            tuple(item.source.occurrence_count for item in self.checkpoints)
            != _CHECKPOINT_OCCURRENCES
        ):
            raise ValueError("checkpoints must use the exact eleven-checkpoint order")
        expected = _cell_digest(
            family=self.family, horizon=self.horizon, checkpoints=tuple(self.checkpoints)
        )
        if self.cell_digest != expected:
            raise ValueError("cell_digest does not bind the composed cell payload")
        return self


class LocalResidualSummary(_StrictFrozenMetricsModel):
    """Separately reported local-table residual range; never a composed error."""

    family: ArtifactFamily
    horizon: HorizonLabel
    minimum: FiniteFloat
    median: FiniteFloat
    maximum: FiniteFloat

    @field_validator("minimum", "median", "maximum", mode="before")
    @classmethod
    def validate_local_residual_float(cls, value: object, info: object) -> float:
        return _strict_finite_float(value, name=getattr(info, "field_name", "local residual"))

    @model_validator(mode="after")
    def validate_local_residual_range(self) -> LocalResidualSummary:
        if self.minimum < 0.0 or not self.minimum <= self.median <= self.maximum:
            raise ValueError("local residual summary must be a nonnegative ordered range")
        return self


class ArtifactIdentityResult(_StrictFrozenMetricsModel):
    """The scientific artifact plus optimizer evidence identity for one target."""

    family: ArtifactFamily
    target_hash: str
    artifact_hash: str
    optimizer_result_hash: str
    parameter_vector: tuple[FiniteFloat, ...]

    @field_validator("target_hash", "artifact_hash", "optimizer_result_hash")
    @classmethod
    def validate_identity_digest(cls, value: str, info: object) -> str:
        return _digest(value, name=getattr(info, "field_name", "artifact digest"))

    @field_validator("parameter_vector", mode="before")
    @classmethod
    def validate_parameter_vector(cls, value: object, info: object) -> tuple[float, ...]:
        return _finite_tuple(_json_array_to_tuple(value, info), name="parameter_vector", length=9)


def _target_checkpoint_digest(checkpoints: tuple[ExactTargetCheckpoint, ...]) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_pasym_swap_summary_targets.v1",
            "checkpoints": tuple(
                {
                    "occurrence_count": item.occurrence_count,
                    "occupancy": tuple(item.occupancy),
                    "evidence_class": item.evidence_class,
                    "exact_reference": item.exact_reference,
                }
                for item in checkpoints
            ),
        }
    )


def _summary_digest(
    *,
    request_hash: str,
    bundle_digest: str,
    target_checkpoint_digest: str,
    seed: int,
    batch_size: int,
    artifact_identities: tuple[ArtifactIdentityResult, ...],
    local_residual_summaries: tuple[LocalResidualSummary, ...],
    cells: tuple[ComposedCellResult, ...],
    comparisons: tuple[PairedCheckpointComparison, ...],
    integrity_acceptance_passed: bool,
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_pasym_swap_summary.v1",
            "request_hash": request_hash,
            "bundle_digest": bundle_digest,
            "target_checkpoint_digest": target_checkpoint_digest,
            "seed": seed,
            "batch_size": batch_size,
            "artifact_identities": artifact_identities,
            "local_residual_summaries": local_residual_summaries,
            "cells": cells,
            "comparisons": comparisons,
            "integrity_acceptance_passed": integrity_acceptance_passed,
        }
    )


class ComposedPAsymSwapSummary(_StrictFrozenMetricsModel):
    """Strict persisted evidence for one seeded full composed-program run."""

    identity_version: Literal["composed_pasym_swap_summary.v1"]
    request_hash: str
    bundle_digest: str
    target_checkpoint_digest: str
    seed: StrictInt
    batch_size: StrictInt
    artifact_identities: tuple[ArtifactIdentityResult, ...]
    local_residual_summaries: tuple[LocalResidualSummary, ...]
    cells: tuple[ComposedCellResult, ...]
    comparisons: tuple[PairedCheckpointComparison, ...]
    integrity_acceptance_passed: StrictBool
    summary_digest: str

    @field_validator("request_hash", "bundle_digest", "target_checkpoint_digest", "summary_digest")
    @classmethod
    def validate_summary_digest_shapes(cls, value: str, info: object) -> str:
        return _digest(value, name=getattr(info, "field_name", "summary digest"))

    @field_validator(
        "artifact_identities", "local_residual_summaries", "cells", "comparisons", mode="before"
    )
    @classmethod
    def normalize_summary_tuples(cls, value: object, info: object) -> object:
        value = _json_array_to_tuple(value, info)
        if not isinstance(value, tuple):
            raise ValueError(f"{getattr(info, 'field_name', 'summary entries')} must be a tuple")
        return value

    @model_validator(mode="after")
    def validate_summary(self) -> ComposedPAsymSwapSummary:
        if self.batch_size != _RELEASE_BATCH_SIZE:
            raise ValueError("summary batch_size must equal the release count 32,768")
        expected_cell_keys = tuple(
            (family, horizon) for family in ARTIFACT_FAMILIES for horizon in HORIZON_LABELS
        )
        if len(self.artifact_identities) != 3 * 37 or tuple(
            item.family for item in self.artifact_identities
        ) != tuple(family for family in ARTIFACT_FAMILIES for _ in range(37)):
            raise ValueError(
                "artifact identities must cover 3 families by 37 targets in family order"
            )
        if (
            tuple((item.family, item.horizon) for item in self.local_residual_summaries)
            != expected_cell_keys
        ):
            raise ValueError("local residual summaries must use family-major horizon-minor order")
        if tuple((item.family, item.horizon) for item in self.cells) != expected_cell_keys:
            raise ValueError("cells must use family-major horizon-minor order")
        expected_comparisons = tuple(
            (horizon, occurrence, label)
            for horizon in HORIZON_LABELS
            for occurrence in _CHECKPOINT_OCCURRENCES
            for label in _PAIR_LABELS
        )
        if (
            tuple((item.horizon, item.occurrence_count, item.label) for item in self.comparisons)
            != expected_comparisons
        ):
            raise ValueError("comparisons must contain all paired labels in canonical order")
        expected = _summary_digest(
            request_hash=self.request_hash,
            bundle_digest=self.bundle_digest,
            target_checkpoint_digest=self.target_checkpoint_digest,
            seed=self.seed,
            batch_size=self.batch_size,
            artifact_identities=tuple(self.artifact_identities),
            local_residual_summaries=tuple(self.local_residual_summaries),
            cells=tuple(self.cells),
            comparisons=tuple(self.comparisons),
            integrity_acceptance_passed=self.integrity_acceptance_passed,
        )
        if self.summary_digest != expected:
            raise ValueError("summary_digest does not bind the composed summary payload")
        return self


def _checked_bundle(bundle: object) -> ComposedArtifactBundle:
    if not isinstance(bundle, ComposedArtifactBundle):
        raise TypeError("bundle must be a ComposedArtifactBundle")
    return ComposedArtifactBundle.model_validate(bundle.model_dump())


def _checked_targets(value: object) -> tuple[ExactTargetCheckpoint, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(item, ExactTargetCheckpoint) for item in value
    ):
        raise TypeError("target_checkpoints must be a tuple of ExactTargetCheckpoint entries")
    checked = tuple(ExactTargetCheckpoint.model_validate(item.model_dump()) for item in value)
    regenerated = validate_exact_target_checkpoints(
        checked, build_paper_fixture(), _CHECKPOINT_OCCURRENCES
    )
    if checked != regenerated:
        raise ValueError("target_checkpoints differ from regenerated exact evidence")
    return regenerated


def _checked_sources(value: object, *, bundle: ComposedArtifactBundle) -> ComposedSamplingSources:
    if not isinstance(value, ComposedSamplingSources):
        raise TypeError("sources must be ComposedSamplingSources")
    checked = ComposedSamplingSources.model_validate(value.model_dump())
    if checked.bundle_digest != bundle.bundle_digest:
        raise ValueError("sources must be bound to the supplied artifact bundle")
    return checked


def _residual_summary(
    *, family: ArtifactFamily, horizon: HorizonLabel, values: object
) -> LocalResidualSummary:
    # The bundle has already fixed shape and finiteness; materialize normal
    # Python float values so persisted strict-float rules remain explicit.
    row = tuple(float(value) for value in values)
    ordered = tuple(sorted(row))
    return LocalResidualSummary(
        family=family,
        horizon=horizon,
        minimum=ordered[0],
        median=ordered[len(ordered) // 2],
        maximum=ordered[-1],
    )


def build_composed_pasym_swap_summary(
    *,
    request_hash: str,
    bundle: ComposedArtifactBundle,
    target_checkpoints: tuple[ExactTargetCheckpoint, ...],
    sources: ComposedSamplingSources,
) -> ComposedPAsymSwapSummary:
    """Build all persisted summary facts from authoritative inputs and integer sources."""

    checked_request_hash = _digest(request_hash, name="request_hash")
    checked_bundle = _checked_bundle(bundle)
    checked_targets = _checked_targets(target_checkpoints)
    checked_sources = _checked_sources(sources, bundle=checked_bundle)
    if checked_sources.batch_size != _RELEASE_BATCH_SIZE:
        raise ValueError("summary sources batch_size must equal the release count 32,768")
    target_by_occurrence = {
        item.occurrence_count: tuple(item.occupancy) for item in checked_targets
    }

    identities = tuple(
        ArtifactIdentityResult(
            family=family,
            target_hash=checked_bundle.target_hashes[target_index],
            artifact_hash=checked_bundle.artifact_hashes[family_index][target_index],
            optimizer_result_hash=checked_bundle.optimizer_evidence_hashes[family_index][
                target_index
            ],
            parameter_vector=tuple(
                float(value)
                for value in checked_bundle.parameter_vectors[family_index][target_index]
            ),
        )
        for family_index, family in enumerate(ARTIFACT_FAMILIES)
        for target_index in range(len(checked_bundle.target_hashes))
    )
    residuals = tuple(
        _residual_summary(
            family=family,
            horizon=horizon,
            values=checked_bundle.local_equilibrium_tv_residuals[family_index, horizon_index],
        )
        for family_index, family in enumerate(ARTIFACT_FAMILIES)
        for horizon_index, horizon in enumerate(HORIZON_LABELS)
    )
    raw_cells = tuple(
        (
            cell.family,
            cell.horizon,
            tuple(
                CheckpointResult(
                    source=checkpoint,
                    exact_target_occupancy=target_by_occurrence[checkpoint.occurrence_count],
                    metrics=validate_checkpoint_metrics(
                        derive_checkpoint_metrics(
                            checkpoint, target_by_occurrence[checkpoint.occurrence_count]
                        ),
                        source=checkpoint,
                        target=target_by_occurrence[checkpoint.occurrence_count],
                    ),
                )
                for checkpoint in cell.checkpoints
            ),
        )
        for cell in checked_sources.cells
    )
    cells = tuple(
        ComposedCellResult(
            family=family,
            horizon=horizon,
            checkpoints=checkpoints,
            cell_digest=_cell_digest(family=family, horizon=horizon, checkpoints=checkpoints),
        )
        for family, horizon, checkpoints in raw_cells
    )
    by_family_horizon = {(cell.family, cell.horizon): cell for cell in cells}
    comparisons = tuple(
        comparison
        for horizon in HORIZON_LABELS
        for checkpoint_index, occurrence_count in enumerate(_CHECKPOINT_OCCURRENCES)
        for comparison in (
            derive_paired_checkpoint_comparison(
                first=by_family_horizon["target_context", horizon]
                .checkpoints[checkpoint_index]
                .metrics,
                second=by_family_horizon["independent", horizon]
                .checkpoints[checkpoint_index]
                .metrics,
                label="target_context_minus_independent",
                horizon=horizon,
                occurrence_count=occurrence_count,
            ),
            derive_paired_checkpoint_comparison(
                first=by_family_horizon["model_context", horizon]
                .checkpoints[checkpoint_index]
                .metrics,
                second=by_family_horizon["target_context", horizon]
                .checkpoints[checkpoint_index]
                .metrics,
                label="model_context_minus_target_context",
                horizon=horizon,
                occurrence_count=occurrence_count,
            ),
            derive_paired_checkpoint_comparison(
                first=by_family_horizon["model_context", horizon]
                .checkpoints[checkpoint_index]
                .metrics,
                second=by_family_horizon["independent", horizon]
                .checkpoints[checkpoint_index]
                .metrics,
                label="model_context_minus_independent",
                horizon=horizon,
                occurrence_count=occurrence_count,
            ),
        )
    )
    target_digest = _target_checkpoint_digest(checked_targets)
    digest = _summary_digest(
        request_hash=checked_request_hash,
        bundle_digest=checked_bundle.bundle_digest,
        target_checkpoint_digest=target_digest,
        seed=checked_sources.seed,
        batch_size=checked_sources.batch_size,
        artifact_identities=identities,
        local_residual_summaries=residuals,
        cells=cells,
        comparisons=comparisons,
        integrity_acceptance_passed=True,
    )
    return ComposedPAsymSwapSummary(
        identity_version="composed_pasym_swap_summary.v1",
        request_hash=checked_request_hash,
        bundle_digest=checked_bundle.bundle_digest,
        target_checkpoint_digest=target_digest,
        seed=checked_sources.seed,
        batch_size=checked_sources.batch_size,
        artifact_identities=identities,
        local_residual_summaries=residuals,
        cells=cells,
        comparisons=comparisons,
        integrity_acceptance_passed=True,
        summary_digest=digest,
    )


def _sources_from_persisted_counts(summary: ComposedPAsymSwapSummary) -> ComposedSamplingSources:
    """Recover only bounded integer checkpoint sources from a parsed summary."""

    return ComposedSamplingSources(
        bundle_digest=summary.bundle_digest,
        seed=summary.seed,
        batch_size=summary.batch_size,
        checkpoint_occurrences=_CHECKPOINT_OCCURRENCES,
        cells=tuple(
            CellTrajectorySource(
                family=cell.family,
                horizon=cell.horizon,
                checkpoints=tuple(checkpoint.source for checkpoint in cell.checkpoints),
            )
            for cell in summary.cells
        ),
    )


def _parse_strict_summary(value: object) -> ComposedPAsymSwapSummary:
    if isinstance(value, (str, bytes, bytearray)):
        return ComposedPAsymSwapSummary.model_validate_json(value)
    return ComposedPAsymSwapSummary.model_validate(value)


def validate_composed_pasym_swap_summary(
    value: object,
    *,
    bundle: ComposedArtifactBundle,
    target_checkpoints: tuple[ExactTargetCheckpoint, ...],
    request_hash: str,
) -> ComposedPAsymSwapSummary:
    """Reload summary evidence only by regenerating it from persisted counts."""

    parsed = _parse_strict_summary(value)
    regenerated = build_composed_pasym_swap_summary(
        request_hash=request_hash,
        bundle=bundle,
        target_checkpoints=target_checkpoints,
        sources=_sources_from_persisted_counts(parsed),
    )
    if parsed != regenerated:
        raise ValueError("persisted composed evidence differs from source reconstruction")
    return regenerated
