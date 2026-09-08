"""Paired-stream sampling for the checked composed PAsymSwap program.

The executor retains only integer checkpoint aggregates.  It deliberately never
persists particle histories or the common-random-number vectors.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator

from thermo_lab.composed_pasym_swap_artifacts import (
    ARTIFACT_FAMILIES,
    HORIZON_LABELS,
    ArtifactFamily,
    ComposedArtifactBundle,
    HorizonLabel,
)
from thermo_lab.hashing import canonical_sha256

_SITE_COUNT = 25
_OCCURRENCE_COUNT = 500
_TARGET_COUNT = 37
_CHECKPOINT_OCCURRENCES = tuple(range(0, _OCCURRENCE_COUNT + 1, 50))
_MAX_SAMPLE_COUNT = 32_768
_TABLE_MASS_TOLERANCE = 1e-12
_MAX_PCG64_SEED = 2**128 - 1

IntegerCounts25: TypeAlias = tuple[StrictInt, ...]
SectorCounts: TypeAlias = tuple[StrictInt, StrictInt, StrictInt]
ParticleCountHistogram: TypeAlias = tuple[StrictInt, ...]


class _StrictFrozenSourceModel(BaseModel):
    """Strict, immutable persisted source payload base."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)


def _strict_integer(
    value: object, *, name: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a strict integer")
    if value < minimum or (maximum is not None and value > maximum):
        range_text = (
            f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        )
        raise ValueError(f"{name} must be {range_text}")
    return value


def _strict_int_tuple(
    value: object, *, name: str, length: int, maximum: int | None = None
) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise ValueError(f"{name} must be a tuple of exactly {length} strict integers")
    return tuple(
        _strict_integer(item, name=f"{name}[{index}]", maximum=maximum)
        for index, item in enumerate(value)
    )


def _json_array_to_tuple(value: object, info: object) -> object:
    """Restore strict immutable tuples when a persisted JSON payload is reloaded."""

    if getattr(info, "mode", None) == "json" and isinstance(value, list):
        return tuple(value)
    return value


def _sha256_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _checkpoint_source_digest(
    *,
    occurrence_count: int,
    sample_count: int,
    occupancy_counts: tuple[int, ...],
    sector_counts: tuple[int, int, int],
    particle_count_histogram: tuple[int, ...],
    particle_count_sum: int,
    particle_count_sum_squares: int,
    ever_left_count: int,
    one_particle_location_counts: tuple[int, ...],
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_sampling_checkpoint_source.v2",
            "occurrence_count": occurrence_count,
            "sample_count": sample_count,
            "occupancy_counts": occupancy_counts,
            "sector_counts": sector_counts,
            "particle_count_histogram": particle_count_histogram,
            "particle_count_sum": particle_count_sum,
            "particle_count_sum_squares": particle_count_sum_squares,
            "ever_left_count": ever_left_count,
            "one_particle_location_counts": one_particle_location_counts,
        }
    )


class CheckpointSource(_StrictFrozenSourceModel):
    """Integer-only sampled state aggregates at one schedule occurrence count."""

    occurrence_count: StrictInt
    sample_count: StrictInt
    occupancy_counts: IntegerCounts25
    sector_counts: SectorCounts
    particle_count_histogram: ParticleCountHistogram
    particle_count_sum: StrictInt
    particle_count_sum_squares: StrictInt
    ever_left_count: StrictInt
    one_particle_location_counts: IntegerCounts25
    source_digest: str | None = None

    @field_validator("occurrence_count", mode="before")
    @classmethod
    def validate_occurrence_count(cls, value: object) -> int:
        return _strict_integer(value, name="occurrence_count", maximum=_OCCURRENCE_COUNT)

    @field_validator("sample_count", mode="before")
    @classmethod
    def validate_sample_count(cls, value: object) -> int:
        return _strict_integer(value, name="sample_count", minimum=1, maximum=_MAX_SAMPLE_COUNT)

    @field_validator("occupancy_counts", "one_particle_location_counts", mode="before")
    @classmethod
    def validate_site_counts(cls, value: object, info: object) -> tuple[int, ...]:
        return _strict_int_tuple(
            _json_array_to_tuple(value, info),
            name=getattr(info, "field_name", "site counts"),
            length=_SITE_COUNT,
        )

    @field_validator("sector_counts", mode="before")
    @classmethod
    def validate_sector_counts(cls, value: object, info: object) -> tuple[int, int, int]:
        checked = _strict_int_tuple(
            _json_array_to_tuple(value, info), name="sector_counts", length=3
        )
        return checked[0], checked[1], checked[2]

    @field_validator("particle_count_histogram", mode="before")
    @classmethod
    def validate_particle_count_histogram(cls, value: object, info: object) -> tuple[int, ...]:
        return _strict_int_tuple(
            _json_array_to_tuple(value, info),
            name="particle_count_histogram",
            length=_SITE_COUNT + 1,
        )

    @field_validator(
        "particle_count_sum", "particle_count_sum_squares", "ever_left_count", mode="before"
    )
    @classmethod
    def validate_nonnegative_count(cls, value: object, info: object) -> int:
        return _strict_integer(value, name=getattr(info, "field_name", "count"))

    @field_validator("source_digest")
    @classmethod
    def validate_digest_shape(cls, value: str | None) -> str | None:
        return None if value is None else _sha256_digest(value, name="source_digest")

    @model_validator(mode="after")
    def validate_source(self) -> CheckpointSource:
        if any(count > self.sample_count for count in self.occupancy_counts):
            raise ValueError("occupancy_counts cannot exceed sample_count")
        if any(count > self.sample_count for count in self.one_particle_location_counts):
            raise ValueError("one_particle_location_counts cannot exceed sample_count")
        if sum(self.sector_counts) != self.sample_count:
            raise ValueError("sector_counts must sum to sample_count")
        if any(count > self.sample_count for count in self.particle_count_histogram):
            raise ValueError("particle_count_histogram cannot exceed sample_count")
        if sum(self.particle_count_histogram) != self.sample_count:
            raise ValueError("particle_count_histogram must sum to sample_count")
        histogram_sector_counts = (
            self.particle_count_histogram[0],
            self.particle_count_histogram[1],
            sum(self.particle_count_histogram[2:]),
        )
        if tuple(self.sector_counts) != histogram_sector_counts:
            raise ValueError("sector_counts must match particle_count_histogram")
        histogram_particle_count_sum = sum(
            mass * count for mass, count in enumerate(self.particle_count_histogram)
        )
        if self.particle_count_sum != histogram_particle_count_sum:
            raise ValueError("particle_count_sum must match particle_count_histogram")
        histogram_particle_count_sum_squares = sum(
            mass**2 * count for mass, count in enumerate(self.particle_count_histogram)
        )
        if self.particle_count_sum_squares != histogram_particle_count_sum_squares:
            raise ValueError("particle_count_sum_squares must match particle_count_histogram")
        if sum(self.occupancy_counts) != self.particle_count_sum:
            raise ValueError("occupancy_counts must sum to particle_count_sum")
        if sum(self.one_particle_location_counts) != self.sector_counts[1]:
            raise ValueError("one_particle_location_counts must sum to the one-particle sector")
        if any(
            one_count > occupancy_count
            for one_count, occupancy_count in zip(
                self.one_particle_location_counts, self.occupancy_counts, strict=True
            )
        ):
            raise ValueError("one_particle_location_counts cannot exceed occupancy_counts")
        if self.ever_left_count > self.sample_count:
            raise ValueError("ever_left_count cannot exceed sample_count")
        if self.ever_left_count < self.sector_counts[0] + self.sector_counts[2]:
            raise ValueError("ever_left_count must cover the current out-of-sector count")
        if self.occurrence_count == 0 and self.ever_left_count != 0:
            raise ValueError("ever_left_count must be zero at occurrence zero")
        multiple_count = self.sector_counts[2]
        minimum_mass = self.sector_counts[1] + 2 * multiple_count
        maximum_mass = self.sector_counts[1] + _SITE_COUNT * multiple_count
        if not minimum_mass <= self.particle_count_sum <= maximum_mass:
            raise ValueError("particle_count_sum is inconsistent with sector_counts")
        minimum_square_sum = self.sector_counts[1] + 4 * multiple_count
        maximum_square_sum = self.sector_counts[1] + _SITE_COUNT**2 * multiple_count
        if not minimum_square_sum <= self.particle_count_sum_squares <= maximum_square_sum:
            raise ValueError("particle_count_sum_squares is inconsistent with sector_counts")
        if self.particle_count_sum_squares < self.particle_count_sum:
            raise ValueError("particle_count_sum_squares cannot be below particle_count_sum")
        if self.sample_count * self.particle_count_sum_squares < self.particle_count_sum**2:
            raise ValueError("particle-count moments violate the nonnegative variance bound")
        if not _is_graphical_histogram(self.particle_count_histogram, self.occupancy_counts):
            raise ValueError("occupancy_counts and particle_count_histogram violate Gale-Ryser")
        residual_occupancy_counts = tuple(
            occupancy_count - one_count
            for occupancy_count, one_count in zip(
                self.occupancy_counts, self.one_particle_location_counts, strict=True
            )
        )
        if any(count < 0 for count in residual_occupancy_counts):
            raise ValueError("one_particle_location_counts cannot exceed occupancy_counts")
        if not _is_graphical_histogram(
            self.particle_count_histogram,
            residual_occupancy_counts,
            minimum_mass=2,
        ):
            raise ValueError(
                "residual multi-particle occupancy and particle_count_histogram violate Gale-Ryser"
            )
        expected = _checkpoint_source_digest(
            occurrence_count=self.occurrence_count,
            sample_count=self.sample_count,
            occupancy_counts=tuple(self.occupancy_counts),
            sector_counts=tuple(self.sector_counts),
            particle_count_histogram=tuple(self.particle_count_histogram),
            particle_count_sum=self.particle_count_sum,
            particle_count_sum_squares=self.particle_count_sum_squares,
            ever_left_count=self.ever_left_count,
            one_particle_location_counts=tuple(self.one_particle_location_counts),
        )
        if self.source_digest is None:
            object.__setattr__(self, "source_digest", expected)
        elif self.source_digest != expected:
            raise ValueError("source_digest does not bind the checkpoint source payload")
        return self


def _largest_histogram_degree_sum(
    histogram: tuple[int, ...], count: int, *, minimum_mass: int
) -> int:
    """Sum the largest ``count`` row degrees without materializing them."""

    remaining = count
    total = 0
    for mass in range(_SITE_COUNT, minimum_mass - 1, -1):
        taken = min(remaining, histogram[mass])
        total += mass * taken
        remaining -= taken
        if remaining == 0:
            break
    return total


def _is_graphical_histogram(
    histogram: tuple[int, ...],
    column_degrees: tuple[int, ...],
    *,
    minimum_mass: int = 1,
) -> bool:
    """Check Gale--Ryser directly from bounded row-degree frequencies."""

    if len(histogram) != _SITE_COUNT + 1 or any(count < 0 for count in histogram):
        return False
    row_count = sum(histogram[minimum_mass:])
    if any(degree < 0 or degree > row_count for degree in column_degrees):
        return False
    row_degree_sum = sum(mass * histogram[mass] for mass in range(minimum_mass, _SITE_COUNT + 1))
    if row_degree_sum != sum(column_degrees):
        return False
    if row_count == 0:
        return not any(column_degrees)
    boundaries = {row_count}
    cumulative_count = 0
    for mass in range(_SITE_COUNT, minimum_mass - 1, -1):
        cumulative_count += histogram[mass]
        if 0 < cumulative_count < row_count:
            boundaries.add(cumulative_count)
    boundaries.update(degree for degree in column_degrees if 0 < degree < row_count)
    return all(
        _largest_histogram_degree_sum(histogram, count, minimum_mass=minimum_mass)
        <= sum(min(count, degree) for degree in column_degrees)
        for count in boundaries
    )


def _cell_source_digest(
    *, family: ArtifactFamily, horizon: HorizonLabel, checkpoints: tuple[CheckpointSource, ...]
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_sampling_cell_source.v1",
            "family": family,
            "horizon": horizon,
            "checkpoints": checkpoints,
        }
    )


class CellTrajectorySource(_StrictFrozenSourceModel):
    """Digest-bound integer checkpoint sources for one family/horizon cell."""

    family: Literal["independent", "target_context", "model_context"]
    horizon: Literal["equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"]
    checkpoints: tuple[CheckpointSource, ...]
    source_digest: str | None = None

    @field_validator("checkpoints", mode="before")
    @classmethod
    def validate_checkpoint_tuple_shape(cls, value: object, info: object) -> object:
        value = _json_array_to_tuple(value, info)
        if not isinstance(value, tuple) or not value:
            raise ValueError("checkpoints must be a nonempty tuple")
        return value

    @field_validator("source_digest")
    @classmethod
    def validate_digest_shape(cls, value: str | None) -> str | None:
        return None if value is None else _sha256_digest(value, name="source_digest")

    @model_validator(mode="after")
    def validate_cell(self) -> CellTrajectorySource:
        occurrences = tuple(checkpoint.occurrence_count for checkpoint in self.checkpoints)
        if occurrences != tuple(sorted(set(occurrences))):
            raise ValueError("checkpoints must have strictly increasing occurrence counts")
        sample_counts = {checkpoint.sample_count for checkpoint in self.checkpoints}
        if len(sample_counts) != 1:
            raise ValueError("checkpoints must have a consistent sample_count")
        ever_left_counts = tuple(checkpoint.ever_left_count for checkpoint in self.checkpoints)
        if ever_left_counts != tuple(sorted(ever_left_counts)):
            raise ValueError("ever_left_count must not decrease across checkpoints")
        expected = _cell_source_digest(
            family=self.family, horizon=self.horizon, checkpoints=self.checkpoints
        )
        if self.source_digest is None:
            object.__setattr__(self, "source_digest", expected)
        elif self.source_digest != expected:
            raise ValueError("source_digest does not bind the cell source payload")
        return self


def _sampling_sources_digest(
    *,
    bundle_digest: str,
    seed: int,
    batch_size: int,
    checkpoint_occurrences: tuple[int, ...],
    cells: tuple[CellTrajectorySource, ...],
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_sampling_sources.v1",
            "bundle_digest": bundle_digest,
            "seed": seed,
            "batch_size": batch_size,
            "checkpoint_occurrences": checkpoint_occurrences,
            "cells": cells,
        }
    )


class ComposedSamplingSources(_StrictFrozenSourceModel):
    """Canonical 21-cell paired-stream source summary for one seeded run."""

    bundle_digest: str
    seed: StrictInt
    batch_size: StrictInt
    checkpoint_occurrences: tuple[StrictInt, ...]
    cells: tuple[CellTrajectorySource, ...]
    source_digest: str | None = None

    @field_validator("bundle_digest")
    @classmethod
    def validate_bundle_digest(cls, value: str) -> str:
        return _sha256_digest(value, name="bundle_digest")

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> int:
        return _strict_integer(value, name="seed", maximum=_MAX_PCG64_SEED)

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, value: object) -> int:
        return _strict_integer(value, name="batch_size", minimum=1)

    @field_validator("checkpoint_occurrences", mode="before")
    @classmethod
    def validate_checkpoint_occurrences(cls, value: object, info: object) -> tuple[int, ...]:
        return _checked_checkpoint_occurrences(_json_array_to_tuple(value, info))

    @field_validator("cells", mode="before")
    @classmethod
    def validate_cells_tuple_shape(cls, value: object, info: object) -> object:
        value = _json_array_to_tuple(value, info)
        if not isinstance(value, tuple):
            raise TypeError("cells must be a tuple")
        return value

    @field_validator("source_digest")
    @classmethod
    def validate_digest_shape(cls, value: str | None) -> str | None:
        return None if value is None else _sha256_digest(value, name="source_digest")

    @model_validator(mode="after")
    def validate_sources(self) -> ComposedSamplingSources:
        expected_cells = tuple(
            (family, horizon) for family in ARTIFACT_FAMILIES for horizon in HORIZON_LABELS
        )
        observed_cells = tuple((cell.family, cell.horizon) for cell in self.cells)
        if observed_cells != expected_cells:
            raise ValueError("cells must use canonical family and horizon order")
        for cell in self.cells:
            if tuple(checkpoint.occurrence_count for checkpoint in cell.checkpoints) != (
                self.checkpoint_occurrences
            ):
                raise ValueError("cell checkpoints must match checkpoint_occurrences")
            if any(checkpoint.sample_count != self.batch_size for checkpoint in cell.checkpoints):
                raise ValueError("cell checkpoint sample_count must equal batch_size")
            initial = cell.checkpoints[0]
            expected_locations = (self.batch_size,) + (0,) * (_SITE_COUNT - 1)
            if (
                initial.occupancy_counts != expected_locations
                or initial.sector_counts != (0, self.batch_size, 0)
                or initial.particle_count_histogram != (0, self.batch_size) + (0,) * 24
                or initial.particle_count_sum != self.batch_size
                or initial.particle_count_sum_squares != self.batch_size
                or initial.ever_left_count != 0
                or initial.one_particle_location_counts != expected_locations
            ):
                raise ValueError("every cell must have the canonical occurrence-zero source")
        expected = _sampling_sources_digest(
            bundle_digest=self.bundle_digest,
            seed=self.seed,
            batch_size=self.batch_size,
            checkpoint_occurrences=self.checkpoint_occurrences,
            cells=self.cells,
        )
        if self.source_digest is None:
            object.__setattr__(self, "source_digest", expected)
        elif self.source_digest != expected:
            raise ValueError("source_digest does not bind the sampling source payload")
        return self


def _checked_uniforms(uniforms: object) -> NDArray[np.float64]:
    if not isinstance(uniforms, np.ndarray) or uniforms.dtype != np.dtype(np.float64):
        raise TypeError("uniforms must be a float64 NumPy vector")
    if uniforms.ndim != 1 or not np.all(np.isfinite(uniforms)):
        raise ValueError("uniforms must be a finite one-dimensional vector")
    if np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
        raise ValueError("uniforms must lie in the half-open interval [0, 1)")
    return uniforms


def _checked_rows(rows: object, sample_count: int) -> NDArray[np.float64]:
    if not isinstance(rows, np.ndarray) or rows.dtype != np.dtype(np.float64):
        raise TypeError("rows must be a float64 NumPy array")
    if rows.shape != (sample_count, 4):
        raise ValueError("rows must have shape (sample_count, 4)")
    if not np.all(np.isfinite(rows)):
        raise ValueError("rows must be finite")
    if np.any(rows < 0.0):
        raise ValueError("rows must be nonnegative")
    masses = rows.sum(axis=1, dtype=np.float64)
    if not np.all(np.isfinite(masses)) or not np.allclose(
        masses, 1.0, rtol=0.0, atol=_TABLE_MASS_TOLERANCE
    ):
        raise ValueError("rows must be row-stochastic")
    return rows


def inverse_cdf_words(
    rows: NDArray[np.float64], uniforms: NDArray[np.float64]
) -> NDArray[np.uint8]:
    """Sample canonical two-bit words from checked float64 conditional rows."""

    checked_uniforms = _checked_uniforms(uniforms)
    checked_rows = _checked_rows(rows, len(checked_uniforms))
    cumulative = np.cumsum(checked_rows, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(cumulative)):
        raise ValueError("inverse-CDF cumulative sums must be finite")
    cumulative[:, -1] = 1.0
    return np.sum(checked_uniforms[:, None] >= cumulative[:, :3], axis=1).astype(np.uint8)


def _checked_tables(tables: object) -> NDArray[np.float64]:
    if not isinstance(tables, np.ndarray) or tables.dtype != np.dtype(np.float64):
        raise TypeError("tables must be a float64 NumPy array")
    if tables.ndim != 3 or tables.shape[1:] != (4, 4) or tables.shape[0] == 0:
        raise ValueError("tables must have shape (target_count, 4, 4)")
    if not np.all(np.isfinite(tables)) or np.any(tables < 0.0):
        raise ValueError("tables must contain finite nonnegative probabilities")
    if not np.allclose(
        tables.sum(axis=-1, dtype=np.float64), 1.0, rtol=0.0, atol=_TABLE_MASS_TOLERANCE
    ):
        raise ValueError("tables must be row-stochastic")
    return tables


def _checked_states(states: object) -> NDArray[np.uint8]:
    if not isinstance(states, np.ndarray) or states.dtype != np.dtype(np.uint8):
        raise TypeError("states must be a uint8 NumPy array")
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] == 0:
        raise ValueError("states must have shape (batch_size, site_count)")
    if not np.all((states == 0) | (states == 1)):
        raise ValueError("states must be binary")
    return states


def _checked_target_index(target_index: object, table_count: int) -> int:
    return _strict_integer(target_index, name="target_index", maximum=table_count - 1)


def _checked_endpoints(endpoints: object, site_count: int) -> tuple[int, int]:
    if not isinstance(endpoints, tuple) or len(endpoints) != 2:
        raise TypeError("endpoints must be a two-site tuple")
    left = _strict_integer(endpoints[0], name="left endpoint", maximum=site_count - 1)
    right = _strict_integer(endpoints[1], name="right endpoint", maximum=site_count - 1)
    if left == right:
        raise ValueError("endpoints must be distinct")
    return left, right


def apply_occurrence(
    states: NDArray[np.uint8],
    tables: NDArray[np.float64],
    *,
    target_index: int,
    endpoints: tuple[int, int],
    uniforms: NDArray[np.float64],
) -> None:
    """Apply one checked two-site conditional to an in-place binary state batch."""

    checked_states = _checked_states(states)
    checked_tables = _checked_tables(tables)
    checked_target = _checked_target_index(target_index, checked_tables.shape[0])
    left, right = _checked_endpoints(endpoints, checked_states.shape[1])
    checked_uniforms = _checked_uniforms(uniforms)
    if len(checked_uniforms) != checked_states.shape[0]:
        raise ValueError("uniforms length must equal state batch size")
    input_indices = 2 * checked_states[:, left] + checked_states[:, right]
    rows = checked_tables[checked_target, input_indices]
    output_indices = inverse_cdf_words(rows, checked_uniforms)
    checked_states[:, left] = output_indices // 2
    checked_states[:, right] = output_indices % 2


def _apply_occurrence_to_cells(
    states: NDArray[np.uint8],
    ever_left: NDArray[np.bool_],
    conditionals: NDArray[np.float64],
    *,
    target_index: int,
    endpoints: tuple[int, int],
    uniforms: NDArray[np.float64],
    cell_indices: tuple[tuple[int, int], ...],
) -> None:
    """Apply one pre-drawn uniform vector to cells in any declared order."""

    for family_index, horizon_index in cell_indices:
        cell_states = states[family_index, horizon_index]
        apply_occurrence(
            cell_states,
            conditionals[family_index, horizon_index],
            target_index=target_index,
            endpoints=endpoints,
            uniforms=uniforms,
        )
        masses = cell_states.sum(axis=1, dtype=np.uint8)
        ever_left[family_index, horizon_index] |= masses != 1


def _checked_checkpoint_occurrences(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError("checkpoint_occurrences must be a nonempty tuple")
    checked = tuple(
        _strict_integer(item, name=f"checkpoint_occurrences[{index}]", maximum=_OCCURRENCE_COUNT)
        for index, item in enumerate(value)
    )
    if checked != _CHECKPOINT_OCCURRENCES:
        raise ValueError("checkpoint_occurrences must equal canonical boundaries 0, 50, ..., 500")
    return checked


def _checked_release_bundle(bundle: object) -> ComposedArtifactBundle:
    if not isinstance(bundle, ComposedArtifactBundle):
        raise TypeError("bundle must be a ComposedArtifactBundle")
    checked = ComposedArtifactBundle.model_validate(bundle.model_dump())
    if checked.families != ARTIFACT_FAMILIES or checked.horizons != HORIZON_LABELS:
        raise ValueError("bundle must use canonical family and horizon order")
    if (
        checked.conditionals.shape != (3, 7, _TARGET_COUNT, 4, 4)
        or checked.conditionals.dtype != np.float64
    ):
        raise ValueError("bundle must contain the checked 3 by 7 by 37 float64 conditional tables")
    if (
        checked.occurrence_target_indices.shape != (_OCCURRENCE_COUNT,)
        or checked.occurrence_target_indices.dtype != np.int16
    ):
        raise ValueError("bundle must contain 500 int16 occurrence target indices")
    if (
        checked.occurrence_site_indices.shape != (_OCCURRENCE_COUNT, 2)
        or checked.occurrence_site_indices.dtype != np.int8
    ):
        raise ValueError("bundle must contain 500 int8 occurrence site pairs")
    return checked


def reduce_checkpoint_source(
    states: NDArray[np.uint8], ever_left: NDArray[np.bool_], occurrence_count: int
) -> CheckpointSource:
    """Reduce one state batch to persistence-safe integer sources only."""

    checked_states = _checked_states(states)
    if checked_states.shape[1] != _SITE_COUNT:
        raise ValueError("checkpoint state batches must use exactly 25 sites")
    if not isinstance(ever_left, np.ndarray) or ever_left.dtype != np.dtype(np.bool_):
        raise TypeError("ever_left must be a bool NumPy vector")
    if ever_left.shape != (checked_states.shape[0],):
        raise ValueError("ever_left must have one value per state")
    checked_occurrence = _strict_integer(
        occurrence_count, name="occurrence_count", maximum=_OCCURRENCE_COUNT
    )
    masses = checked_states.sum(axis=1, dtype=np.int64)
    one_mask = masses == 1
    return CheckpointSource(
        occurrence_count=checked_occurrence,
        sample_count=checked_states.shape[0],
        occupancy_counts=tuple(int(value) for value in checked_states.sum(axis=0, dtype=np.int64)),
        sector_counts=(
            int(np.count_nonzero(masses == 0)),
            int(np.count_nonzero(one_mask)),
            int(np.count_nonzero(masses > 1)),
        ),
        particle_count_histogram=tuple(
            int(np.count_nonzero(masses == mass)) for mass in range(_SITE_COUNT + 1)
        ),
        particle_count_sum=int(masses.sum(dtype=np.int64)),
        particle_count_sum_squares=int(np.square(masses, dtype=np.int64).sum(dtype=np.int64)),
        ever_left_count=int(np.count_nonzero(ever_left)),
        one_particle_location_counts=tuple(
            int(value) for value in checked_states[one_mask].sum(axis=0, dtype=np.int64)
        ),
    )


def _reduce_all_cells(
    states: NDArray[np.uint8], ever_left: NDArray[np.bool_], occurrence_count: int
) -> tuple[tuple[CheckpointSource, ...], ...]:
    return tuple(
        tuple(
            reduce_checkpoint_source(
                states[family_index, horizon_index],
                ever_left[family_index, horizon_index],
                occurrence_count,
            )
            for horizon_index in range(len(HORIZON_LABELS))
        )
        for family_index in range(len(ARTIFACT_FAMILIES))
    )


def _canonical_sources(
    bundle: ComposedArtifactBundle,
    *,
    seed: int,
    batch_size: int,
    checkpoints: tuple[tuple[tuple[CheckpointSource, ...], ...], ...],
) -> ComposedSamplingSources:
    cells = tuple(
        CellTrajectorySource(
            family=family,
            horizon=horizon,
            checkpoints=tuple(
                checkpoint[family_index][horizon_index] for checkpoint in checkpoints
            ),
        )
        for family_index, family in enumerate(bundle.families)
        for horizon_index, horizon in enumerate(bundle.horizons)
    )
    return ComposedSamplingSources(
        bundle_digest=bundle.bundle_digest,
        seed=seed,
        batch_size=batch_size,
        checkpoint_occurrences=tuple(
            checkpoint[0][0].occurrence_count for checkpoint in checkpoints
        ),
        cells=cells,
    )


def sample_composed_program(
    bundle: ComposedArtifactBundle,
    *,
    batch_size: int,
    seed: int,
    checkpoint_occurrences: tuple[int, ...],
) -> ComposedSamplingSources:
    """Sample all 21 artifact cells with one PCG64 uniform vector per occurrence."""

    checked_bundle = _checked_release_bundle(bundle)
    checked_batch_size = _strict_integer(
        batch_size, name="batch_size", minimum=1, maximum=_MAX_SAMPLE_COUNT
    )
    checked_seed = _strict_integer(seed, name="seed", maximum=_MAX_PCG64_SEED)
    checked_checkpoints = _checked_checkpoint_occurrences(checkpoint_occurrences)
    states = np.zeros((3, 7, checked_batch_size, _SITE_COUNT), dtype=np.uint8)
    states[:, :, :, 0] = 1
    ever_left = np.zeros((3, 7, checked_batch_size), dtype=np.bool_)
    rng = np.random.Generator(np.random.PCG64(checked_seed))
    reduced: list[tuple[tuple[CheckpointSource, ...], ...]] = [
        _reduce_all_cells(states, ever_left, 0)
    ]
    requested = set(checked_checkpoints)
    cell_indices = tuple(
        (family_index, horizon_index)
        for family_index in range(len(ARTIFACT_FAMILIES))
        for horizon_index in range(len(HORIZON_LABELS))
    )
    for occurrence_index in range(_OCCURRENCE_COUNT):
        uniforms = rng.random(checked_batch_size, dtype=np.float64)
        _apply_occurrence_to_cells(
            states,
            ever_left,
            checked_bundle.conditionals,
            target_index=int(checked_bundle.occurrence_target_indices[occurrence_index]),
            endpoints=tuple(
                int(value) for value in checked_bundle.occurrence_site_indices[occurrence_index]
            ),
            uniforms=uniforms,
            cell_indices=cell_indices,
        )
        occurrence_count = occurrence_index + 1
        if occurrence_count in requested:
            reduced.append(_reduce_all_cells(states, ever_left, occurrence_count))
    return _canonical_sources(
        checked_bundle,
        seed=checked_seed,
        batch_size=checked_batch_size,
        checkpoints=tuple(reduced),
    )
