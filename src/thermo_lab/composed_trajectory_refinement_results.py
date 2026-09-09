"""Strict persisted evidence for one-step composed trajectory refinement."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from thermo_lab.composed_trajectory_refinement import (
    GroupedGradientSource,
    GroupedParameterUpdate,
    PairedObjectiveEvaluation,
    TerminalOccupancySource,
    project_grouped_parameters,
)
from thermo_lab.hashing import canonical_sha256, to_json_value

_RESULT_SCHEMA_VERSION = "1.0.0"
_N_PARAMETERS = 9


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _tuple_json_lists(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuple_json_lists(item) for item in value)
    return value


def _checked_matrix_shape(values: tuple[tuple[float, ...], ...], *, name: str) -> None:
    if not values or any(len(row) != _N_PARAMETERS for row in values):
        raise ValueError(f"{name} must contain non-empty nine-component rows")


def _matrix_close(
    first: tuple[tuple[float, ...], ...],
    second: tuple[tuple[float, ...], ...],
) -> bool:
    return bool(
        np.array_equal(
            np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64),
        )
    )


class StrictEvidenceModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class TerminalOccupancyResult(StrictEvidenceModel):
    sample_count: StrictInt = Field(gt=0)
    occupancy_counts: tuple[StrictInt, ...]
    source_digest: str

    @field_validator("occupancy_counts", mode="before")
    @classmethod
    def freeze_counts(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @field_validator("source_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_sha256_digest(value):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> TerminalOccupancyResult:
        if not self.occupancy_counts:
            raise ValueError("occupancy_counts must not be empty")
        if any(count < 0 or count > self.sample_count for count in self.occupancy_counts):
            raise ValueError("occupancy_counts must lie within the sample count")
        return self

    @property
    def occupancy(self) -> tuple[float, ...]:
        return tuple(count / self.sample_count for count in self.occupancy_counts)


class GroupedGradientResult(StrictEvidenceModel):
    sample_count: StrictInt = Field(gt=0)
    component_sum: tuple[tuple[StrictFloat, ...], ...]
    component_sum_squares: tuple[tuple[StrictFloat, ...], ...]
    source_digest: str

    @field_validator("component_sum", "component_sum_squares", mode="before")
    @classmethod
    def freeze_matrices(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @field_validator("source_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_sha256_digest(value):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_moments(self) -> GroupedGradientResult:
        _checked_matrix_shape(self.component_sum, name="component_sum")
        _checked_matrix_shape(self.component_sum_squares, name="component_sum_squares")
        if len(self.component_sum_squares) != len(self.component_sum):
            raise ValueError("gradient moment matrices must have identical shapes")
        if any(value < 0.0 for row in self.component_sum_squares for value in row):
            raise ValueError("component_sum_squares must be nonnegative")
        sums = np.asarray(self.component_sum, dtype=np.longdouble)
        squares = np.asarray(self.component_sum_squares, dtype=np.longdouble)
        lower = sums * sums / self.sample_count
        if np.any(lower > squares + 1e-12 * np.maximum(lower, squares)):
            raise ValueError("gradient moments imply a negative variance")
        return self

    @property
    def mean(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(value / self.sample_count for value in row) for row in self.component_sum
        )


class GroupedParameterUpdateResult(StrictEvidenceModel):
    raw_parameters: tuple[tuple[StrictFloat, ...], ...]
    updated_parameters: tuple[tuple[StrictFloat, ...], ...]
    cap_active_mask: tuple[tuple[StrictBool, ...], ...]
    cap_active_parameter_count: StrictInt = Field(ge=0)
    bounds_satisfied: StrictBool
    update_digest: str

    @field_validator(
        "raw_parameters",
        "updated_parameters",
        "cap_active_mask",
        mode="before",
    )
    @classmethod
    def freeze_matrices(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @field_validator("update_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_sha256_digest(value):
            raise ValueError("update_digest must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_update_shape(self) -> GroupedParameterUpdateResult:
        _checked_matrix_shape(self.raw_parameters, name="raw_parameters")
        _checked_matrix_shape(self.updated_parameters, name="updated_parameters")
        if len(self.updated_parameters) != len(self.raw_parameters):
            raise ValueError("updated and raw parameter matrices must have identical shapes")
        if len(self.cap_active_mask) != len(self.raw_parameters) or any(
            len(row) != _N_PARAMETERS for row in self.cap_active_mask
        ):
            raise ValueError("cap_active_mask must match the parameter matrix shape")
        if self.cap_active_parameter_count != sum(
            value for row in self.cap_active_mask for value in row
        ):
            raise ValueError("cap_active_parameter_count must match cap_active_mask")
        return self


class PairedObjectiveResult(StrictEvidenceModel):
    before: TerminalOccupancyResult
    after: TerminalOccupancyResult
    target_occupancy: tuple[StrictFloat, ...]
    objective_before: StrictFloat
    objective_after: StrictFloat
    objective_improvement: StrictFloat
    objective_improved: StrictBool
    result_digest: str

    @field_validator("target_occupancy", mode="before")
    @classmethod
    def freeze_target(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @field_validator("result_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_sha256_digest(value):
            raise ValueError("result_digest must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> PairedObjectiveResult:
        if len(self.before.occupancy_counts) != len(self.target_occupancy):
            raise ValueError("before occupancy and target occupancy must have the same length")
        if len(self.after.occupancy_counts) != len(self.target_occupancy):
            raise ValueError("after occupancy and target occupancy must have the same length")
        if any(value < 0.0 or value > 1.0 for value in self.target_occupancy):
            raise ValueError("target_occupancy values must lie in [0, 1]")
        return self


class ComposedTrajectoryRefinementSummary(StrictEvidenceModel):
    result_schema_version: Literal[_RESULT_SCHEMA_VERSION]
    request_hash: str
    seed: StrictInt = Field(ge=0)
    source_bundle_digest: str
    exact_target_reference: str
    beta: StrictFloat = Field(gt=0)
    schedule_digest: str
    initial_parameters: tuple[tuple[StrictFloat, ...], ...]
    initial_parameter_digest: str
    target_occupancy: tuple[StrictFloat, ...]
    occupancy_seed: StrictInt = Field(ge=0)
    gradient_seed: StrictInt = Field(ge=0)
    evaluation_seed: StrictInt = Field(ge=0)
    occupancy_source: TerminalOccupancyResult
    reward_coefficient: tuple[StrictFloat, ...]
    gradient_source: GroupedGradientResult
    gradient_mean: tuple[tuple[StrictFloat, ...], ...]
    learning_rate: StrictFloat
    parameter_cap: StrictFloat
    update: GroupedParameterUpdateResult
    evaluation: PairedObjectiveResult
    integrity_acceptance_passed: StrictBool
    summary_digest: str

    @field_validator(
        "initial_parameters",
        "target_occupancy",
        "reward_coefficient",
        "gradient_mean",
        mode="before",
    )
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @field_validator(
        "request_hash",
        "source_bundle_digest",
        "exact_target_reference",
        "initial_parameter_digest",
        "summary_digest",
        "schedule_digest",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_sha256_digest(value):
            raise ValueError("summary identities must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def validate_summary_shape(self) -> ComposedTrajectoryRefinementSummary:
        _checked_matrix_shape(self.initial_parameters, name="initial_parameters")
        _checked_matrix_shape(self.gradient_mean, name="gradient_mean")
        if len(self.gradient_mean) != len(self.initial_parameters):
            raise ValueError("gradient_mean must match initial_parameters shape")
        if len(self.target_occupancy) != len(self.reward_coefficient):
            raise ValueError("reward_coefficient must match target occupancy length")
        if self.occupancy_source.sample_count <= 0 or self.gradient_source.sample_count <= 0:
            raise ValueError("sampling sources must be non-empty")
        if self.learning_rate <= 0.0 or self.parameter_cap <= 0.0:
            raise ValueError("learning_rate and parameter_cap must be positive")
        return self


def _terminal_result(source: TerminalOccupancySource) -> TerminalOccupancyResult:
    return TerminalOccupancyResult(
        sample_count=source.sample_count,
        occupancy_counts=source.occupancy_counts,
        source_digest=source.source_digest,
    )


def _gradient_result(source: GroupedGradientSource) -> GroupedGradientResult:
    return GroupedGradientResult(
        sample_count=source.sample_count,
        component_sum=source.component_sum,
        component_sum_squares=source.component_sum_squares,
        source_digest=source.source_digest,
    )


def _update_result(update: GroupedParameterUpdate) -> GroupedParameterUpdateResult:
    return GroupedParameterUpdateResult(
        raw_parameters=update.raw_parameters,
        updated_parameters=update.updated_parameters,
        cap_active_mask=update.cap_active_mask,
        cap_active_parameter_count=update.cap_active_parameter_count,
        bounds_satisfied=update.bounds_satisfied,
        update_digest=update.update_digest,
    )


def _evaluation_result(evaluation: PairedObjectiveEvaluation) -> PairedObjectiveResult:
    return PairedObjectiveResult(
        before=_terminal_result(evaluation.before),
        after=_terminal_result(evaluation.after),
        target_occupancy=evaluation.target_occupancy,
        objective_before=evaluation.objective_before,
        objective_after=evaluation.objective_after,
        objective_improvement=evaluation.objective_improvement,
        objective_improved=evaluation.objective_improved,
        result_digest=evaluation.result_digest,
    )


def _payload_dict(payload: object) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        value = payload.model_dump(mode="json")
    elif isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("summary JSON must be valid") from error
    elif isinstance(payload, Mapping):
        value = to_json_value(dict(payload))
    else:
        raise TypeError("summary payload must be a model, mapping, or JSON string")
    if not isinstance(value, dict):
        raise TypeError("summary payload must decode to an object")
    return value


def composed_trajectory_refinement_summary_digest(payload: object) -> str:
    """Digest the complete summary payload except for its self-referential digest."""

    value = _payload_dict(payload)
    value.pop("summary_digest", None)
    return canonical_sha256(
        {
            "identity_version": "composed_trajectory_refinement_summary.v1",
            "payload": value,
        }
    )


def _objective_from_source(
    source: TerminalOccupancyResult,
    target: tuple[float, ...],
) -> float:
    return math.fsum(
        (occupancy - target_value) ** 2
        for occupancy, target_value in zip(source.occupancy, target, strict=True)
    )


def _deeply_validate_summary(summary: ComposedTrajectoryRefinementSummary) -> None:
    expected_initial_digest = canonical_sha256(summary.initial_parameters)
    if summary.initial_parameter_digest != expected_initial_digest:
        raise ValueError("initial parameter digest does not bind initial parameters")

    if len({summary.occupancy_seed, summary.gradient_seed, summary.evaluation_seed}) != 3:
        raise ValueError("occupancy, gradient, and evaluation role seeds must be distinct")
    if summary.evaluation.before.sample_count != summary.evaluation.after.sample_count:
        raise ValueError("paired evaluation sample counts must agree")

    if len(summary.occupancy_source.occupancy_counts) != len(summary.target_occupancy):
        raise ValueError("occupancy source must match target occupancy length")
    expected_reward = tuple(
        2.0 * (occupancy - target)
        for occupancy, target in zip(
            summary.occupancy_source.occupancy,
            summary.target_occupancy,
            strict=True,
        )
    )
    if summary.reward_coefficient != expected_reward:
        raise ValueError(
            "reward coefficient must be reconstructed from occupancy source and target"
        )

    expected_gradient_mean = summary.gradient_source.mean
    if not _matrix_close(summary.gradient_mean, expected_gradient_mean):
        raise ValueError("gradient mean must be reconstructed from gradient source moments")

    expected_update = project_grouped_parameters(
        summary.initial_parameters,
        summary.gradient_mean,
        learning_rate=summary.learning_rate,
        parameter_cap=summary.parameter_cap,
    )
    persisted_update = summary.update
    if (
        not _matrix_close(persisted_update.raw_parameters, expected_update.raw_parameters)
        or not _matrix_close(
            persisted_update.updated_parameters,
            expected_update.updated_parameters,
        )
        or persisted_update.cap_active_mask != expected_update.cap_active_mask
        or persisted_update.cap_active_parameter_count != expected_update.cap_active_parameter_count
        or persisted_update.bounds_satisfied is not expected_update.bounds_satisfied
        or persisted_update.update_digest != expected_update.update_digest
    ):
        raise ValueError("persisted update does not reconstruct from the gradient mean")

    evaluation = summary.evaluation
    if evaluation.target_occupancy != summary.target_occupancy:
        raise ValueError("evaluation target occupancy must match summary target occupancy")
    before_objective = _objective_from_source(evaluation.before, summary.target_occupancy)
    after_objective = _objective_from_source(evaluation.after, summary.target_occupancy)
    improvement = math.fsum((before_objective, -after_objective))
    if evaluation.objective_before != before_objective:
        raise ValueError("objective_before must reconstruct from held-out occupancy counts")
    if evaluation.objective_after != after_objective:
        raise ValueError("objective_after must reconstruct from held-out occupancy counts")
    if evaluation.objective_improvement != improvement:
        raise ValueError("objective_improvement must equal objective_before minus objective_after")
    if evaluation.objective_improved is not (improvement > 0.0):
        raise ValueError("objective_improved must match the held-out objective difference")

    for source, seed, parameters, role in (
        (
            summary.occupancy_source,
            summary.occupancy_seed,
            summary.initial_parameters,
            "standalone_terminal_occupancy",
        ),
        (evaluation.before, summary.evaluation_seed, summary.initial_parameters, "held_out_before"),
        (
            evaluation.after,
            summary.evaluation_seed,
            summary.update.updated_parameters,
            "held_out_after",
        ),
    ):
        expected_digest = canonical_sha256(
            {
                "identity_version": "composed_equilibrium_terminal_occupancy_source.v1",
                "sample_count": source.sample_count,
                "occupancy_counts": source.occupancy_counts,
                "seed": seed,
                "beta": summary.beta,
                "parameter_digest": canonical_sha256(parameters),
                "schedule_digest": summary.schedule_digest,
                "role": role,
            }
        )
        if source.source_digest != expected_digest:
            raise ValueError(f"{role} source digest does not bind its counts and inputs")
    gradient = summary.gradient_source
    if gradient.source_digest != canonical_sha256(
        {
            "identity_version": "composed_equilibrium_grouped_gradient_source.v1",
            "sample_count": gradient.sample_count,
            "component_sum": gradient.component_sum,
            "component_sum_squares": gradient.component_sum_squares,
            "seed": summary.gradient_seed,
            "beta": summary.beta,
            "parameter_digest": summary.initial_parameter_digest,
            "schedule_digest": summary.schedule_digest,
            "reward_coefficient": summary.reward_coefficient,
            "reference_policy": "independent_same_parent_non_propagated",
        }
    ):
        raise ValueError("gradient source digest does not bind its moments and inputs")
    if evaluation.result_digest != canonical_sha256(
        {
            "identity_version": "composed_equilibrium_paired_objective.v1",
            "before_source_digest": evaluation.before.source_digest,
            "after_source_digest": evaluation.after.source_digest,
            "target_occupancy": summary.target_occupancy,
            "objective_before": before_objective,
            "objective_after": after_objective,
            "objective_improvement": improvement,
            "objective_improved": improvement > 0.0,
            "common_random_numbers": True,
        }
    ):
        raise ValueError("evaluation digest does not bind the paired objective sources")

    if not persisted_update.bounds_satisfied:
        raise ValueError("integrity acceptance requires bounded updated parameters")
    if summary.integrity_acceptance_passed is not True:
        raise ValueError("integrity_acceptance_passed must be true for valid evidence")
    expected_summary_digest = composed_trajectory_refinement_summary_digest(summary)
    if summary.summary_digest != expected_summary_digest:
        raise ValueError("summary digest does not bind the complete summary payload")


def build_composed_trajectory_refinement_summary(
    *,
    request_hash: str,
    seed: int,
    source_bundle_digest: str,
    exact_target_reference: str,
    beta: float,
    schedule_digest: str,
    initial_parameters: object,
    target_occupancy: object,
    occupancy_seed: int,
    gradient_seed: int,
    evaluation_seed: int,
    occupancy_source: TerminalOccupancySource,
    reward_coefficient: object,
    gradient_source: GroupedGradientSource,
    learning_rate: float,
    parameter_cap: float,
    update: GroupedParameterUpdate,
    evaluation: PairedObjectiveEvaluation,
) -> ComposedTrajectoryRefinementSummary:
    """Build and deeply validate one auditable composed refinement result."""

    initial = tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(initial_parameters, dtype=np.float64)
    )
    target = tuple(float(value) for value in target_occupancy)
    reward = tuple(float(value) for value in reward_coefficient)
    gradient = _gradient_result(gradient_source)
    payload: dict[str, Any] = {
        "result_schema_version": _RESULT_SCHEMA_VERSION,
        "request_hash": request_hash,
        "seed": seed,
        "source_bundle_digest": source_bundle_digest,
        "exact_target_reference": exact_target_reference,
        "beta": beta,
        "schedule_digest": schedule_digest,
        "initial_parameters": initial,
        "initial_parameter_digest": canonical_sha256(initial),
        "target_occupancy": target,
        "occupancy_seed": occupancy_seed,
        "gradient_seed": gradient_seed,
        "evaluation_seed": evaluation_seed,
        "occupancy_source": _terminal_result(occupancy_source).model_dump(mode="json"),
        "reward_coefficient": reward,
        "gradient_source": gradient.model_dump(mode="json"),
        "gradient_mean": gradient.mean,
        "learning_rate": float(learning_rate),
        "parameter_cap": float(parameter_cap),
        "update": _update_result(update).model_dump(mode="json"),
        "evaluation": _evaluation_result(evaluation).model_dump(mode="json"),
        "integrity_acceptance_passed": True,
    }
    payload["summary_digest"] = composed_trajectory_refinement_summary_digest(payload)
    summary = ComposedTrajectoryRefinementSummary.model_validate(payload)
    _deeply_validate_summary(summary)
    return summary


def validate_composed_trajectory_refinement_summary(
    data: object,
    *,
    expected_bundle_digest: str,
    expected_initial_parameters: object,
    expected_target_occupancy: object,
    expected_target_reference: str,
) -> ComposedTrajectoryRefinementSummary:
    """Reload and reconstruct persisted evidence against trusted lineage inputs."""

    payload = _payload_dict(data)
    summary = ComposedTrajectoryRefinementSummary.model_validate(payload)
    expected_initial = tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(expected_initial_parameters, dtype=np.float64)
    )
    expected_target = tuple(float(value) for value in expected_target_occupancy)
    if summary.source_bundle_digest != expected_bundle_digest:
        raise ValueError("source bundle digest does not match the trusted prepared bundle")
    if summary.exact_target_reference != expected_target_reference:
        raise ValueError("exact target reference does not match the trusted target checkpoint")
    if not _matrix_close(summary.initial_parameters, expected_initial):
        raise ValueError("initial parameters do not match the trusted model-context bundle")
    if summary.target_occupancy != expected_target:
        raise ValueError("target occupancy does not match the trusted final target checkpoint")
    _deeply_validate_summary(summary)
    return summary
