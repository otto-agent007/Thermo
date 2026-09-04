"""Strict bounded persistence models for paired target-context PAsymSwap runs."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, Self

import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from thermo_lab.config import (
    experiment_config_path,
    independent_pasym_swap_non_seed_config_hash,
    load_experiment_config,
    target_context_pasym_swap_non_seed_config_hash,
)
from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.independent_compiler import loss_and_gradient, project_gradient
from thermo_lab.pasym_swap import PAPER_SOURCE, WORD_ORDER, build_paper_fixture
from thermo_lab.pasym_swap_context import (
    OCCUPANCY_ORDER,
    derive_target_context_trace,
    pool_target_context_profiles,
)
from thermo_lab.pasym_swap_results import (
    ConditionalTable,
    KernelOptimizationAttemptResult,
    KernelOptimizationResult,
    SummaryStatistics,
    summarize_values,
)
from thermo_lab.records import RUN_TIMING_SOURCE, FrozenModel, MetricObservation
from thermo_lab.schemas import (
    PARAMETER_ORDER,
    IndependentCompilerRunConfig,
    PAsymSwapModelConfig,
    TargetContextCompilerRunConfig,
    validate_target_context_pasym_swap_request,
)
from thermo_lab.target_context_compiler import TARGET_CONTEXT_START_ROLES, TargetContextStartRole
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    equilibrium_conditional,
    finite_horizon_conditional,
)

_HORIZONS = (1, 2, 4, 8, 16, 30)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_COLORS = ("H1", "H2", "H3", "V1", "V2", "V3")
_N_PARAMETERS = len(PARAMETER_ORDER)
_FIXED_STARTS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05),
    (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05),
)
_WORDS = ((0, 0), (0, 1), (1, 0), (1, 1))
_ALL_CONTEXT_ROW_COUNT = 37 * 2 * 4
_POSITIVE_SUPPORT_ROW_COUNT = 37 * 4 + 37 * 3
_REQUIRED_METRICS = frozenset(
    {
        "target_context_pasym_swap",
        "baseline_occurrence_weighted_equilibrium_kl",
        "target_context_occurrence_weighted_equilibrium_kl",
        "occurrence_weighted_equilibrium_kl_improvement",
        "baseline_occurrence_weighted_equilibrium_tv",
        "target_context_occurrence_weighted_equilibrium_tv",
        "maximum_paired_k30_equilibrium_residual",
        "maximum_empirical_k30_residual",
        "acceptance_passed",
        "baseline_optimizer_seconds",
        "target_context_optimizer_seconds",
    }
)
_SUMMARY_METHOD = "bounded target-context PAsymSwap summary"
_EXACT_METHOD = "recomputed from exact frozen-model conditionals"
_SAMPLE_METHOD = "independently seeded 4096-chain THRML cross-check"
_ACCEPTANCE_METHOD = "all target-context acceptance gates recomputed"
_BASELINE_OPTIMIZER_METHOD = "wall-clock SciPy optimization across 37 paired uniform baselines"
_TARGET_OPTIMIZER_METHOD = "wall-clock SciPy optimization across 37 target-context profiles"

ParameterVector = tuple[StrictFloat, ...]
ProbabilityVector = tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]
SupportMask = tuple[StrictBool, StrictBool, StrictBool, StrictBool]
CountRow = tuple[StrictInt, StrictInt, StrictInt, StrictInt]
CountTable = tuple[CountRow, CountRow, CountRow, CountRow]


class _StrictFrozenResultModel(FrozenModel):
    """No coercion, unknown payload, or mutable nested persistence state."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)


def _freeze_mapping(values: Mapping[int, Any]) -> Mapping[int, Any]:
    return MappingProxyType(dict(values))


def _json_tuple(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_tuple(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_json_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_json_tuple(item) for item in value)
    return value


def _legacy_json_value(value: object) -> object:
    """Restore legacy strict enum inputs while keeping its persisted tuples frozen."""

    if isinstance(value, Mapping):
        return {
            key: EvidenceClass(item)
            if key == "evidence_class" and isinstance(item, str)
            else _legacy_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_legacy_json_value(item) for item in value)
    return value


def _sha(value: str, *, name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) is not float:
        raise ValueError(f"{name} must be a finite floating-point value")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _vector(value: object, *, name: str, nonnegative: bool = False) -> ProbabilityVector:
    values = _json_tuple(value)
    if not isinstance(values, tuple) or len(values) != 4:
        raise ValueError(f"{name} must contain exactly four values")
    return tuple(
        _finite_float(item, name=f"{name}[{index}]", nonnegative=nonnegative)
        for index, item in enumerate(values)
    )  # type: ignore[return-value]


def _context_weights(value: object, *, name: str) -> ProbabilityVector:
    weights = _vector(value, name=name, nonnegative=True)
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to one within 1e-12")
    return weights


def _support_mask(value: object, *, name: str) -> SupportMask:
    values = _json_tuple(value)
    if (
        not isinstance(values, tuple)
        or len(values) != 4
        or any(type(item) is not bool for item in values)
    ):
        raise ValueError(f"{name} must contain exactly four booleans")
    return values  # type: ignore[return-value]


def _conditional_table(value: object, *, name: str) -> ConditionalTable:
    values = _json_tuple(value)
    if not isinstance(values, tuple) or len(values) != 4:
        raise ValueError(f"{name} must contain exactly four input rows")
    rows: list[ProbabilityVector] = []
    for index, row in enumerate(values):
        checked = _vector(row, name=f"{name}[{index}]", nonnegative=True)
        if not math.isclose(math.fsum(checked), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name}[{index}] must sum to one within 1e-12")
        rows.append(checked)
    return tuple(rows)  # type: ignore[return-value]


def _conditional_row(value: object, *, name: str) -> ProbabilityVector:
    row = _vector(value, name=name, nonnegative=True)
    if not math.isclose(math.fsum(row), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to one within 1e-12")
    return row


def _parse_horizons(value: object, *, name: str) -> dict[int, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    parsed: dict[int, object] = {}
    for key, item in value.items():
        parsed_key = int(key) if isinstance(key, str) and key.isdecimal() else key
        if type(parsed_key) is not int:
            raise ValueError(f"{name} keys must be integer horizons")
        parsed[parsed_key] = item
    if tuple(sorted(parsed)) != _HORIZONS:
        raise ValueError(f"{name} keys must be exactly {_HORIZONS}")
    return parsed


def _parameter_vector(value: object, *, name: str) -> ParameterVector:
    values = _json_tuple(value)
    if not isinstance(values, tuple) or len(values) != _N_PARAMETERS:
        raise ValueError(f"{name} must contain exactly {_N_PARAMETERS} parameters")
    return tuple(_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(values))


def _messages(value: object, *, name: str) -> tuple[str, ...]:
    values = _json_tuple(value)
    if not isinstance(values, tuple) or len(values) > 32:
        raise ValueError(f"{name} must contain at most 32 messages")
    if any(type(item) is not str or not item or len(item) > 1024 for item in values):
        raise ValueError(f"{name} must contain nonempty messages no longer than 1024 characters")
    return values  # type: ignore[return-value]


def _nonnegative_summary_statistics(value: SummaryStatistics, *, name: str) -> None:
    values = (value.minimum, value.median, value.p90, value.maximum)
    if any(not math.isfinite(item) or item < 0.0 for item in values):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if values != tuple(sorted(values)):
        raise ValueError(f"{name} must be ordered minimum, median, p90, maximum")


class OccurrenceContextResult(_StrictFrozenResultModel):
    occurrence_index: StrictInt = Field(ge=0, le=499)
    macrostep: StrictInt = Field(ge=0, le=9)
    layer: StrictInt = Field(ge=0, le=59)
    color: Literal["H1", "H2", "H3", "V1", "V2", "V3"]
    edge: tuple[tuple[StrictInt, StrictInt], tuple[StrictInt, StrictInt]]
    target_hash: str
    context_weights: ProbabilityVector
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]

    @field_validator("edge", mode="before")
    @classmethod
    def normalize_edge(cls, value: object) -> object:
        return _json_tuple(value)

    @field_validator("context_weights", mode="before")
    @classmethod
    def normalize_context(cls, value: object) -> ProbabilityVector:
        return _context_weights(value, name="context_weights")

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        _sha(self.target_hash, name="target_hash")
        if self.edge[0] == self.edge[1] or any(
            item < 0 or item > 4 for pair in self.edge for item in pair
        ):
            raise ValueError("edge must contain two distinct canonical 5 by 5 coordinates")
        if self.color != _COLORS[self.layer % len(_COLORS)]:
            raise ValueError("color must match the canonical layer")
        return self


class PooledContextProfileResult(_StrictFrozenResultModel):
    trace_hash: str
    target_hash: str
    context_reduction: Literal["equal_occurrence_mean_by_target_hash"]
    zero_support_policy: Literal["exact_unsmoothed"]
    occurrence_indices: tuple[StrictInt, ...]
    multiplicity: StrictInt = Field(ge=1, le=500)
    context_weights: ProbabilityVector
    support_mask: SupportMask
    profile_hash: str
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]

    @field_validator("occurrence_indices", mode="before")
    @classmethod
    def normalize_indices(cls, value: object) -> object:
        return _json_tuple(value)

    @field_validator("context_weights", mode="before")
    @classmethod
    def normalize_context(cls, value: object) -> ProbabilityVector:
        return _context_weights(value, name="context_weights")

    @field_validator("support_mask", mode="before")
    @classmethod
    def normalize_support(cls, value: object) -> SupportMask:
        return _support_mask(value, name="support_mask")

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        for name, value in (
            ("trace_hash", self.trace_hash),
            ("target_hash", self.target_hash),
            ("profile_hash", self.profile_hash),
        ):
            _sha(value, name=name)
        if (
            len(self.occurrence_indices) != self.multiplicity
            or tuple(sorted(self.occurrence_indices)) != self.occurrence_indices
            or len(set(self.occurrence_indices)) != len(self.occurrence_indices)
            or any(index < 0 or index >= 500 for index in self.occurrence_indices)
        ):
            raise ValueError(
                "occurrence_indices must be unique canonical indices matching multiplicity"
            )
        if self.support_mask != tuple(weight != 0.0 for weight in self.context_weights):
            raise ValueError("support_mask must exactly match nonzero context_weights")
        return self


class TargetContextInitialState(_StrictFrozenResultModel):
    initial_state: Literal["single_particle"]
    initial_particle_site: tuple[StrictInt, StrictInt]
    initial_occupancy_order: tuple[tuple[StrictInt, StrictInt], ...]
    initial_occupancy: tuple[StrictFloat, ...]
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]

    @field_validator(
        "initial_particle_site", "initial_occupancy_order", "initial_occupancy", mode="before"
    )
    @classmethod
    def normalize_sequences(cls, value: object) -> object:
        return _json_tuple(value)

    @model_validator(mode="after")
    def validate_checked_state(self) -> Self:
        if self.initial_particle_site != (0, 0):
            raise ValueError("initial_particle_site must be (0, 0)")
        if self.initial_occupancy_order != OCCUPANCY_ORDER:
            raise ValueError("initial_occupancy_order must be the canonical 25 coordinates")
        if len(self.initial_occupancy) != 25 or self.initial_occupancy != (1.0,) + (0.0,) * 24:
            raise ValueError("initial_occupancy must be one followed by 24 zeros")
        return self


class OccurrenceArtifactMappingResult(_StrictFrozenResultModel):
    occurrence_index: StrictInt = Field(ge=0, le=499)
    target_hash: str
    profile_hash: str
    baseline_artifact_hash: str
    target_context_artifact_hash: str

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        for name, value in self.__dict__.items():
            if name.endswith("_hash"):
                _sha(value, name=name)
        return self


class ExactKernelEvaluation(_StrictFrozenResultModel):
    target_conditional: ConditionalTable
    equilibrium_conditional: ConditionalTable
    finite_horizon_conditionals: Mapping[StrictInt, ConditionalTable]
    target_to_equilibrium_kl: ProbabilityVector
    target_to_equilibrium_tv: ProbabilityVector
    target_to_finite_horizon_tv: Mapping[StrictInt, ProbabilityVector]
    finite_horizon_to_equilibrium_tv: Mapping[StrictInt, ProbabilityVector]
    equilibrium_normalization_error: ProbabilityVector
    equilibrium_minimum_probability: ProbabilityVector
    finite_horizon_normalization_error: Mapping[StrictInt, ProbabilityVector]
    finite_horizon_minimum_probability: Mapping[StrictInt, ProbabilityVector]
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]

    @field_validator("target_conditional", "equilibrium_conditional", mode="before")
    @classmethod
    def normalize_tables(cls, value: object) -> ConditionalTable:
        return _conditional_table(value, name="conditional")

    @field_validator(
        "target_to_equilibrium_kl",
        "target_to_equilibrium_tv",
        "equilibrium_normalization_error",
        "equilibrium_minimum_probability",
        mode="before",
    )
    @classmethod
    def normalize_diagnostics(cls, value: object) -> ProbabilityVector:
        return _vector(value, name="diagnostic", nonnegative=True)

    @field_validator(
        "finite_horizon_conditionals",
        "target_to_finite_horizon_tv",
        "finite_horizon_to_equilibrium_tv",
        "finite_horizon_normalization_error",
        "finite_horizon_minimum_probability",
        mode="before",
    )
    @classmethod
    def normalize_horizons(cls, value: object, info: Any) -> dict[int, object]:
        parsed = _parse_horizons(value, name=info.field_name)
        if info.field_name == "finite_horizon_conditionals":
            return {
                key: _conditional_table(item, name=f"{info.field_name}[{key}]")
                for key, item in parsed.items()
            }
        return {
            key: _vector(item, name=f"{info.field_name}[{key}]", nonnegative=True)
            for key, item in parsed.items()
        }

    @model_validator(mode="after")
    def freeze_horizons(self) -> Self:
        for name in (
            "finite_horizon_conditionals",
            "target_to_finite_horizon_tv",
            "finite_horizon_to_equilibrium_tv",
            "finite_horizon_normalization_error",
            "finite_horizon_minimum_probability",
        ):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))
        return self

    @field_serializer(
        "finite_horizon_conditionals",
        "target_to_finite_horizon_tv",
        "finite_horizon_to_equilibrium_tv",
        "finite_horizon_normalization_error",
        "finite_horizon_minimum_probability",
    )
    def serialize_horizons(self, value: Mapping[int, Any]) -> dict[str, Any]:
        return {str(key): item for key, item in value.items()}


class SampledK30Evaluation(_StrictFrozenResultModel):
    counts: CountTable
    conditional: ConditionalTable
    empirical_to_exact_k30_tv: ProbabilityVector
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @field_validator("counts", mode="before")
    @classmethod
    def normalize_counts(cls, value: object) -> object:
        values = _json_tuple(value)
        if not isinstance(values, tuple) or len(values) != 4:
            raise ValueError("counts must contain exactly four input rows")
        for index, row in enumerate(values):
            if (
                not isinstance(row, tuple)
                or len(row) != 4
                or any(type(item) is not int or item < 0 for item in row)
            ):
                raise ValueError(f"counts[{index}] must contain four nonnegative integers")
            if sum(row) != 4096:
                raise ValueError(f"counts[{index}] must total exactly 4096")
        return values

    @field_validator("conditional", mode="before")
    @classmethod
    def normalize_conditional(cls, value: object) -> ConditionalTable:
        return _conditional_table(value, name="conditional")

    @field_validator("empirical_to_exact_k30_tv", mode="before")
    @classmethod
    def normalize_tv(cls, value: object) -> ProbabilityVector:
        return _vector(value, name="empirical_to_exact_k30_tv", nonnegative=True)


class TargetContextOptimizationAttemptResult(_StrictFrozenResultModel):
    start_index: StrictInt = Field(ge=0, le=3)
    start_role: TargetContextStartRole
    parameters: ParameterVector
    objective: StrictFloat = Field(ge=0.0)
    raw_gradient_norm: StrictFloat = Field(ge=0.0)
    projected_gradient_norm: StrictFloat = Field(ge=0.0)
    scipy_success: StrictBool
    passed_checks: StrictBool
    iterations: StrictInt = Field(ge=0, le=2000)
    termination: str = Field(min_length=1, max_length=512)
    cap_active_parameter_count: StrictInt = Field(ge=0, le=9)
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @field_validator("parameters", mode="before")
    @classmethod
    def normalize_parameters(cls, value: object) -> ParameterVector:
        return _parameter_vector(value, name="parameters")


class TargetContextOptimizationResult(_StrictFrozenResultModel):
    artifact_hash: str
    start_values: tuple[ParameterVector, ParameterVector, ParameterVector, ParameterVector]
    parameters: ParameterVector
    selected_start_index: StrictInt = Field(ge=0, le=3)
    selected_start_role: TargetContextStartRole
    successful_attempt_count: StrictInt = Field(ge=0, le=4)
    objective: StrictFloat = Field(ge=0.0)
    projected_gradient_norm: StrictFloat = Field(ge=0.0)
    cap_active_parameter_count: StrictInt = Field(ge=0, le=9)
    attempts: tuple[
        TargetContextOptimizationAttemptResult,
        TargetContextOptimizationAttemptResult,
        TargetContextOptimizationAttemptResult,
        TargetContextOptimizationAttemptResult,
    ]
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @field_validator("start_values", mode="before")
    @classmethod
    def normalize_starts(cls, value: object) -> object:
        values = _json_tuple(value)
        if not isinstance(values, tuple) or len(values) != 4:
            raise ValueError("start_values must contain exactly four checked starts")
        return tuple(
            _parameter_vector(item, name=f"start_values[{index}]")
            for index, item in enumerate(values)
        )

    @field_validator("parameters", mode="before")
    @classmethod
    def normalize_parameters(cls, value: object) -> ParameterVector:
        return _parameter_vector(value, name="parameters")

    @field_validator("attempts", mode="before")
    @classmethod
    def normalize_attempts(cls, value: object) -> object:
        return _json_tuple(value)

    @model_validator(mode="after")
    def validate_checked_starts(self) -> Self:
        _sha(self.artifact_hash, name="artifact_hash")
        if (
            tuple(item.start_index for item in self.attempts) != (0, 1, 2, 3)
            or tuple(item.start_role for item in self.attempts) != TARGET_CONTEXT_START_ROLES
        ):
            raise ValueError("target-context attempts must use the checked start order")
        if self.selected_start_role != TARGET_CONTEXT_START_ROLES[self.selected_start_index]:
            raise ValueError("selected target-context start must match the checked start order")
        if self.start_values[1:] != _FIXED_STARTS:
            raise ValueError("final three start values must equal the checked fixed starts")
        return self


class BaselineKernelResult(_StrictFrozenResultModel):
    target_hash: str
    baseline_compiler_request_hash: str
    optimization: KernelOptimizationResult
    exact: ExactKernelEvaluation
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @field_validator("optimization", mode="before")
    @classmethod
    def normalize_legacy_optimization(cls, value: object) -> object:
        return _legacy_json_value(_json_tuple(value))

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        _sha(self.target_hash, name="target_hash")
        _sha(self.baseline_compiler_request_hash, name="baseline_compiler_request_hash")
        _sha(self.optimization.artifact_hash, name="optimization.artifact_hash")
        if self.optimization.objective < 0.0:
            raise ValueError("baseline optimization objective must be nonnegative")
        return self


class TargetContextKernelResult(_StrictFrozenResultModel):
    target_hash: str
    profile_hash: str
    target_compiler_request_hash: str
    baseline_artifact_hash: str
    optimization: TargetContextOptimizationResult
    exact: ExactKernelEvaluation
    sampled_k30: SampledK30Evaluation
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        for name, value in self.__dict__.items():
            if name.endswith("_hash"):
                _sha(value, name=name)
        return self


class PairedProfileMetrics(_StrictFrozenResultModel):
    multiplicity: StrictInt = Field(ge=1, le=500)
    context_weights: ProbabilityVector
    support_mask: SupportMask
    baseline_target_weighted_equilibrium_kl: StrictFloat = Field(ge=0.0)
    target_context_target_weighted_equilibrium_kl: StrictFloat = Field(ge=0.0)
    target_weighted_equilibrium_kl_improvement: StrictFloat
    baseline_target_weighted_equilibrium_tv: StrictFloat = Field(ge=0.0)
    target_context_target_weighted_equilibrium_tv: StrictFloat = Field(ge=0.0)
    baseline_global_kl_contribution: StrictFloat = Field(ge=0.0)
    target_context_global_kl_contribution: StrictFloat = Field(ge=0.0)
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]

    @field_validator("context_weights", mode="before")
    @classmethod
    def normalize_context(cls, value: object) -> ProbabilityVector:
        return _context_weights(value, name="context_weights")

    @field_validator("support_mask", mode="before")
    @classmethod
    def normalize_support(cls, value: object) -> SupportMask:
        return _support_mask(value, name="support_mask")

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        if self.support_mask != tuple(weight != 0.0 for weight in self.context_weights):
            raise ValueError("support_mask must exactly match nonzero context_weights")
        return self


class PairedKernelResult(_StrictFrozenResultModel):
    target_hash: str
    profile_hash: str
    baseline: BaselineKernelResult
    target_context: TargetContextKernelResult
    metrics: PairedProfileMetrics
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        _sha(self.target_hash, name="target_hash")
        _sha(self.profile_hash, name="profile_hash")
        if (
            self.baseline.target_hash != self.target_hash
            or self.target_context.target_hash != self.target_hash
            or self.target_context.profile_hash != self.profile_hash
        ):
            raise ValueError("paired artifact identities must match target and profile hashes")
        if self.target_context.baseline_artifact_hash != self.baseline.optimization.artifact_hash:
            raise ValueError("target context must cite its paired baseline artifact")
        if (
            self.target_context.optimization.start_values[0]
            != self.baseline.optimization.parameters
        ):
            raise ValueError("target-context warm start must equal the paired baseline winner")
        return self


class TargetContextScheduleMetrics(_StrictFrozenResultModel):
    occurrence_count: StrictInt = Field(ge=0, le=500)
    profile_count: StrictInt = Field(ge=0, le=37)
    baseline_occurrence_weighted_equilibrium_kl: StrictFloat = Field(ge=0.0)
    target_context_occurrence_weighted_equilibrium_kl: StrictFloat = Field(ge=0.0)
    occurrence_weighted_equilibrium_kl_improvement: StrictFloat
    baseline_occurrence_weighted_equilibrium_tv: StrictFloat = Field(ge=0.0)
    target_context_occurrence_weighted_equilibrium_tv: StrictFloat = Field(ge=0.0)
    maximum_paired_k30_equilibrium_residual: StrictFloat = Field(ge=0.0)
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]


class AllContextArtifactAssessment(_StrictFrozenResultModel):
    target_hash: str
    profile_hash: str
    artifact_hash: str
    pair_role: Literal["baseline", "target_context"]
    uniform_weighted_equilibrium_kl: StrictFloat = Field(ge=0.0)
    uniform_weighted_equilibrium_tv: StrictFloat = Field(ge=0.0)
    largest_all_row_tv: StrictFloat = Field(ge=0.0)
    largest_positive_support_row_tv: StrictFloat = Field(ge=0.0)
    exceeds_reference_tv_015: StrictBool
    exceeds_reference_tv_035: StrictBool
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        for name, value in self.__dict__.items():
            if name.endswith("_hash"):
                _sha(value, name=name)
        return self


class AllContextDegradationAssessment(_StrictFrozenResultModel):
    baseline_artifacts: tuple[AllContextArtifactAssessment, ...]
    target_context_artifacts: tuple[AllContextArtifactAssessment, ...]
    baseline_uniform_weighted_equilibrium_kl: SummaryStatistics
    baseline_uniform_weighted_equilibrium_tv: SummaryStatistics
    target_context_uniform_weighted_equilibrium_kl: SummaryStatistics
    target_context_uniform_weighted_equilibrium_tv: SummaryStatistics
    all_row_tv: SummaryStatistics
    positive_support_row_tv: SummaryStatistics
    largest_all_row_tv: StrictFloat = Field(ge=0.0)
    largest_positive_support_row_tv: StrictFloat = Field(ge=0.0)
    baseline_artifact_count_above_reference_tv_015: StrictInt = Field(ge=0)
    baseline_artifact_count_above_reference_tv_035: StrictInt = Field(ge=0)
    target_context_artifact_count_above_reference_tv_015: StrictInt = Field(ge=0)
    target_context_artifact_count_above_reference_tv_035: StrictInt = Field(ge=0)
    all_row_count_above_reference_tv_015: StrictInt = Field(ge=0)
    all_row_count_above_reference_tv_035: StrictInt = Field(ge=0)
    positive_support_row_count_above_reference_tv_015: StrictInt = Field(ge=0)
    positive_support_row_count_above_reference_tv_035: StrictInt = Field(ge=0)
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]

    @field_validator("baseline_artifacts", "target_context_artifacts", mode="before")
    @classmethod
    def normalize_artifacts(cls, value: object) -> object:
        return _json_tuple(value)

    @model_validator(mode="after")
    def validate_artifacts(self) -> Self:
        if len(self.baseline_artifacts) != 37 or len(self.target_context_artifacts) != 37:
            raise ValueError(
                "all-context assessment must contain 37 baseline and 37 target artifacts"
            )
        if any(item.pair_role != "baseline" for item in self.baseline_artifacts) or any(
            item.pair_role != "target_context" for item in self.target_context_artifacts
        ):
            raise ValueError("all-context artifact roles must match their assessment collection")
        for name in (
            "baseline_uniform_weighted_equilibrium_kl",
            "baseline_uniform_weighted_equilibrium_tv",
            "target_context_uniform_weighted_equilibrium_kl",
            "target_context_uniform_weighted_equilibrium_tv",
            "all_row_tv",
            "positive_support_row_tv",
        ):
            _nonnegative_summary_statistics(getattr(self, name), name=name)
        artifact_counts = (
            (
                "baseline_artifact_count_above_reference_tv",
                self.baseline_artifact_count_above_reference_tv_015,
                self.baseline_artifact_count_above_reference_tv_035,
            ),
            (
                "target_context_artifact_count_above_reference_tv",
                self.target_context_artifact_count_above_reference_tv_015,
                self.target_context_artifact_count_above_reference_tv_035,
            ),
        )
        row_counts = (
            (
                "all_row_count_above_reference_tv",
                self.all_row_count_above_reference_tv_015,
                self.all_row_count_above_reference_tv_035,
                _ALL_CONTEXT_ROW_COUNT,
            ),
            (
                "positive_support_row_count_above_reference_tv",
                self.positive_support_row_count_above_reference_tv_015,
                self.positive_support_row_count_above_reference_tv_035,
                _POSITIVE_SUPPORT_ROW_COUNT,
            ),
        )
        for name, count_015, count_035 in artifact_counts:
            if count_015 > 37 or count_035 > count_015:
                raise ValueError(f"{name} counts must be bounded by 37 and nested by threshold")
        for name, count_015, count_035, maximum in row_counts:
            if count_015 > maximum or count_035 > count_015:
                raise ValueError(
                    f"{name} counts must be bounded by finite rows and nested by threshold"
                )
        return self


class ZeroSupportRowAssessment(_StrictFrozenResultModel):
    target_hash: str
    profile_hash: str
    artifact_hash: str
    input_index: StrictInt = Field(ge=0, le=3)
    input_word: tuple[StrictInt, StrictInt]
    target_row: ProbabilityVector
    equilibrium_row: ProbabilityVector
    finite_horizon_rows: Mapping[StrictInt, ProbabilityVector]
    equilibrium_kl: StrictFloat = Field(ge=0.0)
    equilibrium_tv: StrictFloat = Field(ge=0.0)
    finite_horizon_kl: Mapping[StrictInt, StrictFloat]
    finite_horizon_tv: Mapping[StrictInt, StrictFloat]
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]

    @field_validator("input_word", mode="before")
    @classmethod
    def normalize_word(cls, value: object) -> object:
        return _json_tuple(value)

    @field_validator("target_row", "equilibrium_row", mode="before")
    @classmethod
    def normalize_rows(cls, value: object) -> ProbabilityVector:
        return _conditional_row(value, name="conditional row")

    @field_validator("finite_horizon_rows", "finite_horizon_kl", "finite_horizon_tv", mode="before")
    @classmethod
    def normalize_horizons(cls, value: object, info: Any) -> dict[int, object]:
        parsed = _parse_horizons(value, name=info.field_name)
        if info.field_name == "finite_horizon_rows":
            return {
                key: _conditional_row(item, name=f"finite_horizon_rows[{key}]")
                for key, item in parsed.items()
            }
        return {
            key: _finite_float(item, name=f"{info.field_name}[{key}]", nonnegative=True)
            for key, item in parsed.items()
        }

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        for name, value in self.__dict__.items():
            if name.endswith("_hash"):
                _sha(value, name=name)
        if self.input_word not in _WORDS or self.input_word != _WORDS[self.input_index]:
            raise ValueError("input_word must match its canonical input index")
        for name in ("finite_horizon_rows", "finite_horizon_kl", "finite_horizon_tv"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))
        return self

    @field_serializer("finite_horizon_rows", "finite_horizon_kl", "finite_horizon_tv")
    def serialize_horizons(self, value: Mapping[int, Any]) -> dict[str, Any]:
        return {str(key): item for key, item in value.items()}


class ZeroSupportAssessment(_StrictFrozenResultModel):
    rows: tuple[ZeroSupportRowAssessment, ...]
    equilibrium_kl: SummaryStatistics
    equilibrium_tv: SummaryStatistics
    finite_horizon_kl: Mapping[StrictInt, SummaryStatistics]
    finite_horizon_tv: Mapping[StrictInt, SummaryStatistics]
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]

    @field_validator("rows", mode="before")
    @classmethod
    def normalize_rows(cls, value: object) -> object:
        return _json_tuple(value)

    @field_validator("finite_horizon_kl", "finite_horizon_tv", mode="before")
    @classmethod
    def normalize_horizon_statistics(cls, value: object, info: Any) -> dict[int, object]:
        return _parse_horizons(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_rows(self) -> Self:
        if len(self.rows) != 37:
            raise ValueError("zero-support assessment must contain exactly 37 ordered rows")
        if tuple(
            (row.target_hash, row.profile_hash, row.artifact_hash, row.input_index)
            for row in self.rows
        ) != tuple(
            sorted(
                (row.target_hash, row.profile_hash, row.artifact_hash, row.input_index)
                for row in self.rows
            )
        ):
            raise ValueError("zero-support rows must use canonical identity order")
        _nonnegative_summary_statistics(self.equilibrium_kl, name="equilibrium_kl")
        _nonnegative_summary_statistics(self.equilibrium_tv, name="equilibrium_tv")
        for name in ("finite_horizon_kl", "finite_horizon_tv"):
            for horizon, statistics in getattr(self, name).items():
                _nonnegative_summary_statistics(statistics, name=f"{name}[{horizon}]")
        object.__setattr__(self, "finite_horizon_kl", _freeze_mapping(self.finite_horizon_kl))
        object.__setattr__(self, "finite_horizon_tv", _freeze_mapping(self.finite_horizon_tv))
        return self

    @field_serializer("finite_horizon_kl", "finite_horizon_tv")
    def serialize_horizon_statistics(self, value: Mapping[int, Any]) -> dict[str, Any]:
        return {str(key): item for key, item in value.items()}


class DeterministicAcceptance(_StrictFrozenResultModel):
    context_derivation_passed: StrictBool
    probability_integrity_passed: StrictBool
    baseline_compilation_and_accuracy_passed: StrictBool
    target_optimizer_passed: StrictBool
    profile_kl_non_regression_passed: StrictBool
    occurrence_weighted_kl_improvement_passed: StrictBool
    k30_equilibrium_mixing_passed: StrictBool
    k30_no_worse_than_k1_passed: StrictBool
    deterministic_consistency_passed: StrictBool
    check_messages: tuple[str, ...]
    passed: StrictBool
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @field_validator("check_messages", mode="before")
    @classmethod
    def normalize_messages(cls, value: object) -> tuple[str, ...]:
        return _messages(value, name="check_messages")

    @model_validator(mode="after")
    def validate_conjunction(self) -> Self:
        checks = (
            self.context_derivation_passed,
            self.probability_integrity_passed,
            self.baseline_compilation_and_accuracy_passed,
            self.target_optimizer_passed,
            self.profile_kl_non_regression_passed,
            self.occurrence_weighted_kl_improvement_passed,
            self.k30_equilibrium_mixing_passed,
            self.k30_no_worse_than_k1_passed,
            self.deterministic_consistency_passed,
        )
        if self.passed != all(checks):
            raise ValueError("deterministic acceptance must equal the conjunction of its checks")
        return self


class SampledFidelityResidual(_StrictFrozenResultModel):
    target_hash: str
    profile_hash: str
    input_index: StrictInt = Field(ge=0, le=3)
    residual: StrictFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        _sha(self.target_hash, name="target_hash")
        _sha(self.profile_hash, name="profile_hash")
        return self


class SampledFidelityAssessment(_StrictFrozenResultModel):
    maximum_empirical_k30_residual: StrictFloat = Field(ge=0.0)
    per_target_profile_input_residuals: tuple[SampledFidelityResidual, ...]
    checked_tolerance: StrictFloat = Field(ge=0.0)
    check_messages: tuple[str, ...]
    passed: StrictBool
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @field_validator("per_target_profile_input_residuals", "check_messages", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object) -> object:
        return _json_tuple(value)

    @field_validator("check_messages", mode="after")
    @classmethod
    def validate_messages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _messages(value, name="check_messages")

    @model_validator(mode="after")
    def validate_residuals(self) -> Self:
        if len(self.per_target_profile_input_residuals) != 148:
            raise ValueError("sampled fidelity must contain exactly 37 by 4 residuals")
        identities = tuple(
            (item.target_hash, item.profile_hash, item.input_index)
            for item in self.per_target_profile_input_residuals
        )
        if identities != tuple(sorted(identities)) or len(set(identities)) != 148:
            raise ValueError("sampled residuals must be unique and canonically ordered")
        if self.passed != (self.maximum_empirical_k30_residual <= self.checked_tolerance):
            raise ValueError("sampled fidelity passed flag must match the checked tolerance")
        return self


class SeedAcceptance(_StrictFrozenResultModel):
    deterministic_acceptance: DeterministicAcceptance
    sampled_fidelity: SampledFidelityAssessment
    check_messages: tuple[str, ...]
    passed: StrictBool
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @field_validator("check_messages", mode="before")
    @classmethod
    def normalize_messages(cls, value: object) -> tuple[str, ...]:
        return _messages(value, name="check_messages")

    @model_validator(mode="after")
    def validate_conjunction(self) -> Self:
        if self.passed != (self.deterministic_acceptance.passed and self.sampled_fidelity.passed):
            raise ValueError("seed acceptance must equal its deterministic and sampled conjunction")
        return self


class OptimizerPhaseResult(_StrictFrozenResultModel):
    seconds: StrictFloat = Field(ge=0.0)
    cache_reused: StrictBool
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @model_validator(mode="after")
    def validate_cache_semantics(self) -> Self:
        if self.cache_reused and self.seconds != 0.0:
            raise ValueError("cached optimizer phases must record exactly zero seconds")
        return self


class TargetContextPAsymSwapSummary(_StrictFrozenResultModel):
    source_reference: Literal[PAPER_SOURCE]
    target_compiler_request_hash: str
    baseline_compiler_request_hash: str
    initial_state: TargetContextInitialState
    context_source: Literal["exact_target_pre_gate"]
    context_reduction: Literal["equal_occurrence_mean_by_target_hash"]
    zero_support_policy: Literal["exact_unsmoothed"]
    warm_start_policy: Literal["paired_uniform_artifact_then_three_fixed_restarts"]
    trace: tuple[OccurrenceContextResult, ...]
    trace_hash: str
    profiles: tuple[PooledContextProfileResult, ...]
    occurrence_mapping: tuple[OccurrenceArtifactMappingResult, ...]
    pairs: tuple[PairedKernelResult, ...]
    schedule_metrics: TargetContextScheduleMetrics
    deterministic_acceptance: DeterministicAcceptance
    sampled_fidelity: SampledFidelityAssessment
    seed_acceptance: SeedAcceptance
    all_context_degradation: AllContextDegradationAssessment
    zero_support_assessment: ZeroSupportAssessment
    baseline_optimizer_phase: OptimizerPhaseResult
    target_context_optimizer_phase: OptimizerPhaseResult
    deterministic_result_hash: str
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @field_validator("trace", "profiles", "occurrence_mapping", "pairs", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object) -> object:
        return _json_tuple(value)

    @model_validator(mode="after")
    def validate_bounded_summary(self) -> Self:
        for name in (
            "target_compiler_request_hash",
            "baseline_compiler_request_hash",
            "trace_hash",
            "deterministic_result_hash",
        ):
            _sha(getattr(self, name), name=name)
        if (len(self.trace), len(self.profiles), len(self.occurrence_mapping), len(self.pairs)) != (
            500,
            37,
            500,
            37,
        ):
            raise ValueError(
                "summary must contain exactly 500/37/500/37 trace, profile, mapping, "
                "and pair records"
            )
        if tuple(item.occurrence_index for item in self.trace) != tuple(range(500)):
            raise ValueError("trace must use occurrence indices exactly 0 through 499")
        profile_targets = tuple(item.target_hash for item in self.profiles)
        pair_targets = tuple(item.target_hash for item in self.pairs)
        if profile_targets != tuple(sorted(profile_targets)) or pair_targets != profile_targets:
            raise ValueError("profiles and pairs must use the same target-hash order")
        if (
            len(set(profile_targets)) != 37
            or len({item.profile_hash for item in self.profiles}) != 37
        ):
            raise ValueError("profiles must have unique target and profile identities")
        if sum(item.multiplicity for item in self.profiles) != 500 or any(
            item.trace_hash != self.trace_hash for item in self.profiles
        ):
            raise ValueError("profiles must reference the trace and total all 500 occurrences")
        if tuple(item.occurrence_index for item in self.occurrence_mapping) != tuple(range(500)):
            raise ValueError("occurrence mapping must use occurrence indices exactly 0 through 499")
        profiles = {item.target_hash: item for item in self.profiles}
        pairs = {item.target_hash: item for item in self.pairs}
        if any(
            item.target_hash not in profiles or item.target_hash not in pairs for item in self.trace
        ):
            raise ValueError("every trace occurrence must resolve to one profile and pair")
        for profile, pair in zip(self.profiles, self.pairs, strict=True):
            if (
                pair.profile_hash != profile.profile_hash
                or pair.baseline.baseline_compiler_request_hash
                != self.baseline_compiler_request_hash
                or pair.target_context.target_compiler_request_hash
                != self.target_compiler_request_hash
            ):
                raise ValueError(
                    "each pair must use its standalone profile and the checked compiler requests"
                )
        for mapping, occurrence in zip(self.occurrence_mapping, self.trace, strict=True):
            profile = profiles.get(mapping.target_hash)
            pair = pairs.get(mapping.target_hash)
            if (
                mapping.target_hash != occurrence.target_hash
                or profile is None
                or pair is None
                or mapping.profile_hash != profile.profile_hash
                or mapping.occurrence_index not in profile.occurrence_indices
                or mapping.baseline_artifact_hash != pair.baseline.optimization.artifact_hash
                or mapping.target_context_artifact_hash
                != pair.target_context.optimization.artifact_hash
            ):
                raise ValueError(
                    "occurrence mapping must resolve to its exact profile and paired artifacts"
                )
        if (
            len({pair.baseline.optimization.artifact_hash for pair in self.pairs}) != 37
            or len({pair.target_context.optimization.artifact_hash for pair in self.pairs}) != 37
        ):
            raise ValueError("paired artifacts must have unique identities")
        expected_baseline_assessments = tuple(
            (
                pair.target_hash,
                pair.profile_hash,
                pair.baseline.optimization.artifact_hash,
                "baseline",
            )
            for pair in self.pairs
        )
        expected_target_assessments = tuple(
            (
                pair.target_hash,
                pair.profile_hash,
                pair.target_context.optimization.artifact_hash,
                "target_context",
            )
            for pair in self.pairs
        )
        observed_baseline_assessments = tuple(
            (item.target_hash, item.profile_hash, item.artifact_hash, item.pair_role)
            for item in self.all_context_degradation.baseline_artifacts
        )
        observed_target_assessments = tuple(
            (item.target_hash, item.profile_hash, item.artifact_hash, item.pair_role)
            for item in self.all_context_degradation.target_context_artifacts
        )
        if (
            observed_baseline_assessments != expected_baseline_assessments
            or observed_target_assessments != expected_target_assessments
        ):
            raise ValueError("all-context assessments must be pair-aligned and uniquely linked")
        expected_zero_support_rows = []
        for profile, pair in zip(self.profiles, self.pairs, strict=True):
            unsupported = tuple(
                index for index, supported in enumerate(profile.support_mask) if not supported
            )
            if unsupported != (3,):
                raise ValueError(
                    "each checked profile must have only canonical input word 11 unsupported"
                )
            expected_zero_support_rows.append(
                (
                    pair.target_hash,
                    profile.profile_hash,
                    pair.target_context.optimization.artifact_hash,
                    3,
                    _WORDS[3],
                )
            )
        observed_zero_support_rows = tuple(
            (row.target_hash, row.profile_hash, row.artifact_hash, row.input_index, row.input_word)
            for row in self.zero_support_assessment.rows
        )
        if observed_zero_support_rows != tuple(expected_zero_support_rows):
            raise ValueError(
                "zero-support rows must be pair-aligned canonical unsupported input rows"
            )
        if (
            self.schedule_metrics.occurrence_count != 500
            or self.schedule_metrics.profile_count != 37
        ):
            raise ValueError(
                "schedule metrics must report the fixed 500 occurrences and 37 profiles"
            )
        if (
            self.seed_acceptance.deterministic_acceptance != self.deterministic_acceptance
            or self.seed_acceptance.sampled_fidelity != self.sampled_fidelity
        ):
            raise ValueError("seed acceptance must embed the summary acceptance layers")
        return self


def _row_tv(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * math.fsum(abs(a - b) for a, b in zip(left, right, strict=True))


def _row_kl(target: Sequence[float], observed: Sequence[float]) -> float:
    terms: list[float] = []
    for expected, actual in zip(target, observed, strict=True):
        if expected == 0.0:
            continue
        if actual <= 0.0:
            return math.inf
        terms.append(expected * (math.log(expected) - math.log(actual)))
    return math.fsum(terms)


def _metric_inputs(
    target: Sequence[Sequence[float]],
    observed: Sequence[Sequence[float]],
    context_weights: Sequence[float],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], tuple[float, ...]]:
    target_rows = tuple(tuple(float(value) for value in row) for row in target)
    observed_rows = tuple(tuple(float(value) for value in row) for row in observed)
    weights = tuple(float(value) for value in context_weights)
    if len(target_rows) != 4 or len(observed_rows) != 4 or len(weights) != 4:
        raise ValueError("conditional tables and context weights must have four rows")
    if any(len(row) != 4 for row in (*target_rows, *observed_rows)):
        raise ValueError("conditional rows must have four outputs")
    values = (
        *weights,
        *(value for row in target_rows for value in row),
        *(value for row in observed_rows for value in row),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("conditional tables and context weights must be finite and nonnegative")
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("context weights must sum to one within 1e-12")
    for name, rows in (("target", target_rows), ("observed", observed_rows)):
        for context, row in enumerate(rows):
            if not math.isclose(math.fsum(row), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{name} context={context} must sum to one within 1e-12")
    return target_rows, observed_rows, weights


def context_weighted_tv(
    target: Sequence[Sequence[float]],
    observed: Sequence[Sequence[float]],
    context_weights: Sequence[float],
) -> float:
    """Return the context-weighted mean of row total-variation distances."""

    target_rows, observed_rows, weights = _metric_inputs(target, observed, context_weights)
    return math.fsum(
        weight * _row_tv(expected, actual)
        for weight, expected, actual in zip(weights, target_rows, observed_rows, strict=True)
    )


def context_weighted_kl(
    target: Sequence[Sequence[float]],
    observed: Sequence[Sequence[float]],
    context_weights: Sequence[float],
) -> float:
    """Return target-to-observed conditional KL in natural-log units (nats)."""

    target_rows, observed_rows, weights = _metric_inputs(target, observed, context_weights)
    result = math.fsum(
        weight * _row_kl(expected, actual)
        for weight, expected, actual in zip(weights, target_rows, observed_rows, strict=True)
        if weight != 0.0
    )
    if not math.isfinite(result):
        raise ValueError("context-weighted KL is infinite because observed support is missing")
    return result


def derive_paired_profile_metrics(
    pair: PairedKernelResult,
    profile: PooledContextProfileResult,
) -> PairedProfileMetrics:
    """Derive one pair's primary target-weighted metrics from exact tables."""

    if pair.target_hash != profile.target_hash or pair.profile_hash != profile.profile_hash:
        raise ValueError(
            "paired metric identity mismatch "
            f"target_hash={pair.target_hash} profile_hash={profile.profile_hash}"
        )
    target = pair.baseline.exact.target_conditional
    if pair.target_context.exact.target_conditional != target:
        raise ValueError(f"paired target tables differ target_hash={pair.target_hash}")
    baseline_kl = context_weighted_kl(
        target, pair.baseline.exact.equilibrium_conditional, profile.context_weights
    )
    target_kl = context_weighted_kl(
        target, pair.target_context.exact.equilibrium_conditional, profile.context_weights
    )
    baseline_tv = context_weighted_tv(
        target, pair.baseline.exact.equilibrium_conditional, profile.context_weights
    )
    target_tv = context_weighted_tv(
        target, pair.target_context.exact.equilibrium_conditional, profile.context_weights
    )
    return PairedProfileMetrics(
        multiplicity=profile.multiplicity,
        context_weights=profile.context_weights,
        support_mask=profile.support_mask,
        baseline_target_weighted_equilibrium_kl=baseline_kl,
        target_context_target_weighted_equilibrium_kl=target_kl,
        target_weighted_equilibrium_kl_improvement=baseline_kl - target_kl,
        baseline_target_weighted_equilibrium_tv=baseline_tv,
        target_context_target_weighted_equilibrium_tv=target_tv,
        baseline_global_kl_contribution=profile.multiplicity * baseline_kl / 500,
        target_context_global_kl_contribution=profile.multiplicity * target_kl / 500,
        evidence_class=EvidenceClass.EXACT_REFERENCE,
    )


def derive_schedule_metrics(pairs: Sequence[PairedKernelResult]) -> TargetContextScheduleMetrics:
    """Reduce pair metrics in sorted target-hash order with occurrence multiplicity."""

    ordered = tuple(sorted(pairs, key=lambda item: item.target_hash))
    if len(ordered) != 37 or len({pair.target_hash for pair in ordered}) != 37:
        raise ValueError("schedule metrics require 37 unique target-hash pairs")
    occurrence_count = sum(pair.metrics.multiplicity for pair in ordered)
    if occurrence_count != 500:
        raise ValueError(f"schedule occurrence count observed={occurrence_count} bound=500")

    def weighted(name: str) -> float:
        return (
            math.fsum(pair.metrics.multiplicity * getattr(pair.metrics, name) for pair in ordered)
            / 500
        )

    baseline_kl = weighted("baseline_target_weighted_equilibrium_kl")
    target_kl = weighted("target_context_target_weighted_equilibrium_kl")
    maximum_residual = max(
        _row_tv(
            exact.finite_horizon_conditionals[30][context], exact.equilibrium_conditional[context]
        )
        for pair in ordered
        for exact in (pair.baseline.exact, pair.target_context.exact)
        for context in range(4)
    )
    return TargetContextScheduleMetrics(
        occurrence_count=occurrence_count,
        profile_count=len(ordered),
        baseline_occurrence_weighted_equilibrium_kl=baseline_kl,
        target_context_occurrence_weighted_equilibrium_kl=target_kl,
        occurrence_weighted_equilibrium_kl_improvement=baseline_kl - target_kl,
        baseline_occurrence_weighted_equilibrium_tv=weighted(
            "baseline_target_weighted_equilibrium_tv"
        ),
        target_context_occurrence_weighted_equilibrium_tv=weighted(
            "target_context_target_weighted_equilibrium_tv"
        ),
        maximum_paired_k30_equilibrium_residual=maximum_residual,
        evidence_class=EvidenceClass.EXACT_REFERENCE,
    )


def derive_all_context_degradation(
    pairs: Sequence[PairedKernelResult],
) -> AllContextDegradationAssessment:
    """Measure baseline and target all-context behavior without creating a gate."""

    ordered = tuple(sorted(pairs, key=lambda item: item.target_hash))
    if len(ordered) != 37:
        raise ValueError("all-context assessment requires exactly 37 pairs")
    uniform = (0.25, 0.25, 0.25, 0.25)
    baseline_items: list[AllContextArtifactAssessment] = []
    target_items: list[AllContextArtifactAssessment] = []
    all_rows: list[float] = []
    positive_rows: list[float] = []
    for pair in ordered:
        for role, kernel, supported in (
            ("baseline", pair.baseline, (True, True, True, True)),
            ("target_context", pair.target_context, pair.metrics.support_mask),
        ):
            exact = kernel.exact
            row_tvs = tuple(
                _row_tv(expected, actual)
                for expected, actual in zip(
                    exact.target_conditional, exact.equilibrium_conditional, strict=True
                )
            )
            all_rows.extend(row_tvs)
            positive = tuple(value for value, keep in zip(row_tvs, supported, strict=True) if keep)
            positive_rows.extend(positive)
            uniform_kl = context_weighted_kl(
                exact.target_conditional, exact.equilibrium_conditional, uniform
            )
            uniform_tv = context_weighted_tv(
                exact.target_conditional, exact.equilibrium_conditional, uniform
            )
            artifact_hash = kernel.optimization.artifact_hash
            item = AllContextArtifactAssessment(
                target_hash=pair.target_hash,
                profile_hash=pair.profile_hash,
                artifact_hash=artifact_hash,
                pair_role=role,
                uniform_weighted_equilibrium_kl=uniform_kl,
                uniform_weighted_equilibrium_tv=uniform_tv,
                largest_all_row_tv=max(row_tvs),
                largest_positive_support_row_tv=max(positive),
                exceeds_reference_tv_015=uniform_tv > 0.15,
                exceeds_reference_tv_035=uniform_tv > 0.35,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
            )
            (baseline_items if role == "baseline" else target_items).append(item)

    def values(items: Sequence[AllContextArtifactAssessment], name: str) -> list[float]:
        return [float(getattr(item, name)) for item in items]

    return AllContextDegradationAssessment(
        baseline_artifacts=tuple(baseline_items),
        target_context_artifacts=tuple(target_items),
        baseline_uniform_weighted_equilibrium_kl=summarize_values(
            values(baseline_items, "uniform_weighted_equilibrium_kl")
        ),
        baseline_uniform_weighted_equilibrium_tv=summarize_values(
            values(baseline_items, "uniform_weighted_equilibrium_tv")
        ),
        target_context_uniform_weighted_equilibrium_kl=summarize_values(
            values(target_items, "uniform_weighted_equilibrium_kl")
        ),
        target_context_uniform_weighted_equilibrium_tv=summarize_values(
            values(target_items, "uniform_weighted_equilibrium_tv")
        ),
        all_row_tv=summarize_values(all_rows),
        positive_support_row_tv=summarize_values(positive_rows),
        largest_all_row_tv=max(all_rows),
        largest_positive_support_row_tv=max(positive_rows),
        baseline_artifact_count_above_reference_tv_015=sum(
            item.exceeds_reference_tv_015 for item in baseline_items
        ),
        baseline_artifact_count_above_reference_tv_035=sum(
            item.exceeds_reference_tv_035 for item in baseline_items
        ),
        target_context_artifact_count_above_reference_tv_015=sum(
            item.exceeds_reference_tv_015 for item in target_items
        ),
        target_context_artifact_count_above_reference_tv_035=sum(
            item.exceeds_reference_tv_035 for item in target_items
        ),
        all_row_count_above_reference_tv_015=sum(value > 0.15 for value in all_rows),
        all_row_count_above_reference_tv_035=sum(value > 0.35 for value in all_rows),
        positive_support_row_count_above_reference_tv_015=sum(
            value > 0.15 for value in positive_rows
        ),
        positive_support_row_count_above_reference_tv_035=sum(
            value > 0.35 for value in positive_rows
        ),
        evidence_class=EvidenceClass.EXACT_REFERENCE,
    )


def derive_zero_support_assessment(
    pairs: Sequence[PairedKernelResult],
) -> ZeroSupportAssessment:
    """Collect diagnostics for rows whose pooled context weight is exactly zero."""

    rows: list[ZeroSupportRowAssessment] = []
    for pair in sorted(pairs, key=lambda item: item.target_hash):
        exact = pair.target_context.exact
        for context, weight in enumerate(pair.metrics.context_weights):
            if weight != 0.0:
                continue
            target_row = exact.target_conditional[context]
            equilibrium_row = exact.equilibrium_conditional[context]
            rows.append(
                ZeroSupportRowAssessment(
                    target_hash=pair.target_hash,
                    profile_hash=pair.profile_hash,
                    artifact_hash=pair.target_context.optimization.artifact_hash,
                    input_index=context,
                    input_word=WORD_ORDER[context],
                    target_row=target_row,
                    equilibrium_row=equilibrium_row,
                    finite_horizon_rows={
                        horizon: exact.finite_horizon_conditionals[horizon][context]
                        for horizon in _HORIZONS
                    },
                    equilibrium_kl=_row_kl(target_row, equilibrium_row),
                    equilibrium_tv=_row_tv(target_row, equilibrium_row),
                    finite_horizon_kl={
                        horizon: _row_kl(
                            target_row, exact.finite_horizon_conditionals[horizon][context]
                        )
                        for horizon in _HORIZONS
                    },
                    finite_horizon_tv={
                        horizon: _row_tv(
                            target_row, exact.finite_horizon_conditionals[horizon][context]
                        )
                        for horizon in _HORIZONS
                    },
                    evidence_class=EvidenceClass.EXACT_REFERENCE,
                )
            )
    if len(rows) != 37 or any(row.input_index != 3 for row in rows):
        raise ValueError(
            "zero-support assessment observed="
            f"{[(row.target_hash, row.input_word) for row in rows]} bound=37 canonical word 11 rows"
        )
    return ZeroSupportAssessment(
        rows=tuple(rows),
        equilibrium_kl=summarize_values([row.equilibrium_kl for row in rows]),
        equilibrium_tv=summarize_values([row.equilibrium_tv for row in rows]),
        finite_horizon_kl={
            horizon: summarize_values([row.finite_horizon_kl[horizon] for row in rows])
            for horizon in _HORIZONS
        },
        finite_horizon_tv={
            horizon: summarize_values([row.finite_horizon_tv[horizon] for row in rows])
            for horizon in _HORIZONS
        },
        evidence_class=EvidenceClass.EXACT_REFERENCE,
    )


def derive_sampled_fidelity(
    pairs: Sequence[PairedKernelResult], run: TargetContextCompilerRunConfig
) -> SampledFidelityAssessment:
    """Derive the sampled-only K30 gate from integer count tables."""

    ordered = tuple(sorted(pairs, key=lambda item: item.target_hash))
    canonical_sampled = tuple(
        derive_sampled_k30_evaluation(
            pair.target_context.sampled_k30.counts,
            pair.target_context.exact.finite_horizon_conditionals[30],
            chain_count=run.chain_count_per_context,
        )
        for pair in ordered
    )
    residuals = tuple(
        SampledFidelityResidual(
            target_hash=pair.target_hash,
            profile_hash=pair.profile_hash,
            input_index=context,
            residual=sampled.empirical_to_exact_k30_tv[context],
        )
        for pair, sampled in zip(ordered, canonical_sampled, strict=True)
        for context in range(4)
    )
    maximum = max(item.residual for item in residuals)
    passed = maximum <= run.thrml_k30_tv_tolerance
    return SampledFidelityAssessment(
        maximum_empirical_k30_residual=maximum,
        per_target_profile_input_residuals=residuals,
        checked_tolerance=run.thrml_k30_tv_tolerance,
        check_messages=(f"sampled_fidelity={'passed' if passed else 'failed'}",),
        passed=passed,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )


def derive_deterministic_acceptance(
    pairs: Sequence[PairedKernelResult],
    schedule: TargetContextScheduleMetrics,
    degradation: AllContextDegradationAssessment,
    run: TargetContextCompilerRunConfig,
    *,
    context_derivation_passed: bool = True,
    probability_integrity_passed: bool = True,
    deterministic_consistency_passed: bool = True,
) -> DeterministicAcceptance:
    """Derive the nine named deterministic gates with baseline-only accuracy limits."""

    ordered = tuple(sorted(pairs, key=lambda item: item.target_hash))
    baseline_compilation = all(
        pair.baseline.optimization.successful_restart_count >= 1
        and pair.baseline.optimization.projected_gradient_norm <= run.projected_gradient_tolerance
        and pair.baseline.optimization.cap_active_parameter_count <= _N_PARAMETERS
        for pair in ordered
    )
    baseline_accuracy = (
        degradation.baseline_uniform_weighted_equilibrium_tv.median
        <= run.baseline_median_equilibrium_tv_tolerance
        and degradation.baseline_uniform_weighted_equilibrium_tv.maximum
        <= run.baseline_worst_equilibrium_tv_tolerance
    )
    baseline_compilation_and_accuracy_passed = baseline_compilation and baseline_accuracy
    target_optimizer_passed = all(
        pair.target_context.optimization.successful_attempt_count >= 1
        and pair.target_context.optimization.projected_gradient_norm
        <= run.projected_gradient_tolerance
        and pair.target_context.optimization.cap_active_parameter_count <= _N_PARAMETERS
        for pair in ordered
    )
    profile_kl_non_regression_passed = all(
        pair.metrics.target_context_target_weighted_equilibrium_kl
        <= pair.metrics.baseline_target_weighted_equilibrium_kl
        + run.profile_kl_non_regression_tolerance
        for pair in ordered
    )
    occurrence_weighted_kl_improvement_passed = (
        schedule.occurrence_weighted_equilibrium_kl_improvement
        >= run.minimum_occurrence_weighted_kl_improvement
    )
    residuals = tuple(
        (
            pair.target_hash,
            role,
            context,
            _row_tv(
                exact.finite_horizon_conditionals[30][context],
                exact.equilibrium_conditional[context],
            ),
            _row_tv(
                exact.finite_horizon_conditionals[1][context],
                exact.equilibrium_conditional[context],
            ),
        )
        for pair in ordered
        for role, exact in (
            ("baseline", pair.baseline.exact),
            ("target_context", pair.target_context.exact),
        )
        for context in range(4)
    )
    k30_equilibrium_mixing_passed = all(
        k30 <= run.k30_equilibrium_tv_tolerance for _, _, _, k30, _ in residuals
    )
    k30_no_worse_than_k1_passed = all(
        k30 <= k1 + run.exact_normalization_tolerance for _, _, _, k30, k1 in residuals
    )
    checks = {
        "context_derivation_passed": context_derivation_passed,
        "probability_integrity_passed": probability_integrity_passed,
        "baseline_compilation_and_accuracy_passed": baseline_compilation_and_accuracy_passed,
        "target_optimizer_passed": target_optimizer_passed,
        "profile_kl_non_regression_passed": profile_kl_non_regression_passed,
        "occurrence_weighted_kl_improvement_passed": occurrence_weighted_kl_improvement_passed,
        "k30_equilibrium_mixing_passed": k30_equilibrium_mixing_passed,
        "k30_no_worse_than_k1_passed": k30_no_worse_than_k1_passed,
        "deterministic_consistency_passed": deterministic_consistency_passed,
    }
    return DeterministicAcceptance(
        **checks,
        check_messages=tuple(
            f"{name}={'passed' if passed else 'failed'}" for name, passed in checks.items()
        ),
        passed=all(checks.values()),
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )


def derive_seed_acceptance(
    deterministic: DeterministicAcceptance, sampled: SampledFidelityAssessment
) -> SeedAcceptance:
    passed = deterministic.passed and sampled.passed
    return SeedAcceptance(
        deterministic_acceptance=deterministic,
        sampled_fidelity=sampled,
        check_messages=(f"seed_acceptance={'passed' if passed else 'failed'}",),
        passed=passed,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )


def derive_exact_kernel_evaluation(
    parameters: Sequence[float], target: Sequence[Sequence[float]], *, beta: float = 1.0
) -> ExactKernelEvaluation:
    """Evaluate exact equilibrium/horizons and every persisted row diagnostic."""

    checked_target, _, _ = _metric_inputs(target, target, (0.25, 0.25, 0.25, 0.25))
    kernel_parameters = KernelParameters(tuple(float(value) for value in parameters))  # type: ignore[arg-type]
    equilibrium_array = equilibrium_conditional(kernel_parameters, beta=beta)
    finite_arrays = finite_horizon_conditional(kernel_parameters, _HORIZONS, beta=beta)
    equilibrium = tuple(tuple(float(value) for value in row) for row in equilibrium_array)
    finite = {
        horizon: tuple(tuple(float(value) for value in row) for row in finite_arrays[horizon])
        for horizon in _HORIZONS
    }
    return ExactKernelEvaluation(
        target_conditional=checked_target,
        equilibrium_conditional=equilibrium,
        finite_horizon_conditionals=finite,
        target_to_equilibrium_kl=tuple(
            _row_kl(expected, actual)
            for expected, actual in zip(checked_target, equilibrium, strict=True)
        ),
        target_to_equilibrium_tv=tuple(
            _row_tv(expected, actual)
            for expected, actual in zip(checked_target, equilibrium, strict=True)
        ),
        target_to_finite_horizon_tv={
            horizon: tuple(
                _row_tv(expected, actual)
                for expected, actual in zip(checked_target, finite[horizon], strict=True)
            )
            for horizon in _HORIZONS
        },
        finite_horizon_to_equilibrium_tv={
            horizon: tuple(
                _row_tv(actual, stationary)
                for actual, stationary in zip(finite[horizon], equilibrium, strict=True)
            )
            for horizon in _HORIZONS
        },
        equilibrium_normalization_error=tuple(abs(math.fsum(row) - 1.0) for row in equilibrium),
        equilibrium_minimum_probability=tuple(min(row) for row in equilibrium),
        finite_horizon_normalization_error={
            horizon: tuple(abs(math.fsum(row) - 1.0) for row in finite[horizon])
            for horizon in _HORIZONS
        },
        finite_horizon_minimum_probability={
            horizon: tuple(min(row) for row in finite[horizon]) for horizon in _HORIZONS
        },
        evidence_class=EvidenceClass.EXACT_REFERENCE,
    )


def derive_sampled_k30_evaluation(
    counts: Sequence[Sequence[int]],
    exact_k30: Sequence[Sequence[float]],
    *,
    chain_count: int = 4096,
) -> SampledK30Evaluation:
    """Derive the empirical table and residuals exclusively from integer counts."""

    count_rows = tuple(tuple(row) for row in counts)
    if len(count_rows) != 4 or any(len(row) != 4 for row in count_rows):
        raise ValueError("sample counts must contain four by four cells")
    if any(type(value) is not int or value < 0 for row in count_rows for value in row):
        raise ValueError("sample counts must be nonnegative integers")
    for context, row in enumerate(count_rows):
        if sum(row) != chain_count:
            raise ValueError(
                f"sample counts context={context} observed={sum(row)} bound={chain_count}"
            )
    conditional = tuple(tuple(value / chain_count for value in row) for row in count_rows)
    exact_rows = tuple(tuple(float(value) for value in row) for row in exact_k30)
    return SampledK30Evaluation(
        counts=count_rows,
        conditional=conditional,
        empirical_to_exact_k30_tv=tuple(
            _row_tv(observed, expected)
            for observed, expected in zip(conditional, exact_rows, strict=True)
        ),
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )


def _compare_float(observed: float, expected: float, *, path: str, seed: int) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{path} seed={seed} observed={observed} bound={expected}")


def _compare_nested(observed: object, expected: object, *, path: str, seed: int) -> None:
    """Compare derived models exactly except for declared float64 roundoff."""

    if hasattr(observed, "model_dump"):
        observed = observed.model_dump(mode="python")  # type: ignore[union-attr]
    if hasattr(expected, "model_dump"):
        expected = expected.model_dump(mode="python")  # type: ignore[union-attr]
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        if tuple(observed) != tuple(expected):
            raise ValueError(
                f"{path} seed={seed} keys observed={tuple(observed)} bound={tuple(expected)}"
            )
        for key in expected:
            _compare_nested(observed[key], expected[key], path=f"{path}.{key}", seed=seed)
        return
    if isinstance(observed, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(observed) != len(expected):
            raise ValueError(
                f"{path} seed={seed} length observed={len(observed)} bound={len(expected)}"
            )
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _compare_nested(left, right, path=f"{path}[{index}]", seed=seed)
        return
    if type(observed) is float and type(expected) is float:
        _compare_float(observed, expected, path=path, seed=seed)
        return
    if observed != expected:
        raise ValueError(f"{path} seed={seed} observed={observed!r} bound={expected!r}")


def _compare_identity_nested(observed: object, expected: object, *, path: str, seed: int) -> None:
    """Compare identity-bearing values exactly, including every binary64 value."""

    if hasattr(observed, "model_dump"):
        observed = observed.model_dump(mode="python")  # type: ignore[union-attr]
    if hasattr(expected, "model_dump"):
        expected = expected.model_dump(mode="python")  # type: ignore[union-attr]
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        if tuple(observed) != tuple(expected):
            raise ValueError(
                f"{path} seed={seed} keys observed={tuple(observed)} bound={tuple(expected)}"
            )
        for key in expected:
            _compare_identity_nested(observed[key], expected[key], path=f"{path}.{key}", seed=seed)
        return
    if isinstance(observed, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(observed) != len(expected):
            raise ValueError(
                f"{path} seed={seed} length observed={len(observed)} bound={len(expected)}"
            )
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _compare_identity_nested(left, right, path=f"{path}[{index}]", seed=seed)
        return
    if observed != expected or type(observed) is not type(expected):
        raise ValueError(f"{path} seed={seed} observed={observed!r} bound={expected!r}")


def _compare_exact_evaluation(
    observed: ExactKernelEvaluation,
    expected: ExactKernelEvaluation,
    *,
    role: str,
    target_hash: str,
    profile_index: int,
    seed: int,
) -> None:
    prefix = (
        f"{role} exact conditional seed={seed} target_hash={target_hash} "
        f"profile_index={profile_index}"
    )
    for name, horizon, observed_table, expected_table in (
        ("target", None, observed.target_conditional, expected.target_conditional),
        ("equilibrium", None, observed.equilibrium_conditional, expected.equilibrium_conditional),
        *(
            (
                "finite_horizon",
                horizon,
                observed.finite_horizon_conditionals[horizon],
                expected.finite_horizon_conditionals[horizon],
            )
            for horizon in _HORIZONS
        ),
    ):
        for context, (observed_row, expected_row) in enumerate(
            zip(observed_table, expected_table, strict=True)
        ):
            for output, (left, right) in enumerate(zip(observed_row, expected_row, strict=True)):
                horizon_text = "" if horizon is None else f" horizon={horizon}"
                _compare_float(
                    left,
                    right,
                    path=(f"{prefix} table={name} context={context}{horizon_text} output={output}"),
                    seed=seed,
                )
    for name in (
        "target_to_equilibrium_kl",
        "target_to_equilibrium_tv",
        "equilibrium_normalization_error",
        "equilibrium_minimum_probability",
    ):
        for context, (left, right) in enumerate(
            zip(getattr(observed, name), getattr(expected, name), strict=True)
        ):
            _compare_float(
                left,
                right,
                path=f"{prefix} diagnostic={name} context={context}",
                seed=seed,
            )
    for name in (
        "target_to_finite_horizon_tv",
        "finite_horizon_to_equilibrium_tv",
        "finite_horizon_normalization_error",
        "finite_horizon_minimum_probability",
    ):
        for horizon in _HORIZONS:
            for context, (left, right) in enumerate(
                zip(getattr(observed, name)[horizon], getattr(expected, name)[horizon], strict=True)
            ):
                _compare_float(
                    left,
                    right,
                    path=(f"{prefix} diagnostic={name} context={context} horizon={horizon}"),
                    seed=seed,
                )
    _compare_nested(
        observed.evidence_class,
        expected.evidence_class,
        path=f"{prefix} evidence_class",
        seed=seed,
    )


def _baseline_artifact_identity(
    pair: PairedKernelResult,
    model: PAsymSwapModelConfig,
    run: IndependentCompilerRunConfig,
) -> dict[str, Any]:
    return {
        "target_hash": pair.target_hash,
        "topology_id": model.topology_id,
        "logical_role_order": (*model.color_a_roles, *model.color_b_roles),
        "parameter_order": PARAMETER_ORDER,
        "dtype": model.exact_dtype,
        "parameters": pair.baseline.optimization.parameters,
        "beta": model.beta,
        "parameter_cap": model.parameter_cap,
        "compiler_settings": {
            "parameter_cap": model.parameter_cap,
            "maxiter": run.maxiter,
            "maxls": run.maxls,
            "ftol": run.ftol,
            "gtol": run.gtol,
            "projected_gradient_tolerance": run.projected_gradient_tolerance,
            "initializations": run.initializations,
            "context_weights": run.context_weights,
        },
    }


def _target_artifact_identity(
    pair: PairedKernelResult,
    profile: PooledContextProfileResult,
    model: PAsymSwapModelConfig,
    run: TargetContextCompilerRunConfig,
) -> dict[str, Any]:
    optimization = pair.target_context.optimization
    return {
        "identity_version": "target_context_artifact.v1",
        "target_hash": profile.target_hash,
        "profile_hash": profile.profile_hash,
        "context_weights": profile.context_weights,
        "baseline_artifact_hash": pair.baseline.optimization.artifact_hash,
        "topology_id": model.topology_id,
        "logical_role_order": (*model.color_a_roles, *model.color_b_roles),
        "parameter_order": PARAMETER_ORDER,
        "dtype": model.exact_dtype,
        "parameters": optimization.parameters,
        "beta": model.beta,
        "parameter_cap": model.parameter_cap,
        "compiler_settings": {
            "optimizer": run.optimizer,
            "maxiter": run.maxiter,
            "maxls": run.maxls,
            "ftol": run.ftol,
            "gtol": run.gtol,
            "projected_gradient_tolerance": run.projected_gradient_tolerance,
            "start_roles": TARGET_CONTEXT_START_ROLES,
            "start_values": optimization.start_values,
            "restart_selection": run.restart_selection,
        },
    }


def _regenerate_optimizer_results(
    pair: PairedKernelResult,
    profile: PooledContextProfileResult,
    model: PAsymSwapModelConfig,
    baseline_run: IndependentCompilerRunConfig,
    run: TargetContextCompilerRunConfig,
    *,
    target: Sequence[Sequence[float]],
) -> tuple[KernelOptimizationResult, TargetContextOptimizationResult]:
    """Re-evaluate optimizer endpoints and select winners from fresh analytic facts."""

    target_array = np.asarray(target, dtype=np.float64)

    def facts(
        parameters: Sequence[float], weights: Sequence[float], tolerance: float
    ) -> tuple[float, float, float, int, bool]:
        parameter_array = np.asarray(parameters, dtype=np.float64)
        objective, gradient = loss_and_gradient(
            parameter_array, target_array, np.asarray(weights, dtype=np.float64)
        )
        raw_norm = float(np.max(np.abs(gradient)))
        projected_norm = float(
            np.max(np.abs(project_gradient(parameter_array, gradient, model.parameter_cap)))
        )
        cap_count = sum(abs(value) >= model.parameter_cap for value in parameters)
        finite = bool(
            np.all(np.isfinite(parameter_array))
            and math.isfinite(objective)
            and math.isfinite(raw_norm)
            and math.isfinite(projected_norm)
        )
        passed = (
            finite
            and float(np.max(np.abs(parameter_array))) <= model.parameter_cap
            and projected_norm <= tolerance
        )
        return objective, raw_norm, projected_norm, cap_count, passed

    baseline_attempts = tuple(
        KernelOptimizationAttemptResult(
            restart_index=index,
            parameters=attempt.parameters,
            objective=(
                evaluated := facts(
                    attempt.parameters,
                    baseline_run.context_weights,
                    baseline_run.projected_gradient_tolerance,
                )
            )[0],
            raw_gradient_norm=evaluated[1],
            projected_gradient_norm=evaluated[2],
            scipy_success=attempt.scipy_success,
            passed_checks=attempt.scipy_success and evaluated[4],
            iterations=attempt.iterations,
            termination=attempt.termination,
            cap_active_parameter_count=evaluated[3],
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        )
        for index, attempt in enumerate(pair.baseline.optimization.attempts)
    )
    baseline_passing = tuple(attempt for attempt in baseline_attempts if attempt.passed_checks)
    if not baseline_passing:
        raise ValueError(
            f"optimizer target_hash={profile.target_hash} role=baseline observed=0 bound=1"
        )
    baseline_winner = min(
        baseline_passing, key=lambda attempt: (attempt.objective, attempt.parameters)
    )
    baseline = KernelOptimizationResult(
        artifact_hash=pair.baseline.optimization.artifact_hash,
        parameters=baseline_winner.parameters,
        selected_restart=baseline_winner.restart_index,
        successful_restart_count=len(baseline_passing),
        objective=baseline_winner.objective,
        projected_gradient_norm=baseline_winner.projected_gradient_norm,
        cap_active_parameter_count=baseline_winner.cap_active_parameter_count,
        attempts=baseline_attempts,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )

    target_attempts = tuple(
        TargetContextOptimizationAttemptResult(
            start_index=index,
            start_role=TARGET_CONTEXT_START_ROLES[index],
            parameters=attempt.parameters,
            objective=(
                evaluated := facts(
                    attempt.parameters,
                    profile.context_weights,
                    run.projected_gradient_tolerance,
                )
            )[0],
            raw_gradient_norm=evaluated[1],
            projected_gradient_norm=evaluated[2],
            scipy_success=attempt.scipy_success,
            passed_checks=attempt.scipy_success and evaluated[4],
            iterations=attempt.iterations,
            termination=attempt.termination,
            cap_active_parameter_count=evaluated[3],
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        )
        for index, attempt in enumerate(pair.target_context.optimization.attempts)
    )
    target_passing = tuple(attempt for attempt in target_attempts if attempt.passed_checks)
    if not target_passing:
        raise ValueError(
            f"optimizer target_hash={profile.target_hash} role=target_context observed=0 bound=1"
        )
    target_winner = min(target_passing, key=lambda attempt: (attempt.objective, attempt.parameters))
    target_result = TargetContextOptimizationResult(
        artifact_hash=pair.target_context.optimization.artifact_hash,
        start_values=(baseline.parameters, *_FIXED_STARTS),
        parameters=target_winner.parameters,
        selected_start_index=target_winner.start_index,
        selected_start_role=target_winner.start_role,
        successful_attempt_count=len(target_passing),
        objective=target_winner.objective,
        projected_gradient_norm=target_winner.projected_gradient_norm,
        cap_active_parameter_count=target_winner.cap_active_parameter_count,
        attempts=target_attempts,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )
    return baseline, target_result


def _regenerate_pair_from_frozen_parameters(
    pair: PairedKernelResult,
    profile: PooledContextProfileResult,
    model: PAsymSwapModelConfig,
    run: TargetContextCompilerRunConfig,
    *,
    target: Sequence[Sequence[float]],
    baseline_run: IndependentCompilerRunConfig | None = None,
    baseline_request_hash: str | None = None,
    target_request_hash: str | None = None,
) -> PairedKernelResult:
    """Build a fresh pair source of truth without reusing persisted derived values."""

    if baseline_run is None:
        independent_config = load_experiment_config(
            experiment_config_path("thrml-independent-pasym-swap.toml")
        )
        baseline_run = IndependentCompilerRunConfig.model_validate(
            to_json_value(independent_config.run_parameters)
        )
    canonical_baseline, canonical_target = _regenerate_optimizer_results(
        pair, profile, model, baseline_run, run, target=target
    )
    baseline_exact = derive_exact_kernel_evaluation(
        canonical_baseline.parameters, target, beta=model.beta
    )
    target_exact = derive_exact_kernel_evaluation(
        canonical_target.parameters, target, beta=model.beta
    )
    sampled = derive_sampled_k30_evaluation(
        pair.target_context.sampled_k30.counts,
        target_exact.finite_horizon_conditionals[30],
        chain_count=run.chain_count_per_context,
    )
    baseline = BaselineKernelResult(
        target_hash=profile.target_hash,
        baseline_compiler_request_hash=(
            pair.baseline.baseline_compiler_request_hash
            if baseline_request_hash is None
            else baseline_request_hash
        ),
        optimization=canonical_baseline,
        exact=baseline_exact,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )
    target_context = TargetContextKernelResult(
        target_hash=profile.target_hash,
        profile_hash=profile.profile_hash,
        target_compiler_request_hash=(
            pair.target_context.target_compiler_request_hash
            if target_request_hash is None
            else target_request_hash
        ),
        baseline_artifact_hash=canonical_baseline.artifact_hash,
        optimization=canonical_target,
        exact=target_exact,
        sampled_k30=sampled,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )
    regenerated = PairedKernelResult(
        target_hash=profile.target_hash,
        profile_hash=profile.profile_hash,
        baseline=baseline,
        target_context=target_context,
        metrics=pair.metrics,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )
    baseline_hash = canonical_sha256(_baseline_artifact_identity(regenerated, model, baseline_run))
    canonical_baseline = canonical_baseline.model_copy(update={"artifact_hash": baseline_hash})
    baseline = baseline.model_copy(update={"optimization": canonical_baseline})
    target_context = target_context.model_copy(update={"baseline_artifact_hash": baseline_hash})
    regenerated = regenerated.model_copy(
        update={"baseline": baseline, "target_context": target_context}
    )
    regenerated = regenerated.model_copy(
        update={"metrics": derive_paired_profile_metrics(regenerated, profile)}
    )
    target_hash = canonical_sha256(_target_artifact_identity(regenerated, profile, model, run))
    target_context = regenerated.target_context.model_copy(
        update={
            "optimization": regenerated.target_context.optimization.model_copy(
                update={"artifact_hash": target_hash}
            )
        }
    )
    return regenerated.model_copy(update={"target_context": target_context})


def _validate_optimizer_endpoints(
    pair: PairedKernelResult,
    canonical_pair: PairedKernelResult,
    profile_index: int,
    seed: int,
) -> None:
    configurations = (
        (
            "baseline",
            pair.baseline.optimization,
            canonical_pair.baseline.optimization,
            "selected_restart",
            "successful_restart_count",
        ),
        (
            "target_context",
            pair.target_context.optimization,
            canonical_pair.target_context.optimization,
            "selected_start_index",
            "successful_attempt_count",
        ),
    )
    for role, optimization, canonical, winner_field, count_field in configurations:
        for attempt_index, (attempt, expected) in enumerate(
            zip(optimization.attempts, canonical.attempts, strict=True)
        ):
            prefix = (
                f"seed={seed} target_hash={pair.target_hash} profile_index={profile_index} "
                f"role={role} attempt={attempt_index}"
            )
            for field, name in (
                ("objective", "attempt objective"),
                ("raw_gradient_norm", "raw gradient"),
                ("projected_gradient_norm", "projected gradient"),
                ("cap_active_parameter_count", "cap-active count"),
                ("passed_checks", "attempt passed checks"),
            ):
                _compare_nested(
                    getattr(attempt, field),
                    getattr(expected, field),
                    path=f"{name} {prefix}",
                    seed=seed,
                )
        prefix = (
            f"seed={seed} target_hash={pair.target_hash} profile_index={profile_index} role={role}"
        )
        _compare_nested(
            getattr(optimization, count_field),
            getattr(canonical, count_field),
            path=f"successful attempt count {prefix}",
            seed=seed,
        )
        _compare_nested(
            getattr(optimization, winner_field),
            getattr(canonical, winner_field),
            path=f"selected winner {prefix}",
            seed=seed,
        )
        if role == "target_context":
            _compare_identity_nested(
                optimization.selected_start_role,
                canonical.selected_start_role,
                path=f"selected start role {prefix}",
                seed=seed,
            )
            _compare_identity_nested(
                optimization.start_values,
                canonical.start_values,
                path=f"checked start values {prefix}",
                seed=seed,
            )
        for field in (
            "parameters",
            "objective",
            "projected_gradient_norm",
            "cap_active_parameter_count",
        ):
            _compare_identity_nested(
                getattr(optimization, field),
                getattr(canonical, field),
                path=f"selected {field} target_hash={pair.target_hash} role={role}",
                seed=seed,
            )


def _deterministic_pair_projection(pair: PairedKernelResult) -> dict[str, Any]:
    return {
        "target_hash": pair.target_hash,
        "profile_hash": pair.profile_hash,
        "baseline": {
            "target_hash": pair.baseline.target_hash,
            "baseline_compiler_request_hash": pair.baseline.baseline_compiler_request_hash,
            "optimization": pair.baseline.optimization.model_dump(mode="json"),
            "exact": pair.baseline.exact.model_dump(mode="json"),
            "evidence_class": pair.baseline.evidence_class,
        },
        "target_context": {
            "target_hash": pair.target_context.target_hash,
            "profile_hash": pair.target_context.profile_hash,
            "target_compiler_request_hash": pair.target_context.target_compiler_request_hash,
            "baseline_artifact_hash": pair.target_context.baseline_artifact_hash,
            "optimization": pair.target_context.optimization.model_dump(mode="json"),
            "exact": pair.target_context.exact.model_dump(mode="json"),
            "evidence_class": pair.target_context.evidence_class,
        },
        "metrics": pair.metrics.model_dump(mode="json"),
        "evidence_class": pair.evidence_class,
    }


def target_context_deterministic_projection(
    summary: TargetContextPAsymSwapSummary,
) -> dict[str, Any]:
    """Return the fixed semantic projection shared by persistence and aggregation."""

    return {
        "identity_version": "target_context_deterministic_result.v1",
        "initial_state": summary.initial_state.model_dump(mode="json"),
        "trace": [item.model_dump(mode="json") for item in summary.trace],
        "trace_hash": summary.trace_hash,
        "profiles": [item.model_dump(mode="json") for item in summary.profiles],
        "occurrence_mapping": [item.model_dump(mode="json") for item in summary.occurrence_mapping],
        "pairs": [_deterministic_pair_projection(pair) for pair in summary.pairs],
        "schedule_metrics": summary.schedule_metrics.model_dump(mode="json"),
        "deterministic_acceptance": summary.deterministic_acceptance.model_dump(mode="json"),
        "all_context_degradation": summary.all_context_degradation.model_dump(mode="json"),
        "zero_support_assessment": summary.zero_support_assessment.model_dump(mode="json"),
    }


def target_context_deterministic_result_hash(summary: TargetContextPAsymSwapSummary) -> str:
    """Hash exactly the deterministic target-context evidence projection."""

    return canonical_sha256(target_context_deterministic_projection(summary))


def _persisted_trace_identity(summary: TargetContextPAsymSwapSummary) -> dict[str, Any]:
    return {
        "identity_version": "target_context_trace.v1",
        "source_reference": summary.source_reference,
        "word_order": WORD_ORDER,
        "initial_state": summary.initial_state.initial_state,
        "initial_particle_site": summary.initial_state.initial_particle_site,
        "initial_occupancy_order": summary.initial_state.initial_occupancy_order,
        "initial_occupancy": summary.initial_state.initial_occupancy,
        "context_source": summary.context_source,
        "zero_support_policy": summary.zero_support_policy,
        "occurrences": tuple(
            {
                "occurrence_index": item.occurrence_index,
                "macrostep": item.macrostep,
                "layer": item.layer,
                "color": item.color,
                "edge": item.edge,
                "target_hash": item.target_hash,
                "context_weights": item.context_weights,
            }
            for item in summary.trace
        ),
    }


def _persisted_profile_identity(profile: PooledContextProfileResult) -> dict[str, Any]:
    return {
        "identity_version": "target_context_profile.v1",
        "trace_hash": profile.trace_hash,
        "target_hash": profile.target_hash,
        "word_order": WORD_ORDER,
        "context_reduction": profile.context_reduction,
        "zero_support_policy": profile.zero_support_policy,
        "occurrence_indices": profile.occurrence_indices,
        "multiplicity": profile.multiplicity,
        "context_weights": profile.context_weights,
        "support_mask": profile.support_mask,
    }


def _strict_validation_context(error: ValidationError) -> tuple[str, str, str]:
    errors = error.errors()
    if not errors:
        return "record", "invalid", "strict validation failed"
    first = errors[0]
    location = ".".join(str(item) for item in first.get("loc", ())) or "record"
    observed = first.get("input", "invalid")
    rendered = repr(observed)
    if len(rendered) > 160:
        rendered = rendered[:157] + "..."
    reason = str(first.get("msg", "strict validation failed"))
    return location, rendered, reason


def deep_validate_target_context_pasym_swap_summary(
    summary: TargetContextPAsymSwapSummary | Mapping[str, Any] | str,
    model: PAsymSwapModelConfig,
    run: TargetContextCompilerRunConfig,
    seed: int,
) -> TargetContextPAsymSwapSummary:
    """Regenerate all deterministic evidence without optimization, THRML, or a backend."""

    validate_target_context_pasym_swap_request(model, run, seed)
    independent_config = load_experiment_config(
        experiment_config_path("thrml-independent-pasym-swap.toml")
    )
    authoritative_model = PAsymSwapModelConfig.model_validate(
        to_json_value(independent_config.model_parameters)
    )
    if model.model_dump(mode="json") != authoritative_model.model_dump(mode="json"):
        raise ValueError(
            f"authoritative model seed={seed} observed={model} bound={authoritative_model}"
        )
    baseline_run = IndependentCompilerRunConfig.model_validate(
        to_json_value(independent_config.run_parameters)
    )
    baseline_request_hash = independent_pasym_swap_non_seed_config_hash(model, baseline_run)
    target_request_hash = target_context_pasym_swap_non_seed_config_hash(model, run)

    if isinstance(summary, str):
        try:
            checked = TargetContextPAsymSwapSummary.model_validate_json(summary)
        except ValidationError as error:
            location, observed, reason = _strict_validation_context(error)
            raise ValueError(
                f"target-context summary seed={seed} component={location} "
                f"reason={reason} observed={observed} bound=valid strict record"
            ) from error
    elif isinstance(summary, TargetContextPAsymSwapSummary):
        checked = summary
    else:
        try:
            checked = TargetContextPAsymSwapSummary.model_validate(summary)
        except Exception as error:
            errors = getattr(error, "errors", lambda: ())()
            location = errors[0].get("loc", ()) if errors else ()
            if len(location) >= 2 and location[0] == "profiles":
                raise ValueError(
                    f"profile profile_index={location[1]} seed={seed} "
                    "observed=invalid bound=canonical"
                ) from error
            message = str(error)
            for key, component, marker in (
                ("trace", "trace", "trace must use occurrence indices"),
                (
                    "occurrence_mapping",
                    "occurrence mapping",
                    "occurrence mapping must use occurrence indices",
                ),
            ):
                records = summary.get(key)
                if marker in message and isinstance(records, (tuple, list)):
                    for index, record in enumerate(records):
                        observed_index = (
                            record.get("occurrence_index")
                            if isinstance(record, Mapping)
                            else record
                        )
                        if observed_index != index:
                            raise ValueError(
                                f"{component} occurrence_index={index} seed={seed} "
                                f"observed={observed_index} bound={index}"
                            ) from error
            raw_pairs_value = summary.get("pairs", ())
            raw_mappings_value = summary.get("occurrence_mapping", ())
            raw_pairs = (
                tuple(item for item in raw_pairs_value if isinstance(item, Mapping))
                if isinstance(raw_pairs_value, (tuple, list))
                else ()
            )
            raw_mappings = (
                tuple(item for item in raw_mappings_value if isinstance(item, Mapping))
                if isinstance(raw_mappings_value, (tuple, list))
                else ()
            )
            for profile_index, pair in enumerate(raw_pairs):
                target_hash = pair.get("target_hash")
                target = pair.get("target_context", {})
                baseline = pair.get("baseline", {})
                if not isinstance(target, Mapping) or not isinstance(baseline, Mapping):
                    continue
                baseline_optimization = baseline.get("optimization", {})
                target_optimization = target.get("optimization", {})
                if not isinstance(baseline_optimization, Mapping) or not isinstance(
                    target_optimization, Mapping
                ):
                    continue
                baseline_hash = baseline_optimization.get("artifact_hash")
                target_hash_value = target_optimization.get("artifact_hash")
                mapped = tuple(
                    item for item in raw_mappings if item.get("target_hash") == target_hash
                )
                baseline_links = {item.get("baseline_artifact_hash") for item in mapped}
                target_links = {item.get("target_context_artifact_hash") for item in mapped}
                if (
                    baseline_hash != target.get("baseline_artifact_hash")
                    and len(baseline_links) == 1
                    and baseline_hash not in baseline_links
                ):
                    raise ValueError(
                        f"baseline artifact seed={seed} target_hash={target_hash} "
                        f"profile_index={profile_index} observed={baseline_hash} "
                        f"bound={next(iter(baseline_links))}"
                    ) from error
                if len(target_links) == 1 and target_hash_value not in target_links:
                    raise ValueError(
                        f"target_context artifact seed={seed} target_hash={target_hash} "
                        f"profile_index={profile_index} observed={target_hash_value} "
                        f"bound={next(iter(target_links))}"
                    ) from error
            if "occurrence mapping must resolve" in message:
                pairs = {pair.get("target_hash"): pair for pair in raw_pairs}
                mismatch = 0
                for index, mapping in enumerate(raw_mappings):
                    pair = pairs.get(mapping.get("target_hash"))
                    baseline = None if pair is None else pair.get("baseline")
                    target = None if pair is None else pair.get("target_context")
                    baseline_optimization = (
                        baseline.get("optimization") if isinstance(baseline, Mapping) else None
                    )
                    target_optimization = (
                        target.get("optimization") if isinstance(target, Mapping) else None
                    )
                    expected_baseline = (
                        baseline_optimization.get("artifact_hash")
                        if isinstance(baseline_optimization, Mapping)
                        else None
                    )
                    expected_target = (
                        target_optimization.get("artifact_hash")
                        if isinstance(target_optimization, Mapping)
                        else None
                    )
                    if (
                        pair is None
                        or mapping.get("profile_hash") != pair.get("profile_hash")
                        or mapping.get("baseline_artifact_hash") != expected_baseline
                        or mapping.get("target_context_artifact_hash") != expected_target
                    ):
                        mismatch = index
                        break
                raise ValueError(
                    f"occurrence mapping occurrence_index={mismatch} seed={seed} "
                    f"observed={raw_mappings[mismatch]} bound=paired artifacts"
                ) from error
            if isinstance(error, ValidationError):
                location_text, observed_text, reason = _strict_validation_context(error)
                raise ValueError(
                    f"target-context summary seed={seed} component={location_text} "
                    f"reason={reason} observed={observed_text} bound=valid strict record"
                ) from error
            raise

    _compare_nested(
        checked.baseline_compiler_request_hash,
        baseline_request_hash,
        path="baseline compiler request hash",
        seed=seed,
    )
    _compare_nested(
        checked.target_compiler_request_hash,
        target_request_hash,
        path="target compiler request hash",
        seed=seed,
    )

    fixture = build_paper_fixture()
    expected_trace = derive_target_context_trace(
        fixture,
        initial_state=run.initial_state,
        initial_particle_site=run.initial_particle_site,
        initial_occupancy=run.initial_occupancy,
        context_source=run.context_source,
        zero_support_policy=run.zero_support_policy,
    )
    expected_profiles = pool_target_context_profiles(
        expected_trace, context_reduction=run.context_reduction
    )
    expected_initial = {
        "initial_state": expected_trace.initial_state,
        "initial_particle_site": expected_trace.initial_particle_site,
        "initial_occupancy_order": expected_trace.initial_occupancy_order,
        "initial_occupancy": expected_trace.initial_occupancy,
        "evidence_class": EvidenceClass.EXACT_REFERENCE,
    }
    _compare_nested(
        checked.initial_state,
        expected_initial,
        path="initial state",
        seed=seed,
    )
    persisted_trace_hash = canonical_sha256(_persisted_trace_identity(checked))
    if checked.trace_hash != persisted_trace_hash:
        raise ValueError(
            f"trace identity seed={seed} observed={checked.trace_hash} bound={persisted_trace_hash}"
        )
    for index, (observed, expected) in enumerate(
        zip(checked.trace, expected_trace.occurrences, strict=True)
    ):
        expected_row = {
            **expected.__dict__,
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        }
        _compare_identity_nested(
            observed,
            expected_row,
            path=f"trace occurrence_index={index}",
            seed=seed,
        )
    _compare_nested(checked.trace_hash, expected_trace.trace_hash, path="trace hash", seed=seed)
    for profile_index, (observed, expected) in enumerate(
        zip(checked.profiles, expected_profiles, strict=True)
    ):
        persisted_profile_hash = canonical_sha256(_persisted_profile_identity(observed))
        if observed.profile_hash != persisted_profile_hash:
            raise ValueError(
                f"profile identity seed={seed} target_hash={observed.target_hash} "
                f"profile_index={profile_index} observed={observed.profile_hash} "
                f"bound={persisted_profile_hash}"
            )
        expected_row = {
            "trace_hash": expected.trace_hash,
            "target_hash": expected.target_hash,
            "context_reduction": expected.context_reduction,
            "zero_support_policy": expected.zero_support_policy,
            "occurrence_indices": expected.occurrence_indices,
            "multiplicity": expected.multiplicity,
            "context_weights": expected.context_weights,
            "support_mask": expected.support_mask,
            "profile_hash": expected.profile_hash,
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        }
        _compare_identity_nested(
            observed,
            expected_row,
            path=f"profile profile_index={profile_index} target_hash={expected.target_hash}",
            seed=seed,
        )

    profile_by_target = {profile.target_hash: profile for profile in checked.profiles}
    pair_by_target = {pair.target_hash: pair for pair in checked.pairs}
    for occurrence_index, (mapping, occurrence) in enumerate(
        zip(checked.occurrence_mapping, checked.trace, strict=True)
    ):
        profile = profile_by_target[occurrence.target_hash]
        pair = pair_by_target[occurrence.target_hash]
        expected_mapping = OccurrenceArtifactMappingResult(
            occurrence_index=occurrence_index,
            target_hash=occurrence.target_hash,
            profile_hash=profile.profile_hash,
            baseline_artifact_hash=pair.baseline.optimization.artifact_hash,
            target_context_artifact_hash=pair.target_context.optimization.artifact_hash,
        )
        _compare_nested(
            mapping,
            expected_mapping,
            path=f"occurrence mapping occurrence_index={occurrence_index}",
            seed=seed,
        )

    fixture_targets = {target.target_hash: target for target in fixture.targets}
    derived_pairs: list[PairedKernelResult] = []
    for profile_index, (pair, profile) in enumerate(
        zip(checked.pairs, checked.profiles, strict=True)
    ):
        context = f"seed={seed} target_hash={pair.target_hash} profile_index={profile_index}"
        expected_target = fixture_targets[pair.target_hash].conditional
        _compare_nested(
            pair.baseline.exact.target_conditional,
            expected_target,
            path=f"baseline target table {context}",
            seed=seed,
        )
        _compare_nested(
            pair.target_context.exact.target_conditional,
            expected_target,
            path=f"target-context target table {context}",
            seed=seed,
        )
        canonical_pair = _regenerate_pair_from_frozen_parameters(
            pair,
            profile,
            model,
            run,
            target=expected_target,
            baseline_run=baseline_run,
            baseline_request_hash=baseline_request_hash,
            target_request_hash=target_request_hash,
        )
        _validate_optimizer_endpoints(pair, canonical_pair, profile_index, seed)
        _compare_nested(
            pair.baseline.optimization.artifact_hash,
            canonical_pair.baseline.optimization.artifact_hash,
            path=f"baseline artifact identity {context}",
            seed=seed,
        )
        _compare_nested(
            pair.target_context.optimization.artifact_hash,
            canonical_pair.target_context.optimization.artifact_hash,
            path=f"target artifact identity {context}",
            seed=seed,
        )
        _compare_exact_evaluation(
            pair.baseline.exact,
            canonical_pair.baseline.exact,
            role="baseline",
            target_hash=pair.target_hash,
            profile_index=profile_index,
            seed=seed,
        )
        _compare_exact_evaluation(
            pair.target_context.exact,
            canonical_pair.target_context.exact,
            role="target",
            target_hash=pair.target_hash,
            profile_index=profile_index,
            seed=seed,
        )
        _compare_nested(
            pair.target_context.sampled_k30,
            canonical_pair.target_context.sampled_k30,
            path=f"sampled K30 {context} horizon=30",
            seed=seed,
        )
        _compare_identity_nested(
            pair.metrics.support_mask,
            canonical_pair.metrics.support_mask,
            path=f"paired metrics {context}.support_mask",
            seed=seed,
        )
        _compare_identity_nested(
            pair.metrics.context_weights,
            canonical_pair.metrics.context_weights,
            path=f"paired metrics {context}.context_weights",
            seed=seed,
        )
        expected_metrics = canonical_pair.metrics
        _compare_nested(
            pair.metrics,
            expected_metrics,
            path=f"paired metrics {context}",
            seed=seed,
        )
        derived_pairs.append(canonical_pair)

    expected_schedule = derive_schedule_metrics(derived_pairs)
    expected_degradation = derive_all_context_degradation(derived_pairs)
    expected_zero = derive_zero_support_assessment(derived_pairs)
    expected_deterministic = derive_deterministic_acceptance(
        derived_pairs, expected_schedule, expected_degradation, run
    )
    expected_sampled_fidelity = derive_sampled_fidelity(derived_pairs, run)
    expected_seed = derive_seed_acceptance(expected_deterministic, expected_sampled_fidelity)
    for path, observed, expected in (
        ("schedule metrics", checked.schedule_metrics, expected_schedule),
        ("all-context degradation", checked.all_context_degradation, expected_degradation),
        ("zero-support assessment", checked.zero_support_assessment, expected_zero),
        ("deterministic acceptance", checked.deterministic_acceptance, expected_deterministic),
        ("sampled fidelity", checked.sampled_fidelity, expected_sampled_fidelity),
        ("seed acceptance", checked.seed_acceptance, expected_seed),
    ):
        _compare_nested(observed, expected, path=path, seed=seed)
    expected_result_hash = target_context_deterministic_result_hash(checked)
    _compare_nested(
        checked.deterministic_result_hash,
        expected_result_hash,
        path="deterministic result hash",
        seed=seed,
    )
    return checked


def _require_target_context_metric(
    metrics: Mapping[str, MetricObservation],
    name: str,
    expected: float | bool,
    *,
    evidence_class: EvidenceClass,
    unit: str | None,
    method: str,
    source: str,
) -> None:
    metric = metrics[name]
    expected_type = bool if type(expected) is bool else float
    if type(metric.value) is not expected_type:
        observed_type = {
            bool: "boolean",
            int: "integer",
        }.get(type(metric.value), type(metric.value).__name__)
        raise ValueError(
            f"metric {name!r} value type must be {expected_type.__name__}; observed={observed_type}"
        )
    if metric.value != expected:
        raise ValueError(
            f"metric {name!r} value does not match regenerated nested evidence: "
            f"observed={metric.value} bound={expected}"
        )
    if metric.evidence_class is not evidence_class:
        raise ValueError(
            f"metric {name!r} evidence_class observed={metric.evidence_class.value!r} "
            f"bound={evidence_class.value!r}"
        )
    if metric.source != source:
        raise ValueError(f"metric {name!r} source observed={metric.source!r} bound={source!r}")
    if metric.method != method:
        raise ValueError(f"metric {name!r} method observed={metric.method!r} bound={method!r}")
    if metric.unit != unit:
        raise ValueError(f"metric {name!r} unit observed={metric.unit!r} bound={unit!r}")


def validate_target_context_pasym_swap_observations(
    metrics: Mapping[str, MetricObservation],
    model: PAsymSwapModelConfig,
    run: TargetContextCompilerRunConfig,
    seed: int,
) -> TargetContextPAsymSwapSummary:
    """Deeply replay a composite metric, then bind its exact scalar envelope."""

    observed_keys = frozenset(metrics)
    if observed_keys != _REQUIRED_METRICS:
        missing = sorted(_REQUIRED_METRICS - observed_keys)
        extra = sorted(observed_keys - _REQUIRED_METRICS)
        raise ValueError(
            "target-context PAsymSwap metrics must contain exactly the required keys: "
            f"missing={missing} extra={extra}"
        )
    checked_metrics = {
        name: value
        if isinstance(value, MetricObservation)
        else MetricObservation.model_validate(to_json_value(value))
        for name, value in metrics.items()
    }
    checked_model = PAsymSwapModelConfig.model_validate(to_json_value(model))
    checked_run = TargetContextCompilerRunConfig.model_validate(to_json_value(run))

    summary_metric = checked_metrics["target_context_pasym_swap"]
    if summary_metric.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError(
            "metric 'target_context_pasym_swap' evidence_class must be software_simulation"
        )
    if summary_metric.source != PAPER_SOURCE:
        raise ValueError(
            "metric 'target_context_pasym_swap' source must be the scientific paper URL"
        )
    if summary_metric.method != _SUMMARY_METHOD:
        raise ValueError(
            "metric 'target_context_pasym_swap' method "
            f"observed={summary_metric.method!r} bound={_SUMMARY_METHOD!r}"
        )
    if summary_metric.unit is not None:
        raise ValueError(
            f"metric 'target_context_pasym_swap' unit observed={summary_metric.unit!r} bound=None"
        )
    summary_json = json.dumps(to_json_value(summary_metric.value), allow_nan=False)
    regenerated = deep_validate_target_context_pasym_swap_summary(
        summary_json, checked_model, checked_run, seed
    )

    schedule = regenerated.schedule_metrics
    exact_values = {
        "baseline_occurrence_weighted_equilibrium_kl": (
            schedule.baseline_occurrence_weighted_equilibrium_kl,
            "nats",
        ),
        "target_context_occurrence_weighted_equilibrium_kl": (
            schedule.target_context_occurrence_weighted_equilibrium_kl,
            "nats",
        ),
        "occurrence_weighted_equilibrium_kl_improvement": (
            schedule.occurrence_weighted_equilibrium_kl_improvement,
            "nats",
        ),
        "baseline_occurrence_weighted_equilibrium_tv": (
            schedule.baseline_occurrence_weighted_equilibrium_tv,
            None,
        ),
        "target_context_occurrence_weighted_equilibrium_tv": (
            schedule.target_context_occurrence_weighted_equilibrium_tv,
            None,
        ),
        "maximum_paired_k30_equilibrium_residual": (
            schedule.maximum_paired_k30_equilibrium_residual,
            None,
        ),
    }
    for name, (value, unit) in exact_values.items():
        _require_target_context_metric(
            checked_metrics,
            name,
            value,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            unit=unit,
            method=_EXACT_METHOD,
            source=PAPER_SOURCE,
        )
    _require_target_context_metric(
        checked_metrics,
        "maximum_empirical_k30_residual",
        regenerated.sampled_fidelity.maximum_empirical_k30_residual,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        unit=None,
        method=_SAMPLE_METHOD,
        source=PAPER_SOURCE,
    )
    _require_target_context_metric(
        checked_metrics,
        "acceptance_passed",
        regenerated.seed_acceptance.passed,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        unit=None,
        method=_ACCEPTANCE_METHOD,
        source=PAPER_SOURCE,
    )
    _require_target_context_metric(
        checked_metrics,
        "baseline_optimizer_seconds",
        regenerated.baseline_optimizer_phase.seconds,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        unit="seconds",
        method=_BASELINE_OPTIMIZER_METHOD,
        source=RUN_TIMING_SOURCE,
    )
    _require_target_context_metric(
        checked_metrics,
        "target_context_optimizer_seconds",
        regenerated.target_context_optimizer_phase.seconds,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        unit="seconds",
        method=_TARGET_OPTIMIZER_METHOD,
        source=RUN_TIMING_SOURCE,
    )
    return regenerated


def build_target_context_pasym_swap_summary(
    *,
    pairs: Sequence[PairedKernelResult],
    model: PAsymSwapModelConfig,
    run: TargetContextCompilerRunConfig,
    seed: int,
    baseline_optimizer_phase: OptimizerPhaseResult,
    target_context_optimizer_phase: OptimizerPhaseResult,
) -> TargetContextPAsymSwapSummary:
    """Assemble a canonical summary and immediately deep-regenerate its evidence."""

    validate_target_context_pasym_swap_request(model, run, seed)
    independent_config = load_experiment_config(
        experiment_config_path("thrml-independent-pasym-swap.toml")
    )
    authoritative_model = PAsymSwapModelConfig.model_validate(
        to_json_value(independent_config.model_parameters)
    )
    if model.model_dump(mode="json") != authoritative_model.model_dump(mode="json"):
        raise ValueError(
            f"authoritative model seed={seed} observed={model} bound={authoritative_model}"
        )
    baseline_run = IndependentCompilerRunConfig.model_validate(
        to_json_value(independent_config.run_parameters)
    )
    baseline_request_hash = independent_pasym_swap_non_seed_config_hash(model, baseline_run)
    target_request_hash = target_context_pasym_swap_non_seed_config_hash(model, run)

    fixture = build_paper_fixture()
    trace = derive_target_context_trace(
        fixture,
        initial_state=run.initial_state,
        initial_particle_site=run.initial_particle_site,
        initial_occupancy=run.initial_occupancy,
        context_source=run.context_source,
        zero_support_policy=run.zero_support_policy,
    )
    pooled = pool_target_context_profiles(trace, context_reduction=run.context_reduction)
    trace_rows = tuple(
        OccurrenceContextResult(
            occurrence_index=item.occurrence_index,
            macrostep=item.macrostep,
            layer=item.layer,
            color=item.color,  # type: ignore[arg-type]
            edge=item.edge,
            target_hash=item.target_hash,
            context_weights=item.context_weights,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
        )
        for item in trace.occurrences
    )
    profiles = tuple(
        PooledContextProfileResult(
            trace_hash=item.trace_hash,
            target_hash=item.target_hash,
            context_reduction="equal_occurrence_mean_by_target_hash",
            zero_support_policy="exact_unsmoothed",
            occurrence_indices=item.occurrence_indices,
            multiplicity=item.multiplicity,
            context_weights=item.context_weights,
            support_mask=item.support_mask,
            profile_hash=item.profile_hash,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
        )
        for item in pooled
    )
    input_pairs = {pair.target_hash: pair for pair in pairs}
    if len(input_pairs) != 37 or set(input_pairs) != {profile.target_hash for profile in profiles}:
        raise ValueError(
            f"pair collection seed={seed} observed={tuple(sorted(input_pairs))} "
            f"bound={tuple(profile.target_hash for profile in profiles)}"
        )
    canonical_pairs: list[PairedKernelResult] = []
    targets_by_hash = {target.target_hash: target.conditional for target in fixture.targets}
    for profile in profiles:
        pair = input_pairs[profile.target_hash]
        canonical_pairs.append(
            _regenerate_pair_from_frozen_parameters(
                pair,
                profile,
                model,
                run,
                target=targets_by_hash[profile.target_hash],
                baseline_run=baseline_run,
                baseline_request_hash=baseline_request_hash,
                target_request_hash=target_request_hash,
            )
        )
    pair_tuple = tuple(canonical_pairs)
    profiles_by_target = {profile.target_hash: profile for profile in profiles}
    pairs_by_target = {pair.target_hash: pair for pair in pair_tuple}
    mappings = tuple(
        OccurrenceArtifactMappingResult(
            occurrence_index=occurrence.occurrence_index,
            target_hash=occurrence.target_hash,
            profile_hash=profiles_by_target[occurrence.target_hash].profile_hash,
            baseline_artifact_hash=pairs_by_target[
                occurrence.target_hash
            ].baseline.optimization.artifact_hash,
            target_context_artifact_hash=pairs_by_target[
                occurrence.target_hash
            ].target_context.optimization.artifact_hash,
        )
        for occurrence in trace_rows
    )
    schedule = derive_schedule_metrics(pair_tuple)
    degradation = derive_all_context_degradation(pair_tuple)
    zero_support = derive_zero_support_assessment(pair_tuple)
    deterministic = derive_deterministic_acceptance(pair_tuple, schedule, degradation, run)
    sampled = derive_sampled_fidelity(pair_tuple, run)
    seed_acceptance = derive_seed_acceptance(deterministic, sampled)
    summary = TargetContextPAsymSwapSummary(
        source_reference=PAPER_SOURCE,
        target_compiler_request_hash=target_request_hash,
        baseline_compiler_request_hash=baseline_request_hash,
        initial_state=TargetContextInitialState(
            initial_state="single_particle",
            initial_particle_site=trace.initial_particle_site,
            initial_occupancy_order=trace.initial_occupancy_order,
            initial_occupancy=trace.initial_occupancy,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
        ),
        context_source="exact_target_pre_gate",
        context_reduction="equal_occurrence_mean_by_target_hash",
        zero_support_policy="exact_unsmoothed",
        warm_start_policy="paired_uniform_artifact_then_three_fixed_restarts",
        trace=trace_rows,
        trace_hash=trace.trace_hash,
        profiles=profiles,
        occurrence_mapping=mappings,
        pairs=pair_tuple,
        schedule_metrics=schedule,
        deterministic_acceptance=deterministic,
        sampled_fidelity=sampled,
        seed_acceptance=seed_acceptance,
        all_context_degradation=degradation,
        zero_support_assessment=zero_support,
        baseline_optimizer_phase=baseline_optimizer_phase,
        target_context_optimizer_phase=target_context_optimizer_phase,
        deterministic_result_hash="sha256:" + "0" * 64,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )
    summary = summary.model_copy(
        update={"deterministic_result_hash": target_context_deterministic_result_hash(summary)}
    )
    return deep_validate_target_context_pasym_swap_summary(summary, model, run, seed)
