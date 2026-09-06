"""Strict, reconstruction-validated trajectory REINFORCE result contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal, Self

import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import WORD_ORDER
from thermo_lab.records import FrozenModel
from thermo_lab.schemas import PARAMETER_ORDER
from thermo_lab.thermodynamic_kernel import KernelParameters
from thermo_lab.trajectory_reinforce import (
    VISIBLE_STATE_ORDER,
    ExactTrajectoryReference,
    OccurrenceGradients,
    TerminalLaw,
    TrajectoryFixture,
    build_checked_fixture,
    build_exact_reference,
)

GradientVector = tuple[
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
]
ProbabilityVector4 = tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]
ProbabilityVector8 = tuple[
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
]

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
MOMENT_ROUNDOFF_FACTOR = 64.0 * np.finfo(np.float64).eps
_ROLE_ORDER = ("input_0", "input_1", "hidden_0", "output_0", "output_1")
_JOINT_OUTCOME_ORDER = tuple(
    (hidden, output_0, output_1) for hidden in (0, 1) for output_0, output_1 in WORD_ORDER
)


class _StrictFrozenResultModel(FrozenModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)


def _json_tuple(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_tuple(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_json_tuple(item) for item in value)
    return value


def _sha(value: str, *, name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _vector(value: object, *, name: str, nonnegative: bool = False) -> GradientVector:
    values = _json_tuple(value)
    if not isinstance(values, tuple) or len(values) != 9:
        raise ValueError(f"{name} must contain exactly nine values")
    return tuple(
        _finite_float(item, name=f"{name}[{index}]", nonnegative=nonnegative)
        for index, item in enumerate(values)
    )  # type: ignore[return-value]


def _float_tuple(
    value: object, *, length: int, name: str, nonnegative: bool = False
) -> tuple[float, ...]:
    values = _json_tuple(value)
    if not isinstance(values, tuple) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(
        _finite_float(item, name=f"{name}[{index}]", nonnegative=nonnegative)
        for index, item in enumerate(values)
    )


def _same(observed: Any, expected: Any, *, path: str) -> None:
    if observed != expected:
        raise ValueError(f"persisted value differs at {path}")


class GradientMoments(_StrictFrozenResultModel):
    """Bounded source aggregates for one nine-component sampled gradient."""

    sample_count: StrictInt = Field(ge=2)
    component_sum: GradientVector
    component_sum_squares: GradientVector

    @field_validator("component_sum", mode="before")
    @classmethod
    def validate_sum(cls, value: object) -> GradientVector:
        return _vector(value, name="component_sum")

    @field_validator("component_sum_squares", mode="before")
    @classmethod
    def validate_sum_squares(cls, value: object) -> GradientVector:
        return _vector(value, name="component_sum_squares", nonnegative=True)


class GradientEstimate(_StrictFrozenResultModel):
    """Source moments and all deterministically derived gradient statistics."""

    moments: GradientMoments
    exact: GradientVector
    mean: GradientVector
    variance: GradientVector
    standard_error: GradientVector
    absolute_error: GradientVector
    maximum_absolute_error: StrictFloat

    @field_validator("exact", "mean", mode="before")
    @classmethod
    def validate_vector(cls, value: object) -> GradientVector:
        return _vector(value, name="gradient vector")

    @field_validator("variance", "standard_error", "absolute_error", mode="before")
    @classmethod
    def validate_nonnegative_vector(cls, value: object) -> GradientVector:
        return _vector(value, name="nonnegative gradient vector", nonnegative=True)

    @field_validator("maximum_absolute_error", mode="before")
    @classmethod
    def validate_maximum(cls, value: object) -> object:
        return _finite_float(value, name="maximum_absolute_error", nonnegative=True)


def _variance(*, sample_count: int, component_sum: float, sum_squares: float) -> float:
    projected = (component_sum / sample_count) * component_sum
    if not math.isfinite(projected):
        raise ValueError("variance calculation produced a non-finite intermediate")
    centered = sum_squares - projected
    if not math.isfinite(centered):
        raise ValueError("variance calculation produced a non-finite intermediate")
    tolerance = MOMENT_ROUNDOFF_FACTOR * max(1.0, abs(sum_squares), abs(projected))
    if not math.isfinite(tolerance):
        raise ValueError("variance calculation produced a non-finite tolerance")
    if centered < -tolerance:
        raise ValueError("source moments imply a materially negative variance")
    variance = max(0.0, centered) / (sample_count - 1)
    if not math.isfinite(variance):
        raise ValueError("variance calculation produced a non-finite result")
    return variance


def build_gradient_estimate(
    moments: GradientMoments, exact: GradientVector | tuple[float, ...]
) -> GradientEstimate:
    """Derive all statistics from count, sum, and sum-of-squares sources."""

    if not isinstance(moments, GradientMoments):
        raise TypeError("moments must be GradientMoments")
    exact_vector = _vector(exact, name="exact")
    count = moments.sample_count
    mean = tuple(value / count for value in moments.component_sum)
    variance = tuple(
        _variance(sample_count=count, component_sum=value, sum_squares=squared)
        for value, squared in zip(moments.component_sum, moments.component_sum_squares, strict=True)
    )
    standard_error = tuple(math.sqrt(value / count) for value in variance)
    absolute_error = tuple(
        abs(observed - expected) for observed, expected in zip(mean, exact_vector, strict=True)
    )
    return GradientEstimate(
        moments=moments,
        exact=exact_vector,
        mean=mean,
        variance=variance,
        standard_error=standard_error,
        absolute_error=absolute_error,
        maximum_absolute_error=max(absolute_error),
    )


def build_shared_gradient_estimate(
    occurrence_0: GradientMoments,
    occurrence_1: GradientMoments,
    cross_products: GradientVector | tuple[float, ...],
    exact: GradientVector | tuple[float, ...],
) -> GradientEstimate:
    """Derive covariance-aware shared moments from the two occurrence sources."""

    if occurrence_0.sample_count != occurrence_1.sample_count:
        raise ValueError("occurrence sample_count values must match")
    build_gradient_estimate(occurrence_0, (0.0,) * 9)
    build_gradient_estimate(occurrence_1, (0.0,) * 9)
    cross = _vector(cross_products, name="cross_products")
    count = occurrence_0.sample_count
    for index, (sum_0, sum_1, q_0, q_1, product_sum) in enumerate(
        zip(
            occurrence_0.component_sum,
            occurrence_1.component_sum,
            occurrence_0.component_sum_squares,
            occurrence_1.component_sum_squares,
            cross,
            strict=True,
        )
    ):
        projected_0 = (sum_0 / count) * sum_0
        projected_1 = (sum_1 / count) * sum_1
        projected_cross = (sum_0 / count) * sum_1
        if not all(math.isfinite(value) for value in (projected_0, projected_1, projected_cross)):
            raise ValueError(
                "centered Cauchy-Schwarz calculation produced a non-finite intermediate"
            )
        centered_0 = q_0 - projected_0
        centered_1 = q_1 - projected_1
        centered_cross = product_sum - projected_cross
        if not all(math.isfinite(value) for value in (centered_0, centered_1, centered_cross)):
            raise ValueError(
                "centered Cauchy-Schwarz calculation produced a non-finite intermediate"
            )
        centered_0 = max(0.0, centered_0)
        centered_1 = max(0.0, centered_1)
        bound = math.sqrt(centered_0) * math.sqrt(centered_1)
        if not math.isfinite(bound):
            raise ValueError("centered Cauchy-Schwarz calculation produced a non-finite bound")
        tolerance = MOMENT_ROUNDOFF_FACTOR * max(1.0, abs(product_sum), abs(projected_cross), bound)
        if not math.isfinite(tolerance):
            raise ValueError("centered Cauchy-Schwarz calculation produced a non-finite tolerance")
        if abs(centered_cross) > bound + tolerance:
            raise ValueError(
                f"cross_products[{index}] violates centered Cauchy-Schwarz consistency"
            )
    shared_sum = tuple(
        left + right
        for left, right in zip(occurrence_0.component_sum, occurrence_1.component_sum, strict=True)
    )
    shared_q = tuple(
        left + right + 2.0 * product_sum
        for left, right, product_sum in zip(
            occurrence_0.component_sum_squares,
            occurrence_1.component_sum_squares,
            cross,
            strict=True,
        )
    )
    if not all(math.isfinite(value) for value in (*shared_sum, *shared_q)):
        raise ValueError("shared moment calculation produced a non-finite result")
    return build_gradient_estimate(
        GradientMoments(
            sample_count=count,
            component_sum=shared_sum,
            component_sum_squares=shared_q,
        ),
        exact,
    )


class TerminalLawResult(_StrictFrozenResultModel):
    probabilities: ProbabilityVector8
    occupancy: tuple[StrictFloat, StrictFloat, StrictFloat]
    expected_mass: StrictFloat
    particle_number_leakage: StrictFloat
    signed_mass_drift: StrictFloat

    @field_validator("probabilities", mode="before")
    @classmethod
    def validate_probabilities(cls, value: object) -> object:
        return _float_tuple(value, length=8, name="probabilities", nonnegative=True)

    @field_validator("occupancy", mode="before")
    @classmethod
    def validate_occupancy(cls, value: object) -> object:
        return _float_tuple(value, length=3, name="occupancy", nonnegative=True)

    @field_validator(
        "expected_mass",
        "particle_number_leakage",
        "signed_mass_drift",
        mode="before",
    )
    @classmethod
    def validate_diagnostics(cls, value: object) -> object:
        return _finite_float(value, name="terminal diagnostic")


class OccurrenceGradientResult(_StrictFrozenResultModel):
    occurrences: tuple[GradientVector, GradientVector]
    shared: GradientVector

    @field_validator("occurrences", mode="before")
    @classmethod
    def validate_occurrences(cls, value: object) -> object:
        values = _json_tuple(value)
        if not isinstance(values, tuple) or len(values) != 2:
            raise ValueError("occurrences must contain exactly two gradient vectors")
        return tuple(
            _vector(item, name=f"occurrences[{index}]") for index, item in enumerate(values)
        )

    @field_validator("shared", mode="before")
    @classmethod
    def validate_shared(cls, value: object) -> GradientVector:
        return _vector(value, name="shared")


class TrajectoryFixtureResult(_StrictFrozenResultModel):
    initial_state: tuple[StrictInt, StrictInt, StrictInt]
    occurrences: tuple[tuple[StrictInt, StrictInt], tuple[StrictInt, StrictInt]]
    target_edge: tuple[tuple[StrictInt, StrictInt], tuple[StrictInt, StrictInt]]
    target_probabilities: tuple[StrictFloat, StrictFloat]
    target_conditional: tuple[
        ProbabilityVector4, ProbabilityVector4, ProbabilityVector4, ProbabilityVector4
    ]
    target_hash: str
    model_parameters: GradientVector
    beta: StrictFloat
    parameter_cap: StrictFloat
    finite_difference_step: StrictFloat
    exact_tolerance: StrictFloat
    finite_difference_tolerance: StrictFloat

    @field_validator("initial_state", "occurrences", "target_edge", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object) -> object:
        return _json_tuple(value)

    @field_validator("target_probabilities", mode="before")
    @classmethod
    def validate_target_probabilities(cls, value: object) -> object:
        return _float_tuple(value, length=2, name="target_probabilities", nonnegative=True)

    @field_validator("target_conditional", mode="before")
    @classmethod
    def validate_target_conditional(cls, value: object) -> object:
        rows = _json_tuple(value)
        if not isinstance(rows, tuple) or len(rows) != 4:
            raise ValueError("target_conditional must contain exactly four rows")
        return tuple(
            _float_tuple(row, length=4, name=f"target_conditional[{index}]", nonnegative=True)
            for index, row in enumerate(rows)
        )

    @field_validator("model_parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: object) -> GradientVector:
        return _vector(value, name="model_parameters")

    @model_validator(mode="after")
    def validate_scalars(self) -> Self:
        _sha(self.target_hash, name="target_hash")
        for name in (
            "beta",
            "parameter_cap",
            "finite_difference_step",
            "exact_tolerance",
            "finite_difference_tolerance",
        ):
            _finite_float(getattr(self, name), name=name, nonnegative=True)
        return self

    @field_validator(
        "beta",
        "parameter_cap",
        "finite_difference_step",
        "exact_tolerance",
        "finite_difference_tolerance",
        mode="before",
    )
    @classmethod
    def reject_coercive_scalars(cls, value: object) -> object:
        return _finite_float(value, name="fixture scalar", nonnegative=True)


class TrajectoryReinforceDeterministicResult(_StrictFrozenResultModel):
    identity_version: Literal["trajectory_reinforce_exact.v1"]
    request_hash: str
    word_order: tuple[tuple[StrictInt, StrictInt], ...]
    visible_state_order: tuple[tuple[StrictInt, StrictInt, StrictInt], ...]
    role_order: tuple[str, ...]
    joint_outcome_order: tuple[tuple[StrictInt, StrictInt, StrictInt], ...]
    parameter_order: tuple[str, ...]
    fixture: TrajectoryFixtureResult
    target_law: TerminalLawResult
    model_law: TerminalLawResult
    objective: StrictFloat
    reward_coefficient: tuple[StrictFloat, StrictFloat, StrictFloat]
    score: OccurrenceGradientResult
    expected_reference: OccurrenceGradientResult
    finite_difference: OccurrenceGradientResult
    finite_difference_tied: GradientVector
    exact_component_errors: OccurrenceGradientResult
    finite_difference_component_errors: OccurrenceGradientResult
    tied_finite_difference_error: GradientVector
    maximum_exact_error: StrictFloat
    maximum_finite_difference_error: StrictFloat
    main_path_count: StrictInt
    augmented_path_count: StrictInt
    accepted: StrictBool
    deterministic_result_digest: str

    @field_validator(
        "word_order",
        "visible_state_order",
        "role_order",
        "joint_outcome_order",
        "parameter_order",
        "reward_coefficient",
        mode="before",
    )
    @classmethod
    def normalize_sequences(cls, value: object) -> object:
        return _json_tuple(value)

    @field_validator("finite_difference_tied", "tied_finite_difference_error", mode="before")
    @classmethod
    def validate_gradient_vectors(cls, value: object) -> GradientVector:
        return _vector(value, name="gradient vector")

    @field_validator("reward_coefficient", mode="before")
    @classmethod
    def validate_reward_coefficient(cls, value: object) -> object:
        return _float_tuple(value, length=3, name="reward_coefficient")

    @field_validator(
        "objective", "maximum_exact_error", "maximum_finite_difference_error", mode="before"
    )
    @classmethod
    def reject_coercive_scalars(cls, value: object) -> object:
        return _finite_float(value, name="deterministic scalar")

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _sha(self.request_hash, name="request_hash")
        _sha(self.deterministic_result_digest, name="deterministic_result_digest")
        return self


class TrajectoryReinforceSampleResult(_StrictFrozenResultModel):
    identity_version: Literal["trajectory_reinforce_sample.v1"]
    deterministic_result_digest: str
    seed: StrictInt = Field(ge=0)
    sample_definition: str = Field(min_length=1)
    occurrence_0: GradientEstimate
    occurrence_1: GradientEstimate
    cross_products: GradientVector
    shared: GradientEstimate
    maximum_absolute_shared_gradient_error: StrictFloat
    sample_digest: str

    @field_validator("cross_products", mode="before")
    @classmethod
    def validate_cross_products(cls, value: object) -> GradientVector:
        return _vector(value, name="cross_products")

    @field_validator("maximum_absolute_shared_gradient_error", mode="before")
    @classmethod
    def validate_maximum_error(cls, value: object) -> object:
        return _finite_float(value, name="maximum_absolute_shared_gradient_error", nonnegative=True)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _sha(self.deterministic_result_digest, name="deterministic_result_digest")
        _sha(self.sample_digest, name="sample_digest")
        if not self.sample_definition:
            raise ValueError("sample_definition must be nonempty")
        return self


class TrajectoryReinforceSummary(_StrictFrozenResultModel):
    identity_version: Literal["trajectory_reinforce_summary.v1"]
    deterministic: TrajectoryReinforceDeterministicResult
    sample: TrajectoryReinforceSampleResult
    acceptance_passed: StrictBool
    summary_digest: str

    @model_validator(mode="after")
    def validate_digest_shape(self) -> Self:
        _sha(self.summary_digest, name="summary_digest")
        return self


def _terminal_payload(law: TerminalLaw) -> dict[str, object]:
    return {
        "probabilities": tuple(float(value) for value in law.probabilities),
        "occupancy": tuple(float(value) for value in law.occupancy),
        "expected_mass": float(law.expected_mass),
        "particle_number_leakage": float(law.particle_number_leakage),
        "signed_mass_drift": float(law.signed_mass_drift),
    }


def _gradient_payload(gradients: OccurrenceGradients) -> dict[str, object]:
    return {
        "occurrences": tuple(
            tuple(float(value) for value in vector) for vector in gradients.occurrences
        ),
        "shared": tuple(float(value) for value in gradients.shared),
    }


def _fixture_payload(fixture: TrajectoryFixture) -> dict[str, object]:
    return {
        "initial_state": fixture.initial_state,
        "occurrences": fixture.occurrences,
        "target_edge": fixture.target_edge,
        "target_probabilities": fixture.target_probabilities,
        "target_conditional": tuple(
            tuple(float(value) for value in row) for row in fixture.target_conditional
        ),
        "target_hash": fixture.target_hash,
        "model_parameters": tuple(float(value) for value in fixture.model_parameters.values),
        "beta": float(fixture.beta),
        "parameter_cap": float(fixture.parameter_cap),
        "finite_difference_step": float(fixture.finite_difference_step),
        "exact_tolerance": float(fixture.exact_tolerance),
        "finite_difference_tolerance": float(fixture.finite_difference_tolerance),
    }


def build_trajectory_reinforce_deterministic_result(
    *, request_hash: str, fixture: TrajectoryFixture, exact: ExactTrajectoryReference
) -> TrajectoryReinforceDeterministicResult:
    """Serialize the complete checked fixture and its independently generated oracles."""

    _sha(request_hash, name="request_hash")
    if not isinstance(fixture, TrajectoryFixture) or not isinstance(
        exact, ExactTrajectoryReference
    ):
        raise TypeError("fixture and exact must be checked trajectory reference objects")
    if _fixture_payload(fixture) != _fixture_payload(build_checked_fixture()):
        raise ValueError("fixture differs from the publication-checked scientific inputs")
    regenerated = build_exact_reference(fixture)
    payload: dict[str, object] = {
        "identity_version": "trajectory_reinforce_exact.v1",
        "request_hash": request_hash,
        "word_order": WORD_ORDER,
        "visible_state_order": VISIBLE_STATE_ORDER,
        "role_order": _ROLE_ORDER,
        "joint_outcome_order": _JOINT_OUTCOME_ORDER,
        "parameter_order": PARAMETER_ORDER,
        "fixture": _fixture_payload(fixture),
        "target_law": _terminal_payload(exact.target_law),
        "model_law": _terminal_payload(exact.model_law),
        "objective": float(exact.objective),
        "reward_coefficient": tuple(float(value) for value in exact.reward_coefficient),
        "score": _gradient_payload(exact.score),
        "expected_reference": _gradient_payload(exact.expected_reference),
        "finite_difference": _gradient_payload(exact.finite_difference),
        "finite_difference_tied": tuple(float(value) for value in exact.finite_difference_tied),
        "exact_component_errors": _gradient_payload(exact.exact_component_errors),
        "finite_difference_component_errors": _gradient_payload(
            exact.finite_difference_component_errors
        ),
        "tied_finite_difference_error": tuple(
            float(value) for value in exact.tied_finite_difference_error
        ),
        "maximum_exact_error": float(exact.maximum_exact_error),
        "maximum_finite_difference_error": float(exact.maximum_finite_difference_error),
        "main_path_count": exact.main_path_count,
        "augmented_path_count": exact.augmented_path_count,
        "accepted": exact.accepted,
    }
    expected_payload = {
        **payload,
        "target_law": _terminal_payload(regenerated.target_law),
        "model_law": _terminal_payload(regenerated.model_law),
        "objective": float(regenerated.objective),
        "reward_coefficient": tuple(float(value) for value in regenerated.reward_coefficient),
        "score": _gradient_payload(regenerated.score),
        "expected_reference": _gradient_payload(regenerated.expected_reference),
        "finite_difference": _gradient_payload(regenerated.finite_difference),
        "finite_difference_tied": tuple(
            float(value) for value in regenerated.finite_difference_tied
        ),
        "exact_component_errors": _gradient_payload(regenerated.exact_component_errors),
        "finite_difference_component_errors": _gradient_payload(
            regenerated.finite_difference_component_errors
        ),
        "tied_finite_difference_error": tuple(
            float(value) for value in regenerated.tied_finite_difference_error
        ),
        "maximum_exact_error": float(regenerated.maximum_exact_error),
        "maximum_finite_difference_error": float(regenerated.maximum_finite_difference_error),
        "main_path_count": regenerated.main_path_count,
        "augmented_path_count": regenerated.augmented_path_count,
        "accepted": regenerated.accepted,
    }
    if payload != expected_payload:
        raise ValueError("exact reference differs from the regenerated checked fixture")
    digest = canonical_sha256(payload)
    return TrajectoryReinforceDeterministicResult(**payload, deterministic_result_digest=digest)


def build_trajectory_reinforce_sample_result(
    *,
    deterministic_result_digest: str,
    seed: int,
    sample_definition: str,
    occurrence_0: GradientMoments,
    occurrence_1: GradientMoments,
    cross_products: GradientVector | tuple[float, ...],
    exact: OccurrenceGradientResult,
) -> TrajectoryReinforceSampleResult:
    """Build one seeded result exclusively from bounded sufficient sources."""

    _sha(deterministic_result_digest, name="deterministic_result_digest")
    if not isinstance(exact, OccurrenceGradientResult):
        raise TypeError("exact must be OccurrenceGradientResult")
    cross = _vector(cross_products, name="cross_products")
    estimate_0 = build_gradient_estimate(occurrence_0, exact.occurrences[0])
    estimate_1 = build_gradient_estimate(occurrence_1, exact.occurrences[1])
    shared = build_shared_gradient_estimate(occurrence_0, occurrence_1, cross, exact.shared)
    payload: dict[str, object] = {
        "identity_version": "trajectory_reinforce_sample.v1",
        "deterministic_result_digest": deterministic_result_digest,
        "seed": seed,
        "sample_definition": sample_definition,
        "occurrence_0": estimate_0,
        "occurrence_1": estimate_1,
        "cross_products": cross,
        "shared": shared,
        "maximum_absolute_shared_gradient_error": shared.maximum_absolute_error,
    }
    return TrajectoryReinforceSampleResult(
        **payload,
        sample_digest=canonical_sha256(payload),
    )


def build_trajectory_reinforce_summary(
    *,
    deterministic: TrajectoryReinforceDeterministicResult,
    sample: TrajectoryReinforceSampleResult,
) -> TrajectoryReinforceSummary:
    """Build the complete exact-plus-sampled result with a non-stochastic pass flag."""

    if sample.deterministic_result_digest != deterministic.deterministic_result_digest:
        raise ValueError("sample deterministic result digest differs from exact result")
    payload: dict[str, object] = {
        "identity_version": "trajectory_reinforce_summary.v1",
        "deterministic": deterministic,
        "sample": sample,
        "acceptance_passed": deterministic.accepted,
    }
    return TrajectoryReinforceSummary(**payload, summary_digest=canonical_sha256(payload))


def _fixture_from_result(result: TrajectoryFixtureResult) -> TrajectoryFixture:
    return TrajectoryFixture(
        initial_state=result.initial_state,
        occurrences=result.occurrences,
        target_edge=result.target_edge,
        target_probabilities=result.target_probabilities,
        target_conditional=np.asarray(result.target_conditional, dtype=np.float64),
        target_hash=result.target_hash,
        model_parameters=KernelParameters(result.model_parameters),
        beta=result.beta,
        parameter_cap=result.parameter_cap,
        finite_difference_step=result.finite_difference_step,
        exact_tolerance=result.exact_tolerance,
        finite_difference_tolerance=result.finite_difference_tolerance,
    )


def validate_trajectory_reinforce_summary(value: object) -> TrajectoryReinforceSummary:
    """Reload and regenerate every scientific and moment-derived result field."""

    if isinstance(value, TrajectoryReinforceSummary):
        parsed = value
    elif isinstance(value, str):
        parsed = TrajectoryReinforceSummary.model_validate_json(value)
    elif isinstance(value, Mapping):
        parsed = TrajectoryReinforceSummary.model_validate(_json_tuple(value))
    else:
        raise TypeError("trajectory REINFORCE summary must be JSON, a mapping, or a model")

    fixture = _fixture_from_result(parsed.deterministic.fixture)
    exact = build_exact_reference(fixture)
    regenerated_deterministic = build_trajectory_reinforce_deterministic_result(
        request_hash=parsed.deterministic.request_hash,
        fixture=fixture,
        exact=exact,
    )
    _same(parsed.deterministic, regenerated_deterministic, path="deterministic")

    regenerated_sample = build_trajectory_reinforce_sample_result(
        deterministic_result_digest=regenerated_deterministic.deterministic_result_digest,
        seed=parsed.sample.seed,
        sample_definition=parsed.sample.sample_definition,
        occurrence_0=parsed.sample.occurrence_0.moments,
        occurrence_1=parsed.sample.occurrence_1.moments,
        cross_products=parsed.sample.cross_products,
        exact=regenerated_deterministic.expected_reference,
    )
    _same(parsed.sample, regenerated_sample, path="sample/shared/derived fields")
    regenerated = build_trajectory_reinforce_summary(
        deterministic=regenerated_deterministic, sample=regenerated_sample
    )
    _same(parsed, regenerated, path="summary digest or acceptance flag")
    return regenerated


def validate_trajectory_reinforce_deterministic_result(
    value: object,
) -> TrajectoryReinforceDeterministicResult:
    """Deeply validate a deterministic result by embedding it in no sampled claims."""

    parsed = (
        value
        if isinstance(value, TrajectoryReinforceDeterministicResult)
        else TrajectoryReinforceDeterministicResult.model_validate(_json_tuple(value))
    )
    fixture = _fixture_from_result(parsed.fixture)
    regenerated = build_trajectory_reinforce_deterministic_result(
        request_hash=parsed.request_hash, fixture=fixture, exact=build_exact_reference(fixture)
    )
    _same(parsed, regenerated, path="deterministic")
    return regenerated


def validate_trajectory_reinforce_sample_result(
    value: object, exact: OccurrenceGradientResult
) -> TrajectoryReinforceSampleResult:
    """Deeply validate a seeded result against its deterministic expected gradient."""

    parsed = (
        value
        if isinstance(value, TrajectoryReinforceSampleResult)
        else TrajectoryReinforceSampleResult.model_validate(_json_tuple(value))
    )
    regenerated = build_trajectory_reinforce_sample_result(
        deterministic_result_digest=parsed.deterministic_result_digest,
        seed=parsed.seed,
        sample_definition=parsed.sample_definition,
        occurrence_0=parsed.occurrence_0.moments,
        occurrence_1=parsed.occurrence_1.moments,
        cross_products=parsed.cross_products,
        exact=exact,
    )
    _same(parsed, regenerated, path="sample")
    return regenerated
