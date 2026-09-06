"""Strict deterministic persistence contracts for model-context PAsymSwap results."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from pydantic import (
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from thermo_lab.hashing import canonical_sha256
from thermo_lab.records import FrozenModel
from thermo_lab.target_context_pasym_swap_results import (
    SampledK30Evaluation,
    context_weighted_kl,
    context_weighted_tv,
    derive_sampled_k30_evaluation,
)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOLERANCE = 1e-12
ProbabilityVector = tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]
ConditionalTable = tuple[ProbabilityVector, ProbabilityVector, ProbabilityVector, ProbabilityVector]


class _StrictFrozenResultModel(FrozenModel):
    """Reject coercion and unknown fields in persisted deterministic evidence."""

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


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _weights(value: object, *, name: str) -> ProbabilityVector:
    values = _json_tuple(value)
    if not isinstance(values, tuple) or len(values) != 4:
        raise ValueError(f"{name} must contain four values")
    checked = tuple(
        _finite(item, name=f"{name}[{index}]", nonnegative=True)
        for index, item in enumerate(values)
    )
    if not math.isclose(math.fsum(checked), 1.0, rel_tol=0.0, abs_tol=_TOLERANCE):
        raise ValueError(f"{name} must sum to one within 1e-12")
    return checked  # type: ignore[return-value]


def _table(value: object, *, name: str) -> ConditionalTable:
    rows = _json_tuple(value)
    if not isinstance(rows, tuple) or len(rows) != 4:
        raise ValueError(f"{name} must contain four rows")
    return tuple(_weights(row, name=f"{name}[{index}]") for index, row in enumerate(rows))  # type: ignore[return-value]


def _same_float(observed: float, expected: float, *, path: str) -> None:
    if observed != expected:
        raise ValueError(f"persisted value differs at {path}")


class ModelContextVariantResult(_StrictFrozenResultModel):
    """Exact local comparison of one frozen artifact under both context profiles."""

    role: Literal["uniform", "target_context", "model_context"]
    artifact_hash: str
    equilibrium_conditional: ConditionalTable
    target_profile_kl: StrictFloat
    target_profile_tv: StrictFloat
    model_profile_kl: StrictFloat
    model_profile_tv: StrictFloat

    @field_validator("equilibrium_conditional", mode="before")
    @classmethod
    def normalize_conditional(cls, value: object) -> ConditionalTable:
        return _table(value, name="equilibrium_conditional")

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        _sha(self.artifact_hash, name="artifact_hash")
        for name in (
            "target_profile_kl",
            "target_profile_tv",
            "model_profile_kl",
            "model_profile_tv",
        ):
            _finite(getattr(self, name), name=name, nonnegative=True)
        return self


class ModelContextProfileResult(_StrictFrozenResultModel):
    """All deterministic evidence for one target hash and model-context profile."""

    target_hash: str
    target_profile_hash: str
    model_profile_hash: str
    model_trace_hash: str
    upstream_target_context_artifact_hash: str
    multiplicity: StrictInt
    target_profile_weights: ProbabilityVector
    model_profile_weights: ProbabilityVector
    target_conditional: ConditionalTable
    uniform: ModelContextVariantResult
    target_context: ModelContextVariantResult
    model_context: ModelContextVariantResult
    model_trace_expected_occupancy_before: StrictFloat
    model_trace_expected_occupancy_after: StrictFloat
    target_context_k1_residual: StrictFloat
    target_context_k30_residual: StrictFloat
    model_context_k1_residual: StrictFloat
    model_context_k30_residual: StrictFloat
    model_context_optimizer_endpoint_passed: StrictBool
    profile_kl_non_regression_tolerance: StrictFloat
    minimum_occurrence_weighted_kl_improvement: StrictFloat
    k30_equilibrium_tv_tolerance: StrictFloat
    target_to_model_context_tv: StrictFloat
    model_profile_kl_improvement: StrictFloat
    target_profile_kl_change: StrictFloat
    model_profile_acceptance_passed: StrictBool
    target_profile_degradation_observed: StrictBool
    exact_k30_acceptance_passed: StrictBool
    deterministic_result_hash: str

    @field_validator("target_profile_weights", "model_profile_weights", mode="before")
    @classmethod
    def normalize_weights(cls, value: object, info: Any) -> ProbabilityVector:
        return _weights(value, name=info.field_name)

    @field_validator("target_conditional", mode="before")
    @classmethod
    def normalize_target(cls, value: object) -> ConditionalTable:
        return _table(value, name="target_conditional")

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        for name in (
            "target_hash",
            "target_profile_hash",
            "model_profile_hash",
            "model_trace_hash",
            "upstream_target_context_artifact_hash",
            "deterministic_result_hash",
        ):
            _sha(getattr(self, name), name=name)
        if self.multiplicity <= 0 or self.multiplicity > 500:
            raise ValueError("multiplicity must be in 1..500")
        if tuple(item.role for item in (self.uniform, self.target_context, self.model_context)) != (
            "uniform",
            "target_context",
            "model_context",
        ):
            raise ValueError("three-way artifact roles must be canonical")
        for name in (
            "model_trace_expected_occupancy_before",
            "model_trace_expected_occupancy_after",
            "target_context_k1_residual",
            "target_context_k30_residual",
            "model_context_k1_residual",
            "model_context_k30_residual",
            "profile_kl_non_regression_tolerance",
            "minimum_occurrence_weighted_kl_improvement",
            "k30_equilibrium_tv_tolerance",
            "target_to_model_context_tv",
            "model_profile_kl_improvement",
            "target_profile_kl_change",
        ):
            _finite(
                getattr(self, name),
                name=name,
                nonnegative=name.endswith("residual") or name.endswith("tv"),
            )
        return self

    def identity_payload(self) -> dict[str, object]:
        return {
            "identity_version": "model_context_profile_result.v1",
            "target_hash": self.target_hash,
            "target_profile_hash": self.target_profile_hash,
            "model_profile_hash": self.model_profile_hash,
            "model_trace_hash": self.model_trace_hash,
            "upstream_target_context_artifact_hash": self.upstream_target_context_artifact_hash,
            "multiplicity": self.multiplicity,
            "target_profile_weights": self.target_profile_weights,
            "model_profile_weights": self.model_profile_weights,
            "target_conditional": self.target_conditional,
            "uniform": self.uniform.model_dump(mode="json"),
            "target_context": self.target_context.model_dump(mode="json"),
            "model_context": self.model_context.model_dump(mode="json"),
            "model_trace_expected_occupancy_before": self.model_trace_expected_occupancy_before,
            "model_trace_expected_occupancy_after": self.model_trace_expected_occupancy_after,
            "target_context_k1_residual": self.target_context_k1_residual,
            "target_context_k30_residual": self.target_context_k30_residual,
            "model_context_k1_residual": self.model_context_k1_residual,
            "model_context_k30_residual": self.model_context_k30_residual,
            "model_context_optimizer_endpoint_passed": self.model_context_optimizer_endpoint_passed,
            "profile_kl_non_regression_tolerance": self.profile_kl_non_regression_tolerance,
            "minimum_occurrence_weighted_kl_improvement": (
                self.minimum_occurrence_weighted_kl_improvement
            ),
            "k30_equilibrium_tv_tolerance": self.k30_equilibrium_tv_tolerance,
            "target_to_model_context_tv": self.target_to_model_context_tv,
            "model_profile_kl_improvement": self.model_profile_kl_improvement,
            "target_profile_kl_change": self.target_profile_kl_change,
            "model_profile_acceptance_passed": self.model_profile_acceptance_passed,
            "target_profile_degradation_observed": self.target_profile_degradation_observed,
            "exact_k30_acceptance_passed": self.exact_k30_acceptance_passed,
        }


def _variant(
    role: Literal["uniform", "target_context", "model_context"],
    artifact_hash: str,
    conditional: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    target_weights: Sequence[float],
    model_weights: Sequence[float],
) -> ModelContextVariantResult:
    table = _table(conditional, name=f"{role}_conditional")
    return ModelContextVariantResult(
        role=role,
        artifact_hash=artifact_hash,
        equilibrium_conditional=table,
        target_profile_kl=context_weighted_kl(target, table, target_weights),
        target_profile_tv=context_weighted_tv(target, table, target_weights),
        model_profile_kl=context_weighted_kl(target, table, model_weights),
        model_profile_tv=context_weighted_tv(target, table, model_weights),
    )


def build_model_context_profile_result(
    *,
    target_hash: str,
    target_profile_hash: str,
    model_profile_hash: str,
    model_trace_hash: str,
    upstream_target_context_artifact_hash: str,
    multiplicity: int,
    target_profile_weights: Sequence[float],
    model_profile_weights: Sequence[float],
    target_conditional: Sequence[Sequence[float]],
    uniform_artifact_hash: str,
    uniform_conditional: Sequence[Sequence[float]],
    target_context_artifact_hash: str,
    target_context_conditional: Sequence[Sequence[float]],
    model_context_artifact_hash: str,
    model_context_conditional: Sequence[Sequence[float]],
    model_trace_expected_occupancy_before: float,
    model_trace_expected_occupancy_after: float,
    target_context_k1_residual: float,
    target_context_k30_residual: float,
    model_context_k1_residual: float,
    model_context_k30_residual: float,
    model_context_optimizer_endpoint_passed: bool,
    non_regression_tolerance: float = 1e-12,
    minimum_improvement: float = 1e-8,
    k30_tolerance: float = 0.05,
) -> ModelContextProfileResult:
    """Build deterministic three-way local evidence without running SciPy or THRML."""

    target = _table(target_conditional, name="target_conditional")
    target_weights = _weights(target_profile_weights, name="target_profile_weights")
    model_weights = _weights(model_profile_weights, name="model_profile_weights")
    uniform = _variant(
        "uniform", uniform_artifact_hash, uniform_conditional, target, target_weights, model_weights
    )
    target_context = _variant(
        "target_context",
        target_context_artifact_hash,
        target_context_conditional,
        target,
        target_weights,
        model_weights,
    )
    model_context = _variant(
        "model_context",
        model_context_artifact_hash,
        model_context_conditional,
        target,
        target_weights,
        model_weights,
    )
    improvement = target_context.model_profile_kl - model_context.model_profile_kl
    change = model_context.target_profile_kl - target_context.target_profile_kl
    acceptance = (
        model_context.model_profile_kl <= target_context.model_profile_kl + non_regression_tolerance
        and improvement >= minimum_improvement
    )
    exact_k30_acceptance = all(
        (k30 <= k30_tolerance and k30 <= k1 + _TOLERANCE)
        for k1, k30 in (
            (target_context_k1_residual, target_context_k30_residual),
            (model_context_k1_residual, model_context_k30_residual),
        )
    )
    payload: dict[str, object] = {
        "target_hash": target_hash,
        "target_profile_hash": target_profile_hash,
        "model_profile_hash": model_profile_hash,
        "model_trace_hash": model_trace_hash,
        "upstream_target_context_artifact_hash": upstream_target_context_artifact_hash,
        "multiplicity": multiplicity,
        "target_profile_weights": target_weights,
        "model_profile_weights": model_weights,
        "target_conditional": target,
        "uniform": uniform,
        "target_context": target_context,
        "model_context": model_context,
        "model_trace_expected_occupancy_before": model_trace_expected_occupancy_before,
        "model_trace_expected_occupancy_after": model_trace_expected_occupancy_after,
        "target_context_k1_residual": target_context_k1_residual,
        "target_context_k30_residual": target_context_k30_residual,
        "model_context_k1_residual": model_context_k1_residual,
        "model_context_k30_residual": model_context_k30_residual,
        "model_context_optimizer_endpoint_passed": model_context_optimizer_endpoint_passed,
        "profile_kl_non_regression_tolerance": non_regression_tolerance,
        "minimum_occurrence_weighted_kl_improvement": minimum_improvement,
        "k30_equilibrium_tv_tolerance": k30_tolerance,
        "target_to_model_context_tv": context_weighted_tv(
            target_context.equilibrium_conditional,
            model_context.equilibrium_conditional,
            target_weights,
        ),
        "model_profile_kl_improvement": improvement,
        "target_profile_kl_change": change,
        "model_profile_acceptance_passed": acceptance,
        "target_profile_degradation_observed": change > 0.0,
        "exact_k30_acceptance_passed": exact_k30_acceptance,
    }
    digest = canonical_sha256({"identity_version": "model_context_profile_result.v1", **payload})
    return ModelContextProfileResult(**payload, deterministic_result_hash=digest)


def validate_model_context_profile_result(value: object) -> ModelContextProfileResult:
    """Deeply reload one profile and recompute every persisted deterministic scalar."""

    if isinstance(value, ModelContextProfileResult):
        parsed = value
    elif isinstance(value, Mapping):
        parsed = ModelContextProfileResult.model_validate(_json_tuple(value))
    else:
        raise TypeError("model-context profile result must be a mapping or result model")
    regenerated = build_model_context_profile_result(
        target_hash=parsed.target_hash,
        target_profile_hash=parsed.target_profile_hash,
        model_profile_hash=parsed.model_profile_hash,
        model_trace_hash=parsed.model_trace_hash,
        upstream_target_context_artifact_hash=parsed.upstream_target_context_artifact_hash,
        multiplicity=parsed.multiplicity,
        target_profile_weights=parsed.target_profile_weights,
        model_profile_weights=parsed.model_profile_weights,
        target_conditional=parsed.target_conditional,
        uniform_artifact_hash=parsed.uniform.artifact_hash,
        uniform_conditional=parsed.uniform.equilibrium_conditional,
        target_context_artifact_hash=parsed.target_context.artifact_hash,
        target_context_conditional=parsed.target_context.equilibrium_conditional,
        model_context_artifact_hash=parsed.model_context.artifact_hash,
        model_context_conditional=parsed.model_context.equilibrium_conditional,
        model_trace_expected_occupancy_before=parsed.model_trace_expected_occupancy_before,
        model_trace_expected_occupancy_after=parsed.model_trace_expected_occupancy_after,
        target_context_k1_residual=parsed.target_context_k1_residual,
        target_context_k30_residual=parsed.target_context_k30_residual,
        model_context_k1_residual=parsed.model_context_k1_residual,
        model_context_k30_residual=parsed.model_context_k30_residual,
        model_context_optimizer_endpoint_passed=parsed.model_context_optimizer_endpoint_passed,
        non_regression_tolerance=parsed.profile_kl_non_regression_tolerance,
        minimum_improvement=parsed.minimum_occurrence_weighted_kl_improvement,
        k30_tolerance=parsed.k30_equilibrium_tv_tolerance,
    )
    for role in ("uniform", "target_context", "model_context"):
        observed, expected = getattr(parsed, role), getattr(regenerated, role)
        for name in (
            "target_profile_kl",
            "target_profile_tv",
            "model_profile_kl",
            "model_profile_tv",
        ):
            _same_float(getattr(observed, name), getattr(expected, name), path=f"{role}.{name}")
    for name in (
        "target_to_model_context_tv",
        "model_profile_kl_improvement",
        "target_profile_kl_change",
    ):
        _same_float(getattr(parsed, name), getattr(regenerated, name), path=name)
    if parsed.model_profile_acceptance_passed != regenerated.model_profile_acceptance_passed:
        raise ValueError("persisted value differs at model_profile_acceptance_passed")
    if (
        parsed.target_profile_degradation_observed
        != regenerated.target_profile_degradation_observed
    ):
        raise ValueError("persisted value differs at target_profile_degradation_observed")
    if parsed.exact_k30_acceptance_passed != regenerated.exact_k30_acceptance_passed:
        raise ValueError("persisted value differs at exact_k30_acceptance_passed")
    if parsed.deterministic_result_hash != regenerated.deterministic_result_hash:
        raise ValueError("persisted value differs at deterministic_result_hash")
    return parsed


class ModelContextAcceptance(_StrictFrozenResultModel):
    profile_non_regression_passed: bool
    occurrence_weighted_improvement: float
    occurrence_weighted_improvement_passed: bool
    passed: bool


def validate_model_context_improvement(
    comparisons: tuple[tuple[int, float, float], ...],
    *,
    non_regression_tolerance: float = 1e-12,
    minimum_improvement: float = 1e-8,
) -> ModelContextAcceptance:
    """Validate model-profile KL against paired target-context artifacts."""

    if not comparisons or sum(item[0] for item in comparisons) != 500:
        raise ValueError("comparisons must cover exactly 500 occurrences")
    if any(
        type(multiplicity) is not int
        or multiplicity <= 0
        or not math.isfinite(previous)
        or not math.isfinite(current)
        or previous < 0.0
        or current < 0.0
        for multiplicity, previous, current in comparisons
    ):
        raise ValueError("comparisons must contain finite nonnegative KL values")
    profile_passed = all(
        current <= previous + non_regression_tolerance for _, previous, current in comparisons
    )
    improvement = (
        math.fsum(
            multiplicity * (previous - current) for multiplicity, previous, current in comparisons
        )
        / 500
    )
    improvement_passed = improvement >= minimum_improvement
    return ModelContextAcceptance(
        profile_non_regression_passed=profile_passed,
        occurrence_weighted_improvement=improvement,
        occurrence_weighted_improvement_passed=improvement_passed,
        passed=profile_passed and improvement_passed,
    )


class ModelContextScheduleAcceptance(_StrictFrozenResultModel):
    """Deterministic schedule-level gates over the canonical 37 model profiles."""

    occurrence_count: StrictInt
    profile_count: StrictInt
    model_context_optimizer_endpoints_passed: StrictBool
    profile_non_regression_passed: StrictBool
    occurrence_weighted_improvement: StrictFloat
    occurrence_weighted_improvement_passed: StrictBool
    exact_k30_acceptance_passed: StrictBool
    target_profile_degradation_count: StrictInt
    passed: StrictBool


def derive_model_context_schedule_acceptance(
    profiles: Sequence[ModelContextProfileResult],
    *,
    non_regression_tolerance: float = 1e-12,
    minimum_improvement: float = 1e-8,
) -> ModelContextScheduleAcceptance:
    """Reduce checked profiles without rerunning optimization or sampling."""

    ordered = tuple(
        sorted(
            (validate_model_context_profile_result(item) for item in profiles),
            key=lambda item: item.target_hash,
        )
    )
    if len(ordered) != 37 or len({item.target_hash for item in ordered}) != 37:
        raise ValueError("schedule acceptance requires 37 unique target profiles")
    occurrence_count = sum(item.multiplicity for item in ordered)
    if occurrence_count != 500:
        raise ValueError("schedule acceptance requires exactly 500 occurrences")
    improvement = validate_model_context_improvement(
        tuple(
            (
                item.multiplicity,
                item.target_context.model_profile_kl,
                item.model_context.model_profile_kl,
            )
            for item in ordered
        ),
        non_regression_tolerance=non_regression_tolerance,
        minimum_improvement=minimum_improvement,
    )
    optimizer = all(item.model_context_optimizer_endpoint_passed for item in ordered)
    exact_k30 = all(item.exact_k30_acceptance_passed for item in ordered)
    degradation_count = sum(item.target_profile_degradation_observed for item in ordered)
    return ModelContextScheduleAcceptance(
        occurrence_count=occurrence_count,
        profile_count=len(ordered),
        model_context_optimizer_endpoints_passed=optimizer,
        profile_non_regression_passed=improvement.profile_non_regression_passed,
        occurrence_weighted_improvement=improvement.occurrence_weighted_improvement,
        occurrence_weighted_improvement_passed=improvement.occurrence_weighted_improvement_passed,
        exact_k30_acceptance_passed=exact_k30,
        target_profile_degradation_count=degradation_count,
        passed=optimizer and improvement.passed and exact_k30,
    )


def validate_model_context_schedule_acceptance(
    value: object, profiles: Sequence[ModelContextProfileResult]
) -> ModelContextScheduleAcceptance:
    """Deeply reload schedule gates from their checked profile evidence."""

    if isinstance(value, ModelContextScheduleAcceptance):
        parsed = value
    elif isinstance(value, Mapping):
        parsed = ModelContextScheduleAcceptance.model_validate(_json_tuple(value))
    else:
        raise TypeError("model-context schedule acceptance must be a mapping or result model")
    expected = derive_model_context_schedule_acceptance(profiles)
    for name in (
        "occurrence_count",
        "profile_count",
        "model_context_optimizer_endpoints_passed",
        "profile_non_regression_passed",
        "occurrence_weighted_improvement",
        "occurrence_weighted_improvement_passed",
        "exact_k30_acceptance_passed",
        "target_profile_degradation_count",
        "passed",
    ):
        if getattr(parsed, name) != getattr(expected, name):
            raise ValueError(f"persisted value differs at {name}")
    return parsed


class ModelContextProfileSampleResult(_StrictFrozenResultModel):
    """Persisted exact reference and sampled K=30 evidence for one model profile."""

    target_hash: str
    profile_hash: str
    model_context_artifact_hash: str
    exact_k30_conditional: ConditionalTable
    sampled_k30: SampledK30Evaluation

    @field_validator("exact_k30_conditional", mode="before")
    @classmethod
    def normalize_exact_k30(cls, value: object) -> ConditionalTable:
        return _table(value, name="exact_k30_conditional")

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        for name in ("target_hash", "profile_hash", "model_context_artifact_hash"):
            _sha(getattr(self, name), name=name)
        return self


class ModelContextPAsymSwapSummary(_StrictFrozenResultModel):
    """Complete exact and sampled evidence for one seeded model-context study."""

    request_hash: str
    model_trace_hash: str
    profile_results: tuple[ModelContextProfileResult, ...]
    profile_samples: tuple[ModelContextProfileSampleResult, ...]
    schedule_acceptance: ModelContextScheduleAcceptance
    maximum_empirical_k30_residual: StrictFloat
    thrml_k30_tv_tolerance: StrictFloat
    exact_acceptance_passed: StrictBool
    empirical_acceptance_passed: StrictBool
    acceptance_passed: StrictBool
    deterministic_result_hash: str
    summary_hash: str

    @field_validator("profile_results", "profile_samples", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object) -> object:
        return _json_tuple(value)

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        for name in (
            "request_hash",
            "model_trace_hash",
            "deterministic_result_hash",
            "summary_hash",
        ):
            _sha(getattr(self, name), name=name)
        _finite(
            self.maximum_empirical_k30_residual,
            name="maximum_empirical_k30_residual",
            nonnegative=True,
        )
        _finite(
            self.thrml_k30_tv_tolerance,
            name="thrml_k30_tv_tolerance",
            nonnegative=True,
        )
        return self


def _profile_sample_reference_payload(
    sample: ModelContextProfileSampleResult,
) -> dict[str, object]:
    return {
        "target_hash": sample.target_hash,
        "profile_hash": sample.profile_hash,
        "model_context_artifact_hash": sample.model_context_artifact_hash,
        "exact_k30_conditional": sample.exact_k30_conditional,
    }


def build_model_context_pasym_swap_summary(
    *,
    request_hash: str,
    profile_results: Sequence[ModelContextProfileResult],
    schedule_acceptance: ModelContextScheduleAcceptance,
    profile_samples: Sequence[ModelContextProfileSampleResult],
    thrml_k30_tv_tolerance: float,
) -> ModelContextPAsymSwapSummary:
    """Build one canonical publication summary without importing backend types."""

    _sha(request_hash, name="request_hash")
    checked_profiles = tuple(
        sorted(
            (validate_model_context_profile_result(item) for item in profile_results),
            key=lambda item: item.target_hash,
        )
    )
    checked_acceptance = validate_model_context_schedule_acceptance(
        schedule_acceptance, checked_profiles
    )
    checked_samples = tuple(sorted(profile_samples, key=lambda item: item.target_hash))
    if len(checked_samples) != 37:
        raise ValueError("publication summary requires exactly 37 profile samples")
    model_trace_hashes = {profile.model_trace_hash for profile in checked_profiles}
    if len(model_trace_hashes) != 1:
        raise ValueError("publication summary requires one model trace hash")
    model_trace_hash = next(iter(model_trace_hashes))
    maximum = max(
        residual
        for sample in checked_samples
        for residual in sample.sampled_k30.empirical_to_exact_k30_tv
    )
    exact_passed = checked_acceptance.passed
    empirical_passed = maximum <= thrml_k30_tv_tolerance
    deterministic_payload = {
        "identity_version": "model_context_pasym_swap_deterministic.v1",
        "request_hash": request_hash,
        "model_trace_hash": model_trace_hash,
        "profile_results": tuple(profile.model_dump(mode="json") for profile in checked_profiles),
        "schedule_acceptance": checked_acceptance.model_dump(mode="json"),
        "profile_sample_references": tuple(
            _profile_sample_reference_payload(sample) for sample in checked_samples
        ),
    }
    deterministic_hash = canonical_sha256(deterministic_payload)
    payload: dict[str, object] = {
        "request_hash": request_hash,
        "model_trace_hash": model_trace_hash,
        "profile_results": checked_profiles,
        "profile_samples": checked_samples,
        "schedule_acceptance": checked_acceptance,
        "maximum_empirical_k30_residual": maximum,
        "thrml_k30_tv_tolerance": thrml_k30_tv_tolerance,
        "exact_acceptance_passed": exact_passed,
        "empirical_acceptance_passed": empirical_passed,
        "acceptance_passed": exact_passed and empirical_passed,
        "deterministic_result_hash": deterministic_hash,
    }
    summary_hash = canonical_sha256(
        {"identity_version": "model_context_pasym_swap_summary.v1", **payload}
    )
    return ModelContextPAsymSwapSummary(**payload, summary_hash=summary_hash)


def validate_model_context_pasym_swap_summary(value: object) -> ModelContextPAsymSwapSummary:
    """Deeply reload and recompute one persisted publication summary."""

    if isinstance(value, ModelContextPAsymSwapSummary):
        parsed = value
    elif isinstance(value, Mapping):
        parsed = ModelContextPAsymSwapSummary.model_validate(_json_tuple(value))
    else:
        raise TypeError("model-context PAsymSwap summary must be a mapping or result model")

    profiles = tuple(validate_model_context_profile_result(item) for item in parsed.profile_results)
    if tuple(profile.target_hash for profile in profiles) != tuple(
        sorted(profile.target_hash for profile in profiles)
    ):
        raise ValueError("profile_results must use canonical target-hash order")
    acceptance = validate_model_context_schedule_acceptance(parsed.schedule_acceptance, profiles)
    if len(parsed.profile_samples) != 37:
        raise ValueError("publication summary requires exactly 37 profile samples")

    regenerated_samples: list[ModelContextProfileSampleResult] = []
    for index, (profile, sample) in enumerate(zip(profiles, parsed.profile_samples, strict=True)):
        if sample.target_hash != profile.target_hash:
            raise ValueError(f"profile_samples[{index}].target_hash differs from exact profile")
        if sample.profile_hash != profile.model_profile_hash:
            raise ValueError(f"profile_samples[{index}].profile_hash differs from exact profile")
        if sample.model_context_artifact_hash != profile.model_context.artifact_hash:
            raise ValueError(
                f"profile_samples[{index}].model_context_artifact_hash differs from exact profile"
            )
        exact_residual = max(
            0.5
            * math.fsum(
                abs(actual - equilibrium) for actual, equilibrium in zip(row, base, strict=True)
            )
            for row, base in zip(
                sample.exact_k30_conditional,
                profile.model_context.equilibrium_conditional,
                strict=True,
            )
        )
        if not math.isclose(
            exact_residual,
            profile.model_context_k30_residual,
            rel_tol=0.0,
            abs_tol=_TOLERANCE,
        ):
            raise ValueError(
                f"profile_samples[{index}].exact_k30_conditional differs from exact residual"
            )
        regenerated_sampled = derive_sampled_k30_evaluation(
            sample.sampled_k30.counts,
            sample.exact_k30_conditional,
        )
        if regenerated_sampled.conditional != sample.sampled_k30.conditional:
            raise ValueError(f"profile_samples[{index}].sampled_k30.conditional is stale")
        if (
            regenerated_sampled.empirical_to_exact_k30_tv
            != sample.sampled_k30.empirical_to_exact_k30_tv
        ):
            raise ValueError(f"profile_samples[{index}].sampled_k30 residual is stale")
        regenerated_samples.append(sample.model_copy(update={"sampled_k30": regenerated_sampled}))

    regenerated = build_model_context_pasym_swap_summary(
        request_hash=parsed.request_hash,
        profile_results=profiles,
        schedule_acceptance=acceptance,
        profile_samples=tuple(regenerated_samples),
        thrml_k30_tv_tolerance=parsed.thrml_k30_tv_tolerance,
    )
    for name in (
        "model_trace_hash",
        "maximum_empirical_k30_residual",
        "exact_acceptance_passed",
        "empirical_acceptance_passed",
        "acceptance_passed",
        "deterministic_result_hash",
        "summary_hash",
    ):
        if getattr(parsed, name) != getattr(regenerated, name):
            raise ValueError(f"persisted value differs at {name}")
    return parsed
