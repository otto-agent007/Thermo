"""Strict, reconstruction-validated one-step refinement evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal, Self

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
from thermo_lab.records import FrozenModel
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference
from thermo_lab.trajectory_reinforce_refinement import OneStepRefinement, build_one_step_refinement
from thermo_lab.trajectory_reinforce_results import (
    GradientVector,
    TerminalLawResult,
    TrajectoryReinforceSummary,
    validate_trajectory_reinforce_summary,
)

CapActiveMask = tuple[
    StrictBool,
    StrictBool,
    StrictBool,
    StrictBool,
    StrictBool,
    StrictBool,
    StrictBool,
    StrictBool,
    StrictBool,
]
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class _StrictFrozenRefinementModel(FrozenModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)


def _json_tuple(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_tuple(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_json_tuple(item) for item in value)
    return value


def _terminal_payload(law: object) -> dict[str, object]:
    return {
        "probabilities": tuple(float(value) for value in law.probabilities),  # type: ignore[attr-defined]
        "occupancy": tuple(float(value) for value in law.occupancy),  # type: ignore[attr-defined]
        "expected_mass": float(law.expected_mass),  # type: ignore[attr-defined]
        "particle_number_leakage": float(law.particle_number_leakage),  # type: ignore[attr-defined]
        "signed_mass_drift": float(law.signed_mass_drift),  # type: ignore[attr-defined]
    }


class TrajectoryReinforceRefinementResult(_StrictFrozenRefinementModel):
    """One sampled update with exact before/after objective evaluation."""

    identity_version: Literal["trajectory_reinforce_refinement_step.v1"]
    source_sample_digest: str
    update_policy: Literal["projected_shared_gradient_descent"]
    gradient_source: Literal["seeded_covariance_aware_shared_gradient_mean"]
    learning_rate: StrictFloat = Field(gt=0.0)
    parameter_cap: StrictFloat = Field(gt=0.0)
    initial_parameters: GradientVector
    sampled_shared_gradient: GradientVector
    raw_parameters: GradientVector
    updated_parameters: GradientVector
    cap_active_mask: CapActiveMask
    cap_active_parameter_count: StrictInt = Field(ge=0, le=9)
    model_law_before: TerminalLawResult
    model_law_after: TerminalLawResult
    objective_before: StrictFloat = Field(ge=0.0)
    objective_after: StrictFloat = Field(ge=0.0)
    objective_improvement: StrictFloat
    relative_objective_improvement: StrictFloat
    objective_improved: StrictBool
    bounds_satisfied: StrictBool
    refinement_digest: str

    @field_validator(
        "initial_parameters",
        "sampled_shared_gradient",
        "raw_parameters",
        "updated_parameters",
        "cap_active_mask",
        mode="before",
    )
    @classmethod
    def normalize_vectors(cls, value: object) -> object:
        return _json_tuple(value)

    @model_validator(mode="after")
    def validate_digest_shapes(self) -> Self:
        for name in ("source_sample_digest", "refinement_digest"):
            value = getattr(self, name)
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        return self


class TrajectoryReinforceRefinementSummary(_StrictFrozenRefinementModel):
    """The unchanged estimator contract plus its one-step refinement result."""

    identity_version: Literal["trajectory_reinforce_refinement_summary.v1"]
    estimator: TrajectoryReinforceSummary
    refinement: TrajectoryReinforceRefinementResult
    acceptance_passed: StrictBool
    summary_digest: str

    @model_validator(mode="after")
    def validate_digest_shape(self) -> Self:
        if not _SHA256.fullmatch(self.summary_digest):
            raise ValueError("summary_digest must be a lowercase SHA-256 digest")
        return self


def _refinement_payload(
    *, estimator: TrajectoryReinforceSummary, refinement: OneStepRefinement
) -> dict[str, object]:
    return {
        "identity_version": "trajectory_reinforce_refinement_step.v1",
        "source_sample_digest": estimator.sample.sample_digest,
        "update_policy": "projected_shared_gradient_descent",
        "gradient_source": "seeded_covariance_aware_shared_gradient_mean",
        "learning_rate": float(refinement.learning_rate),
        "parameter_cap": float(refinement.parameter_cap),
        "initial_parameters": tuple(
            float(value) for value in estimator.deterministic.fixture.model_parameters
        ),
        "sampled_shared_gradient": tuple(
            float(value) for value in refinement.sampled_shared_gradient
        ),
        "raw_parameters": tuple(float(value) for value in refinement.update.raw_parameters),
        "updated_parameters": tuple(float(value) for value in refinement.update.updated_parameters),
        "cap_active_mask": refinement.update.cap_active_mask,
        "cap_active_parameter_count": refinement.update.cap_active_parameter_count,
        "model_law_before": _terminal_payload(refinement.model_law_before),
        "model_law_after": _terminal_payload(refinement.model_law_after),
        "objective_before": float(refinement.objective_before),
        "objective_after": float(refinement.objective_after),
        "objective_improvement": float(refinement.objective_improvement),
        "relative_objective_improvement": float(refinement.relative_objective_improvement),
        "objective_improved": refinement.objective_improved,
        "bounds_satisfied": refinement.bounds_satisfied,
    }


def build_trajectory_reinforce_refinement_summary(
    *, estimator: TrajectoryReinforceSummary, refinement: OneStepRefinement
) -> TrajectoryReinforceRefinementSummary:
    """Build a digest-linked estimator and exact objective-improvement record."""

    if not isinstance(estimator, TrajectoryReinforceSummary):
        raise TypeError("estimator must be a TrajectoryReinforceSummary")
    if not isinstance(refinement, OneStepRefinement):
        raise TypeError("refinement must be a OneStepRefinement")
    if refinement.sampled_shared_gradient != estimator.sample.shared.mean:
        raise ValueError("refinement gradient differs from the sampled shared-gradient mean")
    refinement_payload = _refinement_payload(estimator=estimator, refinement=refinement)
    checked_refinement = TrajectoryReinforceRefinementResult(
        **refinement_payload,
        refinement_digest=canonical_sha256(refinement_payload),
    )
    payload: dict[str, object] = {
        "identity_version": "trajectory_reinforce_refinement_summary.v1",
        "estimator": estimator,
        "refinement": checked_refinement,
        "acceptance_passed": (
            estimator.acceptance_passed
            and checked_refinement.bounds_satisfied
            and checked_refinement.objective_improved
        ),
    }
    return TrajectoryReinforceRefinementSummary(
        **payload,
        summary_digest=canonical_sha256(payload),
    )


def validate_trajectory_reinforce_refinement_summary(
    value: object,
) -> TrajectoryReinforceRefinementSummary:
    """Reload and independently rebuild all estimator and refinement evidence."""

    if isinstance(value, TrajectoryReinforceRefinementSummary):
        parsed = value
    elif isinstance(value, str):
        parsed = TrajectoryReinforceRefinementSummary.model_validate_json(value)
    elif isinstance(value, Mapping):
        parsed = TrajectoryReinforceRefinementSummary.model_validate(_json_tuple(value))
    else:
        raise TypeError("trajectory refinement summary must be JSON, a mapping, or a model")

    estimator = validate_trajectory_reinforce_summary(parsed.estimator)
    fixture = build_checked_fixture()
    refinement = build_one_step_refinement(
        fixture=fixture,
        exact=build_exact_reference(fixture),
        sampled_shared_gradient=estimator.sample.shared.mean,
        learning_rate=parsed.refinement.learning_rate,
    )
    regenerated = build_trajectory_reinforce_refinement_summary(
        estimator=estimator,
        refinement=refinement,
    )
    if parsed != regenerated:
        raise ValueError("persisted trajectory refinement evidence differs from regeneration")
    return regenerated
