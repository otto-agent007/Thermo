"""Strict checked inputs for exact target-context PAsymSwap compilation."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import StrictFloat, StrictInt, field_validator, model_validator

from thermo_lab.schemas import (
    _INITIALIZATIONS,
    PAsymSwapModelConfig,
    StrictSchema,
    _require_json_float,
    _require_json_float_list,
    _require_json_float_matrix,
    _tuple_json_lists,
)

ParameterVector = tuple[
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
CheckedInitializations = tuple[ParameterVector, ParameterVector, ParameterVector]


class TargetContextCompilerRunConfig(StrictSchema):
    """Frozen scientific inputs for the exact target-context experiment."""

    initial_particle_site: tuple[Literal[0], Literal[0]]
    context_source: Literal["exact_target_trajectory"]
    context_aggregation: Literal["mean_over_occurrences_sharing_target_hash"]
    zero_support_policy: Literal["preserve_exact_zero_and_report_off_support"]
    baseline_context_weights: tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]
    optimizer: Literal["scipy_lbfgsb"]
    maxiter: Literal[2000]
    maxls: Literal[50]
    ftol: StrictFloat
    gtol: StrictFloat
    projected_gradient_tolerance: StrictFloat
    initializations: CheckedInitializations
    restart_selection: Literal["minimum_objective_then_lexicographic_parameters"]
    horizons: tuple[StrictInt, StrictInt, StrictInt, StrictInt, StrictInt, StrictInt]
    deployment_horizon: Literal[30]
    reset_distribution: Literal["uniform_over_8_free_states"]
    sweep_order: tuple[Literal["hidden", "outputs"], Literal["hidden", "outputs"]]
    chain_count_per_context: Literal[4096]
    samples_per_chain: Literal[1]
    steps_per_sample: Literal[1]
    key_policy: Literal[
        "fold seed with profile hash then artifact hash then input index; "
        "split init and sampling keys"
    ]
    exact_normalization_tolerance: StrictFloat
    target_cm_not_worse_tolerance: StrictFloat
    median_target_weighted_equilibrium_tv_tolerance: StrictFloat
    worst_target_weighted_equilibrium_tv_tolerance: StrictFloat
    k30_equilibrium_tv_tolerance: StrictFloat
    thrml_k30_tv_tolerance: StrictFloat

    @field_validator(
        "ftol",
        "gtol",
        "projected_gradient_tolerance",
        "exact_normalization_tolerance",
        "target_cm_not_worse_tolerance",
        "median_target_weighted_equilibrium_tv_tolerance",
        "worst_target_weighted_equilibrium_tv_tolerance",
        "k30_equilibrium_tv_tolerance",
        "thrml_k30_tv_tolerance",
        mode="before",
    )
    @classmethod
    def validate_float_encoding(cls, value: object, info) -> object:
        return _require_json_float(value, info.field_name)

    @field_validator("initial_particle_site", mode="before")
    @classmethod
    def validate_initial_site_encoding(cls, value: object) -> object:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(item) is not int for item in value)
        ):
            raise ValueError("initial_particle_site must be encoded as two JSON integers")
        return _tuple_json_lists(value)

    @field_validator("baseline_context_weights", mode="before")
    @classmethod
    def validate_baseline_context_weight_encoding(cls, value: object) -> object:
        return _tuple_json_lists(_require_json_float_list(value, "baseline_context_weights"))

    @field_validator("initializations", mode="before")
    @classmethod
    def validate_initialization_encoding(cls, value: object) -> object:
        return _tuple_json_lists(_require_json_float_matrix(value, "initializations"))

    @field_validator("horizons", "sweep_order", mode="before")
    @classmethod
    def freeze_scientific_sequences(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def validate_target_context_schedule(self) -> TargetContextCompilerRunConfig:
        if self.initial_particle_site != (0, 0):
            raise ValueError("initial_particle_site must be exactly (0, 0)")
        if self.baseline_context_weights != (0.25, 0.25, 0.25, 0.25):
            raise ValueError(
                "baseline_context_weights must be uniform over the four input contexts"
            )
        if self.ftol != 1e-12 or self.gtol != 1e-9 or self.projected_gradient_tolerance != 1e-6:
            raise ValueError("optimizer tolerances must match the checked compiler schedule")
        if self.initializations != _INITIALIZATIONS:
            raise ValueError("initializations must be the three checked deterministic restarts")
        if self.horizons != (1, 2, 4, 8, 16, 30):
            raise ValueError("horizons must be the checked ascending finite-horizon schedule")
        if self.sweep_order != ("hidden", "outputs"):
            raise ValueError("sweep_order must update hidden then outputs")
        expected_tolerances = (1e-12, 1e-10, 0.05, 0.10, 0.05, 0.10)
        observed_tolerances = (
            self.exact_normalization_tolerance,
            self.target_cm_not_worse_tolerance,
            self.median_target_weighted_equilibrium_tv_tolerance,
            self.worst_target_weighted_equilibrium_tv_tolerance,
            self.k30_equilibrium_tv_tolerance,
            self.thrml_k30_tv_tolerance,
        )
        if observed_tolerances != expected_tolerances:
            raise ValueError("acceptance tolerances must match the checked release thresholds")
        if any(not math.isfinite(value) or value <= 0.0 for value in observed_tolerances):
            raise ValueError("tolerances must be positive finite numbers")
        return self


def validate_target_context_pasym_swap_request(
    model: PAsymSwapModelConfig,
    run: TargetContextCompilerRunConfig,
    seed: int,
) -> None:
    """Validate cross-section constraints for an exact target-context request."""
    if not isinstance(model, PAsymSwapModelConfig):
        raise TypeError("model must be a PAsymSwapModelConfig")
    if not isinstance(run, TargetContextCompilerRunConfig):
        raise TypeError("run must be a TargetContextCompilerRunConfig")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    validated_model = PAsymSwapModelConfig.model_validate(model.model_dump(mode="json"))
    validated_run = TargetContextCompilerRunConfig.model_validate(run.model_dump(mode="json"))
    if validated_model.macrosteps != 10 or validated_run.deployment_horizon != 30:
        raise ValueError("PAsymSwap schedule and deployment horizon are fixed")
