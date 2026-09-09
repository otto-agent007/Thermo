"""Strict checked request schema for one-step composed trajectory refinement."""

from __future__ import annotations

from typing import Literal

from pydantic import StrictFloat, StrictInt, field_validator, model_validator

from thermo_lab.hashing import to_json_value
from thermo_lab.records import FrozenModel
from thermo_lab.schemas import PAsymSwapModelConfig

SOURCE_COMPOSED_CONFIG_HASH = (
    "sha256:51f554b4b1e7e747618ec75de5e4aa0a1dda8a93e29967bcd84d2116510f28a4"
)


def _tuple_json_lists(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuple_json_lists(item) for item in value)
    return value


class ComposedTrajectoryRefinementRunConfig(FrozenModel):
    """Immutable schedule for one equilibrium 25-site trajectory update."""

    start_family: Literal["model_context"]
    horizon: Literal["equilibrium"]
    source_composed_config_hash: Literal[SOURCE_COMPOSED_CONFIG_HASH]
    objective_policy: Literal["squared_terminal_occupancy_error"]
    reward_policy: Literal["independent_batch_current_model_occupancy_linearization"]
    reference_policy: Literal["independent_same_parent_non_propagated"]
    occupancy_batch_size: Literal[32768]
    gradient_batch_size: Literal[32768]
    evaluation_batch_size: Literal[32768]
    release_seeds: tuple[StrictInt, StrictInt, StrictInt]
    rng_family: Literal["numpy.random.Generator(PCG64)"]
    role_stream_policy: Literal[
        "SeedSequence(seed).spawn(3) in occupancy, gradient, evaluation order"
    ]
    evaluation_pairing_policy: Literal["common_random_numbers_per_occurrence_before_and_after"]
    update_policy: Literal["one_projected_grouped_gradient_descent_step"]
    learning_rate: StrictFloat
    improvement_policy: Literal["descriptive_non_gating"]

    @field_validator(
        "occupancy_batch_size",
        "gradient_batch_size",
        "evaluation_batch_size",
        mode="before",
    )
    @classmethod
    def validate_integer_encoding(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be encoded as a JSON integer")
        return value

    @field_validator("release_seeds", mode="before")
    @classmethod
    def freeze_release_seeds(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @field_validator("learning_rate", mode="before")
    @classmethod
    def validate_learning_rate_encoding(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("learning_rate must be encoded as a JSON floating-point number")
        return value

    @model_validator(mode="after")
    def validate_checked_schedule(self) -> ComposedTrajectoryRefinementRunConfig:
        if self.release_seeds != (0, 1, 2):
            raise ValueError("release_seeds must use the checked independent replications")
        if self.learning_rate != 0.01:
            raise ValueError("learning_rate must match the checked one-step release")
        return self


def validate_composed_trajectory_refinement_request(
    model: PAsymSwapModelConfig,
    run: ComposedTrajectoryRefinementRunConfig,
    seed: int,
) -> None:
    """Deeply validate the full-program one-step refinement request."""

    if not isinstance(model, PAsymSwapModelConfig):
        raise TypeError("model must be a PAsymSwapModelConfig")
    if not isinstance(run, ComposedTrajectoryRefinementRunConfig):
        raise TypeError("run must be a ComposedTrajectoryRefinementRunConfig")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    validated_model = PAsymSwapModelConfig.model_validate(
        to_json_value(model.model_dump(mode="json"))
    )
    validated_run = ComposedTrajectoryRefinementRunConfig.model_validate(
        to_json_value(run.model_dump(mode="json"))
    )
    if seed not in validated_run.release_seeds:
        raise ValueError("seed must be one of the checked release_seeds")
    if validated_model.beta != 1.0 or validated_model.parameter_cap != 2.0:
        raise ValueError("beta and parameter_cap must match the checked PAsymSwap release")
