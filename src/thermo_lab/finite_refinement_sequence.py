"""Five predeclared finite-sweep updates with replay-validated bounded evidence."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

import numpy as np
from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator

from thermo_lab.composed_trajectory_refinement import (
    _checked_parameter_matrix,
    _checked_site_vector,
    _tuple_matrix,
    calculate_unbiased_population_objective,
    project_grouped_parameters,
)
from thermo_lab.composed_trajectory_refinement_results import (
    GroupedGradientResult,
    GroupedParameterUpdateResult,
    StrictEvidenceModel,
    TerminalOccupancyResult,
    _tuple_json_lists,
)
from thermo_lab.finite_sweep_sampling import (
    _request,
    estimate_finite_sweep_grouped_gradient,
    evaluate_finite_sweep_pair,
    finite_sampling_tables_digest,
    gradient_source_digest,
    sample_finite_sweep_terminal_occupancy,
)
from thermo_lab.frozen_pair_finite_sweeps import HorizonTerminalEvidence
from thermo_lab.hashing import canonical_sha256, to_json_value


class FiniteRefinementProtocol(StrictEvidenceModel):
    identity_version: Literal["bounded_finite_refinement_protocol.v1"] = (
        "bounded_finite_refinement_protocol.v1"
    )
    step_count: StrictInt = Field(default=5, ge=5, le=5)
    horizon: StrictInt = Field(default=4, ge=4, le=4)
    batch_size: StrictInt = Field(default=32768, ge=32768, le=32768)
    learning_rate: StrictFloat = Field(default=0.01, ge=0.01, le=0.01)
    parameter_cap: StrictFloat = Field(default=2.0, ge=2.0, le=2.0)
    beta: StrictFloat = Field(default=1.0, ge=1.0, le=1.0)
    dtype: Literal["float64"] = "float64"
    checkpoint_policy: Literal["always_fifth_update"] = "always_fifth_update"
    role_policy: Literal[
        "SeedSequence([0x4D34, seed]).spawn(11); "
        "alternating occupancy/gradient; final evaluation last"
    ] = (
        "SeedSequence([0x4D34, seed]).spawn(11); "
        "alternating occupancy/gradient; final evaluation last"
    )
    endpoint_policy: Literal[
        "uniform reset; hidden then outputs; hidden-major joint inverse CDF; propagate outputs only"
    ] = "uniform reset; hidden then outputs; hidden-major joint inverse CDF; propagate outputs only"
    gradient_policy: Literal[
        "finite endpoint main-minus-independent-same-parent-reference scores; "
        "sum shared occurrences first"
    ] = (
        "finite endpoint main-minus-independent-same-parent-reference scores; "
        "sum shared occurrences first"
    )
    reward_policy: Literal["two times (independent occupancy estimate minus exact target)"] = (
        "two times (independent occupancy estimate minus exact target)"
    )
    evaluation_policy: Literal[
        "fresh initial/fifth paired evaluation; common random numbers; no checkpoint selection"
    ] = "fresh initial/fifth paired evaluation; common random numbers; no checkpoint selection"
    objective_policy: Literal["unbiased_order_two_u_statistic"] = "unbiased_order_two_u_statistic"
    uncertainty_policy: Literal["paired_delete_one_jackknife_normal_95_approximate"] = (
        "paired_delete_one_jackknife_normal_95_approximate"
    )
    scientific_status: Literal["descriptive_non_gating"] = "descriptive_non_gating"
    sampler_identity_policy: Literal[
        "normalize signed zero in parameter and reward cache identities"
    ] = "normalize signed zero in parameter and reward cache identities"
    replay_relative_tolerance: StrictFloat = Field(default=1e-12, ge=1e-12, le=1e-12)
    replay_absolute_tolerance: StrictFloat = Field(default=1e-10, ge=1e-10, le=1e-10)


def spawn_finite_refinement_roles(seed: int) -> tuple[int, ...]:
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("finite refinement requires a checked seed 0, 1, or 2")
    children = np.random.SeedSequence([0x4D34, seed]).spawn(11)
    roles = tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)
    if len(set(roles)) != 11:
        raise ValueError("finite refinement role seeds must be distinct")
    return roles


class FiniteRefinementStep(StrictEvidenceModel):
    iteration: StrictInt = Field(ge=1, le=5)
    parameter_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    exact_tables_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    occupancy_seed: StrictInt = Field(ge=0)
    gradient_seed: StrictInt = Field(ge=0)
    occupancy: TerminalOccupancyResult
    reward_coefficient: tuple[StrictFloat, ...]
    gradient: GroupedGradientResult
    update: GroupedParameterUpdateResult

    @field_validator("reward_coefficient", mode="before")
    @classmethod
    def freeze_reward(cls, value):
        return _tuple_json_lists(value)

    def training_objective(self, target):
        return calculate_unbiased_population_objective(
            self.occupancy.occupancy_counts,
            sample_count=self.occupancy.sample_count,
            target_occupancy=target,
        )


def sequence_result_digest(value) -> str:
    payload = (
        value.model_dump(mode="json") if isinstance(value, StrictEvidenceModel) else dict(value)
    )
    payload.pop("result_digest", None)
    return canonical_sha256(
        {
            "identity_version": "finite_training_sequence.v1",
            "evidence_class": "software_simulation",
            "sequence": payload,
        }
    )


def require_gradient_replay(stored, expected, parameters, groups, sites, reward, **settings):
    """Check stochastic replay numerically and the stored source identity exactly."""
    if (
        stored.sample_count != settings["batch_size"]
        or len(stored.component_sum) != len(parameters)
        or any(
            not np.allclose(getattr(stored, name), getattr(expected, name), rtol=1e-12, atol=1e-10)
            for name in ("component_sum", "component_sum_squares")
        )
    ):
        raise ValueError("gradient moments do not match deterministic replay")
    digest = gradient_source_digest(
        parameters,
        groups,
        sites,
        reward,
        stored.component_sum,
        stored.component_sum_squares,
        **settings,
    )
    if stored.source_digest != digest:
        raise ValueError("gradient source digest does not bind stored moments and inputs")


class FiniteTrainingSequence(StrictEvidenceModel):
    evidence_class: Literal["software_simulation"] = "software_simulation"
    protocol: FiniteRefinementProtocol
    seed: StrictInt = Field(ge=0, le=2)
    initial_parameters: tuple[tuple[StrictFloat, ...], ...]
    occurrence_target_indices: tuple[StrictInt, ...]
    occurrence_site_indices: tuple[tuple[StrictInt, ...], ...]
    target_occupancy: tuple[StrictFloat, ...]
    steps: tuple[FiniteRefinementStep, ...]
    evaluation_seed: StrictInt = Field(ge=0)
    final_evaluation: HorizonTerminalEvidence
    result_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator(
        "initial_parameters",
        "occurrence_target_indices",
        "occurrence_site_indices",
        "target_occupancy",
        "steps",
        mode="before",
    )
    @classmethod
    def freeze_sequences(cls, value):
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def validate_sequence(self):
        protocol = FiniteRefinementProtocol.model_validate_json(self.protocol.model_dump_json())
        roles = spawn_finite_refinement_roles(self.seed)
        if len(self.steps) != 5 or self.evaluation_seed != roles[-1]:
            raise ValueError("sequence requires five ordered checkpoints and the fresh final role")
        if self.result_digest != sequence_result_digest(self):
            raise ValueError("sequence digest must bind the complete evidence")
        target = _checked_site_vector(
            self.target_occupancy, site_count=len(self.target_occupancy), name="target_occupancy"
        )
        if np.any(target < 0.0) or np.any(target > 1.0):
            raise ValueError("target occupancy must lie in [0, 1]")
        groups, sites = self.occurrence_target_indices, self.occurrence_site_indices
        common = dict(
            site_count=len(target),
            batch_size=protocol.batch_size,
            beta=protocol.beta,
            horizon=protocol.horizon,
        )
        current = self.initial_parameters
        for index, nested in enumerate(self.steps):
            step = FiniteRefinementStep.model_validate_json(nested.model_dump_json())
            if (
                step.iteration != index + 1
                or step.occupancy_seed != roles[2 * index]
                or step.gradient_seed != roles[2 * index + 1]
            ):
                raise ValueError("step order and role seeds must follow the predeclared sequence")
            if step.parameter_digest != canonical_sha256(current):
                raise ValueError("step parameters must be the preceding projected checkpoint")
            if step.exact_tables_digest != finite_sampling_tables_digest(
                current, horizon=4, beta=1.0
            ):
                raise ValueError("step exact tables must reconstruct from current parameters")
            occupancy = sample_finite_sweep_terminal_occupancy(
                current, groups, sites, seed=step.occupancy_seed, **common
            )
            if step.occupancy.model_dump(mode="json") != to_json_value(asdict(occupancy)):
                raise ValueError("occupancy source must match deterministic replay")
            reward = tuple(
                2.0 * (observed - exact)
                for observed, exact in zip(occupancy.occupancy, self.target_occupancy, strict=True)
            )
            if step.reward_coefficient != reward:
                raise ValueError("reward must come only from the independent occupancy role")
            gradient = estimate_finite_sweep_grouped_gradient(
                current, groups, sites, reward, seed=step.gradient_seed, **common
            )
            require_gradient_replay(
                step.gradient,
                gradient,
                current,
                groups,
                sites,
                reward,
                seed=step.gradient_seed,
                **common,
            )
            update = project_grouped_parameters(
                current,
                step.gradient.mean,
                learning_rate=protocol.learning_rate,
                parameter_cap=protocol.parameter_cap,
            )
            if step.update.model_dump(mode="json") != to_json_value(asdict(update)):
                raise ValueError(
                    "projected update must reconstruct exactly from stored gradient moments"
                )
            current = step.update.updated_parameters
        final = HorizonTerminalEvidence.model_validate_json(self.final_evaluation.model_dump_json())
        expected = evaluate_finite_sweep_pair(
            self.initial_parameters, current, groups, sites, seed=self.evaluation_seed, **common
        )
        if final != expected:
            raise ValueError("final initial/fifth checkpoint evidence must match held-out replay")
        final.statistics(self.target_occupancy)
        return self


def run_training_sequence(
    initial_parameters,
    occurrence_target_indices,
    occurrence_site_indices,
    target_occupancy,
    *,
    seed: int,
) -> FiniteTrainingSequence:
    """Execute exactly five updates, freeze checkpoint five, then draw final evaluation."""
    protocol = FiniteRefinementProtocol()
    roles = spawn_finite_refinement_roles(seed)
    values = _checked_parameter_matrix(initial_parameters, name="initial_parameters")
    target = _checked_site_vector(
        target_occupancy, site_count=len(target_occupancy), name="target_occupancy"
    )
    if np.any(target < 0) or np.any(target > 1):
        raise ValueError("target occupancy must lie in [0, 1]")
    common = dict(site_count=len(target), batch_size=32768, beta=1.0, horizon=4)
    checked = _request(
        values, occurrence_target_indices, occurrence_site_indices, seed=roles[0], **common
    )
    groups, sites = checked.groups, checked.sites
    initial = _tuple_matrix(values)
    current = initial
    steps = []
    for index in range(5):
        occupancy_seed, gradient_seed = roles[2 * index : 2 * index + 2]
        occupancy = sample_finite_sweep_terminal_occupancy(
            current, groups, sites, seed=occupancy_seed, **common
        )
        reward = tuple(
            2.0 * (observed - exact)
            for observed, exact in zip(occupancy.occupancy, target, strict=True)
        )
        gradient = estimate_finite_sweep_grouped_gradient(
            current, groups, sites, reward, seed=gradient_seed, **common
        )
        update = project_grouped_parameters(
            current, gradient.mean, learning_rate=0.01, parameter_cap=2.0
        )
        steps.append(
            FiniteRefinementStep.model_validate(
                {
                    "iteration": index + 1,
                    "parameter_digest": canonical_sha256(current),
                    "exact_tables_digest": finite_sampling_tables_digest(
                        current, horizon=4, beta=1.0
                    ),
                    "occupancy_seed": occupancy_seed,
                    "gradient_seed": gradient_seed,
                    "occupancy": asdict(occupancy),
                    "reward_coefficient": reward,
                    "gradient": asdict(gradient),
                    "update": asdict(update),
                }
            )
        )
        current = update.updated_parameters
    final = evaluate_finite_sweep_pair(initial, current, groups, sites, seed=roles[-1], **common)
    payload = {
        "evidence_class": "software_simulation",
        "protocol": protocol.model_dump(mode="json"),
        "seed": seed,
        "initial_parameters": initial,
        "occurrence_target_indices": groups,
        "occurrence_site_indices": sites,
        "target_occupancy": tuple(float(x) for x in target),
        "steps": tuple(step.model_dump(mode="json") for step in steps),
        "evaluation_seed": roles[-1],
        "final_evaluation": final.model_dump(mode="json"),
    }
    payload["result_digest"] = sequence_result_digest(payload)
    return FiniteTrainingSequence.model_validate(payload)
