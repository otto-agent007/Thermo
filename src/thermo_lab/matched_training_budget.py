"""Five-update comparison of finite and equilibrium training at matched draw budgets."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator

from thermo_lab.composed_trajectory_refinement import (
    _build_joint_tables,
    _checked_parameter_matrix,
    _checked_site_vector,
    _schedule_digest,
    estimate_equilibrium_grouped_gradient,
    project_grouped_parameters,
    sample_equilibrium_terminal_occupancy,
)
from thermo_lab.composed_trajectory_refinement_results import StrictEvidenceModel, _tuple_json_lists
from thermo_lab.finite_refinement_sequence import FiniteRefinementStep, require_gradient_replay
from thermo_lab.finite_sweep_sampling import (
    _request,
    estimate_finite_sweep_grouped_gradient,
    evaluate_finite_sweep_pair,
    finite_sampling_tables_digest,
    sample_finite_sweep_terminal_occupancy,
)
from thermo_lab.frozen_pair_finite_sweeps import HorizonTerminalEvidence
from thermo_lab.hashing import canonical_sha256, to_json_value

ArmName = Literal["finite_k4", "equilibrium"]
ARM_NAMES = ("finite_k4", "equilibrium")


class ComparisonProtocol(StrictEvidenceModel):
    identity_version: Literal["matched_training_budget.v1"] = "matched_training_budget.v1"
    step_count: StrictInt = Field(default=5, ge=5, le=5)
    batch_size: StrictInt = Field(default=32768, ge=32768, le=32768)
    evaluation_horizon: StrictInt = Field(default=4, ge=4, le=4)
    learning_rate: StrictFloat = Field(default=0.01, ge=0.01, le=0.01)
    parameter_cap: StrictFloat = Field(default=2.0, ge=2.0, le=2.0)
    beta: StrictFloat = Field(default=1.0, ge=1.0, le=1.0)
    dtype: Literal["float64"] = "float64"
    role_policy: Literal[
        "SeedSequence([0x4D3442, seed]).spawn(21); finite roles 0:10; "
        "equilibrium roles 10:20; evaluation role 20"
    ] = (
        "SeedSequence([0x4D3442, seed]).spawn(21); finite roles 0:10; "
        "equilibrium roles 10:20; evaluation role 20"
    )
    training_policy: Literal[
        "arm-specific occupancy and gradient laws; independent occupancy/gradient roles; "
        "same-parent independent non-propagated references; sum shared occurrences first"
    ] = (
        "arm-specific occupancy and gradient laws; independent occupancy/gradient roles; "
        "same-parent independent non-propagated references; sum shared occurrences first"
    )
    checkpoint_policy: Literal["always_fifth_update"] = "always_fifth_update"
    evaluation_policy: Literal[
        "fresh K4 paired initial/finite, initial/equilibrium, equilibrium/finite; shared stream"
    ] = "fresh K4 paired initial/finite, initial/equilibrium, equilibrium/finite; shared stream"
    primary_contrast: Literal["finite_minus_equilibrium"] = "finite_minus_equilibrium"
    cost_policy: Literal["matched_draws_and_updates_not_hardware_cost"] = (
        "matched_draws_and_updates_not_hardware_cost"
    )
    objective_policy: Literal["unbiased_order_two_u_statistic"] = "unbiased_order_two_u_statistic"
    uncertainty_policy: Literal["paired_delete_one_jackknife_normal_95_approximate"] = (
        "paired_delete_one_jackknife_normal_95_approximate"
    )
    scientific_status: Literal["descriptive_non_gating"] = "descriptive_non_gating"
    replay_relative_tolerance: StrictFloat = Field(default=1e-12, ge=1e-12, le=1e-12)
    replay_absolute_tolerance: StrictFloat = Field(default=1e-10, ge=1e-10, le=1e-10)


def spawn_comparison_roles(seed: int) -> tuple[int, ...]:
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("comparison requires checked seed 0, 1, or 2")
    children = np.random.SeedSequence([0x4D3442, seed]).spawn(21)
    roles = tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)
    if len(set(roles)) != 21:
        raise ValueError("comparison role seeds must be distinct")
    return roles


class TrainingArm(StrictEvidenceModel):
    name: ArmName
    steps: tuple[FiniteRefinementStep, ...]

    @field_validator("steps", mode="before")
    @classmethod
    def freeze_steps(cls, value):
        return _tuple_json_lists(value)


def comparison_digest(value) -> str:
    payload = (
        value.model_dump(mode="json") if isinstance(value, StrictEvidenceModel) else dict(value)
    )
    payload.pop("result_digest", None)
    return canonical_sha256(
        {"identity_version": "training_comparison_result.v1", "result": payload}
    )


def _table_digest(parameters, name):
    if name == "finite_k4":
        return finite_sampling_tables_digest(parameters, horizon=4, beta=1.0)
    values = _checked_parameter_matrix(parameters, name="parameters")
    return canonical_sha256(
        {
            "identity_version": "comparison_equilibrium_tables.v1",
            "evidence_class": "exact_reference",
            "parameters": parameters,
            "beta": 1.0,
            "probabilities": _build_joint_tables(values, beta=1.0).tolist(),
            "score_difference": "beta_times_main_minus_reference_features",
        }
    )


@lru_cache(maxsize=64)
def _equilibrium_occupancy(parameters, groups, sites, site_count, seed):
    return sample_equilibrium_terminal_occupancy(
        parameters, groups, sites, site_count=site_count, batch_size=32768, seed=seed, beta=1.0
    )


@lru_cache(maxsize=64)
def _equilibrium_gradient(parameters, groups, sites, reward, seed):
    return estimate_equilibrium_grouped_gradient(
        parameters,
        groups,
        sites,
        reward,
        site_count=len(reward),
        batch_size=32768,
        seed=seed,
        beta=1.0,
    )


def _occupancy(parameters, groups, sites, site_count, seed, name):
    if name == "equilibrium":
        return _equilibrium_occupancy(parameters, groups, sites, site_count, seed)
    return sample_finite_sweep_terminal_occupancy(
        parameters,
        groups,
        sites,
        site_count=site_count,
        batch_size=32768,
        seed=seed,
        beta=1.0,
        horizon=4,
    )


def _gradient(parameters, groups, sites, reward, seed, name):
    if name == "equilibrium":
        return _equilibrium_gradient(parameters, groups, sites, reward, seed)
    return estimate_finite_sweep_grouped_gradient(
        parameters,
        groups,
        sites,
        reward,
        site_count=len(reward),
        batch_size=32768,
        seed=seed,
        beta=1.0,
        horizon=4,
    )


def _check_gradient(stored, expected, parameters, groups, sites, reward, seed, name):
    if name == "finite_k4":
        require_gradient_replay(
            stored,
            expected,
            parameters,
            groups,
            sites,
            reward,
            seed=seed,
            site_count=len(reward),
            batch_size=32768,
            beta=1.0,
            horizon=4,
        )
        return
    if stored.sample_count != 32768 or any(
        np.asarray(getattr(stored, field)).shape != (len(parameters), 9)
        or not np.allclose(getattr(stored, field), getattr(expected, field), rtol=1e-12, atol=1e-10)
        for field in ("component_sum", "component_sum_squares")
    ):
        raise ValueError("equilibrium gradient moments must match replay")
    digest = canonical_sha256(
        {
            "identity_version": "composed_equilibrium_grouped_gradient_source.v1",
            "sample_count": 32768,
            "component_sum": stored.component_sum,
            "component_sum_squares": stored.component_sum_squares,
            "seed": seed,
            "beta": 1.0,
            "parameter_digest": canonical_sha256(parameters),
            "schedule_digest": _schedule_digest(groups, sites, site_count=len(reward)),
            "reward_coefficient": reward,
            "reference_policy": "independent_same_parent_non_propagated",
        }
    )
    if stored.source_digest != digest:
        raise ValueError("equilibrium gradient digest must bind stored moments")


def _pair(before, after, groups, sites, site_count, seed):
    return evaluate_finite_sweep_pair(
        before,
        after,
        groups,
        sites,
        site_count=site_count,
        batch_size=32768,
        seed=seed,
        beta=1.0,
        horizon=4,
    )


class TrainingComparison(StrictEvidenceModel):
    evidence_class: Literal["software_simulation"] = "software_simulation"
    protocol: ComparisonProtocol
    seed: StrictInt = Field(ge=0, le=2)
    initial_parameters: tuple[tuple[StrictFloat, ...], ...]
    occurrence_target_indices: tuple[StrictInt, ...]
    occurrence_site_indices: tuple[tuple[StrictInt, ...], ...]
    target_occupancy: tuple[StrictFloat, ...]
    arms: tuple[TrainingArm, ...]
    evaluation_seed: StrictInt = Field(ge=0)
    initial_finite: HorizonTerminalEvidence
    initial_equilibrium: HorizonTerminalEvidence
    primary_evaluation: HorizonTerminalEvidence
    result_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator(
        "initial_parameters",
        "occurrence_target_indices",
        "occurrence_site_indices",
        "target_occupancy",
        "arms",
        mode="before",
    )
    @classmethod
    def freeze_sequences(cls, value):
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def reconstruct(self):
        ComparisonProtocol.model_validate_json(self.protocol.model_dump_json())
        if self.result_digest != comparison_digest(self):
            raise ValueError("comparison digest must bind complete evidence")
        roles = spawn_comparison_roles(self.seed)
        if self.evaluation_seed != roles[-1] or tuple(a.name for a in self.arms) != ARM_NAMES:
            raise ValueError("comparison requires ordered arms and fresh final evaluation role")
        target = _checked_site_vector(
            self.target_occupancy, site_count=len(self.target_occupancy), name="target_occupancy"
        )
        if np.any(target < 0) or np.any(target > 1):
            raise ValueError("target occupancy must lie in [0,1]")
        groups, sites = self.occurrence_target_indices, self.occurrence_site_indices
        _request(
            self.initial_parameters,
            groups,
            sites,
            site_count=len(target),
            batch_size=32768,
            seed=self.seed,
            beta=1.0,
            horizon=4,
        )
        finals = []
        for arm_index, arm in enumerate(self.arms):
            current = self.initial_parameters
            if len(arm.steps) != 5:
                raise ValueError("both arms require exactly five updates")
            for index, nested in enumerate(arm.steps):
                step = FiniteRefinementStep.model_validate_json(nested.model_dump_json())
                offset = arm_index * 10 + index * 2
                if (
                    step.iteration != index + 1
                    or (step.occupancy_seed, step.gradient_seed) != roles[offset : offset + 2]
                    or step.parameter_digest != canonical_sha256(current)
                    or step.exact_tables_digest != _table_digest(current, arm.name)
                ):
                    raise ValueError(
                        "arm steps must follow checked roles, law, and parameter chain"
                    )
                occupancy = _occupancy(
                    current, groups, sites, len(target), step.occupancy_seed, arm.name
                )
                if step.occupancy.model_dump(mode="json") != to_json_value(asdict(occupancy)):
                    raise ValueError("arm occupancy must match replay")
                reward = tuple(
                    2.0 * (a - b)
                    for a, b in zip(occupancy.occupancy, self.target_occupancy, strict=True)
                )
                if step.reward_coefficient != reward:
                    raise ValueError("reward must use independent arm-specific occupancy")
                gradient = _gradient(current, groups, sites, reward, step.gradient_seed, arm.name)
                _check_gradient(
                    step.gradient,
                    gradient,
                    current,
                    groups,
                    sites,
                    reward,
                    step.gradient_seed,
                    arm.name,
                )
                update = project_grouped_parameters(
                    current, step.gradient.mean, learning_rate=0.01, parameter_cap=2.0
                )
                if step.update.model_dump(mode="json") != to_json_value(asdict(update)):
                    raise ValueError("projected arm update must reconstruct from stored gradient")
                current = step.update.updated_parameters
            finals.append(current)
        for stored, before, after in (
            (self.initial_finite, self.initial_parameters, finals[0]),
            (self.initial_equilibrium, self.initial_parameters, finals[1]),
            (self.primary_evaluation, finals[1], finals[0]),
        ):
            checked = HorizonTerminalEvidence.model_validate_json(stored.model_dump_json())
            if checked != _pair(before, after, groups, sites, len(target), self.evaluation_seed):
                raise ValueError("K4 paired evaluation must match fresh fixed-checkpoint replay")
            checked.statistics(self.target_occupancy)
        return self


def _run_comparison_payload(initial_parameters, groups, sites, target_occupancy, *, seed):
    roles = spawn_comparison_roles(seed)
    target = _checked_site_vector(
        target_occupancy, site_count=len(target_occupancy), name="target_occupancy"
    )
    if np.any(target < 0) or np.any(target > 1):
        raise ValueError("target occupancy must lie in [0,1]")
    request = _request(
        initial_parameters,
        groups,
        sites,
        site_count=len(target),
        batch_size=32768,
        seed=seed,
        beta=1.0,
        horizon=4,
    )
    initial, groups, sites = request.parameters, request.groups, request.sites
    arms = []
    for arm_index, name in enumerate(ARM_NAMES):
        current = initial
        steps = []
        for index in range(5):
            offset = arm_index * 10 + index * 2
            occ_seed, grad_seed = roles[offset : offset + 2]
            occupancy = _occupancy(current, groups, sites, len(target), occ_seed, name)
            reward = tuple(
                float(2.0 * (a - b)) for a, b in zip(occupancy.occupancy, target, strict=True)
            )
            gradient = _gradient(current, groups, sites, reward, grad_seed, name)
            update = project_grouped_parameters(
                current, gradient.mean, learning_rate=0.01, parameter_cap=2.0
            )
            steps.append(
                FiniteRefinementStep.model_validate(
                    {
                        "iteration": index + 1,
                        "parameter_digest": canonical_sha256(current),
                        "exact_tables_digest": _table_digest(current, name),
                        "occupancy_seed": occ_seed,
                        "gradient_seed": grad_seed,
                        "occupancy": asdict(occupancy),
                        "reward_coefficient": reward,
                        "gradient": asdict(gradient),
                        "update": asdict(update),
                    }
                )
            )
            current = update.updated_parameters
        arms.append(TrainingArm(name=name, steps=tuple(steps)))
    finite, equilibrium = (a.steps[-1].update.updated_parameters for a in arms)
    payload = dict(
        evidence_class="software_simulation",
        protocol=ComparisonProtocol().model_dump(mode="json"),
        seed=seed,
        initial_parameters=initial,
        occurrence_target_indices=groups,
        occurrence_site_indices=sites,
        target_occupancy=tuple(float(x) for x in target),
        arms=tuple(a.model_dump(mode="json") for a in arms),
        evaluation_seed=roles[-1],
        initial_finite=_pair(initial, finite, groups, sites, len(target), roles[-1]).model_dump(
            mode="json"
        ),
        initial_equilibrium=_pair(
            initial, equilibrium, groups, sites, len(target), roles[-1]
        ).model_dump(mode="json"),
        primary_evaluation=_pair(
            equilibrium, finite, groups, sites, len(target), roles[-1]
        ).model_dump(mode="json"),
    )
    payload["result_digest"] = comparison_digest(payload)
    return payload


def run_comparison(initial_parameters, groups, sites, target_occupancy, *, seed):
    return TrainingComparison.model_validate(
        _run_comparison_payload(initial_parameters, groups, sites, target_occupancy, seed=seed)
    )


def clear_generation_caches():
    """Start a timed study run cold; never clear caches during audit reconstruction."""
    from thermo_lab import finite_sweep_sampling as sampling

    for function in (
        sampling._tables,
        sampling._occupancy,
        sampling._gradient,
        sampling._paired,
        _equilibrium_occupancy,
        _equilibrium_gradient,
    ):
        function.cache_clear()


def declared_work(occurrence_count: int) -> dict:
    if type(occurrence_count) is not int or not 1 <= occurrence_count <= 500:
        raise ValueError("work requires 1..500 occurrences")
    training = 5 * 32768 * occurrence_count * 3
    evaluation = 3 * 32768 * occurrence_count * 2
    return {
        "training_endpoint_draws_per_arm": training,
        "three_pair_evaluation_endpoint_draws": evaluation,
        "total_endpoint_draws": 2 * training + evaluation,
        "finite_training_complete_sweep_equivalents": 4 * training,
        "finite_training_free_pbit_update_equivalents": 12 * training,
        "equilibrium_training_complete_sweep_equivalents": None,
        "evaluation_complete_sweep_equivalents": 4 * evaluation,
        "evaluation_free_pbit_update_equivalents": 12 * evaluation,
    }
