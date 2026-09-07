"""One-step refinement for the exact-categorical trajectory fixture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from thermo_lab.thermodynamic_kernel import KernelParameters, equilibrium_joint_conditional
from thermo_lab.trajectory_reinforce import (
    ExactTrajectoryReference,
    TerminalLaw,
    TrajectoryFixture,
    terminal_law,
)


@dataclass(frozen=True)
class ProjectedSharedUpdate:
    """One descent step before and after projection onto the parameter box."""

    raw_parameters: tuple[float, ...]
    updated_parameters: tuple[float, ...]
    cap_active_mask: tuple[bool, ...]
    cap_active_parameter_count: int


@dataclass(frozen=True)
class ExactObjectiveEvaluation:
    """Exact terminal model law and declared scalar objective at one shared vector."""

    model_law: TerminalLaw
    objective: float


@dataclass(frozen=True)
class OneStepRefinement:
    """Auditable exact evaluation of one sampled shared-gradient update."""

    sampled_shared_gradient: tuple[float, ...]
    learning_rate: float
    parameter_cap: float
    update: ProjectedSharedUpdate
    model_law_before: TerminalLaw
    model_law_after: TerminalLaw
    objective_before: float
    objective_after: float
    objective_improvement: float
    relative_objective_improvement: float
    objective_improved: bool
    bounds_satisfied: bool


def _checked_vector(values: object, *, name: str) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != 9:
        raise ValueError(f"{name} must contain exactly nine values")
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} must not contain booleans")
    if any(not isinstance(value, Real) or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite real values")
    return tuple(float(value) for value in values)


def _checked_positive_scalar(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real value")
    checked = float(value)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def project_shared_parameters(
    *,
    initial_parameters: object,
    gradient: object,
    learning_rate: object,
    parameter_cap: object,
) -> ProjectedSharedUpdate:
    """Apply one shared-parameter descent step and project it onto the checked box."""

    initial = _checked_vector(initial_parameters, name="initial_parameters")
    checked_gradient = _checked_vector(gradient, name="gradient")
    rate = _checked_positive_scalar(learning_rate, name="learning_rate")
    cap = _checked_positive_scalar(parameter_cap, name="parameter_cap")
    if any(abs(parameter) > cap for parameter in initial):
        raise ValueError("initial_parameters must be within the parameter cap")
    raw = tuple(
        parameter - rate * component
        for parameter, component in zip(initial, checked_gradient, strict=True)
    )
    if any(not math.isfinite(parameter) for parameter in raw):
        raise ValueError("raw parameters must remain finite before projection")
    updated = tuple(max(-cap, min(cap, parameter)) for parameter in raw)
    active = tuple(
        candidate != projected for candidate, projected in zip(raw, updated, strict=True)
    )
    return ProjectedSharedUpdate(
        raw_parameters=raw,
        updated_parameters=updated,
        cap_active_mask=active,
        cap_active_parameter_count=sum(active),
    )


def evaluate_exact_shared_objective(
    *, fixture: TrajectoryFixture, shared_parameters: object
) -> ExactObjectiveEvaluation:
    """Evaluate the exact objective with one parameter vector tied across both factors."""

    if not isinstance(fixture, TrajectoryFixture):
        raise TypeError("fixture must be a TrajectoryFixture")
    values = _checked_vector(shared_parameters, name="shared_parameters")
    if any(abs(parameter) > fixture.parameter_cap for parameter in values):
        raise ValueError("shared_parameters must be within the parameter cap")
    parameters = KernelParameters(values)  # type: ignore[arg-type]
    joint = equilibrium_joint_conditional(parameters, beta=fixture.beta)
    target_law = terminal_law(
        (fixture.target_conditional, fixture.target_conditional),
        fixture.occurrences,
        fixture.initial_state,
    )
    model_law = terminal_law((joint, joint), fixture.occurrences, fixture.initial_state)
    objective = math.fsum(
        (model_law.occupancy[index] - target_law.occupancy[index]) ** 2 for index in range(3)
    )
    return ExactObjectiveEvaluation(model_law=model_law, objective=objective)


def build_one_step_refinement(
    *,
    fixture: TrajectoryFixture,
    exact: ExactTrajectoryReference,
    sampled_shared_gradient: object,
    learning_rate: object,
) -> OneStepRefinement:
    """Apply one sampled descent step and exactly measure its objective change."""

    if not isinstance(exact, ExactTrajectoryReference):
        raise TypeError("exact must be an ExactTrajectoryReference")
    gradient = _checked_vector(sampled_shared_gradient, name="sampled_shared_gradient")
    rate = _checked_positive_scalar(learning_rate, name="learning_rate")
    before = evaluate_exact_shared_objective(
        fixture=fixture,
        shared_parameters=fixture.model_parameters.values,
    )
    if before.objective != exact.objective:
        raise ValueError("exact reference objective disagrees with the checked fixture")
    update = project_shared_parameters(
        initial_parameters=fixture.model_parameters.values,
        gradient=gradient,
        learning_rate=rate,
        parameter_cap=fixture.parameter_cap,
    )
    after = evaluate_exact_shared_objective(
        fixture=fixture,
        shared_parameters=update.updated_parameters,
    )
    improvement = math.fsum((before.objective, -after.objective))
    bounds_satisfied = all(
        abs(parameter) <= fixture.parameter_cap for parameter in update.updated_parameters
    )
    return OneStepRefinement(
        sampled_shared_gradient=gradient,
        learning_rate=rate,
        parameter_cap=float(fixture.parameter_cap),
        update=update,
        model_law_before=before.model_law,
        model_law_after=after.model_law,
        objective_before=before.objective,
        objective_after=after.objective,
        objective_improvement=improvement,
        relative_objective_improvement=improvement / before.objective,
        objective_improved=improvement > 0.0,
        bounds_satisfied=bounds_satisfied,
    )
