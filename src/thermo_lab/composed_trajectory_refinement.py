"""Equilibrium trajectory-gradient primitives for composed PAsymSwap refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from thermo_lab.hashing import canonical_sha256
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    equilibrium_joint_conditional,
    sufficient_statistics,
)

_N_PARAMETERS = 9
_N_JOINT_OUTCOMES = 8
_N_VISIBLE_OUTCOMES = 4


@dataclass(frozen=True)
class GroupedParameterUpdate:
    """One projected update over target-hash-shared parameter vectors."""

    raw_parameters: tuple[tuple[float, ...], ...]
    updated_parameters: tuple[tuple[float, ...], ...]
    cap_active_mask: tuple[tuple[bool, ...], ...]
    cap_active_parameter_count: int
    bounds_satisfied: bool
    update_digest: str


@dataclass(frozen=True)
class TerminalOccupancySource:
    """Integer sufficient statistics for one terminal occupancy estimate."""

    sample_count: int
    occupancy_counts: tuple[int, ...]
    source_digest: str

    @property
    def occupancy(self) -> tuple[float, ...]:
        return tuple(count / self.sample_count for count in self.occupancy_counts)


@dataclass(frozen=True)
class GroupedGradientSource:
    """Trajectory-level grouped gradient sufficient statistics."""

    sample_count: int
    component_sum: tuple[tuple[float, ...], ...]
    component_sum_squares: tuple[tuple[float, ...], ...]
    source_digest: str

    @property
    def mean(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(value / self.sample_count for value in row) for row in self.component_sum
        )


@dataclass(frozen=True)
class PairedObjectiveEvaluation:
    """Held-out before/after occupancy objective using common random numbers."""

    before: TerminalOccupancySource
    after: TerminalOccupancySource
    target_occupancy: tuple[float, ...]
    objective_before: float
    objective_after: float
    objective_improvement: float
    objective_improved: bool
    result_digest: str


def _checked_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return checked


def _checked_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _checked_seed(seed: object) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _checked_parameter_matrix(values: object, *, name: str) -> NDArray[np.float64]:
    try:
        objects = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if objects.ndim != 2 or objects.shape[1] != _N_PARAMETERS or objects.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n_groups, {_N_PARAMETERS})")
    if any(isinstance(value, (bool, np.bool_)) for value in objects.flat):
        raise ValueError(f"{name} must not contain booleans")
    if any(not isinstance(value, Real) for value in objects.flat):
        raise ValueError(f"{name} must contain finite real numbers")
    checked = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(checked, dtype=np.float64, copy=True)


def _checked_schedule(
    occurrence_target_indices: object,
    occurrence_site_indices: object,
    *,
    group_count: int,
    site_count: object,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    checked_site_count = _checked_positive_int(site_count, name="site_count")
    targets = np.asarray(occurrence_target_indices)
    sites = np.asarray(occurrence_site_indices)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("occurrence_target_indices must be a non-empty vector")
    if sites.shape != (targets.size, 2):
        raise ValueError("occurrence_site_indices must have shape (n_occurrences, 2)")
    if not np.issubdtype(targets.dtype, np.integer) or not np.issubdtype(sites.dtype, np.integer):
        raise ValueError("occurrence indices must use integer dtypes")
    targets = targets.astype(np.int64, copy=False)
    sites = sites.astype(np.int64, copy=False)
    if np.any(targets < 0) or np.any(targets >= group_count):
        raise ValueError("occurrence_target_indices must reference existing parameter groups")
    if np.any(sites < 0) or np.any(sites >= checked_site_count):
        raise ValueError("occurrence_site_indices must reference existing sites")
    if np.any(sites[:, 0] == sites[:, 1]):
        raise ValueError("each occurrence must use two distinct sites")
    return targets, sites, checked_site_count


def _checked_site_vector(values: object, *, site_count: int, name: str) -> NDArray[np.float64]:
    try:
        objects = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite site vector") from error
    if objects.shape != (site_count,):
        raise ValueError(f"{name} must have shape ({site_count},)")
    if any(isinstance(value, (bool, np.bool_)) for value in objects.flat):
        raise ValueError(f"{name} must not contain booleans")
    if any(not isinstance(value, Real) for value in objects.flat):
        raise ValueError(f"{name} must contain finite real numbers")
    checked = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(checked, dtype=np.float64, copy=True)


def _tuple_matrix(values: NDArray[np.float64]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def _build_joint_tables(
    parameters: NDArray[np.float64], *, beta: float
) -> NDArray[np.float64]:
    tables = np.empty((parameters.shape[0], 4, _N_JOINT_OUTCOMES), dtype=np.float64)
    for group, row in enumerate(parameters):
        tables[group] = equilibrium_joint_conditional(
            KernelParameters(tuple(float(value) for value in row)),
            beta=beta,
        )
    return tables


def _feature_table() -> NDArray[np.float64]:
    return np.asarray(
        [
            [
                sufficient_statistics(parent, outcome // _N_VISIBLE_OUTCOMES, outcome % 4)
                for outcome in range(_N_JOINT_OUTCOMES)
            ]
            for parent in range(4)
        ],
        dtype=np.float64,
    )


def _inverse_cdf_rows(
    probabilities: NDArray[np.float64], uniforms: NDArray[np.float64]
) -> NDArray[np.int64]:
    cumulative = np.cumsum(probabilities, axis=1, dtype=np.float64)
    cumulative[:, -1] = 1.0
    return np.sum(uniforms[:, None] >= cumulative[:, :-1], axis=1).astype(
        np.int64, copy=False
    )


def _initial_states(*, batch_size: int, site_count: int) -> NDArray[np.uint8]:
    states = np.zeros((batch_size, site_count), dtype=np.uint8)
    states[:, 0] = 1
    return states


def _terminal_source(
    states: NDArray[np.uint8],
    *,
    seed: int,
    beta: float,
    parameter_digest: str,
    schedule_digest: str,
    role: str,
) -> TerminalOccupancySource:
    counts = tuple(int(value) for value in states.sum(axis=0, dtype=np.int64))
    payload = {
        "identity_version": "composed_equilibrium_terminal_occupancy_source.v1",
        "sample_count": int(states.shape[0]),
        "occupancy_counts": counts,
        "seed": seed,
        "beta": beta,
        "parameter_digest": parameter_digest,
        "schedule_digest": schedule_digest,
        "role": role,
    }
    return TerminalOccupancySource(
        sample_count=states.shape[0],
        occupancy_counts=counts,
        source_digest=canonical_sha256(payload),
    )


def _schedule_digest(
    targets: NDArray[np.int64], sites: NDArray[np.int64], *, site_count: int
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_equilibrium_schedule.v1",
            "site_count": site_count,
            "occurrence_target_indices": tuple(int(value) for value in targets),
            "occurrence_site_indices": tuple(
                tuple(int(value) for value in row) for row in sites
            ),
        }
    )


def project_grouped_parameters(
    initial_parameters: object,
    gradient: object,
    *,
    learning_rate: object,
    parameter_cap: object,
) -> GroupedParameterUpdate:
    """Apply one grouped descent step and project every parameter onto a box."""

    initial = _checked_parameter_matrix(initial_parameters, name="initial_parameters")
    checked_gradient = _checked_parameter_matrix(gradient, name="gradient")
    if checked_gradient.shape != initial.shape:
        raise ValueError("gradient must have the same shape as initial_parameters")
    rate = _checked_positive_float(learning_rate, name="learning_rate")
    cap = _checked_positive_float(parameter_cap, name="parameter_cap")
    if np.any(np.abs(initial) > cap):
        raise ValueError("initial_parameters must already satisfy the parameter cap")
    raw = initial - rate * checked_gradient
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw parameters must remain finite before projection")
    updated = np.clip(raw, -cap, cap)
    active = raw != updated
    raw_tuple = _tuple_matrix(raw)
    updated_tuple = _tuple_matrix(updated)
    active_tuple = tuple(tuple(bool(value) for value in row) for row in active)
    count = int(np.count_nonzero(active))
    bounds_satisfied = bool(np.all(np.abs(updated) <= cap))
    digest = canonical_sha256(
        {
            "identity_version": "composed_grouped_parameter_update.v1",
            "initial_parameters": _tuple_matrix(initial),
            "gradient": _tuple_matrix(checked_gradient),
            "learning_rate": rate,
            "parameter_cap": cap,
            "raw_parameters": raw_tuple,
            "updated_parameters": updated_tuple,
            "cap_active_mask": active_tuple,
            "cap_active_parameter_count": count,
            "bounds_satisfied": bounds_satisfied,
        }
    )
    return GroupedParameterUpdate(
        raw_parameters=raw_tuple,
        updated_parameters=updated_tuple,
        cap_active_mask=active_tuple,
        cap_active_parameter_count=count,
        bounds_satisfied=bounds_satisfied,
        update_digest=digest,
    )


def sample_equilibrium_terminal_occupancy(
    parameter_vectors: object,
    occurrence_target_indices: object,
    occurrence_site_indices: object,
    *,
    site_count: object,
    batch_size: object,
    seed: object,
    beta: object,
) -> TerminalOccupancySource:
    """Sample one equilibrium composed program and retain terminal occupancy counts."""

    parameters = _checked_parameter_matrix(parameter_vectors, name="parameter_vectors")
    targets, sites, checked_site_count = _checked_schedule(
        occurrence_target_indices,
        occurrence_site_indices,
        group_count=parameters.shape[0],
        site_count=site_count,
    )
    checked_batch_size = _checked_positive_int(batch_size, name="batch_size")
    checked_seed = _checked_seed(seed)
    checked_beta = _checked_positive_float(beta, name="beta")
    tables = _build_joint_tables(parameters, beta=checked_beta)
    states = _initial_states(batch_size=checked_batch_size, site_count=checked_site_count)
    rng = np.random.Generator(np.random.PCG64(checked_seed))

    for target, (left, right) in zip(targets, sites, strict=True):
        parent = 2 * states[:, left].astype(np.int64) + states[:, right].astype(np.int64)
        probabilities = tables[target, parent]
        outcome = _inverse_cdf_rows(probabilities, rng.random(checked_batch_size))
        output = outcome % _N_VISIBLE_OUTCOMES
        states[:, left] = (output // 2).astype(np.uint8, copy=False)
        states[:, right] = (output % 2).astype(np.uint8, copy=False)

    return _terminal_source(
        states,
        seed=checked_seed,
        beta=checked_beta,
        parameter_digest=canonical_sha256(_tuple_matrix(parameters)),
        schedule_digest=_schedule_digest(targets, sites, site_count=checked_site_count),
        role="standalone_terminal_occupancy",
    )


def estimate_equilibrium_grouped_gradient(
    parameter_vectors: object,
    occurrence_target_indices: object,
    occurrence_site_indices: object,
    reward_coefficient: object,
    *,
    site_count: object,
    batch_size: object,
    seed: object,
    beta: object,
) -> GroupedGradientSource:
    """Estimate grouped trajectory gradients with independent same-parent references."""

    parameters = _checked_parameter_matrix(parameter_vectors, name="parameter_vectors")
    targets, sites, checked_site_count = _checked_schedule(
        occurrence_target_indices,
        occurrence_site_indices,
        group_count=parameters.shape[0],
        site_count=site_count,
    )
    reward_vector = _checked_site_vector(
        reward_coefficient,
        site_count=checked_site_count,
        name="reward_coefficient",
    )
    checked_batch_size = _checked_positive_int(batch_size, name="batch_size")
    checked_seed = _checked_seed(seed)
    checked_beta = _checked_positive_float(beta, name="beta")
    tables = _build_joint_tables(parameters, beta=checked_beta)
    features = _feature_table()
    states = _initial_states(batch_size=checked_batch_size, site_count=checked_site_count)
    grouped_score = np.zeros(
        (checked_batch_size, parameters.shape[0], _N_PARAMETERS), dtype=np.float64
    )
    main_seed, reference_seed = np.random.SeedSequence(checked_seed).spawn(2)
    main_rng = np.random.Generator(np.random.PCG64(main_seed))
    reference_rng = np.random.Generator(np.random.PCG64(reference_seed))

    for target, (left, right) in zip(targets, sites, strict=True):
        parent = 2 * states[:, left].astype(np.int64) + states[:, right].astype(np.int64)
        probabilities = tables[target, parent]
        main_outcome = _inverse_cdf_rows(probabilities, main_rng.random(checked_batch_size))
        reference_outcome = _inverse_cdf_rows(
            probabilities,
            reference_rng.random(checked_batch_size),
        )
        grouped_score[:, target, :] += checked_beta * (
            features[parent, main_outcome] - features[parent, reference_outcome]
        )
        output = main_outcome % _N_VISIBLE_OUTCOMES
        states[:, left] = (output // 2).astype(np.uint8, copy=False)
        states[:, right] = (output % 2).astype(np.uint8, copy=False)

    reward = states.astype(np.float64, copy=False) @ reward_vector
    contributions = grouped_score * reward[:, None, None]
    component_sum = contributions.sum(axis=0, dtype=np.float64)
    component_sum_squares = (contributions * contributions).sum(axis=0, dtype=np.float64)
    sum_tuple = _tuple_matrix(component_sum)
    squares_tuple = _tuple_matrix(component_sum_squares)
    digest = canonical_sha256(
        {
            "identity_version": "composed_equilibrium_grouped_gradient_source.v1",
            "sample_count": checked_batch_size,
            "component_sum": sum_tuple,
            "component_sum_squares": squares_tuple,
            "seed": checked_seed,
            "beta": checked_beta,
            "parameter_digest": canonical_sha256(_tuple_matrix(parameters)),
            "schedule_digest": _schedule_digest(
                targets, sites, site_count=checked_site_count
            ),
            "reward_coefficient": tuple(float(value) for value in reward_vector),
            "reference_policy": "independent_same_parent_non_propagated",
        }
    )
    return GroupedGradientSource(
        sample_count=checked_batch_size,
        component_sum=sum_tuple,
        component_sum_squares=squares_tuple,
        source_digest=digest,
    )


def evaluate_paired_equilibrium_objective(
    initial_parameters: object,
    updated_parameters: object,
    occurrence_target_indices: object,
    occurrence_site_indices: object,
    target_occupancy: object,
    *,
    site_count: object,
    batch_size: object,
    seed: object,
    beta: object,
) -> PairedObjectiveEvaluation:
    """Evaluate held-out before/after objectives using one shared uniform stream."""

    before_parameters = _checked_parameter_matrix(
        initial_parameters, name="initial_parameters"
    )
    after_parameters = _checked_parameter_matrix(
        updated_parameters, name="updated_parameters"
    )
    if after_parameters.shape != before_parameters.shape:
        raise ValueError("updated_parameters must have the same shape as initial_parameters")
    targets, sites, checked_site_count = _checked_schedule(
        occurrence_target_indices,
        occurrence_site_indices,
        group_count=before_parameters.shape[0],
        site_count=site_count,
    )
    target_vector = _checked_site_vector(
        target_occupancy,
        site_count=checked_site_count,
        name="target_occupancy",
    )
    if np.any(target_vector < 0.0) or np.any(target_vector > 1.0):
        raise ValueError("target_occupancy values must lie in [0, 1]")
    checked_batch_size = _checked_positive_int(batch_size, name="batch_size")
    checked_seed = _checked_seed(seed)
    checked_beta = _checked_positive_float(beta, name="beta")
    before_tables = _build_joint_tables(before_parameters, beta=checked_beta)
    after_tables = _build_joint_tables(after_parameters, beta=checked_beta)
    before_states = _initial_states(
        batch_size=checked_batch_size, site_count=checked_site_count
    )
    after_states = before_states.copy()
    rng = np.random.Generator(np.random.PCG64(checked_seed))

    for target, (left, right) in zip(targets, sites, strict=True):
        uniforms = rng.random(checked_batch_size)
        before_parent = (
            2 * before_states[:, left].astype(np.int64)
            + before_states[:, right].astype(np.int64)
        )
        after_parent = (
            2 * after_states[:, left].astype(np.int64)
            + after_states[:, right].astype(np.int64)
        )
        before_outcome = _inverse_cdf_rows(
            before_tables[target, before_parent], uniforms
        )
        after_outcome = _inverse_cdf_rows(after_tables[target, after_parent], uniforms)
        before_output = before_outcome % _N_VISIBLE_OUTCOMES
        after_output = after_outcome % _N_VISIBLE_OUTCOMES
        before_states[:, left] = (before_output // 2).astype(np.uint8, copy=False)
        before_states[:, right] = (before_output % 2).astype(np.uint8, copy=False)
        after_states[:, left] = (after_output // 2).astype(np.uint8, copy=False)
        after_states[:, right] = (after_output % 2).astype(np.uint8, copy=False)

    schedule_digest = _schedule_digest(targets, sites, site_count=checked_site_count)
    before = _terminal_source(
        before_states,
        seed=checked_seed,
        beta=checked_beta,
        parameter_digest=canonical_sha256(_tuple_matrix(before_parameters)),
        schedule_digest=schedule_digest,
        role="held_out_before",
    )
    after = _terminal_source(
        after_states,
        seed=checked_seed,
        beta=checked_beta,
        parameter_digest=canonical_sha256(_tuple_matrix(after_parameters)),
        schedule_digest=schedule_digest,
        role="held_out_after",
    )
    before_occupancy = np.asarray(before.occupancy, dtype=np.float64)
    after_occupancy = np.asarray(after.occupancy, dtype=np.float64)
    objective_before = math.fsum(
        float(value * value) for value in before_occupancy - target_vector
    )
    objective_after = math.fsum(
        float(value * value) for value in after_occupancy - target_vector
    )
    improvement = math.fsum((objective_before, -objective_after))
    target_tuple = tuple(float(value) for value in target_vector)
    digest = canonical_sha256(
        {
            "identity_version": "composed_equilibrium_paired_objective.v1",
            "before_source_digest": before.source_digest,
            "after_source_digest": after.source_digest,
            "target_occupancy": target_tuple,
            "objective_before": objective_before,
            "objective_after": objective_after,
            "objective_improvement": improvement,
            "objective_improved": improvement > 0.0,
            "common_random_numbers": True,
        }
    )
    return PairedObjectiveEvaluation(
        before=before,
        after=after,
        target_occupancy=target_tuple,
        objective_before=objective_before,
        objective_after=objective_after,
        objective_improvement=improvement,
        objective_improved=improvement > 0.0,
        result_digest=digest,
    )
