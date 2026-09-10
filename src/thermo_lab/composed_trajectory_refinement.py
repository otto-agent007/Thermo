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
_NORMAL_95_CRITICAL_VALUE = 1.959963984540054

POPULATION_OBJECTIVE_ESTIMATOR_POLICY = "unbiased_order_two_u_statistic"
PAIRED_UNCERTAINTY_POLICY = "paired_delete_one_jackknife_normal_95_interval"
POPULATION_OBJECTIVE_CONCLUSION_POLICY = "after_minus_before_interval_strictly_below_or_above_zero"


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
    joined_terminal_second_moment_counts: tuple[tuple[int, ...], ...]
    target_occupancy: tuple[float, ...]
    objective_before: float
    objective_after: float
    objective_improvement: float
    objective_improved: bool
    population_objective_before: float
    population_objective_after: float
    population_objective_difference_after_minus_before: float
    paired_jackknife_standard_error: float
    paired_jackknife_normal_95_interval: tuple[float, float]
    population_objective_conclusion: str
    result_digest: str


@dataclass(frozen=True)
class PairedPopulationObjectiveStatistics:
    """Unbiased paired population objectives and within-evaluation uncertainty."""

    population_objective_before: float
    population_objective_after: float
    population_objective_difference_after_minus_before: float
    paired_jackknife_standard_error: float
    paired_jackknife_normal_95_interval: tuple[float, float]
    population_objective_conclusion: str


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


def _checked_occupancy_counts(
    values: object,
    *,
    sample_count: int,
    name: str = "occupancy_counts",
) -> NDArray[np.int64]:
    counts = np.asarray(values)
    if counts.ndim != 1 or counts.size == 0 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(f"{name} must be a non-empty integer vector")
    if np.issubdtype(counts.dtype, np.bool_):
        raise ValueError(f"{name} must be a non-empty integer vector")
    checked = counts.astype(np.int64, copy=False)
    if np.any(checked < 0) or np.any(checked > sample_count):
        raise ValueError(f"{name} values must lie in [0, sample_count]")
    return checked


def _checked_joined_second_moment_counts(
    values: object,
    *,
    joined_counts: NDArray[np.int64],
    sample_count: int,
) -> NDArray[np.int64]:
    moments = np.asarray(values)
    dimension = joined_counts.size
    if moments.shape != (dimension, dimension) or not np.issubdtype(moments.dtype, np.integer):
        raise ValueError(
            "joined_terminal_second_moment_counts must be a square integer matrix "
            "matching the joined occupancy counts"
        )
    if np.issubdtype(moments.dtype, np.bool_):
        raise ValueError("joined_terminal_second_moment_counts must not contain booleans")
    checked = moments.astype(np.int64, copy=False)
    if not np.array_equal(checked, checked.T):
        raise ValueError("joined_terminal_second_moment_counts must be symmetric")
    if not np.array_equal(np.diag(checked), joined_counts):
        raise ValueError(
            "joined_terminal_second_moment_counts diagonal must equal occupancy counts"
        )
    lower = np.maximum(0, joined_counts[:, None] + joined_counts[None, :] - sample_count)
    upper = np.minimum(joined_counts[:, None], joined_counts[None, :])
    if np.any(checked < lower) or np.any(checked > upper):
        raise ValueError("joined_terminal_second_moment_counts violate count bounds")
    return checked


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


def _tuple_int_matrix(values: NDArray[np.int64]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in values)


def _build_joint_tables(parameters: NDArray[np.float64], *, beta: float) -> NDArray[np.float64]:
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
    return np.sum(uniforms[:, None] >= cumulative[:, :-1], axis=1).astype(np.int64, copy=False)


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
            "occurrence_site_indices": tuple(tuple(int(value) for value in row) for row in sites),
        }
    )


def calculate_unbiased_population_objective(
    occupancy_counts: object,
    *,
    sample_count: object,
    target_occupancy: object,
) -> float:
    """Estimate squared population terminal-occupancy loss without finite-batch bias."""

    checked_sample_count = _checked_positive_int(sample_count, name="sample_count")
    if checked_sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    counts = _checked_occupancy_counts(
        occupancy_counts,
        sample_count=checked_sample_count,
    )
    target = _checked_site_vector(
        target_occupancy,
        site_count=counts.size,
        name="target_occupancy",
    )
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target_occupancy values must lie in [0, 1]")
    denominator = checked_sample_count * (checked_sample_count - 1)
    return math.fsum(
        float(
            count * (count - 1) / denominator
            - 2.0 * target_value * count / checked_sample_count
            + target_value * target_value
        )
        for count, target_value in zip(counts, target, strict=True)
    )


def calculate_paired_population_objective_statistics(
    before_occupancy_counts: object,
    after_occupancy_counts: object,
    joined_terminal_second_moment_counts: object,
    *,
    sample_count: object,
    target_occupancy: object,
) -> PairedPopulationObjectiveStatistics:
    """Derive paired U-statistics and delete-one jackknife uncertainty from counts."""

    checked_sample_count = _checked_positive_int(sample_count, name="sample_count")
    if checked_sample_count < 3:
        raise ValueError("sample_count must be at least 3 for paired jackknife uncertainty")
    before_counts = _checked_occupancy_counts(
        before_occupancy_counts,
        sample_count=checked_sample_count,
        name="before_occupancy_counts",
    )
    after_counts = _checked_occupancy_counts(
        after_occupancy_counts,
        sample_count=checked_sample_count,
        name="after_occupancy_counts",
    )
    if after_counts.shape != before_counts.shape:
        raise ValueError("after_occupancy_counts must match before_occupancy_counts")
    target = _checked_site_vector(
        target_occupancy,
        site_count=before_counts.size,
        name="target_occupancy",
    )
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target_occupancy values must lie in [0, 1]")
    joined_counts = np.concatenate((before_counts, after_counts))
    moments = _checked_joined_second_moment_counts(
        joined_terminal_second_moment_counts,
        joined_counts=joined_counts,
        sample_count=checked_sample_count,
    )

    objective_before = calculate_unbiased_population_objective(
        before_counts,
        sample_count=checked_sample_count,
        target_occupancy=target,
    )
    objective_after = calculate_unbiased_population_objective(
        after_counts,
        sample_count=checked_sample_count,
        target_occupancy=target,
    )
    difference = math.fsum((objective_after, -objective_before))

    site_count = before_counts.size
    paired_cross_diagonal = moments[
        np.arange(site_count),
        site_count + np.arange(site_count),
    ]
    trajectories_are_identical = np.array_equal(before_counts, after_counts) and np.array_equal(
        paired_cross_diagonal,
        before_counts,
    )
    if trajectories_are_identical:
        standard_error = 0.0
    else:
        delete_denominator = (checked_sample_count - 1) * (checked_sample_count - 2)
        before_delete_coefficients = 2.0 * (
            target / (checked_sample_count - 1) - (before_counts - 1) / delete_denominator
        )
        after_delete_coefficients = 2.0 * (
            target / (checked_sample_count - 1) - (after_counts - 1) / delete_denominator
        )
        joined_coefficients = np.concatenate(
            (-before_delete_coefficients, after_delete_coefficients)
        )
        weighted_second_moment = float(joined_coefficients @ moments @ joined_coefficients)
        weighted_sum = float(joined_coefficients @ joined_counts)
        centered_sum_squares = (
            weighted_second_moment - weighted_sum * weighted_sum / checked_sample_count
        )
        roundoff_scale = max(
            1.0,
            abs(weighted_second_moment),
            abs(weighted_sum * weighted_sum / checked_sample_count),
        )
        roundoff_tolerance = 32.0 * np.finfo(np.float64).eps * roundoff_scale
        if centered_sum_squares < -roundoff_tolerance:
            raise ValueError("joined terminal moments imply a negative jackknife variance")
        centered_sum_squares = max(0.0, centered_sum_squares)
        standard_error = math.sqrt(
            (checked_sample_count - 1) / checked_sample_count * centered_sum_squares
        )
    margin = _NORMAL_95_CRITICAL_VALUE * standard_error
    interval = (difference - margin, difference + margin)
    if interval[1] < 0.0:
        conclusion = "improved"
    elif interval[0] > 0.0:
        conclusion = "regressed"
    else:
        conclusion = "inconclusive"
    return PairedPopulationObjectiveStatistics(
        population_objective_before=objective_before,
        population_objective_after=objective_after,
        population_objective_difference_after_minus_before=difference,
        paired_jackknife_standard_error=standard_error,
        paired_jackknife_normal_95_interval=interval,
        population_objective_conclusion=conclusion,
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
            "schedule_digest": _schedule_digest(targets, sites, site_count=checked_site_count),
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

    before_parameters = _checked_parameter_matrix(initial_parameters, name="initial_parameters")
    after_parameters = _checked_parameter_matrix(updated_parameters, name="updated_parameters")
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
    before_states = _initial_states(batch_size=checked_batch_size, site_count=checked_site_count)
    after_states = before_states.copy()
    rng = np.random.Generator(np.random.PCG64(checked_seed))

    for target, (left, right) in zip(targets, sites, strict=True):
        uniforms = rng.random(checked_batch_size)
        before_parent = 2 * before_states[:, left].astype(np.int64) + before_states[
            :, right
        ].astype(np.int64)
        after_parent = 2 * after_states[:, left].astype(np.int64) + after_states[:, right].astype(
            np.int64
        )
        before_outcome = _inverse_cdf_rows(before_tables[target, before_parent], uniforms)
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
    joined_states = np.concatenate((before_states, after_states), axis=1).astype(
        np.int64,
        copy=False,
    )
    joined_terminal_second_moment_counts = _tuple_int_matrix(joined_states.T @ joined_states)
    population_statistics = calculate_paired_population_objective_statistics(
        before.occupancy_counts,
        after.occupancy_counts,
        joined_terminal_second_moment_counts,
        sample_count=checked_batch_size,
        target_occupancy=target_vector,
    )
    before_occupancy = np.asarray(before.occupancy, dtype=np.float64)
    after_occupancy = np.asarray(after.occupancy, dtype=np.float64)
    objective_before = math.fsum(float(value * value) for value in before_occupancy - target_vector)
    objective_after = math.fsum(float(value * value) for value in after_occupancy - target_vector)
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
        joined_terminal_second_moment_counts=joined_terminal_second_moment_counts,
        target_occupancy=target_tuple,
        objective_before=objective_before,
        objective_after=objective_after,
        objective_improvement=improvement,
        objective_improved=improvement > 0.0,
        population_objective_before=population_statistics.population_objective_before,
        population_objective_after=population_statistics.population_objective_after,
        population_objective_difference_after_minus_before=(
            population_statistics.population_objective_difference_after_minus_before
        ),
        paired_jackknife_standard_error=population_statistics.paired_jackknife_standard_error,
        paired_jackknife_normal_95_interval=(
            population_statistics.paired_jackknife_normal_95_interval
        ),
        population_objective_conclusion=(population_statistics.population_objective_conclusion),
        result_digest=digest,
    )
