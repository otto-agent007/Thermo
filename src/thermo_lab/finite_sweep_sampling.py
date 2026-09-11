"""Bounded PCG64 sampling from finite-sweep endpoint laws and their true scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from thermo_lab.composed_trajectory_refinement import (
    GroupedGradientSource,
    TerminalOccupancySource,
    _checked_parameter_matrix,
    _checked_positive_float,
    _checked_positive_int,
    _checked_schedule,
    _checked_seed,
    _checked_site_vector,
    _initial_states,
    _inverse_cdf_rows,
    _tuple_int_matrix,
    _tuple_matrix,
)
from thermo_lab.finite_sweep_gradient_reference import CHECKED_HORIZONS
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.frozen_pair_finite_sweeps import HorizonTerminalEvidence, paired_table_digest
from thermo_lab.hashing import canonical_sha256
from thermo_lab.thermodynamic_kernel import KernelParameters


@dataclass(frozen=True)
class _SamplingRequest:
    parameters: tuple[tuple[float, ...], ...]
    groups: tuple[int, ...]
    sites: tuple[tuple[int, ...], ...]
    site_count: int
    batch_size: int
    seed: int
    beta: float
    horizon: int


def _request(parameters, groups, sites, *, site_count, batch_size, seed, beta, horizon):
    values = _checked_parameter_matrix(parameters, name="parameters")
    values[values == 0.0] = 0.0
    targets, edges, dimension = _checked_schedule(
        groups, sites, group_count=len(values), site_count=site_count
    )
    count = _checked_positive_int(batch_size, name="batch_size")
    if len(values) > 37 or dimension > 25 or len(targets) > 500 or not 3 <= count <= 32768:
        raise ValueError(
            "finite sampling is bounded to 37 groups, 25 sites, 500 occurrences, 32768 samples"
        )
    if np.any(np.abs(values) > 2.0):
        raise ValueError("finite sampling parameters must remain in [-2, 2]")
    if type(horizon) is not int or horizon not in CHECKED_HORIZONS:
        raise ValueError("finite sampling requires a checked complete-sweep horizon")
    return _SamplingRequest(
        _tuple_matrix(values),
        tuple(int(x) for x in targets),
        _tuple_int_matrix(edges),
        dimension,
        count,
        _checked_seed(seed),
        _checked_positive_float(beta, name="beta"),
        horizon,
    )


@lru_cache(maxsize=64)
def _tables(parameters, horizon, beta):
    laws = [finite_sweep_joint_law(KernelParameters(row), horizon, beta=beta) for row in parameters]
    probabilities = np.asarray([law.probabilities for law in laws])
    scores = np.asarray([law.scores for law in laws])
    probabilities.setflags(write=False)
    scores.setflags(write=False)
    return probabilities, scores


def finite_sampling_tables_digest(parameters, *, horizon, beta):
    """Bind exact endpoint probabilities and all nine derivatives to the parameters."""
    values = _checked_parameter_matrix(parameters, name="parameters")
    values[values == 0.0] = 0.0
    if len(values) > 37 or np.any(np.abs(values) > 2.0):
        raise ValueError("finite tables require bounded parameter groups")
    if type(horizon) is not int or horizon not in CHECKED_HORIZONS:
        raise ValueError("finite tables require a checked horizon")
    checked_beta = _checked_positive_float(beta, name="beta")
    probabilities, scores = _tables(_tuple_matrix(values), horizon, checked_beta)
    return canonical_sha256(
        {
            "identity_version": "finite_sampling_tables.v1",
            "evidence_class": "exact_reference",
            "parameters": values.tolist(),
            "horizon": horizon,
            "beta": checked_beta,
            "probabilities": probabilities.tolist(),
            "scores": scores.tolist(),
        }
    )


def _advance(states, left, right, tables, uniforms):
    parents = 2 * states[:, left].astype(np.int64) + states[:, right].astype(np.int64)
    outcomes = _inverse_cdf_rows(tables[parents], uniforms)
    outputs = outcomes % 4
    states[:, left] = outputs // 2
    states[:, right] = outputs % 2
    return parents, outcomes


def _gradient_digest(request, reward, sums, squares):
    return canonical_sha256(
        {
            "identity_version": "finite_grouped_gradient.v1",
            "request": asdict(request),
            "reward_coefficient": tuple(0.0 if x == 0.0 else float(x) for x in reward),
            "reference_policy": "independent_same_parent_non_propagated",
            "score_policy": "finite_endpoint_log_probability_derivative",
            "component_sum": sums,
            "component_sum_squares": squares,
        }
    )


def gradient_source_digest(
    parameters,
    occurrence_target_indices,
    occurrence_site_indices,
    reward_coefficient,
    component_sum,
    component_sum_squares,
    **settings,
):
    """Bind stored gradient moments independently of numerical replay tolerance."""
    request = _request(parameters, occurrence_target_indices, occurrence_site_indices, **settings)
    reward = _checked_site_vector(
        reward_coefficient, site_count=request.site_count, name="reward_coefficient"
    )
    sums = _checked_parameter_matrix(component_sum, name="component_sum")
    squares = _checked_parameter_matrix(component_sum_squares, name="component_sum_squares")
    if sums.shape != (len(request.parameters), 9) or squares.shape != sums.shape:
        raise ValueError("gradient moments must match the parameter groups")
    return _gradient_digest(request, reward, _tuple_matrix(sums), _tuple_matrix(squares))


@lru_cache(maxsize=64)
def _occupancy(request):
    tables, _ = _tables(request.parameters, request.horizon, request.beta)
    states = _initial_states(batch_size=request.batch_size, site_count=request.site_count)
    rng = np.random.Generator(np.random.PCG64(request.seed))
    for group, (left, right) in zip(request.groups, request.sites, strict=True):
        _advance(states, left, right, tables[group], rng.random(request.batch_size))
    counts = tuple(int(value) for value in states.sum(axis=0, dtype=np.int64))
    return TerminalOccupancySource(
        request.batch_size,
        counts,
        canonical_sha256(
            {
                "identity_version": "finite_terminal_occupancy.v1",
                "request": asdict(request),
                "occupancy_counts": counts,
            }
        ),
    )


def sample_finite_sweep_terminal_occupancy(
    parameters,
    occurrence_target_indices,
    occurrence_site_indices,
    *,
    site_count,
    batch_size,
    seed,
    beta,
    horizon,
):
    """Sample terminal occupancy; cached deterministic replays retain only counts."""
    return _occupancy(
        _request(
            parameters,
            occurrence_target_indices,
            occurrence_site_indices,
            site_count=site_count,
            batch_size=batch_size,
            seed=seed,
            beta=beta,
            horizon=horizon,
        )
    )


@lru_cache(maxsize=64)
def _gradient(request, reward_vector):
    tables, scores = _tables(request.parameters, request.horizon, request.beta)
    states = _initial_states(batch_size=request.batch_size, site_count=request.site_count)
    grouped_scores = np.zeros((request.batch_size, len(request.parameters), 9), dtype=np.float64)
    main_seed, reference_seed = np.random.SeedSequence(request.seed).spawn(2)
    main_rng = np.random.Generator(np.random.PCG64(main_seed))
    reference_rng = np.random.Generator(np.random.PCG64(reference_seed))
    for group, (left, right) in zip(request.groups, request.sites, strict=True):
        # Capture the main parent before advancing; reference outcomes never propagate.
        parents, main = _advance(
            states, left, right, tables[group], main_rng.random(request.batch_size)
        )
        reference = _inverse_cdf_rows(
            tables[group, parents], reference_rng.random(request.batch_size)
        )
        grouped_scores[:, group] += scores[group, parents, main] - scores[group, parents, reference]
    rewards = states.astype(np.float64) @ np.asarray(reward_vector)
    contributions = grouped_scores * rewards[:, None, None]
    sums = _tuple_matrix(contributions.sum(axis=0, dtype=np.float64))
    squares = _tuple_matrix((contributions * contributions).sum(axis=0, dtype=np.float64))
    return GroupedGradientSource(
        request.batch_size,
        sums,
        squares,
        _gradient_digest(request, reward_vector, sums, squares),
    )


def estimate_finite_sweep_grouped_gradient(
    parameters,
    occurrence_target_indices,
    occurrence_site_indices,
    reward_coefficient,
    *,
    site_count,
    batch_size,
    seed,
    beta,
    horizon,
):
    """Estimate finite-law derivatives of a fixed linear terminal reward.

    For population loss the coefficient must come from an independent occupancy
    batch. Summing all occurrences before reduction preserves sharing covariance.
    """
    request = _request(
        parameters,
        occurrence_target_indices,
        occurrence_site_indices,
        site_count=site_count,
        batch_size=batch_size,
        seed=seed,
        beta=beta,
        horizon=horizon,
    )
    reward = _checked_site_vector(
        reward_coefficient, site_count=request.site_count, name="reward_coefficient"
    )
    return _gradient(request, tuple(0.0 if x == 0.0 else float(x) for x in reward))


@lru_cache(maxsize=16)
def _paired(request, updated_parameters):
    before_tables, _ = _tables(request.parameters, request.horizon, request.beta)
    after_tables, _ = _tables(updated_parameters, request.horizon, request.beta)
    before = _initial_states(batch_size=request.batch_size, site_count=request.site_count)
    after = before.copy()
    rng = np.random.Generator(np.random.PCG64(request.seed))
    for group, (left, right) in zip(request.groups, request.sites, strict=True):
        uniforms = rng.random(request.batch_size)
        _advance(before, left, right, before_tables[group], uniforms)
        _advance(after, left, right, after_tables[group], uniforms)
    joined = np.concatenate((before, after), axis=1).astype(np.int64)
    first_particles = before.sum(axis=1, dtype=np.int64)
    last_particles = after.sum(axis=1, dtype=np.int64)
    leakage = 2 * (first_particles != 1).astype(np.int64) + (last_particles != 1)
    return HorizonTerminalEvidence(
        horizon=f"k{request.horizon}",
        exact_tables_digest=paired_table_digest(f"k{request.horizon}", before_tables, after_tables),
        sample_count=request.batch_size,
        before_counts=tuple(int(x) for x in before.sum(axis=0)),
        after_counts=tuple(int(x) for x in after.sum(axis=0)),
        joined_moment_counts=_tuple_int_matrix(joined.T @ joined),
        before_particle_histogram=tuple(
            int(x) for x in np.bincount(first_particles, minlength=request.site_count + 1)
        ),
        after_particle_histogram=tuple(
            int(x) for x in np.bincount(last_particles, minlength=request.site_count + 1)
        ),
        paired_leakage_counts=tuple(int(x) for x in np.bincount(leakage, minlength=4)),
    )


def evaluate_finite_sweep_pair(
    initial_parameters,
    updated_parameters,
    occurrence_target_indices,
    occurrence_site_indices,
    *,
    site_count,
    batch_size,
    seed,
    beta,
    horizon,
):
    """Evaluate a frozen initial/final pair with one common uniform per occurrence."""
    before = _request(
        initial_parameters,
        occurrence_target_indices,
        occurrence_site_indices,
        site_count=site_count,
        batch_size=batch_size,
        seed=seed,
        beta=beta,
        horizon=horizon,
    )
    after = _request(
        updated_parameters,
        occurrence_target_indices,
        occurrence_site_indices,
        site_count=site_count,
        batch_size=batch_size,
        seed=seed,
        beta=beta,
        horizon=horizon,
    )
    if len(before.parameters) != len(after.parameters):
        raise ValueError("paired parameter shapes must agree")
    return _paired(before, after.parameters)
