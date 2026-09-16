"""Exact first-exit survival and unmodified endpoint-sampler conservation traces."""

from __future__ import annotations

import numpy as np

from thermo_lab.composed_trajectory_refinement import (
    _build_joint_tables,
    _checked_parameter_matrix,
    _checked_positive_int,
    _checked_schedule,
    _checked_seed,
    _initial_states,
    _inverse_cdf_rows,
)
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.thermodynamic_kernel import KernelParameters

PARTICLES = np.array([0, 1, 1, 2], dtype=np.int64)


def endpoint_tables(parameters, horizon):
    """Return exact joint endpoint laws at beta one for the two declared horizons."""
    values = _checked_parameter_matrix(parameters, name="parameters")
    if len(values) > 37 or np.any(np.abs(values) > 2):
        raise ValueError("diagnostic requires at most 37 parameter groups in [-2,2]")
    if horizon == "equilibrium":
        return _build_joint_tables(values, beta=1.0)
    if type(horizon) is not int or horizon != 4:
        raise ValueError("diagnostic horizon must be 4 or equilibrium")
    return np.asarray(
        [
            finite_sweep_joint_law(KernelParameters(tuple(row)), 4, beta=1.0).probabilities
            for row in values
        ]
    )


def _checked_tables_schedule(tables, groups, sites, site_count):
    values = np.asarray(tables, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[1:] != (4, 8)
        or not 1 <= len(values) <= 37
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.allclose(values.sum(axis=-1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("joint endpoint tables must be finite nonnegative normalized (groups,4,8)")
    targets, edges, dimension = _checked_schedule(
        groups, sites, group_count=len(values), site_count=site_count
    )
    if dimension > 25 or len(targets) > 500:
        raise ValueError("diagnostic is bounded to 25 sites and 500 operations")
    return values, targets, edges, dimension


def local_failure_probabilities(tables):
    """Probability that the two visible outputs change the local particle count."""
    visible = np.asarray(tables).reshape(-1, 4, 2, 4).sum(axis=2)
    invalid = PARTICLES[:, None] != PARTICLES[None, :]
    return (visible * invalid).sum(axis=-1)


def exact_survival(tables, groups, sites, *, site_count):
    """Propagate only paths that have never left the one-particle sector.

    Outside the active edge a particle survives only under 00 -> 00. On the
    edge, either occupied output is allowed. Killed paths can never return.
    """
    tables, groups, sites, dimension = _checked_tables_schedule(tables, groups, sites, site_count)
    visible = tables.reshape(-1, 4, 2, 4).sum(axis=2)
    failures = local_failure_probabilities(tables)
    weights = np.zeros(dimension, dtype=np.float64)
    weights[0] = 1
    rows = []
    for index, (group, (left, right)) in enumerate(zip(groups, sites, strict=True), 1):
        outside = weights.copy()
        outside[[left, right]] = 0
        parent_weights = np.array([outside.sum(), weights[right], weights[left], 0.0])
        first_exit = parent_weights * failures[group]
        flow = parent_weights[:, None] * visible[group]
        created = PARTICLES[None, :] > PARTICLES[:, None]
        destroyed = PARTICLES[None, :] < PARTICLES[:, None]
        after = outside * visible[group, 0, 0]
        after[left] = weights[left] * visible[group, 2, 2] + weights[right] * visible[group, 1, 2]
        after[right] = weights[left] * visible[group, 2, 1] + weights[right] * visible[group, 1, 1]
        weights = after
        rows.append(
            {
                "operation": index,
                "survival_probability": float(weights.sum()),
                "first_exit_by_parent": first_exit.tolist(),
                "first_exit_creation_probability": float(flow[created].sum()),
                "first_exit_destruction_probability": float(flow[destroyed].sum()),
            }
        )
    return rows


def sample_trace(tables, groups, sites, *, site_count, batch_size, seed):
    """Sample all paths, including invalid states and later returns, without repair."""
    tables, groups, sites, dimension = _checked_tables_schedule(tables, groups, sites, site_count)
    count = _checked_positive_int(batch_size, name="batch_size")
    if not 3 <= count <= 32768:
        raise ValueError("diagnostic sample count must be in [3,32768]")
    rng = np.random.Generator(np.random.PCG64(_checked_seed(seed)))
    states = _initial_states(batch_size=count, site_count=dimension)
    particles = np.ones(count, dtype=np.int64)
    ever_exited = np.zeros(count, dtype=bool)
    rows = []
    for index, (group, (left, right)) in enumerate(zip(groups, sites, strict=True), 1):
        parents = 2 * states[:, left].astype(np.int64) + states[:, right].astype(np.int64)
        outcomes = _inverse_cdf_rows(tables[group, parents], rng.random(count)) % 4
        states[:, left], states[:, right] = outcomes // 2, outcomes % 2
        after = particles + PARTICLES[outcomes] - PARTICLES[parents]
        invalid = after != 1
        first_exit = ~ever_exited & invalid
        exits = (particles == 1) & invalid
        returns = (particles != 1) & ~invalid
        ever_exited |= invalid
        transitions = np.bincount(4 * parents + outcomes, minlength=16).reshape(4, 4)
        rows.append(
            {
                "operation": index,
                "particle_histogram": np.bincount(after, minlength=dimension + 1).tolist(),
                "occupancy_counts": states.sum(axis=0, dtype=np.int64).tolist(),
                "local_transition_counts": transitions.tolist(),
                "first_exit_by_parent": np.bincount(parents[first_exit], minlength=4).tolist(),
                "first_exit_creation_count": int((first_exit & (after > particles)).sum()),
                "first_exit_destruction_count": int((first_exit & (after < particles)).sum()),
                "ever_exited_count": int(ever_exited.sum()),
                "exit_count": int(exits.sum()),
                "return_count": int(returns.sum()),
                "valid_after_exit_count": int((ever_exited & ~invalid).sum()),
            }
        )
        particles = after
    return rows
