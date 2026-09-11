"""Independent finite-law and paired-stream checks for the frozen-pair audit."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from thermo_lab.composed_pasym_swap_artifacts import FINITE_HORIZONS, HORIZON_LABELS
from thermo_lab.composed_trajectory_refinement import evaluate_paired_equilibrium_objective
from thermo_lab.frozen_pair_finite_sweeps import (
    HorizonTerminalEvidence,
    build_joint_horizon_tables,
    declared_sampling_work,
    evaluate_frozen_pair_horizons,
)
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    equilibrium_conditional,
    finite_horizon_conditional,
    one_sweep_transition,
)

BEFORE = np.asarray([[0.3, -0.2, 0.4, 0.1, -0.3, 0.2, -0.1, 0.5, -0.4]])
AFTER = np.asarray([[0.1, -0.1, 0.3, 0.2, -0.2, 0.4, -0.2, 0.3, -0.1]])
TARGETS = np.asarray([0, 0], dtype=np.int64)
SITES = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
TARGET = (0.2, 0.3, 0.5)


def _evaluate(before=BEFORE, after=AFTER, **overrides):
    request = {
        "site_count": 3,
        "batch_size": 8192,
        "seed": 17,
        "beta": 1.0,
        "parameter_cap": 2.0,
    }
    request.update(overrides)
    return evaluate_frozen_pair_horizons(before, after, TARGETS, SITES, TARGET, **request)


@pytest.fixture(scope="module")
def cells():
    return _evaluate()


def test_joint_endpoints_match_independent_matrix_powers_and_existing_marginals():
    tables = build_joint_horizon_tables(BEFORE, beta=1.0)
    kernel = KernelParameters(tuple(float(value) for value in BEFORE[0]))
    marginal = finite_horizon_conditional(kernel, (1, 2, 4, 8, 16, 30))
    for index, horizon in enumerate(HORIZON_LABELS):
        expected = (
            equilibrium_conditional(kernel) if index == 0 else marginal[FINITE_HORIZONS[horizon]]
        )
        np.testing.assert_allclose(
            tables[index, 0].reshape(4, 2, 4).sum(axis=1), expected, rtol=0.0, atol=1e-14
        )
        if index:
            for parent in range(4):
                powered = np.linalg.matrix_power(
                    one_sweep_transition(kernel, parent), FINITE_HORIZONS[horizon]
                )
                np.testing.assert_allclose(
                    tables[index, 0, parent],
                    np.full(8, 1.0 / 8.0) @ powered,
                    rtol=0.0,
                    atol=1e-14,
                )
    assert not tables.flags.writeable


def _exact_three_site_law(conditional):
    words = tuple(itertools.product((0, 1), repeat=3))
    law = {(1, 0, 0): 1.0}
    for left, right in SITES:
        next_law = dict.fromkeys(words, 0.0)
        for state, probability in law.items():
            parent = 2 * state[left] + state[right]
            for output in range(4):
                updated = list(state)
                updated[left], updated[right] = divmod(output, 2)
                next_law[tuple(updated)] += probability * conditional[parent, output]
        law = next_law
    occupancy = np.asarray(
        [sum(probability * state[site] for state, probability in law.items()) for site in range(3)]
    )
    leakage = sum(probability for state, probability in law.items() if sum(state) != 1)
    return occupancy, leakage


def test_sampled_endpoints_match_bounded_exact_composition(cells):
    for parameters, member in ((BEFORE, "before"), (AFTER, "after")):
        kernel = KernelParameters(tuple(float(value) for value in parameters[0]))
        finite = finite_horizon_conditional(kernel, (1, 2, 4, 8, 16, 30))
        for cell in cells:
            conditional = (
                equilibrium_conditional(kernel)
                if cell.horizon == "equilibrium"
                else finite[FINITE_HORIZONS[cell.horizon]]
            )
            occupancy, leakage = _exact_three_site_law(conditional)
            observed = np.asarray(getattr(cell, f"{member}_counts")) / cell.sample_count
            bound = 6.0 * np.sqrt(occupancy * (1.0 - occupancy) / cell.sample_count)
            assert np.all(np.abs(observed - occupancy) <= bound + 1.0 / cell.sample_count)
            histogram = getattr(cell, f"{member}_particle_histogram")
            observed_leakage = 1.0 - histogram[1] / cell.sample_count
            leakage_bound = 6.0 * np.sqrt(leakage * (1.0 - leakage) / cell.sample_count)
            assert abs(observed_leakage - leakage) <= leakage_bound + 1.0 / cell.sample_count


def test_equilibrium_control_exactly_replays_the_existing_m1_stream(cells):
    old = evaluate_paired_equilibrium_objective(
        BEFORE,
        AFTER,
        TARGETS,
        SITES,
        TARGET,
        site_count=3,
        batch_size=8192,
        seed=17,
        beta=1.0,
    )
    equilibrium = cells[0]
    assert equilibrium.before_counts == old.before.occupancy_counts
    assert equilibrium.after_counts == old.after.occupancy_counts
    assert equilibrium.joined_moment_counts == old.joined_terminal_second_moment_counts
    statistics = equilibrium.statistics(TARGET)
    assert statistics.population_objective_before == old.population_objective_before
    assert statistics.population_objective_after == old.population_objective_after
    assert statistics.paired_jackknife_normal_95_interval == old.paired_jackknife_normal_95_interval


def test_all_horizons_share_one_stream_and_identical_pairs_have_zero_uncertainty():
    zero = np.zeros((1, 9), dtype=np.float64)
    cells = _evaluate(zero, zero, batch_size=256)
    for cell in cells:
        assert cell.before_counts == cell.after_counts == cells[0].before_counts
        assert cell.joined_moment_counts == cells[0].joined_moment_counts
        statistics = cell.statistics(TARGET)
        assert statistics.population_objective_difference_after_minus_before == 0.0
        assert statistics.paired_jackknife_standard_error == 0.0
        assert statistics.population_objective_conclusion == "inconclusive"
        assert cell.paired_leakage_counts[1:3] == (0, 0)


def test_seed_replay_is_exact_and_inputs_are_unchanged(cells):
    before = BEFORE.copy()
    after = AFTER.copy()
    assert _evaluate(before, after) == cells
    np.testing.assert_array_equal(before, BEFORE)
    np.testing.assert_array_equal(after, AFTER)
    assert _evaluate(seed=18)[0].before_counts != cells[0].before_counts


@pytest.mark.parametrize(
    "change",
    (
        {"batch_size": 2},
        {"batch_size": True},
        {"seed": True},
        {"seed": -1},
        {"beta": 0.0},
        {"parameter_cap": 0.1},
    ),
)
def test_invalid_sampling_requests_are_rejected(change):
    with pytest.raises(ValueError):
        _evaluate(**change)


@pytest.mark.parametrize("mutation", ("histogram", "leakage", "moments", "boolean"))
def test_terminal_evidence_rejects_inconsistent_or_coerced_counts(cells, mutation):
    payload = cells[1].model_dump(mode="json")
    if mutation == "histogram":
        payload["before_particle_histogram"][0] += 1
    elif mutation == "leakage":
        payload["paired_leakage_counts"][0] += 1
    elif mutation == "moments":
        payload["joined_moment_counts"][0][0] += 1
    else:
        payload["sample_count"] = True
    with pytest.raises(ValueError):
        HorizonTerminalEvidence.model_validate(payload)


def test_declared_work_uses_complete_sweeps_and_three_free_pbits():
    for horizon, sweeps in FINITE_HORIZONS.items():
        work = declared_sampling_work(horizon, occurrence_count=500)
        assert work["complete_sweeps_per_trajectory_per_member"] == 500 * sweeps
        assert work["free_pbit_updates_per_trajectory_per_member"] == 1500 * sweeps
        assert "not executed Gibbs updates" in work["execution"]
    work = declared_sampling_work("equilibrium", occurrence_count=500)
    assert work["complete_sweeps_per_trajectory_per_member"] is None
    assert work["free_pbit_updates_per_trajectory_per_member"] is None


@pytest.mark.parametrize("mutation", ("histogram", "identical_vectors", "equal_particle_counts"))
def test_paired_particle_evidence_must_respect_joined_moment_implications(mutation):
    # These moments come from actual paired binary rows, independently of the
    # sampler. Replacing only particle evidence used to admit impossible pairs.
    before = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)), dtype=np.int64)
    after = before.copy()
    if mutation == "equal_particle_counts":
        after[:3] = np.roll(after[:3], 1, axis=1)
    joined = np.concatenate((before, after), axis=1)
    payload = {
        "horizon": "k1",
        "exact_tables_digest": "sha256:" + "0" * 64,
        "sample_count": 4,
        "before_counts": tuple(int(value) for value in before.sum(axis=0)),
        "after_counts": tuple(int(value) for value in after.sum(axis=0)),
        "joined_moment_counts": tuple(
            tuple(int(value) for value in row) for row in joined.T @ joined
        ),
        "before_particle_histogram": (0, 3, 0, 1),
        "after_particle_histogram": (0, 3, 0, 1),
        "paired_leakage_counts": (3, 0, 0, 1),
    }
    assert HorizonTerminalEvidence.model_validate(payload).paired_leakage_counts == (3, 0, 0, 1)
    if mutation == "histogram":
        # The replacement has the same first two particle moments (6 and 12).
        payload["after_particle_histogram"] = (1, 0, 3, 0)
        payload["paired_leakage_counts"] = (0, 3, 0, 1)
    else:
        payload["paired_leakage_counts"] = (2, 1, 1, 0)
    with pytest.raises(ValueError, match="paired.*disagreement"):
        HorizonTerminalEvidence.model_validate(payload)
