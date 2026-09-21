"""Evaluation retains invalid paths and reuses uniforms across distinct cells."""

import numpy as np
import pytest


def test_terminal_sampler_executes_after_particle_loss_and_all_operations():
    from thermo_lab.quality_budget_evaluation import sample_terminal

    # 10 -> 11 creates a second particle; subsequent 11 -> 00 destroys both.
    tables = np.zeros((2, 4, 8))
    tables[0, :, 3] = 1
    tables[1, :, 0] = 1
    states = sample_terminal(tables, [0, 1], [(0, 1), (0, 1)], seed=7)
    assert states.shape == (32768, 25)
    assert np.count_nonzero(states) == 0


@pytest.mark.parametrize("horizon", [1, 2, 4, 8, 16, 30, "equilibrium"])
def test_single_cell_matches_existing_production_occupancy(horizon):
    from thermo_lab.quality_budget_evaluation import joint_tables, sample_terminal
    from thermo_lab.quality_budget_training_laws import sample_training_roles
    from thermo_lab.trajectory_reinforce import build_checked_fixture

    f = build_checked_fixture()
    parameters = (f.model_parameters.values,)
    states = sample_terminal(joint_tables(parameters, horizon), [0, 0], f.occurrences, seed=938)
    expected, _, _ = sample_training_roles(
        parameters,
        [0, 0],
        f.occurrences,
        [0.0] * 25,
        horizon=horizon,
        occupancy_seed=938,
        gradient_seed=939,
    )
    np.testing.assert_array_equal(states.sum(axis=0), expected.occupancy_counts)


def test_joined_evidence_has_finite_minus_equilibrium_sign_and_exact_pairing():
    from thermo_lab.quality_budget_evaluation import paired_evidence

    equilibrium = np.zeros((32768, 25), dtype=np.uint8)
    finite = equilibrium.copy()
    finite[:, 0] = 1
    tables = np.full((1, 4, 8), 0.125)
    result = paired_evidence(equilibrium, finite, tables, tables, 1, [0.0] * 25)
    assert result["statistics"]["population_objective_difference_after_minus_before"] == 1.0
    assert result["evidence"]["paired_leakage_counts"] == [0, 0, 32768, 0]
    assert result["evidence"]["joined_moment_counts"][25][25] == 32768


def test_seed_evaluator_visits_all_twenty_cells_and_pairs_without_resampling():
    from thermo_lab.quality_budget_evaluation import evaluate_seed
    from thermo_lab.quality_budget_training_runner import fixture_request
    from thermo_lab.trajectory_reinforce import build_checked_fixture

    source = fixture_request(seed=0, horizon=1)["inputs"]
    source["target_occupancy"] += [0.0] * 22
    initial = source["initial_parameters"]
    result = evaluate_seed(
        source,
        [initial] * 7,
        [build_checked_fixture().target_conditional],
        seed=0,
        evaluation_seed=1874,
    )
    assert len(result["cells"]) == 20
    assert len(result["pairs"]) == 6
    assert len({c["cell_id"] for c in result["cells"]}) == 20
    assert result["executed_cells"] == 20
    for pair in result["pairs"]:
        assert pair["statistics"]["population_objective_difference_after_minus_before"] == 0
        assert pair["statistics"]["paired_jackknife_standard_error"] == 0
    for cell in result["cells"]:
        assert len(cell["occupancy_counts"]) == 25
        assert sum(cell["particle_histogram"]) == 32768
        assert len(cell["exact_metrics"]["survival"]) == 2
        assert len(cell["decision"]["occupancy_intervals"]) == 25
