"""Independent objective, optimizer-policy, and conservation reference checks."""

import numpy as np
import pytest

from thermo_lab.pasym_swap import build_pasym_swap_conditional

TARGET = np.asarray(build_pasym_swap_conditional(0.08, 0.02))


@pytest.mark.parametrize("penalty", [0.0, 1.0, 10.0])
def test_objective_gradient_matches_central_differences(penalty):
    from thermo_lab.conservation_tradeoff import objective_and_gradient

    parameters = np.array([0.2, -0.3, 0.5, 0.1, -0.4, 0.25, 0.15, -0.2, 0.3])
    _, gradient = objective_and_gradient(parameters, TARGET, penalty)
    numerical = []
    for direction in np.eye(9) * 1e-6:
        plus = objective_and_gradient(parameters + direction, TARGET, penalty)[0]
        minus = objective_and_gradient(parameters - direction, TARGET, penalty)[0]
        numerical.append((plus - minus) / 2e-6)
    np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=0)


def test_uniform_kernel_objective_includes_all_four_contexts():
    from thermo_lab.conservation_tradeoff import objective_and_gradient

    fidelity = np.sum((0.25 - TARGET) ** 2) / 4
    # 00/11 fail with probability 3/4; 01/10 fail with probability 1/2.
    for penalty in (0.0, 1.0, 10.0):
        actual, _ = objective_and_gradient(np.zeros(9), TARGET, penalty)
        assert actual == pytest.approx(fidelity + penalty * 0.625)


def test_fixed_budget_clipping_and_selection(monkeypatch):
    from thermo_lab import conservation_tradeoff as study

    evaluated = []

    def linear_objective(parameters, target, penalty):
        evaluated.append(parameters.copy())
        return -float(parameters.sum()), -np.ones(9) * 100

    monkeypatch.setattr(study, "objective_and_gradient", linear_objective)
    result = study.fit_group(np.full(9, -1.0), TARGET, 10.0)
    assert len(evaluated) == 202  # initial and 100 endpoints, for each of two starts
    assert np.max(np.abs(evaluated)) <= 2
    assert result["selected_candidate"] == 1  # tied endpoints prefer archived start
    assert result["parameters"] == [2.0] * 9
    assert [a["updates"] for a in result["attempts"]] == [100, 100]
    assert result["attempts"][0]["projected_gradient_residual"] == 0


def test_ties_keep_archived_start_and_penalty_normalizes_step(monkeypatch):
    from thermo_lab import conservation_tradeoff as study

    evaluated = []

    def constant_objective(parameters, target, penalty):
        evaluated.append(parameters.copy())
        return 1.0, np.ones(9)

    monkeypatch.setattr(study, "objective_and_gradient", constant_objective)
    result = study.fit_group(np.zeros(9), TARGET, 1.0)
    np.testing.assert_array_equal(evaluated[1], [-0.5] * 9)
    assert result["selected_candidate"] == 0
    assert result["parameters"] == [0.0] * 9


def test_logical_reference_has_zero_error_and_survives_composition():
    from thermo_lab.conservation_tradeoff import measure_tables

    targets = TARGET[None]
    tables = np.concatenate((targets, np.zeros_like(targets)), axis=2)
    measured = measure_tables(tables, targets, [0, 0], [(0, 1), (1, 2)], site_count=3)
    assert measured["mean_conservation_failure"] == 0
    assert measured["mean_row_tv"] == 0
    assert measured["hop_mae"] == 0
    assert measured["conditional_hop_mae"] == 0
    assert measured["asymmetry_mae"] == 0
    assert measured["survival"][-1]["survival_probability"] == pytest.approx(1)


def test_conditional_hops_do_not_conceal_unconditional_failure():
    from thermo_lab.conservation_tradeoff import measure_tables

    targets = TARGET[None]
    visible = targets.copy()
    visible[:, 1:3] *= 0.5
    visible[:, 1:3, 0] = 0.5
    tables = np.concatenate((visible, np.zeros_like(visible)), axis=2)
    measured = measure_tables(tables, targets, [0], [(0, 1)], site_count=3)
    assert measured["conditional_hop_mae"] == 0
    assert measured["hop_mae"] == pytest.approx(0.025)
    assert measured["mean_conservation_failure"] == 0.25
    assert measured["asymmetry_mae"] == pytest.approx(0.03)
    assert measured["survival"][-1]["survival_probability"] == pytest.approx(0.5)
