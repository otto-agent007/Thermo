"""Independent checks for the three-edge exact reference."""

from itertools import product

import numpy as np
import pytest

from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.survival_gradients import survival_gradient
from thermo_lab.thermodynamic_kernel import KernelParameters
from thermo_lab.trajectory_reinforce import build_checked_fixture


def reference_paths(parameters):
    fixture = build_checked_fixture()
    model = finite_sweep_joint_law(KernelParameters(tuple(parameters)), 4)
    visible = model.probabilities.reshape(4, 2, 4).sum(axis=1)
    terminal = np.zeros(8)
    valid = np.zeros(8)
    logical = np.zeros(8)
    target_histories = {index: [] for index in range(8)}
    visited = [{str(row): 0.0 for row in range(4)} for _ in range(3)]
    failed_then_one = 0.0
    for outputs in product(range(4), repeat=3):
        state = [1, 0, 0]
        probability = reference = 1.0
        survived = True
        parents = []
        for occurrence, (left, right) in enumerate(((0, 1), (1, 2), (0, 1))):
            parent = 2 * state[left] + state[right]
            parents.append(parent)
            output = outputs[occurrence]
            probability *= visible[parent, output]
            reference *= fixture.target_conditional[parent, output]
            state[left], state[right] = divmod(output, 2)
            survived &= sum(state) == 1
        index = state[0] * 4 + state[1] * 2 + state[2]
        terminal[index] += probability
        logical[index] += reference
        for occurrence, parent in enumerate(parents):
            visited[occurrence][str(parent)] += reference
        if survived:
            valid[index] += probability
            if reference > 0:
                target_histories[index].append((outputs, reference, probability))
        elif sum(state) == 1:
            failed_then_one += probability
    return terminal, valid, logical, target_histories, visited, failed_then_one


def test_exact_return_paths_merge_and_do_not_revive_failed_mass():
    from thermo_lab.return_fixture_objectives import evaluate

    parameters = build_checked_fixture().model_parameters.values
    terminal, valid, logical, histories, visited, returned = reference_paths(parameters)
    result = evaluate(parameters)
    metrics = result["metrics"]
    assert len(tuple(product(range(4), repeat=3))) == 64
    assert terminal.sum() == pytest.approx(1)
    assert logical.sum() == pytest.approx(1)
    assert len(histories[4]) >= 2
    assert returned > 0
    assert visited[2]["1"] > 0  # the return edge sees parent 01
    np.testing.assert_allclose(metrics["terminal_probabilities"], terminal, atol=1e-12)
    np.testing.assert_allclose(metrics["valid_terminal_masses"], valid, atol=1e-12)
    assert metrics["survival"] == pytest.approx(valid.sum(), abs=1e-12)
    assert metrics["target_visited_rows"][2]["01"] == pytest.approx(visited[2]["1"])
    assert [sum(rows.values()) for rows in metrics["target_visited_rows"]] == pytest.approx(
        [1.0, 1.0, 1.0], abs=1e-12
    )
    assert metrics["reverse_hop_01"] > 0
    visible = finite_sweep_joint_law(KernelParameters(tuple(parameters)), 4).probabilities
    visible = visible.reshape(4, 2, 4).sum(axis=1)
    target = build_checked_fixture().target_conditional
    expected_row_mae = np.abs(visible - target).mean(axis=1)
    for row in range(4):
        assert metrics["row_mae"][format(row, "02b")] == pytest.approx(expected_row_mae[row])
    expected_visited = (
        sum(
            visited[occurrence][str(row)] * expected_row_mae[row]
            for occurrence in range(3)
            for row in range(4)
        )
        / 3
    )
    assert metrics["visited_row_mae"] == pytest.approx(expected_visited, abs=1e-12)
    assert metrics["forward_hop_10"] == pytest.approx(visible[2, 1])
    assert metrics["forward_hop_10_target"] == pytest.approx(target[2, 1])
    independent = survival_gradient(
        np.asarray([parameters]),
        [0, 0, 0],
        [(0, 1), (1, 2), (0, 1)],
        horizon=4,
        site_count=3,
    )
    assert metrics["survival"] == pytest.approx(independent["survival"], abs=1e-12)


@pytest.mark.parametrize(
    "parameters",
    [
        list(build_checked_fixture().model_parameters.values),
        [0.15, -0.22, 0.31, -0.11, 0.27, -0.14, 0.18, -0.26, 0.09],
    ],
)
def test_three_objectives_have_checked_shared_gradients_and_path_gap(parameters):
    from thermo_lab.return_fixture_objectives import evaluate

    result = evaluate(parameters)
    _, valid, logical, histories, _, _ = reference_paths(parameters)
    conditional = 0.0
    for endpoint, paths in histories.items():
        if logical[endpoint] == 0:
            continue
        for _, target_mass, model_mass in paths:
            conditional += target_mass * np.log(
                (target_mass / logical[endpoint]) / (model_mass / valid[endpoint])
            )
    values = result["objectives"]
    assert values["trajectory_kl"] - values["valid_terminal"] == pytest.approx(
        conditional, abs=1e-12
    )
    assert values["trajectory_kl"] >= values["valid_terminal"] - 1e-12
    assert values["valid_terminal"] >= -np.log(result["metrics"]["survival"]) - 1e-12
    assert result["metrics"]["path_objective_gap"] == pytest.approx(conditional, abs=1e-12)
    for objective in ("occupancy", "trajectory_kl", "valid_terminal"):
        for component in range(9):
            delta = np.eye(9)[component] * 1e-6
            finite = (
                evaluate(np.asarray(parameters) + delta)["objectives"][objective]
                - evaluate(np.asarray(parameters) - delta)["objectives"][objective]
            ) / 2e-6
            assert result["gradients"][objective][component] == pytest.approx(finite, abs=1e-7)


@pytest.mark.parametrize(
    "parameters",
    [
        [0] * 8,
        [True] * 9,
        ["0"] * 9,
        [np.nan] * 9,
        [np.inf] * 9,
        [2.01] * 9,
        [[0] * 9],
    ],
)
def test_rejects_invalid_parameters(parameters):
    from thermo_lab.return_fixture_objectives import evaluate

    with pytest.raises(ValueError):
        evaluate(parameters)
