"""Weighted derivatives, exact context accounting, and fidelity-sensitive screening."""

import numpy as np
import pytest

from thermo_lab.pasym_swap import build_pasym_swap_conditional

TARGET = np.asarray(build_pasym_swap_conditional(0.08, 0.02))
PARAMETERS = np.array([0.2, -0.3, 0.5, 0.1, -0.4, 0.25, 0.15, -0.2, 0.3])


@pytest.mark.parametrize("penalty", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("step", [1e-4, 1e-5])
def test_weighted_gradient_matches_central_differences(penalty, step):
    from thermo_lab.context_conservation import weighted_objective_and_gradient

    weights = (0.91, 0.06, 0.03, 0.0)
    _, gradient = weighted_objective_and_gradient(PARAMETERS, TARGET, penalty, weights)
    numerical = []
    for direction in np.eye(9) * step:
        plus = weighted_objective_and_gradient(PARAMETERS + direction, TARGET, penalty, weights)[0]
        minus = weighted_objective_and_gradient(PARAMETERS - direction, TARGET, penalty, weights)[0]
        numerical.append((plus - minus) / (2 * step))
    np.testing.assert_allclose(gradient, numerical, atol=5e-9, rtol=0)


def test_uniform_weights_agree_with_existing_objective():
    from thermo_lab.conservation_tradeoff import objective_and_gradient
    from thermo_lab.context_conservation import weighted_objective_and_gradient

    for penalty in (0.0, 1.0, 10.0):
        expected_value, expected_gradient = objective_and_gradient(PARAMETERS, TARGET, penalty)
        value, gradient = weighted_objective_and_gradient(PARAMETERS, TARGET, penalty, (0.25,) * 4)
        assert value == pytest.approx(expected_value, abs=2e-15)
        np.testing.assert_allclose(gradient, expected_gradient, atol=2e-15, rtol=0)


def test_zero_support_weights_exclude_other_parent_rows():
    from thermo_lab.context_conservation import weighted_objective_and_gradient

    for penalty in (0.0, 1.0, 10.0):
        value, _ = weighted_objective_and_gradient(np.zeros(9), TARGET, penalty, (1, 0, 0, 0))
        assert value == pytest.approx(0.75 * (1 + penalty))


@pytest.mark.parametrize("weights", [(1, 0, 0), (1, 1, 0, 0), (-1, 1, 1, 0), (np.nan, 0, 0, 1)])
def test_invalid_context_weights_rejected(weights):
    from thermo_lab.context_conservation import weighted_objective_and_gradient

    with pytest.raises(ValueError):
        weighted_objective_and_gradient(PARAMETERS, TARGET, 1.0, weights)


def test_weighted_fit_uses_unchanged_fixed_update_and_selection_policy(monkeypatch):
    from thermo_lab import context_conservation as study

    calls = []

    def objective(parameters, target, penalty, weights):
        calls.append((parameters.copy(), tuple(weights)))
        return -float(parameters.sum()), -np.ones(9) * 100

    monkeypatch.setattr(study, "weighted_objective_and_gradient", objective)
    weights = (0.9, 0.06, 0.04, 0)
    fit = study.fit_weighted_group(np.zeros(9), TARGET, 1.0, weights)
    assert len(calls) == 202
    assert all(w == weights for _, w in calls)
    assert fit["selected_candidate"] == 1
    assert fit["parameters"] == [2.0] * 9
    assert [a["updates"] for a in fit["attempts"]] == [100, 100]


def test_derived_contexts_preserve_target_hash_order_and_zero_11_support():
    from thermo_lab.context_conservation import derive_contexts
    from thermo_lab.pasym_swap import build_paper_fixture

    contexts = derive_contexts()
    profiles = contexts["profiles"]
    assert [p["target_hash"] for p in profiles] == [
        t.target_hash for t in build_paper_fixture().targets
    ]
    assert sum(p["multiplicity"] for p in profiles) == 500
    assert all(p["context_weights"][3] == 0 for p in profiles)
    assert all(sum(p["context_weights"]) == pytest.approx(1) for p in profiles)


def test_weighted_measurement_equals_explicit_occurrence_average():
    from thermo_lab.context_conservation import weighted_measurements

    targets = np.array([TARGET, TARGET])
    visible = np.array([np.full((4, 4), 0.25), TARGET])
    weights = np.array([[0.9, 0.06, 0.04, 0], [0.5, 0.2, 0.3, 0]])
    measured = weighted_measurements(visible, targets, weights, [2, 1])
    expected_fidelity = (
        sum(
            weights[g, a] * sum((visible[g, a] - targets[g, a]) ** 2)
            for g in (0, 0, 1)
            for a in range(4)
        )
        / 3
    )
    assert measured["target_context_fidelity"] == pytest.approx(expected_fidelity)
    assert measured["target_context_failure"] == pytest.approx(2 * (0.9 * 0.75 + 0.1 * 0.5) / 3)
    assert measured["mean_empty_failure"] == pytest.approx(0.375)
    assert measured["mean_hop_probability"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    "survival,hop,asymmetry,passes",
    [
        (0.2, 0.1, 0.1, True),
        (0.2, 0.11, 0.1, False),
        (0.2, 0.1, 0.11, False),
        (0.1, 0.1, 0.1, False),
        (0.05, 0.05, 0.05, False),
    ],
)
def test_joint_screen_rejects_survival_only_or_fidelity_only_gains(
    survival, hop, asymmetry, passes
):
    from thermo_lab.context_conservation import compare_metrics

    reference = {"survival": [{"survival_probability": 0.1}], "hop_mae": 0.1, "asymmetry_mae": 0.1}
    candidate = {
        "survival": [{"survival_probability": survival}],
        "hop_mae": hop,
        "asymmetry_mae": asymmetry,
    }
    result = compare_metrics(candidate, reference)
    assert result["survival_gain_without_hop_or_asymmetry_regression"] is passes
    assert result["survival_difference"] == pytest.approx(survival - 0.1)
    assert result["hop_mae_difference"] == pytest.approx(hop - 0.1)
