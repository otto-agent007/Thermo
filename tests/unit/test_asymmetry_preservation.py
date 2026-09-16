"""Finite-budget derivatives and the predeclared asymmetry-preservation screen."""

import numpy as np
import pytest

from thermo_lab.pasym_swap import build_pasym_swap_conditional

TARGET = np.asarray(build_pasym_swap_conditional(0.08, 0.02))
PARAMETERS = np.array([0.2, -0.3, 0.5, 0.1, -0.4, 0.25, 0.15, -0.2, 0.3])
WEIGHTS = (0.91, 0.06, 0.03, 0.0)


@pytest.mark.parametrize("step", [1e-4, 1e-5])
def test_gradient_matches_independent_central_differences(step):
    from thermo_lab.asymmetry_preservation import objective_and_gradient

    _, gradient = objective_and_gradient(PARAMETERS, TARGET, WEIGHTS)
    numerical = [
        (
            objective_and_gradient(PARAMETERS + d, TARGET, WEIGHTS)[0]
            - objective_and_gradient(PARAMETERS - d, TARGET, WEIGHTS)[0]
        )
        / (2 * step)
        for d in np.eye(9) * step
    ]
    np.testing.assert_allclose(gradient, numerical, atol=5e-9, rtol=0)


def test_added_term_and_gradient_use_both_unconditional_directions():
    from thermo_lab.asymmetry_preservation import objective_and_gradient
    from thermo_lab.context_conservation import weighted_objective_and_gradient
    from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
    from thermo_lab.thermodynamic_kernel import KernelParameters

    law = finite_sweep_joint_law(KernelParameters(tuple(PARAMETERS)), 4)
    visible = law.probabilities.reshape(4, 2, 4).sum(axis=1)
    jacobian = law.jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    error = (visible[1, 2] - visible[2, 1]) - (TARGET[1, 2] - TARGET[2, 1])
    # Even a zero logical weight for both directions does not remove the new term.
    weights = (1, 0, 0, 0)
    base, derivative = weighted_objective_and_gradient(PARAMETERS, TARGET, 1.0, weights)
    objective, gradient = objective_and_gradient(PARAMETERS, TARGET, weights)
    assert objective == base + error**2
    np.testing.assert_allclose(
        gradient - derivative,
        2 * error * (jacobian[1, 2] - jacobian[2, 1]),
        atol=1e-15,
        rtol=0,
    )


def test_symmetric_law_and_target_add_zero_penalty():
    from thermo_lab.asymmetry_preservation import objective_and_gradient
    from thermo_lab.context_conservation import weighted_objective_and_gradient

    target = np.asarray(build_pasym_swap_conditional(0.05, 0.05))
    actual = objective_and_gradient(np.zeros(9), target, WEIGHTS)
    expected = weighted_objective_and_gradient(np.zeros(9), target, 1.0, WEIGHTS)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("weights", [(1, 1, 0, 0), (np.nan, 0, 0, 1), (1, 0, 0)])
def test_invalid_weights_are_rejected(weights):
    from thermo_lab.asymmetry_preservation import objective_and_gradient

    with pytest.raises(ValueError):
        objective_and_gradient(PARAMETERS, TARGET, weights)


def test_fit_keeps_the_same_step_caps_budget_and_candidate_selection():
    from thermo_lab.asymmetry_preservation import fit_group, objective_and_gradient

    fit = fit_group(PARAMETERS, TARGET, WEIGHTS)
    for attempt, start in zip(fit["attempts"], (PARAMETERS, np.zeros(9)), strict=True):
        row = start.copy()
        for _ in range(100):
            _, gradient = objective_and_gradient(row, TARGET, WEIGHTS)
            row = np.clip(row - gradient / 2, -2, 2)
        np.testing.assert_array_equal(attempt["final_parameters"], row)
        assert attempt["updates"] == 100
    candidates = [
        a[field] for a in fit["attempts"] for field in ("initial_objective", "final_objective")
    ]
    assert fit["selected_candidate"] == int(np.argmin(candidates))
    assert fit["objective"] == min(candidates)
    assert np.max(np.abs(fit["parameters"])) <= 2


@pytest.mark.parametrize(
    "survival,hop,asymmetry,passes",
    [
        (0.2, 0.03, 0.02, True),
        (0.199, 0.03, 0.02, False),
        (0.2, 0.031, 0.02, False),
        (0.2, 0.03, 0.021, False),
    ],
)
def test_primary_screen_requires_all_three_fixed_reference_thresholds(
    survival, hop, asymmetry, passes
):
    from thermo_lab.asymmetry_preservation import preservation_screen

    def metrics(s, h, a):
        return {"survival": [{"survival_probability": s}], "hop_mae": h, "asymmetry_mae": a}

    screen = preservation_screen(
        metrics(survival, hop, asymmetry), metrics(0.2, 0.03, 0.025), metrics(0.001, 0.04, 0.02)
    )
    assert screen["passes"] is passes
    assert screen["survival_margin"] == pytest.approx(survival - 0.2)
    assert screen["asymmetry_margin"] == pytest.approx(0.02 - asymmetry)
    assert screen["hop_margin"] == pytest.approx(0.03 - hop)
