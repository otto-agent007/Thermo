"""Finite-sweep derivatives must describe the actual reset execution law."""

from dataclasses import replace

import numpy as np
import pytest

from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    finite_horizon_conditional,
    one_sweep_transition,
)
from thermo_lab.trajectory_reinforce import build_checked_fixture


def _law(parameters, horizon, beta=1.0):
    from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law

    return finite_sweep_joint_law(parameters, horizon, beta=beta)


@pytest.mark.parametrize("horizon", (1, 2, 4, 8, 16, 30))
def test_joint_law_matches_existing_finite_execution_and_has_zero_mean_scores(horizon):
    parameters = build_checked_fixture().model_parameters
    law = _law(parameters, horizon)
    expected = finite_horizon_conditional(parameters, (horizon,))[horizon]
    np.testing.assert_allclose(
        law.probabilities.reshape(4, 2, 4).sum(axis=1), expected, rtol=0.0, atol=1e-13
    )
    for parent in range(4):
        expected_joint = np.full(8, 1.0 / 8.0) @ np.linalg.matrix_power(
            one_sweep_transition(parameters, parent), horizon
        )
        np.testing.assert_allclose(law.probabilities[parent], expected_joint, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(law.jacobian.sum(axis=1), 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        (law.probabilities[:, :, None] * law.scores).sum(axis=1), 0.0, rtol=0.0, atol=1e-12
    )
    assert not law.probabilities.flags.writeable
    assert not law.jacobian.flags.writeable
    assert not law.scores.flags.writeable


@pytest.mark.parametrize("horizon,beta", ((1, 1.0), (2, 1.4), (30, 0.7)))
def test_all_nine_derivatives_match_independent_existing_kernel_finite_differences(horizon, beta):
    parameters = build_checked_fixture().model_parameters
    law = _law(parameters, horizon, beta)
    step = 1e-6
    marginal_jacobian = law.jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    for component in range(9):
        delta = np.eye(9)[component] * step
        plus = KernelParameters(tuple(np.asarray(parameters.values) + delta))
        minus = KernelParameters(tuple(np.asarray(parameters.values) - delta))
        expected = (
            finite_horizon_conditional(plus, (horizon,), beta)[horizon]
            - finite_horizon_conditional(minus, (horizon,), beta)[horizon]
        ) / (2 * step)
        np.testing.assert_allclose(
            marginal_jacobian[:, :, component], expected, rtol=0.0, atol=1e-8
        )
        for parent in range(4):
            plus_joint = np.full(8, 1.0 / 8.0) @ np.linalg.matrix_power(
                one_sweep_transition(plus, parent, beta), horizon
            )
            minus_joint = np.full(8, 1.0 / 8.0) @ np.linalg.matrix_power(
                one_sweep_transition(minus, parent, beta), horizon
            )
            np.testing.assert_allclose(
                law.jacobian[parent, :, component],
                (plus_joint - minus_joint) / (2 * step),
                rtol=0.0,
                atol=1e-8,
            )


@pytest.mark.parametrize("horizon", (0, -1, 31, True, 1.0))
def test_unbounded_or_noninteger_horizons_are_rejected(horizon):
    with pytest.raises(ValueError, match="horizon"):
        _law(build_checked_fixture().model_parameters, horizon)


def test_parameter_cap_is_enforced_before_exact_enumeration():
    parameters = replace(build_checked_fixture().model_parameters, values=(2.01,) * 9)
    with pytest.raises(ValueError, match="cap"):
        _law(parameters, 1)


@pytest.fixture(scope="module")
def references():
    from thermo_lab.finite_sweep_gradient_reference import build_finite_sweep_reference

    return tuple(build_finite_sweep_reference(horizon) for horizon in (1, 2, 4, 8, 16, 30))


def test_composed_score_chain_rule_autodiff_and_finite_differences_agree(references):
    for reference in references:
        assert reference.accepted
        assert reference.maximum_exact_error < 1e-12
        assert reference.maximum_finite_difference_error < 1e-7
        for occurrence in range(2):
            np.testing.assert_allclose(
                reference.score.occurrences[occurrence],
                reference.autodiff.occurrences[occurrence],
                rtol=0.0,
                atol=1e-12,
            )
        np.testing.assert_allclose(
            reference.score.shared, reference.finite_difference_tied, rtol=0.0, atol=1e-7
        )


def test_negative_controls_detect_equilibrium_score_substitution_and_missing_occurrence(references):
    first = references[0]
    assert first.equilibrium_score_substitution_error > 1e-4
    assert np.max(np.abs(first.score.shared - first.score.occurrences[0])) > 1e-4
    assert np.max(np.abs(first.score.occurrences[0] - first.score.occurrences[1])) > 1e-4


def test_independent_autodiff_reference_restores_jax_precision_configuration(references):
    import jax

    from thermo_lab.finite_sweep_gradient_reference import build_finite_sweep_reference

    before = jax.config.jax_enable_x64
    reference = build_finite_sweep_reference(1, replace(build_checked_fixture(), beta=1.4))
    assert reference.accepted
    assert jax.config.jax_enable_x64 is before
