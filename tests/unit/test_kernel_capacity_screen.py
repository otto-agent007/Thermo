"""Focused checks for the one-feature kernel-capacity screen (M4I)."""

import numpy as np
import pytest

from thermo_lab import kernel_capacity_screen as screen
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.thermodynamic_kernel import KernelParameters, equilibrium_joint_conditional


def m3_visible(vector, horizon):
    if horizon == "equilibrium":
        joint = equilibrium_joint_conditional(KernelParameters(tuple(vector)))
    else:
        joint = finite_sweep_joint_law(KernelParameters(tuple(vector)), horizon).probabilities
    return joint.reshape(4, 2, 4).sum(axis=1)


@pytest.mark.parametrize("family", list(screen.FAMILIES))
def test_every_family_nests_the_m3_base_law(family):
    rng = np.random.default_rng(5)
    for vector in rng.uniform(-2, 2, (3, 9)):
        padded = np.zeros(screen.parameter_count(family))
        padded[:9] = vector
        for horizon in (1, 4, 30, "equilibrium"):
            law = screen.visible_law(padded, family, horizon, 2.0)
            assert np.max(np.abs(law - m3_visible(vector, horizon))) <= 1e-13


@pytest.mark.parametrize("family", list(screen.FAMILIES))
def test_sweep_leaves_the_boltzmann_conditional_stationary(family):
    rng = np.random.default_rng(9)
    theta = rng.uniform(-4, 4, screen.parameter_count(family))
    phi, _, _ = screen.STRUCTURES[family]
    weights = phi @ theta
    pi = np.exp(weights - weights.max(axis=1, keepdims=True))
    pi /= pi.sum(axis=1, keepdims=True)
    after, rows = screen._one_sweep(pi, theta, family)
    assert np.max(np.abs(after - pi)) <= 1e-12 and rows <= 1e-12
    long_run = screen.finite_law(theta, family, 30, 4.0)[0]
    assert np.allclose(long_run.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_output_coupling_changes_the_law_and_hidden_spin_needs_couplings():
    theta = np.zeros(10)
    theta[9] = -1.5
    law = screen.visible_law(theta, "oo", "equilibrium", 2.0)
    assert law[0, 1] > law[0, 0]  # anticorrelated outputs are favored
    field_only = np.zeros(12)
    field_only[9] = 1.7  # g's field alone cannot reach the outputs
    assert np.allclose(
        screen.visible_law(field_only, "h2", 4, 2.0),
        screen.visible_law(np.zeros(12), "h2", 4, 2.0),
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("family", screen.NEW_FAMILIES)
def test_objective_gradient_matches_centered_differences(family):
    rng = np.random.default_rng(13)
    target = np.zeros((4, 4))
    target[0, 0] = target[3, 3] = 1.0
    target[1, (1, 2)] = (0.9, 0.1)
    target[2, (2, 1)] = (0.8, 0.2)
    weights = np.array([12.0, 1.5, 2.5, 0.0])
    theta = rng.uniform(-3, 3, screen.parameter_count(family))
    _, analytic = screen.objective(theta, family, target, weights, 4.0)
    for k in range(len(theta)):
        step = np.zeros_like(theta)
        step[k] = 1e-6
        numeric = (
            screen.objective(theta + step, family, target, weights, 4.0)[0]
            - screen.objective(theta - step, family, target, weights, 4.0)[0]
        ) / 2e-6
        assert numeric == pytest.approx(analytic[k], rel=1e-6, abs=1e-7)


@pytest.fixture(scope="module")
def inputs():
    return screen.load_inputs()


def test_starts_are_deterministic_nested_and_inside_the_box(inputs):
    for family in screen.NEW_FAMILIES:
        for cap in screen.NEW_CAPS:
            starts = screen._starts(family, cap, 11, inputs)
            again = screen._starts(family, cap, 11, inputs)
            assert len(starts) == 21
            assert starts[0][0] == "m4h_warm_start"
            n = screen.parameter_count(family)
            assert np.array_equal(starts[0][1][:9], inputs["m4h"][cap]["parameters"][11])
            assert not np.any(starts[0][1][9:n])
            for (_, a), (_, b) in zip(starts, again, strict=True):
                assert a.shape == (n,) and np.array_equal(a, b) and np.all(np.abs(a) <= cap)


def test_preflight_passes(inputs):
    assert screen.preflight(inputs)["passed"]


def test_decision_follows_the_frozen_contract():
    def arm(family, cap, classification, status="complete"):
        return {"family": family, "cap": cap, "classification": classification, "status": status}

    good = {"k4": {"survival": 0.95, "hop_mae": 0.01, "asymmetry_mae": 0.01}}
    assert screen.classify(good) == "pass"
    assert screen.classify({"k4": {**good["k4"], "asymmetry_mae": 0.0101}}) == "survival_only"
    assert screen.classify({"k4": {**good["k4"], "survival": 0.9499}}) == "fail"
    assert (
        screen.decide([arm("h2", 2.0, "pass"), arm("oo", 2.0, "pass"), arm("oo", 4.0, "pass")])
        == "pass_oo_cap_2"
    )
    assert screen.decide([arm("h2", 2.0, "pass"), arm("oo", 4.0, "pass")]) == "pass_h2_cap_2"
    assert (
        screen.decide([arm("h2", 4.0, "survival_only"), arm("oo", 4.0, "fail")])
        == "survival_feasible_fidelity_fails"
    )
    assert screen.decide([arm("h2", 4.0, "fail")]) == "survival_fails_in_every_arm"
    assert screen.decide([arm("h2", 4.0, None, "integrity_failure")]) == "integrity_failure"
