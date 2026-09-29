"""Independent finite-state checks of M5b's inner dynamics."""

import importlib.util
from itertools import product

import numpy as np
import pytest


def core():
    assert importlib.util.find_spec("thermo_lab.meta_ebm_thermalization_core"), (
        "M5b numerical implementation is missing"
    )
    from thermo_lab import meta_ebm_thermalization_core

    return meta_ebm_thermalization_core


def joint_fixture(x):
    """Build joint Gibbs probabilities directly from the fixture energy."""
    states = np.array(list(product((-1, 1), repeat=3)))  # y, w0, w1
    y, w0, w1 = states.T
    weight = np.exp(
        (0.2 * x - 0.1) * y
        + (0.3 * x + 0.4) * w0
        + (-0.2 * x - 0.3) * w1
        + (0.8 * w0 - 0.6 * w1) * y
    )
    law = weight / weight.sum()
    transition = np.zeros((8, 8))
    for i, (old_y, _, _) in enumerate(states):
        for j, (_, new_w0, new_w1) in enumerate(states):
            hidden_mask = (states[:, 1] == new_w0) & (states[:, 2] == new_w1)
            old_mask = states[:, 0] == old_y
            transition[i, j] = (
                law[hidden_mask & old_mask].sum()
                / law[old_mask].sum()
                * law[j]
                / law[hidden_mask].sum()
            )
    return states, law, transition


@pytest.mark.parametrize("k", [1, 2, 4])
def test_inner_power_matches_independent_joint_enumeration_and_reset(k):
    api = core()
    structure = {"blanket": [1], "triples": [((1, 2), 0.0), ((1, 3), 0.0)]}
    kernel = api.inner_kernel([0.2, -0.1, 0.3, -0.2, 0.4, -0.3, 0.8, -0.6], structure)
    rates = api.powered_rates(kernel, k)
    for row, x in enumerate((-1, 1)):
        states, law, transition = joint_fixture(x)
        power = np.linalg.matrix_power(transition, k)
        for start in range(8):
            expected_up = power[start, states[:, 0] == 1].sum()
            observed_up = rates[row, 0] if states[start, 0] == -1 else 1 - rates[row, 1]
            assert observed_up == pytest.approx(expected_up, abs=1e-13)
        assert kernel["q_plus"][row] == pytest.approx(law[states[:, 0] == 1].sum())
    assert np.max(1 - rates.sum(axis=1)) > 0  # finite-K remembers incoming y


def test_zero_hidden_spins_give_the_marginal_update_in_one_sweep():
    api = core()
    kernel = api.inner_kernel([0.3], {"blanket": [], "triples": []})
    rates = api.powered_rates(kernel, 1)
    np.testing.assert_allclose(rates, [[0.6456563062257954, 0.35434369377420455]])
    np.testing.assert_allclose(api.powered_rates(kernel, 32), rates)
    with pytest.raises(ValueError, match="positive integer"):
        api.powered_rates(kernel, 0)


def test_strong_coupling_retains_escape_when_lambda_rounds_to_one():
    api = core()
    kernel = api.inner_kernel([0.0, 0.0, 25.0], {"blanket": [], "triples": [((), 0)]})
    rates = api.powered_rates(kernel, 4)
    # Unbiased one-hidden-spin fixture: a=b=2*p*(1-p), p=sigmoid(-50).
    p = np.exp(-50) / (1 + np.exp(-50))
    assert kernel["lambda"][0] == 1.0
    np.testing.assert_allclose(rates[0], [8 * p, 8 * p], rtol=1e-13, atol=0)
    summary = api.mixing_summary(kernel)
    assert summary["k_star"] > 10**20
    assert summary["k_star_contraction"] <= 1e-14
    assert summary["k_tv_1e-6"] > summary["k_tv_1e-3"]


def test_rounding_uses_symmetric_zero_codebook_and_equal_coefficient_weight():
    api = core()
    rounded, summary = api.round_parameters([[0.5, 1.5, 2.5, -0.5, -1.5, 0, 7, -7], [0.1]], 7, 4)
    np.testing.assert_array_equal(rounded[0], [0, 2, 2, 0, -2, 0, 7, -7])
    assert summary["max_absolute"] == 0.5
    assert summary["mean_absolute"] == pytest.approx(2.6 / 9)
    assert summary["mean_cap_normalized"] == pytest.approx(2.6 / 63)
    assert summary["sites"][1]["mean_absolute"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="cap"):
        api.round_parameters([[7.1]], 7, 4)


def test_sweep_retains_joint_stationary_law_and_incoming_output():
    api = core()
    assert hasattr(api, "sweep"), "M5b current-output sweep is missing"
    spins = ((np.arange(4)[:, None] >> np.arange(2)) & 1) * 2 - 1
    target = np.exp(0.7 * spins.prod(axis=1) + spins @ [0.2, -0.3])
    target /= target.sum()
    rates, expected, ideal = [], np.eye(4), np.eye(4)
    for site, (field, escape) in enumerate(((0.2, 0.4), (-0.3, 0.6))):
        plus = 1 / (1 + np.exp(-2 * (field + 0.7 * spins[:, 1 - site])))
        rates.append(np.column_stack((escape * plus, escape * (1 - plus))))
        single, marginal = np.zeros((4, 4)), np.zeros((4, 4))
        for start in range(4):
            low, high = start & ~(1 << site), start | (1 << site)
            marginal[start, high] = plus[start]
            marginal[start, low] = 1 - plus[start]
        single = escape * marginal + (1 - escape) * np.eye(4)
        expected = expected @ single
        ideal = ideal @ marginal
    observed = api.sweep(np.eye(4), rates)
    np.testing.assert_allclose(observed, expected, atol=1e-15)
    np.testing.assert_allclose(target @ observed, target, atol=1e-15)
    assert np.max(np.abs(observed - ideal)) > 0.1
    np.testing.assert_allclose(api.stationary_law(observed), target, atol=1e-14)
    metrics = api.chain_metrics(observed, ideal, ideal, target, spins, horizon=4)
    assert metrics["bias"] < 1e-13
    assert metrics["tv_target"][0] == pytest.approx(0.5 * np.abs(0.25 - target).sum())
    assert metrics["tv_own_stationary"][4] < metrics["tv_own_stationary"][0]
    assert metrics["tv_ideal"][0] == 0
    assert metrics["tv_ideal"][1] > 0


def test_stationary_solver_rejects_nonunique_or_invalid_chains():
    api = core()
    assert hasattr(api, "stationary_law"), "M5b stationary solver is missing"
    for matrix in (
        np.eye(2),
        np.array([[0.1, 0.3], [0.2, 0.8]]),
        np.array([[1.1, -0.1], [0.5, 0.5]]),
    ):
        with pytest.raises(api.NumericalIntegrityError):
            api.stationary_law(matrix)


def test_runtime_controls_detect_a_sweep_that_drops_incoming_output(monkeypatch):
    api = core()
    assert hasattr(api, "integrity_controls"), "runtime integrity controls are missing"
    assert api.integrity_controls()["passed"]

    def wrong_sweep(rows, rates):
        return np.full_like(rows, 1 / rows.shape[1])

    monkeypatch.setattr(api, "sweep", wrong_sweep)
    with pytest.raises(api.NumericalIntegrityError, match="control"):
        api.integrity_controls()


def test_maximizing_input_uses_resolved_rate_when_lambdas_round_equal():
    api = core()
    kernel = api.inner_kernel([0, 0, 1, -1, 25], {"blanket": [1], "triples": [((), 0)]})
    assert kernel["lambda"].tolist() == [1.0, 1.0]
    assert kernel["rate"][1] < kernel["rate"][0]
    assert api.mixing_summary(kernel)["maximizing_input"] == 1
