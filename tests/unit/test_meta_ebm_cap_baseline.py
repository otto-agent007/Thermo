"""Focused checks for the M5a meta-EBM cap baseline."""

import numpy as np
import pytest

from thermo_lab import meta_ebm_cap_baseline as m5a


def _random_kernel(rng, k=4, nh=3, scale=3.0):
    structure = {"blanket": list(range(k)), "triples": [((0, 1), 0.0)] * nh}
    vector = rng.uniform(-scale, scale, m5a.parameter_count(structure))
    return structure, vector, m5a.blanket_inputs(structure)


def test_target_recipe_is_deterministic_distinct_and_reading_b_rescales_a():
    a, again, b = m5a.make_target(3, "A"), m5a.make_target(3, "A"), m5a.make_target(3, "B")
    assert a == again
    assert len({tuple(p) for p in a["pairs"]}) == m5a.PAIR_COUNT
    assert len({tuple(t) for t in a["triples"]}) == m5a.TRIPLE_COUNT
    assert a["pairs"] == b["pairs"] and a["triples"] == b["triples"]
    assert a["fields"] == b["fields"]
    assert np.array_equal(np.array(a["pair_weights"]) * 0.5, b["pair_weights"])
    assert np.array_equal(np.array(a["triple_weights"]) * (1.0 / 6.0), b["triple_weights"])


def test_closed_form_matches_enumeration_of_hidden_spins():
    rng = np.random.default_rng(1)
    for _ in range(5):
        structure, vector, inputs = _random_kernel(rng)
        closed = m5a.expit(2.0 * m5a.kernel_logit(vector, structure, inputs))
        brute = m5a.brute_force_probability(vector, structure, inputs)
        assert np.max(np.abs(closed - brute)) <= 1e-12


def test_without_hidden_biases_the_logit_has_no_even_part():
    """The research note's finding: bias-free kernels cannot express x_m x_m'."""
    rng = np.random.default_rng(2)
    structure, vector, inputs = _random_kernel(rng)
    _, _, _, b, _ = m5a.unpack(vector, structure)
    b[:] = 0.0  # unpack returns views into the parameter array
    vector = np.asarray(vector)
    even = m5a.kernel_logit(vector, structure, inputs) + m5a.kernel_logit(
        vector, structure, -inputs
    )
    assert np.ptp(even) <= 1e-12


def test_objective_gradient_matches_centered_differences():
    rng = np.random.default_rng(3)
    structure, vector, inputs = _random_kernel(rng, scale=1.5)
    target_logit = rng.normal(0, 1, len(inputs))
    _, analytic = m5a.objective(vector, structure, inputs, target_logit)
    for k in range(len(vector)):
        step = np.zeros_like(vector)
        step[k] = 1e-6
        numeric = (
            m5a.objective(vector + step, structure, inputs, target_logit)[0]
            - m5a.objective(vector - step, structure, inputs, target_logit)[0]
        ) / 2e-6
        assert numeric == pytest.approx(analytic[k], rel=1e-6, abs=1e-9)


def test_constructive_recipe_improves_with_the_cap_and_stays_inside_it():
    structure = m5a.structures(m5a.make_target(1, "B"))[0]
    inputs = m5a.blanket_inputs(structure)
    exact = m5a.expit(2.0 * m5a.exact_logit(structure, inputs))
    errors = []
    for cap in m5a.CAPS:
        vector = m5a.constructive_parameters(structure, cap)
        assert np.max(np.abs(vector)) <= cap
        closed = m5a.expit(2.0 * m5a.kernel_logit(vector, structure, inputs))
        errors.append(float(np.max(np.abs(closed - exact))))
    assert errors[-1] < 1e-6 < errors[0]


def test_ideal_chain_is_exact_for_the_target():
    checks = m5a.ideal_checks(m5a.make_target(0, "A"))
    assert checks["stationary_residual"] <= m5a.TOLERANCE["stationary"]
    assert checks["detailed_balance_residual"] <= m5a.TOLERANCE["detailed_balance"]


def test_compiled_chain_obeys_the_proven_bound():
    target = m5a.make_target(1, "B")
    parameters = [
        m5a.constructive_parameters(structure, 0.5).tolist() for structure in m5a.structures(target)
    ]
    # A loose rho keeps the test fast; the bound only gets weaker as rho grows.
    chain = m5a.evaluate_chain(
        ("B", 1, 0.5, "constructive", parameters, {"dobrushin": 0.95, "slem": 0.5})
    )
    assert chain["bound_slack_min"] >= -m5a.TOLERANCE["bound"]
    assert chain["bias"] <= chain["floor_dobrushin"] + m5a.TOLERANCE["bound"]
    assert chain["stationary_residual"] <= m5a.TOLERANCE["stationary"]
    assert max(site["brute_force_error"] for site in chain["sites"]) <= 1e-12
    assert chain["delta"][0] == 0.0 and len(chain["delta"]) == m5a.HORIZON + 1
