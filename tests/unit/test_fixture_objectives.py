import importlib

import numpy as np
import pytest

from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.survival_gradients import survival_gradient
from thermo_lab.thermodynamic_kernel import KernelParameters
from thermo_lab.trajectory_reinforce import build_checked_fixture, terminal_law


def api():
    assert importlib.util.find_spec("thermo_lab.fixture_objectives"), "missing exact evaluator"
    return importlib.import_module("thermo_lab.fixture_objectives")


def test_objectives_and_all_shared_derivatives():
    evaluate = api().evaluate
    f = build_checked_fixture()
    p = np.array(f.model_parameters.values)
    r = evaluate(p)
    law = finite_sweep_joint_law(KernelParameters(tuple(p)), 4).probabilities
    target = terminal_law((f.target_conditional, f.target_conditional))
    full = terminal_law((law, law))
    expected = np.sum((np.array(full.occupancy) - target.occupancy) ** 2)
    assert r["objectives"]["occupancy"] == pytest.approx(expected, abs=1e-12)
    s = survival_gradient(p[None], [0, 0], [(0, 1), (1, 2)], horizon=4, site_count=3)
    assert r["metrics"]["survival"] == pytest.approx(s["survival"], abs=1e-12)
    assert r["metrics"]["terminal_leakage"] == pytest.approx(full.particle_number_leakage)
    assert r["metrics"]["survival"] < 1 - r["metrics"]["terminal_leakage"]
    assert r["objectives"]["trajectory_kl"] >= r["objectives"]["valid_terminal"]
    assert r["objectives"]["valid_terminal"] >= -np.log(r["metrics"]["survival"])
    assert r["metrics"]["survival_lower_bound"] <= r["metrics"]["survival"] + 1e-12
    for key in r["objectives"]:
        for j in range(9):
            delta = np.eye(9)[j] * 1e-6
            fd = (
                evaluate(p + delta)["objectives"][key] - evaluate(p - delta)["objectives"][key]
            ) / 2e-6
            assert r["gradients"][key][j] == pytest.approx(fd, abs=1e-7)


def test_zero_parameters_known_survival_and_kl_chain_rule():
    r = api().evaluate(np.zeros(9))
    assert r["metrics"]["survival"] == pytest.approx(3 / 16)
    assert r["objectives"]["trajectory_kl"] == pytest.approx(
        sum(r["metrics"]["occurrence_kl"]), abs=1e-12
    )
    assert sum(r["metrics"]["terminal_probabilities"]) == pytest.approx(1)
    assert sum(r["metrics"]["valid_terminal_masses"]) == pytest.approx(3 / 16)


def test_visible_kl_matches_independent_local_chain_rule():
    f = build_checked_fixture()
    r = api().evaluate(f.model_parameters.values)
    v = finite_sweep_joint_law(f.model_parameters, 4).probabilities.reshape(4, 2, 4).sum(1)
    target = f.target_conditional
    local = np.array(
        [
            sum(t * np.log(t / m) for t, m in zip(tr, vr, strict=True) if t > 0)
            for tr, vr in zip(target, v, strict=True)
        ]
    )
    second = target[2, ::2].sum() * local[0] + target[2, 1::2].sum() * local[2]
    np.testing.assert_allclose(r["metrics"]["occurrence_kl"], [local[2], second], atol=1e-12)


@pytest.mark.parametrize("parameters", [[0.0] * 9, [0.5] * 9, [-1.5] * 9])
def test_unique_valid_path_per_endpoint_makes_path_objectives_equivalent(parameters):
    r = api().evaluate(parameters)
    assert r["objectives"]["trajectory_kl"] == pytest.approx(
        r["objectives"]["valid_terminal"], abs=1e-12
    )
    np.testing.assert_allclose(
        r["gradients"]["trajectory_kl"], r["gradients"]["valid_terminal"], atol=1e-12
    )


@pytest.mark.parametrize(
    "p", [[0] * 8, [True] * 9, ["0"] * 9, [np.nan] * 9, [np.inf] * 9, [2.01] * 9, [[0] * 9]]
)
def test_rejects_invalid_parameters(p):
    with pytest.raises(ValueError):
        api().evaluate(p)
