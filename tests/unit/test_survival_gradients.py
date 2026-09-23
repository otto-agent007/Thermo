"""Detect missing outside-edge survival, occurrence reduction and wrong derivative laws."""

import importlib

import numpy as np
import pytest

from thermo_lab.conservation_diagnostic import exact_survival
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.thermodynamic_kernel import KernelParameters, equilibrium_joint_conditional


def audit_api():
    spec = importlib.util.find_spec("thermo_lab.survival_gradients")
    assert spec is not None, "survival gradient implementation is missing"
    return importlib.import_module("thermo_lab.survival_gradients").survival_gradient


@pytest.mark.parametrize("horizon", [1, 2, 4, 8, 16, 30, "equilibrium"])
def test_shared_gradient_matches_independent_finite_differences(horizon):
    fn = audit_api()
    p = np.array([[0.1, -0.2, 0.3, -0.1, 0.2, -0.15, 0.25, 0.05, -0.3]])
    groups, sites = [0, 0], [(0, 1), (1, 2)]

    def value(x):
        param = KernelParameters(tuple(x[0]))
        table = (
            equilibrium_joint_conditional(param)
            if horizon == "equilibrium"
            else finite_sweep_joint_law(param, horizon).probabilities
        )
        return np.log(
            exact_survival(table[None], groups, sites, site_count=3)[-1]["survival_probability"]
        )

    result = fn(p, groups, sites, horizon=horizon, site_count=3)
    np.testing.assert_allclose(result["log_survival"], value(p), atol=1e-12)
    for j in range(9):
        delta = np.zeros_like(p)
        delta[0, j] = 1e-6
        fd = (value(p + delta) - value(p - delta)) / 2e-6
        np.testing.assert_allclose(result["gradient"][0][j], fd, atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        np.sum(result["occurrence_gradients"], axis=0), result["gradient"][0], atol=1e-12
    )
    assert (
        abs(
            result["survival"]
            + result["first_exit_creation"]
            + result["first_exit_destruction"]
            - 1
        )
        < 1e-12
    )


def test_uniform_outputs_kill_outside_edge_paths():
    r = audit_api()(np.zeros((1, 9)), [0, 0], [(0, 1), (1, 2)], horizon=1, site_count=3)
    # First operation: half survives. Second: left particle survives with 1/4,
    # right particle with 1/2, each previously has mass 1/4.
    assert r["survival"] == pytest.approx(3 / 16)


@pytest.mark.parametrize("horizon", [True, 0, 31, "4"])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError):
        audit_api()(np.zeros((1, 9)), [0], [(0, 1)], horizon=horizon, site_count=3)


def test_multiple_groups_route_gradients_and_long_chain_stays_scaled():
    fn = audit_api()
    p = np.array([[0.1] * 9, [-0.2] * 9])
    groups, sites = [0, 1, 0, 1], [(0, 1), (1, 2), (0, 2), (0, 1)]
    r = fn(p, groups, sites, horizon=4, site_count=3)
    for group in range(2):
        np.testing.assert_allclose(
            np.array(r["occurrence_gradients"])[np.array(groups) == group].sum(axis=0),
            r["gradient"][group],
            atol=1e-12,
        )
        delta = np.zeros_like(p)
        delta[group, 3] = 1e-6

        def value(x):
            tables = np.array(
                [finite_sweep_joint_law(KernelParameters(tuple(row)), 4).probabilities for row in x]
            )
            return np.log(
                exact_survival(tables, groups, sites, site_count=3)[-1]["survival_probability"]
            )

        np.testing.assert_allclose(
            r["gradient"][group][3], (value(p + delta) - value(p - delta)) / 2e-6, atol=1e-7, rtol=0
        )
    long = fn(np.zeros((1, 9)), [0] * 500, [(0, 1)] * 500, horizon=1, site_count=3)
    assert long["log_survival"] == pytest.approx(-500 * np.log(2), abs=1e-11)
    assert long["survival"] == pytest.approx(2.0**-500, rel=1e-12, abs=0)
    assert np.all(np.isfinite(long["gradient"]))
