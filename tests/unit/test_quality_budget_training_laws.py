"""Catch law substitution, shared-score loss and reuse of stochastic roles."""

import numpy as np
import pytest

from thermo_lab.trajectory_reinforce import build_checked_fixture


@pytest.mark.parametrize("horizon", [1, 2, 4, 8, 16, 30, "equilibrium"])
def test_law_validation_retains_independent_gradient_oracles(horizon):
    from thermo_lab.quality_budget_training_laws import validate_training_law

    result = validate_training_law(horizon)
    assert result["maximum_exact_error"] <= 1e-12
    assert result["maximum_finite_difference_error"] <= 1e-7
    assert result["maximum_occupancy_replay_error"] == 0
    assert result["maximum_gradient_replay_error"] <= 1e-10
    assert len(result["diagnostics"]) == 3
    assert all(d["occupancy"]["sample_count"] == 32768 for d in result["diagnostics"])
    assert all(d["gradient"]["sample_count"] == 32768 for d in result["diagnostics"])
    assert result["exact_reference"]["mean"] == pytest.approx(
        np.sum(result["exact_reference"]["occurrence_mean"], axis=0), abs=1e-12
    )


@pytest.mark.parametrize(
    "invalid", [True, 1.0, 0, 3, "4", None, np.array("equilibrium"), np.array(["equilibrium"])]
)
def test_training_adapter_rejects_undeclared_laws(invalid):
    from thermo_lab.quality_budget_training_laws import sample_training_roles

    f = build_checked_fixture()
    with pytest.raises(ValueError, match="law"):
        sample_training_roles(
            (f.model_parameters.values,),
            (0, 0),
            f.occurrences,
            (0.1, 0.2, 0.7),
            horizon=invalid,
            occupancy_seed=11,
            gradient_seed=12,
        )


def test_training_adapter_rejects_dependent_roles():
    from thermo_lab.quality_budget_training_laws import sample_training_roles

    f = build_checked_fixture()
    with pytest.raises(ValueError, match="distinct"):
        sample_training_roles(
            (f.model_parameters.values,),
            (0, 0),
            f.occurrences,
            (0.1, 0.2, 0.7),
            horizon=1,
            occupancy_seed=11,
            gradient_seed=11,
        )


@pytest.mark.parametrize("horizon", [1, 2, 4, 8, 16, 30, "equilibrium"])
def test_both_laws_enforce_parameter_caps_before_sampling(horizon):
    from thermo_lab.quality_budget_training_laws import sample_training_roles

    f = build_checked_fixture()
    parameters = ((2.1, *f.model_parameters.values[1:]),)
    with pytest.raises(ValueError, match=r"\[-2, 2\]"):
        sample_training_roles(
            parameters,
            (0, 0),
            f.occurrences,
            (0.1, 0.2, 0.7),
            horizon=horizon,
            occupancy_seed=11,
            gradient_seed=12,
        )


@pytest.mark.parametrize("horizon", [1, 2, 4, 8, 16, 30, "equilibrium"])
@pytest.mark.parametrize("dimension", ["groups", "sites", "operations"])
def test_both_laws_reject_oversized_requests(horizon, dimension):
    from thermo_lab.quality_budget_training_laws import sample_training_roles

    f = build_checked_fixture()
    parameters, groups, sites, target = (
        (f.model_parameters.values,),
        (0, 0),
        f.occurrences,
        (0.1, 0.2, 0.7),
    )
    if dimension == "groups":
        parameters *= 38
    elif dimension == "sites":
        target = (0.0,) * 26
    else:
        groups, sites = (0,) * 501, ((0, 1),) * 501
    with pytest.raises(ValueError, match="bounded"):
        sample_training_roles(
            parameters, groups, sites, target, horizon=horizon, occupancy_seed=11, gradient_seed=12
        )


def test_law_substitution_is_detected_by_realized_draw_replay(monkeypatch):
    import thermo_lab.quality_budget_training_laws as laws

    original = laws.sample_training_roles

    def wrong_law(*args, **kwargs):
        return original(*args, **{**kwargs, "horizon": "equilibrium"})

    monkeypatch.setattr(laws, "sample_training_roles", wrong_law)
    with pytest.raises(ValueError, match="replay"):
        laws.validate_training_law(1)
