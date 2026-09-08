"""One-step shared-parameter refinement behavior."""

from __future__ import annotations

import importlib
import math

import pytest

from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference


def test_projected_shared_update_applies_descent_and_enforces_cap() -> None:
    refinement = importlib.import_module("thermo_lab.trajectory_reinforce_refinement")

    update = refinement.project_shared_parameters(
        initial_parameters=(0.25, -0.35, 0.20, 0.45, -0.30, -0.40, 0.25, 0.30, -0.20),
        gradient=(-10.0, 10.0, 0.5, 0.0, -0.5, 1.0, -1.0, 2.0, -2.0),
        learning_rate=0.25,
        parameter_cap=2.0,
    )

    assert update.raw_parameters == (
        2.75,
        -2.85,
        0.07500000000000001,
        0.45,
        -0.175,
        -0.65,
        0.5,
        -0.2,
        0.3,
    )
    assert update.updated_parameters == (
        2.0,
        -2.0,
        0.07500000000000001,
        0.45,
        -0.175,
        -0.65,
        0.5,
        -0.2,
        0.3,
    )
    assert update.cap_active_mask == (True, True, False, False, False, False, False, False, False)
    assert update.cap_active_parameter_count == 2


def test_projected_shared_update_rejects_nonfinite_gradient() -> None:
    refinement = importlib.import_module("thermo_lab.trajectory_reinforce_refinement")

    with pytest.raises(ValueError, match="gradient.*finite"):
        refinement.project_shared_parameters(
            initial_parameters=(0.0,) * 9,
            gradient=(math.nan,) + (0.0,) * 8,
            learning_rate=0.25,
            parameter_cap=2.0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"initial_parameters": (0.0,) * 8}, "initial_parameters.*nine"),
        ({"gradient": (0.0,) * 8}, "gradient.*nine"),
        ({"initial_parameters": (math.inf,) + (0.0,) * 8}, "initial_parameters.*finite"),
        ({"gradient": (False,) + (0.0,) * 8}, "gradient.*booleans"),
        ({"learning_rate": 0.0}, "learning_rate.*positive"),
        ({"learning_rate": True}, "learning_rate.*finite real"),
        ({"parameter_cap": 0.0}, "parameter_cap.*positive"),
        ({"parameter_cap": math.nan}, "parameter_cap.*finite real"),
        ({"initial_parameters": (2.01,) + (0.0,) * 8}, "initial_parameters.*cap"),
    ],
)
def test_projected_shared_update_rejects_invalid_inputs(
    overrides: dict[str, object], message: str
) -> None:
    refinement = importlib.import_module("thermo_lab.trajectory_reinforce_refinement")
    arguments: dict[str, object] = {
        "initial_parameters": (0.0,) * 9,
        "gradient": (0.0,) * 9,
        "learning_rate": 0.25,
        "parameter_cap": 2.0,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        refinement.project_shared_parameters(**arguments)


def test_exact_objective_evaluation_ties_updated_parameters_across_both_occurrences() -> None:
    refinement = importlib.import_module("thermo_lab.trajectory_reinforce_refinement")
    fixture = build_checked_fixture()
    updated_parameters = (
        0.2578964039775758,
        -0.3991077845810621,
        0.07684419619207026,
        0.5343311739208323,
        -0.2585365897347461,
        -0.35089221541893795,
        0.37315580380792975,
        0.2886649427783426,
        -0.2317278058001982,
    )

    evaluation = refinement.evaluate_exact_shared_objective(
        fixture=fixture,
        shared_parameters=updated_parameters,
    )

    assert evaluation.objective == pytest.approx(0.35555284701837758, rel=0.0, abs=1e-15)


def test_one_step_refinement_records_exact_before_and_after_improvement() -> None:
    refinement = importlib.import_module("thermo_lab.trajectory_reinforce_refinement")
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)

    result = refinement.build_one_step_refinement(
        fixture=fixture,
        exact=exact,
        sampled_shared_gradient=tuple(float(value) for value in exact.score.shared),
        learning_rate=0.25,
    )

    assert result.objective_before == pytest.approx(0.5246570826850282, rel=0.0, abs=1e-15)
    assert result.objective_after == pytest.approx(0.35555284701837758, rel=0.0, abs=1e-15)
    assert result.objective_improvement == pytest.approx(0.16910423566665062, rel=0.0, abs=1e-15)
    assert result.relative_objective_improvement == pytest.approx(
        0.32231383364011573, rel=0.0, abs=1e-15
    )
    assert result.objective_improved is True
    assert result.bounds_satisfied is True
    assert result.update.cap_active_parameter_count == 0


def test_projected_shared_update_rejects_overflow_before_projection() -> None:
    refinement = importlib.import_module("thermo_lab.trajectory_reinforce_refinement")

    with pytest.raises(ValueError, match="raw parameter.*finite"):
        refinement.project_shared_parameters(
            initial_parameters=(0.0,) * 9,
            gradient=(1e308,) + (0.0,) * 8,
            learning_rate=1e308,
            parameter_cap=2.0,
        )
