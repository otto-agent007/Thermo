"""Independent full-conditional and complete-record checks for the row pilot."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.thermodynamic_kernel import KernelParameters
from thermo_lab.trajectory_reinforce import build_checked_fixture


def assert_equivalent_trace(actual, expected):
    """Match archived numeric traces while allowing tiny CPU rounding."""
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equivalent_trace(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            assert_equivalent_trace(value, reference)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-12, rel=0)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "parameters",
    [
        list(build_checked_fixture().model_parameters.values),
        [0.15, -0.22, 0.31, -0.11, 0.27, -0.14, 0.18, -0.26, 0.09],
    ],
)
def test_full_row_penalty_includes_all_sixteen_outcomes_and_tied_gradient(parameters):
    # Dropping row 11 or an invalid output changes the objective and fails here.
    from thermo_lab.return_fixture_full_row_pilot import evaluate_full_row

    p = np.asarray(parameters)
    baseline = evaluate_full_row(p, 0)
    weighted = evaluate_full_row(p, 10)
    conditional = finite_sweep_joint_law(KernelParameters(tuple(p)), 4)
    visible = conditional.probabilities.reshape(4, 2, 4).sum(axis=1)
    target = build_checked_fixture().target_conditional
    errors = np.abs(visible - target)
    assert weighted["objective"] - baseline["objective"] == pytest.approx(
        10 * np.square(errors).sum(), abs=1e-12
    )
    np.testing.assert_allclose(weighted["conditional"], visible, atol=1e-12)
    for row in range(4):
        name = f"{row:02b}"
        assert weighted["row_max_abs"][name] == pytest.approx(errors[row].max())
        assert weighted["row_total_variation"][name] == pytest.approx(errors[row].sum() / 2)
    assert weighted["max_abs_error"] == pytest.approx(errors.max())
    for component in range(9):
        delta = np.eye(9)[component] * 1e-6
        difference = (
            evaluate_full_row(p + delta, 10)["objective"]
            - evaluate_full_row(p - delta, 10)["objective"]
        ) / 2e-6
        assert weighted["gradient"][component] == pytest.approx(difference, abs=1e-7)


def test_zero_forward_hop_cannot_qualify_even_with_high_survival():
    from thermo_lab.return_fixture_full_row_pilot import qualifies

    target = build_checked_fixture().target_conditional.copy()
    model = target.copy()
    model[2, 2] += model[2, 1]
    model[2, 1] = 0
    assert qualifies(0.99, target, target)
    assert not qualifies(0.99, model, target)
    assert not qualifies(0.94, target, target)


@pytest.fixture(scope="module")
def study():
    from thermo_lab.return_fixture_full_row_pilot import build_study

    return build_study()


def test_predeclared_budget_and_armijo_reconstruction(study):
    assert study["request"]["weights"] == [0, 1, 10, 100]
    assert study["request"]["qualification"] == {
        "survival_min": 0.95,
        "max_all_row_absolute_error": 0.005,
    }
    assert study["accounting"]["arm_evaluations"] == 804
    assert study["accounting"]["samples"] == 0
    assert study["gradient_preflight"]["checked_components"] == 18
    assert len(study["arms"]) == 4
    for arm in study["arms"]:
        assert arm["evaluations"] == 201
        state = arm["initial"]
        p = np.asarray(study["request"]["initial_parameters"])
        assert len(arm["rounds"]) == 25
        for round_row in arm["rounds"]:
            gradient = np.asarray(state["gradient"])
            accepted = []
            assert len(round_row["trials"]) == 8
            for index, trial in enumerate(round_row["trials"]):
                proposal = np.clip(p - 2.0**-index * gradient, -2, 2)
                np.testing.assert_array_equal(trial["parameters"], proposal)
                slope = float(gradient @ (proposal - p))
                if (
                    slope < 0
                    and trial["evaluation"]["objective"] <= state["objective"] + 1e-4 * slope
                ):
                    accepted.append(index)
            chosen = accepted[0] if accepted else None
            assert round_row["selected_index"] == chosen
            if chosen is not None:
                p = np.asarray(round_row["trials"][chosen]["parameters"])
                state = round_row["trials"][chosen]["evaluation"]
            np.testing.assert_array_equal(round_row["parameters"], p)
        assert state == arm["final"]
        assert arm["final"]["qualifies"] == (
            arm["final"]["base"]["metrics"]["survival"] >= 0.95
            and arm["final"]["max_abs_error"] <= 0.005
        )


def test_weight_zero_reproduces_archived_path_kl_choices(study):
    from thermo_lab.return_fixture_full_row_pilot import WEIGHTS

    assert list(WEIGHTS) == [0, 1, 10, 100]
    unchanged = study["arms"][0]
    archived_study = json.loads(
        gzip.decompress(
            (
                Path(__file__).parents[2]
                / "docs/experiment-reports/2026-09-23-three-operation-return-fixture/study.json.gz"
            ).read_bytes()
        )
    )
    archived = next(
        arm
        for arm in archived_study["arms"]
        if arm["objective"] == "trajectory_kl" and arm["step_rule"] == "backtracking"
    )
    assert unchanged["final"]["base"]["metrics"]["survival"] == pytest.approx(0.934739588, abs=1e-9)
    assert_equivalent_trace(unchanged["initial"]["base"], archived["initial"])
    assert_equivalent_trace(unchanged["final"]["base"], archived["final"])
    for current, earlier in zip(unchanged["rounds"], archived["rounds"], strict=True):
        assert current["selected_index"] == earlier["selected_index"]
        assert_equivalent_trace(current["parameters"], earlier["parameters"])
        for trial, reference in zip(current["trials"], earlier["trials"], strict=True):
            assert_equivalent_trace(trial["parameters"], reference["parameters"])
            assert_equivalent_trace(trial["evaluation"]["base"], reference["evaluation"])


def test_rehashed_proposal_mutation_fails_complete_replay(study):
    from thermo_lab.hashing import canonical_sha256
    from thermo_lab.return_fixture_full_row_pilot import validate_study

    forged = copy.deepcopy(study)
    forged["arms"][1]["rounds"][3]["trials"][0]["evaluation"]["conditional"][1][0] += 0.01
    forged["result_digest"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "result_digest"}
    )
    with pytest.raises(ValueError, match="replay"):
        validate_study(forged)


def test_report_tables_have_one_delimiter_per_column(study):
    from thermo_lab.return_fixture_full_row_pilot import render_report

    lines = [line for line in render_report(study).splitlines() if line.startswith("|")]
    assert len(lines) == 12
    for start, columns in ((0, 10), (6, 17)):
        for row in lines[start : start + 6]:
            assert len(row.strip("|").split("|")) == columns
