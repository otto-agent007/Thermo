"""Exact local-gradient and complete-record checks for the fidelity pilot."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_lab.trajectory_reinforce import build_checked_fixture


@pytest.mark.parametrize(
    "parameters",
    [
        list(build_checked_fixture().model_parameters.values),
        [0.15, -0.22, 0.31, -0.11, 0.27, -0.14, 0.18, -0.26, 0.09],
    ],
)
def test_both_hop_penalties_have_correct_tied_gradient(parameters):
    # A wrong row/output index or a missing shared derivative must fail here.
    from thermo_lab.return_fixture_fidelity_pilot import evaluate_penalty

    vector = np.asarray(parameters)
    base = evaluate_penalty(vector, 0)
    weighted = evaluate_penalty(vector, 10)
    assert weighted["objective"] - base["objective"] == pytest.approx(
        10 * (weighted["forward_error"] ** 2 + weighted["reverse_error"] ** 2),
        abs=1e-12,
    )
    for component in range(9):
        displacement = np.eye(9)[component] * 1e-6
        central = (
            evaluate_penalty(vector + displacement, 10)["objective"]
            - evaluate_penalty(vector - displacement, 10)["objective"]
        ) / 2e-6
        assert weighted["gradient"][component] == pytest.approx(central, abs=1e-7)


def test_weight_zero_reproduces_saved_path_kl_backtracking():
    from thermo_lab.return_fixture_fidelity_pilot import run_arm

    arm = run_arm(0)
    previous = json.loads(
        gzip.decompress(
            (
                Path(__file__).parents[2]
                / "docs/experiment-reports/2026-09-23-three-operation-return-fixture/study.json.gz"
            ).read_bytes()
        )
    )
    archived = next(
        row
        for row in previous["arms"]
        if row["objective"] == "trajectory_kl" and row["step_rule"] == "backtracking"
    )
    assert arm["evaluations"] == 201
    assert len(arm["rounds"]) == 25
    assert all(len(round_row["trials"]) == 8 for round_row in arm["rounds"])
    assert arm["initial"]["base"] == archived["initial"]
    assert arm["final"]["base"] == archived["final"]
    for new, old in zip(arm["rounds"], archived["rounds"], strict=True):
        assert new["selected_index"] == old["selected_index"]
        assert new["parameters"] == old["parameters"]
        for trial, previous_trial in zip(new["trials"], old["trials"], strict=True):
            assert trial["parameters"] == previous_trial["parameters"]
            assert trial["evaluation"]["base"] == previous_trial["evaluation"]
    final = arm["final"]["base"]
    assert final["metrics"]["survival"] == pytest.approx(0.934739588, abs=1e-9)
    assert final["metrics"]["forward_hop_10"] == pytest.approx(0.000411512102, abs=1e-11)
    assert final["metrics"]["reverse_hop_01"] == pytest.approx(0.00469740014, abs=1e-11)


@pytest.fixture(scope="module")
def study():
    from thermo_lab.return_fixture_fidelity_pilot import build_study

    return build_study()


def test_complete_arm_budget_and_predeclared_decision(study):
    from thermo_lab.return_fixture_fidelity_pilot import WEIGHTS

    assert study["request"]["weights"] == list(WEIGHTS) == [0, 1, 10, 100]
    assert study["request"]["qualification"] == {
        "survival_min": 0.95,
        "per_hop_absolute_error_max": 0.01,
    }
    assert study["accounting"]["arm_evaluations"] == 804
    assert study["accounting"]["samples"] == 0
    for arm in study["arms"]:
        assert arm["evaluations"] == 201
        assert len(arm["rounds"]) == 25
        state = arm["initial"]
        parameters = np.array(study["request"]["initial_parameters"])
        for row in arm["rounds"]:
            assert len(row["trials"]) == 8
            gradient = np.array(state["gradient"])
            eligible = []
            for index, trial in enumerate(row["trials"]):
                raw = parameters - 2.0**-index * gradient
                proposal = np.clip(raw, -2, 2)
                np.testing.assert_array_equal(trial["parameters"], proposal)
                slope = float(gradient @ (proposal - parameters))
                if (
                    slope < 0
                    and trial["evaluation"]["objective"] <= state["objective"] + 1e-4 * slope
                ):
                    eligible.append(index)
            assert row["selected_index"] == (eligible[0] if eligible else None)
            if eligible:
                parameters = np.asarray(row["trials"][eligible[0]]["parameters"])
                state = row["trials"][eligible[0]]["evaluation"]
            np.testing.assert_array_equal(row["parameters"], parameters)
        assert arm["final"] == state
        metric = state["base"]["metrics"]
        qualifies = (
            metric["survival"] >= 0.95
            and abs(state["forward_error"]) <= 0.01
            and abs(state["reverse_error"]) <= 0.01
        )
        assert state["qualifies"] is qualifies


def test_rehashed_metric_forgery_fails_complete_numerical_replay(study):
    from thermo_lab.hashing import canonical_sha256
    from thermo_lab.return_fixture_fidelity_pilot import validate_study

    forged = copy.deepcopy(study)
    forged["arms"][1]["rounds"][3]["trials"][0]["evaluation"]["reverse_error"] += 0.002
    forged["result_digest"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "result_digest"}
    )
    with pytest.raises(ValueError, match="replay"):
        validate_study(forged)
