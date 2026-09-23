"""Contract and replay checks for the frozen three-operation comparison."""

import copy
import json

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256


@pytest.fixture(scope="module")
def study():
    from thermo_lab.return_fixture_study import build_study

    return build_study()


def test_exact_budget_and_arm_selection(study):
    assert study["request"]["schema"] == "return_fixture_objective_step.v1"
    assert study["request"]["occurrences"] == [[0, 1], [1, 2], [0, 1]]
    assert study["accounting"] == {
        "arm_evaluations": 1206,
        "preflight_evaluations": 19,
        "generation_evaluations": 1225,
        "samples": 0,
    }
    assert len(study["arms"]) == 6
    assert len({(arm["objective"], arm["step_rule"]) for arm in study["arms"]}) == 6
    for arm in study["arms"]:
        assert arm["evaluations"] == 201
        assert len(arm["rounds"]) == 25
        parameters = np.asarray(study["request"]["initial_parameters"])
        state = arm["initial"]
        for row in arm["rounds"]:
            assert len(row["trials"]) == 8
            assert row["gradient_norm"] == pytest.approx(
                np.linalg.norm(state["gradients"][arm["objective"]])
            )
            if arm["step_rule"] == "fixed":
                for trial in row["trials"]:
                    raw = parameters - 0.01 * np.asarray(state["gradients"][arm["objective"]])
                    proposal = np.clip(raw, -2, 2)
                    np.testing.assert_array_equal(trial["parameters"], proposal)
                    assert trial["clipped_components"] == int(np.count_nonzero(raw != proposal))
                    parameters, state = proposal, trial["evaluation"]
                assert row["selected_index"] == 7
            else:
                gradient = np.asarray(state["gradients"][arm["objective"]])
                value = state["objectives"][arm["objective"]]
                eligible = []
                for index, trial in enumerate(row["trials"]):
                    raw = parameters - 2.0**-index * gradient
                    proposal = np.clip(raw, -2, 2)
                    np.testing.assert_array_equal(trial["parameters"], proposal)
                    assert trial["clipped_components"] == int(np.count_nonzero(raw != proposal))
                    slope = float(gradient @ (proposal - parameters))
                    if (
                        slope < 0
                        and trial["evaluation"]["objectives"][arm["objective"]]
                        <= value + 1e-4 * slope
                    ):
                        eligible.append(index)
                assert row["selected_index"] == (eligible[0] if eligible else None)
                if eligible:
                    parameters = np.asarray(row["trials"][eligible[0]]["parameters"])
                    state = row["trials"][eligible[0]]["evaluation"]
            np.testing.assert_array_equal(row["parameters"], parameters)
        assert arm["final"] == state


@pytest.mark.parametrize("mutation", ["selection", "metric", "budget", "schedule", "numeric_type"])
def test_rehashed_forgery_is_rejected(study, mutation):
    from thermo_lab.return_fixture_study import validate_study

    bad = copy.deepcopy(study)
    if mutation == "selection":
        bad["arms"][1]["rounds"][0]["selected_index"] = 99
    elif mutation == "metric":
        bad["arms"][0]["final"]["metrics"]["survival"] += 1e-5
    elif mutation == "budget":
        bad["accounting"]["arm_evaluations"] -= 1
    elif mutation == "schedule":
        bad["request"]["occurrences"][2] = [1, 2]
        bad["request_digest"] = canonical_sha256(bad["request"])
    else:
        bad["arms"][0]["rounds"][0]["selected_index"] = 7.0
    bad["result_digest"] = canonical_sha256({k: v for k, v in bad.items() if k != "result_digest"})
    with pytest.raises(ValueError):
        validate_study(bad)


def test_zero_gradient_keeps_parameters_and_evaluates_all_candidates(monkeypatch):
    import thermo_lab.return_fixture_study as module

    real = module.evaluate
    calls = []

    def flat(parameters):
        calls.append(tuple(parameters))
        result = real(parameters)
        result["gradients"]["occupancy"] = [0.0] * 9
        return result

    monkeypatch.setattr(module, "evaluate", flat)
    arm = module.run_arm("occupancy", "backtracking")
    assert len(calls) == 201
    assert all(row["selected_index"] is None for row in arm["rounds"])
    assert arm["final"] == arm["initial"]


def test_completion_is_written_after_persisted_replay(tmp_path, monkeypatch):
    import thermo_lab.return_fixture_study as module

    destination = tmp_path / "study"
    render = module.render_report

    def assert_persisted(record):
        assert (destination / "study.json").exists()
        assert not (destination / "completion.json").exists()
        return render(record)

    monkeypatch.setattr(module, "render_report", assert_persisted)
    module.run_study(destination)
    completion = json.loads((destination / "completion.json").read_text())
    assert completion["status"] == "return_fixture_study_complete"
    assert completion["arms"] == 6 and completion["samples"] == 0
    with pytest.raises(FileExistsError):
        module.run_study(destination)


def test_failed_replay_leaves_no_completion(tmp_path, monkeypatch):
    import thermo_lab.return_fixture_study as module

    def fail(record):
        raise ValueError("replay failed")

    monkeypatch.setattr(module, "render_report", fail)
    destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="replay failed"):
        module.run_study(destination)
    assert not (destination / "completion.json").exists()


def test_report_shows_all_visited_rows_both_hops_and_optimizer_activity(study):
    from thermo_lab.return_fixture_study import render_report

    report = render_report(study)
    assert "Row 00" in report and "Row 01" in report and "Row 10" in report
    assert "forward 10" in report and "reverse 01" in report
    assert "visited row mae" in report.lower()
    assert "gradient norm" in report and "clipped components" in report
    for arm in study["arms"]:
        metric = arm["final"]["metrics"]
        assert f"{metric['row_mae']['00']:.9g}" in report
        assert f"{metric['forward_hop_10']:.9g}" in report
        assert (
            str(
                sum(trial["clipped_components"] for row in arm["rounds"] for trial in row["trials"])
            )
            in report
        )
