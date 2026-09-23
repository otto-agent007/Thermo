import copy
import importlib
import json

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256


def api():
    assert importlib.util.find_spec("thermo_lab.fixture_objective_study"), "missing study runner"
    return importlib.import_module("thermo_lab.fixture_objective_study")


@pytest.fixture(scope="module")
def study():
    return api().build_study()


def test_exact_budget_and_selected_updates(study):
    assert study["request"]["path_objective_equivalence"] == "unique valid path per terminal state"
    assert study["accounting"] == {
        "arm_evaluations": 1206,
        "preflight_evaluations": 19,
        "generation_evaluations": 1225,
        "samples": 0,
    }
    assert len(study["arms"]) == 6
    assert len({(a["objective"], a["step_rule"]) for a in study["arms"]}) == 6
    for arm in study["arms"]:
        assert arm["evaluations"] == 201
        assert len(arm["rounds"]) == 25
        current = np.array(study["request"]["initial_parameters"])
        state = arm["initial"]
        for row in arm["rounds"]:
            assert len(row["trials"]) == 8
            if arm["step_rule"] == "fixed":
                for trial in row["trials"]:
                    current = np.clip(
                        current - 0.01 * np.array(state["gradients"][arm["objective"]]), -2, 2
                    )
                    np.testing.assert_array_equal(trial["parameters"], current)
                    state = trial["evaluation"]
                assert row["selected_index"] == 7
            else:
                eligible = []
                g = np.array(state["gradients"][arm["objective"]])
                f = state["objectives"][arm["objective"]]
                for index, trial in enumerate(row["trials"]):
                    proposal = np.clip(current - 2.0 ** (-index) * g, -2, 2)
                    np.testing.assert_array_equal(trial["parameters"], proposal)
                    slope = float(g @ (proposal - current))
                    value = trial["evaluation"]["objectives"][arm["objective"]]
                    if slope < 0 and value <= f + 1e-4 * slope:
                        eligible.append(index)
                chosen = eligible[0] if eligible else None
                assert row["selected_index"] == chosen
                if chosen is not None:
                    current = np.array(row["trials"][chosen]["parameters"])
                    state = row["trials"][chosen]["evaluation"]
            np.testing.assert_array_equal(row["parameters"], current)
        assert arm["final"] == state
        assert np.max(np.abs(current)) <= 2


@pytest.mark.parametrize("mutation", ["selection", "metric", "budget", "request", "numeric_type"])
def test_rehashed_forgery_rejected(study, mutation):
    bad = copy.deepcopy(study)
    if mutation == "selection":
        bad["arms"][1]["rounds"][0]["selected_index"] = 99
    elif mutation == "metric":
        bad["arms"][0]["final"]["metrics"]["survival"] += 1e-5
    elif mutation == "budget":
        bad["accounting"]["arm_evaluations"] -= 1
    elif mutation == "request":
        bad["request"]["horizon"] = 30
        bad["request_digest"] = canonical_sha256(bad["request"])
    else:
        bad["arms"][0]["rounds"][0]["selected_index"] = 7.0
    bad["result_digest"] = canonical_sha256({k: v for k, v in bad.items() if k != "result_digest"})
    with pytest.raises(ValueError):
        api().validate_study(bad)


def test_no_descent_retains_parameters_and_evaluates_every_proposal(monkeypatch):
    module = api()
    real = module.evaluate
    calls = []

    def flat(p):
        calls.append(list(p))
        result = real(p)
        result["gradients"]["occupancy"] = [0.0] * 9
        return result

    monkeypatch.setattr(module, "evaluate", flat)
    arm = module.run_arm("occupancy", "backtracking")
    assert len(calls) == 201
    assert all(row["selected_index"] is None for row in arm["rounds"])
    assert arm["final"] == arm["initial"]


def test_release_reloads_before_completion(tmp_path, monkeypatch):
    module = api()
    destination = tmp_path / "run"
    original = module.render_report

    def checked(record):
        assert (destination / "study.json").exists()
        assert not (destination / "completion.json").exists()
        return original(record)

    monkeypatch.setattr(module, "render_report", checked)
    module.run_study(destination)
    completion = json.loads((destination / "completion.json").read_text())
    persisted = json.loads((destination / "study.json").read_text())
    assert completion["result_digest"] == persisted["result_digest"]
    assert completion["arms"] == 6
    with pytest.raises(FileExistsError):
        module.run_study(destination)


def test_failed_report_leaves_no_completion(tmp_path, monkeypatch):
    module = api()

    def fail(record):
        raise ValueError("replay failed")

    monkeypatch.setattr(module, "render_report", fail)
    with pytest.raises(ValueError, match="replay failed"):
        module.run_study(tmp_path / "failed")
    assert not (tmp_path / "failed" / "completion.json").exists()
