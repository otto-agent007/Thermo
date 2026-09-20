"""Runner preflight retains all diagnostic steps without running study fits."""

import copy
import json

import pytest

from thermo_lab.hashing import canonical_sha256


@pytest.fixture(scope="module")
def evidence():
    from thermo_lab.quality_budget_runner_preflight import build_preflight

    return build_preflight()


def test_all_evolving_laws_replay_independently(evidence):
    assert len(evidence["fixture_fits"]) == 21
    assert evidence["fixture_updates"] == 105
    assert evidence["study_fits_executed"] == evidence["evaluation_cells_executed"] == 0
    assert evidence["full_m4g_ready"] is False
    assert evidence["independent_replay"]["steps_checked"] == 105
    assert evidence["independent_replay"]["maximum_occupancy_error"] == 0
    assert evidence["independent_replay"]["maximum_gradient_error"] < 1e-10
    fixture_roles = {
        s[k]
        for f in evidence["fixture_fits"]
        for s in f["steps"]
        for k in ("occupancy_seed", "gradient_seed")
    }
    study_roles = {
        s[k]
        for r in evidence["study_requests"]
        for s in r["roles"]
        for k in ("occupancy_seed", "gradient_seed")
    }
    assert len(fixture_roles) == len(study_roles) == 210
    assert fixture_roles.isdisjoint(study_roles)


@pytest.mark.parametrize("field", ["source", "gradient", "replay", "checkpoint", "ready"])
def test_report_rejects_rehashed_changes(evidence, field):
    from thermo_lab.quality_budget_runner_preflight import render_report

    value = copy.deepcopy(evidence)
    if field == "source":
        value["study_requests"][0]["inputs"]["initial_parameters"][0][0] += 0.01
    elif field == "gradient":
        value["fixture_fits"][0]["steps"][0]["gradient"]["component_sum"][0][0] += 1
    elif field == "replay":
        value["independent_replay"]["steps_checked"] = 104
    elif field == "checkpoint":
        value["fixture_fits"][0]["selected_parameters"] = value["fixture_fits"][0]["request"][
            "inputs"
        ]["initial_parameters"]
    else:
        value["full_m4g_ready"] = True
    value["result_digest"] = canonical_sha256(
        {k: v for k, v in value.items() if k != "result_digest"}
    )
    with pytest.raises(ValueError):
        render_report(value)


def test_persisted_report_and_completion(tmp_path):
    from thermo_lab.quality_budget_runner_preflight import write_preflight

    output = tmp_path / "runner"
    write_preflight(output)
    completion = json.loads((output / "completion.json").read_text())
    assert completion["status"] == "training_runner_component_complete"
    assert completion["study_fits_executed"] == 0
    assert (output / "report.md").exists()
    with pytest.raises(FileExistsError):
        write_preflight(output)


def test_replay_failure_does_not_publish_completion(tmp_path, monkeypatch):
    import thermo_lab.quality_budget_runner_preflight as gate

    def reject(*args, **kwargs):
        raise ValueError("injected replay failure")

    monkeypatch.setattr(gate, "render_report", reject)
    with pytest.raises(ValueError, match="injected"):
        gate.write_preflight(tmp_path / "failed")
    assert not (tmp_path / "failed" / "completion.json").exists()
