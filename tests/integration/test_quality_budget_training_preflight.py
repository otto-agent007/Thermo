"""Persisted law-validation evidence must replay before release."""

import copy
import json

import pytest

from thermo_lab.hashing import canonical_sha256


@pytest.fixture(scope="module")
def evidence():
    from thermo_lab.quality_budget_training_preflight import build_preflight

    return build_preflight()


def test_all_laws_validate_without_consuming_study_roles(evidence):
    from thermo_lab.quality_budget_protocol import fit_manifest, role_schedule

    assert [c["horizon"] for c in evidence["laws"]] == [1, 2, 4, 8, 16, 30, "equilibrium"]
    assert evidence["fits_executed"] == evidence["evaluation_cells_executed"] == 0
    assert evidence["full_m4g_ready"] is False
    study = {role_schedule(s)["evaluation_seed"] for s in (0, 1, 2)}
    study.update(
        r[k]
        for f in fit_manifest()
        for r in f["steps"]
        for k in ("occupancy_seed", "gradient_seed")
    )
    validation = [
        d[k]
        for c in evidence["laws"]
        for d in c["diagnostics"]
        for k in ("occupancy_seed", "gradient_seed")
    ]
    assert len(set(validation)) == 42
    assert not study.intersection(validation)
    assert evidence["negative_controls"]["mixed_occupancy_law_gradient_error_k1"] > 1e-7


@pytest.mark.parametrize("field", ["gradient", "role", "source", "ready", "contract"])
def test_rehashed_tampering_cannot_be_reported(evidence, field):
    from thermo_lab.quality_budget_training_preflight import render_report

    changed = copy.deepcopy(evidence)
    if field == "gradient":
        changed["laws"][0]["diagnostics"][0]["gradient"]["component_sum"][0][0] += 1
    elif field == "role":
        changed["laws"][0]["diagnostics"][0]["gradient_seed"] += 1
    elif field == "source":
        changed["sources"][0]["initial_parameter_digest"] = "sha256:" + "0" * 64
    elif field == "contract":
        changed["validation_contract"]["batch_size"] = 16
    else:
        changed["full_m4g_ready"] = True
    changed["result_digest"] = canonical_sha256(
        {k: v for k, v in changed.items() if k != "result_digest"}
    )
    with pytest.raises(ValueError, match="replay"):
        render_report(changed)


def test_write_completion_last_and_refuse_overwrite(tmp_path):
    from thermo_lab.quality_budget_training_preflight import write_preflight

    output = tmp_path / "training-laws"
    write_preflight(output)
    completion = json.loads((output / "completion.json").read_text())
    assert completion["status"] == "training_law_component_complete"
    assert completion["full_m4g_ready"] is False
    assert completion["fits_executed"] == completion["evaluation_cells_executed"] == 0
    with pytest.raises(FileExistsError):
        write_preflight(output)


def test_failed_report_validation_never_writes_completion(tmp_path, monkeypatch):
    import thermo_lab.quality_budget_training_preflight as gate

    def reject(*args, **kwargs):
        raise ValueError("injected replay failure")

    monkeypatch.setattr(gate, "render_report", reject)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected replay"):
        gate.write_preflight(output)
    assert (output / "preflight.json").exists()
    assert not (output / "completion.json").exists()
