"""Pinned consumed references and strict end-to-end M4F evidence replay."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256

REFERENCE = Path(__file__).resolve().parents[2] / (
    "docs/experiment-reports/2026-09-16-context-weighted-conservation/study.json"
)


@pytest.fixture(scope="module")
def audit():
    from thermo_lab.asymmetry_preservation_audit import build_audit

    return build_audit(json.loads(REFERENCE.read_text()))


def test_complete_study_replays_and_objectives_reconstruct(audit):
    from thermo_lab.asymmetry_preservation_audit import validate_audit
    from thermo_lab.asymmetry_reference import REFERENCE_DIGEST

    assert validate_audit(json.loads(json.dumps(audit))) == audit
    assert canonical_sha256(audit["reference"]) == REFERENCE_DIGEST
    assert len(audit["candidate"]["fits"]) == 37
    assert len(audit["candidate"]["measurements"]["survival"]) == 500
    assert len(audit["evaluations"]) == 4
    assert len(audit["comparisons"]) == 2
    visible = np.asarray(audit["candidate"]["measurements"]["visible_tables"])
    ref = audit["reference"]
    target = np.asarray([t["conditional"] for t in ref["control"]["request"]["targets"]])
    weights = np.asarray([p["context_weights"] for p in ref["contexts"]["profiles"]])
    particles = np.array([0, 1, 1, 2])
    failure = particles[:, None] != particles[None, :]
    error = visible[:, 1, 2] - visible[:, 2, 1] - (target[:, 1, 2] - target[:, 2, 1])
    objective = (
        np.sum(weights[:, :, None] * ((visible - target) ** 2 + visible * failure), axis=(1, 2))
        + error**2
    )
    np.testing.assert_allclose(
        [f["objective"] for f in audit["candidate"]["fits"]], objective, atol=1e-14, rtol=0
    )
    for fit in audit["candidate"]["fits"]:
        assert [a["updates"] for a in fit["attempts"]] == [100, 100]
        assert fit["objective"] <= fit["attempts"][0]["initial_objective"]
    m = audit["candidate"]["measurements"]
    weighted = ref["cells"][1]["measurements"]
    frozen = ref["control"]["cells"][1]["measurements"]
    assert audit["decision"]["passes"] == (
        m["survival"][-1]["survival_probability"]
        >= weighted["survival"][-1]["survival_probability"]
        and m["asymmetry_mae"] <= frozen["asymmetry_mae"]
        and m["hop_mae"] <= weighted["hop_mae"]
    )


@pytest.mark.parametrize("field", ["coefficient", "reference", "selection", "decision"])
def test_rehashed_evidence_cannot_change_declared_inputs_or_results(audit, field):
    from thermo_lab.asymmetry_preservation_audit import audit_digest, render_report

    changed = copy.deepcopy(audit)
    if field == "coefficient":
        changed["request"]["asymmetry_coefficient"] = 2
        changed["request_hash"] = canonical_sha256(changed["request"])
    elif field == "reference":
        changed["reference"]["cells"][1]["parameters"][0][0] += 1e-15
    elif field == "selection":
        changed["candidate"]["fits"][0]["selected_candidate"] = -1
    else:
        changed["decision"]["passes"] = not changed["decision"]["passes"]
    changed["result_digest"] = audit_digest(changed)
    with pytest.raises(ValueError):
        render_report(changed)


def test_existing_output_is_preserved(tmp_path):
    from thermo_lab.asymmetry_preservation_audit import run_study

    marker = tmp_path / "completion.json"
    marker.write_text("original evidence")
    with pytest.raises(FileExistsError):
        run_study(REFERENCE, tmp_path)
    assert marker.read_text() == "original evidence"


def test_failed_report_replay_cannot_write_completion(audit, tmp_path, monkeypatch):
    from thermo_lab import asymmetry_preservation_audit as study

    def fail(_):
        raise ValueError("injected replay failure")

    monkeypatch.setattr(study, "build_audit", lambda _: audit)
    monkeypatch.setattr(study, "render_report", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected replay"):
        study.run_study(REFERENCE, output)
    assert (output / "study.json").exists()
    assert not (output / "completion.json").exists()
