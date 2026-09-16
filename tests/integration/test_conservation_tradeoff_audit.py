"""Full-budget replay, immutable source lineage, and publication boundary tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256
from thermo_lab.records import RunRecord

ARCHIVE = Path(__file__).resolve().parents[2] / (
    "docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement"
)


@pytest.fixture(scope="module")
def records():
    return [
        RunRecord.model_validate(
            json.loads((ARCHIVE / f"seed-{seed:010d}.json").read_text())["source_record"]
        )
        for seed in range(3)
    ]


@pytest.fixture(scope="module")
def audit(records):
    from thermo_lab.conservation_tradeoff_audit import build_audit

    return build_audit(records)


def test_full_grid_round_trip_and_predeclared_selection(audit):
    from thermo_lab.conservation_tradeoff_audit import validate_audit

    assert validate_audit(json.loads(json.dumps(audit))) == audit
    assert len(audit["cells"]) == 5
    logical = audit["cells"][0]["measurements"]
    assert logical["mean_conservation_failure"] == 0
    assert logical["survival"][-1]["survival_probability"] == pytest.approx(1, abs=1e-13)
    for cell in audit["cells"][2:]:
        assert len(cell["fits"]) == 37
        for fit in cell["fits"]:
            assert fit["objective"] <= fit["attempts"][0]["initial_objective"]
            assert np.max(np.abs(fit["parameters"])) <= 2
            assert [attempt["updates"] for attempt in fit["attempts"]] == [100, 100]
            objectives = [
                attempt[field]
                for attempt in fit["attempts"]
                for field in ("initial_objective", "final_objective")
            ]
            assert fit["selected_candidate"] == int(np.argmin(objectives))
        assert len(cell["measurements"]["survival"]) == 500


def test_repaired_digest_cannot_authorize_changed_optimizer_evidence(audit):
    from thermo_lab.conservation_tradeoff_audit import audit_digest, validate_audit

    changed = copy.deepcopy(audit)
    changed["cells"][2]["fits"][0]["attempts"][0]["updates"] = 99
    changed["result_digest"] = audit_digest(changed)
    with pytest.raises(ValueError, match="numerical replay"):
        validate_audit(changed)


@pytest.mark.parametrize("mutation", ["policy", "source", "measurement", "missing_seed"])
def test_input_and_evidence_corruption_rejected(audit, mutation):
    from thermo_lab.conservation_tradeoff_audit import audit_digest, render_report

    changed = copy.deepcopy(audit)
    if mutation == "policy":
        changed["request"]["updates_per_start"] = 99
        changed["request_hash"] = canonical_sha256(changed["request"])
        changed["result_digest"] = audit_digest(changed)
    elif mutation == "source":
        changed["source_records"][0]["provenance"]["platform"] += "-changed"
    elif mutation == "missing_seed":
        changed["source_records"].pop()
    else:
        changed["cells"][2]["measurements"]["mean_conservation_failure"] = 0
    with pytest.raises(ValueError):
        render_report(changed)


def test_preexisting_destination_is_untouched(tmp_path):
    from thermo_lab.conservation_tradeoff_audit import run_study

    marker = tmp_path / "completion.json"
    marker.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        run_study(ARCHIVE, tmp_path)
    assert marker.read_text() == "existing evidence"


def test_failed_report_replay_cannot_write_completion(audit, tmp_path, monkeypatch):
    from thermo_lab import conservation_tradeoff_audit as study

    def fail(_audit):
        raise ValueError("injected replay failure")

    monkeypatch.setattr(study, "build_audit", lambda records: audit)
    monkeypatch.setattr(study, "render_report", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected replay failure"):
        study.run_study(ARCHIVE, output)
    assert (output / "study.json").is_file()
    assert not (output / "completion.json").exists()
