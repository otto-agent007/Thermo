"""Persistence must reconstruct finite-sweep gradient evidence before publication."""

import json

import pytest

from thermo_lab.hashing import canonical_sha256


@pytest.fixture(scope="module")
def audit():
    from thermo_lab.finite_sweep_gradient_audit import build_gradient_audit

    return build_gradient_audit()


def test_audit_round_trip_binds_all_six_horizons_and_keeps_exact_evidence(audit):
    from thermo_lab.finite_sweep_gradient_audit import (
        FiniteSweepGradientAudit,
        render_gradient_audit,
    )

    reloaded = FiniteSweepGradientAudit.model_validate_json(audit.model_dump_json())
    assert reloaded == audit
    assert tuple(cell.horizon for cell in audit.cells) == (1, 2, 4, 8, 16, 30)
    assert all(cell.accepted for cell in audit.cells)
    assert audit.evidence_class == "exact_reference"
    report = render_gradient_audit(audit)
    assert "finite-sweep" in report
    assert "No parameter update" in report


@pytest.mark.parametrize(
    "mutation", ("request", "gradient", "objective", "order", "digest", "boolean")
)
def test_rehashed_corrupt_evidence_is_rejected(audit, mutation):
    from thermo_lab.finite_sweep_gradient_audit import (
        FiniteSweepGradientAudit,
        gradient_result_digest,
    )

    payload = json.loads(audit.model_dump_json())
    if mutation == "request":
        payload["request"]["reset_policy"] = "carry hidden state"
        payload["request_hash"] = canonical_sha256(payload["request"])
    elif mutation == "gradient":
        payload["cells"][0]["score_occurrences"][0][0] += 0.01
    elif mutation == "objective":
        payload["cells"][0]["objective"] += 0.01
    elif mutation == "order":
        payload["cells"].reverse()
    elif mutation == "boolean":
        payload["cells"][0]["score_occurrences"][0][0] = True
    if mutation == "digest":
        payload["result_digest"] = "sha256:" + "0" * 64
    else:
        payload["result_digest"] = gradient_result_digest(payload["request_hash"], payload["cells"])
    with pytest.raises(ValueError):
        FiniteSweepGradientAudit.model_validate_json(json.dumps(payload))


def test_report_revalidates_model_copy_bypasses(audit):
    from thermo_lab.finite_sweep_gradient_audit import render_gradient_audit

    changed = audit.cells[0].model_copy(update={"objective": -1.0})
    corrupt = audit.model_copy(update={"cells": (changed, *audit.cells[1:])})
    with pytest.raises(ValueError):
        render_gradient_audit(corrupt)


def test_cli_publishes_reloaded_evidence_and_protects_existing_output(tmp_path):
    from thermo_lab.cli import main
    from thermo_lab.finite_sweep_gradient_audit import FiniteSweepGradientAudit

    output = tmp_path / "contract"
    assert main(["check-finite-sweep-gradients", "--output-dir", str(output)]) == 0
    payload = (output / "audit.json").read_text()
    audit = FiniteSweepGradientAudit.model_validate_json(payload)
    completion = json.loads((output / "completion.json").read_text())
    assert completion["status"] == "complete"
    assert completion["result_digest"] == audit.result_digest
    assert completion["horizons"] == [1, 2, 4, 8, 16, 30]
    assert (output / "audit.schema.json").is_file()
    with pytest.raises(FileExistsError):
        main(["check-finite-sweep-gradients", "--output-dir", str(output)])
    assert (output / "audit.json").read_text() == payload


def test_report_write_failure_cannot_publish_completion(tmp_path, monkeypatch, audit):
    import thermo_lab.finite_sweep_gradient_audit as module

    original_write = module.atomic_write_text

    def fail_report(path, content):
        if path.name == "report.md":
            raise OSError("report destination unavailable")
        return original_write(path, content)

    monkeypatch.setattr(module, "atomic_write_text", fail_report)
    output = tmp_path / "partial"
    with pytest.raises(OSError, match="unavailable"):
        module.run_gradient_audit(output)
    assert (output / "audit.json").is_file()
    assert not (output / "completion.json").exists()
