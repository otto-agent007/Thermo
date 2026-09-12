"""Real archived-source execution, persistence, lineage and completion boundaries."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement"


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    from thermo_lab.cli import main

    root = tmp_path_factory.mktemp("matched-budget")
    source = root / "source.json"
    original = json.loads((ARCHIVE / "seed-0000000000.json").read_text())
    source.write_text(json.dumps(original["source_record"]))
    output = root / "release"
    assert main(["compare-training-laws", str(source), "--output-dir", str(output)]) == 0
    return source, output


def test_real_source_release_replays_and_labels_partial(release):
    from thermo_lab.matched_training_audit import TrainingComparisonAudit, render_comparison_report

    _, output = release
    audit = TrainingComparisonAudit.model_validate_json(
        (output / "seed-0000000000.json").read_text()
    )
    assert audit.request.seed == 0
    assert len(audit.comparison.initial_parameters) == 37
    assert len(audit.comparison.occurrence_target_indices) == 500
    assert (output / "report.md").read_text() == render_comparison_report((audit,))
    completion = json.loads((output / "completion.json").read_text())
    assert completion["full_three_seed_release"] is False
    assert completion["result_digests"] == [audit.result_digest]
    assert completion["matched_hardware_cost"] is False


@pytest.mark.parametrize("mutation", ["request", "source", "comparison"])
def test_rehashed_audit_rejects_lineage_or_nested_forgery(release, mutation):
    from thermo_lab.hashing import canonical_sha256
    from thermo_lab.matched_training_audit import TrainingComparisonAudit, audit_digest
    from thermo_lab.matched_training_budget import comparison_digest

    _, output = release
    p = json.loads((output / "seed-0000000000.json").read_text())
    if mutation == "request":
        p["request"]["source_summary_digest"] = "sha256:" + "0" * 64
    elif mutation == "source":
        other = json.loads((ARCHIVE / "seed-0000000001.json").read_text())
        p["source_record"] = other["source_record"]
    else:
        p["comparison"]["arms"].reverse()
        p["comparison"]["result_digest"] = comparison_digest(p["comparison"])
    p["request_hash"] = canonical_sha256(p["request"])
    p["result_digest"] = audit_digest(p["request_hash"], p["comparison"]["result_digest"])
    with pytest.raises(ValueError):
        TrainingComparisonAudit.model_validate(p)


def test_no_overwrite_duplicate_or_empty_publication(release, tmp_path):
    from thermo_lab.matched_training_audit import run_training_comparison

    source, output = release
    original = (output / "completion.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_training_comparison((source,), output)
    assert (output / "completion.json").read_bytes() == original
    with pytest.raises(ValueError, match="duplicate"):
        run_training_comparison((source, source), tmp_path / "duplicates")
    with pytest.raises(ValueError, match="source"):
        run_training_comparison((), tmp_path / "empty")
    assert not (tmp_path / "duplicates").exists()
    assert not (tmp_path / "empty").exists()


def test_reporting_revalidates_model_copy_forgery(release):
    from thermo_lab.matched_training_audit import TrainingComparisonAudit, render_comparison_report

    _, output = release
    audit = TrainingComparisonAudit.model_validate_json(
        (output / "seed-0000000000.json").read_text()
    )
    forged = audit.model_copy(update={"request_hash": "sha256:" + "0" * 64})
    with pytest.raises(ValueError):
        render_comparison_report((forged,))


def test_failed_publication_has_no_completion(release, tmp_path, monkeypatch):
    from thermo_lab import matched_training_audit as module

    source, _ = release
    real_write = module.atomic_write_text

    def fail_report(path, text):
        if path.name == "report.md":
            raise OSError("simulated report persistence failure")
        return real_write(path, text)

    monkeypatch.setattr(module, "atomic_write_text", fail_report)
    output = tmp_path / "failed"
    with pytest.raises(OSError, match="persistence"):
        module.run_training_comparison((source,), output)
    assert not (output / "completion.json").exists()
