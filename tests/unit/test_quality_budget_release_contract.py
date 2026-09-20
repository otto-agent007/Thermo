"""Execution metadata cannot masquerade as the reviewed final scientific release."""

import copy
from pathlib import Path

import pytest

from thermo_lab.quality_budget_gate_plan import gate_arguments


def command_for(name):
    root = Path(__file__).resolve().parents[2]
    return [
        "uv",
        *[
            "results/output"
            if x == "<output>"
            else f"results/source/seed-{int(x[-2]):010d}.json"
            if x.startswith("<source")
            else x
            for x in gate_arguments(name, root)
        ],
    ]


def test_execution_record_binds_provenance_and_leaves_release_pending():
    from thermo_lab.hashing import canonical_sha256
    from thermo_lab.quality_budget_release import execution_record

    study = {
        "request_hash": "request",
        "result_digest": "result",
        "decision": {"comparison": {"decision": "both_quality_failure"}},
    }
    preflight = {
        "protocol_commit": "commit",
        "implementation_digest": "code",
        "result_digest": "preflight",
    }
    result = execution_record(study, preflight, {"review": "approved"}, {"runtime": "observed"})
    assert result["status"] == "study_execution_complete"
    assert result["full_m4g_complete"] is False
    assert result["provenance_digest"] == canonical_sha256({"runtime": "observed"})
    assert result["remaining_gates"] == [
        "independent_production_evidence_review",
        "repository_gates",
    ]


def test_gate_record_requires_every_current_repository_gate():
    from thermo_lab.quality_budget_release_contract import REQUIRED_GATES, validate_gate_record

    record = {
        "schema_version": "quality_budget_repository_gates.v1",
        "implementation_digest": "code",
        "commands": [
            {"name": n, "command": command_for(n), "returncode": 0, "seconds": 0.1}
            for n in REQUIRED_GATES
        ],
    }
    assert validate_gate_record(record, "code") == record
    for mutate in (
        lambda x: x["commands"].pop(),
        lambda x: x["commands"][0].update(returncode=1),
        lambda x: x["commands"][0].update(command=["true"]),
        lambda x: x["commands"][0].update(returncode=False),
        lambda x: x.update(implementation_digest="stale"),
    ):
        bad = copy.deepcopy(record)
        mutate(bad)
        with pytest.raises(ValueError):
            validate_gate_record(bad, "code")


def test_provenance_requires_complete_finite_observed_timing_rows():
    from thermo_lab.hashing import to_json_value
    from thermo_lab.provenance import collect_runtime_provenance
    from thermo_lab.quality_budget_release_contract import validate_provenance

    def row(n):
        return {"calls": n, "inclusive_seconds": 1.0, "exclusive_seconds": 0.5}

    provenance = {
        "schema_version": "quality_budget_runtime.v1",
        "implementation_digest": "code",
        "runtime": to_json_value(collect_runtime_provenance()),
        "numeric_packages": {k: "1" for k in ("numpy", "scipy", "jax", "jaxlib")},
        "evidence_class": "software_simulation",
        "timing_policy": "nested_cpu_wall_time.v1",
        "timing_excludes": "provenance collection and final metadata I/O",
        "prerequisite_work": {},
        "generation_work": {
            "training_update_chain": row(21),
            "training_sampling": row(105),
            "evaluation_sampling": row(60),
            "paired_statistics": row(18),
        },
        "persisted_replay_and_reporting_work": {
            "evaluation_sampling": row(60),
            "paired_statistics": row(18),
        },
        "infrastructure_retries": [],
    }
    assert validate_provenance(provenance, "code") == provenance
    for mutate in (
        lambda x: x["generation_work"]["evaluation_sampling"].update(calls=59),
        lambda x: x["generation_work"]["training_sampling"].update(exclusive_seconds=-1.0),
        lambda x: x["generation_work"]["training_sampling"].update(inclusive_seconds=float("nan")),
        lambda x: x.update(evidence_class="physical_hardware"),
        lambda x: x.pop("runtime"),
    ):
        bad = copy.deepcopy(provenance)
        mutate(bad)
        with pytest.raises(ValueError):
            validate_provenance(bad, "code")


def test_review_requires_distinct_trimmed_identities_and_current_results():
    from thermo_lab.quality_budget_release_contract import validate_evidence_review

    review = {
        "schema_version": "quality_budget_evidence_review.v1",
        "implementation_digest": "code",
        "study_result_digest": "study",
        "reviews": [
            {"role": role, "reviewer": role, "verdict": "approved", "notes": "Reviewed"}
            for role in ("statistical", "implementation")
        ],
    }
    assert validate_evidence_review(review, "code", "study") == review
    for mutate in (
        lambda x: x["reviews"][1].update(reviewer="statistical"),
        lambda x: x["reviews"][1].update(reviewer=" statistical "),
        lambda x: x["reviews"][1].update(verdict="pending"),
        lambda x: x.update(study_result_digest="stale"),
    ):
        bad = copy.deepcopy(review)
        mutate(bad)
        with pytest.raises(ValueError):
            validate_evidence_review(bad, "code", "study")


def test_release_finalization_requires_persisted_reviews_and_successful_gates(
    tmp_path, monkeypatch
):
    # Numerical replay has separate real-fixture integration coverage. This test
    # isolates the release state machine after a successful numerical replay.
    from thermo_lab import quality_budget_release as release
    from thermo_lab.quality_budget_release_contract import REQUIRED_GATES

    execution = {
        "implementation_digest": "code",
        "result_digest": "study",
        "scientific_decision": "both_quality_failure",
    }
    monkeypatch.setattr(
        release, "validate_execution", lambda *args: ({"result_digest": "study"}, execution)
    )
    review = {
        "schema_version": "quality_budget_evidence_review.v1",
        "implementation_digest": "code",
        "study_result_digest": "study",
        "reviews": [
            {"role": role, "reviewer": role, "verdict": "approved", "notes": "Reviewed"}
            for role in ("statistical", "implementation")
        ],
    }
    gates = {
        "schema_version": "quality_budget_repository_gates.v1",
        "implementation_digest": "code",
        "commands": [
            {"name": name, "command": command_for(name), "returncode": 0, "seconds": 0.1}
            for name in REQUIRED_GATES
        ],
    }
    review_path, gates_path = tmp_path / "input-review.json", tmp_path / "input-gates.json"
    with pytest.raises(FileNotFoundError):
        release.finalize_release(tmp_path, review_path, gates_path)
    assert not (tmp_path / "completion.json").exists()
    release._write(review_path, review)
    bad = copy.deepcopy(gates)
    bad["commands"][0]["returncode"] = 1
    release._write(gates_path, bad)
    with pytest.raises(ValueError):
        release.finalize_release(tmp_path, review_path, gates_path)
    assert not (tmp_path / "completion.json").exists()
    release._write(gates_path, gates)
    release.finalize_release(tmp_path, review_path, gates_path)
    assert release.validate_release(tmp_path)["result_digest"] == "study"
    completion = release._read(tmp_path / "completion.json")
    completion["full_m4g_complete"] = 1
    release._write(tmp_path / "completion.json", completion)
    with pytest.raises(ValueError):
        release.validate_release(tmp_path)
    with pytest.raises(FileExistsError):
        release.finalize_release(tmp_path, review_path, gates_path)


def test_identity_comparison_rejects_numeric_type_substitution():
    from thermo_lab.quality_budget_release import _require_equal

    for actual, expected in ((21.0, 21), (0, False)):
        with pytest.raises(ValueError):
            _require_equal(actual, expected, "record")
