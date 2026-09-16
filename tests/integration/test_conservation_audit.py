"""Publication replays frozen evidence; terminal correctness cannot hide first exits."""

import copy
import json
from pathlib import Path

import pytest

from thermo_lab.hashing import canonical_sha256
from thermo_lab.records import RunRecord

ARCHIVE = Path(__file__).resolve().parents[2] / (
    "docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement"
)


@pytest.fixture(scope="module")
def audit():
    from thermo_lab.conservation_audit import build_audit

    raw = json.loads((ARCHIVE / "seed-0000000000.json").read_text())["source_record"]
    return build_audit(RunRecord.model_validate(raw))


def test_real_frozen_program_round_trip_and_exact_sampled_first_exit_agreement(audit):
    from thermo_lab.conservation_audit import validate_audit

    reloaded = validate_audit(json.loads(json.dumps(audit)))
    assert reloaded == audit
    assert len(audit["cells"]) == 2
    for cell in audit["cells"]:
        assert len(cell["exact"]) == len(cell["sampled"]) == 500
        previous_leakage = 0
        cumulative_first_exit = 0
        for exact, sampled in zip(cell["exact"], cell["sampled"], strict=True):
            cumulative_first_exit += sum(sampled["first_exit_by_parent"])
            assert cumulative_first_exit == sampled["ever_exited_count"]
            leakage = 32768 - sampled["particle_histogram"][1]
            assert leakage - previous_leakage == sampled["exit_count"] - sampled["return_count"]
            assert sampled["valid_after_exit_count"] == cumulative_first_exit - leakage
            previous_leakage = leakage
            # Fixed-seed calibration guard, not a scientific acceptance threshold.
            assert abs(1 - cumulative_first_exit / 32768 - exact["survival_probability"]) < 0.025


@pytest.mark.parametrize("mutation", ["first_exit", "histogram", "policy", "source"])
def test_repaired_digest_does_not_authorize_modified_evidence(audit, mutation):
    from thermo_lab.conservation_audit import audit_digest, validate_audit

    changed = copy.deepcopy(audit)
    if mutation == "first_exit":
        changed["cells"][0]["sampled"][0]["first_exit_by_parent"][0] += 1
    elif mutation == "histogram":
        changed["cells"][0]["sampled"][-1]["particle_histogram"][1] += 1
    elif mutation == "policy":
        changed["request"]["sample_count"] = 16384
        changed["request_hash"] = canonical_sha256(changed["request"])
    else:
        changed["source_record"]["provenance"]["platform"] += "-changed"
    changed["result_digest"] = audit_digest(changed)
    with pytest.raises(ValueError):
        validate_audit(changed)


def test_report_rejects_duplicate_seeds_and_tampering(audit):
    from thermo_lab.conservation_audit import render_report

    with pytest.raises(ValueError):
        render_report([audit, audit])
    changed = copy.deepcopy(audit)
    changed["cells"][0]["exact"][0]["survival_probability"] = 1.0
    with pytest.raises(ValueError):
        render_report([changed])
