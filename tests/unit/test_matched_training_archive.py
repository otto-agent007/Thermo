"""Frozen-source authentication is independent of unused historical numerics."""

import copy
import json
from pathlib import Path

import pytest

from thermo_lab.hashing import canonical_sha256
from thermo_lab.matched_training_archive import ARCHIVE_DIGESTS, import_archived_training_input
from thermo_lab.records import RunRecord

ARCHIVE = Path(__file__).resolve().parents[2] / (
    "docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement"
)


def payload(seed):
    return json.loads((ARCHIVE / f"seed-{seed:010d}.json").read_text())["source_record"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fixed_sources_authenticate_and_rebuild_inputs_without_legacy_replay(seed, monkeypatch):
    from thermo_lab import composed_trajectory_refinement_reporting as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("archive import must not recompile or replay unused M1 statistics")

    monkeypatch.setattr(legacy, "_reconstruction_backend", forbidden)
    monkeypatch.setattr(legacy, "validate_persisted_composed_refinement_record", forbidden)
    raw = payload(seed)
    before = copy.deepcopy(raw)
    source = import_archived_training_input(RunRecord.model_validate(raw))
    assert raw == before
    assert source.archive_digest == ARCHIVE_DIGESTS[seed]
    assert len(source.initial_parameters) == 37
    assert len(source.occurrence_target_indices) == len(source.occurrence_site_indices) == 500
    assert len(source.target_occupancy) == 25


@pytest.mark.parametrize("mutation", ["statistic", "parameters", "provenance", "spec"])
def test_complete_archive_pin_rejects_changed_fields_even_with_repaired_summary(mutation):
    raw = payload(0)
    summary = raw["metrics"]["composed_trajectory_refinement_summary"]["value"]
    if mutation == "statistic":
        summary["evaluation"]["paired_jackknife_standard_error"] += 1e-15
    elif mutation == "parameters":
        summary["initial_parameters"][0][0] += 1e-8
        summary["initial_parameter_digest"] = canonical_sha256(summary["initial_parameters"])
    elif mutation == "provenance":
        raw["provenance"]["platform"] += "-changed"
    else:
        raw["spec"]["seed"] = 1
    # Even replacing the internal summary checksum cannot change the trusted outer pin.
    summary["summary_digest"] = canonical_sha256(
        {k: v for k, v in summary.items() if k != "summary_digest"}
    )
    with pytest.raises(ValueError):
        import_archived_training_input(RunRecord.model_validate(raw))
