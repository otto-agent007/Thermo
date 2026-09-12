"""Historical machine identity may differ; package and execution identities may not."""

import json
from pathlib import Path

import pytest

from thermo_lab.composed_pasym_swap_reporting import _checked_runtime_provenance
from thermo_lab.records import RunRecord


def test_archived_platform_is_preserved_and_only_explicitly_portable():
    path = Path(__file__).resolve().parents[2] / (
        "docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement/seed-0000000000.json"
    )
    record = RunRecord.model_validate(json.loads(path.read_text())["source_record"])
    other = record.model_copy(
        update={
            "provenance": record.provenance.model_copy(
                update={"platform": "Linux-historical-other-host"}
            )
        }
    )
    with pytest.raises(ValueError, match="runtime provenance"):
        _checked_runtime_provenance(other)
    _checked_runtime_provenance(other, allow_historical_platform=True)
    assert other.provenance.platform == "Linux-historical-other-host"
    wrong = other.model_copy(
        update={"provenance": other.provenance.model_copy(update={"jax_backend": "gpu"})}
    )
    with pytest.raises(ValueError, match="runtime provenance"):
        _checked_runtime_provenance(wrong, allow_historical_platform=True)
    package = other.provenance.packages[0].model_copy(update={"version": "0.0.0"})
    wrong = other.model_copy(
        update={
            "provenance": other.provenance.model_copy(
                update={"packages": (package, *other.provenance.packages[1:])}
            )
        }
    )
    with pytest.raises(ValueError, match="runtime provenance"):
        _checked_runtime_provenance(wrong, allow_historical_platform=True)
