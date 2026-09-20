"""The component gate must replay evidence and never masquerade as a trained study."""

import copy
import json

import pytest

from thermo_lab.hashing import canonical_sha256
from thermo_lab.quality_budget_preflight import (
    build_preflight,
    render_report,
    validate_preflight,
    write_preflight,
)


@pytest.fixture(scope="module")
def preflight():
    return build_preflight()


def test_preflight_authenticates_sources_and_completes_declared_statistical_grid(preflight):
    assert len(preflight["sources"]) == 3
    assert len({s["initial_parameter_digest"] for s in preflight["sources"]}) == 1
    coverage = preflight["statistics"]["coverage"]
    assert len(coverage) == 55
    assert {r["n"] for r in coverage} == {1, 2, 8, 32, 32768}
    assert min(r["coverage"] for r in coverage) >= 1 - 0.05 / 1560 - 1e-12
    assert preflight["statistics"]["joint_fixture"]["ordered_batches"] == 4096
    assert preflight["statistics"]["budget_patterns"] == 729
    assert preflight["full_m4g_ready"] is False
    assert preflight["fits_executed"] == preflight["evaluation_cells_executed"] == 0


@pytest.mark.parametrize("field", ["threshold", "role", "cost", "coverage", "source", "ready"])
def test_rehashed_tampering_is_rejected(preflight, field):
    changed = copy.deepcopy(preflight)
    if field == "threshold":
        changed["request"]["survival_min"] = 0.9
        changed["request_hash"] = canonical_sha256(changed["request"])
    elif field == "role":
        changed["fits"][0]["steps"][0]["occupancy_seed"] += 1
    elif field == "cost":
        changed["costs"]["training"]["endpoint_draws"] -= 1
    elif field == "coverage":
        changed["statistics"]["coverage"][0]["coverage"] = 0.0
    elif field == "source":
        changed["sources"][0]["initial_parameter_digest"] = "sha256:" + "0" * 64
    else:
        changed["full_m4g_ready"] = True
    changed["result_digest"] = canonical_sha256(
        {k: v for k, v in changed.items() if k != "result_digest"}
    )
    with pytest.raises(ValueError):
        validate_preflight(changed)
    with pytest.raises(ValueError):
        render_report(changed)


def test_persistence_replays_before_completion_and_refuses_occupied_destination(tmp_path):
    output = tmp_path / "preflight"
    write_preflight(output)
    completion = json.loads((output / "completion.json").read_text())
    assert completion["status"] == "component_preflight_complete"
    assert completion["full_m4g_ready"] is False
    evidence = json.loads((output / "preflight.json").read_text())
    assert completion["result_digest"] == evidence["result_digest"]
    assert (output / "report.md").read_text() == render_report(evidence)
    with pytest.raises(FileExistsError):
        write_preflight(output)
