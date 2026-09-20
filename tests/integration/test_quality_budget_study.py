"""Full result replay rejects coherent tampering and scope substitution."""

import copy

import pytest

from thermo_lab.hashing import canonical_sha256


@pytest.fixture(scope="module")
def evidence():
    from thermo_lab.quality_budget_study import build_fixture_study

    return build_fixture_study()


def test_complete_fixture_study_replays_all_cells(evidence):
    from thermo_lab.quality_budget_study import validate_study

    checked = validate_study(evidence, scope="fixture")
    assert len(checked["fits"]) == 21
    assert sum(e["executed_cells"] for e in checked["evaluations"]) == 60
    assert sum(len(e["pairs"]) for e in checked["evaluations"]) == 18
    assert checked["executions"]["production_fits"] == 0
    assert checked["executions"]["production_evaluation_cells"] == 0
    assert len(checked["decision"]["method_statuses"]["finite"]) == 6


@pytest.mark.parametrize(
    "field", ["counts", "paired", "exact", "fit", "seed", "decision", "omission", "extra"]
)
def test_rehashed_study_changes_are_rejected(evidence, field):
    from thermo_lab.quality_budget_study import study_digest, validate_study

    value = copy.deepcopy(evidence)
    first = value["evaluations"][0]
    if field == "counts":
        first["cells"][0]["occupancy_counts"][0] += 1
    elif field == "paired":
        first["pairs"][0]["evidence"]["joined_moment_counts"][0][1] += 1
    elif field == "exact":
        first["cells"][0]["exact_metrics"]["hop_mae"] = 0.0
    elif field == "fit":
        value["fits"][0]["request"]["inputs"]["initial_parameters"][0][0] += 0.1
    elif field == "seed":
        first["cells"][0]["evaluation_seed"] += 1
    elif field == "decision":
        value["decision"]["comparison"]["decision"] = "lower_budget_demonstrated"
    elif field == "omission":
        first["cells"].pop()
    else:
        value["unrequested"] = True
    value["result_digest"] = study_digest(value)
    with pytest.raises(ValueError):
        validate_study(value, scope="fixture")


def test_fixture_evidence_cannot_be_promoted_to_production(evidence):
    from thermo_lab.quality_budget_study import validate_study

    with pytest.raises(ValueError):
        validate_study(evidence, scope="production")


def test_fixture_roles_are_distinct_from_all_production_roles(evidence):
    from thermo_lab.quality_budget_protocol import fit_manifest, role_schedule

    production = {role_schedule(s)["evaluation_seed"] for s in (0, 1, 2)}
    production.update(
        r[k]
        for f in fit_manifest()
        for r in f["steps"]
        for k in ("occupancy_seed", "gradient_seed")
    )
    diagnostic = [e["evaluation_seed"] for e in evidence["evaluations"]]
    assert len(set(diagnostic)) == 3
    assert production.isdisjoint(diagnostic)
    assert evidence["request_hash"] == canonical_sha256(evidence["request"])
