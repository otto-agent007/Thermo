import json

import pytest
from pydantic import ValidationError

from thermo_lab.improvement_harness.plan import Plan, load_plan, plan_digest


def valid_plan(**changes):
    data = {
        "schema_version": 1,
        "track": "dashboard",
        "objective": "labels",
        "baseline_commit": "a" * 40,
        "allowed_paths": ["dashboard/src/"],
        "primary_metric": "checks",
        "direction": "pass",
        "threshold": 1,
        "max_candidates": 1,
        "wall_seconds": 120,
    }
    data.update(changes)
    return data


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": True},
        {"allowed_paths": ["dashboard/src/", "dashboard/src/"]},
        {"allowed_paths": ["/dashboard/src/"]},
        {"allowed_paths": ["dashboard/../src/"]},
        {"baseline_commit": "main"},
        {"max_candidates": 0},
        {"wall_seconds": 0},
    ],
)
def test_plan_rejects_invalid_frozen_inputs(changes):
    with pytest.raises(ValidationError):
        Plan.model_validate(valid_plan(**changes))


def test_plan_digest_ignores_json_key_order(tmp_path):
    data = valid_plan()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(data))
    b.write_text(json.dumps(dict(reversed(list(data.items())))))
    assert plan_digest(load_plan(a)) == plan_digest(load_plan(b))
    assert plan_digest(load_plan(a)).startswith("sha256:")


def test_loaded_plan_is_frozen_and_digest_has_no_runtime_state(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(valid_plan()))
    plan = load_plan(path)
    digest = plan_digest(plan)
    with pytest.raises(ValidationError):
        plan.objective = "changed"
    assert digest == plan_digest(load_plan(path))
