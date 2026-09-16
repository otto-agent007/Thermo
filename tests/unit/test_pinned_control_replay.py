"""Pinned historical evidence permits bounded runtime roundoff, never input edits."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256

CONTROL = Path(__file__).resolve().parents[2] / (
    "docs/experiment-reports/2026-09-16-local-conservation-tradeoff/study.json"
)


@pytest.fixture
def control():
    return json.loads(CONTROL.read_text())


def test_runtime_roundoff_preserves_the_exact_archived_control(control):
    from thermo_lab.pinned_control_replay import check_control_replay

    runtime = copy.deepcopy(control)
    fit = runtime["cells"][2]["fits"][0]
    fit["parameters"][0] = float(np.nextafter(fit["parameters"][0], np.inf))
    # Digests of newly computed floating arrays need not equal historical digests.
    runtime["cells"][2]["tables_digest"] = canonical_sha256("runtime table")
    runtime["result_digest"] = canonical_sha256("runtime result")
    original = copy.deepcopy(control)
    check_control_replay(control, runtime)
    assert control == original


@pytest.mark.parametrize("delta", [1e-10, float("nan"), float("inf")])
def test_runtime_numeric_disagreement_is_rejected(control, delta):
    from thermo_lab.pinned_control_replay import check_control_replay

    runtime = copy.deepcopy(control)
    runtime["cells"][2]["fits"][0]["objective"] += delta
    with pytest.raises(ValueError, match="control replay.*cells"):
        check_control_replay(control, runtime)


@pytest.mark.parametrize(
    "mutation", ["selection", "updates", "type", "bool", "penalty", "missing", "extra", "request"]
)
def test_runtime_structure_and_discrete_policy_are_exact(control, mutation):
    from thermo_lab.pinned_control_replay import check_control_replay

    runtime = copy.deepcopy(control)
    fit = runtime["cells"][2]["fits"][0]
    if mutation == "selection":
        fit["selected_candidate"] = (fit["selected_candidate"] + 1) % 4
    elif mutation == "updates":
        fit["attempts"][0]["updates"] = 99
    elif mutation == "type":
        fit["attempts"][0]["updates"] = 100.0
    elif mutation == "bool":
        fit["selected_candidate"] = bool(fit["selected_candidate"])
    elif mutation == "penalty":
        runtime["cells"][2]["penalty"] = float(np.nextafter(0.0, np.inf))
    elif mutation == "missing":
        runtime["cells"][2]["measurements"]["survival"].pop()
    elif mutation == "extra":
        fit["ignored"] = 0.0
    else:
        runtime["request"]["cap"] = float(np.nextafter(2.0, np.inf))
    with pytest.raises(ValueError, match="control replay"):
        check_control_replay(control, runtime)


@pytest.mark.parametrize("above", [False, True])
def test_absolute_tolerance_boundary_has_no_relative_allowance(control, above):
    from thermo_lab.pinned_control_replay import REPLAY_ATOL, check_control_replay

    runtime = copy.deepcopy(control)
    # Logical-reference error is exactly zero, so the boundary is unambiguous.
    runtime["cells"][0]["measurements"]["asymmetry_mae"] = (
        float(np.nextafter(REPLAY_ATOL, np.inf)) if above else REPLAY_ATOL
    )
    if above:
        with pytest.raises(ValueError, match="control replay"):
            check_control_replay(control, runtime)
    else:
        check_control_replay(control, runtime)


def test_new_replay_policy_is_bound_without_rewriting_legacy_identity(control):
    from thermo_lab.context_conservation_audit import _request

    historical = json.loads(
        (
            CONTROL.parent.parent / "2026-09-16-context-weighted-conservation/protocol.json"
        ).read_text()
    )
    assert _request(control, legacy=True) == historical
    current = _request(control)
    assert current["identity_version"] == "context_weighted_conservation.v2"
    assert current["control_replay"]["numeric_atol"] == 1e-12
    assert current["control_replay"]["numeric_rtol"] == 0
    assert canonical_sha256(current) != canonical_sha256(historical)


def test_even_roundoff_sized_archive_mutation_fails_the_pin(control):
    from thermo_lab.conservation_tradeoff_audit import audit_digest
    from thermo_lab.pinned_control_replay import check_control_replay

    runtime = copy.deepcopy(control)
    fit = control["cells"][2]["fits"][0]
    fit["objective"] = float(np.nextafter(fit["objective"], np.inf))
    control["result_digest"] = audit_digest(control)
    with pytest.raises(ValueError, match="pinned complete artifact"):
        check_control_replay(control, runtime)
