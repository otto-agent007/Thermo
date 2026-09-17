"""A pinned context reference remains immutable across compatible CPU replay."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

REFERENCE = Path(__file__).resolve().parents[2] / (
    "docs/experiment-reports/2026-09-16-context-weighted-conservation/study.json"
)


@pytest.fixture
def reference():
    return json.loads(REFERENCE.read_text())


def test_reference_authentication_rejects_even_a_rehashed_small_edit(reference):
    from thermo_lab.asymmetry_reference import reference_inputs
    from thermo_lab.context_conservation_audit import audit_digest

    reference["cells"][1]["parameters"][0][0] = float(
        np.nextafter(reference["cells"][1]["parameters"][0][0], np.inf)
    )
    reference["result_digest"] = audit_digest(reference)
    with pytest.raises(ValueError, match="pinned"):
        reference_inputs(reference)


def test_context_replay_preserves_types_support_and_bounded_weights(reference):
    from thermo_lab.asymmetry_reference import compare_contexts

    original = reference["contexts"]
    runtime = copy.deepcopy(original)
    runtime["trace_hash"] = "new derived trace hash"
    runtime["profiles"][0]["profile_hash"] = "new derived profile hash"
    runtime["profiles"][0]["context_weights"][0] = float(
        np.nextafter(runtime["profiles"][0]["context_weights"][0], 0)
    )
    compare_contexts(original, runtime)
    runtime["profiles"][0]["context_weights"][0] += 1e-8
    with pytest.raises(ValueError, match="replay"):
        compare_contexts(original, runtime)
    runtime = copy.deepcopy(original)
    runtime["profiles"][0]["support_mask"][0] = not runtime["profiles"][0]["support_mask"][0]
    with pytest.raises(ValueError, match="replay"):
        compare_contexts(original, runtime)


def test_cell_replay_does_not_tolerate_changed_penalty_labels(reference):
    from thermo_lab.asymmetry_reference import compare_cell

    cell = reference["cells"][1]
    runtime = copy.deepcopy(cell)
    runtime["penalty"] = float(np.nextafter(1.0, np.inf))
    with pytest.raises(ValueError, match="penalty"):
        compare_cell(cell, runtime)
