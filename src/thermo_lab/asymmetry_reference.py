"""Authenticate and replay only the historical cells consumed by M4F."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thermo_lab.conservation_diagnostic import endpoint_tables
from thermo_lab.conservation_tradeoff import measure_tables
from thermo_lab.conservation_tradeoff_audit import _request as uniform_request
from thermo_lab.context_conservation import derive_contexts, fit_weighted_group
from thermo_lab.hashing import canonical_sha256
from thermo_lab.pinned_control_replay import CONTROL_DIGEST, _compare_values
from thermo_lab.records import RunRecord

REFERENCE_PATH = Path("docs/experiment-reports/2026-09-16-context-weighted-conservation/study.json")
REFERENCE_DIGEST = "sha256:d01e83fe9a9fb790910ddf964448aa19a98bb506cd547d3204d3b7e11139d816"


def reference_inputs(reference):
    """Pin all archive bytes and independently authenticate sources, targets and schedule."""
    if canonical_sha256(reference) != REFERENCE_DIGEST:
        raise ValueError("context reference differs from the pinned complete artifact")
    control = reference["control"]
    if canonical_sha256(control) != CONTROL_DIGEST:
        raise ValueError("embedded uniform control differs from its pin")
    request = uniform_request([RunRecord.model_validate(r) for r in control["source_records"]])
    if canonical_sha256(request) != canonical_sha256(control["request"]):
        raise ValueError("reference request does not match authenticated inputs")
    return request["sources"][0], np.asarray([t["conditional"] for t in request["targets"]])


def compare_contexts(archived, runtime):
    """Derived hashes can differ; numerical weights and exact profile policies must agree."""
    if archived.keys() != runtime.keys() or len(archived["profiles"]) != len(runtime["profiles"]):
        raise ValueError("context replay structure mismatch")
    for index, (old, new) in enumerate(zip(archived["profiles"], runtime["profiles"], strict=True)):
        if old.keys() != new.keys():
            raise ValueError("context replay profile structure mismatch")
        _compare_values(
            {k: v for k, v in old.items() if k not in ("trace_hash", "profile_hash")},
            {k: v for k, v in new.items() if k not in ("trace_hash", "profile_hash")},
            f"contexts/profiles/{index}",
        )


def compare_cell(archived, runtime):
    if archived.keys() != runtime.keys():
        raise ValueError("reference replay cell structure mismatch")
    if canonical_sha256(archived["penalty"]) != canonical_sha256(runtime["penalty"]):
        raise ValueError("reference replay penalty mismatch")
    _compare_values(
        {k: v for k, v in archived.items() if k != "tables_digest"},
        {k: v for k, v in runtime.items() if k != "tables_digest"},
        f"reference/{archived['name']}",
    )


def make_cell(name, tables, targets, source, *, parameters=None, penalty=None, fits=None):
    return {
        "name": name,
        "penalty": penalty,
        "parameters": parameters,
        "fits": fits,
        "tables_digest": canonical_sha256(tables.tolist()),
        "measurements": measure_tables(
            tables,
            targets,
            source["occurrence_target_indices"],
            source["occurrence_site_indices"],
            site_count=25,
        ),
    }


def replay_reference(reference):
    """Replay three consumed cells and 7,400 weighted-control updates, never unused grids."""
    source, targets = reference_inputs(reference)
    compare_contexts(reference["contexts"], derive_contexts())
    weights = [p["context_weights"] for p in reference["contexts"]["profiles"]]
    initial = source["initial_parameters"]
    logical = np.concatenate((targets, np.zeros_like(targets)), axis=2)
    compare_cell(
        reference["control"]["cells"][0], make_cell("logical_reference", logical, targets, source)
    )
    compare_cell(
        reference["control"]["cells"][1],
        make_cell(
            "frozen_initial",
            endpoint_tables(initial, 4),
            targets,
            source,
            parameters=initial,
        ),
    )
    fits = [
        fit_weighted_group(row, target, 1.0, weight)
        for row, target, weight in zip(initial, targets, weights, strict=True)
    ]
    parameters = [fit["parameters"] for fit in fits]
    compare_cell(
        reference["cells"][1],
        make_cell(
            "target_context_1",
            endpoint_tables(parameters, 4),
            targets,
            source,
            parameters=parameters,
            penalty=1.0,
            fits=fits,
        ),
    )
    return source, targets
