"""Auditable component preflight for M4G; does not train or evaluate any model."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from thermo_lab.hashing import canonical_json, canonical_sha256, to_json_value
from thermo_lab.matched_training_archive import import_archived_training_input
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import collect_runtime_provenance, find_repository_root
from thermo_lab.quality_budget_protocol import (
    PROTOCOL_BLOB,
    QualityBudgetProtocol,
    cost_ledger,
    evaluation_manifest,
    fit_manifest,
)
from thermo_lab.quality_budget_statistics import statistical_preflight, validation_contract
from thermo_lab.records import RunRecord
from thermo_lab.runtime_work import timed_work


@timed_work("source_authentication")
def _sources(repository_root):
    root = repository_root or find_repository_root(Path(__file__).resolve())
    if root is None:
        raise ValueError("M4G preflight requires the repository's pinned source archives")
    root = Path(root)
    protocol = (root / "docs/experiments/task-quality-inference-budget.md").read_bytes()
    blob = hashlib.sha1(f"blob {len(protocol)}\0".encode() + protocol).hexdigest()
    if blob != PROTOCOL_BLOB:
        raise ValueError("M4G protocol differs from the frozen commit")
    folder = root / "docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement"
    sources = []
    for seed in (0, 1, 2):
        raw = json.loads((folder / f"seed-{seed:010d}.json").read_text())
        record = RunRecord.model_validate(raw["source_record"])
        source = import_archived_training_input(record)
        if source.seed != seed:
            raise ValueError("M4G source order differs from predeclared seed order")
        sources.append(asdict(source))
    if len({s["initial_parameter_digest"] for s in sources}) != 1:
        raise ValueError("M4G sources must share the pinned initial matrix")
    return to_json_value(sources)


def build_preflight(repository_root=None):
    request = QualityBudgetProtocol()
    contract = validation_contract()
    result = {
        "schema_version": "quality_budget_component_preflight.v1",
        "request": request.model_dump(mode="json"),
        "request_hash": request.request_hash,
        "validation_contract": contract,
        "validation_contract_hash": canonical_sha256(contract),
        "sources": _sources(repository_root),
        "fits": fit_manifest(),
        "evaluation_cells": evaluation_manifest(),
        "costs": cost_ledger(),
        "statistics": statistical_preflight(),
        "fits_executed": 0,
        "evaluation_cells_executed": 0,
        "full_m4g_ready": False,
        "remaining_gates": [
            "matched seven-law runner and law-specific training validation",
            "complete runner integrity preflight and independent review",
            "21 fits and 60 held-out cells with complete persisted replay",
            "full-study evidence review and recorded M4G decision",
        ],
    }
    return {**result, "result_digest": canonical_sha256(result)}


def validate_preflight(evidence, repository_root=None):
    # No cache of supplied evidence: rebuild from the protocol and authenticated inputs.
    expected = build_preflight(repository_root)
    if canonical_json(evidence) != canonical_json(expected):
        raise ValueError("M4G component preflight differs from complete deterministic replay")
    return expected


def render_report(evidence, repository_root=None):
    checked = validate_preflight(evidence, repository_root)
    stats = checked["statistics"]
    minimum_coverage = min(r["coverage"] for r in stats["coverage"])
    return (
        "# M4G decision-component preflight\n\n"
        "**Component preflight passed. Full M4G runner preflight remains pending.** "
        "No model fitting or held-out task evaluation was performed.\n\n"
        f"Request: `{checked['request_hash']}`\n\n"
        f"Result: `{checked['result_digest']}`\n\n"
        "- Three pinned archives authenticated; one common initialization.\n"
        "- 21 planned fits, 60 planned evaluation cells and 213 distinct role seeds.\n"
        "- 55 coverage checks; all counts enumerated at N=1,2,8,32,32768.\n"
        f"- Minimum checked marginal coverage: {minimum_coverage:.12f}.\n"
        f"- {len(stats['independent_inversions'])} independent binomial-tail inversions; "
        "interval nesting and literal threshold boundaries checked.\n"
        "- 4,096 joint binary batches; perfectly correlated interval family; "
        f"729 horizon patterns and {stats['distinct_bracket_pairs']} distinct bracket pairs.\n\n"
        "| Planned work | Endpoint draws |\n| --- | ---: |\n"
        "| Finite training grid (18 fits) | 4,423,680,000 |\n"
        "| Equilibrium training (3 fits) | 737,280,000 |\n"
        "| Total training | 5,160,960,000 |\n"
        "| All evaluation | 983,040,000 |\n"
        "| Equilibrium diagnostic evaluation subset | 98,304,000 |\n\n"
        "These are declared modeled operations, not physical measurements. Equilibrium "
        "sweep costs are null, not zero. Independent trajectory counts do not fall with K.\n\n"
        f"{stats['coverage_proof']}\n\n"
        "Next: implement the matched runner, validate both training laws at every K, "
        "and complete its integrity preflight before executing the fixed study. "
        "No task-quality result or inference saving follows from this component gate.\n"
    )


def write_preflight(output_dir, repository_root=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    evidence = build_preflight(repository_root)
    atomic_write_text(
        output / "preflight.json", json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )
    persisted = json.loads((output / "preflight.json").read_text())
    report = render_report(persisted, repository_root)
    atomic_write_text(output / "report.md", report)
    root = (
        Path(repository_root) if repository_root else find_repository_root(Path(__file__).resolve())
    )
    elapsed = perf_counter() - start
    provenance = {
        "runtime": to_json_value(collect_runtime_provenance(root)),
        "numeric_packages": {name: version(name) for name in ("numpy", "scipy")},
        "elapsed_seconds": elapsed,
        "timing_includes": "source authentication, statistics, write, full replay, report",
        "timing_excludes": "runtime provenance collection and final completion I/O",
        "timing_evidence_class": "software_simulation",
    }
    atomic_write_text(output / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    completion = {
        "status": "component_preflight_complete",
        "full_m4g_ready": False,
        "fits_executed": 0,
        "evaluation_cells_executed": 0,
        "request_hash": evidence["request_hash"],
        "result_digest": evidence["result_digest"],
    }
    atomic_write_text(output / "completion.json", json.dumps(completion, indent=2) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_preflight(args.output_dir)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
