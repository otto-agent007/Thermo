"""Reviewed, completion-last execution of the frozen M4G experiment."""

from __future__ import annotations

import argparse
import csv
import io
import json
from importlib.metadata import version
from pathlib import Path

from thermo_lab.hashing import canonical_json, canonical_sha256, to_json_value
from thermo_lab.matched_training_budget import clear_generation_caches
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import collect_runtime_provenance, find_repository_root
from thermo_lab.quality_budget_full_preflight import validate_preflight, validate_review
from thermo_lab.quality_budget_release_contract import (
    validate_evidence_review,
    validate_gate_record,
    validate_provenance,
)
from thermo_lab.quality_budget_study import build_production_study, render_report
from thermo_lab.quality_budget_training_runner import _sample_cached
from thermo_lab.runtime_work import collect_work, timed_work


@timed_work("prerequisite_validation")
def load_prerequisites(preflight_dir, review_path, repository_root=None):
    folder = Path(preflight_dir)
    evidence = _read(folder / "preflight.json")
    checked = validate_preflight(evidence, repository_root)
    completion = _read(folder / "completion.json")
    expected = {
        "status": "integrated_preflight_complete",
        "result_digest": checked["result_digest"],
        "implementation_digest": checked["implementation_digest"],
        "production_fits": 0,
        "production_cells": 0,
        "full_m4g_ready": False,
    }
    if canonical_json(completion) != canonical_json(expected):
        raise ValueError("integrated preflight completion differs")
    review = validate_review(_read(Path(review_path)), checked, repository_root)
    return checked, review


@timed_work("evidence_io")
def _write(path, value):
    atomic_write_text(path, canonical_json(value) + "\n")


@timed_work("evidence_io")
def _read(path):
    return json.loads(Path(path).read_text())


@timed_work("evidence_io")
def _read_text(path):
    return Path(path).read_text()


@timed_work("evidence_io")
def _write_text(path, value):
    atomic_write_text(path, value)


@timed_work("csv_reporting")
def metrics_csv(evidence):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "seed",
            "member",
            "horizon",
            "loss_estimate",
            "loss_lower",
            "loss_upper",
            "leakage_count",
            "leakage_lower",
            "leakage_upper",
            "survival",
            "hop_mae",
            "asymmetry_mae",
            "status",
        ]
    )
    for evaluation in evidence["evaluations"]:
        for cell in evaluation["cells"]:
            exact, decision = cell["exact_metrics"], cell["decision"]
            writer.writerow(
                [
                    cell["seed"],
                    cell["member"],
                    cell["horizon"],
                    cell["population_loss_estimate"],
                    *decision["loss_bounds"],
                    cell["leakage_count"],
                    *decision["leakage_interval"],
                    exact["survival"][-1]["survival_probability"],
                    exact["hop_mae"],
                    exact["asymmetry_mae"],
                    decision["status"],
                ]
            )
    return stream.getvalue()


def execution_record(evidence, preflight, review, provenance):
    return {
        "status": "study_execution_complete",
        "full_m4g_complete": False,
        "remaining_gates": ["independent_production_evidence_review", "repository_gates"],
        "provenance_digest": canonical_sha256(provenance),
        "protocol_commit": preflight["protocol_commit"],
        "implementation_digest": preflight["implementation_digest"],
        "preflight_result_digest": preflight["result_digest"],
        "review_digest": canonical_sha256(review),
        "request_hash": evidence["request_hash"],
        "result_digest": evidence["result_digest"],
        "fits": 21,
        "updates": 105,
        "evaluation_cells": 60,
        "paired_diagnostics": 18,
        "selected_checkpoint": 5,
        "scientific_decision": evidence["decision"]["comparison"]["decision"],
        "hardware_measurement": False,
    }


def _require_equal(actual, expected, label):
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f"{label} differs from replay")


@timed_work("production_validation")
def validate_execution(output_dir, repository_root=None):
    output = Path(output_dir)
    preflight = validate_preflight(_read(output / "preflight.json"), repository_root)
    review = validate_review(_read(output / "review.json"), preflight, repository_root)
    provenance = validate_provenance(
        _read(output / "provenance.json"), preflight["implementation_digest"]
    )
    evidence = _read(output / "study.json")
    # Reporting fully validates all training and held-out numerical evidence.
    report = render_report(evidence, scope="production", repository_root=repository_root)
    _require_equal(
        _read_text(output / "metrics.csv"),
        metrics_csv(evidence).replace("\r\n", "\n"),
        "release metrics",
    )
    _require_equal(_read_text(output / "report.md"), report, "release report")
    execution = execution_record(evidence, preflight, review, provenance)
    _require_equal(_read(output / "execution.json"), execution, "execution record")
    return evidence, execution


def completion_record(execution, evidence_review, gates):
    return {
        "status": "full_m4g_study_complete",
        "full_m4g_complete": True,
        "implementation_digest": execution["implementation_digest"],
        "result_digest": execution["result_digest"],
        "execution_digest": canonical_sha256(execution),
        "evidence_review_digest": canonical_sha256(evidence_review),
        "repository_gates_digest": canonical_sha256(gates),
        "scientific_decision": execution["scientific_decision"],
        "hardware_measurement": False,
    }


def _release_reviews(output, execution):
    review = validate_evidence_review(
        _read(output / "evidence-review.json"),
        execution["implementation_digest"],
        execution["result_digest"],
    )
    gates = validate_gate_record(_read(output / "gates.json"), execution["implementation_digest"])
    return review, gates


def finalize_release(output_dir, evidence_review_path, gate_record_path, repository_root=None):
    output = Path(output_dir)
    if (output / "completion.json").exists():
        raise FileExistsError(output / "completion.json")
    evidence, execution = validate_execution(output, repository_root)
    review = validate_evidence_review(
        _read(evidence_review_path), execution["implementation_digest"], execution["result_digest"]
    )
    gates = validate_gate_record(_read(gate_record_path), execution["implementation_digest"])
    _write(output / "evidence-review.json", review)
    _write(output / "gates.json", gates)
    persisted_review, persisted_gates = _release_reviews(output, execution)
    _require_equal(persisted_review, review, "persisted evidence review")
    _require_equal(persisted_gates, gates, "persisted repository gates")
    _write(output / "completion.json", completion_record(execution, review, gates))
    return evidence


def validate_release(output_dir, repository_root=None):
    output = Path(output_dir)
    evidence, execution = validate_execution(output, repository_root)
    review, gates = _release_reviews(output, execution)
    _require_equal(
        _read(output / "completion.json"),
        completion_record(execution, review, gates),
        "release completion",
    )
    return evidence


def run_study(preflight_dir, review_path, output_dir, repository_root=None):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    with collect_work() as prerequisites_work:
        preflight, review = load_prerequisites(preflight_dir, review_path, repository_root)
    # No output directory, fitting or evaluation is created before both prerequisites pass.
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    _write(output / "preflight.json", preflight)
    _write(output / "review.json", review)

    def on_fit(index, fit):
        _write(checkpoints / f"fit-{index:02d}.json", fit)
        print(
            f"Frozen fit {index + 1}/21: seed {fit['request']['seed']}, "
            f"law {fit['request']['horizon']}",
            flush=True,
        )

    clear_generation_caches()
    _sample_cached.cache_clear()
    with collect_work() as generation_work:
        evidence = build_production_study(repository_root, on_fit=on_fit)
        _write(output / "study.json", evidence)
    print("All 21 fits and 60 cells generated; replaying persisted evidence", flush=True)
    with collect_work() as replay_work:
        persisted = _read(output / "study.json")
        report = render_report(persisted, scope="production", repository_root=repository_root)
        _write_text(output / "report.md", report)
        _write_text(output / "metrics.csv", metrics_csv(persisted))
        _require_equal(_read_text(output / "report.md"), report, "persisted report")
        _require_equal(
            _read_text(output / "metrics.csv"),
            metrics_csv(persisted).replace("\r\n", "\n"),
            "persisted metrics",
        )
        _require_equal(_read(output / "preflight.json"), preflight, "persisted preflight")
        _require_equal(_read(output / "review.json"), review, "persisted review")
    root = (
        Path(repository_root) if repository_root else find_repository_root(Path(__file__).resolve())
    )
    provenance = {
        "schema_version": "quality_budget_runtime.v1",
        "implementation_digest": preflight["implementation_digest"],
        "runtime": to_json_value(collect_runtime_provenance(root)),
        "numeric_packages": {n: version(n) for n in ("numpy", "scipy", "jax", "jaxlib")},
        "evidence_class": "software_simulation",
        "timing_policy": "nested_cpu_wall_time.v1",
        "timing_excludes": "provenance collection and final metadata I/O",
        "prerequisite_work": prerequisites_work,
        "generation_work": generation_work,
        "persisted_replay_and_reporting_work": replay_work,
        "infrastructure_retries": [],
    }
    validate_provenance(provenance, preflight["implementation_digest"])
    _write(output / "provenance.json", provenance)
    _require_equal(_read(output / "provenance.json"), provenance, "persisted provenance")
    _write(output / "execution.json", execution_record(evidence, preflight, review, provenance))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--review-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_study(args.preflight_dir, args.review_record, args.output_dir)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
