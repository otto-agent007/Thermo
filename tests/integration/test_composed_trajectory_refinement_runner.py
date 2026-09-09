"""Full checked refinement persistence and adversarial publication boundaries."""

from pathlib import Path

import pytest

from thermo_lab.aggregate import CompletionState, aggregate_run_records
from thermo_lab.composed_trajectory_refinement_reporting import (
    REFINEMENT_SCALARS,
    validate_persisted_composed_refinement_record,
)
from thermo_lab.composed_trajectory_refinement_results import (
    composed_trajectory_refinement_summary_digest,
)
from thermo_lab.config import experiment_config_path
from thermo_lab.records import RunRecord
from thermo_lab.reporting import render_report
from thermo_lab.runner import run_experiment

CONFIG = experiment_config_path("numpy-composed-pasym-swap-trajectory-refinement-one-step.toml")


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    output = tmp_path_factory.mktemp("composed-refinement")
    aggregate = run_experiment(CONFIG, output, seeds=(0,))
    assert aggregate.completion_state is CompletionState.COMPLETE, aggregate.failures
    records = tuple(
        RunRecord.model_validate_json((output / path).read_text())
        for path in aggregate.run_record_paths
    )
    return output, aggregate, records


def _aggregate(records):
    return aggregate_run_records(
        records,
        requested_seeds=(0,),
        run_record_paths=("runs/seed-0000000000.json",),
        source_config=str(CONFIG),
    )


def test_checked_refinement_round_trip_and_report(release):
    output, aggregate, records = release
    summary = validate_persisted_composed_refinement_record(records[0])
    assert summary.seed == 0
    assert len(summary.initial_parameters) == 37
    assert len(summary.target_occupancy) == 25
    assert summary.update.bounds_satisfied
    assert summary.occupancy_source.sample_count == 32768
    assert summary.gradient_source.sample_count == 32768
    assert summary.evaluation.before.sample_count == 32768
    assert summary.evaluation.after.sample_count == 32768
    assert set(aggregate.metric_aggregates) == REFINEMENT_SCALARS
    assert all(
        metric.confidence_interval is None for metric in aggregate.metric_aggregates.values()
    )
    report = render_report(aggregate, records)
    assert (output / "report.md").read_text() == report
    assert "One-step 25-site trajectory refinement" in report
    assert repr(summary.evaluation.objective_before) in report
    assert repr(summary.evaluation.objective_after) in report
    assert "never an integrity gate" in report
    assert "finite-batch bias" in report


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_summary",
        "scalar",
        "boolean_scalar",
        "extra_metric",
        "timing",
        "evidence",
        "request_hash",
        "seed",
        "role_seed",
        "batch",
        "rate",
        "source_digest",
        "evaluation_digest",
    ],
)
def test_aggregation_and_report_reject_forged_refinement_records(release, mutation):
    _, aggregate, records = release
    payload = records[0].model_dump(mode="json")
    metrics = payload["metrics"]
    summary = metrics["composed_trajectory_refinement_summary"]["value"]
    if mutation == "missing_summary":
        del metrics["composed_trajectory_refinement_summary"]
    elif mutation == "scalar":
        metrics["objective_after"]["value"] = 123.0
    elif mutation == "boolean_scalar":
        metrics["cap_active_parameter_count"]["value"] = False
    elif mutation == "extra_metric":
        metrics["unapproved_scalar"] = metrics["objective_after"]
    elif mutation == "timing":
        payload["timing"]["timing_method"] += "; changed boundary"
    elif mutation == "evidence":
        metrics["objective_before"]["evidence_class"] = "exact_reference"
    else:
        if mutation == "request_hash":
            summary["request_hash"] = "sha256:" + "0" * 64
        elif mutation == "seed":
            summary["seed"] = 1
        elif mutation == "role_seed":
            summary["occupancy_seed"] = summary["gradient_seed"]
        elif mutation == "batch":
            summary["evaluation"]["after"]["sample_count"] = 16384
        elif mutation == "rate":
            summary["learning_rate"] = 0.1
        elif mutation == "source_digest":
            summary["gradient_source"]["source_digest"] = "sha256:" + "0" * 64
        elif mutation == "evaluation_digest":
            summary["evaluation"]["result_digest"] = "sha256:" + "0" * 64
        summary["summary_digest"] = composed_trajectory_refinement_summary_digest(summary)
    forged = RunRecord.model_validate(payload)
    with pytest.raises(ValueError):
        _aggregate((forged,))
    with pytest.raises(ValueError):
        render_report(aggregate, (forged,))


def test_unsupported_seed_is_an_auditable_failure(tmp_path: Path):
    aggregate = run_experiment(CONFIG, tmp_path, seeds=(3,))
    assert aggregate.completion_state is CompletionState.FAILED
    assert aggregate.failed_runs == 1
    assert aggregate.failures[0].seed == 3
    assert aggregate.run_record_paths == ()
    assert (
        "Unavailable because no seeded execution completed." in (tmp_path / "report.md").read_text()
    )
