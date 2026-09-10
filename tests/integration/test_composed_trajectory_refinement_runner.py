"""Full checked refinement persistence and adversarial publication boundaries."""

import math
from pathlib import Path

import pytest

from thermo_lab.aggregate import CompletionState, aggregate_run_records
from thermo_lab.composed_trajectory_refinement import sample_equilibrium_terminal_occupancy
from thermo_lab.composed_trajectory_refinement_reporting import (
    REFINEMENT_SCALARS,
    _reconstruction_backend,
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
    # Reconstruct the pre-M1 plug-in statistic through the separate single-member
    # sampler. Fresh SciPy compilation is not a portable bitwise golden fixture:
    # the archived release numbers belong to their recorded parameter lineage.
    prepared = _reconstruction_backend().prepare(records[0].spec)
    legacy_objectives = []
    for parameters, observed in (
        (summary.initial_parameters, summary.evaluation.before),
        (summary.update.updated_parameters, summary.evaluation.after),
    ):
        replay = sample_equilibrium_terminal_occupancy(
            parameters,
            prepared.bundle.occurrence_target_indices,
            prepared.bundle.occurrence_site_indices,
            site_count=25,
            batch_size=32768,
            seed=summary.evaluation_seed,
            beta=summary.beta,
        )
        assert replay.occupancy_counts == observed.occupancy_counts
        residuals = tuple(
            count / replay.sample_count - target
            for count, target in zip(
                replay.occupancy_counts, summary.target_occupancy, strict=True
            )
        )
        legacy_objectives.append(math.fsum(value * value for value in residuals))
    assert summary.evaluation.objective_before == legacy_objectives[0]
    assert summary.evaluation.objective_after == legacy_objectives[1]
    assert summary.evaluation.objective_improvement == math.fsum(
        (legacy_objectives[0], -legacy_objectives[1])
    )
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
    assert "order-two U-statistic" in report
    assert "approximate normal 95%" in report
    assert "Near-zero population loss can cause severe undercoverage" in report
    assert "8/16 (50%)" in report
    assert "within-evaluation" in report
    assert "across-seed" in report
    assert "After minus before" in report
    assert "descriptive and non-gating" in report
    assert "inconclusive" in report
    assert repr(summary.evaluation.population_objective_before) in report
    assert repr(summary.evaluation.population_objective_after) in report
    assert repr(summary.evaluation.population_objective_difference_after_minus_before) in report
    assert repr(summary.evaluation.paired_jackknife_standard_error) in report
    assert repr(summary.evaluation.paired_jackknife_normal_95_interval) in report
    assert summary.evaluation.population_objective_conclusion in report


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
        "population_scalar",
        "uncertainty_metadata",
        "population_estimate",
        "jackknife_interval",
        "conclusion",
        "joined_moments",
        "audit_policy",
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
    elif mutation == "population_scalar":
        metrics["population_objective_after"]["value"] = 123.0
    elif mutation == "uncertainty_metadata":
        metrics["population_objective_after"]["method"] = "exact objective"
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
        elif mutation == "population_estimate":
            summary["evaluation"]["population_objective_after"] = 0.5
        elif mutation == "jackknife_interval":
            summary["evaluation"]["paired_jackknife_normal_95_interval"] = [-0.5, 0.5]
        elif mutation == "conclusion":
            summary["evaluation"]["population_objective_conclusion"] = "regressed"
        elif mutation == "joined_moments":
            summary["evaluation"]["joined_terminal_second_moment_counts"][0][0] += 1
        elif mutation == "audit_policy":
            summary["evaluation"]["paired_uncertainty_policy"] = "unpaired"
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
