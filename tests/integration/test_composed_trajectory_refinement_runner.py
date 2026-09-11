"""Full checked refinement persistence and adversarial publication boundaries."""

import json
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
def bounded_finite_release(release):
    from thermo_lab.cli import main
    from thermo_lab.finite_refinement_audit import FiniteRefinementAudit

    source_dir, aggregate, _ = release
    source = source_dir / aggregate.run_record_paths[0]
    original = source.read_bytes()
    output = source_dir / "bounded-finite-refinement"
    assert main(["refine-finite-sweeps", str(source), "--output-dir", str(output)]) == 0
    assert source.read_bytes() == original
    audit = FiniteRefinementAudit.model_validate_json((output / "seed-0000000000.json").read_text())
    return output, source, audit


def test_bounded_finite_cli_persists_predeclared_protocol_and_five_updates(bounded_finite_release):
    output, _, audit = bounded_finite_release
    source = validate_persisted_composed_refinement_record(audit.source_record)
    assert audit.sequence.initial_parameters == source.initial_parameters
    assert len(audit.sequence.steps) == 5
    assert len(audit.sequence.initial_parameters) == 37
    assert audit.sequence.final_evaluation.horizon == "k4"
    completion = json.loads((output / "completion.json").read_text())
    assert completion["status"] == "complete"
    assert completion["full_three_seed_release"] is False
    assert completion["updates_per_run"] == 5
    assert (output / "protocol.json").is_file()
    assert (output / "sampler-validation.json").is_file()
    report = (output / "report.md").read_text()
    assert "partial release" in report
    assert "not held-out checkpoint comparisons" in report
    assert "does not establish convergence" in report


@pytest.mark.parametrize(
    "mutation", ("request", "source", "sampler", "sequence", "provenance", "digest")
)
def test_bounded_finite_audit_rejects_rehashed_tampering(bounded_finite_release, mutation):
    from thermo_lab.finite_refinement_audit import FiniteRefinementAudit, refinement_result_digest
    from thermo_lab.hashing import canonical_sha256

    _, _, audit = bounded_finite_release
    payload = audit.model_dump(mode="json")
    if mutation == "request":
        payload["request"]["source_summary_digest"] = "sha256:" + "0" * 64
        payload["request_hash"] = canonical_sha256(payload["request"])
    elif mutation == "source":
        payload["source_record"]["spec"]["seed"] = 1
    elif mutation == "sampler":
        payload["sampler_audit"]["cells"][0]["sampling_seed"] += 1
    elif mutation == "sequence":
        payload["sequence"]["evaluation_seed"] += 1
    elif mutation == "provenance":
        payload["provenance"]["packages"][0]["version"] = "0.0.0-forged"
    else:
        payload["result_digest"] = "sha256:" + "0" * 64
    if mutation != "digest":
        payload["result_digest"] = refinement_result_digest(
            payload["request_hash"],
            payload["sampler_audit"]["result_digest"],
            payload["sequence"]["result_digest"],
        )
    with pytest.raises(ValueError):
        FiniteRefinementAudit.model_validate(payload)


def test_bounded_finite_output_is_fresh_and_completion_is_last(
    bounded_finite_release, tmp_path, monkeypatch
):
    from thermo_lab import finite_refinement_audit as module

    output, source, audit = bounded_finite_release
    with pytest.raises(FileExistsError):
        module.run_finite_refinement((source,), output)
    with pytest.raises(ValueError, match="duplicate"):
        module.run_finite_refinement((source, source), tmp_path / "duplicate")
    original = module.atomic_write_text

    def fail_report(path, content):
        if path.name == "report.md":
            raise OSError("injected report-write failure")
        original(path, content)

    monkeypatch.setattr(module, "atomic_write_text", fail_report)
    broken = tmp_path / "report-failure"
    with pytest.raises(OSError, match="report-write"):
        module.run_finite_refinement((source,), broken)
    assert (broken / "seed-0000000000.json").exists()
    assert not (broken / "completion.json").exists()


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
            for count, target in zip(replay.occupancy_counts, summary.target_occupancy, strict=True)
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


@pytest.fixture(scope="module")
def finite_sweep_release(release):
    from thermo_lab.cli import main
    from thermo_lab.frozen_pair_audit import FrozenPairAudit

    source_dir, aggregate, _ = release
    source_path = source_dir / aggregate.run_record_paths[0]
    source_bytes = source_path.read_bytes()
    output = source_dir / "finite-sweep-audit"
    assert main(["audit-finite-sweeps", str(source_path), "--output-dir", str(output)]) == 0
    assert source_path.read_bytes() == source_bytes
    audit = FrozenPairAudit.model_validate_json(
        (output / "seed-0000000000.json").read_text(encoding="utf-8")
    )
    return output, source_path, audit


def test_frozen_pair_cli_round_trip_and_equilibrium_identity(finite_sweep_release):
    from thermo_lab.composed_pasym_swap_artifacts import HORIZON_LABELS
    from thermo_lab.frozen_pair_audit import render_frozen_pair_audit

    output, _, audit = finite_sweep_release
    source = validate_persisted_composed_refinement_record(audit.source_record)
    assert audit.request.source_summary_digest == source.summary_digest
    assert tuple(cell.horizon for cell in audit.cells) == HORIZON_LABELS
    assert audit.cells[0].before_counts == source.evaluation.before.occupancy_counts
    assert audit.cells[0].after_counts == source.evaluation.after.occupancy_counts
    assert (
        audit.cells[0].joined_moment_counts
        == source.evaluation.joined_terminal_second_moment_counts
    )
    assert all(cell.sample_count == 32768 for cell in audit.cells)
    completion = json.loads((output / "completion.json").read_text())
    assert completion["status"] == "complete"
    assert completion["seeds"] == [0]
    assert completion["full_three_seed_release"] is False
    assert completion["horizon_cells"] == 7
    report = render_frozen_pair_audit((audit,))
    assert (output / "report.md").read_text() == report
    assert "not simultaneous" in report
    assert "not fresh independent confirmation" in report
    assert "Historical PR #20 pair" in report
    assert "does not silently substitute a historical pair" in report
    assert "descriptive and non-gating" in report
    assert "15000" in report and "45000" in report
    assert (output / "audit.schema.json").is_file()


@pytest.mark.parametrize(
    "mutation",
    (
        "source_scalar",
        "source_seed",
        "request_seed",
        "evaluation_seed",
        "horizon_order",
        "missing_horizon",
        "table_digest",
        "counts",
        "extra_metric",
        "timing",
        "result_digest",
    ),
)
def test_frozen_pair_reload_rejects_tampered_evidence(finite_sweep_release, mutation):
    from thermo_lab.frozen_pair_audit import FrozenPairAudit

    _, _, audit = finite_sweep_release
    payload = audit.model_dump(mode="json")
    if mutation == "source_scalar":
        payload["source_record"]["metrics"]["objective_before"]["value"] += 0.01
    elif mutation == "source_seed":
        payload["source_record"]["spec"]["seed"] = 1
    elif mutation == "request_seed":
        payload["request"]["seed"] = 1
    elif mutation == "evaluation_seed":
        payload["request"]["evaluation_seed"] += 1
    elif mutation == "horizon_order":
        payload["cells"][1], payload["cells"][2] = payload["cells"][2], payload["cells"][1]
    elif mutation == "missing_horizon":
        payload["cells"].pop()
    elif mutation == "table_digest":
        payload["cells"][1]["exact_tables_digest"] = "sha256:" + "0" * 64
    elif mutation == "counts":
        payload["cells"][1]["before_counts"][0] += 1
    elif mutation == "extra_metric":
        payload["cells"][1]["population_objective_before"] = -100.0
    elif mutation == "timing":
        payload["timing"]["timing_method"] = "hardware"
    else:
        payload["result_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        FrozenPairAudit.model_validate_json(json.dumps(payload))


def test_frozen_pair_reporting_rejects_validation_bypass(finite_sweep_release):
    from thermo_lab.frozen_pair_audit import render_frozen_pair_audit

    _, _, audit = finite_sweep_release
    forged = audit.model_copy(update={"result_digest": "sha256:" + "0" * 64})
    with pytest.raises(ValueError):
        render_frozen_pair_audit((forged,))
    with pytest.raises(ValueError, match="unique"):
        render_frozen_pair_audit((audit, audit))


def test_frozen_pair_output_protects_inputs_and_previous_results(finite_sweep_release, tmp_path):
    from thermo_lab.frozen_pair_audit import run_frozen_pair_audit

    output, source_path, _ = finite_sweep_release
    original = (output / "report.md").read_bytes()
    with pytest.raises(FileExistsError):
        run_frozen_pair_audit((source_path,), output)
    assert (output / "report.md").read_bytes() == original
    with pytest.raises(ValueError, match="duplicate"):
        run_frozen_pair_audit((source_path, source_path), tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


def test_failed_frozen_pair_audit_cannot_publish_completion(
    finite_sweep_release, tmp_path, monkeypatch
):
    import thermo_lab.frozen_pair_audit as module

    _, source_path, _ = finite_sweep_release

    def fail(record):
        raise RuntimeError("injected execution failure")

    monkeypatch.setattr(module, "audit_frozen_record", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        module.run_frozen_pair_audit((source_path,), output)
    assert not (output / "completion.json").exists()
    assert not (output / "report.md").exists()
