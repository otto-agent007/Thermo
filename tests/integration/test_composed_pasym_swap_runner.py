"""Persisted domain reporting: complete evidence, honest signs, and deep validation."""

from pathlib import Path

import pytest

from thermo_lab.aggregate import AggregateRecord, CompletionState, aggregate_run_records
from thermo_lab.backends.numpy_composed_pasym_swap import NumpyComposedPAsymSwapBackend
from thermo_lab.composed_pasym_swap import CellTrajectorySource, CheckpointSource
from thermo_lab.composed_pasym_swap_reporting import (
    build_composed_metric_observations,
    validate_persisted_composed_pasym_swap_record,
)
from thermo_lab.composed_pasym_swap_results import (
    _sources_from_persisted_counts,
    build_composed_pasym_swap_summary,
    validate_composed_pasym_swap_summary,
)
from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.hashing import canonical_json
from thermo_lab.records import RunRecord
from thermo_lab.reporting import render_report
from thermo_lab.runner import _backend, run_experiment

CONFIG = experiment_config_path("numpy-composed-pasym-swap-finite-gibbs.toml")


@pytest.fixture(scope="module")
def composed_release(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("composed-release")
    aggregate = run_experiment(CONFIG, output_dir, seeds=(0, 1, 2))
    records = tuple(
        RunRecord.model_validate_json((output_dir / path).read_text(encoding="utf-8"))
        for path in aggregate.run_record_paths
    )
    return output_dir, aggregate, records


@pytest.fixture(scope="module")
def composed_records(composed_release):
    backend = NumpyComposedPAsymSwapBackend(Path(__file__).resolve().parents[2])
    return backend, composed_release[2]


def test_runner_writes_three_valid_composed_records_aggregate_and_report(composed_release):
    output_dir, aggregate, records = composed_release
    assert aggregate.completion_state is CompletionState.COMPLETE
    assert aggregate.completed_runs == 3
    assert aggregate.failed_runs == 0
    assert aggregate.failures == ()
    assert aggregate.seeds == (0, 1, 2)
    assert {record.spec.seed for record in records} == {0, 1, 2}
    assert (
        AggregateRecord.model_validate_json(
            (output_dir / "aggregate.json").read_text(encoding="utf-8")
        )
        == aggregate
    )
    summaries = [validate_persisted_composed_pasym_swap_record(record)[0] for record in records]
    assert {summary.seed for summary in summaries} == {0, 1, 2}
    for identity in ("request_hash", "bundle_digest", "target_checkpoint_digest"):
        assert len({getattr(summary, identity) for summary in summaries}) == 1
    assert len({summary.summary_digest for summary in summaries}) == 3
    assert len({summary.cells[0].cell_digest for summary in summaries}) == 3
    for summary in summaries:
        assert summary.integrity_acceptance_passed is True
        assert len(summary.cells) == 21
        assert all(len(cell.checkpoints) == 11 for cell in summary.cells)
    assert len(aggregate.metric_aggregates) == 147
    assert all(metric.count == 3 for metric in aggregate.metric_aggregates.values())
    assert (output_dir / "report.md").read_text(encoding="utf-8") == render_report(
        aggregate, records
    )


def test_composed_dispatch_is_dedicated_and_wrong_backend_is_rejected(tmp_path):
    # Routing NumPy's composed ID through its estimator adapter must fail this contract.
    config = load_experiment_config(CONFIG)
    assert isinstance(_backend(config, None), NumpyComposedPAsymSwapBackend)
    wrong = tmp_path / "wrong-backend.toml"
    wrong.write_text(
        CONFIG.read_text(encoding="utf-8").replace("numpy_exact_categorical", "thrml_local"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires backend"):
        run_experiment(wrong, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_unsupported_composed_seed_is_persisted_as_failed(tmp_path):
    aggregate = run_experiment(CONFIG, tmp_path, seeds=(3,))
    assert aggregate.completion_state is CompletionState.FAILED
    assert aggregate.completed_runs == 0
    assert aggregate.failed_runs == 1
    assert aggregate.failures[0].seed == 3
    assert aggregate.failures[0].error_type == "ValueError"
    assert "seed" in aggregate.failures[0].message.lower()
    assert aggregate.run_record_paths == ()
    assert "Unavailable because no seeded execution completed." in (
        tmp_path / "report.md"
    ).read_text(encoding="utf-8")


def test_partial_composed_run_can_be_replaced_and_complete_requires_safe_overwrite(tmp_path):
    # A real unsupported seed exercises failure aggregation without replacing the backend.
    partial = run_experiment(CONFIG, tmp_path, seeds=(0, 3))
    assert partial.completion_state is CompletionState.PARTIAL
    assert (partial.completed_runs, partial.failed_runs) == (1, 1)
    assert partial.failures[0].seed == 3
    assert partial.run_record_paths == ("runs/seed-0000000000.json",)
    records = tuple(
        RunRecord.model_validate_json((tmp_path / path).read_text(encoding="utf-8"))
        for path in partial.run_record_paths
    )
    assert validate_persisted_composed_pasym_swap_record(records[0])[0].seed == 0
    assert all(metric.count == 1 for metric in partial.metric_aggregates.values())
    complete = run_experiment(CONFIG, tmp_path, seeds=(1,))
    assert complete.completion_state is CompletionState.COMPLETE
    assert not (tmp_path / partial.run_record_paths[0]).exists()
    before = (tmp_path / "aggregate.json").read_bytes()
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_experiment(CONFIG, tmp_path, seeds=(2,))
    assert (tmp_path / "aggregate.json").read_bytes() == before
    unrelated = tmp_path / "user-notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    replaced = run_experiment(CONFIG, tmp_path, seeds=(3,), overwrite=True)
    assert replaced.completion_state is CompletionState.FAILED
    assert not (tmp_path / complete.run_record_paths[0]).exists()
    assert unrelated.read_text(encoding="utf-8") == "keep me"


def _aggregate(records):
    return aggregate_run_records(
        records,
        requested_seeds=tuple(record.spec.seed for record in records),
        run_record_paths=tuple(f"runs/seed-{record.spec.seed:010d}.json" for record in records),
        source_config=str(CONFIG),
    )


def _section(report, heading):
    """Select only this labeled Markdown section, including its subsections."""
    lines = report.splitlines()
    start = lines.index(heading) + 1
    level = len(heading) - len(heading.lstrip("#"))
    end = next(
        (
            index
            for index in range(start, len(lines))
            if lines[index].startswith("#")
            and len(lines[index]) - len(lines[index].lstrip("#")) <= level
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _reported_number(value, *, error=False):
    if value is None:
        return "unavailable (no one-particle samples)"
    if not error:
        return repr(value)
    outcome = "improved" if value < 0 else "worsened" if value > 0 else "unchanged"
    return f"{value!r} ({outcome})"


def test_report_exposes_macrostep_drift_endpoint_comparisons_and_regressions(composed_records):
    """Missing dispatch or omitted evidence must fail at the rendered report boundary."""
    _, records = composed_records
    aggregate = _aggregate(records)
    report = render_report(aggregate, records)
    for required in (
        "## Full 25-site composed finite-Gibbs comparison",
        "500 canonical occurrences",
        "32,768 complete trajectories",
        "Common random numbers",
        "### Macrostep occupancy half-L1",
        "### Macrostep particle-number leakage",
        "### Macrostep ever-left-sector probability",
        "### Local kernel residuals (not composed error)",
        "### Final 25-site occupancy",
        "### Final particle sectors and mass",
        "### Final conditional one-particle location",
        "target_context - independent",
        "model_context - target_context",
        "model_context - independent",
        "negative means improvement",
        "Scientific performance is non-gating",
        "not live THRML",
        "not 25-site parameter refinement",
        "not official Thermalizers",
        "hosted",
        "hardware",
        "Seeds are the replication units",
        "trajectories are within-batch samples",
        "exact_reference",
        "software_simulation",
        "PCG64",
        "hidden-before-outputs",
        "modulo_5",
        "https://arxiv.org/abs/2608.01615v2",
        "[aggregate.json](aggregate.json)",
        "[run-record.schema.json](schemas/run-record.schema.json)",
        "[aggregate-record.schema.json](schemas/aggregate-record.schema.json)",
    ):
        assert required in report
    for seed in (0, 1, 2):
        assert f"[Seed {seed} run](runs/seed-{seed:010d}.json)" in report
        assert f"| {seed} | yes |" in report
    # Count is the number of independent batches, never the trajectory count.
    name = "final_model_context_minus_target_context_k1_particle_number_leakage_difference"
    assert aggregate.metric_aggregates[name].count == 3
    # Exact nested numbers must reach their distinct domain tables without rounding.
    summary = records[0].metrics["composed_pasym_swap_summary"].value
    for residual in summary["local_residual_summaries"]:
        assert (
            f"| {residual['family']} | {residual['horizon']} | {residual['minimum']!r} | "
            f"{residual['median']!r} | {residual['maximum']!r} |"
        ) in _section(report, "### Local kernel residuals (not composed error)")
    # Preserve the seed/family/horizon/site mapping, not just numbers somewhere in text.
    for record in records:
        seed = record.spec.seed
        cells = record.metrics["composed_pasym_swap_summary"].value["cells"]
        for cell in cells:
            for measurement, heading in (
                ("occupancy_half_l1_error", "### Macrostep occupancy half-L1"),
                ("particle_number_leakage", "### Macrostep particle-number leakage"),
                ("ever_left_sector_probability", "### Macrostep ever-left-sector probability"),
            ):
                values = " | ".join(
                    repr(checkpoint["metrics"][measurement]) for checkpoint in cell["checkpoints"]
                )
                assert f"| {seed} | {cell['family']} | {cell['horizon']} | {values} |" in (
                    _section(report, heading)
                )
            final_metrics = cell["checkpoints"][-1]["metrics"]
            values = " | ".join(
                _reported_number(value)
                for value in (
                    *final_metrics["sector_probabilities"],
                    *(
                        final_metrics[key]
                        for key in (
                            "expected_particle_count",
                            "signed_mass_drift",
                            "particle_count_variance",
                            "maximum_site_occupancy_error",
                            "conditional_location_half_l1_error",
                        )
                    ),
                )
            )
            assert f"| {seed} | {cell['family']} | {cell['horizon']} | {values} |" in (
                _section(report, "### Final particle sectors and mass")
            )
        for horizon in ("equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"):
            final = {
                cell["family"]: cell["checkpoints"][-1]
                for cell in cells
                if cell["horizon"] == horizon
            }
            for field, heading in (
                ("occupancy", "### Final 25-site occupancy"),
                ("conditional_location", "### Final conditional one-particle location"),
            ):
                section = _section(_section(report, heading), f"#### {horizon}")
                for index in range(25):
                    target = final["independent"]["exact_target_occupancy"][index]
                    values = " | ".join(
                        _reported_number(
                            None
                            if final[family]["metrics"][field] is None
                            else final[family]["metrics"][field][index]
                        )
                        for family in ("independent", "target_context", "model_context")
                    )
                    assert (
                        f"| {seed} | ({index // 5}, {index % 5}) | {target!r} | {values} |"
                    ) in section
        comparisons = record.metrics["composed_pasym_swap_summary"].value["comparisons"]
        for horizon in ("equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"):
            for label in (
                "target_context_minus_independent",
                "model_context_minus_target_context",
                "model_context_minus_independent",
            ):
                paired = sorted(
                    (
                        item
                        for item in comparisons
                        if item["horizon"] == horizon and item["label"] == label
                    ),
                    key=lambda item: item["occurrence_count"],
                )
                prefix = f"| {seed} | {horizon} | {label.replace('_minus_', ' - ')} | "
                for measurement in paired[0]["differences"]:
                    error = measurement in {
                        "occupancy_half_l1_error",
                        "maximum_site_occupancy_error",
                        "particle_number_leakage",
                        "ever_left_sector_probability",
                        "conditional_location_half_l1_error",
                    }
                    values = " | ".join(
                        _reported_number(item["differences"][measurement], error=error)
                        for item in paired
                    )
                    section = _section(
                        _section(report, "### All paired checkpoint scalar differences"),
                        f"#### {measurement}",
                    )
                    assert prefix + values + " |" in section
                final_values = " | ".join(
                    _reported_number(paired[-1]["differences"][key], error=True)
                    for key in ("occupancy_half_l1_error", "particle_number_leakage")
                )
                assert prefix + final_values + " |" in _section(
                    report, "### Final scientific outcomes (500 occurrences)"
                )


def test_valid_worsened_endpoint_remains_integrity_accepted_and_conditional_unavailable(
    composed_records,
):
    """Positive scientific differences must not be hidden or turned into acceptance failures."""
    backend, records = composed_records
    record = records[0]
    prepared = backend.prepare(record.spec)
    summary = validate_composed_pasym_swap_summary(
        canonical_json(record.metrics["composed_pasym_swap_summary"].value),
        bundle=prepared.bundle,
        target_checkpoints=prepared.target_checkpoints,
        request_hash=backend.checked_request(record.spec)[2],
    )
    sources = _sources_from_persisted_counts(summary)
    cells = []
    for cell in sources.cells:
        if cell.family == "model_context":
            checkpoints = (cell.checkpoints[0],) + tuple(
                CheckpointSource(
                    occurrence_count=occurrence,
                    sample_count=32768,
                    occupancy_counts=(32768,) * 25,
                    sector_counts=(0, 0, 32768),
                    particle_count_histogram=(0,) * 25 + (32768,),
                    particle_count_sum=25 * 32768,
                    particle_count_sum_squares=625 * 32768,
                    ever_left_count=32768,
                    one_particle_location_counts=(0,) * 25,
                )
                for occurrence in range(50, 501, 50)
            )
            cell = CellTrajectorySource(
                family=cell.family, horizon=cell.horizon, checkpoints=checkpoints
            )
        cells.append(cell)
    synthetic = build_composed_pasym_swap_summary(
        request_hash=summary.request_hash,
        bundle=prepared.bundle,
        target_checkpoints=prepared.target_checkpoints,
        sources=sources.model_copy(update={"cells": tuple(cells), "source_digest": None}),
    )
    changed = RunRecord.model_validate_json(
        record.model_copy(
            update={"metrics": build_composed_metric_observations(synthetic)}
        ).model_dump_json()
    )
    report = render_report(_aggregate((changed,)), (changed,))
    difference = changed.metrics[
        "final_model_context_minus_target_context_k1_occupancy_half_l1_difference"
    ].value
    assert difference > 0.0
    assert f"{difference!r} (worsened)" in report
    assert "0.0 (unchanged)" in report
    assert "| 0 | yes |" in report
    assert "unavailable (no one-particle samples)" in report


def test_empty_composed_section_is_explicitly_unavailable():
    from thermo_lab import composed_pasym_swap_reporting

    section = composed_pasym_swap_reporting.render_composed_pasym_swap_section(())
    assert "Unavailable because no seeded execution completed." in section


def test_report_rejects_tampered_later_seed_before_returning_text(composed_records):
    """Checking just the first run would publish invalid later-seed endpoint evidence."""
    _, records = composed_records
    aggregate = _aggregate(records)
    last = records[-1]
    name = "final_model_context_k1_particle_number_leakage"
    metrics = {
        **last.metrics,
        name: last.metrics[name].model_copy(update={"value": 123.0}),
    }
    with pytest.raises(ValueError, match="standalone scalar"):
        render_report(aggregate, (*records[:-1], last.model_copy(update={"metrics": metrics})))


@pytest.mark.parametrize(
    "difference, expected", [(-1e-300, "improved"), (0.0, "unchanged"), (1e-300, "worsened")]
)
def test_comparison_sign_has_no_tolerance(difference, expected):
    from thermo_lab import composed_pasym_swap_reporting

    assert composed_pasym_swap_reporting.comparison_outcome(difference) == expected
