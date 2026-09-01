"""Persisted runner coverage for the checked independent PAsymSwap compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from thermo_lab.aggregate import AggregateRecord, CompletionState, StatisticalSemantics
from thermo_lab.config import load_experiment_config
from thermo_lab.records import RunRecord
from thermo_lab.runner import run_experiment

pytestmark = pytest.mark.slow

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/thrml-independent-pasym-swap.toml"


def _records(output: Path, aggregate: AggregateRecord) -> tuple[RunRecord, ...]:
    return tuple(
        RunRecord.model_validate_json((output / path).read_text(encoding="utf-8"))
        for path in aggregate.run_record_paths
    )


@pytest.fixture(scope="module")
def run_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("independent-pasym-swap")
    aggregate = run_experiment(CONFIG, output, seeds=(0, 1, 2))

    assert aggregate.completion_state is CompletionState.COMPLETE
    return output


def test_runner_persists_three_independent_cross_checks(run_output: Path) -> None:
    aggregate = AggregateRecord.model_validate_json(
        (run_output / "aggregate.json").read_text(encoding="utf-8")
    )
    records = _records(run_output, aggregate)

    assert aggregate.seeds == (0, 1, 2)
    assert aggregate.run_record_paths == (
        "runs/seed-0000000000.json",
        "runs/seed-0000000001.json",
        "runs/seed-0000000002.json",
    )
    assert aggregate.statistical_semantics is StatisticalSemantics.INDEPENDENT_SEEDED_REPLICATIONS
    assert all(
        record.spec.seed == seed for record, seed in zip(records, aggregate.seeds, strict=True)
    )
    assert all(record.backend_id.value == "thrml_local" for record in records)
    assert records[0].metrics["deterministic_optimizer_seconds"].value > 0.0
    assert [record.metrics["deterministic_optimizer_seconds"].value for record in records[1:]] == [
        0.0,
        0.0,
    ]
    assert records[0].timing.compile_seconds > 0.0
    assert [record.timing.compile_seconds for record in records[1:]] == [0.0, 0.0]

    summaries = [record.metrics["independent_pasym_swap"].value for record in records]
    artifact_sets = [
        tuple(item["optimization"]["artifact_hash"] for item in summary["artifacts"])
        for summary in summaries
    ]
    assert len(artifact_sets[0]) == 37
    assert artifact_sets[0] == artifact_sets[1] == artifact_sets[2]
    assert any(
        left["conditionals"]["empirical_k30_counts"]
        != right["conditionals"]["empirical_k30_counts"]
        for left, right in zip(summaries[0]["artifacts"], summaries[1]["artifacts"], strict=True)
    )


def test_runner_persists_snapshot_schemas_and_only_scalar_seed_aggregates(run_output: Path) -> None:
    aggregate = AggregateRecord.model_validate_json(
        (run_output / "aggregate.json").read_text(encoding="utf-8")
    )

    assert load_experiment_config(run_output / "config.snapshot.toml") == load_experiment_config(
        CONFIG
    )
    assert (run_output / "schemas/run-record.schema.json").exists()
    assert (run_output / "schemas/aggregate-record.schema.json").exists()
    assert "independent_pasym_swap" not in aggregate.metric_aggregates
    assert aggregate.omitted_metrics["independent_pasym_swap"] == (
        "non-scalar metric retained only in per-run records"
    )
    assert all(metric.count == 3 for metric in aggregate.metric_aggregates.values())
    assert all(
        metric.interval_method.startswith("two-sided Student-t across independent seeds")
        for metric in aggregate.metric_aggregates.values()
    )
    forbidden_identity_terms = ("kernel", "context", "horizon", "probability", "occurrence")
    assert not any(
        any(term in name for term in forbidden_identity_terms)
        for name in aggregate.metric_aggregates
    )


def test_runner_preserves_completed_outputs_without_overwrite(run_output: Path) -> None:
    with pytest.raises(FileExistsError, match="completed run"):
        run_experiment(CONFIG, run_output, seeds=(0, 1, 2))


def test_runner_records_one_failed_seed_and_never_claims_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import thermo_lab.backends.thrml_independent_pasym_swap as backend_module

    real_execute = backend_module.ThrmlIndependentPAsymSwapBackend.execute

    def fail_only_seed_one(self, spec):
        result = real_execute(self, spec)
        if spec.seed == 1:
            raise ValueError("sampled THRML cross-check residual=0.100001 bound=0.1")
        return result

    monkeypatch.setattr(
        backend_module.ThrmlIndependentPAsymSwapBackend,
        "execute",
        fail_only_seed_one,
    )
    aggregate = run_experiment(CONFIG, tmp_path, seeds=(0, 1, 2))

    assert aggregate.completion_state is CompletionState.PARTIAL
    assert aggregate.completed_runs == 2
    assert aggregate.failed_runs == 1
    assert aggregate.run_record_paths == (
        "runs/seed-0000000000.json",
        "runs/seed-0000000002.json",
    )
    assert aggregate.failures[0].seed == 1
    assert "residual=0.100001" in aggregate.failures[0].message
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Completion state: `partial`" in report
    assert "Passed: `yes`" not in report
