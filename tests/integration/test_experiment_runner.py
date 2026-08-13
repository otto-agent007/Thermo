import json
from pathlib import Path

import pytest

from thermo_lab.aggregate import AggregateRecord, CompletionState, StatisticalSemantics
from thermo_lab.backends import TorxStateVectorBackend
from thermo_lab.record_schemas import schema_json
from thermo_lab.records import RunRecord
from thermo_lab.runner import run_experiment

ROOT = Path(__file__).parents[2]
TORX_CONFIG = ROOT / "configs/experiments/torx-two-gate.toml"


def test_single_seed_run_writes_validated_portable_artifacts(tmp_path: Path) -> None:
    source_before = TORX_CONFIG.read_bytes()

    aggregate = run_experiment(TORX_CONFIG, tmp_path, seeds=(3,))

    assert aggregate.completion_state is CompletionState.COMPLETE
    assert aggregate.seeds == (3,)
    assert TORX_CONFIG.read_bytes() == source_before
    run_path = tmp_path / "runs/seed-0000000003.json"
    persisted_run = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
    persisted_aggregate = AggregateRecord.model_validate_json(
        (tmp_path / "aggregate.json").read_text(encoding="utf-8")
    )
    assert persisted_run.spec.seed == 3
    assert persisted_aggregate == aggregate
    assert persisted_aggregate.run_record_paths == ("runs/seed-0000000003.json",)
    assert (tmp_path / "config.snapshot.toml").exists()
    assert (tmp_path / "report.md").exists()


def test_multi_seed_run_emits_deterministic_schemas_and_report(tmp_path: Path) -> None:
    aggregate = run_experiment(TORX_CONFIG, tmp_path, seeds=(2, 0, 1))

    assert aggregate.seeds == (2, 0, 1)
    assert aggregate.completed_runs == 3
    assert aggregate.statistical_semantics is StatisticalSemantics.INDEPENDENT_SEEDED_REPLICATIONS
    run_schema = tmp_path / "schemas/run-record.schema.json"
    aggregate_schema = tmp_path / "schemas/aggregate-record.schema.json"
    assert run_schema.read_text(encoding="utf-8") == schema_json(RunRecord)
    assert aggregate_schema.read_text(encoding="utf-8") == schema_json(AggregateRecord)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "exact_reference" in report
    assert aggregate.model_hash in report
    assert aggregate.run_config_hash in report
    assert "3 independent seeded runs" in report
    assert "two-sided 95% Student-t interval across independently seeded runs" in report
    assert "not a physical Z1 or TSU hardware measurement" in report
    assert "Exact final probability mass" in report
    assert "[aggregate.json](aggregate.json)" in report
    assert "synchronized steady-state backend interval only" in report
    assert "excludes compilation, untimed warm launch, configuration loading" in report
    assert "`software_simulation`" in report
    assert "`seconds`" in report


def test_completed_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    first = run_experiment(TORX_CONFIG, tmp_path, seeds=(0,))
    aggregate_path = tmp_path / "aggregate.json"
    current_aggregate = aggregate_path.read_bytes()

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_experiment(TORX_CONFIG, tmp_path, seeds=(1,))

    assert aggregate_path.read_bytes() == current_aggregate
    second = run_experiment(TORX_CONFIG, tmp_path, seeds=(1,), overwrite=True)
    assert second.aggregate_id != first.aggregate_id
    assert not (tmp_path / "runs/seed-0000000000.json").exists()


def test_overwrite_replaces_unsupported_predecessor_aggregate_without_parsing(
    tmp_path: Path,
) -> None:
    predecessor = '{"schema_version":"1.0.0","completion_state":"complete"}\n'
    (tmp_path / "aggregate.json").write_text(predecessor, encoding="utf-8")

    aggregate = run_experiment(TORX_CONFIG, tmp_path, seeds=(1,), overwrite=True)

    assert aggregate.schema_version == "1.1.0"
    assert aggregate.seeds == (1,)
    assert (tmp_path / "aggregate.json").read_text(encoding="utf-8") != predecessor


def test_source_config_cannot_be_an_output_artifact(tmp_path: Path) -> None:
    source = tmp_path / "config.snapshot.toml"
    source.write_bytes(TORX_CONFIG.read_bytes())
    before = source.read_bytes()

    with pytest.raises(ValueError, match="source config"):
        run_experiment(source, tmp_path)

    assert source.read_bytes() == before


def test_failed_seed_persists_partial_not_complete_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_backend = TorxStateVectorBackend()

    class FailingSeedBackend:
        def execute(self, spec):
            if spec.seed == 1:
                raise RuntimeError("injected seed failure")
            return real_backend.execute(spec)

    monkeypatch.setattr(
        "thermo_lab.runner._backend", lambda config, repository_root: FailingSeedBackend()
    )

    aggregate = run_experiment(TORX_CONFIG, tmp_path, seeds=(0, 1, 2))

    assert aggregate.completion_state is CompletionState.PARTIAL
    assert aggregate.completed_runs == 2
    assert aggregate.failed_runs == 1
    assert aggregate.failures[0].seed == 1
    assert aggregate.run_record_paths == (
        "runs/seed-0000000000.json",
        "runs/seed-0000000002.json",
    )
    persisted = AggregateRecord.model_validate_json(
        (tmp_path / "aggregate.json").read_text(encoding="utf-8")
    )
    assert persisted.completion_state is CompletionState.PARTIAL


def test_schema_output_is_deterministic_json() -> None:
    first = schema_json(AggregateRecord)
    second = schema_json(AggregateRecord)

    assert first == second
    assert json.loads(first)["title"] == "AggregateRecord"
