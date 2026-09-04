import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import thermo_lab.runner as runner_module
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
    predecessor_run = '{"schema_version":"1.0.0","timing":{}}\n'
    (tmp_path / "aggregate.json").write_text(predecessor, encoding="utf-8")
    old_run = tmp_path / "runs/seed-0000000000.json"
    old_run.parent.mkdir()
    old_run.write_text(predecessor_run, encoding="utf-8")

    aggregate = run_experiment(TORX_CONFIG, tmp_path, seeds=(1,), overwrite=True)

    assert aggregate.schema_version == "1.1.0"
    assert aggregate.seeds == (1,)
    assert (tmp_path / "aggregate.json").read_text(encoding="utf-8") != predecessor
    assert not old_run.exists()
    replacement_run = json.loads(
        (tmp_path / "runs/seed-0000000001.json").read_text(encoding="utf-8")
    )
    assert replacement_run["schema_version"] == "1.1.0"
    assert replacement_run["timing"]["evidence_class"] == "software_simulation"


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


def test_report_validation_failure_leaves_seed_records_but_no_derived_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner_module, "render_report", Mock(side_effect=ValueError("bad report")), raising=False
    )

    with pytest.raises(ValueError, match="bad report"):
        run_experiment(TORX_CONFIG, tmp_path, seeds=(0,))

    assert tuple((tmp_path / "runs").glob("seed-*.json"))
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "aggregate.json").exists()


def test_aggregate_validation_failure_leaves_seed_records_but_no_derived_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner_module,
        "aggregate_run_records",
        Mock(side_effect=ValueError("bad aggregate")),
    )

    with pytest.raises(ValueError, match="bad aggregate"):
        run_experiment(TORX_CONFIG, tmp_path, seeds=(0,))

    assert tuple((tmp_path / "runs").glob("seed-*.json"))
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "aggregate.json").exists()


def test_report_is_atomically_published_before_aggregate_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_atomic_write = runner_module.atomic_write_text
    real_aggregate_write = AggregateRecord.write_json

    def tracked_atomic_write(path, content):
        if path.name == "report.md":
            events.append("report")
        return real_atomic_write(path, content)

    def tracked_aggregate_write(self, path):
        events.append("aggregate")
        return real_aggregate_write(self, path)

    monkeypatch.setattr(runner_module, "atomic_write_text", tracked_atomic_write)
    monkeypatch.setattr(AggregateRecord, "write_json", tracked_aggregate_write)

    run_experiment(TORX_CONFIG, tmp_path, seeds=(0,))

    assert events == ["report", "aggregate"]


def test_aggregate_and_report_are_derived_from_reloaded_seed_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend_records: list[RunRecord] = []
    aggregated_records: list[RunRecord] = []
    reported_records: list[RunRecord] = []
    real_backend = TorxStateVectorBackend()
    real_aggregate = runner_module.aggregate_run_records
    real_render = runner_module.render_report
    real_run_write = RunRecord.write_json
    persisted_python = "persisted-python-version"

    class TrackingBackend:
        def execute(self, spec):
            result = real_backend.execute(spec)
            backend_records.append(result.record)
            return result

    def tracked_aggregate(records, **kwargs):
        aggregated_records.extend(records)
        return real_aggregate(records, **kwargs)

    def tracked_render(aggregate, records):
        reported_records.extend(records)
        return real_render(aggregate, records)

    def mutate_persisted_record(self, path):
        real_run_write(self, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["provenance"]["python_version"] = persisted_python
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        runner_module, "_backend", lambda config, repository_root: TrackingBackend()
    )
    monkeypatch.setattr(runner_module, "aggregate_run_records", tracked_aggregate)
    monkeypatch.setattr(runner_module, "render_report", tracked_render)
    monkeypatch.setattr(RunRecord, "write_json", mutate_persisted_record)

    aggregate = run_experiment(TORX_CONFIG, tmp_path, seeds=(0, 1))

    assert [record.spec.seed for record in aggregated_records] == [0, 1]
    assert reported_records == aggregated_records
    assert aggregate.provenance_summary is not None
    assert aggregate.provenance_summary.python_version == persisted_python
    assert all(record.provenance.python_version == persisted_python for record in reported_records)
    assert all(record.provenance.python_version != persisted_python for record in backend_records)
    assert all(
        persisted is not returned
        for persisted, returned in zip(aggregated_records, backend_records, strict=True)
    )


def test_schema_output_is_deterministic_json() -> None:
    first = schema_json(AggregateRecord)
    second = schema_json(AggregateRecord)

    assert first == second
    assert json.loads(first)["title"] == "AggregateRecord"


def test_unreadable_predecessor_aggregate_is_refused_without_overwrite(tmp_path: Path) -> None:
    predecessor = '{"schema_version":"1.0.0","completion_state":"complete"}\n'
    (tmp_path / "aggregate.json").write_text(predecessor, encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_experiment(TORX_CONFIG, tmp_path, seeds=(1,))

    assert (tmp_path / "aggregate.json").read_text(encoding="utf-8") == predecessor
    assert not (tmp_path / "runs").exists()


def test_multiline_failure_message_stays_on_one_report_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingBackend:
        def execute(self, spec):
            raise RuntimeError("first line\n- second line | with table syntax")

    monkeypatch.setattr("thermo_lab.runner._backend", lambda config, root: FailingBackend())

    aggregate = run_experiment(TORX_CONFIG, tmp_path, seeds=(0,))

    assert aggregate.completion_state is CompletionState.FAILED
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "- Seed 0: `RuntimeError` — first line / \\- second line \\| with table syntax" in report
    assert "\n- second line" not in report


def test_git_provenance_is_independent_of_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    from_root = run_experiment(TORX_CONFIG, tmp_path / "root", seeds=(0,))
    monkeypatch.chdir(ROOT / "configs")
    from_subdirectory = run_experiment(TORX_CONFIG, tmp_path / "subdirectory", seeds=(0,))

    assert from_root.provenance_summary is not None
    assert from_subdirectory.provenance_summary is not None
    assert (
        from_subdirectory.provenance_summary.git_commit == from_root.provenance_summary.git_commit
    )
    assert from_subdirectory.provenance_summary.git_dirty == from_root.provenance_summary.git_dirty
