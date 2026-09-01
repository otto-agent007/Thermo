"""Persisted runner coverage for the checked independent PAsymSwap compiler."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from thermo_lab.aggregate import AggregateRecord, CompletionState, StatisticalSemantics
from thermo_lab.backends.base import ExecutionResult
from thermo_lab.config import load_experiment_config
from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap_reporting import render_independent_pasym_swap_section
from thermo_lab.records import RunRecord
from thermo_lab.reporting import write_report_from_persisted
from thermo_lab.runner import _backend, run_experiment

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/thrml-independent-pasym-swap.toml"


def _records(output: Path, aggregate: AggregateRecord) -> tuple[RunRecord, ...]:
    return tuple(
        RunRecord.model_validate_json((output / path).read_text(encoding="utf-8"))
        for path in aggregate.run_record_paths
    )


@pytest.fixture(scope="module")
def run_artifacts(tmp_path_factory: pytest.TempPathFactory):
    import thermo_lab.backends.thrml_independent_pasym_swap as backend_module

    instances = []
    real_init = backend_module.ThrmlIndependentPAsymSwapBackend.__init__
    patch = pytest.MonkeyPatch()

    def recording_init(self, *args, **kwargs) -> None:
        real_init(self, *args, **kwargs)
        instances.append(self)

    patch.setattr(backend_module.ThrmlIndependentPAsymSwapBackend, "__init__", recording_init)
    output = tmp_path_factory.mktemp("independent-pasym-swap")
    try:
        aggregate = run_experiment(CONFIG, output, seeds=(0, 1, 2))

        assert aggregate.completion_state is CompletionState.COMPLETE
        assert len(instances) == 1
        return SimpleNamespace(output=output, backend=instances[0])
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def run_output(run_artifacts) -> Path:
    return run_artifacts.output


def test_backend_dispatches_independent_compiler_exact_id() -> None:
    from thermo_lab.backends.thrml_independent_pasym_swap import ThrmlIndependentPAsymSwapBackend

    assert isinstance(
        _backend(load_experiment_config(CONFIG), ROOT), ThrmlIndependentPAsymSwapBackend
    )


@pytest.mark.slow
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
    assert "lower().compile() measured once" in records[0].timing.timing_method
    assert all(
        "reused from in-process shape cache" in record.timing.timing_method
        for record in records[1:]
    )

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


@pytest.mark.slow
def test_runner_persists_snapshot_schemas_and_only_scalar_seed_aggregates(run_output: Path) -> None:
    aggregate = AggregateRecord.model_validate_json(
        (run_output / "aggregate.json").read_text(encoding="utf-8")
    )

    assert load_experiment_config(run_output / "config.snapshot.toml") == load_experiment_config(
        CONFIG
    )
    assert (run_output / "schemas/run-record.schema.json").exists()
    assert (run_output / "schemas/aggregate-record.schema.json").exists()
    assert set(aggregate.metric_aggregates) == {"maximum_empirical_k30_residual"}
    assert (
        aggregate.metric_aggregates["maximum_empirical_k30_residual"].interval_method
        == "two-sided Student-t across independent seeds"
    )
    assert set(aggregate.omitted_metrics) == {
        "acceptance_passed",
        "deterministic_optimizer_seconds",
        "independent_pasym_swap",
        "maximum_k30_equilibrium_residual",
        "median_equilibrium_tv",
        "successful_artifact_count",
        "timing.compile_seconds",
        "timing.execution_seconds",
        "total_cap_active_parameter_count",
        "worst_equilibrium_tv",
    }
    assert all(
        "independently seeded sampled cross-check" in reason
        or reason == "non-scalar metric retained only in per-run records"
        for reason in aggregate.omitted_metrics.values()
    )


@pytest.mark.slow
def test_runner_preserves_completed_outputs_without_overwrite(run_output: Path) -> None:
    with pytest.raises(FileExistsError, match="completed run"):
        run_experiment(CONFIG, run_output, seeds=(0, 1, 2))


@pytest.mark.slow
def test_report_separates_paper_values_from_thermo_conventions(run_output: Path) -> None:
    """A missing dedicated section would conceal the experiment boundary."""

    report = (run_output / "report.md").read_text(encoding="utf-8")

    for required_text in (
        "Independent PAsymSwap thermodynamic kernels",
        "arXiv:2608.01615v2",
        "Thermo convention",
        "synthetic `K_(3,2)`",
        "uniform reset",
        "4,096 chains per input context",
        "context matching was not evaluated",
        "trajectory-level REINFORCE was not evaluated",
        "not a Z1 placement",
        "not official Thermalizers compatibility",
        "Seeds vary only the sampled cross-check and timing",
        "do not receive Student-t intervals",
    ):
        assert required_text in report


@pytest.mark.slow
def test_pasym_report_escapes_persisted_timing_text(run_output: Path) -> None:
    """Persisted prose cannot create Markdown table cells or headings."""

    aggregate = AggregateRecord.model_validate_json(
        (run_output / "aggregate.json").read_text(encoding="utf-8")
    )
    record = _records(run_output, aggregate)[0]
    escaped_record = record.model_copy(
        update={
            "timing": record.timing.model_copy(
                update={"timing_method": "unsafe | [link]\n# heading"}
            )
        }
    )

    section = "\n".join(render_independent_pasym_swap_section(escaped_record))

    assert r"unsafe \| \[link\] / \# heading" in section


@pytest.mark.slow
@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        (
            lambda payload: payload["metrics"]["independent_pasym_swap"]["value"]["artifacts"][0][
                "conditionals"
            ]["equilibrium_conditional"][0].__setitem__(0, 0.0),
            "equilibrium target_hash",
        ),
        (
            lambda payload: payload["metrics"]["independent_pasym_swap"]["value"]["occurrences"][
                0
            ].__setitem__("target_hash", "tampered-target-hash"),
            "every occurrence target hash",
        ),
        (
            lambda payload: payload["metrics"].__setitem__(
                "median_equilibrium_tv",
                {
                    **payload["metrics"]["median_equilibrium_tv"],
                    "value": payload["metrics"]["median_equilibrium_tv"]["value"] + 0.01,
                },
            ),
            "median_equilibrium_tv",
        ),
        (
            lambda payload: payload["metrics"].__setitem__(
                "maximum_empirical_k30_residual",
                {
                    **payload["metrics"]["maximum_empirical_k30_residual"],
                    "evidence_class": "exact_reference",
                },
            ),
            "maximum_empirical_k30_residual",
        ),
    ),
    ids=("nested_equilibrium", "occurrence_hash", "scalar_summary", "evidence_class"),
)
def test_report_rejects_tampered_persisted_pasym_swap_data_before_replacing_report(
    run_output: Path,
    tmp_path: Path,
    mutation,
    expected_error: str,
) -> None:
    """A report regeneration must validate nested evidence, not trust its aggregate."""

    output = tmp_path / "tampered-report"
    shutil.copytree(run_output, output)
    report_path = output / "report.md"
    original_report = report_path.read_bytes()
    aggregate = AggregateRecord.model_validate_json(
        (output / "aggregate.json").read_text(encoding="utf-8")
    )
    record_path = output / aggregate.run_record_paths[0]
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    mutation(payload)
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        write_report_from_persisted(output)

    assert report_path.read_bytes() == original_report


@pytest.mark.slow
def test_runner_records_one_failed_seed_and_never_claims_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_artifacts
) -> None:
    import thermo_lab.backends.thrml_independent_pasym_swap as backend_module

    successful = _records(
        run_artifacts.output,
        AggregateRecord.model_validate_json(
            (run_artifacts.output / "aggregate.json").read_text(encoding="utf-8")
        ),
    )
    records_by_seed = {record.spec.seed: record for record in successful}
    real_validate = backend_module.validate_independent_pasym_swap_observations
    validated_seeds: list[int] = []

    def corrupt_seed_one_before_validation(metrics, model, run, seed):
        validated_seeds.append(seed)
        if seed == 1:
            summary = to_json_value(metrics["independent_pasym_swap"].value)
            conditionals = summary["artifacts"][0]["conditionals"]
            finite = conditionals["finite_horizon_conditionals"]["30"][0]
            output_index = min(range(4), key=finite.__getitem__)
            counts = [0, 0, 0, 0]
            counts[output_index] = run.chain_count_per_context
            conditionals["empirical_k30_counts"][0] = counts
            conditionals["empirical_k30_conditional"][0] = [
                count / run.chain_count_per_context for count in counts
            ]
            metrics["independent_pasym_swap"] = metrics["independent_pasym_swap"].model_copy(
                update={"value": summary}
            )
        return real_validate(metrics, model, run, seed)

    class ReplaySuccessfulSeeds:
        def execute(self, spec):
            if spec.seed in (0, 2):
                return ExecutionResult.build(records_by_seed[spec.seed])
            return run_artifacts.backend.execute(spec)

    monkeypatch.setattr(
        backend_module,
        "validate_independent_pasym_swap_observations",
        corrupt_seed_one_before_validation,
    )
    monkeypatch.setattr("thermo_lab.runner._backend", lambda *_: ReplaySuccessfulSeeds())
    aggregate = run_experiment(CONFIG, tmp_path, seeds=(0, 1, 2))

    assert aggregate.completion_state is CompletionState.PARTIAL
    assert aggregate.completed_runs == 2
    assert aggregate.failed_runs == 1
    assert aggregate.run_record_paths == (
        "runs/seed-0000000000.json",
        "runs/seed-0000000002.json",
    )
    assert aggregate.failures[0].seed == 1
    assert validated_seeds == [1]
    assert "target_hash=" in aggregate.failures[0].message
    assert "observed=" in aggregate.failures[0].message
    assert "bound=0.1" in aggregate.failures[0].message
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Completion state: `partial`" in report
    assert "Passed: `yes`" not in report
