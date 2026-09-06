"""Runner and persisted-report coverage for model-context PAsymSwap."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from thermo_lab.aggregate import AggregateRecord, CompletionState
from thermo_lab.backends.thrml_model_context_pasym_swap import (
    ThrmlModelContextPAsymSwapBackend,
)
from thermo_lab.config import load_experiment_config
from thermo_lab.records import RunRecord
from thermo_lab.reporting import write_report_from_persisted
from thermo_lab.runner import _backend, run_experiment

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/thrml-model-context-pasym-swap.toml"


@dataclass(frozen=True)
class CompletedModelContextRun:
    output: Path
    record: RunRecord
    aggregate: AggregateRecord


@pytest.fixture(scope="module")
def completed_model_context_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompletedModelContextRun:
    output = tmp_path_factory.mktemp("model-context-report")
    aggregate = run_experiment(CONFIG, output)
    record = RunRecord.model_validate_json(
        (output / aggregate.run_record_paths[0]).read_text(encoding="utf-8")
    )
    assert aggregate.completion_state is CompletionState.COMPLETE
    return CompletedModelContextRun(output=output, record=record, aggregate=aggregate)


def test_backend_dispatches_model_context_exact_id() -> None:
    configured = load_experiment_config(CONFIG)

    assert isinstance(_backend(configured, ROOT), ThrmlModelContextPAsymSwapBackend)


def test_model_context_aggregate_exposes_only_seeded_empirical_residual(
    completed_model_context_run: CompletedModelContextRun,
) -> None:
    aggregate = completed_model_context_run.aggregate

    assert set(aggregate.metric_aggregates) == {"maximum_empirical_k30_residual"}
    assert aggregate.omitted_metrics["model_context_pasym_swap_summary"] == (
        "nested model-context evidence is retained only in per-run records"
    )
    assert aggregate.omitted_metrics["timing.compile_seconds"] == (
        "per-seed synchronized JAX timing is not an independently seeded sampled cross-check"
    )
    assert aggregate.omitted_metrics["timing.execution_seconds"] == (
        "per-seed synchronized JAX timing is not an independently seeded sampled cross-check"
    )


def test_model_context_runner_writes_reload_validated_evidence_report(
    completed_model_context_run: CompletedModelContextRun,
) -> None:
    report = (completed_model_context_run.output / "report.md").read_text(encoding="utf-8")

    required = (
        "## Model-context PAsymSwap study",
        "37 profiles / 500 occurrences",
        "26 × 10, 9 × 20, and 2 × 30",
        "Model-profile KL improvement",
        "Exact model K=30 residual",
        "Empirical model K=30 residual",
        "148 keyed 4096-chain",
        "Profiles, input contexts, and chains are not independent replications",
        "not a physical Z1 or TSU hardware measurement",
    )
    for phrase in required:
        assert phrase in report


def _model_context_summary(payload: dict) -> dict:
    return payload["metrics"]["model_context_pasym_swap_summary"]["value"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: (
            _model_context_summary(payload)["profile_samples"][0]["sampled_k30"]["counts"][
                0
            ].__setitem__(
                0,
                _model_context_summary(payload)["profile_samples"][0]["sampled_k30"]["counts"][0][0]
                + 1,
            ),
            _model_context_summary(payload)["profile_samples"][0]["sampled_k30"]["counts"][
                0
            ].__setitem__(
                1,
                _model_context_summary(payload)["profile_samples"][0]["sampled_k30"]["counts"][0][1]
                - 1,
            ),
        ),
        lambda payload: _model_context_summary(payload)["profile_samples"][0].__setitem__(
            "model_context_artifact_hash", "sha256:" + "0" * 64
        ),
        lambda payload: _model_context_summary(payload).__setitem__("acceptance_passed", False),
        lambda payload: payload["metrics"]["maximum_empirical_k30_residual"].__setitem__(
            "value",
            payload["metrics"]["maximum_empirical_k30_residual"]["value"] + 0.001,
        ),
    ),
    ids=("count", "artifact", "acceptance", "scalar"),
)
def test_model_context_report_rejects_tampering_before_replacing_existing_report(
    completed_model_context_run: CompletedModelContextRun,
    tmp_path: Path,
    mutation,
) -> None:
    output = tmp_path / "tampered"
    shutil.copytree(completed_model_context_run.output, output)
    report_path = output / "report.md"
    expected_report = report_path.read_text(encoding="utf-8")
    record_path = output / completed_model_context_run.aggregate.run_record_paths[0]
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    mutation(payload)
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        write_report_from_persisted(output)

    assert report_path.read_text(encoding="utf-8") == expected_report
