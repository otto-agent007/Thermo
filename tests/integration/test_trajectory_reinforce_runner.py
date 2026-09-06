"""Runner and aggregate integration for trajectory REINFORCE evidence."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

import thermo_lab.runner as runner_module
from thermo_lab.aggregate import AggregateRecord, CompletionState, aggregate_run_records
from thermo_lab.backends.numpy_exact_categorical import NumpyExactCategoricalBackend
from thermo_lab.cli import main
from thermo_lab.config import load_experiment_config
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.records import ExperimentSpec, RunRecord
from thermo_lab.reporting import write_report_from_persisted
from thermo_lab.runner import _backend, _failed_identity
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference
from thermo_lab.trajectory_reinforce_reporting import (
    validate_persisted_trajectory_reinforce_record,
)
from thermo_lab.trajectory_reinforce_results import (
    GradientMoments,
    build_trajectory_reinforce_deterministic_result,
    build_trajectory_reinforce_sample_result,
    build_trajectory_reinforce_summary,
    validate_trajectory_reinforce_summary,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml"


@dataclass(frozen=True)
class CompletedTrajectoryRun:
    output: Path
    aggregate: AggregateRecord
    records: tuple[RunRecord, RunRecord, RunRecord]


@pytest.fixture(scope="module")
def completed_trajectory_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompletedTrajectoryRun:
    output = tmp_path_factory.mktemp("trajectory-reinforce")
    exit_code = main(
        [
            "run",
            str(CONFIG),
            "--seeds",
            "0,1,2",
            "--output-dir",
            str(output),
        ]
    )
    aggregate = AggregateRecord.model_validate_json(
        (output / "aggregate.json").read_text(encoding="utf-8")
    )
    records = tuple(
        RunRecord.model_validate_json((output / path).read_text(encoding="utf-8"))
        for path in aggregate.run_record_paths
    )
    assert exit_code == 0
    assert len(records) == 3
    return CompletedTrajectoryRun(
        output=output,
        aggregate=aggregate,
        records=(records[0], records[1], records[2]),
    )


def _aggregate(records: tuple[RunRecord, ...]) -> AggregateRecord:
    seeds = tuple(record.spec.seed for record in records)
    return aggregate_run_records(
        records,
        requested_seeds=seeds,
        run_record_paths=tuple(f"runs/seed-{seed:010d}.json" for seed in seeds),
        source_config="configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml",
    )


def _rebuilt_summary_record(
    record: RunRecord,
    *,
    request_hash: str | None = None,
    sample_definition: str | None = None,
) -> RunRecord:
    original = validate_trajectory_reinforce_summary(
        record.metrics["trajectory_reinforce_summary"].value
    )
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    deterministic = build_trajectory_reinforce_deterministic_result(
        request_hash=request_hash or original.deterministic.request_hash,
        fixture=fixture,
        exact=exact,
    )
    sampled = build_trajectory_reinforce_sample_result(
        deterministic_result_digest=deterministic.deterministic_result_digest,
        seed=original.sample.seed,
        sample_definition=sample_definition or original.sample.sample_definition,
        occurrence_0=original.sample.occurrence_0.moments,
        occurrence_1=original.sample.occurrence_1.moments,
        cross_products=original.sample.cross_products,
        exact=deterministic.expected_reference,
    )
    rebuilt = build_trajectory_reinforce_summary(
        deterministic=deterministic,
        sample=sampled,
    )
    return record.model_copy(
        update={
            "metrics": {
                **dict(record.metrics),
                "trajectory_reinforce_summary": record.metrics[
                    "trajectory_reinforce_summary"
                ].model_copy(update={"value": rebuilt}),
            }
        }
    )


def _rebuilt_wrong_count_record(record: RunRecord) -> RunRecord:
    original = validate_trajectory_reinforce_summary(
        record.metrics["trajectory_reinforce_summary"].value
    )
    wrong_count = original.sample.occurrence_0.moments.sample_count - 1

    def changed(moments: GradientMoments) -> GradientMoments:
        return moments.model_copy(update={"sample_count": wrong_count})

    sampled = build_trajectory_reinforce_sample_result(
        deterministic_result_digest=original.deterministic.deterministic_result_digest,
        seed=original.sample.seed,
        sample_definition=original.sample.sample_definition,
        occurrence_0=changed(original.sample.occurrence_0.moments),
        occurrence_1=changed(original.sample.occurrence_1.moments),
        cross_products=original.sample.cross_products,
        exact=original.deterministic.expected_reference,
    )
    rebuilt = build_trajectory_reinforce_summary(
        deterministic=original.deterministic,
        sample=sampled,
    )
    return record.model_copy(
        update={
            "metrics": {
                **dict(record.metrics),
                "trajectory_reinforce_summary": record.metrics[
                    "trajectory_reinforce_summary"
                ].model_copy(update={"value": rebuilt}),
                "maximum_absolute_shared_gradient_error": record.metrics[
                    "maximum_absolute_shared_gradient_error"
                ].model_copy(update={"value": sampled.maximum_absolute_shared_gradient_error}),
            }
        }
    )


def test_runner_dispatches_exact_trajectory_experiment_before_generic_backends() -> None:
    config = load_experiment_config(CONFIG)

    assert isinstance(_backend(config, ROOT), NumpyExactCategoricalBackend)


def test_cli_persists_three_seeded_records_and_complete_aggregate(
    completed_trajectory_run: CompletedTrajectoryRun,
) -> None:
    aggregate = completed_trajectory_run.aggregate

    assert aggregate.completion_state is CompletionState.COMPLETE
    assert aggregate.seeds == (0, 1, 2)
    assert aggregate.completed_runs == 3
    assert aggregate.failed_runs == 0
    assert (completed_trajectory_run.output / "config.snapshot.toml").is_file()
    assert (completed_trajectory_run.output / "report.md").is_file()
    assert tuple(
        path.name for path in sorted((completed_trajectory_run.output / "runs").glob("*.json"))
    ) == (
        "seed-0000000000.json",
        "seed-0000000001.json",
        "seed-0000000002.json",
    )


def test_runner_failure_identity_remains_software_simulation() -> None:
    config = load_experiment_config(CONFIG)

    experiment_id, backend, evidence, _, _ = _failed_identity(config)

    assert experiment_id == "numpy.trajectory_reinforce_pasym_swap_estimator.v1"
    assert backend is BackendId.NUMPY_EXACT_CATEGORICAL
    assert evidence is EvidenceClass.SOFTWARE_SIMULATION


def test_runner_persists_truthful_all_failed_software_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingBackend:
        def execute(self, spec):
            raise RuntimeError(f"seed {spec.seed} failed")

    monkeypatch.setattr(runner_module, "_backend", lambda *_: FailingBackend())
    output = tmp_path / "failed"

    aggregate = runner_module.run_experiment(CONFIG, output, seeds=(0, 1, 2))

    assert aggregate.completion_state is CompletionState.FAILED
    assert aggregate.backend_id is BackendId.NUMPY_EXACT_CATEGORICAL
    assert aggregate.evidence_class is EvidenceClass.SOFTWARE_SIMULATION
    assert aggregate.completed_runs == 0
    assert aggregate.failed_runs == 3
    assert aggregate.run_record_paths == ()
    assert tuple(failure.seed for failure in aggregate.failures) == (0, 1, 2)
    persisted = AggregateRecord.model_validate_json(
        (output / "aggregate.json").read_text(encoding="utf-8")
    )
    assert persisted.completion_state is CompletionState.FAILED


def test_aggregate_rejects_tampered_nested_summary(
    completed_trajectory_run: CompletedTrajectoryRun,
) -> None:
    payload = copy.deepcopy(
        completed_trajectory_run.records[0].model_dump(mode="json", by_alias=True)
    )
    payload["metrics"]["trajectory_reinforce_summary"]["value"]["sample"][
        "maximum_absolute_shared_gradient_error"
    ] += 0.01
    tampered = RunRecord.model_validate(payload)

    with pytest.raises(ValueError, match="sample|derived|summary"):
        _aggregate((tampered,))


def test_wrong_self_consistent_sample_count_is_rejected_at_every_record_consumer(
    completed_trajectory_run: CompletedTrajectoryRun,
    tmp_path: Path,
) -> None:
    forged = _rebuilt_wrong_count_record(completed_trajectory_run.records[0])

    with pytest.raises(ValueError, match="sample_count.*batch_size"):
        validate_persisted_trajectory_reinforce_record(forged)
    with pytest.raises(ValueError, match="sample_count.*batch_size"):
        _aggregate((forged,))

    output = tmp_path / "wrong-count"
    shutil.copytree(completed_trajectory_run.output, output)
    record_path = output / completed_trajectory_run.aggregate.run_record_paths[0]
    record_path.write_text(forged.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="sample_count.*batch_size"):
        write_report_from_persisted(output)


def test_aggregate_rejects_cross_seed_deterministic_digest_mismatch(
    completed_trajectory_run: CompletedTrajectoryRun,
) -> None:
    first, second, _ = completed_trajectory_run.records
    payload = copy.deepcopy(second.model_dump(mode="json", by_alias=True))
    payload["metrics"]["trajectory_reinforce_summary"]["value"]["deterministic"][
        "deterministic_result_digest"
    ] = "sha256:" + "0" * 64
    drifted = RunRecord.model_validate(payload)

    with pytest.raises(ValueError, match="deterministic|digest"):
        _aggregate((first, drifted))


def test_aggregate_rejects_internally_valid_non_seed_request_hash_mismatch(
    completed_trajectory_run: CompletedTrajectoryRun,
) -> None:
    mismatched = _rebuilt_summary_record(
        completed_trajectory_run.records[0],
        request_hash="sha256:" + "0" * 64,
    )

    with pytest.raises(ValueError, match="request hash"):
        _aggregate((mismatched,))


def test_aggregate_rejects_internally_valid_sample_definition_mismatch(
    completed_trajectory_run: CompletedTrajectoryRun,
) -> None:
    mismatched = _rebuilt_summary_record(
        completed_trajectory_run.records[0],
        sample_definition="forged but internally valid sample definition",
    )

    with pytest.raises(ValueError, match="sampled definition"):
        _aggregate((mismatched,))


def test_trajectory_seed_validation_precedes_output_mutation(tmp_path: Path) -> None:
    output = tmp_path / "untouched"

    with pytest.raises(ValueError, match="non-empty and unique"):
        runner_module.run_experiment(CONFIG, output, seeds=(0, 0))

    assert not output.exists()


def test_report_publishes_validated_exact_and_per_seed_trajectory_evidence(
    completed_trajectory_run: CompletedTrajectoryRun,
) -> None:
    report = (completed_trajectory_run.output / "report.md").read_text(encoding="utf-8")

    required_text = (
        "three sites / two overlapping occurrences / one shared kernel",
        "references are independently sampled from the same parent and are never propagated",
        "Target and model terminal laws",
        "Particle-number leakage",
        "Signed mass drift",
        "Exact trajectory-score shared gradient",
        "Exact expected-reference shared gradient",
        "Untied occurrence finite-difference sum",
        "Independent tied finite-difference gradient",
        "Maximum exact discrepancy",
        "Maximum finite-difference discrepancy",
        "Seed 0",
        "Seed 1",
        "Seed 2",
        "Sampled mean",
        "Standard error",
        "Absolute error",
        "Monte Carlo comparison is non-gating",
        "exact m(phi) defines the stopped reward coefficient",
        "No parameters were updated and no improvement was tested",
        "not THRML or a finite-Gibbs gradient",
        "not official Thermalizers compatibility or hosted simulation",
        "not a 25-site refinement or physical Z1/TSU measurement",
        (
            "Exact identities and exact vectors are deterministic; sampled vector estimates "
            "are per-run software_simulation evidence and are excluded from cross-seed aggregation"
        ),
    )
    for text in required_text:
        assert text in report
    assert report.index("Seed 0") < report.index("Seed 1") < report.index("Seed 2")
    assert report.count("Target and model terminal laws") == 1
    assert "Recorded Markov-chain states" not in report
    assert "empirical THRML residual" not in report


_REPORT_TAMPER_FAMILIES = (
    "experiment identity",
    "backend",
    "run evidence",
    "sample definition",
    "metric set",
    "request hash",
    "timing prefix",
    "runtime provenance",
    "deep summary",
    "standalone sampled scalar",
    "standalone acceptance",
)


def _tamper_report_source(payload: dict, family: str) -> None:
    if family == "experiment identity":
        payload["spec"]["experiment_id"] = "forged.experiment.v1"
        payload["run_config_hash"] = ExperimentSpec.model_validate(payload["spec"]).run_config_hash
    elif family == "backend":
        payload["backend_id"] = "thrml_local"
    elif family == "run evidence":
        payload["evidence_class"] = "exact_reference"
    elif family == "sample definition":
        payload["spec"]["sample_definition"] = "forged sample definition"
        payload["run_config_hash"] = ExperimentSpec.model_validate(payload["spec"]).run_config_hash
    elif family == "metric set":
        payload["metrics"]["forged_metric"] = copy.deepcopy(
            payload["metrics"]["maximum_absolute_shared_gradient_error"]
        )
    elif family == "request hash":
        payload["metrics"]["trajectory_reinforce_summary"]["value"]["deterministic"][
            "request_hash"
        ] = "sha256:" + "0" * 64
    elif family == "timing prefix":
        payload["timing"]["timing_method"] = "forged timing method"
    elif family == "runtime provenance":
        payload["provenance"]["jax_backend"] = "cpu"
    elif family == "deep summary":
        payload["metrics"]["trajectory_reinforce_summary"]["value"]["sample"][
            "maximum_absolute_shared_gradient_error"
        ] += 0.01
    elif family == "standalone sampled scalar":
        payload["metrics"]["maximum_absolute_shared_gradient_error"]["value"] += 0.01
    elif family == "standalone acceptance":
        payload["metrics"]["acceptance_passed"]["value"] = False
    else:  # pragma: no cover - the parametrization above is closed
        raise AssertionError(f"unknown tamper family: {family}")


@pytest.mark.parametrize(
    "family",
    (
        "backend",
        "sample definition",
        "metric set",
        "request hash",
        "timing prefix",
        "runtime provenance",
        "deep summary",
        "standalone sampled scalar",
        "standalone acceptance",
    ),
)
def test_trajectory_aggregate_enforces_the_complete_checked_record_contract(
    completed_trajectory_run: CompletedTrajectoryRun,
    family: str,
) -> None:
    original = completed_trajectory_run.records[0]
    payload = copy.deepcopy(original.model_dump(mode="json", by_alias=True))
    if family == "request hash":
        forged = _rebuilt_summary_record(original, request_hash="sha256:" + "0" * 64)
    else:
        _tamper_report_source(payload, family)
        forged = RunRecord.model_validate(payload)

    with pytest.raises(ValueError):
        _aggregate((forged,))


@pytest.mark.parametrize("family", _REPORT_TAMPER_FAMILIES)
def test_report_rejects_tampered_sources_before_atomic_publication(
    completed_trajectory_run: CompletedTrajectoryRun,
    tmp_path: Path,
    family: str,
) -> None:
    output = tmp_path / family.replace(" ", "-")
    shutil.copytree(completed_trajectory_run.output, output)
    report_path = output / "report.md"
    previous_report = report_path.read_bytes()
    record_path = output / completed_trajectory_run.aggregate.run_record_paths[0]
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if family == "request hash":
        rebuilt = _rebuilt_summary_record(
            RunRecord.model_validate(payload), request_hash="sha256:" + "0" * 64
        )
        payload = rebuilt.model_dump(mode="json", by_alias=True)
    else:
        _tamper_report_source(payload, family)
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        write_report_from_persisted(output)

    assert report_path.read_bytes() == previous_report
