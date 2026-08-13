from pathlib import Path

import pytest
from pydantic import ValidationError

from thermo_lab.aggregate import (
    AggregateRecord,
    CompletionState,
    RunFailure,
    StatisticalSemantics,
    aggregate_run_records,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.records import (
    ExperimentSpec,
    MetricObservation,
    PackageProvenance,
    RuntimeProvenance,
    RunTiming,
    build_run_record,
)


def _record(
    seed: int,
    value: float = 1.0,
    *,
    backend: BackendId = BackendId.THRML_LOCAL,
    ess: float | None = None,
    experiment_id: str = "test.aggregate.v1",
):
    spec = ExperimentSpec(
        experiment_id=experiment_id,
        seed=seed,
        model_config={"numeric_dtype": "float32", "weight": 1.0},
        run_config={"n_samples": 20},
        sample_definition="one correlated recorded state",
    )
    provenance = RuntimeProvenance(
        python_version="3.11.9",
        platform="test",
        jax_version="0.7.1",
        jaxlib_version="0.7.1",
        jax_backend="cpu",
        jax_devices=("cpu:test",),
        git_commit="a" * 40,
        git_dirty=False,
        jax_enable_x64=False,
        packages=(
            PackageProvenance(
                distribution="thrml",
                version="0.1.4",
                artifact_verification="locked",
            ),
        ),
    )
    evidence = (
        EvidenceClass.SOFTWARE_SIMULATION
        if backend is BackendId.THRML_LOCAL
        else EvidenceClass.EXACT_REFERENCE
    )
    metrics = {
        "scalar": MetricObservation(
            value=value,
            evidence_class=evidence,
            method="scalar method",
        ),
        "vector": MetricObservation(
            value=[value, value + 1],
            evidence_class=evidence,
            method="vector method",
        ),
        "recorded_states": MetricObservation(
            value=20,
            evidence_class=evidence,
            method="recorded count",
        ),
    }
    if ess is not None:
        metrics["minimum_spin_ess"] = MetricObservation(
            value=ess,
            evidence_class=evidence,
            method="IPS",
        )
    return build_run_record(
        backend_id=backend,
        evidence_class=evidence,
        spec=spec,
        provenance=provenance,
        timing=RunTiming(
            compile_seconds=0.2 + seed / 100,
            execution_seconds=0.01,
            synchronized=True,
            timing_method="synchronized test",
        ),
        metrics=metrics,
    )


def test_one_seed_interval_is_unavailable_and_vectors_are_not_flattened() -> None:
    aggregate = aggregate_run_records(
        [_record(7, 3.0)],
        requested_seeds=(7,),
        run_record_paths=("runs/seed-0000000007.json",),
        source_config="configs/test.toml",
    )

    scalar = aggregate.metric_aggregates["scalar"]
    assert scalar.count == 1
    assert scalar.mean == 3.0
    assert scalar.confidence_interval is None
    assert scalar.interval_unavailable_reason == "requires at least two independent seeded runs"
    assert scalar.confidence_level == 0.95
    assert aggregate.statistical_semantics is StatisticalSemantics.INDEPENDENT_SEEDED_REPLICATIONS
    assert "vector" not in aggregate.metric_aggregates


def test_multiple_seeds_use_sample_std_and_student_t_interval() -> None:
    records = [_record(seed, value) for seed, value in enumerate((1.0, 2.0, 3.0, 4.0))]

    aggregate = aggregate_run_records(
        records,
        requested_seeds=(0, 1, 2, 3),
        run_record_paths=tuple(f"runs/seed-{seed:010d}.json" for seed in range(4)),
        source_config="configs/test.toml",
    )

    scalar = aggregate.metric_aggregates["scalar"]
    assert scalar.mean == pytest.approx(2.5)
    assert scalar.standard_deviation == pytest.approx(1.2909944487)
    assert scalar.confidence_interval is not None
    assert scalar.confidence_interval.lower == pytest.approx(0.4457, abs=1e-3)
    assert scalar.confidence_interval.upper == pytest.approx(4.5543, abs=1e-3)
    assert scalar.interval_method == "two-sided Student-t across independent seeds"
    assert aggregate.statistical_semantics is StatisticalSemantics.INDEPENDENT_SEEDED_REPLICATIONS


def test_deterministic_identity_has_no_replication_interval_contract() -> None:
    aggregate = aggregate_run_records(
        [
            _record(
                0,
                3.0,
                backend=BackendId.TORX_STATEVECTOR,
                experiment_id="torx.weighted_graph_walk.v1",
            )
        ],
        requested_seeds=(0,),
        run_record_paths=("runs/seed-0000000000.json",),
        source_config="configs/experiments/torx-weighted-graph-walk.toml",
    )

    scalar = aggregate.metric_aggregates["scalar"]
    assert aggregate.statistical_semantics is StatisticalSemantics.DETERMINISTIC_IDENTITY
    assert scalar.standard_deviation is None
    assert scalar.confidence_interval is None
    assert scalar.confidence_level is None
    assert scalar.interval_method == "not applicable for deterministic execution identity"
    assert scalar.interval_unavailable_reason == (
        "confidence intervals are not applicable to deterministic identity fields"
    )


def test_ess_interval_is_capped_at_recorded_state_count() -> None:
    records = [_record(seed, ess=ess) for seed, ess in enumerate((20.0, 20.0, 19.0, 20.0))]

    aggregate = aggregate_run_records(
        records,
        requested_seeds=(0, 1, 2, 3),
        run_record_paths=tuple(f"runs/seed-{seed:010d}.json" for seed in range(4)),
        source_config="configs/test.toml",
    )

    interval = aggregate.metric_aggregates["minimum_spin_ess"].confidence_interval
    assert interval is not None
    assert interval.upper == 20.0
    assert (
        "truncated to [0, recorded_states]"
        in aggregate.metric_aggregates["minimum_spin_ess"].interval_method
    )


def test_requested_seed_order_is_preserved() -> None:
    aggregate = aggregate_run_records(
        [_record(9), _record(2)],
        requested_seeds=(9, 2),
        run_record_paths=("runs/seed-0000000009.json", "runs/seed-0000000002.json"),
        source_config="config.toml",
    )

    assert aggregate.seeds == (9, 2)
    assert aggregate.run_record_paths == (
        "runs/seed-0000000009.json",
        "runs/seed-0000000002.json",
    )


def test_incompatible_runs_are_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        aggregate_run_records(
            [_record(0), _record(1, backend=BackendId.TORX_STATEVECTOR)],
            requested_seeds=(0, 1),
            run_record_paths=("runs/seed-0000000000.json", "runs/seed-0000000001.json"),
            source_config="config.toml",
        )


def test_incompatible_runtime_provenance_is_rejected() -> None:
    first = _record(0)
    payload = _record(1).model_dump(mode="python", by_alias=True)
    payload["provenance"]["jax_backend"] = "gpu"
    second = type(first).model_validate(payload)

    with pytest.raises(ValueError, match="JAX backend"):
        aggregate_run_records(
            [first, second],
            requested_seeds=(0, 1),
            run_record_paths=("runs/seed-0000000000.json", "runs/seed-0000000001.json"),
            source_config="config.toml",
        )


@pytest.mark.parametrize(
    ("failures", "expected"),
    [
        ((RunFailure(seed=2, error_type="RuntimeError", message="bad"),), CompletionState.PARTIAL),
        ((RunFailure(seed=1, error_type="RuntimeError", message="bad"),), CompletionState.FAILED),
    ],
)
def test_partial_and_failed_completion_states(
    failures: tuple[RunFailure, ...], expected: CompletionState
) -> None:
    records = [_record(1)] if expected is CompletionState.PARTIAL else []
    paths = ("runs/seed-0000000001.json",) if records else ()
    requested = (1, 2) if records else (1,)

    aggregate = aggregate_run_records(
        records,
        requested_seeds=requested,
        run_record_paths=paths,
        source_config="config.toml",
        failures=failures,
        failed_identity=(
            "test.aggregate.v1",
            BackendId.THRML_LOCAL,
            EvidenceClass.SOFTWARE_SIMULATION,
            "sha256:" + "0" * 64,
            "sha256:" + "1" * 64,
        )
        if not records
        else None,
    )

    assert aggregate.completion_state is expected
    assert aggregate.completed_runs == len(records)
    assert aggregate.failed_runs == len(failures)


def test_aggregate_round_trips_and_rejects_absolute_record_paths(tmp_path: Path) -> None:
    aggregate = aggregate_run_records(
        [_record(1)],
        requested_seeds=(1,),
        run_record_paths=("runs/seed-0000000001.json",),
        source_config="config.toml",
    )
    path = tmp_path / "aggregate.json"
    aggregate.write_json(path)

    assert AggregateRecord.model_validate_json(path.read_text(encoding="utf-8")) == aggregate
    payload = aggregate.model_dump(mode="python")
    payload["run_record_paths"] = ("/absolute/run.json",)
    with pytest.raises(ValidationError, match="relative"):
        AggregateRecord.model_validate(payload)

    payload = aggregate.model_dump(mode="python")
    payload["schema_version"] = "2.0.0"
    with pytest.raises(ValidationError, match="schema_version"):
        AggregateRecord.model_validate(payload)
