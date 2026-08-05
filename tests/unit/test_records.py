from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.records import (
    ExperimentSpec,
    MetricObservation,
    PackageProvenance,
    RunRecord,
    RuntimeProvenance,
    RunTiming,
    build_run_record,
)


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="test.v1",
        seed=4,
        model_config={"weights": [1.0], "beta": 1.0},
        run_config={"samples": 10},
        sample_definition="one exact probability vector",
    )


def _provenance() -> RuntimeProvenance:
    return RuntimeProvenance(
        python_version="3.11.9",
        platform="test",
        jax_version="0.test",
        jaxlib_version="0.test",
        jax_backend="cpu",
        jax_devices=("cpu:test",),
        git_commit="a" * 40,
        git_dirty=False,
        jax_enable_x64=False,
        packages=(
            PackageProvenance(
                distribution="thermo-lab",
                version="0.1.0",
                artifact_verification="local_editable_install",
            ),
        ),
    )


def _timing() -> RunTiming:
    return RunTiming(
        compile_seconds=0.1,
        execution_seconds=0.01,
        synchronized=True,
        timing_method="test clock after block_until_ready",
    )


def test_record_derives_hashes_and_round_trips() -> None:
    spec = _spec()
    record = build_run_record(
        backend_id=BackendId.TORX_STATEVECTOR,
        evidence_class=EvidenceClass.EXACT_REFERENCE,
        spec=spec,
        provenance=_provenance(),
        timing=_timing(),
        metrics={
            "probability_sum": MetricObservation(
                value=1.0,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method="exact state vector",
            )
        },
    )

    assert record.model_hash == spec.model_hash
    assert record.run_config_hash == spec.run_config_hash
    assert RunRecord.model_validate_json(record.model_dump_json()) == record


def test_experiment_seed_rejects_integer_coercion() -> None:
    payload = _spec().model_dump(mode="python", by_alias=True)
    payload["seed"] = "4"

    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_record_rejects_manufactured_physical_hardware_claim() -> None:
    with pytest.raises(ValidationError, match="cannot emit 'physical_hardware'"):
        RunRecord(
            created_at_utc=datetime.now(UTC),
            backend_id=BackendId.THRML_LOCAL,
            evidence_class=EvidenceClass.PHYSICAL_HARDWARE,
            spec=_spec(),
            model_hash=_spec().model_hash,
            run_config_hash=_spec().run_config_hash,
            provenance=_provenance(),
            timing=_timing(),
            metrics={
                "value": MetricObservation(
                    value=1,
                    evidence_class=EvidenceClass.PHYSICAL_HARDWARE,
                    method="false claim",
                )
            },
        )


def test_record_rejects_physical_metric_hidden_in_software_run() -> None:
    with pytest.raises(ValidationError, match="cannot contain a 'physical_hardware' metric"):
        build_run_record(
            backend_id=BackendId.THRML_LOCAL,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            spec=_spec(),
            provenance=_provenance(),
            timing=_timing(),
            metrics={
                "value": MetricObservation(
                    value=1,
                    evidence_class=EvidenceClass.PHYSICAL_HARDWARE,
                    method="false claim",
                )
            },
        )


def test_record_rejects_unsynchronized_timing() -> None:
    with pytest.raises(ValidationError, match="must synchronize"):
        build_run_record(
            backend_id=BackendId.TORX_STATEVECTOR,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            spec=_spec(),
            provenance=_provenance(),
            timing=RunTiming(
                compile_seconds=0,
                execution_seconds=0,
                synchronized=False,
                timing_method="dispatch only",
            ),
            metrics={
                "value": MetricObservation(
                    value=1,
                    evidence_class=EvidenceClass.EXACT_REFERENCE,
                    method="exact",
                )
            },
        )


def test_experiment_and_record_containers_are_deeply_immutable() -> None:
    spec = _spec()
    record = build_run_record(
        backend_id=BackendId.TORX_STATEVECTOR,
        evidence_class=EvidenceClass.EXACT_REFERENCE,
        spec=spec,
        provenance=_provenance(),
        timing=_timing(),
        metrics={
            "value": MetricObservation(
                value={"distribution": [1.0, 0.0]},
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method="exact",
            )
        },
    )

    with pytest.raises(TypeError, match="immutable"):
        spec.model_parameters["beta"] = 2.0
    with pytest.raises(TypeError):
        dict.__setitem__(spec.model_parameters, "beta", 2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="immutable"):
        record.metrics["forged"] = MetricObservation(
            value=1,
            evidence_class=EvidenceClass.PHYSICAL_HARDWARE,
            method="forged",
        )
    with pytest.raises(TypeError):
        dict.__setitem__(  # type: ignore[arg-type]
            record.metrics,
            "forged",
            MetricObservation(
                value=1,
                evidence_class=EvidenceClass.PHYSICAL_HARDWARE,
                method="forged",
            ),
        )
    distribution = record.metrics["value"].value["distribution"]
    assert isinstance(distribution, tuple)
    with pytest.raises(TypeError):
        distribution[0] = 0.0


def test_timing_rejects_infinity() -> None:
    with pytest.raises(ValidationError):
        RunTiming(
            compile_seconds=float("inf"),
            execution_seconds=0,
            synchronized=True,
            timing_method="invalid",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("compile_seconds", "0"),
        ("execution_seconds", "1"),
        ("synchronized", "yes"),
        ("synchronized", 1),
    ],
)
def test_timing_rejects_evidence_coercions(field: str, value: object) -> None:
    timing: dict[str, object] = {
        "compile_seconds": 0.0,
        "execution_seconds": 1.0,
        "synchronized": True,
        "timing_method": "strict timing",
    }
    timing[field] = value

    with pytest.raises(ValidationError):
        RunTiming.model_validate(timing)


@pytest.mark.parametrize(
    ("field", "value"),
    [("git_dirty", "no"), ("jax_enable_x64", 0)],
)
def test_runtime_provenance_rejects_boolean_coercions(field: str, value: object) -> None:
    provenance = _provenance().model_dump(mode="python")
    provenance[field] = value

    with pytest.raises(ValidationError):
        RuntimeProvenance.model_validate(provenance)
