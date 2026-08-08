"""Immutable experiment inputs and observed run records."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from thermo_lab.evidence import (
    BackendId,
    EvidenceClass,
    validate_backend_evidence,
    validate_metric_evidence,
)
from thermo_lab.hashing import canonical_sha256, to_json_value


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class FrozenDict(Mapping[str, Any]):
    """Tuple-backed JSON mapping with no mutable ``dict`` base to bypass."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any]):
        object.__setattr__(self, "_items", tuple(values.items()))

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self)!r})"

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("experiment JSON values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


class ExperimentSpec(FrozenModel):
    """Requested experiment inputs; no observations or runtime metadata."""

    experiment_id: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    model_parameters: Mapping[str, Any] = Field(alias="model_config")
    run_parameters: Mapping[str, Any] = Field(alias="run_config")
    sample_definition: str = Field(min_length=1)

    @field_validator("model_parameters", "run_parameters", mode="before")
    @classmethod
    def normalize_parameters(cls, value: Any) -> dict[str, Any]:
        normalized = to_json_value(value)
        if not isinstance(normalized, dict):
            raise TypeError("Experiment model and run configurations must be JSON objects")
        return normalized

    @field_validator("model_parameters", "run_parameters", mode="after")
    @classmethod
    def freeze_parameters(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        # Freeze after Pydantic's mapping normalization so the outermost
        # mapping is immutable as well as its nested values.
        return _freeze_json(value)

    @field_serializer("model_parameters", "run_parameters")
    def serialize_parameters(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return to_json_value(value)

    @property
    def model_hash(self) -> str:
        return canonical_sha256(self.model_parameters)

    @property
    def run_config_hash(self) -> str:
        return canonical_sha256(
            {
                "experiment_id": self.experiment_id,
                "seed": self.seed,
                "run_config": self.run_parameters,
                "sample_definition": self.sample_definition,
            }
        )

    @property
    def non_seed_run_config_hash(self) -> str:
        """Hash run inputs shared by independent replications, excluding seed."""

        return canonical_sha256(
            {
                "experiment_id": self.experiment_id,
                "run_config": self.run_parameters,
                "sample_definition": self.sample_definition,
            }
        )


class MetricObservation(FrozenModel):
    """One claim and the evidence supporting it."""

    value: Any
    evidence_class: EvidenceClass
    unit: str | None = None
    method: str = Field(min_length=1)
    source: str | None = None
    notes: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value(cls, value: Any) -> Any:
        return _freeze_json(to_json_value(value))

    @field_serializer("value")
    def serialize_value(self, value: Any) -> Any:
        return to_json_value(value)


class PackageProvenance(FrozenModel):
    distribution: str
    version: str
    release_source_repository: str | None = None
    release_source_commit: str | None = None
    expected_wheel_sha256: str | None = None
    artifact_verification: str


class RuntimeProvenance(FrozenModel):
    python_version: str
    platform: str
    jax_version: str
    jaxlib_version: str
    jax_backend: str
    jax_devices: tuple[str, ...]
    git_commit: str | None
    git_dirty: StrictBool | None
    jax_enable_x64: StrictBool
    packages: tuple[PackageProvenance, ...]


class RunTiming(FrozenModel):
    compile_seconds: float = Field(ge=0)
    execution_seconds: float = Field(ge=0)
    synchronized: StrictBool
    timing_method: str = Field(min_length=1)

    @field_validator("compile_seconds", "execution_seconds", mode="before")
    @classmethod
    def reject_duration_coercions(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Timing durations must be real numbers, not coerced values")
        return value


class RunRecord(FrozenModel):
    """Observed output from exactly one backend execution."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    backend_id: BackendId
    evidence_class: EvidenceClass
    spec: ExperimentSpec
    model_hash: str
    run_config_hash: str
    provenance: RuntimeProvenance
    timing: RunTiming
    metrics: Mapping[str, MetricObservation]

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(
        cls, value: Mapping[str, MetricObservation]
    ) -> Mapping[str, MetricObservation]:
        return FrozenDict(value)

    @field_serializer("metrics")
    def serialize_metrics(self, value: Mapping[str, MetricObservation]) -> dict[str, Any]:
        return {key: to_json_value(metric) for key, metric in value.items()}

    @model_validator(mode="after")
    def validate_record(self) -> RunRecord:
        validate_backend_evidence(self.backend_id, self.evidence_class)
        if self.model_hash != self.spec.model_hash:
            raise ValueError("model_hash does not match canonical experiment model input")
        if self.run_config_hash != self.spec.run_config_hash:
            raise ValueError("run_config_hash does not match canonical experiment run input")
        if not self.timing.synchronized:
            raise ValueError("Run timing must synchronize asynchronous accelerator work")
        if not self.metrics:
            raise ValueError("A run record must contain at least one observed metric")
        for metric in self.metrics.values():
            validate_metric_evidence(self.backend_id, metric.evidence_class)
        return self

    def write_json(self, path: Path) -> None:
        # Revalidate the complete payload at the persistence boundary. This is
        # Defense in depth at the persistence boundary.
        validated = RunRecord.model_validate(self.model_dump(mode="python", by_alias=True))
        from thermo_lab.persistence import atomic_write_text

        payload = validated.model_dump(mode="json", by_alias=True)
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_run_record(
    *,
    backend_id: BackendId,
    evidence_class: EvidenceClass,
    spec: ExperimentSpec,
    provenance: RuntimeProvenance,
    timing: RunTiming,
    metrics: dict[str, MetricObservation],
) -> RunRecord:
    """Construct a record while deriving, rather than trusting, input hashes."""

    return RunRecord(
        backend_id=backend_id,
        evidence_class=evidence_class,
        spec=spec,
        model_hash=spec.model_hash,
        run_config_hash=spec.run_config_hash,
        provenance=provenance,
        timing=timing,
        metrics=metrics,
    )
