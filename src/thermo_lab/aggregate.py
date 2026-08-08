"""Compatibility-checked aggregation across independent seeded executions."""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, StrictInt, field_serializer, field_validator, model_validator

from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.records import FrozenDict, FrozenModel, RunRecord

AGGREGATE_SCHEMA_VERSION = "1.0.0"
CONFIDENCE_LEVEL = 0.95


class CompletionState(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ConfidenceInterval(FrozenModel):
    lower: float
    upper: float


class ScalarAggregate(FrozenModel):
    count: StrictInt = Field(ge=1)
    mean: float
    standard_deviation: float | None
    median: float
    minimum: float
    maximum: float
    confidence_interval: ConfidenceInterval | None
    confidence_level: float = CONFIDENCE_LEVEL
    interval_method: str
    interval_unavailable_reason: str | None = None
    unit: str | None = None
    evidence_class: EvidenceClass | None = None
    method: str = Field(min_length=1)


class PackageVersion(FrozenModel):
    distribution: str
    version: str


class ProvenanceCompatibilitySummary(FrozenModel):
    python_version: str
    platform: str
    jax_version: str
    jaxlib_version: str
    jax_backend: str
    jax_devices: tuple[str, ...]
    jax_enable_x64: bool
    numeric_dtype: str
    git_commit: str | None
    git_dirty: bool | None
    packages: tuple[PackageVersion, ...]


class RunFailure(FrozenModel):
    seed: StrictInt = Field(ge=0)
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ReportGenerationMetadata(FrozenModel):
    generator: str = "thermo_lab.reporting.render_report"
    generated_from_persisted_records: bool = True
    report_path: str = "report.md"


class AggregateRecord(FrozenModel):
    """Immutable summary whose replication unit is one independently seeded run."""

    schema_version: Literal[AGGREGATE_SCHEMA_VERSION] = AGGREGATE_SCHEMA_VERSION
    aggregate_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    experiment_id: str = Field(min_length=1)
    backend_id: BackendId
    evidence_class: EvidenceClass
    model_hash: str
    run_config_hash: str
    source_config: str = Field(min_length=1)
    seeds: tuple[StrictInt, ...]
    requested_runs: StrictInt = Field(ge=1)
    completed_runs: StrictInt = Field(ge=0)
    failed_runs: StrictInt = Field(ge=0)
    run_record_paths: tuple[str, ...]
    failures: tuple[RunFailure, ...] = ()
    provenance_summary: ProvenanceCompatibilitySummary | None
    metric_aggregates: Mapping[str, ScalarAggregate]
    omitted_metrics: Mapping[str, str]
    report_generation: ReportGenerationMetadata = Field(default_factory=ReportGenerationMetadata)
    completion_state: CompletionState

    @field_validator("metric_aggregates", "omitted_metrics", mode="after")
    @classmethod
    def freeze_mappings(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return FrozenDict(value)

    @field_serializer("metric_aggregates", "omitted_metrics")
    def serialize_mappings(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(value)

    @field_validator("run_record_paths")
    @classmethod
    def validate_relative_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        for path in paths:
            parsed = PurePosixPath(path)
            if parsed.is_absolute() or ".." in parsed.parts:
                raise ValueError("Run record paths must be relative and portable")
        return paths

    @model_validator(mode="after")
    def validate_counts_and_state(self) -> AggregateRecord:
        if len(self.seeds) != self.requested_runs or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain each requested seed exactly once")
        if self.completed_runs + self.failed_runs != self.requested_runs:
            raise ValueError("completed and failed run counts must equal requested_runs")
        if len(self.run_record_paths) != self.completed_runs:
            raise ValueError("run_record_paths count must equal completed_runs")
        if len(self.failures) != self.failed_runs:
            raise ValueError("failure details count must equal failed_runs")
        expected_state = (
            CompletionState.COMPLETE
            if self.failed_runs == 0
            else CompletionState.FAILED
            if self.completed_runs == 0
            else CompletionState.PARTIAL
        )
        if self.completion_state is not expected_state:
            raise ValueError(f"completion_state must be {expected_state.value!r} for these counts")
        if self.completed_runs and self.provenance_summary is None:
            raise ValueError("Successful aggregates require a provenance compatibility summary")
        return self

    def write_json(self, path: Path) -> None:
        validated = AggregateRecord.model_validate(self.model_dump(mode="python"))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(validated.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.replace(temporary, path)


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    table = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    if degrees_of_freedom <= len(table):
        return table[degrees_of_freedom - 1]
    # Third-order expansion of the t quantile about the standard normal.
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def _summarize_scalar(
    values: Sequence[float],
    *,
    unit: str | None,
    evidence_class: EvidenceClass | None,
    method: str,
    interval_bounds: tuple[float, float] | None = None,
) -> ScalarAggregate:
    count = len(values)
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if count >= 2 else None
    interval = None
    reason = None
    if standard_deviation is None:
        reason = "requires at least two independent seeded runs"
    else:
        critical = _student_t_critical_95(count - 1)
        margin = critical * standard_deviation / math.sqrt(count)
        lower = mean - margin
        upper = mean + margin
        if interval_bounds is not None:
            lower = max(interval_bounds[0], lower)
            upper = min(interval_bounds[1], upper)
        interval = ConfidenceInterval(lower=lower, upper=upper)
    interval_method = "two-sided Student-t across independent seeds"
    if interval_bounds is not None:
        interval_method += "; truncated to [0, recorded_states]"
    return ScalarAggregate(
        count=count,
        mean=mean,
        standard_deviation=standard_deviation,
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
        confidence_interval=interval,
        interval_method=interval_method,
        interval_unavailable_reason=reason,
        unit=unit,
        evidence_class=evidence_class,
        method=method,
    )


def _compatibility_signature(record: RunRecord) -> tuple[Any, ...]:
    package_versions = tuple(
        sorted((package.distribution, package.version) for package in record.provenance.packages)
    )
    return (
        record.spec.experiment_id,
        record.backend_id,
        record.evidence_class,
        record.model_hash,
        record.spec.non_seed_run_config_hash,
        record.spec.sample_definition,
        package_versions,
        record.provenance.python_version,
        record.provenance.platform,
        record.provenance.jax_version,
        record.provenance.jaxlib_version,
        record.provenance.jax_backend,
        record.provenance.jax_devices,
        record.spec.model_parameters.get("numeric_dtype"),
        record.provenance.jax_enable_x64,
        record.timing.timing_method,
    )


def _provenance_summary(record: RunRecord) -> ProvenanceCompatibilitySummary:
    packages = tuple(
        PackageVersion(distribution=name, version=version)
        for name, version in sorted(
            (package.distribution, package.version) for package in record.provenance.packages
        )
    )
    return ProvenanceCompatibilitySummary(
        python_version=record.provenance.python_version,
        platform=record.provenance.platform,
        jax_version=record.provenance.jax_version,
        jaxlib_version=record.provenance.jaxlib_version,
        jax_backend=record.provenance.jax_backend,
        jax_devices=record.provenance.jax_devices,
        jax_enable_x64=record.provenance.jax_enable_x64,
        numeric_dtype=str(record.spec.model_parameters.get("numeric_dtype")),
        git_commit=record.provenance.git_commit,
        git_dirty=record.provenance.git_dirty,
        packages=packages,
    )


def aggregate_run_records(
    records: Sequence[RunRecord],
    *,
    requested_seeds: tuple[int, ...],
    run_record_paths: tuple[str, ...],
    source_config: str,
    failures: tuple[RunFailure, ...] = (),
    failed_identity: tuple[str, BackendId, EvidenceClass, str, str] | None = None,
) -> AggregateRecord:
    """Aggregate compatible scalar metrics using independent seeds as replications."""

    if not requested_seeds or len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("Requested seeds must be non-empty and unique")
    if len(records) != len(run_record_paths):
        raise ValueError("Every successful record requires one relative record path")
    successful_seeds = tuple(record.spec.seed for record in records)
    failed_seeds = tuple(failure.seed for failure in failures)
    if tuple(seed for seed in requested_seeds if seed not in failed_seeds) != successful_seeds:
        raise ValueError("Successful records must preserve requested seed ordering")
    if set(successful_seeds).intersection(failed_seeds):
        raise ValueError("A seed cannot be both successful and failed")

    if records:
        first = records[0]
        signature = _compatibility_signature(first)
        labels = (
            "experiment_id",
            "backend",
            "evidence_class",
            "model_hash",
            "non-seed run configuration",
            "sample_definition",
            "package versions",
            "Python version",
            "platform",
            "JAX version",
            "JAXLIB version",
            "JAX backend",
            "JAX devices",
            "numeric dtype",
            "JAX x64 setting",
            "timing method",
        )
        for record in records[1:]:
            candidate = _compatibility_signature(record)
            for label, expected, actual in zip(labels, signature, candidate, strict=True):
                if actual != expected:
                    raise ValueError(
                        f"Cannot aggregate incompatible {label}: {expected!r} != {actual!r}"
                    )
        experiment_id = first.spec.experiment_id
        backend_id = first.backend_id
        evidence_class = first.evidence_class
        model_hash = first.model_hash
        run_config_hash = first.spec.non_seed_run_config_hash
        provenance_summary = _provenance_summary(first)
    elif failed_identity is not None:
        experiment_id, backend_id, evidence_class, model_hash, run_config_hash = failed_identity
        provenance_summary = None
    else:
        raise ValueError("All-failed aggregation requires checked configuration identity")

    metric_aggregates: dict[str, ScalarAggregate] = {}
    omitted_metrics: dict[str, str] = {}
    if records:
        common_names = set(records[0].metrics)
        for record in records[1:]:
            common_names.intersection_update(record.metrics)
        all_names = set().union(*(set(record.metrics) for record in records))
        for name in sorted(all_names - common_names):
            omitted_metrics[name] = "metric is not present in every successful run"
        for name in sorted(common_names):
            observations = [record.metrics[name] for record in records]
            values = [observation.value for observation in observations]
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
            ):
                omitted_metrics[name] = "non-scalar metric retained only in per-run records"
                continue
            metadata = {
                (observation.unit, observation.evidence_class, observation.method)
                for observation in observations
            }
            if len(metadata) != 1:
                raise ValueError(f"Cannot aggregate metric {name!r} with incompatible metadata")
            unit, metric_evidence, method = next(iter(metadata))
            interval_bounds = None
            if name in {"minimum_spin_ess", "median_spin_ess", "magnetization_trace_ess"}:
                recorded_counts = [record.metrics.get("recorded_states") for record in records]
                if any(item is None or not isinstance(item.value, int) for item in recorded_counts):
                    raise ValueError(f"Cannot bound ESS metric {name!r} without recorded_states")
                upper_bound = float(min(item.value for item in recorded_counts if item is not None))
                if any(float(value) < 0 or float(value) > upper_bound for value in values):
                    raise ValueError(f"ESS metric {name!r} exceeds recorded-state bounds")
                interval_bounds = (0.0, upper_bound)
            metric_aggregates[name] = _summarize_scalar(
                [float(value) for value in values],
                unit=unit,
                evidence_class=metric_evidence,
                method=method,
                interval_bounds=interval_bounds,
            )
        timing_method = records[0].timing.timing_method
        metric_aggregates["timing.compile_seconds"] = _summarize_scalar(
            [record.timing.compile_seconds for record in records],
            unit="seconds",
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method=(
                f"{timing_method}; compilation interval only; excludes execution, "
                "configuration loading, provenance collection, persistence, aggregation, "
                "and reporting; source=RunTiming"
            ),
        )
        metric_aggregates["timing.execution_seconds"] = _summarize_scalar(
            [record.timing.execution_seconds for record in records],
            unit="seconds",
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method=(
                f"{timing_method}; synchronized steady-state backend interval only; "
                "excludes compilation, untimed warm launch, configuration loading, "
                "provenance collection, persistence, aggregation, and reporting; "
                "source=RunTiming"
            ),
        )

    state = (
        CompletionState.COMPLETE
        if not failures
        else CompletionState.FAILED
        if not records
        else CompletionState.PARTIAL
    )
    return AggregateRecord(
        experiment_id=experiment_id,
        backend_id=backend_id,
        evidence_class=evidence_class,
        model_hash=model_hash,
        run_config_hash=run_config_hash,
        source_config=source_config,
        seeds=requested_seeds,
        requested_runs=len(requested_seeds),
        completed_runs=len(records),
        failed_runs=len(failures),
        run_record_paths=run_record_paths,
        failures=failures,
        provenance_summary=provenance_summary,
        metric_aggregates=metric_aggregates,
        omitted_metrics=omitted_metrics,
        completion_state=state,
    )
