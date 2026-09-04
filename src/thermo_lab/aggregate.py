"""Compatibility-checked aggregation with explicit persisted statistical semantics."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from thermo_lab.config import TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.persistence import atomic_write_text
from thermo_lab.records import RUN_TIMING_SOURCE, FrozenDict, FrozenModel, RunRecord
from thermo_lab.schemas import (
    WEIGHTED_GRAPH_WALK_EXPERIMENT_ID,
    PAsymSwapModelConfig,
    TargetContextCompilerRunConfig,
    validate_target_context_pasym_swap_request,
)
from thermo_lab.target_context_pasym_swap_results import (
    validate_target_context_pasym_swap_observations,
)

AGGREGATE_SCHEMA_VERSION = "1.1.0"
CONFIDENCE_LEVEL = 0.95
_INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID = "thrml.independent_pasym_swap_compilation.v1"
_INDEPENDENT_PASYM_SWAP_SAMPLED_METRICS = frozenset({"maximum_empirical_k30_residual"})
_INDEPENDENT_PASYM_SWAP_OMITTED_METRIC_REASONS = {
    "median_equilibrium_tv": (
        "deterministic exact conditional diagnostic is not an independently seeded sampled "
        "cross-check"
    ),
    "worst_equilibrium_tv": (
        "deterministic exact conditional diagnostic is not an independently seeded sampled "
        "cross-check"
    ),
    "maximum_k30_equilibrium_residual": (
        "deterministic exact conditional diagnostic is not an independently seeded sampled "
        "cross-check"
    ),
    "successful_artifact_count": (
        "deterministic compiler identity is not an independently seeded sampled cross-check"
    ),
    "total_cap_active_parameter_count": (
        "deterministic compiler identity is not an independently seeded sampled cross-check"
    ),
    "acceptance_passed": (
        "deterministic acceptance identity is not an independently seeded sampled cross-check"
    ),
    "deterministic_optimizer_seconds": (
        "deterministic compiler/cache timing is not an independently seeded sampled cross-check"
    ),
}
_INDEPENDENT_PASYM_SWAP_TIMING_OMISSION_REASON = (
    "per-seed cache timing is not an independently seeded sampled cross-check"
)
_INDEPENDENT_PASYM_SWAP_TIMING_METHOD_PREFIX = (
    "cached shared jax.jit(jax.vmap(single_chain)) executable; one untimed synchronized "
    "warm launch, then aggregate synchronized steady-state execution"
)
_TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID = "thrml.target_context_pasym_swap_compilation.v1"
_TARGET_CONTEXT_PASYM_SWAP_SAMPLED_METRICS = frozenset({"maximum_empirical_k30_residual"})
_TARGET_CONTEXT_PASYM_SWAP_OMITTED_METRIC_REASONS = {
    "target_context_pasym_swap": (
        "nested target-context evidence is retained only in per-run records"
    ),
    "baseline_occurrence_weighted_equilibrium_kl": (
        "deterministic exact target-context diagnostic is not an independently seeded sampled "
        "cross-check"
    ),
    "target_context_occurrence_weighted_equilibrium_kl": (
        "deterministic exact target-context diagnostic is not an independently seeded sampled "
        "cross-check"
    ),
    "occurrence_weighted_equilibrium_kl_improvement": (
        "deterministic exact target-context diagnostic is not an independently seeded sampled "
        "cross-check"
    ),
    "baseline_occurrence_weighted_equilibrium_tv": (
        "deterministic exact target-context diagnostic is not an independently seeded sampled "
        "cross-check"
    ),
    "target_context_occurrence_weighted_equilibrium_tv": (
        "deterministic exact target-context diagnostic is not an independently seeded sampled "
        "cross-check"
    ),
    "maximum_paired_k30_equilibrium_residual": (
        "deterministic exact target-context diagnostic is not an independently seeded sampled "
        "cross-check"
    ),
    "acceptance_passed": (
        "deterministic acceptance identity is not an independently seeded sampled cross-check"
    ),
    "baseline_optimizer_seconds": (
        "per-seed optimizer/cache timing is not an independently seeded sampled cross-check"
    ),
    "target_context_optimizer_seconds": (
        "per-seed optimizer/cache timing is not an independently seeded sampled cross-check"
    ),
}
_TARGET_CONTEXT_PASYM_SWAP_TIMING_OMISSION_REASON = (
    "per-seed synchronized JAX timing is not an independently seeded sampled cross-check"
)
_TARGET_CONTEXT_PASYM_SWAP_TIMING_METHOD_PREFIX = (
    "cached shared jax.jit(jax.vmap(single_chain)) executable; one untimed synchronized "
    "warm launch, then aggregate synchronized steady-state execution"
)
_TARGET_CONTEXT_PASYM_SWAP_TIMING_METHOD_SUFFIXES = (
    "; JAX lower().compile() measured once for shared shapes",
    "; JAX executable reused from in-process shape cache",
)


class CompletionState(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class StatisticalSemantics(StrEnum):
    """Persisted contract identifying what, if anything, is a replication unit."""

    INDEPENDENT_SEEDED_REPLICATIONS = "independent_seeded_replications"
    DETERMINISTIC_IDENTITY = "deterministic_identity"


class ConfidenceInterval(FrozenModel):
    lower: StrictFloat
    upper: StrictFloat


class ScalarAggregate(FrozenModel):
    count: StrictInt = Field(ge=1)
    mean: StrictFloat
    standard_deviation: StrictFloat | None
    median: StrictFloat
    minimum: StrictFloat
    maximum: StrictFloat
    confidence_interval: ConfidenceInterval | None
    confidence_level: StrictFloat | None
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


@dataclass(frozen=True)
class AggregateDerivedFields:
    """Purely derived aggregate claims, excluding identity and creation metadata."""

    experiment_id: str
    statistical_semantics: StatisticalSemantics
    backend_id: BackendId
    evidence_class: EvidenceClass
    model_hash: str
    run_config_hash: str
    source_config: str
    seeds: tuple[int, ...]
    requested_runs: int
    completed_runs: int
    failed_runs: int
    run_record_paths: tuple[str, ...]
    failures: tuple[RunFailure, ...]
    provenance_summary: ProvenanceCompatibilitySummary | None
    metric_aggregates: Mapping[str, ScalarAggregate]
    omitted_metrics: Mapping[str, str]
    completion_state: CompletionState


class AggregateRecord(FrozenModel):
    """Immutable summary with an experiment-selected statistical contract."""

    schema_version: Literal[AGGREGATE_SCHEMA_VERSION] = AGGREGATE_SCHEMA_VERSION
    aggregate_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    experiment_id: str = Field(min_length=1)
    statistical_semantics: StatisticalSemantics
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
        expected_semantics = _statistical_semantics_for_experiment(self.experiment_id)
        if self.statistical_semantics is not expected_semantics:
            raise ValueError(
                "statistical_semantics must match the checked experiment identity: "
                f"expected {expected_semantics.value!r}"
            )
        if self.statistical_semantics is StatisticalSemantics.DETERMINISTIC_IDENTITY and (
            self.seeds != (0,) or self.requested_runs != 1
        ):
            raise ValueError(
                "deterministic identity aggregates require exactly the seed-zero identity"
            )
        for name, metric in self.metric_aggregates.items():
            if metric.count != self.completed_runs:
                raise ValueError(
                    f"metric {name!r} count must equal the completed deterministic or seeded runs"
                )
            if self.statistical_semantics is StatisticalSemantics.DETERMINISTIC_IDENTITY:
                if (
                    metric.standard_deviation is not None
                    or metric.confidence_interval is not None
                    or metric.confidence_level is not None
                    or metric.interval_method
                    != "not applicable for deterministic execution identity"
                    or metric.interval_unavailable_reason
                    != "confidence intervals are not applicable to deterministic identity fields"
                ):
                    raise ValueError(
                        f"deterministic metric {name!r} must persist not-applicable "
                        "confidence-interval metadata"
                    )
            elif (
                metric.confidence_level != CONFIDENCE_LEVEL
                or not metric.interval_method.startswith(
                    "two-sided Student-t across independent seeds"
                )
            ):
                raise ValueError(
                    f"independent-seed metric {name!r} must persist the 95% Student-t contract"
                )
            elif metric.count == 1 and (
                metric.standard_deviation is not None
                or metric.confidence_interval is not None
                or metric.interval_unavailable_reason
                != "requires at least two independent seeded runs"
            ):
                raise ValueError(
                    f"one-run independent-seed metric {name!r} must persist its unavailable reason"
                )
            elif metric.count >= 2 and (
                metric.standard_deviation is None
                or metric.confidence_interval is None
                or metric.interval_unavailable_reason is not None
            ):
                raise ValueError(
                    f"replicated independent-seed metric {name!r} must persist its interval"
                )
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
        payload = json.dumps(validated.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        atomic_write_text(path, payload)


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


def _statistical_semantics_for_experiment(experiment_id: str) -> StatisticalSemantics:
    if experiment_id == WEIGHTED_GRAPH_WALK_EXPERIMENT_ID:
        return StatisticalSemantics.DETERMINISTIC_IDENTITY
    return StatisticalSemantics.INDEPENDENT_SEEDED_REPLICATIONS


def _summarize_scalar(
    values: Sequence[float],
    *,
    unit: str | None,
    evidence_class: EvidenceClass | None,
    method: str,
    statistical_semantics: StatisticalSemantics,
    interval_bounds: tuple[float, float] | None = None,
) -> ScalarAggregate:
    count = len(values)
    mean = statistics.fmean(values)
    standard_deviation = None
    interval = None
    confidence_level = None
    if statistical_semantics is StatisticalSemantics.DETERMINISTIC_IDENTITY:
        reason = "confidence intervals are not applicable to deterministic identity fields"
        interval_method = "not applicable for deterministic execution identity"
    elif count < 2:
        reason = "requires at least two independent seeded runs"
        confidence_level = CONFIDENCE_LEVEL
        interval_method = "two-sided Student-t across independent seeds"
    else:
        reason = None
        confidence_level = CONFIDENCE_LEVEL
        standard_deviation = statistics.stdev(values)
        critical = _student_t_critical_95(count - 1)
        margin = critical * standard_deviation / math.sqrt(count)
        lower = mean - margin
        upper = mean + margin
        if interval_bounds is not None:
            lower = max(interval_bounds[0], lower)
            upper = min(interval_bounds[1], upper)
        interval = ConfidenceInterval(lower=lower, upper=upper)
        interval_method = "two-sided Student-t across independent seeds"
    if (
        statistical_semantics is StatisticalSemantics.INDEPENDENT_SEEDED_REPLICATIONS
        and interval_bounds is not None
    ):
        interval_method += "; truncated to [0, recorded_states]"
    return ScalarAggregate(
        count=count,
        mean=mean,
        standard_deviation=standard_deviation,
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
        confidence_interval=interval,
        confidence_level=confidence_level,
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
    timing_method = record.timing.timing_method
    if record.spec.experiment_id == _INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID:
        for suffix in (
            "; JAX lower().compile() measured once for shared shapes",
            "; JAX executable reused from in-process shape cache",
        ):
            if timing_method == _INDEPENDENT_PASYM_SWAP_TIMING_METHOD_PREFIX + suffix:
                timing_method = _INDEPENDENT_PASYM_SWAP_TIMING_METHOD_PREFIX
                break
    deterministic_identity: tuple[str, ...] | str | None = None
    if record.spec.experiment_id == _INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID:
        deterministic_identity = _independent_pasym_swap_artifact_identity(record)
    elif record.spec.experiment_id == _TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
        deterministic_identity = _target_context_deterministic_identity(record)
        timing_method = _target_context_timing_method(record)
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
        _dtype_compatibility_signature(record),
        record.provenance.jax_enable_x64,
        record.timing.evidence_class,
        record.timing.unit,
        record.timing.source,
        timing_method,
        deterministic_identity,
    )


def _independent_pasym_swap_artifact_identity(record: RunRecord) -> tuple[str, ...]:
    """Extract the ordered compiled-artifact identity from one bounded record."""

    observation = record.metrics.get("independent_pasym_swap")
    value = observation.value if observation is not None else None
    if not isinstance(value, Mapping):
        raise ValueError(
            "independent PAsymSwap deterministic artifact identity requires a nested summary"
        )
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)) or not artifacts:
        raise ValueError(
            "independent PAsymSwap deterministic artifact identity requires compiled artifacts"
        )

    artifact_hashes: list[str] = []
    for artifact in artifacts:
        optimization = artifact.get("optimization") if isinstance(artifact, Mapping) else None
        artifact_hash = (
            optimization.get("artifact_hash") if isinstance(optimization, Mapping) else None
        )
        if not isinstance(artifact_hash, str) or not artifact_hash:
            raise ValueError(
                "independent PAsymSwap deterministic artifact identity requires artifact hashes"
            )
        artifact_hashes.append(artifact_hash)
    return tuple(artifact_hashes)


def _target_context_timing_method(record: RunRecord) -> str:
    for suffix in _TARGET_CONTEXT_PASYM_SWAP_TIMING_METHOD_SUFFIXES:
        if record.timing.timing_method == _TARGET_CONTEXT_PASYM_SWAP_TIMING_METHOD_PREFIX + suffix:
            return _TARGET_CONTEXT_PASYM_SWAP_TIMING_METHOD_PREFIX
    raise ValueError(
        "target-context timing method must use the synchronized sampling prefix and one "
        "checked JAX compile/reuse suffix"
    )


def _target_context_deterministic_identity(record: RunRecord) -> str:
    """Deeply validate one checked target record before exposing aggregate scalars."""

    if record.spec.experiment_id != _TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
        raise ValueError("record is not the checked target-context PAsymSwap experiment")
    if record.spec.sample_definition != TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION:
        raise ValueError("target-context sample definition differs from the checked value")
    if record.backend_id is not BackendId.THRML_LOCAL:
        raise ValueError("target-context records require the thrml_local backend")
    if record.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("target-context records require software_simulation evidence")
    if record.timing.synchronized is not True:
        raise ValueError("target-context records require synchronized timing")
    if record.timing.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("target-context timing requires software_simulation evidence")
    if record.timing.source != RUN_TIMING_SOURCE:
        raise ValueError("target-context timing source differs from the checked value")
    if record.timing.unit != "seconds":
        raise ValueError("target-context timing unit must be seconds")
    _target_context_timing_method(record)
    if not any(
        package.distribution == "thrml" and package.version == "0.1.4"
        for package in record.provenance.packages
    ):
        raise ValueError(
            "target-context runtime provenance requires the pinned THRML 0.1.4 package"
        )

    model = PAsymSwapModelConfig.model_validate(to_json_value(record.spec.model_parameters))
    run = TargetContextCompilerRunConfig.model_validate(to_json_value(record.spec.run_parameters))
    validate_target_context_pasym_swap_request(model, run, record.spec.seed)
    summary = validate_target_context_pasym_swap_observations(
        record.metrics, model, run, record.spec.seed
    )
    return summary.deterministic_result_hash


def _dtype_compatibility_signature(record: RunRecord) -> str:
    """Return the declared numeric representation without conflating exact and THRML paths."""

    if record.spec.experiment_id in {
        _INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID,
        _TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID,
    }:
        return (
            f"exact={record.spec.model_parameters.get('exact_dtype')}; "
            f"thrml={record.spec.model_parameters.get('thrml_dtype')}"
        )
    return str(record.spec.model_parameters.get("numeric_dtype"))


def _independent_pasym_swap_omission_reason(name: str) -> str | None:
    return _INDEPENDENT_PASYM_SWAP_OMITTED_METRIC_REASONS.get(
        name,
        "metric is not declared an independently seeded sampled cross-check",
    )


def _target_context_pasym_swap_omission_reason(name: str) -> str:
    return _TARGET_CONTEXT_PASYM_SWAP_OMITTED_METRIC_REASONS.get(
        name,
        "metric is not declared an independently seeded sampled cross-check",
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
        numeric_dtype=_dtype_compatibility_signature(record),
        git_commit=record.provenance.git_commit,
        git_dirty=record.provenance.git_dirty,
        packages=packages,
    )


def derive_aggregate_fields(
    records: Sequence[RunRecord],
    *,
    requested_seeds: tuple[int, ...],
    run_record_paths: tuple[str, ...],
    source_config: str,
    failures: tuple[RunFailure, ...] = (),
    failed_identity: tuple[str, BackendId, EvidenceClass, str, str] | None = None,
) -> AggregateDerivedFields:
    """Purely derive compatible aggregate claims from run records and request identity."""

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
    expected_paths = tuple(f"runs/seed-{record.spec.seed:010d}.json" for record in records)
    if run_record_paths != expected_paths:
        raise ValueError("aggregate run record path does not match persisted run seed")

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
            "timing evidence class",
            "timing unit",
            "timing source",
            "timing method",
            "deterministic artifact identity",
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
    statistical_semantics = _statistical_semantics_for_experiment(experiment_id)

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
            if (
                experiment_id == _TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID
                and name not in _TARGET_CONTEXT_PASYM_SWAP_SAMPLED_METRICS
            ):
                omitted_metrics[name] = _target_context_pasym_swap_omission_reason(name)
                continue
            omission_reason = _INDEPENDENT_PASYM_SWAP_OMITTED_METRIC_REASONS.get(name)
            if (
                experiment_id == _INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID
                and name not in _INDEPENDENT_PASYM_SWAP_SAMPLED_METRICS
                and omission_reason is not None
            ):
                omitted_metrics[name] = omission_reason
                continue
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
            ):
                omitted_metrics[name] = "non-scalar metric retained only in per-run records"
                continue
            if (
                experiment_id == _INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID
                and name not in _INDEPENDENT_PASYM_SWAP_SAMPLED_METRICS
            ):
                omitted_metrics[name] = _independent_pasym_swap_omission_reason(name)
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
                statistical_semantics=statistical_semantics,
                interval_bounds=interval_bounds,
            )
        if experiment_id == _INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID:
            omitted_metrics["timing.compile_seconds"] = (
                _INDEPENDENT_PASYM_SWAP_TIMING_OMISSION_REASON
            )
            omitted_metrics["timing.execution_seconds"] = (
                _INDEPENDENT_PASYM_SWAP_TIMING_OMISSION_REASON
            )
        elif experiment_id == _TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
            omitted_metrics["timing.compile_seconds"] = (
                _TARGET_CONTEXT_PASYM_SWAP_TIMING_OMISSION_REASON
            )
            omitted_metrics["timing.execution_seconds"] = (
                _TARGET_CONTEXT_PASYM_SWAP_TIMING_OMISSION_REASON
            )
        else:
            timing_method = records[0].timing.timing_method
            timing_evidence = records[0].timing.evidence_class
            timing_unit = records[0].timing.unit
            timing_source = records[0].timing.source
            metric_aggregates["timing.compile_seconds"] = _summarize_scalar(
                [record.timing.compile_seconds for record in records],
                unit=timing_unit,
                evidence_class=timing_evidence,
                statistical_semantics=statistical_semantics,
                method=(
                    f"{timing_method}; compilation interval only; excludes execution, "
                    "configuration loading, provenance collection, persistence, aggregation, "
                    f"and reporting; source={timing_source}"
                ),
            )
            metric_aggregates["timing.execution_seconds"] = _summarize_scalar(
                [record.timing.execution_seconds for record in records],
                unit=timing_unit,
                evidence_class=timing_evidence,
                statistical_semantics=statistical_semantics,
                method=(
                    f"{timing_method}; synchronized steady-state backend interval only; "
                    "excludes compilation, untimed warm launch, configuration loading, "
                    "provenance collection, persistence, aggregation, and reporting; "
                    f"source={timing_source}"
                ),
            )

    state = (
        CompletionState.COMPLETE
        if not failures
        else CompletionState.FAILED
        if not records
        else CompletionState.PARTIAL
    )
    return AggregateDerivedFields(
        experiment_id=experiment_id,
        statistical_semantics=statistical_semantics,
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
        metric_aggregates=FrozenDict(metric_aggregates),
        omitted_metrics=FrozenDict(omitted_metrics),
        completion_state=state,
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
    """Build an aggregate from the same pure derivation used for report validation."""

    derived = derive_aggregate_fields(
        records,
        requested_seeds=requested_seeds,
        run_record_paths=run_record_paths,
        source_config=source_config,
        failures=failures,
        failed_identity=failed_identity,
    )
    return AggregateRecord(
        experiment_id=derived.experiment_id,
        statistical_semantics=derived.statistical_semantics,
        backend_id=derived.backend_id,
        evidence_class=derived.evidence_class,
        model_hash=derived.model_hash,
        run_config_hash=derived.run_config_hash,
        source_config=derived.source_config,
        seeds=derived.seeds,
        requested_runs=derived.requested_runs,
        completed_runs=derived.completed_runs,
        failed_runs=derived.failed_runs,
        run_record_paths=derived.run_record_paths,
        failures=derived.failures,
        provenance_summary=derived.provenance_summary,
        metric_aggregates=derived.metric_aggregates,
        omitted_metrics=derived.omitted_metrics,
        completion_state=derived.completion_state,
    )


def validate_aggregate_against_records(
    aggregate: AggregateRecord,
    records: Sequence[RunRecord],
) -> None:
    """Re-derive and compare every aggregate claim that comes from persisted runs."""

    if len(records) != aggregate.completed_runs:
        raise ValueError("aggregate completed run count does not match persisted run records")
    if len(records) != len(aggregate.run_record_paths):
        raise ValueError("aggregate run record paths do not match persisted run records")

    failed_identity = (
        aggregate.experiment_id,
        aggregate.backend_id,
        aggregate.evidence_class,
        aggregate.model_hash,
        aggregate.run_config_hash,
    )
    derived = derive_aggregate_fields(
        records,
        requested_seeds=aggregate.seeds,
        run_record_paths=aggregate.run_record_paths,
        source_config=aggregate.source_config,
        failures=aggregate.failures,
        failed_identity=failed_identity,
    )
    compared_fields = {
        "experiment_id": "experiment id",
        "statistical_semantics": "statistical semantics",
        "backend_id": "backend",
        "evidence_class": "evidence class",
        "model_hash": "model hash",
        "run_config_hash": "run configuration hash",
        "seeds": "seeds",
        "requested_runs": "requested run count",
        "completed_runs": "completed run count",
        "failed_runs": "failed run count",
        "run_record_paths": "run record paths",
        "failures": "failure details",
        "provenance_summary": "provenance summary",
        "metric_aggregates": "metric aggregates",
        "omitted_metrics": "omitted metrics",
        "completion_state": "completion state",
    }
    for field_name, label in compared_fields.items():
        if getattr(aggregate, field_name) != getattr(derived, field_name):
            raise ValueError(
                f"aggregate {label} does not match values derived from persisted run records"
            )
