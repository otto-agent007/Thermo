"""Bounded persisted results and acceptance checks for PAsymSwap compilation.

The nested conditional tables are the source of truth.  This module deliberately
recomputes all diagnostics when a result is assembled or reloaded, so a stale
``passed`` flag or aggregate cannot make an invalid run publishable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.pasym_swap import PAPER_SOURCE, WORD_ORDER, GateOccurrence
from thermo_lab.records import FrozenModel, MetricObservation
from thermo_lab.schemas import (
    PARAMETER_ORDER,
    IndependentCompilerRunConfig,
    PAsymSwapModelConfig,
    validate_independent_pasym_swap_request,
)

_HORIZONS = (1, 2, 4, 8, 16, 30)
_CONTEXTS = range(4)
_OUTPUTS = range(4)
_CHECKS = (
    "conditional_validity",
    "optimizer_convergence",
    "median_equilibrium_tv",
    "worst_equilibrium_tv",
    "k30_equilibrium_residual",
    "k30_not_worse_than_k1",
    "empirical_k30_residual",
    "aggregate_agreement",
)

ConditionalTable = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def _freeze_mapping(values: Mapping[int, Any]) -> Mapping[int, Any]:
    return MappingProxyType(dict(values))


def _parse_horizon_mapping(value: object) -> object:
    """Restore integer horizon keys after a JSON metric-value round trip."""

    if not isinstance(value, Mapping):
        return value
    parsed: dict[object, Any] = {}
    for key, item in value.items():
        parsed[int(key) if isinstance(key, str) and key.isdecimal() else key] = item
    return parsed


def _finite(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _table(value: object, *, name: str, nonnegative: bool = True) -> ConditionalTable:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must contain exactly four input rows")
    rows: list[tuple[float, float, float, float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise ValueError(f"{name} context={row_index} must contain exactly four outputs")
        checked: list[float] = []
        for output_index, item in enumerate(row):
            if type(item) is not float or not math.isfinite(item):
                raise ValueError(
                    f"{name} context={row_index} output={output_index} must be finite floats"
                )
            if nonnegative and item < 0.0:
                raise ValueError(
                    f"{name} context={row_index} output={output_index} must be nonnegative"
                )
            checked.append(item)
        rows.append(tuple(checked))  # type: ignore[arg-type]
    return tuple(rows)  # type: ignore[return-value]


def _probability_table(value: object, *, name: str, tolerance: float) -> ConditionalTable:
    table = _table(value, name=name)
    for context, row in enumerate(table):
        total = sum(row)
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"{name} context={context} observed={total} bound={tolerance}: row must normalize"
            )
    return table


def _counts(
    value: object, *, name: str, chain_count: int
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must contain exactly four input rows")
    rows: list[tuple[int, int, int, int]] = []
    for context, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise ValueError(f"{name} context={context} must contain exactly four outputs")
        if any(type(item) is not int or item < 0 for item in row):
            raise ValueError(f"{name} context={context} must contain nonnegative integer counts")
        if sum(row) != chain_count:
            raise ValueError(
                f"{name} context={context} observed={sum(row)} bound={chain_count}: chain count"
            )
        rows.append(tuple(row))  # type: ignore[arg-type]
    return tuple(rows)  # type: ignore[return-value]


class SummaryStatistics(FrozenModel):
    """Explicit order-statistics used in reports and acceptance summaries."""

    minimum: StrictFloat
    median: StrictFloat
    p90: StrictFloat
    maximum: StrictFloat


def summarize_values(values: Sequence[float]) -> SummaryStatistics:
    """Return min, even-aware median, nearest-rank p90, and max.

    P90 is the value at one-based rank ``ceil(.90*n)``; it is intentionally
    not an interpolated library percentile.
    """

    if not values:
        raise ValueError("summary values must not be empty")
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("summary values must be finite floats")
    ordered = sorted(values)
    length = len(ordered)
    middle = length // 2
    median = ordered[middle] if length % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return SummaryStatistics(
        minimum=ordered[0],
        median=median,
        p90=ordered[math.ceil(0.90 * length) - 1],
        maximum=ordered[-1],
    )


class KernelConditionalResult(FrozenModel):
    """Exact and sampled conditionals for one frozen five-spin kernel."""

    target_conditional: ConditionalTable
    equilibrium_conditional: ConditionalTable
    finite_horizon_conditionals: Mapping[StrictInt, ConditionalTable]
    empirical_k30_counts: tuple[
        tuple[StrictInt, StrictInt, StrictInt, StrictInt],
        tuple[StrictInt, StrictInt, StrictInt, StrictInt],
        tuple[StrictInt, StrictInt, StrictInt, StrictInt],
        tuple[StrictInt, StrictInt, StrictInt, StrictInt],
    ]
    empirical_k30_conditional: ConditionalTable
    target_evidence_class: EvidenceClass = EvidenceClass.EXACT_REFERENCE
    equilibrium_evidence_class: EvidenceClass = EvidenceClass.EXACT_REFERENCE
    finite_horizon_evidence_class: EvidenceClass = EvidenceClass.EXACT_REFERENCE
    empirical_k30_evidence_class: EvidenceClass = EvidenceClass.SOFTWARE_SIMULATION

    @field_validator("finite_horizon_conditionals", mode="before")
    @classmethod
    def parse_horizon_keys(cls, value: object) -> object:
        return _parse_horizon_mapping(value)

    @model_validator(mode="after")
    def validate_bounded_tables(self) -> KernelConditionalResult:
        _table(self.target_conditional, name="target_conditional")
        _table(self.equilibrium_conditional, name="equilibrium_conditional")
        _table(self.empirical_k30_conditional, name="empirical_k30_conditional")
        _counts(
            self.empirical_k30_counts,
            name="empirical_k30_counts",
            chain_count=4096,
        )
        horizons = tuple(self.finite_horizon_conditionals)
        if set(horizons) != set(_HORIZONS) or len(horizons) != len(_HORIZONS):
            raise ValueError(f"finite_horizon_conditionals keys must be exactly {_HORIZONS}")
        normalized = {
            horizon: _table(table, name=f"finite_horizon_conditionals horizon={horizon}")
            for horizon, table in self.finite_horizon_conditionals.items()
        }
        object.__setattr__(self, "finite_horizon_conditionals", _freeze_mapping(normalized))
        if self.target_evidence_class is not EvidenceClass.EXACT_REFERENCE:
            raise ValueError("target conditional must use exact_reference evidence")
        if self.equilibrium_evidence_class is not EvidenceClass.EXACT_REFERENCE:
            raise ValueError("equilibrium conditional must use exact_reference evidence")
        if self.finite_horizon_evidence_class is not EvidenceClass.EXACT_REFERENCE:
            raise ValueError("finite-horizon conditional must use exact_reference evidence")
        if self.empirical_k30_evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
            raise ValueError("empirical K30 conditional must use software_simulation evidence")
        return self

    @field_serializer("finite_horizon_conditionals")
    def serialize_finite_horizons(self, value: Mapping[int, ConditionalTable]) -> dict[str, Any]:
        return {str(horizon): table for horizon, table in value.items()}


class KernelOptimizationResult(FrozenModel):
    """Bounded optimizer winner information; histories are intentionally absent."""

    artifact_hash: str = Field(min_length=1)
    parameters: tuple[
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
    ]
    selected_restart: StrictInt = Field(ge=0, le=2)
    successful_restart_count: StrictInt = Field(ge=0, le=3)
    objective: StrictFloat
    projected_gradient_norm: StrictFloat = Field(ge=0)
    cap_active_parameter_count: StrictInt = Field(ge=0, le=9)
    evidence_class: EvidenceClass = EvidenceClass.SOFTWARE_SIMULATION

    @model_validator(mode="after")
    def validate_evidence(self) -> KernelOptimizationResult:
        if self.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
            raise ValueError("optimizer results must use software_simulation evidence")
        return self


class CompiledKernelResult(FrozenModel):
    """One canonical target and the immutable compiled artifact that realizes it."""

    target_hash: str = Field(min_length=1)
    compiler_request_hash: str = Field(min_length=1)
    optimization: KernelOptimizationResult
    conditionals: KernelConditionalResult
    evidence_class: EvidenceClass = EvidenceClass.SOFTWARE_SIMULATION

    @model_validator(mode="after")
    def validate_composite_evidence(self) -> CompiledKernelResult:
        if self.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
            raise ValueError(
                "optimizer/sample-derived compiled kernel must use software_simulation evidence"
            )
        return self

    @property
    def artifact_hash(self) -> str:
        return self.optimization.artifact_hash


class PAsymSwapAcceptance(FrozenModel):
    """Persisted acceptance record, cross-checked rather than trusted."""

    passed: StrictBool
    checks: tuple[str, ...]
    evidence_class: EvidenceClass = EvidenceClass.SOFTWARE_SIMULATION

    @model_validator(mode="after")
    def validate_shape(self) -> PAsymSwapAcceptance:
        if self.checks != _CHECKS:
            raise ValueError("acceptance checks must list the eight canonical gates in order")
        if self.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
            raise ValueError("acceptance composite must use software_simulation evidence")
        return self


class IndependentPAsymSwapSummary(FrozenModel):
    """Complete bounded outcome for one sampled independent-compiler seed."""

    source_reference: str = Field(min_length=1)
    artifacts: tuple[CompiledKernelResult, ...]
    occurrences: tuple[GateOccurrence, ...]
    equilibrium_kl: SummaryStatistics
    equilibrium_tv: SummaryStatistics
    finite_horizon_kl: Mapping[StrictInt, SummaryStatistics]
    finite_horizon_tv: Mapping[StrictInt, SummaryStatistics]
    maximum_finite_horizon_equilibrium_residual: Mapping[StrictInt, StrictFloat]
    maximum_empirical_k30_residual: StrictFloat = Field(ge=0)
    successful_artifact_count: StrictInt = Field(ge=0)
    total_cap_active_parameter_count: StrictInt = Field(ge=0)
    acceptance: PAsymSwapAcceptance
    evidence_class: EvidenceClass = EvidenceClass.SOFTWARE_SIMULATION

    @field_validator(
        "finite_horizon_kl",
        "finite_horizon_tv",
        "maximum_finite_horizon_equilibrium_residual",
        mode="before",
    )
    @classmethod
    def parse_horizon_keys(cls, value: object) -> object:
        return _parse_horizon_mapping(value)

    @model_validator(mode="after")
    def validate_summary_shape(self) -> IndependentPAsymSwapSummary:
        if self.source_reference != PAPER_SOURCE:
            raise ValueError("PAsymSwap summary source must be the checked paper reference")
        if not self.artifacts:
            raise ValueError("PAsymSwap summary must contain at least one artifact")
        if len(self.occurrences) != 500:
            raise ValueError("PAsymSwap summary must contain exactly 500 occurrences")
        artifact_hashes = [artifact.artifact_hash for artifact in self.artifacts]
        if len(set(artifact_hashes)) != len(artifact_hashes):
            raise ValueError("PAsymSwap artifacts must have unique artifact hashes")
        target_hashes = {artifact.target_hash for artifact in self.artifacts}
        if any(occurrence.target_hash not in target_hashes for occurrence in self.occurrences):
            raise ValueError("every occurrence target hash must resolve to an included artifact")
        for name, mapping in (
            ("finite_horizon_kl", self.finite_horizon_kl),
            ("finite_horizon_tv", self.finite_horizon_tv),
            (
                "maximum_finite_horizon_equilibrium_residual",
                self.maximum_finite_horizon_equilibrium_residual,
            ),
        ):
            if set(mapping) != set(_HORIZONS) or len(mapping) != len(_HORIZONS):
                raise ValueError(f"{name} keys must be exactly {_HORIZONS}")
        object.__setattr__(self, "finite_horizon_kl", _freeze_mapping(self.finite_horizon_kl))
        object.__setattr__(self, "finite_horizon_tv", _freeze_mapping(self.finite_horizon_tv))
        object.__setattr__(
            self,
            "maximum_finite_horizon_equilibrium_residual",
            _freeze_mapping(self.maximum_finite_horizon_equilibrium_residual),
        )
        if self.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
            raise ValueError("compiled summary must use software_simulation evidence")
        return self

    @field_serializer(
        "finite_horizon_kl",
        "finite_horizon_tv",
        "maximum_finite_horizon_equilibrium_residual",
    )
    def serialize_horizon_mappings(self, value: Mapping[int, Any]) -> dict[str, Any]:
        return {str(horizon): item for horizon, item in value.items()}


def _tv(left: ConditionalTable, right: ConditionalTable, context: int) -> float:
    return 0.5 * sum(abs(left[context][output] - right[context][output]) for output in _OUTPUTS)


def _kl(target: ConditionalTable, observed: ConditionalTable, context: int) -> float:
    result = 0.0
    for output in _OUTPUTS:
        probability = target[context][output]
        if probability:
            result += probability * (math.log(probability) - math.log(observed[context][output]))
    return result


def _artifact_identity(
    artifact: CompiledKernelResult, model: PAsymSwapModelConfig, run: IndependentCompilerRunConfig
) -> dict[str, Any]:
    return {
        "target_hash": artifact.target_hash,
        "topology_id": model.topology_id,
        "logical_role_order": (*model.color_a_roles, *model.color_b_roles),
        "parameter_order": PARAMETER_ORDER,
        "dtype": model.exact_dtype,
        "parameters": artifact.optimization.parameters,
        "beta": model.beta,
        "parameter_cap": model.parameter_cap,
        "compiler_settings": {
            "parameter_cap": model.parameter_cap,
            "maxiter": run.maxiter,
            "maxls": run.maxls,
            "ftol": run.ftol,
            "gtol": run.gtol,
            "projected_gradient_tolerance": run.projected_gradient_tolerance,
            "initializations": tuple(tuple(values) for values in run.initializations),
            "context_weights": tuple(run.context_weights),
        },
    }


def _check_tables(artifact: CompiledKernelResult, run: IndependentCompilerRunConfig) -> None:
    conditionals = artifact.conditionals
    target = _probability_table(
        conditionals.target_conditional,
        name=f"target target_hash={artifact.target_hash}",
        tolerance=run.exact_normalization_tolerance,
    )
    expected_target_hash = canonical_sha256({"word_order": WORD_ORDER, "conditional": target})
    if artifact.target_hash != expected_target_hash:
        raise ValueError(
            "target hash mismatch "
            f"target_hash={artifact.target_hash} observed={expected_target_hash}"
        )
    _probability_table(
        conditionals.equilibrium_conditional,
        name=f"equilibrium target_hash={artifact.target_hash}",
        tolerance=run.exact_normalization_tolerance,
    )
    for horizon in _HORIZONS:
        _probability_table(
            conditionals.finite_horizon_conditionals[horizon],
            name=f"finite horizon target_hash={artifact.target_hash} horizon={horizon}",
            tolerance=run.exact_normalization_tolerance,
        )
    counts = _counts(
        conditionals.empirical_k30_counts,
        name=f"empirical counts target_hash={artifact.target_hash}",
        chain_count=run.chain_count_per_context,
    )
    empirical = _probability_table(
        conditionals.empirical_k30_conditional,
        name=f"empirical target_hash={artifact.target_hash} horizon=30",
        tolerance=run.exact_normalization_tolerance,
    )
    for context in _CONTEXTS:
        for output in _OUTPUTS:
            expected = counts[context][output] / run.chain_count_per_context
            if empirical[context][output] != expected:
                raise ValueError(
                    f"empirical target_hash={artifact.target_hash} context={context} horizon=30 "
                    f"observed={empirical[context][output]} bound={expected}: counts disagree"
                )


def _check_optimizer(
    artifact: CompiledKernelResult, model: PAsymSwapModelConfig, run: IndependentCompilerRunConfig
) -> None:
    optimization = artifact.optimization
    if optimization.successful_restart_count < 1:
        raise ValueError(f"optimizer target_hash={artifact.target_hash} observed=0 bound=1")
    if optimization.projected_gradient_norm > run.projected_gradient_tolerance:
        raise ValueError(
            f"optimizer target_hash={artifact.target_hash} "
            f"observed={optimization.projected_gradient_norm} "
            f"bound={run.projected_gradient_tolerance}"
        )
    if any(abs(value) > model.parameter_cap for value in optimization.parameters):
        maximum = max(abs(value) for value in optimization.parameters)
        raise ValueError(
            f"parameter cap target_hash={artifact.target_hash} "
            f"observed={maximum} bound={model.parameter_cap}"
        )
    expected_hash = canonical_sha256(_artifact_identity(artifact, model, run))
    if optimization.artifact_hash != expected_hash:
        raise ValueError(
            f"artifact hash mismatch target_hash={artifact.target_hash} "
            f"observed={optimization.artifact_hash} bound={expected_hash}"
        )


def _derived(artifacts: Sequence[CompiledKernelResult]) -> dict[str, Any]:
    equilibrium_kls: list[float] = []
    equilibrium_tvs: list[float] = []
    horizon_kls = {horizon: [] for horizon in _HORIZONS}
    horizon_tvs = {horizon: [] for horizon in _HORIZONS}
    maximum_residuals = {horizon: 0.0 for horizon in _HORIZONS}
    maximum_empirical = 0.0
    for artifact in artifacts:
        conditionals = artifact.conditionals
        target = conditionals.target_conditional
        equilibrium = conditionals.equilibrium_conditional
        for context in _CONTEXTS:
            equilibrium_kls.append(_kl(target, equilibrium, context) * 0.25)
            equilibrium_tvs.append(_tv(target, equilibrium, context) * 0.25)
            for horizon in _HORIZONS:
                finite = conditionals.finite_horizon_conditionals[horizon]
                horizon_kls[horizon].append(_kl(target, finite, context) * 0.25)
                horizon_tvs[horizon].append(_tv(target, finite, context) * 0.25)
                maximum_residuals[horizon] = max(
                    maximum_residuals[horizon], _tv(finite, equilibrium, context)
                )
            maximum_empirical = max(
                maximum_empirical,
                _tv(
                    conditionals.empirical_k30_conditional,
                    conditionals.finite_horizon_conditionals[30],
                    context,
                ),
            )
    # The values above are collected per context.  A canonical artifact's
    # uniform-weighted diagnostic is one value, not four pseudo-replicates.
    per_artifact_kl = []
    per_artifact_tv = []
    per_horizon_kl = {horizon: [] for horizon in _HORIZONS}
    per_horizon_tv = {horizon: [] for horizon in _HORIZONS}
    for artifact in artifacts:
        conditional = artifact.conditionals
        per_artifact_kl.append(
            sum(
                _kl(conditional.target_conditional, conditional.equilibrium_conditional, c)
                for c in _CONTEXTS
            )
            / 4.0
        )
        per_artifact_tv.append(
            sum(
                _tv(conditional.target_conditional, conditional.equilibrium_conditional, c)
                for c in _CONTEXTS
            )
            / 4.0
        )
        for horizon in _HORIZONS:
            finite = conditional.finite_horizon_conditionals[horizon]
            per_horizon_kl[horizon].append(
                sum(_kl(conditional.target_conditional, finite, c) for c in _CONTEXTS) / 4.0
            )
            per_horizon_tv[horizon].append(
                sum(_tv(conditional.target_conditional, finite, c) for c in _CONTEXTS) / 4.0
            )
    return {
        "equilibrium_kl": summarize_values(per_artifact_kl),
        "equilibrium_tv": summarize_values(per_artifact_tv),
        "finite_horizon_kl": _freeze_mapping(
            {horizon: summarize_values(values) for horizon, values in per_horizon_kl.items()}
        ),
        "finite_horizon_tv": _freeze_mapping(
            {horizon: summarize_values(values) for horizon, values in per_horizon_tv.items()}
        ),
        "maximum_finite_horizon_equilibrium_residual": _freeze_mapping(maximum_residuals),
        "maximum_empirical_k30_residual": maximum_empirical,
    }


def summarize_artifacts(
    artifacts: Sequence[CompiledKernelResult],
    occurrences: Sequence[GateOccurrence],
    model: PAsymSwapModelConfig,
    run: IndependentCompilerRunConfig,
) -> IndependentPAsymSwapSummary:
    """Recompute all eight acceptance gates from bounded nested values."""

    validate_independent_pasym_swap_request(model, run, seed=0)
    artifacts = tuple(artifacts)
    occurrences = tuple(occurrences)
    if not artifacts:
        raise ValueError("PAsymSwap summary requires at least one artifact")
    if len(occurrences) != 500:
        raise ValueError("PAsymSwap summary requires exactly 500 occurrences")
    if len({artifact.target_hash for artifact in artifacts}) != len(artifacts):
        raise ValueError("PAsymSwap artifacts must contain one artifact per target hash")
    target_hashes = {artifact.target_hash for artifact in artifacts}
    for occurrence in occurrences:
        if occurrence.target_hash not in target_hashes:
            raise ValueError(
                f"occurrence target_hash={occurrence.target_hash} does not resolve "
                "to an included artifact"
            )
    for artifact in artifacts:
        _check_tables(artifact, run)
        _check_optimizer(artifact, model, run)
    derived = _derived(artifacts)
    if derived["equilibrium_tv"].median > run.median_equilibrium_tv_tolerance:
        raise ValueError(
            f"median_equilibrium_tv observed={derived['equilibrium_tv'].median} "
            f"bound={run.median_equilibrium_tv_tolerance}"
        )
    if derived["equilibrium_tv"].maximum > run.worst_equilibrium_tv_tolerance:
        raise ValueError(
            f"worst_equilibrium_tv observed={derived['equilibrium_tv'].maximum} "
            f"bound={run.worst_equilibrium_tv_tolerance}"
        )
    for artifact in artifacts:
        for context in _CONTEXTS:
            conditionals = artifact.conditionals
            k30 = _tv(
                conditionals.finite_horizon_conditionals[30],
                conditionals.equilibrium_conditional,
                context,
            )
            k1 = _tv(
                conditionals.finite_horizon_conditionals[1],
                conditionals.equilibrium_conditional,
                context,
            )
            if k30 > run.k30_equilibrium_tv_tolerance:
                raise ValueError(
                    "K30 equilibrium residual "
                    f"target_hash={artifact.target_hash} context={context} "
                    f"horizon=30 observed={k30} bound={run.k30_equilibrium_tv_tolerance}"
                )
            if k30 > k1:
                raise ValueError(
                    f"K30 versus K1 target_hash={artifact.target_hash} context={context} "
                    "horizon=30 "
                    f"observed={k30} bound={k1}"
                )
            empirical = _tv(
                conditionals.empirical_k30_conditional,
                conditionals.finite_horizon_conditionals[30],
                context,
            )
            if empirical > run.thrml_k30_tv_tolerance:
                raise ValueError(
                    "empirical K30 residual "
                    f"target_hash={artifact.target_hash} context={context} horizon=30 "
                    f"observed={empirical} bound={run.thrml_k30_tv_tolerance}"
                )
    return IndependentPAsymSwapSummary(
        source_reference=model.source_reference,
        artifacts=artifacts,
        occurrences=occurrences,
        equilibrium_kl=derived["equilibrium_kl"],
        equilibrium_tv=derived["equilibrium_tv"],
        finite_horizon_kl=derived["finite_horizon_kl"],
        finite_horizon_tv=derived["finite_horizon_tv"],
        maximum_finite_horizon_equilibrium_residual=derived[
            "maximum_finite_horizon_equilibrium_residual"
        ],
        maximum_empirical_k30_residual=derived["maximum_empirical_k30_residual"],
        successful_artifact_count=sum(
            artifact.optimization.successful_restart_count >= 1 for artifact in artifacts
        ),
        total_cap_active_parameter_count=sum(
            artifact.optimization.cap_active_parameter_count for artifact in artifacts
        ),
        acceptance=PAsymSwapAcceptance(passed=True, checks=_CHECKS),
    )


_REQUIRED_METRICS = frozenset(
    {
        "independent_pasym_swap",
        "median_equilibrium_tv",
        "worst_equilibrium_tv",
        "maximum_k30_equilibrium_residual",
        "maximum_empirical_k30_residual",
        "successful_artifact_count",
        "total_cap_active_parameter_count",
        "acceptance_passed",
    }
)


def _require_metric(
    metrics: Mapping[str, MetricObservation],
    name: str,
    expected: float | int | bool,
    evidence: EvidenceClass,
) -> None:
    metric = metrics[name]
    if metric.evidence_class is not evidence:
        raise ValueError(f"metric {name!r} must use {evidence.value} evidence")
    if metric.value != expected:
        raise ValueError(
            f"{name} does not match recomputed nested artifacts: "
            f"observed={metric.value} bound={expected}"
        )


def validate_independent_pasym_swap_observations(
    metrics: Mapping[str, MetricObservation],
    model: PAsymSwapModelConfig,
    run: IndependentCompilerRunConfig,
    seed: int,
) -> IndependentPAsymSwapSummary:
    """Validate persisted metrics by reconstructing the complete summary.

    The persisted summary, acceptance booleans, and scalar metrics are all
    treated as claims to be checked against nested artifact values.
    """

    validate_independent_pasym_swap_request(model, run, seed)
    missing = sorted(_REQUIRED_METRICS.difference(metrics))
    if missing:
        raise ValueError(f"independent PAsymSwap record is missing required metrics: {missing}")
    summary_metric = metrics["independent_pasym_swap"]
    if summary_metric.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("independent_pasym_swap composite must use software_simulation evidence")
    if summary_metric.source != model.source_reference:
        raise ValueError("independent_pasym_swap source differs from persisted model")
    summary = IndependentPAsymSwapSummary.model_validate(to_json_value(summary_metric.value))
    regenerated = summarize_artifacts(summary.artifacts, summary.occurrences, model, run)
    if summary.model_dump(mode="json") != regenerated.model_dump(mode="json"):
        raise ValueError("persisted PAsymSwap summary disagrees with nested artifacts")
    _require_metric(
        metrics,
        "median_equilibrium_tv",
        regenerated.equilibrium_tv.median,
        EvidenceClass.EXACT_REFERENCE,
    )
    _require_metric(
        metrics,
        "worst_equilibrium_tv",
        regenerated.equilibrium_tv.maximum,
        EvidenceClass.EXACT_REFERENCE,
    )
    _require_metric(
        metrics,
        "maximum_k30_equilibrium_residual",
        regenerated.maximum_finite_horizon_equilibrium_residual[30],
        EvidenceClass.EXACT_REFERENCE,
    )
    _require_metric(
        metrics,
        "maximum_empirical_k30_residual",
        regenerated.maximum_empirical_k30_residual,
        EvidenceClass.SOFTWARE_SIMULATION,
    )
    _require_metric(
        metrics,
        "successful_artifact_count",
        regenerated.successful_artifact_count,
        EvidenceClass.SOFTWARE_SIMULATION,
    )
    _require_metric(
        metrics,
        "total_cap_active_parameter_count",
        regenerated.total_cap_active_parameter_count,
        EvidenceClass.SOFTWARE_SIMULATION,
    )
    _require_metric(metrics, "acceptance_passed", True, EvidenceClass.SOFTWARE_SIMULATION)
    return regenerated
