"""Persisted-only Markdown rendering for target-context PAsymSwap evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from thermo_lab.config import (
    TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID,
    TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.records import RUN_TIMING_SOURCE, MetricObservation, RunRecord
from thermo_lab.schemas import (
    PAsymSwapModelConfig,
    TargetContextCompilerRunConfig,
    validate_target_context_pasym_swap_request,
)
from thermo_lab.target_context_pasym_swap_results import (
    TargetContextPAsymSwapSummary,
    validate_target_context_pasym_swap_observations,
)

_HORIZON_KEYS = ("1", "2", "4", "8", "16", "30")


def _canonical_horizon_order(value: Any) -> Any:
    """Restore numeric horizon order in a validation-only JSON copy.

    Run records are written with sorted JSON keys, which places ``16`` before
    ``2``. Task 8's deep validator deliberately compares the declared horizon
    order, so persisted records need this semantic JSON normalization before
    they are handed back to it.
    """

    if isinstance(value, Mapping):
        keys = tuple(value)
        ordered_keys = _HORIZON_KEYS if set(keys) == set(_HORIZON_KEYS) else keys
        return {key: _canonical_horizon_order(value[key]) for key in ordered_keys}
    if isinstance(value, list):
        return [_canonical_horizon_order(item) for item in value]
    return value


def canonical_target_context_record(record: RunRecord) -> RunRecord:
    """Return a strictly parsed validation copy with canonical horizon mappings."""

    payload = _canonical_horizon_order(to_json_value(record))
    return RunRecord.model_validate(payload)


def validate_persisted_target_context_pasym_swap_record(
    record: RunRecord,
) -> tuple[
    TargetContextPAsymSwapSummary,
    PAsymSwapModelConfig,
    TargetContextCompilerRunConfig,
]:
    """Deeply validate one persisted target-context record without mutating it."""

    if record.spec.experiment_id != TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
        raise ValueError("record is not a target-context PAsymSwap experiment")
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
    if not any(
        package.distribution == "thrml" and package.version == "0.1.4"
        for package in record.provenance.packages
    ):
        raise ValueError(
            "target-context runtime provenance requires the pinned THRML 0.1.4 package"
        )

    checked = canonical_target_context_record(record)
    model = PAsymSwapModelConfig.model_validate(to_json_value(checked.spec.model_parameters))
    run = TargetContextCompilerRunConfig.model_validate(to_json_value(checked.spec.run_parameters))
    validate_target_context_pasym_swap_request(model, run, checked.spec.seed)
    metrics = {
        name: MetricObservation.model_validate(to_json_value(observation))
        for name, observation in checked.metrics.items()
    }
    summary = validate_target_context_pasym_swap_observations(
        metrics, model, run, checked.spec.seed
    )
    return summary, model, run


def _statistics_text(statistics: Any, format_number) -> str:
    return " / ".join(
        format_number(value)
        for value in (
            statistics.minimum,
            statistics.median,
            statistics.p90,
            statistics.maximum,
        )
    )


def _optimizer_phase_text(phase: Any, format_number) -> str:
    if phase.cache_reused:
        return (
            "reused; no optimizer work in this seed (the persisted zero is a cache sentinel, "
            "not a zero-second benchmark)"
        )
    return f"executed in {format_number(phase.seconds)} seconds"


def render_target_context_pasym_swap_section(record: RunRecord) -> list[str]:
    """Render a target-context section only after complete persisted validation."""

    # This call must remain first: no report text is constructed from unvalidated
    # persisted values, and validation works on a fresh JSON copy.
    summary, model, run = validate_persisted_target_context_pasym_swap_record(record)

    from thermo_lab.reporting import _format_number, _markdown_code_span, _markdown_text

    schedule = summary.schedule_metrics
    degradation = summary.all_context_degradation
    zero_support = summary.zero_support_assessment
    multiplicities = Counter(profile.multiplicity for profile in summary.profiles)
    lines = [
        "## Exact target-context PAsymSwap paired evidence",
        "",
        (
            "The target-context artifacts improve occurrence-weighted target-to-model KL by "
            f"{_format_number(schedule.occurrence_weighted_equilibrium_kl_improvement)} nats "
            "under the exact target input distribution. This conclusion is limited to that "
            "frozen distribution and the paired software-derived kernels."
        ),
        "",
        "### Occurrence-weighted paired KL and TV",
        "",
        "| Metric | Paired uniform baseline | Target-context artifact | Improvement |",
        "|---|---:|---:|---:|",
        (
            "| Target-to-equilibrium KL (nats) | "
            f"{_format_number(schedule.baseline_occurrence_weighted_equilibrium_kl)} | "
            f"{_format_number(schedule.target_context_occurrence_weighted_equilibrium_kl)} | "
            f"{_format_number(schedule.occurrence_weighted_equilibrium_kl_improvement)} |"
        ),
        (
            "| Target-to-equilibrium TV | "
            f"{_format_number(schedule.baseline_occurrence_weighted_equilibrium_tv)} | "
            f"{_format_number(schedule.target_context_occurrence_weighted_equilibrium_tv)} | "
            "descriptive paired difference |"
        ),
        "",
        "| Target/profile | Multiplicity | Baseline KL | Target-context KL | KL improvement | "
        "Baseline TV | Target-context TV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in summary.pairs:
        metric = pair.metrics
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_code_span(pair.target_hash),
                    str(metric.multiplicity),
                    _format_number(metric.baseline_target_weighted_equilibrium_kl),
                    _format_number(metric.target_context_target_weighted_equilibrium_kl),
                    _format_number(metric.target_weighted_equilibrium_kl_improvement),
                    _format_number(metric.baseline_target_weighted_equilibrium_tv),
                    _format_number(metric.target_context_target_weighted_equilibrium_tv),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "### All-context degradation (non-gating)",
            "",
            (
                "Uniform all-context and positive-support diagnostics are exact evaluations of "
                "the paired frozen artifacts. They are deliberately non-gating for the checked "
                "target-context objective."
            ),
            "",
            "| Diagnostic | Minimum / median / p90 / maximum |",
            "|---|---:|",
            "| Baseline uniform-weighted equilibrium KL | "
            + _statistics_text(degradation.baseline_uniform_weighted_equilibrium_kl, _format_number)
            + " |",
            "| Baseline uniform-weighted equilibrium TV | "
            + _statistics_text(degradation.baseline_uniform_weighted_equilibrium_tv, _format_number)
            + " |",
            "| Target-context uniform-weighted equilibrium KL | "
            + _statistics_text(
                degradation.target_context_uniform_weighted_equilibrium_kl, _format_number
            )
            + " |",
            "| Target-context uniform-weighted equilibrium TV | "
            + _statistics_text(
                degradation.target_context_uniform_weighted_equilibrium_tv, _format_number
            )
            + " |",
            "| All-row TV | " + _statistics_text(degradation.all_row_tv, _format_number) + " |",
            "| Positive-support-row TV | "
            + _statistics_text(degradation.positive_support_row_tv, _format_number)
            + " |",
            "",
            (
                "Reference-level breaches: target artifacts above TV 0.15/0.35 = "
                f"{degradation.target_context_artifact_count_above_reference_tv_015}/"
                f"{degradation.target_context_artifact_count_above_reference_tv_035}; "
                "all rows above 0.15/0.35 = "
                f"{degradation.all_row_count_above_reference_tv_015}/"
                f"{degradation.all_row_count_above_reference_tv_035}; positive-support rows = "
                f"{degradation.positive_support_row_count_above_reference_tv_015}/"
                f"{degradation.positive_support_row_count_above_reference_tv_035}."
            ),
            "",
            "### Zero-support degradation (non-gating)",
            "",
            (
                "Input word 11 has exact zero target-context weight in every pooled profile. "
                "It therefore has no contribution to the target-weighted objective, and its "
                "target-accuracy degradation is non-gating. Its exact mixing and sampling "
                "gates remain required."
            ),
            "",
            "| Diagnostic | Minimum / median / p90 / maximum |",
            "|---|---:|",
            "| Equilibrium KL | "
            + _statistics_text(zero_support.equilibrium_kl, _format_number)
            + " |",
            "| Equilibrium TV | "
            + _statistics_text(zero_support.equilibrium_tv, _format_number)
            + " |",
        )
    )
    for horizon in run.horizons:
        lines.append(
            f"| K={horizon} finite-horizon TV | "
            + _statistics_text(zero_support.finite_horizon_tv[horizon], _format_number)
            + " |"
        )
    lines.extend(
        (
            "",
            "### Source and conventions",
            "",
            "| Item | Paper/source value | Checked Thermo convention |",
            "|---|---|---|",
            (
                "| Primary source | [arXiv:2608.01615v2]"
                "(https://arxiv.org/abs/2608.01615v2) | Persisted source identity is "
                f"{_markdown_code_span(summary.source_reference)}. |"
            ),
            (
                "| Program fixture | 5×5 periodic torus; 10 macrosteps; 500 atomic "
                "occurrences | Six ordered colors "
                f"{_markdown_code_span(', '.join(model.color_order))}; 37 shared target "
                "identities. |"
            ),
            (
                "| Word and storage order | Inputs and outputs (00, 01, 10, 11) | "
                f"{_markdown_code_span(model.matrix_storage)} and "
                f"{_markdown_code_span(model.bit_to_spin)}. |"
            ),
            (
                "| Five-spin model | Atomic PAsymSwap context | Synthetic "
                f"{_markdown_code_span(model.topology_id)}; exact=float64, THRML=float32; "
                f"parameter cap ±{_format_number(model.parameter_cap)}. |"
            ),
            (
                "| Sampling | K=30 complete two-color sweeps | "
                f"{run.chain_count_per_context} chains per input context, one sample per chain, "
                f"reset={_markdown_code_span(run.reset_distribution)}. |"
            ),
            "",
            "### Initial state and context policies",
            "",
            (
                f"- Initial state: {_markdown_code_span(summary.initial_state.initial_state)} "
                "with one particle at "
                f"{_markdown_code_span(str(summary.initial_state.initial_particle_site))}."
            ),
            "- Occupancy order: "
            f"{_markdown_code_span(run.initial_occupancy_order)}; persisted vector has "
            "one followed by 24 zeros.",
            f"- Context source: {_markdown_code_span(summary.context_source)}.",
            f"- Pooling: {_markdown_code_span(summary.context_reduction)}.",
            f"- Zero-support policy: {_markdown_code_span(summary.zero_support_policy)}.",
            f"- Warm-start policy: {_markdown_code_span(summary.warm_start_policy)}.",
            "",
            "### Deterministic trace and profile identities",
            "",
            (
                "The persisted deterministic identity contains 500 ordered occurrences and "
                "37 pooled profiles. Their checked multiplicities are 26 × 10, 9 × 20, and "
                "2 × 30."
            ),
            f"- Trace hash: {_markdown_code_span(summary.trace_hash)}.",
            "- Deterministic result hash: "
            f"{_markdown_code_span(summary.deterministic_result_hash)}.",
            "- Observed multiplicities: "
            + ", ".join(f"{multiplicities[count]} × {count}" for count in sorted(multiplicities))
            + ".",
            "",
            "| Target hash | Profile hash | Multiplicity | Context weights 00/01/10/11 |",
            "|---|---|---:|---|",
        )
    )
    lines.extend(
        "| "
        + " | ".join(
            (
                _markdown_code_span(profile.target_hash),
                _markdown_code_span(profile.profile_hash),
                str(profile.multiplicity),
                "/".join(_format_number(value) for value in profile.context_weights),
            )
        )
        + " |"
        for profile in summary.profiles
    )
    lines.extend(
        (
            "",
            "### Optimizer starts, convergence, winners, and cap activity",
            "",
            (
                f"- Baseline optimizer phase: "
                f"{_optimizer_phase_text(summary.baseline_optimizer_phase, _format_number)}."
            ),
            (
                f"- Target-context optimizer phase: "
                f"{_optimizer_phase_text(summary.target_context_optimizer_phase, _format_number)}."
            ),
            f"- Optimizer: {_markdown_code_span(run.optimizer)}; maxiter={run.maxiter}; "
            f"maxls={run.maxls}; winner={_markdown_code_span(run.restart_selection)}.",
            "",
            "| Target | Start | Role | SciPy success | Converged/passed | Selected | "
            "Iterations | Objective | Projected gradient | Cap-active | Termination |",
            "|---|---:|---|---|---|---|---:|---:|---:|---:|---|",
        )
    )
    for pair in summary.pairs:
        optimization = pair.target_context.optimization
        for attempt in optimization.attempts:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_code_span(pair.target_hash),
                        str(attempt.start_index),
                        _markdown_code_span(attempt.start_role),
                        "yes" if attempt.scipy_success else "no",
                        "yes" if attempt.passed_checks else "no",
                        "yes" if attempt.start_index == optimization.selected_start_index else "no",
                        str(attempt.iterations),
                        _format_number(attempt.objective),
                        _format_number(attempt.projected_gradient_norm),
                        str(attempt.cap_active_parameter_count),
                        _markdown_text(attempt.termination),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "### Exact paired finite-horizon mixing",
            "",
            (
                "Each value is the maximum finite-horizon-to-equilibrium TV over 37 paired "
                "artifacts and all four input contexts. Horizons are complete two-color sweeps."
            ),
            "",
            "| K | Paired uniform baseline | Target-context artifact |",
            "|---:|---:|---:|",
        )
    )
    for horizon in run.horizons:
        baseline_residual = max(
            value
            for pair in summary.pairs
            for value in pair.baseline.exact.finite_horizon_to_equilibrium_tv[horizon]
        )
        target_residual = max(
            value
            for pair in summary.pairs
            for value in pair.target_context.exact.finite_horizon_to_equilibrium_tv[horizon]
        )
        lines.append(
            f"| {horizon} | {_format_number(baseline_residual)} | "
            f"{_format_number(target_residual)} |"
        )
    lines.extend(
        (
            "",
            f"### Selected seed {record.spec.seed}: target-only THRML counts and "
            "empirical residuals",
            "",
            (
                "Counts contain 4,096 target-context samples per input row at K=30. "
                "Paired baselines were not sampled."
            ),
            "",
            "| Target | Input | Counts 00/01/10/11 | Empirical-to-exact K30 TV |",
            "|---|---:|---|---:|",
        )
    )
    for pair in summary.pairs:
        sampled = pair.target_context.sampled_k30
        for input_index, (counts, residual) in enumerate(
            zip(sampled.counts, sampled.empirical_to_exact_k30_tv, strict=True)
        ):
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_code_span(pair.target_hash),
                        str(input_index),
                        "/".join(str(count) for count in counts),
                        _format_number(residual),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "### Cache and synchronized timing semantics",
            "",
            (
                "SciPy optimizer intervals include only the paired kernel optimization work "
                "and are measured separately from JAX. A reused phase means no optimizer work "
                "occurred in this seed; its zero sentinel is not a benchmark."
            ),
            (
                "JAX compilation includes only `lower().compile()` for the shared shapes. The "
                "untimed synchronized warm launch is excluded from both compilation and "
                "steady-state execution."
            ),
            (
                "The steady-state interval includes the 148 target-context executions and final "
                "synchronization. It excludes conversion of returned samples into persisted "
                "integer counts."
            ),
            (
                "All reported optimizer and JAX intervals exclude fixture and context derivation, "
                "configuration loading, provenance collection, persistence/I/O, aggregation, and "
                "report rendering."
            ),
            (
                f"- JAX lowering/compilation: {_format_number(record.timing.compile_seconds)} "
                f"{record.timing.unit}; synchronized execution: "
                f"{_format_number(record.timing.execution_seconds)} {record.timing.unit}."
            ),
            "- Timing evidence/source: "
            f"{_markdown_code_span(record.timing.evidence_class.value)} / "
            f"{_markdown_code_span(record.timing.source)}; synchronized=yes.",
            f"- Persisted timing method: {_markdown_text(record.timing.timing_method)}.",
            "",
            f"### Selected seed {record.spec.seed}: acceptance layers",
            "",
            "- Deterministic acceptance: "
            f"{'passed' if summary.deterministic_acceptance.passed else 'failed'}.",
            "- Sampled fidelity acceptance: "
            f"{'passed' if summary.sampled_fidelity.passed else 'failed'}; maximum residual="
            f"{_format_number(summary.sampled_fidelity.maximum_empirical_k30_residual)}, "
            f"tolerance={_format_number(summary.sampled_fidelity.checked_tolerance)}.",
            f"- Seed acceptance: {'passed' if summary.seed_acceptance.passed else 'failed'}.",
        )
    )
    return lines
