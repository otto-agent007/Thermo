"""Markdown reports rendered only from persisted machine-readable records."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from thermo_lab.aggregate import (
    AggregateRecord,
    CompletionState,
    StatisticalSemantics,
    validate_aggregate_against_records,
)
from thermo_lab.graph_walk_results import (
    WeightedGraphWalkSummary,
    validate_weighted_graph_walk_observations,
)
from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap_reporting import (
    render_independent_pasym_swap_section,
    validate_persisted_independent_pasym_swap_record,
)
from thermo_lab.persistence import atomic_write_text
from thermo_lab.records import RunRecord
from thermo_lab.schemas import (
    WEIGHTED_GRAPH_WALK_EXPERIMENT_ID,
    WeightedGraphModelConfig,
    WeightedGraphRunConfig,
)
from thermo_lab.target_context_pasym_swap_reporting import (
    canonical_target_context_record,
    render_target_context_pasym_swap_section,
    validate_persisted_target_context_pasym_swap_record,
)

_INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID = "thrml.independent_pasym_swap_compilation.v1"
_TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID = "thrml.target_context_pasym_swap_compilation.v1"
_LEGACY_SAMPLE_DEFINITIONS = frozenset(
    (
        "Exact final probability mass over basis states [00, 01, 10, 11]; not a Monte Carlo "
        "sample.",
        "One deterministic family of complete Torx state-vector trajectories over declared "
        "Trotter resolutions and edge orders; variants and program depths are not independent "
        "samples or replications.",
        "One independently seeded THRML cross-check using 4,096 chains per input context over "
        "every frozen compiled kernel at 30 complete two-color Gibbs sweeps.",
        "One independently seeded THRML cross-check using 4,096 chains per input context over "
        "every frozen target-context kernel at 30 complete two-color Gibbs sweeps.",
        "One recorded full five-spin state after two complete ordered block-Gibbs sweeps; "
        "recorded-state count is not an effective-independent-sample count.",
    )
)
_ORDINARY_TABLE_UNIT = re.compile(r"[A-Za-z][A-Za-z0-9_./ \-]*\Z")


def _format_number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.8g}"


def _normalize_markdown_value(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " / ")


def _markdown_numeric_code(value: str) -> str:
    normalized = _normalize_markdown_value(value)
    visible_characters: list[str] = []
    for index, character in enumerate(normalized):
        codepoint = ord(character)
        category = unicodedata.category(character)
        is_noncharacter = 0xFDD0 <= codepoint <= 0xFDEF or codepoint & 0xFFFF in (
            0xFFFE,
            0xFFFF,
        )
        is_collapsible_space = character == " " and (
            index == 0
            or index == len(normalized) - 1
            or (index > 0 and normalized[index - 1] == " ")
        )
        if is_noncharacter:
            label = "NONCHARACTER"
        elif category.startswith("C"):
            label = {
                0x0000: "NULL",
                0x0009: "CHARACTER TABULATION",
            }.get(codepoint, unicodedata.name(character, "CONTROL"))
        elif is_collapsible_space:
            label = "SPACE"
        else:
            visible_characters.append(character)
            continue
        visible_characters.append(f"[U+{codepoint:04X} {label}]")
    payload = "".join(f"&#{ord(character)};" for character in "".join(visible_characters))
    return f"<code>{payload}</code>"


def _markdown_text(value: str) -> str:
    """Keep persisted text within one Markdown block and escape active syntax."""

    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    escaped_lines = []
    for line in normalized.split("\n"):
        escaped = line.replace("\\", "\\\\")
        for character in ("`", "*", "_", "~", "[", "]", "<", ">", "|", "&"):
            escaped = escaped.replace(character, f"\\{character}")
        escaped_lines.append(re.sub(r"^([ ]{0,3})([#\-+])(?=\s)", r"\1\\\2", escaped))
    return " / ".join(escaped_lines)


def _markdown_block_text(value: str) -> str:
    """Render persisted text as one inert standalone Markdown block."""

    if value in _LEGACY_SAMPLE_DEFINITIONS:
        return _markdown_text(value)
    return _markdown_numeric_code(value)


def _markdown_code_span(value: str) -> str:
    """Render arbitrary persisted text as a single safe CommonMark code span."""

    normalized = _normalize_markdown_value(value)
    longest_run = current_run = 0
    for character in normalized:
        if character == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    delimiter = "`" * (longest_run + 1)
    padding = " " if normalized.startswith("`") or normalized.endswith("`") else ""
    return f"{delimiter}{padding}{normalized}{padding}{delimiter}"


def _markdown_table_code_span(value: str) -> str:
    """Render a safe code span inside a GFM table cell."""

    if value == value.strip() and "  " not in value and _ORDINARY_TABLE_UNIT.fullmatch(value):
        return _markdown_code_span(value)
    return _markdown_numeric_code(value)


def _metric_table(aggregate: AggregateRecord) -> list[str]:
    interval_heading = (
        "Confidence interval"
        if aggregate.statistical_semantics is StatisticalSemantics.DETERMINISTIC_IDENTITY
        else "95% interval"
    )
    lines = [
        "| Metric | Evidence | Unit | Count | Mean | Std. dev. | Median | Min | Max | "
        f"{interval_heading} | Method/source |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, metric in aggregate.metric_aggregates.items():
        if metric.confidence_interval is None:
            interval = (
                "unavailable ("
                f"{_markdown_text(metric.interval_unavailable_reason or 'unspecified')})"
            )
        else:
            interval = (
                f"[{_format_number(metric.confidence_interval.lower)}, "
                f"{_format_number(metric.confidence_interval.upper)}]"
            )
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_text(name),
                    _markdown_code_span(metric.evidence_class.value)
                    if metric.evidence_class is not None
                    else "unavailable",
                    _markdown_table_code_span(metric.unit)
                    if metric.unit is not None
                    else "dimensionless",
                    str(metric.count),
                    _format_number(metric.mean),
                    _format_number(metric.standard_deviation),
                    _format_number(metric.median),
                    _format_number(metric.minimum),
                    _format_number(metric.maximum),
                    interval,
                    _markdown_text(metric.method),
                )
            )
            + " |"
        )
    return lines


def _run_set_description(aggregate: AggregateRecord) -> str:
    if aggregate.statistical_semantics is StatisticalSemantics.DETERMINISTIC_IDENTITY:
        return f"{aggregate.requested_runs} deterministic execution"
    return f"{aggregate.requested_runs} independent seeded runs"


def _validated_weighted_graph_walk_data(
    record: RunRecord,
) -> tuple[WeightedGraphWalkSummary, WeightedGraphModelConfig]:
    """Validate persisted graph observations against their checked requested inputs."""

    if record.spec.experiment_id != WEIGHTED_GRAPH_WALK_EXPERIMENT_ID:
        raise ValueError("weighted graph-walk summary belongs to a different experiment")
    model = WeightedGraphModelConfig.model_validate(to_json_value(record.spec.model_parameters))
    run = WeightedGraphRunConfig.model_validate(to_json_value(record.spec.run_parameters))
    summary = validate_weighted_graph_walk_observations(
        record.metrics,
        model,
        run,
        seed=record.spec.seed,
    )
    return summary, model


def _weighted_graph_walk_section(record: RunRecord) -> list[str]:
    """Render graph convergence details from the persisted run record only."""

    summary, model = _validated_weighted_graph_walk_data(record)

    canonical_order = ", ".join(
        f"{source}-{target}" for source, target in model.canonical_edge_order
    )
    variants = sorted(
        summary.variants,
        key=lambda item: (item.resolution, 0 if item.order == "canonical" else 1),
    )
    lines = [
        "## Weighted graph-walk convergence",
        "",
        f"- Primary source: <{summary.source_reference}>",
        f"- Canonical edge order: {_markdown_code_span(canonical_order)}",
        "- Resolution/order variants are not independent replications.",
        "",
        "### Source graph fixture",
        "",
        "| Edge | Weight |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {_markdown_text(edge.source)}–{_markdown_text(edge.target)} | {edge.weight:.2f} |"
        for edge in model.edges
    )
    lines.extend(
        (
            "",
            "### Exact final occupancy",
            "",
            "| Node | Exact final occupancy |",
            "|---|---:|",
        )
    )
    lines.extend(
        f"| {_markdown_text(node)} | {_format_number(occupancy)} |"
        for node, occupancy in zip(summary.node_labels, summary.exact_final_occupancy, strict=True)
    )
    lines.extend(
        (
            "",
            "### Resolution variants",
            "",
            "| N | Order | Final half-L1 | Max trajectory half-L1 | Final max abs. | "
            "Max leakage | Max normalization error | Min probability |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    lines.extend(
        "| "
        + " | ".join(
            (
                str(variant.resolution),
                f"`{variant.order}`",
                _format_number(variant.final_half_l1),
                _format_number(variant.max_trajectory_half_l1),
                _format_number(variant.final_max_abs_error),
                _format_number(variant.max_one_particle_leakage),
                _format_number(variant.max_normalization_error),
                _format_number(variant.minimum_state_probability),
            )
        )
        + " |"
        for variant in variants
    )
    lines.extend(
        (
            "",
            "### Edge-order sensitivity",
            "",
            "| N | Final half-L1 | Max trajectory half-L1 |",
            "|---:|---:|---:|",
        )
    )
    lines.extend(
        "| "
        + " | ".join(
            (
                str(sensitivity.resolution),
                _format_number(sensitivity.final_half_l1),
                _format_number(sensitivity.max_trajectory_half_l1),
            )
        )
        + " |"
        for sensitivity in sorted(summary.order_sensitivity, key=lambda item: item.resolution)
    )
    lines.extend(
        (
            "",
            "### Acceptance checks",
            "",
            f"- Passed: `{'yes' if summary.acceptance.passed else 'no'}`",
        )
    )
    lines.extend(f"- {_markdown_text(check)}" for check in summary.acceptance.checks)
    lines.extend(
        (
            "",
            "### Checkpoint occupancy",
            "",
            "| N | Order | Time | "
            + " | ".join(_markdown_text(label) for label in summary.node_labels)
            + " |",
            "|---:|---|---:|" + "---:|" * len(summary.node_labels),
        )
    )
    for variant in variants:
        for time, occupancy in zip(
            summary.checkpoint_times, variant.checkpoint_occupancies, strict=True
        ):
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(variant.resolution),
                        f"`{variant.order}`",
                        _format_number(time),
                        *(_format_number(value) for value in occupancy),
                    )
                )
                + " |"
            )
    return lines


def render_report(aggregate: AggregateRecord, records: tuple[RunRecord, ...]) -> str:
    """Render an evidence-safe report from already persisted validated data."""

    is_weighted_graph_walk = aggregate.experiment_id == WEIGHTED_GRAPH_WALK_EXPERIMENT_ID
    is_independent_pasym_swap = aggregate.experiment_id == _INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID
    is_target_context_pasym_swap = (
        aggregate.experiment_id == _TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID
    )
    is_deterministic = (
        aggregate.statistical_semantics is StatisticalSemantics.DETERMINISTIC_IDENTITY
    )
    target_summaries = []
    records_for_validation = records
    if is_independent_pasym_swap:
        # Aggregate scalar rederivation deliberately omits deterministic nested
        # identities, so validate every persisted successful run before any
        # report text is returned to the atomic writer.
        expected_artifact_identity: tuple[str, ...] | None = None
        for record in records:
            summary, _, _ = validate_persisted_independent_pasym_swap_record(record)
            artifact_identity = tuple(artifact.artifact_hash for artifact in summary.artifacts)
            if expected_artifact_identity is None:
                expected_artifact_identity = artifact_identity
            elif artifact_identity != expected_artifact_identity:
                raise ValueError(
                    "Cannot report incompatible independent PAsymSwap deterministic artifact "
                    "identity across seeds"
                )
    elif is_target_context_pasym_swap:
        # Persistence sorts keys lexicographically. Restore only the six
        # known finite-horizon mappings on strict validation copies before
        # invoking Task 8's order-sensitive deep validator.
        canonical_records = tuple(canonical_target_context_record(record) for record in records)
        expected_result_hash: str | None = None
        for record in canonical_records:
            summary, _, _ = validate_persisted_target_context_pasym_swap_record(record)
            target_summaries.append(summary)
            if expected_result_hash is None:
                expected_result_hash = summary.deterministic_result_hash
            elif summary.deterministic_result_hash != expected_result_hash:
                raise ValueError(
                    "Cannot report incompatible target-context deterministic result hash "
                    "across seeds"
                )
        records_for_validation = canonical_records
    validate_aggregate_against_records(aggregate, records_for_validation)
    sample_definition = (
        records[0].spec.sample_definition
        if records
        else (
            "Unavailable because no deterministic execution completed."
            if is_deterministic
            else "Unavailable because no seeded execution completed."
        )
    )
    seed_list = ", ".join(str(seed) for seed in aggregate.seeds)
    lines = [
        "# Thermo experiment report",
        "",
        f"- Experiment: {_markdown_code_span(aggregate.experiment_id)}",
        f"- Backend: {_markdown_code_span(aggregate.backend_id.value)}",
        f"- Evidence class: {_markdown_code_span(aggregate.evidence_class.value)}",
        f"- Statistical semantics: {_markdown_code_span(aggregate.statistical_semantics.value)}",
        f"- Completion state: {_markdown_code_span(aggregate.completion_state.value)}",
        f"- Model hash: {_markdown_code_span(aggregate.model_hash)}",
        f"- Non-seed configuration hash: {_markdown_code_span(aggregate.run_config_hash)}",
        f"- Seeds: {seed_list} ({_run_set_description(aggregate)})",
        f"- Completed/failed: {aggregate.completed_runs}/{aggregate.failed_runs}",
        "",
        "## Sample definition",
        "",
        _markdown_block_text(sample_definition),
        "",
        (
            "This record contains deterministic complete-distribution variants. Resolution, "
            "edge order, program depth, and node coordinates are not replication units, and "
            "no confidence interval is inferred from them."
            if is_deterministic
            else (
                "Seeds vary only the sampled cross-check and timing; targets, compiler, "
                "compiled artifacts, and exact horizons are deterministic identity fields and "
                "do not receive Student-t intervals."
                if is_independent_pasym_swap
                else (
                    "Only empirical THRML evidence, sampled acceptance, cache state, and timing "
                    "vary by seed; the target trace, profiles, paired artifacts, optimizer "
                    "observations, and exact evaluations are deterministic identity fields."
                    if is_target_context_pasym_swap
                    else "Recorded Markov-chain states are not described as independent samples. "
                    "Independent seeded runs are the replication unit for confidence intervals."
                )
            )
        ),
        "",
        "## Runtime provenance",
        "",
    ]
    if aggregate.provenance_summary is None:
        lines.append("Unavailable because no run completed.")
    else:
        provenance = aggregate.provenance_summary
        lines.extend(
            (
                f"- Python: {_markdown_code_span(provenance.python_version)}",
                f"- Platform: {_markdown_code_span(provenance.platform)}",
                "- JAX/JAXLIB: "
                f"{_markdown_code_span(provenance.jax_version)} / "
                f"{_markdown_code_span(provenance.jaxlib_version)}",
                f"- JAX backend: {_markdown_code_span(provenance.jax_backend)}",
                "- JAX devices: "
                + ", ".join(_markdown_code_span(item) for item in provenance.jax_devices),
                "- JAX x64 enabled: " + _markdown_code_span(str(provenance.jax_enable_x64).lower()),
                f"- Numeric dtype: {_markdown_code_span(provenance.numeric_dtype)}",
                "- Git commit: " + _markdown_code_span(provenance.git_commit or "unavailable"),
                "- Git dirty: "
                + _markdown_code_span(
                    str(provenance.git_dirty).lower()
                    if provenance.git_dirty is not None
                    else "unavailable"
                ),
                "- Packages: "
                + ", ".join(
                    _markdown_code_span(f"{package.distribution}=={package.version}")
                    for package in provenance.packages
                ),
            )
        )
        if is_independent_pasym_swap:
            lines.extend(
                (
                    "",
                    "Only the independently seeded THRML sampled cross-check scalar is eligible "
                    "for a two-sided 95% Student-t interval. Deterministic compiler and "
                    "exact-evaluation identity fields remain in the validated per-run section "
                    "below.",
                )
            )
            if records:
                lines.extend(("", *render_independent_pasym_swap_section(records[0])))
                if aggregate.completion_state is not CompletionState.COMPLETE:
                    lines.extend(
                        (
                            "",
                            "Aggregate completion is "
                            f"{_markdown_code_span(aggregate.completion_state.value)}; "
                            "failed seeds "
                            "are excluded from this selected successful record's gate table and "
                            "from seed aggregate calculations.",
                        )
                    )
    if is_target_context_pasym_swap:
        lines.extend(
            (
                "",
                "Only the empirical THRML K=30 residual is eligible for a cross-seed "
                "interval. Deterministic evidence is rendered once from the first successful "
                "persisted record, when one exists, after every successful record has passed "
                "deep validation and deterministic-hash comparison.",
            )
        )
        if records:
            lines.extend(("", *render_target_context_pasym_swap_section(records[0])))
        completed_seeds = tuple(record.spec.seed for record in records)
        failed_seeds = tuple(failure.seed for failure in aggregate.failures)
        all_requested_passed = (
            aggregate.completion_state is CompletionState.COMPLETE
            and len(target_summaries) == aggregate.requested_runs
            and all(summary.seed_acceptance.passed for summary in target_summaries)
        )
        acceptance_statement = (
            "- All requested seeds completed and passed: yes."
            if all_requested_passed
            else "- Acceptance applies only to completed seeds; one or more requested seeds failed."
            if records
            else "- No requested seed completed; no seed acceptance evidence is available."
        )
        lines.extend(
            (
                "",
                "### Acceptance and seed completeness",
                "",
                f"- Seed partition: requested={aggregate.seeds}; completed={completed_seeds}; "
                f"failed={failed_seeds}.",
                f"- Completion state: {_markdown_code_span(aggregate.completion_state.value)}.",
                acceptance_statement,
                (
                    "- Only empirical THRML evidence, sampled acceptance, cache state, and "
                    "timing vary by seed. Deterministic values are not treated as replicated "
                    "statistics."
                ),
                "",
                "### Evidence classes",
                "",
                (
                    "Exact evaluations are `exact_reference` for frozen software-derived "
                    "models. The target marginal, occurrence trace, and pooled profiles are "
                    "exact references for the checked target process and initial state."
                ),
                (
                    "Optimization, THRML sampling, and timing are `software_simulation`. "
                    "All-context and zero-support diagnostics are exact evaluations of those "
                    "software-derived artifacts, not physical measurements."
                ),
                "",
                "### Deferred scope and explicit exclusions",
                "",
                "- model-context matching was not evaluated.",
                "- REINFORCE and other trajectory-level refinement were not evaluated.",
                "- A complete compiled 25-site rollout was not executed.",
                "- This is not official Thermalizers compatibility or hosted simulation.",
                "- This is not Z1 or other physical hardware evidence.",
            )
        )
    lines.extend(
        (
            "",
            (
                "## Scalar results"
                if is_weighted_graph_walk
                else "## Sampled scalar results across seeds"
                if is_target_context_pasym_swap
                else "## Scalar results across seeds"
            ),
            "",
        )
    )
    lines.extend(_metric_table(aggregate))
    if is_weighted_graph_walk:
        lines.extend(
            (
                "",
                "This baseline contains no THRML, Thermalizers, Z1 projection, "
                "or physical-hardware evidence.",
            )
        )
        if records:
            lines.extend(("", *_weighted_graph_walk_section(records[0])))
    elif is_target_context_pasym_swap:
        lines.extend(
            (
                "",
                "The interval contract above applies only to the independently seeded empirical "
                "THRML residual. A single successful seed receives no manufactured interval.",
            )
        )
    else:
        lines.extend(
            (
                "",
                "Intervals use a two-sided 95% Student-t interval across independently seeded "
                "runs. "
                "No interval is manufactured for a single successful seed.",
                "",
                "THRML reports include exact-reference marginal comparisons, total variation, "
                "and within-chain autocorrelation/ESS summaries when those metrics are present.",
            )
        )
    if aggregate.omitted_metrics:
        lines.extend(("", "## Omitted aggregate metrics", ""))
        lines.extend(
            f"- {_markdown_code_span(name)}: {_markdown_text(reason)}"
            for name, reason in aggregate.omitted_metrics.items()
        )
    if aggregate.failures:
        lines.extend(("", "## Failures", ""))
        lines.extend(
            f"- Seed {failure.seed}: {_markdown_code_span(failure.error_type)} — "
            f"{_markdown_text(failure.message)}"
            for failure in aggregate.failures
        )
    lines.extend(
        (
            "",
            "## Evidence caveat",
            "",
            f"This report contains {_markdown_code_span(aggregate.evidence_class.value)} evidence "
            f"from {_markdown_code_span(aggregate.backend_id.value)}. It is not a physical Z1 or "
            "TSU hardware measurement. "
            "Any calibrated projection must remain in a separate projection record.",
            "",
            "## Machine-readable artifacts",
            "",
            "- [config.snapshot.toml](config.snapshot.toml)",
            "- [aggregate.json](aggregate.json)",
            "- [run-record.schema.json](schemas/run-record.schema.json)",
            "- [aggregate-record.schema.json](schemas/aggregate-record.schema.json)",
        )
    )
    lines.extend(
        f"- [Seed {record.spec.seed} run]({path})"
        for record, path in zip(records, aggregate.run_record_paths, strict=True)
    )
    return "\n".join(lines) + "\n"


def write_report_from_persisted(output_dir: Path) -> None:
    aggregate = AggregateRecord.model_validate_json(
        (output_dir / "aggregate.json").read_text(encoding="utf-8")
    )
    records = tuple(
        RunRecord.model_validate_json((output_dir / path).read_text(encoding="utf-8"))
        for path in aggregate.run_record_paths
    )
    atomic_write_text(
        output_dir / aggregate.report_generation.report_path, render_report(aggregate, records)
    )
