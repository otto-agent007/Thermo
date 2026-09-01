"""Markdown reports rendered only from persisted machine-readable records."""

from __future__ import annotations

import re
from pathlib import Path

from thermo_lab.aggregate import (
    AggregateRecord,
    StatisticalSemantics,
    validate_aggregate_against_records,
)
from thermo_lab.graph_walk_results import (
    WeightedGraphWalkSummary,
    validate_weighted_graph_walk_observations,
)
from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap_reporting import render_independent_pasym_swap_section
from thermo_lab.persistence import atomic_write_text
from thermo_lab.records import RunRecord
from thermo_lab.schemas import WeightedGraphModelConfig, WeightedGraphRunConfig

_WEIGHTED_GRAPH_WALK_EXPERIMENT_ID = "torx.weighted_graph_walk.v1"
_INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID = "thrml.independent_pasym_swap_compilation.v1"


def _format_number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.8g}"


def _markdown_text(value: str) -> str:
    """Keep persisted text within one Markdown block and escape active syntax."""

    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    escaped_lines = []
    for line in normalized.split("\n"):
        escaped = line.replace("\\", "\\\\")
        for character in ("`", "*", "_", "[", "]", "<", ">", "|", "&"):
            escaped = escaped.replace(character, f"\\{character}")
        escaped_lines.append(re.sub(r"^([ ]{0,3})([#\-+])(?=\s)", r"\1\\\2", escaped))
    return " / ".join(escaped_lines)


def _markdown_code_span(value: str) -> str:
    """Render arbitrary persisted text as a single safe CommonMark code span."""

    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " / ")
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
            interval = f"unavailable ({metric.interval_unavailable_reason})"
        else:
            interval = (
                f"[{_format_number(metric.confidence_interval.lower)}, "
                f"{_format_number(metric.confidence_interval.upper)}]"
            )
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    f"`{metric.evidence_class.value}`"
                    if metric.evidence_class is not None
                    else "unavailable",
                    f"`{metric.unit}`" if metric.unit is not None else "dimensionless",
                    str(metric.count),
                    _format_number(metric.mean),
                    _format_number(metric.standard_deviation),
                    _format_number(metric.median),
                    _format_number(metric.minimum),
                    _format_number(metric.maximum),
                    interval,
                    metric.method,
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

    if record.spec.experiment_id != _WEIGHTED_GRAPH_WALK_EXPERIMENT_ID:
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

    validate_aggregate_against_records(aggregate, records)
    is_weighted_graph_walk = aggregate.experiment_id == _WEIGHTED_GRAPH_WALK_EXPERIMENT_ID
    is_independent_pasym_swap = aggregate.experiment_id == _INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID
    is_deterministic = (
        aggregate.statistical_semantics is StatisticalSemantics.DETERMINISTIC_IDENTITY
    )
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
        f"- Experiment: `{aggregate.experiment_id}`",
        f"- Backend: `{aggregate.backend_id.value}`",
        f"- Evidence class: `{aggregate.evidence_class.value}`",
        f"- Statistical semantics: `{aggregate.statistical_semantics.value}`",
        f"- Completion state: `{aggregate.completion_state.value}`",
        f"- Model hash: `{aggregate.model_hash}`",
        f"- Non-seed configuration hash: `{aggregate.run_config_hash}`",
        f"- Seeds: {seed_list} ({_run_set_description(aggregate)})",
        f"- Completed/failed: {aggregate.completed_runs}/{aggregate.failed_runs}",
        "",
        "## Sample definition",
        "",
        sample_definition,
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
                else "Recorded Markov-chain states are not described as independent samples. "
                "Independent seeded runs are the replication unit for confidence intervals."
            )
        ),
        "",
        "## Runtime provenance",
        "",
    ]
    if aggregate.provenance_summary is None:
        lines.append("Unavailable because no run completed.")
    elif is_independent_pasym_swap:
        lines.extend(
            (
                "",
                "Only the independently seeded THRML sampled cross-check scalar is eligible for "
                "a two-sided 95% Student-t interval. Deterministic compiler and exact-evaluation "
                "identity fields remain in the validated per-run section below.",
            )
        )
        if records:
            lines.extend(("", *render_independent_pasym_swap_section(records[0])))
    else:
        provenance = aggregate.provenance_summary
        lines.extend(
            (
                f"- Python: `{provenance.python_version}`",
                f"- Platform: `{provenance.platform}`",
                f"- JAX/JAXLIB: `{provenance.jax_version}` / `{provenance.jaxlib_version}`",
                f"- JAX backend: `{provenance.jax_backend}`",
                "- JAX devices: " + ", ".join(f"`{item}`" for item in provenance.jax_devices),
                f"- JAX x64 enabled: `{str(provenance.jax_enable_x64).lower()}`",
                f"- Numeric dtype: `{provenance.numeric_dtype}`",
                f"- Git commit: `{provenance.git_commit or 'unavailable'}`",
                "- Git dirty: `"
                + (
                    str(provenance.git_dirty).lower()
                    if provenance.git_dirty is not None
                    else "unavailable"
                )
                + "`",
                "- Packages: "
                + ", ".join(
                    f"`{package.distribution}=={package.version}`"
                    for package in provenance.packages
                ),
            )
        )
    lines.extend(
        (
            "",
            "## Scalar results" if is_weighted_graph_walk else "## Scalar results across seeds",
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
        lines.extend(f"- `{name}`: {reason}" for name, reason in aggregate.omitted_metrics.items())
    if aggregate.failures:
        lines.extend(("", "## Failures", ""))
        lines.extend(
            f"- Seed {failure.seed}: `{failure.error_type}` — {failure.message}"
            for failure in aggregate.failures
        )
    lines.extend(
        (
            "",
            "## Evidence caveat",
            "",
            f"This report contains `{aggregate.evidence_class.value}` evidence from "
            f"`{aggregate.backend_id.value}`. It is not a physical Z1 or TSU hardware measurement. "
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
