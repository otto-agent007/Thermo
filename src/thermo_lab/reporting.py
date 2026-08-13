"""Markdown reports rendered only from persisted machine-readable records."""

from __future__ import annotations

import math
from pathlib import Path

from thermo_lab.aggregate import AggregateRecord
from thermo_lab.graph_walk_results import WeightedGraphWalkSummary
from thermo_lab.hashing import to_json_value
from thermo_lab.persistence import atomic_write_text
from thermo_lab.records import RunRecord
from thermo_lab.schemas import WeightedGraphModelConfig, WeightedGraphRunConfig

_WEIGHTED_GRAPH_WALK_EXPERIMENT_ID = "torx.weighted_graph_walk.v1"


def _format_number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.8g}"


def _metric_table(aggregate: AggregateRecord) -> list[str]:
    lines = [
        "| Metric | Evidence | Unit | Count | Mean | Std. dev. | Median | Min | Max | "
        "95% interval | Method/source |",
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
    if aggregate.experiment_id == _WEIGHTED_GRAPH_WALK_EXPERIMENT_ID:
        return f"{aggregate.requested_runs} deterministic execution"
    return f"{aggregate.requested_runs} independent seeded runs"


def _validate_aggregate_records(aggregate: AggregateRecord, records: tuple[RunRecord, ...]) -> None:
    """Reject records that do not match the aggregate that references them."""

    if len(records) != aggregate.completed_runs:
        raise ValueError("aggregate completed run count does not match persisted run records")
    if len(records) != len(aggregate.run_record_paths):
        raise ValueError("aggregate run record paths do not match persisted run records")
    failed_seeds = {failure.seed for failure in aggregate.failures}
    expected_seeds = tuple(seed for seed in aggregate.seeds if seed not in failed_seeds)
    actual_seeds = tuple(record.spec.seed for record in records)
    if actual_seeds != expected_seeds:
        raise ValueError("aggregate seeds do not match persisted run records")

    for record, path in zip(records, aggregate.run_record_paths, strict=True):
        if aggregate.experiment_id != record.spec.experiment_id:
            raise ValueError("aggregate experiment id does not match persisted run record")
        if aggregate.backend_id != record.backend_id:
            raise ValueError("aggregate backend does not match persisted run record")
        if aggregate.evidence_class != record.evidence_class:
            raise ValueError("aggregate evidence class does not match persisted run record")
        if aggregate.model_hash != record.model_hash:
            raise ValueError("aggregate model hash does not match persisted run record")
        if aggregate.run_config_hash != record.spec.non_seed_run_config_hash:
            raise ValueError("aggregate run configuration hash does not match persisted run record")
        expected_path = f"runs/seed-{record.spec.seed:010d}.json"
        if path != expected_path:
            raise ValueError("aggregate run record path does not match persisted run seed")


def _validated_weighted_graph_walk_data(
    record: RunRecord,
) -> tuple[WeightedGraphWalkSummary, WeightedGraphModelConfig]:
    """Validate persisted graph observations against their checked requested inputs."""

    if record.spec.experiment_id != _WEIGHTED_GRAPH_WALK_EXPERIMENT_ID:
        raise ValueError("weighted graph-walk summary belongs to a different experiment")
    try:
        metric = record.metrics["weighted_graph_walk"]
    except KeyError as error:
        raise ValueError("Weighted graph-walk record is missing its persisted summary") from error
    summary = WeightedGraphWalkSummary.model_validate(to_json_value(metric.value))
    model = WeightedGraphModelConfig.model_validate(to_json_value(record.spec.model_parameters))
    run = WeightedGraphRunConfig.model_validate(to_json_value(record.spec.run_parameters))
    if metric.source != model.source_reference:
        raise ValueError("weighted graph-walk metric source differs from the persisted model")
    if summary.node_labels != tuple(model.nodes):
        raise ValueError("Weighted graph-walk summary node labels differ from the persisted model")
    if summary.source_reference != model.source_reference:
        raise ValueError("Weighted graph-walk summary source differs from the persisted model")
    if summary.declared_resolutions != tuple(run.resolutions):
        raise ValueError("weighted graph-walk summary resolutions differ from the persisted run")
    if summary.checkpoint_times != tuple(run.checkpoint_times):
        raise ValueError(
            "weighted graph-walk summary checkpoint times differ from the persisted run"
        )
    exact_final_error = max(
        abs(observed - requested)
        for observed, requested in zip(
            summary.exact_final_occupancy,
            run.expected_exact_final_occupancy,
            strict=True,
        )
    )
    if not math.isclose(
        exact_final_error,
        0.0,
        rel_tol=0.0,
        abs_tol=run.exact_invariant_tolerance,
    ):
        raise ValueError(
            "weighted graph-walk exact final occupancy differs from the requested endpoint"
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
        f"- Canonical edge order: `{canonical_order}`",
        "- Resolution/order variants are not independent replications.",
        "",
        "### Source graph fixture",
        "",
        "| Edge | Weight |",
        "|---|---:|",
    ]
    lines.extend(f"| {edge.source}–{edge.target} | {edge.weight:.2f} |" for edge in model.edges)
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
        f"| {node} | {_format_number(occupancy)} |"
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
    lines.extend(f"- {check}" for check in summary.acceptance.checks)
    lines.extend(
        (
            "",
            "### Checkpoint occupancy",
            "",
            "| N | Order | Time | " + " | ".join(summary.node_labels) + " |",
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

    _validate_aggregate_records(aggregate, records)
    is_weighted_graph_walk = aggregate.experiment_id == _WEIGHTED_GRAPH_WALK_EXPERIMENT_ID
    sample_definition = (
        records[0].spec.sample_definition
        if records
        else (
            "Unavailable because no deterministic execution completed."
            if is_weighted_graph_walk
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
            if is_weighted_graph_walk
            else "Recorded Markov-chain states are not described as independent samples. "
            "Independent seeded runs are the replication unit for confidence intervals."
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
