"""Markdown reports rendered only from persisted machine-readable records."""

from __future__ import annotations

from pathlib import Path

from thermo_lab.aggregate import AggregateRecord
from thermo_lab.persistence import atomic_write_text
from thermo_lab.records import RunRecord


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


def render_report(aggregate: AggregateRecord, records: tuple[RunRecord, ...]) -> str:
    """Render an evidence-safe report from already persisted validated data."""

    sample_definition = (
        records[0].spec.sample_definition
        if records
        else "Unavailable because no seeded execution completed."
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
        f"- Seeds: {seed_list} ({aggregate.requested_runs} independent seeded runs)",
        f"- Completed/failed: {aggregate.completed_runs}/{aggregate.failed_runs}",
        "",
        "## Sample definition",
        "",
        sample_definition,
        "",
        "Recorded Markov-chain states are not described as independent samples. "
        "Independent seeded runs are the replication unit for confidence intervals.",
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
    lines.extend(("", "## Scalar results across seeds", ""))
    lines.extend(_metric_table(aggregate))
    lines.extend(
        (
            "",
            "Intervals use a two-sided 95% Student-t interval across independently seeded runs. "
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
