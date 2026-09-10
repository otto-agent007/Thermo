"""Checked M2 audit of explicitly supplied frozen M1 parameter pairs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from thermo_lab.composed_pasym_swap_artifacts import HORIZON_LABELS, HorizonLabel
from thermo_lab.composed_pasym_swap_reporting import _checked_runtime_provenance
from thermo_lab.composed_trajectory_refinement_reporting import (
    _reconstruction_backend,
    validate_persisted_composed_refinement_record,
)
from thermo_lab.composed_trajectory_refinement_results import (
    ComposedTrajectoryRefinementSummary,
    StrictEvidenceModel,
)
from thermo_lab.evidence import EvidenceClass
from thermo_lab.frozen_pair_finite_sweeps import (
    HorizonTerminalEvidence,
    build_joint_horizon_tables,
    declared_sampling_work,
    evaluate_frozen_pair_horizons,
    paired_table_digest,
)
from thermo_lab.hashing import canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import find_repository_root
from thermo_lab.records import RUN_TIMING_SOURCE, RunRecord, RuntimeProvenance, RunTiming

HISTORICAL_M1_SUMMARIES = {
    0: "sha256:24bd10653d1b7f5e32c14d1959a2f784e690a2eaf58c368e8cd73d6af6cb09bf",
    1: "sha256:c43b40ecd64994a250107ca4e3feb3b4fca7be412d4953aef42cbfb501a8ac1c",
    2: "sha256:2b8322b44525a727716e2b73a7ef315c4c2887ff5ff79ab9cbe8827a7b9d5be1",
}
AUDIT_TIMING_METHOD = (
    "execution_seconds measures synchronous NumPy float64 local joint-table construction, "
    "PCG64 endpoint sampling over all 14 frozen-pair/horizon cells, and bounded terminal "
    "reductions; compile_seconds is zero because table construction is included in execution; "
    "excludes M1 source validation and lineage reconstruction, provenance, artifact validation, "
    "persistence, aggregation, and reporting; no live Gibbs, THRML, hosted or hardware execution"
)


class FrozenPairAuditRequest(StrictEvidenceModel):
    identity_version: Literal["frozen_pair_finite_sweep_request.v1"] = (
        "frozen_pair_finite_sweep_request.v1"
    )
    source_summary_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_bundle_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    schedule_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    exact_target_reference: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    initial_parameter_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    updated_parameter_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    seed: StrictInt = Field(ge=0, le=2)
    evaluation_seed: StrictInt = Field(ge=0)
    batch_size: StrictInt = Field(ge=32768, le=32768)
    site_count: StrictInt = Field(ge=25, le=25)
    occurrence_count: StrictInt = Field(ge=500, le=500)
    horizons: tuple[HorizonLabel, ...] = HORIZON_LABELS
    dtype: Literal["float64"] = "float64"
    rng: Literal["numpy.Generator(PCG64)"] = "numpy.Generator(PCG64)"
    randomness_policy: Literal["reuse M1 held-out seed; one uniform vector per occurrence"] = (
        "reuse M1 held-out seed; one uniform vector per occurrence"
    )
    reset_policy: Literal["uniform over eight free states at every occurrence"] = (
        "uniform over eight free states at every occurrence"
    )
    sweep_policy: Literal["hidden then both outputs, inputs clamped"] = (
        "hidden then both outputs, inputs clamped"
    )
    endpoint_policy: Literal["hidden-major joint inverse CDF; propagate outputs only"] = (
        "hidden-major joint inverse CDF; propagate outputs only"
    )
    estimator_policy: Literal["unbiased_order_two_u_statistic"] = "unbiased_order_two_u_statistic"
    uncertainty_policy: Literal["paired_delete_one_jackknife_normal_95_approximate"] = (
        "paired_delete_one_jackknife_normal_95_approximate"
    )
    conclusion_policy: Literal["interval_sign_with_zero_inconclusive"] = (
        "interval_sign_with_zero_inconclusive"
    )
    scientific_status: Literal["descriptive_non_gating"] = "descriptive_non_gating"
    sample_definition: Literal["one complete before/after trajectory pair per horizon"] = (
        "one complete before/after trajectory pair per horizon"
    )

    @field_validator("horizons", mode="before")
    @classmethod
    def freeze_horizons(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_horizons(self) -> FrozenPairAuditRequest:
        if self.horizons != HORIZON_LABELS:
            raise ValueError("audit requires the seven canonical ordered horizons")
        return self


def _checked_source(record: RunRecord) -> ComposedTrajectoryRefinementSummary:
    summary = validate_persisted_composed_refinement_record(record)
    return summary


def _request(summary: ComposedTrajectoryRefinementSummary) -> FrozenPairAuditRequest:
    return FrozenPairAuditRequest(
        source_summary_digest=summary.summary_digest,
        source_request_hash=summary.request_hash,
        source_bundle_digest=summary.source_bundle_digest,
        schedule_digest=summary.schedule_digest,
        exact_target_reference=summary.exact_target_reference,
        initial_parameter_digest=summary.initial_parameter_digest,
        updated_parameter_digest=canonical_sha256(summary.update.updated_parameters),
        seed=summary.seed,
        evaluation_seed=summary.evaluation_seed,
        batch_size=summary.evaluation.before.sample_count,
        site_count=len(summary.target_occupancy),
        occurrence_count=500,
    )


def audit_result_digest(request_hash: str, cells: tuple[HorizonTerminalEvidence, ...]) -> str:
    """Bind bounded observations separately from the requested-input hash and timings."""
    return canonical_sha256(
        {
            "identity_version": "frozen_pair_finite_sweep_result.v1",
            "request_hash": request_hash,
            "evidence_class": "software_simulation",
            "cells": tuple(cell.model_dump(mode="json") for cell in cells),
        }
    )


class FrozenPairAudit(StrictEvidenceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evidence_class: Literal["software_simulation"] = "software_simulation"
    source_record: RunRecord
    request: FrozenPairAuditRequest
    request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    cells: tuple[HorizonTerminalEvidence, ...]
    provenance: RuntimeProvenance
    timing: RunTiming
    result_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("cells", mode="before")
    @classmethod
    def freeze_cells(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_audit(self) -> FrozenPairAudit:
        source = _checked_source(self.source_record)
        if self.request != _request(source):
            raise ValueError("audit request must reconstruct from the supplied checked M1 source")
        if self.request_hash != canonical_sha256(self.request.model_dump(mode="json")):
            raise ValueError("audit request hash must bind canonical requested inputs")
        if tuple(cell.horizon for cell in self.cells) != HORIZON_LABELS:
            raise ValueError("audit must contain exactly one cell per canonical horizon")
        if any(
            cell.sample_count != self.request.batch_size
            or len(cell.before_counts) != self.request.site_count
            for cell in self.cells
        ):
            raise ValueError("audit counts must match the checked trajectory budget and sites")
        before = build_joint_horizon_tables(source.initial_parameters, beta=source.beta)
        after = build_joint_horizon_tables(source.update.updated_parameters, beta=source.beta)
        for index, cell in enumerate(self.cells):
            if cell.exact_tables_digest != paired_table_digest(
                cell.horizon, before[index], after[index]
            ):
                raise ValueError("audit tables must reconstruct from the frozen parameters")
            cell.statistics(source.target_occupancy)
        equilibrium = self.cells[0]
        if (
            equilibrium.before_counts != source.evaluation.before.occupancy_counts
            or equilibrium.after_counts != source.evaluation.after.occupancy_counts
            or equilibrium.joined_moment_counts
            != source.evaluation.joined_terminal_second_moment_counts
        ):
            raise ValueError("equilibrium control must exactly reproduce M1 held-out counts")
        if self.result_digest != audit_result_digest(self.request_hash, self.cells):
            raise ValueError("audit result digest must bind all bounded terminal evidence")
        timing = self.timing
        if (
            timing.compile_seconds != 0.0
            or timing.execution_seconds <= 0.0
            or timing.timing_method != AUDIT_TIMING_METHOD
            or timing.source != RUN_TIMING_SOURCE
            or timing.synchronized is not True
        ):
            raise ValueError("audit timing must use the declared synchronous NumPy boundary")
        _checked_runtime_provenance(
            self.source_record.model_copy(update={"provenance": self.provenance})
        )
        return self


def audit_frozen_record(record: RunRecord) -> FrozenPairAudit:
    """Evaluate an already trained, strictly validated M1 pair; never run another update."""
    from thermo_lab.backends.numpy_composed_pasym_swap import _composed_provenance

    summary = _checked_source(record)
    prepared = _reconstruction_backend().prepare(record.spec)
    request = _request(summary)
    request_hash = canonical_sha256(request.model_dump(mode="json"))
    started = time.perf_counter()
    cells = evaluate_frozen_pair_horizons(
        summary.initial_parameters,
        summary.update.updated_parameters,
        prepared.bundle.occurrence_target_indices,
        prepared.bundle.occurrence_site_indices,
        summary.target_occupancy,
        site_count=request.site_count,
        batch_size=request.batch_size,
        seed=request.evaluation_seed,
        beta=summary.beta,
        parameter_cap=summary.parameter_cap,
    )
    elapsed = time.perf_counter() - started
    return FrozenPairAudit(
        source_record=record,
        request=request,
        request_hash=request_hash,
        cells=cells,
        provenance=_composed_provenance(find_repository_root(Path.cwd())),
        timing=RunTiming(
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            unit="seconds",
            source=RUN_TIMING_SOURCE,
            compile_seconds=0.0,
            execution_seconds=elapsed,
            synchronized=True,
            timing_method=AUDIT_TIMING_METHOD,
        ),
        result_digest=audit_result_digest(request_hash, cells),
    )


def render_frozen_pair_audit(audits: tuple[FrozenPairAudit, ...]) -> str:
    """Revalidate reloaded evidence before publication, including model_copy bypasses."""
    checked = tuple(
        FrozenPairAudit.model_validate_json(audit.model_dump_json()) for audit in audits
    )
    seeds = tuple(audit.request.seed for audit in checked)
    if not seeds or seeds != tuple(sorted(set(seeds))):
        raise ValueError("publication requires nonempty, unique, sorted source seeds")
    lines = [
        "# Frozen-pair finite-sweep transfer audit (M2)",
        "",
        "All full-program results are software_simulation. Exact local joint endpoint tables "
        "and the target reference are exact_reference. Explicitly supplied frozen M1 pairs are "
        "evaluated without additional training. The equilibrium control reproduces M1 exactly.",
        "",
        "Each cell uses 32,768 complete trajectory pairs over 25 sites and 500 occurrences. "
        "One PCG64 uniform vector per occurrence is shared across all 14 member/horizon cells. "
        "Each finite kernel resets uniformly and represents complete hidden-then-output sweeps. "
        "NumPy samples exact finite-horizon joint endpoints; no live Gibbs chain is executed.",
        "",
        "The order-two U-statistic estimates squared population occupancy loss without "
        "finite-batch bias. Negative after-minus-before differences favor the updated pair. "
        "Approximate normal 95% intervals use paired delete-one jackknife uncertainty "
        "conditional on each frozen pair. Near-zero and tiny-sample coverage can be far below "
        "95% (M1 toy examples: 50% and 62.5%). These are pointwise intervals, not simultaneous "
        "bands. All improved/regressed/inconclusive conclusions are descriptive and non-gating.",
        "",
        "Horizons and paired members are correlated, not independent replications. Independent "
        "source seeds are the cross-run replication units. The reused M1 held-out stream is "
        "an equilibrium reproduction control, not fresh independent confirmation.",
        "",
        "| Seed | Horizon | Before loss | After loss | After minus before | "
        "Approximate 95% interval | Conclusion | Leakage before | Leakage after | Leakage delta |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for audit in checked:
        target = _checked_source(audit.source_record).target_occupancy
        for cell in audit.cells:
            m = cell.metrics(target)
            interval = m["paired_jackknife_normal_95_interval"]
            lines.append(
                f"| {audit.request.seed} | {cell.horizon} | "
                f"{m['population_objective_before']!r} | "
                f"{m['population_objective_after']!r} | "
                f"{m['population_objective_difference_after_minus_before']!r} | "
                f"{interval!r} | {m['population_objective_conclusion']} | "
                f"{m['particle_leakage_before']!r} | {m['particle_leakage_after']!r} | "
                f"{m['particle_leakage_difference_after_minus_before']!r} |"
            )
    lines.extend(
        (
            "",
            "## Across-seed descriptive means",
            "",
            "Each mean weights the supplied independent source seeds equally. No per-run SEs "
            "or intervals are averaged, and no across-seed confidence claim is made.",
            "",
            "| Horizon | Independent seeds | Mean loss difference (after minus before) |",
            "| --- | --- | --- |",
        )
    )
    for index, horizon in enumerate(HORIZON_LABELS):
        differences = [
            audit.cells[index]
            .statistics(_checked_source(audit.source_record).target_occupancy)
            .population_objective_difference_after_minus_before
            for audit in checked
        ]
        lines.append(f"| {horizon} | {len(checked)} | {math.fsum(differences) / len(checked)!r} |")
    lines.extend(
        (
            "",
            "## Declared sampling work",
            "",
            "Counts below are per complete trajectory per member of the frozen pair. "
            "Multiply by 2 x 32,768 for both members at one finite horizon and source seed. "
            "Three free pbits update per complete sweep. Equilibrium has no finite sweep budget.",
            "",
            "| Horizon | Complete sweeps | Free pbit updates |",
            "| --- | --- | --- |",
        )
    )
    for horizon in HORIZON_LABELS:
        work = declared_sampling_work(horizon, occurrence_count=500)
        sweeps = work["complete_sweeps_per_trajectory_per_member"]
        updates = work["free_pbit_updates_per_trajectory_per_member"]
        lines.append(f"| {horizon} | {sweeps} | {updates} |")
    lines.extend(
        (
            "",
            "These are declared algorithmic counts, not measured device operations, energy, "
            "or latency. They exclude reset, clamp, parameter writes, readout, host I/O, "
            "and table setup. NumPy executes endpoint draws rather than these Gibbs updates.",
            "",
            "## Identities and execution timing",
            "",
            f"Timing boundary: {AUDIT_TIMING_METHOD}",
            "",
            "| Seed | Historical PR #20 pair | M1 source summary | M2 request | "
            "M2 result | NumPy seconds |",
            "| --- | --- | --- | --- | --- | --- |",
        )
    )
    for audit in checked:
        historical = (
            audit.request.source_summary_digest == HISTORICAL_M1_SUMMARIES[audit.request.seed]
        )
        lines.append(
            f"| {audit.request.seed} | {historical} | {audit.request.source_summary_digest} | "
            f"{audit.request_hash} | {audit.result_digest} | {audit.timing.execution_seconds!r} |"
        )
    lines.extend(
        (
            "",
            "A regenerated numerical compiler lineage need not match the historical release "
            "bit for bit. The table explicitly marks historical identity matches. M2 freezes "
            "the supplied validated source parameters and reproduces that source's equilibrium "
            "control exactly; it does not silently substitute a historical pair.",
            "",
            "Moment feasibility and digest consistency are necessary audit checks, not complete "
            "binary realizability or proof of execution. This study does not establish "
            "finite-sweep gradient correctness, iterative convergence, official Thermalizers "
            "compatibility, hosted simulation, or physical Z1/TSU performance.",
            "",
        )
    )
    return "\n".join(lines)


def run_frozen_pair_audit(
    source_paths: tuple[Path, ...], output_dir: Path
) -> tuple[FrozenPairAudit, ...]:
    """Audit explicitly supplied M1 records and publish completion only after round-trip checks."""
    if not source_paths:
        raise ValueError("at least one explicit M1 source record is required")
    records = tuple(
        RunRecord.model_validate_json(path.read_text(encoding="utf-8")) for path in source_paths
    )
    summaries = tuple(_checked_source(record) for record in records)
    seeds = tuple(summary.seed for summary in summaries)
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate source seeds are not independent replications")
    ordered = sorted(zip(seeds, records, strict=True), key=lambda pair: pair[0])
    # A fresh destination protects both input artifacts and previous/partial audit outputs.
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        output_dir / "audit.schema.json",
        json.dumps(FrozenPairAudit.model_json_schema(), indent=2, sort_keys=True) + "\n",
    )
    audits = []
    for seed, record in ordered:
        audit = audit_frozen_record(record)
        path = output_dir / f"seed-{seed:010d}.json"
        atomic_write_text(path, audit.model_dump_json(indent=2) + "\n")
        audits.append(FrozenPairAudit.model_validate_json(path.read_text(encoding="utf-8")))
    result = tuple(audits)
    atomic_write_text(output_dir / "report.md", render_frozen_pair_audit(result))
    atomic_write_text(
        output_dir / "completion.json",
        json.dumps(
            {
                "status": "complete",
                "seeds": [audit.request.seed for audit in result],
                "full_three_seed_release": tuple(seed for seed, _ in ordered) == (0, 1, 2),
                "completed_runs": len(result),
                "horizon_cells": len(result) * len(HORIZON_LABELS),
                "scientific_status": "descriptive_non_gating",
                "result_digests": [audit.result_digest for audit in result],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return result
