"""Archived M1 lineage, reporting, and completion-last M4B publication."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field, StrictFloat, StrictInt, model_validator

from thermo_lab.composed_pasym_swap_reporting import _checked_runtime_provenance
from thermo_lab.composed_trajectory_refinement_reporting import (
    _reconstruction_backend,
    validate_persisted_composed_refinement_record,
)
from thermo_lab.composed_trajectory_refinement_results import StrictEvidenceModel
from thermo_lab.hashing import canonical_sha256
from thermo_lab.matched_training_budget import (
    ComparisonProtocol,
    TrainingComparison,
    _run_comparison_payload,
    clear_generation_caches,
    declared_work,
)
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import find_repository_root
from thermo_lab.records import RunRecord, RuntimeProvenance

SOURCE_DIGESTS = (
    "sha256:24bd10653d1b7f5e32c14d1959a2f784e690a2eaf58c368e8cd73d6af6cb09bf",
    "sha256:c43b40ecd64994a250107ca4e3feb3b4fca7be412d4953aef42cbfb501a8ac1c",
    "sha256:2b8322b44525a727716e2b73a7ef315c4c2887ff5ff79ab9cbe8827a7b9d5be1",
)


class ComparisonRequest(StrictEvidenceModel):
    identity_version: Literal["matched_training_request.v1"] = "matched_training_request.v1"
    protocol: ComparisonProtocol
    seed: StrictInt = Field(ge=0, le=2)
    source_summary_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    initial_parameter_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    exact_target_reference: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    schedule_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_policy: Literal["original_M1_sources_embedded_in_M4_at_e6ce897"] = (
        "original_M1_sources_embedded_in_M4_at_e6ce897"
    )


def _request(source):
    if source.seed not in (0, 1, 2) or source.summary_digest != SOURCE_DIGESTS[source.seed]:
        raise ValueError("comparison requires the predeclared archived M1 source identity")
    return ComparisonRequest(
        protocol=ComparisonProtocol(),
        seed=source.seed,
        source_summary_digest=source.summary_digest,
        source_request_hash=source.request_hash,
        initial_parameter_digest=source.initial_parameter_digest,
        exact_target_reference=source.exact_target_reference,
        schedule_digest=source.schedule_digest,
    )


def audit_digest(request_hash, result_digest):
    return canonical_sha256(
        {
            "identity_version": "matched_training_audit.v1",
            "evidence_class": "software_simulation",
            "request_hash": request_hash,
            "comparison_result_digest": result_digest,
        }
    )


class TrainingComparisonAudit(StrictEvidenceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evidence_class: Literal["software_simulation"] = "software_simulation"
    source_record: RunRecord
    request: ComparisonRequest
    request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    comparison: TrainingComparison
    result_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    execution_seconds: StrictFloat = Field(ge=0, allow_inf_nan=False)
    timing_boundary: Literal[
        "cold synchronous NumPy tables, sampling, reductions and bounded terminal checks; "
        "excludes source preparation, sequence replay, enclosing audit validation, persistence "
        "and reporting; not hardware latency"
    ] = (
        "cold synchronous NumPy tables, sampling, reductions and bounded terminal checks; "
        "excludes source preparation, sequence replay, enclosing audit validation, persistence "
        "and reporting; not hardware latency"
    )
    provenance: RuntimeProvenance

    @model_validator(mode="after")
    def reconstruct(self):
        source = validate_persisted_composed_refinement_record(self.source_record)
        request = ComparisonRequest.model_validate_json(self.request.model_dump_json())
        if request != _request(source):
            raise ValueError("comparison request must bind the frozen source lineage")
        if self.request_hash != canonical_sha256(request.model_dump(mode="json")):
            raise ValueError("comparison request hash must bind all requested inputs")
        sequence = TrainingComparison.model_validate_json(self.comparison.model_dump_json())
        prepared = _reconstruction_backend().prepare(self.source_record.spec)
        if (
            sequence.protocol != request.protocol
            or sequence.seed != request.seed
            or sequence.initial_parameters != source.initial_parameters
            or sequence.target_occupancy != source.target_occupancy
            or sequence.occurrence_target_indices
            != tuple(int(x) for x in prepared.bundle.occurrence_target_indices)
            or sequence.occurrence_site_indices
            != tuple(tuple(int(x) for x in row) for row in prepared.bundle.occurrence_site_indices)
            or len(sequence.initial_parameters) != 37
            or len(sequence.target_occupancy) != 25
            or len(sequence.occurrence_target_indices) != 500
        ):
            raise ValueError("comparison must use the complete archived source program")
        if self.result_digest != audit_digest(self.request_hash, sequence.result_digest):
            raise ValueError("audit result must bind the request and complete comparison")
        _checked_runtime_provenance(
            self.source_record.model_copy(update={"provenance": self.provenance})
        )
        return self


def build_comparison_audit(record):
    from thermo_lab.backends.numpy_composed_pasym_swap import _composed_provenance

    source = validate_persisted_composed_refinement_record(record)
    request = _request(source)
    request_hash = canonical_sha256(request.model_dump(mode="json"))
    prepared = _reconstruction_backend().prepare(record.spec)
    clear_generation_caches()
    start = perf_counter()
    payload = _run_comparison_payload(
        source.initial_parameters,
        prepared.bundle.occurrence_target_indices,
        prepared.bundle.occurrence_site_indices,
        source.target_occupancy,
        seed=source.seed,
    )
    seconds = perf_counter() - start
    comparison = TrainingComparison.model_validate(payload)
    return TrainingComparisonAudit(
        source_record=record,
        request=request,
        request_hash=request_hash,
        comparison=comparison,
        result_digest=audit_digest(request_hash, comparison.result_digest),
        execution_seconds=seconds,
        provenance=_composed_provenance(find_repository_root(Path.cwd())),
    )


def render_comparison_report(audits):
    if not audits:
        raise ValueError("at least one run is required")
    checked = tuple(
        TrainingComparisonAudit.model_validate_json(a.model_dump_json()) for a in audits
    )
    seeds = tuple(a.request.seed for a in checked)
    if seeds != tuple(sorted(set(seeds))):
        raise ValueError("report requires unique ordered seeds")
    lines = [
        "# Matched-training-budget comparison (M4B)",
        "",
        "Full three-seed release."
        if seeds == (0, 1, 2)
        else "Diagnostic partial release; not the full three-seed study.",
        "",
        "Five updates per arm from identical archived initial parameters; learning rate 0.01, "
        "bounds [-2,2], beta 1, float64. Every update uses independent 32,768-trajectory "
        "occupancy and gradient batches. Finite training uses K=4; equilibrium training uses "
        "the exact local equilibrium law. All final comparisons execute at K=4.",
        "",
        "Updates and logical training draws are matched, not wall-clock time or hardware cost. "
        "Both training occupancy and gradient laws change between arms; this is not a score-only "
        "ablation. The fifth checkpoint is fixed in advance, with a fresh final stream.",
        "",
        "## Primary contrast: finite minus equilibrium",
        "",
        "Negative favors finite training. Approximate paired jackknife normal 95% intervals are "
        "conditional, pointwise, descriptive and non-gating. Known small-sample/near-zero "
        "undercoverage remains; these are not simultaneous bands or formal acceptance tests. "
        "Three independent seeds are the replication units, not the correlated comparisons "
        "or trajectories. No convergence or inference-sample saving is established.",
        "",
        "| Seed | Equilibrium-trained loss | Finite-trained loss | Finite minus equilibrium | "
        "Approximate 95% interval | Conclusion | Equilibrium leakage | Finite leakage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    primary_differences = []
    for audit in checked:
        seq = audit.comparison
        m = seq.primary_evaluation.metrics(seq.target_occupancy)
        lo, hi = m["paired_jackknife_normal_95_interval"]
        primary_differences.append(m["population_objective_difference_after_minus_before"])
        lines.append(
            f"| {seq.seed} | {m['population_objective_before']:.9g} | "
            f"{m['population_objective_after']:.9g} | "
            f"{m['population_objective_difference_after_minus_before']:.9g} | "
            f"[{lo:.9g}, {hi:.9g}] | {m['population_objective_conclusion']} | "
            f"{m['particle_leakage_before']:.6g} | {m['particle_leakage_after']:.6g} |"
        )
    lines += [
        "",
        f"Descriptive mean difference across {len(checked)} seeds: "
        f"{sum(primary_differences) / len(checked):.9g}. No across-seed confidence claim.",
        "",
        "## Each arm versus initial at K=4",
        "",
        "| Seed | Arm | Initial loss | Final loss | Difference | Approximate 95% interval | "
        "Conclusion | Initial leakage | Final leakage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for audit in checked:
        seq = audit.comparison
        for name, pair in (
            ("finite_k4", seq.initial_finite),
            ("equilibrium", seq.initial_equilibrium),
        ):
            m = pair.metrics(seq.target_occupancy)
            lo, hi = m["paired_jackknife_normal_95_interval"]
            lines.append(
                f"| {seq.seed} | {name} | {m['population_objective_before']:.9g} | "
                f"{m['population_objective_after']:.9g} | "
                f"{m['population_objective_difference_after_minus_before']:.9g} | "
                f"[{lo:.9g}, {hi:.9g}] | {m['population_objective_conclusion']} | "
                f"{m['particle_leakage_before']:.6g} | {m['particle_leakage_after']:.6g} |"
            )
    lines += [
        "",
        "## Projection diagnostics",
        "",
        "Counts of clipped parameters by update; training diagnostics do not select checkpoints.",
        "",
        "| Seed | Arm | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for audit in checked:
        for arm in audit.comparison.arms:
            lines.append(
                f"| {audit.request.seed} | {arm.name} | "
                + " | ".join(str(s.update.cap_active_parameter_count) for s in arm.steps)
                + " |"
            )
    lines += ["", "## Declared work per seed", "", "| Quantity | Count |", "| --- | --- |"]
    lines += [
        f"| {key} | {value if value is not None else 'undefined: equilibrium oracle'} |"
        for key, value in declared_work(500).items()
    ]
    lines += [
        "",
        "Counts describe algorithmic equivalents. NumPy draws joint endpoints; it does "
        "not execute these Gibbs sweeps. The three explicitly evaluated pairs repeat work "
        "despite sharing randomness. Excludes source preparation, table setup, replay, "
        "reset/clamp, host I/O and physical embedding costs. No measured device energy "
        "or latency is reported.",
        "",
        "Sampled results are software_simulation. Exact local tables are exact_reference. "
        "Lower occupancy loss does not imply particle conservation or full-distribution "
        "fidelity. Source lineage, every step, final histograms and joined moments are "
        "retained in bounded JSON and replayed before reporting.",
        "",
        "## Identity and simulator timing",
        "",
        checked[0].timing_boundary,
        "",
        "| Seed | Source | Request | Result | Simulator seconds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for audit in checked:
        lines.append(
            f"| {audit.request.seed} | `{audit.request.source_summary_digest}` | "
            f"`{audit.request_hash}` | `{audit.result_digest}` | {audit.execution_seconds:.6g} |"
        )
    return "\n".join(lines) + "\n"


def run_training_comparison(source_paths: tuple[Path, ...], output_dir: Path):
    if not source_paths:
        raise ValueError("at least one explicit archived M1 source is required")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    records = tuple(
        RunRecord.model_validate_json(p.read_text(encoding="utf-8")) for p in source_paths
    )
    requests = tuple(_request(validate_persisted_composed_refinement_record(r)) for r in records)
    seeds = tuple(r.seed for r in requests)
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate source seeds are not independent replications")
    ordered = sorted(zip(seeds, records, requests, strict=True), key=lambda row: row[0])
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        output_dir / "protocol.json",
        json.dumps({"requests": [r.model_dump(mode="json") for _, _, r in ordered]}, indent=2)
        + "\n",
    )
    atomic_write_text(
        output_dir / "audit.schema.json",
        json.dumps(TrainingComparisonAudit.model_json_schema(), indent=2) + "\n",
    )
    audits = []
    for seed, record, _ in ordered:
        audit = build_comparison_audit(record)
        path = output_dir / f"seed-{seed:010d}.json"
        atomic_write_text(path, audit.model_dump_json(indent=2) + "\n")
        audits.append(TrainingComparisonAudit.model_validate_json(path.read_text(encoding="utf-8")))
    result = tuple(audits)
    atomic_write_text(output_dir / "report.md", render_comparison_report(result))
    atomic_write_text(
        output_dir / "completion.json",
        json.dumps(
            {
                "status": "complete",
                "seeds": [a.request.seed for a in result],
                "full_three_seed_release": tuple(a.request.seed for a in result) == (0, 1, 2),
                "completed_runs": len(result),
                "updates_per_arm": 5,
                "evaluation_horizon": 4,
                "selected_checkpoint": 5,
                "matched_hardware_cost": False,
                "scientific_status": "descriptive_non_gating",
                "result_digests": [a.result_digest for a in result],
            },
            indent=2,
        )
        + "\n",
    )
    return result
