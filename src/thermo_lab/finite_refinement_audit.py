"""Checked M4 source lineage, five-update study, reporting, and publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from thermo_lab.composed_pasym_swap_reporting import _checked_runtime_provenance
from thermo_lab.composed_trajectory_refinement_reporting import (
    _reconstruction_backend,
    validate_persisted_composed_refinement_record,
)
from thermo_lab.composed_trajectory_refinement_results import StrictEvidenceModel
from thermo_lab.finite_refinement_sequence import (
    FiniteRefinementProtocol,
    FiniteTrainingSequence,
    run_training_sequence,
)
from thermo_lab.finite_sampler_audit import FiniteSamplerAudit, build_sampler_audit
from thermo_lab.finite_sweep_estimator_reference import exact_estimator_moments
from thermo_lab.finite_sweep_gradient_reference import CHECKED_HORIZONS
from thermo_lab.hashing import canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import find_repository_root
from thermo_lab.records import RunRecord, RuntimeProvenance


class FiniteRefinementRequest(StrictEvidenceModel):
    identity_version: Literal["bounded_finite_refinement_request.v1"] = (
        "bounded_finite_refinement_request.v1"
    )
    protocol: FiniteRefinementProtocol
    source_summary_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_bundle_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    initial_parameter_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    exact_target_reference: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    schedule_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    sampler_request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    seed: StrictInt = Field(ge=0, le=2)
    site_count: StrictInt = Field(default=25, ge=25, le=25)
    occurrence_count: StrictInt = Field(default=500, ge=500, le=500)
    parameter_group_count: StrictInt = Field(default=37, ge=37, le=37)
    start_policy: Literal["supplied_M1_initial_model_context_parameters"] = (
        "supplied_M1_initial_model_context_parameters"
    )


def _request(source, sampler):
    return FiniteRefinementRequest(
        protocol=FiniteRefinementProtocol(),
        source_summary_digest=source.summary_digest,
        source_request_hash=source.request_hash,
        source_bundle_digest=source.source_bundle_digest,
        initial_parameter_digest=source.initial_parameter_digest,
        exact_target_reference=source.exact_target_reference,
        schedule_digest=source.schedule_digest,
        sampler_request_hash=sampler.request_hash,
        seed=source.seed,
    )


def refinement_result_digest(request_hash, sampler_result_digest, sequence_result_digest):
    return canonical_sha256(
        {
            "identity_version": "bounded_finite_refinement_result.v1",
            "evidence_class": "software_simulation",
            "request_hash": request_hash,
            "sampler_result_digest": sampler_result_digest,
            "sequence_result_digest": sequence_result_digest,
        }
    )


class FiniteRefinementAudit(StrictEvidenceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evidence_class: Literal["software_simulation"] = "software_simulation"
    source_record: RunRecord
    sampler_audit: FiniteSamplerAudit
    request: FiniteRefinementRequest
    request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    sequence: FiniteTrainingSequence
    result_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    provenance: RuntimeProvenance

    @model_validator(mode="after")
    def reconstruct(self):
        source = validate_persisted_composed_refinement_record(self.source_record)
        sampler = FiniteSamplerAudit.model_validate_json(self.sampler_audit.model_dump_json())
        request = FiniteRefinementRequest.model_validate_json(self.request.model_dump_json())
        if request != _request(source, sampler):
            raise ValueError("M4 request must reconstruct from its actual supplied M1 lineage")
        if self.request_hash != canonical_sha256(request.model_dump(mode="json")):
            raise ValueError("M4 request hash must bind the complete declared inputs")
        sequence = FiniteTrainingSequence.model_validate_json(self.sequence.model_dump_json())
        prepared = _reconstruction_backend().prepare(self.source_record.spec)
        if (
            sequence.protocol != request.protocol
            or sequence.seed != request.seed
            or sequence.initial_parameters != source.initial_parameters
            or sequence.target_occupancy != source.target_occupancy
            or sequence.occurrence_target_indices
            != tuple(int(index) for index in prepared.bundle.occurrence_target_indices)
            or sequence.occurrence_site_indices
            != tuple(
                tuple(int(site) for site in row) for row in prepared.bundle.occurrence_site_indices
            )
            or len(sequence.initial_parameters) != 37
            or len(sequence.target_occupancy) != 25
            or len(sequence.occurrence_target_indices) != 500
        ):
            raise ValueError(
                "M4 sequence must use the supplied source's initial parameters, target, and "
                "schedule"
            )
        if self.result_digest != refinement_result_digest(
            self.request_hash, sampler.result_digest, sequence.result_digest
        ):
            raise ValueError("M4 result digest must bind sampler validation and all five updates")
        _checked_runtime_provenance(
            self.source_record.model_copy(update={"provenance": self.provenance})
        )
        return self


def build_finite_refinement_run(
    record: RunRecord, sampler_audit: FiniteSamplerAudit
) -> FiniteRefinementAudit:
    from thermo_lab.backends.numpy_composed_pasym_swap import _composed_provenance

    source = validate_persisted_composed_refinement_record(record)
    sampler = FiniteSamplerAudit.model_validate_json(sampler_audit.model_dump_json())
    request = _request(source, sampler)
    request_hash = canonical_sha256(request.model_dump(mode="json"))
    prepared = _reconstruction_backend().prepare(record.spec)
    sequence = run_training_sequence(
        source.initial_parameters,
        prepared.bundle.occurrence_target_indices,
        prepared.bundle.occurrence_site_indices,
        source.target_occupancy,
        seed=source.seed,
    )
    return FiniteRefinementAudit(
        source_record=record,
        sampler_audit=sampler,
        request=request,
        request_hash=request_hash,
        sequence=sequence,
        result_digest=refinement_result_digest(
            request_hash, sampler.result_digest, sequence.result_digest
        ),
        provenance=_composed_provenance(find_repository_root(Path.cwd())),
    )


def render_finite_refinement_report(audits: tuple[FiniteRefinementAudit, ...]) -> str:
    if not audits:
        raise ValueError("at least one validated run is required")
    checked = tuple(
        FiniteRefinementAudit.model_validate_json(audit.model_dump_json()) for audit in audits
    )
    seeds = tuple(audit.request.seed for audit in checked)
    if seeds != tuple(sorted(set(seeds))):
        raise ValueError("report runs require unique ordered seeds")
    if len({audit.sampler_audit.result_digest for audit in checked}) != 1:
        raise ValueError("runs must bind the same checked sampler diagnostics")
    full = seeds == (0, 1, 2)
    lines = [
        "# Bounded finite-sweep refinement (M4)",
        "",
        "Full three-seed release."
        if full
        else "Diagnostic partial release; this is not the full three-seed study.",
        "",
        "Five updates at four complete sweeps, learning rate 0.01, beta 1, float64, and bounds "
        "[-2, 2]. "
        "Each update uses independent 32,768-trajectory occupancy and gradient roles. "
        "The fifth checkpoint is selected in advance; 32,768 fresh paired trajectories "
        "evaluate initial versus final parameters.",
        "",
        "The initial parameters and target are frozen from each supplied M1 source. "
        "Its one-step updated parameters and held-out draws are not reused for M4 training or "
        "final evaluation.",
        "",
        "## Held-out initial versus fifth update",
        "",
        "Signed differences are after minus before. Approximate 95% intervals use the paired "
        "delete-one jackknife. "
        "Conclusions are descriptive and non-gating, retain small-sample/near-zero coverage "
        "limitations, "
        "and are not simultaneous confidence bands. This bounded study does not establish "
        "convergence.",
        "",
        "| Seed | Initial U-loss | Final U-loss | Difference | Approximate 95% interval | "
        "Conclusion | Initial leakage | Final leakage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for audit in checked:
        sequence = audit.sequence
        metrics = sequence.final_evaluation.metrics(sequence.target_occupancy)
        lo, hi = metrics["paired_jackknife_normal_95_interval"]
        lines.append(
            f"| {audit.request.seed} | {metrics['population_objective_before']:.9g} | "
            f"{metrics['population_objective_after']:.9g} | "
            f"{metrics['population_objective_difference_after_minus_before']:.9g} | "
            f"[{lo:.9g}, {hi:.9g}] | {metrics['population_objective_conclusion']} | "
            f"{metrics['particle_leakage_before']:.6g} | {metrics['particle_leakage_after']:.6g} |"
        )
    lines += [
        "",
        "## Training diagnostics",
        "",
        "These are unbiased order-two occupancy losses from the training role before each update, "
        "not held-out checkpoint comparisons. Different rows use fresh randomness and "
        "adaptively updated parameters; "
        "their fluctuations are not evidence of monotonic learning. No row selects a "
        "checkpoint or changes the protocol.",
        "",
        "| Seed | Update | Pre-update training U-loss | Parameters clipped |",
        "| --- | --- | --- | --- |",
    ]
    for audit in checked:
        for step in audit.sequence.steps:
            lines.append(
                f"| {audit.request.seed} | {step.iteration} | "
                f"{step.training_objective(audit.sequence.target_occupancy):.9g} | "
                f"{step.update.cap_active_parameter_count} |"
            )
    lines += [
        "",
        "## Sampler validation",
        "",
        "The exact estimator mean agrees with M3's independently checked gradient. Exact "
        "reference second moments "
        "include independent same-parent reference variance and covariance between shared "
        "occurrences. "
        "The table reports maximum absolute component errors divided by exact standard errors "
        "for a fixed "
        "reward coefficient. These sampled diagnostics are descriptive, not simultaneous "
        "acceptance tests. "
        "They do not include uncertainty from an estimated occupancy coefficient.",
        "",
        "| Sweeps | Maximum exact mean discrepancy | Seed 0 standardized error | Seed 1 "
        "standardized error | Seed 2 standardized error |",
        "| --- | --- | --- | --- | --- |",
    ]
    sampler = checked[0].sampler_audit
    for horizon in CHECKED_HORIZONS:
        cells = tuple(c for c in sampler.cells if c.horizon == horizon)
        lines.append(
            f"| {horizon} | {exact_estimator_moments(horizon).maximum_m3_error:.9g} | "
            + " | ".join(f"{c.maximum_standardized_mean_error:.6g}" for c in cells)
            + " |"
        )
    draws = 32768 * 500 * (5 + 2 * 5 + 2)
    lines += [
        "",
        "## Evidence and declared work",
        "",
        "All sampled outcomes are software_simulation; local probability/derivative tables and "
        "microcircuit "
        "expectations are exact_reference. NumPy samples precomputed joint endpoints. No "
        "hardware measurements are made.",
        "",
        f"Per study seed: {draws:,} logical endpoint draws across occupancy, main/reference "
        f"gradient, and paired final roles. "
        f"At K=4 this represents {4 * draws:,} declared complete-sweep equivalents and "
        f"{12 * draws:,} free-pbit-update equivalents. "
        "These are algorithmic counts, not executed Gibbs updates or measured latency/energy. "
        "They exclude source "
        "generation/validation, table construction, microcircuit diagnostics, artifact replay, "
        "reset/clamp, and I/O.",
        "",
        "JSON evidence retains all five updates and bounded moments, source records, role "
        "seeds, and final joined "
        "counts/histograms. Reloading replays the supplied sampling inputs and reconstructs "
        "the chain. "
        "Cached replays retain only immutable summaries; raw trajectories are not persisted.",
        "",
        "| Seed | Source summary | Request | Result |",
        "| --- | --- | --- | --- |",
    ]
    for audit in checked:
        lines.append(
            f"| {audit.request.seed} | `{audit.request.source_summary_digest}` | "
            f"`{audit.request_hash}` | `{audit.result_digest}` |"
        )
    lines += [
        "",
        f"Sampler request: `{sampler.request_hash}`",
        f"Sampler result: `{sampler.result_digest}`",
        "",
    ]
    return "\n".join(lines)


def run_finite_refinement(
    source_paths: tuple[Path, ...], output_dir: Path
) -> tuple[FiniteRefinementAudit, ...]:
    if not source_paths:
        raise ValueError("at least one explicit M1 source record is required")
    records = tuple(
        RunRecord.model_validate_json(path.read_text(encoding="utf-8")) for path in source_paths
    )
    summaries = tuple(validate_persisted_composed_refinement_record(record) for record in records)
    seeds = tuple(source.seed for source in summaries)
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate source seeds are not independent replications")
    ordered = sorted(zip(seeds, records, strict=True), key=lambda pair: pair[0])
    output_dir.mkdir(parents=True, exist_ok=False)
    sampler = build_sampler_audit()
    atomic_write_text(
        output_dir / "sampler-validation.json", sampler.model_dump_json(indent=2) + "\n"
    )
    atomic_write_text(
        output_dir / "sampler.schema.json",
        json.dumps(FiniteSamplerAudit.model_json_schema(), indent=2) + "\n",
    )
    atomic_write_text(
        output_dir / "audit.schema.json",
        json.dumps(FiniteRefinementAudit.model_json_schema(), indent=2) + "\n",
    )
    requests = [
        _request(source, sampler).model_dump(mode="json")
        for source in sorted(summaries, key=lambda s: s.seed)
    ]
    atomic_write_text(
        output_dir / "protocol.json",
        json.dumps(
            {"requests": requests, "sampler_request": sampler.request.model_dump(mode="json")},
            indent=2,
        )
        + "\n",
    )
    audits = []
    for seed, record in ordered:
        audit = build_finite_refinement_run(record, sampler)
        path = output_dir / f"seed-{seed:010d}.json"
        atomic_write_text(path, audit.model_dump_json(indent=2) + "\n")
        audits.append(FiniteRefinementAudit.model_validate_json(path.read_text(encoding="utf-8")))
    result = tuple(audits)
    atomic_write_text(output_dir / "report.md", render_finite_refinement_report(result))
    atomic_write_text(
        output_dir / "completion.json",
        json.dumps(
            {
                "status": "complete",
                "seeds": [seed for seed, _ in ordered],
                "full_three_seed_release": tuple(seed for seed, _ in ordered) == (0, 1, 2),
                "completed_runs": len(result),
                "updates_per_run": 5,
                "horizon": 4,
                "selected_checkpoint": 5,
                "scientific_status": "descriptive_non_gating",
                "sampler_result_digest": sampler.result_digest,
                "result_digests": [audit.result_digest for audit in result],
            },
            indent=2,
        )
        + "\n",
    )
    return result
