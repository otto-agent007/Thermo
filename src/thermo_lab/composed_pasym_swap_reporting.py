"""Fixed scalar surface and reconstruction-based persisted composed evidence checks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from thermo_lab.composed_pasym_swap_artifacts import ARTIFACT_FAMILIES, HORIZON_LABELS
from thermo_lab.composed_pasym_swap_results import (
    ComposedPAsymSwapSummary,
    validate_composed_pasym_swap_summary,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import canonical_json, to_json_value
from thermo_lab.pasym_swap import PAPER_SOURCE
from thermo_lab.pasym_swap_context import OCCUPANCY_ORDER
from thermo_lab.records import RUN_TIMING_SOURCE, MetricObservation, RunRecord
from thermo_lab.schemas import ComposedPAsymSwapRunConfig, PAsymSwapModelConfig

if TYPE_CHECKING:
    from thermo_lab.backends.numpy_composed_pasym_swap import NumpyComposedPAsymSwapBackend

FINAL_MEASUREMENTS = (
    "occupancy_half_l1_error",
    "maximum_site_occupancy_error",
    "particle_number_leakage",
    "signed_mass_drift",
    "ever_left_sector_probability",
)
PAIRED_MEASUREMENTS = (
    "occupancy_half_l1_difference",
    "particle_number_leakage_difference",
)
PAIRED_LABELS = (
    "target_context_minus_independent",
    "model_context_minus_target_context",
    "model_context_minus_independent",
)
_PAIRED_SOURCES = {
    "occupancy_half_l1_difference": "occupancy_half_l1_error",
    "particle_number_leakage_difference": "particle_number_leakage",
}
COMPOSED_PASYM_SWAP_TIMING_METHOD = (
    "compile_seconds measures deterministic NumPy/SciPy lineage, finite-table bundle, and exact "
    "target preparation on cache miss (zero on a non-seed request cache hit); execution_seconds "
    "measures synchronous NumPy PCG64 common-random-number sampling of 32,768 complete "
    "25-site trajectories over all 500 occurrences and 21 family-horizon cells, including "
    "sampler setup, bounded checkpoint reduction, and source validation; excludes top-level "
    "request validation, final summary/run-record construction and validation, provenance "
    "collection, persistence, aggregation, and reporting; no JAX execution, THRML sampler, "
    "hosted simulation, or physical hardware"
)


def comparison_outcome(difference: float) -> Literal["improved", "unchanged", "worsened"]:
    """Classify a validated error/leak delta by its exact sign, never a tolerance."""

    if difference < 0.0:
        return "improved"
    if difference > 0.0:
        return "worsened"
    return "unchanged"


def composed_scalar_metric_names() -> frozenset[str]:
    """The approved 105 cell metrics and 42 paired metrics, with no nullable scalars."""

    return frozenset(
        {
            f"final_{family}_{horizon}_{measurement}"
            for family in ARTIFACT_FAMILIES
            for horizon in HORIZON_LABELS
            for measurement in FINAL_MEASUREMENTS
        }
        | {
            f"final_{pair}_{horizon}_{measurement}"
            for pair in PAIRED_LABELS
            for horizon in HORIZON_LABELS
            for measurement in PAIRED_MEASUREMENTS
        }
    )


def composed_metric_metadata(name: str) -> dict[str, object]:
    """Exact evidence metadata shared by emission and persisted validation."""

    notes = None
    if name == "composed_pasym_swap_summary":
        method = "frozen-artifact NumPy composed finite-Gibbs checkpoint evidence"
    elif name == "integrity_acceptance_passed":
        method = "checked lineage, full-schedule sources, and reconstructed metrics"
        notes = "Scientific improvements and regressions are non-gating."
    elif name in composed_scalar_metric_names():
        method = (
            "final-checkpoint first-minus-second paired composed metric difference"
            if any(name.startswith(f"final_{pair}_") for pair in PAIRED_LABELS)
            else "final-checkpoint composed metric reconstructed from bounded trajectory counts"
        )
    else:
        raise ValueError(f"Unknown composed metric: {name}")
    return {
        "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
        "method": method,
        "source": PAPER_SOURCE,
        "unit": None,
        "notes": notes,
    }


def _final_scalar_values(summary: ComposedPAsymSwapSummary) -> dict[str, float]:
    values = {
        f"final_{cell.family}_{cell.horizon}_{measurement}": getattr(
            cell.checkpoints[-1].metrics, measurement
        )
        for cell in summary.cells
        for measurement in FINAL_MEASUREMENTS
    }
    for comparison in summary.comparisons:
        if comparison.occurrence_count == 500:
            for measurement, source in _PAIRED_SOURCES.items():
                value = comparison.differences[source]
                if type(value) is not float:
                    raise ValueError("Standalone paired composed metrics must be strict floats")
                values[f"final_{comparison.label}_{comparison.horizon}_{measurement}"] = value
    return values


def build_composed_metric_observations(
    summary: ComposedPAsymSwapSummary,
) -> dict[str, MetricObservation]:
    """Expose only the approved final scalar surface alongside the rich summary."""

    values = {
        "composed_pasym_swap_summary": summary,
        "integrity_acceptance_passed": summary.integrity_acceptance_passed,
        **_final_scalar_values(summary),
    }
    return {
        name: MetricObservation(value=value, **composed_metric_metadata(name))
        for name, value in values.items()
    }


def _checked_runtime_provenance(record: RunRecord) -> None:
    from thermo_lab.backends.numpy_composed_pasym_swap import _composed_provenance

    observed = record.provenance
    expected = _composed_provenance(None)
    # Historical Git identity is meaningful provenance, not today's checkout identity.
    if (
        observed.git_commit is not None
        and re.fullmatch(r"[0-9a-f]{40}", observed.git_commit) is None
    ):
        raise ValueError("Composed Git provenance requires a full commit identity")
    if (observed.git_commit is None) != (observed.git_dirty is None):
        raise ValueError("Composed Git commit and dirty-state availability must agree")
    if observed.model_copy(update={"git_commit": None, "git_dirty": None}) != expected:
        raise ValueError("Composed runtime provenance differs from checked runtime identities")
    if any(package.version == "not-installed" for package in observed.packages):
        raise ValueError("Composed lineage and execution packages must be installed")


@lru_cache(maxsize=1)
def _reconstruction_backend() -> NumpyComposedPAsymSwapBackend:
    from thermo_lab.backends.numpy_composed_pasym_swap import NumpyComposedPAsymSwapBackend

    # This cache holds deterministic inputs only; every summary is independently rebuilt.
    return NumpyComposedPAsymSwapBackend()


def validate_persisted_composed_pasym_swap_record(
    record: RunRecord,
) -> tuple[ComposedPAsymSwapSummary, PAsymSwapModelConfig, ComposedPAsymSwapRunConfig]:
    """Validate authoritative inputs, provenance and every persisted value without sampling.

    Runtime identities must match the checking environment. Counts are reconciled and all
    derived evidence is reconstructed, but this is not a cryptographic proof of execution.
    Scientific improvement or regression never affects integrity acceptance.
    """

    # model_copy/model_construct bypass validation; always deeply parse a fresh JSON tree.
    checked = RunRecord.model_validate(to_json_value(record))
    if (
        checked.backend_id is not BackendId.NUMPY_EXACT_CATEGORICAL
        or checked.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
    ):
        raise ValueError("Composed records require NumPy software_simulation evidence")
    expected_names = composed_scalar_metric_names() | {
        "composed_pasym_swap_summary",
        "integrity_acceptance_passed",
    }
    if set(checked.metrics) != expected_names:
        raise ValueError("Composed record differs from the exact fixed metric set")
    for name, observation in checked.metrics.items():
        if observation.model_dump(exclude={"value"}) != composed_metric_metadata(name):
            raise ValueError(f"Composed metric metadata differs for {name}")
    timing = checked.timing
    if (
        timing.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
        or timing.source != RUN_TIMING_SOURCE
        or timing.unit != "seconds"
        or timing.synchronized is not True
        or timing.execution_seconds <= 0.0
        or timing.timing_method != COMPOSED_PASYM_SWAP_TIMING_METHOD
    ):
        raise ValueError("Composed timing differs from the checked execution boundary")
    _checked_runtime_provenance(checked)
    backend = _reconstruction_backend()
    model, run, request_hash = backend.checked_request(checked.spec)
    prepared = backend._prepared(model, run, request_hash)
    summary = validate_composed_pasym_swap_summary(
        canonical_json(checked.metrics["composed_pasym_swap_summary"].value),
        bundle=prepared.bundle,
        target_checkpoints=prepared.target_checkpoints,
        request_hash=request_hash,
    )
    if summary.seed != checked.spec.seed or summary.batch_size != run.trajectory_batch_size:
        raise ValueError("Composed summary seed/batch differs from the checked request")
    if (
        summary.integrity_acceptance_passed is not True
        or checked.metrics["integrity_acceptance_passed"].value is not True
    ):
        raise ValueError("Composed persisted evidence must pass integrity acceptance")
    for name, expected in _final_scalar_values(summary).items():
        actual = checked.metrics[name].value
        if type(actual) is not float or actual != expected:
            raise ValueError(f"Composed standalone scalar differs from its final source: {name}")
    return summary, model, run


def _number(value: float | None) -> str:
    # Round-trip precision keeps even tiny signed differences visible as reported.
    return "unavailable (no one-particle samples)" if value is None else repr(value)


def _table(headings: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(headings) + " |",
        "|" + "|".join("---" for _ in headings) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _error_difference(value: float | None) -> str:
    return _number(value) if value is None else f"{_number(value)} ({comparison_outcome(value)})"


def _macrostep_tables(summaries: Sequence[ComposedPAsymSwapSummary]) -> list[str]:
    lines = []
    checkpoints = summaries[0].cells[0].checkpoints
    headings = ["Seed", "Family", "Horizon"] + [
        f"M{index} ({checkpoint.source.occurrence_count})"
        for index, checkpoint in enumerate(checkpoints)
    ]
    for heading, measurement in (
        ("Macrostep occupancy half-L1", "occupancy_half_l1_error"),
        ("Macrostep particle-number leakage", "particle_number_leakage"),
        ("Macrostep ever-left-sector probability", "ever_left_sector_probability"),
    ):
        lines.extend(
            (
                "",
                f"### {heading}",
                "",
                "Per-seed software_simulation; columns are macrosteps (canonical occurrences).",
                "",
                *_table(
                    headings,
                    (
                        [str(summary.seed), cell.family, cell.horizon]
                        + [
                            _number(getattr(checkpoint.metrics, measurement))
                            for checkpoint in cell.checkpoints
                        ]
                        for summary in summaries
                        for cell in summary.cells
                    ),
                ),
            )
        )
    return lines


def _site_tables(summaries: Sequence[ComposedPAsymSwapSummary], *, conditional: bool) -> list[str]:
    lines = [
        "",
        "### Final conditional one-particle location"
        if conditional
        else "### Final 25-site occupancy",
        "",
        "At occurrence 500: exact_reference target versus per-seed software_simulation. "
        + (
            "Conditioning uses only current N=1 samples, not trajectories that never left N=1. "
            "No one-particle samples means unavailable, not zero."
            if conditional
            else "Site order is x-major, y-minor; occupancy is not normalized after leakage."
        ),
    ]
    field = "conditional_location" if conditional else "occupancy"
    for horizon in HORIZON_LABELS:
        rows = []
        for summary in summaries:
            cells = {cell.family: cell for cell in summary.cells if cell.horizon == horizon}
            target = cells["independent"].checkpoints[-1].exact_target_occupancy
            vectors = [
                getattr(cells[family].checkpoints[-1].metrics, field)
                for family in ARTIFACT_FAMILIES
            ]
            for index, (x, y) in enumerate(OCCUPANCY_ORDER):
                rows.append(
                    [str(summary.seed), f"({x}, {y})", _number(target[index])]
                    + [_number(None if vector is None else vector[index]) for vector in vectors]
                )
        lines.extend(
            (
                "",
                f"#### {horizon}",
                "",
                *_table(["Seed", "Site", "Target", *ARTIFACT_FAMILIES], rows),
            )
        )
    return lines


def _paired_tables(summaries: Sequence[ComposedPAsymSwapSummary]) -> list[str]:
    lines = ["", "### All paired checkpoint scalar differences", ""]
    keys = tuple(summaries[0].comparisons[0].differences)
    occurrences = [item.source.occurrence_count for item in summaries[0].cells[0].checkpoints]
    error_keys = {
        "occupancy_half_l1_error",
        "maximum_site_occupancy_error",
        "particle_number_leakage",
        "ever_left_sector_probability",
        "conditional_location_half_l1_error",
    }
    for key in keys:
        rows = []
        formatter = _error_difference if key in error_keys else _number
        for summary in summaries:
            lookup = {
                (item.horizon, item.label, item.occurrence_count): item.differences[key]
                for item in summary.comparisons
            }
            for horizon in HORIZON_LABELS:
                for label in PAIRED_LABELS:
                    rows.append(
                        [str(summary.seed), horizon, label.replace("_minus_", " - ")]
                        + [
                            formatter(lookup[horizon, label, occurrence])
                            for occurrence in occurrences
                        ]
                    )
        lines.extend(
            (
                "",
                f"#### {key}",
                "",
                "First - second; negative means improvement for errors/leakage and ever-left "
                "probability. Mass, signed drift, and variance differences are signed changes, "
                "not better/worse judgments. Outcomes use the exact sign, with no tolerance. "
                "All values are per-seed software_simulation; unavailable propagates when "
                "either conditional location is unavailable.",
                "",
                *_table(["Seed", "Horizon", "First - second", *map(str, occurrences)], rows),
            )
        )
    return lines


def render_composed_pasym_swap_section(records: Sequence[RunRecord]) -> list[str]:
    """Render domain evidence only after deeply validating every supplied record."""

    validated = [validate_persisted_composed_pasym_swap_record(record) for record in records]
    lines = ["## Full 25-site composed finite-Gibbs comparison", ""]
    if not validated:
        return [*lines, "Unavailable because no seeded execution completed."]
    summaries = sorted((item[0] for item in validated), key=lambda item: item.seed)
    summary, model, run = validated[0]
    identities = {
        (item.request_hash, item.bundle_digest, item.target_checkpoint_digest) for item in summaries
    }
    if len(identities) != 1:
        raise ValueError("Cannot report incompatible composed deterministic identities")
    lines.extend(
        (
            f"- Primary source: [PAsymSwap paper]({model.source_reference}).",
            f"- Fixture: {model.torus_side} by {model.torus_side} torus; boundary "
            f"{model.periodic_boundary}; coordinates {model.coordinate_order}; initial particle "
            f"at (0, 0); gamma={model.gamma!r}, delta_t={model.delta_t!r}, beta={model.beta!r}.",
            f"- Schedule: {model.macrosteps} macrosteps, 500 canonical occurrences, 37 target "
            f"channels; color order {', '.join(model.color_order)}.",
            f"- Frozen families: {', '.join(run.artifact_families)}; horizons: "
            f"{', '.join(run.horizon_labels)}. kN means N complete Gibbs sweeps.",
            f"- Each seed: {run.trajectory_batch_size:,} complete trajectories for every "
            "family/horizon cell (21 paired cells, not 21 independent replications).",
            f"- Common random numbers: {run.rng_family}; {run.stream_policy}.",
            f"- Local transition policy: {run.local_transition_policy}.",
            "- Evidence: analytic target trace and frozen local tables are exact_reference "
            f"({model.exact_dtype}); sampled composed counts, errors, paired differences, "
            "and timings are software_simulation. Originating artifact optimization "
            "histories remain software_simulation, not exact optimization claims.",
            "- Scientific performance is non-gating. Integrity acceptance checks lineage, "
            "sources and metric reconstruction, not improvement. It is not proof that "
            "self-consistent persisted counts were actually sampled.",
            "- Seeds are the replication units; trajectories are within-batch samples. "
            "The generic cross-seed table summarizes the 147 approved final scalars with "
            "mean, sample standard deviation, median, extrema and 95% Student-t intervals. "
            "Deterministic target/local identities receive no intervals; one seed receives "
            "no manufactured interval.",
            "- Occupancy half-L1 is not distribution total variation when expected mass "
            "differs from one. Leakage is P(N != 1); ever-left also includes trajectories "
            "that returned to the one-particle sector.",
            "- Scope: not live THRML; not 25-site parameter refinement; not an unbiased "
            "finite-Gibbs gradient result; not official Thermalizers or an implementation "
            "of unpublished Thermalizers; not hosted Extropic simulation, calibrated "
            "projection, or physical Z1/TSU hardware evidence.",
            "",
            "### Integrity acceptance by seed",
            "",
            *_table(
                ["Seed", "Integrity acceptance", "Summary digest"],
                (
                    (
                        str(item.seed),
                        "yes" if item.integrity_acceptance_passed else "no",
                        item.summary_digest,
                    )
                    for item in summaries
                ),
            ),
            "",
            "### Final scientific outcomes (500 occurrences)",
            "",
            "First - second; negative means improvement for error/leakage. Positive means "
            "worsened and zero means unchanged, without tolerance-based relabeling. "
            "Mixed outcomes are reported separately, never collapsed into a success label.",
            "",
            *_table(
                [
                    "Seed",
                    "Horizon",
                    "First - second",
                    "Occupancy half-L1 difference",
                    "Leakage difference",
                ],
                (
                    (
                        str(item.seed),
                        pair.horizon,
                        pair.label.replace("_minus_", " - "),
                        _error_difference(pair.differences["occupancy_half_l1_error"]),
                        _error_difference(pair.differences["particle_number_leakage"]),
                    )
                    for item in summaries
                    for pair in item.comparisons
                    if pair.occurrence_count == 500
                ),
            ),
            "",
            "### Deterministic target and local-table identities",
            "",
            f"- Checked request: `{summary.request_hash}`.",
            f"- Frozen local-table/schedule bundle: `{summary.bundle_digest}`.",
            f"- Exact target checkpoint trace: `{summary.target_checkpoint_digest}`.",
            "- These bind the complete schedule, three families by 37 target channels, "
            "frozen parameters, local tables and exact target checkpoints; they are "
            "identities, not replicated sampled metrics.",
            "",
            *_table(
                ["Family", "Target hash", "Artifact hash", "Optimizer result hash"],
                (
                    (item.family, item.target_hash, item.artifact_hash, item.optimizer_result_hash)
                    for item in summary.artifact_identities
                ),
            ),
            "",
            "### Local kernel residuals (not composed error)",
            "",
            "Deterministic exact_reference, once across seeds: min/median/max over 37 "
            "target channels of half the sum of absolute differences across the entire "
            "local conditional table (all input rows), versus the same frozen artifact's "
            "equilibrium table. Equilibrium residual is zero by identity. These are not "
            "500-occurrence composed errors against the analytic target.",
            "",
            *_table(
                ["Family", "Horizon", "Min", "Median", "Max"],
                (
                    (
                        item.family,
                        item.horizon,
                        _number(item.minimum),
                        _number(item.median),
                        _number(item.maximum),
                    )
                    for item in summary.local_residual_summaries
                ),
            ),
        )
    )
    lines.extend(_macrostep_tables(summaries))
    lines.extend(_site_tables(summaries, conditional=False))
    lines.extend(
        (
            "",
            "### Final particle sectors and mass",
            "",
            "Per-seed software_simulation at occurrence 500. Exact target: P(N=0)=0, "
            "P(N=1)=1, P(N>=2)=0, E[N]=1, signed mass drift=0, Var[N]=0.",
            "",
            *_table(
                [
                    "Seed",
                    "Family",
                    "Horizon",
                    "P(N=0)",
                    "P(N=1)",
                    "P(N>=2)",
                    "E[N]",
                    "Signed mass drift",
                    "Var[N]",
                    "Maximum site error",
                    "Conditional location half-L1 error",
                ],
                (
                    [str(item.seed), cell.family, cell.horizon]
                    + [
                        _number(value)
                        for value in cell.checkpoints[-1].metrics.sector_probabilities
                    ]
                    + [
                        _number(getattr(cell.checkpoints[-1].metrics, key))
                        for key in (
                            "expected_particle_count",
                            "signed_mass_drift",
                            "particle_count_variance",
                            "maximum_site_occupancy_error",
                            "conditional_location_half_l1_error",
                        )
                    ]
                    for item in summaries
                    for cell in item.cells
                ),
            ),
        )
    )
    lines.extend(_site_tables(summaries, conditional=True))
    lines.extend(_paired_tables(summaries))
    return lines
