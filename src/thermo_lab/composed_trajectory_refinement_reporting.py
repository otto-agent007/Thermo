"""Reconstruct checked refinement records before aggregation or publication."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from thermo_lab.composed_pasym_swap_reporting import _checked_runtime_provenance
from thermo_lab.composed_trajectory_refinement import _schedule_digest
from thermo_lab.composed_trajectory_refinement_results import (
    ComposedTrajectoryRefinementSummary,
    validate_composed_trajectory_refinement_summary,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap import PAPER_SOURCE
from thermo_lab.records import RUN_TIMING_SOURCE, MetricObservation, RunRecord

REFINEMENT_SCALARS = frozenset(
    {
        "objective_before",
        "objective_after",
        "objective_improvement",
        "cap_active_parameter_count",
        "population_objective_before",
        "population_objective_after",
        "population_objective_difference_after_minus_before",
    }
)
REFINEMENT_TIMING_METHOD = (
    "compile_seconds measures deterministic NumPy/SciPy composed-lineage preparation on a "
    "non-seed cache miss (zero on cache hit); execution_seconds measures synchronous NumPy "
    "PCG64 occupancy and grouped-gradient batches, projected update, and held-out paired "
    "equilibrium evaluation, including table setup and bounded reductions; excludes request "
    "validation, summary/run-record construction and validation, provenance, persistence, "
    "aggregation, and reporting; no JAX execution, THRML sampling, hosted or physical hardware"
)


def refinement_metric_observations(summary: ComposedTrajectoryRefinementSummary):
    """Keep emitted scalar copies and their evidence metadata on one fixed contract."""
    values = {
        "composed_trajectory_refinement_summary": (
            summary,
            "sample-split one-step equilibrium full-program trajectory refinement",
            "Held-out objective improvement is descriptive and non-gating.",
        ),
        "objective_before": (
            summary.evaluation.objective_before,
            "historical plug-in squared sampled terminal occupancy objective before update",
            None,
        ),
        "objective_after": (
            summary.evaluation.objective_after,
            "historical plug-in squared sampled terminal occupancy objective after update",
            None,
        ),
        "objective_improvement": (
            summary.evaluation.objective_improvement,
            "historical plug-in before-minus-after objective with common random numbers",
            None,
        ),
        "cap_active_parameter_count": (
            summary.update.cap_active_parameter_count,
            "count of grouped parameters changed by box projection",
            None,
        ),
        "population_objective_before": (
            summary.evaluation.population_objective_before,
            "unbiased order-two U-statistic of population terminal occupancy objective "
            "before update",
            "An unbiased finite-sample estimate may be negative; descriptive and non-gating.",
        ),
        "population_objective_after": (
            summary.evaluation.population_objective_after,
            "unbiased order-two U-statistic of population terminal occupancy objective "
            "after update",
            "An unbiased finite-sample estimate may be negative; descriptive and non-gating.",
        ),
        "population_objective_difference_after_minus_before": (
            summary.evaluation.population_objective_difference_after_minus_before,
            "unbiased after-minus-before population objective difference "
            "with common random numbers",
            "Negative favors the updated parameters; descriptive and non-gating.",
        ),
    }
    return {
        name: MetricObservation(
            value=value,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method=method,
            source=PAPER_SOURCE,
            notes=notes,
        )
        for name, (value, method, notes) in values.items()
    }


@lru_cache(maxsize=1)
def _reconstruction_backend():
    from thermo_lab.backends.numpy_composed_trajectory_refinement import (
        NumpyComposedTrajectoryRefinementBackend,
    )

    return NumpyComposedTrajectoryRefinementBackend()


def validate_persisted_composed_refinement_record(
    record: RunRecord,
    *,
    allow_historical_platform: bool = False,
) -> ComposedTrajectoryRefinementSummary:
    """Rebuild evidence from bounded sources and trusted inputs, without resampling.

    Hash and moment consistency is an audit of the record, not proof of execution.
    """
    from thermo_lab.backends.numpy_composed_trajectory_refinement import spawn_refinement_role_seeds

    checked = RunRecord.model_validate(to_json_value(record))
    if (
        checked.backend_id is not BackendId.NUMPY_EXACT_CATEGORICAL
        or checked.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
    ):
        raise ValueError("Refinement records require NumPy software_simulation evidence")
    if set(checked.metrics) != REFINEMENT_SCALARS | {"composed_trajectory_refinement_summary"}:
        raise ValueError("Refinement record differs from the fixed metric set")
    timing = checked.timing
    if (
        timing.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
        or timing.source != RUN_TIMING_SOURCE
        or timing.unit != "seconds"
        or timing.synchronized is not True
        or timing.execution_seconds <= 0.0
        or timing.timing_method != REFINEMENT_TIMING_METHOD
    ):
        raise ValueError("Refinement timing differs from the checked execution boundary")
    _checked_runtime_provenance(checked, allow_historical_platform=allow_historical_platform)
    backend = _reconstruction_backend()
    model, run, request_hash = backend.checked_request(checked.spec)
    prepared = backend._prepared(model, run, request_hash)
    summary = validate_composed_trajectory_refinement_summary(
        checked.metrics["composed_trajectory_refinement_summary"].value,
        expected_bundle_digest=prepared.bundle.bundle_digest,
        expected_initial_parameters=prepared.initial_parameters,
        expected_target_occupancy=prepared.target_checkpoint.occupancy,
        expected_target_reference=prepared.target_checkpoint.exact_reference,
    )
    if (
        summary.request_hash != request_hash
        or summary.seed != checked.spec.seed
        or summary.beta != model.beta
        or summary.parameter_cap != model.parameter_cap
        or summary.learning_rate != run.learning_rate
    ):
        raise ValueError("Refinement summary differs from the checked request")
    if (summary.occupancy_seed, summary.gradient_seed, summary.evaluation_seed) != (
        spawn_refinement_role_seeds(checked.spec.seed)
    ):
        raise ValueError("Refinement role seeds differ from the checked request")
    if (
        summary.occupancy_source.sample_count != run.occupancy_batch_size
        or summary.gradient_source.sample_count != run.gradient_batch_size
        or summary.evaluation.before.sample_count != run.evaluation_batch_size
        or summary.evaluation.after.sample_count != run.evaluation_batch_size
    ):
        raise ValueError("Refinement sample counts differ from the checked request")
    targets = np.asarray(prepared.bundle.occurrence_target_indices)
    if summary.schedule_digest != _schedule_digest(
        targets,
        np.asarray(prepared.bundle.occurrence_site_indices),
        site_count=25,
    ):
        raise ValueError("Refinement schedule differs from the trusted composed program")
    # Every trajectory contribution is reward * beta * sum(main_feature-reference_feature).
    bound = (
        2.0
        * model.beta
        * np.bincount(targets, minlength=len(summary.initial_parameters))
        * sum(abs(value) for value in summary.reward_coefficient)
    )[:, None]
    gradient = summary.gradient_source
    if np.any(
        np.abs(np.asarray(gradient.component_sum)) > run.gradient_batch_size * bound * (1.0 + 1e-12)
    ) or np.any(
        np.asarray(gradient.component_sum_squares)
        > run.gradient_batch_size * bound**2 * (1.0 + 1e-12)
    ):
        raise ValueError("Refinement gradient moments exceed trajectory contribution bounds")
    for name, expected in refinement_metric_observations(summary).items():
        observed = checked.metrics[name]
        if observed.model_dump(exclude={"value"}) != expected.model_dump(exclude={"value"}):
            raise ValueError(f"Refinement metric metadata differs: {name}")
        if name in REFINEMENT_SCALARS and (
            type(observed.value) is not type(expected.value) or observed.value != expected.value
        ):
            raise ValueError(f"Refinement standalone scalar differs from its source: {name}")
    return summary


def render_composed_refinement_section(records: tuple[RunRecord, ...]) -> list[str]:
    summaries = [validate_persisted_composed_refinement_record(record) for record in records]
    lines = [
        "## One-step 25-site trajectory refinement",
        "",
        "Each seed updates 37 shared nine-parameter groups over 500 occurrences at equilibrium. "
        "Independent occupancy and gradient batches use 32,768 trajectories each; a separate "
        "32,768-trajectory held-out batch evaluates before/after with common random numbers. "
        "The objective is the sum of squared terminal occupancy errors against the exact target. "
        "These are software_simulation estimates, not exact full-program model objectives. "
        "The historical plug-in statistic squares sampled occupancies and has finite-batch bias. "
        "Its before-minus-after improvement is descriptive and non-gating, "
        "never an integrity gate.",
        "",
        "### Historical plug-in statistic",
        "",
        "| Seed | Before | After | Before minus after | Plug-in outcome | Projected parameters |",
        "|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        evaluation = summary.evaluation
        outcome = (
            "improved"
            if evaluation.objective_improvement > 0
            else "worsened"
            if evaluation.objective_improvement < 0
            else "unchanged"
        )
        lines.append(
            f"| {summary.seed} | {evaluation.objective_before!r} | "
            f"{evaluation.objective_after!r} | {evaluation.objective_improvement!r} | "
            f"{outcome} | {summary.update.cap_active_parameter_count} |"
        )
    if not summaries:
        lines.extend(("", "Unavailable because no seeded execution completed."))
    lines.extend(
        (
            "",
            "### Population-objective audit",
            "",
            "The unbiased order-two U-statistic estimates the squared population occupancy "
            "objective; its finite-sample value may be negative. The signed difference is "
            "after minus before. The paired delete-one jackknife uses complete trajectory pairs "
            "and all cross-site/before-after second moments. Its approximate normal 95% "
            "interval describes within-evaluation uncertainty conditional on the frozen "
            "parameter pair, not uncertainty from training. An interval entirely below zero "
            "is improved; entirely above zero is regressed; otherwise it is inconclusive. "
            "Near-zero population loss can cause severe undercoverage: with n=4, before=0, "
            "after Bernoulli(1/2), and target=1/2, exhaustive coverage is 8/16 (50%). "
            "This approximate interval has no finite-sample coverage guarantee. "
            "These conclusions are descriptive and non-gating. Per-run jackknife quantities "
            "stay nested; across-seed aggregate intervals use independent seeded runs and "
            "are distinct from the within-evaluation intervals.",
            "",
            "| Seed | Unbiased before | Unbiased after | After minus before | Paired jackknife SE "
            "| Approximate normal 95% interval | Conclusion | Bounds satisfied |",
            "|---|---|---|---|---|---|---|---|",
        )
    )
    for summary in summaries:
        evaluation = summary.evaluation
        lines.append(
            f"| {summary.seed} | {evaluation.population_objective_before!r} | "
            f"{evaluation.population_objective_after!r} | "
            f"{evaluation.population_objective_difference_after_minus_before!r} | "
            f"{evaluation.paired_jackknife_standard_error!r} | "
            f"{evaluation.paired_jackknife_normal_95_interval!r} | "
            f"{evaluation.population_objective_conclusion} | {summary.update.bounds_satisfied} |"
        )
    lines.extend(
        (
            "",
            "Learning rate: 0.01; projection bounds: [-2, 2]. Only independent seeds "
            "are replication units; timing is not a scientific replication metric. "
            "The exact three-site gradient oracle remains a separate validation gate. "
            "No finite-Gibbs gradient, iterative optimization, official Thermalizers, "
            "hosted simulator, or physical hardware result is claimed.",
            "",
        )
    )
    for summary in summaries:
        lines.append(
            f"- Seed {summary.seed}: request `{summary.request_hash}`; "
            f"source bundle `{summary.source_bundle_digest}`; exact target "
            f"`{summary.exact_target_reference}`; paired result "
            f"`{summary.evaluation.result_digest}`; summary `{summary.summary_digest}`."
        )
    return lines
