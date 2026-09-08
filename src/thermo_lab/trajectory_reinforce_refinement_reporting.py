"""Reload-validated reporting for one-step trajectory REINFORCE refinement."""

from __future__ import annotations

from collections.abc import Sequence

from thermo_lab.config import (
    TRAJECTORY_REINFORCE_REFINEMENT_EXPERIMENT_ID,
    TRAJECTORY_REINFORCE_REFINEMENT_SAMPLE_DEFINITION,
    trajectory_reinforce_refinement_non_seed_config_hash,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap import PAPER_SOURCE
from thermo_lab.records import RUN_TIMING_SOURCE, RunRecord
from thermo_lab.schemas import (
    TrajectoryReinforceModelConfig,
    TrajectoryReinforceRefinementRunConfig,
    validate_trajectory_reinforce_refinement_request,
)
from thermo_lab.trajectory_reinforce_refinement_results import (
    TrajectoryReinforceRefinementSummary,
    validate_trajectory_reinforce_refinement_summary,
)
from thermo_lab.trajectory_reinforce_reporting import _checked_runtime_provenance

TRAJECTORY_REFINEMENT_TIMING_METHOD = (
    "NumPy PCG64 inverse-CDF augmented trajectory batch, covariance-aware shared-gradient "
    "reduction, projected parameter update, and exact post-update objective enumeration; "
    "excludes checked fixture construction, deterministic gradient oracles, provenance, "
    "persistence, aggregation, and reporting"
)
_METRICS = frozenset(
    {
        "trajectory_reinforce_refinement_summary",
        "maximum_absolute_shared_gradient_error",
        "objective_before",
        "objective_after",
        "objective_improvement",
        "relative_objective_improvement",
        "cap_active_parameter_count",
        "acceptance_passed",
    }
)
_METHODS = {
    "trajectory_reinforce_refinement_summary": (
        EvidenceClass.SOFTWARE_SIMULATION,
        "sampled shared-gradient update with exact objective readout",
    ),
    "maximum_absolute_shared_gradient_error": (
        EvidenceClass.SOFTWARE_SIMULATION,
        "NumPy PCG64 inverse-CDF sampled shared-gradient error",
    ),
    "objective_before": (
        EvidenceClass.EXACT_REFERENCE,
        "exact enumeration of the initial tied-parameter circuit",
    ),
    "objective_after": (
        EvidenceClass.SOFTWARE_SIMULATION,
        "exact enumeration at the seed-derived updated parameters",
    ),
    "objective_improvement": (
        EvidenceClass.SOFTWARE_SIMULATION,
        "exact objective before minus exact objective after sampled update",
    ),
    "relative_objective_improvement": (
        EvidenceClass.SOFTWARE_SIMULATION,
        "exact objective improvement divided by exact initial objective",
    ),
    "cap_active_parameter_count": (
        EvidenceClass.SOFTWARE_SIMULATION,
        "count of sampled raw parameters projected onto the declared cap",
    ),
    "acceptance_passed": (
        EvidenceClass.SOFTWARE_SIMULATION,
        "exact gradient checks, bounded update, and strict objective decrease",
    ),
}


def validate_persisted_trajectory_refinement_record(
    record: RunRecord,
) -> tuple[
    TrajectoryReinforceRefinementSummary,
    TrajectoryReinforceModelConfig,
    TrajectoryReinforceRefinementRunConfig,
]:
    """Validate one persisted refinement record without resampling."""

    if record.spec.experiment_id != TRAJECTORY_REINFORCE_REFINEMENT_EXPERIMENT_ID:
        raise ValueError("record is not the checked trajectory refinement experiment")
    if record.backend_id is not BackendId.NUMPY_EXACT_CATEGORICAL:
        raise ValueError("trajectory refinement requires the numpy_exact_categorical backend")
    if record.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("trajectory refinement requires software_simulation run evidence")
    if record.spec.sample_definition != TRAJECTORY_REINFORCE_REFINEMENT_SAMPLE_DEFINITION:
        raise ValueError("trajectory refinement sample definition differs from the checked value")
    if (
        record.model_hash != record.spec.model_hash
        or record.run_config_hash != record.spec.run_config_hash
    ):
        raise ValueError("trajectory refinement record hashes differ from the persisted request")
    if set(record.metrics) != _METRICS:
        raise ValueError("trajectory refinement record has an unexpected metric set")
    for name, (evidence_class, method) in _METHODS.items():
        observation = record.metrics[name]
        if (
            observation.evidence_class is not evidence_class
            or observation.method != method
            or observation.source != PAPER_SOURCE
            or observation.unit is not None
            or observation.notes is not None
        ):
            raise ValueError(f"trajectory refinement metric metadata differs for {name}")
    timing = record.timing
    if (
        timing.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
        or timing.source != RUN_TIMING_SOURCE
        or timing.unit != "seconds"
        or timing.compile_seconds != 0.0
        or timing.synchronized is not True
        or timing.timing_method != TRAJECTORY_REFINEMENT_TIMING_METHOD
    ):
        raise ValueError("trajectory refinement timing differs from the checked boundary")
    _checked_runtime_provenance(record)

    model = TrajectoryReinforceModelConfig.model_validate(
        to_json_value(record.spec.model_parameters)
    )
    run = TrajectoryReinforceRefinementRunConfig.model_validate(
        to_json_value(record.spec.run_parameters)
    )
    validate_trajectory_reinforce_refinement_request(model, run, record.spec.seed)
    summary = validate_trajectory_reinforce_refinement_summary(
        record.metrics["trajectory_reinforce_refinement_summary"].value
    )
    if summary.estimator.sample.seed != record.spec.seed:
        raise ValueError("trajectory refinement sampled seed differs from the record")
    if summary.estimator.sample.sample_definition != record.spec.sample_definition:
        raise ValueError("trajectory refinement sampled definition differs from the record")
    sample_counts = (
        summary.estimator.sample.occurrence_0.moments.sample_count,
        summary.estimator.sample.occurrence_1.moments.sample_count,
        summary.estimator.sample.shared.moments.sample_count,
    )
    if any(sample_count != run.batch_size for sample_count in sample_counts):
        raise ValueError("trajectory refinement sample_count must equal run.batch_size")
    expected_hash = trajectory_reinforce_refinement_non_seed_config_hash(model, run)
    if summary.estimator.deterministic.request_hash != expected_hash:
        raise ValueError("trajectory refinement request hash differs from checked inputs")
    if summary.refinement.learning_rate != run.learning_rate:
        raise ValueError("trajectory refinement learning rate differs from checked inputs")
    expected_scalars = {
        "maximum_absolute_shared_gradient_error": (
            summary.estimator.sample.maximum_absolute_shared_gradient_error
        ),
        "objective_before": summary.refinement.objective_before,
        "objective_after": summary.refinement.objective_after,
        "objective_improvement": summary.refinement.objective_improvement,
        "relative_objective_improvement": summary.refinement.relative_objective_improvement,
        "cap_active_parameter_count": summary.refinement.cap_active_parameter_count,
        "acceptance_passed": summary.acceptance_passed,
    }
    for name, expected in expected_scalars.items():
        if record.metrics[name].value != expected:
            raise ValueError(f"trajectory refinement standalone metric differs for {name}")
    return summary, model, run


def _number(value: float) -> str:
    return f"{value:.10g}"


def render_trajectory_refinement_section(records: Sequence[RunRecord]) -> list[str]:
    """Render gradient validity and exact objective change for every completed seed."""

    validated = tuple(validate_persisted_trajectory_refinement_record(record) for record in records)
    if len({record.spec.seed for record in records}) != len(records):
        raise ValueError("trajectory refinement report requires unique completed seeds")
    deterministic_identities = {
        (
            summary.estimator.deterministic.request_hash,
            summary.estimator.deterministic.deterministic_result_digest,
        )
        for summary, _, _ in validated
    }
    if len(deterministic_identities) > 1:
        raise ValueError("trajectory refinement deterministic identities differ across seeds")

    lines = [
        "## One-step trajectory REINFORCE refinement",
        "",
        (
            "Each seed uses the same three-site, two-operation circuit and the same shared "
            "parameter vector at both occurrences. Seeds vary the sampled update; exact "
            "enumeration measures the declared objective before and after that update."
        ),
    ]
    if not validated:
        return [*lines, "", "No successful refinement record is available."]
    deterministic = validated[0][0].estimator.deterministic
    lines.extend(
        (
            "",
            "### Retained gradient checks",
            "",
            f"- Maximum exact gradient discrepancy: {_number(deterministic.maximum_exact_error)}.",
            "- Maximum finite-difference discrepancy: "
            f"{_number(deterministic.maximum_finite_difference_error)}.",
            "",
            "### Exact objective change by seed",
            "",
            "| Seed | Exact objective before | Exact objective after | Objective improvement | "
            "Relative improvement | Strict objective decrease | Parameter bounds satisfied | "
            "Cap-active parameters |",
            "|---:|---:|---:|---:|---:|---|---|---:|",
        )
    )
    for summary, _, _ in sorted(validated, key=lambda item: item[0].estimator.sample.seed):
        step = summary.refinement
        lines.append(
            f"| {summary.estimator.sample.seed} | {_number(step.objective_before)} | "
            f"{_number(step.objective_after)} | {_number(step.objective_improvement)} | "
            f"{_number(step.relative_objective_improvement)} | "
            f"{'yes' if step.objective_improved else 'no'} | "
            f"{'yes' if step.bounds_satisfied else 'no'} | "
            f"{step.cap_active_parameter_count} |"
        )
    lines.extend(("", "### Audited shared-parameter update", ""))
    for summary, _, _ in sorted(validated, key=lambda item: item[0].estimator.sample.seed):
        step = summary.refinement
        lines.extend(
            (
                f"#### Seed {summary.estimator.sample.seed}",
                "",
                "| Parameter | Initial | Sampled gradient | Raw candidate | Projected update | "
                "Bound active |",
                "|---|---:|---:|---:|---:|---|",
            )
        )
        for index, parameter in enumerate(summary.estimator.deterministic.parameter_order):
            lines.append(
                f"| {parameter} | {_number(step.initial_parameters[index])} | "
                f"{_number(step.sampled_shared_gradient[index])} | "
                f"{_number(step.raw_parameters[index])} | "
                f"{_number(step.updated_parameters[index])} | "
                f"{'yes' if step.cap_active_mask[index] else 'no'} |"
            )
        lines.append("")
    lines.extend(
        (
            "The exact readout is exact for the frozen categorical circuit. Because each updated "
            "parameter vector comes from a finite sampled batch, the before/after refinement "
            "comparison is software_simulation evidence.",
            "",
            "This does not establish finite-Gibbs-horizon unbiasedness, a 25-site refinement, "
            "official Thermalizers compatibility, hosted simulation, or physical Z1/TSU evidence.",
        )
    )
    return lines
