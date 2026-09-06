"""Reload-validated reporting for the checked trajectory REINFORCE study."""

from __future__ import annotations

from collections.abc import Sequence

from thermo_lab.config import (
    TRAJECTORY_REINFORCE_EXPERIMENT_ID,
    TRAJECTORY_REINFORCE_SAMPLE_DEFINITION,
    trajectory_reinforce_non_seed_config_hash,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap import PAPER_SOURCE
from thermo_lab.records import RUN_TIMING_SOURCE, RunRecord
from thermo_lab.schemas import (
    TrajectoryReinforceModelConfig,
    TrajectoryReinforceRunConfig,
    validate_trajectory_reinforce_request,
)
from thermo_lab.trajectory_reinforce_results import (
    TrajectoryReinforceSummary,
    validate_trajectory_reinforce_summary,
)

_SUMMARY_METHOD = "structured exact and NumPy exact-categorical estimator result"
_SAMPLED_SCALAR_METHOD = "NumPy PCG64 inverse-CDF sampled shared-gradient error"
_ACCEPTANCE_METHOD = "exact identities and deterministic finite-difference acceptance"
_ACCEPTANCE_NOTES = "Independent of Monte Carlo error and uncertainty."
_TIMING_PREFIX = (
    "NumPy Generator(PCG64) inverse-CDF exact categorical sampling of 65,536 augmented "
    "trajectories using four independent streams in main_0, reference_0, main_1, reference_1 "
    "order"
)
_METRICS = frozenset(
    {
        "trajectory_reinforce_summary",
        "maximum_absolute_shared_gradient_error",
        "acceptance_passed",
    }
)


def _checked_metric_metadata(record: RunRecord) -> None:
    summary = record.metrics["trajectory_reinforce_summary"]
    scalar = record.metrics["maximum_absolute_shared_gradient_error"]
    acceptance = record.metrics["acceptance_passed"]
    if (
        summary.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
        or summary.method != _SUMMARY_METHOD
        or summary.source != PAPER_SOURCE
        or summary.unit is not None
        or summary.notes is not None
    ):
        raise ValueError("trajectory summary metric metadata differs from the checked contract")
    if (
        scalar.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
        or scalar.method != _SAMPLED_SCALAR_METHOD
        or scalar.source != PAPER_SOURCE
        or scalar.unit is not None
        or scalar.notes is not None
    ):
        raise ValueError("trajectory sampled scalar metadata differs from the checked contract")
    if (
        acceptance.evidence_class is not EvidenceClass.EXACT_REFERENCE
        or acceptance.method != _ACCEPTANCE_METHOD
        or acceptance.source != PAPER_SOURCE
        or acceptance.unit is not None
        or acceptance.notes != _ACCEPTANCE_NOTES
    ):
        raise ValueError("trajectory acceptance metric metadata differs from the checked contract")


def _checked_runtime_provenance(record: RunRecord) -> None:
    provenance = record.provenance
    if not provenance.python_version or not provenance.platform:
        raise ValueError("trajectory runtime provenance requires Python and platform identities")
    if (
        provenance.jax_version != "not-used"
        or provenance.jaxlib_version != "not-used"
        or provenance.jax_backend != "not-used"
        or provenance.jax_devices != ()
        or provenance.jax_enable_x64 is not False
    ):
        raise ValueError("trajectory runtime provenance must identify JAX as not used")
    if tuple(package.distribution for package in provenance.packages) != (
        "numpy",
        "thermo-lab",
    ):
        raise ValueError("trajectory runtime provenance requires NumPy and thermo-lab packages")
    if any(
        not package.version
        or package.artifact_verification
        != "runtime_import_metadata; artifact hash not runtime reverified"
        for package in provenance.packages
    ):
        raise ValueError("trajectory package provenance differs from the checked contract")


def validate_persisted_trajectory_reinforce_record(
    record: RunRecord,
) -> tuple[
    TrajectoryReinforceSummary,
    TrajectoryReinforceModelConfig,
    TrajectoryReinforceRunConfig,
]:
    """Validate one complete persisted record without sampling or optimization."""

    if record.spec.experiment_id != TRAJECTORY_REINFORCE_EXPERIMENT_ID:
        raise ValueError("record is not the checked trajectory REINFORCE experiment")
    if record.backend_id is not BackendId.NUMPY_EXACT_CATEGORICAL:
        raise ValueError("trajectory records require the numpy_exact_categorical backend")
    if record.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("trajectory records require software_simulation run evidence")
    if record.spec.sample_definition != TRAJECTORY_REINFORCE_SAMPLE_DEFINITION:
        raise ValueError("trajectory sample definition differs from the checked value")
    if (
        record.model_hash != record.spec.model_hash
        or record.run_config_hash != record.spec.run_config_hash
    ):
        raise ValueError("trajectory record hashes differ from the persisted request")
    if set(record.metrics) != _METRICS:
        raise ValueError("trajectory record must contain exactly the checked metric set")
    _checked_metric_metadata(record)

    timing = record.timing
    if (
        timing.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
        or timing.source != RUN_TIMING_SOURCE
        or timing.unit != "seconds"
        or timing.compile_seconds != 0.0
        or timing.synchronized is not True
        or not timing.timing_method.startswith(_TIMING_PREFIX)
    ):
        raise ValueError("trajectory timing differs from the checked sampling boundary")
    _checked_runtime_provenance(record)

    model = TrajectoryReinforceModelConfig.model_validate(
        to_json_value(record.spec.model_parameters)
    )
    run = TrajectoryReinforceRunConfig.model_validate(to_json_value(record.spec.run_parameters))
    validate_trajectory_reinforce_request(model, run, record.spec.seed)
    summary = validate_trajectory_reinforce_summary(
        record.metrics["trajectory_reinforce_summary"].value
    )
    sample_counts = (
        summary.sample.occurrence_0.moments.sample_count,
        summary.sample.occurrence_1.moments.sample_count,
        summary.sample.shared.moments.sample_count,
    )
    if any(sample_count != run.batch_size for sample_count in sample_counts):
        raise ValueError("trajectory sample_count must equal the checked run.batch_size")
    expected_request_hash = trajectory_reinforce_non_seed_config_hash(model, run)
    if summary.deterministic.request_hash != expected_request_hash:
        raise ValueError("trajectory deterministic request hash differs from checked inputs")
    if summary.sample.seed != record.spec.seed:
        raise ValueError("trajectory sampled seed differs from the record seed")
    if summary.sample.sample_definition != record.spec.sample_definition:
        raise ValueError("trajectory sampled definition differs from the record definition")
    if record.metrics["maximum_absolute_shared_gradient_error"].value != (
        summary.sample.maximum_absolute_shared_gradient_error
    ):
        raise ValueError("standalone sampled scalar differs from strict summary")
    if record.metrics["acceptance_passed"].value is not summary.acceptance_passed:
        raise ValueError("standalone acceptance differs from strict summary")
    if not summary.acceptance_passed:
        raise ValueError("trajectory persisted summary did not pass deterministic acceptance")
    return summary, model, run


def _vector(values: Sequence[float], format_number) -> str:
    return "[" + ", ".join(format_number(value) for value in values) + "]"


def render_trajectory_reinforce_section(records: Sequence[RunRecord]) -> list[str]:
    """Render one deterministic reference and every validated seed in seed order."""

    validated = tuple(validate_persisted_trajectory_reinforce_record(record) for record in records)
    if len({record.spec.seed for record in records}) != len(records):
        raise ValueError("trajectory report requires unique completed seeds")
    identities = {
        (
            summary.deterministic.request_hash,
            summary.deterministic.deterministic_result_digest,
        )
        for summary, _, _ in validated
    }
    if len(identities) > 1:
        raise ValueError("trajectory deterministic identities differ across seeds")

    from thermo_lab.reporting import _format_number, _markdown_code_span, _markdown_text

    lines = [
        "## Trajectory-level REINFORCE estimator study",
        "",
        (
            "This checked microcircuit has three sites / two overlapping occurrences / one "
            "shared kernel."
        ),
        (
            "The shared parameter vector is used at both occurrences, and occurrence gradients "
            "are summed."
        ),
        "The references are independently sampled from the same parent and are never propagated.",
        "Monte Carlo comparison is non-gating; deterministic exact identities define acceptance.",
        "The exact m(phi) defines the stopped reward coefficient.",
        "No parameters were updated and no improvement was tested.",
        (
            "The sampled estimator is exact-categorical software evidence, not THRML or a "
            "finite-Gibbs gradient."
        ),
        "It is not official Thermalizers compatibility or hosted simulation.",
        "It is not a 25-site refinement or physical Z1/TSU measurement.",
    ]
    if not validated:
        lines.extend(("", "No successful trajectory record is available."))
        return lines

    deterministic = validated[0][0].deterministic
    lines.extend(
        (
            "",
            "### Checked deterministic identity",
            "",
            f"- Request hash: {_markdown_code_span(deterministic.request_hash)}.",
            "- Deterministic result digest: "
            f"{_markdown_code_span(deterministic.deterministic_result_digest)}.",
            f"- Exact objective: {_format_number(deterministic.objective)}.",
            "- Exact acceptance: "
            f"{_markdown_code_span('yes' if deterministic.accepted else 'no')}.",
            f"- Maximum exact discrepancy: {_format_number(deterministic.maximum_exact_error)}.",
            "- Maximum finite-difference discrepancy: "
            f"{_format_number(deterministic.maximum_finite_difference_error)}.",
            "",
            "### Target and model terminal laws",
            "",
            "| Law | Occupancy | Expected mass | Particle-number leakage | Signed mass drift |",
            "|---|---|---:|---:|---:|",
            "| Target | "
            f"{_vector(deterministic.target_law.occupancy, _format_number)} | "
            f"{_format_number(deterministic.target_law.expected_mass)} | "
            f"{_format_number(deterministic.target_law.particle_number_leakage)} | "
            f"{_format_number(deterministic.target_law.signed_mass_drift)} |",
            "| Model | "
            f"{_vector(deterministic.model_law.occupancy, _format_number)} | "
            f"{_format_number(deterministic.model_law.expected_mass)} | "
            f"{_format_number(deterministic.model_law.particle_number_leakage)} | "
            f"{_format_number(deterministic.model_law.signed_mass_drift)} |",
            "",
            "Terminal probabilities use the checked visible-state order: "
            f"target={_vector(deterministic.target_law.probabilities, _format_number)}; "
            f"model={_vector(deterministic.model_law.probabilities, _format_number)}.",
            "",
            "### Shared gradients and deterministic discrepancies",
            "",
            "| Parameter | Exact trajectory-score shared gradient | "
            "Exact expected-reference shared gradient | Untied occurrence finite-difference sum | "
            "Independent tied finite-difference gradient | Exact discrepancy | "
            "Untied finite-difference discrepancy | Tied finite-difference discrepancy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for index, parameter in enumerate(deterministic.parameter_order):
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_text(parameter),
                    _format_number(deterministic.score.shared[index]),
                    _format_number(deterministic.expected_reference.shared[index]),
                    _format_number(deterministic.finite_difference.shared[index]),
                    _format_number(deterministic.finite_difference_tied[index]),
                    _format_number(deterministic.exact_component_errors.shared[index]),
                    _format_number(deterministic.finite_difference_component_errors.shared[index]),
                    _format_number(deterministic.tied_finite_difference_error[index]),
                )
            )
            + " |"
        )

    lines.extend(("", "### Per-seed sampled shared-gradient evidence", ""))
    for summary, _, _ in sorted(validated, key=lambda item: item[0].sample.seed):
        sample = summary.sample
        lines.extend(
            (
                f"#### Seed {sample.seed}",
                "",
                f"- Sample digest: {_markdown_code_span(sample.sample_digest)}.",
                "- Maximum sampled shared-gradient error: "
                f"{_format_number(sample.maximum_absolute_shared_gradient_error)}.",
                "- Deterministic acceptance: "
                f"{_markdown_code_span('yes' if summary.acceptance_passed else 'no')}; "
                "Monte Carlo error and uncertainty do not gate it.",
                "",
                "| Parameter | Sampled mean | Standard error | Absolute error |",
                "|---|---:|---:|---:|",
            )
        )
        for index, parameter in enumerate(deterministic.parameter_order):
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_text(parameter),
                        _format_number(sample.shared.mean[index]),
                        _format_number(sample.shared.standard_error[index]),
                        _format_number(sample.shared.absolute_error[index]),
                    )
                )
                + " |"
            )
        lines.append("")
    return lines[:-1]
