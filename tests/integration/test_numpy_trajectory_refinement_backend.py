"""Backend contract for one-step exact-categorical trajectory refinement."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import pytest

from thermo_lab.backends.numpy_exact_categorical import sample_augmented_gradient_sources
from thermo_lab.config import load_experiment_config
from thermo_lab.records import RunRecord, RuntimeProvenance
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference
from thermo_lab.trajectory_reinforce_refinement import build_one_step_refinement
from thermo_lab.trajectory_reinforce_refinement_reporting import (
    render_trajectory_refinement_section,
    validate_persisted_trajectory_refinement_record,
)
from thermo_lab.trajectory_reinforce_refinement_results import (
    build_trajectory_reinforce_refinement_summary,
    validate_trajectory_reinforce_refinement_summary,
)
from thermo_lab.trajectory_reinforce_results import (
    build_trajectory_reinforce_sample_result,
    build_trajectory_reinforce_summary,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/numpy-trajectory-reinforce-pasym-swap-one-step.toml"


@pytest.fixture(scope="module")
def refinement_record() -> RunRecord:
    backend_module = importlib.import_module("thermo_lab.backends.numpy_exact_categorical")
    configured = load_experiment_config(CONFIG)
    return (
        backend_module.NumpyTrajectoryRefinementBackend(ROOT).execute(configured.to_spec()).record
    )


def test_backend_persists_sampled_update_and_exact_objective_improvement(
    refinement_record: RunRecord,
) -> None:
    record = refinement_record

    summary = validate_trajectory_reinforce_refinement_summary(
        record.metrics["trajectory_reinforce_refinement_summary"].value
    )
    refinement = summary.refinement

    assert summary.estimator.deterministic.accepted is True
    assert summary.estimator.deterministic.maximum_exact_error <= 1e-12
    assert summary.estimator.deterministic.maximum_finite_difference_error <= 1e-7
    assert refinement.objective_before == pytest.approx(0.5246570826850282, rel=0.0, abs=1e-15)
    assert refinement.objective_after == pytest.approx(0.35496303115459227, rel=0.0, abs=1e-15)
    assert refinement.objective_improvement == pytest.approx(
        0.16969405153043593, rel=0.0, abs=1e-15
    )
    assert refinement.objective_improved is True
    assert refinement.bounds_satisfied is True
    assert refinement.cap_active_parameter_count == 0
    assert max(abs(value) for value in refinement.updated_parameters) <= 2.0
    assert summary.acceptance_passed is True
    assert record.metrics["objective_improvement"].value == refinement.objective_improvement


def test_persisted_validator_rejects_tampered_top_level_hash(
    refinement_record: RunRecord,
) -> None:
    tampered = refinement_record.model_copy(update={"run_config_hash": "sha256:" + "0" * 64})

    with pytest.raises(ValueError, match="hashes"):
        validate_persisted_trajectory_refinement_record(tampered)


@pytest.mark.parametrize(
    "provenance",
    (
        lambda record: record.provenance.model_copy(update={"python_version": ""}),
        lambda record: record.provenance.model_copy(
            update={
                "packages": (
                    record.provenance.packages[0].model_copy(update={"version": ""}),
                    record.provenance.packages[1],
                )
            }
        ),
        lambda record: record.provenance.model_copy(
            update={
                "packages": (
                    record.provenance.packages[0].model_copy(
                        update={"artifact_verification": "unverified"}
                    ),
                    record.provenance.packages[1],
                )
            }
        ),
    ),
)
def test_persisted_validator_rejects_tampered_runtime_provenance(
    refinement_record: RunRecord,
    provenance: Callable[[RunRecord], RuntimeProvenance],
) -> None:
    tampered = refinement_record.model_copy(update={"provenance": provenance(refinement_record)})

    with pytest.raises(ValueError, match="provenance"):
        validate_persisted_trajectory_refinement_record(tampered)


def test_persisted_validator_requires_the_declared_training_batch_size(
    refinement_record: RunRecord,
) -> None:
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    original = validate_trajectory_reinforce_refinement_summary(
        refinement_record.metrics["trajectory_reinforce_refinement_summary"].value
    )
    sources = sample_augmented_gradient_sources(
        fixture=fixture,
        exact=exact,
        batch_size=17,
        seed=refinement_record.spec.seed,
    )
    sample = build_trajectory_reinforce_sample_result(
        deterministic_result_digest=original.estimator.deterministic.deterministic_result_digest,
        seed=refinement_record.spec.seed,
        sample_definition=refinement_record.spec.sample_definition,
        occurrence_0=sources.occurrence_0,
        occurrence_1=sources.occurrence_1,
        cross_products=sources.cross_products,
        exact=original.estimator.deterministic.expected_reference,
    )
    estimator = build_trajectory_reinforce_summary(
        deterministic=original.estimator.deterministic,
        sample=sample,
    )
    refinement = build_one_step_refinement(
        fixture=fixture,
        exact=exact,
        sampled_shared_gradient=sample.shared.mean,
        learning_rate=0.25,
    )
    replacement = build_trajectory_reinforce_refinement_summary(
        estimator=estimator,
        refinement=refinement,
    )
    metrics = dict(refinement_record.metrics)
    replacements = {
        "trajectory_reinforce_refinement_summary": replacement,
        "maximum_absolute_shared_gradient_error": sample.maximum_absolute_shared_gradient_error,
        "objective_before": refinement.objective_before,
        "objective_after": refinement.objective_after,
        "objective_improvement": refinement.objective_improvement,
        "relative_objective_improvement": refinement.relative_objective_improvement,
        "cap_active_parameter_count": refinement.update.cap_active_parameter_count,
        "acceptance_passed": replacement.acceptance_passed,
    }
    for name, value in replacements.items():
        metrics[name] = metrics[name].model_copy(update={"value": value})
    tampered = refinement_record.model_copy(update={"metrics": metrics})

    with pytest.raises(ValueError, match="sample_count"):
        validate_persisted_trajectory_refinement_record(tampered)


def test_report_retains_a_non_improving_update(refinement_record: RunRecord) -> None:
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    original = validate_trajectory_reinforce_refinement_summary(
        refinement_record.metrics["trajectory_reinforce_refinement_summary"].value
    )
    occurrence_0 = original.estimator.sample.occurrence_0.moments.model_copy(
        update={
            "component_sum": tuple(
                -value for value in original.estimator.sample.occurrence_0.moments.component_sum
            )
        }
    )
    occurrence_1 = original.estimator.sample.occurrence_1.moments.model_copy(
        update={
            "component_sum": tuple(
                -value for value in original.estimator.sample.occurrence_1.moments.component_sum
            )
        }
    )
    sample = build_trajectory_reinforce_sample_result(
        deterministic_result_digest=original.estimator.deterministic.deterministic_result_digest,
        seed=refinement_record.spec.seed,
        sample_definition=refinement_record.spec.sample_definition,
        occurrence_0=occurrence_0,
        occurrence_1=occurrence_1,
        cross_products=original.estimator.sample.cross_products,
        exact=original.estimator.deterministic.expected_reference,
    )
    estimator = build_trajectory_reinforce_summary(
        deterministic=original.estimator.deterministic,
        sample=sample,
    )
    refinement = build_one_step_refinement(
        fixture=fixture,
        exact=exact,
        sampled_shared_gradient=sample.shared.mean,
        learning_rate=0.25,
    )
    replacement = build_trajectory_reinforce_refinement_summary(
        estimator=estimator,
        refinement=refinement,
    )
    metrics = dict(refinement_record.metrics)
    replacements = {
        "trajectory_reinforce_refinement_summary": replacement,
        "maximum_absolute_shared_gradient_error": sample.maximum_absolute_shared_gradient_error,
        "objective_before": refinement.objective_before,
        "objective_after": refinement.objective_after,
        "objective_improvement": refinement.objective_improvement,
        "relative_objective_improvement": refinement.relative_objective_improvement,
        "cap_active_parameter_count": refinement.update.cap_active_parameter_count,
        "acceptance_passed": replacement.acceptance_passed,
    }
    for name, value in replacements.items():
        metrics[name] = metrics[name].model_copy(update={"value": value})
    record = refinement_record.model_copy(update={"metrics": metrics})

    rendered = "\n".join(render_trajectory_refinement_section((record,)))

    assert replacement.acceptance_passed is False
    assert replacement.refinement.objective_after > replacement.refinement.objective_before
    assert "| no | yes |" in rendered
