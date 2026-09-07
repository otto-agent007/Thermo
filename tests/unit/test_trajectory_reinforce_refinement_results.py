"""Strict persistence for one-step trajectory refinement evidence."""

from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from thermo_lab.backends.numpy_exact_categorical import sample_augmented_gradient_sources
from thermo_lab.hashing import canonical_sha256
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference
from thermo_lab.trajectory_reinforce_refinement import build_one_step_refinement
from thermo_lab.trajectory_reinforce_results import (
    TrajectoryReinforceSummary,
    build_trajectory_reinforce_deterministic_result,
    build_trajectory_reinforce_sample_result,
    build_trajectory_reinforce_summary,
)


def _estimator_summary(*, ascent: bool = False) -> TrajectoryReinforceSummary:
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    deterministic = build_trajectory_reinforce_deterministic_result(
        request_hash=canonical_sha256({"request": "one-step-refinement"}),
        fixture=fixture,
        exact=exact,
    )
    sources = sample_augmented_gradient_sources(
        fixture=fixture,
        exact=exact,
        batch_size=17,
        seed=2,
    )
    occurrence_0 = sources.occurrence_0
    occurrence_1 = sources.occurrence_1
    if ascent:
        occurrence_0 = occurrence_0.model_copy(
            update={"component_sum": tuple(-value for value in occurrence_0.component_sum)}
        )
        occurrence_1 = occurrence_1.model_copy(
            update={"component_sum": tuple(-value for value in occurrence_1.component_sum)}
        )
    sample = build_trajectory_reinforce_sample_result(
        deterministic_result_digest=deterministic.deterministic_result_digest,
        seed=2,
        sample_definition="one checked augmented training batch",
        occurrence_0=occurrence_0,
        occurrence_1=occurrence_1,
        cross_products=sources.cross_products,
        exact=deterministic.expected_reference,
    )
    return build_trajectory_reinforce_summary(deterministic=deterministic, sample=sample)


def test_refinement_summary_round_trips_exact_objective_evidence() -> None:
    results = importlib.import_module("thermo_lab.trajectory_reinforce_refinement_results")
    estimator = _estimator_summary()
    fixture = build_checked_fixture()
    refinement = build_one_step_refinement(
        fixture=fixture,
        exact=build_exact_reference(fixture),
        sampled_shared_gradient=estimator.sample.shared.mean,
        learning_rate=0.25,
    )

    summary = results.build_trajectory_reinforce_refinement_summary(
        estimator=estimator,
        refinement=refinement,
    )

    assert summary.refinement.objective_before == refinement.objective_before
    assert summary.refinement.objective_after == refinement.objective_after
    assert summary.refinement.objective_improved is refinement.objective_improved
    assert summary.refinement.bounds_satisfied is True
    assert (
        results.validate_trajectory_reinforce_refinement_summary(summary.model_dump_json())
        == summary
    )


def test_refinement_result_rejects_malformed_sha256_digest() -> None:
    results = importlib.import_module("thermo_lab.trajectory_reinforce_refinement_results")
    estimator = _estimator_summary()
    fixture = build_checked_fixture()
    refinement = build_one_step_refinement(
        fixture=fixture,
        exact=build_exact_reference(fixture),
        sampled_shared_gradient=estimator.sample.shared.mean,
        learning_rate=0.25,
    )
    summary = results.build_trajectory_reinforce_refinement_summary(
        estimator=estimator,
        refinement=refinement,
    )
    payload = summary.refinement.model_dump(mode="python")
    payload["refinement_digest"] = "sha256:" + "g" * 64

    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        results.TrajectoryReinforceRefinementResult.model_validate(payload)


def test_refinement_summary_retains_non_improving_exact_outcome() -> None:
    results = importlib.import_module("thermo_lab.trajectory_reinforce_refinement_results")
    estimator = _estimator_summary(ascent=True)
    fixture = build_checked_fixture()
    refinement = build_one_step_refinement(
        fixture=fixture,
        exact=build_exact_reference(fixture),
        sampled_shared_gradient=estimator.sample.shared.mean,
        learning_rate=0.25,
    )

    summary = results.build_trajectory_reinforce_refinement_summary(
        estimator=estimator,
        refinement=refinement,
    )

    assert summary.refinement.objective_before == pytest.approx(0.5246570826850282)
    assert summary.refinement.objective_after > summary.refinement.objective_before
    assert summary.refinement.objective_improvement < 0.0
    assert summary.refinement.objective_improved is False
    assert summary.refinement.bounds_satisfied is True
    assert summary.acceptance_passed is False
