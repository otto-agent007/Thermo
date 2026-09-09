"""Strict evidence contract for one-step composed trajectory refinement."""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

from thermo_lab.composed_trajectory_refinement import (
    _schedule_digest,
    estimate_equilibrium_grouped_gradient,
    evaluate_paired_equilibrium_objective,
    project_grouped_parameters,
    sample_equilibrium_terminal_occupancy,
)
from thermo_lab.hashing import canonical_sha256
from thermo_lab.trajectory_reinforce import build_checked_fixture


def _results_module():
    spec = importlib.util.find_spec("thermo_lab.composed_trajectory_refinement_results")
    assert spec is not None
    return importlib.import_module("thermo_lab.composed_trajectory_refinement_results")


def _micro_summary():
    results = _results_module()
    fixture = build_checked_fixture()
    parameters = np.asarray((fixture.model_parameters.values,), dtype=np.float64)
    target_indices = np.asarray((0, 0), dtype=np.int16)
    site_indices = np.asarray(((0, 1), (1, 2)), dtype=np.int8)
    occupancy = sample_equilibrium_terminal_occupancy(
        parameters,
        target_indices,
        site_indices,
        site_count=3,
        batch_size=4096,
        seed=101,
        beta=fixture.beta,
    )
    target_occupancy = occupancy.occupancy
    reward = tuple(0.0 for _ in range(3))
    gradient = estimate_equilibrium_grouped_gradient(
        parameters,
        target_indices,
        site_indices,
        reward,
        site_count=3,
        batch_size=4096,
        seed=202,
        beta=fixture.beta,
    )
    update = project_grouped_parameters(
        parameters,
        gradient.mean,
        learning_rate=0.01,
        parameter_cap=fixture.parameter_cap,
    )
    evaluation = evaluate_paired_equilibrium_objective(
        parameters,
        update.updated_parameters,
        target_indices,
        site_indices,
        target_occupancy,
        site_count=3,
        batch_size=4096,
        seed=303,
        beta=fixture.beta,
    )
    bundle_digest = canonical_sha256({"bundle": "micro"})
    target_reference = canonical_sha256({"target": target_occupancy})
    summary = results.build_composed_trajectory_refinement_summary(
        request_hash=canonical_sha256({"request": "micro"}),
        seed=7,
        source_bundle_digest=bundle_digest,
        exact_target_reference=target_reference,
        beta=fixture.beta,
        schedule_digest=_schedule_digest(target_indices, site_indices, site_count=3),
        initial_parameters=parameters,
        target_occupancy=target_occupancy,
        occupancy_seed=101,
        gradient_seed=202,
        evaluation_seed=303,
        occupancy_source=occupancy,
        reward_coefficient=reward,
        gradient_source=gradient,
        learning_rate=0.01,
        parameter_cap=fixture.parameter_cap,
        update=update,
        evaluation=evaluation,
    )
    return results, summary, parameters, target_occupancy, bundle_digest, target_reference


def test_summary_round_trips_reconstructed_evidence() -> None:
    results, summary, parameters, target, bundle_digest, target_reference = _micro_summary()

    reloaded = results.validate_composed_trajectory_refinement_summary(
        summary.model_dump_json(),
        expected_bundle_digest=bundle_digest,
        expected_initial_parameters=parameters,
        expected_target_occupancy=target,
        expected_target_reference=target_reference,
    )

    assert reloaded == summary
    assert summary.integrity_acceptance_passed is True
    assert summary.evaluation.objective_improved is False
    assert summary.evaluation.objective_improvement == 0.0


def test_scientific_non_improvement_is_not_an_integrity_failure() -> None:
    _, summary, *_ = _micro_summary()

    assert summary.evaluation.objective_improved is False
    assert summary.integrity_acceptance_passed is True


def test_summary_rejects_tampered_reward_even_with_top_digest_removed() -> None:
    results, summary, parameters, target, bundle_digest, target_reference = _micro_summary()
    payload = summary.model_dump(mode="json")
    reward = list(payload["reward_coefficient"])
    reward[0] = 0.125
    payload["reward_coefficient"] = reward
    payload["summary_digest"] = results.composed_trajectory_refinement_summary_digest(payload)

    with pytest.raises(ValueError, match="reward coefficient"):
        results.validate_composed_trajectory_refinement_summary(
            payload,
            expected_bundle_digest=bundle_digest,
            expected_initial_parameters=parameters,
            expected_target_occupancy=target,
            expected_target_reference=target_reference,
        )


def test_summary_rejects_tampered_gradient_mean_after_redigest() -> None:
    results, summary, parameters, target, bundle_digest, target_reference = _micro_summary()
    payload = summary.model_dump(mode="json")
    mean = [list(row) for row in payload["gradient_mean"]]
    mean[0][0] = 0.5
    payload["gradient_mean"] = mean
    payload["summary_digest"] = results.composed_trajectory_refinement_summary_digest(payload)

    with pytest.raises(ValueError, match="gradient mean"):
        results.validate_composed_trajectory_refinement_summary(
            payload,
            expected_bundle_digest=bundle_digest,
            expected_initial_parameters=parameters,
            expected_target_occupancy=target,
            expected_target_reference=target_reference,
        )


@pytest.mark.parametrize("source", ["occupancy_source", "gradient_source", "before", "after"])
def test_summary_rejects_forged_source_digest_after_redigest(source) -> None:
    results, summary, parameters, target, bundle_digest, target_reference = _micro_summary()
    payload = summary.model_dump(mode="json")
    container = payload["evaluation"] if source in ("before", "after") else payload
    container[source]["source_digest"] = "sha256:" + "0" * 64
    payload["summary_digest"] = results.composed_trajectory_refinement_summary_digest(payload)
    with pytest.raises(ValueError, match="source digest"):
        results.validate_composed_trajectory_refinement_summary(
            payload,
            expected_bundle_digest=bundle_digest,
            expected_initial_parameters=parameters,
            expected_target_occupancy=target,
            expected_target_reference=target_reference,
        )


def test_summary_rejects_forged_evaluation_digest_after_redigest() -> None:
    results, summary, parameters, target, bundle_digest, target_reference = _micro_summary()
    payload = summary.model_dump(mode="json")
    payload["evaluation"]["result_digest"] = "sha256:" + "0" * 64
    payload["summary_digest"] = results.composed_trajectory_refinement_summary_digest(payload)
    with pytest.raises(ValueError, match="evaluation digest"):
        results.validate_composed_trajectory_refinement_summary(
            payload,
            expected_bundle_digest=bundle_digest,
            expected_initial_parameters=parameters,
            expected_target_occupancy=target,
            expected_target_reference=target_reference,
        )


def test_gradient_moments_reject_impossible_variance() -> None:
    results = _results_module()
    with pytest.raises(ValueError, match="moments"):
        results.GroupedGradientResult(
            sample_count=2,
            component_sum=((1.0,) * 9,),
            component_sum_squares=((0.0,) * 9,),
            source_digest="sha256:" + "0" * 64,
        )
