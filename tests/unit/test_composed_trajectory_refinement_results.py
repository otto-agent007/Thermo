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


def test_v2_persists_population_audit_without_changing_legacy_fixture() -> None:
    _, summary, *_ = _micro_summary()
    assert summary.result_schema_version == "2.0.0"
    evaluation = summary.evaluation
    assert evaluation.objective_before == 0.0005123019218444824
    assert evaluation.objective_after == 0.0005123019218444824
    assert evaluation.objective_improvement == 0.0
    assert evaluation.objective_improved is False
    assert summary.update.update_digest == (
        "sha256:14f7e96faad7418381edef6a14a6d36b313a1128b04a68efc1f3abc0ae758af8"
    )
    assert evaluation.population_objective_before < evaluation.objective_before
    assert evaluation.population_objective_after == evaluation.population_objective_before
    assert evaluation.population_objective_difference_after_minus_before == 0.0
    assert evaluation.paired_jackknife_standard_error == 0.0
    assert evaluation.paired_jackknife_normal_95_interval == (0.0, 0.0)
    assert evaluation.population_objective_conclusion == "inconclusive"
    assert len(evaluation.joined_terminal_second_moment_counts) == 6
    payload = summary.model_dump(mode="json")
    _redigest_evaluation(payload)
    assert payload["evaluation"]["result_digest"] == evaluation.result_digest


def _redigest_evaluation(payload):
    evaluation = payload["evaluation"]
    digest_payload = {
        key: value
        for key, value in evaluation.items()
        if key not in {"before", "after", "result_digest"}
    }
    digest_payload.update(
        identity_version="composed_equilibrium_paired_objective.v2",
        before_source_digest=evaluation["before"]["source_digest"],
        after_source_digest=evaluation["after"]["source_digest"],
        common_random_numbers=True,
    )
    evaluation["result_digest"] = canonical_sha256(digest_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("population_objective_before", 0.5),
        ("population_objective_after", 0.5),
        ("population_objective_difference_after_minus_before", -0.5),
        ("paired_jackknife_standard_error", 0.5),
        ("paired_jackknife_normal_95_interval", [-0.5, 0.5]),
        ("population_objective_conclusion", "improved"),
    ],
)
def test_population_audit_rejects_tampering_after_both_outer_redigests(field, value):
    results, summary, parameters, target, bundle_digest, target_reference = _micro_summary()
    payload = summary.model_dump(mode="json")
    assert field in payload["evaluation"]
    payload["evaluation"][field] = value
    _redigest_evaluation(payload)
    payload["summary_digest"] = results.composed_trajectory_refinement_summary_digest(payload)
    with pytest.raises(ValueError, match=field):
        results.validate_composed_trajectory_refinement_summary(
            payload,
            expected_bundle_digest=bundle_digest,
            expected_initial_parameters=parameters,
            expected_target_occupancy=target,
            expected_target_reference=target_reference,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("shape", "square"),
        ("symmetry", "symmetric"),
        ("diagonal", "diagonal"),
        ("lower", "count bounds"),
        ("upper", "count bounds"),
        ("boolean", "integer"),
    ],
)
def test_persisted_joined_moments_reject_impossible_counts(mutation, message):
    results, summary, *_ = _micro_summary()
    payload = summary.evaluation.model_dump(mode="json")
    assert "joined_terminal_second_moment_counts" in payload
    moments = payload["joined_terminal_second_moment_counts"]
    if mutation == "shape":
        moments.pop()
    elif mutation == "symmetry":
        moments[0][1] += 1
    elif mutation == "diagonal":
        moments[0][0] -= 1
    elif mutation == "lower":
        # count[0] + count[3] - n = 2222, so zero violates Frechet.
        moments[0][3] = moments[3][0] = 0
    elif mutation == "upper":
        moments[0][1] = moments[1][0] = 4096
    else:
        moments[0][0] = True
    with pytest.raises(ValueError, match=message):
        results.PairedObjectiveResult.model_validate(payload)


def test_v1_payload_is_not_silently_reinterpreted_as_a_population_audit():
    results, summary, *_ = _micro_summary()
    payload = summary.model_dump(mode="json")
    payload["result_schema_version"] = "1.0.0"
    with pytest.raises(ValueError, match="result_schema_version"):
        results.ComposedTrajectoryRefinementSummary.model_validate(payload)


def test_aggregate_scalars_expose_both_estimators_but_keep_jackknife_nested():
    from thermo_lab.composed_trajectory_refinement_reporting import (
        REFINEMENT_SCALARS,
        refinement_metric_observations,
    )

    _, summary, *_ = _micro_summary()
    metrics = refinement_metric_observations(summary)
    for name in (
        "population_objective_before",
        "population_objective_after",
        "population_objective_difference_after_minus_before",
    ):
        assert name in REFINEMENT_SCALARS
        assert metrics[name].value == getattr(summary.evaluation, name)
        assert metrics[name].evidence_class.value == "software_simulation"
        assert "unbiased" in metrics[name].method
    assert "paired_jackknife_standard_error" not in metrics
    assert "paired_jackknife_normal_95_interval" not in metrics
    assert "plug-in" in metrics["objective_before"].method


def test_valid_joined_moment_edit_is_bound_by_the_evaluation_digest():
    results, summary, parameters, target, bundle_digest, target_reference = _micro_summary()
    payload = summary.model_dump(mode="json")
    moments = payload["evaluation"]["joined_terminal_second_moment_counts"]
    # Preserve identical paired trajectories and all count bounds, but alter
    # their cross-site joint occupancy consistently in every block.
    for first, second in ((0, 1), (3, 4), (0, 4), (3, 1)):
        moments[first][second] -= 1
        moments[second][first] -= 1
    results.PairedObjectiveResult.model_validate(payload["evaluation"])
    payload["summary_digest"] = results.composed_trajectory_refinement_summary_digest(payload)
    with pytest.raises(ValueError, match="evaluation digest"):
        results.validate_composed_trajectory_refinement_summary(
            payload,
            expected_bundle_digest=bundle_digest,
            expected_initial_parameters=parameters,
            expected_target_occupancy=target,
            expected_target_reference=target_reference,
        )


def test_summary_rejects_impossible_single_block_moments_after_both_outer_redigests():
    results, summary, parameters, target, bundle_digest, target_reference = _micro_summary()
    payload = summary.model_dump(mode="json")
    moments = payload["evaluation"]["joined_terminal_second_moment_counts"]
    # Unlike the valid edit above, only one block changes: equal before/after
    # columns now contradict each other. Counts and all six derived values stay fixed.
    moments[0][1] -= 1
    moments[1][0] -= 1
    _redigest_evaluation(payload)
    payload["summary_digest"] = results.composed_trajectory_refinement_summary_digest(payload)
    with pytest.raises(ValueError, match="identical columns.*identical moment rows"):
        results.validate_composed_trajectory_refinement_summary(
            payload,
            expected_bundle_digest=bundle_digest,
            expected_initial_parameters=parameters,
            expected_target_occupancy=target,
            expected_target_reference=target_reference,
        )


@pytest.mark.parametrize(
    "policy",
    [
        "population_objective_estimator_policy",
        "paired_uncertainty_policy",
        "population_objective_conclusion_policy",
        "improvement_policy",
    ],
)
def test_persisted_audit_rejects_policy_reinterpretation_after_redigest(policy):
    results, summary, *_ = _micro_summary()
    payload = summary.model_dump(mode="json")
    payload["evaluation"][policy] = "unchecked"
    _redigest_evaluation(payload)
    payload["summary_digest"] = results.composed_trajectory_refinement_summary_digest(payload)
    with pytest.raises(ValueError, match=policy):
        results.ComposedTrajectoryRefinementSummary.model_validate(payload)


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
