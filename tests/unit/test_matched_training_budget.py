"""Catch arm substitution, evaluation reuse, broken chains and forged evidence."""

from dataclasses import asdict

import numpy as np
import pytest

from thermo_lab.composed_trajectory_refinement import (
    estimate_equilibrium_grouped_gradient,
    sample_equilibrium_terminal_occupancy,
)
from thermo_lab.finite_sweep_sampling import (
    estimate_finite_sweep_grouped_gradient,
    evaluate_finite_sweep_pair,
    sample_finite_sweep_terminal_occupancy,
)
from thermo_lab.hashing import to_json_value
from thermo_lab.trajectory_reinforce import build_checked_fixture, terminal_law


@pytest.fixture(scope="module")
def comparison():
    from thermo_lab.matched_training_budget import run_comparison

    fixture = build_checked_fixture()
    target = terminal_law((fixture.target_conditional,) * 2).occupancy
    return run_comparison(
        (fixture.model_parameters.values,), (0, 0), fixture.occurrences, target, seed=0
    )


def test_equilibrium_arm_uses_equilibrium_law_and_both_arms_project_five_updates(comparison):
    groups = comparison.occurrence_target_indices
    sites = comparison.occurrence_site_indices
    eq = comparison.arms[1]
    initial = comparison.initial_parameters
    first = eq.steps[0]
    common = dict(site_count=3, batch_size=32768, beta=1.0)
    occupancy = sample_equilibrium_terminal_occupancy(
        initial, groups, sites, seed=first.occupancy_seed, **common
    )
    assert first.occupancy.model_dump(mode="json") == to_json_value(asdict(occupancy))
    gradient = estimate_equilibrium_grouped_gradient(
        initial, groups, sites, first.reward_coefficient, seed=first.gradient_seed, **common
    )
    np.testing.assert_allclose(first.gradient.component_sum, gradient.component_sum, atol=1e-10)
    assert comparison.arms[0].steps[0].exact_tables_digest != first.exact_tables_digest
    for arm in comparison.arms:
        assert len(arm.steps) == 5
        current = np.array(initial)
        for step in arm.steps:
            raw = current - 0.01 * np.array(step.gradient.mean)
            np.testing.assert_array_equal(step.update.raw_parameters, raw)
            current = np.clip(raw, -2, 2)
            np.testing.assert_array_equal(step.update.updated_parameters, current)


def test_primary_comparison_is_finite_minus_equilibrium_at_k4(comparison):
    expected = evaluate_finite_sweep_pair(
        comparison.arms[1].steps[-1].update.updated_parameters,
        comparison.arms[0].steps[-1].update.updated_parameters,
        comparison.occurrence_target_indices,
        comparison.occurrence_site_indices,
        site_count=3,
        batch_size=32768,
        beta=1.0,
        horizon=4,
        seed=comparison.evaluation_seed,
    )
    assert comparison.primary_evaluation == expected
    assert comparison.initial_finite.before_counts == comparison.initial_equilibrium.before_counts
    assert comparison.initial_finite.after_counts == expected.after_counts
    assert comparison.initial_equilibrium.after_counts == expected.before_counts


def test_finite_arm_uses_checked_k4_occupancy_and_gradient(comparison):
    step = comparison.arms[0].steps[0]
    common = dict(site_count=3, batch_size=32768, beta=1.0, horizon=4)
    args = (
        comparison.initial_parameters,
        comparison.occurrence_target_indices,
        comparison.occurrence_site_indices,
    )
    occupancy = sample_finite_sweep_terminal_occupancy(*args, seed=step.occupancy_seed, **common)
    assert step.occupancy.model_dump(mode="json") == to_json_value(asdict(occupancy))
    reward = tuple(
        2 * (a - b) for a, b in zip(occupancy.occupancy, comparison.target_occupancy, strict=True)
    )
    assert step.reward_coefficient == reward
    gradient = estimate_finite_sweep_grouped_gradient(
        *args, reward, seed=step.gradient_seed, **common
    )
    for name in ("component_sum", "component_sum_squares"):
        np.testing.assert_array_equal(getattr(step.gradient, name), getattr(gradient, name))


@pytest.mark.parametrize("repair_source", [False, True])
def test_equilibrium_second_moment_replay_and_exact_digest_binding(comparison, repair_source):
    from thermo_lab.composed_trajectory_refinement import _schedule_digest
    from thermo_lab.hashing import canonical_sha256
    from thermo_lab.matched_training_budget import TrainingComparison, comparison_digest

    p = comparison.model_dump(mode="json")
    step = p["arms"][1]["steps"][0]
    gradient = step["gradient"]
    # Large, rehashed forgery must fail numerical replay; a sub-tolerance
    # difference must still fail the exact source binding if the digest is stale.
    gradient["component_sum_squares"][0][0] += 1.0 if repair_source else 1e-11
    if repair_source:
        gradient["source_digest"] = canonical_sha256(
            {
                "identity_version": "composed_equilibrium_grouped_gradient_source.v1",
                "sample_count": 32768,
                "component_sum": gradient["component_sum"],
                "component_sum_squares": gradient["component_sum_squares"],
                "seed": step["gradient_seed"],
                "beta": 1.0,
                "parameter_digest": canonical_sha256(comparison.initial_parameters),
                "schedule_digest": _schedule_digest(
                    comparison.occurrence_target_indices,
                    comparison.occurrence_site_indices,
                    site_count=3,
                ),
                "reward_coefficient": step["reward_coefficient"],
                "reference_policy": "independent_same_parent_non_propagated",
            }
        )
    p["result_digest"] = comparison_digest(p)
    with pytest.raises(ValueError, match="replay" if repair_source else "digest.*stored"):
        TrainingComparison.model_validate(p)


def test_roles_are_disjoint_across_arms_seeds_and_historical_m4(comparison):
    from thermo_lab.finite_refinement_sequence import spawn_finite_refinement_roles
    from thermo_lab.matched_training_budget import spawn_comparison_roles

    roles = [r for seed in (0, 1, 2) for r in spawn_comparison_roles(seed)]
    old = [r for seed in (0, 1, 2) for r in spawn_finite_refinement_roles(seed)]
    assert len(set(roles)) == 63
    assert not set(roles).intersection(old)
    stored = [
        r for arm in comparison.arms for s in arm.steps for r in (s.occupancy_seed, s.gradient_seed)
    ] + [comparison.evaluation_seed]
    assert tuple(stored) == spawn_comparison_roles(0)


@pytest.mark.parametrize("mutation", ["arm", "seed", "gradient", "chain", "final", "budget"])
def test_rehashed_forged_comparison_rejected(comparison, mutation):
    from thermo_lab.matched_training_budget import TrainingComparison, comparison_digest

    p = comparison.model_dump(mode="json")
    if mutation == "arm":
        p["arms"].reverse()
    elif mutation == "seed":
        p["evaluation_seed"] = p["arms"][0]["steps"][0]["occupancy_seed"]
    elif mutation == "gradient":
        p["arms"][1]["steps"][0]["gradient"]["component_sum"][0][0] += 1.0
    elif mutation == "chain":
        p["arms"][1]["steps"].reverse()
    elif mutation == "final":
        p["primary_evaluation"] = p["initial_finite"]
    else:
        p["protocol"]["step_count"] = 4
    p["result_digest"] = comparison_digest(p)
    with pytest.raises(ValueError):
        TrainingComparison.model_validate(p)


def test_comparison_round_trip_and_bounded_work_accounting(comparison):
    from thermo_lab.matched_training_budget import TrainingComparison, declared_work

    assert TrainingComparison.model_validate_json(comparison.model_dump_json()) == comparison
    work = declared_work(500)
    assert work["training_endpoint_draws_per_arm"] == 245760000
    assert work["three_pair_evaluation_endpoint_draws"] == 98304000
    assert work["finite_training_complete_sweep_equivalents"] == 983040000
    assert work["equilibrium_training_complete_sweep_equivalents"] is None


@pytest.mark.parametrize("seed", [True, -1, 3])
def test_reject_unchecked_seed(seed):
    from thermo_lab.matched_training_budget import spawn_comparison_roles

    with pytest.raises(ValueError):
        spawn_comparison_roles(seed)
