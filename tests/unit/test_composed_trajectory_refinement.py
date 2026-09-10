from __future__ import annotations

import importlib
import importlib.util
import itertools
import math

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference


def _refinement_module():
    spec = importlib.util.find_spec("thermo_lab.composed_trajectory_refinement")
    assert spec is not None
    return importlib.import_module("thermo_lab.composed_trajectory_refinement")


def _micro_schedule() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((0, 0), dtype=np.int16),
        np.asarray(((0, 1), (1, 2)), dtype=np.int8),
    )


def test_population_objective_is_unbiased_at_exact_bernoulli_target() -> None:
    refinement = _refinement_module()
    estimates = []
    plugin_losses = []

    for sample in itertools.product((0, 1), repeat=2):
        count = sum(sample)
        estimates.append(
            refinement.calculate_unbiased_population_objective(
                (count,),
                sample_count=2,
                target_occupancy=(0.5,),
            )
        )
        plugin_losses.append((count / 2 - 0.5) ** 2)

    assert estimates == [0.25, -0.25, -0.25, 0.25]
    assert math.fsum(estimates) / 4 == 0.0
    assert math.fsum(plugin_losses) / 4 == 0.125


def test_population_objective_avoids_integer_overflow_for_large_valid_counts() -> None:
    refinement = _refinement_module()

    result = refinement.calculate_unbiased_population_objective(
        (4_000_000_000,),
        sample_count=4_000_000_000,
        target_occupancy=(0.0,),
    )

    assert result == 1.0


def test_population_estimators_require_enough_independent_trajectories() -> None:
    refinement = _refinement_module()

    with pytest.raises(ValueError, match="sample_count must be at least 2"):
        refinement.calculate_unbiased_population_objective(
            (1,),
            sample_count=1,
            target_occupancy=(0.5,),
        )
    with pytest.raises(
        ValueError,
        match="sample_count must be at least 3 for paired jackknife uncertainty",
    ):
        refinement.calculate_paired_population_objective_statistics(
            (1,),
            (1,),
            ((1, 1), (1, 1)),
            sample_count=2,
            target_occupancy=(0.5,),
        )


def test_paired_jackknife_preserves_pairing_and_cross_site_covariance() -> None:
    refinement = _refinement_module()
    identical_moments = (
        (2, 2, 2, 2),
        (2, 2, 2, 2),
        (2, 2, 2, 2),
        (2, 2, 2, 2),
    )
    changed_moments = (
        (2, 2, 2, 0),
        (2, 2, 2, 0),
        (2, 2, 2, 0),
        (0, 0, 0, 2),
    )

    identical = refinement.calculate_paired_population_objective_statistics(
        (2, 2),
        (2, 2),
        identical_moments,
        sample_count=4,
        target_occupancy=(0.0, 0.0),
    )
    changed = refinement.calculate_paired_population_objective_statistics(
        (2, 2),
        (2, 2),
        changed_moments,
        sample_count=4,
        target_occupancy=(0.0, 0.0),
    )

    assert identical.population_objective_difference_after_minus_before == 0.0
    assert identical.paired_jackknife_standard_error == 0.0
    assert identical.paired_jackknife_normal_95_interval == (0.0, 0.0)
    assert identical.population_objective_conclusion == "inconclusive"
    assert changed.population_objective_difference_after_minus_before == 0.0
    assert changed.paired_jackknife_standard_error == pytest.approx(math.sqrt(1.0 / 3.0))
    assert changed.paired_jackknife_normal_95_interval == pytest.approx(
        (-1.1315857340761717, 1.1315857340761717)
    )
    assert changed.population_objective_conclusion == "inconclusive"


@pytest.mark.parametrize(
    ("before_counts", "after_counts", "moments", "expected_difference", "conclusion"),
    (
        ((0,), (4,), ((0, 0), (0, 4)), -1.0, "improved"),
        ((4,), (0,), ((4, 0), (0, 0)), 1.0, "regressed"),
    ),
)
def test_paired_population_conclusion_uses_strict_interval_sign(
    before_counts: tuple[int, ...],
    after_counts: tuple[int, ...],
    moments: tuple[tuple[int, ...], ...],
    expected_difference: float,
    conclusion: str,
) -> None:
    refinement = _refinement_module()

    result = refinement.calculate_paired_population_objective_statistics(
        before_counts,
        after_counts,
        moments,
        sample_count=4,
        target_occupancy=(1.0,),
    )

    assert result.population_objective_difference_after_minus_before == expected_difference
    assert result.paired_jackknife_standard_error == 0.0
    assert result.paired_jackknife_normal_95_interval == (
        expected_difference,
        expected_difference,
    )
    assert result.population_objective_conclusion == conclusion


def test_project_grouped_parameters_clips_and_records_caps() -> None:
    refinement = _refinement_module()
    initial = np.zeros((2, 9), dtype=np.float64)
    initial[0, 0] = 1.95
    gradient = np.zeros((2, 9), dtype=np.float64)
    gradient[0, 0] = -1.0
    gradient[1, 1] = 2.0

    result = refinement.project_grouped_parameters(
        initial,
        gradient,
        learning_rate=0.1,
        parameter_cap=2.0,
    )

    assert result.raw_parameters[0][0] == 2.05
    assert result.updated_parameters[0][0] == 2.0
    assert result.updated_parameters[1][1] == -0.2
    assert result.cap_active_parameter_count == 1
    assert result.bounds_satisfied is True


def test_equilibrium_terminal_occupancy_is_seed_deterministic() -> None:
    refinement = _refinement_module()
    fixture = build_checked_fixture()
    target_indices, site_indices = _micro_schedule()
    parameters = np.asarray((fixture.model_parameters.values,), dtype=np.float64)

    first = refinement.sample_equilibrium_terminal_occupancy(
        parameters,
        target_indices,
        site_indices,
        site_count=3,
        batch_size=4096,
        seed=17,
        beta=fixture.beta,
    )
    second = refinement.sample_equilibrium_terminal_occupancy(
        parameters,
        target_indices,
        site_indices,
        site_count=3,
        batch_size=4096,
        seed=17,
        beta=fixture.beta,
    )

    assert first == second
    assert first.sample_count == 4096
    assert len(first.occupancy_counts) == 3
    assert first.source_digest.startswith("sha256:")


def test_grouped_gradient_matches_exact_three_site_reference() -> None:
    refinement = _refinement_module()
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    target_indices, site_indices = _micro_schedule()
    parameters = np.asarray((fixture.model_parameters.values,), dtype=np.float64)

    result = refinement.estimate_equilibrium_grouped_gradient(
        parameters,
        target_indices,
        site_indices,
        exact.reward_coefficient,
        site_count=3,
        batch_size=65536,
        seed=23,
        beta=fixture.beta,
    )
    sampled = np.asarray(result.component_sum, dtype=np.float64)[0] / result.sample_count

    assert np.max(np.abs(sampled - exact.expected_reference.shared)) < 0.03
    assert result.sample_count == 65536
    assert result.source_digest.startswith("sha256:")


def test_paired_objective_uses_common_random_numbers() -> None:
    refinement = _refinement_module()
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    target_indices, site_indices = _micro_schedule()
    parameters = np.asarray((fixture.model_parameters.values,), dtype=np.float64)

    result = refinement.evaluate_paired_equilibrium_objective(
        parameters,
        parameters.copy(),
        target_indices,
        site_indices,
        exact.target_law.occupancy,
        site_count=3,
        batch_size=4096,
        seed=31,
        beta=fixture.beta,
    )

    assert result.before.occupancy_counts == result.after.occupancy_counts
    assert result.objective_before == 0.5196119244151355
    assert result.objective_after == 0.5196119244151355
    assert result.objective_improvement == 0.0
    assert result.objective_improved is False
    assert (
        tuple(result.joined_terminal_second_moment_counts[index][index] for index in range(6))
        == result.before.occupancy_counts + result.after.occupancy_counts
    )
    assert result.population_objective_before == result.population_objective_after
    assert result.population_objective_difference_after_minus_before == 0.0
    assert result.paired_jackknife_standard_error == 0.0
    assert result.paired_jackknife_normal_95_interval == (0.0, 0.0)
    assert result.population_objective_conclusion == "inconclusive"
    assert result.result_digest == canonical_sha256(
        {
            "identity_version": "composed_equilibrium_paired_objective.v1",
            "before_source_digest": result.before.source_digest,
            "after_source_digest": result.after.source_digest,
            "target_occupancy": result.target_occupancy,
            "objective_before": result.objective_before,
            "objective_after": result.objective_after,
            "objective_improvement": result.objective_improvement,
            "objective_improved": result.objective_improved,
            "common_random_numbers": True,
        }
    )
