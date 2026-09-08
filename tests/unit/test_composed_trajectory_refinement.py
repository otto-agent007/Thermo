from __future__ import annotations

import numpy as np

import thermo_lab.composed_pasym_swap as composed
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference


def _micro_schedule() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((0, 0), dtype=np.int16),
        np.asarray(((0, 1), (1, 2)), dtype=np.int8),
    )


def test_project_grouped_parameters_clips_and_records_caps() -> None:
    assert hasattr(composed, "project_grouped_parameters")
    initial = np.zeros((2, 9), dtype=np.float64)
    initial[0, 0] = 1.95
    gradient = np.zeros((2, 9), dtype=np.float64)
    gradient[0, 0] = -1.0
    gradient[1, 1] = 2.0

    result = composed.project_grouped_parameters(
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
    assert hasattr(composed, "sample_equilibrium_terminal_occupancy")
    fixture = build_checked_fixture()
    target_indices, site_indices = _micro_schedule()
    parameters = np.asarray((fixture.model_parameters.values,), dtype=np.float64)

    first = composed.sample_equilibrium_terminal_occupancy(
        parameters,
        target_indices,
        site_indices,
        site_count=3,
        batch_size=4096,
        seed=17,
        beta=fixture.beta,
    )
    second = composed.sample_equilibrium_terminal_occupancy(
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
    assert hasattr(composed, "estimate_equilibrium_grouped_gradient")
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    target_indices, site_indices = _micro_schedule()
    parameters = np.asarray((fixture.model_parameters.values,), dtype=np.float64)

    result = composed.estimate_equilibrium_grouped_gradient(
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
    assert hasattr(composed, "evaluate_paired_equilibrium_objective")
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    target_indices, site_indices = _micro_schedule()
    parameters = np.asarray((fixture.model_parameters.values,), dtype=np.float64)

    result = composed.evaluate_paired_equilibrium_objective(
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
    assert result.objective_before == result.objective_after
    assert result.objective_improvement == 0.0
    assert result.objective_improved is False
    assert result.result_digest.startswith("sha256:")
