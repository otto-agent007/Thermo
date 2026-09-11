import numpy as np
import pytest

from thermo_lab.finite_sweep_gradient_reference import build_finite_sweep_reference


@pytest.mark.parametrize("invalid", (True, 1.0))
def test_warm_exact_reference_cache_cannot_bypass_horizon_type(invalid):
    from thermo_lab.finite_sweep_estimator_reference import exact_estimator_moments

    exact_estimator_moments(horizon=1)
    with pytest.raises(ValueError):
        exact_estimator_moments(horizon=invalid)


@pytest.mark.parametrize("horizon", (1, 2, 4, 8, 16, 30))
def test_exact_sampled_estimator_mean_matches_all_m3_oracles(horizon):
    from thermo_lab.finite_sweep_estimator_reference import exact_estimator_moments

    result = exact_estimator_moments(horizon)
    reference = build_finite_sweep_reference(horizon)
    np.testing.assert_allclose(result.occurrence_mean, reference.autodiff.occurrences, atol=1e-12)
    np.testing.assert_allclose(result.mean, reference.finite_difference_tied, atol=1e-7)
    assert np.all(np.asarray(result.second_moment) >= np.asarray(result.mean) ** 2)


def test_checked_sampler_diagnostics_cover_all_seeds_and_horizons():
    from thermo_lab.finite_sweep_estimator_reference import checked_sampler_diagnostics

    checks = checked_sampler_diagnostics()
    assert [(c.horizon, c.seed) for c in checks] == [
        (k, s) for k in (1, 2, 4, 8, 16, 30) for s in (0, 1, 2)
    ]
    assert all(c.sample.sample_count == 32768 for c in checks)
    assert len({c.sampling_seed for c in checks}) == 18
    assert all(np.isfinite(c.maximum_standardized_mean_error) for c in checks)


def test_exact_second_moments_match_independent_exhaustive_reference_draws():
    from thermo_lab.finite_sweep_estimator_reference import exact_estimator_moments
    from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
    from thermo_lab.trajectory_reinforce import build_checked_fixture

    fixture = build_checked_fixture()
    law = finite_sweep_joint_law(fixture.model_parameters, 1)
    reference = build_finite_sweep_reference(1)
    reward = 2 * (np.asarray(reference.model_law.occupancy) - reference.target_law.occupancy)
    mean = np.zeros(9)
    second = np.zeros(9)
    for first in range(8):
        left, middle = divmod(first % 4, 2)
        parent = 2 * middle
        for last in range(8):
            end_middle, right = divmod(last % 4, 2)
            value = reward @ (left, end_middle, right)
            for first_ref in range(8):
                for last_ref in range(8):
                    p = law.probabilities[2, first] * law.probabilities[parent, last]
                    p *= law.probabilities[2, first_ref] * law.probabilities[parent, last_ref]
                    g = value * (
                        law.scores[2, first]
                        - law.scores[2, first_ref]
                        + law.scores[parent, last]
                        - law.scores[parent, last_ref]
                    )
                    mean += p * g
                    second += p * g**2
    result = exact_estimator_moments(1)
    np.testing.assert_allclose(result.mean, mean, atol=1e-12)
    np.testing.assert_allclose(result.second_moment, second, atol=1e-12)
