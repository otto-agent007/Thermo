from itertools import product

import numpy as np
import pytest

from thermo_lab.finite_sweep_gradient_reference import build_finite_sweep_reference
from thermo_lab.trajectory_reinforce import build_checked_fixture


def test_uniform_exhaustive_draws_reproduce_shared_gradient_and_second_moments(monkeypatch):
    from thermo_lab.finite_sweep_sampling import estimate_finite_sweep_grouped_gradient

    # Every main/reference endpoint at both occurrences: all 8**4 equal-probability paths.
    words = np.asarray(tuple(product(range(8), repeat=4)))
    streams = [iter((words[:, 0], words[:, 2])), iter((words[:, 1], words[:, 3]))]

    class Draws:
        def __init__(self, stream):
            self.stream = stream

        def random(self, count):
            assert count == len(words)
            return (next(self.stream) + 0.5) / 8

    generators = iter(Draws(stream) for stream in streams)
    monkeypatch.setattr(np.random, "Generator", lambda _: next(generators))
    result = estimate_finite_sweep_grouped_gradient(
        ((0.0,) * 9,),
        (0, 0),
        ((0, 1), (1, 2)),
        (0.7, -0.2, 0.4),
        site_count=3,
        batch_size=len(words),
        seed=991,
        beta=1.0,
        horizon=1,
    )

    # At zero parameters, one-sweep scores follow independent Bernoulli outputs.
    # Hidden bias and input-hidden correlations have no terminal effect; output
    # fields have derivative 2*sigma, input/output couplings add the clamped spin.
    def score(parent, outcome):
        x = 2 * np.asarray(divmod(parent, 2)) - 1
        hidden = 2 * (outcome // 4) - 1
        y = 2 * np.asarray(divmod(outcome % 4, 2)) - 1
        return np.asarray(
            (
                hidden,
                y[0],
                y[1],
                x[0] * y[0],
                x[0] * y[1],
                x[1] * y[0],
                x[1] * y[1],
                hidden * y[0],
                hidden * y[1],
            )
        )

    contributions = []
    for first, ref_first, second, ref_second in words:
        left, middle = divmod(first % 4, 2)
        final_middle, right = divmod(second % 4, 2)
        reward = 0.7 * left - 0.2 * final_middle + 0.4 * right
        contributions.append(
            reward
            * (
                score(2, first)
                - score(2, ref_first)
                + score(2 * middle, second)
                - score(2 * middle, ref_second)
            )
        )
    expected = np.asarray(contributions)
    np.testing.assert_allclose(result.component_sum[0], expected.sum(axis=0), atol=1e-10)
    np.testing.assert_allclose(
        result.component_sum_squares[0], (expected**2).sum(axis=0), atol=1e-10
    )


@pytest.mark.parametrize("horizon", (1, 4, 30))
def test_sampled_occurrences_and_shared_sum_agree_with_m3(horizon):
    from thermo_lab.finite_sweep_sampling import estimate_finite_sweep_grouped_gradient

    fixture = build_checked_fixture()
    reference = build_finite_sweep_reference(horizon)
    reward = 2 * (np.asarray(reference.model_law.occupancy) - reference.target_law.occupancy)
    args = dict(site_count=3, batch_size=32768, seed=41, beta=1.0, horizon=horizon)
    untied = estimate_finite_sweep_grouped_gradient(
        (fixture.model_parameters.values,) * 2, (0, 1), fixture.occurrences, reward, **args
    )
    shared = estimate_finite_sweep_grouped_gradient(
        (fixture.model_parameters.values,), (0, 0), fixture.occurrences, reward, **args
    )
    np.testing.assert_allclose(np.asarray(untied.mean).sum(axis=0), shared.mean[0], atol=1e-12)
    np.testing.assert_allclose(untied.mean, reference.score.occurrences, atol=0.025, rtol=0)


def test_finite_occupancy_and_paired_evaluation_use_the_same_endpoint_law():
    from thermo_lab.finite_sweep_sampling import (
        evaluate_finite_sweep_pair,
        sample_finite_sweep_terminal_occupancy,
    )

    fixture = build_checked_fixture()
    parameters = (fixture.model_parameters.values,)
    args = dict(site_count=3, batch_size=512, seed=17, beta=1.0, horizon=4)
    source = sample_finite_sweep_terminal_occupancy(parameters, (0, 0), fixture.occurrences, **args)
    cell = evaluate_finite_sweep_pair(parameters, parameters, (0, 0), fixture.occurrences, **args)
    assert cell.horizon == "k4"
    assert cell.before_counts == cell.after_counts == source.occupancy_counts
    assert cell.paired_leakage_counts[1:3] == (0, 0)
    assert cell.statistics((0.2, 0.3, 0.5)).population_objective_difference_after_minus_before == 0


@pytest.mark.parametrize(
    "override",
    (
        {"horizon": True},
        {"horizon": 0},
        {"horizon": 31},
        {"batch_size": 32769},
        {"batch_size": True},
        {"seed": -1},
    ),
)
def test_sampler_rejects_invalid_or_unbounded_requests(override):
    from thermo_lab.finite_sweep_sampling import sample_finite_sweep_terminal_occupancy

    args = dict(site_count=3, batch_size=16, seed=0, beta=1.0, horizon=4)
    args.update(override)
    with pytest.raises(ValueError):
        sample_finite_sweep_terminal_occupancy(((0.0,) * 9,), (0, 0), ((0, 1), (1, 2)), **args)


def test_signed_zero_requests_have_cache_independent_identities():
    from thermo_lab.finite_sweep_sampling import (
        _gradient,
        _occupancy,
        estimate_finite_sweep_grouped_gradient,
        sample_finite_sweep_terminal_occupancy,
    )

    positive = ((0.0,) * 9,)
    negative = ((-0.0,) + (0.0,) * 8,)
    args = dict(site_count=3, batch_size=16, seed=2048, beta=1.0, horizon=1)

    def occupancy(parameters):
        return sample_finite_sweep_terminal_occupancy(parameters, (0, 0), ((0, 1), (1, 2)), **args)

    occupancy(positive)
    warm = occupancy(negative)
    _occupancy.cache_clear()
    assert occupancy(negative) == warm == occupancy(positive)

    def gradient(parameters, reward):
        return estimate_finite_sweep_grouped_gradient(
            parameters, (0, 0), ((0, 1), (1, 2)), reward, **args
        )

    gradient(positive, (0.0, 0.2, 0.3))
    warm_gradient = gradient(negative, (-0.0, 0.2, 0.3))
    _gradient.cache_clear()
    assert gradient(negative, (-0.0, 0.2, 0.3)) == warm_gradient
    assert gradient(positive, (0.0, 0.2, 0.3)) == warm_gradient


def test_nonuniform_reference_parents_and_nonpropagation_match_explicit_oracle(monkeypatch):
    from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
    from thermo_lab.finite_sweep_sampling import estimate_finite_sweep_grouped_gradient

    fixture = build_checked_fixture()
    law = finite_sweep_joint_law(fixture.model_parameters, 1)
    words = np.asarray(tuple(product(range(8), repeat=4)))
    uniforms = (words + 0.5) / 8

    class Draws:
        def __init__(self, columns):
            self.columns = iter(columns)

        def random(self, count):
            assert count == len(words)
            return uniforms[:, next(self.columns)]

    generators = iter((Draws((0, 2)), Draws((1, 3))))
    monkeypatch.setattr(np.random, "Generator", lambda _: next(generators))
    actual = estimate_finite_sweep_grouped_gradient(
        (fixture.model_parameters.values,),
        (0, 0),
        fixture.occurrences,
        (0.7, -0.2, 0.4),
        site_count=3,
        batch_size=len(words),
        seed=993,
        beta=1.0,
        horizon=1,
    )
    cdf = law.probabilities.cumsum(axis=1)
    cdf[:, -1] = 1.0

    def contribution(draw, wrong_reference_parent=False, propagate_reference=False):
        first = int(np.searchsorted(cdf[2], draw[0], side="right"))
        first_ref_parent = 0 if wrong_reference_parent else 2
        first_ref = int(np.searchsorted(cdf[first_ref_parent], draw[1], side="right"))
        left, middle = divmod((first_ref if propagate_reference else first) % 4, 2)
        parent = 2 * middle
        second = int(np.searchsorted(cdf[parent], draw[2], side="right"))
        second_ref_parent = 0 if wrong_reference_parent else parent
        second_ref = int(np.searchsorted(cdf[second_ref_parent], draw[3], side="right"))
        last_middle, right = divmod(second % 4, 2)
        return (0.7 * left - 0.2 * last_middle + 0.4 * right) * (
            law.scores[2, first]
            - law.scores[first_ref_parent, first_ref]
            + law.scores[parent, second]
            - law.scores[second_ref_parent, second_ref]
        )

    correct = np.asarray([contribution(u) for u in uniforms])
    np.testing.assert_allclose(actual.component_sum[0], correct.sum(axis=0), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        actual.component_sum_squares[0], (correct**2).sum(axis=0), atol=1e-10, rtol=0
    )
    for wrong in ({"wrong_reference_parent": True}, {"propagate_reference": True}):
        erroneous = np.asarray([contribution(u, **wrong) for u in uniforms])
        assert np.max(np.abs(erroneous.sum(axis=0) - correct.sum(axis=0))) > 1.0
