"""Exact first/second moments and non-gating diagnostics for the finite sampler."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from thermo_lab.composed_trajectory_refinement import GroupedGradientSource, _tuple_matrix
from thermo_lab.finite_sweep_gradient_reference import (
    CHECKED_HORIZONS,
    build_finite_sweep_reference,
)
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.finite_sweep_sampling import estimate_finite_sweep_grouped_gradient
from thermo_lab.trajectory_reinforce import build_checked_fixture


@dataclass(frozen=True)
class ExactEstimatorMoments:
    horizon: int
    reward_coefficient: tuple[float, ...]
    occurrence_mean: tuple[tuple[float, ...], ...]
    mean: tuple[float, ...]
    second_moment: tuple[float, ...]
    maximum_m3_error: float


@lru_cache(maxsize=6, typed=True)
def exact_estimator_moments(horizon: int) -> ExactEstimatorMoments:
    """Sum 64 main paths; integrate the two independent references analytically."""
    if type(horizon) is not int or horizon not in CHECKED_HORIZONS:
        raise ValueError("estimator reference requires a checked horizon")
    fixture = build_checked_fixture()
    reference = build_finite_sweep_reference(horizon)
    if not reference.accepted:
        raise ValueError("M3 exact gradient gate failed")
    law = finite_sweep_joint_law(fixture.model_parameters, horizon, beta=fixture.beta)
    probabilities, scores = law.probabilities, law.scores
    reward = 2 * (np.asarray(reference.model_law.occupancy) - reference.target_law.occupancy)
    score_mean = np.einsum("po,poj->pj", probabilities, scores)
    score_variance = np.einsum("po,poj->pj", probabilities, scores**2) - score_mean**2
    occurrences = np.zeros((2, 9))
    second_moment = np.zeros(9)
    for first, second in product(range(8), repeat=2):
        left, middle = divmod(first % 4, 2)
        last_middle, right = divmod(second % 4, 2)
        parent = 2 * middle
        probability = probabilities[2, first] * probabilities[parent, second]
        value = reward @ (left, last_middle, right)
        centered = np.asarray(
            (scores[2, first] - score_mean[2], scores[parent, second] - score_mean[parent])
        )
        occurrences += probability * value * centered
        second_moment += (
            probability
            * value**2
            * (centered.sum(axis=0) ** 2 + score_variance[2] + score_variance[parent])
        )
    mean = occurrences.sum(axis=0)
    error = max(
        float(np.max(np.abs(occurrences - reference.autodiff.occurrences))),
        float(np.max(np.abs(mean - reference.autodiff.shared))),
    )
    if error > 1e-12 or np.max(np.abs(score_mean)) > 1e-12:
        raise ValueError("finite sampled-estimator expectation disagrees with M3")
    return ExactEstimatorMoments(
        horizon,
        tuple(float(x) for x in reward),
        _tuple_matrix(occurrences),
        tuple(float(x) for x in mean),
        tuple(float(x) for x in second_moment),
        error,
    )


@dataclass(frozen=True)
class SamplerDiagnostic:
    horizon: int
    seed: int
    sampling_seed: int
    reference: ExactEstimatorMoments
    sample: GroupedGradientSource

    @property
    def maximum_standardized_mean_error(self) -> float:
        mean = np.asarray(self.reference.mean)
        variance = np.maximum(0.0, np.asarray(self.reference.second_moment) - mean**2)
        error = np.abs(np.asarray(self.sample.mean)[0] - mean)
        return float(
            np.max(error / np.maximum(np.sqrt(variance / self.sample.sample_count), 1e-30))
        )


@lru_cache(maxsize=1)
def checked_sampler_diagnostics() -> tuple[SamplerDiagnostic, ...]:
    """Fixed seeds/budgets; Monte Carlo agreement is descriptive, never a tuning gate."""
    fixture = build_checked_fixture()
    checks = []
    for horizon in CHECKED_HORIZONS:
        moments = exact_estimator_moments(horizon)
        reward = moments.reward_coefficient
        for seed in (0, 1, 2):
            sampling_seed = int(
                np.random.SeedSequence([0x4D34, 0x56414C, seed, horizon]).generate_state(
                    1, dtype=np.uint64
                )[0]
            )
            sample = estimate_finite_sweep_grouped_gradient(
                (fixture.model_parameters.values,),
                (0, 0),
                fixture.occurrences,
                reward,
                site_count=3,
                batch_size=32768,
                seed=sampling_seed,
                beta=fixture.beta,
                horizon=horizon,
            )
            checks.append(SamplerDiagnostic(horizon, seed, sampling_seed, moments, sample))
    return tuple(checks)
