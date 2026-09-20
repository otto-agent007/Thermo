"""M4G law-specific sampling and bounded, independently replayed validation."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from itertools import product

import numpy as np

from thermo_lab.composed_trajectory_refinement import (
    _checked_site_vector,
    estimate_equilibrium_grouped_gradient,
    sample_equilibrium_terminal_occupancy,
)
from thermo_lab.finite_sweep_gradient_reference import build_finite_sweep_reference
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.finite_sweep_sampling import (
    _request,
    estimate_finite_sweep_grouped_gradient,
    sample_finite_sweep_terminal_occupancy,
)
from thermo_lab.hashing import to_json_value
from thermo_lab.quality_budget_protocol import HORIZONS
from thermo_lab.thermodynamic_kernel import equilibrium_joint_conditional, sufficient_statistics
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference

LAWS = (*HORIZONS, "equilibrium")


def _check_law(horizon):
    if not (
        (type(horizon) is int and horizon in HORIZONS)
        or (type(horizon) is str and horizon == "equilibrium")
    ):
        raise ValueError("training law must be a declared finite K or equilibrium")


def sample_training_roles(
    parameters, groups, sites, target, *, horizon, occupancy_seed, gradient_seed
):
    """Sample independent occupancy/gradient roles at one fixed law; perform no update."""
    _check_law(horizon)
    if any(type(s) is not int or s < 0 for s in (occupancy_seed, gradient_seed)):
        raise ValueError("training roles require nonnegative integer seeds")
    if occupancy_seed == gradient_seed:
        raise ValueError("occupancy and gradient roles must be distinct")
    target = _checked_site_vector(target, site_count=len(target), name="target")
    if np.any(target < 0) or np.any(target > 1):
        raise ValueError("target occupancy must lie in [0,1]")
    options = dict(site_count=len(target), batch_size=32768, beta=1.0)
    # Reuse common bounded-input validation for both laws. K1 here validates
    # only the request; it constructs no tables and consumes no randomness.
    checked = _request(parameters, groups, sites, seed=occupancy_seed, horizon=1, **options)
    parameters, groups, sites = checked.parameters, checked.groups, checked.sites
    if horizon == "equilibrium":
        occupancy_fn = sample_equilibrium_terminal_occupancy
        gradient_fn = estimate_equilibrium_grouped_gradient
    else:
        occupancy_fn = sample_finite_sweep_terminal_occupancy
        gradient_fn = estimate_finite_sweep_grouped_gradient
        options["horizon"] = horizon
    occupancy = occupancy_fn(parameters, groups, sites, seed=occupancy_seed, **options)
    reward = tuple(2.0 * (a - b) for a, b in zip(occupancy.occupancy, target, strict=True))
    gradient = gradient_fn(parameters, groups, sites, reward, seed=gradient_seed, **options)
    return occupancy, reward, gradient


def validation_roles(horizon, seed):
    _check_law(horizon)
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("validation requires seed 0, 1 or 2")
    # A separate namespace: never consume any of the 213 production roles.
    children = np.random.SeedSequence([0x4D3447, 0x56414C, seed, LAWS.index(horizon)]).spawn(2)
    return tuple(int(c.generate_state(1, dtype=np.uint64)[0]) for c in children)


@lru_cache(maxsize=7, typed=True)
def _reference(horizon):
    """Cache only frozen internally built references, never external evidence."""
    fixture = build_checked_fixture()
    if horizon == "equilibrium":
        reference = build_exact_reference(fixture)
        probabilities = equilibrium_joint_conditional(fixture.model_parameters, beta=1.0)
        features = np.asarray(
            [[sufficient_statistics(p, o // 4, o % 4) for o in range(8)] for p in range(4)]
        )
        scores = features - np.einsum("po,poj->pj", probabilities, features)[:, None, :]
        replay_scores = features
    else:
        reference = build_finite_sweep_reference(horizon, fixture)
        law = finite_sweep_joint_law(fixture.model_parameters, horizon, beta=1.0)
        probabilities, scores = law.probabilities, law.scores
        replay_scores = scores
    if not reference.accepted:
        raise ValueError("existing exact gradient contract failed")
    for values in (probabilities, scores, replay_scores):
        values.setflags(write=False)
    return reference, probabilities, scores, replay_scores


def _moments(probabilities, scores, reward):
    """Enumerate 64 main paths and integrate independent same-parent references."""
    mean_score = np.einsum("po,poj->pj", probabilities, scores)
    occurrences = np.zeros((2, 9))
    occupancy = np.zeros(3)
    for first, last in product(range(8), repeat=2):
        left, middle = divmod(first % 4, 2)
        end_middle, right = divmod(last % 4, 2)
        parent = 2 * middle
        p = probabilities[2, first] * probabilities[parent, last]
        state = np.asarray((left, end_middle, right))
        value = reward @ state
        centered = np.asarray(
            (scores[2, first] - mean_score[2], scores[parent, last] - mean_score[parent])
        )
        occupancy += p * state
        occurrences += p * value * centered
    return occupancy, occurrences


def _fixture_replay(probabilities, scores, seed, reward=None):
    """Independent parent-partition/searchsorted replay, without production samplers."""
    n = 32768
    states = np.tile(np.asarray((1, 0, 0), dtype=np.int64), (n, 1))
    if reward is None:
        main = np.random.Generator(np.random.PCG64(seed))
    else:
        main_seed, ref_seed = np.random.SeedSequence(seed).spawn(2)
        main = np.random.Generator(np.random.PCG64(main_seed))
        reference = np.random.Generator(np.random.PCG64(ref_seed))
    accumulated = np.zeros((n, 9))

    def draw(parents, uniforms):
        result = np.empty(n, dtype=np.int64)
        for parent in range(4):
            mask = parents == parent
            cdf = np.cumsum(probabilities[parent])
            cdf[-1] = 1.0
            result[mask] = np.searchsorted(cdf, uniforms[mask], side="right")
        return result

    for left, right in ((0, 1), (1, 2)):
        parents = 2 * states[:, left] + states[:, right]
        outcomes = draw(parents, main.random(n))
        if reward is not None:
            ref = draw(parents, reference.random(n))
            accumulated += scores[parents, outcomes] - scores[parents, ref]
        states[:, left], states[:, right] = outcomes % 4 // 2, outcomes % 2
    if reward is None:
        return states.sum(axis=0)
    contribution = accumulated * (states @ np.asarray(reward))[:, None]
    return contribution.sum(axis=0), (contribution**2).sum(axis=0)


def validate_training_law(horizon):
    _check_law(horizon)
    fixture = build_checked_fixture()
    ref, probabilities, scores, replay_scores = _reference(horizon)
    reward = 2 * (np.asarray(ref.model_law.occupancy) - ref.target_law.occupancy)
    occupancy, occurrences = _moments(probabilities, scores, reward)
    exact_error = max(
        ref.maximum_exact_error,
        float(np.max(np.abs(occupancy - ref.model_law.occupancy))),
        float(np.max(np.abs(occurrences - ref.score.occurrences))),
        float(np.max(np.abs(occurrences.sum(axis=0) - ref.score.shared))),
        float(np.max(np.abs(np.einsum("po,poj->pj", probabilities, scores)))),
    )
    if exact_error > 1e-12:
        raise ValueError("training-law exact estimator expectation failed")
    diagnostics = []
    max_occ_error = max_grad_error = 0.0
    for seed in (0, 1, 2):
        occupancy_seed, gradient_seed = validation_roles(horizon, seed)
        observed, coefficient, gradient = sample_training_roles(
            (fixture.model_parameters.values,),
            (0, 0),
            fixture.occurrences,
            ref.target_law.occupancy,
            horizon=horizon,
            occupancy_seed=occupancy_seed,
            gradient_seed=gradient_seed,
        )
        expected_counts = _fixture_replay(probabilities, replay_scores, occupancy_seed)
        expected_reward = 2 * (expected_counts / 32768 - ref.target_law.occupancy)
        expected_sum, expected_squares = _fixture_replay(
            probabilities, replay_scores, gradient_seed, expected_reward
        )
        if (
            observed.sample_count != 32768
            or gradient.sample_count != 32768
            or not np.array_equal(observed.occupancy_counts, expected_counts)
            or not np.array_equal(coefficient, expected_reward)
            or not np.allclose(
                gradient.component_sum, expected_sum[None, :], rtol=1e-12, atol=1e-10
            )
            or not np.allclose(
                gradient.component_sum_squares, expected_squares[None, :], rtol=1e-12, atol=1e-10
            )
        ):
            raise ValueError("law-specific occupancy/gradient replay failed")
        max_occ_error = max(
            max_occ_error,
            float(np.max(np.abs(np.asarray(observed.occupancy_counts) - expected_counts))),
        )
        max_grad_error = max(
            max_grad_error,
            float(np.max(np.abs(np.asarray(gradient.component_sum) - expected_sum))),
            float(np.max(np.abs(np.asarray(gradient.component_sum_squares) - expected_squares))),
        )
        diagnostics.append(
            {
                "seed": seed,
                "occupancy_seed": occupancy_seed,
                "gradient_seed": gradient_seed,
                "evidence_class": "software_simulation",
                "occupancy": asdict(observed),
                "reward_coefficient": coefficient,
                "gradient": asdict(gradient),
            }
        )
    return to_json_value(
        {
            "horizon": horizon,
            "exact_reference": {
                "evidence_class": "exact_reference",
                "occupancy": occupancy.tolist(),
                "occurrence_mean": occurrences.tolist(),
                "mean": occurrences.sum(axis=0).tolist(),
                "finite_difference": ref.finite_difference_tied.tolist(),
            },
            "maximum_exact_error": exact_error,
            "maximum_finite_difference_error": ref.maximum_finite_difference_error,
            "maximum_occupancy_replay_error": max_occ_error,
            "maximum_gradient_replay_error": max_grad_error,
            "diagnostics": diagnostics,
        }
    )


def negative_controls():
    finite, probabilities, scores, _ = _reference(1)
    equilibrium, _, _, _ = _reference("equilibrium")
    wrong_reward = 2 * (np.asarray(equilibrium.model_law.occupancy) - finite.target_law.occupancy)
    _, wrong_occurrences = _moments(probabilities, scores, wrong_reward)
    errors = {
        "mixed_occupancy_law_gradient_error_k1": float(
            np.max(np.abs(wrong_occurrences.sum(axis=0) - finite.score.shared))
        ),
        "equilibrium_score_substitution_error_k1": finite.equilibrium_score_substitution_error,
        "missing_shared_occurrence_error_k1": float(np.max(np.abs(finite.score.occurrences[1]))),
    }
    if any(error <= 1e-7 for error in errors.values()):
        raise ValueError("training-law negative control is not discriminating")
    return errors
