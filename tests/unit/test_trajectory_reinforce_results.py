"""Strict persistence tests for trajectory-level REINFORCE evidence."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from thermo_lab.hashing import canonical_sha256
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference
from thermo_lab.trajectory_reinforce_results import (
    GradientMoments,
    TrajectoryReinforceSummary,
    build_gradient_estimate,
    build_shared_gradient_estimate,
    build_trajectory_reinforce_deterministic_result,
    build_trajectory_reinforce_sample_result,
    build_trajectory_reinforce_summary,
    validate_trajectory_reinforce_summary,
)

OBSERVATIONS_0 = (
    (1.0, 2.0, -1.0, 4.0, 0.5, 3.0, -2.0, 5.0, 0.0),
    (3.0, -2.0, 2.0, 0.0, 1.5, -1.0, 4.0, 1.0, 2.0),
)
OBSERVATIONS_1 = (
    (2.0, -1.0, 4.0, 1.0, 2.0, -2.0, 3.0, 2.0, 1.0),
    (6.0, 3.0, -2.0, 5.0, 0.0, 4.0, -1.0, 4.0, -3.0),
)


def _componentwise_sum(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(sum(row[index] for row in rows) for index in range(9))


VECTOR_0 = _componentwise_sum(OBSERVATIONS_0)
VECTOR_1 = _componentwise_sum(OBSERVATIONS_1)
SQUARES_0 = _componentwise_sum(
    tuple(tuple(value * value for value in row) for row in OBSERVATIONS_0)
)
SQUARES_1 = _componentwise_sum(
    tuple(tuple(value * value for value in row) for row in OBSERVATIONS_1)
)
CROSS = tuple(
    sum(
        left[index] * right[index]
        for left, right in zip(OBSERVATIONS_0, OBSERVATIONS_1, strict=True)
    )
    for index in range(9)
)


def _moments(component_sum: tuple[float, ...], squares: tuple[float, ...]) -> GradientMoments:
    return GradientMoments(
        sample_count=2,
        component_sum=component_sum,
        component_sum_squares=squares,
    )


def _summary() -> TrajectoryReinforceSummary:
    exact = build_exact_reference(build_checked_fixture())
    deterministic = build_trajectory_reinforce_deterministic_result(
        request_hash=canonical_sha256({"request": "checked"}),
        fixture=build_checked_fixture(),
        exact=exact,
    )
    sample = build_trajectory_reinforce_sample_result(
        deterministic_result_digest=deterministic.deterministic_result_digest,
        seed=7,
        sample_definition="one augmented trajectory",
        occurrence_0=_moments(VECTOR_0, SQUARES_0),
        occurrence_1=_moments(VECTOR_1, SQUARES_1),
        cross_products=CROSS,
        exact=deterministic.expected_reference,
    )
    return build_trajectory_reinforce_summary(deterministic=deterministic, sample=sample)


def _payload() -> dict[str, object]:
    return json.loads(_summary().model_dump_json())


def _rehash(payload: dict[str, object]) -> None:
    deterministic = payload["deterministic"]
    assert isinstance(deterministic, dict)
    deterministic_without_digest = {
        key: value for key, value in deterministic.items() if key != "deterministic_result_digest"
    }
    deterministic["deterministic_result_digest"] = canonical_sha256(
        {"identity_version": "trajectory_reinforce_exact.v1", **deterministic_without_digest}
    )
    sample = payload["sample"]
    assert isinstance(sample, dict)
    sample["deterministic_result_digest"] = deterministic["deterministic_result_digest"]
    sample_without_digest = {key: value for key, value in sample.items() if key != "sample_digest"}
    sample["sample_digest"] = canonical_sha256(
        {"identity_version": "trajectory_reinforce_sample.v1", **sample_without_digest}
    )
    payload_without_digest = {
        key: value for key, value in payload.items() if key != "summary_digest"
    }
    payload["summary_digest"] = canonical_sha256(
        {"identity_version": "trajectory_reinforce_summary.v1", **payload_without_digest}
    )


def test_gradient_estimate_derives_unbiased_variance_and_standard_error() -> None:
    moments = _moments(VECTOR_0, SQUARES_0)
    estimate = build_gradient_estimate(moments, (0.0,) * 9)

    expected_variance = tuple(q - s * s / 2.0 for s, q in zip(VECTOR_0, SQUARES_0, strict=True))
    assert estimate.mean == tuple(value / 2.0 for value in VECTOR_0)
    assert estimate.variance == pytest.approx(expected_variance)
    assert estimate.standard_error == pytest.approx(
        tuple(math.sqrt(value / 2.0) for value in expected_variance)
    )
    assert estimate.absolute_error == tuple(abs(value / 2.0) for value in VECTOR_0)
    assert estimate.maximum_absolute_error == max(estimate.absolute_error)


def test_shared_estimate_includes_twice_the_correlated_cross_products() -> None:
    first = _moments(VECTOR_0, SQUARES_0)
    second = _moments(VECTOR_1, SQUARES_1)

    shared = build_shared_gradient_estimate(first, second, CROSS, (0.0,) * 9)

    assert shared.moments.component_sum == tuple(
        left + right for left, right in zip(VECTOR_0, VECTOR_1, strict=True)
    )
    assert shared.moments.component_sum_squares == tuple(
        left + right + 2.0 * cross
        for left, right, cross in zip(SQUARES_0, SQUARES_1, CROSS, strict=True)
    )
    centered_cross = tuple(
        product_sum - sum_0 * sum_1 / 2.0
        for product_sum, sum_0, sum_1 in zip(CROSS, VECTOR_0, VECTOR_1, strict=True)
    )
    assert any(value != 0.0 for value in centered_cross)
    paired_shared = tuple(
        tuple(left + right for left, right in zip(row_0, row_1, strict=True))
        for row_0, row_1 in zip(OBSERVATIONS_0, OBSERVATIONS_1, strict=True)
    )
    expected_variance = tuple(
        ((paired_shared[0][index] - paired_shared[1][index]) ** 2) / 2.0 for index in range(9)
    )
    assert shared.variance == pytest.approx(expected_variance)
    assert shared.standard_error == pytest.approx(
        tuple(math.sqrt(value / 2.0) for value in expected_variance)
    )


@pytest.mark.parametrize("sample_count", [0, 1, -1, True, 2.0])
def test_gradient_moments_rejects_invalid_sample_counts(sample_count: object) -> None:
    with pytest.raises(ValidationError):
        GradientMoments(
            sample_count=sample_count,
            component_sum=(0.0,) * 9,
            component_sum_squares=(0.0,) * 9,
        )


@pytest.mark.parametrize("invalid", [True, 1, "1.0", math.nan, math.inf])
def test_gradient_vectors_reject_coercive_or_nonfinite_values(invalid: object) -> None:
    values = [0.0] * 9
    values[3] = invalid
    with pytest.raises(ValidationError):
        GradientMoments(
            sample_count=2,
            component_sum=values,
            component_sum_squares=[0.0] * 9,
        )


def test_gradient_vectors_require_exactly_nine_components_and_nonnegative_squares() -> None:
    with pytest.raises(ValidationError):
        GradientMoments(sample_count=2, component_sum=(0.0,) * 8, component_sum_squares=(0.0,) * 9)
    with pytest.raises(ValidationError, match="nonnegative"):
        GradientMoments(
            sample_count=2, component_sum=(0.0,) * 9, component_sum_squares=(-1.0,) + (0.0,) * 8
        )


def test_roundoff_only_negative_variance_is_zeroed_but_material_failure_is_rejected() -> None:
    tiny = math.nextafter(0.5, 0.0)
    estimate = build_gradient_estimate(
        GradientMoments(
            sample_count=2, component_sum=(1.0,) * 9, component_sum_squares=(tiny,) * 9
        ),
        (0.0,) * 9,
    )
    assert estimate.variance == (0.0,) * 9
    with pytest.raises(ValueError, match="negative variance"):
        build_gradient_estimate(
            GradientMoments(
                sample_count=2, component_sum=(2.0,) * 9, component_sum_squares=(1.0,) * 9
            ),
            (0.0,) * 9,
        )


def test_extreme_finite_moments_cannot_overflow_into_an_accepted_zero_variance() -> None:
    extreme = (1e308,) * 9
    with pytest.raises(ValueError, match="non-finite"):
        build_gradient_estimate(
            GradientMoments(
                sample_count=2,
                component_sum=extreme,
                component_sum_squares=extreme,
            ),
            (0.0,) * 9,
        )


def test_extreme_finite_cross_moments_cannot_overflow_cauchy_schwarz_checks() -> None:
    occurrence = GradientMoments(
        sample_count=2,
        component_sum=(1e154,) * 9,
        component_sum_squares=(1e308,) * 9,
    )
    with pytest.raises(ValueError, match="non-finite"):
        build_shared_gradient_estimate(
            occurrence,
            occurrence,
            (1e308,) * 9,
            (0.0,) * 9,
        )


def test_shared_estimate_rejects_count_mismatch_and_impossible_centered_covariance() -> None:
    first = _moments((0.0,) * 9, (1.0,) * 9)
    other_count = GradientMoments(
        sample_count=3, component_sum=(0.0,) * 9, component_sum_squares=(1.0,) * 9
    )
    with pytest.raises(ValueError, match="sample_count"):
        build_shared_gradient_estimate(first, other_count, (0.0,) * 9, (0.0,) * 9)
    with pytest.raises(ValueError, match="Cauchy-Schwarz"):
        build_shared_gradient_estimate(first, first, (2.0,) * 9, (0.0,) * 9)


def test_checked_summary_is_frozen_and_survives_json_round_trip() -> None:
    summary = _summary()
    reloaded = validate_trajectory_reinforce_summary(summary.model_dump_json())
    assert reloaded == summary
    with pytest.raises(ValidationError):
        summary.sample.seed = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("deterministic", "target_law", "probabilities", 0), 0.25),
        (("deterministic", "model_law", "particle_number_leakage"), 0.25),
        (("deterministic", "score", "occurrences", 0, 0), 0.25),
        (("deterministic", "finite_difference_tied", 0), 0.25),
        (("sample", "cross_products", 0), 0.25),
        (("sample", "shared", "standard_error", 0), 0.25),
        (("deterministic", "fixture", "exact_tolerance"), 0.25),
        (("deterministic", "deterministic_result_digest"), "sha256:" + "0" * 64),
        (("acceptance_passed",), False),
    ],
)
def test_deep_validation_rejects_each_tampered_source_or_derived_field(
    path: tuple[object, ...], replacement: object
) -> None:
    payload = _payload()
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]
    if path not in {
        ("deterministic", "deterministic_result_digest"),
        ("acceptance_passed",),
    }:
        _rehash(payload)

    with pytest.raises((ValueError, ValidationError)):
        validate_trajectory_reinforce_summary(payload)


def test_deep_validation_rejects_stale_shared_moments_even_with_fresh_digests() -> None:
    payload = _payload()
    payload["sample"]["shared"]["moments"]["component_sum_squares"][0] += 1.0  # type: ignore[index]
    _rehash(payload)
    with pytest.raises(ValueError, match="shared"):
        validate_trajectory_reinforce_summary(payload)


def test_models_reject_unknown_fields() -> None:
    payload = _payload()
    payload["unexpected"] = "field"
    with pytest.raises(ValidationError):
        validate_trajectory_reinforce_summary(payload)


@pytest.mark.parametrize("invalid", [True, 1, "1.0", math.nan, math.inf])
def test_persisted_scalar_fields_reject_coercive_or_nonfinite_values(invalid: object) -> None:
    payload = _payload()
    payload["deterministic"]["objective"] = invalid  # type: ignore[index]
    with pytest.raises(ValidationError):
        validate_trajectory_reinforce_summary(payload)
