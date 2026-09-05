"""Pure acceptance math for one-pass model-context local comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContextAcceptance:
    profile_non_regression_passed: bool
    occurrence_weighted_improvement: float
    occurrence_weighted_improvement_passed: bool
    passed: bool


def validate_model_context_improvement(
    comparisons: tuple[tuple[int, float, float], ...],
    *,
    non_regression_tolerance: float = 1e-12,
    minimum_improvement: float = 1e-8,
) -> ModelContextAcceptance:
    """Validate model-profile KL against paired target-context artifacts."""

    if not comparisons or sum(item[0] for item in comparisons) != 500:
        raise ValueError("comparisons must cover exactly 500 occurrences")
    if any(
        type(multiplicity) is not int
        or multiplicity <= 0
        or not math.isfinite(previous)
        or not math.isfinite(current)
        or previous < 0.0
        or current < 0.0
        for multiplicity, previous, current in comparisons
    ):
        raise ValueError("comparisons must contain finite nonnegative KL values")
    profile_passed = all(
        current <= previous + non_regression_tolerance for _, previous, current in comparisons
    )
    improvement = (
        math.fsum(
            multiplicity * (previous - current) for multiplicity, previous, current in comparisons
        )
        / 500
    )
    improvement_passed = improvement >= minimum_improvement
    return ModelContextAcceptance(
        profile_passed, improvement, improvement_passed, profile_passed and improvement_passed
    )
