"""Acceptance contracts for model-context KL comparisons."""

import pytest

from thermo_lab.model_context_pasym_swap_results import validate_model_context_improvement


def test_model_context_acceptance_uses_occurrence_weighted_improvement() -> None:
    result = validate_model_context_improvement(((250, 1.0, 0.9), (250, 2.0, 1.8)))

    assert result.profile_non_regression_passed
    assert result.occurrence_weighted_improvement == pytest.approx(0.15, abs=1e-15)
    assert result.passed
