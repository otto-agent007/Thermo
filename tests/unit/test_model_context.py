"""Contracts for deterministic first-moment model-context traces."""

import pytest

from thermo_lab.model_context import (
    ModelContextArtifact,
    derive_model_context_trace,
    pool_model_context_profiles,
)
from thermo_lab.pasym_swap import build_paper_fixture


def test_model_context_module_exposes_trace_derivation() -> None:
    assert callable(derive_model_context_trace)


def _identity_artifacts():
    fixture = build_paper_fixture()
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return fixture, {
        target.target_hash: ModelContextArtifact(f"artifact:{target.target_hash}", identity)
        for target in fixture.targets
    }


def test_model_trace_factorizes_endpoint_means_and_pools_canonically() -> None:
    fixture, artifacts = _identity_artifacts()
    trace = derive_model_context_trace(fixture, artifacts, initial_occupancy=(1.0,) + (0.0,) * 24)

    assert trace.occurrences[0].context_weights == (0.0, 0.0, 1.0, 0.0)
    assert trace.occurrences[0].source_mean_after == 1.0
    assert trace.occurrences[0].target_mean_after == 0.0
    profiles = pool_model_context_profiles(trace)
    assert len(trace.occurrences) == 500
    assert len(profiles) == 37
    assert sum(profile.multiplicity for profile in profiles) == 500


def test_model_trace_uses_upstream_conditional_and_rejects_missing_artifact() -> None:
    fixture, artifacts = _identity_artifacts()
    first_hash = fixture.occurrences[0].target_hash
    artifacts[first_hash] = ModelContextArtifact("changed", ((0.0, 0.0, 0.0, 1.0),) * 4)
    trace = derive_model_context_trace(fixture, artifacts, initial_occupancy=(1.0,) + (0.0,) * 24)
    assert trace.occurrences[0].target_mean_after == 1.0
    del artifacts[first_hash]
    with pytest.raises(ValueError, match="missing target-context artifact"):
        derive_model_context_trace(fixture, artifacts, initial_occupancy=(1.0,) + (0.0,) * 24)
