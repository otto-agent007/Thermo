"""Contracts for deterministic model-context backend preparation."""

import pytest

from thermo_lab.backends.thrml_model_context_pasym_swap import (
    ThrmlModelContextPAsymSwapBackend,
)
from thermo_lab.config import model_context_pasym_swap_non_seed_config_hash
from thermo_lab.experiments.model_context_pasym_swap import model_context_pasym_swap_spec
from thermo_lab.experiments.target_context_pasym_swap import target_context_pasym_swap_spec


def test_model_context_backend_accepts_only_the_checked_model_context_request() -> None:
    backend = ThrmlModelContextPAsymSwapBackend()

    model, run, request_hash, upstream_spec = backend.checked_request(
        model_context_pasym_swap_spec(seed=2)
    )

    assert request_hash == model_context_pasym_swap_non_seed_config_hash(model, run)
    assert upstream_spec == target_context_pasym_swap_spec(seed=2)
    assert run.initial_occupancy == (1.0,) + (0.0,) * 24
    assert model.torus_side == 5

    with pytest.raises(ValueError, match="Unexpected experiment request"):
        backend.checked_request(target_context_pasym_swap_spec(seed=2))


def test_model_context_backend_rebuilds_one_upstream_lineage_and_compiles_37_profiles() -> None:
    prepared = ThrmlModelContextPAsymSwapBackend().prepare(model_context_pasym_swap_spec(seed=0))

    assert len(prepared.target_context_artifacts) == 37
    assert len(prepared.model_profiles) == 37
    assert len(prepared.model_context_artifacts) == 37
    assert len(prepared.model_trace.occurrences) == 500
    assert tuple(item.target_hash for item in prepared.model_context_artifacts) == tuple(
        profile.target_hash for profile in prepared.model_profiles
    )
    assert all(
        artifact.target_context_artifact_hash == profile.upstream_artifact_hash
        for artifact, profile in zip(
            prepared.model_context_artifacts, prepared.model_profiles, strict=True
        )
    )
