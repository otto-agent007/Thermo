"""Contracts for deterministic model-context backend preparation."""

import pytest

from thermo_lab.backends.thrml_model_context_pasym_swap import (
    ThrmlModelContextPAsymSwapBackend,
)
from thermo_lab.config import model_context_pasym_swap_non_seed_config_hash
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.experiments.model_context_pasym_swap import model_context_pasym_swap_spec
from thermo_lab.experiments.target_context_pasym_swap import target_context_pasym_swap_spec
from thermo_lab.model_context_pasym_swap_results import (
    validate_model_context_pasym_swap_summary,
    validate_model_context_profile_result,
    validate_model_context_schedule_acceptance,
)


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


def test_model_context_backend_derives_checked_exact_profile_and_schedule_evidence() -> None:
    evidence = ThrmlModelContextPAsymSwapBackend().evaluate(model_context_pasym_swap_spec(seed=0))

    assert len(evidence.profile_results) == 37
    assert evidence.acceptance.occurrence_count == 500
    assert evidence.acceptance.profile_count == 37
    assert evidence.acceptance.model_context_optimizer_endpoints_passed
    assert evidence.acceptance.exact_k30_acceptance_passed
    assert evidence.acceptance.passed
    assert all(
        validate_model_context_profile_result(profile) == profile
        for profile in evidence.profile_results
    )
    assert (
        validate_model_context_schedule_acceptance(evidence.acceptance, evidence.profile_results)
        == evidence.acceptance
    )


def test_model_context_backend_cross_checks_all_model_kernels_with_thrml_sampling() -> None:
    sampled = ThrmlModelContextPAsymSwapBackend().sample(model_context_pasym_swap_spec(seed=0))

    assert len(sampled.profile_samples) == 37
    assert sampled.maximum_empirical_k30_residual <= 0.10
    assert sampled.passed
    assert all(
        sample.model_context_artifact_hash
        and all(sum(row) == 4096 for row in sample.sampled_k30.counts)
        for sample in sampled.profile_samples
    )


def test_model_context_backend_emits_one_standard_run_record() -> None:
    result = ThrmlModelContextPAsymSwapBackend().execute(model_context_pasym_swap_spec(seed=0))
    record = result.record
    summary = validate_model_context_pasym_swap_summary(
        record.metrics["model_context_pasym_swap_summary"].value
    )

    assert record.backend_id is BackendId.THRML_LOCAL
    assert record.evidence_class is EvidenceClass.SOFTWARE_SIMULATION
    assert set(record.metrics) == {
        "model_context_pasym_swap_summary",
        "maximum_empirical_k30_residual",
    }
    assert record.metrics["maximum_empirical_k30_residual"].value == (
        summary.maximum_empirical_k30_residual
    )
    assert summary.acceptance_passed
    assert record.timing.synchronized
    assert "148 keyed 4096-chain" in record.timing.timing_method
    assert "excludes compilation" in record.timing.timing_method
