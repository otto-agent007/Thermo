"""Acceptance contracts for model-context PAsymSwap persisted results."""

import pytest

import thermo_lab.model_context_pasym_swap_results as results_module
from thermo_lab.hashing import canonical_sha256
from thermo_lab.model_context_pasym_swap_results import (
    build_model_context_profile_result,
    derive_model_context_schedule_acceptance,
    validate_model_context_improvement,
    validate_model_context_profile_result,
    validate_model_context_schedule_acceptance,
)
from thermo_lab.target_context_pasym_swap_results import derive_sampled_k30_evaluation


def test_model_context_acceptance_uses_occurrence_weighted_improvement() -> None:
    result = validate_model_context_improvement(((250, 1.0, 0.9), (250, 2.0, 1.8)))

    assert result.profile_non_regression_passed
    assert result.occurrence_weighted_improvement == pytest.approx(0.15, abs=1e-15)
    assert result.passed


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _identity_table() -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _profile_result(label: str = "one", multiplicity: int = 10, *, trace_hash: str | None = None):
    target = _identity_table()
    target_context = ((0.75, 0.25, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0), target[2], target[3])
    model_context = ((0.5, 0.5, 0.0, 0.0), target[1], target[2], target[3])
    return build_model_context_profile_result(
        target_hash=_sha(f"target:{label}"),
        target_profile_hash=_sha(f"target-profile:{label}"),
        model_profile_hash=_sha(f"model-profile:{label}"),
        model_trace_hash=trace_hash or _sha(f"trace:{label}"),
        upstream_target_context_artifact_hash=_sha(f"upstream:{label}"),
        multiplicity=multiplicity,
        target_profile_weights=(1.0, 0.0, 0.0, 0.0),
        model_profile_weights=(0.0, 1.0, 0.0, 0.0),
        target_conditional=target,
        uniform_artifact_hash=_sha(f"uniform:{label}"),
        uniform_conditional=target,
        target_context_artifact_hash=_sha(f"target-context:{label}"),
        target_context_conditional=target_context,
        model_context_artifact_hash=_sha(f"model-context:{label}"),
        model_context_conditional=model_context,
        model_trace_expected_occupancy_before=1.0,
        model_trace_expected_occupancy_after=1.1,
        target_context_k1_residual=0.03,
        target_context_k30_residual=0.02,
        model_context_k1_residual=0.04,
        model_context_k30_residual=0.03,
        model_context_optimizer_endpoint_passed=True,
    )


def test_profile_result_persists_three_variants_and_keeps_target_degradation_non_gating() -> None:
    result = _profile_result()

    assert result.model_context.model_profile_kl == pytest.approx(0.0)
    assert result.target_context.model_profile_kl > result.model_context.model_profile_kl
    assert result.model_profile_kl_improvement > 1e-8
    assert result.target_profile_kl_change > 0.0
    assert result.target_to_model_context_tv == pytest.approx(0.25)
    assert result.model_profile_acceptance_passed
    assert result.target_profile_degradation_observed
    assert result.deterministic_result_hash.startswith("sha256:")


def test_profile_result_deep_validation_rejects_persisted_metric_tampering() -> None:
    result = _profile_result()
    payload = result.model_dump(mode="json")
    payload["model_context"]["model_profile_kl"] = 0.5

    with pytest.raises(ValueError, match="model_context.model_profile_kl"):
        validate_model_context_profile_result(payload)


def test_profile_result_requires_exact_k30_to_mix_and_not_regress_from_k1() -> None:
    result = _profile_result()
    assert result.exact_k30_acceptance_passed

    degraded = _profile_result().model_copy(
        update={"model_context_k30_residual": 0.06, "exact_k30_acceptance_passed": True}
    )
    with pytest.raises(ValueError, match="exact_k30_acceptance_passed"):
        validate_model_context_profile_result(degraded)


def test_profile_result_persists_checked_acceptance_thresholds_for_deep_reload() -> None:
    base = _profile_result()
    result = build_model_context_profile_result(
        target_hash=base.target_hash,
        target_profile_hash=base.target_profile_hash,
        model_profile_hash=base.model_profile_hash,
        model_trace_hash=base.model_trace_hash,
        upstream_target_context_artifact_hash=base.upstream_target_context_artifact_hash,
        multiplicity=base.multiplicity,
        target_profile_weights=base.target_profile_weights,
        model_profile_weights=base.model_profile_weights,
        target_conditional=base.target_conditional,
        uniform_artifact_hash=base.uniform.artifact_hash,
        uniform_conditional=base.uniform.equilibrium_conditional,
        target_context_artifact_hash=base.target_context.artifact_hash,
        target_context_conditional=base.target_context.equilibrium_conditional,
        model_context_artifact_hash=base.model_context.artifact_hash,
        model_context_conditional=base.model_context.equilibrium_conditional,
        model_trace_expected_occupancy_before=base.model_trace_expected_occupancy_before,
        model_trace_expected_occupancy_after=base.model_trace_expected_occupancy_after,
        target_context_k1_residual=0.07,
        target_context_k30_residual=0.06,
        model_context_k1_residual=0.07,
        model_context_k30_residual=0.06,
        model_context_optimizer_endpoint_passed=True,
        k30_tolerance=0.1,
    )

    assert result.exact_k30_acceptance_passed
    assert validate_model_context_profile_result(result) == result


def test_schedule_acceptance_requires_all_37_profiles_and_500_occurrences() -> None:
    profiles = (
        tuple(_profile_result(f"ten:{index}", 10) for index in range(26))
        + tuple(_profile_result(f"twenty:{index}", 20) for index in range(9))
        + tuple(_profile_result(f"thirty:{index}", 30) for index in range(2))
    )

    acceptance = derive_model_context_schedule_acceptance(profiles)
    assert acceptance.passed
    assert acceptance.occurrence_weighted_improvement > 1e-8
    assert acceptance.target_profile_degradation_count == 37

    forged = acceptance.model_copy(update={"target_profile_degradation_count": 0})
    with pytest.raises(ValueError, match="target_profile_degradation_count"):
        validate_model_context_schedule_acceptance(forged, profiles)


def _publication_evidence():
    trace_hash = _sha("publication-trace")
    profiles = (
        tuple(_profile_result(f"ten:{index}", 10, trace_hash=trace_hash) for index in range(26))
        + tuple(_profile_result(f"twenty:{index}", 20, trace_hash=trace_hash) for index in range(9))
        + tuple(_profile_result(f"thirty:{index}", 30, trace_hash=trace_hash) for index in range(2))
    )
    ordered = tuple(sorted(profiles, key=lambda item: item.target_hash))
    exact_k30 = (
        (0.47, 0.53, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    counts = (
        (1925, 2171, 0, 0),
        (0, 4096, 0, 0),
        (0, 0, 4096, 0),
        (0, 0, 0, 4096),
    )
    samples = tuple(
        results_module.ModelContextProfileSampleResult(
            target_hash=profile.target_hash,
            profile_hash=profile.model_profile_hash,
            model_context_artifact_hash=profile.model_context.artifact_hash,
            exact_k30_conditional=exact_k30,
            sampled_k30=derive_sampled_k30_evaluation(counts, exact_k30),
        )
        for profile in ordered
    )
    return ordered, derive_model_context_schedule_acceptance(ordered), samples


def _publication_summary():
    profiles, acceptance, samples = _publication_evidence()
    return results_module.build_model_context_pasym_swap_summary(
        request_hash=_sha("request"),
        profile_results=profiles,
        schedule_acceptance=acceptance,
        profile_samples=samples,
        thrml_k30_tv_tolerance=0.10,
    )


def test_model_context_publication_summary_round_trips_strict_evidence() -> None:
    summary = _publication_summary()

    assert (
        results_module.validate_model_context_pasym_swap_summary(summary.model_dump(mode="json"))
        == summary
    )
    assert len(summary.profile_results) == 37
    assert len(summary.profile_samples) == 37
    assert summary.model_trace_hash == _sha("publication-trace")
    assert summary.acceptance_passed


def test_model_context_publication_summary_rejects_changed_integer_count() -> None:
    payload = _publication_summary().model_dump(mode="json")
    payload["profile_samples"][0]["sampled_k30"]["counts"][0][0] -= 1
    payload["profile_samples"][0]["sampled_k30"]["counts"][0][1] += 1

    with pytest.raises(ValueError, match="conditional|residual"):
        results_module.validate_model_context_pasym_swap_summary(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["profile_samples"][0].__setitem__(
                "model_context_artifact_hash", _sha("forged-artifact")
            ),
            "model_context_artifact_hash",
        ),
        (
            lambda payload: payload.__setitem__("maximum_empirical_k30_residual", 0.0),
            "maximum_empirical_k30_residual",
        ),
        (
            lambda payload: payload.__setitem__("empirical_acceptance_passed", False),
            "empirical_acceptance_passed",
        ),
        (
            lambda payload: payload.__setitem__("thrml_k30_tv_tolerance", 0.01),
            "summary_hash",
        ),
    ],
)
def test_model_context_publication_summary_rejects_top_level_tampering(
    mutate, message: str
) -> None:
    payload = _publication_summary().model_dump(mode="json")
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        results_module.validate_model_context_pasym_swap_summary(payload)
