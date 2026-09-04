"""Checked paired compilation for pooled PAsymSwap target contexts."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import thermo_lab.target_context_compiler as target_context_compiler
from thermo_lab.independent_compiler import CompilerSettings, compile_target
from thermo_lab.pasym_swap import WORD_ORDER, build_pasym_swap_conditional
from thermo_lab.pasym_swap_context import PooledTargetContextProfile
from thermo_lab.target_context_compiler import (
    TARGET_CONTEXT_START_ROLES,
    TargetContextOptimizationAttempt,
    compile_paired_target,
    compile_target_context,
    evaluate_target_context_artifact,
)

TARGET_HASH = "target-hash"
TARGET = np.asarray(build_pasym_swap_conditional(0.03, 0.07))
PROFILE = PooledTargetContextProfile(
    trace_hash="trace-hash",
    target_hash=TARGET_HASH,
    word_order=WORD_ORDER,
    context_reduction="equal_occurrence_mean_by_target_hash",
    zero_support_policy="exact_unsmoothed",
    occurrence_indices=(0,),
    multiplicity=1,
    context_weights=(0.60, 0.25, 0.15, 0.0),
    support_mask=(True, True, True, False),
)


def checked_baseline_settings() -> CompilerSettings:
    return CompilerSettings()


def checked_target_settings() -> CompilerSettings:
    return dataclasses.replace(checked_baseline_settings(), context_weights=PROFILE.context_weights)


@pytest.fixture(scope="module")
def direct_baseline():
    """Compile the canonical baseline once: compiler paths are intentionally expensive."""

    return compile_target(TARGET_HASH, TARGET, checked_baseline_settings())


@pytest.fixture(scope="module")
def compiled_pair():
    """Compile the baseline plus exactly four target-context attempts once."""

    return compile_paired_target(
        TARGET_HASH,
        TARGET,
        PROFILE,
        checked_baseline_settings(),
        checked_target_settings(),
    )


def test_paired_compile_preserves_direct_independent_baseline_identity(
    direct_baseline, compiled_pair
) -> None:
    """Changing the paired path must not alter the established uniform compiler."""

    assert compiled_pair.baseline == direct_baseline
    assert compiled_pair.baseline.artifact_hash == direct_baseline.artifact_hash


def test_target_compiler_runs_four_labeled_starts_in_exact_order(compiled_pair) -> None:
    """Dropping or reordering a target attempt changes the checked optimization protocol."""

    assert tuple(item.start_role for item in compiled_pair.target_context.attempts) == (
        TARGET_CONTEXT_START_ROLES
    )
    assert compiled_pair.target_context.start_values == (
        compiled_pair.baseline.parameters.values,
        *checked_target_settings().initializations,
    )


def test_target_context_evaluation_needs_only_the_frozen_artifact(compiled_pair) -> None:
    observed = evaluate_target_context_artifact(compiled_pair.target_context)

    assert observed.shape == (4, 4)
    assert observed.dtype == np.float64
    with pytest.raises(TypeError):
        evaluate_target_context_artifact(compiled_pair.baseline)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target_hash, profile, baseline_change, settings, match",
    [
        ("other-target", PROFILE, None, checked_target_settings(), "profile target_hash"),
        (
            TARGET_HASH,
            dataclasses.replace(PROFILE, target_hash="other-target"),
            None,
            checked_target_settings(),
            "profile target_hash",
        ),
        (
            TARGET_HASH,
            PROFILE,
            {"target_hash": "other-target"},
            checked_target_settings(),
            "baseline",
        ),
        (
            TARGET_HASH,
            PROFILE,
            {
                "settings": dataclasses.replace(
                    checked_baseline_settings(), context_weights=(0.4, 0.2, 0.2, 0.2)
                )
            },
            checked_target_settings(),
            "uniform",
        ),
        (
            TARGET_HASH,
            PROFILE,
            None,
            dataclasses.replace(
                checked_target_settings(), context_weights=(0.25, 0.25, 0.25, 0.25)
            ),
            "context_weights",
        ),
        (
            TARGET_HASH,
            PROFILE,
            None,
            dataclasses.replace(checked_target_settings(), maxiter=7),
            "maxiter",
        ),
        (
            TARGET_HASH,
            PROFILE,
            None,
            dataclasses.replace(
                checked_target_settings(),
                initializations=((0.01,) * 9, *checked_target_settings().initializations[1:]),
            ),
            "initializations",
        ),
    ],
)
def test_target_context_rejects_mismatched_paired_inputs_before_optimization(
    direct_baseline, target_hash, profile, baseline_change, settings, match
) -> None:
    """A mismatched pair must fail before it can produce a misleading comparison."""

    baseline = (
        dataclasses.replace(direct_baseline, **baseline_change)
        if baseline_change is not None
        else direct_baseline
    )
    with pytest.raises((TypeError, ValueError), match=match):
        compile_target_context(target_hash, TARGET, profile, baseline, settings)


def test_target_context_recomputes_endpoint_math_instead_of_trusting_scipy(
    monkeypatch: pytest.MonkeyPatch, direct_baseline
) -> None:
    """A bogus SciPy objective/Jacobian cannot become stored scientific evidence."""

    endpoint = np.full(9, 0.2)
    monkeypatch.setattr(
        target_context_compiler,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, fun=999.0, jac=np.full(9, 777.0), x=endpoint, nit=5, message="reported"
        ),
    )
    monkeypatch.setattr(
        target_context_compiler,
        "loss_and_gradient",
        lambda values, target, weights: (0.25, np.zeros(9)),
    )

    artifact = compile_target_context(
        TARGET_HASH, TARGET, PROFILE, direct_baseline, checked_target_settings()
    )

    assert artifact.objective == 0.25
    assert all(item.objective == 0.25 for item in artifact.attempts)
    assert all(item.raw_gradient_norm == 0.0 for item in artifact.attempts)
    assert all(item.projected_gradient_norm == 0.0 for item in artifact.attempts)


def test_target_context_rejects_scipy_failure_even_when_endpoint_gradient_is_small(
    monkeypatch: pytest.MonkeyPatch, direct_baseline
) -> None:
    monkeypatch.setattr(
        target_context_compiler,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, fun=0.0, x=np.zeros(9), nit=1, message="failed"
        ),
    )
    monkeypatch.setattr(
        target_context_compiler,
        "loss_and_gradient",
        lambda values, target, weights: (0.0, np.zeros(9)),
    )

    with pytest.raises(RuntimeError, match="No target-context optimizer endpoint passed"):
        compile_target_context(
            TARGET_HASH, TARGET, PROFILE, direct_baseline, checked_target_settings()
        )


def test_target_context_rejects_a_malformed_endpoint_even_if_later_starts_pass(
    monkeypatch: pytest.MonkeyPatch, direct_baseline
) -> None:
    """Malformed SciPy parameters must not be replaced by a fabricated endpoint."""

    results = iter(
        (
            SimpleNamespace(success=False, fun=0.0, x=np.zeros(8), nit=1, message="malformed"),
            *(
                SimpleNamespace(success=True, fun=0.0, x=np.zeros(9), nit=1, message="passing")
                for _ in range(3)
            ),
        )
    )
    monkeypatch.setattr(
        target_context_compiler,
        "minimize",
        lambda *args, **kwargs: next(results),
    )
    monkeypatch.setattr(
        target_context_compiler,
        "loss_and_gradient",
        lambda values, target, weights: (0.0, np.zeros(9)),
    )

    with pytest.raises(ValueError, match="malformed optimizer endpoint"):
        compile_target_context(
            TARGET_HASH, TARGET, PROFILE, direct_baseline, checked_target_settings()
        )


def test_target_context_retains_a_finite_failed_endpoint_while_another_start_passes(
    monkeypatch: pytest.MonkeyPatch, direct_baseline
) -> None:
    """A valid endpoint survives recording even when SciPy reports failure."""

    failed_endpoint = np.full(9, 0.1)
    results = iter(
        (
            SimpleNamespace(
                success=False, fun=0.0, x=failed_endpoint, nit=1, message="reported failure"
            ),
            *(
                SimpleNamespace(success=True, fun=0.0, x=np.zeros(9), nit=1, message="passing")
                for _ in range(3)
            ),
        )
    )
    monkeypatch.setattr(
        target_context_compiler,
        "minimize",
        lambda *args, **kwargs: next(results),
    )
    monkeypatch.setattr(
        target_context_compiler,
        "loss_and_gradient",
        lambda values, target, weights: (0.0, np.zeros(9)),
    )

    artifact = compile_target_context(
        TARGET_HASH, TARGET, PROFILE, direct_baseline, checked_target_settings()
    )

    assert artifact.attempts[0].parameters == (0.1,) * 9
    assert not artifact.attempts[0].scipy_success
    assert not artifact.attempts[0].passed_checks


def test_target_context_selects_exact_objective_then_lexicographic_parameters(
    monkeypatch: pytest.MonkeyPatch, direct_baseline
) -> None:
    """Equal objectives must not make selection depend on optimizer return order."""

    endpoints = iter(
        (
            np.full(9, 0.1),
            np.full(9, 0.2),
            np.full(9, 0.3),
            np.full(9, -0.3),
        )
    )
    monkeypatch.setattr(
        target_context_compiler,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, fun=-999.0, x=next(endpoints), nit=2, message="reported"
        ),
    )
    objectives = {0.0: 4.0, 0.1: 2.0, 0.2: 1.0, 0.3: 1.0, -0.3: 1.0}
    monkeypatch.setattr(
        target_context_compiler,
        "loss_and_gradient",
        lambda values, target, weights: (objectives[round(float(values[0]), 1)], np.zeros(9)),
    )

    artifact = compile_target_context(
        TARGET_HASH, TARGET, PROFILE, direct_baseline, checked_target_settings()
    )

    assert artifact.selected_start_index == 3
    assert artifact.selected_start_role == "fixed_antithetic_negative"
    assert artifact.parameters.values == (-0.3,) * 9


def test_target_context_identity_has_exact_scope_and_excludes_attempt_diagnostics(
    compiled_pair,
) -> None:
    artifact = compiled_pair.target_context
    payload = artifact.identity_payload()
    changed = dataclasses.replace(
        artifact,
        attempts=(
            dataclasses.replace(artifact.attempts[0], iterations=999, termination="different"),
            *artifact.attempts[1:],
        ),
        objective=999.0,
        projected_gradient_norm=2.0,
        cap_active_parameter_count=7,
    )

    assert set(payload) == {
        "identity_version",
        "target_hash",
        "profile_hash",
        "context_weights",
        "baseline_artifact_hash",
        "topology_id",
        "logical_role_order",
        "parameter_order",
        "dtype",
        "parameters",
        "beta",
        "parameter_cap",
        "compiler_settings",
    }
    assert set(payload["compiler_settings"]) == {
        "optimizer",
        "maxiter",
        "maxls",
        "ftol",
        "gtol",
        "projected_gradient_tolerance",
        "start_roles",
        "start_values",
        "restart_selection",
    }
    assert payload["identity_version"] == "target_context_artifact.v1"
    assert artifact.artifact_hash == changed.artifact_hash
    assert payload == changed.identity_payload()


def test_target_context_artifact_defensively_copies_nested_start_and_attempt_records(
    compiled_pair,
) -> None:
    artifact = compiled_pair.target_context
    mutable_starts = [list(values) for values in artifact.start_values]
    mutable_parameters = list(artifact.attempts[0].parameters)
    mutable_attempt = TargetContextOptimizationAttempt(
        start_index=0,
        start_role="uniform_baseline_warm_start",
        objective=artifact.attempts[0].objective,
        parameters=mutable_parameters,  # type: ignore[arg-type]
        raw_gradient_norm=artifact.attempts[0].raw_gradient_norm,
        projected_gradient_norm=artifact.attempts[0].projected_gradient_norm,
        scipy_success=True,
        passed_checks=True,
        iterations=1,
        termination="copied",
        cap_active_parameter_count=0,
    )
    mutable_attempts = [mutable_attempt, *artifact.attempts[1:]]
    copied = dataclasses.replace(
        artifact,
        start_values=mutable_starts,  # type: ignore[arg-type]
        attempts=mutable_attempts,  # type: ignore[arg-type]
    )
    identity = copied.identity_payload()
    artifact_hash = copied.artifact_hash
    mutable_starts[0][0] = 1.5
    mutable_parameters[0] = 1.5
    mutable_attempts.clear()

    assert copied.start_values[0] == artifact.start_values[0]
    assert len(copied.attempts) == 4
    assert copied.attempts[0] == mutable_attempt
    assert mutable_attempt.parameters[0] != 1.5
    assert copied.identity_payload() == identity
    assert copied.artifact_hash == artifact_hash


def test_target_context_compiles_against_cached_baseline_without_recompiling_it(
    monkeypatch: pytest.MonkeyPatch, direct_baseline
) -> None:
    """Cached backend calls must not trigger uniform compilation a second time."""

    monkeypatch.setattr(
        target_context_compiler,
        "compile_target",
        lambda *args, **kwargs: pytest.fail("compile_target must not be called"),
    )
    monkeypatch.setattr(
        target_context_compiler,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, fun=0.0, x=np.zeros(9), nit=1, message="reported"
        ),
    )
    monkeypatch.setattr(
        target_context_compiler,
        "loss_and_gradient",
        lambda values, target, weights: (0.0, np.zeros(9)),
    )

    artifact = compile_target_context(
        TARGET_HASH, TARGET, PROFILE, direct_baseline, checked_target_settings()
    )

    assert len(artifact.attempts) == 4
