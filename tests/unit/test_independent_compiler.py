import dataclasses
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

import thermo_lab.independent_compiler as independent_compiler
from thermo_lab.hashing import canonical_sha256
from thermo_lab.independent_compiler import (
    CompiledKernelArtifact,
    CompilerSettings,
    OptimizationAttempt,
    compile_target,
    evaluate_artifact,
    loss_and_gradient,
    project_gradient,
)
from thermo_lab.pasym_swap import build_paper_fixture, build_pasym_swap_conditional
from thermo_lab.schemas import PARAMETER_ORDER
from thermo_lab.thermodynamic_kernel import KernelParameters

PARAMETERS = np.asarray((0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9))
TARGET = np.asarray(build_pasym_swap_conditional(0.03, 0.07))


def central_difference(function: object, *, epsilon: float = 1e-6) -> np.ndarray:
    values = PARAMETERS.copy()
    numeric = np.empty(len(values))
    for index in range(len(values)):
        offset = np.zeros(len(values))
        offset[index] = epsilon
        numeric[index] = (function(values + offset) - function(values - offset)) / (2.0 * epsilon)  # type: ignore[operator]
    return numeric


def _legacy_checked_pasym_target(target: object) -> np.ndarray:
    """Test-local copy of the pre-hardening valid-target validator."""

    checked = np.asarray(target, dtype=np.float64)
    if checked.shape != (4, 4) or not np.all(np.isfinite(checked)):
        raise ValueError("target must be a finite (4, 4) PAsymSwap conditional")
    if np.any(checked < 0.0) or not np.allclose(checked.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("target must be a stochastic PAsymSwap conditional")
    if not (
        np.array_equal(checked[0], np.asarray((1.0, 0.0, 0.0, 0.0)))
        and np.array_equal(checked[3], np.asarray((0.0, 0.0, 0.0, 1.0)))
        and checked[1, 0] == 0.0
        and checked[1, 3] == 0.0
        and checked[2, 0] == 0.0
        and checked[2, 3] == 0.0
        and 0.0 < checked[1, 2] < 1.0
        and 0.0 < checked[2, 1] < 1.0
    ):
        raise ValueError("target must use the checked two-bit PAsymSwap support")
    return checked


@pytest.fixture(scope="module")
def paper_artifact_pairs() -> tuple[tuple[CompiledKernelArtifact, CompiledKernelArtifact], ...]:
    """Compile the canonical targets once per validator path for exact regressions."""

    fixture = build_paper_fixture()
    settings = checked_compiler_settings()
    hardened_validator = independent_compiler._checked_pasym_target
    try:
        independent_compiler._checked_pasym_target = _legacy_checked_pasym_target
        legacy = tuple(
            compile_target(target.target_hash, np.asarray(target.conditional), settings)
            for target in fixture.targets
        )
    finally:
        independent_compiler._checked_pasym_target = hardened_validator
    hardened = tuple(
        compile_target(target.target_hash, np.asarray(target.conditional), settings)
        for target in fixture.targets
    )
    return tuple(zip(legacy, hardened, strict=True))


def checked_compiler_settings() -> CompilerSettings:
    alternating = tuple(0.05 if index % 2 == 0 else -0.05 for index in range(9))
    return CompilerSettings(
        parameter_cap=2.0,
        maxiter=2000,
        maxls=50,
        ftol=1e-12,
        gtol=1e-9,
        projected_gradient_tolerance=1e-6,
        initializations=((0.0,) * 9, alternating, tuple(-value for value in alternating)),
    )


def test_exact_gradient_matches_central_difference() -> None:
    values = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9])
    target = np.asarray(build_pasym_swap_conditional(0.03, 0.07))
    loss, gradient = loss_and_gradient(values, target, np.full(4, 0.25))
    epsilon = 1e-6
    numeric = np.empty(9)
    for index in range(9):
        offset = np.zeros(9)
        offset[index] = epsilon
        plus = loss_and_gradient(values + offset, target, np.full(4, 0.25))[0]
        minus = loss_and_gradient(values - offset, target, np.full(4, 0.25))[0]
        numeric[index] = (plus - minus) / (2.0 * epsilon)
    assert math.isfinite(loss)
    assert gradient.dtype == np.float64
    np.testing.assert_allclose(gradient, numeric, atol=2e-7, rtol=2e-6)


def test_pasym_target_validation_uses_zero_relative_tolerance() -> None:
    target = np.asarray(build_pasym_swap_conditional(0.03, 0.07))
    target[1, 1] += 1e-8

    with pytest.raises(ValueError, match="stochastic"):
        loss_and_gradient(np.zeros(9), target, np.full(4, 0.25))


def test_nonuniform_zero_context_gradient_matches_central_difference() -> None:
    weights = np.asarray((0.60, 0.25, 0.15, 0.0))

    observed_loss, observed_gradient = loss_and_gradient(PARAMETERS, TARGET, weights)
    numeric = central_difference(lambda values: loss_and_gradient(values, TARGET, weights)[0])

    assert math.isfinite(observed_loss)
    np.testing.assert_allclose(observed_gradient, numeric, rtol=1e-5, atol=1e-7)


def test_absolute_target_validation_preserves_all_canonical_artifact_identities(
    paper_artifact_pairs: tuple[tuple[CompiledKernelArtifact, CompiledKernelArtifact], ...],
) -> None:
    assert len(paper_artifact_pairs) == 37
    for legacy, hardened in paper_artifact_pairs:
        assert hardened.parameters.values == legacy.parameters.values
        assert hardened.objective == legacy.objective
        assert hardened.attempts == legacy.attempts
        assert hardened.selected_restart == legacy.selected_restart
        assert hardened.artifact_hash == legacy.artifact_hash


def test_projected_gradient_zeros_only_blocked_descent_components() -> None:
    values = np.asarray([-4.0, 4.0, 0.0])
    gradient = np.asarray([2.0, -3.0, 5.0])
    np.testing.assert_array_equal(project_gradient(values, gradient, 4.0), [0.0, 0.0, 5.0])


def test_compiler_is_deterministic_and_freezes_artifact_identity() -> None:
    target = np.asarray(build_pasym_swap_conditional(0.03, 0.07))
    settings = checked_compiler_settings()
    first = compile_target("target-hash", target, settings)
    second = compile_target("target-hash", target, settings)

    assert first == second
    assert first.selected_restart in {0, 1, 2}
    assert first.projected_gradient_norm <= 1e-6
    assert max(abs(value) for value in first.parameters.values) <= 2.0
    assert first.artifact_hash == canonical_sha256(first.identity_payload())
    assert first.attempts[first.selected_restart].scipy_success
    assert first.attempts[first.selected_restart].passed_checks
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.selected_restart = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("change", "expected_identity_key"),
    [
        (lambda artifact: dataclasses.replace(artifact, target_hash="other-target"), "target_hash"),
        (
            lambda artifact: dataclasses.replace(artifact, topology_id="other-topology"),
            "topology_id",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact,
                logical_role_order=("input_1", "input_0", "hidden_0", "output_0", "output_1"),
            ),
            "logical_role_order",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact, parameter_order=tuple(reversed(PARAMETER_ORDER))
            ),
            "parameter_order",
        ),
        (lambda artifact: dataclasses.replace(artifact, dtype="float32"), "dtype"),
        (lambda artifact: dataclasses.replace(artifact, beta=0.5), "beta"),
        (lambda artifact: dataclasses.replace(artifact, parameter_cap=3.5), "parameter_cap"),
        (
            lambda artifact: dataclasses.replace(
                artifact, parameters=KernelParameters((0.1,) + artifact.parameters.values[1:])
            ),
            "parameters",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact,
                settings=dataclasses.replace(
                    artifact.settings, maxiter=artifact.settings.maxiter + 1
                ),
            ),
            "compiler_settings",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact,
                settings=dataclasses.replace(artifact.settings, parameter_cap=3.5),
            ),
            "compiler_settings",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact,
                settings=dataclasses.replace(artifact.settings, maxls=artifact.settings.maxls + 1),
            ),
            "compiler_settings",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact,
                settings=dataclasses.replace(artifact.settings, ftol=1e-11),
            ),
            "compiler_settings",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact,
                settings=dataclasses.replace(artifact.settings, gtol=1e-8),
            ),
            "compiler_settings",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact,
                settings=dataclasses.replace(artifact.settings, projected_gradient_tolerance=1e-5),
            ),
            "compiler_settings",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact,
                settings=dataclasses.replace(
                    artifact.settings,
                    initializations=((0.01,) * 9, *artifact.settings.initializations[1:]),
                ),
            ),
            "compiler_settings",
        ),
        (
            lambda artifact: dataclasses.replace(
                artifact,
                settings=dataclasses.replace(
                    artifact.settings, context_weights=(0.2, 0.3, 0.2, 0.3)
                ),
            ),
            "compiler_settings",
        ),
    ],
)
def test_artifact_identity_hash_covers_each_scientific_input(
    change: object, expected_identity_key: str
) -> None:
    artifact = example_artifact()
    changed = change(artifact)  # type: ignore[operator]

    assert expected_identity_key in artifact.identity_payload()
    assert canonical_sha256(artifact.identity_payload()) != canonical_sha256(
        changed.identity_payload()
    )


def test_artifact_identity_excludes_optimizer_diagnostics() -> None:
    artifact = example_artifact()
    changed_attempt = dataclasses.replace(
        artifact.attempts[0], iterations=999, termination="different timing and termination"
    )
    changed = dataclasses.replace(
        artifact,
        attempts=(changed_attempt, *artifact.attempts[1:]),
        objective=42.0,
        projected_gradient_norm=2.0,
    )

    assert artifact.identity_payload() == changed.identity_payload()
    assert canonical_sha256(artifact.identity_payload()) == canonical_sha256(
        changed.identity_payload()
    )


def test_artifact_derives_hash_and_copies_nested_records() -> None:
    artifact = example_artifact()
    forged = dataclasses.replace(artifact, target_hash="forged-target")
    mutable_parameters = [0.0] * 9
    attempt = OptimizationAttempt(
        restart_index=0,
        objective=0.1,
        parameters=mutable_parameters,  # type: ignore[arg-type]
        raw_gradient_norm=0.01,
        projected_gradient_norm=0.01,
        scipy_success=True,
        passed_checks=True,
        iterations=3,
        termination="converged",
        cap_active_parameter_count=0,
    )
    mutable_attempts = [attempt]
    copied = dataclasses.replace(artifact, attempts=mutable_attempts)  # type: ignore[arg-type]
    mutable_parameters[0] = 1.0
    mutable_attempts.clear()

    assert forged.artifact_hash == canonical_sha256(forged.identity_payload())
    assert attempt.parameters == (0.0,) * 9
    assert copied.attempts == (attempt,)


def test_artifact_copies_kernel_parameters_from_mutable_backing_storage() -> None:
    mutable_values = [0.0] * 9
    artifact = example_artifact(parameters=KernelParameters(mutable_values))
    identity = artifact.identity_payload()
    artifact_hash = artifact.artifact_hash
    mutable_values[0] = 1.0

    assert artifact.parameters.values == (0.0,) * 9
    assert artifact.identity_payload() == identity
    assert artifact.artifact_hash == artifact_hash == canonical_sha256(artifact.identity_payload())


def test_two_target_compilations_have_isolated_attempts_and_parameters() -> None:
    settings = checked_compiler_settings()
    first = compile_target("first", np.asarray(build_pasym_swap_conditional(0.03, 0.07)), settings)
    second = compile_target(
        "second", np.asarray(build_pasym_swap_conditional(0.07, 0.03)), settings
    )

    assert first.target_hash != second.target_hash
    assert first.attempts is not second.attempts
    assert first.parameters.values is not second.parameters.values


def test_evaluation_accepts_only_artifact_without_optimizer_context() -> None:
    assert tuple(inspect.signature(evaluate_artifact).parameters) == ("artifact",)
    observed = evaluate_artifact(example_artifact())

    assert observed.shape == (4, 4)
    assert observed.dtype == np.float64
    with pytest.raises(TypeError):
        evaluate_artifact(KernelParameters((0.0,) * 9))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(
            success=True,
            fun=float("nan"),
            x=np.zeros(9),
            nit=2,
            message="bad objective",
        ),
        SimpleNamespace(
            success=False,
            fun=1.0,
            x=np.zeros(9),
            nit=2,
            message="status failed",
        ),
        SimpleNamespace(
            success=True,
            fun=1.0,
            x=np.zeros(9),
            nit=2,
            message="gradient too large",
        ),
    ],
)
def test_compile_target_rejects_every_unchecked_restart(
    monkeypatch: pytest.MonkeyPatch, result: SimpleNamespace
) -> None:
    monkeypatch.setattr("thermo_lab.independent_compiler.minimize", lambda *args, **kwargs: result)
    target = np.asarray(build_pasym_swap_conditional(0.03, 0.07))

    with pytest.raises(ValueError, match="target-for-failure.*no checked restart passed"):
        compile_target("target-for-failure", target, checked_compiler_settings())


def example_artifact(parameters: KernelParameters | None = None) -> CompiledKernelArtifact:
    settings = checked_compiler_settings()
    checked_parameters = parameters or KernelParameters((0.0,) * 9)
    attempt = OptimizationAttempt(
        restart_index=0,
        objective=0.1,
        parameters=checked_parameters.values,
        raw_gradient_norm=0.01,
        projected_gradient_norm=0.01,
        scipy_success=True,
        passed_checks=True,
        iterations=3,
        termination="converged",
        cap_active_parameter_count=0,
    )
    artifact = CompiledKernelArtifact(
        target_hash="target-hash",
        topology_id="thermo_k3_2_v1",
        logical_role_order=("input_0", "input_1", "hidden_0", "output_0", "output_1"),
        parameter_order=PARAMETER_ORDER,
        dtype="float64",
        parameters=checked_parameters,
        beta=1.0,
        parameter_cap=2.0,
        settings=settings,
        attempts=(attempt,),
        selected_restart=0,
        objective=0.1,
        projected_gradient_norm=0.01,
        cap_active_parameter_count=0,
    )
    return artifact
