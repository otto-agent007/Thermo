"""Deterministic exact compiler for the checked two-bit PAsymSwap targets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import WORD_ORDER
from thermo_lab.schemas import PARAMETER_ORDER
from thermo_lab.thermodynamic_kernel import KernelParameters, equilibrium_conditional

_LOGICAL_ROLE_ORDER = ("input_0", "input_1", "hidden_0", "output_0", "output_1")
_TOPOLOGY_ID = "thermo_k3_2_v1"
_DTYPE = "float64"
_N_PARAMETERS = len(PARAMETER_ORDER)


def _as_float64_vector(values: object, *, name: str, length: int) -> NDArray[np.float64]:
    try:
        checked = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector") from error
    if checked.shape != (length,) or not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return checked


def _checked_pasym_target(target: object) -> NDArray[np.float64]:
    try:
        checked = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("target must be a numeric PAsymSwap conditional") from error
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


def _checked_context_weights(context_weights: object) -> NDArray[np.float64]:
    checked = _as_float64_vector(context_weights, name="context_weights", length=4)
    if np.any(checked < 0.0) or not np.isclose(checked.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("context_weights must be nonnegative and sum to one")
    return checked


@dataclass(frozen=True)
class CompilerSettings:
    """Exact, deterministic settings for the bounded three-restart optimizer."""

    parameter_cap: float = 4.0
    maxiter: int = 2000
    maxls: int = 50
    ftol: float = 1e-12
    gtol: float = 1e-9
    projected_gradient_tolerance: float = 1e-6
    initializations: tuple[tuple[float, ...], ...] = (
        (0.0,) * _N_PARAMETERS,
        (0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05),
        (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05),
    )
    context_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    def __post_init__(self) -> None:
        scalar_values = (
            self.parameter_cap,
            self.ftol,
            self.gtol,
            self.projected_gradient_tolerance,
        )
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in scalar_values):
            raise ValueError("compiler settings must use real scalar values")
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in scalar_values):
            raise ValueError("compiler settings must be finite and positive")
        if (
            type(self.maxiter) is not int
            or type(self.maxls) is not int
            or self.maxiter <= 0
            or self.maxls <= 0
        ):
            raise ValueError("maxiter and maxls must be positive integers")
        if len(self.initializations) != 3:
            raise ValueError("exactly three deterministic initializations are required")
        checked_initializations = tuple(
            tuple(
                float(value)
                for value in _as_float64_vector(initial, name="initialization", length=9)
            )
            for initial in self.initializations
        )
        if any(
            abs(value) > float(self.parameter_cap)
            for initial in checked_initializations
            for value in initial
        ):
            raise ValueError("initializations must respect parameter_cap")
        checked_weights = _checked_context_weights(self.context_weights)
        object.__setattr__(self, "parameter_cap", float(self.parameter_cap))
        object.__setattr__(self, "ftol", float(self.ftol))
        object.__setattr__(self, "gtol", float(self.gtol))
        object.__setattr__(
            self, "projected_gradient_tolerance", float(self.projected_gradient_tolerance)
        )
        object.__setattr__(self, "initializations", checked_initializations)
        object.__setattr__(
            self, "context_weights", tuple(float(value) for value in checked_weights)
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "parameter_cap": self.parameter_cap,
            "maxiter": self.maxiter,
            "maxls": self.maxls,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "projected_gradient_tolerance": self.projected_gradient_tolerance,
            "initializations": self.initializations,
            "context_weights": self.context_weights,
        }


@dataclass(frozen=True)
class OptimizationAttempt:
    """Bounded, immutable observation of one optimizer restart."""

    restart_index: int
    objective: float
    parameters: tuple[float, ...]
    raw_gradient_norm: float
    projected_gradient_norm: float
    scipy_success: bool
    passed_checks: bool
    iterations: int
    termination: str
    cap_active_parameter_count: int

    def __post_init__(self) -> None:
        try:
            parameters = tuple(float(value) for value in self.parameters)
        except (TypeError, ValueError) as error:
            raise ValueError("attempt parameters must be numeric") from error
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True)
class CompiledKernelArtifact:
    """Frozen learned kernel and bounded optimizer observations."""

    target_hash: str
    topology_id: str
    logical_role_order: tuple[str, str, str, str, str]
    parameter_order: tuple[str, ...]
    dtype: str
    parameters: KernelParameters
    beta: float
    parameter_cap: float
    settings: CompilerSettings
    attempts: tuple[OptimizationAttempt, ...]
    selected_restart: int
    objective: float
    projected_gradient_norm: float
    cap_active_parameter_count: int
    artifact_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, KernelParameters):
            raise TypeError("parameters must be KernelParameters")
        if not isinstance(self.settings, CompilerSettings):
            raise TypeError("settings must be CompilerSettings")
        attempts = tuple(self.attempts)
        if any(not isinstance(attempt, OptimizationAttempt) for attempt in attempts):
            raise TypeError("attempts must contain OptimizationAttempt records")
        logical_role_order = tuple(self.logical_role_order)
        parameter_order = tuple(self.parameter_order)
        parameters = KernelParameters(tuple(float(value) for value in self.parameters.values))
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "logical_role_order", logical_role_order)
        object.__setattr__(self, "parameter_order", parameter_order)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "artifact_hash", canonical_sha256(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        """Return only the scientific identity, excluding optimizer observations."""

        return {
            "target_hash": self.target_hash,
            "topology_id": self.topology_id,
            "logical_role_order": self.logical_role_order,
            "parameter_order": self.parameter_order,
            "dtype": self.dtype,
            "parameters": self.parameters.values,
            "beta": self.beta,
            "parameter_cap": self.parameter_cap,
            "compiler_settings": self.settings.identity_payload(),
        }


def _sufficient_statistics(
    input_index: int, hidden_spin: float, output_index: int
) -> NDArray[np.float64]:
    input_0, input_1 = 2.0 * np.asarray(WORD_ORDER[input_index], dtype=np.float64) - 1.0
    output_0, output_1 = 2.0 * np.asarray(WORD_ORDER[output_index], dtype=np.float64) - 1.0
    return np.asarray(
        (
            hidden_spin,
            output_0,
            output_1,
            input_0 * output_0,
            input_0 * output_1,
            input_1 * output_0,
            input_1 * output_1,
            hidden_spin * output_0,
            hidden_spin * output_1,
        ),
        dtype=np.float64,
    )


def _model_statistics(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return model p(y|x) and E[f | x,y] by exact hidden enumeration."""

    conditional = np.empty((4, 4), dtype=np.float64)
    conditioned_statistics = np.empty((4, 4, _N_PARAMETERS), dtype=np.float64)
    for input_index in range(4):
        log_weights = np.empty((4, 2), dtype=np.float64)
        statistics = np.empty((4, 2, _N_PARAMETERS), dtype=np.float64)
        for output_index in range(4):
            for hidden_index, hidden_spin in enumerate((-1.0, 1.0)):
                feature = _sufficient_statistics(input_index, hidden_spin, output_index)
                statistics[output_index, hidden_index] = feature
                log_weights[output_index, hidden_index] = float(np.dot(values, feature))
            hidden_log_normalizer = np.logaddexp.reduce(log_weights[output_index])
            hidden_probabilities = np.exp(log_weights[output_index] - hidden_log_normalizer)
            conditioned_statistics[input_index, output_index] = (
                hidden_probabilities @ statistics[output_index]
            )
        output_log_weights = np.logaddexp.reduce(log_weights, axis=1)
        conditional[input_index] = np.exp(
            output_log_weights - np.logaddexp.reduce(output_log_weights)
        )
    return conditional, conditioned_statistics


def loss_and_gradient(
    values: NDArray[np.generic], target: NDArray[np.generic], context_weights: NDArray[np.generic]
) -> tuple[float, NDArray[np.float64]]:
    """Return exact target-to-model KL and its float64 sufficient-statistic gradient."""

    checked_values = _as_float64_vector(values, name="values", length=_N_PARAMETERS)
    checked_target = _checked_pasym_target(target)
    checked_weights = _checked_context_weights(context_weights)
    model, conditioned_statistics = _model_statistics(checked_values)
    positive_target = checked_target > 0.0
    terms = np.zeros((4, 4), dtype=np.float64)
    terms[positive_target] = checked_target[positive_target] * (
        np.log(checked_target[positive_target]) - np.log(model[positive_target])
    )
    loss = float(np.dot(checked_weights, terms.sum(axis=1)))
    model_statistics = np.einsum("xy,xyf->xf", model, conditioned_statistics)
    target_statistics = np.einsum("xy,xyf->xf", checked_target, conditioned_statistics)
    gradient = np.einsum(
        "x,xf->f", checked_weights, model_statistics - target_statistics, dtype=np.float64
    )
    return loss, np.asarray(gradient, dtype=np.float64)


def project_gradient(
    values: NDArray[np.generic], gradient: NDArray[np.generic], parameter_cap: float
) -> NDArray[np.float64]:
    """Zero only components whose steepest descent direction points out of an active bound."""

    checked_values = _as_float64_vector(values, name="values", length=len(np.asarray(values)))
    checked_gradient = _as_float64_vector(gradient, name="gradient", length=len(checked_values))
    if not isinstance(parameter_cap, Real) or isinstance(parameter_cap, bool) or parameter_cap <= 0:
        raise ValueError("parameter_cap must be positive")
    cap = float(parameter_cap)
    if not math.isfinite(cap):
        raise ValueError("parameter_cap must be finite")
    projected = checked_gradient.copy()
    projected[(checked_values <= -cap) & (checked_gradient > 0.0)] = 0.0
    projected[(checked_values >= cap) & (checked_gradient < 0.0)] = 0.0
    return projected


def _attempt_from_result(
    result: Any, restart_index: int, target: NDArray[np.float64], settings: CompilerSettings
) -> OptimizationAttempt:
    result_values = np.asarray(getattr(result, "x", ()), dtype=np.float64)
    valid_values = result_values.shape == (_N_PARAMETERS,) and np.all(np.isfinite(result_values))
    if valid_values:
        objective, gradient = loss_and_gradient(
            result_values, target, np.asarray(settings.context_weights)
        )
        raw_norm = float(np.max(np.abs(gradient)))
        projected_norm = float(
            np.max(np.abs(project_gradient(result_values, gradient, settings.parameter_cap)))
        )
        values = tuple(float(value) for value in result_values)
    else:
        objective = float("nan")
        raw_norm = float("nan")
        projected_norm = float("nan")
        values = tuple(float(value) for value in result_values.ravel())
    reported_objective = float(getattr(result, "fun", float("nan")))
    observations_finite = (
        all(
            math.isfinite(value)
            for value in (reported_objective, objective, raw_norm, projected_norm)
        )
        and valid_values
    )
    scipy_success = bool(getattr(result, "success", False))
    checked_success = (
        scipy_success
        and observations_finite
        and projected_norm <= settings.projected_gradient_tolerance
        and all(abs(value) <= settings.parameter_cap for value in values)
    )
    return OptimizationAttempt(
        restart_index=restart_index,
        objective=objective,
        parameters=values,
        raw_gradient_norm=raw_norm,
        projected_gradient_norm=projected_norm,
        scipy_success=scipy_success,
        passed_checks=checked_success,
        iterations=int(getattr(result, "nit", 0)),
        termination=str(getattr(result, "message", "")),
        cap_active_parameter_count=sum(abs(value) >= settings.parameter_cap for value in values),
    )


def compile_target(
    target_hash: str, target: NDArray[np.generic], settings: CompilerSettings
) -> CompiledKernelArtifact:
    """Compile one target using all checked bounded deterministic restarts."""

    if not isinstance(target_hash, str) or not target_hash:
        raise ValueError("target_hash must be a nonempty string")
    if not isinstance(settings, CompilerSettings):
        raise TypeError("settings must be CompilerSettings")
    checked_target = _checked_pasym_target(target)
    attempts: list[OptimizationAttempt] = []
    bounds = [(-settings.parameter_cap, settings.parameter_cap)] * _N_PARAMETERS
    context_weights = np.asarray(settings.context_weights, dtype=np.float64)
    for restart_index, initialization in enumerate(settings.initializations):
        result = minimize(
            fun=lambda values: loss_and_gradient(values, checked_target, context_weights)[0],
            x0=np.asarray(initialization, dtype=np.float64),
            jac=lambda values: loss_and_gradient(values, checked_target, context_weights)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": settings.maxiter,
                "maxls": settings.maxls,
                "ftol": settings.ftol,
                "gtol": settings.gtol,
            },
        )
        attempts.append(_attempt_from_result(result, restart_index, checked_target, settings))
    passing = [attempt for attempt in attempts if attempt.passed_checks]
    if not passing:
        raise ValueError(f"target {target_hash}: no checked restart passed")
    winner = min(passing, key=lambda attempt: (attempt.objective, attempt.parameters))
    parameters = KernelParameters(tuple(float(value) for value in winner.parameters))
    artifact = CompiledKernelArtifact(
        target_hash=target_hash,
        topology_id=_TOPOLOGY_ID,
        logical_role_order=_LOGICAL_ROLE_ORDER,
        parameter_order=PARAMETER_ORDER,
        dtype=_DTYPE,
        parameters=parameters,
        beta=1.0,
        parameter_cap=settings.parameter_cap,
        settings=settings,
        attempts=tuple(attempts),
        selected_restart=winner.restart_index,
        objective=winner.objective,
        projected_gradient_norm=winner.projected_gradient_norm,
        cap_active_parameter_count=winner.cap_active_parameter_count,
    )
    return artifact


def evaluate_artifact(artifact: CompiledKernelArtifact) -> NDArray[np.float64]:
    """Evaluate a frozen artifact without optimizer or target-state access."""

    if not isinstance(artifact, CompiledKernelArtifact):
        raise TypeError("artifact must be CompiledKernelArtifact")
    return equilibrium_conditional(artifact.parameters, beta=artifact.beta)
