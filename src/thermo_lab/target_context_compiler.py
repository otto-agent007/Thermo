"""Checked paired compilation for pooled PAsymSwap target contexts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from thermo_lab.hashing import canonical_sha256
from thermo_lab.independent_compiler import (
    CompiledKernelArtifact,
    CompilerSettings,
    compile_target,
    loss_and_gradient,
    project_gradient,
)
from thermo_lab.pasym_swap_context import ContextWeights, PooledTargetContextProfile
from thermo_lab.schemas import PARAMETER_ORDER
from thermo_lab.thermodynamic_kernel import KernelParameters, equilibrium_conditional

TargetContextStartRole = Literal[
    "uniform_baseline_warm_start",
    "fixed_zero",
    "fixed_positive",
    "fixed_antithetic_negative",
]
TARGET_CONTEXT_START_ROLES: tuple[TargetContextStartRole, ...] = (
    "uniform_baseline_warm_start",
    "fixed_zero",
    "fixed_positive",
    "fixed_antithetic_negative",
)

_UNIFORM_CONTEXT_WEIGHTS: ContextWeights = (0.25, 0.25, 0.25, 0.25)
_LOGICAL_ROLE_ORDER = ("input_0", "input_1", "hidden_0", "output_0", "output_1")
_TOPOLOGY_ID = "thermo_k3_2_v1"
_DTYPE = "float64"
_N_PARAMETERS = len(PARAMETER_ORDER)


def _vector(values: object, *, name: str) -> NDArray[np.float64]:
    try:
        checked = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector") from error
    if checked.shape != (_N_PARAMETERS,) or not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must be a finite vector of length {_N_PARAMETERS}")
    return checked


def _context_weights(values: object) -> ContextWeights:
    try:
        checked = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("context_weights must be numeric") from error
    if (
        checked.shape != (4,)
        or not np.all(np.isfinite(checked))
        or np.any(checked < 0.0)
        or not np.isclose(checked.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("context_weights must be nonnegative and sum to one")
    return tuple(float(value) for value in checked)  # type: ignore[return-value]


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


@dataclass(frozen=True)
class TargetContextOptimizationAttempt:
    """Immutable checked observation of one target-context optimizer start."""

    start_index: int
    start_role: TargetContextStartRole
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
        if type(self.start_index) is not int or self.start_index < 0:
            raise ValueError("start_index must be a nonnegative integer")
        if self.start_role not in TARGET_CONTEXT_START_ROLES:
            raise ValueError("start_role must be a checked target-context start role")
        checked_parameters = _vector(self.parameters, name="attempt parameters")
        object.__setattr__(self, "parameters", tuple(float(value) for value in checked_parameters))


@dataclass(frozen=True)
class TargetContextCompiledKernelArtifact:
    """Frozen target-context kernel with bounded four-start observations."""

    target_hash: str
    profile_hash: str
    context_weights: ContextWeights
    baseline_artifact_hash: str
    topology_id: str
    logical_role_order: tuple[str, ...]
    parameter_order: tuple[str, ...]
    dtype: str
    parameters: KernelParameters
    beta: float
    parameter_cap: float
    settings: CompilerSettings
    start_values: tuple[tuple[float, ...], ...]
    attempts: tuple[TargetContextOptimizationAttempt, ...]
    selected_start_index: int
    selected_start_role: TargetContextStartRole
    objective: float
    projected_gradient_norm: float
    cap_active_parameter_count: int
    artifact_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty_string(self.target_hash, name="target_hash")
        _nonempty_string(self.profile_hash, name="profile_hash")
        _nonempty_string(self.baseline_artifact_hash, name="baseline_artifact_hash")
        if not isinstance(self.parameters, KernelParameters):
            raise TypeError("parameters must be KernelParameters")
        if not isinstance(self.settings, CompilerSettings):
            raise TypeError("settings must be CompilerSettings")
        if self.selected_start_role not in TARGET_CONTEXT_START_ROLES:
            raise ValueError("selected_start_role must be a checked target-context start role")
        if type(self.selected_start_index) is not int:
            raise TypeError("selected_start_index must be an integer")
        start_values = tuple(
            tuple(float(value) for value in _vector(start, name="start value"))
            for start in self.start_values
        )
        attempts = tuple(self.attempts)
        if len(start_values) != len(TARGET_CONTEXT_START_ROLES):
            raise ValueError("exactly four target-context start values are required")
        if len(attempts) != len(TARGET_CONTEXT_START_ROLES) or any(
            not isinstance(attempt, TargetContextOptimizationAttempt) for attempt in attempts
        ):
            raise ValueError("exactly four target-context optimization attempts are required")
        expected_starts = tuple(range(len(TARGET_CONTEXT_START_ROLES)))
        if (
            tuple(attempt.start_index for attempt in attempts) != expected_starts
            or tuple(attempt.start_role for attempt in attempts) != TARGET_CONTEXT_START_ROLES
        ):
            raise ValueError("target-context attempts must use the checked start order")
        if self.selected_start_index not in expected_starts or (
            self.selected_start_role != TARGET_CONTEXT_START_ROLES[self.selected_start_index]
        ):
            raise ValueError("selected target-context start must match the checked start order")
        if (
            not isinstance(self.parameter_cap, Real)
            or isinstance(self.parameter_cap, bool)
            or not math.isfinite(float(self.parameter_cap))
            or float(self.parameter_cap) <= 0.0
        ):
            raise ValueError("parameter_cap must be finite and positive")
        context_weights = _context_weights(self.context_weights)
        parameters = KernelParameters(tuple(float(value) for value in self.parameters.values))
        object.__setattr__(self, "context_weights", context_weights)
        object.__setattr__(self, "logical_role_order", tuple(self.logical_role_order))
        object.__setattr__(self, "parameter_order", tuple(self.parameter_order))
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "start_values", start_values)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "parameter_cap", float(self.parameter_cap))
        object.__setattr__(self, "artifact_hash", canonical_sha256(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        """Return the exact v1 target-context scientific identity payload."""

        return {
            "identity_version": "target_context_artifact.v1",
            "target_hash": self.target_hash,
            "profile_hash": self.profile_hash,
            "context_weights": self.context_weights,
            "baseline_artifact_hash": self.baseline_artifact_hash,
            "topology_id": self.topology_id,
            "logical_role_order": self.logical_role_order,
            "parameter_order": self.parameter_order,
            "dtype": self.dtype,
            "parameters": self.parameters.values,
            "beta": self.beta,
            "parameter_cap": self.parameter_cap,
            "compiler_settings": {
                "optimizer": "scipy_lbfgsb",
                "maxiter": self.settings.maxiter,
                "maxls": self.settings.maxls,
                "ftol": self.settings.ftol,
                "gtol": self.settings.gtol,
                "projected_gradient_tolerance": self.settings.projected_gradient_tolerance,
                "start_roles": TARGET_CONTEXT_START_ROLES,
                "start_values": self.start_values,
                "restart_selection": "minimum_objective_then_lexicographic_parameters",
            },
        }


@dataclass(frozen=True)
class PairedCompiledKernelArtifacts:
    """The authoritative uniform baseline paired with one target-context kernel."""

    baseline: CompiledKernelArtifact
    target_context: TargetContextCompiledKernelArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, CompiledKernelArtifact):
            raise TypeError("baseline must be CompiledKernelArtifact")
        if not isinstance(self.target_context, TargetContextCompiledKernelArtifact):
            raise TypeError("target_context must be TargetContextCompiledKernelArtifact")
        if self.baseline.target_hash != self.target_context.target_hash:
            raise ValueError("paired artifacts must share target_hash")


def _require_paired_settings(
    profile: PooledTargetContextProfile,
    baseline_settings: CompilerSettings,
    target_settings: CompilerSettings,
) -> None:
    if not isinstance(profile, PooledTargetContextProfile):
        raise TypeError("profile must be PooledTargetContextProfile")
    if not isinstance(baseline_settings, CompilerSettings):
        raise TypeError("baseline_settings must be CompilerSettings")
    if not isinstance(target_settings, CompilerSettings):
        raise TypeError("target settings must be CompilerSettings")
    if baseline_settings.context_weights != _UNIFORM_CONTEXT_WEIGHTS:
        raise ValueError("baseline settings must use exactly uniform context weights")
    if target_settings.context_weights != profile.context_weights:
        raise ValueError("target settings context_weights must exactly match the profile")
    for name in (
        "parameter_cap",
        "maxiter",
        "maxls",
        "ftol",
        "gtol",
        "projected_gradient_tolerance",
        "initializations",
    ):
        if getattr(baseline_settings, name) != getattr(target_settings, name):
            raise ValueError(f"baseline and target settings must share {name}")


def _run_checked_attempt(
    start_index: int,
    start_role: TargetContextStartRole,
    start: tuple[float, ...],
    target: NDArray[np.float64],
    context_weights: ContextWeights,
    settings: CompilerSettings,
) -> TargetContextOptimizationAttempt:
    bounds = [(-settings.parameter_cap, settings.parameter_cap)] * _N_PARAMETERS
    weights = np.asarray(context_weights, dtype=np.float64)
    result = minimize(
        fun=lambda values: loss_and_gradient(values, target, weights)[0],
        x0=np.asarray(start, dtype=np.float64),
        jac=lambda values: loss_and_gradient(values, target, weights)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": settings.maxiter,
            "maxls": settings.maxls,
            "ftol": settings.ftol,
            "gtol": settings.gtol,
        },
    )
    try:
        endpoint = np.asarray(getattr(result, "x", ()), dtype=np.float64)
    except (TypeError, ValueError):
        endpoint = np.asarray((), dtype=np.float64)
    valid_endpoint = endpoint.shape == (_N_PARAMETERS,) and np.all(np.isfinite(endpoint))
    if not valid_endpoint:
        raise ValueError("target-context optimizer returned malformed optimizer endpoint")
    objective, gradient = loss_and_gradient(endpoint, target, weights)
    raw_gradient_norm = float(np.max(np.abs(gradient)))
    projected_gradient_norm = float(
        np.max(np.abs(project_gradient(endpoint, gradient, settings.parameter_cap)))
    )
    parameters = tuple(float(value) for value in endpoint)
    try:
        scipy_success = bool(getattr(result, "success", False))
    except (TypeError, ValueError):
        scipy_success = False
    passed_checks = (
        scipy_success
        and all(
            math.isfinite(value)
            for value in (objective, raw_gradient_norm, projected_gradient_norm)
        )
        and projected_gradient_norm <= settings.projected_gradient_tolerance
        and all(abs(value) <= settings.parameter_cap for value in parameters)
    )
    try:
        iterations = int(getattr(result, "nit", 0))
    except (TypeError, ValueError):
        iterations = 0
    return TargetContextOptimizationAttempt(
        start_index=start_index,
        start_role=start_role,
        objective=objective,
        parameters=parameters,
        raw_gradient_norm=raw_gradient_norm,
        projected_gradient_norm=projected_gradient_norm,
        scipy_success=scipy_success,
        passed_checks=passed_checks,
        iterations=iterations,
        termination=str(getattr(result, "message", "")),
        cap_active_parameter_count=sum(
            abs(value) >= settings.parameter_cap for value in parameters
        ),
    )


def compile_target_context(
    target_hash: str,
    target: NDArray[np.generic],
    profile: PooledTargetContextProfile,
    baseline_artifact: CompiledKernelArtifact,
    settings: CompilerSettings,
) -> TargetContextCompiledKernelArtifact:
    """Compile exactly four checked target-context starts using a cached baseline."""

    target_hash = _nonempty_string(target_hash, name="target_hash")
    if not isinstance(baseline_artifact, CompiledKernelArtifact):
        raise TypeError("baseline_artifact must be CompiledKernelArtifact")
    _require_paired_settings(profile, baseline_artifact.settings, settings)
    if profile.target_hash != target_hash:
        raise ValueError("profile target_hash must match target_hash")
    if baseline_artifact.target_hash != target_hash:
        raise ValueError("baseline target_hash must match target_hash")
    if baseline_artifact.parameter_cap != baseline_artifact.settings.parameter_cap:
        raise ValueError("baseline parameter_cap must match baseline settings")

    weights = np.asarray(profile.context_weights, dtype=np.float64)
    # This public primitive performs the established strict target validation.
    loss_and_gradient(np.zeros(_N_PARAMETERS, dtype=np.float64), target, weights)
    checked_target = np.asarray(target, dtype=np.float64)
    start_values = (baseline_artifact.parameters.values, *settings.initializations)
    attempts = tuple(
        _run_checked_attempt(index, role, start, checked_target, profile.context_weights, settings)
        for index, (role, start) in enumerate(
            zip(TARGET_CONTEXT_START_ROLES, start_values, strict=True)
        )
    )
    passing = tuple(attempt for attempt in attempts if attempt.passed_checks)
    if not passing:
        raise RuntimeError("No target-context optimizer endpoint passed checked convergence")
    winner = min(passing, key=lambda attempt: (attempt.objective, attempt.parameters))
    return TargetContextCompiledKernelArtifact(
        target_hash=target_hash,
        profile_hash=profile.profile_hash,
        context_weights=profile.context_weights,
        baseline_artifact_hash=baseline_artifact.artifact_hash,
        topology_id=_TOPOLOGY_ID,
        logical_role_order=_LOGICAL_ROLE_ORDER,
        parameter_order=PARAMETER_ORDER,
        dtype=_DTYPE,
        parameters=KernelParameters(winner.parameters),
        beta=1.0,
        parameter_cap=settings.parameter_cap,
        settings=settings,
        start_values=start_values,
        attempts=attempts,
        selected_start_index=winner.start_index,
        selected_start_role=winner.start_role,
        objective=winner.objective,
        projected_gradient_norm=winner.projected_gradient_norm,
        cap_active_parameter_count=winner.cap_active_parameter_count,
    )


def compile_paired_target(
    target_hash: str,
    target: NDArray[np.generic],
    profile: PooledTargetContextProfile,
    baseline_settings: CompilerSettings,
    target_settings: CompilerSettings,
) -> PairedCompiledKernelArtifacts:
    """Compile the authoritative uniform baseline once and pair its warm start."""

    target_hash = _nonempty_string(target_hash, name="target_hash")
    _require_paired_settings(profile, baseline_settings, target_settings)
    if profile.target_hash != target_hash:
        raise ValueError("profile target_hash must match target_hash")
    baseline = compile_target(target_hash, target, baseline_settings)
    return PairedCompiledKernelArtifacts(
        baseline=baseline,
        target_context=compile_target_context(
            target_hash, target, profile, baseline, target_settings
        ),
    )


def evaluate_target_context_artifact(
    artifact: TargetContextCompiledKernelArtifact,
) -> NDArray[np.float64]:
    """Evaluate a frozen target-context artifact without optimizer state."""

    if not isinstance(artifact, TargetContextCompiledKernelArtifact):
        raise TypeError("artifact must be TargetContextCompiledKernelArtifact")
    return equilibrium_conditional(artifact.parameters, beta=artifact.beta)
