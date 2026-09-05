"""Checked compilation primitives for one-pass model-context artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

from thermo_lab.hashing import canonical_sha256
from thermo_lab.independent_compiler import CompilerSettings, loss_and_gradient, project_gradient
from thermo_lab.model_context import PooledModelContextProfile
from thermo_lab.target_context_compiler import TargetContextCompiledKernelArtifact
from thermo_lab.thermodynamic_kernel import KernelParameters, equilibrium_conditional

ModelContextStartRole = Literal[
    "target_context_warm_start", "fixed_zero", "fixed_positive", "fixed_antithetic_negative"
]
MODEL_CONTEXT_START_ROLES: tuple[ModelContextStartRole, ...] = (
    "target_context_warm_start",
    "fixed_zero",
    "fixed_positive",
    "fixed_antithetic_negative",
)


@dataclass(frozen=True)
class ModelContextOptimizationAttempt:
    start_index: int
    start_role: ModelContextStartRole
    objective: float
    parameters: tuple[float, ...]
    raw_gradient_norm: float
    projected_gradient_norm: float
    scipy_success: bool
    passed_checks: bool
    iterations: int
    termination: str
    cap_active_parameter_count: int


@dataclass(frozen=True)
class ModelContextCompiledKernelArtifact:
    target_hash: str
    profile_hash: str
    context_weights: tuple[float, float, float, float]
    target_context_artifact_hash: str
    parameters: KernelParameters
    beta: float
    parameter_cap: float
    settings: CompilerSettings
    start_values: tuple[tuple[float, ...], ...]
    attempts: tuple[ModelContextOptimizationAttempt, ...]
    selected_start_index: int
    selected_start_role: ModelContextStartRole
    objective: float
    projected_gradient_norm: float
    cap_active_parameter_count: int
    artifact_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.target_hash or not self.profile_hash or not self.target_context_artifact_hash:
            raise ValueError("artifact identity strings must be nonempty")
        if len(self.context_weights) != 4 or not math.isclose(
            math.fsum(self.context_weights), 1.0, abs_tol=1e-12, rel_tol=0.0
        ):
            raise ValueError("context_weights must sum to one")
        if (
            len(self.attempts) != 4
            or tuple(a.start_role for a in self.attempts) != MODEL_CONTEXT_START_ROLES
        ):
            raise ValueError("attempts must use checked start order")
        object.__setattr__(self, "artifact_hash", canonical_sha256(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "identity_version": "model_context_artifact.v1",
            "target_hash": self.target_hash,
            "profile_hash": self.profile_hash,
            "context_weights": self.context_weights,
            "target_context_artifact_hash": self.target_context_artifact_hash,
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
                "start_roles": MODEL_CONTEXT_START_ROLES,
                "start_values": self.start_values,
                "restart_selection": "minimum_objective_then_lexicographic_parameters",
            },
        }


def compile_model_context(
    target_hash: str,
    target: np.ndarray,
    profile: PooledModelContextProfile,
    target_context_artifact: TargetContextCompiledKernelArtifact,
    settings: CompilerSettings,
) -> ModelContextCompiledKernelArtifact:
    if profile.target_hash != target_hash or target_context_artifact.target_hash != target_hash:
        raise ValueError("target hash must match profile and upstream artifact")
    if settings.context_weights != profile.context_weights:
        raise ValueError("settings must use profile weights")
    starts = (target_context_artifact.parameters.values, *settings.initializations)
    attempts: list[ModelContextOptimizationAttempt] = []
    for index, (role, start) in enumerate(zip(MODEL_CONTEXT_START_ROLES, starts, strict=True)):
        weights = np.asarray(profile.context_weights)
        result = minimize(
            fun=lambda value, weights=weights: loss_and_gradient(value, target, weights)[0],
            x0=np.asarray(start),
            jac=lambda value, weights=weights: loss_and_gradient(value, target, weights)[1],
            method="L-BFGS-B",
            bounds=[(-settings.parameter_cap, settings.parameter_cap)] * 9,
            options={
                "maxiter": settings.maxiter,
                "maxls": settings.maxls,
                "ftol": settings.ftol,
                "gtol": settings.gtol,
            },
        )
        values = tuple(float(value) for value in np.asarray(result.x, dtype=np.float64))
        objective, gradient = loss_and_gradient(np.asarray(values), target, weights)
        projected = float(
            np.max(np.abs(project_gradient(np.asarray(values), gradient, settings.parameter_cap)))
        )
        raw = float(np.max(np.abs(gradient)))
        success = bool(result.success)
        attempts.append(
            ModelContextOptimizationAttempt(
                index,
                role,
                objective,
                values,
                raw,
                projected,
                success,
                success and projected <= settings.projected_gradient_tolerance,
                int(result.nit),
                str(result.message),
                sum(abs(value) >= settings.parameter_cap for value in values),
            )
        )
    passing = tuple(item for item in attempts if item.passed_checks)
    if not passing:
        raise RuntimeError("No model-context optimizer endpoint passed checked convergence")
    winner = min(passing, key=lambda item: (item.objective, item.parameters))
    return ModelContextCompiledKernelArtifact(
        target_hash,
        profile.profile_hash,
        profile.context_weights,
        target_context_artifact.artifact_hash,
        KernelParameters(winner.parameters),
        1.0,
        settings.parameter_cap,
        settings,
        starts,
        tuple(attempts),
        winner.start_index,
        winner.start_role,
        winner.objective,
        winner.projected_gradient_norm,
        winner.cap_active_parameter_count,
    )


def evaluate_model_context_artifact(artifact: ModelContextCompiledKernelArtifact) -> np.ndarray:
    return equilibrium_conditional(artifact.parameters, beta=artifact.beta)
