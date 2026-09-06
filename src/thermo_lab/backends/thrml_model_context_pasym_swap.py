"""Deterministic preparation for the checked model-context PAsymSwap study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import thrml

from thermo_lab.backends.thrml_target_context_pasym_swap import (
    ThrmlTargetContextPAsymSwapBackend,
)
from thermo_lab.config import (
    MODEL_CONTEXT_PASYM_SWAP_EXPERIMENT_ID,
    MODEL_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION,
    experiment_config_path,
    load_experiment_config,
    model_context_pasym_swap_non_seed_config_hash,
)
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.independent_compiler import CompilerSettings
from thermo_lab.model_context import (
    ModelContextArtifact,
    ModelContextTrace,
    PooledModelContextProfile,
    derive_model_context_trace,
    pool_model_context_profiles,
)
from thermo_lab.model_context_compiler import (
    MODEL_CONTEXT_START_ROLES,
    ModelContextCompiledKernelArtifact,
    compile_model_context,
)
from thermo_lab.pasym_swap import PAsymSwapTarget, build_paper_fixture
from thermo_lab.pasym_swap_context import derive_target_context_trace, pool_target_context_profiles
from thermo_lab.records import ExperimentSpec
from thermo_lab.schemas import (
    ModelContextCompilerRunConfig,
    PAsymSwapModelConfig,
    validate_model_context_pasym_swap_request,
)
from thermo_lab.target_context_compiler import TargetContextCompiledKernelArtifact
from thermo_lab.thermodynamic_kernel import equilibrium_conditional


@dataclass(frozen=True)
class ModelContextPreparedArtifacts:
    """Checked, in-memory inputs for the later sampling and reporting slice."""

    target_context_artifacts: tuple[TargetContextCompiledKernelArtifact, ...]
    model_trace: ModelContextTrace
    model_profiles: tuple[PooledModelContextProfile, ...]
    model_context_artifacts: tuple[ModelContextCompiledKernelArtifact, ...]


def _model_settings(
    model: PAsymSwapModelConfig,
    run: ModelContextCompilerRunConfig,
    profile: PooledModelContextProfile,
) -> CompilerSettings:
    return CompilerSettings(
        parameter_cap=model.parameter_cap,
        maxiter=run.maxiter,
        maxls=run.maxls,
        ftol=run.ftol,
        gtol=run.gtol,
        projected_gradient_tolerance=run.projected_gradient_tolerance,
        initializations=tuple(tuple(values) for values in run.initializations),
        context_weights=profile.context_weights,
    )


def _checked_model_artifact(
    artifact: ModelContextCompiledKernelArtifact,
    *,
    profile: PooledModelContextProfile,
    upstream: TargetContextCompiledKernelArtifact,
    settings: CompilerSettings,
) -> ModelContextCompiledKernelArtifact:
    """Validate the complete checked identity of one compiled model-context artifact."""

    if not isinstance(artifact, ModelContextCompiledKernelArtifact):
        raise TypeError(
            f"model artifact target_hash={profile.target_hash} must be a compiled artifact"
        )
    if (
        artifact.target_hash != profile.target_hash
        or artifact.profile_hash != profile.profile_hash
        or artifact.context_weights != profile.context_weights
        or artifact.target_context_artifact_hash != upstream.artifact_hash
        or artifact.settings != settings
    ):
        raise ValueError(
            f"model artifact target_hash={profile.target_hash} has mismatched checked inputs"
        )
    if artifact.artifact_hash != canonical_sha256(artifact.identity_payload()):
        raise ValueError(f"model artifact target_hash={profile.target_hash} has stale identity")
    if (
        tuple(attempt.start_index for attempt in artifact.attempts) != (0, 1, 2, 3)
        or tuple(attempt.start_role for attempt in artifact.attempts) != MODEL_CONTEXT_START_ROLES
        or artifact.selected_start_index not in range(4)
        or not artifact.attempts[artifact.selected_start_index].passed_checks
    ):
        raise ValueError(
            f"model artifact target_hash={profile.target_hash} has invalid optimizer records"
        )
    return artifact


class ThrmlModelContextPAsymSwapBackend:
    """Own the checked deterministic lineage before sampling and runner integration."""

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = repository_root
        self._target_backend = ThrmlTargetContextPAsymSwapBackend(repository_root)

    def checked_request(
        self, spec: ExperimentSpec
    ) -> tuple[PAsymSwapModelConfig, ModelContextCompilerRunConfig, str, ExperimentSpec]:
        """Validate one exact checked model-context request and its target-context ancestor."""

        if thrml.__version__ != "0.1.4":
            raise RuntimeError(f"Expected THRML 0.1.4, found {thrml.__version__}")
        expected = load_experiment_config(
            experiment_config_path("thrml-model-context-pasym-swap.toml")
        )
        if spec.experiment_id != MODEL_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
            raise ValueError(
                "Unexpected experiment request for model-context PAsymSwap THRML backend"
            )
        model_json = to_json_value(spec.model_parameters)
        run_json = to_json_value(spec.run_parameters)
        model = PAsymSwapModelConfig.model_validate(model_json)
        run = ModelContextCompilerRunConfig.model_validate(run_json)
        validate_model_context_pasym_swap_request(model, run, spec.seed)
        if (
            spec.sample_definition != MODEL_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION
            or model_json != to_json_value(expected.model_parameters)
            or run_json != to_json_value(expected.run_parameters)
        ):
            raise ValueError(
                "Model-context backend accepts only the exact checked experiment request"
            )
        if canonical_sha256(model.model_dump(mode="json")) != spec.model_hash:
            raise ValueError(
                "Validated model-context model differs from the canonically hashed request"
            )
        if canonical_sha256(run.model_dump(mode="json")) != canonical_sha256(spec.run_parameters):
            raise ValueError(
                "Validated model-context run differs from the canonically hashed request"
            )
        request_hash = model_context_pasym_swap_non_seed_config_hash(model, run)
        if request_hash != expected.non_seed_config_hash:
            raise ValueError(
                "Checked model-context request hash differs from authoritative configuration"
            )

        upstream = load_experiment_config(
            experiment_config_path("thrml-target-context-pasym-swap.toml")
        ).to_spec(seed=spec.seed)
        if to_json_value(upstream.model_parameters) != model_json:
            raise ValueError(
                "Model-context and upstream target-context model JSON must match exactly"
            )
        return model, run, request_hash, upstream

    def prepare(self, spec: ExperimentSpec) -> ModelContextPreparedArtifacts:
        """Rebuild exact upstream artifacts and compile the one-pass model-context study.

        This deliberately stops before THRML sampling, result validation, and runner protocol
        integration, which belong to the next slice.
        """

        model, run, _, upstream_spec = self.checked_request(spec)
        (
            upstream_model,
            upstream_run,
            upstream_request_hash,
            baseline_run,
            baseline_request_hash,
        ) = self._target_backend._checked_request(upstream_spec)
        if upstream_model != model:
            raise ValueError(
                "checked upstream target-context model differs from model-context model"
            )

        fixture = build_paper_fixture()
        target_trace = derive_target_context_trace(
            fixture,
            initial_state=upstream_run.initial_state,
            initial_particle_site=upstream_run.initial_particle_site,
            initial_occupancy=upstream_run.initial_occupancy,
            context_source=upstream_run.context_source,
            zero_support_policy=upstream_run.zero_support_policy,
        )
        target_profiles = pool_target_context_profiles(
            target_trace, context_reduction=upstream_run.context_reduction
        )
        targets: dict[str, PAsymSwapTarget] = {
            target.target_hash: target for target in fixture.targets
        }
        baselines, _ = self._target_backend._baselines(
            profiles=target_profiles,
            targets=targets,
            model=model,
            run=baseline_run,
            request_hash=baseline_request_hash,
        )
        target_artifacts, _ = self._target_backend._targets(
            profiles=target_profiles,
            targets=targets,
            baselines=baselines,
            model=model,
            run=upstream_run,
            baseline_request_hash=baseline_request_hash,
            target_request_hash=upstream_request_hash,
            trace_hash=target_trace.trace_hash,
        )
        upstream_by_target = {artifact.target_hash: artifact for artifact in target_artifacts}
        model_trace = derive_model_context_trace(
            fixture,
            {
                target_hash: ModelContextArtifact(
                    artifact_hash=artifact.artifact_hash,
                    conditional=tuple(
                        tuple(float(value) for value in row)
                        for row in equilibrium_conditional(artifact.parameters, beta=artifact.beta)
                    ),
                )
                for target_hash, artifact in upstream_by_target.items()
            },
            initial_occupancy=run.initial_occupancy,
        )
        model_profiles = pool_model_context_profiles(model_trace)
        model_artifacts = tuple(
            _checked_model_artifact(
                compile_model_context(
                    profile.target_hash,
                    np.asarray(targets[profile.target_hash].conditional, dtype=np.float64),
                    profile,
                    upstream_by_target[profile.target_hash],
                    _model_settings(model, run, profile),
                ),
                profile=profile,
                upstream=upstream_by_target[profile.target_hash],
                settings=_model_settings(model, run, profile),
            )
            for profile in model_profiles
        )
        return ModelContextPreparedArtifacts(
            target_context_artifacts=target_artifacts,
            model_trace=model_trace,
            model_profiles=model_profiles,
            model_context_artifacts=model_artifacts,
        )
