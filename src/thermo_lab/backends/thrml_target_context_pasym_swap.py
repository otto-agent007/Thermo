"""THRML execution for the checked exact target-context PAsymSwap compiler."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import thrml

from thermo_lab.backends.base import ExecutionResult
from thermo_lab.backends.thrml_pasym_swap import (
    CHAIN_COUNT as _CHAIN_COUNT,
)
from thermo_lab.backends.thrml_pasym_swap import (
    SAMPLER_CACHE_KEY as _SAMPLER_CACHE_KEY,
)
from thermo_lab.backends.thrml_pasym_swap import (
    compiled_sampler,
    fold_digest,
    output_word_counts,
    parameters_for_thrml,
    shared_sampler,
    synchronize_tree,
    uniform_free_state,
)
from thermo_lab.config import (
    TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID,
    TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION,
    experiment_config_path,
    independent_pasym_swap_non_seed_config_hash,
    load_experiment_config,
    target_context_pasym_swap_non_seed_config_hash,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.independent_compiler import CompiledKernelArtifact, CompilerSettings, compile_target
from thermo_lab.pasym_swap import PAPER_SOURCE, WORD_ORDER, PAsymSwapTarget, build_paper_fixture
from thermo_lab.pasym_swap_context import (
    PooledTargetContextProfile,
    derive_target_context_trace,
    pool_target_context_profiles,
)
from thermo_lab.pasym_swap_results import (
    KernelOptimizationAttemptResult,
    KernelOptimizationResult,
)
from thermo_lab.provenance import collect_runtime_provenance
from thermo_lab.records import (
    RUN_TIMING_SOURCE,
    ExperimentSpec,
    MetricObservation,
    RunRecord,
    RunTiming,
    build_run_record,
)
from thermo_lab.schemas import (
    IndependentCompilerRunConfig,
    PAsymSwapModelConfig,
    TargetContextCompilerRunConfig,
    validate_target_context_pasym_swap_request,
)
from thermo_lab.target_context_compiler import (
    TARGET_CONTEXT_START_ROLES,
    TargetContextCompiledKernelArtifact,
    compile_target_context,
)
from thermo_lab.target_context_pasym_swap_results import (
    BaselineKernelResult,
    OptimizerPhaseResult,
    PairedKernelResult,
    PairedProfileMetrics,
    PooledContextProfileResult,
    TargetContextKernelResult,
    TargetContextOptimizationAttemptResult,
    TargetContextOptimizationResult,
    build_target_context_pasym_swap_summary,
    derive_exact_kernel_evaluation,
    derive_paired_profile_metrics,
    derive_sampled_k30_evaluation,
    validate_target_context_pasym_swap_observations,
)

_SUMMARY_METHOD = "bounded target-context PAsymSwap summary"
_EXACT_METHOD = "recomputed from exact frozen-model conditionals"
_SAMPLE_METHOD = "independently seeded 4096-chain THRML cross-check"
_ACCEPTANCE_METHOD = "all target-context acceptance gates recomputed"
_BASELINE_OPTIMIZER_METHOD = "wall-clock SciPy optimization across 37 paired uniform baselines"
_TARGET_OPTIMIZER_METHOD = "wall-clock SciPy optimization across 37 target-context profiles"
_TIMING_PREFIX = (
    "cached shared jax.jit(jax.vmap(single_chain)) executable; one untimed synchronized "
    "warm launch, then aggregate synchronized steady-state execution"
)
TargetCacheKey = tuple[str, str, str, str, str]


def target_context_artifact_keys(
    root_key: jax.Array,
    target_hash: str,
    profile_hash: str,
    input_index: int,
) -> tuple[jax.Array, jax.Array]:
    """Derive stable target/profile/input initialization and sampling keys."""

    if type(input_index) is not int or input_index not in range(4):
        raise ValueError("input_index must be a canonical two-bit input index")
    key = fold_digest(root_key, target_hash, name="target_hash")
    key = fold_digest(key, profile_hash, name="profile_hash")
    key = jax.random.fold_in(key, input_index)
    init_key, sample_key = jax.random.split(key)
    return init_key, sample_key


def _shared_sampler() -> Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]:
    """Keep sampler construction patchable at the dedicated-backend boundary."""

    return shared_sampler()


def _baseline_settings(
    model: PAsymSwapModelConfig, run: IndependentCompilerRunConfig
) -> CompilerSettings:
    return CompilerSettings(
        parameter_cap=model.parameter_cap,
        maxiter=run.maxiter,
        maxls=run.maxls,
        ftol=run.ftol,
        gtol=run.gtol,
        projected_gradient_tolerance=run.projected_gradient_tolerance,
        initializations=tuple(tuple(values) for values in run.initializations),
        context_weights=tuple(run.context_weights),
    )


def _target_settings(
    model: PAsymSwapModelConfig,
    run: TargetContextCompilerRunConfig,
    profile: PooledTargetContextProfile,
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


def _checked_baseline_artifact(
    artifact: CompiledKernelArtifact,
    *,
    target_hash: str,
    settings: CompilerSettings,
) -> CompiledKernelArtifact:
    """Apply one structural/identity validator to fresh and cached baselines."""

    if not isinstance(artifact, CompiledKernelArtifact):
        raise TypeError(f"baseline cache target_hash={target_hash} must contain compiled artifacts")
    if artifact.target_hash != target_hash or artifact.settings != settings:
        raise ValueError(f"baseline cache target_hash={target_hash} has mismatched checked inputs")
    if artifact.artifact_hash != canonical_sha256(artifact.identity_payload()):
        raise ValueError(f"baseline cache target_hash={target_hash} has stale artifact identity")
    if (
        tuple(attempt.restart_index for attempt in artifact.attempts) != (0, 1, 2)
        or artifact.selected_restart not in range(3)
        or not artifact.attempts[artifact.selected_restart].passed_checks
    ):
        raise ValueError(f"baseline cache target_hash={target_hash} has invalid optimizer records")
    return artifact


def _checked_target_artifact(
    artifact: TargetContextCompiledKernelArtifact,
    *,
    profile: PooledTargetContextProfile,
    baseline: CompiledKernelArtifact,
    settings: CompilerSettings,
) -> TargetContextCompiledKernelArtifact:
    """Apply one structural/identity validator to fresh and cached target artifacts."""

    if not isinstance(artifact, TargetContextCompiledKernelArtifact):
        raise TypeError(
            f"target cache target_hash={profile.target_hash} must contain compiled artifacts"
        )
    if (
        artifact.target_hash != profile.target_hash
        or artifact.profile_hash != profile.profile_hash
        or artifact.context_weights != profile.context_weights
        or artifact.baseline_artifact_hash != baseline.artifact_hash
        or artifact.settings != settings
    ):
        raise ValueError(
            f"target cache target_hash={profile.target_hash} profile_hash={profile.profile_hash} "
            "has mismatched checked inputs"
        )
    if artifact.artifact_hash != canonical_sha256(artifact.identity_payload()):
        raise ValueError(
            f"target cache target_hash={profile.target_hash} profile_hash={profile.profile_hash} "
            "has stale artifact identity"
        )
    if (
        tuple(attempt.start_index for attempt in artifact.attempts) != (0, 1, 2, 3)
        or tuple(attempt.start_role for attempt in artifact.attempts) != TARGET_CONTEXT_START_ROLES
        or artifact.selected_start_index not in range(4)
        or not artifact.attempts[artifact.selected_start_index].passed_checks
    ):
        raise ValueError(
            f"target cache target_hash={profile.target_hash} profile_hash={profile.profile_hash} "
            "has invalid optimizer records"
        )
    return artifact


def _profile_result(profile: PooledTargetContextProfile) -> PooledContextProfileResult:
    return PooledContextProfileResult(
        trace_hash=profile.trace_hash,
        target_hash=profile.target_hash,
        context_reduction="equal_occurrence_mean_by_target_hash",
        zero_support_policy="exact_unsmoothed",
        occurrence_indices=profile.occurrence_indices,
        multiplicity=profile.multiplicity,
        context_weights=profile.context_weights,
        support_mask=profile.support_mask,
        profile_hash=profile.profile_hash,
        evidence_class=EvidenceClass.EXACT_REFERENCE,
    )


def _baseline_optimization(artifact: CompiledKernelArtifact) -> KernelOptimizationResult:
    return KernelOptimizationResult(
        artifact_hash=artifact.artifact_hash,
        parameters=artifact.parameters.values,
        selected_restart=artifact.selected_restart,
        successful_restart_count=sum(attempt.passed_checks for attempt in artifact.attempts),
        objective=artifact.objective,
        projected_gradient_norm=artifact.projected_gradient_norm,
        cap_active_parameter_count=artifact.cap_active_parameter_count,
        attempts=tuple(
            KernelOptimizationAttemptResult(
                restart_index=attempt.restart_index,
                parameters=attempt.parameters,
                objective=attempt.objective,
                raw_gradient_norm=attempt.raw_gradient_norm,
                projected_gradient_norm=attempt.projected_gradient_norm,
                scipy_success=attempt.scipy_success,
                passed_checks=attempt.passed_checks,
                iterations=attempt.iterations,
                termination=attempt.termination,
                cap_active_parameter_count=attempt.cap_active_parameter_count,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            )
            for attempt in artifact.attempts
        ),
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )


def _target_optimization(
    artifact: TargetContextCompiledKernelArtifact,
) -> TargetContextOptimizationResult:
    return TargetContextOptimizationResult(
        artifact_hash=artifact.artifact_hash,
        start_values=artifact.start_values,  # type: ignore[arg-type]
        parameters=artifact.parameters.values,
        selected_start_index=artifact.selected_start_index,
        selected_start_role=artifact.selected_start_role,
        successful_attempt_count=sum(attempt.passed_checks for attempt in artifact.attempts),
        objective=artifact.objective,
        projected_gradient_norm=artifact.projected_gradient_norm,
        cap_active_parameter_count=artifact.cap_active_parameter_count,
        attempts=tuple(
            TargetContextOptimizationAttemptResult(
                start_index=attempt.start_index,
                start_role=attempt.start_role,
                parameters=attempt.parameters,
                objective=attempt.objective,
                raw_gradient_norm=attempt.raw_gradient_norm,
                projected_gradient_norm=attempt.projected_gradient_norm,
                scipy_success=attempt.scipy_success,
                passed_checks=attempt.passed_checks,
                iterations=attempt.iterations,
                termination=attempt.termination,
                cap_active_parameter_count=attempt.cap_active_parameter_count,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            )
            for attempt in artifact.attempts
        ),  # type: ignore[arg-type]
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )


def _pair_result(
    *,
    target: PAsymSwapTarget,
    profile: PooledTargetContextProfile,
    baseline_artifact: CompiledKernelArtifact,
    target_artifact: TargetContextCompiledKernelArtifact,
    counts: tuple[tuple[int, int, int, int], ...],
    baseline_request_hash: str,
    target_request_hash: str,
    model: PAsymSwapModelConfig,
) -> PairedKernelResult:
    baseline_exact = derive_exact_kernel_evaluation(
        baseline_artifact.parameters.values, target.conditional, beta=model.beta
    )
    target_exact = derive_exact_kernel_evaluation(
        target_artifact.parameters.values, target.conditional, beta=model.beta
    )
    baseline = BaselineKernelResult(
        target_hash=profile.target_hash,
        baseline_compiler_request_hash=baseline_request_hash,
        optimization=_baseline_optimization(baseline_artifact),
        exact=baseline_exact,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )
    target_context = TargetContextKernelResult(
        target_hash=profile.target_hash,
        profile_hash=profile.profile_hash,
        target_compiler_request_hash=target_request_hash,
        baseline_artifact_hash=baseline_artifact.artifact_hash,
        optimization=_target_optimization(target_artifact),
        exact=target_exact,
        sampled_k30=derive_sampled_k30_evaluation(
            counts,
            target_exact.finite_horizon_conditionals[30],
            chain_count=_CHAIN_COUNT,
        ),
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )
    placeholder = PairedProfileMetrics(
        multiplicity=profile.multiplicity,
        context_weights=profile.context_weights,
        support_mask=profile.support_mask,
        baseline_target_weighted_equilibrium_kl=0.0,
        target_context_target_weighted_equilibrium_kl=0.0,
        target_weighted_equilibrium_kl_improvement=0.0,
        baseline_target_weighted_equilibrium_tv=0.0,
        target_context_target_weighted_equilibrium_tv=0.0,
        baseline_global_kl_contribution=0.0,
        target_context_global_kl_contribution=0.0,
        evidence_class=EvidenceClass.EXACT_REFERENCE,
    )
    pair = PairedKernelResult(
        target_hash=profile.target_hash,
        profile_hash=profile.profile_hash,
        baseline=baseline,
        target_context=target_context,
        metrics=placeholder,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )
    return pair.model_copy(
        update={"metrics": derive_paired_profile_metrics(pair, _profile_result(profile))}
    )


class ThrmlTargetContextPAsymSwapBackend:
    """Compile paired kernels deterministically and sample only target artifacts."""

    backend_id = BackendId.THRML_LOCAL
    evidence_class = EvidenceClass.SOFTWARE_SIMULATION

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = repository_root
        self._baseline_cache: dict[str, tuple[CompiledKernelArtifact, ...]] = {}
        self._target_cache: dict[TargetCacheKey, TargetContextCompiledKernelArtifact] = {}
        self._sampler_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}

    def run(self, spec: ExperimentSpec) -> RunRecord:
        return self.execute(spec).record

    def _checked_request(
        self, spec: ExperimentSpec
    ) -> tuple[
        PAsymSwapModelConfig,
        TargetContextCompilerRunConfig,
        str,
        IndependentCompilerRunConfig,
        str,
    ]:
        if thrml.__version__ != "0.1.4":
            raise RuntimeError(f"Expected THRML 0.1.4, found {thrml.__version__}")
        expected = load_experiment_config(
            experiment_config_path("thrml-target-context-pasym-swap.toml")
        )
        if spec.experiment_id != TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
            raise ValueError(
                "Unexpected experiment request for target-context PAsymSwap THRML backend"
            )
        model_json = to_json_value(spec.model_parameters)
        run_json = to_json_value(spec.run_parameters)
        model = PAsymSwapModelConfig.model_validate(model_json)
        run = TargetContextCompilerRunConfig.model_validate(run_json)
        validate_target_context_pasym_swap_request(model, run, spec.seed)
        if (
            spec.sample_definition != TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION
            or model_json != to_json_value(expected.model_parameters)
            or run_json != to_json_value(expected.run_parameters)
        ):
            raise ValueError(
                "Target-context PAsymSwap backend accepts only the exact checked experiment request"
            )
        if canonical_sha256(model.model_dump(mode="json")) != spec.model_hash:
            raise ValueError(
                "Validated target-context PAsymSwap model differs from the "
                "canonically hashed request"
            )
        if canonical_sha256(run.model_dump(mode="json")) != canonical_sha256(spec.run_parameters):
            raise ValueError(
                "Validated target-context PAsymSwap run differs from the canonically hashed request"
            )
        target_request_hash = target_context_pasym_swap_non_seed_config_hash(model, run)
        if target_request_hash != expected.non_seed_config_hash:
            raise ValueError(
                "Checked target-context request hash differs from authoritative configuration"
            )

        independent = load_experiment_config(
            experiment_config_path("thrml-independent-pasym-swap.toml")
        )
        independent_model_json = to_json_value(independent.model_parameters)
        if independent_model_json != model_json:
            raise ValueError(
                "Target-context and authoritative independent model JSON must match exactly"
            )
        baseline_run = IndependentCompilerRunConfig.model_validate(
            to_json_value(independent.run_parameters)
        )
        baseline_request_hash = independent_pasym_swap_non_seed_config_hash(model, baseline_run)
        if baseline_request_hash != independent.non_seed_config_hash:
            raise ValueError(
                "Checked baseline request hash differs from authoritative independent configuration"
            )
        return model, run, target_request_hash, baseline_run, baseline_request_hash

    def _baselines(
        self,
        *,
        profiles: tuple[PooledTargetContextProfile, ...],
        targets: dict[str, PAsymSwapTarget],
        model: PAsymSwapModelConfig,
        run: IndependentCompilerRunConfig,
        request_hash: str,
    ) -> tuple[tuple[CompiledKernelArtifact, ...], OptimizerPhaseResult]:
        settings = _baseline_settings(model, run)
        cached = self._baseline_cache.get(request_hash)
        if cached is None:
            started = time.perf_counter()
            artifacts = tuple(
                compile_target(
                    profile.target_hash,
                    np.asarray(targets[profile.target_hash].conditional, dtype=np.float64),
                    settings,
                )
                for profile in profiles
            )
            seconds = time.perf_counter() - started
            cache_reused = False
        else:
            artifacts = cached
            seconds = 0.0
            cache_reused = True
        if len(artifacts) != 37:
            raise ValueError(
                f"baseline cache request_hash={request_hash} observed={len(artifacts)} bound=37"
            )
        checked = tuple(
            _checked_baseline_artifact(
                artifact,
                target_hash=profile.target_hash,
                settings=settings,
            )
            for profile, artifact in zip(profiles, artifacts, strict=True)
        )
        if not cache_reused:
            self._baseline_cache[request_hash] = checked
        return checked, OptimizerPhaseResult(
            seconds=seconds,
            cache_reused=cache_reused,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        )

    def _targets(
        self,
        *,
        profiles: tuple[PooledTargetContextProfile, ...],
        targets: dict[str, PAsymSwapTarget],
        baselines: tuple[CompiledKernelArtifact, ...],
        model: PAsymSwapModelConfig,
        run: TargetContextCompilerRunConfig,
        baseline_request_hash: str,
        target_request_hash: str,
        trace_hash: str,
    ) -> tuple[tuple[TargetContextCompiledKernelArtifact, ...], OptimizerPhaseResult]:
        started = time.perf_counter()
        compiled_any = False
        artifacts: list[TargetContextCompiledKernelArtifact] = []
        for profile, baseline in zip(profiles, baselines, strict=True):
            key = (
                baseline_request_hash,
                target_request_hash,
                trace_hash,
                profile.profile_hash,
                profile.target_hash,
            )
            settings = _target_settings(model, run, profile)
            artifact = self._target_cache.get(key)
            artifact_reused = artifact is not None
            if artifact is None:
                artifact = compile_target_context(
                    profile.target_hash,
                    np.asarray(targets[profile.target_hash].conditional, dtype=np.float64),
                    profile,
                    baseline,
                    settings,
                )
                compiled_any = True
            checked = _checked_target_artifact(
                artifact,
                profile=profile,
                baseline=baseline,
                settings=settings,
            )
            if not artifact_reused:
                self._target_cache[key] = checked
            artifacts.append(checked)
        seconds = time.perf_counter() - started if compiled_any else 0.0
        return tuple(artifacts), OptimizerPhaseResult(
            seconds=seconds,
            cache_reused=not compiled_any,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        )

    def _executable(
        self, exemplar: TargetContextCompiledKernelArtifact
    ) -> tuple[Callable[..., jax.Array], float, bool]:
        cached = self._sampler_cache.get(_SAMPLER_CACHE_KEY)
        if cached is not None:
            return cached, 0.0, True
        biases, weights = parameters_for_thrml(exemplar.parameters)
        return compiled_sampler(
            self._sampler_cache,
            biases,
            weights,
            sampler_factory=_shared_sampler,
        )

    def _sample(
        self,
        *,
        spec: ExperimentSpec,
        run: TargetContextCompilerRunConfig,
        profiles: tuple[PooledTargetContextProfile, ...],
        artifacts: tuple[TargetContextCompiledKernelArtifact, ...],
        executable: Callable[..., jax.Array],
    ) -> tuple[dict[str, tuple[tuple[int, int, int, int], ...]], float]:
        root_key = jax.random.key(spec.seed)
        launches: list[
            tuple[
                PooledTargetContextProfile,
                int,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ]
        ] = []
        for profile, artifact in zip(profiles, artifacts, strict=True):
            biases, weights = parameters_for_thrml(artifact.parameters)
            for input_index, bits in enumerate(WORD_ORDER):
                init_key, sampling_key = target_context_artifact_keys(
                    root_key, profile.target_hash, profile.profile_hash, input_index
                )
                hidden, outputs = uniform_free_state(
                    init_key, chain_count=run.chain_count_per_context
                )
                keys = jax.random.split(sampling_key, run.chain_count_per_context)
                clamp = jnp.broadcast_to(
                    jnp.asarray(bits, dtype=jnp.bool_),
                    (run.chain_count_per_context, 2),
                )
                launches.append(
                    (profile, input_index, biases, weights, keys, hidden, outputs, clamp)
                )
        if len(launches) != 148:
            raise RuntimeError(f"target-context launch count observed={len(launches)} bound=148")

        _, _, biases, weights, keys, hidden, outputs, clamp = launches[0]
        representative = executable(biases, weights, keys, hidden, outputs, clamp)
        representative.block_until_ready()
        started = time.perf_counter()
        measured: list[tuple[PooledTargetContextProfile, int, jax.Array]] = []
        for profile, input_index, biases, weights, keys, hidden, outputs, clamp in launches:
            measured.append(
                (
                    profile,
                    input_index,
                    executable(biases, weights, keys, hidden, outputs, clamp),
                )
            )
        synchronize_tree([observed for _, _, observed in measured])
        execution_seconds = time.perf_counter() - started

        empirical: dict[str, list[tuple[int, int, int, int]]] = {
            profile.target_hash: [(0, 0, 0, 0)] * 4 for profile in profiles
        }
        for profile, input_index, observed in measured:
            empirical[profile.target_hash][input_index] = output_word_counts(
                observed, chain_count=run.chain_count_per_context
            )
        return (
            {target_hash: tuple(rows) for target_hash, rows in empirical.items()},
            execution_seconds,
        )

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        model, run, target_request_hash, baseline_run, baseline_request_hash = (
            self._checked_request(spec)
        )
        fixture = build_paper_fixture()
        trace = derive_target_context_trace(
            fixture,
            initial_state=run.initial_state,
            initial_particle_site=run.initial_particle_site,
            initial_occupancy=run.initial_occupancy,
            context_source=run.context_source,
            zero_support_policy=run.zero_support_policy,
        )
        profiles = pool_target_context_profiles(trace, context_reduction=run.context_reduction)
        targets = {target.target_hash: target for target in fixture.targets}
        baselines, baseline_phase = self._baselines(
            profiles=profiles,
            targets=targets,
            model=model,
            run=baseline_run,
            request_hash=baseline_request_hash,
        )
        target_artifacts, target_phase = self._targets(
            profiles=profiles,
            targets=targets,
            baselines=baselines,
            model=model,
            run=run,
            baseline_request_hash=baseline_request_hash,
            target_request_hash=target_request_hash,
            trace_hash=trace.trace_hash,
        )
        executable, compile_seconds, reused_executable = self._executable(target_artifacts[0])
        empirical, execution_seconds = self._sample(
            spec=spec,
            run=run,
            profiles=profiles,
            artifacts=target_artifacts,
            executable=executable,
        )
        pairs = tuple(
            _pair_result(
                target=targets[profile.target_hash],
                profile=profile,
                baseline_artifact=baseline,
                target_artifact=target_artifact,
                counts=empirical[profile.target_hash],
                baseline_request_hash=baseline_request_hash,
                target_request_hash=target_request_hash,
                model=model,
            )
            for profile, baseline, target_artifact in zip(
                profiles, baselines, target_artifacts, strict=True
            )
        )
        summary = build_target_context_pasym_swap_summary(
            pairs=pairs,
            model=model,
            run=run,
            seed=spec.seed,
            baseline_optimizer_phase=baseline_phase,
            target_context_optimizer_phase=target_phase,
        )
        schedule = summary.schedule_metrics
        metrics = {
            "target_context_pasym_swap": MetricObservation(
                value=summary,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=_SUMMARY_METHOD,
                source=PAPER_SOURCE,
            ),
            "baseline_occurrence_weighted_equilibrium_kl": MetricObservation(
                value=schedule.baseline_occurrence_weighted_equilibrium_kl,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                unit="nats",
                method=_EXACT_METHOD,
                source=PAPER_SOURCE,
            ),
            "target_context_occurrence_weighted_equilibrium_kl": MetricObservation(
                value=schedule.target_context_occurrence_weighted_equilibrium_kl,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                unit="nats",
                method=_EXACT_METHOD,
                source=PAPER_SOURCE,
            ),
            "occurrence_weighted_equilibrium_kl_improvement": MetricObservation(
                value=schedule.occurrence_weighted_equilibrium_kl_improvement,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                unit="nats",
                method=_EXACT_METHOD,
                source=PAPER_SOURCE,
            ),
            "baseline_occurrence_weighted_equilibrium_tv": MetricObservation(
                value=schedule.baseline_occurrence_weighted_equilibrium_tv,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method=_EXACT_METHOD,
                source=PAPER_SOURCE,
            ),
            "target_context_occurrence_weighted_equilibrium_tv": MetricObservation(
                value=schedule.target_context_occurrence_weighted_equilibrium_tv,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method=_EXACT_METHOD,
                source=PAPER_SOURCE,
            ),
            "maximum_paired_k30_equilibrium_residual": MetricObservation(
                value=schedule.maximum_paired_k30_equilibrium_residual,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method=_EXACT_METHOD,
                source=PAPER_SOURCE,
            ),
            "maximum_empirical_k30_residual": MetricObservation(
                value=summary.sampled_fidelity.maximum_empirical_k30_residual,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=_SAMPLE_METHOD,
                source=PAPER_SOURCE,
            ),
            "acceptance_passed": MetricObservation(
                value=summary.seed_acceptance.passed,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=_ACCEPTANCE_METHOD,
                source=PAPER_SOURCE,
            ),
            "baseline_optimizer_seconds": MetricObservation(
                value=baseline_phase.seconds,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                unit="seconds",
                method=_BASELINE_OPTIMIZER_METHOD,
                source=RUN_TIMING_SOURCE,
            ),
            "target_context_optimizer_seconds": MetricObservation(
                value=target_phase.seconds,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                unit="seconds",
                method=_TARGET_OPTIMIZER_METHOD,
                source=RUN_TIMING_SOURCE,
            ),
        }
        validated = validate_target_context_pasym_swap_observations(metrics, model, run, spec.seed)
        if not validated.seed_acceptance.passed:
            messages = (
                *validated.deterministic_acceptance.check_messages,
                *validated.sampled_fidelity.check_messages,
                *validated.seed_acceptance.check_messages,
            )
            detail = "; ".join(messages)[:1536]
            raise RuntimeError(f"target-context acceptance failed seed={spec.seed}: {detail}")

        timing_method = _TIMING_PREFIX
        if reused_executable:
            timing_method += "; JAX executable reused from in-process shape cache"
        else:
            timing_method += "; JAX lower().compile() measured once for shared shapes"
        record = build_run_record(
            backend_id=self.backend_id,
            evidence_class=self.evidence_class,
            spec=spec,
            provenance=collect_runtime_provenance(self.repository_root),
            timing=RunTiming(
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                unit="seconds",
                source=RUN_TIMING_SOURCE,
                compile_seconds=compile_seconds,
                execution_seconds=execution_seconds,
                synchronized=True,
                timing_method=timing_method,
            ),
            metrics=metrics,
        )
        return ExecutionResult.build(record)
