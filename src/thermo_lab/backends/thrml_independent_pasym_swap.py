"""THRML execution for independently compiled two-bit PAsymSwap kernels."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import thrml
from thrml import sample_states

from thermo_lab.backends.base import ExecutionResult
from thermo_lab.backends.thrml_pasym_swap import (
    CHAIN_COUNT as _CHAIN_COUNT,
)
from thermo_lab.backends.thrml_pasym_swap import (
    SAMPLER_CACHE_KEY as _SAMPLER_CACHE_KEY,
)
from thermo_lab.backends.thrml_pasym_swap import (
    compiled_sampler,
    digest_words,
    output_word_counts,
    parameters_for_thrml,
    shared_sampler,
    synchronize_tree,
    uniform_free_state,
)
from thermo_lab.config import (
    INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID,
    INDEPENDENT_PASYM_SWAP_SAMPLE_DEFINITION,
    experiment_config_path,
    independent_pasym_swap_non_seed_config_hash,
    load_experiment_config,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.independent_compiler import CompiledKernelArtifact, CompilerSettings, compile_target
from thermo_lab.pasym_swap import PAPER_SOURCE, WORD_ORDER, PAsymSwapTarget, build_paper_fixture
from thermo_lab.pasym_swap_results import (
    CompiledKernelResult,
    KernelConditionalResult,
    KernelOptimizationAttemptResult,
    KernelOptimizationResult,
    summarize_artifacts,
    validate_independent_pasym_swap_observations,
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
    validate_independent_pasym_swap_request,
)
from thermo_lab.thermodynamic_kernel import equilibrium_conditional, finite_horizon_conditional


def _digest_words(target_hash: str) -> tuple[int, int, int, int, int, int, int, int]:
    """Parse either a canonical ``sha256:`` digest or raw 64-hex test digest."""

    return digest_words(target_hash, name="target_hash")


def artifact_keys(
    root_key: jax.Array, target_hash: str, input_index: int
) -> tuple[jax.Array, jax.Array]:
    """Derive iteration-order-independent initialization and sampling keys."""

    if type(input_index) is not int or input_index not in range(4):
        raise ValueError("input_index must be a canonical two-bit input index")
    key = root_key
    for word in _digest_words(target_hash):
        key = jax.random.fold_in(key, word)
    key = jax.random.fold_in(key, input_index)
    init_key, sample_key = jax.random.split(key)
    return init_key, sample_key


_artifact_keys = artifact_keys


def _settings(model: PAsymSwapModelConfig, run: IndependentCompilerRunConfig) -> CompilerSettings:
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


def _table(values: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(float(item) for item in row) for row in values)


def _parameters_for_thrml(artifact: CompiledKernelArtifact) -> tuple[jax.Array, jax.Array]:
    """Preserve the independent backend's artifact-typed conversion wrapper."""

    return parameters_for_thrml(artifact.parameters)


def _shared_sampler() -> Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]:
    """Preserve the independent backend's patchable sampler-construction seam."""

    return shared_sampler(sample_states_function=sample_states)


@dataclass(frozen=True)
class _DeterministicFixture:
    targets: tuple[PAsymSwapTarget, ...]
    artifacts: tuple[CompiledKernelArtifact, ...]
    optimizer_seconds: float
    cache_note: str


class ThrmlIndependentPAsymSwapBackend:
    """Execute the checked compiler then sample each frozen kernel with THRML."""

    backend_id = BackendId.THRML_LOCAL
    evidence_class = EvidenceClass.SOFTWARE_SIMULATION

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = repository_root
        self._fixture_cache: dict[str, _DeterministicFixture] = {}
        self._sampler_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}

    def run(self, spec: ExperimentSpec) -> RunRecord:
        return self.execute(spec).record

    def _checked_request(
        self, spec: ExperimentSpec
    ) -> tuple[PAsymSwapModelConfig, IndependentCompilerRunConfig, str]:
        if thrml.__version__ != "0.1.4":
            raise RuntimeError(f"Expected THRML 0.1.4, found {thrml.__version__}")
        expected = load_experiment_config(
            experiment_config_path("thrml-independent-pasym-swap.toml")
        )
        if spec.experiment_id != INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID:
            raise ValueError("Unexpected experiment request for PAsymSwap THRML backend")
        model = PAsymSwapModelConfig.model_validate(to_json_value(spec.model_parameters))
        run = IndependentCompilerRunConfig.model_validate(to_json_value(spec.run_parameters))
        validate_independent_pasym_swap_request(model, run, spec.seed)
        expected_model = to_json_value(expected.model_parameters)
        expected_run = to_json_value(expected.run_parameters)
        if (
            spec.sample_definition != INDEPENDENT_PASYM_SWAP_SAMPLE_DEFINITION
            or to_json_value(spec.model_parameters) != expected_model
            or to_json_value(spec.run_parameters) != expected_run
        ):
            raise ValueError("PAsymSwap backend accepts only the exact checked experiment request")
        if canonical_sha256(model.model_dump(mode="json")) != spec.model_hash:
            raise ValueError(
                "Validated PAsymSwap model differs from the canonically hashed request"
            )
        if canonical_sha256(run.model_dump(mode="json")) != canonical_sha256(spec.run_parameters):
            raise ValueError("Validated PAsymSwap run differs from the canonically hashed request")
        request_hash = independent_pasym_swap_non_seed_config_hash(model, run)
        if request_hash != expected.non_seed_config_hash:
            raise ValueError(
                "Checked PAsymSwap request hash differs from authoritative configuration"
            )
        return model, run, request_hash

    def _fixture(
        self,
        model: PAsymSwapModelConfig,
        run: IndependentCompilerRunConfig,
        request_hash: str,
    ) -> _DeterministicFixture:
        existing = self._fixture_cache.get(request_hash)
        if existing is not None:
            return _DeterministicFixture(
                targets=existing.targets,
                artifacts=existing.artifacts,
                optimizer_seconds=0.0,
                cache_note=(
                    "deterministic compiled fixture reused from in-process non-seed request cache"
                ),
            )
        fixture = build_paper_fixture()
        started = time.perf_counter()
        compiled = {
            target.target_hash: compile_target(
                target.target_hash,
                np.asarray(target.conditional, dtype=np.float64),
                _settings(model, run),
            )
            for target in sorted(fixture.targets, key=lambda item: item.target_hash)
        }
        elapsed = time.perf_counter() - started
        result = _DeterministicFixture(
            targets=tuple(fixture.targets),
            artifacts=tuple(compiled[target.target_hash] for target in fixture.targets),
            optimizer_seconds=elapsed,
            cache_note=(
                "deterministic compiled fixture populated in in-process non-seed request cache"
            ),
        )
        self._fixture_cache[request_hash] = result
        return result

    def _executable(
        self, exemplar: CompiledKernelArtifact
    ) -> tuple[Callable[..., jax.Array], float, bool]:
        cached = self._sampler_cache.get(_SAMPLER_CACHE_KEY)
        if cached is not None:
            return cached, 0.0, True
        biases, weights = _parameters_for_thrml(exemplar)
        return compiled_sampler(
            self._sampler_cache,
            biases,
            weights,
            sampler_factory=_shared_sampler,
        )

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        model, run, request_hash = self._checked_request(spec)
        deterministic = self._fixture(model, run, request_hash)
        executable, compile_seconds, reused_executable = self._executable(
            deterministic.artifacts[0]
        )

        root_key = jax.random.key(spec.seed)
        launches: list[
            tuple[
                CompiledKernelArtifact,
                PAsymSwapTarget,
                int,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ]
        ] = []
        for target, artifact in sorted(
            zip(deterministic.targets, deterministic.artifacts, strict=True),
            key=lambda item: item[0].target_hash,
        ):
            biases, weights = _parameters_for_thrml(artifact)
            for input_index, bits in enumerate(WORD_ORDER):
                init_key, sampling_key = artifact_keys(root_key, target.target_hash, input_index)
                hidden, outputs = uniform_free_state(
                    init_key, chain_count=run.chain_count_per_context
                )
                keys = jax.random.split(sampling_key, run.chain_count_per_context)
                clamp = jnp.broadcast_to(jnp.asarray(bits, dtype=jnp.bool_), (_CHAIN_COUNT, 2))
                launches.append(
                    (
                        artifact,
                        target,
                        input_index,
                        biases,
                        weights,
                        keys,
                        hidden,
                        outputs,
                        clamp,
                    )
                )

        # One representative untimed/synchronized launch warms the shared executable.
        _, _, _, biases, weights, keys, hidden, outputs, clamp = launches[0]
        warm = executable(biases, weights, keys, hidden, outputs, clamp)
        warm.block_until_ready()
        started = time.perf_counter()
        measured: list[tuple[CompiledKernelArtifact, PAsymSwapTarget, int, jax.Array]] = []
        for (
            artifact,
            target,
            input_index,
            biases,
            weights,
            keys,
            hidden,
            outputs,
            clamp,
        ) in launches:
            measured.append(
                (
                    artifact,
                    target,
                    input_index,
                    executable(biases, weights, keys, hidden, outputs, clamp),
                )
            )
        synchronize_tree([item[3] for item in measured])
        execution_seconds = time.perf_counter() - started

        empirical: dict[str, list[tuple[int, int, int, int]]] = {
            target.target_hash: [(0, 0, 0, 0)] * 4 for target in deterministic.targets
        }
        for _, target, input_index, observed in measured:
            counts = output_word_counts(observed)
            empirical[target.target_hash][input_index] = counts  # type: ignore[assignment]

        results: list[CompiledKernelResult] = []
        for target, artifact in zip(deterministic.targets, deterministic.artifacts, strict=True):
            counts = tuple(empirical[target.target_hash])
            results.append(
                CompiledKernelResult(
                    target_hash=target.target_hash,
                    compiler_request_hash=request_hash,
                    optimization=KernelOptimizationResult(
                        artifact_hash=artifact.artifact_hash,
                        parameters=artifact.parameters.values,
                        selected_restart=artifact.selected_restart,
                        successful_restart_count=sum(
                            item.passed_checks for item in artifact.attempts
                        ),
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
                            )
                            for attempt in artifact.attempts
                        ),
                    ),
                    conditionals=KernelConditionalResult(
                        target_conditional=target.conditional,
                        equilibrium_conditional=_table(
                            equilibrium_conditional(artifact.parameters, model.beta)
                        ),
                        finite_horizon_conditionals={
                            horizon: _table(table)
                            for horizon, table in finite_horizon_conditional(
                                artifact.parameters, run.horizons, model.beta
                            ).items()
                        },
                        empirical_k30_counts=counts,  # type: ignore[arg-type]
                        empirical_k30_conditional=tuple(
                            tuple(float(value) / _CHAIN_COUNT for value in row) for row in counts
                        ),
                    ),
                )
            )
        summary = summarize_artifacts(results, build_paper_fixture().occurrences, model, run)
        metrics = {
            "independent_pasym_swap": MetricObservation(
                value=summary,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="bounded independent compiler summary",
                source=PAPER_SOURCE,
            ),
            "median_equilibrium_tv": MetricObservation(
                value=summary.equilibrium_tv.median,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method="recomputed from exact frozen-model conditionals",
                source=PAPER_SOURCE,
            ),
            "worst_equilibrium_tv": MetricObservation(
                value=summary.equilibrium_tv.maximum,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method="recomputed from exact frozen-model conditionals",
                source=PAPER_SOURCE,
            ),
            "maximum_k30_equilibrium_residual": MetricObservation(
                value=summary.maximum_finite_horizon_equilibrium_residual[30],
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method="recomputed from exact finite-horizon conditionals",
                source=PAPER_SOURCE,
            ),
            "maximum_empirical_k30_residual": MetricObservation(
                value=summary.maximum_empirical_k30_residual,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="independently seeded 4096-chain synthetic cross-check",
                source=PAPER_SOURCE,
            ),
            "successful_artifact_count": MetricObservation(
                value=summary.successful_artifact_count,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="bounded optimizer winner checks",
                source=PAPER_SOURCE,
            ),
            "total_cap_active_parameter_count": MetricObservation(
                value=summary.total_cap_active_parameter_count,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="bounded optimizer winner checks",
                source=PAPER_SOURCE,
            ),
            "acceptance_passed": MetricObservation(
                value=True,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="all eight acceptance gates recomputed",
                source=PAPER_SOURCE,
            ),
            "deterministic_optimizer_seconds": MetricObservation(
                value=deterministic.optimizer_seconds,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                unit="seconds",
                method="wall-clock deterministic SciPy optimization across canonical target hashes",
                source=PAPER_SOURCE,
                notes=deterministic.cache_note,
            ),
        }
        validate_independent_pasym_swap_observations(metrics, model, run, spec.seed)
        timing_method = (
            "cached shared jax.jit(jax.vmap(single_chain)) executable; one untimed synchronized "
            "warm launch, then aggregate synchronized steady-state execution"
        )
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
