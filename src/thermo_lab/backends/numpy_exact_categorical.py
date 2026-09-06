"""NumPy exact-categorical backend for the checked trajectory estimator study."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from thermo_lab.backends.base import ExecutionResult
from thermo_lab.config import (
    TRAJECTORY_REINFORCE_EXPERIMENT_ID,
    TRAJECTORY_REINFORCE_SAMPLE_DEFINITION,
    experiment_config_path,
    load_experiment_config,
    trajectory_reinforce_non_seed_config_hash,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.pasym_swap import PAPER_SOURCE, WORD_ORDER
from thermo_lab.provenance import find_repository_root
from thermo_lab.records import (
    RUN_TIMING_SOURCE,
    ExperimentSpec,
    MetricObservation,
    PackageProvenance,
    RunRecord,
    RuntimeProvenance,
    RunTiming,
    build_run_record,
)
from thermo_lab.schemas import (
    TrajectoryReinforceModelConfig,
    TrajectoryReinforceRunConfig,
    validate_trajectory_reinforce_request,
)
from thermo_lab.thermodynamic_kernel import equilibrium_joint_conditional, sufficient_statistics
from thermo_lab.trajectory_reinforce import (
    ExactTrajectoryReference,
    TrajectoryFixture,
    build_checked_fixture,
    build_exact_reference,
)
from thermo_lab.trajectory_reinforce_reporting import (
    validate_persisted_trajectory_reinforce_record,
)
from thermo_lab.trajectory_reinforce_results import (
    GradientMoments,
    build_trajectory_reinforce_deterministic_result,
    build_trajectory_reinforce_sample_result,
    build_trajectory_reinforce_summary,
)

_TIMING_METHOD = (
    "NumPy Generator(PCG64) inverse-CDF exact categorical sampling of 65,536 augmented "
    "trajectories using four independent streams in main_0, reference_0, main_1, reference_1 "
    "order; synchronized CPU execution includes sampler setup and bounded-source model "
    "validation; excludes deterministic reference construction and final sampled-summary/"
    "run-record construction; does not use JAX, THRML, hosted simulation, or physical hardware"
)


@dataclass(frozen=True)
class SampledGradientSources:
    """Online sufficient aggregates; no sampled trajectory is retained."""

    occurrence_0: GradientMoments
    occurrence_1: GradientMoments
    cross_products: tuple[float, ...]


def _draw_inverse_cdf(rng: np.random.Generator, probabilities: NDArray[np.float64]) -> int:
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0
    return int(np.searchsorted(cumulative, rng.random(), side="right"))


def _feature_table() -> NDArray[np.float64]:
    return np.asarray(
        [
            [sufficient_statistics(parent, outcome // 4, outcome % 4) for outcome in range(8)]
            for parent in range(4)
        ],
        dtype=np.float64,
    )


def _spawn_role_generators(
    seed: int,
    *,
    seed_sequence_factory: Any = np.random.SeedSequence,
    generator_factory: Any = np.random.default_rng,
) -> tuple[Any, Any, Any, Any]:
    """Build the four role-ordered streams through one testable construction boundary."""

    children = seed_sequence_factory(seed).spawn(4)
    return tuple(generator_factory(child) for child in children)  # type: ignore[return-value]


def sample_augmented_gradient_sources(
    *,
    fixture: TrajectoryFixture,
    exact: ExactTrajectoryReference,
    batch_size: int,
    seed: int,
) -> SampledGradientSources:
    """Sample augmented trajectories and retain only float64 moment sources."""

    if not isinstance(fixture, TrajectoryFixture):
        raise TypeError("fixture must be a TrajectoryFixture")
    if not isinstance(exact, ExactTrajectoryReference):
        raise TypeError("exact must be an ExactTrajectoryReference")
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("batch_size must be an integer of at least two")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    joint = equilibrium_joint_conditional(fixture.model_parameters, beta=fixture.beta)
    features = _feature_table()
    main_0_rng, reference_0_rng, main_1_rng, reference_1_rng = _spawn_role_generators(seed)
    main_rngs = (main_0_rng, main_1_rng)
    reference_rngs = (reference_0_rng, reference_1_rng)
    sums = [np.zeros(9, dtype=np.float64), np.zeros(9, dtype=np.float64)]
    sum_squares = [np.zeros(9, dtype=np.float64), np.zeros(9, dtype=np.float64)]
    cross_products = np.zeros(9, dtype=np.float64)

    for _ in range(batch_size):
        state = [*fixture.initial_state]
        parents = [0, 0]
        main_outcomes = [0, 0]
        reference_outcomes = [0, 0]
        for occurrence, (left, right) in enumerate(fixture.occurrences):
            parent = WORD_ORDER.index((state[left], state[right]))
            main_outcome = _draw_inverse_cdf(main_rngs[occurrence], joint[parent])
            reference_outcome = _draw_inverse_cdf(reference_rngs[occurrence], joint[parent])
            parents[occurrence] = parent
            main_outcomes[occurrence] = main_outcome
            reference_outcomes[occurrence] = reference_outcome
            state[left], state[right] = WORD_ORDER[main_outcome % 4]

        reward = float(exact.reward_coefficient @ np.asarray(state, dtype=np.float64))
        gradient_0 = (
            reward
            * fixture.beta
            * (features[parents[0], main_outcomes[0]] - features[parents[0], reference_outcomes[0]])
        )
        gradient_1 = (
            reward
            * fixture.beta
            * (features[parents[1], main_outcomes[1]] - features[parents[1], reference_outcomes[1]])
        )
        sums[0] += gradient_0
        sums[1] += gradient_1
        sum_squares[0] += gradient_0 * gradient_0
        sum_squares[1] += gradient_1 * gradient_1
        cross_products += gradient_0 * gradient_1

    return SampledGradientSources(
        occurrence_0=GradientMoments(
            sample_count=batch_size,
            component_sum=tuple(float(value) for value in sums[0]),
            component_sum_squares=tuple(float(value) for value in sum_squares[0]),
        ),
        occurrence_1=GradientMoments(
            sample_count=batch_size,
            component_sum=tuple(float(value) for value in sums[1]),
            component_sum_squares=tuple(float(value) for value in sum_squares[1]),
        ),
        cross_products=tuple(float(value) for value in cross_products),
    )


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_metadata(repository_root: Path | None) -> tuple[str | None, bool | None]:
    if repository_root is None:
        return None, None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def _numpy_provenance(repository_root: Path | None) -> RuntimeProvenance:
    commit, dirty = _git_metadata(repository_root)
    packages = tuple(
        PackageProvenance(
            distribution=name,
            version=_distribution_version(name),
            artifact_verification="runtime_import_metadata; artifact hash not runtime reverified",
        )
        for name in ("numpy", "thermo-lab")
    )
    return RuntimeProvenance(
        python_version=platform.python_version(),
        platform=platform.platform(),
        jax_version="not-used",
        jaxlib_version="not-used",
        jax_backend="not-used",
        jax_devices=(),
        git_commit=commit,
        git_dirty=dirty,
        jax_enable_x64=False,
        packages=packages,
    )


class NumpyExactCategoricalBackend:
    """Execute the checked exact-categorical trajectory estimator experiment."""

    backend_id = BackendId.NUMPY_EXACT_CATEGORICAL
    evidence_class = EvidenceClass.SOFTWARE_SIMULATION

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = repository_root

    def checked_request(
        self, spec: ExperimentSpec
    ) -> tuple[TrajectoryReinforceModelConfig, TrajectoryReinforceRunConfig, str]:
        """Accept only the exact publication-checked trajectory request."""

        expected = load_experiment_config(
            experiment_config_path("numpy-trajectory-reinforce-pasym-swap.toml")
        )
        if spec.experiment_id != TRAJECTORY_REINFORCE_EXPERIMENT_ID:
            raise ValueError("Unexpected experiment request for NumPy exact-categorical backend")
        model_json = to_json_value(spec.model_parameters)
        run_json = to_json_value(spec.run_parameters)
        model = TrajectoryReinforceModelConfig.model_validate(model_json)
        run = TrajectoryReinforceRunConfig.model_validate(run_json)
        validate_trajectory_reinforce_request(model, run, spec.seed)
        if (
            spec.sample_definition != TRAJECTORY_REINFORCE_SAMPLE_DEFINITION
            or model_json != to_json_value(expected.model_parameters)
            or run_json != to_json_value(expected.run_parameters)
        ):
            raise ValueError("NumPy backend accepts only the exact checked experiment request")
        if canonical_sha256(model.model_dump(mode="json")) != spec.model_hash:
            raise ValueError("Validated trajectory model differs from the hashed request")
        if canonical_sha256(run.model_dump(mode="json")) != canonical_sha256(spec.run_parameters):
            raise ValueError("Validated trajectory run differs from the hashed request")
        request_hash = trajectory_reinforce_non_seed_config_hash(model, run)
        if request_hash != expected.non_seed_config_hash:
            raise ValueError("Checked trajectory request hash differs from authoritative config")
        return model, run, request_hash

    def run(self, spec: ExperimentSpec) -> RunRecord:
        return self.execute(spec).record

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        _, run, request_hash = self.checked_request(spec)
        fixture = build_checked_fixture()
        exact = build_exact_reference(fixture)
        deterministic = build_trajectory_reinforce_deterministic_result(
            request_hash=request_hash, fixture=fixture, exact=exact
        )

        started = time.perf_counter()
        sources = sample_augmented_gradient_sources(
            fixture=fixture,
            exact=exact,
            batch_size=run.batch_size,
            seed=spec.seed,
        )
        execution_seconds = time.perf_counter() - started
        sampled = build_trajectory_reinforce_sample_result(
            deterministic_result_digest=deterministic.deterministic_result_digest,
            seed=spec.seed,
            sample_definition=spec.sample_definition,
            occurrence_0=sources.occurrence_0,
            occurrence_1=sources.occurrence_1,
            cross_products=sources.cross_products,
            exact=deterministic.expected_reference,
        )
        summary = build_trajectory_reinforce_summary(deterministic=deterministic, sample=sampled)
        if not summary.acceptance_passed:
            raise RuntimeError("exact trajectory REINFORCE acceptance failed")

        record = build_run_record(
            backend_id=self.backend_id,
            evidence_class=self.evidence_class,
            spec=spec,
            provenance=_numpy_provenance(self.repository_root or find_repository_root(Path.cwd())),
            timing=RunTiming(
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                unit="seconds",
                source=RUN_TIMING_SOURCE,
                compile_seconds=0.0,
                execution_seconds=execution_seconds,
                synchronized=True,
                timing_method=_TIMING_METHOD,
            ),
            metrics={
                "trajectory_reinforce_summary": MetricObservation(
                    value=summary,
                    evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                    method="structured exact and NumPy exact-categorical estimator result",
                    source=PAPER_SOURCE,
                ),
                "maximum_absolute_shared_gradient_error": MetricObservation(
                    value=summary.sample.maximum_absolute_shared_gradient_error,
                    evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                    method="NumPy PCG64 inverse-CDF sampled shared-gradient error",
                    source=PAPER_SOURCE,
                ),
                "acceptance_passed": MetricObservation(
                    value=summary.acceptance_passed,
                    evidence_class=EvidenceClass.EXACT_REFERENCE,
                    method="exact identities and deterministic finite-difference acceptance",
                    source=PAPER_SOURCE,
                    notes="Independent of Monte Carlo error and uncertainty.",
                ),
            },
        )
        validate_persisted_trajectory_reinforce_record(record)
        return ExecutionResult.build(record)
