"""Checked NumPy backend for one-step full-program equilibrium refinement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from thermo_lab.backends.base import ExecutionResult
from thermo_lab.backends.numpy_composed_pasym_swap import (
    NumpyComposedPAsymSwapBackend,
    _composed_provenance,
)
from thermo_lab.composed_pasym_swap_artifacts import ComposedArtifactBundle, ExactTargetCheckpoint
from thermo_lab.composed_trajectory_refinement import (
    _schedule_digest,
    estimate_equilibrium_grouped_gradient,
    evaluate_paired_equilibrium_objective,
    project_grouped_parameters,
    sample_equilibrium_terminal_occupancy,
)
from thermo_lab.composed_trajectory_refinement_reporting import (
    REFINEMENT_TIMING_METHOD as COMPOSED_TRAJECTORY_REFINEMENT_TIMING_METHOD,
)
from thermo_lab.composed_trajectory_refinement_reporting import (
    refinement_metric_observations,
)
from thermo_lab.composed_trajectory_refinement_results import (
    ComposedTrajectoryRefinementSummary,
    build_composed_trajectory_refinement_summary,
    validate_composed_trajectory_refinement_summary,
)
from thermo_lab.composed_trajectory_refinement_schema import (
    ComposedTrajectoryRefinementRunConfig,
    validate_composed_trajectory_refinement_request,
)
from thermo_lab.config import (
    COMPOSED_TRAJECTORY_REFINEMENT_EXPERIMENT_ID,
    COMPOSED_TRAJECTORY_REFINEMENT_SAMPLE_DEFINITION,
    experiment_config_path,
    load_experiment_config,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.provenance import find_repository_root
from thermo_lab.records import (
    RUN_TIMING_SOURCE,
    ExperimentSpec,
    RunRecord,
    RunTiming,
    build_run_record,
)
from thermo_lab.schemas import PAsymSwapModelConfig


@dataclass(frozen=True)
class PreparedComposedTrajectoryRefinement:
    """Trusted lineage inputs for one full-program update."""

    bundle: ComposedArtifactBundle
    initial_parameters: tuple[tuple[float, ...], ...]
    target_checkpoint: ExactTargetCheckpoint


def spawn_refinement_role_seeds(seed: int) -> tuple[int, int, int]:
    """Derive deterministic, independent occupancy/gradient/evaluation role seeds."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    children = np.random.SeedSequence(seed).spawn(3)
    values = tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)
    if len(set(values)) != 3:
        raise RuntimeError("refinement role seeds must be distinct")
    return values  # type: ignore[return-value]


class NumpyComposedTrajectoryRefinementBackend:
    """Apply one trajectory-gradient update to the audited model-context program."""

    backend_id = BackendId.NUMPY_EXACT_CATEGORICAL
    evidence_class = EvidenceClass.SOFTWARE_SIMULATION

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = repository_root
        self._composed_backend = NumpyComposedPAsymSwapBackend(repository_root)
        self._prepared_cache: dict[str, PreparedComposedTrajectoryRefinement] = {}

    def checked_request(
        self, spec: ExperimentSpec
    ) -> tuple[PAsymSwapModelConfig, ComposedTrajectoryRefinementRunConfig, str]:
        """Require the exact checked request and its audited composed lineage identity."""

        expected = load_experiment_config(
            experiment_config_path("numpy-composed-pasym-swap-trajectory-refinement-one-step.toml")
        )
        if spec.experiment_id != COMPOSED_TRAJECTORY_REFINEMENT_EXPERIMENT_ID:
            raise ValueError("Unexpected experiment request for composed trajectory refinement")
        model_json = to_json_value(spec.model_parameters)
        run_json = to_json_value(spec.run_parameters)
        model = PAsymSwapModelConfig.model_validate(model_json)
        run = ComposedTrajectoryRefinementRunConfig.model_validate(run_json)
        validate_composed_trajectory_refinement_request(model, run, spec.seed)
        if (
            spec.sample_definition != COMPOSED_TRAJECTORY_REFINEMENT_SAMPLE_DEFINITION
            or model_json != to_json_value(expected.model_parameters)
            or run_json != to_json_value(expected.run_parameters)
        ):
            raise ValueError("Refinement backend accepts only the exact checked experiment request")
        if canonical_sha256(model.model_dump(mode="json")) != spec.model_hash:
            raise ValueError("Validated refinement model differs from the hashed request")
        source = load_experiment_config(
            experiment_config_path("numpy-composed-pasym-swap-finite-gibbs.toml")
        )
        if run.source_composed_config_hash != source.non_seed_config_hash:
            raise ValueError("Refinement request is not bound to the authoritative composed study")
        request_hash = expected.non_seed_config_hash
        # The summary binds the complete config; ExperimentSpec's hash covers
        # only the run inputs. Compare hashes with the same input boundary.
        if expected.to_spec().non_seed_run_config_hash != spec.non_seed_run_config_hash:
            raise ValueError("Refinement request hash differs from the authoritative config")
        return model, run, request_hash

    def _prepared(
        self,
        model: PAsymSwapModelConfig,
        run: ComposedTrajectoryRefinementRunConfig,
        request_hash: str,
    ) -> PreparedComposedTrajectoryRefinement:
        if request_hash not in self._prepared_cache:
            source = load_experiment_config(
                experiment_config_path("numpy-composed-pasym-swap-finite-gibbs.toml")
            )
            source_prepared = self._composed_backend.prepare(source.to_spec(seed=0))
            if source_prepared.bundle.families[2] != run.start_family:
                raise ValueError("model_context must remain the checked refinement start family")
            if source_prepared.bundle.horizons[0] != run.horizon:
                raise ValueError("equilibrium must remain the checked refinement horizon")
            target = next(
                (
                    checkpoint
                    for checkpoint in source_prepared.target_checkpoints
                    if checkpoint.occurrence_count == 500
                ),
                None,
            )
            if target is None:
                raise ValueError("prepared composed lineage is missing the final target checkpoint")
            initial = source_prepared.bundle.parameter_vectors[2]
            if any(abs(value) > model.parameter_cap for row in initial for value in row):
                raise ValueError(
                    "initial model-context parameters violate the checked parameter cap"
                )
            self._prepared_cache[request_hash] = PreparedComposedTrajectoryRefinement(
                bundle=source_prepared.bundle,
                initial_parameters=initial,
                target_checkpoint=target,
            )
        return self._prepared_cache[request_hash]

    def prepare(self, spec: ExperimentSpec) -> PreparedComposedTrajectoryRefinement:
        """Validate the checked request before preparing its trusted lineage."""

        return self._prepared(*self.checked_request(spec))

    def run(self, spec: ExperimentSpec) -> RunRecord:
        return self.execute(spec).record

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        model, run, request_hash = self.checked_request(spec)
        cache_hit = request_hash in self._prepared_cache
        started = time.perf_counter()
        prepared = self._prepared(model, run, request_hash)
        compile_seconds = 0.0 if cache_hit else time.perf_counter() - started
        occupancy_seed, gradient_seed, evaluation_seed = spawn_refinement_role_seeds(spec.seed)

        started = time.perf_counter()
        occupancy_source = sample_equilibrium_terminal_occupancy(
            prepared.initial_parameters,
            prepared.bundle.occurrence_target_indices,
            prepared.bundle.occurrence_site_indices,
            site_count=25,
            batch_size=run.occupancy_batch_size,
            seed=occupancy_seed,
            beta=model.beta,
        )
        reward = tuple(
            2.0 * (observed - target)
            for observed, target in zip(
                occupancy_source.occupancy,
                prepared.target_checkpoint.occupancy,
                strict=True,
            )
        )
        gradient_source = estimate_equilibrium_grouped_gradient(
            prepared.initial_parameters,
            prepared.bundle.occurrence_target_indices,
            prepared.bundle.occurrence_site_indices,
            reward,
            site_count=25,
            batch_size=run.gradient_batch_size,
            seed=gradient_seed,
            beta=model.beta,
        )
        update = project_grouped_parameters(
            prepared.initial_parameters,
            gradient_source.mean,
            learning_rate=run.learning_rate,
            parameter_cap=model.parameter_cap,
        )
        evaluation = evaluate_paired_equilibrium_objective(
            prepared.initial_parameters,
            update.updated_parameters,
            prepared.bundle.occurrence_target_indices,
            prepared.bundle.occurrence_site_indices,
            prepared.target_checkpoint.occupancy,
            site_count=25,
            batch_size=run.evaluation_batch_size,
            seed=evaluation_seed,
            beta=model.beta,
        )
        execution_seconds = time.perf_counter() - started
        summary = build_composed_trajectory_refinement_summary(
            request_hash=request_hash,
            seed=spec.seed,
            source_bundle_digest=prepared.bundle.bundle_digest,
            exact_target_reference=prepared.target_checkpoint.exact_reference,
            beta=model.beta,
            schedule_digest=_schedule_digest(
                np.asarray(prepared.bundle.occurrence_target_indices),
                np.asarray(prepared.bundle.occurrence_site_indices),
                site_count=25,
            ),
            initial_parameters=prepared.initial_parameters,
            target_occupancy=prepared.target_checkpoint.occupancy,
            occupancy_seed=occupancy_seed,
            gradient_seed=gradient_seed,
            evaluation_seed=evaluation_seed,
            occupancy_source=occupancy_source,
            reward_coefficient=reward,
            gradient_source=gradient_source,
            learning_rate=run.learning_rate,
            parameter_cap=model.parameter_cap,
            update=update,
            evaluation=evaluation,
        )
        self._validate_summary(summary, prepared)
        if not summary.integrity_acceptance_passed:
            raise RuntimeError("composed trajectory refinement integrity acceptance failed")

        record = build_run_record(
            backend_id=self.backend_id,
            evidence_class=self.evidence_class,
            spec=spec,
            provenance=_composed_provenance(
                self.repository_root or find_repository_root(Path.cwd())
            ),
            timing=RunTiming(
                evidence_class=self.evidence_class,
                unit="seconds",
                source=RUN_TIMING_SOURCE,
                compile_seconds=compile_seconds,
                execution_seconds=execution_seconds,
                synchronized=True,
                timing_method=COMPOSED_TRAJECTORY_REFINEMENT_TIMING_METHOD,
            ),
            metrics=refinement_metric_observations(summary),
        )
        return ExecutionResult.build(record)

    @staticmethod
    def _validate_summary(
        summary: ComposedTrajectoryRefinementSummary,
        prepared: PreparedComposedTrajectoryRefinement,
    ) -> None:
        validate_composed_trajectory_refinement_summary(
            summary,
            expected_bundle_digest=prepared.bundle.bundle_digest,
            expected_initial_parameters=prepared.initial_parameters,
            expected_target_occupancy=prepared.target_checkpoint.occupancy,
            expected_target_reference=prepared.target_checkpoint.exact_reference,
        )
