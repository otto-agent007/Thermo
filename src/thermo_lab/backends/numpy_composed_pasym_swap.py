"""Checked NumPy composed finite-Gibbs evaluation of frozen compiler artifacts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from thermo_lab.backends.base import ExecutionResult
from thermo_lab.backends.numpy_exact_categorical import _distribution_version, _numpy_provenance
from thermo_lab.backends.thrml_model_context_pasym_swap import ThrmlModelContextPAsymSwapBackend
from thermo_lab.composed_pasym_swap import sample_composed_program
from thermo_lab.composed_pasym_swap_artifacts import (
    ComposedArtifactBundle,
    ExactTargetCheckpoint,
    build_composed_artifact_bundle,
    derive_exact_target_checkpoints,
)
from thermo_lab.composed_pasym_swap_reporting import (
    COMPOSED_PASYM_SWAP_TIMING_METHOD,
    build_composed_metric_observations,
)
from thermo_lab.composed_pasym_swap_results import (
    build_composed_pasym_swap_summary,
    validate_composed_pasym_swap_summary,
)
from thermo_lab.config import (
    COMPOSED_PASYM_SWAP_EXPERIMENT_ID,
    COMPOSED_PASYM_SWAP_SAMPLE_DEFINITION,
    composed_pasym_swap_non_seed_config_hash,
    experiment_config_path,
    load_experiment_config,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import canonical_json, canonical_sha256, to_json_value
from thermo_lab.pasym_swap import build_paper_fixture
from thermo_lab.provenance import find_repository_root
from thermo_lab.records import (
    RUN_TIMING_SOURCE,
    ExperimentSpec,
    PackageProvenance,
    RunRecord,
    RuntimeProvenance,
    RunTiming,
    build_run_record,
)
from thermo_lab.schemas import (
    ComposedPAsymSwapRunConfig,
    PAsymSwapModelConfig,
    validate_composed_pasym_swap_request,
)


@dataclass(frozen=True)
class PreparedComposedInputs:
    """Immutable deterministic inputs reused across independent release seeds."""

    bundle: ComposedArtifactBundle
    target_checkpoints: tuple[ExactTargetCheckpoint, ...]


def _composed_provenance(repository_root: Path | None) -> RuntimeProvenance:
    # The lineage adapter prepares via NumPy/SciPy; importing JAX is not execution.
    base = _numpy_provenance(repository_root)
    return base.model_copy(
        update={
            "packages": tuple(
                PackageProvenance(
                    distribution=name,
                    version=_distribution_version(name),
                    artifact_verification=(
                        "runtime_import_metadata; artifact hash not runtime reverified"
                    ),
                )
                for name in ("numpy", "scipy", "thrml", "thermo-lab")
            )
        }
    )


class NumpyComposedPAsymSwapBackend:
    """Run the complete checked composed schedule without scientific acceptance gates."""

    backend_id = BackendId.NUMPY_EXACT_CATEGORICAL
    evidence_class = EvidenceClass.SOFTWARE_SIMULATION

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = repository_root
        self._lineage_backend = ThrmlModelContextPAsymSwapBackend(repository_root)
        self._bundle_cache: dict[str, PreparedComposedInputs] = {}

    def checked_request(
        self, spec: ExperimentSpec
    ) -> tuple[PAsymSwapModelConfig, ComposedPAsymSwapRunConfig, str]:
        """Require authoritative model/run/sample JSON and all three lineage identities."""

        expected = load_experiment_config(
            experiment_config_path("numpy-composed-pasym-swap-finite-gibbs.toml")
        )
        if spec.experiment_id != COMPOSED_PASYM_SWAP_EXPERIMENT_ID:
            raise ValueError("Unexpected experiment request for composed PAsymSwap backend")
        model_json = to_json_value(spec.model_parameters)
        run_json = to_json_value(spec.run_parameters)
        model = PAsymSwapModelConfig.model_validate(model_json)
        run = ComposedPAsymSwapRunConfig.model_validate(run_json)
        validate_composed_pasym_swap_request(model, run, spec.seed)
        if (
            spec.sample_definition != COMPOSED_PASYM_SWAP_SAMPLE_DEFINITION
            or model_json != to_json_value(expected.model_parameters)
            or run_json != to_json_value(expected.run_parameters)
        ):
            raise ValueError("Composed backend accepts only the exact checked experiment request")
        if canonical_sha256(model.model_dump(mode="json")) != spec.model_hash:
            raise ValueError("Validated composed model differs from the hashed request")
        if canonical_sha256(run.model_dump(mode="json")) != canonical_sha256(spec.run_parameters):
            raise ValueError("Validated composed run differs from the hashed request")

        ancestors = tuple(
            load_experiment_config(experiment_config_path(f"thrml-{family}-pasym-swap.toml"))
            for family in ("independent", "target-context", "model-context")
        )
        if any(to_json_value(item.model_parameters) != model_json for item in ancestors):
            raise ValueError("Composed and upstream lineage model JSON must match exactly")
        request_hash = composed_pasym_swap_non_seed_config_hash(
            model, run, tuple(item.non_seed_config_hash for item in ancestors)
        )
        if request_hash != expected.non_seed_config_hash:
            raise ValueError("Checked composed request hash differs from authoritative config")
        return model, run, request_hash

    def _prepared(
        self,
        model: PAsymSwapModelConfig,
        run: ComposedPAsymSwapRunConfig,
        request_hash: str,
    ) -> PreparedComposedInputs:
        if request_hash not in self._bundle_cache:
            upstream = load_experiment_config(
                experiment_config_path("thrml-model-context-pasym-swap.toml")
            ).to_spec(seed=0)
            lineage = self._lineage_backend.prepare(upstream)
            fixture = build_paper_fixture()
            bundle = build_composed_artifact_bundle(
                fixture, lineage, beta=model.beta, horizons=run.horizon_labels
            )
            targets = derive_exact_target_checkpoints(fixture, run.checkpoint_occurrences)
            self._bundle_cache[request_hash] = PreparedComposedInputs(bundle, targets)
        return self._bundle_cache[request_hash]

    def prepare(self, spec: ExperimentSpec) -> PreparedComposedInputs:
        """Check the full request before preparing or reusing its deterministic lineage."""

        return self._prepared(*self.checked_request(spec))

    def run(self, spec: ExperimentSpec) -> RunRecord:
        return self.execute(spec).record

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        model, run, request_hash = self.checked_request(spec)
        cache_hit = request_hash in self._bundle_cache
        started = time.perf_counter()
        prepared = self._prepared(model, run, request_hash)
        compile_seconds = 0.0 if cache_hit else time.perf_counter() - started

        started = time.perf_counter()
        sources = sample_composed_program(
            prepared.bundle,
            batch_size=run.trajectory_batch_size,
            seed=spec.seed,
            checkpoint_occurrences=run.checkpoint_occurrences,
        )
        execution_seconds = time.perf_counter() - started
        summary = build_composed_pasym_swap_summary(
            request_hash=request_hash,
            bundle=prepared.bundle,
            target_checkpoints=prepared.target_checkpoints,
            sources=sources,
        )
        if not summary.integrity_acceptance_passed:
            raise RuntimeError("Composed evidence integrity acceptance failed")
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
                timing_method=COMPOSED_PASYM_SWAP_TIMING_METHOD,
            ),
            metrics=build_composed_metric_observations(summary),
        )
        validate_composed_pasym_swap_summary(
            canonical_json(record.metrics["composed_pasym_swap_summary"].value),
            bundle=prepared.bundle,
            target_checkpoints=prepared.target_checkpoints,
            request_hash=request_hash,
        )
        return ExecutionResult.build(record)
