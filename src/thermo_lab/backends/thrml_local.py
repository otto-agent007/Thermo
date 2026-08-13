"""Local THRML Ising sampling validated against exact enumeration."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import thrml
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

from thermo_lab.backends.base import ExecutionResult
from thermo_lab.diagnostics import summarize_chain
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.exact import IsingModel, enumerate_ising
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.provenance import collect_runtime_provenance
from thermo_lab.records import (
    RUN_TIMING_SOURCE,
    ExperimentSpec,
    MetricObservation,
    RunRecord,
    RunTiming,
    build_run_record,
)
from thermo_lab.schemas import ThrmlRunConfig


def _validate_partition(model: IsingModel, partition: list[list[int]]) -> None:
    flat = [node for block in partition for node in block]
    if sorted(flat) != list(range(model.n_nodes)) or len(set(flat)) != len(flat):
        raise ValueError("Block partition must cover each model node exactly once")
    block_of = {node: block_index for block_index, block in enumerate(partition) for node in block}
    for left, right in model.edges:
        if block_of[left] == block_of[right]:
            raise ValueError(f"Connected nodes {(left, right)} cannot share a Gibbs block")


def _empirical_tv(samples: np.ndarray, exact_states: np.ndarray, exact_probs: np.ndarray) -> float:
    counts = Counter(tuple(int(value) for value in row) for row in samples)
    total = samples.shape[0]
    empirical = np.asarray([counts.get(tuple(row), 0) / total for row in exact_states])
    return float(0.5 * np.abs(empirical - exact_probs).sum())


class ThrmlLocalBackend:
    backend_id = BackendId.THRML_LOCAL
    evidence_class = EvidenceClass.SOFTWARE_SIMULATION

    def __init__(self, repository_root: Path | None = None):
        self.repository_root = repository_root

    def run(self, spec: ExperimentSpec) -> RunRecord:
        return self.execute(spec).record

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        if thrml.__version__ != "0.1.4":
            raise RuntimeError(f"Expected THRML 0.1.4, found {thrml.__version__}")

        requested_model = to_json_value(spec.model_parameters)
        requested_run = to_json_value(spec.run_parameters)
        model = IsingModel.from_config(requested_model)
        run_config = ThrmlRunConfig.model_validate(requested_run)
        if canonical_sha256(model.as_config()) != spec.model_hash:
            raise ValueError("Validated Ising model differs from the canonically hashed request")
        if canonical_sha256(run_config.model_dump(mode="json")) != canonical_sha256(requested_run):
            raise ValueError(
                "Validated THRML run config differs from the canonically hashed request"
            )

        partition = run_config.block_partition
        _validate_partition(model, partition)
        exact = enumerate_ising(model)

        nodes = [SpinNode() for _ in range(model.n_nodes)]
        edges = [(nodes[left], nodes[right]) for left, right in model.edges]
        ebm = IsingEBM(
            nodes,
            edges,
            jnp.asarray(model.biases, dtype=jnp.float32),
            jnp.asarray(model.weights, dtype=jnp.float32),
            jnp.asarray(model.beta, dtype=jnp.float32),
        )
        free_blocks = [Block([nodes[index] for index in block]) for block in partition]
        program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks=[])
        schedule = SamplingSchedule(
            n_warmup=run_config.n_warmup,
            n_samples=run_config.n_samples,
            steps_per_sample=run_config.steps_per_sample,
        )
        if schedule.n_samples < 2 or schedule.n_warmup < 0 or schedule.steps_per_sample < 1:
            raise ValueError(
                "THRML schedule requires samples>=2, warmup>=0, and steps_per_sample>=1"
            )

        root_key = jax.random.key(spec.seed)
        init_key, sample_key = jax.random.split(root_key, 2)
        initial_state = hinton_init(init_key, ebm, free_blocks, ())
        initial_state = jax.tree.map(lambda value: value.block_until_ready(), initial_state)

        sample_fn = jax.jit(
            lambda key, state: sample_states(
                key,
                program,
                schedule,
                state,
                [],
                [Block(nodes)],
            )
        )
        compile_started = time.perf_counter()
        executable = sample_fn.lower(sample_key, initial_state).compile()
        compile_seconds = time.perf_counter() - compile_started

        warm_observed = executable(sample_key, initial_state)
        jax.tree.map(lambda value: value.block_until_ready(), warm_observed)
        execution_started = time.perf_counter()
        observed = executable(sample_key, initial_state)
        observed = jax.tree.map(lambda value: value.block_until_ready(), observed)
        execution_seconds = time.perf_counter() - execution_started

        bool_samples = np.asarray(observed[0], dtype=bool)
        samples = 2 * bool_samples.astype(np.int8) - 1
        sampled_mean = samples.mean(axis=0)
        max_marginal_error = float(np.max(np.abs(sampled_mean - exact.mean_spins)))
        total_variation = _empirical_tv(samples, exact.states, exact.probabilities)
        sampled_energy = float(model.energies(samples).mean())
        diagnostics = summarize_chain(
            samples, complete_sweeps_per_state=run_config.steps_per_sample
        )

        marginal_tolerance = run_config.max_marginal_error_tolerance
        tv_tolerance = run_config.total_variation_tolerance
        if max_marginal_error > marginal_tolerance:
            raise RuntimeError(
                f"THRML marginal error {max_marginal_error:.6f} exceeded {marginal_tolerance}"
            )
        if total_variation > tv_tolerance:
            raise RuntimeError(
                f"THRML total-variation error {total_variation:.6f} exceeded {tv_tolerance}"
            )

        sampled_method = (
            "THRML 0.1.4 float32 two-color block Gibbs; JAX work synchronized before observation"
        )
        exact_method = f"complete NumPy float64 enumeration of 2^{model.n_nodes} spin states"
        metrics = {
            "sampled_mean_spins": MetricObservation(
                value=sampled_mean,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=sampled_method,
            ),
            "exact_mean_spins": MetricObservation(
                value=exact.mean_spins,
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method=exact_method,
            ),
            "max_marginal_error": MetricObservation(
                value=max_marginal_error,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="maximum absolute sampled-vs-exact single-spin mean error",
            ),
            "empirical_total_variation": MetricObservation(
                value=total_variation,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="half-L1 distance between empirical and exact state distributions",
            ),
            "sampled_mean_energy": MetricObservation(
                value=sampled_energy,
                unit="dimensionless_energy",
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=sampled_method,
            ),
            "exact_expected_energy": MetricObservation(
                value=exact.expected_energy,
                unit="dimensionless_energy",
                evidence_class=EvidenceClass.EXACT_REFERENCE,
                method=exact_method,
            ),
            "recorded_states": MetricObservation(
                value=schedule.n_samples,
                unit="recorded_states",
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="THRML SamplingSchedule.n_samples",
                notes="Not an effective-independent-sample count.",
            ),
            "complete_gibbs_sweeps_per_recorded_state": MetricObservation(
                value=diagnostics.complete_sweeps_per_recorded_state,
                unit="complete_gibbs_sweeps_per_recorded_state",
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="THRML SamplingSchedule.steps_per_sample",
            ),
            "lag_1_autocorrelation_by_spin": MetricObservation(
                value=[item.lag_1_autocorrelation for item in diagnostics.spin_coordinates],
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=diagnostics.spin_coordinates[0].estimator,
                notes="Coordinate-wise values; recorded states are correlated chain states.",
            ),
            "integrated_autocorrelation_time_by_spin": MetricObservation(
                value=[
                    item.integrated_autocorrelation_time for item in diagnostics.spin_coordinates
                ],
                unit="recorded_states",
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=diagnostics.spin_coordinates[0].estimator,
            ),
            "effective_sample_size_by_spin": MetricObservation(
                value=[item.ess for item in diagnostics.spin_coordinates],
                unit="effective_samples",
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=diagnostics.spin_coordinates[0].estimator,
                notes="Diagnostic only; never exceeds the recorded-state count.",
            ),
            "minimum_spin_ess": MetricObservation(
                value=diagnostics.minimum_spin_ess,
                unit="effective_samples",
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=diagnostics.spin_coordinates[0].estimator,
            ),
            "median_spin_ess": MetricObservation(
                value=diagnostics.median_spin_ess,
                unit="effective_samples",
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=diagnostics.spin_coordinates[0].estimator,
            ),
            "magnetization_trace_ess": MetricObservation(
                value=diagnostics.magnetization.ess,
                unit="effective_samples",
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method=diagnostics.magnetization.estimator,
                notes=f"status={diagnostics.magnetization.status}",
            ),
        }
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
                timing_method=(
                    "jax.lower().compile(), one untimed synchronized launch, then synchronized "
                    "steady-state execution"
                ),
            ),
            metrics=metrics,
        )
        return ExecutionResult.build(record, {"spin_states": samples})
