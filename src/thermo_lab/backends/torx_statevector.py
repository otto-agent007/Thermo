"""Exact Torx state-vector execution."""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torx
from torx import psc

from thermo_lab.backends.base import ExecutionResult
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.provenance import collect_runtime_provenance
from thermo_lab.records import (
    ExperimentSpec,
    MetricObservation,
    RunRecord,
    RunTiming,
    build_run_record,
)
from thermo_lab.schemas import TorxGateConfig, TorxModelConfig, TorxRunConfig


def _gate(config: TorxGateConfig) -> psc.AbstractDiscreteGate:
    gate_type = config.type
    sites = config.sites
    if gate_type == "pnot":
        return psc.PNOT(sites[0])
    return psc.PCNOT(sites)


class TorxStateVectorBackend:
    backend_id = BackendId.TORX_STATEVECTOR
    evidence_class = EvidenceClass.EXACT_REFERENCE

    def __init__(self, repository_root: Path | None = None):
        self.repository_root = repository_root

    def run(self, spec: ExperimentSpec) -> RunRecord:
        return self.execute(spec).record

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        if torx.__version__ != "0.0.1":
            raise RuntimeError(f"Expected Torx 0.0.1, found {torx.__version__}")

        requested_model = to_json_value(spec.model_parameters)
        requested_run = to_json_value(spec.run_parameters)
        model_config = TorxModelConfig.model_validate(requested_model)
        run_config = TorxRunConfig.model_validate(requested_run)
        if canonical_sha256(model_config.model_dump(mode="json")) != spec.model_hash:
            raise ValueError("Validated Torx model differs from the canonically hashed request")
        if canonical_sha256(run_config.model_dump(mode="json")) != canonical_sha256(requested_run):
            raise ValueError(
                "Validated Torx run config differs from the canonically hashed request"
            )

        gates = [_gate(config) for config in model_config.gates]
        thetas = [jnp.asarray([config.theta], dtype=jnp.float32) for config in model_config.gates]
        circuit = psc.DiscretePCircuit(gates)
        simulator = psc.StateVectorSimulator()
        compiled_circuit = simulator.build_circuit(circuit, thetas)

        initial = jnp.asarray(model_config.initial_distribution, dtype=jnp.float32)
        expected_size = int(np.prod(compiled_circuit.dims))
        if initial.shape != (expected_size,):
            raise ValueError(
                f"Initial distribution has shape {initial.shape}; expected {(expected_size,)}"
            )
        initial_np = np.asarray(initial)
        if (
            not np.all(np.isfinite(initial_np))
            or np.min(initial_np) < 0
            or not np.isclose(initial_np.sum(), 1.0, atol=1e-7)
        ):
            raise ValueError(
                "Initial Torx state vector must be a normalized probability distribution"
            )

        density_fn = jax.jit(lambda state: simulator.density(compiled_circuit, state))
        compile_started = time.perf_counter()
        executable = density_fn.lower(initial).compile()
        compile_seconds = time.perf_counter() - compile_started

        warm_density = executable(initial)
        warm_density.block_until_ready()
        execution_started = time.perf_counter()
        density = executable(initial)
        density.block_until_ready()
        execution_seconds = time.perf_counter() - execution_started

        density_np = np.asarray(density)
        if (
            not np.all(np.isfinite(density_np))
            or np.min(density_np) < -1e-7
            or not np.isclose(density_np.sum(), 1.0, atol=1e-6)
        ):
            raise RuntimeError("Torx produced an invalid probability distribution")

        expected = np.asarray(run_config.expected_distribution, dtype=float)
        if expected.shape != density_np.shape:
            raise ValueError(
                f"Expected distribution has shape {expected.shape}; received {density_np.shape}"
            )
        if (
            not np.all(np.isfinite(expected))
            or np.min(expected) < 0
            or not np.isclose(expected.sum(), 1.0, atol=1e-12)
        ):
            raise ValueError("Analytic comparison must be a normalized probability distribution")
        max_abs_error = float(np.max(np.abs(density_np - expected)))
        tolerance = run_config.absolute_tolerance
        if tolerance < 0:
            raise ValueError("Torx absolute tolerance must be finite and non-negative")
        if max_abs_error > tolerance:
            raise RuntimeError(
                f"Torx exact distribution error {max_abs_error} exceeded {tolerance}"
            )

        exact_method = "Torx 0.0.1 StateVectorSimulator with fixed float32 gate parameters"
        metrics = {
            "final_distribution": MetricObservation(
                value=density_np,
                evidence_class=self.evidence_class,
                method=exact_method,
            ),
            "probability_sum": MetricObservation(
                value=float(density_np.sum()),
                evidence_class=self.evidence_class,
                method=exact_method,
            ),
            "minimum_probability": MetricObservation(
                value=float(density_np.min()),
                evidence_class=self.evidence_class,
                method=exact_method,
            ),
            "max_abs_error_vs_analytic": MetricObservation(
                value=max_abs_error,
                evidence_class=self.evidence_class,
                method="comparison with hand-derived transition probabilities",
            ),
        }
        record = build_run_record(
            backend_id=self.backend_id,
            evidence_class=self.evidence_class,
            spec=spec,
            provenance=collect_runtime_provenance(self.repository_root),
            timing=RunTiming(
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
        return ExecutionResult.build(record)
