"""Deterministic Torx state-vector adapter for weighted graph walks."""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torx
from numpy.typing import NDArray
from torx import psc

from thermo_lab.backends.base import ExecutionResult
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.exact import (
    build_generator,
    euler_occupancies,
    exact_occupancies,
    validate_exact_trajectory,
)
from thermo_lab.graph_walk_results import (
    GraphWalkAcceptance,
    GraphWalkOrderSensitivity,
    GraphWalkVariantResult,
    WeightedGraphWalkSummary,
)
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.provenance import collect_runtime_provenance
from thermo_lab.records import (
    ExperimentSpec,
    MetricObservation,
    RunRecord,
    RunTiming,
    build_run_record,
)
from thermo_lab.schemas import (
    TORX_GRAPH_WALK_SOURCE,
    WeightedGraphModelConfig,
    WeightedGraphRunConfig,
    validate_weighted_graph_request,
)


def _summarize_state_trajectory(
    states: NDArray[np.floating], node_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Collapse C-order basis-state probabilities to node occupancies and leakage."""

    basis_bits = np.asarray(tuple(np.ndindex(*(2,) * node_count)), dtype=np.float64)
    states_array = np.asarray(states, dtype=np.float64)
    occupancies = states_array @ basis_bits
    leakage = states_array[:, basis_bits.sum(axis=1) != 1].sum(axis=1)
    return occupancies, leakage


class TorxWeightedGraphWalkBackend:
    backend_id = BackendId.TORX_STATEVECTOR
    evidence_class = EvidenceClass.EXACT_REFERENCE

    def __init__(self, repository_root: Path | None = None):
        self.repository_root = repository_root

    def run(self, spec: ExperimentSpec) -> RunRecord:
        return self.execute(spec).record

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        requested_model = to_json_value(spec.model_parameters)
        requested_run = to_json_value(spec.run_parameters)
        model = WeightedGraphModelConfig.model_validate(requested_model)
        run = WeightedGraphRunConfig.model_validate(requested_run)
        validate_weighted_graph_request(model, run, spec.seed)
        if torx.__version__ != "0.0.1":
            raise RuntimeError(f"Expected Torx 0.0.1, found {torx.__version__}")
        if canonical_sha256(model.model_dump(mode="json")) != spec.model_hash:
            raise ValueError("Validated weighted graph model differs from the hashed request")
        if canonical_sha256(run.model_dump(mode="json")) != canonical_sha256(requested_run):
            raise ValueError("Validated weighted graph run config differs from the hashed request")

        node_index = {label: index for index, label in enumerate(model.nodes)}
        initial_state = np.zeros(2 ** len(model.nodes), dtype=np.float32)
        for node, mass in enumerate(model.initial_occupancy):
            one_particle_bits = tuple(
                1 if index == node else 0 for index in range(len(model.nodes))
            )
            flat_index = np.ravel_multi_index(one_particle_bits, (2,) * len(model.nodes))
            initial_state[flat_index] = mass
        initial = jnp.asarray(initial_state, dtype=jnp.float32)

        edge_weights = {frozenset((edge.source, edge.target)): edge.weight for edge in model.edges}
        orders = (
            ("canonical", model.canonical_edge_order),
            ("reverse", list(reversed(model.canonical_edge_order))),
        )
        simulator = psc.StateVectorSimulator()
        compiled_variants = []
        compile_seconds = 0.0
        for resolution in run.resolutions:
            for order_name, edge_order in orders:
                gates = []
                thetas = []
                for source, target in edge_order:
                    gates.append(psc.PSWAP([node_index[source], node_index[target]]))
                    probability = (
                        edge_weights[frozenset((source, target))] * run.final_time / resolution
                    )
                    theta = np.log(probability) - np.log1p(-probability)
                    thetas.append(jnp.asarray([theta], dtype=jnp.float32))
                circuit = psc.DiscretePCircuit(gates, reps=1)
                compiled_layer = simulator.build_circuit(circuit, thetas)

                def trajectory(
                    state,
                    compiled_layer=compiled_layer,
                    resolution=resolution,
                ):
                    def step(carry, _):
                        next_state = simulator.density(compiled_layer, carry)
                        return next_state, next_state

                    _, states = jax.lax.scan(step, state, xs=None, length=resolution)
                    return jnp.concatenate((state[None, :], states), axis=0)

                trajectory_fn = jax.jit(trajectory)
                compile_started = time.perf_counter()
                executable = trajectory_fn.lower(initial).compile()
                compile_seconds += time.perf_counter() - compile_started
                compiled_variants.append((resolution, order_name, edge_order, executable))

        warm_trajectories = [executable(initial) for *_, executable in compiled_variants]
        for trajectory in warm_trajectories:
            trajectory.block_until_ready()

        execution_started = time.perf_counter()
        trajectories = [executable(initial) for *_, executable in compiled_variants]
        for trajectory in trajectories:
            trajectory.block_until_ready()
        execution_seconds = time.perf_counter() - execution_started

        exact_final = None
        variant_results = []
        trajectory_by_variant = {}
        for (resolution, order_name, edge_order, _), state_trajectory in zip(
            compiled_variants, trajectories, strict=True
        ):
            states = np.asarray(state_trajectory, dtype=np.float64)
            if not np.all(np.isfinite(states)):
                raise RuntimeError(
                    f"Torx trajectory N={resolution} order={order_name} contains non-finite values"
                )
            occupancies, leakage = _summarize_state_trajectory(states, node_count=len(model.nodes))
            times = np.linspace(0.0, run.final_time, resolution + 1)
            exact = exact_occupancies(model, times)
            validate_exact_trajectory(build_generator(model), exact, run.exact_invariant_tolerance)
            numpy_euler = euler_occupancies(model, run.final_time, resolution, edge_order)
            half_l1 = 0.5 * np.abs(occupancies - exact).sum(axis=1)
            checkpoint_indices = tuple(
                int(round(checkpoint * resolution / run.final_time))
                for checkpoint in run.checkpoint_times
            )

            numpy_euler_error = float(np.max(np.abs(occupancies - numpy_euler)))
            if numpy_euler_error > run.numpy_euler_tolerance:
                raise RuntimeError(
                    f"Torx/NumPy Euler difference N={resolution} order={order_name} "
                    f"value={numpy_euler_error} exceeded bound={run.numpy_euler_tolerance}"
                )
            max_normalization_error = float(np.max(np.abs(states.sum(axis=1) - 1.0)))
            if max_normalization_error > run.torx_normalization_tolerance:
                raise RuntimeError(
                    f"Torx normalization error N={resolution} order={order_name} "
                    f"value={max_normalization_error} exceeded "
                    f"bound={run.torx_normalization_tolerance}"
                )
            minimum_state_probability = float(states.min())
            if minimum_state_probability < run.torx_minimum_probability_floor:
                raise RuntimeError(
                    f"Torx minimum probability N={resolution} order={order_name} "
                    f"value={minimum_state_probability} fell below "
                    f"bound={run.torx_minimum_probability_floor}"
                )
            max_leakage = float(leakage.max())
            if max_leakage > run.one_particle_leakage_tolerance:
                raise RuntimeError(
                    f"Torx one-particle leakage N={resolution} order={order_name} "
                    f"value={max_leakage} exceeded bound={run.one_particle_leakage_tolerance}"
                )

            exact_final = exact[-1]
            variant_results.append(
                GraphWalkVariantResult(
                    resolution=resolution,
                    order=order_name,
                    final_occupancy=tuple(float(value) for value in occupancies[-1]),
                    checkpoint_occupancies=tuple(
                        tuple(float(value) for value in occupancies[index])
                        for index in checkpoint_indices
                    ),
                    final_half_l1=float(half_l1[-1]),
                    max_trajectory_half_l1=float(half_l1.max()),
                    final_max_abs_error=float(np.max(np.abs(occupancies[-1] - exact[-1]))),
                    max_one_particle_leakage=max_leakage,
                    max_normalization_error=max_normalization_error,
                    minimum_state_probability=minimum_state_probability,
                )
            )
            trajectory_by_variant[(resolution, order_name)] = occupancies

        if exact_final is None:
            raise RuntimeError("Weighted graph walk produced no variants")
        expected_exact_final = np.asarray(run.expected_exact_final_occupancy, dtype=np.float64)
        exact_final_error = float(np.max(np.abs(exact_final - expected_exact_final)))
        if exact_final_error > run.exact_invariant_tolerance:
            raise RuntimeError(
                f"Exact final occupancy difference value={exact_final_error} exceeded "
                f"bound={run.exact_invariant_tolerance}"
            )

        finest_resolution = run.resolutions[-1]
        finest_canonical = next(
            item
            for item in variant_results
            if item.resolution == finest_resolution and item.order == "canonical"
        )
        if finest_canonical.final_half_l1 > run.finest_final_half_l1_tolerance:
            raise RuntimeError(
                f"Finest final half-L1 N={finest_resolution} order=canonical "
                f"value={finest_canonical.final_half_l1} exceeded "
                f"bound={run.finest_final_half_l1_tolerance}"
            )
        if finest_canonical.max_trajectory_half_l1 > run.finest_max_trajectory_half_l1_tolerance:
            raise RuntimeError(
                f"Finest maximum trajectory half-L1 N={finest_resolution} order=canonical "
                f"value={finest_canonical.max_trajectory_half_l1} exceeded "
                f"bound={run.finest_max_trajectory_half_l1_tolerance}"
            )

        for order_name, _ in orders:
            ordered_variants = sorted(
                (item for item in variant_results if item.order == order_name),
                key=lambda item: item.resolution,
            )
            final_three = ordered_variants[-3:]
            for metric_name in ("final_half_l1", "max_trajectory_half_l1"):
                values = tuple(getattr(item, metric_name) for item in final_three)
                if not values[0] > values[1] > values[2]:
                    resolutions = tuple(item.resolution for item in final_three)
                    raise RuntimeError(
                        f"{metric_name} did not strictly decrease for order={order_name} "
                        f"over resolutions={resolutions}: values={values}"
                    )

        sensitivity_results = []
        for resolution in run.resolutions:
            canonical = trajectory_by_variant[(resolution, "canonical")]
            reverse = trajectory_by_variant[(resolution, "reverse")]
            sensitivity = 0.5 * np.abs(canonical - reverse).sum(axis=1)
            sensitivity_results.append(
                GraphWalkOrderSensitivity(
                    resolution=resolution,
                    final_half_l1=float(sensitivity[-1]),
                    max_trajectory_half_l1=float(sensitivity.max()),
                )
            )

        summary = WeightedGraphWalkSummary(
            source_reference=TORX_GRAPH_WALK_SOURCE,
            node_labels=tuple(model.nodes),
            declared_resolutions=tuple(run.resolutions),
            checkpoint_times=tuple(run.checkpoint_times),
            exact_final_occupancy=tuple(float(value) for value in exact_final),
            variants=tuple(variant_results),
            order_sensitivity=tuple(sensitivity_results),
            acceptance=GraphWalkAcceptance(
                passed=True,
                checks=(
                    "exact trajectory invariants and declared final occupancy passed",
                    "Torx trajectories matched independent NumPy Euler trajectories",
                    "Torx normalization, probability-floor, and one-particle checks passed",
                    "finest canonical error thresholds passed",
                    "both error metrics strictly decreased over the final three resolutions",
                ),
            ),
        )
        exact_method = (
            "Torx 0.0.1 float32 StateVectorSimulator deterministic PSWAP trajectories, "
            "independently compared with NumPy float64 Euler and eigendecomposition references"
        )
        maximum_leakage = max(item.max_one_particle_leakage for item in variant_results)
        metrics = {
            "weighted_graph_walk": MetricObservation(
                value=summary.model_dump(mode="json"),
                evidence_class=self.evidence_class,
                method=exact_method,
                source=TORX_GRAPH_WALK_SOURCE,
            ),
            "finest_canonical_final_half_l1": MetricObservation(
                value=finest_canonical.final_half_l1,
                evidence_class=self.evidence_class,
                method="final half-L1 distance from the NumPy float64 exact trajectory",
                source=TORX_GRAPH_WALK_SOURCE,
            ),
            "finest_canonical_max_trajectory_half_l1": MetricObservation(
                value=finest_canonical.max_trajectory_half_l1,
                evidence_class=self.evidence_class,
                method="maximum half-L1 distance from the NumPy float64 exact trajectory",
                source=TORX_GRAPH_WALK_SOURCE,
            ),
            "maximum_one_particle_leakage": MetricObservation(
                value=maximum_leakage,
                evidence_class=self.evidence_class,
                method="maximum state-vector probability mass outside the one-particle basis",
                source=TORX_GRAPH_WALK_SOURCE,
            ),
            "acceptance_passed": MetricObservation(
                value=summary.acceptance.passed,
                evidence_class=self.evidence_class,
                method="conjunction of all predeclared weighted graph walk acceptance checks",
                source=TORX_GRAPH_WALK_SOURCE,
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
                    f"jax.lower().compile() for {len(compiled_variants)} deterministic variants; "
                    "one untimed synchronized launch of every executable; then one timed, "
                    "synchronized complete pass over all variants. Excludes configuration, "
                    "provenance, persistence, aggregation, and reporting."
                ),
            ),
            metrics=metrics,
        )
        return ExecutionResult.build(record)
