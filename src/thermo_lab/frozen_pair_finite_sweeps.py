"""Frozen-parameter, finite-sweep endpoint sampling and bounded terminal evidence."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictInt, field_validator, model_validator

from thermo_lab.composed_pasym_swap_artifacts import FINITE_HORIZONS, HORIZON_LABELS, HorizonLabel
from thermo_lab.composed_trajectory_refinement import (
    PairedPopulationObjectiveStatistics,
    _checked_parameter_matrix,
    _checked_positive_float,
    _checked_positive_int,
    _checked_schedule,
    _checked_seed,
    _checked_site_vector,
    _initial_states,
    _inverse_cdf_rows,
    calculate_paired_population_objective_statistics,
)
from thermo_lab.composed_trajectory_refinement_results import (
    StrictEvidenceModel,
    _tuple_json_lists,
)
from thermo_lab.hashing import canonical_sha256
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    equilibrium_joint_conditional,
    one_sweep_transition,
)


class HorizonTerminalEvidence(StrictEvidenceModel):
    """Only bounded integer summaries; scientific scalars are reconstructed on demand."""

    horizon: HorizonLabel
    table_evidence_class: Literal["exact_reference"] = "exact_reference"
    exact_tables_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    sample_count: StrictInt = Field(ge=3)
    before_counts: tuple[StrictInt, ...]
    after_counts: tuple[StrictInt, ...]
    joined_moment_counts: tuple[tuple[StrictInt, ...], ...]
    before_particle_histogram: tuple[StrictInt, ...]
    after_particle_histogram: tuple[StrictInt, ...]
    paired_leakage_counts: tuple[StrictInt, StrictInt, StrictInt, StrictInt]

    @field_validator(
        "before_counts",
        "after_counts",
        "joined_moment_counts",
        "before_particle_histogram",
        "after_particle_histogram",
        "paired_leakage_counts",
        mode="before",
    )
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def validate_moments(self) -> HorizonTerminalEvidence:
        dimension = len(self.before_counts)
        if not dimension or len(self.after_counts) != dimension:
            raise ValueError("paired occupancy dimensions must agree and be nonempty")
        # Reuse M1's count, identical-column, and exact centered-Gram feasibility guards.
        self.statistics((0.0,) * dimension)
        n = self.sample_count
        for counts, histogram, offset in (
            (self.before_counts, self.before_particle_histogram, 0),
            (self.after_counts, self.after_particle_histogram, dimension),
        ):
            if len(histogram) != dimension + 1 or any(value < 0 for value in histogram):
                raise ValueError("particle histogram must have one nonnegative count per sector")
            if sum(histogram) != n:
                raise ValueError("particle histogram must sum to the trajectory count")
            if sum(k * value for k, value in enumerate(histogram)) != sum(counts):
                raise ValueError("particle histogram must reproduce occupancy counts")
            moment_sum = sum(
                self.joined_moment_counts[i][j]
                for i in range(offset, offset + dimension)
                for j in range(offset, offset + dimension)
            )
            if sum(k * k * value for k, value in enumerate(histogram)) != moment_sum:
                raise ValueError("particle histogram must reproduce total second moment")
        paired = self.paired_leakage_counts
        if any(value < 0 for value in paired) or sum(paired) != n:
            raise ValueError("paired leakage table must count all trajectories")
        if paired[0] + paired[1] != self.before_particle_histogram[1]:
            raise ValueError("paired leakage table must match the before histogram")
        if paired[0] + paired[2] != self.after_particle_histogram[1]:
            raise ValueError("paired leakage table must match the after histogram")
        return self

    def statistics(self, target_occupancy: object) -> PairedPopulationObjectiveStatistics:
        return calculate_paired_population_objective_statistics(
            self.before_counts,
            self.after_counts,
            self.joined_moment_counts,
            sample_count=self.sample_count,
            target_occupancy=target_occupancy,
        )

    def metrics(self, target_occupancy: object) -> dict[str, object]:
        """Reconstruct scientific scalars; horizons are not independent replications."""
        n = self.sample_count
        before = 1.0 - self.before_particle_histogram[1] / n
        after = 1.0 - self.after_particle_histogram[1] / n
        # Paired indicator difference, with one complete trajectory pair as the sample.
        delta = (self.paired_leakage_counts[1] - self.paired_leakage_counts[2]) / n
        return {
            **asdict(self.statistics(target_occupancy)),
            "particle_leakage_before": before,
            "particle_leakage_after": after,
            "particle_leakage_difference_after_minus_before": delta,
        }


def build_joint_horizon_tables(parameters: object, *, beta: object) -> NDArray[np.float64]:
    """Enumerate local joint hidden/output endpoints, including uniform resets.

    Shape is (horizon, shared group, input word, joint hidden/output word).
    Finite endpoints represent exact complete hidden-then-output sweeps.
    Sampling these tables does not execute a live Gibbs chain.
    """
    checked = _checked_parameter_matrix(parameters, name="parameters")
    checked_beta = _checked_positive_float(beta, name="beta")
    tables = np.empty((len(HORIZON_LABELS), len(checked), 4, 8), dtype=np.float64)
    for group, row in enumerate(checked):
        kernel = KernelParameters(tuple(float(value) for value in row))
        tables[0, group] = equilibrium_joint_conditional(kernel, beta=checked_beta)
        for parent in range(4):
            transition = one_sweep_transition(kernel, parent, beta=checked_beta)
            distribution = np.full(8, 1.0 / 8.0, dtype=np.float64)
            previous = 0
            for index, horizon in enumerate(HORIZON_LABELS[1:], start=1):
                sweeps = FINITE_HORIZONS[horizon]
                for _ in range(sweeps - previous):
                    distribution = distribution @ transition
                tables[index, group, parent] = distribution
                previous = sweeps
    if (
        not np.all(np.isfinite(tables))
        or np.any(tables < 0.0)
        or not np.allclose(tables.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("joint endpoint tables must be finite and row-stochastic")
    tables.setflags(write=False)
    return tables


def paired_table_digest(
    horizon: HorizonLabel,
    before: NDArray[np.float64],
    after: NDArray[np.float64],
) -> str:
    return canonical_sha256(
        {
            "identity_version": "frozen_pair_joint_endpoint_tables.v1",
            "evidence_class": "exact_reference",
            "horizon": horizon,
            "before": before,
            "after": after,
            "joint_order": "hidden-major, output words 00,01,10,11",
            "reset": "uniform over eight free states at every occurrence",
            "sweep": "hidden then both outputs, inputs clamped",
        }
    )


def declared_sampling_work(horizon: HorizonLabel, *, occurrence_count: int) -> dict[str, object]:
    """Algorithmic work per trajectory per parameter vector, not measured device work."""
    if horizon not in HORIZON_LABELS:
        raise ValueError("unsupported horizon")
    count = _checked_positive_int(occurrence_count, name="occurrence_count")
    sweeps = None if horizon == "equilibrium" else FINITE_HORIZONS[horizon] * count
    return {
        "complete_sweeps_per_trajectory_per_member": sweeps,
        "free_pbit_updates_per_trajectory_per_member": None if sweeps is None else 3 * sweeps,
        "basis": "declared algorithmic counts; three free pbits per complete sweep",
        "exclusions": "reset, clamp, parameter writes, readout, host I/O, table setup",
        "execution": "NumPy draws joint endpoints; counts are not executed Gibbs updates",
    }


def evaluate_frozen_pair_horizons(
    initial_parameters: object,
    updated_parameters: object,
    occurrence_target_indices: object,
    occurrence_site_indices: object,
    target_occupancy: object,
    *,
    site_count: object,
    batch_size: object,
    seed: object,
    beta: object,
    parameter_cap: object,
) -> tuple[HorizonTerminalEvidence, ...]:
    """Use one PCG64 uniform vector per occurrence across all 14 comparison cells."""
    before_parameters = _checked_parameter_matrix(initial_parameters, name="initial_parameters")
    after_parameters = _checked_parameter_matrix(updated_parameters, name="updated_parameters")
    if before_parameters.shape != after_parameters.shape:
        raise ValueError("initial and updated parameter shapes must agree")
    cap = _checked_positive_float(parameter_cap, name="parameter_cap")
    if np.any(np.abs(before_parameters) > cap) or np.any(np.abs(after_parameters) > cap):
        raise ValueError("frozen parameters must satisfy the declared cap")
    targets, sites, dimension = _checked_schedule(
        occurrence_target_indices,
        occurrence_site_indices,
        group_count=len(before_parameters),
        site_count=site_count,
    )
    target = _checked_site_vector(target_occupancy, site_count=dimension, name="target_occupancy")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target occupancies must lie in [0, 1]")
    count = _checked_positive_int(batch_size, name="batch_size")
    if count < 3:
        raise ValueError("paired uncertainty requires at least three independent trajectories")
    rng = np.random.Generator(np.random.PCG64(_checked_seed(seed)))
    tables = (
        build_joint_horizon_tables(before_parameters, beta=beta),
        build_joint_horizon_tables(after_parameters, beta=beta),
    )
    states = np.empty((2, len(HORIZON_LABELS), count, dimension), dtype=np.uint8)
    states[:] = _initial_states(batch_size=count, site_count=dimension)
    for group, (left, right) in zip(targets, sites, strict=True):
        uniforms = rng.random(count)
        for member in range(2):
            for index in range(len(HORIZON_LABELS)):
                current = states[member, index]
                parents = 2 * current[:, left].astype(np.int64) + current[:, right].astype(np.int64)
                output = _inverse_cdf_rows(tables[member][index, group, parents], uniforms) % 4
                current[:, left] = output // 2
                current[:, right] = output % 2

    evidence = []
    for index, horizon in enumerate(HORIZON_LABELS):
        before, after = states[:, index]
        joined = np.concatenate((before, after), axis=1).astype(np.int64)
        particles_before = before.sum(axis=1, dtype=np.int64)
        particles_after = after.sum(axis=1, dtype=np.int64)
        leakage_word = 2 * (particles_before != 1).astype(np.int64) + (particles_after != 1)
        cell = HorizonTerminalEvidence(
            horizon=horizon,
            exact_tables_digest=paired_table_digest(horizon, tables[0][index], tables[1][index]),
            sample_count=count,
            before_counts=tuple(int(value) for value in before.sum(axis=0)),
            after_counts=tuple(int(value) for value in after.sum(axis=0)),
            joined_moment_counts=tuple(
                tuple(int(value) for value in row) for row in joined.T @ joined
            ),
            before_particle_histogram=tuple(
                int(value) for value in np.bincount(particles_before, minlength=dimension + 1)
            ),
            after_particle_histogram=tuple(
                int(value) for value in np.bincount(particles_after, minlength=dimension + 1)
            ),
            paired_leakage_counts=tuple(
                int(value) for value in np.bincount(leakage_word, minlength=4)
            ),
        )
        cell.statistics(target)
        evidence.append(cell)
    return tuple(evidence)
