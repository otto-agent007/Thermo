"""Carefully bounded Z1 topology facts and Appendix-B cost projection.

This module is not a physical-hardware backend. The constants come from the
refined SPICE-based model in the first Thermalizers paper and therefore produce
calibrated projections with explicit exclusions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral

from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import canonical_sha256


class ClaimRelation(StrEnum):
    EQUAL = "equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"


@dataclass(frozen=True)
class RateClaim:
    value_hz: float
    relation: ClaimRelation
    source_term: str


@dataclass(frozen=True)
class Z1HardwareProfile:
    """Sealed published profile; a future profile receives a new type/version."""

    profile_id: str = field(default="z1-thermalizers-v1-2026-08", init=False)
    physical_pbits: int = field(default=269_568, init=False)
    cores: int = field(default=8, init=False)
    announced_coupling_parameters: int = field(default=215_904, init=False)
    coupling_parameter_semantics: str = field(
        default="opaque_announced_count_not_edge_count", init=False
    )
    interaction_order: int = field(default=2, init=False)
    colors: int = field(default=2, init=False)
    interior_degree: int = field(default=16, init=False)
    connection_rules: tuple[tuple[int, int], ...] = field(
        default=((1, 0), (2, 1), (2, 3), (4, 1)), init=False
    )
    directional_couplings_supported: bool = field(default=True, init=False)
    symmetric_couplings_required_for_boltzmann_equilibrium: bool = field(default=True, init=False)
    physical_grid_shape: tuple[int, int] | None = field(default=None, init=False)
    exact_physical_graph_available: bool = field(default=False, init=False)
    advertised_sampling_rate: RateClaim = field(
        default=RateClaim(
            value_hz=50e6,
            relation=ClaimRelation.GREATER_THAN,
            source_term="sampling rate",
        ),
        init=False,
    )
    cost_model_max_complete_sweep_rate: RateClaim = field(
        default=RateClaim(
            value_hz=50e6,
            relation=ClaimRelation.LESS_THAN_OR_EQUAL,
            source_term="complete two-color Gibbs sweeps per second",
        ),
        init=False,
    )
    gibbs_node_update_energy_j: float = field(default=7.09e-15, init=False)
    node_read_energy_j: float = field(default=1.692e-12, init=False)
    node_full_sram_write_energy_j: float = field(default=153.6e-12, init=False)
    io_cost_model: str = field(default="idealized_node_access_excluding_host", init=False)
    source_model: str = field(default="spice_based_refined_projection", init=False)
    source_references: tuple[str, ...] = field(
        default=(
            "https://extropic.ai/writing/from-one-to-one-billion/",
            "https://arxiv.org/pdf/2608.01615v1",
        ),
        init=False,
    )

    def interior_offsets(self) -> frozenset[tuple[int, int]]:
        """Expand each published connection rule through four rotations."""

        offsets: set[tuple[int, int]] = set()
        for dx, dy in self.connection_rules:
            offsets.update(((dx, dy), (-dy, dx), (-dx, -dy), (dy, -dx)))
        return frozenset(offsets)

    def projection_basis(self) -> dict[str, object]:
        """Return every assumption that controls the calibrated calculation."""

        return {
            "profile_id": self.profile_id,
            "physical_pbits": self.physical_pbits,
            "cores": self.cores,
            "colors": self.colors,
            "connection_rules": self.connection_rules,
            "cost_model_max_complete_sweep_rate": self.cost_model_max_complete_sweep_rate,
            "gibbs_node_update_energy_j": self.gibbs_node_update_energy_j,
            "node_read_energy_j": self.node_read_energy_j,
            "node_full_sram_write_energy_j": self.node_full_sram_write_energy_j,
            "io_cost_model": self.io_cost_model,
            "source_model": self.source_model,
            "source_references": self.source_references,
        }

    @property
    def profile_hash(self) -> str:
        return canonical_sha256(self.projection_basis())


def _nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class Z1OperationCounts:
    """Auditable counts for complete two-color sweep projections.

    One modeled complete sweep contains two color-block phases and updates each
    participating, non-clamped p-bit at most once.
    """

    logical_pbits: int
    physical_pbits_used: int
    participating_free_pbits: int
    gibbs_node_updates: int
    elapsed_complete_sweeps: int
    node_reads: int
    node_full_sram_writes: int
    clamp_state_changes: int = 0
    coupling_reflashes: int = 0
    host_round_trips: int = 0

    def __post_init__(self) -> None:
        for name in (
            "logical_pbits",
            "physical_pbits_used",
            "participating_free_pbits",
            "gibbs_node_updates",
            "elapsed_complete_sweeps",
            "node_reads",
            "node_full_sram_writes",
            "clamp_state_changes",
            "coupling_reflashes",
            "host_round_trips",
        ):
            _nonnegative_integer(name, getattr(self, name))

        capacity = Z1HardwareProfile().physical_pbits
        if self.physical_pbits_used > capacity:
            raise ValueError(
                f"physical_pbits_used exceeds the canonical single-chip capacity {capacity}"
            )
        if self.logical_pbits > self.physical_pbits_used:
            raise ValueError("logical_pbits cannot exceed physical_pbits_used")
        if self.participating_free_pbits > self.physical_pbits_used:
            raise ValueError("participating_free_pbits cannot exceed physical_pbits_used")
        if self.physical_pbits_used == 0 and any(
            (
                self.gibbs_node_updates,
                self.node_reads,
                self.node_full_sram_writes,
                self.clamp_state_changes,
                self.coupling_reflashes,
            )
        ):
            raise ValueError("Node operations require at least one physical p-bit")
        maximum_updates = self.participating_free_pbits * self.elapsed_complete_sweeps
        if self.gibbs_node_updates > maximum_updates:
            raise ValueError(
                "gibbs_node_updates cannot exceed participating_free_pbits times "
                "elapsed_complete_sweeps"
            )
        if self.clamp_state_changes > self.node_full_sram_writes:
            raise ValueError(
                "Every modeled clamp-state change must be included in node_full_sram_writes"
            )
        if self.coupling_reflashes > self.node_full_sram_writes:
            raise ValueError(
                "Every coupling-reflash event must include at least one affected "
                "node_full_sram_write"
            )

    @property
    def color_block_phases(self) -> int:
        return 2 * self.elapsed_complete_sweeps

    @classmethod
    def constant_participation(
        cls,
        *,
        logical_pbits: int,
        physical_pbits_used: int,
        participating_free_pbits: int,
        elapsed_complete_sweeps: int,
        node_reads: int = 0,
        node_full_sram_writes: int = 0,
        clamp_state_changes: int = 0,
        coupling_reflashes: int = 0,
        host_round_trips: int = 0,
    ) -> Z1OperationCounts:
        """Assume every participating free p-bit updates once per full sweep."""

        return cls(
            logical_pbits=logical_pbits,
            physical_pbits_used=physical_pbits_used,
            participating_free_pbits=participating_free_pbits,
            gibbs_node_updates=participating_free_pbits * elapsed_complete_sweeps,
            elapsed_complete_sweeps=elapsed_complete_sweeps,
            node_reads=node_reads,
            node_full_sram_writes=node_full_sram_writes,
            clamp_state_changes=clamp_state_changes,
            coupling_reflashes=coupling_reflashes,
            host_round_trips=host_round_trips,
        )


@dataclass(frozen=True)
class Z1Projection:
    profile_id: str
    profile_hash: str
    source_references: tuple[str, ...]
    operation_counts: Z1OperationCounts
    gibbs_node_update_energy_j: float
    node_read_energy_j: float
    node_full_sram_write_energy_j: float
    assumed_max_complete_sweep_rate_hz: float
    sampling_energy_j: float
    read_energy_j: float
    write_energy_j: float
    modeled_total_energy_j: float
    sampling_time_at_assumed_max_clock_s: float
    sampling_time_relation: ClaimRelation = field(
        default=ClaimRelation.GREATER_THAN_OR_EQUAL, init=False
    )
    evidence_class: EvidenceClass = field(default=EvidenceClass.CALIBRATED_PROJECTION, init=False)
    assumptions: tuple[str, ...] = field(
        default=(
            "one complete sweep is two color-block phases",
            "each participating non-clamped p-bit updates at most once per complete sweep",
            "sampling clock operates at the Appendix-B assumed maximum of 50 MHz",
            "node-level I/O is idealized",
            "coupling_reflashes counts events; energy is priced through affected node writes",
        ),
        init=False,
    )
    excluded_costs: tuple[str, ...] = field(
        default=(
            "host_energy",
            "host_latency",
            "io_latency",
            "idle_energy",
            "current_core_wide_io_access_amplification",
            "board_and_system_power",
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        profile = Z1HardwareProfile()
        if self.profile_id != profile.profile_id or self.profile_hash != profile.profile_hash:
            raise ValueError("Z1 projection must use the sealed canonical profile")
        if self.source_references != profile.source_references:
            raise ValueError("Z1 projection sources do not match the sealed canonical profile")
        expected_constants = (
            profile.gibbs_node_update_energy_j,
            profile.node_read_energy_j,
            profile.node_full_sram_write_energy_j,
            profile.cost_model_max_complete_sweep_rate.value_hz,
        )
        observed_constants = (
            self.gibbs_node_update_energy_j,
            self.node_read_energy_j,
            self.node_full_sram_write_energy_j,
            self.assumed_max_complete_sweep_rate_hz,
        )
        if observed_constants != expected_constants:
            raise ValueError("Projection constants do not match the sealed canonical profile")
        numeric_values = (
            self.sampling_energy_j,
            self.read_energy_j,
            self.write_energy_j,
            self.modeled_total_energy_j,
            self.sampling_time_at_assumed_max_clock_s,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric_values):
            raise ValueError("Projection results must be finite and non-negative")
        expected_sampling = (
            self.operation_counts.gibbs_node_updates * self.gibbs_node_update_energy_j
        )
        expected_read = self.operation_counts.node_reads * self.node_read_energy_j
        expected_write = (
            self.operation_counts.node_full_sram_writes * self.node_full_sram_write_energy_j
        )
        if not math.isclose(self.sampling_energy_j, expected_sampling, rel_tol=1e-15):
            raise ValueError("sampling_energy_j is inconsistent with operation counts")
        if not math.isclose(self.read_energy_j, expected_read, rel_tol=1e-15):
            raise ValueError("read_energy_j is inconsistent with operation counts")
        if not math.isclose(self.write_energy_j, expected_write, rel_tol=1e-15):
            raise ValueError("write_energy_j is inconsistent with operation counts")
        if not math.isclose(
            self.modeled_total_energy_j,
            expected_sampling + expected_read + expected_write,
            rel_tol=1e-15,
            abs_tol=0.0,
        ):
            raise ValueError("modeled_total_energy_j is not component-additive")
        expected_sampling_time = (
            self.operation_counts.elapsed_complete_sweeps / self.assumed_max_complete_sweep_rate_hz
        )
        if not math.isclose(
            self.sampling_time_at_assumed_max_clock_s,
            expected_sampling_time,
            rel_tol=1e-15,
            abs_tol=0.0,
        ):
            raise ValueError(
                "sampling_time_at_assumed_max_clock_s is inconsistent with operation counts"
            )


def project_z1_operations(
    counts: Z1OperationCounts,
) -> Z1Projection:
    """Apply the versioned Appendix-B idealized node-operation cost model."""

    profile = Z1HardwareProfile()
    sampling_energy = counts.gibbs_node_updates * profile.gibbs_node_update_energy_j
    read_energy = counts.node_reads * profile.node_read_energy_j
    write_energy = counts.node_full_sram_writes * profile.node_full_sram_write_energy_j
    return Z1Projection(
        profile_id=profile.profile_id,
        profile_hash=profile.profile_hash,
        source_references=profile.source_references,
        operation_counts=counts,
        gibbs_node_update_energy_j=profile.gibbs_node_update_energy_j,
        node_read_energy_j=profile.node_read_energy_j,
        node_full_sram_write_energy_j=profile.node_full_sram_write_energy_j,
        assumed_max_complete_sweep_rate_hz=(profile.cost_model_max_complete_sweep_rate.value_hz),
        sampling_energy_j=sampling_energy,
        read_energy_j=read_energy,
        write_energy_j=write_energy,
        modeled_total_energy_j=sampling_energy + read_energy + write_energy,
        sampling_time_at_assumed_max_clock_s=(
            counts.elapsed_complete_sweeps / profile.cost_model_max_complete_sweep_rate.value_hz
        ),
    )
