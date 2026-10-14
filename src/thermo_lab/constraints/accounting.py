"""Z1 operation counts and calibrated pricing for constraint-layer runs.

Counts are algorithmic (updates, node reads, full SRAM writes, clamp changes),
not device operations. Pricing goes through the sealed ``Z1HardwareProfile``
and is a ``calibrated_projection``, never a hardware measurement.
"""

from __future__ import annotations

import dataclasses

from thermo_lab.constraints.request import ConstraintRequest
from thermo_lab.hardware.z1 import Z1OperationCounts, project_z1_operations


def count(
    request: ConstraintRequest,
    *,
    logical_pbits: int,
    physical_pbits_used: int,
    free_pbits: int,
    sweeps: int,
    node_reads: int,
    node_full_sram_writes: int,
    clamp_state_changes: int = 0,
    active_updates: int | None = None,
) -> Z1OperationCounts:
    """Counts under the request's ``update_rule``.

    ``whole_array``: every participating free p-bit updates once per complete sweep
    (``Z1OperationCounts.constant_participation``). ``active_kernel``: only the
    active kernel's free p-bits update, so ``active_updates`` must be given.
    """
    counts = Z1OperationCounts.constant_participation(
        logical_pbits=logical_pbits,
        physical_pbits_used=physical_pbits_used,
        participating_free_pbits=free_pbits,
        elapsed_complete_sweeps=sweeps,
        node_reads=node_reads,
        node_full_sram_writes=node_full_sram_writes,
        clamp_state_changes=clamp_state_changes,
    )
    if request.update_rule == "whole_array":
        if active_updates is not None:
            raise ValueError("active_updates applies only to the active_kernel update rule")
        return counts
    if active_updates is None:
        raise ValueError("the active_kernel update rule needs active_updates")
    return dataclasses.replace(counts, gibbs_node_updates=active_updates)


def price(request: ConstraintRequest, counts: Z1OperationCounts) -> dict:
    """Calibrated Z1 projection of one count record, tagged with the request."""
    projection = project_z1_operations(counts)
    total = projection.modeled_total_energy_j
    io = projection.read_energy_j + projection.write_energy_j
    return {
        "request_digest": request.digest(),
        "relaxed": list(request.relaxed),
        "update_rule": request.update_rule,
        "counts": dataclasses.asdict(counts),
        "sampling_energy_j": projection.sampling_energy_j,
        "read_energy_j": projection.read_energy_j,
        "write_energy_j": projection.write_energy_j,
        "modeled_total_energy_j": total,
        "io_share": io / total if total else 0.0,
        "sampling_time_at_assumed_max_clock_s": projection.sampling_time_at_assumed_max_clock_s,
        "profile_hash": projection.profile_hash,
        "assumptions": list(projection.assumptions),
        "excluded_costs": list(projection.excluded_costs),
        "evidence_class": str(projection.evidence_class),
        "note": "algorithmic counts priced by the sealed Z1 profile; not device operations",
    }
