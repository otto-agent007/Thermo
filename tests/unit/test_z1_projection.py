import pytest

from thermo_lab.evidence import EvidenceClass
from thermo_lab.hardware import (
    Z1HardwareProfile,
    Z1OperationCounts,
    project_z1_operations,
)


def test_published_topology_rules_expand_to_sixteen_bipartite_offsets() -> None:
    profile = Z1HardwareProfile()
    offsets = profile.interior_offsets()

    assert len(offsets) == profile.interior_degree == 16
    assert all((dx + dy) % 2 != 0 for dx, dy in offsets)
    assert max(dx * dx + dy * dy for dx, dy in offsets) == 17
    assert profile.physical_grid_shape is None
    assert profile.exact_physical_graph_available is False
    assert profile.coupling_parameter_semantics == "opaque_announced_count_not_edge_count"


def test_projection_components_use_published_si_constants() -> None:
    projection = project_z1_operations(
        Z1OperationCounts(
            logical_pbits=1,
            physical_pbits_used=1,
            participating_free_pbits=1,
            gibbs_node_updates=1,
            elapsed_complete_sweeps=1,
            node_reads=1,
            node_full_sram_writes=1,
        )
    )

    assert projection.sampling_energy_j == pytest.approx(7.09e-15)
    assert projection.read_energy_j == pytest.approx(1.692e-12)
    assert projection.write_energy_j == pytest.approx(153.6e-12)
    assert projection.modeled_total_energy_j == pytest.approx(7.09e-15 + 1.692e-12 + 153.6e-12)
    assert projection.sampling_time_at_assumed_max_clock_s == pytest.approx(20e-9)
    assert projection.evidence_class is EvidenceClass.CALIBRATED_PROJECTION
    assert "host_energy" in projection.excluded_costs
    assert projection.profile_hash.startswith("sha256:")
    assert projection.source_references


def test_whole_chip_cycle_is_projection_not_total_chip_power() -> None:
    profile = Z1HardwareProfile()
    counts = Z1OperationCounts.constant_participation(
        logical_pbits=profile.physical_pbits,
        physical_pbits_used=profile.physical_pbits,
        participating_free_pbits=profile.physical_pbits,
        elapsed_complete_sweeps=1,
    )
    projection = project_z1_operations(counts)

    assert projection.sampling_energy_j == pytest.approx(1.91123712e-9)
    modeled_sampling_power = (
        projection.sampling_energy_j * profile.cost_model_max_complete_sweep_rate.value_hz
    )
    assert modeled_sampling_power == pytest.approx(0.095561856)
    assert projection.evidence_class is EvidenceClass.CALIBRATED_PROJECTION


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"gibbs_node_updates": -1}, ValueError),
        ({"node_reads": 1.5}, TypeError),
        ({"elapsed_complete_sweeps": True}, TypeError),
        ({"physical_pbits_used": 269_569}, ValueError),
        ({"gibbs_node_updates": 2}, ValueError),
    ],
)
def test_operation_counts_reject_invalid_values(
    kwargs: dict[str, object], exception: type[Exception]
) -> None:
    values: dict[str, object] = {
        "logical_pbits": 1,
        "physical_pbits_used": 1,
        "participating_free_pbits": 1,
        "gibbs_node_updates": 0,
        "elapsed_complete_sweeps": 1,
        "node_reads": 0,
        "node_full_sram_writes": 0,
    }
    values.update(kwargs)
    with pytest.raises(exception):
        Z1OperationCounts(**values)  # type: ignore[arg-type]


def test_node_operations_require_a_nonempty_physical_region() -> None:
    with pytest.raises(ValueError, match="at least one physical p-bit"):
        Z1OperationCounts(
            logical_pbits=0,
            physical_pbits_used=0,
            participating_free_pbits=0,
            gibbs_node_updates=0,
            elapsed_complete_sweeps=0,
            node_reads=1,
            node_full_sram_writes=0,
        )


def test_reflash_energy_requires_affected_node_writes() -> None:
    with pytest.raises(ValueError, match="coupling-reflash"):
        Z1OperationCounts(
            logical_pbits=1,
            physical_pbits_used=1,
            participating_free_pbits=0,
            gibbs_node_updates=0,
            elapsed_complete_sweeps=0,
            node_reads=0,
            node_full_sram_writes=0,
            coupling_reflashes=1,
        )


def test_canonical_profile_and_projection_evidence_cannot_be_overridden() -> None:
    with pytest.raises(TypeError):
        Z1HardwareProfile(physical_pbits=1)  # type: ignore[call-arg]

    counts = Z1OperationCounts.constant_participation(
        logical_pbits=1,
        physical_pbits_used=1,
        participating_free_pbits=1,
        elapsed_complete_sweeps=1,
    )
    projection = project_z1_operations(counts)
    with pytest.raises(TypeError):
        type(projection)(
            **{
                **projection.__dict__,
                "evidence_class": EvidenceClass.PHYSICAL_HARDWARE,
            }
        )
