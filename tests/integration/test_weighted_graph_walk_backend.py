import numpy as np

from thermo_lab.backends import TorxWeightedGraphWalkBackend
from thermo_lab.evidence import EvidenceClass
from thermo_lab.experiments import weighted_graph_walk_spec
from thermo_lab.graph_walk_results import WeightedGraphWalkSummary
from thermo_lab.schemas import TORX_GRAPH_WALK_SOURCE


def test_weighted_graph_backend_passes_declared_sweep() -> None:
    spec = weighted_graph_walk_spec()
    execution = TorxWeightedGraphWalkBackend().execute(spec)
    record = execution.record
    summary = WeightedGraphWalkSummary.model_validate(record.metrics["weighted_graph_walk"].value)
    assert record.evidence_class is EvidenceClass.EXACT_REFERENCE
    assert summary.acceptance.passed
    assert summary.declared_resolutions == tuple(spec.run_parameters["resolutions"])
    assert summary.checkpoint_times == tuple(spec.run_parameters["checkpoint_times"])
    assert len(summary.variants) == 12
    finest = next(
        item for item in summary.variants if item.resolution == 128 and item.order == "canonical"
    )
    assert finest.final_half_l1 <= 0.003
    assert finest.max_trajectory_half_l1 <= 0.006
    assert finest.max_one_particle_leakage <= 1e-6
    assert execution.diagnostic_series == {}
    assert "per_layer" not in record.model_dump_json()

    assert set(record.metrics) == {
        "weighted_graph_walk",
        "finest_canonical_final_half_l1",
        "finest_canonical_max_trajectory_half_l1",
        "maximum_one_particle_leakage",
        "acceptance_passed",
    }
    for metric in record.metrics.values():
        assert metric.evidence_class is EvidenceClass.EXACT_REFERENCE
        assert metric.source == TORX_GRAPH_WALK_SOURCE
        assert metric.method
    assert record.timing.synchronized
    assert record.timing.compile_seconds >= 0.0
    assert record.timing.execution_seconds >= 0.0
    assert "12 deterministic variants" in record.timing.timing_method
    assert "synchronized complete pass" in record.timing.timing_method
    assert "Excludes configuration, provenance, persistence, aggregation, and reporting" in (
        record.timing.timing_method
    )


def test_basis_collapse_preserves_node_order_and_detects_leakage() -> None:
    from thermo_lab.backends.torx_weighted_graph_walk import _summarize_state_trajectory

    states = np.zeros((2, 32), dtype=np.float64)
    states[0, np.ravel_multi_index((1, 0, 0, 0, 0), (2,) * 5)] = 1.0
    states[1, np.ravel_multi_index((0, 1, 0, 0, 0), (2,) * 5)] = 0.75
    states[1, np.ravel_multi_index((1, 1, 0, 0, 0), (2,) * 5)] = 0.25
    occupancies, leakage = _summarize_state_trajectory(states, node_count=5)
    np.testing.assert_allclose(occupancies[0], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(occupancies[1], [0.25, 1.0, 0, 0, 0])
    np.testing.assert_allclose(leakage, [0.0, 0.25])
