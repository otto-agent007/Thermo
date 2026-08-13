import pytest
from pydantic import ValidationError

from thermo_lab.graph_walk_results import (
    GraphWalkAcceptance,
    GraphWalkOrderSensitivity,
    GraphWalkVariantResult,
    WeightedGraphWalkSummary,
)


def variant(order: str = "canonical") -> GraphWalkVariantResult:
    return GraphWalkVariantResult(
        resolution=128,
        order=order,
        final_occupancy=(0.2, 0.2, 0.2, 0.2, 0.2),
        checkpoint_occupancies=((1.0, 0.0, 0.0, 0.0, 0.0), (0.2,) * 5),
        final_half_l1=0.002,
        max_trajectory_half_l1=0.005,
        final_max_abs_error=0.001,
        max_one_particle_leakage=0.0,
        max_normalization_error=1e-7,
        minimum_state_probability=0.0,
    )


def test_summary_round_trips_as_bounded_json() -> None:
    summary = WeightedGraphWalkSummary(
        source_reference="https://arxiv.org/pdf/2608.01612v1#page=10",
        node_labels=("A", "B", "C", "D", "E"),
        checkpoint_times=(0.0, 10.0),
        exact_final_occupancy=(0.2,) * 5,
        variants=(variant(), variant("reverse")),
        order_sensitivity=(
            GraphWalkOrderSensitivity(
                resolution=128,
                final_half_l1=0.001,
                max_trajectory_half_l1=0.002,
            ),
        ),
        acceptance=GraphWalkAcceptance(passed=True, checks=("all checks passed",)),
    )
    assert WeightedGraphWalkSummary.model_validate_json(summary.model_dump_json()) == summary
    assert "per_layer" not in summary.model_dump_json()


def test_variant_rejects_negative_distance() -> None:
    payload = variant().model_dump(mode="python")
    payload["final_half_l1"] = -0.1
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        GraphWalkVariantResult.model_validate(payload)
