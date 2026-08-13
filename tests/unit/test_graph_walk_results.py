import pytest
from pydantic import ValidationError

from thermo_lab.graph_walk_results import (
    GraphWalkAcceptance,
    GraphWalkOrderSensitivity,
    GraphWalkVariantResult,
    WeightedGraphWalkSummary,
)


def variant(resolution: int = 128, order: str = "canonical") -> GraphWalkVariantResult:
    return GraphWalkVariantResult(
        resolution=resolution,
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


def summary() -> WeightedGraphWalkSummary:
    return WeightedGraphWalkSummary(
        source_reference="https://arxiv.org/pdf/2608.01612v1#page=10",
        node_labels=("A", "B", "C", "D", "E"),
        declared_resolutions=(64, 128),
        checkpoint_times=(0.0, 10.0),
        exact_final_occupancy=(0.2,) * 5,
        variants=(
            variant(64),
            variant(64, "reverse"),
            variant(),
            variant(order="reverse"),
        ),
        order_sensitivity=(
            GraphWalkOrderSensitivity(
                resolution=64,
                final_half_l1=0.002,
                max_trajectory_half_l1=0.003,
            ),
            GraphWalkOrderSensitivity(
                resolution=128,
                final_half_l1=0.001,
                max_trajectory_half_l1=0.002,
            ),
        ),
        acceptance=GraphWalkAcceptance(passed=True, checks=("all checks passed",)),
    )


def test_summary_round_trips_as_bounded_json() -> None:
    result = summary()
    assert WeightedGraphWalkSummary.model_validate_json(result.model_dump_json()) == result
    assert "per_layer" not in result.model_dump_json()


@pytest.mark.parametrize(
    "field_name",
    (
        "node_labels",
        "declared_resolutions",
        "checkpoint_times",
        "exact_final_occupancy",
        "variants",
        "order_sensitivity",
    ),
)
def test_summary_rejects_empty_structural_fields(field_name: str) -> None:
    payload = summary().model_dump(mode="python")
    payload[field_name] = ()
    with pytest.raises(ValidationError, match=f"{field_name} must not be empty"):
        WeightedGraphWalkSummary.model_validate(payload)


def test_summary_rejects_jointly_omitted_declared_resolution() -> None:
    payload = summary().model_dump(mode="python")
    payload["variants"] = tuple(item for item in payload["variants"] if item["resolution"] != 64)
    payload["order_sensitivity"] = tuple(
        item for item in payload["order_sensitivity"] if item["resolution"] != 64
    )
    with pytest.raises(ValidationError, match="declared resolutions"):
        WeightedGraphWalkSummary.model_validate(payload)


def test_summary_rejects_empty_acceptance_checks() -> None:
    payload = summary().model_dump(mode="python")
    payload["acceptance"]["checks"] = ()
    with pytest.raises(ValidationError, match="acceptance checks must not be empty"):
        WeightedGraphWalkSummary.model_validate(payload)


def test_variant_rejects_negative_distance() -> None:
    payload = variant().model_dump(mode="python")
    payload["final_half_l1"] = -0.1
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        GraphWalkVariantResult.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["checkpoint_times"].__setitem__(0, "0.0"),
        lambda payload: payload["exact_final_occupancy"].__setitem__(0, True),
        lambda payload: payload["variants"][0].__setitem__("final_half_l1", "0.002"),
        lambda payload: payload["variants"][0]["final_occupancy"].__setitem__(0, False),
        lambda payload: payload["variants"][0]["checkpoint_occupancies"][0].__setitem__(0, "1.0"),
        lambda payload: payload["order_sensitivity"][0].__setitem__(
            "max_trajectory_half_l1", "0.003"
        ),
    ),
)
def test_summary_rejects_coerced_observed_numbers(mutation) -> None:
    payload = summary().model_dump(mode="json")
    mutation(payload)

    with pytest.raises(ValidationError):
        WeightedGraphWalkSummary.model_validate(payload)


def test_graph_result_numbers_accept_json_integers() -> None:
    payload = variant().model_dump(mode="json")
    payload.update(
        final_occupancy=[1, 0, 0, 0, 0],
        checkpoint_occupancies=[[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
        final_half_l1=0,
        max_trajectory_half_l1=0,
        final_max_abs_error=0,
        max_one_particle_leakage=0,
        max_normalization_error=0,
        minimum_state_probability=0,
    )

    result = GraphWalkVariantResult.model_validate(payload)

    assert result.final_occupancy == (1.0, 0.0, 0.0, 0.0, 0.0)
