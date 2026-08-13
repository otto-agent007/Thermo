from copy import deepcopy

import pytest
from pydantic import ValidationError

from thermo_lab.schemas import (
    TORX_GRAPH_WALK_SOURCE,
    WeightedGraphModelConfig,
    WeightedGraphRunConfig,
    validate_weighted_graph_request,
)


def valid_model() -> dict[str, object]:
    return {
        "source_reference": TORX_GRAPH_WALK_SOURCE,
        "nodes": ["A", "B", "C", "D", "E"],
        "edges": [
            {"source": "A", "target": "B", "weight": 0.30},
            {"source": "A", "target": "C", "weight": 0.20},
            {"source": "B", "target": "C", "weight": 0.10},
            {"source": "B", "target": "D", "weight": 0.15},
            {"source": "C", "target": "E", "weight": 0.10},
        ],
        "canonical_edge_order": [["A", "C"], ["B", "C"], ["A", "B"], ["B", "D"], ["C", "E"]],
        "initial_occupancy": [1.0, 0.0, 0.0, 0.0, 0.0],
        "numeric_dtype": "float32",
    }


def valid_run() -> dict[str, object]:
    return {
        "final_time": 10.0,
        "resolutions": [4, 8, 16, 32, 64, 128],
        "checkpoint_times": [0.0, 2.5, 5.0, 7.5, 10.0],
        "expected_exact_final_occupancy": [
            0.235791407046705,
            0.225498386178227,
            0.217953975322491,
            0.183734148661745,
            0.137022082790832,
        ],
        "exact_invariant_tolerance": 1e-12,
        "torx_normalization_tolerance": 1e-6,
        "torx_minimum_probability_floor": -1e-7,
        "one_particle_leakage_tolerance": 1e-6,
        "finest_final_half_l1_tolerance": 0.003,
        "finest_max_trajectory_half_l1_tolerance": 0.006,
        "numpy_euler_tolerance": 2e-6,
    }


def test_checked_graph_request_is_valid() -> None:
    model = WeightedGraphModelConfig.model_validate(valid_model())
    run = WeightedGraphRunConfig.model_validate(valid_run())
    validate_weighted_graph_request(model, run, seed=0)


def test_graph_model_accepts_a_normalized_occupancy_mixture() -> None:
    payload = valid_model()
    payload["initial_occupancy"] = [0.5, 0.25, 0.125, 0.075, 0.05]

    model = WeightedGraphModelConfig.model_validate(payload)

    assert model.initial_occupancy == [0.5, 0.25, 0.125, 0.075, 0.05]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["nodes"].append("A"), "unique"),
        (
            lambda value: value["edges"].append({"source": "A", "target": "A", "weight": 0.1}),
            "self-loop",
        ),
        (
            lambda value: value["edges"].__setitem__(
                0, {"source": "A", "target": "B", "weight": 0.0}
            ),
            "positive",
        ),
        (lambda value: value.__setitem__("canonical_edge_order", [["A", "B"]]), "permutation"),
    ],
)
def test_graph_model_rejects_invalid_structure(mutation, message: str) -> None:
    payload = deepcopy(valid_model())
    mutation(payload)
    with pytest.raises(ValidationError, match=message):
        WeightedGraphModelConfig.model_validate(payload)


@pytest.mark.parametrize(
    "initial_occupancy",
    (
        [1.1, -0.1, 0.0, 0.0, 0.0],
        [float("nan"), 1.0, 0.0, 0.0, 0.0],
        [float("inf"), 0.0, 0.0, 0.0, 0.0],
    ),
)
def test_graph_model_rejects_negative_or_nonfinite_initial_occupancy(
    initial_occupancy: list[float],
) -> None:
    payload = valid_model()
    payload["initial_occupancy"] = initial_occupancy

    with pytest.raises(ValidationError):
        WeightedGraphModelConfig.model_validate(payload)


def test_request_rejects_nonzero_seed_and_invalid_euler_probability() -> None:
    model = WeightedGraphModelConfig.model_validate(valid_model())
    run = WeightedGraphRunConfig.model_validate(valid_run())
    with pytest.raises(ValueError, match="seed zero"):
        validate_weighted_graph_request(model, run, seed=1)
    coarse = WeightedGraphRunConfig.model_validate(
        {
            **valid_run(),
            "final_time": 40.0,
            "resolutions": [8, 16, 32],
            "checkpoint_times": [0.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    with pytest.raises(ValueError, match="Euler probability"):
        validate_weighted_graph_request(model, coarse, seed=0)
