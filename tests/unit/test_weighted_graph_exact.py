import numpy as np
import pytest
from scipy.linalg import expm

from thermo_lab.exact.weighted_graph import (
    build_generator,
    euler_occupancies,
    exact_occupancies,
    validate_exact_trajectory,
)
from thermo_lab.experiments import weighted_graph_walk_spec
from thermo_lab.hashing import to_json_value
from thermo_lab.schemas import WeightedGraphModelConfig, WeightedGraphRunConfig

EXPECTED_FINAL = np.array(
    [
        0.235791407046705,
        0.225498386178227,
        0.217953975322491,
        0.183734148661745,
        0.137022082790832,
    ]
)


def checked_inputs():
    spec = weighted_graph_walk_spec()
    return (
        WeightedGraphModelConfig.model_validate(to_json_value(spec.model_parameters)),
        WeightedGraphRunConfig.model_validate(to_json_value(spec.run_parameters)),
    )


def test_two_node_reference_matches_closed_form() -> None:
    model, _ = checked_inputs()
    two_node = WeightedGraphModelConfig.model_validate(
        {
            **model.model_dump(mode="python"),
            "nodes": ["A", "B"],
            "edges": [{"source": "A", "target": "B", "weight": 0.3}],
            "canonical_edge_order": [["A", "B"]],
            "initial_occupancy": [1.0, 0.0],
        }
    )
    observed = exact_occupancies(two_node, np.array([0.0, 2.0]))
    swap = 0.5 * (1.0 - np.exp(-2.0 * 0.3 * 2.0))
    np.testing.assert_allclose(observed[1], [1.0 - swap, swap], atol=1e-12)


def test_paper_fixture_matches_independent_matrix_exponential() -> None:
    model, run = checked_inputs()
    generator = build_generator(model)
    observed = exact_occupancies(model, np.array([0.0, run.final_time]))
    independent = expm(generator * run.final_time) @ np.asarray(model.initial_occupancy)
    np.testing.assert_allclose(observed[-1], EXPECTED_FINAL, atol=1e-12)
    np.testing.assert_allclose(observed[-1], independent, atol=1e-12)
    validate_exact_trajectory(generator, observed, run.exact_invariant_tolerance)


def test_euler_error_decreases_at_fine_resolutions() -> None:
    model, run = checked_inputs()
    errors = []
    for resolution in (32, 64, 128):
        approximate = euler_occupancies(
            model, run.final_time, resolution, model.canonical_edge_order
        )
        exact = exact_occupancies(model, np.linspace(0.0, run.final_time, resolution + 1))
        errors.append(float(np.max(0.5 * np.abs(approximate - exact).sum(axis=1))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] <= run.finest_max_trajectory_half_l1_tolerance


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_validate_exact_trajectory_rejects_nonfinite_generator(
    invalid_value: float,
) -> None:
    model, run = checked_inputs()
    generator = build_generator(model)
    generator[0, 1] = invalid_value
    occupancies = exact_occupancies(model, np.array([0.0, run.final_time]))

    with pytest.raises(ValueError, match="generator must contain only finite values"):
        validate_exact_trajectory(generator, occupancies, run.exact_invariant_tolerance)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_validate_exact_trajectory_rejects_nonfinite_occupancies(
    invalid_value: float,
) -> None:
    model, run = checked_inputs()
    generator = build_generator(model)
    occupancies = exact_occupancies(model, np.array([0.0, run.final_time]))
    occupancies[1, 0] = invalid_value

    with pytest.raises(ValueError, match="occupancies must contain only finite values"):
        validate_exact_trajectory(generator, occupancies, run.exact_invariant_tolerance)


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -1e-12])
def test_validate_exact_trajectory_rejects_invalid_tolerance(tolerance: float) -> None:
    model, run = checked_inputs()
    generator = build_generator(model)
    occupancies = exact_occupancies(model, np.array([0.0, run.final_time]))

    with pytest.raises(ValueError, match="tolerance must be finite and nonnegative"):
        validate_exact_trajectory(generator, occupancies, tolerance)
