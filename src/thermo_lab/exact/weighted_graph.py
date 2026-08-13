from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from thermo_lab.schemas import WeightedGraphModelConfig


def build_generator(model: WeightedGraphModelConfig) -> NDArray[np.float64]:
    node_index = {label: index for index, label in enumerate(model.nodes)}
    generator = np.zeros((len(model.nodes), len(model.nodes)), dtype=np.float64)
    for edge in model.edges:
        i, j = node_index[edge.source], node_index[edge.target]
        direction = np.zeros(len(model.nodes), dtype=np.float64)
        direction[i], direction[j] = 1.0, -1.0
        generator -= edge.weight * np.outer(direction, direction)
    return generator


def exact_occupancies(
    model: WeightedGraphModelConfig,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    generator = build_generator(model)
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    initial_modes = eigenvectors.T @ np.asarray(model.initial_occupancy, dtype=np.float64)
    return np.stack([eigenvectors @ (np.exp(eigenvalues * time) * initial_modes) for time in times])


def euler_occupancies(
    model: WeightedGraphModelConfig,
    final_time: float,
    resolution: int,
    edge_order: Sequence[Sequence[str]],
) -> NDArray[np.float64]:
    node_index = {label: index for index, label in enumerate(model.nodes)}
    edge_weights = {frozenset((edge.source, edge.target)): edge.weight for edge in model.edges}
    state = np.asarray(model.initial_occupancy, dtype=np.float64)
    trajectory = np.empty((resolution + 1, len(model.nodes)), dtype=np.float64)
    trajectory[0] = state
    dt = final_time / resolution
    for step in range(1, resolution + 1):
        for source, target in edge_order:
            direction = np.zeros(len(model.nodes), dtype=np.float64)
            direction[node_index[source]] = 1.0
            direction[node_index[target]] = -1.0
            probability = edge_weights[frozenset((source, target))] * dt
            gate = np.eye(len(model.nodes)) - probability * np.outer(direction, direction)
            state = gate @ state
        trajectory[step] = state
    return trajectory


def validate_exact_trajectory(
    generator: NDArray[np.float64],
    occupancies: NDArray[np.float64],
    tolerance: float,
) -> None:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not np.all(np.isfinite(generator)):
        raise ValueError("generator must contain only finite values")
    if not np.all(np.isfinite(occupancies)):
        raise ValueError("occupancies must contain only finite values")
    symmetry_error = float(np.max(np.abs(generator - generator.T)))
    if symmetry_error > tolerance:
        raise RuntimeError(f"Exact generator symmetry error {symmetry_error} exceeded {tolerance}")
    row_sum_error = float(np.max(np.abs(generator.sum(axis=1))))
    column_sum_error = float(np.max(np.abs(generator.sum(axis=0))))
    if max(row_sum_error, column_sum_error) > tolerance:
        raise RuntimeError(
            f"Exact generator sum error {max(row_sum_error, column_sum_error)} exceeded {tolerance}"
        )
    off_diagonal = generator[~np.eye(generator.shape[0], dtype=bool)]
    minimum_rate = float(off_diagonal.min())
    if minimum_rate < -tolerance:
        raise RuntimeError(
            f"Exact generator minimum off-diagonal rate {minimum_rate} is below {-tolerance}"
        )
    normalization_error = float(np.max(np.abs(occupancies.sum(axis=1) - 1.0)))
    if normalization_error > tolerance:
        raise RuntimeError(
            f"Exact occupancy normalization error {normalization_error} exceeded {tolerance}"
        )
    minimum_probability = float(occupancies.min())
    if minimum_probability < -tolerance:
        raise RuntimeError(f"Exact minimum probability {minimum_probability} is below {-tolerance}")
