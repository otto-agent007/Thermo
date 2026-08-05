"""Small-system exact enumeration for THRML-compatible Ising models."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from thermo_lab.hashing import to_json_value
from thermo_lab.schemas import ISING_ENERGY_CONVENTION, JAX_NUMERIC_DTYPE, IsingModelConfig

MAX_EXACT_NODES = 16


@dataclass(frozen=True)
class IsingModel:
    """Ising parameters using spin values in {-1, +1}.

    The THRML-compatible energy convention is
    E(s) = -beta * (sum_i b_i s_i + sum_(i,j) J_ij s_i s_j).
    """

    biases: tuple[float, ...]
    edges: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]
    beta: float

    def __post_init__(self) -> None:
        if not self.biases:
            raise ValueError("An Ising model needs at least one node")
        if len(self.edges) != len(self.weights):
            raise ValueError("Each edge must have exactly one weight")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if not np.all(np.isfinite(self.biases)) or not np.all(np.isfinite(self.weights)):
            raise ValueError("Ising biases and weights must be finite")
        for left, right in self.edges:
            if left == right:
                raise ValueError("Self-edges are not supported")
            if left < 0 or right < 0 or left >= self.n_nodes or right >= self.n_nodes:
                raise ValueError(f"Edge {(left, right)} references a node outside the model")

    @property
    def n_nodes(self) -> int:
        return len(self.biases)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> IsingModel:
        validated = IsingModelConfig.model_validate(to_json_value(config))
        return cls(
            biases=tuple(validated.biases),
            edges=tuple((edge[0], edge[1]) for edge in validated.edges),
            weights=tuple(validated.weights),
            beta=validated.beta,
        )

    def as_config(self) -> dict[str, object]:
        return {
            "biases": list(self.biases),
            "edges": [list(edge) for edge in self.edges],
            "weights": list(self.weights),
            "beta": self.beta,
            "spin_values": [-1, 1],
            "energy_convention": ISING_ENERGY_CONVENTION,
            "numeric_dtype": JAX_NUMERIC_DTYPE,
        }

    def energies(self, states: NDArray[np.int8]) -> NDArray[np.float64]:
        fields = states @ np.asarray(self.biases, dtype=np.float64)
        interactions = np.zeros(states.shape[0], dtype=np.float64)
        for (left, right), weight in zip(self.edges, self.weights, strict=True):
            interactions += weight * states[:, left] * states[:, right]
        return -self.beta * (fields + interactions)


@dataclass(frozen=True)
class ExactIsingResult:
    states: NDArray[np.int8]
    energies: NDArray[np.float64]
    probabilities: NDArray[np.float64]

    @property
    def mean_spins(self) -> NDArray[np.float64]:
        return self.probabilities @ self.states

    @property
    def expected_energy(self) -> float:
        return float(self.probabilities @ self.energies)


def enumerate_ising(model: IsingModel, *, max_nodes: int = MAX_EXACT_NODES) -> ExactIsingResult:
    """Enumerate a bounded Ising state space and normalize stably."""

    if model.n_nodes > max_nodes:
        raise ValueError(
            f"Exact enumeration is limited to {max_nodes} nodes; received {model.n_nodes}"
        )
    states = np.asarray(list(itertools.product((-1, 1), repeat=model.n_nodes)), dtype=np.int8)
    energies = model.energies(states)
    log_weights = -energies
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    probabilities = weights / weights.sum()
    return ExactIsingResult(states=states, energies=energies, probabilities=probabilities)
