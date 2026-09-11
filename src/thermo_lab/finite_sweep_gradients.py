"""Exact derivatives of the bounded uniform-reset, finite-Gibbs endpoint law."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from thermo_lab.thermodynamic_kernel import KernelParameters, _checked_beta, sufficient_statistics
from thermo_lab.trajectory_reinforce import _readonly_float64


@dataclass(frozen=True)
class FiniteSweepJointLaw:
    """Joint endpoint probabilities and derivatives in hidden-major state order."""

    probabilities: NDArray[np.float64]
    jacobian: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name, shape in (("probabilities", (4, 8)), ("jacobian", (4, 8, 9))):
            object.__setattr__(
                self, name, _readonly_float64(getattr(self, name), shape=shape, name=name)
            )
        if np.any(self.probabilities < 0.0) or not np.allclose(
            self.probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("endpoint probabilities must be row-stochastic")
        if not np.allclose(self.jacobian.sum(axis=1), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("endpoint derivatives must preserve normalization")
        if np.any(self.jacobian[self.probabilities == 0.0] != 0.0):
            raise ValueError("zero-probability endpoints must have zero represented derivatives")

    @property
    def scores(self) -> NDArray[np.float64]:
        scores = np.divide(
            self.jacobian,
            self.probabilities[:, :, None],
            out=np.zeros_like(self.jacobian),
            where=self.probabilities[:, :, None] > 0.0,
        )
        scores.setflags(write=False)
        return scores


def _block_law_and_jacobian(
    features: NDArray[np.float64], values: NDArray[np.float64], beta: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    log_weights = beta * (features @ values)
    if not np.all(np.isfinite(log_weights)):
        raise ValueError("block log weights must remain finite")
    probabilities = np.exp(log_weights - np.logaddexp.reduce(log_weights))
    centered = features - probabilities @ features
    return probabilities, beta * probabilities[:, None] * centered


def _sweep_with_jacobian(
    parameters: KernelParameters, beta: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(parameters.values, dtype=np.float64)
    transition = np.empty((4, 8, 8), dtype=np.float64)
    derivative = np.empty((4, 8, 8, 9), dtype=np.float64)
    for parent in range(4):
        features = np.asarray(
            [
                [sufficient_statistics(parent, hidden, output) for output in range(4)]
                for hidden in range(2)
            ]
        )
        for old in range(8):
            hidden_p, hidden_jacobian = _block_law_and_jacobian(features[:, old % 4], values, beta)
            for hidden in range(2):
                output_p, output_jacobian = _block_law_and_jacobian(features[hidden], values, beta)
                section = slice(4 * hidden, 4 * hidden + 4)
                transition[parent, old, section] = hidden_p[hidden] * output_p
                derivative[parent, old, section] = (
                    hidden_jacobian[hidden] * output_p[:, None] + hidden_p[hidden] * output_jacobian
                )
    return transition, derivative


def finite_sweep_joint_law(
    parameters: KernelParameters, horizon: int, *, beta: float = 1.0
) -> FiniteSweepJointLaw:
    """Differentiate p[k+1] = p[k] T, including every sweep and a fixed reset.

    The reset is uniform over eight free states and has zero parameter
    derivative. Inputs remain clamped; every sweep updates hidden then outputs.
    This bounded exact reference neither samples paths nor trains parameters.
    """
    if not isinstance(parameters, KernelParameters):
        raise TypeError("parameters must be KernelParameters")
    if type(horizon) is not int or not 1 <= horizon <= 30:
        raise ValueError("horizon must be an integer from 1 through 30")
    if any(abs(value) > 2.0 for value in parameters.values):
        raise ValueError("parameters must satisfy the checked cap [-2, 2]")
    checked_beta = _checked_beta(beta)
    transition, derivative = _sweep_with_jacobian(parameters, checked_beta)
    probabilities = np.full((4, 8), 1.0 / 8.0, dtype=np.float64)
    jacobian = np.zeros((4, 8, 9), dtype=np.float64)
    for _ in range(horizon):
        jacobian = np.einsum("pik,pij->pjk", jacobian, transition) + np.einsum(
            "pi,pijk->pjk", probabilities, derivative
        )
        probabilities = np.einsum("pi,pij->pj", probabilities, transition)
    return FiniteSweepJointLaw(probabilities, jacobian)
