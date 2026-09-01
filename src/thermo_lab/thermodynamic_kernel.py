"""Exact float64 equilibrium model for the declared five-spin K_(3,2) kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from thermo_lab.pasym_swap import WORD_ORDER
from thermo_lab.schemas import PARAMETER_ORDER

ParameterVector = tuple[float, float, float, float, float, float, float, float, float]

_N_PARAMETERS = len(PARAMETER_ORDER)


@dataclass(frozen=True)
class KernelParameters:
    """The nine finite fields and couplings in canonical parameter order."""

    values: ParameterVector

    def __post_init__(self) -> None:
        if len(self.values) != _N_PARAMETERS:
            raise ValueError(f"KernelParameters requires exactly {_N_PARAMETERS} values")
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in self.values
        ):
            raise ValueError("KernelParameters values must be finite real numbers")


def bits_to_spins(bits: tuple[int, int]) -> NDArray[np.int8]:
    """Map one canonical two-bit word to bipolar spins using ``2*b - 1``."""

    if type(bits) is not tuple or len(bits) != 2:
        raise ValueError("bits must be a two-item tuple")
    if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")
    return (2 * np.asarray(bits, dtype=np.int8) - 1).astype(np.int8, copy=False)


def _checked_spins(spins: NDArray[np.generic]) -> NDArray[np.float64]:
    values = np.asarray(spins)
    if values.shape != (5,):
        raise ValueError("spins must have shape (5,)")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isin(values, (-1, 1))):
        raise ValueError("spins must contain only -1 and 1")
    return values.astype(np.float64, copy=False)


def joint_energy(parameters: KernelParameters, spins: NDArray[np.generic]) -> float:
    """Return energy in role order ``(input_0, input_1, hidden, output_0, output_1)``."""

    if not isinstance(parameters, KernelParameters):
        raise TypeError("parameters must be KernelParameters")
    input_0, input_1, hidden, output_0, output_1 = _checked_spins(spins)
    (
        h_hidden,
        h_output_0,
        h_output_1,
        j_input_0_output_0,
        j_input_0_output_1,
        j_input_1_output_0,
        j_input_1_output_1,
        j_hidden_output_0,
        j_hidden_output_1,
    ) = parameters.values
    affinity = (
        h_hidden * hidden
        + h_output_0 * output_0
        + h_output_1 * output_1
        + j_input_0_output_0 * input_0 * output_0
        + j_input_0_output_1 * input_0 * output_1
        + j_input_1_output_0 * input_1 * output_0
        + j_input_1_output_1 * input_1 * output_1
        + j_hidden_output_0 * hidden * output_0
        + j_hidden_output_1 * hidden * output_1
    )
    return -float(affinity)


def equilibrium_conditional(parameters: KernelParameters, beta: float = 1.0) -> NDArray[np.float64]:
    """Enumerate the exact input-major equilibrium output conditional in float64."""

    if not isinstance(parameters, KernelParameters):
        raise TypeError("parameters must be KernelParameters")
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a positive finite number")

    conditional = np.empty((4, 4), dtype=np.float64)
    for input_index, input_bits in enumerate(WORD_ORDER):
        input_spins = bits_to_spins(input_bits)
        log_affinities = np.empty(4, dtype=np.float64)
        for output_index, output_bits in enumerate(WORD_ORDER):
            output_spins = bits_to_spins(output_bits)
            hidden_negative = np.asarray(
                (input_spins[0], input_spins[1], -1, output_spins[0], output_spins[1]),
                dtype=np.int8,
            )
            hidden_positive = np.asarray(
                (input_spins[0], input_spins[1], 1, output_spins[0], output_spins[1]),
                dtype=np.int8,
            )
            log_affinities[output_index] = np.logaddexp(
                -beta * joint_energy(parameters, hidden_negative),
                -beta * joint_energy(parameters, hidden_positive),
            )
        conditional[input_index] = np.exp(log_affinities - np.logaddexp.reduce(log_affinities))
    return conditional


def _checked_conditional(values: NDArray[np.generic], *, name: str) -> NDArray[np.float64]:
    try:
        conditional = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric conditional table") from error
    if conditional.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(conditional)):
        raise ValueError(f"{name} must contain only finite probabilities")
    if np.any(conditional < 0.0):
        raise ValueError(f"{name} must contain nonnegative probabilities")
    if not np.allclose(conditional.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} rows must sum to one")
    return conditional


def uniform_context_kl(target: NDArray[np.generic], model: NDArray[np.generic]) -> float:
    """Return uniform-context target-to-model KL without target smoothing."""

    checked_target = _checked_conditional(target, name="target")
    checked_model = _checked_conditional(model, name="model")
    if np.any(checked_model <= 0.0):
        raise ValueError("model probabilities must be strictly positive")

    terms = np.zeros((4, 4), dtype=np.float64)
    positive_target = checked_target > 0.0
    terms[positive_target] = checked_target[positive_target] * (
        np.log(checked_target[positive_target]) - np.log(checked_model[positive_target])
    )
    return float(np.dot(np.full(4, 0.25, dtype=np.float64), terms.sum(axis=1)))


def conditional_tv(left: NDArray[np.generic], right: NDArray[np.generic]) -> NDArray[np.float64]:
    """Return total variation distance for each canonical input context."""

    checked_left = _checked_conditional(left, name="left")
    checked_right = _checked_conditional(right, name="right")
    return 0.5 * np.abs(checked_left - checked_right).sum(axis=1)
