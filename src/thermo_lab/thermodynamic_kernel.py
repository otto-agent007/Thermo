"""Exact float64 equilibrium model for the declared five-spin K_(3,2) kernel."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from thermo_lab.pasym_swap import WORD_ORDER
from thermo_lab.schemas import PARAMETER_ORDER

ParameterVector = tuple[float, float, float, float, float, float, float, float, float]

_N_PARAMETERS = len(PARAMETER_ORDER)
_N_FREE_STATES = 8


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


def _checked_beta(beta: float) -> float:
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a positive finite number")
    return float(beta)


def _checked_input_index(input_index: int) -> int:
    if type(input_index) is not int or input_index not in range(len(WORD_ORDER)):
        raise ValueError("input_index must be a canonical input word index")
    return input_index


def _normalized_probabilities(
    log_weights: NDArray[np.float64], *, name: str
) -> NDArray[np.float64]:
    probabilities = np.exp(log_weights - np.logaddexp.reduce(log_weights))
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"{name} probabilities must be finite and nonnegative")
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} probabilities must sum to one")
    return probabilities


def _checked_transition(transition: NDArray[np.generic]) -> NDArray[np.float64]:
    checked = np.asarray(transition, dtype=np.float64)
    if checked.shape != (_N_FREE_STATES, _N_FREE_STATES):
        raise ValueError("transition must have shape (8, 8)")
    if not np.all(np.isfinite(checked)):
        raise ValueError("transition must contain only finite probabilities")
    if np.any(checked < 0.0):
        raise ValueError("transition must contain nonnegative probabilities")
    if not np.allclose(checked.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("transition rows must sum to one")
    return checked


def _spins_for_state(
    input_bits: tuple[int, int], hidden_bit: int, output_bits: tuple[int, int]
) -> NDArray[np.int8]:
    return 2 * np.asarray((*input_bits, hidden_bit, *output_bits), dtype=np.int8) - 1


def one_sweep_transition(
    parameters: KernelParameters, input_index: int, beta: float = 1.0
) -> NDArray[np.float64]:
    """Return one complete hidden-then-output Gibbs sweep for one clamped input."""

    if not isinstance(parameters, KernelParameters):
        raise TypeError("parameters must be KernelParameters")
    checked_input_index = _checked_input_index(input_index)
    checked_beta = _checked_beta(beta)
    input_bits = WORD_ORDER[checked_input_index]
    transition = np.empty((_N_FREE_STATES, _N_FREE_STATES), dtype=np.float64)

    for hidden_index, _ in enumerate((0, 1)):
        for output_index, current_output_bits in enumerate(WORD_ORDER):
            start_index = hidden_index * len(WORD_ORDER) + output_index
            hidden_log_weights = np.asarray(
                [
                    -checked_beta
                    * joint_energy(
                        parameters,
                        _spins_for_state(input_bits, next_hidden_bit, current_output_bits),
                    )
                    for next_hidden_bit in (0, 1)
                ],
                dtype=np.float64,
            )
            hidden_probabilities = _normalized_probabilities(
                hidden_log_weights, name="hidden conditional"
            )
            for next_hidden_index, next_hidden_bit in enumerate((0, 1)):
                output_log_weights = np.asarray(
                    [
                        -checked_beta
                        * joint_energy(
                            parameters,
                            _spins_for_state(input_bits, next_hidden_bit, next_output_bits),
                        )
                        for next_output_bits in WORD_ORDER
                    ],
                    dtype=np.float64,
                )
                output_probabilities = _normalized_probabilities(
                    output_log_weights, name="output conditional"
                )
                next_start = next_hidden_index * len(WORD_ORDER)
                transition[start_index, next_start : next_start + len(WORD_ORDER)] = (
                    hidden_probabilities[next_hidden_index] * output_probabilities
                )

    return _checked_transition(transition)


def _checked_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    try:
        requested = tuple(horizons)
    except TypeError as error:
        raise ValueError("horizons must be an iterable of positive integers") from error
    if not requested or any(type(horizon) is not int or horizon <= 0 for horizon in requested):
        raise ValueError("horizons must contain only positive integers")
    return tuple(sorted(set(requested)))


def _checked_distribution(distribution: NDArray[np.generic]) -> NDArray[np.float64]:
    checked = np.asarray(distribution, dtype=np.float64)
    if checked.shape != (_N_FREE_STATES,):
        raise ValueError("free-state distribution must have shape (8,)")
    if not np.all(np.isfinite(checked)) or np.any(checked < 0.0):
        raise ValueError("free-state distribution must be finite and nonnegative")
    if not np.isclose(checked.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("free-state distribution must sum to one")
    return checked


def finite_horizon_conditional(
    parameters: KernelParameters, horizons: Iterable[int], beta: float = 1.0
) -> dict[int, NDArray[np.float64]]:
    """Return uniform-reset output conditionals after exact complete-sweep horizons."""

    if not isinstance(parameters, KernelParameters):
        raise TypeError("parameters must be KernelParameters")
    checked_horizons = _checked_horizons(horizons)
    checked_beta = _checked_beta(beta)
    conditionals = {
        horizon: np.empty((len(WORD_ORDER), len(WORD_ORDER)), dtype=np.float64)
        for horizon in checked_horizons
    }

    for input_index in range(len(WORD_ORDER)):
        transition = one_sweep_transition(parameters, input_index, beta=checked_beta)
        distribution = np.full(_N_FREE_STATES, 1.0 / _N_FREE_STATES, dtype=np.float64)
        previous_horizon = 0
        for horizon in checked_horizons:
            for _ in range(horizon - previous_horizon):
                distribution = distribution @ transition
            distribution = _checked_distribution(distribution)
            conditionals[horizon][input_index] = distribution.reshape(2, len(WORD_ORDER)).sum(
                axis=0
            )
            previous_horizon = horizon

    return {
        horizon: _checked_conditional(conditional, name=f"horizon {horizon} conditional")
        for horizon, conditional in conditionals.items()
    }


def equilibrium_conditional(parameters: KernelParameters, beta: float = 1.0) -> NDArray[np.float64]:
    """Enumerate the exact input-major equilibrium output conditional in float64."""

    if not isinstance(parameters, KernelParameters):
        raise TypeError("parameters must be KernelParameters")
    checked_beta = _checked_beta(beta)

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
                -checked_beta * joint_energy(parameters, hidden_negative),
                -checked_beta * joint_energy(parameters, hidden_positive),
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


def _checked_context_distribution(values: NDArray[np.generic]) -> NDArray[np.float64]:
    try:
        checked = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("context_weights must be a numeric distribution") from error
    if checked.shape != (4,):
        raise ValueError("context_weights must contain exactly four weights")
    if not np.all(np.isfinite(checked)) or np.any(checked < 0.0):
        raise ValueError("context_weights must be finite and nonnegative")
    if not np.isclose(checked.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("context_weights must sum to one")
    return checked


def context_weighted_kl(
    target: NDArray[np.generic], model: NDArray[np.generic], context_weights: NDArray[np.generic]
) -> float:
    """Return target-context-weighted target-to-model KL in natural-log nats."""

    checked_target = _checked_conditional(target, name="target")
    checked_model = _checked_conditional(model, name="model")
    weights = _checked_context_distribution(context_weights)
    positive_target = (checked_target > 0.0) & (weights[:, np.newaxis] > 0.0)
    if np.any(checked_model[positive_target] <= 0.0):
        raise ValueError("model probabilities must be strictly positive on weighted target support")

    terms = np.zeros((4, 4), dtype=np.float64)
    terms[positive_target] = checked_target[positive_target] * (
        np.log(checked_target[positive_target]) - np.log(checked_model[positive_target])
    )
    return math.fsum(
        float(weight * value) for weight, value in zip(weights, terms.sum(axis=1), strict=True)
    )


def context_weighted_tv(
    target: NDArray[np.generic], model: NDArray[np.generic], context_weights: NDArray[np.generic]
) -> float:
    """Return target-context-weighted mean of canonical row total variations."""

    checked_target = _checked_conditional(target, name="target")
    checked_model = _checked_conditional(model, name="model")
    row_tv = 0.5 * np.abs(checked_target - checked_model).sum(axis=1)
    weights = _checked_context_distribution(context_weights)
    return math.fsum(float(weight * value) for weight, value in zip(weights, row_tv, strict=True))


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
