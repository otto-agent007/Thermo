import math

import numpy as np
import pytest

from thermo_lab.pasym_swap import WORD_ORDER
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    bits_to_spins,
    conditional_tv,
    equilibrium_conditional,
    joint_energy,
    uniform_context_kl,
)


def test_joint_energy_uses_canonical_parameter_order() -> None:
    params = KernelParameters((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
    spins = np.asarray([-1, 1, -1, 1, -1], dtype=np.int8)
    expected = -(
        1.0 * -1
        + 2.0 * 1
        + 3.0 * -1
        + 4.0 * (-1 * 1)
        + 5.0 * (-1 * -1)
        + 6.0 * (1 * 1)
        + 7.0 * (1 * -1)
        + 8.0 * (-1 * 1)
        + 9.0 * (-1 * -1)
    )

    assert joint_energy(params, spins) == expected


def test_bit_word_to_spin_mapping_is_pinned() -> None:
    np.testing.assert_array_equal(bits_to_spins((0, 1)), np.asarray([-1, 1], dtype=np.int8))


def brute_force_conditional(params: KernelParameters, beta: float = 1.0) -> np.ndarray:
    """Deliberately independent direct-space oracle for the 32-state conditional."""

    result = np.zeros((4, 4), dtype=np.float64)
    for input_index, input_bits in enumerate(WORD_ORDER):
        weights: list[float] = []
        for output_bits in WORD_ORDER:
            weight = 0.0
            for hidden_bit in (0, 1):
                bits = (*input_bits, hidden_bit, *output_bits)
                spins = 2 * np.asarray(bits, dtype=np.int8) - 1
                weight += math.exp(-beta * joint_energy(params, spins))
            weights.append(weight)
        result[input_index] = np.asarray(weights) / sum(weights)
    return result


def test_equilibrium_conditional_matches_independent_brute_force_oracle() -> None:
    params = KernelParameters((0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9))

    np.testing.assert_allclose(
        equilibrium_conditional(params), brute_force_conditional(params), atol=1e-14
    )


def test_equilibrium_conditional_is_input_major_and_row_stochastic() -> None:
    conditional = equilibrium_conditional(KernelParameters((0.0,) * 9))

    assert conditional.shape == (4, 4)
    assert conditional.dtype == np.float64
    np.testing.assert_allclose(conditional, np.full((4, 4), 0.25), atol=1e-15)
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0, atol=1e-15)


def test_equilibrium_conditional_remains_finite_at_declared_parameter_cap() -> None:
    conditional = equilibrium_conditional(
        KernelParameters((4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0))
    )

    assert np.all(np.isfinite(conditional))
    assert np.all(conditional > 0.0)
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0, atol=1e-15)


@pytest.mark.parametrize("beta", [0.0, -1.0, float("inf"), float("nan")])
def test_equilibrium_conditional_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        equilibrium_conditional(KernelParameters((0.0,) * 9), beta=beta)


@pytest.mark.parametrize(
    "values",
    [
        (0.0,) * 8,
        (0.0,) * 10,
        (0.0, 0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, "not-a-number", 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_kernel_parameters_reject_invalid_length_or_nonfinite_values(
    values: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError):
        KernelParameters(values)  # type: ignore[arg-type]


@pytest.mark.parametrize("bits", [(0, 2), (1, -1), (True, 0), (0,), [0, 1]])
def test_bits_to_spins_rejects_noncanonical_bit_words(bits: object) -> None:
    with pytest.raises(ValueError):
        bits_to_spins(bits)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "spins",
    [
        np.asarray([-1, 1, -1, 1], dtype=np.int8),
        np.asarray([-1, 1, 0, 1, -1], dtype=np.int8),
        np.asarray([[-1, 1, -1, 1, -1]], dtype=np.int8),
    ],
)
def test_joint_energy_rejects_invalid_spin_shape_or_domain(spins: np.ndarray) -> None:
    with pytest.raises(ValueError, match="spins"):
        joint_energy(KernelParameters((0.0,) * 9), spins)


def test_uniform_context_kl_uses_only_positive_target_entries() -> None:
    target = np.asarray([[0.5, 0.5, 0.0, 0.0]] * 4, dtype=np.float64)
    model = np.asarray([[0.5, 0.5, 1e-300, 1e-300]] * 4, dtype=np.float64)

    assert uniform_context_kl(target, model) == pytest.approx(0.0)


def test_uniform_context_kl_applies_uniform_context_weighting() -> None:
    target = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float64)
    model = np.asarray([[0.5, 0.25, 0.125, 0.125]] * 4, dtype=np.float64)

    assert uniform_context_kl(target, model) == pytest.approx(math.log(2.0))


def test_uniform_context_kl_rejects_a_nonpositive_model_probability() -> None:
    target = np.asarray([[0.5, 0.5, 0.0, 0.0]] * 4, dtype=np.float64)
    model = np.asarray([[0.5, 0.5, 0.0, 0.0]] * 4, dtype=np.float64)

    with pytest.raises(ValueError, match="model"):
        uniform_context_kl(target, model)


def test_conditional_tv_returns_one_value_per_normalized_context() -> None:
    left = np.full((4, 4), 0.25, dtype=np.float64)
    right = np.asarray(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=np.float64,
    )

    observed = conditional_tv(left, right)

    assert observed.shape == (4,)
    np.testing.assert_allclose(observed, np.asarray([0.5, 0.75, 0.0, 0.2]))


@pytest.mark.parametrize(
    "conditional",
    [
        np.full((4, 3), 1.0 / 3.0),
        np.full((4, 4), 0.2),
        np.asarray([[0.25, 0.25, 0.25, -0.25]] * 4),
    ],
)
def test_conditional_metrics_reject_nonstochastic_inputs(conditional: np.ndarray) -> None:
    valid = np.full((4, 4), 0.25)

    with pytest.raises(ValueError):
        uniform_context_kl(conditional, valid)
    with pytest.raises(ValueError):
        conditional_tv(valid, conditional)
