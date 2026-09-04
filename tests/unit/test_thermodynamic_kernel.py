import math

import numpy as np
import pytest

from thermo_lab.pasym_swap import WORD_ORDER
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    bits_to_spins,
    conditional_tv,
    context_weighted_kl,
    context_weighted_tv,
    equilibrium_conditional,
    finite_horizon_conditional,
    joint_energy,
    one_sweep_transition,
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
        KernelParameters((2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0))
    )

    assert np.all(np.isfinite(conditional))
    assert np.all(conditional > 0.0)
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0, atol=1e-15)


def test_zero_parameter_sweep_maps_every_start_to_uniform() -> None:
    transition = one_sweep_transition(KernelParameters((0.0,) * 9), input_index=0)

    assert transition.shape == (8, 8)
    assert transition.dtype == np.float64
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(transition, np.full((8, 8), 1.0 / 8.0), atol=1e-15)


def test_finite_horizon_zero_model_is_uniform_for_every_k() -> None:
    observed = finite_horizon_conditional(KernelParameters((0.0,) * 9), (1, 2, 30))

    assert tuple(observed) == (1, 2, 30)
    for conditional in observed.values():
        assert conditional.shape == (4, 4)
        assert conditional.dtype == np.float64
        np.testing.assert_allclose(conditional, np.full((4, 4), 0.25), atol=1e-15)


def direct_sweep_transition(
    params: KernelParameters, input_index: int, beta: float = 1.0
) -> np.ndarray:
    """Independent state-loop oracle in free-state order (hidden, output_0, output_1)."""

    input_bits = WORD_ORDER[input_index]
    transition = np.zeros((8, 8), dtype=np.float64)
    for hidden_index, _hidden_bit in enumerate((0, 1)):
        for output_index, output_bits in enumerate(WORD_ORDER):
            start_index = hidden_index * 4 + output_index
            hidden_weights = np.asarray(
                [
                    math.exp(
                        -beta
                        * joint_energy(
                            params,
                            2 * np.asarray((*input_bits, next_hidden, *output_bits), dtype=np.int8)
                            - 1,
                        )
                    )
                    for next_hidden in (0, 1)
                ],
                dtype=np.float64,
            )
            hidden_probabilities = hidden_weights / hidden_weights.sum()
            for next_hidden_index, next_hidden in enumerate((0, 1)):
                output_weights = np.asarray(
                    [
                        math.exp(
                            -beta
                            * joint_energy(
                                params,
                                2
                                * np.asarray(
                                    (*input_bits, next_hidden, *next_output), dtype=np.int8
                                )
                                - 1,
                            )
                        )
                        for next_output in WORD_ORDER
                    ],
                    dtype=np.float64,
                )
                output_probabilities = output_weights / output_weights.sum()
                transition[start_index, next_hidden_index * 4 : (next_hidden_index + 1) * 4] = (
                    hidden_probabilities[next_hidden_index] * output_probabilities
                )
    return transition


def test_one_sweep_transition_matches_independent_hidden_then_output_state_loop() -> None:
    params = KernelParameters((0.15, -0.25, 0.35, -0.45, 0.2, 0.3, -0.4, 0.5, -0.1))

    for input_index in range(4):
        np.testing.assert_allclose(
            one_sweep_transition(params, input_index, beta=0.75),
            direct_sweep_transition(params, input_index, beta=0.75),
            atol=1e-14,
        )


def test_finite_horizon_matches_direct_loop_and_matrix_powers() -> None:
    params = KernelParameters((0.2, -0.15, 0.3, -0.25, 0.4, -0.35, 0.1, 0.2, -0.3))
    horizons = (4, 1, 8, 2)
    observed = finite_horizon_conditional(params, horizons, beta=0.8)

    assert tuple(observed) == (1, 2, 4, 8)
    for input_index in range(4):
        transition = direct_sweep_transition(params, input_index, beta=0.8)
        for horizon, conditional in observed.items():
            direct_distribution = np.full(8, 1.0 / 8.0, dtype=np.float64)
            for _ in range(horizon):
                direct_distribution = direct_distribution @ transition
            matrix_power_distribution = np.full(
                8, 1.0 / 8.0, dtype=np.float64
            ) @ np.linalg.matrix_power(transition, horizon)
            np.testing.assert_allclose(direct_distribution, matrix_power_distribution, atol=1e-14)
            np.testing.assert_allclose(
                conditional[input_index],
                direct_distribution.reshape(2, 4).sum(axis=0),
                atol=1e-14,
            )


@pytest.mark.parametrize(
    "params",
    [
        KernelParameters((0.1, -0.2, 0.3, -0.25, 0.15, 0.2, -0.3, 0.25, -0.1)),
        KernelParameters((-0.3, 0.2, -0.15, 0.35, -0.2, 0.1, 0.25, -0.3, 0.2)),
        KernelParameters((0.25, 0.1, -0.2, -0.15, 0.3, -0.25, 0.2, 0.1, -0.35)),
    ],
)
def test_thirty_sweeps_approaches_equilibrium_for_nonzero_fixtures(
    params: KernelParameters,
) -> None:
    finite = finite_horizon_conditional(params, (30,))[30]

    np.testing.assert_allclose(finite, equilibrium_conditional(params), atol=1e-10)


@pytest.mark.parametrize("input_index", [-1, 4, True, 1.0, "1"])
def test_one_sweep_transition_rejects_noncanonical_input_indexes(input_index: object) -> None:
    with pytest.raises(ValueError, match="input_index"):
        one_sweep_transition(KernelParameters((0.0,) * 9), input_index)  # type: ignore[arg-type]


@pytest.mark.parametrize("horizons", [(), (0,), (-1,), (True,), (1, 1.5), (1, "2")])
def test_finite_horizon_rejects_nonpositive_or_noninteger_horizons(horizons: object) -> None:
    with pytest.raises(ValueError, match="horizons"):
        finite_horizon_conditional(KernelParameters((0.0,) * 9), horizons)  # type: ignore[arg-type]


@pytest.mark.parametrize("beta", [0.0, -1.0, float("inf"), float("nan")])
def test_finite_sweep_interfaces_reject_invalid_beta(beta: float) -> None:
    params = KernelParameters((0.0,) * 9)

    with pytest.raises(ValueError, match="beta"):
        one_sweep_transition(params, 0, beta=beta)
    with pytest.raises(ValueError, match="beta"):
        finite_horizon_conditional(params, (1,), beta=beta)


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


def test_context_weighted_kl_uses_natural_logs_and_zero_context_contributions() -> None:
    target = np.asarray(
        (
            (0.5, 0.5, 0.0, 0.0),
            (0.0, 0.5, 0.5, 0.0),
            (0.0, 0.25, 0.75, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    model = np.asarray(
        (
            (0.25, 0.75, 0.0, 0.0),
            (0.25, 0.25, 0.5, 0.0),
            (0.0, 0.25, 0.75, 0.0),
            (0.0, 0.5, 0.5, 0.0),
        ),
        dtype=np.float64,
    )
    weights = np.asarray((0.6, 0.25, 0.15, 0.0), dtype=np.float64)
    expected = math.fsum(
        weight
        * math.fsum(
            probability * (math.log(probability) - math.log(model_row[index]))
            for index, probability in enumerate(target_row)
            if probability > 0.0
        )
        for weight, target_row, model_row in zip(weights, target, model, strict=True)
        if weight > 0.0
    )

    assert context_weighted_kl(target, model, weights) == pytest.approx(expected, abs=1e-15)


def test_context_weighted_metrics_use_fixed_order_without_mutating_fresh_payloads() -> None:
    target = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.8, 0.2, 0.0),
            (0.0, 0.4, 0.6, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    model = np.asarray(
        (
            (0.7, 0.1, 0.1, 0.1),
            (0.1, 0.4, 0.4, 0.1),
            (0.2, 0.2, 0.5, 0.1),
            (0.1, 0.1, 0.1, 0.7),
        ),
        dtype=np.float64,
    )
    weights = np.asarray((0.6, 0.25, 0.15, 0.0), dtype=np.float64)
    target_before, model_before, weights_before = target.copy(), model.copy(), weights.copy()
    row_tv = tuple(
        0.5 * math.fsum(abs(left - right) for left, right in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(target, model, strict=True)
    )
    expected_tv = math.fsum(weight * value for weight, value in zip(weights, row_tv, strict=True))
    legacy_summary = uniform_context_kl(target, model)
    legacy_summary_text = f"{legacy_summary:.17g}"

    assert context_weighted_tv(target, model, weights) == pytest.approx(expected_tv, abs=1e-15)
    context_weighted_kl(target, model, weights)
    assert uniform_context_kl(target, model) == legacy_summary
    assert f"{uniform_context_kl(target, model):.17g}" == legacy_summary_text
    np.testing.assert_array_equal(target, target_before)
    np.testing.assert_array_equal(model, model_before)
    np.testing.assert_array_equal(weights, weights_before)


def test_context_weighted_kl_rejects_zero_model_on_positive_weighted_target_support() -> None:
    target = np.asarray(((0.5, 0.5, 0.0, 0.0),) * 4, dtype=np.float64)
    model = np.asarray(((1.0, 0.0, 0.0, 0.0),) * 4, dtype=np.float64)

    with pytest.raises(ValueError, match="model"):
        context_weighted_kl(target, model, np.asarray((0.25, 0.25, 0.25, 0.25)))


@pytest.mark.parametrize(
    "weights",
    [
        np.asarray((0.25, 0.25, 0.25)),
        np.asarray((0.25, 0.25, 0.25, 0.25000001)),
        np.asarray((0.25, 0.25, -0.25, 0.75)),
        np.asarray((0.25, 0.25, 0.25, float("nan"))),
        np.asarray((0.25, 0.25, 0.25, float("inf"))),
    ],
)
def test_context_weighted_metrics_reject_invalid_context_distributions(weights: np.ndarray) -> None:
    conditional = np.full((4, 4), 0.25, dtype=np.float64)

    with pytest.raises(ValueError, match="context_weights"):
        context_weighted_kl(conditional, conditional, weights)
    with pytest.raises(ValueError, match="context_weights"):
        context_weighted_tv(conditional, conditional, weights)


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
