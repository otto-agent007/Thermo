"""Exact checks for the bounded trajectory-level REINFORCE fixture."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import (
    WORD_ORDER,
    build_pasym_swap_conditional,
    hop_probability,
)
from thermo_lab.thermodynamic_kernel import (
    equilibrium_joint_conditional,
    sufficient_statistics,
)
from thermo_lab.trajectory_reinforce import (
    VISIBLE_STATE_ORDER,
    build_checked_fixture,
    build_exact_reference,
    terminal_law,
    terminal_law_from_probabilities,
)

CHECKED_PARAMETERS = (0.25, -0.35, 0.20, 0.45, -0.30, -0.40, 0.25, 0.30, -0.20)
EXPECTED_STATES = tuple(product((0, 1), repeat=3))


def _direct_terminal_probabilities(conditionals: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    result = np.zeros(8, dtype=np.float64)
    for first_output_index, second_output_index in product(range(4), repeat=2):
        first_output = WORD_ORDER[first_output_index]
        parent_1 = (first_output[1], 0)
        terminal_state = (first_output[0], *WORD_ORDER[second_output_index])
        probability = (
            conditionals[0][WORD_ORDER.index((1, 0)), first_output_index]
            * conditionals[1][WORD_ORDER.index(parent_1), second_output_index]
        )
        result[EXPECTED_STATES.index(terminal_state)] += probability
    return result


def test_checked_fixture_pins_the_three_site_scientific_inputs() -> None:
    fixture = build_checked_fixture()
    source = (0, 0)
    destination = (1, 0)
    p_ij = hop_probability(source, destination)
    p_ji = hop_probability(destination, source)
    independently_built_target = np.asarray(
        build_pasym_swap_conditional(p_ij, p_ji), dtype=np.float64
    )

    assert VISIBLE_STATE_ORDER == EXPECTED_STATES
    assert fixture.initial_state == (1, 0, 0)
    assert fixture.occurrences == ((0, 1), (1, 2))
    assert fixture.target_edge == (source, destination)
    assert fixture.model_parameters.values == CHECKED_PARAMETERS
    assert fixture.beta == 1.0
    assert fixture.parameter_cap == 2.0
    assert fixture.finite_difference_step == 1e-6
    assert fixture.exact_tolerance == 1e-12
    assert fixture.finite_difference_tolerance == 1e-7
    np.testing.assert_array_equal(fixture.target_conditional, independently_built_target)
    assert fixture.target_hash == canonical_sha256(
        {"word_order": WORD_ORDER, "conditional": tuple(map(tuple, independently_built_target))}
    )
    assert fixture.target_conditional.dtype == np.float64
    assert not fixture.target_conditional.flags.writeable


def test_terminal_law_propagates_outputs_and_preserves_untouched_sites() -> None:
    fixture = build_checked_fixture()
    first = np.zeros((4, 4), dtype=np.float64)
    second = np.zeros((4, 4), dtype=np.float64)
    first[:, WORD_ORDER.index((0, 1))] = 1.0
    second[:, WORD_ORDER.index((1, 1))] = 1.0

    law = terminal_law((first, second), fixture.occurrences, fixture.initial_state)

    assert law.probabilities[EXPECTED_STATES.index((0, 1, 1))] == 1.0
    assert law.occupancy == (0.0, 1.0, 1.0)


def test_target_terminal_law_matches_independent_channel_composition() -> None:
    fixture = build_checked_fixture()
    law = terminal_law(
        (fixture.target_conditional, fixture.target_conditional),
        fixture.occurrences,
        fixture.initial_state,
    )
    expected = _direct_terminal_probabilities(
        (fixture.target_conditional, fixture.target_conditional)
    )

    np.testing.assert_allclose(law.probabilities, expected, rtol=0.0, atol=1e-15)
    assert law.particle_number_leakage == 0.0
    assert law.expected_mass == pytest.approx(1.0, abs=1e-15)
    assert law.signed_mass_drift == pytest.approx(0.0, abs=1e-15)


def test_terminal_law_distinguishes_leakage_from_mass_drift() -> None:
    probabilities = np.zeros(8, dtype=np.float64)
    probabilities[EXPECTED_STATES.index((0, 0, 0))] = 0.5
    probabilities[EXPECTED_STATES.index((1, 1, 0))] = 0.5

    law = terminal_law_from_probabilities(probabilities)

    assert law.particle_number_leakage == 1.0
    assert law.expected_mass == 1.0
    assert law.signed_mass_drift == 0.0


def test_terminal_law_arrays_are_copied_finite_float64_and_read_only() -> None:
    probabilities = np.full(8, 0.125, dtype=np.float32)
    law = terminal_law_from_probabilities(probabilities)
    probabilities[0] = 1.0

    assert law.probabilities.dtype == np.float64
    assert not law.probabilities.flags.writeable
    assert law.probabilities[0] == 0.125
    with pytest.raises(ValueError, match="finite"):
        terminal_law_from_probabilities((math.nan,) + (0.0,) * 7)
    with pytest.raises(ValueError, match="boolean"):
        terminal_law_from_probabilities((True,) + (0.0,) * 7)


def test_terminal_law_rejects_negative_joint_entries_before_marginalization() -> None:
    fixture = build_checked_fixture()
    joint = equilibrium_joint_conditional(fixture.model_parameters, fixture.beta).copy()
    joint[0] = 0.0
    joint[0, 0] = -1.0
    joint[0, 4] = 2.0

    assert math.isclose(float(joint[0].sum()), 1.0)
    assert np.allclose(joint.reshape(4, 2, 4).sum(axis=1)[0], (1.0, 0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="row-stochastic"):
        terminal_law((joint, joint), fixture.occurrences, fixture.initial_state)


def test_terminal_law_rejects_boolean_joint_tables() -> None:
    fixture = build_checked_fixture()
    joint = np.zeros((4, 8), dtype=bool)
    joint[:, 0] = True

    with pytest.raises(ValueError, match="real|boolean"):
        terminal_law((joint, joint), fixture.occurrences, fixture.initial_state)


@pytest.mark.parametrize("columns", [4, 8])
def test_terminal_law_rejects_mixed_boolean_float_nested_tables(columns: int) -> None:
    fixture = build_checked_fixture()
    conditional: list[list[float | bool]] = [[1.0, *([0.0] * (columns - 1))] for _ in range(4)]
    conditional[0][1] = False

    with pytest.raises(ValueError, match="boolean"):
        terminal_law((conditional, conditional), fixture.occurrences, fixture.initial_state)


@pytest.mark.parametrize("invalid", ["1.0", 1.0 + 0.0j])
def test_terminal_law_rejects_coercible_nonreal_probabilities(invalid: object) -> None:
    with pytest.raises(ValueError, match="real numbers"):
        terminal_law_from_probabilities((invalid,) + (0.0,) * 7)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"beta": 0.0}, "beta"),
        ({"finite_difference_step": 0.0}, "step"),
        ({"parameter_cap": 0.25}, "cap"),
        ({"occurrences": ((1, 2), (0, 1))}, "occurrences"),
    ],
)
def test_fixture_rejects_inputs_that_break_the_exact_oracle(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(build_checked_fixture(), **change)


def test_fixture_rejects_a_nonstochastic_target() -> None:
    target = np.array(build_checked_fixture().target_conditional, copy=True)
    target[0, 0] = 0.5

    with pytest.raises(ValueError, match="stochastic"):
        replace(build_checked_fixture(), target_conditional=target)


def test_fixture_rejects_target_metadata_that_disagrees_with_its_channel() -> None:
    fixture = build_checked_fixture()

    with pytest.raises(ValueError, match="hash"):
        replace(fixture, target_hash="0" * 64)
    with pytest.raises(ValueError, match="probabilities"):
        replace(fixture, target_probabilities=(0.25, 0.5))
    with pytest.raises(ValueError, match="edge"):
        replace(fixture, target_edge=((1, 0), (0, 0)))


def test_fixture_rejects_coordinated_target_substitution_for_the_checked_edge() -> None:
    fixture = build_checked_fixture()
    substituted_probabilities = (0.25, 0.5)
    substituted_conditional = build_pasym_swap_conditional(*substituted_probabilities)
    substituted_hash = canonical_sha256(
        {"word_order": WORD_ORDER, "conditional": substituted_conditional}
    )

    with pytest.raises(ValueError, match="edge probabilities"):
        replace(
            fixture,
            target_probabilities=substituted_probabilities,
            target_conditional=np.asarray(substituted_conditional),
            target_hash=substituted_hash,
        )


def test_fixture_copies_target_probabilities_to_an_immutable_tuple() -> None:
    fixture = build_checked_fixture()
    mutable_probabilities = list(fixture.target_probabilities)

    copied = replace(fixture, target_probabilities=mutable_probabilities)  # type: ignore[arg-type]
    mutable_probabilities[0] = 0.5

    assert copied.target_probabilities == fixture.target_probabilities
    assert type(copied.target_probabilities) is tuple


def _feature(input_index: int, outcome_index: int) -> np.ndarray:
    return sufficient_statistics(input_index, outcome_index // 4, outcome_index % 4)


def _independent_reference_expectation(
    *, beta: float, wrong_parent: bool = False, propagate_reference: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Direct 8^4 loop, with switches representing two realistic implementation bugs."""

    fixture = replace(build_checked_fixture(), beta=beta)
    joint = equilibrium_joint_conditional(fixture.model_parameters, beta=beta)
    target = terminal_law(
        (fixture.target_conditional, fixture.target_conditional),
        fixture.occurrences,
        fixture.initial_state,
    )
    model = terminal_law((joint, joint), fixture.occurrences, fixture.initial_state)
    coefficient = 2.0 * (np.asarray(model.occupancy) - np.asarray(target.occupancy))
    contributions = [np.zeros(9, dtype=np.float64), np.zeros(9, dtype=np.float64)]
    input_0 = WORD_ORDER.index((1, 0))

    for main_0, main_1, reference_0, reference_1 in product(range(8), repeat=4):
        main_output_0 = WORD_ORDER[main_0 % 4]
        reference_output_0 = WORD_ORDER[reference_0 % 4]
        propagated_output = reference_output_0 if propagate_reference else main_output_0
        input_1 = WORD_ORDER.index((propagated_output[1], 0))
        recorded_input_1 = WORD_ORDER.index((main_output_0[1], 0))
        reference_input_1 = input_0 if wrong_parent else recorded_input_1
        terminal_output = WORD_ORDER[main_1 % 4]
        terminal_state = (propagated_output[0], *terminal_output)
        reward = float(coefficient @ np.asarray(terminal_state, dtype=np.float64))
        probability = (
            joint[input_0, main_0]
            * joint[input_1, main_1]
            * joint[input_0, reference_0]
            * joint[reference_input_1, reference_1]
        )
        contributions[0] += (
            probability
            * reward
            * beta
            * (_feature(input_0, main_0) - _feature(input_0, reference_0))
        )
        contributions[1] += (
            probability
            * reward
            * beta
            * (_feature(input_1, main_1) - _feature(reference_input_1, reference_1))
        )
    return contributions[0], contributions[1]


def test_exact_reference_matches_three_independent_gradient_oracles() -> None:
    reference = build_exact_reference(build_checked_fixture())

    assert reference.main_path_count == 64
    assert reference.augmented_path_count == 4096
    assert reference.objective == pytest.approx(0.5246570826850282, abs=1e-15)
    assert np.all(np.asarray(reference.score.shared) != 0.0)
    for occurrence in range(2):
        np.testing.assert_allclose(
            reference.score.occurrences[occurrence],
            reference.expected_reference.occurrences[occurrence],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            reference.score.occurrences[occurrence],
            reference.finite_difference.occurrences[occurrence],
            rtol=0.0,
            atol=1e-7,
        )
    np.testing.assert_allclose(
        reference.score.shared,
        np.asarray(reference.score.occurrences[0]) + np.asarray(reference.score.occurrences[1]),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        reference.finite_difference.shared,
        reference.finite_difference_tied,
        rtol=0.0,
        atol=1e-7,
    )
    assert reference.maximum_exact_error < 1e-12
    assert reference.maximum_finite_difference_error < 1e-7
    assert reference.accepted


def test_expected_reference_enumerates_same_parent_nonpropagated_draws() -> None:
    reference = build_exact_reference(build_checked_fixture())
    expected = _independent_reference_expectation(beta=1.0)
    wrong_parent = _independent_reference_expectation(beta=1.0, wrong_parent=True)
    propagated = _independent_reference_expectation(beta=1.0, propagate_reference=True)

    for occurrence in range(2):
        np.testing.assert_allclose(
            reference.expected_reference.occurrences[occurrence],
            expected[occurrence],
            rtol=1e-12,
            atol=1e-12,
        )
    assert not np.allclose(reference.expected_reference.shared, sum(wrong_parent), atol=1e-6)
    assert not np.allclose(reference.expected_reference.shared, sum(propagated), atol=1e-6)


def test_nonunit_beta_is_applied_to_scores_and_finite_differences() -> None:
    fixture = replace(build_checked_fixture(), beta=0.7)
    reference = build_exact_reference(fixture)
    expected = _independent_reference_expectation(beta=0.7)

    np.testing.assert_allclose(
        reference.expected_reference.shared, sum(expected), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        reference.score.shared,
        reference.finite_difference_tied,
        rtol=0.0,
        atol=1e-7,
    )


def test_exact_gradient_arrays_are_finite_float64_and_read_only() -> None:
    reference = build_exact_reference(build_checked_fixture())

    for gradients in (
        reference.score,
        reference.expected_reference,
        reference.finite_difference,
    ):
        assert gradients.shared.dtype == np.float64
        assert np.all(np.isfinite(gradients.shared))
        assert not gradients.shared.flags.writeable
        assert all(not occurrence.flags.writeable for occurrence in gradients.occurrences)
    assert not reference.finite_difference_tied.flags.writeable
