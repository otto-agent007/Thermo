"""Exact three-site oracle for the trajectory-level REINFORCE estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import WORD_ORDER, build_pasym_swap_conditional, hop_probability
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    equilibrium_joint_conditional,
    sufficient_statistics,
)

VisibleState = tuple[int, int, int]
Occurrence = tuple[int, int]

VISIBLE_STATE_ORDER: tuple[VisibleState, ...] = tuple(product((0, 1), repeat=3))
CHECKED_PARAMETERS = (0.25, -0.35, 0.20, 0.45, -0.30, -0.40, 0.25, 0.30, -0.20)


def _readonly_float64(values: object, *, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    try:
        objects = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite real numbers") from error
    if any(isinstance(value, (bool, np.bool_)) for value in objects.flat):
        raise ValueError(f"{name} must not contain booleans")
    if any(not isinstance(value, Real) for value in objects.flat):
        raise ValueError(f"{name} must contain finite real numbers")
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite real numbers") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TrajectoryFixture:
    """Immutable scientific inputs for the checked three-site fixture."""

    initial_state: VisibleState
    occurrences: tuple[Occurrence, Occurrence]
    target_edge: tuple[tuple[int, int], tuple[int, int]]
    target_probabilities: tuple[float, float]
    target_conditional: NDArray[np.float64]
    target_hash: str
    model_parameters: KernelParameters
    beta: float
    parameter_cap: float
    finite_difference_step: float
    exact_tolerance: float
    finite_difference_tolerance: float

    def __post_init__(self) -> None:
        if self.initial_state != (1, 0, 0):
            raise ValueError("initial_state must be the checked three-site state")
        if self.occurrences != ((0, 1), (1, 2)):
            raise ValueError("occurrences must be the checked ordered pair")
        if self.target_edge != ((0, 0), (1, 0)):
            raise ValueError("target edge must be the checked oriented edge")
        if not isinstance(self.model_parameters, KernelParameters):
            raise TypeError("model_parameters must be KernelParameters")
        try:
            target_probabilities = tuple(self.target_probabilities)
        except TypeError as error:
            raise ValueError("target probabilities must contain exactly two values") from error
        if len(target_probabilities) != 2:
            raise ValueError("target probabilities must contain exactly two values")
        object.__setattr__(self, "target_probabilities", target_probabilities)
        numeric = (
            self.beta,
            self.parameter_cap,
            self.finite_difference_step,
            self.exact_tolerance,
            self.finite_difference_tolerance,
            *self.target_probabilities,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in numeric
        ):
            raise ValueError("fixture numeric values must be finite real numbers")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.parameter_cap <= 0.0:
            raise ValueError("parameter_cap must be positive")
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        if self.exact_tolerance < 0.0 or self.finite_difference_tolerance < 0.0:
            raise ValueError("gradient tolerances must be nonnegative")
        if any(
            abs(value) + self.finite_difference_step >= self.parameter_cap
            for value in self.model_parameters.values
        ):
            raise ValueError("finite differences must remain strictly inside the parameter cap")
        expected_probabilities = (
            hop_probability(*self.target_edge),
            hop_probability(*reversed(self.target_edge)),
        )
        if self.target_probabilities != expected_probabilities:
            raise ValueError("target edge probabilities disagree with the checked edge")
        object.__setattr__(
            self,
            "target_conditional",
            _readonly_float64(
                self.target_conditional, shape=(len(WORD_ORDER), len(WORD_ORDER)), name="target"
            ),
        )
        if np.any(self.target_conditional < 0.0) or not np.allclose(
            self.target_conditional.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("target conditional must be row-stochastic")
        expected_target = build_pasym_swap_conditional(*self.target_probabilities)
        if not np.array_equal(self.target_conditional, np.asarray(expected_target)):
            raise ValueError("target probabilities disagree with target conditional")
        expected_hash = canonical_sha256({"word_order": WORD_ORDER, "conditional": expected_target})
        if self.target_hash != expected_hash:
            raise ValueError("target hash disagrees with target conditional")


@dataclass(frozen=True)
class TerminalLaw:
    """Exact terminal distribution and diagnostics in canonical visible-state order."""

    probabilities: NDArray[np.float64]
    occupancy: tuple[float, float, float]
    expected_mass: float
    particle_number_leakage: float
    signed_mass_drift: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probabilities",
            _readonly_float64(self.probabilities, shape=(8,), name="terminal probabilities"),
        )
        scalars = (
            *self.occupancy,
            self.expected_mass,
            self.particle_number_leakage,
            self.signed_mass_drift,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in scalars
        ):
            raise ValueError("terminal diagnostics must be finite real numbers")


@dataclass(frozen=True)
class OccurrenceGradients:
    """Two occurrence-local vectors and their shared-parameter sum."""

    occurrences: tuple[NDArray[np.float64], NDArray[np.float64]]
    shared: NDArray[np.float64]

    def __post_init__(self) -> None:
        if type(self.occurrences) is not tuple or len(self.occurrences) != 2:
            raise ValueError("occurrence gradients must contain exactly two vectors")
        checked_occurrences = tuple(
            _readonly_float64(values, shape=(9,), name=f"occurrence {index} gradient")
            for index, values in enumerate(self.occurrences)
        )
        object.__setattr__(self, "occurrences", checked_occurrences)
        object.__setattr__(
            self, "shared", _readonly_float64(self.shared, shape=(9,), name="shared gradient")
        )


@dataclass(frozen=True)
class ExactTrajectoryReference:
    """All deterministic oracles and their checked comparison diagnostics."""

    target_law: TerminalLaw
    model_law: TerminalLaw
    objective: float
    reward_coefficient: NDArray[np.float64]
    score: OccurrenceGradients
    expected_reference: OccurrenceGradients
    finite_difference: OccurrenceGradients
    finite_difference_tied: NDArray[np.float64]
    exact_component_errors: OccurrenceGradients
    finite_difference_component_errors: OccurrenceGradients
    tied_finite_difference_error: NDArray[np.float64]
    maximum_exact_error: float
    maximum_finite_difference_error: float
    main_path_count: int
    augmented_path_count: int
    accepted: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reward_coefficient",
            _readonly_float64(self.reward_coefficient, shape=(3,), name="reward coefficient"),
        )
        object.__setattr__(
            self,
            "finite_difference_tied",
            _readonly_float64(
                self.finite_difference_tied, shape=(9,), name="tied finite difference"
            ),
        )
        object.__setattr__(
            self,
            "tied_finite_difference_error",
            _readonly_float64(
                self.tied_finite_difference_error,
                shape=(9,),
                name="tied finite-difference error",
            ),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in (
                self.objective,
                self.maximum_exact_error,
                self.maximum_finite_difference_error,
            )
        ):
            raise ValueError("exact reference scalars must be finite real numbers")
        if type(self.main_path_count) is not int or self.main_path_count <= 0:
            raise ValueError("main_path_count must be a positive integer")
        if type(self.augmented_path_count) is not int or self.augmented_path_count <= 0:
            raise ValueError("augmented_path_count must be a positive integer")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be boolean")


def build_checked_fixture() -> TrajectoryFixture:
    """Build the frozen, optimizer-independent three-site validation fixture."""

    edge = ((0, 0), (1, 0))
    p_ij = hop_probability(*edge)
    p_ji = hop_probability(*reversed(edge))
    conditional = build_pasym_swap_conditional(p_ij, p_ji)
    return TrajectoryFixture(
        initial_state=(1, 0, 0),
        occurrences=((0, 1), (1, 2)),
        target_edge=edge,
        target_probabilities=(p_ij, p_ji),
        target_conditional=np.asarray(conditional, dtype=np.float64),
        target_hash=canonical_sha256({"word_order": WORD_ORDER, "conditional": conditional}),
        model_parameters=KernelParameters(CHECKED_PARAMETERS),
        beta=1.0,
        parameter_cap=2.0,
        finite_difference_step=1e-6,
        exact_tolerance=1e-12,
        finite_difference_tolerance=1e-7,
    )


def terminal_law_from_probabilities(probabilities: object) -> TerminalLaw:
    """Derive terminal occupancy, leakage, and signed mass drift from a law."""

    checked = _readonly_float64(probabilities, shape=(8,), name="terminal probabilities")
    if np.any(checked < 0.0):
        raise ValueError("terminal probabilities must be nonnegative")
    if not math.isclose(
        math.fsum(float(value) for value in checked), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("terminal probabilities must sum to one")
    rows = tuple(zip(VISIBLE_STATE_ORDER, checked, strict=True))
    occupancy = tuple(
        math.fsum(float(probability) * state[site] for state, probability in rows)
        for site in range(3)
    )
    expected_mass = math.fsum(float(probability) * sum(state) for state, probability in rows)
    leakage = math.fsum(float(probability) for state, probability in rows if sum(state) != 1)
    return TerminalLaw(
        probabilities=checked,
        occupancy=occupancy,
        expected_mass=expected_mass,
        particle_number_leakage=leakage,
        signed_mass_drift=expected_mass - 1.0,
    )


def _checked_output_conditional(values: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.shape not in {(4, 4), (4, 8)}:
        raise ValueError(f"{name} must have shape (4, 4) or (4, 8)")
    source = _readonly_float64(values, shape=array.shape, name=name)
    if np.any(source < 0.0) or not np.allclose(source.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be row-stochastic")
    if source.shape == (4, 8):
        return _readonly_float64(source.reshape(4, 2, 4).sum(axis=1), shape=(4, 4), name=name)
    return source


def terminal_law(
    conditionals: tuple[object, object],
    occurrences: tuple[Occurrence, Occurrence] = ((0, 1), (1, 2)),
    initial_state: VisibleState = (1, 0, 0),
) -> TerminalLaw:
    """Compose two local conditionals, propagating only visible output bits."""

    if occurrences != ((0, 1), (1, 2)):
        raise ValueError("occurrences must be the canonical ordered pair")
    if (
        type(initial_state) is not tuple
        or len(initial_state) != 3
        or any(type(bit) is not int or bit not in (0, 1) for bit in initial_state)
    ):
        raise ValueError("initial_state must be a canonical three-bit state")
    checked = tuple(
        _checked_output_conditional(conditional, name=f"occurrence {index} conditional")
        for index, conditional in enumerate(conditionals)
    )
    distribution: dict[VisibleState, list[float]] = {initial_state: [1.0]}
    for occurrence, conditional in zip(occurrences, checked, strict=True):
        next_distribution: dict[VisibleState, list[float]] = {}
        left, right = occurrence
        for state in VISIBLE_STATE_ORDER:
            parent_terms = distribution.get(state)
            if parent_terms is None:
                continue
            parent_probability = math.fsum(parent_terms)
            input_index = WORD_ORDER.index((state[left], state[right]))
            for output_index, output in enumerate(WORD_ORDER):
                next_state_values = list(state)
                next_state_values[left], next_state_values[right] = output
                next_state = tuple(next_state_values)
                next_distribution.setdefault(next_state, []).append(
                    parent_probability * float(conditional[input_index, output_index])
                )
        distribution = next_distribution
    probabilities = np.asarray(
        [math.fsum(distribution.get(state, ())) for state in VISIBLE_STATE_ORDER],
        dtype=np.float64,
    )
    return terminal_law_from_probabilities(probabilities)


def _componentwise_fsum(vectors: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
    return np.asarray(
        [math.fsum(float(vector[index]) for vector in vectors) for index in range(9)],
        dtype=np.float64,
    )


def _outcome_feature(input_index: int, outcome_index: int) -> NDArray[np.float64]:
    return sufficient_statistics(input_index, outcome_index // 4, outcome_index % 4)


def _main_trajectory_rows(
    joint: NDArray[np.float64], reward_coefficient: NDArray[np.float64]
) -> tuple[tuple[int, int, int, float, float], ...]:
    """Return ``(outcome_0, parent_1, outcome_1, probability, reward)`` rows."""

    rows: list[tuple[int, int, int, float, float]] = []
    input_0 = WORD_ORDER.index((1, 0))
    for outcome_0, outcome_1 in product(range(8), repeat=2):
        first_output = WORD_ORDER[outcome_0 % 4]
        input_1 = WORD_ORDER.index((first_output[1], 0))
        second_output = WORD_ORDER[outcome_1 % 4]
        terminal_state = np.asarray((first_output[0], *second_output), dtype=np.float64)
        rows.append(
            (
                outcome_0,
                input_1,
                outcome_1,
                float(joint[input_0, outcome_0] * joint[input_1, outcome_1]),
                float(reward_coefficient @ terminal_state),
            )
        )
    return tuple(rows)


def _score_gradients(
    joint: NDArray[np.float64],
    reward_coefficient: NDArray[np.float64],
    *,
    beta: float,
) -> OccurrenceGradients:
    input_0 = WORD_ORDER.index((1, 0))
    expected_features = tuple(
        np.asarray(
            [
                math.fsum(
                    float(joint[input_index, outcome])
                    * float(_outcome_feature(input_index, outcome)[component])
                    for outcome in range(8)
                )
                for component in range(9)
            ],
            dtype=np.float64,
        )
        for input_index in range(4)
    )
    terms: tuple[list[list[float]], list[list[float]]] = (
        [[] for _ in range(9)],
        [[] for _ in range(9)],
    )
    for outcome_0, input_1, outcome_1, probability, reward in _main_trajectory_rows(
        joint, reward_coefficient
    ):
        scores = (
            beta * (_outcome_feature(input_0, outcome_0) - expected_features[input_0]),
            beta * (_outcome_feature(input_1, outcome_1) - expected_features[input_1]),
        )
        for occurrence in range(2):
            for component in range(9):
                terms[occurrence][component].append(
                    probability * reward * float(scores[occurrence][component])
                )
    occurrences = tuple(
        np.asarray([math.fsum(component) for component in occurrence], dtype=np.float64)
        for occurrence in terms
    )
    return OccurrenceGradients(
        occurrences=occurrences,  # type: ignore[arg-type]
        shared=_componentwise_fsum(occurrences),
    )


def _reference_gradients(
    joint: NDArray[np.float64],
    reward_coefficient: NDArray[np.float64],
    *,
    beta: float,
) -> OccurrenceGradients:
    """Enumerate the actual independent same-parent reference outcomes."""

    input_0 = WORD_ORDER.index((1, 0))
    terms: tuple[list[list[float]], list[list[float]]] = (
        [[] for _ in range(9)],
        [[] for _ in range(9)],
    )
    for outcome_0, input_1, outcome_1, main_probability, reward in _main_trajectory_rows(
        joint, reward_coefficient
    ):
        main_features = (
            _outcome_feature(input_0, outcome_0),
            _outcome_feature(input_1, outcome_1),
        )
        for reference_0, reference_1 in product(range(8), repeat=2):
            probability = (
                main_probability
                * float(joint[input_0, reference_0])
                * float(joint[input_1, reference_1])
            )
            reference_features = (
                _outcome_feature(input_0, reference_0),
                _outcome_feature(input_1, reference_1),
            )
            for occurrence in range(2):
                difference = beta * (main_features[occurrence] - reference_features[occurrence])
                for component in range(9):
                    terms[occurrence][component].append(
                        probability * reward * float(difference[component])
                    )
    occurrences = tuple(
        np.asarray([math.fsum(component) for component in occurrence], dtype=np.float64)
        for occurrence in terms
    )
    return OccurrenceGradients(
        occurrences=occurrences,  # type: ignore[arg-type]
        shared=_componentwise_fsum(occurrences),
    )


def _objective_for_parameters(
    occurrence_parameters: tuple[KernelParameters, KernelParameters],
    *,
    target_occupancy: tuple[float, float, float],
    beta: float,
) -> float:
    conditionals = tuple(
        equilibrium_joint_conditional(parameters, beta=beta) for parameters in occurrence_parameters
    )
    model = terminal_law(conditionals)  # type: ignore[arg-type]
    return math.fsum((model.occupancy[index] - target_occupancy[index]) ** 2 for index in range(3))


def _finite_difference_gradients(
    fixture: TrajectoryFixture, target_occupancy: tuple[float, float, float]
) -> tuple[OccurrenceGradients, NDArray[np.float64]]:
    values = fixture.model_parameters.values
    step = fixture.finite_difference_step
    occurrence_vectors: list[NDArray[np.float64]] = []
    for occurrence in range(2):
        components: list[float] = []
        for component in range(9):
            plus = list(values)
            minus = list(values)
            plus[component] += step
            minus[component] -= step
            plus_parameters = [fixture.model_parameters, fixture.model_parameters]
            minus_parameters = [fixture.model_parameters, fixture.model_parameters]
            plus_parameters[occurrence] = KernelParameters(tuple(plus))  # type: ignore[arg-type]
            minus_parameters[occurrence] = KernelParameters(tuple(minus))  # type: ignore[arg-type]
            plus_objective = _objective_for_parameters(
                tuple(plus_parameters),  # type: ignore[arg-type]
                target_occupancy=target_occupancy,
                beta=fixture.beta,
            )
            minus_objective = _objective_for_parameters(
                tuple(minus_parameters),  # type: ignore[arg-type]
                target_occupancy=target_occupancy,
                beta=fixture.beta,
            )
            components.append((plus_objective - minus_objective) / (2.0 * step))
        occurrence_vectors.append(np.asarray(components, dtype=np.float64))

    tied_components: list[float] = []
    for component in range(9):
        plus = list(values)
        minus = list(values)
        plus[component] += step
        minus[component] -= step
        plus_parameters = KernelParameters(tuple(plus))  # type: ignore[arg-type]
        minus_parameters = KernelParameters(tuple(minus))  # type: ignore[arg-type]
        plus_objective = _objective_for_parameters(
            (plus_parameters, plus_parameters),
            target_occupancy=target_occupancy,
            beta=fixture.beta,
        )
        minus_objective = _objective_for_parameters(
            (minus_parameters, minus_parameters),
            target_occupancy=target_occupancy,
            beta=fixture.beta,
        )
        tied_components.append((plus_objective - minus_objective) / (2.0 * step))

    occurrences = tuple(occurrence_vectors)
    return (
        OccurrenceGradients(
            occurrences=occurrences,  # type: ignore[arg-type]
            shared=_componentwise_fsum(occurrences),
        ),
        np.asarray(tied_components, dtype=np.float64),
    )


def _absolute_errors(
    actual: OccurrenceGradients, expected: OccurrenceGradients
) -> OccurrenceGradients:
    occurrences = tuple(
        np.abs(actual.occurrences[index] - expected.occurrences[index]) for index in range(2)
    )
    return OccurrenceGradients(
        occurrences=occurrences,  # type: ignore[arg-type]
        shared=np.abs(actual.shared - expected.shared),
    )


def build_exact_reference(
    fixture: TrajectoryFixture | None = None,
) -> ExactTrajectoryReference:
    """Build score, independent-reference, and finite-difference gradient oracles."""

    checked_fixture = fixture or build_checked_fixture()
    if not isinstance(checked_fixture, TrajectoryFixture):
        raise TypeError("fixture must be TrajectoryFixture")
    joint = equilibrium_joint_conditional(
        checked_fixture.model_parameters, beta=checked_fixture.beta
    )
    target = terminal_law(
        (checked_fixture.target_conditional, checked_fixture.target_conditional),
        checked_fixture.occurrences,
        checked_fixture.initial_state,
    )
    model = terminal_law((joint, joint), checked_fixture.occurrences, checked_fixture.initial_state)
    reward_coefficient = 2.0 * (
        np.asarray(model.occupancy, dtype=np.float64)
        - np.asarray(target.occupancy, dtype=np.float64)
    )
    objective = math.fsum(
        (model.occupancy[index] - target.occupancy[index]) ** 2 for index in range(3)
    )
    score = _score_gradients(joint, reward_coefficient, beta=checked_fixture.beta)
    expected_reference = _reference_gradients(joint, reward_coefficient, beta=checked_fixture.beta)
    finite_difference, tied = _finite_difference_gradients(checked_fixture, target.occupancy)
    exact_errors = _absolute_errors(score, expected_reference)
    finite_errors = _absolute_errors(score, finite_difference)
    tied_error = np.abs(score.shared - tied)
    maximum_exact_error = max(
        float(np.max(exact_errors.shared)),
        *(float(np.max(vector)) for vector in exact_errors.occurrences),
    )
    maximum_finite_difference_error = max(
        float(np.max(finite_errors.shared)),
        float(np.max(tied_error)),
        *(float(np.max(vector)) for vector in finite_errors.occurrences),
    )
    accepted = (
        maximum_exact_error <= checked_fixture.exact_tolerance
        and maximum_finite_difference_error <= checked_fixture.finite_difference_tolerance
    )
    return ExactTrajectoryReference(
        target_law=target,
        model_law=model,
        objective=objective,
        reward_coefficient=reward_coefficient,
        score=score,
        expected_reference=expected_reference,
        finite_difference=finite_difference,
        finite_difference_tied=tied,
        exact_component_errors=exact_errors,
        finite_difference_component_errors=finite_errors,
        tied_finite_difference_error=tied_error,
        maximum_exact_error=maximum_exact_error,
        maximum_finite_difference_error=maximum_finite_difference_error,
        main_path_count=8**2,
        augmented_path_count=8**4,
        accepted=accepted,
    )
