"""Independent exact checks for finite-sweep gradients of the three-site circuit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from numpy.typing import NDArray

from thermo_lab.finite_sweep_gradients import FiniteSweepJointLaw, finite_sweep_joint_law
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    finite_horizon_conditional,
    sufficient_statistics,
)
from thermo_lab.trajectory_reinforce import (
    VISIBLE_STATE_ORDER,
    OccurrenceGradients,
    TerminalLaw,
    TrajectoryFixture,
    _readonly_float64,
    build_checked_fixture,
    terminal_law,
)

CHECKED_HORIZONS = (1, 2, 4, 8, 16, 30)


def _gradients(values: object) -> OccurrenceGradients:
    matrix = np.asarray(values, dtype=np.float64)
    return OccurrenceGradients((matrix[0], matrix[1]), matrix.sum(axis=0))


@dataclass(frozen=True)
class FiniteSweepGradientReference:
    horizon: int
    target_law: TerminalLaw
    model_law: TerminalLaw
    objective: float
    score: OccurrenceGradients
    chain_rule: OccurrenceGradients
    autodiff: OccurrenceGradients
    finite_difference: OccurrenceGradients
    finite_difference_tied: NDArray[np.float64]
    equilibrium_form_score: OccurrenceGradients
    maximum_exact_error: float
    maximum_finite_difference_error: float
    equilibrium_score_substitution_error: float
    accepted: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "finite_difference_tied",
            _readonly_float64(self.finite_difference_tied, shape=(9,), name="tied derivative"),
        )


def _score_expectation(
    joint: NDArray[np.float64], scores: NDArray[np.float64], reward: NDArray[np.float64]
) -> OccurrenceGradients:
    terms = [[], []]
    for first, second in product(range(8), repeat=2):
        first_left, first_right = divmod(first % 4, 2)
        second_left, second_right = divmod(second % 4, 2)
        parent = 2 * first_right
        probability = joint[2, first] * joint[parent, second]
        value = reward @ np.asarray((first_left, second_left, second_right))
        terms[0].append(probability * value * scores[2, first])
        terms[1].append(probability * value * scores[parent, second])
    return _gradients(
        [[math.fsum(float(row[i]) for row in terms[k]) for i in range(9)] for k in range(2)]
    )


def _terminal_chain_rule(
    law: FiniteSweepJointLaw, reward: NDArray[np.float64]
) -> OccurrenceGradients:
    probabilities = law.probabilities.reshape(4, 2, 4).sum(axis=1)
    derivative = law.jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    terminal_jacobian = np.zeros((8, 2, 9), dtype=np.float64)
    for first, second in product(range(4), repeat=2):
        left, middle = divmod(first, 2)
        parent = 2 * middle
        terminal = 4 * left + second
        terminal_jacobian[terminal, 0] += derivative[2, first] * probabilities[parent, second]
        terminal_jacobian[terminal, 1] += probabilities[2, first] * derivative[parent, second]
    terminal_reward = np.asarray(VISIBLE_STATE_ORDER) @ reward
    return _gradients(np.einsum("s,ski->ki", terminal_reward, terminal_jacobian))


def _independent_autodiff(
    values: tuple[float, ...], horizon: int, beta: float, target: tuple[float, ...]
) -> tuple[OccurrenceGradients, NDArray[np.float64]]:
    """Differentiate individual Bernoulli spin updates and full visible matrices.

    This reference uses neither the production transition/derivative builders
    nor sufficient-statistic scores. Precision changes are scoped and restored.
    """
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        words = jnp.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=jnp.float64)
        spins = 2 * words - 1
        hidden_spins = jnp.asarray((-1.0, 1.0), dtype=jnp.float64)
        states = jnp.asarray(VISIBLE_STATE_ORDER, dtype=jnp.int32)
        joint_hidden = jnp.arange(8) // 4
        joint_output = jnp.arange(8) % 4

        def endpoint(parameters):
            hidden_field = parameters[0] + spins @ parameters[7:9]
            hidden_p = jax.nn.sigmoid(2 * beta * hidden_field[:, None] * hidden_spins[None, :])
            output_fields = (
                parameters[1:3][None, None, :]
                + (spins @ parameters[3:7].reshape(2, 2))[:, None, :]
                + hidden_spins[None, :, None] * parameters[7:9][None, None, :]
            )
            output_p = jax.nn.sigmoid(
                2 * beta * output_fields[:, :, None, :] * spins[None, None, :, :]
            ).prod(axis=-1)
            transition = (
                hidden_p[joint_output[:, None], joint_hidden[None, :]][None, :, :]
                * output_p[:, joint_hidden, joint_output][:, None, :]
            )
            distribution = jax.lax.fori_loop(
                0,
                horizon,
                lambda _, p: jnp.einsum("pi,pij->pj", p, transition),
                jnp.full((4, 8), 1.0 / 8.0, dtype=jnp.float64),
            )
            return distribution.reshape(4, 2, 4).sum(axis=1)

        def loss_and_law(parameters):
            tables = jax.vmap(endpoint)(parameters)
            distribution = jax.nn.one_hot(4, 8, dtype=jnp.float64)
            for index, (left, right, untouched) in enumerate(((0, 1, 2), (1, 2, 0))):
                parents = 2 * states[:, left] + states[:, right]
                outputs = 2 * states[:, left] + states[:, right]
                visible_transition = tables[index][parents[:, None], outputs[None, :]] * (
                    states[:, untouched, None] == states[None, :, untouched]
                )
                distribution = distribution @ visible_transition
            residual = distribution @ states - jnp.asarray(target, dtype=jnp.float64)
            return jnp.sum(residual * residual), distribution

        gradient, probabilities = jax.jacfwd(loss_and_law, has_aux=True)(
            jnp.asarray((values, values), dtype=jnp.float64)
        )
        return _gradients(np.asarray(gradient)), np.asarray(probabilities)


@lru_cache(maxsize=128)
def _existing_conditionals(
    values: tuple[float, ...], beta: float
) -> dict[int, NDArray[np.float64]]:
    return finite_horizon_conditional(KernelParameters(values), CHECKED_HORIZONS, beta)


def _finite_differences(
    fixture: TrajectoryFixture, horizon: int, target: tuple[float, ...]
) -> tuple[OccurrenceGradients, NDArray[np.float64]]:
    def objective(parameters):
        tables = tuple(
            _existing_conditionals(tuple(row), fixture.beta)[horizon] for row in parameters
        )
        law = terminal_law(tables)
        return math.fsum((x - y) ** 2 for x, y in zip(law.occupancy, target, strict=True))

    values = np.asarray((fixture.model_parameters.values,) * 2, dtype=np.float64)
    step = fixture.finite_difference_step
    local = np.empty((2, 9), dtype=np.float64)
    tied = np.empty(9, dtype=np.float64)
    for component in range(9):
        for occurrence in range(2):
            delta = np.zeros((2, 9))
            delta[occurrence, component] = step
            local[occurrence, component] = (
                objective(values + delta) - objective(values - delta)
            ) / (2 * step)
        delta = np.zeros((2, 9))
        delta[:, component] = step
        tied[component] = (objective(values + delta) - objective(values - delta)) / (2 * step)
    return _gradients(local), tied


def _gradient_error(left: OccurrenceGradients, right: OccurrenceGradients) -> float:
    return max(
        float(np.max(np.abs(left.shared - right.shared))),
        *(
            float(np.max(np.abs(a - b)))
            for a, b in zip(left.occurrences, right.occurrences, strict=True)
        ),
    )


def build_finite_sweep_reference(
    horizon: int, fixture: TrajectoryFixture | None = None
) -> FiniteSweepGradientReference:
    if type(horizon) is not int or horizon not in CHECKED_HORIZONS:
        raise ValueError("reference horizon must be one of 1, 2, 4, 8, 16, 30")
    checked = build_checked_fixture() if fixture is None else fixture
    if not isinstance(checked, TrajectoryFixture):
        raise TypeError("fixture must be TrajectoryFixture")
    law = finite_sweep_joint_law(checked.model_parameters, horizon, beta=checked.beta)
    target = terminal_law((checked.target_conditional,) * 2)
    model = terminal_law((law.probabilities,) * 2)
    reward = 2 * (np.asarray(model.occupancy) - np.asarray(target.occupancy))
    objective = math.fsum(
        (a - b) ** 2 for a, b in zip(model.occupancy, target.occupancy, strict=True)
    )
    score = _score_expectation(law.probabilities, law.scores, reward)
    chain_rule = _terminal_chain_rule(law, reward)
    autodiff, autodiff_law = _independent_autodiff(
        checked.model_parameters.values, horizon, checked.beta, target.occupancy
    )
    finite_difference, tied = _finite_differences(checked, horizon, target.occupancy)
    features = np.asarray(
        [
            [sufficient_statistics(parent, outcome // 4, outcome % 4) for outcome in range(8)]
            for parent in range(4)
        ]
    )
    equilibrium_form = checked.beta * (
        features - np.einsum("po,poi->pi", law.probabilities, features)[:, None, :]
    )
    wrong = _score_expectation(law.probabilities, equilibrium_form, reward)
    existing_law = terminal_law(
        (_existing_conditionals(checked.model_parameters.values, checked.beta)[horizon],) * 2
    )
    exact_error = max(
        _gradient_error(score, chain_rule),
        _gradient_error(score, autodiff),
        float(np.max(np.abs(model.probabilities - autodiff_law))),
        float(np.max(np.abs(model.probabilities - existing_law.probabilities))),
        float(np.max(np.abs(law.jacobian.sum(axis=1)))),
    )
    finite_error = max(
        _gradient_error(score, finite_difference), float(np.max(np.abs(score.shared - tied)))
    )
    return FiniteSweepGradientReference(
        horizon,
        target,
        model,
        objective,
        score,
        chain_rule,
        autodiff,
        finite_difference,
        tied,
        wrong,
        exact_error,
        finite_error,
        _gradient_error(score, wrong),
        exact_error <= checked.exact_tolerance
        and finite_error <= checked.finite_difference_tolerance,
    )
