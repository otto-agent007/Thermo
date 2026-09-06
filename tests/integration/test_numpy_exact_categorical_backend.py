"""Contracts for the exact-categorical trajectory REINFORCE backend."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from thermo_lab.backends import numpy_exact_categorical as backend_module
from thermo_lab.backends.numpy_exact_categorical import (
    NumpyExactCategoricalBackend,
    sample_augmented_gradient_sources,
)
from thermo_lab.config import TRAJECTORY_REINFORCE_SAMPLE_DEFINITION
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.experiments.trajectory_reinforce_pasym_swap import (
    trajectory_reinforce_pasym_swap_spec,
)
from thermo_lab.pasym_swap import WORD_ORDER
from thermo_lab.thermodynamic_kernel import equilibrium_joint_conditional, sufficient_statistics
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference
from thermo_lab.trajectory_reinforce_results import validate_trajectory_reinforce_summary


def _inverse_cdf(rng: np.random.Generator, probabilities: np.ndarray) -> int:
    return int(np.searchsorted(np.cumsum(probabilities), rng.random(), side="right"))


def _scalar_sources(batch_size: int, seed: int) -> tuple[np.ndarray, ...]:
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    joint = equilibrium_joint_conditional(fixture.model_parameters, beta=fixture.beta)
    generators = tuple(
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)
    )
    sums = [np.zeros(9), np.zeros(9)]
    squares = [np.zeros(9), np.zeros(9)]
    cross = np.zeros(9)
    for _ in range(batch_size):
        state = [1, 0, 0]
        gradients = []
        main_outcomes = []
        parents = []
        references = []
        for occurrence, (left, right) in enumerate(fixture.occurrences):
            parent = WORD_ORDER.index((state[left], state[right]))
            main = _inverse_cdf(generators[2 * occurrence], joint[parent])
            reference = _inverse_cdf(generators[2 * occurrence + 1], joint[parent])
            parents.append(parent)
            main_outcomes.append(main)
            references.append(reference)
            state[left], state[right] = WORD_ORDER[main % 4]
        reward = float(exact.reward_coefficient @ np.asarray(state, dtype=np.float64))
        for occurrence in range(2):
            gradient = (
                reward
                * fixture.beta
                * (
                    sufficient_statistics(
                        parents[occurrence],
                        main_outcomes[occurrence] // 4,
                        main_outcomes[occurrence] % 4,
                    )
                    - sufficient_statistics(
                        parents[occurrence],
                        references[occurrence] // 4,
                        references[occurrence] % 4,
                    )
                )
            )
            gradients.append(gradient)
            sums[occurrence] += gradient
            squares[occurrence] += gradient * gradient
        cross += gradients[0] * gradients[1]
    return sums[0], squares[0], sums[1], squares[1], cross


def test_sampler_matches_scalar_main_propagation_and_same_parent_references() -> None:
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    expected = _scalar_sources(batch_size=17, seed=2)

    first = sample_augmented_gradient_sources(fixture=fixture, exact=exact, batch_size=17, seed=2)
    np.testing.assert_array_equal(first.occurrence_0.component_sum, expected[0])
    np.testing.assert_array_equal(first.occurrence_0.component_sum_squares, expected[1])
    np.testing.assert_array_equal(first.occurrence_1.component_sum, expected[2])
    np.testing.assert_array_equal(first.occurrence_1.component_sum_squares, expected[3])
    np.testing.assert_array_equal(first.cross_products, expected[4])
    assert set(vars(first)) == {"occurrence_0", "occurrence_1", "cross_products"}


def test_sampler_repeats_one_seed_and_separates_different_seeds() -> None:
    fixture = build_checked_fixture()
    exact = build_exact_reference(fixture)
    first = sample_augmented_gradient_sources(fixture=fixture, exact=exact, batch_size=17, seed=2)
    repeated = sample_augmented_gradient_sources(
        fixture=fixture, exact=exact, batch_size=17, seed=2
    )
    different = sample_augmented_gradient_sources(
        fixture=fixture, exact=exact, batch_size=17, seed=1
    )

    assert first == repeated
    assert first != different


def test_stream_factory_spawns_exactly_four_children_once_in_role_order() -> None:
    children = tuple(object() for _ in range(4))
    constructed_seeds: list[int] = []
    spawn_counts: list[int] = []
    generated_from: list[object] = []

    class RecordingSeedSequence:
        def __init__(self, seed: int) -> None:
            constructed_seeds.append(seed)

        def spawn(self, count: int) -> tuple[object, ...]:
            spawn_counts.append(count)
            return children

    def recording_generator(child: object) -> str:
        generated_from.append(child)
        return f"generator-{children.index(child)}"

    generators = backend_module._spawn_role_generators(
        7,
        seed_sequence_factory=RecordingSeedSequence,
        generator_factory=recording_generator,
    )

    assert constructed_seeds == [7]
    assert spawn_counts == [4]
    assert generated_from == list(children)
    assert generators == (
        "generator-0",
        "generator-1",
        "generator-2",
        "generator-3",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", True, "batch_size"),
        ("batch_size", 1, "batch_size"),
        ("seed", True, "seed"),
        ("seed", -1, "seed"),
    ],
)
def test_sampler_rejects_unchecked_requests(field: str, value: object, message: str) -> None:
    arguments = {
        "fixture": build_checked_fixture(),
        "exact": build_exact_reference(build_checked_fixture()),
        "batch_size": 2,
        "seed": 0,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        sample_augmented_gradient_sources(**arguments)


def test_backend_accepts_only_the_checked_request() -> None:
    backend = NumpyExactCategoricalBackend()
    model, run, request_hash = backend.checked_request(trajectory_reinforce_pasym_swap_spec())
    assert request_hash.startswith("sha256:")
    assert model.shared_parameter_vector == build_checked_fixture().model_parameters.values
    assert run.batch_size == 65536

    spec = trajectory_reinforce_pasym_swap_spec()
    payload = spec.model_dump(mode="python", by_alias=True)
    payload["run_config"] = deepcopy(dict(spec.run_parameters))
    payload["run_config"]["batch_size"] = 32768
    with pytest.raises(ValueError):
        backend.checked_request(type(spec).model_validate(payload))


def test_backend_full_checked_batch_emits_strict_non_gating_evidence() -> None:
    result = NumpyExactCategoricalBackend().execute(trajectory_reinforce_pasym_swap_spec())
    record = result.record
    summary = validate_trajectory_reinforce_summary(
        record.metrics["trajectory_reinforce_summary"].value
    )

    assert record.backend_id is BackendId.NUMPY_EXACT_CATEGORICAL
    assert result.diagnostic_series == {}
    assert record.evidence_class is EvidenceClass.SOFTWARE_SIMULATION
    assert set(record.metrics) == {
        "trajectory_reinforce_summary",
        "maximum_absolute_shared_gradient_error",
        "acceptance_passed",
    }
    assert summary.sample.occurrence_0.moments.sample_count == 65536
    assert summary.sample.sample_definition == TRAJECTORY_REINFORCE_SAMPLE_DEFINITION
    assert record.metrics["maximum_absolute_shared_gradient_error"].value == (
        summary.sample.maximum_absolute_shared_gradient_error
    )
    assert record.metrics["acceptance_passed"].value is summary.deterministic.accepted is True
    assert summary.acceptance_passed == summary.deterministic.accepted
    assert summary.sample.maximum_absolute_shared_gradient_error >= 0.0
    assert record.metrics["acceptance_passed"].evidence_class is EvidenceClass.EXACT_REFERENCE
    assert record.timing.compile_seconds == 0.0
    assert record.timing.synchronized
    assert "65,536 augmented trajectories" in record.timing.timing_method
    assert "four independent streams" in record.timing.timing_method
    assert "includes sampler setup and bounded-source model validation" in (
        record.timing.timing_method
    )
    assert (
        "excludes deterministic reference construction and final sampled-summary/run-record "
        "construction" in record.timing.timing_method
    )
    assert "does not use JAX, THRML, hosted simulation, or physical hardware" in (
        record.timing.timing_method
    )
    packages = {item.distribution: item.version for item in record.provenance.packages}
    assert set(packages) == {"numpy", "thermo-lab"}
    assert packages["numpy"] == np.__version__
    assert record.provenance.jax_backend == "not-used"
    assert record.provenance.jax_devices == ()
