import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256
from thermo_lab.trajectory_reinforce import build_checked_fixture, terminal_law


@pytest.fixture(scope="module")
def sequence():
    from thermo_lab.finite_refinement_sequence import run_training_sequence

    fixture = build_checked_fixture()
    target = terminal_law((fixture.target_conditional,) * 2).occupancy
    return run_training_sequence(
        (fixture.model_parameters.values,), (0, 0), fixture.occurrences, target, seed=0
    )


def test_five_updates_and_final_evaluation_reconstruct(sequence):
    from thermo_lab.finite_refinement_sequence import FiniteTrainingSequence

    assert len(sequence.steps) == sequence.protocol.step_count == 5
    assert sequence.protocol.horizon == 4
    assert sequence.protocol.checkpoint_policy == "always_fifth_update"
    assert sequence.final_evaluation.sample_count == 32768
    current = np.asarray(sequence.initial_parameters)
    for iteration, step in enumerate(sequence.steps, 1):
        assert step.iteration == iteration
        assert step.parameter_digest == canonical_sha256(
            tuple(tuple(float(x) for x in row) for row in current)
        )
        expected_raw = current - 0.01 * np.asarray(step.gradient.mean)
        np.testing.assert_array_equal(step.update.raw_parameters, expected_raw)
        current = np.clip(expected_raw, -2.0, 2.0)
        np.testing.assert_array_equal(step.update.updated_parameters, current)
        assert step.update.bounds_satisfied
    assert FiniteTrainingSequence.model_validate_json(sequence.model_dump_json()) == sequence


def test_all_training_roles_and_final_evaluation_are_disjoint(sequence):
    from thermo_lab.finite_refinement_sequence import spawn_finite_refinement_roles

    seeds = [
        value for step in sequence.steps for value in (step.occupancy_seed, step.gradient_seed)
    ]
    seeds.append(sequence.evaluation_seed)
    assert tuple(seeds) == spawn_finite_refinement_roles(0)
    all_study_roles = [s for seed in (0, 1, 2) for s in spawn_finite_refinement_roles(seed)]
    assert len(set(all_study_roles)) == 33


@pytest.mark.parametrize(
    "path,value",
    [
        (("protocol", "step_count"), 4),
        (("protocol", "horizon"), 1),
        (("protocol", "learning_rate"), 0.02),
        (("protocol", "batch_size"), 1024),
        (("evaluation_seed",), 0),
        (("seed",), True),
        (("steps", 0, "occupancy_seed"), 0),
        (("steps", 0, "gradient_seed"), 0),
        (("steps", 0, "reward_coefficient", 0), 4.0),
    ],
)
def test_rehashed_protocol_and_role_tampering_is_rejected(sequence, path, value):
    from thermo_lab.finite_refinement_sequence import FiniteTrainingSequence, sequence_result_digest

    payload = sequence.model_dump(mode="json")
    pointer = payload
    for part in path[:-1]:
        pointer = pointer[part]
    pointer[path[-1]] = value
    payload["result_digest"] = sequence_result_digest(payload)
    with pytest.raises(ValueError):
        FiniteTrainingSequence.model_validate(payload)


def test_gradient_replay_rejects_a_rehashed_forged_source(sequence):
    from thermo_lab.finite_refinement_sequence import FiniteTrainingSequence, sequence_result_digest
    from thermo_lab.finite_sweep_sampling import gradient_source_digest

    payload = sequence.model_dump(mode="json")
    step = payload["steps"][0]
    step["gradient"]["component_sum"][0][0] += 0.01
    step["gradient"]["source_digest"] = gradient_source_digest(
        sequence.initial_parameters,
        sequence.occurrence_target_indices,
        sequence.occurrence_site_indices,
        step["reward_coefficient"],
        step["gradient"]["component_sum"],
        step["gradient"]["component_sum_squares"],
        site_count=len(sequence.target_occupancy),
        batch_size=32768,
        seed=step["gradient_seed"],
        beta=1.0,
        horizon=4,
    )
    payload["result_digest"] = sequence_result_digest(payload)
    with pytest.raises(ValueError, match="gradient.*replay"):
        FiniteTrainingSequence.model_validate(payload)


def test_sequence_cannot_drop_or_reorder_checkpoints(sequence):
    from thermo_lab.finite_refinement_sequence import FiniteTrainingSequence, sequence_result_digest

    for steps in (sequence.steps[:-1], tuple(reversed(sequence.steps))):
        forged = sequence.model_copy(update={"steps": steps})
        payload = forged.model_dump(mode="json")
        payload["result_digest"] = sequence_result_digest(payload)
        with pytest.raises(ValueError):
            FiniteTrainingSequence.model_validate(payload)
