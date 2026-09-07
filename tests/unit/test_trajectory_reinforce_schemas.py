from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from thermo_lab.config import (
    TRAJECTORY_REINFORCE_EXPERIMENT_ID,
    TRAJECTORY_REINFORCE_SAMPLE_DEFINITION,
    ExperimentConfig,
    dump_experiment_config,
    load_experiment_config,
    trajectory_reinforce_non_seed_config_hash,
)
from thermo_lab.evidence import BackendId
from thermo_lab.experiments.trajectory_reinforce_pasym_swap import (
    trajectory_reinforce_pasym_swap_spec,
)
from thermo_lab.hashing import to_json_value
from thermo_lab.schemas import (
    PARAMETER_ORDER,
    TrajectoryReinforceModelConfig,
    TrajectoryReinforceRunConfig,
    validate_trajectory_reinforce_request,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml"

EXPECTED_SAMPLE_DEFINITION = (
    "One independently seeded batch of 65,536 augmented two-occurrence trajectories; "
    "each sample contains two propagated main exact-categorical joint-kernel draws and "
    "one independent same-parent non-propagated reference draw per occurrence."
)
EXPECTED_MODEL = {
    "source_reference": "https://arxiv.org/abs/2608.01615v2",
    "site_order": ["site_0", "site_1", "site_2"],
    "word_order": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "role_order": ["input_0", "input_1", "hidden_0", "output_0", "output_1"],
    "joint_outcome_order": [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    "parameter_order": list(PARAMETER_ORDER),
    "shared_parameter_vector": [
        0.25,
        -0.35,
        0.20,
        0.45,
        -0.30,
        -0.40,
        0.25,
        0.30,
        -0.20,
    ],
    "beta": 1.0,
    "parameter_cap": 2.0,
    "exact_dtype": "float64",
    "target_edge": [[0, 0], [1, 0]],
    "target_probabilities": [0.009628878136877513, 0.0903711218631225],
    "target_conditional": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.9096288781368775, 0.0903711218631225, 0.0],
        [0.0, 0.009628878136877513, 0.9903711218631225, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "target_hash": "sha256:7230c4f09bb3a6de72f8cd59b27d0719895e10a564bbab71a914aab370fc8a85",
}
EXPECTED_RUN = {
    "initial_state": [1, 0, 0],
    "occurrences": [[0, 1], [1, 2]],
    "objective_policy": "squared_terminal_occupancy_error",
    "reward_policy": "exact_current_model_occupancy_linearization",
    "reference_policy": "independent_same_parent_non_propagated",
    "main_path_count": 64,
    "augmented_path_count": 4096,
    "finite_difference_step": 1e-6,
    "exact_tolerance": 1e-12,
    "finite_difference_tolerance": 1e-7,
    "batch_size": 65536,
    "release_seeds": [0, 1, 2],
    "rng_family": "numpy.random.Generator(PCG64)",
    "stream_policy": (
        "SeedSequence(seed).spawn(4) in main_0, reference_0, main_1, reference_1 order"
    ),
    "moment_policy": "component sums, sum squares, and occurrence cross-products in float64",
}


def _payload() -> dict[str, object]:
    configured = load_experiment_config(CONFIG)
    return configured.model_dump(mode="python", by_alias=True)


def test_checked_request_has_exact_identity_and_fixture_literals() -> None:
    configured = load_experiment_config(CONFIG)

    assert TRAJECTORY_REINFORCE_EXPERIMENT_ID == (
        "numpy.trajectory_reinforce_pasym_swap_estimator.v1"
    )
    assert TRAJECTORY_REINFORCE_SAMPLE_DEFINITION == EXPECTED_SAMPLE_DEFINITION
    assert configured.backend is BackendId.NUMPY_EXACT_CATEGORICAL
    assert configured.experiment_id == TRAJECTORY_REINFORCE_EXPERIMENT_ID
    assert configured.sample_definition == EXPECTED_SAMPLE_DEFINITION
    assert configured.seed == 0
    assert to_json_value(configured.model_parameters) == EXPECTED_MODEL
    assert to_json_value(configured.run_parameters) == EXPECTED_RUN


def test_typed_request_round_trips_and_rebuilds_target_identity() -> None:
    configured = load_experiment_config(CONFIG)
    model = TrajectoryReinforceModelConfig.model_validate(EXPECTED_MODEL)
    run = TrajectoryReinforceRunConfig.model_validate(EXPECTED_RUN)

    validate_trajectory_reinforce_request(model, run, configured.seed)

    assert model.model_dump(mode="json") == EXPECTED_MODEL
    assert run.model_dump(mode="json") == EXPECTED_RUN
    assert trajectory_reinforce_non_seed_config_hash(model, run) == configured.non_seed_config_hash


def test_seed_is_excluded_from_non_seed_identity() -> None:
    configured = load_experiment_config(CONFIG)

    assert configured.with_overrides(seed=1).non_seed_config_hash == configured.non_seed_config_hash
    assert configured.with_overrides(seed=1).to_spec().seed == 1


def test_public_validator_and_hash_revalidate_constructed_models() -> None:
    model = TrajectoryReinforceModelConfig.model_validate(EXPECTED_MODEL)
    run = TrajectoryReinforceRunConfig.model_validate(EXPECTED_RUN)
    invalid_model = model.model_copy(update={"beta": 0.9})
    invalid_run = run.model_copy(update={"batch_size": 32768})

    with pytest.raises(ValidationError):
        validate_trajectory_reinforce_request(invalid_model, run, 0)
    with pytest.raises(ValidationError):
        validate_trajectory_reinforce_request(model, invalid_run, 0)
    with pytest.raises(ValidationError):
        trajectory_reinforce_non_seed_config_hash(invalid_model, run)
    with pytest.raises(ValidationError):
        trajectory_reinforce_non_seed_config_hash(model, invalid_run)


@pytest.mark.parametrize("field", ["main_path_count", "augmented_path_count", "batch_size"])
@pytest.mark.parametrize("invalid_type", [float, bool])
def test_integer_literals_reject_numerically_equal_non_integer_encodings(
    field: str, invalid_type: type[float] | type[bool]
) -> None:
    run_payload = deepcopy(EXPECTED_RUN)
    run_payload[field] = invalid_type(run_payload[field])

    with pytest.raises(ValidationError):
        TrajectoryReinforceRunConfig.model_validate(run_payload)


@pytest.mark.parametrize("field", ["main_path_count", "augmented_path_count", "batch_size"])
@pytest.mark.parametrize("invalid_type", [float, bool])
def test_public_validator_rejects_unchecked_non_integer_literal_copies(
    field: str, invalid_type: type[float] | type[bool]
) -> None:
    model = TrajectoryReinforceModelConfig.model_validate(EXPECTED_MODEL)
    run = TrajectoryReinforceRunConfig.model_validate(EXPECTED_RUN)
    invalid_run = run.model_copy(update={field: invalid_type(getattr(run, field))})

    with pytest.raises(ValidationError):
        validate_trajectory_reinforce_request(model, invalid_run, 0)
    with pytest.raises(ValidationError):
        trajectory_reinforce_non_seed_config_hash(model, invalid_run)


def test_config_rejects_approximate_sample_definition() -> None:
    payload = _payload()
    payload["sample_definition"] = "one approximate trajectory batch"

    with pytest.raises(ValidationError, match="sample_definition"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("model", "site_order", ["site_1", "site_0", "site_2"]),
        ("run", "occurrences", [[1, 2], [0, 1]]),
        ("model", "target_edge", [[1, 0], [0, 0]]),
        ("model", "parameter_order", list(reversed(PARAMETER_ORDER))),
        (
            "model",
            "shared_parameter_vector",
            [0.24] + EXPECTED_MODEL["shared_parameter_vector"][1:],
        ),
        ("model", "beta", 0.99),
        ("model", "parameter_cap", 2.1),
        ("run", "exact_tolerance", 2e-12),
        ("run", "finite_difference_step", 2e-6),
        ("run", "finite_difference_tolerance", 2e-7),
        ("run", "batch_size", 32768),
        ("run", "rng_family", "numpy.random.RandomState(MT19937)"),
        ("run", "stream_policy", "one shared stream"),
        ("run", "reward_policy", "same_batch_occupancy_linearization"),
    ],
)
def test_non_seed_identity_includes_each_scientific_field(
    section: str, field: str, replacement: object
) -> None:
    configured = load_experiment_config(CONFIG)
    model = deepcopy(EXPECTED_MODEL)
    run = deepcopy(EXPECTED_RUN)
    scientific = model if section == "model" else run
    scientific[field] = replacement
    unchecked = configured.model_copy(update={"model_parameters": model, "run_parameters": run})

    assert unchecked.non_seed_config_hash != configured.non_seed_config_hash


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("model", "source_reference", "https://arxiv.org/abs/2608.01615v1"),
        ("model", "site_order", ["site_1", "site_0", "site_2"]),
        ("model", "word_order", [[0, 1], [0, 0], [1, 0], [1, 1]]),
        (
            "model",
            "role_order",
            ["input_1", "input_0", "hidden_0", "output_0", "output_1"],
        ),
        (
            "model",
            "joint_outcome_order",
            list(reversed(EXPECTED_MODEL["joint_outcome_order"])),
        ),
        ("run", "occurrences", [[1, 2], [0, 1]]),
        ("run", "initial_state", [0, 1, 0]),
        ("model", "target_edge", [[1, 0], [0, 0]]),
        ("model", "parameter_order", list(reversed(PARAMETER_ORDER))),
        (
            "model",
            "shared_parameter_vector",
            [0.24] + EXPECTED_MODEL["shared_parameter_vector"][1:],
        ),
        ("model", "beta", 0.99),
        ("model", "parameter_cap", 2.1),
        ("model", "exact_dtype", "float32"),
        ("run", "objective_policy", "terminal_kl_divergence"),
        ("run", "exact_tolerance", 2e-12),
        ("run", "finite_difference_step", 2e-6),
        ("run", "finite_difference_tolerance", 2e-7),
        ("run", "batch_size", 32768),
        ("run", "rng_family", "numpy.random.RandomState(MT19937)"),
        ("run", "stream_policy", "one shared stream"),
        ("run", "reward_policy", "same_batch_occupancy_linearization"),
        ("run", "reference_policy", "propagated_reference"),
        ("run", "moment_policy", "means only"),
    ],
)
def test_every_scientific_field_mutation_is_rejected(
    section: str, field: str, replacement: object
) -> None:
    payload = deepcopy(_payload())
    table = dict(payload[section])
    table[field] = replacement
    payload[section] = table

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("model", "target_hash", "sha256:" + "0" * 64),
        ("model", "target_probabilities", [0.01, 0.09]),
        ("model", "target_conditional", [[1.0, 0.0, 0.0, 0.0]] * 4),
        ("run", "main_path_count", 63),
        ("run", "augmented_path_count", 4095),
        ("run", "release_seeds", [0, 1, 3]),
    ],
)
def test_derived_or_enumeration_identity_mutation_is_rejected(
    section: str, field: str, replacement: object
) -> None:
    payload = deepcopy(_payload())
    table = dict(payload[section])
    table[field] = replacement
    payload[section] = table

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


def test_snapshot_round_trip_and_factory_equality(tmp_path: Path) -> None:
    configured = load_experiment_config(CONFIG)
    snapshot = tmp_path / "trajectory.toml"
    snapshot.write_text(dump_experiment_config(configured), encoding="utf-8")

    reloaded = load_experiment_config(snapshot)

    assert reloaded == configured
    assert trajectory_reinforce_pasym_swap_spec() == configured.to_spec()
    with pytest.raises(TypeError, match="immutable"):
        trajectory_reinforce_pasym_swap_spec().run_parameters["batch_size"] = 1


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("model", "beta", 1),
        ("model", "shared_parameter_vector", [0] * 9),
        ("run", "finite_difference_step", 1),
        ("run", "release_seeds", [False, 1, 2]),
    ],
)
def test_request_rejects_coercive_numeric_encodings(
    section: str, field: str, replacement: object
) -> None:
    payload = deepcopy(_payload())
    table = dict(payload[section])
    table[field] = replacement
    payload[section] = table

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)
