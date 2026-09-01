from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from thermo_lab.config import ExperimentConfig, dump_experiment_config, load_experiment_config
from thermo_lab.evidence import BackendId
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.pasym_swap import COLOR_ORDER, COORDINATE_PAIR_CLASSES, PAPER_SOURCE, WORD_ORDER
from thermo_lab.schemas import (
    PARAMETER_ORDER,
    IndependentCompilerRunConfig,
    PAsymSwapModelConfig,
    validate_independent_pasym_swap_request,
)

CONFIG_PATH = Path("configs/experiments/thrml-independent-pasym-swap.toml")


def checked_payload() -> dict[str, object]:
    return load_experiment_config(CONFIG_PATH).model_dump(mode="python", by_alias=True)


def checked_model() -> dict[str, object]:
    return deepcopy(to_json_value(checked_payload()["model"]))


def checked_run() -> dict[str, object]:
    return deepcopy(to_json_value(checked_payload()["run"]))


def test_checked_pasym_swap_config_declares_every_scientific_choice() -> None:
    config = load_experiment_config(CONFIG_PATH)

    assert config.experiment_id == "thrml.independent_pasym_swap_compilation.v1"
    assert config.backend is BackendId.THRML_LOCAL
    model = PAsymSwapModelConfig.model_validate(to_json_value(config.model_parameters))
    run = IndependentCompilerRunConfig.model_validate(to_json_value(config.run_parameters))
    assert model.source_reference == PAPER_SOURCE
    assert model.color_order == COLOR_ORDER
    assert [item.coordinate_pairs for item in model.color_classes] == [
        tuple(COORDINATE_PAIR_CLASSES[name]) for name in COLOR_ORDER
    ]
    assert model.word_order == WORD_ORDER
    assert model.parameter_order == PARAMETER_ORDER
    assert model.parameter_cap == 2.0
    assert run.horizons == (1, 2, 4, 8, 16, 30)
    assert run.chain_count_per_context == 4096
    assert run.initializations[1] == (
        0.05,
        -0.05,
        0.05,
        -0.05,
        0.05,
        -0.05,
        0.05,
        -0.05,
        0.05,
    )


def test_checked_pasym_swap_sequences_are_deeply_immutable_and_json_stable() -> None:
    requested_model = checked_model()
    requested_run = checked_run()
    model = PAsymSwapModelConfig.model_validate(requested_model)
    run = IndependentCompilerRunConfig.model_validate(requested_run)

    assert isinstance(model.color_order, tuple)
    assert isinstance(model.color_classes, tuple)
    assert isinstance(model.color_classes[0].coordinate_pairs, tuple)
    assert isinstance(model.color_classes[0].coordinate_pairs[0], tuple)
    assert isinstance(model.word_order, tuple)
    assert isinstance(model.word_order[0], tuple)
    assert isinstance(model.topology_edges, tuple)
    assert isinstance(model.topology_edges[0], tuple)
    assert isinstance(run.context_weights, tuple)
    assert isinstance(run.initializations, tuple)
    assert isinstance(run.initializations[0], tuple)
    assert isinstance(run.horizons, tuple)
    assert isinstance(run.sweep_order, tuple)
    with pytest.raises(TypeError):
        model.color_order[0] = "H2"  # type: ignore[index]
    with pytest.raises(AttributeError):
        model.color_classes.append(model.color_classes[0])  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        model.color_classes[0].coordinate_pairs[0][0] = 4  # type: ignore[index]
    with pytest.raises(TypeError):
        model.topology_edges[0][0] = "hidden_0"  # type: ignore[index]
    with pytest.raises(TypeError):
        run.context_weights[0] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        run.initializations[0][0] = 0.1  # type: ignore[index]
    with pytest.raises(TypeError):
        run.horizons[0] = 30  # type: ignore[index]
    with pytest.raises(TypeError):
        run.sweep_order[0] = "outputs"  # type: ignore[index]
    assert model.model_dump(mode="json") == requested_model
    assert run.model_dump(mode="json") == requested_run


MODEL_MUTATIONS = [
    ("source_reference", "https://arxiv.org/abs/0000.00000v1"),
    ("torus_side", 6),
    ("coordinate_order", "(y,x), each coordinate in 0..4"),
    ("periodic_boundary", "modulo_6"),
    ("gamma", 3.0),
    ("delta_t", 0.06),
    ("macrosteps", 9),
    ("color_order", ["H2", "H1", "H3", "V1", "V2", "V3"]),
    (
        "color_classes",
        [
            {"name": "H1", "axis": "vertical", "coordinate_pairs": [[0, 1], [2, 3]]},
            {"name": "H2", "axis": "horizontal", "coordinate_pairs": [[1, 2], [3, 4]]},
            {"name": "H3", "axis": "horizontal", "coordinate_pairs": [[4, 0]]},
            {"name": "V1", "axis": "vertical", "coordinate_pairs": [[0, 1], [2, 3]]},
            {"name": "V2", "axis": "vertical", "coordinate_pairs": [[1, 2], [3, 4]]},
            {"name": "V3", "axis": "vertical", "coordinate_pairs": [[4, 0]]},
        ],
    ),
    ("word_order", [[0, 0], [1, 0], [0, 1], [1, 1]]),
    ("matrix_storage", "conditional[output_index][input_index]"),
    ("bit_to_spin", "s = b"),
    ("color_a_roles", ["hidden_0", "input_0", "input_1"]),
    ("color_b_roles", ["output_1", "output_0"]),
    ("topology_id", "thermo_k3_2_v2"),
    (
        "topology_edges",
        [
            ["input_0", "output_1"],
            ["input_0", "output_0"],
            ["input_1", "output_0"],
            ["input_1", "output_1"],
            ["hidden_0", "output_0"],
            ["hidden_0", "output_1"],
        ],
    ),
    ("parameter_order", list(reversed(PARAMETER_ORDER))),
    ("beta", 0.9),
    ("parameter_cap", 4.0),
    ("exact_dtype", "float32"),
    ("thrml_dtype", "float64"),
]

RUN_MUTATIONS = [
    ("context_weights", [0.4, 0.2, 0.2, 0.2]),
    ("optimizer", "scipy_cg"),
    ("maxiter", 2001),
    ("maxls", 51),
    ("ftol", 2e-12),
    ("gtol", 2e-9),
    ("projected_gradient_tolerance", 2e-6),
    (
        "initializations",
        [
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05],
            [-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05],
        ],
    ),
    ("restart_selection", "minimum_objective"),
    ("horizons", [1, 2, 8, 4, 16, 30]),
    ("deployment_horizon", 29),
    ("reset_distribution", "uniform_over_4_output_states"),
    ("sweep_order", ["outputs", "hidden"]),
    ("chain_count_per_context", 4095),
    ("samples_per_chain", 2),
    ("steps_per_sample", 2),
    ("key_policy", "split root key"),
    ("exact_normalization_tolerance", 2e-12),
    ("median_equilibrium_tv_tolerance", 0.16),
    ("worst_equilibrium_tv_tolerance", 0.36),
    ("k30_equilibrium_tv_tolerance", 0.06),
    ("thrml_k30_tv_tolerance", 0.11),
]


def assert_rejection_or_hash_change(
    payload: dict[str, object], *, section: str, configured: ExperimentConfig
) -> None:
    try:
        changed = ExperimentConfig.model_validate(payload)
    except ValidationError:
        return

    if section == "model":
        assert changed.to_spec().model_hash != configured.to_spec().model_hash
    else:
        assert (
            changed.to_spec().non_seed_run_config_hash
            != configured.to_spec().non_seed_run_config_hash
        )


@pytest.mark.parametrize(("field", "replacement"), MODEL_MUTATIONS)
def test_every_model_scientific_input_rejects_mutation_or_changes_model_hash(
    field: str, replacement: object
) -> None:
    configured = load_experiment_config(CONFIG_PATH)
    payload = checked_payload()
    model = checked_model()
    model[field] = replacement
    payload["model"] = model

    assert_rejection_or_hash_change(payload, section="model", configured=configured)


@pytest.mark.parametrize(("field", "replacement"), RUN_MUTATIONS)
def test_every_run_scientific_input_rejects_mutation_or_changes_non_seed_run_hash(
    field: str, replacement: object
) -> None:
    configured = load_experiment_config(CONFIG_PATH)
    payload = checked_payload()
    run = checked_run()
    run[field] = replacement
    payload["run"] = run

    assert_rejection_or_hash_change(payload, section="run", configured=configured)


def test_every_sample_definition_mutation_changes_non_seed_run_hash() -> None:
    configured = load_experiment_config(CONFIG_PATH)
    payload = checked_payload()
    payload["sample_definition"] = "A materially different sample definition."

    assert_rejection_or_hash_change(payload, section="run", configured=configured)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ftol", 1),
        ("gtol", 1),
        ("projected_gradient_tolerance", 1),
        ("exact_normalization_tolerance", 1),
    ],
)
def test_run_schema_rejects_integer_encoded_floats(field: str, value: int) -> None:
    run = checked_run()
    run[field] = value

    with pytest.raises(ValidationError):
        IndependentCompilerRunConfig.model_validate(run)


def test_checked_schemas_reject_unknown_keys_and_invalid_seed() -> None:
    model = checked_model()
    model["unknown"] = "value"
    with pytest.raises(ValidationError):
        PAsymSwapModelConfig.model_validate(model)

    run = checked_run()
    run["unknown"] = "value"
    with pytest.raises(ValidationError):
        IndependentCompilerRunConfig.model_validate(run)

    model = PAsymSwapModelConfig.model_validate(checked_model())
    run = IndependentCompilerRunConfig.model_validate(checked_run())
    with pytest.raises(ValueError, match="nonnegative"):
        validate_independent_pasym_swap_request(model, run, seed=-1)


def test_public_request_validation_rejects_constructed_schema_bypasses() -> None:
    model = PAsymSwapModelConfig.model_validate(checked_model())
    run = IndependentCompilerRunConfig.model_validate(checked_run())
    forged_model_payload = dict(model.__dict__)
    forged_model_payload["parameter_cap"] = 4.0
    forged_model = PAsymSwapModelConfig.model_construct(**forged_model_payload)
    forged_run_payload = dict(run.__dict__)
    forged_run_payload["context_weights"] = (0.0, 0.5, 0.25, 0.25)
    forged_run = IndependentCompilerRunConfig.model_construct(**forged_run_payload)

    with pytest.raises(ValueError, match="parameter_cap"):
        validate_independent_pasym_swap_request(forged_model, run, seed=0)
    with pytest.raises(ValueError, match="context_weights"):
        validate_independent_pasym_swap_request(model, forged_run, seed=0)
    with pytest.raises(TypeError, match="PAsymSwapModelConfig"):
        validate_independent_pasym_swap_request(checked_model(), run, seed=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="IndependentCompilerRunConfig"):
        validate_independent_pasym_swap_request(model, checked_run(), seed=0)  # type: ignore[arg-type]


def test_checked_config_rejects_non_thrml_backend() -> None:
    payload = checked_payload()
    payload["backend"] = "torx_statevector"

    with pytest.raises(ValidationError, match="requires backend"):
        ExperimentConfig.model_validate(payload)


def test_strict_schema_dumps_preserve_requested_hash_material() -> None:
    configured = load_experiment_config(CONFIG_PATH)
    requested_model = to_json_value(configured.model_parameters)
    requested_run = to_json_value(configured.run_parameters)
    model = PAsymSwapModelConfig.model_validate(requested_model)
    run = IndependentCompilerRunConfig.model_validate(requested_run)

    assert canonical_sha256(model.model_dump(mode="json")) == canonical_sha256(requested_model)
    assert canonical_sha256(run.model_dump(mode="json")) == canonical_sha256(requested_run)


def test_checked_pasym_swap_config_snapshot_round_trips(tmp_path: Path) -> None:
    configured = load_experiment_config(CONFIG_PATH)
    snapshot = tmp_path / "pasym-swap.toml"
    snapshot.write_text(dump_experiment_config(configured), encoding="utf-8")

    assert load_experiment_config(snapshot) == configured
