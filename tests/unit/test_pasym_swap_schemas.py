from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from thermo_lab.config import ExperimentConfig, dump_experiment_config, load_experiment_config
from thermo_lab.evidence import BackendId
from thermo_lab.hashing import to_json_value
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
    assert model.color_order == list(COLOR_ORDER)
    assert [item.coordinate_pairs for item in model.color_classes] == [
        [list(pair) for pair in COORDINATE_PAIR_CLASSES[name]] for name in COLOR_ORDER
    ]
    assert model.word_order == [list(word) for word in WORD_ORDER]
    assert model.parameter_order == list(PARAMETER_ORDER)
    assert run.horizons == [1, 2, 4, 8, 16, 30]
    assert run.chain_count_per_context == 4096
    assert run.initializations[1] == [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05]


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        ("model", lambda value: value.__setitem__("gamma", 2)),
        ("model", lambda value: value.__setitem__("coordinate_order", "(y,x)")),
        ("model", lambda value: value.__setitem__("periodic_boundary", "open")),
        ("model", lambda value: value["color_classes"][0].__setitem__("axis", "vertical")),
        ("model", lambda value: value["color_classes"][0]["coordinate_pairs"][0].__setitem__(0, 4)),
        ("model", lambda value: value.__setitem__("word_order", [[0, 0], [1, 0], [0, 1], [1, 1]])),
        (
            "model",
            lambda value: value.__setitem__(
                "color_a_roles", ["hidden_0", "input_0", "input_1"]
            ),
        ),
        ("model", lambda value: value["topology_edges"].__setitem__(0, ["output_0", "input_0"])),
        (
            "model",
            lambda value: value.__setitem__("parameter_order", list(reversed(PARAMETER_ORDER))),
        ),
        ("model", lambda value: value.__setitem__("beta", 0.9)),
        ("model", lambda value: value.__setitem__("parameter_cap", 3.0)),
        ("run", lambda value: value.__setitem__("context_weights", [0.4, 0.2, 0.2, 0.2])),
        ("run", lambda value: value.__setitem__("horizons", [1, 2, 8, 4, 16, 30])),
        ("run", lambda value: value.__setitem__("horizons", [1, 2, 4, 8, 16])),
        ("run", lambda value: value.__setitem__("maxiter", 1999)),
        ("run", lambda value: value.__setitem__("initializations", [[0.0] * 8] * 3)),
        ("run", lambda value: value.__setitem__("exact_normalization_tolerance", 1e-11)),
        ("run", lambda value: value.__setitem__("median_equilibrium_tv_tolerance", 0.16)),
    ],
)
def test_checked_scientific_inputs_reject_mutations(section: str, mutate) -> None:
    value = checked_model() if section == "model" else checked_run()
    mutate(value)

    with pytest.raises(ValidationError):
        if section == "model":
            PAsymSwapModelConfig.model_validate(value)
        else:
            IndependentCompilerRunConfig.model_validate(value)


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


def test_checked_config_rejects_non_thrml_backend() -> None:
    payload = checked_payload()
    payload["backend"] = "torx_statevector"

    with pytest.raises(ValidationError, match="requires backend"):
        ExperimentConfig.model_validate(payload)


def test_request_hashes_change_for_allowed_input_mutations() -> None:
    configured = load_experiment_config(CONFIG_PATH)
    payload = checked_payload()
    payload["sample_definition"] = "A materially different sample definition."
    changed = ExperimentConfig.model_validate(payload)

    assert changed.non_seed_config_hash != configured.non_seed_config_hash


def test_checked_pasym_swap_config_snapshot_round_trips(tmp_path: Path) -> None:
    configured = load_experiment_config(CONFIG_PATH)
    snapshot = tmp_path / "pasym-swap.toml"
    snapshot.write_text(dump_experiment_config(configured), encoding="utf-8")

    assert load_experiment_config(snapshot) == configured
