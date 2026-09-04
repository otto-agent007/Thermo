from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

import thermo_lab.config as config_module
import thermo_lab.experiments as experiments_module
import thermo_lab.schemas as schemas_module
from thermo_lab.evidence import BackendId
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.pasym_swap import COLOR_ORDER, PAPER_SOURCE, WORD_ORDER

CONFIG_PATH = Path("configs/experiments/thrml-target-context-pasym-swap.toml")
EXPERIMENT_ID = "thrml.target_context_pasym_swap_compilation.v1"
SAMPLE_DEFINITION = (
    "One independently seeded THRML cross-check using 4,096 chains per input context "
    "over every exact target-context compiled kernel at 30 complete two-color Gibbs "
    "sweeps; the exact target trajectory and both compiler variants are deterministic."
)


def checked_payload() -> dict[str, object]:
    return config_module.load_experiment_config(CONFIG_PATH).model_dump(
        mode="python", by_alias=True
    )


def checked_model() -> dict[str, object]:
    return deepcopy(to_json_value(checked_payload()["model"]))


def checked_run() -> dict[str, object]:
    return deepcopy(to_json_value(checked_payload()["run"]))


def target_run_config():
    return schemas_module.TargetContextCompilerRunConfig


def test_target_context_public_identity_and_checked_file_are_registered() -> None:
    assert config_module.TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID == EXPERIMENT_ID
    assert config_module.TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION == SAMPLE_DEFINITION
    assert (
        config_module.experiment_config_path("thrml-target-context-pasym-swap.toml").read_bytes()
        == CONFIG_PATH.read_bytes()
    )


def test_checked_target_context_config_declares_every_scientific_choice() -> None:
    config = config_module.load_experiment_config(CONFIG_PATH)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.backend is BackendId.THRML_LOCAL
    assert config.sample_definition == SAMPLE_DEFINITION
    model = schemas_module.PAsymSwapModelConfig.model_validate(
        to_json_value(config.model_parameters)
    )
    run = target_run_config().model_validate(to_json_value(config.run_parameters))
    assert model.source_reference == PAPER_SOURCE
    assert model.color_order == COLOR_ORDER
    assert model.word_order == WORD_ORDER
    assert model.parameter_order == schemas_module.PARAMETER_ORDER
    assert model.parameter_cap == 2.0
    assert run.initial_particle_site == (0, 0)
    assert run.context_source == "exact_target_trajectory"
    assert run.context_aggregation == "mean_over_occurrences_sharing_target_hash"
    assert run.zero_support_policy == "preserve_exact_zero_and_report_off_support"
    assert run.baseline_context_weights == (0.25, 0.25, 0.25, 0.25)
    assert run.horizons == (1, 2, 4, 8, 16, 30)
    assert run.chain_count_per_context == 4096
    assert run.target_cm_not_worse_tolerance == 1e-10
    assert run.median_target_weighted_equilibrium_tv_tolerance == 0.05
    assert run.worst_target_weighted_equilibrium_tv_tolerance == 0.10
    assert run.k30_equilibrium_tv_tolerance == 0.05
    assert run.thrml_k30_tv_tolerance == 0.10


def test_checked_target_context_sequences_are_immutable_and_json_stable() -> None:
    requested_model = checked_model()
    requested_run = checked_run()
    model = schemas_module.PAsymSwapModelConfig.model_validate(requested_model)
    run = target_run_config().model_validate(requested_run)

    assert isinstance(run.initial_particle_site, tuple)
    assert isinstance(run.baseline_context_weights, tuple)
    assert isinstance(run.initializations, tuple)
    assert isinstance(run.initializations[0], tuple)
    assert isinstance(run.horizons, tuple)
    assert isinstance(run.sweep_order, tuple)
    with pytest.raises(TypeError):
        run.initial_particle_site[0] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        run.baseline_context_weights[0] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        run.initializations[0][0] = 0.1  # type: ignore[index]
    assert model.model_dump(mode="json") == requested_model
    assert run.model_dump(mode="json") == requested_run


def test_target_context_non_seed_hash_matches_checked_config() -> None:
    config = config_module.load_experiment_config(CONFIG_PATH)
    model = schemas_module.PAsymSwapModelConfig.model_validate(
        to_json_value(config.model_parameters)
    )
    run = target_run_config().model_validate(to_json_value(config.run_parameters))

    assert (
        config_module.target_context_pasym_swap_non_seed_config_hash(model, run)
        == config.non_seed_config_hash
    )


RUN_MUTATIONS = [
    ("initial_particle_site", [1, 0]),
    ("context_source", "sampled_target_trajectory"),
    ("context_aggregation", "occurrence_specific_parameters"),
    ("zero_support_policy", "epsilon_floor"),
    ("baseline_context_weights", [0.4, 0.2, 0.2, 0.2]),
    ("optimizer", "scipy_cg"),
    ("maxiter", 2001),
    ("maxls", 51),
    ("ftol", 2e-12),
    ("gtol", 2e-9),
    ("projected_gradient_tolerance", 2e-6),
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
    ("target_cm_not_worse_tolerance", 2e-10),
    ("median_target_weighted_equilibrium_tv_tolerance", 0.06),
    ("worst_target_weighted_equilibrium_tv_tolerance", 0.11),
    ("k30_equilibrium_tv_tolerance", 0.06),
    ("thrml_k30_tv_tolerance", 0.11),
]


@pytest.mark.parametrize(("field", "replacement"), RUN_MUTATIONS)
def test_every_target_context_run_input_rejects_mutation_or_changes_hash(
    field: str, replacement: object
) -> None:
    configured = config_module.load_experiment_config(CONFIG_PATH)
    payload = checked_payload()
    run = checked_run()
    run[field] = replacement
    payload["run"] = run

    try:
        changed = config_module.ExperimentConfig.model_validate(payload)
    except ValidationError:
        return
    assert changed.non_seed_config_hash != configured.non_seed_config_hash


@pytest.mark.parametrize(
    "field",
    [
        "ftol",
        "gtol",
        "projected_gradient_tolerance",
        "exact_normalization_tolerance",
        "target_cm_not_worse_tolerance",
        "median_target_weighted_equilibrium_tv_tolerance",
        "worst_target_weighted_equilibrium_tv_tolerance",
        "k30_equilibrium_tv_tolerance",
        "thrml_k30_tv_tolerance",
    ],
)
def test_target_context_schema_rejects_integer_encoded_floats(field: str) -> None:
    run = checked_run()
    run[field] = 1

    with pytest.raises(ValidationError):
        target_run_config().model_validate(run)


def test_target_context_schema_rejects_unknown_keys_and_invalid_seed() -> None:
    run = checked_run()
    run["unknown"] = "value"
    with pytest.raises(ValidationError):
        target_run_config().model_validate(run)

    model = schemas_module.PAsymSwapModelConfig.model_validate(checked_model())
    validated_run = target_run_config().model_validate(checked_run())
    with pytest.raises(ValueError, match="nonnegative"):
        schemas_module.validate_target_context_pasym_swap_request(model, validated_run, seed=-1)


def test_target_context_request_validation_rejects_constructed_bypasses() -> None:
    model = schemas_module.PAsymSwapModelConfig.model_validate(checked_model())
    run = target_run_config().model_validate(checked_run())
    forged_model_payload = dict(model.__dict__)
    forged_model_payload["parameter_cap"] = 4.0
    forged_model = schemas_module.PAsymSwapModelConfig.model_construct(**forged_model_payload)
    forged_run_payload = dict(run.__dict__)
    forged_run_payload["zero_support_policy"] = "epsilon_floor"
    forged_run = target_run_config().model_construct(**forged_run_payload)

    with pytest.raises(ValueError, match="parameter_cap"):
        schemas_module.validate_target_context_pasym_swap_request(forged_model, run, seed=0)
    with pytest.raises(ValueError, match="zero_support_policy"):
        schemas_module.validate_target_context_pasym_swap_request(model, forged_run, seed=0)
    with pytest.raises(TypeError, match="PAsymSwapModelConfig"):
        schemas_module.validate_target_context_pasym_swap_request(checked_model(), run, seed=0)
    with pytest.raises(TypeError, match="TargetContextCompilerRunConfig"):
        schemas_module.validate_target_context_pasym_swap_request(model, checked_run(), seed=0)


def test_target_context_checked_config_round_trips(tmp_path: Path) -> None:
    configured = config_module.load_experiment_config(CONFIG_PATH)
    snapshot = tmp_path / "target-context.toml"
    snapshot.write_text(config_module.dump_experiment_config(configured), encoding="utf-8")

    assert config_module.load_experiment_config(snapshot) == configured
    assert canonical_sha256(to_json_value(configured.run_parameters)) == canonical_sha256(
        checked_run()
    )


def test_target_context_factory_uses_checked_config() -> None:
    assert experiments_module.target_context_pasym_swap_spec() == (
        config_module.load_experiment_config(CONFIG_PATH).to_spec()
    )
    assert experiments_module.target_context_pasym_swap_spec(seed=9).seed == 9
