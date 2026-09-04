"""Checked-input contracts for the target-context PAsymSwap compiler."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from thermo_lab.config import (
    TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID,
    TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION,
    ExperimentConfig,
    independent_pasym_swap_non_seed_config_hash,
    load_experiment_config,
    target_context_pasym_swap_non_seed_config_hash,
)
from thermo_lab.evidence import BackendId
from thermo_lab.experiments.target_context_pasym_swap import target_context_pasym_swap_spec
from thermo_lab.hashing import to_json_value
from thermo_lab.schemas import (
    IndependentCompilerRunConfig,
    PAsymSwapModelConfig,
    TargetContextCompilerRunConfig,
    validate_target_context_pasym_swap_request,
)

ROOT = Path(__file__).parents[2]
INDEPENDENT_CONFIG = ROOT / "configs/experiments/thrml-independent-pasym-swap.toml"
TARGET_CONFIG = ROOT / "configs/experiments/thrml-target-context-pasym-swap.toml"


def target_payload() -> dict[str, object]:
    return load_experiment_config(TARGET_CONFIG).model_dump(mode="python", by_alias=True)


def target_model() -> dict[str, object]:
    return deepcopy(to_json_value(target_payload()["model"]))


def target_run() -> dict[str, object]:
    return deepcopy(to_json_value(target_payload()["run"]))


def test_checked_target_context_config_has_exact_schema_and_shared_model() -> None:
    target = load_experiment_config(TARGET_CONFIG)
    independent = load_experiment_config(INDEPENDENT_CONFIG)

    assert target.experiment_id == TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID
    assert target.backend is BackendId.THRML_LOCAL
    assert target.sample_definition == TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION
    assert target.model_parameters == independent.model_parameters
    run = TargetContextCompilerRunConfig.model_validate(to_json_value(target.run_parameters))
    assert run.initial_particle_site == (0, 0)
    assert run.initial_occupancy == (1.0,) + (0.0,) * 24
    assert run.baseline_context_weights == (0.25,) * 4
    assert "context_weights" not in TargetContextCompilerRunConfig.model_fields
    assert "median_equilibrium_tv_tolerance" not in TargetContextCompilerRunConfig.model_fields
    assert "worst_equilibrium_tv_tolerance" not in TargetContextCompilerRunConfig.model_fields


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("initial_state", "empty"),
        ("initial_particle_site", [0, 1]),
        ("initial_occupancy_order", "(x,y)"),
        ("initial_occupancy", [1.0] + [0.0] * 23),
        ("context_source", "post_gate"),
        ("context_reduction", "uniform_profiles"),
        ("zero_support_policy", "smoothed"),
        ("warm_start_policy", "fixed_restarts"),
        ("optimizer", "scipy_cg"),
        ("maxiter", 1999),
        ("maxls", 49),
        ("ftol", 2e-12),
        ("gtol", 2e-9),
        ("projected_gradient_tolerance", 2e-6),
        ("initializations", [[0.0] * 9, [0.05] * 9, [-0.05] * 9]),
        ("restart_selection", "minimum_objective"),
        ("horizons", [1, 2, 8, 4, 16, 30]),
        ("deployment_horizon", 29),
        ("reset_distribution", "uniform_over_4_output_states"),
        ("sweep_order", ["outputs", "hidden"]),
        ("chain_count_per_context", 4095),
        ("samples_per_chain", 2),
        ("steps_per_sample", 2),
        ("key_policy", "split root key"),
        ("baseline_context_weights", [0.4, 0.2, 0.2, 0.2]),
        ("exact_normalization_tolerance", 2e-12),
        ("baseline_median_equilibrium_tv_tolerance", 0.16),
        ("baseline_worst_equilibrium_tv_tolerance", 0.36),
        ("k30_equilibrium_tv_tolerance", 0.06),
        ("thrml_k30_tv_tolerance", 0.11),
        ("profile_kl_non_regression_tolerance", 2e-12),
        ("minimum_occurrence_weighted_kl_improvement", 2e-8),
    ],
)
def test_target_schema_rejects_each_changed_checked_policy_or_schedule(
    field: str, replacement: object
) -> None:
    run = target_run()
    run[field] = replacement

    with pytest.raises(ValidationError):
        TargetContextCompilerRunConfig.model_validate(run)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("ftol", 1),
        ("initial_occupancy", [1] + [0.0] * 24),
        ("baseline_context_weights", [0.25, 0.25, 0.25, 1]),
        ("initializations", [[0] * 9, [0.05] * 9, [-0.05] * 9]),
    ],
)
def test_target_schema_requires_json_float_encoding(field: str, replacement: object) -> None:
    run = target_run()
    run[field] = replacement

    with pytest.raises(ValidationError):
        TargetContextCompilerRunConfig.model_validate(run)


@pytest.mark.parametrize(
    "field",
    [
        "context_weights",
        "median_equilibrium_tv_tolerance",
        "worst_equilibrium_tv_tolerance",
    ],
)
def test_target_schema_rejects_legacy_unscoped_fields(field: str) -> None:
    run = target_run()
    run[field] = 0.25 if field == "context_weights" else 0.15

    with pytest.raises(ValidationError, match=field):
        TargetContextCompilerRunConfig.model_validate(run)


def test_independent_schema_rejects_target_only_fields() -> None:
    independent = load_experiment_config(INDEPENDENT_CONFIG)
    run = to_json_value(independent.run_parameters)
    assert isinstance(run, dict)
    run["context_source"] = "exact_target_pre_gate"

    with pytest.raises(ValidationError, match="context_source"):
        IndependentCompilerRunConfig.model_validate(run)


def test_target_request_revalidates_forged_instances_and_rejects_invalid_seeds() -> None:
    model = PAsymSwapModelConfig.model_validate(target_model())
    run = TargetContextCompilerRunConfig.model_validate(target_run())
    forged_model = PAsymSwapModelConfig.model_construct(macrosteps=9)
    forged_run = TargetContextCompilerRunConfig.model_construct(deployment_horizon=29)

    validate_target_context_pasym_swap_request(model, run, seed=0)
    for seed in (True, -1, 0.0):
        with pytest.raises(ValueError, match="nonnegative integer"):
            validate_target_context_pasym_swap_request(model, run, seed=seed)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        validate_target_context_pasym_swap_request(forged_model, run, seed=0)
    with pytest.raises(ValidationError):
        validate_target_context_pasym_swap_request(model, forged_run, seed=0)
    with pytest.raises(TypeError):
        validate_target_context_pasym_swap_request(target_model(), run, seed=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_target_context_pasym_swap_request(model, target_run(), seed=0)  # type: ignore[arg-type]


def test_target_config_rejects_backend_sample_definition_seed_and_unknown_fields() -> None:
    payload = target_payload()
    for field, replacement in (
        ("backend", BackendId.TORX_STATEVECTOR),
        ("sample_definition", "one sample"),
        ("seed", True),
        ("seed", -1),
        ("seed", 0.0),
    ):
        invalid = deepcopy(payload)
        invalid[field] = replacement
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(invalid)
    invalid = deepcopy(payload)
    invalid["run"]["unexpected"] = "field"  # type: ignore[index]
    with pytest.raises(ValidationError, match="unexpected"):
        ExperimentConfig.model_validate(invalid)


def test_target_hashes_are_distinct_from_independent_and_ignore_seed() -> None:
    target = load_experiment_config(TARGET_CONFIG)
    independent = load_experiment_config(INDEPENDENT_CONFIG)
    model = PAsymSwapModelConfig.model_validate(to_json_value(target.model_parameters))
    run = TargetContextCompilerRunConfig.model_validate(to_json_value(target.run_parameters))
    independent_run = IndependentCompilerRunConfig.model_validate(
        to_json_value(independent.run_parameters)
    )

    assert hashlib.sha256(INDEPENDENT_CONFIG.read_bytes()).hexdigest() == (
        "7222466ee092c79a3930c547fd2284db3fa118ec12742c60a71339b52e95a8ac"
    )
    assert independent.non_seed_config_hash == (
        "sha256:ef8890e5d0350df60afd2b534f11d32aed317e1ab37d4e786a9e4c221b747e70"
    )
    assert independent.model_hash == (
        "sha256:b28ffb03b70f63dfe2765b2a91477dfc72df2e4ff7fd313ec8a150558b64fe57"
    )
    assert independent.to_spec().non_seed_run_config_hash == (
        "sha256:8b36c17bf74581ba4b1d557201d15bb0c598669129d6b05c9195adf96a747cbc"
    )
    assert target_context_pasym_swap_non_seed_config_hash(model, run) == target.non_seed_config_hash
    assert independent_pasym_swap_non_seed_config_hash(model, independent_run) == (
        independent.non_seed_config_hash
    )
    assert target.non_seed_config_hash != target.to_spec().non_seed_run_config_hash
    assert target.non_seed_config_hash != independent.non_seed_config_hash
    assert target.with_overrides(seed=7).non_seed_config_hash == target.non_seed_config_hash
    changed_run = run.model_copy(update={"minimum_occurrence_weighted_kl_improvement": 2e-8})
    assert target_context_pasym_swap_non_seed_config_hash(model, changed_run) != (
        target.non_seed_config_hash
    )


def test_target_factory_loads_checked_config_with_requested_seed() -> None:
    assert target_context_pasym_swap_spec() == load_experiment_config(TARGET_CONFIG).to_spec()
    assert target_context_pasym_swap_spec(seed=2).seed == 2
