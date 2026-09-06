"""Checked-input contracts for the model-context PAsymSwap compiler."""

from pathlib import Path

from thermo_lab.config import (
    MODEL_CONTEXT_PASYM_SWAP_EXPERIMENT_ID,
    load_experiment_config,
)
from thermo_lab.experiments.model_context_pasym_swap import model_context_pasym_swap_spec
from thermo_lab.hashing import to_json_value
from thermo_lab.schemas import ModelContextCompilerRunConfig

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/thrml-model-context-pasym-swap.toml"


def test_checked_model_context_config_loads_and_has_fixed_policies() -> None:
    config = load_experiment_config(CONFIG)
    run = ModelContextCompilerRunConfig.model_validate(to_json_value(config.run_parameters))

    assert config.experiment_id == MODEL_CONTEXT_PASYM_SWAP_EXPERIMENT_ID
    assert run.context_source == "mean_field_model_pre_gate"
    assert run.model_trace_policy == "one_pass_first_moment_factorization"
    assert run.upstream_artifact_policy == "rebuild_checked_target_context_artifacts"
    assert run.warm_start_policy == "paired_target_context_artifact_then_three_fixed_restarts"
    assert model_context_pasym_swap_spec(seed=2).seed == 2
