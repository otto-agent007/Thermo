"""Checked request identity for one-step trajectory refinement."""

from __future__ import annotations

import importlib
from pathlib import Path

from thermo_lab.evidence import BackendId
from thermo_lab.hashing import to_json_value

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/numpy-trajectory-reinforce-pasym-swap-one-step.toml"


def test_checked_one_step_request_declares_the_update_and_exact_readout() -> None:
    config_module = importlib.import_module("thermo_lab.config")
    configured = config_module.load_experiment_config(CONFIG)

    assert config_module.TRAJECTORY_REINFORCE_REFINEMENT_EXPERIMENT_ID == (
        "numpy.trajectory_reinforce_pasym_swap_one_step.v1"
    )
    assert configured.experiment_id == config_module.TRAJECTORY_REINFORCE_REFINEMENT_EXPERIMENT_ID
    assert configured.backend is BackendId.NUMPY_EXACT_CATEGORICAL
    assert configured.seed == 0
    run = to_json_value(configured.run_parameters)
    assert run["update_policy"] == "one_projected_shared_gradient_descent_step"
    assert run["gradient_source"] == "seeded_covariance_aware_shared_gradient_mean"
    assert run["learning_rate"] == 0.25
    assert run["objective_evaluation_policy"] == "exact_enumeration_before_and_after"
    assert run["improvement_policy"] == "strict_objective_decrease"
    assert configured.with_overrides(seed=1).non_seed_config_hash == configured.non_seed_config_hash
