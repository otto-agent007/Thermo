"""Checked public boundary for the one-step 25-site refinement study."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from thermo_lab.evidence import BackendId

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/numpy-composed-pasym-swap-trajectory-refinement-one-step.toml"
EXPERIMENT_ID = "numpy.composed_pasym_swap_trajectory_refinement_one_step.v1"


def _backend_module():
    spec = importlib.util.find_spec("thermo_lab.backends.numpy_composed_trajectory_refinement")
    assert spec is not None
    return importlib.import_module("thermo_lab.backends.numpy_composed_trajectory_refinement")


def test_checked_config_loads_with_exact_release_contract() -> None:
    assert CONFIG.is_file()
    config_module = importlib.import_module("thermo_lab.config")
    assert config_module.COMPOSED_TRAJECTORY_REFINEMENT_EXPERIMENT_ID == EXPERIMENT_ID
    configured = config_module.load_experiment_config(CONFIG)

    assert configured.experiment_id == EXPERIMENT_ID
    assert configured.backend is BackendId.NUMPY_EXACT_CATEGORICAL
    assert configured.seed == 0
    run = dict(configured.run_parameters)
    assert run["start_family"] == "model_context"
    assert run["horizon"] == "equilibrium"
    assert run["occupancy_batch_size"] == 32768
    assert run["gradient_batch_size"] == 32768
    assert run["evaluation_batch_size"] == 32768
    assert tuple(run["release_seeds"]) == (0, 1, 2)
    assert run["learning_rate"] == 0.01
    assert run["improvement_policy"] == "descriptive_non_gating"


def test_refinement_role_seeds_are_deterministic_and_distinct() -> None:
    backend_module = _backend_module()
    first = backend_module.spawn_refinement_role_seeds(7)
    second = backend_module.spawn_refinement_role_seeds(7)

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert all(type(seed) is int and seed >= 0 for seed in first)


def test_backend_checked_request_accepts_authoritative_hash_semantics() -> None:
    config_module = importlib.import_module("thermo_lab.config")
    backend_module = _backend_module()
    configured = config_module.load_experiment_config(CONFIG)
    backend = backend_module.NumpyComposedTrajectoryRefinementBackend()

    _, _, request_hash = backend.checked_request(configured.to_spec(seed=0))

    assert request_hash == configured.non_seed_config_hash


def test_runner_dispatches_the_dedicated_checked_backend() -> None:
    config_module = importlib.import_module("thermo_lab.config")
    runner_module = importlib.import_module("thermo_lab.runner")
    backend_module = _backend_module()
    configured = config_module.load_experiment_config(CONFIG)

    backend = runner_module._backend(configured, None)

    assert isinstance(backend, backend_module.NumpyComposedTrajectoryRefinementBackend)


def test_ci_packages_and_runs_the_checked_refinement_config() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    config = "configs/experiments/numpy-composed-pasym-swap-trajectory-refinement-one-step.toml"

    assert f'"{config}",' in workflow
    assert "name: composed-trajectory-refinement-one-step" in workflow
    assert config in workflow
    assert "--seeds 0,1,2" in workflow
