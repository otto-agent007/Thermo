"""Exact-categorical trajectory REINFORCE specification loaded from checked TOML."""

from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.records import ExperimentSpec


def trajectory_reinforce_pasym_swap_spec() -> ExperimentSpec:
    """Return the immutable checked three-site estimator-validation request."""

    return load_experiment_config(
        experiment_config_path("numpy-trajectory-reinforce-pasym-swap.toml")
    ).to_spec()
