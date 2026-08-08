"""Exact two-gate Torx smoke specification loaded from checked TOML."""

from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.records import ExperimentSpec

_CONFIG = experiment_config_path("torx-two-gate.toml")


def torx_smoke_spec(seed: int = 0) -> ExperimentSpec:
    """PNOT(0), then PCNOT(0,1), each applied with probability one-half."""

    return load_experiment_config(_CONFIG).with_overrides(seed=seed).to_spec()
