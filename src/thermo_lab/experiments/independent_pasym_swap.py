"""Independent PAsymSwap compiler specification loaded from checked TOML."""

from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.records import ExperimentSpec

_CONFIG = experiment_config_path("thrml-independent-pasym-swap.toml")


def independent_pasym_swap_spec(seed: int = 0) -> ExperimentSpec:
    return load_experiment_config(_CONFIG).with_overrides(seed=seed).to_spec()
