"""Target-context PAsymSwap compiler specification loaded from checked TOML."""

from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.records import ExperimentSpec

_CONFIG = experiment_config_path("thrml-target-context-pasym-swap.toml")


def target_context_pasym_swap_spec(seed: int = 0) -> ExperimentSpec:
    """Return the checked exact target-context specification for one seed."""
    return load_experiment_config(_CONFIG).with_overrides(seed=seed).to_spec()
