"""Model-context PAsymSwap compiler specification loaded from checked TOML."""

from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.records import ExperimentSpec

_CONFIG = experiment_config_path("thrml-model-context-pasym-swap.toml")


def model_context_pasym_swap_spec(seed: int = 0) -> ExperimentSpec:
    return load_experiment_config(_CONFIG).to_spec(seed=seed)
