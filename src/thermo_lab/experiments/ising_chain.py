"""Small Ising-chain specification loaded from checked TOML."""

from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.records import ExperimentSpec

_CONFIG = experiment_config_path("thrml-ising-chain.toml")


def ising_chain_spec(seed: int = 7, n_samples: int = 2_500) -> ExperimentSpec:
    return (
        load_experiment_config(_CONFIG)
        .with_overrides(seed=seed, run={"n_samples": n_samples})
        .to_spec()
    )
