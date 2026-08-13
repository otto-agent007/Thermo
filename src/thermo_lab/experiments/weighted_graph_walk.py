"""Weighted graph-walk specification loaded from checked TOML."""

from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.records import ExperimentSpec

WEIGHTED_GRAPH_WALK_EXPERIMENT_ID = "torx.weighted_graph_walk.v1"
_CONFIG = experiment_config_path("torx-weighted-graph-walk.toml")


def weighted_graph_walk_spec() -> ExperimentSpec:
    return load_experiment_config(_CONFIG).to_spec()
