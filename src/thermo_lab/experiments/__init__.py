"""Checked research experiment specifications."""

from thermo_lab.experiments.independent_pasym_swap import independent_pasym_swap_spec
from thermo_lab.experiments.ising_chain import ising_chain_spec
from thermo_lab.experiments.target_context_pasym_swap import (
    target_context_pasym_swap_spec,
)
from thermo_lab.experiments.torx_smoke import torx_smoke_spec
from thermo_lab.experiments.weighted_graph_walk import weighted_graph_walk_spec

__all__ = [
    "independent_pasym_swap_spec",
    "ising_chain_spec",
    "target_context_pasym_swap_spec",
    "torx_smoke_spec",
    "weighted_graph_walk_spec",
]
