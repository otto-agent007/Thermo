"""Hardware-constraint layer (queue row ``constraint-layer``, P-0004).

Declares which Z1-shaped constraints a model runs under and applies them:
placement on the 16-offset lattice with a two-colour schedule, codebook
quantization under a declared cap rule, static per-node beta jitter, chromatic
block Gibbs only with no replica exchange, and separate update, read, write and
clamp counts priced by the sealed Z1 profile. Every relaxed constraint is
named, with a reason, in the hashed request.

Evidence classes: rounding, placement and exact chromatic laws are
``exact_reference`` (float64); THRML runs are ``software_simulation`` (CPU,
float32 unless declared); pricing is a ``calibrated_projection`` of
algorithmic counts. Nothing here is a hardware measurement.

The package imports hash-bound study modules read-only and edits none of them.
THRML is imported only by ``thermo_lab.constraints.thrml_lowering``.
"""

from thermo_lab.constraints.accounting import count, price
from thermo_lab.constraints.gibbs import MAX_EXACT_FREE, exact_chromatic_kernels, propagate
from thermo_lab.constraints.jitter import request_gains, static_gains
from thermo_lab.constraints.lattice import (
    LatticeReport,
    check,
    colour,
    lower_m5c_layout,
    place_m5c,
    planar_target,
    require_placement,
    unplaced,
    unplaced_m5a_kernel,
)
from thermo_lab.constraints.model import PairwiseModel
from thermo_lab.constraints.quantize import (
    cap_for,
    codebook,
    quantize,
    quantize_array,
    quantize_terms,
    rounding_summary,
)
from thermo_lab.constraints.request import CONSTRAINTS, ConstraintRequest

__all__ = [
    "CONSTRAINTS",
    "MAX_EXACT_FREE",
    "ConstraintRequest",
    "LatticeReport",
    "PairwiseModel",
    "cap_for",
    "check",
    "codebook",
    "colour",
    "count",
    "exact_chromatic_kernels",
    "lower_m5c_layout",
    "place_m5c",
    "planar_target",
    "price",
    "propagate",
    "quantize",
    "quantize_array",
    "quantize_terms",
    "request_gains",
    "require_placement",
    "rounding_summary",
    "static_gains",
    "unplaced",
    "unplaced_m5a_kernel",
]
