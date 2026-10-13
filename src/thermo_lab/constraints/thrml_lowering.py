"""THRML lowering of constraint-layer models (the only module that calls THRML).

Programs are float32 on CPU unless the request declares otherwise, as every
archived THRML study ran. With static beta jitter the free blocks use
``am_coupling_bits.GainSpinConditional`` (imported, not copied), so jittered
programs share the associative-memory study's arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, BlockGibbsSpec, FactorSamplingProgram, SamplingSchedule, SpinNode
from thrml import sample_states as thrml_sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from thermo_lab import am_coupling_bits as am
from thermo_lab.constraints.lattice import require_placement
from thermo_lab.constraints.model import PairwiseModel
from thermo_lab.constraints.request import ConstraintRequest


def colour_blocks(model: PairwiseModel) -> list[list[int]]:
    """Free nodes of the first colour, then of the second, each in ``free`` order."""
    colours = model.node_colours
    return [
        [n for n in model.free if colours[n] == c]
        for c in (model.first_colour, 1 - model.first_colour)
    ]


def lower(model: PairwiseModel, request: ConstraintRequest, gains: np.ndarray | None = None):
    """Model -> (IsingEBM, sampling program, free colour blocks).

    One free block per colour (first colour first) and one clamped block in
    ``clamped`` order. ``gains`` (one per node, any order of the model's nodes)
    are required when the request's jitter is positive and forbidden otherwise.
    """
    require_placement(model, request)
    dtype = jnp.dtype(request.dtype)
    nodes = [SpinNode() for _ in range(model.nodes)]
    ebm = IsingEBM(
        nodes,
        [(nodes[u], nodes[v]) for u, v in model.edges],
        jnp.asarray(model.bias, dtype=dtype),
        jnp.asarray(model.weights, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
    )
    blocks = [Block([nodes[n] for n in group]) for group in colour_blocks(model)]
    clamped = [Block([nodes[c] for c in model.clamped])] if model.clamped else []
    if request.jitter == 0.0:
        if gains is not None:
            raise ValueError("gains given but the request declares no jitter")
        return ebm, IsingSamplingProgram(ebm, blocks, clamped_blocks=clamped), blocks
    if gains is None or np.shape(gains) != (model.nodes,):
        raise ValueError("a jittered request needs one gain per node")
    spec = BlockGibbsSpec(blocks, clamped, ebm.node_shape_dtypes)
    samplers = [
        am.GainSpinConditional(jnp.asarray(np.asarray(gains)[group], dtype))
        for group in colour_blocks(model)
    ]
    return ebm, FactorSamplingProgram(spec, samplers, ebm.factors, []), blocks


def lower_am(zq: dict, gains: Sequence[np.ndarray], p: int, n: int, cue: int):
    """The associative-memory study's own gain program from layer terms and gains.

    Bitwise reproduction of its sampled counts needs its node and factor order,
    which ``lower`` does not promise, so the layer hands its quantized terms and
    static gains to ``am_coupling_bits.gain_structure(...).gain_program``.
    Returns ``(structure, program)``.
    """
    gv, gh = gains
    structure = am.gain_structure(p, n, cue)
    return structure, structure.gain_program(zq, gv, gh)


def final_states(
    program,
    blocks: list,
    sweeps: int,
    keys: jax.Array,
    clamped_values: jax.Array,
    initial: Sequence[np.ndarray],
) -> np.ndarray:
    """Free-node states after exactly ``sweeps`` sweeps, one chain per key.

    Each chain key is split into (init, sample); the init half is unused because
    ``initial`` (one bool array per free block) is fixed. Output columns follow
    the free blocks in order. Synchronized before return.
    """
    schedule = SamplingSchedule(n_warmup=sweeps, n_samples=1, steps_per_sample=1)
    init = [jnp.asarray(np.asarray(s, dtype=bool)) for s in initial]

    def one_chain(key, x):
        _, sample_key = jax.random.split(key, 2)
        observed = thrml_sample_states(sample_key, program, schedule, init, [x], blocks)
        return jnp.concatenate([o[0] for o in observed])

    out = jax.jit(jax.vmap(one_chain))(keys, clamped_values)
    return np.asarray(out.block_until_ready())
