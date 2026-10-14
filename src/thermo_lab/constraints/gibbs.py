"""Exact float64 chromatic block-Gibbs kernels for small pairwise models.

One sweep updates the ``first_colour`` block, then the other, each node from
its THRML spin conditional P(s = +1) = sigmoid(2 * field). There is no replica
exchange and no per-replica temperature anywhere in this module.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from thermo_lab.constraints.model import PairwiseModel

MAX_EXACT_FREE = 12  # AGENTS.md: keep exact enumerators deliberately bounded


def free_states(n_free: int) -> np.ndarray:
    """(2^n, n) +-1 table; bit k of the row index is free node k (E0's joint order)."""
    return ((np.arange(1 << n_free)[:, None] >> np.arange(n_free)) & 1) * 2.0 - 1.0


def exact_chromatic_kernels(
    model: PairwiseModel, x_batch: np.ndarray, gains: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One-sweep transition matrix on the free nodes for every clamped input row.

    ``x_batch`` has one row per input and one +-1 column per input index named in
    ``clamped_input``. ``gains`` (one per free node, optional) multiply each free
    node's field, as static beta jitter does. Returns ``(kernels, states)`` with
    kernels of shape (inputs, 2^n_free, 2^n_free).
    """
    free = model.free
    nf = len(free)
    if nf > MAX_EXACT_FREE:
        raise ValueError(f"exact enumeration is bounded to {MAX_EXACT_FREE} free nodes")
    x_batch = np.atleast_2d(np.asarray(x_batch, dtype=np.float64))
    slot = {n: k for k, n in enumerate(free)}
    clamp_pos = {c: k for k, c in enumerate(model.clamped)}
    states = free_states(nf)
    pair = np.zeros((nf, nf))
    ext = np.zeros((len(x_batch), nf))
    ext += model.bias[list(free)][None]
    for (u, v), w in zip(model.edges, model.weights, strict=True):
        if u in slot and v in slot:
            pair[slot[u], slot[v]] += w
            pair[slot[v], slot[u]] += w
        elif u in slot or v in slot:
            f, c = (u, v) if u in slot else (v, u)
            if c not in clamp_pos:
                raise ValueError("an edge leaves the model's free and clamped nodes")
            ext[:, slot[f]] += w * x_batch[:, model.clamped_input[clamp_pos[c]]]
    g = np.ones(nf) if gains is None else np.asarray(gains, dtype=np.float64)
    if g.shape != (nf,):
        raise ValueError("one gain per free node")
    colours = np.array([model.node_colours[n] for n in free])
    if model.free_colour_violations():
        raise ValueError("two coupled free nodes share a colour")
    out = []
    for e in ext:
        kernel = np.eye(1 << nf)
        for c in (model.first_colour, 1 - model.first_colour):
            block = colours == c
            field = g[None, :] * (e[None, :] + states @ pair)
            p_up = expit(2.0 * field)
            prob = np.where(states[None, :, :] > 0, p_up[:, None, :], 1 - p_up[:, None, :])
            keep = np.all(states[:, None, ~block] == states[None, :, ~block], axis=2)
            kernel = kernel @ np.where(keep, np.prod(prob[:, :, block], axis=2), 0.0)
        out.append(kernel)
    return np.stack(out), states


def propagate(initial: np.ndarray, kernels: np.ndarray, sweeps: int) -> np.ndarray:
    """Law after ``sweeps`` sweeps for each input: (inputs, states)."""
    law = np.broadcast_to(np.asarray(initial, dtype=np.float64), kernels.shape[:2])
    for _ in range(sweeps):
        law = np.einsum("is,ist->it", law, kernels)
    return np.asarray(law)
