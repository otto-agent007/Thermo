"""Static per-node beta jitter: one gain per node, fixed for the whole request."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from thermo_lab.constraints.request import ConstraintRequest


def static_gains(words: Sequence[int] | None, sizes: Sequence[int], jitter: float) -> list:
    """One ``default_rng(words)``, then ``uniform(1 - j, 1 + j, n)`` per group in order.

    This is ``am_coupling_bits.gains``' draw order, so its words reproduce its gains
    bitwise. Jitter 0 gives unit gains and draws nothing.
    """
    if not 0.0 <= jitter < 1.0:
        raise ValueError("jitter must lie in [0, 1)")
    if jitter == 0:
        return [np.ones(int(n)) for n in sizes]
    if not words:
        raise ValueError("jitter > 0 needs seed words")
    rng = np.random.default_rng([int(w) for w in words])
    return [rng.uniform(1 - jitter, 1 + jitter, int(n)) for n in sizes]


def request_gains(request: ConstraintRequest, sizes: Sequence[int]) -> list:
    """Gains declared by a request (unit gains when ``beta_jitter`` is relaxed)."""
    return static_gains(request.jitter_words, sizes, request.jitter)
