"""Pairwise spin models the constraint layer places, quantizes, samples and prices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _frozen(values, name: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class PairwiseModel:
    """A pairwise +-1 spin model with optional lattice sites.

    Energy convention (THRML ``IsingEBM``): E(s) = -sum_e w_e s_u s_v - sum_i b_i s_i.
    ``sites`` gives each node's (u, v) lattice site, or ``None`` for an unplaced
    model, which must then carry ``colours`` (0 or 1 per node). ``clamped[k]``
    holds input column ``clamped_input[k]``. ``first_colour`` is the colour block
    updated first in each sweep.
    """

    sites: tuple[tuple[int, int], ...] | None
    edges: tuple[tuple[int, int], ...]
    weights: np.ndarray
    bias: np.ndarray
    free: tuple[int, ...]
    clamped: tuple[int, ...]
    clamped_input: tuple[int, ...]
    first_colour: int
    colours: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "edges", tuple((int(u), int(v)) for u, v in self.edges))
        set_(self, "weights", _frozen(self.weights, "weights"))
        set_(self, "bias", _frozen(self.bias, "bias"))
        set_(self, "free", tuple(int(i) for i in self.free))
        set_(self, "clamped", tuple(int(i) for i in self.clamped))
        set_(self, "clamped_input", tuple(int(i) for i in self.clamped_input))
        n = len(self.bias)
        if self.sites is not None:
            set_(self, "sites", tuple((int(a), int(b)) for a, b in self.sites))
            if len(self.sites) != n:
                raise ValueError("one site per node")
            derived = tuple((a + b) % 2 for a, b in self.sites)
            if self.colours is not None and tuple(self.colours) != derived:
                raise ValueError("colours disagree with the sites' (u + v) mod 2")
            set_(self, "colours", derived)
        elif self.colours is None:
            raise ValueError("an unplaced model needs a colour per node")
        else:
            set_(self, "colours", tuple(int(c) for c in self.colours))
        colours = self.node_colours
        if len(colours) != n or any(c not in (0, 1) for c in colours):
            raise ValueError("colours must be 0 or 1, one per node")
        if len(self.weights) != len(self.edges):
            raise ValueError("one weight per edge")
        seen = set()
        for u, v in self.edges:
            if u == v:
                raise ValueError("self-edge")
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError("edge endpoint out of range")
            key = (min(u, v), max(u, v))
            if key in seen:
                raise ValueError("duplicate edge")
            seen.add(key)
        if len(set(self.free)) != len(self.free) or len(set(self.clamped)) != len(self.clamped):
            raise ValueError("free and clamped nodes must be unique")
        if set(self.free) & set(self.clamped):
            raise ValueError("free and clamped nodes must be disjoint")
        if not all(0 <= i < n for i in self.free + self.clamped):
            raise ValueError("node index out of range")
        if len(self.clamped_input) != len(self.clamped):
            raise ValueError("one input column per clamped node")
        if self.first_colour not in (0, 1):
            raise ValueError("first_colour must be 0 or 1")

    @property
    def nodes(self) -> int:
        return len(self.bias)

    @property
    def placed(self) -> bool:
        return self.sites is not None

    @property
    def node_colours(self) -> tuple[int, ...]:
        if self.colours is None:  # unreachable after construction
            raise ValueError("model has no colours")
        return self.colours

    def degrees(self) -> np.ndarray:
        degree = np.zeros(self.nodes, dtype=np.int64)
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
        return degree

    def free_colour_violations(self) -> int:
        """Free-free edges inside one colour block (they break a two-colour sweep)."""
        free, colours = set(self.free), self.node_colours
        return sum(u in free and v in free and colours[u] == colours[v] for u, v in self.edges)

    def with_parameters(self, weights, bias) -> PairwiseModel:
        return PairwiseModel(
            sites=self.sites,
            edges=self.edges,
            weights=weights,
            bias=bias,
            free=self.free,
            clamped=self.clamped,
            clamped_input=self.clamped_input,
            first_colour=self.first_colour,
            colours=None if self.sites is not None else self.colours,
        )
