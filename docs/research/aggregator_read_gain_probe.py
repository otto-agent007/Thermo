"""Information-per-read probe for a directed aggregator p-bit.

Asks: if one p-bit listens one-way to k independent model sites (each +1 with
probability theta) and fires with probability sigmoid(2 * (w * sum(s) + b)),
how much Fisher information about theta does one read of it carry, relative to
one read of a single site?  Exact, no sampling.  Exploration, not evidence.

Within-block correlations are NOT modelled here; see proposal 1 in
docs/research/2026-11-12-mechanism-proposals.md.

Run from a Thermo checkout:
    uv run python docs/research/aggregator_read_gain_probe.py
Pure NumPy/SciPy.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def fire_prob(theta: float, k: int, w: float, b: float) -> float:
    """P(aggregator = +1) for k independent sites with P(site = +1) = theta."""
    up = np.arange(k + 1)
    spin_sum = 2 * up - k
    return float((binom.pmf(up, k, theta) / (1 + np.exp(-2 * (w * spin_sum + b)))).sum())


def gain(theta: float, k: int, w: float, b: float, d: float = 1e-5) -> float:
    """Fisher information per aggregator read over that of one site read."""
    p = fire_prob(theta, k, w, b)
    dp = (fire_prob(theta + d, k, w, b) - fire_prob(theta - d, k, w, b)) / (2 * d)
    return dp * dp / (p * (1 - p)) * theta * (1 - theta)


def two_level_gain(theta: float, w1: float, w2: float, d: float = 1e-5) -> float:
    """16 aggregators of 16 sites each feeding one top unit (256 sites per read)."""

    def top(t: float) -> float:
        return fire_prob(fire_prob(t, 16, w1, 0.0), 16, w2, 0.0)

    p = top(theta)
    dp = (top(theta + d) - top(theta - d)) / (2 * d)
    return dp * dp / (p * (1 - p)) * theta * (1 - theta)


if __name__ == "__main__":
    k = 16
    print("gain over one site read; threshold off-centre by 0, 1, 2, 4 sites")
    for theta in (0.5, 0.2):
        for w in (0.125, 0.25, 0.5, 1, 2):
            centred = -w * (2 * k * theta - k)
            row = [gain(theta, k, w, centred + 2 * w * off) for off in (0, 1, 2, 4)]
            print(f"theta={theta} w={w:<5} {np.round(row, 2)}")
    print("two-level tree at theta=0.5 (maximum 256)")
    for w1, w2 in ((0.5, 0.5), (1, 0.5), (2, 1), (2, 2)):
        print(f"w1={w1} w2={w2} gain={two_level_gain(0.5, w1, w2):.1f}")
