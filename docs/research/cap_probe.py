"""Leakage-capacity probe for Thermo's five-spin PAsymSwap kernel.

Asks: under a field/coupling cap c, how low can a single nine-parameter kernel
push leakage on empty edges (input 00) while holding leakage on occupied edges
(inputs 01/10) below eps?  Evaluated at equilibrium and at K uniform-reset
hidden-then-output Gibbs sweeps, matching thermo_lab.thermodynamic_kernel.

Run from a Thermo checkout:  uv run python docs/research/cap_probe.py
Pure NumPy/SciPy; the repo import is used only for the cross-check.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# Word spins in canonical WORD_ORDER ((0,0),(0,1),(1,0),(1,1)) with s = 2b - 1.
S = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
H = np.array([-1.0, 1.0])


def conditional(p: np.ndarray, K: int | None) -> np.ndarray:
    """4x4 input-major output conditional; K=None means equilibrium."""
    hh, h0, h1, j00, j01, j10, j11, jh0, jh1 = p
    J = np.array([[j00, j01], [j10, j11]])  # J[input bit, output bit]
    a = np.array([h0, h1]) + S @ J  # (input, output) field from inputs
    jh = np.array([jh0, jh1])
    lo = a[:, None, :] + H[None, :, None] * jh  # (input, hidden, output bit)
    if K is None:
        lw = hh * H[None, :, None] + (S[None, None] * lo[:, :, None, :]).sum(-1)
        w = np.exp(lw - lw.max(axis=(1, 2), keepdims=True)).sum(1)
        return w / w.sum(1, keepdims=True)
    p_h = 1 / (1 + np.exp(-2 * (hh + S @ jh)))  # p(hidden=+1 | outputs)
    Ph = np.stack([1 - p_h, p_h], 1)
    pu = 1 / (1 + np.exp(-2 * lo))
    Po = np.prod(np.where(S[None, None] > 0, pu[:, :, None, :], 1 - pu[:, :, None, :]), -1)
    T = np.einsum("oh,xhp->xohp", Ph, Po)
    T = np.broadcast_to(T[:, None], (4, 2, 4, 2, 4)).reshape(4, 8, 8)
    d = np.full((4, 8), 1 / 8)  # uniform reset over (hidden, outputs)
    for _ in range(K):
        d = np.einsum("xs,xst->xt", d, T)
    return d.reshape(4, 2, 4).sum(1)


def leaks(C: np.ndarray) -> np.ndarray:
    """Particle-count leakage for inputs 00, 01, 10."""
    return np.array([1 - C[0, 0], C[1, 0] + C[1, 3], C[2, 0] + C[2, 3]])


def cross_check() -> float:
    from thermo_lab.thermodynamic_kernel import (
        KernelParameters,
        equilibrium_conditional,
        finite_horizon_conditional,
    )

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        p = rng.uniform(-2, 2, 9)
        P = KernelParameters(tuple(float(v) for v in p))
        worst = max(worst, np.abs(conditional(p, None) - equilibrium_conditional(P)).max())
        for K in (1, 4, 30):
            ref = finite_horizon_conditional(P, [K])[K]
            worst = max(worst, np.abs(conditional(p, K) - ref).max())
    return worst


def min_empty_leak(cap: float, eps: float, K: int | None, starts: int, seed: int) -> float:
    """Best empty-edge leakage found with occupied-edge leakage <= eps."""
    rng = np.random.default_rng(seed)
    best = np.inf
    for i in range(starts):
        x0 = rng.choice([-cap, cap], 9).astype(float) if i % 2 else rng.uniform(-cap, cap, 9)

        def f(p: np.ndarray) -> float:
            L = leaks(conditional(p, K))
            excess = max(0.0, np.log(max(L[1], L[2])) - np.log(eps))
            return np.log(max(L[0], 1e-300)) + 1e3 * excess**2

        r = minimize(f, x0, method="L-BFGS-B", bounds=[(-cap, cap)] * 9)
        L = leaks(conditional(r.x, K))
        if max(L[1], L[2]) <= eps * 1.01 and L[0] < best:
            best = L[0]
    return best


if __name__ == "__main__":
    print(f"cross-check vs thermo_lab (max abs diff): {cross_check():.1e}")
    print("cap  K    eps      min empty-edge leak   product")
    for cap in (2, 3, 4, 6):
        for K in (None, 4):
            for eps in (1e-2, 1e-3):
                v = min_empty_leak(cap, eps, K, starts=60, seed=cap)
                print(f"{cap:<4} {str(K):<4} {eps:<8g} {v:<21.3e} {v * eps:.1e}", flush=True)
