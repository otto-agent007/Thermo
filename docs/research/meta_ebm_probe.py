"""Exploratory probe (not evidence): d=12 three-body meta-EBM, exact cap sweep.

Usage: uv run python docs/research/meta_ebm_probe.py SEED [--prefactor] [--mixing-only]

--prefactor applies 1/2 and 1/3! to unordered pair and triple sums (reading B in
2026-09-26-meta-ebm-target-reading.md); the default is unit weights (reading A).
The Dobrushin scan uses float32 and six fit starts; it informs the M5a protocol
and is not recorded evidence.
"""

import itertools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

D = 12
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(seed)
pairs = [tuple(p) for p in rng.choice(list(itertools.combinations(range(D), 2)), 18, replace=False)]
triples = [
    tuple(t) for t in rng.choice(list(itertools.combinations(range(D), 3)), 20, replace=False)
]
W1 = rng.normal(0, 0.6, D)
W2 = rng.normal(0, 0.6, len(pairs))
W3 = rng.normal(0, 0.6, len(triples))
if "--prefactor" in sys.argv:  # reading B: 1/2 and 1/3! applied to unordered sums
    W2, W3 = W2 / 2, W3 / 6

# States: row index bits -> spins in {-1,+1}; site n is bit n.
S = np.array(list(itertools.product([-1, 1], repeat=D)))[:, ::-1].astype(float)
energy = -(S @ W1)
for (m, n), w in zip(pairs, W2, strict=True):
    energy -= w * S[:, m] * S[:, n]
for (a, b, c), w in zip(triples, W3, strict=True):
    energy -= w * S[:, a] * S[:, b] * S[:, c]
pi = np.exp(-(energy - energy.min()))
pi /= pi.sum()


def theta_exact(n, X):
    """Half log-odds of site n given full-state rows X (x_n ignored)."""
    t = np.full(len(X), W1[n])
    for (a, b), w in zip(pairs, W2, strict=True):
        if n in (a, b):
            t += w * X[:, b if a == n else a]
    for tri, w in zip(triples, W3, strict=True):
        if n in tri:
            o = [s for s in tri if s != n]
            t += w * X[:, o[0]] * X[:, o[1]]
    return t


blankets = []
for n in range(D):
    nb = set()
    for p in pairs + triples:
        if n in p:
            nb |= set(p)
    nb.discard(n)
    blankets.append(sorted(nb))


def sweep(q, conds):
    """One systematic sweep; conds[n] = P(x_n=+1 | state) over all 4096 states."""
    q = q.copy()
    for n in range(D):
        bit = 1 << n
        idx0 = np.arange(len(q))[(np.arange(len(q)) & bit) == 0]
        mass = q[idx0] + q[idx0 | bit]  # marginalize x_n; blanket read from idx0 row
        p1 = conds[n][idx0]
        q[idx0], q[idx0 | bit] = mass * (1 - p1), mass * p1
    return q


def sig(z):
    return 1 / (1 + np.exp(-z))


def sp(z):
    return np.logaddexp(0, z)


ideal = [sig(2 * theta_exact(n, S)) for n in range(D)]

# Ideal sweep matrix: columns = sweep applied to unit vectors (dense 4096x4096).
t0 = time.time()
P = np.stack([sweep(e, ideal) for e in np.eye(len(S))])  # row x -> law after sweep
rows = P
rho_dob = 0.0
rows32 = rows.astype(np.float32)
for i in range(0, len(S), 32):
    block = rows32[i : i + 32]
    for j0 in range(i, len(S), 512):
        tv = 0.5 * np.abs(block[:, None, :] - rows32[None, j0 : j0 + 512, :]).sum(-1)
        rho_dob = max(rho_dob, tv.max())
ev = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
print(f"seed {seed}: Dobrushin rho0={rho_dob:.4f} SLEM={ev[1]:.4f} ({time.time() - t0:.0f}s)")
if "--mixing-only" in sys.argv:
    raise SystemExit
print(
    "blanket sizes",
    [len(b) for b in blankets],
    "n_h",
    [sum(n in t for t in triples) for n in range(D)],
)


def kernel_theta(p, Xb, nh):
    k = Xb.shape[1]
    J, h = p[:k], p[k]
    A = p[k + 1 : k + 1 + nh * k].reshape(nh, k)
    B = p[k + 1 + nh * k : k + 1 + nh * k + nh]
    b = p[k + 1 + nh * k + nh :]
    s = Xb @ A.T + b
    return Xb @ J + h + (0.5 * (sp(-2 * (s - B)) - sp(-2 * (s + B)))).sum(1)


def compile_site(n, cap, restarts=6):
    bl = blankets[n]
    Xb = np.array(list(itertools.product([-1, 1], repeat=len(bl))), float)
    full = np.zeros((len(Xb), D))
    full[:, bl] = Xb
    target = sig(2 * theta_exact(n, full))
    nh = sum(n in t for t in triples)
    k = len(bl)
    npar = k + 1 + nh * k + 2 * nh

    Xj, tj = jnp.asarray(Xb), jnp.asarray(target)

    def kl_j(p):
        k_ = Xj.shape[1]
        J, h = p[:k_], p[k_]
        A = p[k_ + 1 : k_ + 1 + nh * k_].reshape(nh, k_)
        B = p[k_ + 1 + nh * k_ : k_ + 1 + nh * k_ + nh]
        b = p[k_ + 1 + nh * k_ + nh :]
        s_ = Xj @ A.T + b
        th = (
            Xj @ J
            + h
            + (0.5 * (jax.nn.softplus(-2 * (s_ - B)) - jax.nn.softplus(-2 * (s_ + B)))).sum(1)
        )
        # KL(target || model) for Bernoulli with logit 2*theta, numerically stable
        lp1, lp0 = -jax.nn.softplus(-2 * th), -jax.nn.softplus(2 * th)
        return jnp.mean(tj * (jnp.log(tj) - lp1) + (1 - tj) * (jnp.log(1 - tj) - lp0))

    vg = jax.jit(jax.value_and_grad(kl_j))

    def kl(p):
        v, g = vg(jnp.asarray(p))
        return float(v), np.asarray(g, dtype=float)

    best = None
    r = np.random.default_rng(1000 + n)
    for _ in range(restarts):
        x0 = r.uniform(-min(cap, 1), min(cap, 1), npar)
        res = minimize(kl, x0, jac=True, method="L-BFGS-B", bounds=[(-cap, cap)] * npar)
        if best is None or res.fun < best.fun:
            best = res
    return bl, Xb, best.x, nh


slem = ev[1]
print(
    f"{'cap':>5} {'eps_bar':>8} {'bnd_dob':>8} {'bnd_slem':>8} {'plateau':>8}"
    f" {'r_slem':>6} {'t90':>4} {'site_err':>9}"
)
q0 = np.full(len(S), 1 / len(S))
for cap in (0.3, 0.5, 0.75, 1, 1.5, 2, 3, 6, 10):
    conds, eps = [], 0.0
    for n in range(D):
        bl, Xb, p, nh = compile_site(n, cap)
        # map every full state to its blanket row index
        code = np.zeros(len(S), int)
        for j, s in enumerate(bl):
            code |= ((S[:, s] > 0).astype(int)) << (len(bl) - 1 - j)
        c = sig(2 * kernel_theta(p, Xb, nh))[code]
        eps = max(eps, np.abs(c - ideal[n]).max())
        conds.append(c)
    q, qt, errs = q0.copy(), q0.copy(), []
    for _ in range(40):
        q, qt = sweep(q, ideal), sweep(qt, conds)
        errs.append(0.5 * np.abs(q - qt).sum())
    plateau = errs[-1]
    t90 = next(i + 1 for i, e in enumerate(errs) if abs(e - plateau) <= 0.1 * plateau + 1e-12)
    site_err = np.mean(np.abs(S.T @ qt - S.T @ pi))
    bound, bs = eps / (1 - rho_dob), eps / (1 - slem)
    print(
        f"{cap:5} {eps:8.4f} {bound:8.4f} {bs:8.4f} {plateau:8.4f}"
        f" {plateau / bs:6.2f} {t90:4d} {site_err:9.2e}"
    )
