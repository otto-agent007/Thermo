"""Exploratory probe: can pairwise binary units emulate a categorical hidden unit?

EXPLORATION ONLY. Output of this script is not recorded evidence and must not
be cited as a study result. It checks, before an associative-memory protocol is
drafted, what a pairwise "one-hot" emulation of the dense memory's categorical
hidden unit recovers at equilibrium and how fast a Gibbs sampler gets there.

Model. N visible spins sigma in {-1,+1}, P binary hidden units h_mu in {0,1}:
    log w(sigma, h) = sum_mu h_mu a_mu(sigma) - lam * (sum_mu h_mu - 1)^2,
    a_mu = beta * xi_mu . sigma / sqrt(N).
(sum h - 1)^2 expands to pairwise inhibition 2 * lam between hidden pairs plus
a bias, so the model is pairwise. lam = 0 gives independent hidden units;
lam -> infinity keeps only one-hot h and the visible marginal becomes the dense
(log-sum-exp) memory exactly. The hidden layer is summed out exactly for any P
with elementary symmetric polynomials over the active count k:
    Z_h(sigma) = sum_k e_k(exp(a)) * exp(-lam (k - 1)^2).

Part 1: exact equilibrium recall of the missing bits given a cue, against the
categorical dense memory and Hebbian Hopfield. Part 2: recall after K Gibbs
sweeps (visible block, then hidden units one at a time, since inhibition
couples them), to see whether large lam traps the dynamics.

Pattern seeds 9700+ are probe-only; no protocol may reuse them.

Run: uv run python docs/research/am_emulation_probe.py
"""

from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.special import logsumexp

N = 24
LAMS = (0.0, 1.0, 4.0, 16.0, 64.0)
BETAS = (2.0, 4.0, 8.0, 16.0)


def patterns(p: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1, 1], size=(p, N)).astype(np.float64)


def completions(xi: np.ndarray, target: int, cue: int) -> tuple[np.ndarray, np.ndarray]:
    miss = np.array(list(itertools.product((-1.0, 1.0), repeat=N - cue)))
    full = np.concatenate([np.repeat(xi[target, :cue][None], len(miss), 0), miss], axis=1)
    return miss, full @ xi.T  # overlaps (states, P)


def log_elementary(overlaps: np.ndarray, beta: float) -> np.ndarray:
    """log e_k(exp(a)) for k = 0..P, by a log-space DP; independent of lam."""
    a = beta * overlaps / np.sqrt(N)
    s, p = a.shape
    log_e = np.full((s, p + 1), -np.inf)
    log_e[:, 0] = 0.0
    for mu in range(p):
        log_e[:, 1 : mu + 2] = np.logaddexp(
            log_e[:, 1 : mu + 2], a[:, mu : mu + 1] + log_e[:, 0 : mu + 1]
        )
    return log_e


def recall_from(lw: np.ndarray, miss: np.ndarray, truth: np.ndarray) -> float:
    w = np.exp(lw - lw.max())
    return float((w / w.sum()) @ (miss == truth).mean(axis=1))


def recall_table(xi, target, cue, beta):
    """Exact recall for Hopfield, every one-hot lam, and the categorical limit (k = 1 term)."""
    miss, ov = completions(xi, target, cue)
    truth = xi[target, cue:]
    out = {"hopfield": recall_from(beta / (2 * N) * (ov**2).sum(axis=1), miss, truth)}
    log_e = log_elementary(ov, beta)
    k = np.arange(log_e.shape[1])
    for lam in LAMS:
        out[f"lam={lam:g}"] = recall_from(
            logsumexp(log_e - lam * (k - 1.0) ** 2, axis=1), miss, truth
        )
    out["dense"] = recall_from(log_e[:, 1], miss, truth)
    return out


def part_one() -> None:
    ps = (8, 32, 128)
    names = ["hopfield", "dense"] + [f"lam={lam:g}" for lam in LAMS]
    for cue in (12, 8):
        print(f"\nPart 1: exact equilibrium recall, N={N}, cue {cue} bits, best beta in {BETAS}")
        print("  model        " + "  ".join(f"P={p:<9d}" for p in ps))
        t0 = time.perf_counter()
        best = {n: [] for n in names}
        for p in ps:
            acc = {(n, b): [] for n in names for b in BETAS}
            for d in range(2):
                xi = patterns(p, 9700 + 37 * p + d)
                for t in range(min(p, 3)):
                    for beta in BETAS:
                        for n, v in recall_table(xi, t, cue, beta).items():
                            acc[(n, beta)].append(v)
            for n in names:
                b = max(BETAS, key=lambda bb: np.mean(acc[(n, bb)]))
                best[n].append((float(np.mean(acc[(n, b)])), b))
        for n in names:
            print(f"  {n:12s} " + "  ".join(f"{a:.3f} @b{b:<4g}" for a, b in best[n]))
        print(f"  ({time.perf_counter() - t0:.0f}s)")


def gibbs_recall(xi, target, cue, beta, lam, sweeps, chains, rng, random_order=False):
    """Visible missing block, then hidden units sequentially; start: random missing bits, h = 0."""
    p = xi.shape[0]
    sig = np.repeat(xi[target][None], chains, 0)
    sig[:, cue:] = rng.choice([-1.0, 1.0], size=(chains, N - cue))
    h = np.zeros((chains, p))
    scale = beta / np.sqrt(N)
    for _ in range(sweeps):
        field = scale * (h @ xi[:, cue:])  # (chains, missing)
        sig[:, cue:] = np.where(rng.random(field.shape) < 1 / (1 + np.exp(-2 * field)), 1.0, -1.0)
        a = scale * (sig @ xi.T)  # (chains, P)
        order = rng.permutation(p) if random_order else range(p)
        for mu in order:
            s_other = h.sum(axis=1) - h[:, mu]
            logit = a[:, mu] - lam * (2 * s_other - 1)
            h[:, mu] = (rng.random(chains) < 1 / (1 + np.exp(-logit))).astype(float)
    return float((sig[:, cue:] == xi[target, cue:]).mean())


def part_two() -> None:
    cue, p, beta = 12, 32, 8.0
    rng = np.random.default_rng(1)
    xi = patterns(p, 9799)
    print(f"\nPart 2: recall after K sweeps, N={N}, cue {cue}, P={p}, beta={beta}, 2000 chains")
    table = recall_table(xi, 5, cue, beta)
    exact = {lam: table[f"lam={lam:g}"] for lam in LAMS}
    exact_dense = table["dense"]
    print(f"  exact dense (categorical) equilibrium: {exact_dense:.3f}")
    for order in (False, True):
        print(f"  hidden update order: {'random each sweep' if order else 'fixed 0..P-1'}")
        for lam in LAMS:
            vals = [
                gibbs_recall(xi, 5, cue, beta, lam, k, 2000, rng, order)
                for k in (1, 4, 16, 64, 256)
            ]
            print(
                f"    lam={lam:>4g}  exact eq {exact[lam]:.3f}   K=1,4,16,64,256: "
                + "  ".join(f"{v:.3f}" for v in vals)
            )


# --- Part 3: domain-wall and negative-bias hidden layers (added after a literature scan)


def dw_log_marginal(overlaps: np.ndarray, beta: float, j: float) -> np.ndarray:
    """Domain-wall chain z_1..z_{P-1}, fixed z_0 = +1 and z_P = -1, label indicator
    delta_mu = (z_{mu-1} - z_mu)/2, log w = sum_mu delta_mu a_mu + j sum_k z_k z_{k+1}.
    Exact sum over the chain by a transfer recursion, vectorized over visible states."""
    a = beta * overlaps / np.sqrt(N)
    s, p = a.shape
    vals = np.array([1.0, -1.0])
    # site field on z_k (k = 1..P-1): (a_{k+1} - a_k)/2 with a indexed from 1; boundary
    # terms a_1/2 (z_0 = +1) and a_P/2 (z_P = -1) are constants.
    const = 0.5 * (a[:, 0] + a[:, p - 1])
    # alpha over z_1 given z_0 = +1
    field = 0.5 * (a[:, 1:] - a[:, :-1])  # (s, P-1), field on z_1..z_{P-1}
    alpha = j * vals[None, :] + field[:, 0:1] * vals[None, :]  # (s, 2)
    for k in range(1, p - 1):
        trans = j * np.outer(vals, vals)  # (2, 2)
        alpha = (
            logsumexp(alpha[:, :, None] + trans[None], axis=1) + field[:, k : k + 1] * vals[None, :]
        )
    alpha = alpha + j * vals[None, :] * (-1.0)  # bond to z_P = -1
    return logsumexp(alpha, axis=1) + const


def bias_log_marginal(overlaps: np.ndarray, beta: float, theta: float) -> np.ndarray:
    a = beta * overlaps / np.sqrt(N)
    return np.logaddexp(0.0, a - theta).sum(axis=1)


def gibbs_dw(xi, target, cue, beta, j, sweeps, chains, rng):
    p = xi.shape[0]
    scale = beta / np.sqrt(N)
    sig = np.repeat(xi[target][None], chains, 0)
    sig[:, cue:] = rng.choice([-1.0, 1.0], size=(chains, N - cue))
    z = np.ones((chains, p + 1))  # z_0..z_P
    z[:, p] = -1.0
    z[:, 1:p] = rng.choice([-1.0, 1.0], size=(chains, p - 1))
    for _ in range(sweeps):
        delta = 0.5 * (z[:, :-1] - z[:, 1:])  # (chains, P) label indicators
        field = scale * (delta @ xi[:, cue:])
        sig[:, cue:] = np.where(rng.random(field.shape) < 1 / (1 + np.exp(-2 * field)), 1.0, -1.0)
        a = scale * (sig @ xi.T)
        for parity in (1, 0):
            ks = np.arange(1, p)
            ks = ks[ks % 2 == parity]
            h = j * (z[:, ks - 1] + z[:, ks + 1]) + 0.5 * (a[:, ks] - a[:, ks - 1])
            z[:, ks] = np.where(rng.random(h.shape) < 1 / (1 + np.exp(-2 * h)), 1.0, -1.0)
    return float((sig[:, cue:] == xi[target, cue:]).mean())


def gibbs_bias(xi, target, cue, beta, theta, sweeps, chains, rng):
    p = xi.shape[0]
    scale = beta / np.sqrt(N)
    sig = np.repeat(xi[target][None], chains, 0)
    sig[:, cue:] = rng.choice([-1.0, 1.0], size=(chains, N - cue))
    h = np.zeros((chains, p))
    for _ in range(sweeps):
        field = scale * (h @ xi[:, cue:])
        sig[:, cue:] = np.where(rng.random(field.shape) < 1 / (1 + np.exp(-2 * field)), 1.0, -1.0)
        a = scale * (sig @ xi.T) - theta
        h = (rng.random(a.shape) < 1 / (1 + np.exp(-a))).astype(float)
    return float((sig[:, cue:] == xi[target, cue:]).mean())


def part_three() -> None:
    cue, p, beta = 12, 32, 8.0
    xi = patterns(p, 9799)
    miss, ov = completions(xi, 5, cue)
    truth = xi[5, cue:]
    rng = np.random.default_rng(2)
    print(f"\nPart 3: domain-wall and negative-bias layers, N={N}, cue {cue}, P={p}, beta={beta}")
    dense_eq = recall_from(log_elementary(ov, beta)[:, 1], miss, truth)
    print(f"  exact dense (categorical) equilibrium: {dense_eq:.3f}")
    for j in (1.0, 4.0, 16.0, 64.0):
        eq = recall_from(dw_log_marginal(ov, beta, j), miss, truth)
        vals = [gibbs_dw(xi, 5, cue, beta, j, k, 2000, rng) for k in (1, 4, 16, 64, 256)]
        print(
            f"  domain-wall J={j:>4g}  exact eq {eq:.3f}   K=1,4,16,64,256: "
            + "  ".join(f"{v:.3f}" for v in vals)
        )
    for theta in (0.0, 8.0, 16.0, 24.0):
        eq = recall_from(bias_log_marginal(ov, beta, theta), miss, truth)
        vals = [gibbs_bias(xi, 5, cue, beta, theta, k, 2000, rng) for k in (1, 4, 16, 64, 256)]
        print(
            f"  bias theta={theta:>4g}  exact eq {eq:.3f}   K=1,4,16,64,256: "
            + "  ".join(f"{v:.3f}" for v in vals)
        )


if __name__ == "__main__":
    import sys

    parts = sys.argv[1:] or ["1", "2", "3"]
    if "1" in parts:
        part_one()
    if "2" in parts:
        part_two()
    if "3" in parts:
        part_three()
