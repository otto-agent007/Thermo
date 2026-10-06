"""Exploratory probe: can a short pilot detect that ordinary Gibbs will trap on a Potts target?

EXPLORATION ONLY. Output of this script is not recorded evidence and must not
be cited as a study result. Potts stage B found that tempering plus the label
symmetry estimator wins when ordinary chains trap (beta 16) and loses when
they mix (beta 8), but it identified trapping only against exact references.
This probe checks, before any protocol is drafted, whether reference-free
signals from a short pilot of the five independent cold chains predict which
sampler is better at T = 4096, and how short the pilot can be.

Signals (all invariant under label permutation, so symmetric modes do not
count as disagreement):
- edge_rhat: largest over edges of the between/within-chain variance ratio
  (Gelman-Rubin R-hat) of the indicator c_a == c_b across the five chains;
- score_rhat: the same for the chain score sum_e K_e delta(c_a, c_b);
- sym_spread: mean pairwise TV between the five chains' symmetrized joint
  histograms of sites 0-3;
- pair_rhat: largest R-hat over all 66 site pairs of c_i == c_j (edges alone
  cannot separate colour classes: in near-proper colourings every edge
  disagrees in every chain);
- spread_ratio: sym_spread divided by the mean within-chain spread between
  each chain's first and second half, which cancels most sample-count noise.

Uses graph seeds 910-915 at beta 8, 12 and 16, which no protocol may reuse.

Run: JAX_PLATFORMS=cpu uv run python docs/research/potts_trapping_probe.py
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from thermo_lab import potts_symmetry_tempering as b

SEEDS = (910, 911, 912, 913, 914, 915)
BETAS = (8.0, 12.0, 16.0)
PILOTS = (64, 256, 1024)
HORIZON = 4096


def make_request() -> dict:
    targets = []
    for seed in SEEDS:
        edges, weights = b.cubic_graph(b.N_SITES, seed)
        for beta in BETAS:
            targets.append(
                {
                    "id": f"g{seed}-b{beta:g}",
                    "seed": seed,
                    "beta": beta,
                    "edges": [list(e) for e in edges],
                    "couplings": [-w / 5 for w in weights],
                    "blocks": b.three_colouring(b.N_SITES, edges),
                }
            )
    return {
        "n": b.N_SITES,
        "targets": targets,
        "root_seed": 99_000,
        "trials": 16,
        "samplers": list(b.SAMPLERS),
    }


def rhat(x: np.ndarray) -> np.ndarray:
    """x: (chains, draws, k). Classic R-hat per column; inf when chains are frozen apart."""
    n = x.shape[1]
    means = x.mean(axis=1)
    within = x.var(axis=1, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    pooled = (n - 1) / n * within + between / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(pooled / within)
    r = np.where(within > 0, r, np.where(between > 0, np.inf, 1.0))
    return r


def signals(chains: np.ndarray, edges, couplings, s: np.ndarray) -> dict:
    """chains: (5, draws, n) labels after burn-in."""
    a = np.array([e[0] for e in edges])
    c = np.array([e[1] for e in edges])
    agree = (chains[:, :, a] == chains[:, :, c]).astype(float)
    score = agree @ np.asarray(couplings)
    hists = []
    for chain in chains:
        h = np.bincount(b.joint_codes(chain), minlength=b.Q**b.JOINT_SITES) / len(chain)
        hists.append(h @ s)
    spread = np.mean(
        [0.5 * np.abs(hists[i] - hists[j]).sum() for i, j in itertools.combinations(range(5), 2)]
    )
    half = chains.shape[1] // 2
    within = []
    for chain in chains:
        h1, h2 = (
            np.bincount(b.joint_codes(part), minlength=b.Q**b.JOINT_SITES) / len(part) @ s
            for part in (chain[:half], chain[half : 2 * half])
        )
        within.append(0.5 * np.abs(h1 - h2).sum())
    pairs = np.array(list(itertools.combinations(range(chains.shape[2]), 2)))
    pair_agree = (chains[:, :, pairs[:, 0]] == chains[:, :, pairs[:, 1]]).astype(float)
    return {
        "edge_rhat": float(rhat(agree).max()),
        "score_rhat": float(rhat(score[:, :, None])[0]),
        "sym_spread": float(spread),
        "pair_rhat": float(rhat(pair_agree).max()),
        "spread_ratio": float(spread / max(np.mean(within), 1e-12)),
    }


def sym_error(kept: np.ndarray, length: int, joint: np.ndarray, s: np.ndarray) -> np.ndarray:
    window = kept[:, length // 4 : length].reshape(kept.shape[0], -1, b.N_SITES).astype(np.int64)
    out = []
    for trial in window:
        h = np.bincount(b.joint_codes(trial), minlength=b.Q**b.JOINT_SITES) / len(trial)
        out.append(0.5 * np.abs(h @ s - joint).sum())
    return np.array(out)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def main() -> None:
    request = make_request()
    s = b.symmetry_matrix()
    t0 = time.perf_counter()
    refs = b.exact_references(request)
    print(f"exact side {time.perf_counter() - t0:.1f}s")
    rows = []
    for t_index, target in enumerate(request["targets"]):
        joint = np.array(refs[target["id"]]["joint"])
        kept = {}
        for sampler in ("independent", "tempering"):
            fn, r = b.compile_sampler(target, sampler, HORIZON)
            init_keys, sample_keys = b.trial_keys(request, t_index, sampler)
            kept[sampler] = np.asarray(fn(sample_keys, b.initial_labels(init_keys, r))[0])
        err_ind = sym_error(kept["independent"], HORIZON, joint, s)
        err_tmp = sym_error(kept["tempering"], HORIZON, joint, s)
        for trial in range(request["trials"]):
            row = {
                "target": target["id"],
                "beta": target["beta"],
                "err_ind": err_ind[trial],
                "err_tmp": err_tmp[trial],
            }
            for pilot in PILOTS:
                chains = kept["independent"][trial, pilot // 4 : pilot].transpose(1, 0, 2)
                for k, v in signals(chains, target["edges"], target["couplings"], s).items():
                    row[f"{k}@{pilot}"] = v
            rows.append(row)
        print(
            f"{target['id']:10s} beta {target['beta']:4.0f}  indep+sym {err_ind.mean():.3f}  "
            f"temp+sym {err_tmp.mean():.3f}  "
            + "  ".join(
                f"{k}@{p} {np.median([x[f'{k}@{p}'] for x in rows[-16:]]):.3g}"
                for p in (64, 256)
                for k in ("pair_rhat", "spread_ratio")
            )
        )
    # per trial: does the signal predict that tempering+sym beats independent+sym at T=4096?
    names = ("edge_rhat", "score_rhat", "sym_spread", "pair_rhat", "spread_ratio")
    for label_name, labels in (
        ("tempering+sym better", np.array([x["err_tmp"] < x["err_ind"] for x in rows])),
        ("independent+sym fails 0.05", np.array([x["err_ind"] > 0.05 for x in rows])),
    ):
        print(f"\ntrials where {label_name} at T={HORIZON}: {labels.sum()} / {len(labels)}")
        for pilot in PILOTS:
            line = []
            for k in names:
                scores = np.array([x[f"{k}@{pilot}"] for x in rows])
                scores = np.where(np.isfinite(scores), scores, 1e9)
                line.append(f"{k} {auc(scores, labels):.3f}")
            print(f"  AUC @{pilot}: " + "  ".join(line))
    # per target: median signal against mean outcome
    print("\nper target (mean errors, median pilot signals at 256):")
    for target in request["targets"]:
        sub = [x for x in rows if x["target"] == target["id"]]
        print(
            f"  {target['id']:10s} ind {np.mean([x['err_ind'] for x in sub]):.3f} "
            f"tmp {np.mean([x['err_tmp'] for x in sub]):.3f}  "
            f"edge_rhat {np.median([x['edge_rhat@256'] for x in sub]):.2f}  "
            f"score_rhat {np.median([x['score_rhat@256'] for x in sub]):.2f}  "
            f"sym_spread {np.median([x['sym_spread@256'] for x in sub]):.3f}  "
            f"pair_rhat {np.median([x['pair_rhat@256'] for x in sub]):.2f}  "
            f"spread_ratio {np.median([x['spread_ratio@256'] for x in sub]):.2f}"
        )


if __name__ == "__main__":
    main()
