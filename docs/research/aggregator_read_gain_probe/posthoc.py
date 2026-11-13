"""Post-hoc checks for the P-0012 probe (written after probe.json existed).

Exploration, not evidence. Reuses ``probe.py``'s functions on the same frozen,
hash-checked inputs. Three questions the declared probe raised but did not
answer, none of which changes a declared verdict:

1. Combined jitter bias. ``probe.json`` reports the worst single block. The
   round-robin estimator pools the nine blocks, so its bias is the root of the
   pooled, misspecified score equation (host model g = 1, device g = 0.9 or
   1.1). Computed for the primary geometry, rule and caps.
2. Bias-folded gain. If the tolerance is read as root-mean-square error 0.01,
   reads scale as 1 / (I (tol^2 - bias^2)); the baseline is unbiased.
3. Placement. On the archived map an aggregator in a free slot sees at most a
   few pore sites. The gain of an aggregator over exactly the sites one free
   slot can see (float, half-rate rule, g = 1).

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
      uv run python docs/research/aggregator_read_gain_probe/posthoc.py \\
      --output docs/research/aggregator_read_gain_probe/posthoc.json
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p0012_probe", HERE / "probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

FD = probe.FD
TOL = probe.TOL


def law_and_derivative(mu: float, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    law = probe.block_law(mu, mask)
    dlaw = (probe.block_law(mu + FD, mask) - probe.block_law(mu - FD, mask)) / (2 * FD)
    return law, dlaw


def rates(law: np.ndarray, k: int, w: float, b: float, g: float) -> np.ndarray:
    return expit(2 * g * (w * (2 * np.arange(k + 1) - k) + b))


def combined_bias(mu, masks, wq, biases, g) -> float:
    """Root of the pooled score with host g = 1; returns the loading bias."""
    k = probe.K
    actual = [
        float(probe.block_law(mu, m) @ rates(None, k, wq, b, g))
        for m, b in zip(masks, biases, strict=True)
    ]

    def score(m: float) -> float:
        total = 0.0
        for mask, b, pa in zip(masks, biases, actual, strict=True):
            law, dlaw = law_and_derivative(m, mask)
            s = rates(None, k, wq, b, 1.0)
            p, dp = float(law @ s), float(dlaw @ s)
            total += (pa - p) * dp / (p * (1 - p))
        return total

    lo, hi = mu - 1.0, mu + 1.0
    if score(lo) * score(hi) > 0:
        return float("nan")
    mh = brentq(score, lo, hi, xtol=1e-9)
    return probe.loading(mh) - probe.loading(mu)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if Path(args.output).exists():
        raise SystemExit("refusing to overwrite")
    t0 = time.process_time()
    rec = json.loads(gzip.decompress((HERE / "probe.json.gz").read_bytes()))
    masks = probe.blocks("columns")
    out = {"label": "post-hoc exploration, not evidence", "combined_bias": [], "bias_caps": {}}

    # Own bias cap per w (largest float half-rate |b| over blocks and points).
    for c in rec["results"]["columns"]["cells"]:
        if c["caps"] == "own" and c["precision"] == "float" and c["gain_g"] == 1.0:
            out["bias_caps"][str(c["w"])] = max(
                abs(b) for p in c["points"] for b in p["half_biases"]
            )

    # 1 and 2: pooled bias and bias-folded gain, primary cells.
    for c in rec["results"]["columns"]["cells"]:
        if c["caps"] != "own" or c["gain_g"] == 1.0:
            continue
        row = {"w": c["w"], "precision": c["precision"], "gain_g": c["gain_g"], "points": []}
        for p in c["points"]:
            bias = combined_bias(p["mu"], masks, c["w_programmed"], p["half_biases"], c["gain_g"])
            gain = p["rules"]["half"]["gain_rr"]
            folded = gain * (1 - (bias / TOL) ** 2) if abs(bias) < TOL else 0.0
            row["points"].append(
                {"mu": p["mu"], "loading_bias": bias, "gain": gain, "gain_rmse_folded": folded}
            )
        out["combined_bias"].append(row)
        print(
            row["w"],
            row["precision"],
            row["gain_g"],
            [(round(q["loading_bias"], 5), round(q["gain_rmse_folded"], 2)) for q in row["points"]],
            flush=True,
        )

    # 3: placement-limited aggregators on the archived map.
    ref = probe.apr.load_reference()
    coords, _, _ = ref.geometry(*probe.SHAPE)
    slots = {(int(x + 4 * z), int(y + z)): i for i, (x, y, z) in enumerate(coords)}
    xs = [s[0] for s in slots]
    ys = [s[1] for s in slots]
    seen = {}
    for sx in range(min(xs) - 5, max(xs) + 6):
        for sy in range(min(ys) - 5, max(ys) + 6):
            if (sx, sy) in slots:
                continue
            sites = tuple(
                sorted(
                    slots[(sx + dx, sy + dy)]
                    for dx, dy in probe.OFFSETS
                    if (sx + dx, sy + dy) in slots
                )
            )
            if sites:
                seen.setdefault(sites, (sx, sy))
    sizes = np.bincount([len(s) for s in seen])
    placement = {"distinct_site_sets_by_size": sizes.tolist(), "points": []}
    for p in rec["points"]:
        mu = p["mu"]
        dl = p["dL_dmu_fd"]
        entry = {
            "mu": mu,
            "B_rr": p["fisher_per_read"]["B_rr"],
            "B_best": p["fisher_per_read"]["B_best"],
        }
        for w in (0.5, 1.0):
            vals = {}
            for sites in seen:
                k = len(sites)
                mask = np.zeros(probe.NSITES, bool)
                mask[list(sites)] = True
                law, dlaw = law_and_derivative(mu, mask)
                spin = 2 * np.arange(k + 1) - k
                b = brentq(
                    lambda b, law=law, spin=spin, w=w: float(law @ expit(2 * (w * spin + b))) - 0.5,
                    -w * k - 40,
                    w * k + 40,
                    xtol=1e-13,
                )
                s = expit(2 * (w * spin + b))
                pp, dp = float(law @ s), float(dlaw @ s)
                vals[sites] = dp * dp / (pp * (1 - pp)) / dl**2
            kmax = max(len(s) for s in seen)
            top = [v for s, v in vals.items() if len(s) == kmax]
            entry[f"w{w}"] = {
                "best_set_gain_vs_best_site": max(vals.values()) / entry["B_best"],
                "mean_kmax_set_gain_vs_B_rr": float(np.mean(top)) / entry["B_rr"],
                "kmax": kmax,
                "n_kmax_sets": len(top),
            }
        placement["points"].append(entry)
        print("placement", mu, {k: v for k, v in entry.items() if k.startswith("w")}, flush=True)
    out["placement_limited"] = placement
    out["cpu_seconds"] = time.process_time() - t0
    Path(args.output).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("cpu", out["cpu_seconds"])


if __name__ == "__main__":
    main()
