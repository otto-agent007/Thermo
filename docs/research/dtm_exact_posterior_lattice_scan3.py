"""Supplementary exact check for P-0008 round 2: `smooth8` bias design at 8, 6, 4 bits.

Chooses b0 for the centered bias design on round 1's development set (fine
grid around scan 2's optimum, float), then scores that b0 on round 1's
500-image test set at float and 8, 6, 4 bits under `split`. Reports the MPM gap
and the n-draw majority gaps (16 and 64 draws, the owner's estimate size) of
the model's exact equilibrium. `exact_reference`, float64. Exploration, not
evidence. Run:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
    uv run python docs/research/dtm_exact_posterior_lattice_scan3.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dtm_exact_posterior_lattice_probe as probe  # noqa: E402

NAME = "smooth8_h05_q035"


def maj_gap(p_up, x, bayes_img, draws):
    p_pred_up = binom.sf(draws // 2 - 1, draws, p_up)
    wrong = np.where(x > 0, 1.0 - p_pred_up, p_pred_up).mean(1)
    return float((wrong - bayes_img).mean())


def score(p, x, y, bayes_img):
    pu, wts, S = probe.exact_equilibrium(p, y)
    on = (S > 0).sum(1)
    mpm = (np.where(pu > 0.5, 1.0, -1.0) != x).mean(1) - bayes_img
    return {
        "mpm_gap_to_bayes": float(mpm.mean()),
        "mpm_gap_se": float(mpm.std(ddof=1) / np.sqrt(len(x))),
        "maj16_gap_to_bayes": maj_gap(pu, x, bayes_img, 16),
        "maj64_gap_to_bayes": maj_gap(pu, x, bayes_img, 64),
        "hidden_mass_none_on": float(wts[:, on == 0].sum(1).mean()),
        "hidden_mass_one_on": float(wts[:, on == 1].sum(1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    J, h, q = probe.family(NAME)
    out = {
        "label": "exploration, not evidence",
        "evidence_class": "exact_reference",
        "family": NAME,
        "dev": [],
        "test": [],
    }
    mix, xd, yd = probe.test_set(NAME, 300, 2)
    bd = (np.where(mix.posterior(yd, q) > 0.5, 1.0, -1.0) != xd).mean(1)
    for b0 in (-15.5, -15.0, -14.5):
        row = {"b0": b0, **score(probe.build(J, h, q, "bias", "centered", b0=b0), xd, yd, bd)}
        out["dev"].append(row)
        print("dev", b0, round(row["mpm_gap_to_bayes"], 4), flush=True)
    best = min(out["dev"], key=lambda r: r["mpm_gap_to_bayes"])["b0"]
    out["chosen_b0"] = best
    mix, x, y = probe.test_set(NAME, 500, 1)
    post = mix.posterior(y, q)
    bt = (np.where(post > 0.5, 1.0, -1.0) != x).mean(1)
    out["bayes_mpm_test"] = float(bt.mean())
    out["bayes_maj64_floor_gap_test"] = maj_gap(post, x, bt, 64)
    for bits in (None, 8, 6, 4):
        p = probe.quantize(probe.build(J, h, q, "bias", "centered", b0=best), bits)
        row = {"b0": best, "bits": bits, "caps": p.get("caps"), **score(p, x, y, bt)}
        out["test"].append(row)
        print("test", bits, round(row["mpm_gap_to_bayes"], 4), flush=True)
    out["cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")


if __name__ == "__main__":
    main()
