"""Supplementary exact scan for P-0008 round 2 (exploration, not evidence).

The round-1 probe cut the `smooth8` bias-design b0 scan at [-10, 0], where it
was still falling, and tuned one-hot only on the float error. This scan
settles both before a protocol fixes the design:

- bias design, centered, b0 in [-24, -11] (development set, as round 1);
- one-hot, centered, lambda in {12, 16, 24}, scored at float and 8 bits under
  `split` on the development set (round 1 chose lambda on float error only).

Every number is `exact_reference` (float64 enumeration over the 2^8 hidden
states times the transfer matrix). Functions are imported from the round-1
probe unchanged. Run:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
    uv run python docs/research/dtm_exact_posterior_lattice_scan2.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dtm_exact_posterior_lattice_probe as probe  # noqa: E402

NAME = "smooth8_h05_q035"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-dev", type=int, default=300)
    a = ap.parse_args()
    t0 = time.process_time()
    J, h, q = probe.family(NAME)
    mix, xd, yd = probe.test_set(NAME, a.n_dev, 2)  # round 1's development salt
    post = mix.posterior(yd, q)
    bayes = (np.where(post > 0.5, 1.0, -1.0) != xd).mean(1)
    out = {
        "label": "exploration, not evidence",
        "evidence_class": "exact_reference",
        "family": NAME,
        "n_dev": a.n_dev,
        "bayes_mpm_dev": float(bayes.mean()),
        "bias_centered": [],
        "onehot_centered": [],
    }
    for b0 in np.arange(-24.0, -10.5, 1.0):
        p = probe.build(J, h, q, "bias", "centered", b0=float(b0))
        pu, wts, S = probe.exact_equilibrium(p, yd)
        on = (S > 0).sum(1)
        row = {
            "b0": float(b0),
            **probe.summarize(pu, xd, bayes),
            "hidden_mass_none_on": float(wts[:, on == 0].sum(1).mean()),
            "hidden_mass_one_on": float(wts[:, on == 1].sum(1).mean()),
        }
        out["bias_centered"].append(row)
        print("bias", b0, round(row["mpm_gap_to_bayes"], 4), flush=True)
    for lam in (12.0, 16.0, 24.0):
        for bits in (None, 8):
            p = probe.quantize(probe.build(J, h, q, "onehot", "centered", lam=lam), bits)
            pu, wts, S = probe.exact_equilibrium(p, yd)
            on = (S > 0).sum(1)
            row = {
                "lambda": lam,
                "bits": bits,
                **probe.summarize(pu, xd, bayes),
                "hidden_mass_one_on": float(wts[:, on == 1].sum(1).mean()),
                "caps": p.get("caps"),
            }
            out["onehot_centered"].append(row)
            print("onehot", lam, bits, round(row["mpm_gap_to_bayes"], 4), flush=True)
    out["cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")


if __name__ == "__main__":
    main()
