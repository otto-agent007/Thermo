"""Exact majority-estimate floors for P-0008 (`dtm-exact-posterior-lattice`).

Companion to ``dtm_exact_posterior_lattice_probe.py``. Exploration, not evidence.
For the probe's own test images (same seeds, same salt), the error of an
n-draw pixelwise majority (ties to +1) of independent draws from the exact
posterior, against the Bayes MPM error. This is the best any sampler can do
with an n-draw estimate, so a tolerance below it cannot be met by sampling.
Pure arithmetic on the exact posterior (``exact_reference``), no sampler.

Run:
    OPENBLAS_NUM_THREADS=1 uv run python \\
      docs/research/dtm_exact_posterior_lattice_floors.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).parent))
import dtm_exact_posterior_lattice_probe as probe  # noqa: E402

DRAWS = (1, 4, 16, 32, 64, 128, 256)
N_TEST = {"t4_h05_q035": 1000, "smooth8_h05_q035": 500}  # the probe's test sets


def floor(p_up, x, n):
    """Expected n-draw majority error, ties to +1 (as the probe's estimate)."""
    p_pred_up = binom.sf(int(np.ceil(n / 2)) - 1, n, p_up)  # P(#up >= n/2)
    return float(np.where(x > 0, 1.0 - p_pred_up, p_pred_up).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    out = {
        "label": "exploration, not evidence",
        "evidence_class": "exact_reference",
        "families": [],
    }
    for name, n_test in N_TEST.items():
        J, h, q = probe.family(name)
        mix, x, y = probe.test_set(name, n_test, 1)
        post = mix.posterior(y, q)
        bayes = float((np.where(post > 0.5, 1.0, -1.0) != x).mean())
        rows = [
            {"draws": n, "error": floor(post, x, n), "gap_to_bayes": floor(post, x, n) - bayes}
            for n in DRAWS
        ]
        out["families"].append(
            {"name": name, "n_test": n_test, "bayes_mpm": bayes, "majority_floors": rows}
        )
        for r in rows:
            print(name, r["draws"], round(r["gap_to_bayes"], 4), flush=True)
    out["cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
