"""Post-hoc spread check for the P-0010 probe (written after probe.json existed).

Exploration, not evidence. Recomputes four rows of ``probe.py`` (same functions,
same archived inputs) and reports the per-image spread of the MPM gap: the
standard error of the mean over the 1,000 archived test images, and a 95
percent bootstrap interval (2,000 resamples, seed 20261105). The gaps are exact
for these images; the spread says how far another 1,000 images from the same
family could move them. It does not change the probe's point-estimate reading,
which follows P-0008's stage E rule.

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
      uv run python docs/research/rounding_rule_transfer_probe/posthoc_spread.py \\
      --output docs/research/rounding_rule_transfer_probe/posthoc_spread.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from thermo_lab import dtm_exact_posterior_lattice as dl
from thermo_lab.constraints.quantize import cap_for

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p0010_probe", HERE / "probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

ROWS = (("nearest", 4), ("nearest", 5), ("diffuse_class", 4), ("diffuse_class", 5))
SEED = 20261105
RESAMPLES = 2000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    arch = probe.load_archive()
    req = arch["request"]
    ev = arch["study"]["evaluation"]["t4_h05_q035"]
    fam = req["families"][0]
    mix, x, y, _ = dl.images(req, 0, "test")
    ne = req["n_test_exact"]
    x, y = x[:ne], y[:ne]
    _, bw = dl.bayes(mix, x, y, fam["q"])
    base, units, model, nfree, kinds = dl.placed(req, 0, ev["carried_layout"], ev["w_star"])
    wts = np.asarray(model.weights, dtype=float)
    fields = np.asarray(model.bias[:nfree], dtype=float)
    caps = cap_for("split", wts, fields, 8)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, ne, size=(RESAMPLES, ne))
    out = {"label": "post-hoc exploration, not evidence", "seed": SEED, "rows": {}}
    per_image = {}
    for rule, bits in ROWS:
        wq, fq = probe.rounded(rule, bits, wts, fields, caps)
        pg = probe.collapsed_general(base, units, model, kinds, wq, fq)
        pe = probe.exact_equilibrium_general(pg, y, prune=req["prune"])
        mpm = (np.where(pe > 0.5, 1.0, -1.0) != x).sum(1)
        d = (mpm - bw) / dl.N
        per_image[(rule, bits)] = d
        boots = d[idx].mean(1)
        out["rows"][f"{rule}/{bits}"] = {
            "mpm_gap": float(d.mean()),
            "sem": float(d.std(ddof=1) / np.sqrt(ne)),
            "bootstrap_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        }
        print(rule, bits, out["rows"][f"{rule}/{bits}"], flush=True)
    diff = per_image[("diffuse_class", 4)] - per_image[("nearest", 5)]
    out["diffuse4_minus_nearest5"] = {
        "mean": float(diff.mean()),
        "sem": float(diff.std(ddof=1) / np.sqrt(ne)),
    }
    Path(args.output).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
