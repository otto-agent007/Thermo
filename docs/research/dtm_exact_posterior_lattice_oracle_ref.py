"""Exact references for the oracle-start arm, P-0008 round 3 (exploration).

An oracle start sets every copy (and relay) of each unit to the generating
template's hidden state. If the copies stay there, the chain samples the
placed model **conditioned on that hidden state**, which carries information
the noisy image does not. Its error is then compared, not with Bayes, but with
the tie-limit model conditioned on the generating hidden state (and, for
scale, with the true posterior conditioned on the generating component).

Writes, per family, on the round-3 probe's 200 images:

- ``oracle_bayes_gap``: MPM error of p(x | y, c_true) minus Bayes MPM error;
- ``tie_limit_conditioned_maj64_gap``: the tie-limit 8-bit model's exact
  64-draw majority gap with the hidden units held at the oracle state;

for the index-5 and fewest-free-node index-4 layouts at w = 2. Both are
``exact_reference`` (float64). Run after
``dtm_exact_posterior_lattice_layout4.py``:

    uv run python docs/research/dtm_exact_posterior_lattice_oracle_ref.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dtm_exact_posterior_lattice_layout4 as l4  # noqa: E402
import dtm_exact_posterior_lattice_placed as placed  # noqa: E402
import dtm_exact_posterior_lattice_probe as probe  # noqa: E402

from thermo_lab.dtm_small_image import GridIsing  # noqa: E402

N, L, DRAWS = probe.N, probe.L, 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    out = {
        "label": "exploration, not evidence",
        "evidence_class": "exact_reference",
        "families": [],
    }
    for name, b0 in (("t4_h05_q035", -5.0), ("smooth8_h05_q035", -15.0)):
        J, h, q = probe.family(name)
        C = len(h)
        n_img = 200
        mix, x, y = probe.test_set(name, 1000 if C <= 4 else 500, 1)
        comp = np.random.default_rng([probe.PROBE_SEED, 1, list(probe.FAMILIES).index(name)])
        comp = comp.integers(0, C, len(x))[:n_img]
        x, y = x[:n_img], y[:n_img]
        post = mix.posterior(y, q)
        bayes = (np.where(post > 0.5, 1.0, -1.0) != x).mean(1)
        g = 0.5 * np.log((1 - q) / q)
        grid = GridIsing(L, J)
        pc, _ = grid.marginals(h[comp] + g * y)
        rec = {
            "family": name,
            "images": n_img,
            "oracle_bayes_gap": float(
                ((np.where(pc > 0.5, 1.0, -1.0) != x).mean(1) - bayes).mean()
            ),
            "oracle_bayes_maj64_gap": float((placed.maj_wrong(pc, x, DRAWS) - bayes).mean()),
            "layouts": {},
        }
        base = probe.build(J, h, q, "bias", "centered", b0=b0)
        for bname in ("index5", "index4"):
            basis, cands = l4.configs(bname, C)
            best = None
            for clamp, ureps, order in cands:
                units = l4.build_layout(basis, C, clamp, ureps, order)
                nf = sum(len(u["copies"]) + len(u["relay_nodes"]) for u in units)
                if best is None or nf < best[0]:
                    best = (nf, clamp, units)
            _, clamp, units = best
            model, nfree, kinds = l4.to_pairwise(units, basis, clamp, base, 2.0)
            p8, _ = l4.collapsed(base, units, model, nfree, kinds, 8)
            S = np.where(np.arange(C)[None] == comp[:, None], 1.0, -1.0)
            fields = p8["bp"][None] + S @ p8["W"].T + p8["g"] * y
            pm, _ = GridIsing(L, p8["J"]).marginals(fields)
            rec["layouts"][bname] = {
                "free_nodes": nfree,
                "tie_limit_conditioned_mpm_gap": float(
                    ((np.where(pm > 0.5, 1.0, -1.0) != x).mean(1) - bayes).mean()
                ),
                "tie_limit_conditioned_maj64_gap": float(
                    (placed.maj_wrong(pm, x, DRAWS) - bayes).mean()
                ),
            }
        print(json.dumps(rec), flush=True)
        out["families"].append(rec)
    out["total_cpu_s"] = time.process_time() - t0
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
