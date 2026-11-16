"""Replay the pre-registered comparison for the Z1T noise quality follow-up.

Reading rule: docs/research/2026-11-15-z1t-noise-quality-followup.md (PR #170).
Inputs are committed byte copies only: this directory's two runs and the first
run's snapshot in ../2026-11-15-z1t-noise-results/. No model is evaluated.
Evidence class of every number: software_simulation (exploration).

    python docs/research/2026-11-15-z1t-noise-quality-results/compare.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / "2026-11-15-z1t-noise-results"
RUNS = ("noscale", "distill")
WINDOW_TOL = 1e-3  # rule: step-0 clean per-window losses must agree to 1e-3 nats


def per(path: Path) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, np.float64) for k, v in json.loads(path.read_text())["per"].items()}


def mean_sem(x: np.ndarray) -> list[float]:
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))]


def compare() -> dict:
    s0 = per(FIRST / "start.json")
    first_noisy = per(FIRST / "noisy" / "final.json")
    first_ctrl = per(FIRST / "control" / "final.json")
    out: dict = {"evidence_class": "software_simulation", "runs": {}}
    for run in RUNS:
        st = per(HERE / run / "start.json")
        fin = per(HERE / run / "noisy" / "final.json")
        check = {
            k: float(np.abs(st[k] - s0[k]).max())
            for k in ("owt_clean", "wt103_clean", "owt_N32", "wt103_N32")
        }
        r: dict = {
            "step0_max_abs_diff": check,
            "paired_valid": all(check[k] <= WINDOW_TOL for k in ("owt_clean", "wt103_clean")),
        }
        for ds in ("owt", "wt103"):
            c0, n0 = s0[f"{ds}_clean"], s0[f"{ds}_N32"]
            fc, fn = fin[f"{ds}_clean"], fin[f"{ds}_N32"]
            r[ds] = {
                "final_clean": mean_sem(fc),
                "final_N32": mean_sem(fn),
                "recovery": float((n0.mean() - fn.mean()) / (n0.mean() - c0.mean())),
                "cost_vs_start": mean_sem(fc - c0),
                "clean_minus_first_noisy": mean_sem(fc - first_noisy[f"{ds}_clean"]),
                "N32_minus_first_noisy": mean_sem(fn - first_noisy[f"{ds}_N32"]),
                "N32_minus_first_control": mean_sem(fn - first_ctrl[f"{ds}_N32"]),
            }
        r["owt"]["curve"] = {f"N{n}": mean_sem(fin[f"owt_N{n}"]) for n in (128, 512, 2048)}
        summary = json.loads((HERE / run / "summary.json").read_text())["recovered_fraction"]
        for ds in ("owt", "wt103"):
            assert abs(summary[f"{ds}: noisy"] - r[ds]["recovery"]) < 1e-12, (run, ds)
        out["runs"][run] = r
    return out


if __name__ == "__main__":
    result = compare()
    target = HERE / "comparison.json"
    text = json.dumps(result, indent=2) + "\n"
    if target.exists() and target.read_text() != text:
        raise SystemExit("comparison.json differs from this replay")
    target.write_text(text)
    print(text)
