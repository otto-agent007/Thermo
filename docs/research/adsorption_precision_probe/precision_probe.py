"""Exploratory precision probe for the held adsorption model (loop P-0009).

Exploration, not a recorded study. Exact references only: no sampling, no
jitter, no cost. Run from the repository root:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      uv run python docs/research/adsorption_precision_probe/precision_probe.py \
      --output docs/research/adsorption_precision_probe/probe.json

Fixed before any rounding result was computed (committed with this script):

* Model: the frozen ``native-adsorption-eligibility`` lattice gas, imported
  from the archived helper ``native_adsorption_probe/adsorption_reference.py``
  (authenticated by SHA-256, not edited). Coupling cap 0.25 (exact at every
  bit count), field cap 1.9904094766263793 from the archived probe JSON.
* The three frozen chemical potentials are read from the archived JSON. On
  them the probe evaluates ONLY the archived control (nearest rounding on
  the family-wide split caps) at 6 and 8 bits, to reproduce the archive, plus
  nominal-model linear response (no rounding). Every other rule and bit count
  is evaluated on six development chemical potentials (target loadings
  0.25, 0.35, ..., 0.75), so the frozen points stay unseen for the protocol.
* Rules, bits 4 to 12, all on fields only (couplings 0.25 sit on the codebook):
  - ``nearest_family``: codebook nearest on the family cap (archived rule).
  - ``diffuse_site_family``: one-dimensional error diffusion in site order
    (z, then y, then x), carrying the full residual to the next site, then
    the family codebook.
  - ``diffuse_class_family``: the same, with the residual carried only within
    sites that share a nominal field value (a field class), in site order.
  - ``nearest_point``: nearest on a per-point split cap (max |field| of that
    chemical potential's own model).
* Tolerances: loading and working capacity (highest minus lowest point of a
  set) at most 0.01, aggregate pair occupancy at most 0.02, all absolute,
  against the full-precision exact model at the same chemical potential.
* Linear response: d(loading)/d(mu) and d(pair)/d(mu) by central difference
  (step 1e-5) on the nominal model. Raising every field by delta equals
  raising mu by 2*delta, and every site susceptibility is nonnegative in this
  ferromagnetic model (FKG), so (step/2) * 2 * dL/dmu = step * dL/dmu is the
  first-order bound on the loading shift for ANY rounding with per-site error
  at most step/2. The same holds for pair occupancy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

HERE = Path(__file__).resolve().parent
ARCHIVE_DIR = HERE.parent / "native_adsorption_probe"
sys.path.insert(0, str(ARCHIVE_DIR))

import adsorption_reference as reference  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from thermo_lab.constraints.quantize import codebook, quantize_array  # noqa: E402

ARCHIVE_JSON_SHA256 = "c4db957d7f672eb4e22400eaf610a7db549125871fe193e5da1b051f2a12a7ef"
SHAPE = (3, 3, 16)
COUPLING_CAP = 0.25
BITS = tuple(range(4, 13))
DEV_TARGETS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
RULES = ("nearest_family", "diffuse_site_family", "diffuse_class_family", "nearest_point")
TOL = {"loading": 0.01, "capacity": 0.01, "pair": 0.02}
FD_STEP = 1e-5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_archive() -> dict:
    path = ARCHIVE_DIR / "probe.json"
    digest = sha256(path)
    if digest != ARCHIVE_JSON_SHA256:
        raise SystemExit(f"archived probe JSON hash mismatch: {digest}")
    return json.loads(path.read_text())


def aggregate(weights, bias) -> tuple[float, float]:
    out = reference.exact_aggregate(SHAPE, weights, bias)
    return out["mean_occupancy"], out["pair_occupancy"]


def nominal(mu: float) -> dict:
    return reference.adsorption_parameters(mu, shape=SHAPE)


def solve_mu(target: float) -> float:
    def residual(mu: float) -> float:
        p = nominal(mu)
        return aggregate(p["weights"], p["bias"])[0] - target

    return float(brentq(residual, -10, 10, xtol=1e-12))


def field_classes(bias: np.ndarray) -> np.ndarray:
    _, labels = np.unique(np.round(bias, 12), return_inverse=True)
    return labels


def diffuse(bias: np.ndarray, cap: float, bits: int, labels: np.ndarray | None) -> np.ndarray:
    """Error diffusion in site order; residual carried globally or per class."""
    out = np.empty_like(bias)
    carry: dict[int, float] = {}
    for i, value in enumerate(bias):
        key = 0 if labels is None else int(labels[i])
        v = value + carry.get(key, 0.0)
        q = float(codebook(np.array([v]), cap, bits)[0])
        out[i] = q
        carry[key] = v - q
    return out


def rounded_fields(rule: str, bias: np.ndarray, bits: int, family_cap: float) -> np.ndarray:
    if rule == "nearest_family":
        return quantize_array(bias, family_cap, bits)
    if rule == "diffuse_site_family":
        return diffuse(bias, family_cap, bits, None)
    if rule == "diffuse_class_family":
        return diffuse(bias, family_cap, bits, field_classes(bias))
    if rule == "nearest_point":
        return quantize_array(bias, float(np.max(np.abs(bias))), bits)
    raise ValueError(rule)


def evaluate_point(mu: float, rule: str, bits: int, family_cap: float, full: tuple) -> dict:
    p = nominal(mu)
    weights = quantize_array(p["weights"], COUPLING_CAP, bits)
    assert np.array_equal(weights, p["weights"]), "couplings must be exact"
    fields = rounded_fields(rule, p["bias"], bits, family_cap)
    loading, pair = aggregate(weights, fields)
    delta = fields - p["bias"]
    return {
        "loading": loading,
        "pair": pair,
        "loading_error": loading - full[0],
        "pair_error": pair - full[1],
        "max_abs_field_error": float(np.max(np.abs(delta))),
        "mean_field_error": float(np.mean(delta)),
        "sum_field_error": float(np.sum(delta)),
    }


def response(mu: float) -> dict:
    lo = nominal(mu - FD_STEP)
    hi = nominal(mu + FD_STEP)
    l_lo, p_lo = aggregate(lo["weights"], lo["bias"])
    l_hi, p_hi = aggregate(hi["weights"], hi["bias"])
    return {
        "dloading_dmu": (l_hi - l_lo) / (2 * FD_STEP),
        "dpair_dmu": (p_hi - p_lo) / (2 * FD_STEP),
    }


def bound_bits(responses: list[dict], cap: float) -> dict:
    """Smallest bits at which the first-order any-rule bound meets every tolerance."""
    rows = {}
    for bits in BITS:
        step = cap / (2 ** (bits - 1) - 1)
        loading = [step * r["dloading_dmu"] for r in responses]
        pair = [step * r["dpair_dmu"] for r in responses]
        capacity = step * (responses[0]["dloading_dmu"] + responses[-1]["dloading_dmu"])
        rows[str(bits)] = {
            "step": step,
            "max_loading_bound": max(loading),
            "max_pair_bound": max(pair),
            "capacity_bound": capacity,
            "passes": max(loading) <= TOL["loading"]
            and max(pair) <= TOL["pair"]
            and capacity <= TOL["capacity"],
        }
    passing = [b for b in BITS if rows[str(b)]["passes"]]
    return {"by_bits": rows, "smallest_passing_bits": passing[0] if passing else None}


def verdicts(points: list[dict]) -> dict:
    """Per bits: max errors over the set and pass/fail; points ordered by mu."""
    out = {}
    for bits in BITS:
        key = str(bits)
        loading = [abs(pt["by_bits"][key]["loading_error"]) for pt in points]
        pair = [abs(pt["by_bits"][key]["pair_error"]) for pt in points]
        capacity = abs(
            points[-1]["by_bits"][key]["loading_error"] - points[0]["by_bits"][key]["loading_error"]
        )
        out[key] = {
            "max_abs_loading_error": max(loading),
            "max_abs_pair_error": max(pair),
            "abs_capacity_error": capacity,
            "points_failing_loading": sum(v > TOL["loading"] for v in loading),
            "passes": max(loading) <= TOL["loading"]
            and max(pair) <= TOL["pair"]
            and capacity <= TOL["capacity"],
        }
    passing = [b for b in BITS if out[str(b)]["passes"]]
    first = passing[0] if passing else None
    stays = first is not None and all(out[str(b)]["passes"] for b in BITS if b >= first)
    return {"by_bits": out, "smallest_passing_bits": first, "passes_at_every_higher_bit": stays}


def provenance() -> dict:
    root = HERE.parents[2]

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    versions = {}
    for package in ("numpy", "scipy", "thermo-lab"):
        versions[package] = importlib.metadata.version(package)
    return {
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "head_sha": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain").splitlines(),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(Path(reference.__file__)),
        "archive_json_sha256": ARCHIVE_JSON_SHA256,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
    }


def run() -> dict:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    archive = load_archive()
    family_cap = float(archive["family_quantization"]["field_cap"])
    frozen_mu = [float(m) for m in archive["derived_numerical_inputs"]["chemical_potentials"]]

    # Control: archived rule at 6 and 8 bits on the frozen points.
    control = []
    for index, mu in enumerate(frozen_mu):
        p = nominal(mu)
        full = aggregate(p["weights"], p["bias"])
        row = {"mu_index": index, "mu": mu, "full": {"loading": full[0], "pair": full[1]}}
        for bits in (6, 8):
            result = evaluate_point(mu, "nearest_family", bits, family_cap, full)
            archived = archive["exact_references"][index]["bits"][str(bits)]
            result["archived_loading"] = archived["mean_occupancy"]
            result["archived_pair"] = archived["pair_occupancy"]
            result["max_abs_difference_from_archive"] = max(
                abs(result["loading"] - archived["mean_occupancy"]),
                abs(result["pair"] - archived["pair_occupancy"]),
            )
            row[str(bits)] = result
        control.append(row)
    control_max = max(r[b]["max_abs_difference_from_archive"] for r in control for b in ("6", "8"))
    if control_max > 1e-12:
        raise SystemExit(f"control failed to reproduce archive: {control_max}")

    frozen_response = [response(mu) for mu in frozen_mu]

    # Development points: all rules and bits.
    dev_mu = [solve_mu(t) for t in DEV_TARGETS]
    dev_response = [response(mu) for mu in dev_mu]
    rules = {}
    timing = []
    for rule in RULES:
        points = []
        for target, mu in zip(DEV_TARGETS, dev_mu, strict=True):
            p = nominal(mu)
            full = aggregate(p["weights"], p["bias"])
            row = {
                "target_loading": target,
                "mu": mu,
                "full": {"loading": full[0], "pair": full[1]},
            }
            row["by_bits"] = {}
            for bits in BITS:
                tick = time.process_time()
                row["by_bits"][str(bits)] = evaluate_point(mu, rule, bits, family_cap, full)
                timing.append(time.process_time() - tick)
            points.append(row)
        rules[rule] = {"points": points, "verdict": verdicts(points)}

    per_point_caps = [float(np.max(np.abs(nominal(mu)["bias"]))) for mu in dev_mu]
    return {
        "evidence_class": "exact_reference",
        "label": "exploration, not a recorded study",
        "inputs": {
            "shape": list(SHAPE),
            "coupling_cap": COUPLING_CAP,
            "family_field_cap": family_cap,
            "frozen_mu": frozen_mu,
            "dev_targets": list(DEV_TARGETS),
            "dev_mu": dev_mu,
            "dev_per_point_field_caps": per_point_caps,
            "bits": list(BITS),
            "rules": list(RULES),
            "tolerances": TOL,
            "fd_step": FD_STEP,
        },
        "control_frozen_points": {"rows": control, "max_abs_difference_from_archive": control_max},
        "linear_response": {
            "frozen_points": frozen_response,
            "frozen_bound_family_cap": bound_bits(frozen_response, family_cap),
            "dev_points": dev_response,
        },
        "development": rules,
        "runtime": {
            "cpu_seconds": time.process_time() - started_cpu,
            "wall_seconds": time.perf_counter() - started_wall,
            "median_cpu_seconds_per_exact_cell": float(np.median(timing)),
            "exact_cells": len(timing),
            "provenance": provenance(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    record = run()
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.output}; cpu {record['runtime']['cpu_seconds']:.1f} s")


if __name__ == "__main__":
    main()
