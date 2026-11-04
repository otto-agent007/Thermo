"""Adsorption precision requirement (loop P-0009): field bits per rounding rule.

Study-local runner for the frozen protocol
``docs/experiments/adsorption-precision-requirement.md``. It imports the archived
512-state transfer helper ``docs/research/native_adsorption_probe/adsorption_reference.py``
unedited, authenticated by SHA-256, and reads the archived probe JSON (also
SHA-256 checked) for the frozen chemical potentials, the family field cap and
the control values. Codebook rounding is ``thermo_lab.constraints.quantize``.

Every observable is ``exact_reference``: the exact equilibrium of a rounded
model against the exact equilibrium of the unrounded model at the same
chemical potential. No sampling, no jitter, no seeds, no cost and no hardware
claim. The first-order linear-response bound is analysis, labelled as such.

One recorded cell is one exact transfer evaluation of (loading, aggregate pair
occupancy) for one rule, bit count and chemical potential.

Run (under a CPU-minute; no autosave needed)::

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
      uv run python -m thermo_lab.adsorption_precision_requirement \\
      --output-dir results/adsorption-precision-requirement

Replay an archive (CI)::

    ... -m thermo_lab.adsorption_precision_requirement --replay <study.json.gz> --output-dir <new>

Replay tolerances. Rounded field arrays are elementwise IEEE arithmetic on the
archived chemical potentials, so their SHA-256 is compared bitwise on any host.
Loadings and pair occupancies pass through a BLAS product, so they are
compared at 1e-12 absolute. The secondary chemical potentials are re-solved
by Brent (xtol 1e-12) and compared at 1e-9; the cells are then recomputed at
the archived chemical potentials, not the re-solved ones. Verdicts are
recomputed from the archived cells and must match
exactly; verdicts from the recomputed cells must make the same pass/fail
decisions. The finite-difference susceptibilities in the analysis block divide
a BLAS-level difference by 2e-5, so they are compared at 1e-9.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy.optimize import brentq

from thermo_lab.constraints.quantize import codebook, quantize_array
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/experiments/adsorption-precision-requirement.md"
HELPER = "docs/research/native_adsorption_probe/adsorption_reference.py"
ARCHIVE_JSON = "docs/research/native_adsorption_probe/probe.json"
PROTOCOL_SHA256 = "43483357b0398b9a6057acbb6029ca9e9d12102405837d103fad22349f797c74"
HELPER_SHA256 = "6f1f9022543c115b4d74718ba5d08d84bd31997e8e230be50f6385958e352dff"
ARCHIVE_JSON_SHA256 = "c4db957d7f672eb4e22400eaf610a7db549125871fe193e5da1b051f2a12a7ef"

SHAPE = (3, 3, 16)
COUPLING_CAP = 0.25
BITS = tuple(range(4, 13))
# Protocol numbering: rules 1 to 3 compete for the recommendation; rule 4 is diagnostic.
RULES = ("nearest_family", "diffuse_class_family", "nearest_point", "diffuse_site_family")
RECOMMENDABLE = RULES[:3]
FROZEN_TARGETS = (0.2, 0.5, 0.8)
SECONDARY_TARGETS = (0.15, 0.3, 0.4, 0.6, 0.7, 0.85)
TOL = {"loading": 0.01, "capacity": 0.01, "pair": 0.02}
CLASS_DECIMALS = 12
BRENT = {"bracket": [-10.0, 10.0], "xtol": 1e-12}
FD_STEP = 1e-5
REPLAY_ATOL = 1e-12
ANALYSIS_ATOL = 1e-9
RESIDUAL_SLACK = 1e-12
STATUS = "adsorption_precision_requirement_complete"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_single_thread_blas() -> None:
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"set {name}=1 before running or replaying this study")


def load_reference() -> ModuleType:
    """Import the archived transfer helper after checking its SHA-256."""
    path = (ROOT / HELPER).resolve()
    if sha256_file(path) != HELPER_SHA256:
        raise ValueError(f"archived helper hash mismatch: {HELPER}")
    module = sys.modules.get("adsorption_reference")
    if module is not None and Path(module.__file__).resolve() == path:
        return module
    spec = importlib.util.spec_from_file_location("adsorption_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["adsorption_reference"] = module
    spec.loader.exec_module(module)
    return module


def load_archive() -> dict:
    path = ROOT / ARCHIVE_JSON
    if sha256_file(path) != ARCHIVE_JSON_SHA256:
        raise ValueError(f"archived probe JSON hash mismatch: {ARCHIVE_JSON}")
    return json.loads(path.read_text())


# --- model and rounding rules -------------------------------------------------------------


def nominal(mu: float) -> dict:
    return load_reference().adsorption_parameters(mu, shape=SHAPE)


def aggregate(weights, bias) -> tuple[float, float]:
    out = load_reference().exact_aggregate(SHAPE, weights, bias)
    return float(out["mean_occupancy"]), float(out["pair_occupancy"])


def solve_mu(target: float) -> float:
    def residual(mu: float) -> float:
        p = nominal(mu)
        return aggregate(p["weights"], p["bias"])[0] - target

    lo, hi = BRENT["bracket"]
    return float(brentq(residual, lo, hi, xtol=BRENT["xtol"]))


def field_classes(bias: np.ndarray) -> np.ndarray:
    """Class label per site: sites sharing a nominal field (rounded to 12 decimals)."""
    _, labels = np.unique(np.round(bias, CLASS_DECIMALS), return_inverse=True)
    return labels


def diffuse(
    bias: np.ndarray, cap: float, bits: int, labels: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Error diffusion in site order; residual carried globally or within a class.

    Returns (rounded fields, pre-rounding values, carried residual after each site).
    """
    out = np.empty_like(bias)
    pre = np.empty_like(bias)
    carries = np.empty_like(bias)
    carry: dict[int, float] = {}
    for i, value in enumerate(bias):
        key = 0 if labels is None else int(labels[i])
        v = value + carry.get(key, 0.0)
        q = float(codebook(np.array([v]), cap, bits)[0])
        out[i], pre[i] = q, v
        carry[key] = v - q
        carries[i] = carry[key]
    return out, pre, carries


def rounded_fields(
    rule: str, bias: np.ndarray, bits: int, family_cap: float
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    """(fields, pre-rounding values, field cap, class labels or None) for one rule."""
    if rule == "nearest_family":
        return quantize_array(bias, family_cap, bits), bias, family_cap, None
    if rule == "nearest_point":
        cap = float(np.max(np.abs(bias)))
        return quantize_array(bias, cap, bits), bias, cap, None
    if rule == "diffuse_class_family":
        labels = field_classes(bias)
        fields, pre, _ = diffuse(bias, family_cap, bits, labels)
        return fields, pre, family_cap, labels
    if rule == "diffuse_site_family":
        fields, pre, _ = diffuse(bias, family_cap, bits, None)
        return fields, pre, family_cap, np.zeros(len(bias), dtype=int)
    raise ValueError(f"unknown rule {rule!r}")


def fields_digest(fields: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(fields, dtype="<f8").tobytes()).hexdigest()


def max_running_residual(bias: np.ndarray, fields: np.ndarray, labels: np.ndarray) -> float:
    """Largest |running sum of (nominal - rounded)| within any class, in site order."""
    worst = 0.0
    for label in np.unique(labels):
        delta = (bias - fields)[labels == label]
        worst = max(worst, float(np.max(np.abs(np.cumsum(delta)))))
    return worst


def evaluate_cell(mu: float, rule: str, bits: int, family_cap: float, full: dict) -> dict:
    p = nominal(mu)
    weights = quantize_array(p["weights"], COUPLING_CAP, bits)
    fields, pre, cap, labels = rounded_fields(rule, p["bias"], bits, family_cap)
    levels = 2 ** (bits - 1) - 1
    step = cap / levels
    clipped = int(np.sum(np.abs(np.rint(pre / step)) > levels))
    residual_ok = None
    if labels is not None and clipped == 0:
        residual_ok = max_running_residual(p["bias"], fields, labels) <= step / 2 + RESIDUAL_SLACK
    loading, pair = aggregate(weights, fields)
    delta = fields - p["bias"]
    return {
        "loading": loading,
        "pair": pair,
        "loading_error": loading - full["loading"],
        "pair_error": pair - full["pair"],
        "field_cap": cap,
        "max_abs_field_error": float(np.max(np.abs(delta))),
        "mean_field_error": float(np.mean(delta)),
        "clipped_sites": clipped,
        "fields_sha256": fields_digest(fields),
        "couplings_exact": bool(np.array_equal(weights, p["weights"])),
        "fields_on_codebook": bool(np.array_equal(codebook(fields, cap, bits), fields)),
        "residual_within_half_step": residual_ok,
    }


# --- request ------------------------------------------------------------------------------


def study_request() -> dict:
    archive = load_archive()
    frozen_mu = [float(m) for m in archive["derived_numerical_inputs"]["chemical_potentials"]]
    family_cap = float(archive["family_quantization"]["field_cap"])
    return {
        "schema": "adsorption_precision_requirement.request.v1",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "helper": HELPER,
        "helper_sha256": HELPER_SHA256,
        "archive_json": ARCHIVE_JSON,
        "archive_json_sha256": ARCHIVE_JSON_SHA256,
        "model": {
            "shape": list(SHAPE),
            "beta": 1.0,
            "epsilon": 1.0,
            "lateral_wall_phi": 3.0,
            "spin_map": "J = beta*epsilon/4; h_i = beta*((mu - phi_i)/2 + epsilon*degree_i/4)",
            "site_order": "z, then y, then x",
            "numeric_dtype": "float64",
            "placement": "archived (x + 4z, y + z), offsets (1,0), (0,1), (4,1); no copies",
        },
        "frozen": {"target_loadings": list(FROZEN_TARGETS), "mu": frozen_mu},
        "secondary": {
            "target_loadings": list(SECONDARY_TARGETS),
            "solver": "brentq",
            **BRENT,
        },
        "caps": {
            "coupling": COUPLING_CAP,
            "family_field": family_cap,
            "per_point_field": "max |h| of that point's own nominal model",
        },
        "rules": list(RULES),
        "recommendable_rules": list(RECOMMENDABLE),
        "class_rounding_decimals": CLASS_DECIMALS,
        "bits": list(BITS),
        "tolerances": TOL,
        "fd_step": FD_STEP,
        "replay_atol": REPLAY_ATOL,
        "analysis_atol": ANALYSIS_ATOL,
    }


# --- preflight ----------------------------------------------------------------------------

# Hand-built six-site check (cap 3, 3 bits: levels -3..3, step 1). Classes are
# {0.4: sites 0, 2, 4} and {1.4: sites 1, 3, 5}. Worked by hand:
#   class: 0.4->0 c.4; 1.4->1 c.4; 0.8->1 c-.2; 1.8->2 c-.2; 0.2->0 c.2; 1.2->1 c.2
#   site:  0.4->0 c.4; 1.8->2 c-.2; 0.2->0 c.2; 1.6->2 c-.4; 0.0->0 c0; 1.4->1 c.4
HAND_FIELDS = (0.4, 1.4, 0.4, 1.4, 0.4, 1.4)
HAND_CAP, HAND_BITS = 3.0, 3
HAND_EXPECTED = {
    "class": {"fields": (0, 1, 1, 2, 0, 1), "carries": (0.4, 0.4, -0.2, -0.2, 0.2, 0.2)},
    "site": {"fields": (0, 2, 0, 2, 0, 1), "carries": (0.4, -0.2, 0.2, -0.4, 0.0, 0.4)},
}


def hand_diffusion_check() -> dict:
    bias = np.array(HAND_FIELDS)
    out = {}
    for name, labels in (("class", field_classes(bias)), ("site", None)):
        fields, _, carries = diffuse(bias, HAND_CAP, HAND_BITS, labels)
        want = HAND_EXPECTED[name]
        ok = bool(
            np.allclose(fields, want["fields"], rtol=0, atol=1e-12)
            and np.allclose(carries, want["carries"], rtol=0, atol=1e-12)
        )
        out[name] = {"fields": fields.tolist(), "carries": carries.tolist(), "pass": ok}
    out["passed"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def control(request: dict, archive: dict) -> dict:
    """The archived rule at 6 and 8 bits on the frozen points against the archive."""
    rows = []
    worst = 0.0
    for index, mu in enumerate(request["frozen"]["mu"]):
        p = nominal(mu)
        for bits in (6, 8):
            fields = quantize_array(p["bias"], request["caps"]["family_field"], bits)
            weights = quantize_array(p["weights"], COUPLING_CAP, bits)
            loading, pair = aggregate(weights, fields)
            archived = archive["exact_references"][index]["bits"][str(bits)]
            diff = max(
                abs(loading - archived["mean_occupancy"]), abs(pair - archived["pair_occupancy"])
            )
            worst = max(worst, diff)
            rows.append(
                {
                    "mu_index": index,
                    "bits": bits,
                    "loading": loading,
                    "pair": pair,
                    "archived_loading": archived["mean_occupancy"],
                    "archived_pair": archived["pair_occupancy"],
                }
            )
    return {"rows": rows, "max_abs_difference": worst, "passed": worst <= 1e-12}


def preflight(request: dict) -> dict:
    hashes = {
        "protocol": sha256_file(ROOT / PROTOCOL) == PROTOCOL_SHA256,
        "helper": sha256_file(ROOT / HELPER) == HELPER_SHA256,
        "archive_json": sha256_file(ROOT / ARCHIVE_JSON) == ARCHIVE_JSON_SHA256,
    }
    if not all(hashes.values()):
        return {"hashes": hashes, "passed": False}
    reference = load_reference()
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            reference.self_test()
        match = re.search(r"PASS (\d+) enumeration cases", buffer.getvalue())
        self_test = {"passed": match is not None, "cases": int(match.group(1)) if match else 0}
    except AssertionError as error:
        self_test = {"passed": False, "cases": 0, "error": str(error)[:500]}
    self_test["passed"] = self_test["passed"] and self_test["cases"] == 24
    ctrl = control(request, load_archive())
    hand = hand_diffusion_check()
    return {
        "hashes": hashes,
        "self_test": self_test,
        "control": ctrl,
        "hand_diffusion": hand,
        "passed": all(hashes.values())
        and self_test["passed"]
        and ctrl["passed"]
        and hand["passed"],
    }


# --- cells, verdicts, analysis ------------------------------------------------------------


def full_cells(mus: list[float]) -> list[dict]:
    out = []
    for mu in mus:
        p = nominal(mu)
        loading, pair = aggregate(p["weights"], p["bias"])
        out.append(
            {
                "loading": loading,
                "pair": pair,
                "max_abs_field": float(np.max(np.abs(p["bias"]))),
            }
        )
    return out


def point_cells(request: dict, mus: list[float], full: list[dict], timing: list) -> dict:
    """{rule: {bits: [cell per point]}} for one point set."""
    cap = request["caps"]["family_field"]
    out: dict = {}
    for rule in request["rules"]:
        out[rule] = {}
        for bits in request["bits"]:
            row = []
            for mu, ref in zip(mus, full, strict=True):
                tick = time.process_time()
                row.append(evaluate_cell(mu, rule, bits, cap, ref))
                timing.append(time.process_time() - tick)
            out[rule][str(bits)] = row
    return out


def verdicts(request: dict, cells: dict) -> dict:
    """Per rule: per-bit errors and pass flags, b*, b_stable. Points ordered by loading."""
    tol = request["tolerances"]
    out = {}
    for rule in request["rules"]:
        by_bits = {}
        for bits in request["bits"]:
            row = cells[rule][str(bits)]
            loading = max(abs(c["loading_error"]) for c in row)
            pair = max(abs(c["pair_error"]) for c in row)
            capacity = abs(row[-1]["loading_error"] - row[0]["loading_error"])
            by_bits[str(bits)] = {
                "max_abs_loading_error": loading,
                "abs_capacity_error": capacity,
                "max_abs_pair_error": pair,
                "points_failing_loading": sum(
                    abs(c["loading_error"]) > tol["loading"] for c in row
                ),
                "clipped_sites": sum(c["clipped_sites"] for c in row),
                "passes": loading <= tol["loading"]
                and capacity <= tol["capacity"]
                and pair <= tol["pair"],
            }
        bits = request["bits"]
        passing = [b for b in bits if by_bits[str(b)]["passes"]]
        stable = None
        for b in reversed(bits):
            if not by_bits[str(b)]["passes"]:
                break
            stable = b
        out[rule] = {
            "by_bits": by_bits,
            "b_star": passing[0] if passing else None,
            "b_stable": stable,
            "non_monotone_loading": any(
                by_bits[str(b + 1)]["max_abs_loading_error"]
                > by_bits[str(b)]["max_abs_loading_error"]
                for b in bits[:-1]
            ),
            "passes_then_fails": passing != [] and passing != list(range(passing[0], bits[-1] + 1)),
        }
    return out


def decide(request: dict, frozen: dict, secondary: dict) -> dict:
    """Protocol primary rule 3 and 4: recommended rule and generalization labels."""
    labels = {}
    for rule in request["rules"]:
        b = frozen[rule]["b_stable"]
        if b is None:
            labels[rule] = "no passing bit count on the frozen points within 12 bits"
        elif secondary[rule]["by_bits"][str(b)]["passes"]:
            labels[rule] = "passes the frozen points and the secondary grid at its b_stable"
        else:
            labels[rule] = "passes the frozen points only"
    candidates = [
        (frozen[r]["b_stable"], i, r)
        for i, r in enumerate(request["recommendable_rules"])
        if frozen[r]["b_stable"] is not None
        and secondary[r]["by_bits"][str(frozen[r]["b_stable"])]["passes"]
    ]
    if candidates:
        b, _, rule = min(candidates)
        fallback = False
    else:
        rule, b, fallback = "nearest_family", frozen["nearest_family"]["b_stable"], True
    return {
        "recommended_rule": rule,
        "recommended_b_stable": b,
        "fallback_no_rule_generalized": fallback,
        "rule_labels": labels,
    }


def response(mu: float) -> dict:
    lo, hi = nominal(mu - FD_STEP), nominal(mu + FD_STEP)
    l_lo, p_lo = aggregate(lo["weights"], lo["bias"])
    l_hi, p_hi = aggregate(hi["weights"], hi["bias"])
    return {
        "dloading_dmu": (l_hi - l_lo) / (2 * FD_STEP),
        "dpair_dmu": (p_hi - p_lo) / (2 * FD_STEP),
    }


def linear_bound(request: dict) -> dict:
    """Analysis, not an exact observable: the first-order any-rule bound on the frozen points.

    A field shift delta equals a chemical-potential shift 2*delta; with per-site error at
    most step/2 and nonnegative site susceptibilities, loading moves at most
    step * dL/dmu to first order. Same for pair occupancy.
    """
    responses = [response(mu) for mu in request["frozen"]["mu"]]
    cap = request["caps"]["family_field"]
    rows = {}
    for bits in request["bits"]:
        step = cap / (2 ** (bits - 1) - 1)
        loading = max(step * r["dloading_dmu"] for r in responses)
        pair = max(step * r["dpair_dmu"] for r in responses)
        capacity = step * (responses[0]["dloading_dmu"] + responses[-1]["dloading_dmu"])
        rows[str(bits)] = {
            "step": step,
            "max_loading_bound": loading,
            "max_pair_bound": pair,
            "capacity_bound": capacity,
            "passes": loading <= TOL["loading"]
            and pair <= TOL["pair"]
            and capacity <= TOL["capacity"],
        }
    passing = [b for b in request["bits"] if rows[str(b)]["passes"]]
    return {
        "label": "analysis (first-order linear response), not an exact observable",
        "responses": responses,
        "by_bits": rows,
        "smallest_passing_bits": passing[0] if passing else None,
    }


def expectations(record: dict) -> dict:
    """Pre-registered expectations (protocol section), reported, non-gating."""
    f = record["verdicts"]["frozen"]
    return {
        "1_control_reproduces_archive": record["preflight"]["control"]["passed"],
        "2_nearest_family_b_star_9": f["nearest_family"]["b_star"] == 9,
        "2_first_order_bound_passes_at_9": record["analysis"]["smallest_passing_bits"] == 9,
        "3_diffuse_class_family_b_star_5": f["diffuse_class_family"]["b_star"] == 5,
        "4_nearest_point_b_star_7_or_8": f["nearest_point"]["b_star"] in (7, 8),
        "5_diffuse_site_family_b_star_5_or_6": f["diffuse_site_family"]["b_star"] in (5, 6),
        "6_non_monotone_for_a_nearest_rule": f["nearest_family"]["non_monotone_loading"]
        or f["nearest_point"]["non_monotone_loading"],
    }


def checks(record: dict) -> dict:
    out = {"cells": 0, "couplings_exact": True, "fields_on_codebook": True, "residual": True}
    for name in ("frozen", "secondary"):
        for rule_cells in record["cells"][name].values():
            for row in rule_cells.values():
                for c in row:
                    out["cells"] += 1
                    out["couplings_exact"] &= c["couplings_exact"]
                    out["fields_on_codebook"] &= c["fields_on_codebook"]
                    out["residual"] &= c["residual_within_half_step"] is not False
    out["passed"] = out["couplings_exact"] and out["fields_on_codebook"] and out["residual"]
    return out


def result_payload(record: dict) -> dict:
    return {
        "secondary_mu": record["secondary_mu"],
        "full": record["full"],
        "cells": record["cells"],
        "verdicts": record["verdicts"],
        "decision": record["decision"],
    }


# --- driver -------------------------------------------------------------------------------


def compute(request: dict, log=lambda msg: None) -> dict:
    """Everything after preflight; returns the result part of the record and CPU timings."""
    timing: list[float] = []
    frozen_mu = request["frozen"]["mu"]
    frozen_full = full_cells(frozen_mu)
    log("full-precision frozen cells: 3")
    secondary_mu = [solve_mu(t) for t in request["secondary"]["target_loadings"]]
    secondary_full = full_cells(secondary_mu)
    log(f"secondary chemical potentials solved: {len(secondary_mu)}")
    frozen = point_cells(request, frozen_mu, frozen_full, timing)
    log(f"frozen cells: {len(timing)}")
    secondary = point_cells(request, secondary_mu, secondary_full, timing)
    log(f"secondary cells: {len(timing) - 108}")
    record = {
        "secondary_mu": secondary_mu,
        "full": {"frozen": frozen_full, "secondary": secondary_full},
        "cells": {"frozen": frozen, "secondary": secondary},
    }
    record["verdicts"] = {
        "frozen": verdicts(request, frozen),
        "secondary": verdicts(request, secondary),
    }
    record["decision"] = decide(
        request, record["verdicts"]["frozen"], record["verdicts"]["secondary"]
    )
    record["analysis"] = linear_bound(request)
    return record, timing


def run_study(output_dir: str | Path) -> dict:
    require_single_thread_blas()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    log_path = root / "run.log"

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            fh.write(f"{stamp} P-0009 {msg}\n")

    request = study_request()
    log(f"request {canonical_sha256(request)}")
    check = preflight(request)
    log(
        f"preflight passed={check['passed']} control max diff "
        f"{check.get('control', {}).get('max_abs_difference')}"
    )
    if not check["passed"]:
        atomic_write_text(root / "preflight.json", canonical_json(check) + "\n")
        raise SystemExit("preflight failed; stopping before any new cell")
    result, timing = compute(request, log)
    record = {
        "schema": "adsorption_precision_requirement.record.v1",
        "evidence": {
            "cells": "exact_reference (float64 transfer, archived helper)",
            "full_precision": "exact_reference",
            "analysis": "analysis (first-order linear-response bound), not evidence",
            "hardware": "none; no sampling, no jitter, no cost",
        },
        "request": request,
        "request_digest": canonical_sha256(request),
        "preflight": check,
        **result,
    }
    record["checks"] = checks(record)
    record["expectations"] = expectations(record)
    record["result_digest"] = canonical_sha256(result_payload(record))
    log(
        f"checks passed={record['checks']['passed']}; recommended "
        f"{record['decision']['recommended_rule']} at b_stable "
        f"{record['decision']['recommended_b_stable']}"
    )
    if not record["checks"]["passed"]:
        raise SystemExit("cell checks failed (coupling, codebook or residual)")
    archive_bytes = gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    (root / "study.json.gz").write_bytes(archive_bytes)
    persisted = json.loads(gzip.decompress((root / "study.json.gz").read_bytes()))
    replay_seconds = time.monotonic()
    replay(persisted)
    log(f"replay passed in {time.monotonic() - replay_seconds:.1f}s")
    from thermo_lab.provenance import collect_runtime_provenance

    provenance = {
        "schema": "adsorption_precision_requirement.provenance.v1",
        "request_digest": persisted["request_digest"],
        "result_digest": persisted["result_digest"],
        "runtime": json.loads(canonical_json(collect_runtime_provenance(ROOT).model_dump())),
        "thread_environment": {
            k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.monotonic() - started_wall,
        "exact_cells": len(timing),
        "median_cpu_seconds_per_cell": float(np.median(timing)),
    }
    atomic_write_text(root / "summary.md", render_report(persisted))
    atomic_write_text(root / "provenance.json", canonical_json(provenance) + "\n")
    log(f"completion published after {time.monotonic() - started_wall:.1f}s")
    atomic_write_text(
        root / "completion.json",
        canonical_json(completion(persisted, provenance, archive_bytes)) + "\n",
    )
    return persisted


# --- replay -------------------------------------------------------------------------------


def match(new, old, path: str, atol: float = REPLAY_ATOL) -> None:
    """Recomputed against archived: floats to ``atol`` absolute, everything else exactly."""
    if isinstance(old, dict):
        if not isinstance(new, dict) or set(new) != set(old):
            raise ValueError(f"{path} keys drifted")
        for key in old:
            match(new[key], old[key], f"{path}/{key}", atol)
    elif isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            raise ValueError(f"{path} length drifted")
        for i, (a, b) in enumerate(zip(new, old, strict=True)):
            match(a, b, f"{path}[{i}]", atol)
    elif isinstance(old, float) and not isinstance(old, bool):
        if not isinstance(new, (int, float)) or isinstance(new, bool) or abs(new - old) > atol:
            raise ValueError(f"{path} drifted: {new!r} against archived {old!r}")
    elif new != old or type(new) is not type(old):
        raise ValueError(f"{path} drifted: {new!r} against archived {old!r}")


def _decisions(verdict: dict) -> dict:
    return {
        rule: {
            "b_star": v["b_star"],
            "b_stable": v["b_stable"],
            "passes": {b: row["passes"] for b, row in v["by_bits"].items()},
        }
        for rule, v in verdict.items()
    }


def replay(record: dict) -> None:
    require_single_thread_blas()
    request = study_request()
    if record["request"] != request or record["request_digest"] != canonical_sha256(request):
        raise ValueError("archived request does not match this code's request")
    if not record["preflight"]["passed"]:
        raise ValueError("archived preflight did not pass")
    check = preflight(request)
    if not check["passed"]:
        raise ValueError("preflight does not pass on replay")
    match(check["control"], record["preflight"]["control"], "preflight/control")
    match(check["hand_diffusion"], record["preflight"]["hand_diffusion"], "preflight/hand")
    # Re-solve the secondary points, then recompute every cell from the archived inputs.
    solved = [solve_mu(t) for t in request["secondary"]["target_loadings"]]
    match(solved, record["secondary_mu"], "secondary_mu", ANALYSIS_ATOL)
    mus = {"frozen": request["frozen"]["mu"], "secondary": record["secondary_mu"]}
    timing: list[float] = []
    recomputed_verdicts = {}
    for name, points in mus.items():
        full = full_cells(points)
        match(full, record["full"][name], f"full/{name}")
        cells = point_cells(request, points, record["full"][name], timing)
        match(cells, record["cells"][name], f"cells/{name}")
        recomputed_verdicts[name] = verdicts(request, cells)
    # Verdicts from the archived cells must be identical; from recomputed cells, same decisions.
    for name in mus:
        archived_verdict = verdicts(request, record["cells"][name])
        if archived_verdict != record["verdicts"][name]:
            match(archived_verdict, record["verdicts"][name], f"verdicts/{name}", 0.0)
            raise ValueError(f"verdicts/{name} drifted")
        if _decisions(recomputed_verdicts[name]) != _decisions(archived_verdict):
            raise ValueError(f"verdicts/{name} decision drifted on recomputation")
    decision = decide(request, record["verdicts"]["frozen"], record["verdicts"]["secondary"])
    if decision != record["decision"]:
        raise ValueError("decision drifted")
    match(linear_bound(request), record["analysis"], "analysis", ANALYSIS_ATOL)
    if checks(record) != record["checks"] or not record["checks"]["passed"]:
        raise ValueError("cell checks drifted or did not pass")
    if expectations(record) != record["expectations"]:
        raise ValueError("expectations drifted")
    if canonical_sha256(result_payload(record)) != record["result_digest"]:
        raise ValueError("result digest does not replay")


# --- report -------------------------------------------------------------------------------


def _fmt(v) -> str:
    return "none" if v is None else str(v)


def render_report(record: dict) -> str:
    req, d = record["request"], record["decision"]
    f, s = record["verdicts"]["frozen"], record["verdicts"]["secondary"]
    lines = [
        "# Adsorption precision requirement (P-0009): runner summary",
        "",
        "Every number is `exact_reference` (exact equilibrium of a rounded model against",
        "full precision at the same chemical potential). No sampling, no hardware claim.",
        "",
        f"Request `{record['request_digest']}`; result `{record['result_digest']}`.",
        "",
        f"Recommended rule: `{d['recommended_rule']}` at b_stable "
        f"{_fmt(d['recommended_b_stable'])}"
        + (" (fallback: no rule generalized)." if d["fallback_no_rule_generalized"] else "."),
        "",
        "| rule | b* frozen | b_stable frozen | b* secondary | b_stable secondary | label |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rule in req["rules"]:
        lines.append(
            f"| `{rule}` | {_fmt(f[rule]['b_star'])} | {_fmt(f[rule]['b_stable'])} "
            f"| {_fmt(s[rule]['b_star'])} | {_fmt(s[rule]['b_stable'])} "
            f"| {d['rule_labels'][rule]} |"
        )
    for name, v in (("Frozen points", f), ("Secondary grid", s)):
        lines += [
            "",
            f"## {name}: max loading / capacity / max pair error (pass marked *)",
            "",
            "| bits | " + " | ".join(f"`{r}`" for r in req["rules"]) + " |",
            "| --- |" + " --- |" * len(req["rules"]),
        ]
        for bits in req["bits"]:
            row = []
            for rule in req["rules"]:
                c = v[rule]["by_bits"][str(bits)]
                mark = "*" if c["passes"] else ""
                row.append(
                    f"{c['max_abs_loading_error']:.4f} / {c['abs_capacity_error']:.4f} / "
                    f"{c['max_abs_pair_error']:.4f}{mark}"
                )
            lines.append(f"| {bits} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def completion(record: dict, provenance: dict, archive_bytes: bytes) -> dict:
    f, s, d = record["verdicts"]["frozen"], record["verdicts"]["secondary"], record["decision"]
    n_frozen = sum(len(r) for rule in record["cells"]["frozen"].values() for r in rule.values())
    n_sec = sum(len(r) for rule in record["cells"]["secondary"].values() for r in rule.values())
    return {
        "status": STATUS,
        "rules": len(record["request"]["rules"]),
        "bits": len(record["request"]["bits"]),
        "frozen_cells": n_frozen,
        "secondary_cells": n_sec,
        "full_precision_cells": len(record["full"]["frozen"]),
        "secondary_full_precision_cells": len(record["full"]["secondary"]),
        "control_passed": record["preflight"]["control"]["passed"],
        "preflight_passed": record["preflight"]["passed"],
        "checks_passed": record["checks"]["passed"],
        "replayed": True,
        "integrity": True,
        "b_star": {
            "frozen": {r: f[r]["b_star"] for r in f},
            "secondary": {r: s[r]["b_star"] for r in s},
        },
        "b_stable": {
            "frozen": {r: f[r]["b_stable"] for r in f},
            "secondary": {r: s[r]["b_stable"] for r in s},
        },
        "recommended_rule": d["recommended_rule"],
        "recommended_b_stable": d["recommended_b_stable"],
        "fallback_no_rule_generalized": d["fallback_no_rule_generalized"],
        "evidence_class": "exact_reference",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def replay_archive(archive: str | Path, output_dir: str | Path) -> dict:
    archive_bytes = Path(archive).read_bytes()
    record = json.loads(gzip.decompress(archive_bytes))
    replay(record)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": "adsorption_precision_requirement.provenance.v1",
        "mode": "replay_only",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
    }
    atomic_write_text(dest / "summary.md", render_report(record))
    atomic_write_text(
        dest / "completion.json",
        canonical_json(completion(record, provenance, archive_bytes)) + "\n",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay", help="replay this study.json.gz instead of running")
    args = parser.parse_args()
    if args.replay:
        replay_archive(args.replay, args.output_dir)
    else:
        run_study(args.output_dir)


if __name__ == "__main__":
    main()
