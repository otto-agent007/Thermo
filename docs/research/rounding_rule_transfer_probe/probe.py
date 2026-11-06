"""Exploratory probe for queue row ``rounding-rule-transfer`` (loop P-0010).

Exploration, not a recorded study. Exact references only: no sampling, no
jitter, no training, no new layout, no cost. Run from the repository root:

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
      uv run python docs/research/rounding_rule_transfer_probe/probe.py \\
      --output docs/research/rounding_rule_transfer_probe/probe.json

Fixed before any class-diffusion result was computed (committed with this
script, before it ran):

* Target: P-0008's archived placed tie-limit model of ``t4_h05_q035``'s
  posterior on the carried layout ``index4`` at w* = 1.75, the 1,000 archived
  exact test images, Bayes floor recomputed from the archived generator. The
  archive (``docs/experiment-reports/2026-10-17-dtm-exact-posterior-lattice``)
  is authenticated three ways before anything runs: the evidence tarball
  against ``manifest.json``; every on-disk source against the archived
  request's ``sources`` hashes; and the request rebuilt by
  ``thermo_lab.dtm_exact_posterior_lattice.make_request()`` against the
  archived ``request_digest``. The module is imported, never edited.
* Caps: the archived ``split`` caps of the placed model (one coupling cap =
  max |coupling| = w* = 1.75, one field cap = max |free-node field| = 0.5),
  computed by the imported ``cap_for``. Both rules use the same caps.
* Rules:
  - ``nearest``: the archived rule, ``quantize_array`` on each vector. The
    control reproduces the archived exact rows at 4, 6 and 8 bits (MPM gap
    and per-image error counts, exactly), then runs 5 bits.
  - ``diffuse_class``: P-0009's ``diffuse_class_family`` extended to every
    coefficient class. Two vectors, as in ``split``: the coupling vector
    (grid, clamp, copy and tie couplings in the archived edge order) and the
    free-node field vector (pixel fields, copy biases, relay zeros in the
    archived node order). Within each vector, a class is the set of entries
    sharing a nominal value rounded to 12 decimals (P-0009's
    ``CLASS_DECIMALS``), regardless of kind. Each class carries its own
    rounding residual in vector order through the imported ``codebook``
    (clipping at the cap). The function is P-0009's ``diffuse``, imported
    unedited from ``thermo_lab.adsorption_precision_requirement``.
* Exact evaluation of a rounded model with non-uniform couplings: the
  archived enumerator assumes one grid coupling J and one clamp coupling g
  (``collapsed`` reads the first entry of each). Class diffusion makes both
  vary by edge, so the probe uses study-local copies of ``collapsed``,
  ``exact_equilibrium`` and the transfer-matrix grid with per-edge vertical
  and horizontal couplings and a per-pixel clamp coupling. The copies must
  reproduce the imported functions bitwise on the archived nearest rows
  (checked: same per-image counts and the same gap to the bit).
* Bits: 4, 5, 6 and 8 (the row); 3 as exploration only.
* Metric: exact MPM pixel error minus the Bayes MPM error, mean over the
  1,000 exact images; tolerance 0.01. The 64-draw majority gap is reported
  alongside and does not gate.
* Mechanism report at 4 bits: every class with its size, kinds, nominal
  value and summed rounding error under each rule; and one-kind ablations
  (round one coefficient kind at 4 bits under each rule, everything else
  float) so the probe names the kind that carries the 4-bit miss.
* am-coupling-bits class count: stage A's bias design on the 16 held pattern
  sets of each of the six cells (archived configs, hash-checked stage A
  archive), ``split`` caps; per cell, class sizes and how many classes round
  off the codebook at 6, 8 and 10 bits. Counting only; no recall evaluated.
* Adsorption pore at 2 and 3 bits (exploration the row allows): P-0009's
  four rules on its three frozen chemical potentials through its own
  imported ``evaluate_cell`` and tolerances.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from thermo_lab import adsorption_precision_requirement as apr
from thermo_lab import am_binary_emulation as stage_a
from thermo_lab import am_coupling_bits as amcb
from thermo_lab import dtm_exact_posterior_lattice as dl
from thermo_lab.am_categorical_reference import load_stage_a
from thermo_lab.constraints.quantize import cap_for, codebook, quantize_array
from thermo_lab.dtm_small_image import GridIsing
from thermo_lab.hashing import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "docs/experiment-reports/2026-10-17-dtm-exact-posterior-lattice"
BITS = (3, 4, 5, 6, 8)
ROW_BITS = (4, 5, 6, 8)
ARCHIVED_BITS = (4, 6, 8)
RULES = ("nearest", "diffuse_class")
TOL = 0.01
CLASS_DECIMALS = apr.CLASS_DECIMALS
L, N = dl.L, dl.N


# --- archive authentication ---------------------------------------------------------------


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_archive() -> dict:
    manifest = json.loads((REPORT / "manifest.json").read_text())
    blob = (REPORT / manifest["archive"]).read_bytes()
    if sha_bytes(blob) != manifest["sha256"]:
        raise SystemExit("P-0008 evidence tarball does not match manifest.json")
    tar = tarfile.open(fileobj=io.BytesIO(blob))
    prefix = "dtm-exact-posterior-lattice/"
    members = {}
    for name, digest in manifest["members"].items():
        data = tar.extractfile(prefix + name).read()
        if sha_bytes(data) != digest:
            raise SystemExit(f"archive member {name} does not match manifest.json")
        members[name] = data
    request = json.loads(members["request.json"])
    study = json.loads(gzip.decompress(members["study.json.gz"]))
    bad = [p for p, h in request["sources"].items() if sha_bytes((ROOT / p).read_bytes()) != h]
    if bad:
        raise SystemExit(f"on-disk sources differ from the archived hashes: {bad}")
    rebuilt = dl.make_request()
    if canonical_sha256(rebuilt) != study["request_digest"]:
        raise SystemExit("make_request() does not rebuild the archived request digest")
    return {
        "request": rebuilt,
        "study": study,
        "tarball_sha256": manifest["sha256"],
        "request_digest": study["request_digest"],
    }


# --- study-local exact evaluation with per-edge couplings -----------------------------------


class EdgeGrid(GridIsing):
    """GridIsing with per-pixel couplings to the pixel above (ju) and to the left (jl)."""

    def __init__(self, ju: np.ndarray, jl: np.ndarray):
        super().__init__(L, 0.0)
        self.ju, self.jl = np.asarray(ju, dtype=float), np.asarray(jl, dtype=float)

    def _ju_jl(self, k: int) -> tuple[float, float]:
        r, c = divmod(k, self.L)
        return (float(self.ju[k]) if r > 0 else 0.0), (float(self.jl[k]) if c > 0 else 0.0)


def grid_arrays(model, wq: np.ndarray, kinds) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel up/left couplings from the archived edge list (grid edges only)."""
    ju, jl = np.zeros(N), np.zeros(N)
    for (a, b), v, k in zip(model.edges, wq, kinds, strict=True):
        if k != "grid":
            continue
        if b == a + L:
            ju[b] = v
        elif b == a + 1:
            jl[b] = v
        else:
            raise ValueError("unexpected grid edge")
    return ju, jl


def collapsed_general(p, units, model, kinds, wq, fields):
    """Tie-limit model from rounded vectors; per-edge grid and per-pixel clamp couplings."""
    kinds_a = np.array(kinds)
    out = dict(p)
    ju, jl = grid_arrays(model, wq, kinds)
    out["grid"] = EdgeGrid(ju, jl)
    clamp_edges = [e for e, k in zip(model.edges, kinds, strict=True) if k == "clamp"]
    g = np.empty(N)
    for (a, _), v in zip(clamp_edges, wq[kinds_a == "clamp"], strict=True):
        g[a] = v
    out["g"] = g
    W = np.zeros((N, p["C"]))
    seen = np.zeros((N, p["C"]), dtype=int)
    cw = wq[kinds_a == "copy"]
    e, k = 0, N
    bh = np.zeros(p["C"])
    for c, u in enumerate(units):
        for _, px in u["copies"]:
            for q in px:
                W[q, c] = cw[e]
                seen[q, c] += 1
                e += 1
            bh[c] += fields[k]
            k += 1
    if not (seen == 1).all():
        raise ValueError("a pixel does not couple to exactly one copy of every unit")
    out["W"], out["bp"], out["bh"] = W, fields[:N], bh
    return out


def exact_equilibrium_general(p, y, prune=1e-9, chunk=128):
    """``dl.exact_equilibrium`` with the grid object and a per-pixel clamp coupling."""
    C = p["C"]
    S = dl.hidden_states(C)
    hid = S @ p["bh"] + p["whh"] * ((S.sum(1) ** 2 - C) / 2.0)
    grid = p["grid"]
    base = p["bp"][None] + S @ p["W"].T
    out = np.empty((len(y), N))
    wts = np.empty((len(y), len(S)))
    for i in range(0, len(y), chunk):
        yy = y[i : i + chunk]
        fields = base[:, None, :] + p["g"] * yy[None]
        lz = grid.forward(fields.reshape(-1, N))[0].reshape(len(S), len(yy))
        lw = hid[:, None] + lz
        w = np.exp(lw - logsumexp(lw, axis=0))
        wts[i : i + chunk] = w.T
        acc = np.zeros((len(yy), N))
        for si in range(len(S)):
            keep = np.flatnonzero(w[si] > prune)
            if len(keep):
                pm, _ = grid.marginals(fields[si, keep])
                acc[keep] += w[si, keep, None] * pm
        out[i : i + chunk] = acc / wts[i : i + chunk].sum(1, keepdims=True).clip(1e-300)
    return out


def score(pe, x, bw, draws):
    mpm = (np.where(pe > 0.5, 1.0, -1.0) != x).sum(1)
    maj = dl.maj_wrong(pe, x, draws)
    return {
        "mpm_gap": float(((mpm - bw) / N).mean()),
        "maj_gap": float((maj - bw / N).mean()),
        "mpm_wrong": mpm.astype(int),
    }


# --- rounding rules ---------------------------------------------------------------------------


def classes(values: np.ndarray) -> np.ndarray:
    _, labels = np.unique(np.round(values, CLASS_DECIMALS), return_inverse=True)
    return labels


def round_vector(rule: str, values: np.ndarray, cap: float, bits: int) -> np.ndarray:
    if rule == "nearest":
        return quantize_array(values, cap, bits)
    if rule == "diffuse_class":
        out, _, _ = apr.diffuse(values, cap, bits, classes(values))
        return out
    raise ValueError(rule)


def rounded(rule, bits, wts, fields, caps, mask_w=None, mask_f=None):
    """Round both vectors; entries outside the masks stay float (ablations)."""
    cc, cf = caps
    wq = round_vector(rule, wts, cc, bits)
    fq = round_vector(rule, fields, cf, bits)
    if mask_w is not None:
        wq = np.where(mask_w, wq, wts)
    if mask_f is not None:
        fq = np.where(mask_f, fq, fields)
    return wq, fq


def class_table(values, rounded_by_rule, kind_names, cap, bits):
    """Per class: size, kinds, nominal, summed error per rule (bounded)."""
    labels = classes(values)
    step = cap / (2 ** (bits - 1) - 1)
    rows = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        kinds = sorted({kind_names[i] for i in idx})
        row = {
            "size": int(len(idx)),
            "kinds": kinds,
            "nominal": float(values[idx[0]]),
            "nominal_in_steps": float(values[idx[0]] / step),
        }
        for rule, q in rounded_by_rule.items():
            err = q[idx] - values[idx]
            row[rule] = {
                "sum_error": float(err.sum()),
                "sum_error_in_steps": float(err.sum() / step),
                "max_abs_error": float(np.abs(err).max()),
                "distinct_levels": int(len(np.unique(np.round(q[idx], 12)))),
            }
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["nearest"]["sum_error"]))
    return rows


# --- part 1: the placed tie-limit model --------------------------------------------------------


def placed_model_part(arch: dict, log) -> dict:
    req = arch["request"]
    study = arch["study"]
    ev = study["evaluation"]["t4_h05_q035"]
    w = ev["w_star"]
    lname = ev["carried_layout"]
    fam = req["families"][0]
    mix, x, y, _ = dl.images(req, 0, "test")
    ne = req["n_test_exact"]
    x, y = x[:ne], y[:ne]
    _, bw = dl.bayes(mix, x, y, fam["q"])
    bw = bw.astype(float)
    archived_rows = study["families"]["0"]["etest"]["rows"]
    bayes_ok = bool(
        np.array_equal(
            bw.astype(int), np.array(study["families"]["0"]["etest"]["bayes_wrong"][:ne])
        )
    )
    base, units, model, nfree, kinds = dl.placed(req, 0, lname, w)
    wts = np.asarray(model.weights, dtype=float)
    fields = np.asarray(model.bias[:nfree], dtype=float)
    kinds_a = np.array(kinds)
    field_kind = np.array(
        ["pixel"] * N
        + ["copy_bias"] * sum(len(u["copies"]) for u in units)
        + ["relay"] * sum(len(u["relay_nodes"]) for u in units)
    )
    assert len(field_kind) == nfree
    draws, prune = req["draws"], req["prune"]
    out = {
        "layout": lname,
        "w_star": w,
        "free_nodes": nfree,
        "bayes_mpm_error": float(bw.mean() / N),
        "bayes_recomputed_matches_archive": bayes_ok,
        "coefficient_counts": {
            "couplings": {k: int((kinds_a == k).sum()) for k in ("grid", "clamp", "copy", "tie")},
            "fields": {k: int((field_kind == k).sum()) for k in ("pixel", "copy_bias", "relay")},
        },
    }

    # control A: imported functions reproduce the archived rows (nearest, 4/6/8 and float)
    control = {}
    t0 = time.process_time()
    for bits in (None, *ARCHIVED_BITS):
        pc, caps = dl.collapsed(base, units, model, nfree, kinds, bits)
        row, pe_imp = dl.exact_row(pc, x, y, bw, draws, prune)
        arc = archived_rows[f"placed_{bits}"]
        # the archived 8-bit row ran on all 1,500 images and was scored on the first 1,000
        same_counts = row["mpm_wrong"] == arc["mpm_wrong"]
        control[str(bits)] = {
            "mpm_gap": row["mpm_gap"],
            "archived_mpm_gap": arc["mpm_gap"],
            "maj_gap": row["maj_gap"],
            "archived_maj_gap": arc["maj_gap"],
            "mpm_gap_equal": row["mpm_gap"] == arc["mpm_gap"],
            "maj_gap_abs_diff": abs(row["maj_gap"] - arc["maj_gap"]),
            "per_image_counts_equal": bool(same_counts),
            "caps": caps,
        }
        # control B: the study-local general evaluator reproduces the imported one bitwise
        if bits is None:
            wq, fq = wts, fields
        else:
            wq, fq = rounded("nearest", bits, wts, fields, (caps["coupling"], caps["field"]))
        pg = collapsed_general(base, units, model, kinds, wq, fq)
        pe_gen = exact_equilibrium_general(pg, y, prune=prune)
        control[str(bits)]["general_evaluator_bitwise"] = bool(np.array_equal(pe_gen, pe_imp))
        log(f"control bits={bits} gap={row['mpm_gap']:.6f} archived={arc['mpm_gap']:.6f}")
    out["control"] = control
    out["control_passed"] = bool(
        bayes_ok
        and all(
            c["mpm_gap_equal"]
            and c["per_image_counts_equal"]
            and c["maj_gap_abs_diff"] <= 1e-12
            and c["general_evaluator_bitwise"]
            for c in control.values()
        )
    )
    out["control_cpu_s"] = time.process_time() - t0
    if not out["control_passed"]:
        return out

    caps = cap_for("split", wts, fields, 8)
    out["caps"] = {"coupling": caps[0], "field": caps[1]}

    # main grid: both rules at every bit count
    grid = {}
    timing = []
    for rule in RULES:
        grid[rule] = {}
        for bits in BITS:
            t1 = time.process_time()
            wq, fq = rounded(rule, bits, wts, fields, caps)
            pg = collapsed_general(base, units, model, kinds, wq, fq)
            pe = exact_equilibrium_general(pg, y, prune=prune)
            sc = score(pe, x, bw, draws)
            levels = 2 ** (bits - 1) - 1
            grid[rule][str(bits)] = {
                "mpm_gap": sc["mpm_gap"],
                "maj_gap": sc["maj_gap"],
                "passes": sc["mpm_gap"] <= TOL,
                "exploration_only": bits not in ROW_BITS,
                "clipped_couplings": int(np.sum(np.abs(wq) > caps[0] * (1 + 1e-12))),
                "couplings_on_codebook": bool(
                    np.allclose(codebook(wq, caps[0], bits), wq, 0, 1e-12)
                ),
                "fields_on_codebook": bool(np.allclose(codebook(fq, caps[1], bits), fq, 0, 1e-12)),
                "levels": levels,
                "grid_coupling_mean": float(wq[kinds_a == "grid"].mean()),
                "clamp_coupling_mean": float(wq[kinds_a == "clamp"].mean()),
                "copy_bias_sums": [float(v) for v in pg["bh"]],
            }
            timing.append(time.process_time() - t1)
            log(f"{rule} bits={bits} gap={sc['mpm_gap']:.6f}")
    out["grid"] = grid
    out["nominal"] = {
        "J": float(base["J"]),
        "g": float(base["g"]),
        "bh_unit_sums": [float(v) for v in base["bh"]],
    }

    def first_pass(rule, stable=False):
        ok = [b for b in ROW_BITS if grid[rule][str(b)]["passes"]]
        if not stable:
            return ok[0] if ok else None
        s = None
        for b in reversed(ROW_BITS):
            if not grid[rule][str(b)]["passes"]:
                break
            s = b
        return s

    out["b_star"] = {r: first_pass(r) for r in RULES}
    out["b_stable"] = {r: first_pass(r, True) for r in RULES}

    # mechanism at 4 bits: class tables and one-kind ablations
    bits = 4
    rw = {r: round_vector(r, wts, caps[0], bits) for r in RULES}
    rf = {r: round_vector(r, fields, caps[1], bits) for r in RULES}
    out["classes_4bit"] = {
        "couplings": class_table(wts, rw, list(kinds_a), caps[0], bits),
        "fields": class_table(fields, rf, list(field_kind), caps[1], bits),
    }
    ablation = {}
    for kind in ("grid", "clamp", "copy", "tie", "pixel", "copy_bias"):
        if kind in ("pixel", "copy_bias"):
            mw, mf = np.zeros(len(wts), bool), field_kind == kind
        else:
            mw, mf = kinds_a == kind, np.zeros(nfree, bool)
        ablation[kind] = {}
        for rule in RULES:
            wq, fq = rounded(rule, bits, wts, fields, caps, mw, mf)
            pg = collapsed_general(base, units, model, kinds, wq, fq)
            pe = exact_equilibrium_general(pg, y, prune=prune)
            sc = score(pe, x, bw, draws)
            ablation[kind][rule] = {"mpm_gap": sc["mpm_gap"], "maj_gap": sc["maj_gap"]}
            log(f"ablation {kind} {rule} gap={sc['mpm_gap']:.6f}")
    out["ablation_4bit_one_kind_rounded"] = ablation
    out["cpu_s_per_exact_row"] = {
        "median": float(np.median(timing)),
        "max": float(np.max(timing)),
        "rows": len(timing),
    }
    return out


# --- part 2: am-coupling-bits class sizes -----------------------------------------------------


def am_part(log) -> dict:
    record, sha = load_stage_a()
    req = amcb.study_request()
    if sha != req["stage_a_archive_sha256"]:
        raise SystemExit("stage A archive SHA-256 differs")
    out = {"stage_a_archive_sha256": sha, "cells": {}}
    for cell in amcb.cells(req):
        cfg = record["evaluation"]["equilibrium"][f"{cell['id']}/bias"]["config"]
        beta, theta = amcb.parse_config(cfg)
        per_bits = {b: [] for b in (6, 8, 10)}
        sizes = []
        for seed in req["held_seeds"]:
            xi = stage_a.patterns(seed, cell["p"])
            z = amcb.nominal_terms(beta, theta, xi)
            coup = z["j_vh"].ravel()
            flds = np.concatenate([z["b_v"], z["b_h"]])
            for name, vec in (("couplings", coup), ("fields", flds)):
                lab = classes(vec)
                cnt = np.bincount(lab)
                sizes.append((name, int(len(cnt)), int(cnt.max()), len(vec)))
            for b in per_bits:
                cc, cf = amcb.caps_for(z, "split", b)
                for name, vec, cap in (("couplings", coup, cc), ("fields", flds, cf)):
                    lab = classes(vec)
                    q = amcb.codebook(vec, cap, b)
                    err = q - vec
                    step = cap / (2 ** (b - 1) - 1)
                    off = []
                    for lab_i in np.unique(lab):
                        idx = lab == lab_i
                        if np.abs(err[idx]).max() > 1e-12:
                            off.append((int(idx.sum()), float(abs(err[idx].sum()) / step)))
                    per_bits[b].append(
                        {
                            "vector": name,
                            "off_codebook_classes": len(off),
                            "largest_off_class": max((s for s, _ in off), default=0),
                            "max_class_sum_error_steps": max((e for _, e in off), default=0.0),
                        }
                    )
        summary = {}
        for name in ("couplings", "fields"):
            rows = [s for s in sizes if s[0] == name]
            summary[name] = {
                "entries": rows[0][3],
                "classes_range": [min(r[1] for r in rows), max(r[1] for r in rows)],
                "largest_class_range": [min(r[2] for r in rows), max(r[2] for r in rows)],
            }
            for b, recs in per_bits.items():
                rr = [r for r in recs if r["vector"] == name]
                summary[name][f"bits_{b}"] = {
                    "off_codebook_classes_max": max(r["off_codebook_classes"] for r in rr),
                    "largest_off_class_max": max(r["largest_off_class"] for r in rr),
                    "max_class_sum_error_steps": max(r["max_class_sum_error_steps"] for r in rr),
                }
        out["cells"][cell["id"]] = {"config": cfg, "held_sets": len(req["held_seeds"]), **summary}
        log(f"am {cell['id']} done")
    return out


# --- part 3: adsorption pore at 2 and 3 bits --------------------------------------------------


def adsorption_part(log) -> dict:
    request = apr.study_request()
    mus = request["frozen"]["mu"]
    full = apr.full_cells(mus)
    cap = request["caps"]["family_field"]
    out = {}
    for rule in apr.RULES:
        out[rule] = {}
        for bits in (2, 3, 4):
            row = [
                apr.evaluate_cell(mu, rule, bits, cap, ref)
                for mu, ref in zip(mus, full, strict=True)
            ]
            loading = max(abs(c["loading_error"]) for c in row)
            pair = max(abs(c["pair_error"]) for c in row)
            capacity = abs(row[-1]["loading_error"] - row[0]["loading_error"])
            out[rule][str(bits)] = {
                "max_abs_loading_error": loading,
                "abs_capacity_error": capacity,
                "max_abs_pair_error": pair,
                "clipped_sites": sum(c["clipped_sites"] for c in row),
                "couplings_exact": all(c["couplings_exact"] for c in row),
                "passes": loading <= apr.TOL["loading"]
                and capacity <= apr.TOL["capacity"]
                and pair <= apr.TOL["pair"],
            }
        log(f"adsorption {rule} done")
    return {"frozen_mu": mus, "tolerances": apr.TOL, "rules": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    t0 = time.time()
    c0 = time.process_time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    arch = load_archive()
    log(f"archive authenticated {arch['tarball_sha256']}")
    record = {
        "label": "exploration, not evidence (loop P-0010 probe)",
        "evidence_class": "exact_reference (float64 enumeration times transfer matrix, CPU)",
        "hardware_claim": False,
        "archive": {
            "path": str(REPORT.relative_to(ROOT)),
            "tarball_sha256": arch["tarball_sha256"],
            "request_digest": arch["request_digest"],
        },
        "rules": list(RULES),
        "bits": list(BITS),
        "row_bits": list(ROW_BITS),
        "tolerance": TOL,
        "class_decimals": CLASS_DECIMALS,
    }
    record["placed_tie_limit"] = placed_model_part(arch, log)
    if record["placed_tie_limit"]["control_passed"]:
        record["am_coupling_bits_classes"] = am_part(log)
        record["adsorption_low_bits"] = adsorption_part(log)
    record["cpu_seconds"] = time.process_time() - c0
    record["wall_seconds"] = time.time() - t0
    Path(args.output).write_text(json.dumps(record, indent=1, sort_keys=True, default=_js) + "\n")
    log(f"wrote {args.output}")


def _js(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))


if __name__ == "__main__":
    main()
