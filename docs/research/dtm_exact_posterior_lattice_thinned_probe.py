"""Thinned-draw probe on P-0008's archived model (exploration, not recorded evidence).

Question (direction review D-0005, premise option 2): may one clamp load serve the
64 draws of a majority estimate? P-0008 priced one clamp load per draw because its
estimate is the pixelwise majority of 64 independent chains, each loading the noisy
image. This probe runs one chain per image, loads the clamp once, and reads the
pixel states every ``lag`` sweeps, 64 times, so the sweep count per estimate equals
the archived cell's at K = lag. The thinned majority is compared, image by image,
with the archived 64-chain majority on the same images under the same placed,
quantized, jittered model (``t4_h05_q035``, index 4, w* = 1.75, 8-bit ``split``,
10 percent jitter, operating arm), and the thinned path is priced on the sealed
profile with the layer's ``count``/``price``.

Everything sampled here is ``software_simulation`` (NumPy two-colour Gibbs, the
runner's own loop, not the lowered THRML program). Joules are
``calibrated_projection``. Nothing ran on hardware. Fresh RNG seeds; nothing here
touches the archive or any hash-bound source.

Run from the repository root::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
        docs/research/dtm_exact_posterior_lattice_thinned_probe.py --out <dir> [--smoke]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import multiprocessing
import os
import tarfile
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from thermo_lab import dtm_exact_posterior_lattice as x8
from thermo_lab.constraints import count, price, request_gains, require_placement

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/experiment-reports/2026-10-17-dtm-exact-posterior-lattice/evidence.tar.gz"
PROBE_SEED = 20261318
FAMILY = 0  # t4_h05_q035
LAYOUT = "index4"
W_STAR = 1.75
N = x8.N
_STATE: dict = {}


def load_archive():
    """Per-image majority counts of the archived cells and the Bayes counts."""
    with tarfile.open(ARCHIVE) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("study.json.gz"))
        raw = tf.extractfile(member).read()
    study = json.loads(gzip.decompress(raw))
    fam = study["families"][str(FAMILY)]
    cells = {k: np.asarray(v["maj_wrong"], dtype=np.float64) for k, v in fam["cells"].items()}
    bayes_wrong = np.asarray(fam["etest"]["bayes_wrong"], dtype=np.float64)
    return cells, bayes_wrong


def init_worker():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    req = x8.make_request()
    _, units, model, nfree, kinds = x8.placed(req, FAMILY, LAYOUT, W_STAR)
    _, x, y, comp = x8.images(req, FAMILY, "test")
    _STATE.update(req=req, units=units, model=model, nfree=nfree, kinds=kinds, x=x, y=y)


def thinned_chain(m, x, y, lag, start, rng, streams, per, draws, gains, copy_units):
    """One chain per image; clamp loaded once; read every ``lag`` sweeps, ``draws`` times.

    ``random``: every image starts its chain from a fresh uniform state (all images run
    as parallel rows). ``continue``: one chain per stream carries its state from the
    last read of the previous image, exactly as the protocol's continuation start,
    so streams run as rows and images step in order.
    """
    n = m["n"]
    blocks = [np.flatnonzero(m["colour"] == c) for c in (0, 1)]
    WT = [m["W"][blk].T.tocsr() for blk in blocks]
    G = None if gains is None else [np.asarray(gains)[blk] for blk in blocks]
    n_img = len(y)
    if start == "random":
        rows, order = n_img, [np.arange(n_img)]
    else:
        rows, order = streams, [np.arange(streams) * per + j for j in range(per)]
    S = rng.integers(0, 2, (rows, n)) * 2.0 - 1.0
    maj_wrong = np.empty(n_img)
    read_err = np.empty((n_img, draws))
    flips = np.empty(n_img)
    agree = np.empty(n_img)
    for idx in order:
        if start == "random":
            S = rng.integers(0, 2, (rows, n)) * 2.0 - 1.0
        ext = np.tile(m["b"], (rows, 1))
        ext[:, :N] += m["g"] * y[idx]
        up = np.zeros((rows, N))
        prev = None
        nflip = np.zeros(rows)
        agr = np.zeros(rows)
        for r in range(draws):
            for _ in range(lag):
                for bi, (blk, wt) in enumerate(zip(blocks, WT, strict=True)):
                    f = (wt.T @ S.T).T + ext[:, blk]
                    if G is not None:
                        f = G[bi][None] * f
                    S[:, blk] = np.where(rng.random((rows, len(blk))) < expit(2 * f), 1.0, -1.0)
            pix = S[:, :N]
            up += pix > 0
            read_err[idx, r] = (np.where(pix > 0, 1.0, -1.0) != x[idx]).mean(1)
            unit_mean = np.stack([S[:, sl].mean(1) for sl in copy_units], 1)
            dom = unit_mean.argmax(1)
            if prev is not None:
                nflip += dom != prev
            prev = dom
            fr = []
            for sl in copy_units:
                v = S[:, sl]
                mj = np.where(v.sum(1, keepdims=True) >= 0, 1.0, -1.0)
                fr.append((v == mj).mean(1))
            agr += np.mean(fr, axis=0)
        pred = np.where(up >= draws // 2, 1.0, -1.0)  # ties to +1, as the runner
        maj_wrong[idx] = (pred != x[idx]).sum(1)
        flips[idx] = nflip / (draws - 1)
        agree[idx] = agr / draws
    return maj_wrong, read_err, flips, agree


def run_job(job):
    st = _STATE
    req, s, lag, start = st["req"], job["s"], job["lag"], job["start"]
    creq = x8.arm_request(req, FAMILY, s, "op")
    require_placement(st["model"], creq)
    m = x8.quantized_sampler(st["model"], st["nfree"], st["kinds"], creq.bits)
    gains = request_gains(creq, [st["nfree"]])[0]
    sl = x8.seed_slice(req, s)
    x, y = st["x"][sl], st["y"][sl]
    n_img = job.get("n_img")
    streams, per = req["streams"], req["images_per_stream"]
    if n_img:  # smoke: whole streams only
        streams = max(1, n_img // per)
        x, y = x[: streams * per], y[: streams * per]
    rng = np.random.default_rng([PROBE_SEED, FAMILY, s, lag, 0 if start == "random" else 1])
    t0 = time.process_time()
    mw, read_err, flips, agree = thinned_chain(
        m,
        x,
        y,
        lag,
        start,
        rng,
        streams,
        per,
        req["draws"],
        gains,
        x8.unit_slices(st["units"])[: len(st["units"])],
    )
    cpu = time.process_time() - t0
    return {
        "s": s,
        "lag": lag,
        "start": start,
        "n_img": int(len(x)),
        "streams": streams,
        "maj_wrong": mw.astype(int).tolist(),
        "read_err_curve": read_err.mean(0).tolist(),
        "flip_rate": float(flips.mean()),
        "agreement": float(agree.mean()),
        "cpu_s": cpu,
        "sweeps_per_image": int(req["draws"] * lag),
    }


def thinned_price(req, nfree, lag, start):
    """Per 64-draw estimate on the thinned path: one clamp load, 64 reads per draw."""
    creq = x8.arm_request(req, FAMILY, req["sampler_seeds"][0], "op")
    draws = req["draws"]
    fw = {"continue": 0, "random": nfree}[start]
    c = count(
        creq,
        logical_pbits=nfree + N,
        physical_pbits_used=nfree + N,
        free_pbits=nfree,
        sweeps=draws * lag,
        node_reads=N * draws,
        node_full_sram_writes=N + fw,
    )
    p = price(creq, c)
    hi = p["modeled_total_energy_j"]
    lo = hi - N * x8.WRITE_J + N * x8.READ_J
    return {
        "clamp_loads": N,
        "reads": int(c.node_reads),
        "writes": int(c.node_full_sram_writes),
        "free_node_writes": fw,
        "updates": int(c.gibbs_node_updates),
        "joule_per_estimate_write_constant": hi,
        "joule_per_estimate_clamp_at_read": lo,
        "evidence_class": p["evidence_class"],
    }


def stream_boot(d, streams, per, rng, n=2000):
    g = d.reshape(streams, per).mean(1)
    b = g[rng.integers(0, streams, (n, streams))].mean(1)
    return float(g.mean()), np.quantile(b, [0.025, 0.975]).tolist()


def summarize(req, nfree, results, archive, bayes_wrong):
    per = req["images_per_stream"]
    rng = np.random.default_rng([PROBE_SEED, 7])
    rows = []
    for r in results:
        s, lag, start = r["s"], r["lag"], r["start"]
        sl = x8.seed_slice(req, s)
        bw = bayes_wrong[sl][: r["n_img"]]
        mw = np.asarray(r["maj_wrong"], dtype=np.float64)
        gap, gap_ci = stream_boot((mw - bw) / N, r["streams"], per, rng)
        ref_name = f"cell__f{FAMILY}_s{s}_op_{start}_K{lag}"
        if ref_name not in archive and lag == req["K_ref"] and start == "random":
            ref_name = f"cell__f{FAMILY}_s{s}_pair_random_K{lag}"
        row = {
            "seed": s,
            "lag": lag,
            "start": start,
            "n_img": r["n_img"],
            "thinned_gap": gap,
            "thinned_gap_ci95": gap_ci,
            "flip_rate_between_reads": r["flip_rate"],
            "copy_agreement": r["agreement"],
            "first_read_err_minus_bayes": float(r["read_err_curve"][0] - bw.mean() / N),
            "last_read_err_minus_bayes": float(r["read_err_curve"][-1] - bw.mean() / N),
            "cpu_s": r["cpu_s"],
            "reference": None,
        }
        if ref_name in archive:
            ref = archive[ref_name][: r["n_img"]]
            rgap, rci = stream_boot((ref - bw) / N, r["streams"], per, rng)
            dgap, dci = stream_boot((mw - ref) / N, r["streams"], per, rng)
            row["reference"] = {
                "cell": ref_name,
                "gap": rgap,
                "gap_ci95": rci,
                "thinned_minus_reference": dgap,
                "thinned_minus_reference_ci95": dci,
            }
        rows.append(row)
    pooled = {}
    for lag in sorted({r["lag"] for r in rows}):
        for start in ("random", "continue"):
            sel = [r for r in rows if r["lag"] == lag and r["start"] == start]
            if not sel:
                continue
            key = f"{start}_lag{lag}"
            entry = {
                "thinned_pooled_gap": float(np.mean([r["thinned_gap"] for r in sel])),
                "per_seed": {str(r["seed"]): r["thinned_gap"] for r in sel},
                "price_per_estimate": thinned_price(req, nfree, lag, start),
            }
            refs = [r["reference"] for r in sel if r["reference"]]
            if refs:
                entry["reference_pooled_gap"] = float(np.mean([q["gap"] for q in refs]))
                entry["reference_cells"] = [q["cell"] for q in refs]
                entry["thinned_minus_reference_pooled"] = float(
                    np.mean([q["thinned_minus_reference"] for q in refs])
                )
            pooled[key] = entry
    # archived per-estimate prices for the same K, from the runner's own pricing
    archived_price = {
        f"{start}_K{K}": x8.price_sample(req, FAMILY, nfree, K, start)
        for K in sorted({r["lag"] for r in rows})
        for start in ("continue", "random")
    }
    return {"cells": rows, "pooled": pooled, "archived_independent_price": archived_price}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lags", default="64,256,1024,4096")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lags = [int(v) for v in args.lags.split(",")]
    init_worker()
    req, nfree = _STATE["req"], _STATE["nfree"]
    archive, bayes_wrong = load_archive()
    jobs = []
    for s in req["sampler_seeds"]:
        for lag in lags:
            for start in ("random", "continue"):
                if lag == 64 and start == "continue":
                    continue
                jobs.append({"s": s, "lag": lag, "start": start})
    if args.smoke:
        jobs = [
            {"s": 0, "lag": 64, "start": "random", "n_img": 40},
            {"s": 0, "lag": 64, "start": "continue", "n_img": 40},
        ]
    jobs.sort(key=lambda j: -j["lag"] * (2 if j["start"] == "continue" else 1))
    t0 = time.time()
    results = []
    ctx = multiprocessing.get_context("spawn")
    pool = concurrent.futures.ProcessPoolExecutor
    with pool(args.workers, mp_context=ctx, initializer=init_worker) as ex:
        for r in ex.map(run_job, jobs):
            results.append(r)
            (out / f"cell_s{r['s']}_{r['start']}_lag{r['lag']}.json").write_text(json.dumps(r))
            print(
                f"done s{r['s']} {r['start']} lag {r['lag']} cpu {r['cpu_s']:.0f}s "
                f"flip {r['flip_rate']:.3f} agree {r['agreement']:.3f}",
                flush=True,
            )
    summary = summarize(req, nfree, results, archive, bayes_wrong)
    summary["wall_s"] = time.time() - t0
    summary["evidence_class"] = {
        "gaps": "software_simulation (NumPy two-colour Gibbs; not the lowered THRML program)",
        "joules": "calibrated_projection (sealed Z1 profile)",
        "label": "exploration, not recorded evidence",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for key, e in summary["pooled"].items():
        ref = e.get("reference_pooled_gap")
        d = e.get("thinned_minus_reference_pooled")
        pr = e["price_per_estimate"]
        hi = pr["joule_per_estimate_write_constant"]
        lo = pr["joule_per_estimate_clamp_at_read"]
        line = f"{key:18s} thinned {e['thinned_pooled_gap']:+.4f}"
        if ref is not None:
            line += f"  ref {ref:+.4f}  diff {d:+.4f}"
        print(line + f"  J/est {hi:.2e} / {lo:.2e}")


if __name__ == "__main__":
    main()
