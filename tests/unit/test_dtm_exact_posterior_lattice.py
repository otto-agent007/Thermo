"""Checks for the DTM exact-posterior lattice study (P-0008).

Probe equality is bitwise (same host, same code path). The small-run test runs the
CLI under the single-thread BLAS pin, interrupts it, resumes it, and checks that
the resumed record is identical and that a tampered archive fails replay. Unit
tests use the probe seed and small requests only; the production seeds
(202613xx) appear only in the request-shape test. Comparisons against an archive
written on another host are bitwise only with ``THERMO_ARCHIVE_HOST=1``;
otherwise floats compare at the request's replay tolerance.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import dtm_exact_posterior_lattice as ep
from thermo_lab import dtm_small_image as dtm

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "docs/research"
PINNED_ENV = {"JAX_PLATFORMS": "cpu", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
ARCHIVE_HOST = os.environ.get("THERMO_ARCHIVE_HOST") == "1"
SEED = ep.PROBE_SEED


def _probes():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import dtm_exact_posterior_lattice_layout4 as l4
        import dtm_exact_posterior_lattice_placed as placed
        import dtm_exact_posterior_lattice_probe as probe
    finally:
        sys.path.remove(str(PROBE_DIR))
    return probe, placed, l4


def _cli(arguments, env=None, check=True):
    result = subprocess.run(
        [sys.executable, "-m", "thermo_lab.dtm_exact_posterior_lattice", *arguments],
        env=env or {**os.environ, **PINNED_ENV},
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stdout[-3000:] + result.stderr[-4000:])
    return result


def small_request(**extra):
    base = dict(
        families=["t4_h05_q035"],
        w_grid=[1.5, 2.0],
        draws=4,
        n_dev=10,
        dev_streams=2,
        n_test_exact=20,
        streams=2,
        images_per_stream=5,
        sampler_seeds=[0, 1],
        K_grid=[2, 4],
        K_dev=2,
        K_ref=4,
        bootstrap_resamples=50,
        preflight={
            "seed": SEED,
            "kernel_chains": 40000,
            "kernel_sweeps": [1, 2],
            "max_z": 5.0,
            "oracle_images": 10,
            "calibration_streams": 2,
            "calibration_K": 4,
        },
        seeds={
            "test": SEED + 1,
            "dev": SEED + 2,
            "dev_sampler": SEED + 3,
            "stage_s": SEED + 4,
            "float_ref": SEED + 5,
            "jitter_words": SEED + 6,
            "bootstrap": SEED + 7,
            "oracle_pair": SEED + 8,
        },
    )
    return ep.make_request(**(base | extra))


def test_single_thread_blas_pin(tmp_path):
    unpinned = {**os.environ, **PINNED_ENV, "OPENBLAS_NUM_THREADS": "4"}
    failed = _cli(["--output-dir", str(tmp_path / "x"), "--replay"], env=unpinned, check=False)
    assert failed.returncode != 0
    assert "OPENBLAS_NUM_THREADS=1" in failed.stderr
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "run pytest under the BLAS pin"


def test_production_request_matches_protocol():
    r = ep.make_request()
    assert r["protocol_commit"] == "d4f8b2a7f498743ce15170d45a14d555bb1a21fd"
    assert [f["name"] for f in r["families"]] == ["t4_h05_q035", "smooth8_h05_q035"]
    assert [f["components"] for f in r["families"]] == [4, 8]
    assert [f["b0"] for f in r["families"]] == [-5.0, -15.0]
    assert [(f["J"], f["h0"], f["q"]) for f in r["families"]] == [(0.35, 0.5, 0.35)] * 2
    assert [f["admission_requires_maj"] for f in r["families"]] == [False, True]
    assert r["w_grid"] == [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    assert (r["bits_operating"], r["jitter"], r["tol"], r["per_seed_bound"]) == (
        8,
        0.1,
        0.01,
        0.015,
    )
    assert (r["draws"], r["n_dev"], r["n_test_exact"], r["n_test_total"]) == (64, 300, 1000, 1500)
    assert (r["streams"], r["images_per_stream"], r["sampler_seeds"]) == (25, 20, [0, 1, 2])
    assert (r["K_grid"], r["K_dev"], r["K_ref"], r["reset_every"]) == (
        [64, 256, 1024],
        1024,
        4096,
        5,
    )
    assert r["cpu_stop_hours"] == 10.0 and r["oracle_agree_bound"] == 0.01
    assert r["seeds"] == {
        "test": 20261301,
        "dev": 20261302,
        "dev_sampler": 20261303,
        "stage_s": 20261304,
        "float_ref": 20261305,
        "jitter_words": 20261306,
        "bootstrap": 20261307,
        "oracle_pair": 20261308,
    }
    free = {
        ln: {k: v["free_nodes"] for k, v in s["families"].items()} for ln, s in r["layouts"].items()
    }
    assert free == {
        "index5": {"t4_h05_q035": 197, "smooth8_h05_q035": 339},
        "index4": {"t4_h05_q035": 189, "smooth8_h05_q035": 332},
    }
    cells = ep.stage_s_cells(r, 0, "index4", 2.0)
    assert len(cells) == 90
    assert sum(c["kind"] == "s" for c in cells) == 81
    assert sum(c["kind"] == "float" for c in cells) == 3
    assert sum(c["kind"] == "pair" for c in cells) == 6
    op = ep.arm_request(r, 0, 1, "op")
    assert op.relaxed == () and op.bits == 8 and op.cap_rule == "split"
    assert op.jitter == 0.1 and op.jitter_words == (20261306, 0, 1)
    pair = ep.arm_request(r, 0, 1, "pair").to_canonical()
    assert {k: v for k, v in pair.items() if k != "model_id"} == {
        k: v for k, v in op.to_canonical().items() if k != "model_id"
    }, "the oracle pair runs on the operating arm (same bits, jitter and jitter words)"
    assert ep.arm_request(r, 0, 1, "jit0").relaxed == ("beta_jitter",)
    b6 = ep.arm_request(r, 0, 1, "b6")
    assert b6.bits == 6 and b6.relaxed == ("beta_jitter",)
    fl = ep.arm_request(r, 0, 1, "float")
    assert fl.relaxed == ("beta_jitter", "quantization") and fl.bits is None


def test_model_equals_probe_bitwise():
    probe, _, _ = _probes()
    for name, b0 in (("t4_h05_q035", -5.0), ("smooth8_h05_q035", -15.0)):
        J, h, q = probe.family(name)
        mine = ep.build(J, h, q, b0)
        theirs = probe.build(J, h, q, "bias", "centered", b0=b0)
        for bits in (None, 8, 6, 4):
            a, b = ep.quantize_unplaced(mine, bits), probe.quantize(theirs, bits)
            for k in ("J", "g", "W", "bp", "bh", "whh", "C"):
                assert np.array_equal(a[k], b[k]), (name, bits, k)
        oh, toh = ep.build_onehot(J, h, q), probe.build(J, h, q, "onehot", "centered", lam=0.0)
        for k in ("J", "g", "W", "bp", "bh", "whh"):
            assert np.array_equal(oh[k], toh[k])
    J, h, q = probe.family("t4_h05_q035")
    _, x, y = probe.test_set("t4_h05_q035", 24, 1)
    p = probe.quantize(probe.build(J, h, q, "bias", "centered", b0=-5.0), 8)
    a, wa, _ = ep.exact_equilibrium(p, y)
    b, wb, _ = probe.exact_equilibrium(p, y)
    assert np.array_equal(a, b) and np.array_equal(wa, wb)
    _, placed, _ = _probes()
    assert np.array_equal(ep.maj_wrong(a, x, 64), placed.maj_wrong(a, x, 64))


@pytest.mark.parametrize("name", ["t4_h05_q035", "smooth8_h05_q035"])
def test_layout_and_tie_limit_equal_probe_bitwise(name):
    probe, _, l4 = _probes()
    J, h, q = probe.family(name)
    base = probe.build(J, h, q, "bias", "centered", b0=-5.0)
    for lname in ("index5", "index4"):
        basis, clamp, ureps, order = ep.layout_config(lname, len(h))
        tb, cands = l4.configs(lname, len(h))
        theirs_cfg = next(
            c for c in cands if tuple(c[0]) == tuple(clamp) and c[2][: len(h)] == order[: len(h)]
        )
        tunits = l4.build_layout(tb, len(h), *theirs_cfg)
        units = ep.build_layout(basis, len(h), clamp, ureps, order)
        assert units == tunits
        for w in (1.5, 2.25):
            model, nf, kinds = ep.to_pairwise(units, basis, clamp, base, w)
            tmodel, tnf, tkinds = l4.to_pairwise(tunits, tb, clamp, base, w)
            assert (nf, kinds, model.sites, model.edges) == (
                tnf,
                tkinds,
                tmodel.sites,
                tmodel.edges,
            )
            assert np.array_equal(model.weights, tmodel.weights)
            assert np.array_equal(model.bias, tmodel.bias)
            for bits in (8, None, 6, 4):
                a, ca = ep.collapsed(base, units, model, nf, kinds, bits)
                b, cb = l4.collapsed(base, tunits, tmodel, tnf, tkinds, bits)
                assert ca == cb
                for k in ("J", "g", "W", "bp", "bh"):
                    assert np.array_equal(a[k], b[k]), (lname, w, bits, k)
        comp = np.arange(30) % len(h)
        assert np.array_equal(
            ep.oracle_states(units, comp, nf, 30)[:, ep.N :],
            l4.oracle_states(tunits, comp, tnf, 30)[:, ep.N :],
        )


def test_gibbs_equals_probe_bitwise():
    probe, _, l4 = _probes()
    J, h, q = probe.family("t4_h05_q035")
    base = probe.build(J, h, q, "bias", "centered", b0=-5.0)
    basis, clamp, ureps, order = ep.layout_config("index4", 4)
    units = ep.build_layout(basis, 4, clamp, ureps, order)
    model, nf, kinds = ep.to_pairwise(units, basis, clamp, base, 2.0)
    m = ep.quantized_sampler(model, nf, kinds, 8)
    tm = l4.quantized_sampler(model, nf, kinds, 8)
    assert (m["W"] != tm["W"]).nnz == 0 and np.array_equal(m["b"], tm["b"])
    _, x, y = probe.test_set("t4_h05_q035", 1000, 1)
    y = y[:6]
    comp = np.arange(6) % 4
    orc = ep.oracle_states(units, comp, nf, 6)
    for start in ("random", "oracle"):
        a, _ = ep.gibbs(m, y, 3, start, np.random.default_rng([SEED, 1]), 2, 3, 64, oracle=orc)
        b = l4.gibbs(m, y, 3, start, np.random.default_rng([SEED, 1]), 2, 3, orc)
        assert np.array_equal(a, b), start
        c, _ = ep.gibbs(
            m,
            y,
            3,
            start,
            np.random.default_rng([SEED, 1]),
            2,
            3,
            64,
            oracle=orc,
            gains=np.ones(nf),
        )
        assert np.array_equal(a, c), "unit gains must not change the draw"
    # reset every 1 image is a random start; reset every > per images is continuation
    a, _ = ep.gibbs(m, y, 2, "random", np.random.default_rng([SEED, 2]), 2, 3, 8)
    b, _ = ep.gibbs(m, y, 2, "reset", np.random.default_rng([SEED, 2]), 2, 3, 8, reset_every=1)
    assert np.array_equal(a, b)
    # continuation never redraws; reset with a period longer than the stream equals it
    a, _ = ep.gibbs(m, y, 2, "continue", np.random.default_rng([SEED, 3]), 2, 3, 8)
    b, _ = ep.gibbs(m, y, 2, "reset", np.random.default_rng([SEED, 3]), 2, 3, 8, reset_every=99)
    c, _ = ep.gibbs(m, y, 2, "continue", np.random.default_rng([SEED, 3]), 2, 3, 8)
    assert np.array_equal(a, c)
    assert not np.array_equal(a, b), "reset redraws the state at image 0, consuming the rng"


def test_pricing_matches_probe_and_hand_count():
    probe, _, _ = _probes()
    r = small_request()
    assert ep.pricing_check(r)["passed"]
    for start, reset in (("continue", None), ("random", None), ("reset", 5)):
        mine = ep.price_sample(ep.make_request(), 0, 189, 1024, start)
        theirs = probe.price(189, 1024, start, reset)
        for k, t in (
            ("joule_per_sample_write_constant", "joule_per_sample_write_constant"),
            ("joule_per_sample_clamp_at_read", "joule_per_sample_clamp_at_read"),
            ("writes", "writes"),
            ("free_node_writes", "free_node_writes"),
            ("updates", "updates"),
        ):
            assert abs(mine[k] - theirs[t]) <= 1e-12 * max(1.0, abs(theirs[t])), (start, k)
    # protocol numbers: t4 index 4, random start, 2.58e-6 J per 64-draw estimate at K = 1024
    p = ep.price_sample(ep.make_request(), 0, 189, 1024, "random")
    assert abs(p["joule_per_estimate_write_constant"] - 2.58e-6) < 0.01e-6


def _strip(study):
    return {k: v for k, v in study["evaluation"].items()}


def test_small_run_replays_resumes_and_detects_tampering(tmp_path):
    request = small_request()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    full = tmp_path / "full"
    _cli(["--output-dir", str(full), "--request", str(path), "--workers", "2"])
    completion = json.loads((full / "completion.json").read_text())
    assert completion["status"] == "dtm_exact_posterior_lattice_complete"
    assert completion["replayed"] and completion["prefix_check_passed"]
    assert completion["preflight_passed"] and completion["placement_passed"]
    assert completion["references_recomputed"]
    assert completion["families_run"] == 1 and completion["exact_rows"] == 2 * 2 + 8
    v = completion["verdicts"]["t4_h05_q035"]
    n_dev = len(v["admitted"][v["carried_layout"]])
    split = completion["sampled_cells_by_family"]["t4_h05_q035"]
    assert split["development_cells"] == n_dev
    assert split["stage_s_and_reference_cells"] == (0 if v["stop_a"] else 2 * (3 * 3 * 2 + 3))
    assert completion["sampled_cells"] == n_dev + split["stage_s_and_reference_cells"]
    # interrupted then resumed run gives an identical record
    part = tmp_path / "part"
    stopped = _cli(
        ["--output-dir", str(part), "--request", str(path), "--workers", "1", "--stop-after", "4"],
        check=False,
    )
    assert stopped.returncode == 130
    assert not (part / "completion.json").exists()
    changed = small_request(n_test_exact=10)
    path2 = tmp_path / "changed.json"
    path2.write_text(json.dumps(changed))
    refused = _cli(["--output-dir", str(part), "--request", str(path2), "--resume"], check=False)
    assert refused.returncode != 0
    assert "stored request differs" in refused.stderr
    _cli(["--output-dir", str(part), "--request", str(path), "--workers", "2", "--resume"])
    assert "attempt 2 (resume)" in (part / "run.log").read_text()
    a = dtm.read_gz_json(full / "study.json.gz")
    b = dtm.read_gz_json(part / "study.json.gz")
    assert a["result_digest"] == b["result_digest"]
    assert _strip(a) == _strip(b)
    light = _cli(["--output-dir", str(full), "--replay", "--light"])
    assert json.loads(light.stdout)["replayed"]
    # a tampered archive fails replay
    bad = tmp_path / "bad"
    shutil.copytree(full, bad)
    study = dtm.read_gz_json(bad / "study.json.gz")
    fam = study["families"]["0"]
    if "cells" in fam:
        name = next(iter(fam["cells"]))
        fam["cells"][name]["maj_wrong"][0] = (fam["cells"][name]["maj_wrong"][0] + 1) % 64
    else:
        w = next(iter(fam["dev"]))
        fam["dev"][w][0] = (fam["dev"][w][0] + 1) % 64
    dtm.write_gz_json(bad / "study.json.gz", study)
    failed = _cli(["--output-dir", str(bad), "--replay", "--light"], check=False)
    assert failed.returncode != 0
    assert "digest" in failed.stderr


REPORT = sorted((ROOT / "docs/experiment-reports").glob("*-dtm-exact-posterior-lattice"))


@pytest.mark.skipif(not REPORT, reason="no committed archive yet")
def test_committed_archive_light_replay(tmp_path):
    report = REPORT[-1]
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    out = next(p.parent for p in tmp_path.rglob("study.json.gz"))
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    completion = ep.replay(out, light=True)
    committed = json.loads((report / "completion.json").read_text())
    for key in ("result_digest", "families_run", "exact_rows", "sampled_cells", "verdicts"):
        assert completion[key] == committed[key]
    assert ARCHIVE_HOST or completion["references_recomputed"]
