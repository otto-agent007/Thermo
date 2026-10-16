"""Checks for the DTM baseline-headroom study: probe equality, THRML bitwise, replay, resume.

Every runner call passes the single-thread BLAS pin. Unit tests use probe seeds
and a small request only; the production seeds (202612xx) appear nowhere here.
Ladder fits run in worker processes pinned to one CPU, which makes their
float32 results bitwise reproducible on one host; comparisons against floats
written on another host (the committed archive) are bitwise only with
``THERMO_ARCHIVE_HOST=1``.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from thermo_lab import dtm_baseline_headroom as hr
from thermo_lab import dtm_small_image as dtm

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "docs/research"
PINNED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
ARCHIVE_HOST = os.environ.get("THERMO_ARCHIVE_HOST") == "1"
SEED = hr.PROBE_SEED


def _probe():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import dtm_baseline_headroom_probe as probe
    finally:
        sys.path.remove(str(PROBE_DIR))
    return probe


def small_request(**extra):
    base = dict(
        L=4,
        family_table=(
            ("anchor_p0006", 0.55, 0.2, 0.2),
            ("t4_h05_q035", 0.35, 0.5, 0.35),
            ("parity_J03_h06_q035", 0.30, 0.6, 0.35),
        ),
        rungs=("logistic_7x7", "cnn2_w8_k5_gpool"),
        n_pool=200,
        n_val=24,
        n_test=24,
        ladder={"rates": [0.03, 0.1], "seeds": [0, 1], "steps": 150},
        dtm={
            "seeds": [0],
            "n_train": 100,
            "steps_per_layer": 4,
            "K_train": 4,
            "operating_point": {"bits": 4, "cap_rule": "split", "jitter": 0.1, "K": 2},
            "reference_sweeps": 4,
            "draws": 4,
            "streams": 4,
            "images_per_stream": 6,
            "max_candidates": 1,
        },
        bootstrap={"resamples": 50},
        preflight={
            "steps_per_layer": 4,
            "n_train": 100,
            "images": 2,
            "chains": 256,
            "sweeps": 4,
        },
        seeds={
            "pool": [SEED, 1],
            "validation": [SEED, 2],
            "test": [SEED, 3],
            "ladder_batches": [SEED, 4],
            "ladder_init": SEED + 5,
            "dtm_training": [SEED, 6],
            "thrml_root": SEED + 7,
            "jitter_words": [SEED, 8],
            "stream_states": [SEED, 9],
            "bootstrap": [SEED, 10],
        },
    )
    return hr.make_request(**(base | extra))


def _cli(arguments, env=None, check=True):
    result = subprocess.run(
        [sys.executable, "-m", "thermo_lab.dtm_baseline_headroom", *arguments],
        env=env or {**os.environ, **PINNED_ENV},
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stdout[-3000:] + result.stderr[-4000:])
    return result


def test_single_thread_blas_pin(tmp_path):
    unpinned = {**os.environ, **PINNED_ENV, "OPENBLAS_NUM_THREADS": "4"}
    failed = _cli(["--output-dir", str(tmp_path / "x"), "--replay"], env=unpinned, check=False)
    assert failed.returncode != 0
    assert "OPENBLAS_NUM_THREADS=1" in failed.stderr
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "run pytest under the BLAS pin"
    assert jax.default_backend() == "cpu"


def test_production_request_matches_protocol():
    r = hr.make_request()
    names = [f["name"] for f in r["families"]]
    assert names == [
        "anchor_p0006",
        "t4_h05_q035",
        "smooth8_h05_q035",
        "smooth16_h05_q035",
        "parity_J03_h06_q035",
        "cornerkey_J03_h06_q035",
    ]
    assert [f["components"] for f in r["families"]] == [4, 4, 8, 16, 4, 4]
    assert [(f["J"], f["h0"], f["q"]) for f in r["families"]] == [
        (0.55, 0.2, 0.2),
        (0.35, 0.5, 0.35),
        (0.35, 0.5, 0.35),
        (0.35, 0.5, 0.35),
        (0.3, 0.6, 0.35),
        (0.3, 0.6, 0.35),
    ]
    lad = r["ladder"]
    assert len(lad["rungs"]) == 9
    assert (lad["rates"], lad["seeds"], lad["steps"], lad["batch"]) == (
        [0.003, 0.01, 0.03],
        [0, 1, 2],
        6000,
        128,
    )
    fits = len(r["families"]) * len(lad["rungs"]) * (len(lad["rates"]) + len(lad["seeds"]) - 1)
    assert fits == 270
    macs = {g["name"]: g["macs_per_image"] for g in lad["rungs"]}
    assert macs["logistic_7x7"] == 3136
    assert macs["cnn2_w8_k5"] == 128000
    assert macs["cnn2_w8_k5_gpool"] == 243200
    assert macs["cnn2_w32_k5"] == 1740800
    d = r["dtm"]
    assert (d["T"], d["n_train"], d["lr"], d["batch"], d["K_train"], d["steps_per_layer"]) == (
        2,
        4000,
        0.03,
        64,
        64,
        1600,
    )
    assert d["operating_point"] == {"bits": 4, "cap_rule": "split", "jitter": 0.1, "K": 64}
    assert (d["reference_sweeps"], d["draws"], d["streams"], d["images_per_stream"]) == (
        1024,
        16,
        50,
        20,
    )
    assert (r["n_val"], r["n_test"], r["n_pool"]) == (1000, 1000, 20000)
    assert len(r["constraint_requests"]) == 5 * 3 * 2 * 2
    op = hr.arm_request(r, 4, 2, 1, "op")
    assert op.relaxed == () and op.bits == 4 and op.cap_rule == "split"
    assert op.jitter == 0.1 and op.jitter_words == (20261208, 4, 2, 1)
    ref = hr.arm_request(r, 4, 2, 1, "float_ref")
    assert ref.relaxed == ("beta_jitter", "quantization") and ref.bits is None
    assert r["seeds"]["thrml_root"] == 20261207 and r["seeds"]["ladder_init"] == 20261205
    assert r["cpu_stop_hours"] == 10.0


def test_families_equal_probe_bitwise():
    probe = _probe()
    specs = {s["name"]: s for s in probe.family_specs(8)}
    r = hr.make_request()
    for fam in r["families"]:
        theirs = specs[fam["name"]]
        assert (fam["J"], fam["h0"], fam["q"]) == (theirs["J"], theirs["h0"], theirs["q"])
        assert np.array_equal(np.asarray(fam["fields"]), theirs["fields"])
    fam = r["families"][3]
    mine = hr.FieldMixture(8, fam["J"], np.asarray(fam["fields"]))
    theirs = probe.FieldMixture(8, fam["J"], specs[fam["name"]]["fields"])
    assert np.array_equal(mine.logz, theirs.logz)
    a = mine.sample(40, np.random.default_rng(SEED))
    assert np.array_equal(a, theirs.sample(40, np.random.default_rng(SEED)))
    y = dtm.flip(a, 0.35, np.random.default_rng([SEED, 1]))
    assert np.array_equal(mine.posterior(y, 0.35), theirs.posterior(y, 0.35))


@pytest.mark.parametrize("rung", ["logistic_7x7", "cnn1_w4_k5", "cnn2_w8_k5_gpool"])
def test_ladder_equals_probe_bitwise(rung):
    probe = _probe()
    L = 8
    widths, ks, gp = next((w, k, g) for n, w, k, g in hr.rung_table(L) if n == rung)
    spec = {s["name"]: s for s in probe.family_specs(L)}["parity_J03_h06_q035"]
    mix = probe.FieldMixture(L, spec["J"], spec["fields"])
    pool = mix.sample(300, np.random.default_rng(SEED))
    seed = 3
    theirs = probe.train_model(L, widths, ks, 0.35, pool, 12, 32, 0.01, seed, gp)
    mine = hr.train_model(
        L,
        widths,
        ks,
        0.35,
        pool,
        12,
        32,
        0.01,
        jax.random.PRNGKey(seed),
        np.random.default_rng([probe.PROBE_SEED, seed, L]),
        gp,
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(mine), jax.tree_util.tree_leaves(theirs), strict=True
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    x = mix.sample(50, np.random.default_rng([SEED, 1]))
    y = dtm.flip(x, 0.35, np.random.default_rng([SEED, 2]))
    assert np.array_equal(hr.errors(mine, L, x, y, gp), probe.errors(theirs, L, x, y, gp))
    assert hr.counts(L, widths, ks, gp) == probe.counts(L, widths, ks, gp)
    assert hr.counts(L, widths, ks, gp) == hr.hand_counts(L, widths, ks, gp)


def test_thrml_stream_equals_dtm_thrml_xs_bitwise():
    dreq = dtm.make_request(
        bits=(4,),
        n_test=6,
        draws=2,
        streams=2,
        images_per_stream=3,
        training_seeds=[0],
        steps=[2],
        sweeps=[2],
        reference_sweeps=4,
        seeds={"jax_root": SEED, "jitter_words": [SEED, 5], "stream_states": [SEED, 7]},
    )
    rng = np.random.default_rng(SEED)
    layers = [
        dtm.layer_from(8, 0.7, rng.normal(0, 0.3, 876 - 64), rng.normal(0, 0.1, 256))
        for _ in range(2)
    ]
    y = rng.integers(0, 2, (6, 64)) * 2.0 - 1.0
    for arm, code in (("b4_j0.1", 3), ("float", 0)):
        ref, _ = dtm.thrml_xs(dreq, 0, 2, code, layers, arm, 2, y)
        lowered = [
            hr._Lowered(lay, dtm.arm_request(dreq, 0, 2, li, arm)) for li, lay in enumerate(layers)
        ]
        mine, _ = hr.thrml_stream_xs(
            lowered,
            y,
            2,
            2,
            2,
            3,
            np.random.default_rng([SEED, 7, 0, 2, code]),
            SEED,
            lambda j, li, code=code: (0, 2, code, j, li),
        )
        assert np.array_equal(mine, ref)


def test_pricing_counts_and_transfer_checks():
    r = hr.make_request()
    p = hr.pricing_check(r)
    assert p["passed"]
    assert (p["reads"], p["writes"], p["updates"], p["x_rounded"]) == (128, 128, 24576, 320826)
    assert abs(p["per_estimate_j"][1] - 3.21e-7) < 0.01e-7
    assert abs(p["per_estimate_j"][0] - 9.7e-9) < 0.1e-9
    assert hr.counts_check(r)["passed"]
    t = hr.transfer_check(r)
    assert t["passed"] and len(t["rows"]) == 12


def _strip_cpu(study):
    ev = study["evaluation"]
    ev.pop("cpu_s_per_estimate_ladder")
    for fam in ev["stage2"].values():
        fam.pop("cpu_s_per_estimate", None)
    return ev


def test_small_run_replays_resumes_and_detects_tampering(tmp_path):
    request = small_request()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    full = tmp_path / "full"
    _cli(["--output-dir", str(full), "--request", str(path), "--workers", "2"])
    completion = json.loads((full / "completion.json").read_text())
    assert completion["status"] == "dtm_baseline_headroom_complete"
    assert completion["replayed"] and completion["prefix_check_passed"]
    assert completion["preflight_passed"] and completion["references_recomputed"]
    assert completion["families"] == 3 and completion["ladder_fits"] == 3 * 2 * 3
    assert completion["anchor_control_passed"], "small-run anchor must exercise stage 2"
    assert completion["candidates"] == 1
    assert completion["dtm_cells"] == 4 * completion["candidates"]
    # interrupted then resumed run gives an identical record
    part = tmp_path / "part"
    stopped = _cli(
        ["--output-dir", str(part), "--request", str(path), "--workers", "1", "--stop-after", "12"],
        check=False,
    )
    assert stopped.returncode == 130
    assert not (part / "completion.json").exists()
    # a changed request cannot reuse the saved units
    changed = small_request(n_val=30)
    path2 = tmp_path / "changed.json"
    path2.write_text(json.dumps(changed))
    refused = _cli(["--output-dir", str(part), "--request", str(path2), "--resume"], check=False)
    assert refused.returncode != 0
    assert "stored request differs" in refused.stderr
    _cli(["--output-dir", str(part), "--request", str(path), "--workers", "2", "--resume"])
    log = (part / "run.log").read_text()
    assert "attempt 2 (resume)" in log
    a = dtm.read_gz_json(full / "study.json.gz")
    b = dtm.read_gz_json(part / "study.json.gz")
    assert a["result_digest"] == b["result_digest"]
    assert _strip_cpu(a) == _strip_cpu(b)
    # light replay of the finished run
    light = _cli(["--output-dir", str(full), "--replay", "--light"])
    assert json.loads(light.stdout)["replayed"]
    # a tampered archive fails replay
    bad = tmp_path / "bad"
    shutil.copytree(full, bad)
    study = dtm.read_gz_json(bad / "study.json.gz")
    name = next(iter(study["ladder_fits"]))
    study["ladder_fits"][name]["test_wrong"][0] += 1
    dtm.write_gz_json(bad / "study.json.gz", study)
    failed = _cli(["--output-dir", str(bad), "--replay", "--light"], check=False)
    assert failed.returncode != 0
    assert "digest" in failed.stderr


REPORT = sorted((ROOT / "docs/experiment-reports").glob("*-dtm-baseline-headroom"))


@pytest.mark.skipif(not REPORT, reason="no committed archive yet")
def test_committed_archive_light_replay(tmp_path):
    import hashlib
    import tarfile

    report = REPORT[-1]
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    out = next(p.parent for p in tmp_path.rglob("study.json.gz"))
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    completion = hr.replay(out, light=True)
    committed = json.loads((report / "completion.json").read_text())
    for key in ("result_digest", "candidate_list", "family_verdicts", "seed_verdicts"):
        assert completion[key] == committed[key]
    assert ARCHIVE_HOST or completion["references_recomputed"]
