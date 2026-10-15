"""Checks for the DTM small-image study: probe equality, THRML bitwise, replay, resume.

Every runner call passes the single-thread BLAS pin. Unit tests use probe seeds
and a small request only; the production seeds (2026110x) appear nowhere here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from thermo_lab import dtm_small_image as dtm
from thermo_lab.constraints import ConstraintRequest
from thermo_lab.constraints.thrml_lowering import final_states

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "docs/research"
PINNED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
# Bitwise comparisons of floats written on another host need the archive host.
ARCHIVE_HOST = os.environ.get("THERMO_ARCHIVE_HOST") == "1"
SEED = dtm.PROBE_SEED


def _probe():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import dtm_small_image_probe as p0
        import dtm_small_image_probe_r1 as p1
    finally:
        sys.path.remove(str(PROBE_DIR))
    return p0, p1


def small_request(**extra):
    base = dict(
        bits=(3,),
        n_train=200,
        n_test=24,
        draws=4,
        streams=4,
        images_per_stream=6,
        training_seeds=[0],
        steps=[1, 2],
        training={"steps_per_layer": 6, "K_train": 4},
        logistic={"steps": 10},
        sweeps=[2, 4],
        reference_sweeps=8,
        bootstrap={"resamples": 50},
        preflight={
            "steps_per_layer": 4,
            "n_train": 100,
            "images": 2,
            "chains": 256,
            "sweeps": 4,
        },
        seeds={
            "training_data": [SEED, 1],
            "test": SEED + 2,
            "training_chains": [SEED, 3],
            "jax_root": SEED + 4,
            "jitter_words": [SEED, 5],
            "logistic": [SEED, 6],
            "stream_states": [SEED, 7],
            "bootstrap": [SEED, 8],
        },
    )
    return dtm.make_request(**(base | extra))


def _cli(arguments, env=None, check=True):
    result = subprocess.run(
        [sys.executable, "-m", "thermo_lab.dtm_small_image", *arguments],
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
    r = dtm.make_request()
    assert r["family"] == {"L": 8, "J": 0.55, "h0": 0.2, "q": 0.2, "components": 4}
    assert (r["n_train"], r["n_test"], r["draws"], r["streams"]) == (4000, 1000, 16, 50)
    assert r["arms"] == ["float"] + [f"b{b}_j{j}" for b in (3, 4, 6, 8) for j in (0.0, 0.1)]
    cells = dtm.cells(r)
    thrml = [c for c in cells if c["kind"] == "thrml"]
    assert len(thrml) == 168
    assert len(cells) - len(thrml) == 36
    assert len(r["constraint_requests"]) == 3 * 3 * 9
    assert r["seeds"]["jax_root"] == 20261104
    float_req = dtm.arm_request(r, 0, 1, 0, "float")
    assert float_req.relaxed == ("beta_jitter", "quantization")
    jit = dtm.arm_request(r, 2, 2, 1, "b6_j0.1")
    assert jit.relaxed == () and jit.jitter_words == (20261105, 2, 2, 1) and jit.bits == 6
    assert jit.cap_rule == "split" and jit.update_rule == "active_kernel"
    assert jit.dtype == "float64"


def test_data_model_layout_and_baseline_equal_probe_bitwise():
    p0, _ = _probe()
    mine, theirs = dtm.Mixture(8, 0.55, 0.2), p0.Mixture(8, 0.55, 0.2)
    assert np.array_equal(mine.logz, theirs.logz)
    a = mine.sample(30, np.random.default_rng(SEED))
    b = theirs.sample(30, np.random.default_rng(SEED))
    assert np.array_equal(a, b)
    y = dtm.flip(a, 0.2, np.random.default_rng([SEED, 1]))
    assert np.array_equal(mine.posterior(y, 0.2), theirs.posterior(y, 0.2))
    assert np.array_equal(mine.loglik(a), theirs.loglik(a))
    assert dtm.layout(8) == p0.layout(8)
    w1 = dtm.train_logistic(8, 0.2, a, np.random.default_rng(SEED), steps=5, r=3)
    w2 = p0.train_logistic(8, 0.2, a, np.random.default_rng(SEED), steps=5, r=3)
    assert np.array_equal(w1, w2)


def test_trainer_equals_probe_train_dtm_bitwise():
    p0, _ = _probe()
    data = dtm.Mixture(8, 0.55, 0.2).sample(64, np.random.default_rng(SEED))
    mine = dtm.train_layer(8, 1, 1, 0.2, data, 3, 16, 4, np.random.default_rng(SEED))
    (theirs,), _, _ = p0.train_dtm(8, 1, 0.2, data, 3, 16, 4, np.random.default_rng(SEED))
    assert np.array_equal(mine.w, theirs.w) and np.array_equal(mine.b, theirs.b)
    assert mine.gamma == theirs.gamma


@pytest.mark.parametrize("start", ["continue", "random"])
def test_numpy_streams_equal_probe_r1_bitwise(start):
    p0, p1 = _probe()
    rng = np.random.default_rng(SEED)
    lay = p0.Layer(8, 0.7, rng)
    lay.w = rng.normal(0, 0.3, len(lay.w))
    mine_lay = dtm.layer_from(8, lay.gamma, lay.w, lay.b)
    y = dtm.flip(np.ones((p1.N_TEST, 64)), 0.3, np.random.default_rng([SEED, 2]))
    a = dtm.numpy_streams(
        [mine_lay],
        y,
        1,
        np.random.default_rng([SEED, 3]),
        start,
        p1.DRAWS,
        p1.STREAMS,
        p1.IMAGES_PER_STREAM,
    )
    b = p1.denoise_streams([lay], y, 1, np.random.default_rng([SEED, 3]), start)
    assert np.array_equal(a, b)


def test_thrml_runner_equals_final_states_bitwise_and_carries_state():
    rng = np.random.default_rng(SEED)
    lay = dtm.layer_from(8, 0.7, rng.normal(0, 0.3, 876 - 64), rng.normal(0, 0.1, 256))
    creq = ConstraintRequest(
        model_id="test",
        update_rule="active_kernel",
        dtype="float64",
        relaxed=("beta_jitter", "quantization"),
        reasons={"beta_jitter": "test", "quantization": "test"},
    )
    low = dtm.LoweredLayer(lay, creq)
    chains = 64
    with jax.enable_x64(True):
        keys = jax.random.split(jax.random.key(SEED), chains)
        x = jnp.asarray(rng.integers(0, 2, (chains, 64)) > 0)
        shared = [rng.integers(0, 2, n) > 0 for n in low.sizes]
        fn = dtm.make_thrml_runner(low.program, low.blocks, 3)
        mine = np.asarray(fn(keys, x, *[jnp.asarray(np.tile(s, (chains, 1))) for s in shared]))
        ref = final_states(low.program, low.blocks, 3, keys, x, shared)
        assert np.array_equal(mine, ref)
        # per-chain initial states: chain k equals final_states run from its own state
        per = [rng.integers(0, 2, (chains, n)) > 0 for n in low.sizes]
        many = np.asarray(fn(keys, x, *[jnp.asarray(p) for p in per]))
        for k in (0, 17):
            one = final_states(
                low.program, low.blocks, 3, keys[k : k + 1], x[k : k + 1], [p[k] for p in per]
            )
            assert np.array_equal(many[k : k + 1], one)


def test_pricing_reproduces_protocol_expectations():
    p = dtm.price_sample(8, 1, 256)
    assert (p["reads"], p["writes"], p["free_node_writes"], p["clamp_state_changes"]) == (
        64,
        64,
        0,
        0,
    )
    assert p["updates"] == 3 * 64 * 256
    assert abs(p["energy_j"] - 1.03e-8) < 0.01e-8
    assert abs(p["energy_j_clamp_loads_at_read_constant_outside_profile"] - 5.7e-10) < 0.1e-10
    assert abs(dtm.price_sample(8, 2, 64)["energy_j"] - 2.01e-8) < 0.01e-8
    r = dtm.price_sample(8, 1, 64, "random")
    assert r["free_node_writes"] == 192
    assert abs(r["energy_j"] - 3.95e-8) < 0.01e-8
    assert dtm.pricing_check(dtm.make_request())["passed"]


def test_transfer_matrix_equals_brute_force():
    assert dtm.transfer_check(1e-12)["passed"]


def test_small_run_replays_and_resume_is_identical(tmp_path):
    request = small_request()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    full = tmp_path / "full"
    _cli(["--output-dir", str(full), "--request", str(path), "--workers", "1"])
    completion = json.loads((full / "completion.json").read_text())
    assert completion["status"] == "dtm_small_image_complete"
    assert completion["replayed"] and completion["prefix_check_passed"]
    assert completion["thrml_cells"] == 2 * (3 * 2 + 1)
    assert completion["numpy_cells"] == 2 * 2 * 2
    assert completion["preflight_passed"]
    # interrupted then resumed run gives an identical record
    part = tmp_path / "part"
    stopped = _cli(
        ["--output-dir", str(part), "--request", str(path), "--workers", "1", "--stop-after", "9"],
        check=False,
    )
    assert stopped.returncode == 130
    assert not (part / "completion.json").exists()
    _cli(["--output-dir", str(part), "--request", str(path), "--workers", "1", "--resume"])
    a = dtm.read_gz_json(full / "study.json.gz")
    b = dtm.read_gz_json(part / "study.json.gz")
    assert a["result_digest"] == b["result_digest"]
    for record in (a, b):
        record["evaluation"]["operating_point"].pop("cpu_s_per_estimate")
    assert a["evaluation"] == b["evaluation"]
    # light replay of the finished run
    light = _cli(["--output-dir", str(full), "--replay", "--light"])
    assert json.loads(light.stdout)["replayed"]


def test_tampered_archive_fails_replay(tmp_path):
    request = small_request(n_test=8, streams=2, images_per_stream=4, steps=[1])
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    out = tmp_path / "run"
    _cli(["--output-dir", str(out), "--request", str(path), "--workers", "1"])
    study = dtm.read_gz_json(out / "study.json.gz")
    name = next(iter(study["cells"]))
    study["cells"][name]["maj_wrong"][0] += 1
    dtm.write_gz_json(out / "study.json.gz", study)
    failed = _cli(["--output-dir", str(out), "--replay", "--light"], check=False)
    assert failed.returncode != 0
    assert "digest" in failed.stderr


REPORT = sorted((ROOT / "docs/experiment-reports").glob("*-dtm-small-image"))


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
    completion = dtm.replay(out, light=True)
    committed = json.loads((report / "completion.json").read_text())
    for key in ("result_digest", "k_star", "b_star", "jitter_verdict", "baseline_verdict"):
        assert completion[key] == committed[key]
    assert ARCHIVE_HOST or completion["references_recomputed"]
