"""Checks for the Z1 acceptance kit study (P-0015).

The kernel is checked bitwise against the round-2 probe sampler it derives from
(same host, same code path), and its calibration arithmetic against the archived
round-3 probe values. The small-run test runs the CLI under the single-thread BLAS
pin, interrupts it, resumes it, compares the resumed record with an uninterrupted
run, replays the bounded archive and checks that a tampered unit fails replay.
Comparisons against an archive written on another host are bitwise only with
``THERMO_ARCHIVE_HOST=1``; otherwise floats compare at the runner's replay
tolerance (2e-12 relative). Unit tests use small requests only.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import planar_16_offset_ferro as p16
from thermo_lab import z1_acceptance_kit as z

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "docs/research"
PINNED_ENV = {"JAX_PLATFORMS": "cpu", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
ARCHIVE = ROOT / "docs/experiment-reports/2026-11-17-z1-acceptance-kit/evidence.tar.gz"


def _probe_r2():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import z1_acceptance_kit_probe_r2 as r2
    finally:
        sys.path.remove(str(PROBE_DIR))
    return r2


def _pin(monkeypatch):
    for key, value in PINNED_ENV.items():
        monkeypatch.setenv(key, value)


def _probe_order_edge_sums(target, beta, chains, burn, window, seed):
    """The runner's kernel driven by one generator in the round-2 probe's draw order."""
    array = z.make_array(
        target["n"],
        p16.target_positions(target),
        p16.edge_tuples(target),
        target["couplings"],
        beta=beta,
    )
    rng = np.random.default_rng(seed)
    state = rng.choice([-1.0, 1.0], size=(chains, target["n"]))
    acc = np.zeros((chains, len(target["edges"])))
    for it in range(burn + window):
        for idx in (array["a"], array["b"]):
            z.half_sweep(state, array, idx, rng.random((chains, idx.size)))
        if it >= burn:
            acc += state[:, array["ei"]] * state[:, array["ej"]]
    return acc


def test_kernel_bitwise_equals_probe_sampler(monkeypatch):
    _pin(monkeypatch)
    r2 = _probe_r2()
    target = p16.build_target(8, "greedy-long", 4511)
    for beta in (0.5, 0.75):
        mine = _probe_order_edge_sums(target, beta, 12, 7, 40, 99) / 40
        probe = r2.gibbs_edge_means(target, beta, 12, 7, 40, 99)
        assert np.array_equal(mine, probe)


def test_calibration_reproduces_round3_probe(monkeypatch):
    """Round-3 probe part R: beta 0.5, 128 chains, 4,096-read window, t(127) n_scan."""
    _pin(monkeypatch)
    r2 = _probe_r2()
    target = p16.build_target(8, "greedy-long", 4510)
    sums = _probe_order_edge_sums(target, 0.5, 128, 512, 4096, r2.SEED_CAL + 50)
    _, c0 = p16.kac_ward(
        p16.target_positions(target), p16.edge_tuples(target), 0.5 * np.asarray(target["couplings"])
    )
    c1, _ = z.missing_edge_correlations(
        p16.target_positions(target), p16.edge_tuples(target), 0.5 * np.asarray(target["couplings"])
    )
    summary = z.calibration_summary(target, 0.5, 4096, sums, c0, c1)["summary"]
    assert summary["n_scan"] == 5402
    assert summary["scan_independent_reads"] == 2679


def test_per_replicate_streams_are_independent_of_batch(monkeypatch):
    _pin(monkeypatch)
    target = p16.build_target(8, "greedy-long", 4512)
    array = z.make_array(
        target["n"],
        p16.target_positions(target),
        p16.edge_tuples(target),
        target["couplings"],
        beta=0.6,
    )
    rngs = [np.random.default_rng([5, r]) for r in range(4)]
    batch, _ = z.simulate(array, rngs, 3, 37)
    alone, _ = z.simulate(array, [np.random.default_rng([5, 2])], 3, 37)
    assert np.array_equal(batch[2], alone[0])


def test_missing_edge_correlations_against_direct_and_central(monkeypatch):
    _pin(monkeypatch)
    target = p16.build_target(16, "greedy-long", 4510)
    pos, edges = p16.target_positions(target), p16.edge_tuples(target)
    k = 0.75 * np.asarray(target["couplings"], dtype=float)
    c1, c0_alt = z.missing_edge_correlations(pos, edges, k)
    _, c0 = p16.kac_ward(pos, edges, k)
    assert np.max(np.abs(c0_alt - c0)) <= 1e-12
    for e in z.declared_edges(len(edges), 5):
        assert abs(z.missing_edge_direct(pos, edges, k, e) - c1[e]) <= 1e-12
        # central difference carries an h^2 truncation term up to about 1.3e-9 here
        assert abs(z.missing_edge_central(pos, edges, k, e) - c1[e]) <= 2e-9


def test_closed_form_counts_match_preregistered_values():
    assert z.reads_needed(float(z.sig(8.0)), 0.5) == 5
    assert z.reads_needed(float(z.sig(8.0)), 0.5, alpha=z.ALPHA / 1984) == 8
    err = float(z.quantize_array(np.array([-0.2]), 1.0, 8)[0] + 0.2)
    assert z.reads_needed(float(z.sig(8 * 0.2)), float(z.sig(8 * (0.2 - err))), per_read=511) == 238
    assert z.two_sided_reads(1.15, 0.1, z.ALPHA) == 3368
    assert z.two_sided_reads(1.0, 0.3, z.ALPHA) == 469
    assert int(np.ceil(1 + np.log2(1 / z.ALPHA))) == 6


def test_enumeration_and_patch_checks_pass():
    assert z.enumeration_checks()["passed"]
    assert z.patch_checks([0.5])["passed"]


def test_production_request_shape():
    request = z.make_request()
    z.validate_request(request)
    assert len(request["targets"]) == 9
    assert len(request["reference_ids"]) == 18
    assert request["candidate_betas"] == [0.5, 0.6, 0.75]
    assert request["sampled_checks"] == 41
    assert (request["max_ideal_rejections"], request["min_defect_rejections"]) == (74, 926)
    assert request["attribution_limit"] == 0.065
    assert request["seeds"]["calibration_root"] == 20261121
    assert request["seeds"]["tested_root"] == 20261118
    assert request["seeds"]["gain_monte_carlo_root"] == 20261122
    assert request["calibration_window"] == 8192 and request["recalibration_window"] == 16384


def test_runner_requires_single_thread_blas(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    with pytest.raises(RuntimeError, match="OPENBLAS_NUM_THREADS"):
        z.run_study(tmp_path / "out", requested=small_request())


def small_request():
    l8 = [t for t in z.greedy_targets() if t["size"] == 8]
    return z.make_request(
        targets=[l8[0]],
        simple_targets=l8,
        simple_array_target=l8[0]["id"],
        reference_ids=["L8-greedy-long-s4510"],
        candidate_betas=[0.5, 0.6],
        calibration_chains=8,
        calibration_window=128,
        recalibration_window=256,
        replicates=24,
        bootstrap_resamples=40,
        gain_mc_replicates=600,
        bisection_steps=3,
        direct_check_edges=3,
    )


def _cli(arguments, check=True):
    env = {**os.environ, **PINNED_ENV}
    result = subprocess.run(
        [sys.executable, "-m", "thermo_lab.z1_acceptance_kit", *arguments],
        env=env,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stdout[-3000:] + result.stderr[-4000:])
    return result


def _record(out):
    result = json.loads((out / "results.json").read_text())
    for key in ("unit_timings_seconds", "generation_seconds_this_attempt"):
        result.pop(key)
    return result


def test_small_run_resume_replay_and_archive(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(small_request()))
    whole, split = tmp_path / "whole", tmp_path / "split"
    _cli(["--output-dir", str(whole), "--request", str(request_path), "--workers", "1"])
    stopped = _cli(
        [
            "--output-dir",
            str(split),
            "--request",
            str(request_path),
            "--workers",
            "1",
            "--stop-after",
            "6",
        ],
        check=False,
    )
    assert stopped.returncode == 130
    assert not (split / "completion.json").exists()
    _cli(["--output-dir", str(split), "--request", str(request_path), "--workers", "1", "--resume"])
    assert "resume attempt 2" in (split / "run.log").read_text()
    assert _record(whole) == _record(split)
    assert (whole / "statistics.npz").read_bytes() == (split / "statistics.npz").read_bytes()
    completion = json.loads((whole / "completion.json").read_text())
    assert completion["status"] == z.STATUS and completion["replayed"] is True
    assert completion["simple_checks"] == 14 and completion["collective_checks"] == 3

    archive = tmp_path / "evidence.tar.gz"
    _cli(["--output-dir", str(whole), "--write-archive", str(archive)])
    light = json.loads(_cli(["--replay-archive", str(archive)]).stdout)
    assert light["archive_replayed"] and light["decisions"] == 17

    unit = sorted((whole / "units").glob("collective__*ideal.npz"))[0]
    unit.write_bytes(unit.read_bytes()[:-8] + b"tampered")
    (whole / "completion.json").unlink()
    failed = _cli(["--output-dir", str(whole), "--replay"], check=False)
    assert failed.returncode != 0 and "digest mismatch" in failed.stderr


@pytest.mark.skipif(not ARCHIVE.exists(), reason="production archive not committed yet")
def test_committed_archive_replays():
    summary = z.replay_archive(ARCHIVE)
    assert summary["archive_replayed"] and summary["decisions"] == 41
