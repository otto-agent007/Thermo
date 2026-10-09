"""Checks for the planar Ising exact reference, two-colour sampler and study contract."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from thermo_lab import planar_ising_scaling as study

REPORT = Path(__file__).resolve().parents[2] / "docs/experiment-reports"


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("variant", study.VARIANTS)
def test_kac_ward_matches_brute_force(size, variant):
    rng = np.random.default_rng(size * 10 + len(variant))
    m = len(study.grid_edges(size))
    weights = rng.integers(1, 6, m) / 5
    signs = rng.choice([-1.0, 1.0], m)
    interactions = study.COLD_BETA * (-weights if variant == "ferro" else signs * weights)
    log_z, edge = study.kac_ward(size, interactions)
    exact_log_z, exact_edge = study.brute_force(size, interactions)
    assert log_z == pytest.approx(exact_log_z, abs=1e-10)
    np.testing.assert_allclose(edge, exact_edge, atol=1e-10, rtol=0)


def test_kac_ward_matches_transfer_matrix_at_study_sizes():
    for target in study.grid_targets():
        if target["size"] > 16:
            continue
        exact = study.reference(target)
        assert "transfer_matrix_log_z" in exact
        assert exact["transfer_matrix_log_z"] == pytest.approx(exact["log_z"], rel=1e-10)
        assert np.all(np.abs(exact["edge"]) <= 1)
        assert exact["finite_difference_max_abs_error"] < 1e-5


def test_targets_cover_three_sizes_two_variants_three_seeds():
    targets = study.grid_targets()
    assert len(targets) == 18
    assert {t["size"] for t in targets} == {8, 16, 32}
    for target in targets:
        horizontal, vertical = study.coupling_arrays(target)
        assert horizontal.shape == (target["size"], target["size"] - 1)
        assert vertical.shape == (target["size"] - 1, target["size"])
        if target["variant"] == "ferro":
            assert np.all(np.asarray(target["couplings"]) < 0)
        else:
            signs = np.sign(target["couplings"])
            assert (signs > 0).any() and (signs < 0).any()
    assert study.LADDERS[5] == tuple(study.base.LADDER)
    assert study.LADDERS[9][0] == pytest.approx(0.25) and study.LADDERS[9][-1] == 4.0


def test_two_colour_kernel_is_exactly_stationary():
    checks = study.fixture_checks()
    assert checks["gibbs_stationarity_residual"] <= 1e-12


def test_local_fields_match_dense_definition():
    target = study.grid_targets()[0]
    size = target["size"]
    horizontal, vertical = study.coupling_arrays(target)
    state = np.random.default_rng(3).integers(0, 2, (2, size, size)).astype(bool)
    spins = 2 * state.astype(float) - 1
    edges = study.grid_edges(size)
    matrix = np.zeros((size * size, size * size))
    for (i, j), coupling in zip(edges, target["couplings"], strict=True):
        matrix[i, j] = matrix[j, i] = coupling
    expected = (spins.reshape(2, -1) @ matrix).reshape(2, size, size)
    actual = np.asarray(
        study.local_fields(jnp.asarray(state), jnp.asarray(horizontal), jnp.asarray(vertical))
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    products = np.asarray(study.edge_products(jnp.asarray(state), size))
    flat = spins.reshape(2, -1)
    np.testing.assert_array_equal(
        products, flat[:, [e[0] for e in edges]] * flat[:, [e[1] for e in edges]]
    )


def test_sampler_prefix_sums_and_exchange_schedule():
    target = study.grid_targets()[0]
    size = target["size"]
    horizontal, vertical = (jnp.asarray(x) for x in study.coupling_arrays(target))
    initial, keys = study.key_inputs(0, size, 9, 3, 11)
    plan = study.snapshot_plan("tempering", [16, 64])
    assert plan == [4, 16, 64]
    fn = study.compile_sampler("tempering", 5, 4, size, 64, plan)
    snaps, accepted = map(np.asarray, fn(keys, initial, horizontal, vertical))
    assert snaps.shape == (3, 3, len(study.grid_edges(size)))
    assert accepted.shape == (3, 64, 2)
    due = (np.arange(64) + 1) % 4 == 0
    assert not accepted[:, ~due].any()
    # Prefix sums are monotone in window length and bounded by it.
    assert np.all(np.abs(snaps[:, 0]) <= 4) and np.all(np.abs(snaps[:, 2]) <= 64)
    sums = study.window_sums(snaps, plan, "tempering", 64)
    assert np.all(np.abs(sums) <= 48)
    estimate = study.window_estimate(sums, "tempering", 1, 64)
    assert np.all(np.abs(estimate) <= 1)
    nine = study.compile_sampler(
        "tempering", 9, 1, size, 16, study.snapshot_plan("tempering", [16])
    )
    _, accepted9 = map(np.asarray, nine(keys, initial, horizontal, vertical))
    assert accepted9.shape == (3, 16, 4)
    long = study.compile_sampler("long", 1, 0, size, 16, study.snapshot_plan("long", [16]))
    snaps_long, accepted_long = map(np.asarray, long(keys, initial, horizontal, vertical))
    assert study.snapshot_plan("long", [16]) == [20, 80]
    assert not accepted_long.any()
    assert np.all(np.abs(study.window_sums(snaps_long, [20, 80], "long", 16)) <= 60)


def test_pricing_uses_replica_count_and_pair_tries():
    target = study.grid_targets()[0]
    n = target["n"]
    priced = study.price(target, "tempering", 9, 4, 64, np.array([3, 0]))
    assert priced["physical_pbits_used"] == 9 * n
    assert priced["pair_tries_per_trial"] == 16 * 4
    assert priced["host_round_trips_per_trial"] == 16
    assert priced["elapsed_complete_sweeps"] == 64
    assert (
        priced["energy"]["beta_knob"]["mean_total_energy_j"]
        < (priced["energy"]["published"]["mean_total_energy_j"])
    )
    long = study.price(target, "long", 1, 0, 64, np.zeros(2, int))
    assert long["elapsed_complete_sweeps"] == 320 and long["physical_pbits_used"] == n
    assert long["energy"]["published"]["mean_total_energy_j"] == pytest.approx(320 * n * 7.09e-15)


def test_small_run_replays_and_rejects_changed_sums(tmp_path):
    request = study.make_request()
    request["targets"] = [t for t in study.grid_targets() if t["id"] == "L8-mixed-s400"]
    request["budgets"] = [16, 64]
    request["trials"] = 3
    output = tmp_path / "study"
    study.run_study(output, request)
    assert not (output / "completion.json").exists()
    complete = study.replay(output)
    assert complete["status"] == "planar_ising_scaling_complete"
    assert complete["cells_replayed"] == 12
    assert complete["decisions_replayed"] == 6
    with pytest.raises(FileExistsError):
        study.run_study(output, request)
    result = json.loads((output / "results.json").read_text())
    bad = copy.deepcopy(result)
    next(c for c in bad["cells"] if c["arm"] == "tempering9-k4")["means"]["edge_mae"] = 0.0
    (output / "results.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="edge_mae"):
        study.replay(output)


def test_archived_planar_ising_evidence_replays(tmp_path):
    report = REPORT / "2026-10-07-planar-ising-scaling"
    if not report.exists():
        pytest.skip("study not yet archived")
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    output = tmp_path / "planar-ising-scaling"
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    subprocess.run(
        [
            sys.executable,
            "-m",
            "thermo_lab.planar_ising_portable_replay",
            "--output-dir",
            str(output),
        ],
        check=True,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "false",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
    )
    complete = json.loads((output / "portable-completion.json").read_text())
    assert complete["status"] == "planar_ising_scaling_portable_replay_complete"
    assert complete["targets"] == 18
    assert complete["cells_replayed"] == 540
    assert complete["decisions_replayed"] == 108
    assert complete["decisions_unchanged_by_recomputed_references"] is True


def test_jax_platform_is_cpu():
    assert jax.default_backend() == "cpu"
