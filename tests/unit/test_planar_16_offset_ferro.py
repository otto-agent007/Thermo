"""Checks for the planar 16-offset ferro study: graphs, exact reference, sampler, autosave."""

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

from thermo_lab import planar_16_offset_ferro as ferro
from thermo_lab import planar_ising_scaling as study

REPORT = Path(__file__).resolve().parents[2] / "docs/experiment-reports"
PINNED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "false",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}


def _cli(arguments, env=None, check=True):
    """Run the study module in a subprocess under the single-thread BLAS pin."""
    result = subprocess.run(
        [sys.executable, "-m", "thermo_lab.planar_16_offset_ferro", *arguments],
        env=env or {**os.environ, **PINNED_ENV},
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stdout[-2000:] + result.stderr[-4000:])
    return result


def test_single_thread_blas_pin_for_replay(tmp_path):
    # The ill-conditioned Kac-Ward inverse differs at 1e-5 under other BLAS thread
    # counts, and replay compares at 2e-12, so the runner refuses to run or
    # replay without the pin; every runner call in this file passes PINNED_ENV.
    unpinned = {**os.environ, **PINNED_ENV, "OPENBLAS_NUM_THREADS": "4"}
    failed = _cli(["--output-dir", str(tmp_path / "x"), "--replay"], env=unpinned, check=False)
    assert failed.returncode != 0
    assert "OPENBLAS_NUM_THREADS=1" in failed.stderr
    assert jax.default_backend() == "cpu"
    assert not jax.config.jax_enable_x64


def test_offsets_are_the_z1_interior_rule_and_odd():
    assert len(ferro.OFFSETS) == 16
    assert {tuple(sorted(map(abs, o))) for o in ferro.OFFSETS} == {(0, 1), (1, 2), (2, 3), (1, 4)}
    assert all((dx + dy) % 2 for dx, dy in ferro.OFFSETS)


def test_greedy_long_reproduces_probe_graph_and_is_planar_bipartite():
    edges, positions = ferro.planar_subgraph(32, 32, "greedy-long", 7)
    stats = ferro.graph_stats(edges, positions)
    # The probe's greedy-long L = 32 graph (graph seed 7).
    assert stats["edges"] == 1982
    assert stats["edges_by_offset_class"] == {"0,1": 1149, "1,2": 252, "1,4": 580, "2,3": 1}
    assert stats["isolated_sites"] == 0
    assert stats["bipartite_checkerboard"]
    assert stats["edges"] <= stats["planar_bipartite_bound_2n_minus_4"]


def _crosses(p, q, a, b):
    def orient(u, v, w):
        return np.sign((v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]))

    return orient(p, q, a) * orient(p, q, b) < 0 and orient(a, b, p) * orient(a, b, q) < 0


def test_greedy_subgraph_has_no_crossing_and_is_maximal_on_a_small_patch():
    edges, positions = ferro.planar_subgraph(8, 8, "greedy-long", 4510)
    segments = [(positions[i], positions[j]) for i, j in edges]
    for x in range(len(segments)):
        for y in range(x + 1, len(segments)):
            assert not _crosses(*segments[x], *segments[y])
    kept = set(edges)
    for r in range(8):
        for c in range(8):
            for dx, dy in ferro.OFFSETS:
                if 0 <= r + dy < 8 and 0 <= c + dx < 8:
                    a, b = r * 8 + c, (r + dy) * 8 + c + dx
                    e = (min(a, b), max(a, b))
                    if e not in kept:
                        assert any(_crosses(positions[e[0]], positions[e[1]], *s) for s in segments)


def test_grid_targets_pair_with_planar_scaling_couplings():
    grid = ferro.build_target(8, "grid", 4510)
    assert [tuple(e) for e in grid["edges"]] == study.grid_edges(8)
    rng = np.random.default_rng(4510 * 1000 + 8)
    expected = -(rng.integers(1, 6, len(grid["edges"])).astype(np.float32) / 5)
    assert grid["couplings"] == expected.tolist()
    targets = ferro.study_targets()
    assert len(targets) == 18
    assert {(t["size"], t["graph"]) for t in targets} == {
        (s, g) for s in ferro.SIZES for g in ferro.GRAPHS
    }
    assert all(np.all(np.asarray(t["couplings"]) < 0) for t in targets)


def test_kac_ward_equals_imported_function_bitwise_on_grids():
    rng = np.random.default_rng(9)
    for size in (3, 5, 8):
        edges = study.grid_edges(size)
        k = study.COLD_BETA * -(rng.integers(1, 6, len(edges)) / 5)
        positions = ferro.positions_for(size, size)
        mine = ferro.kac_ward(positions, edges, k)
        theirs = study.kac_ward(size, k)
        assert mine[0] == theirs[0]
        assert np.array_equal(mine[1], theirs[1])
        assert np.array_equal(
            ferro.kac_ward_matrix(positions, edges, k)[0], study.kac_ward_matrix(size, k)[0]
        )


def test_kac_ward_matches_brute_force_on_16_offset_patches():
    checks = ferro.reference_checks()
    assert checks["passed"]
    assert len(checks["brute_force_patches"]) >= 12
    assert all(p["non_grid_edges"] > 0 for p in checks["brute_force_patches"])
    assert checks["max_log_z_error"] <= 1e-9 and checks["max_edge_error"] <= 1e-9


def test_two_colour_sweep_is_exactly_stationary_on_16_offset_patches():
    for rows, cols, seed in ((2, 5, 1), (3, 3, 2)):
        check = ferro.stationarity_check(rows, cols, "greedy-long", seed)
        assert check["non_grid_edges"] > 0
        assert check["gibbs_stationarity_residual"] <= 1e-12
        assert check["kernel_row_sum_error"] <= 1e-12


def test_neighbour_sampler_equals_planar_scaling_sampler_bitwise():
    equivalence = ferro.grid_equivalence_check(ferro.ROOT_SEED, size=8, horizon=64, trials=3)
    for arm, row in equivalence["arms"].items():
        assert row["snapshot_sums_identical"], arm
        assert row["exchange_flags_identical"], arm


def test_counts_round_trip_to_window_sums():
    target = ferro.build_target(8, "greedy-long", 4510)
    request = _small_request()
    request["targets"] = [target]
    arrays, _ = ferro.run_unit(request, 0, "independent")
    method, replicas, _ = ferro.ARMS["independent"]
    initial, keys = ferro.unit_keys(0, target, method, request["trials"], request["root_seed"])
    plan = study.snapshot_plan(method, request["budgets"])
    fn = ferro.compile_sampler(method, replicas, 0, target, max(request["budgets"]), plan)
    snaps, _ = map(np.asarray, fn(keys, initial))
    for i, b in enumerate(request["budgets"]):
        expected = study.window_estimate(
            study.window_sums(snaps, plan, method, b), method, replicas, b
        )
        actual = ferro.estimates_from_counts(arrays["plus_counts"][i], method, replicas, b)
        assert np.array_equal(actual, expected)
    assert arrays["plus_counts"].dtype == np.uint16


def test_classification_rule():
    assert ferro.classify([0, 0, 1]) == "holds"
    assert ferro.classify([0, 0, 2]) == "mixed"
    assert ferro.classify([1, 1, 0]) == "slips"
    assert ferro.classify([-1, -2, 0]) == "improves"
    assert ferro.classify([1, -1, 0]) == "mixed"
    assert ferro.budget_step({"status": "not_reached"}, [64, 256]) == 2


def _small_request():
    request = ferro.make_request()
    request["targets"] = [ferro.build_target(8, graph, 4510) for graph in ("grid", "greedy-long")]
    request["budgets"] = [16, 64]
    request["trials"] = 3
    request["empirical_check_horizon"] = 4096
    return request


def _science(out):
    result = json.loads((out / "results.json").read_text())
    return {k: result[k] for k in ("cells", "decisions", "summary", "counts_sha256")}


def _run(tmp_path, out, request, *extra, check=True):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    return _cli(["--output-dir", str(out), "--request", str(path), *extra], check=check)


def test_small_run_replays_resumes_bitwise_and_rejects_changes(tmp_path):
    request = _small_request()
    whole = tmp_path / "whole"
    _run(tmp_path, whole, request, "--workers", "2")
    complete = json.loads((whole / "completion.json").read_text())
    assert complete["status"] == "planar_16_offset_ferro_complete"
    assert complete["cells_replayed"] == 2 * 6 * 2
    assert complete["decisions_replayed"] == 12
    assert complete["references_recomputed"] == 2
    assert complete["graph_rebuild_passed"] and complete["kernel_checks_passed"]
    assert complete["evidence_class"]["sweeps"] == "software_simulation"
    assert complete["evidence_class"]["references"] == "exact_reference"
    assert complete["row_verdict"] == "not_applicable_partial_request"
    assert "fresh output directory" in _run(tmp_path, whole, request, check=False).stderr

    # Interrupt after one unit, refuse a changed request, then resume serially:
    # the archive must equal the uninterrupted two-worker run bitwise.
    broken = tmp_path / "broken"
    stopped = _run(tmp_path, broken, request, "--workers", "1", "--stop-after", "1", check=False)
    assert stopped.returncode == 130
    assert not (broken / "completion.json").exists()
    assert len(json.loads((broken / "units/index.json").read_text())) == 1
    changed = json.loads(json.dumps(request))
    changed["trials"] = 4
    refused = _run(tmp_path, broken, changed, "--resume", check=False)
    assert refused.returncode != 0 and "request" in refused.stderr
    _run(tmp_path, broken, request, "--resume", "--workers", "1")
    resumed = json.loads((broken / "completion.json").read_text())
    assert resumed["counts_sha256"] == complete["counts_sha256"]
    assert _science(broken) == _science(whole)
    assert (broken / "counts.npz").read_bytes() == (whole / "counts.npz").read_bytes()
    assert "resume attempt 2" in (broken / "run.log").read_text()

    result = json.loads((whole / "results.json").read_text())
    result["cells"][5]["means"]["edge_mae"] = 0.0
    (whole / "results.json").write_text(json.dumps(result))
    tampered = _cli(["--output-dir", str(whole), "--replay"], check=False)
    assert tampered.returncode != 0 and "edge_mae" in tampered.stderr


def test_resume_reruns_a_unit_whose_digest_fails(tmp_path):
    request = _small_request()
    request["targets"] = request["targets"][:1]
    out = tmp_path / "run"
    stopped = _run(tmp_path, out, request, "--workers", "1", "--stop-after", "2", check=False)
    assert stopped.returncode == 130
    index = json.loads((out / "units/index.json").read_text())
    first = sorted(index)[0]
    (out / "units" / f"{first}.npz").write_bytes(b"corrupt")
    _run(tmp_path, out, request, "--resume", "--workers", "1")
    complete = json.loads((out / "completion.json").read_text())
    assert complete["cells_replayed"] == 12
    assert "failed digest verification" in (out / "run.log").read_text()


def test_archived_planar_16_offset_evidence_replays(tmp_path):
    reports = sorted(REPORT.glob("*-planar-16-offset-ferro"))
    if not reports:
        pytest.skip("study not yet archived")
    report = reports[-1]
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    output = tmp_path / "planar-16-offset-ferro"
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    archived = json.loads((output / "completion.json").read_text())
    _cli(["--output-dir", str(output), "--replay"])
    complete = json.loads((output / "completion.json").read_text())
    assert complete["status"] == "planar_16_offset_ferro_complete"
    assert complete["targets"] == 18
    assert complete["cells_replayed"] == 540
    assert complete["decisions_replayed"] == 108
    assert complete["references_recomputed"] == 18
    assert complete["reference_checks_passed"] and complete["kernel_checks_passed"]
    assert complete["graph_rebuild_passed"]
    assert complete["row_verdict"] == archived["row_verdict"]
    assert complete["counts_sha256"] == archived["counts_sha256"]


def test_jax_platform_is_cpu():
    assert jax.default_backend() == "cpu"
    assert jnp.zeros(1).dtype == jnp.float32
