"""Checks for the exact references, the scheduled sampler and the planar annealing contract."""

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

from thermo_lab import planar_annealing as study
from thermo_lab import planar_ising_scaling as scaling

REPORT = Path(__file__).resolve().parents[2] / "docs/experiment-reports"


@pytest.mark.parametrize("size", [2, 3, 4])
def test_transfer_matrices_match_brute_force(size):
    rng = np.random.default_rng(size)
    m = len(scaling.grid_edges(size))
    couplings = rng.choice([-1.0, 1.0], m) * rng.integers(1, 6, m) / 5
    assert study.ground_state_energy(size, couplings) == pytest.approx(
        study.brute_force_ground_state(size, couplings), abs=1e-12
    )
    for beta in (4.0, 8.0, 16.0):
        log_z, edge = scaling.brute_force(size, beta * couplings)
        assert study.transfer_log_z(size, beta * couplings) == pytest.approx(log_z, abs=1e-9)
        assert study.thermal_energy(size, couplings, beta) == pytest.approx(
            float(couplings @ edge) / (size * size), abs=1e-12
        )


def test_reference_checks_pass_and_thermal_energy_is_below_the_ground_state():
    checks = study.reference_checks()
    assert checks["max_ground_state_error"] < 1e-12
    assert checks["max_thermal_energy_error"] < 1e-9
    target = next(t for t in study.grid_targets() if t["id"] == "L8-mixed-s400")
    exact = study.reference(target)
    gs = exact["ground_state_per_spin"]
    assert exact["thermal_per_spin"]["8.0"] < exact["thermal_per_spin"]["16.0"] < gs
    assert gs - exact["thermal_per_spin"]["16.0"] < 1e-3


def test_targets_and_schedule_shape():
    targets = study.grid_targets()
    assert [t["size"] for t in targets] == [8, 8, 8, 16, 16, 16, 24, 24, 24]
    for target in targets:
        signs = np.sign(target["couplings"])
        assert (signs > 0).any() and (signs < 0).any()
    ramp = study.schedule("anneal", 16.0, 1024)
    assert ramp.shape == (1024,)
    assert ramp[0] == pytest.approx(0.5) and ramp[767] == pytest.approx(16.0, rel=1e-5)
    assert np.all(ramp[768:] == np.float32(16.0))
    assert np.all(np.diff(ramp[:768]) > 0)
    cold = study.schedule("cold", 16.0, 64)
    assert np.all(cold == np.float32(16.0))


def test_scheduled_sampler_hold_sums_are_exact_and_bounded():
    target = next(t for t in study.grid_targets() if t["id"] == "L8-mixed-s400")
    horizontal, vertical = (jnp.asarray(x) for x in scaling.coupling_arrays(target))
    initial, keys = scaling.key_inputs(0, 8, 9, 3, 5)
    fn = study.compile_scheduled_sampler(8, 4, study.schedule("anneal", 16.0, 64))
    sums = np.asarray(fn(keys, initial, horizontal, vertical))
    assert sums.shape == (3, 4, 112) and sums.dtype == np.int32
    assert np.all(np.abs(sums) <= 16)  # hold phase is 16 sweeps of +-1 products
    energy = study.hold_energy(sums, target, 64)
    assert energy.shape == (3, 4)
    exact = study.reference(target, thermal=False)
    assert np.all(energy <= exact["ground_state_per_spin"] + 1e-12)


def test_cell_reports_best_chain_and_prices_only_updates_for_annealing():
    target = next(t for t in study.grid_targets() if t["id"] == "L8-mixed-s400")
    exact = {"ground_state_per_spin": 0.8}
    energies = np.array([[0.70, 0.79], [0.60, 0.65]])
    cell = study.make_cell(target, "restart16x4", 256, energies, exact)
    assert cell["per_trial"]["energy_per_spin"] == [0.79, 0.65]
    assert cell["means"]["gap"] == pytest.approx((0.01 + 0.15) / 2)
    assert cell["fraction_within"]["0.001"] == 0.0
    assert cell["z1"]["physical_pbits_used"] == 4 * 64
    assert cell["z1"]["mean_total_energy_j"]["published"] == pytest.approx(4 * 64 * 256 * 7.09e-15)
    with pytest.raises(ValueError, match="above the exact ground state"):
        study.make_cell(target, "anneal8", 256, np.array([[0.81]]), exact)


def test_small_run_resumes_replays_and_rejects_changed_energies(tmp_path):
    request = study.make_request()
    request["targets"] = [t for t in study.grid_targets() if t["id"] == "L8-mixed-s400"]
    request["budgets"] = [64, 256]
    request["trials"] = 3
    output = tmp_path / "study"
    study.run_study(output, request)
    assert not (output / "completion.json").exists()
    complete = study.replay(output)
    assert complete["status"] == "planar_annealing_complete"
    assert complete["cells_replayed"] == 12
    assert complete["decisions_replayed"] == 6
    assert complete["references_recomputed"] == 1
    with pytest.raises(FileExistsError):
        study.run_study(output, request)
    # Resume with the same request reuses the saved unit without resampling.
    resumed = study.run_study(output, request, resume=True)
    assert resumed["cells"] == json.loads((output / "results.json").read_text())["cells"]
    result = json.loads((output / "results.json").read_text())
    bad = copy.deepcopy(result)
    next(c for c in bad["cells"] if c["arm"] == "anneal16")["means"]["gap"] = 0.0
    (output / "results.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="gap"):
        study.replay(output)


def test_archived_planar_annealing_evidence_replays(tmp_path):
    report = REPORT / "2026-10-08-planar-annealing"
    if not report.exists():
        pytest.skip("study not yet archived")
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    output = tmp_path / "planar-annealing"
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    subprocess.run(
        [
            sys.executable,
            "-m",
            "thermo_lab.planar_annealing",
            "--output-dir",
            str(output),
            "--replay",
            "--light",
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
    complete = json.loads((output / "completion.json").read_text())
    assert complete["status"] == "planar_annealing_complete"
    assert complete["targets"] == 9
    assert complete["cells_replayed"] == 270
    assert complete["decisions_replayed"] == 54
    assert complete["references_recomputed"] == 6
    assert complete["references_verified_by_digest"] == 3


def test_jax_platform_is_cpu():
    assert jax.default_backend() == "cpu"
