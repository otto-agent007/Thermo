"""Scientific checks for the fixed-budget sampler and its persisted evidence."""

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


def test_gibbs_and_exchange_preserve_exact_target():
    from thermo_lab.fixed_budget_sampling import exact_checks

    checks = exact_checks()
    assert checks["gibbs_stationarity_residual"] < 1e-12
    assert checks["exchange_balance_residual"] < 1e-12
    assert checks["reversed_sign_residual"] > 0.001


def test_swap_acceptance_sign():
    from thermo_lab.fixed_budget_sampling import swap_log_acceptance

    # Moving the higher-score cold state to the hot slot is unfavorable.
    assert float(swap_log_acceptance(0.5, 4.0, -2.0, 1.0)) == -10.5
    assert float(swap_log_acceptance(0.5, 4.0, 1.0, -2.0)) == 0.0


def test_empirical_cold_distributions_match_enumeration():
    from thermo_lab.fixed_budget_sampling import validate_fixture

    checks, _ = validate_fixture()
    assert set(checks["empirical_tv"]) == {"long", "independent", "tempering"}
    assert max(checks["empirical_tv"].values()) < 0.04


def test_saved_evidence_replays_and_tampering_fails(tmp_path):
    from thermo_lab.fixed_budget_sampling import make_request, replay, run_study

    requested = make_request()
    requested["targets"] = requested["targets"][:1]
    requested["trials"] = 4
    requested["budgets_replica_sweeps"] = [4, 16]
    out = tmp_path / "run"
    run_study(out, requested)
    assert not (out / "completion.json").exists()
    completion = replay(out)
    assert completion["cells_replayed"] == 6
    result = json.loads((out / "results.json").read_text())
    for row in result["cells"]:
        assert row["spin_updates_per_trial"] == 5 * row["budget"] * 12
        assert row["swap_attempts_per_trial"] == (
            2 * row["budget"] if row["method"] == "tempering" else 0
        )
    with pytest.raises(FileExistsError):
        run_study(out, requested)
    path = out / "traces.npz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="trace hash"):
        replay(out)


def test_zero_coupling_sampler_redraws_unbiased_spins():
    from thermo_lab.fixed_budget_sampling import compile_sampler

    initial = jnp.zeros((16, 5, 3), dtype=jnp.bool_)
    keys = jax.random.split(jax.random.key(19), 16)
    for method in ("long", "independent", "tempering"):
        fn = compile_sampler(method, 3, 512)
        states, accepted = fn(keys, initial, jnp.zeros(3), jnp.zeros((3, 3)))
        values = np.asarray(states)[:, 1:].mean(axis=(0, 1, 2))
        np.testing.assert_allclose(values, 0.5, atol=0.025)
        if method == "tempering":
            assert np.asarray(accepted).all()


def test_archived_fresh_seed_evidence_replays(tmp_path):
    report = Path(__file__).resolve().parents[2] / (
        "docs/experiment-reports/2026-10-01-fixed-budget-sampling"
    )
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    output = tmp_path / "fixed-budget-sampling"
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    # Replay is bitwise: isolate the declared BLAS/JAX settings before import.
    # Changing BLAS reduction order changes exact-reference last bits (~1e-14).
    subprocess.run(
        [
            sys.executable,
            "-m",
            "thermo_lab.fixed_budget_sampling",
            "--output-dir",
            str(output),
            "--replay",
        ],
        env={
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "false",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    completion = json.loads((output / "completion.json").read_text())
    assert completion["request_digest"] == (
        "sha256:7fe6b11624be109e5382f2892fee95c525e57e1f5b7b15437bef4200b89b383d"
    )
    assert completion["cells_replayed"] == 108
