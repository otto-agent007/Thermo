"""Independent observable identities and paired-estimator archive integrity."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest


def test_conditional_observables_match_direct_enumeration():
    from thermo_lab import conditional_estimation as study

    n = 4
    states = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(bool)
    spins = 2 * states.astype(float) - 1
    h = np.array([0.12, -0.08, 0.04, -0.01])
    matrix = np.array([[0, 0.13, 0, 0], [0.13, 0, -0.07, 0], [0, -0.07, 0, 0.09], [0, 0, 0.09, 0]])
    score = 4 * (spins @ h + np.einsum("ki,ij,kj->k", spins, matrix, spins) / 2)
    probability = np.exp(score - score.max())
    probability /= probability.sum()
    q = study.conditional_probabilities(states[None], h[None], matrix[None])[0]
    for row, state in enumerate(states):
        for site in range(n):
            matches = np.all(np.delete(states, site, axis=1) == np.delete(state, site), axis=1)
            direct = probability[matches & states[:, site]].sum() / probability[matches].sum()
            assert q[row, site] == pytest.approx(direct, abs=2e-15)
    edges = [(0, 1), (1, 2), (2, 3)]
    # One state per batch member exposes each conditional observable, then
    # weight those values by an independently enumerated target law.
    conditional = study.estimate(
        states[:, None],
        np.tile(h, (len(states), 1)),
        np.tile(matrix, (len(states), 1, 1)),
        edges,
        "conditional",
        alarm_threshold=2,
    )
    np.testing.assert_allclose(
        probability @ conditional["marginal"], probability @ states, atol=1e-14
    )
    expected_edges = np.column_stack([spins[:, i] * spins[:, j] for i, j in edges])
    np.testing.assert_allclose(
        probability @ conditional["edge"], probability @ expected_edges, atol=1e-14
    )
    assert probability @ conditional["alarm"] == pytest.approx(probability @ (states.sum(1) >= 2))
    assert np.all(
        probability @ (q * q) - (probability @ q) ** 2
        <= (probability @ states) * (1 - probability @ states) + 1e-14
    )


def test_uncoupled_conditionals_are_exact_from_constant_bad_states():
    from thermo_lab import conditional_estimation as study

    kept = np.zeros((2, 3, 12), bool)
    h = np.full((2, 12), np.log(3) / 8)
    matrices = np.zeros((2, 12, 12))
    empirical = study.estimate(kept, h, matrices, [(0, 1)], "empirical")
    conditional = study.estimate(kept, h, matrices, [(0, 1)], "conditional")
    np.testing.assert_array_equal(empirical["marginal"], 0)
    np.testing.assert_allclose(conditional["marginal"], 0.75, atol=1e-15)
    # Marginal correction does not make the joint event estimate exact.
    np.testing.assert_array_equal(conditional["alarm"], 0)


def test_tempering_estimator_ignores_hot_replicas_and_burnin():
    from thermo_lab import conditional_estimation as study

    states = np.zeros((2, 5, 5, 12), bool)
    h = np.zeros((2, 12))
    matrices = np.ones((2, 12, 12)) * 0.03
    for matrix in matrices:
        np.fill_diagonal(matrix, 0)
    original = study.estimate_trace(states, "tempering", 4, h, matrices, [(0, 1)], "conditional")
    states[:, :, :-1] = True
    states[:, :2] = True
    changed = study.estimate_trace(states, "tempering", 4, h, matrices, [(0, 1)], "conditional")
    for name in original:
        np.testing.assert_array_equal(changed[name], original[name])


def test_small_paired_study_replays_and_rejects_changed_metrics(tmp_path):
    from thermo_lab import conditional_estimation as study

    request = study.make_request()
    request.update(
        seeds=[94, 95], conditions=[request["conditions"][0]], budgets=[4], timing_repeats=1
    )
    out = tmp_path / "paired"
    study.run_study(out, request)
    complete = study.replay(out)
    assert complete["trajectory_cells_replayed"] == 2
    assert complete["estimator_cells_replayed"] == 4
    assert complete["query_estimates_replayed"] == 200
    result = json.loads((out / "results.json").read_text())
    result["cells"][0]["arms"]["conditional"]["metrics"]["marginal_mae"][0][0] += 0.1
    (out / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="marginal_mae"):
        study.replay(out)


def test_archived_conditional_estimation_replays(tmp_path):
    report = Path(__file__).resolve().parents[2] / (
        "docs/experiment-reports/2026-10-02-conditional-estimation"
    )
    manifest = json.loads((report / "manifest.json").read_text())
    path = report / manifest["archive"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(path) as archive:
        archive.extractall(tmp_path, filter="data")
    out = tmp_path / "conditional-estimation"
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    subprocess.run(
        [
            sys.executable,
            "-m",
            "thermo_lab.conditional_estimation",
            "--output-dir",
            str(out),
            "--replay",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "false",
        },
    )
    complete = json.loads((out / "completion.json").read_text())
    assert complete["trajectory_cells_replayed"] == 96
    assert complete["estimator_cells_replayed"] == 192
    assert complete["query_estimates_replayed"] == 76800
