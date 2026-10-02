"""Causal streaming, exact targets and intervention contracts."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from thermo_lab import fixed_budget_sampling as base


def test_batched_sampler_matches_archived_sampler_bitwise():
    from thermo_lab import changing_evidence as study

    initial, keys = base.key_inputs(31, 3, 2)
    h, matrix = base.fixture_model()
    for algorithm in ("gibbs", "tempering"):
        fn = study.make_sampler(algorithm, 3, 8)
        states, accepted, final = fn(
            keys, initial, jnp.tile(h, (2, 1)), jnp.tile(matrix, (2, 1, 1))
        )
        expected, flags = base.compile_sampler(
            "independent" if algorithm == "gibbs" else algorithm, 3, 8
        )(keys, initial, h, matrix)
        np.testing.assert_array_equal(states, expected)
        np.testing.assert_array_equal(accepted, flags)
        np.testing.assert_array_equal(final, expected[:, -1])


def test_references_match_enumeration_and_round_trip_targets():
    from thermo_lab import changing_evidence as study

    inputs = study.make_inputs(study.conditions()[3], [90, 91])
    reference, probability, _ = study.references(inputs, repeats=1)
    np.testing.assert_array_equal(inputs["fields"][:12], inputs["fields"][24:12:-1])
    np.testing.assert_allclose(reference["marginal"][:12], reference["marginal"][24:12:-1])
    for t in (0, 12, 24):
        states, p, _ = base.enumerate_target(inputs["fields"][t, 0], inputs["matrices"][0])
        np.testing.assert_allclose(probability[t, 0], p, atol=1e-14, rtol=0)
        np.testing.assert_allclose(reference["marginal"][t, 0], p @ states, atol=1e-14)
    np.testing.assert_allclose(reference["input_rms"][0], 0)


def test_features_are_causal_and_decision_regret_uses_exact_risk():
    from thermo_lab import changing_evidence as study

    previous = np.array([[0.9, 0.1]])
    x = study.features(np.array([[-0.2, 0.2]]), np.array([[0.2, -0.2]]), previous, [0.5], 2)
    np.testing.assert_allclose(x, [[0.4, 0.32, 0.36, 0.5, 1 / 3]])
    np.testing.assert_allclose(study.regret([0.1, 0.9, 0.2], [0.8, 0.1, 0.2]), [3, 0.5, 0])
    states = np.zeros((2, 9, 5, 12), bool)
    states[:, :, -1] = True
    estimate = study.estimates(states, "tempering", 8, [(0, 1)])
    np.testing.assert_array_equal(estimate["marginal"], 1)
    np.testing.assert_array_equal(estimate["alarm"], 1)


def test_predictor_handles_separable_and_constant_labels():
    from thermo_lab import changing_evidence as study

    x = np.tile(np.arange(-10, 11)[:, None], (1, 5)).astype(float)
    y = x[:, 0] > 0
    model = study.fit_predictor(x, y, [0, 1, 2, 3, 4])
    assert study.prediction_metrics(y, study.predict(model, x))["auc"] == 1
    for value in (0, 1):
        constant = study.fit_predictor(x, np.full(len(x), value), [0])
        np.testing.assert_array_equal(study.predict(constant, x), value)


def test_small_study_replays_and_rejects_modified_derived_error(tmp_path):
    from thermo_lab import changing_evidence as study

    request = study.make_request()
    request.update(
        development_seeds=[90, 91],
        heldout_seeds=[92, 93],
        conditions=[study.conditions()[0]],
        budgets=[4],
        policy_budget=4,
        timing_repeats=1,
    )
    out = tmp_path / "study"
    study.run_study(out, request)
    complete = study.replay(out)
    assert complete["cells_replayed"] == 10
    assert complete["query_estimates_replayed"] == 500
    data = json.loads((out / "results.json").read_text())
    for algorithm in ("gibbs", "tempering"):
        first = [
            c
            for c in data["cells"]
            if c["split"] == "heldout"
            and c["method"] in (f"retain-{algorithm}", f"restart-{algorithm}")
        ]
        np.testing.assert_array_equal(first[0]["marginal"][0], first[1]["marginal"][0])
    data["cells"][0]["metrics"]["marginal_mae"][0][0] += 0.1
    (out / "results.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="marginal_mae"):
        study.replay(out)


def test_archived_changing_evidence_replays(tmp_path):
    report = Path(__file__).resolve().parents[2] / (
        "docs/experiment-reports/2026-10-02-changing-evidence"
    )
    manifest = json.loads((report / "manifest.json").read_text())
    path = report / manifest["archive"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(path) as archive:
        archive.extractall(tmp_path, filter="data")
    out = tmp_path / "changing-evidence"
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    subprocess.run(
        [
            sys.executable,
            "-m",
            "thermo_lab.changing_evidence",
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
    result = json.loads((out / "completion.json").read_text())
    assert result["cells_replayed"] == 416
    assert result["query_estimates_replayed"] == 83200
