"""Accuracy, symmetry, censoring and evidence checks for the timing follow-up."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import fixed_budget_sampling as base


def test_estimates_agree_with_existing_evaluator_and_explicit_flip():
    from thermo_lab.sampling_time_to_accuracy import accuracy, estimates, reference

    target = base.make_request()["targets"][0]
    states = np.random.default_rng(4).integers(0, 2, (4, 17, 5, 12)).astype(bool)
    accepted = np.zeros((4, 16, 2), bool)
    exact = reference(target)
    raw = estimates(target, "independent", states, 16)
    checked = accuracy(raw, exact)
    old = base.score(target, "independent", states, accepted, 16)
    assert checked["per_trial"] == old["per_trial"]
    symmetric = estimates(target, "independent-flip", states, 16)
    explicit = estimates(target, "independent", np.concatenate([states, ~states], axis=2), 16)
    for name in symmetric:
        np.testing.assert_allclose(symmetric[name], explicit[name], atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="zero fields"):
        estimates(base.make_request()["targets"][1], "independent-flip", states, 16)


def test_denoising_is_the_untempered_posterior_after_parameter_scaling():
    from thermo_lab.sampling_time_to_accuracy import denoising_targets

    targets = denoising_targets()
    assert len(targets) == 6
    for target in targets:
        fields, matrix = base.model(target)
        states, p, _ = base.enumerate_target(fields, matrix)
        spins = 2 * states.astype(float) - 1
        observed = np.asarray(target["observed"])
        h = ((2 * observed - 1) * 0.5 * np.log((1 - target["noise"]) / target["noise"])).astype(
            np.float32
        )
        j = float(np.float32(0.4))
        edges = np.asarray(target["edges"])
        logp = spins @ h.astype(float) + j * (spins[:, edges[:, 0]] * spins[:, edges[:, 1]]).sum(
            axis=1
        )
        expected = np.exp(logp - logp.max())
        expected /= expected.sum()
        np.testing.assert_allclose(p, expected, atol=1e-14, rtol=0)


def test_threshold_requires_both_metrics_and_a_sustained_crossing():
    from thermo_lab.sampling_time_to_accuracy import qualification

    cells = [
        {
            "budget": b,
            "means": {"joint_tv": tv, "edge_mae": edge},
            "pipeline_median_seconds": b / 1000,
        }
        for b, tv, edge in [
            (16, 0.04, 0.03),
            (64, 0.02, 0.07),
            (256, 0.04, 0.03),
            (1024, 0.02, 0.02),
        ]
    ]
    assert qualification(cells, 0.05) == {
        "status": "reached",
        "budget": 256,
        "warm_batch_seconds": 0.256,
    }
    cells[-1]["means"]["edge_mae"] = 0.06
    assert qualification(cells, 0.05)["status"] == "not_reached"


def test_replay_tolerates_roundoff_but_rejects_changed_values():
    from thermo_lab.sampling_time_to_accuracy import check_equal

    check_equal({"value": [0.5]}, {"value": [0.5 + 1e-14]})
    with pytest.raises(ValueError):
        check_equal({"value": [0.5]}, {"value": [0.51]})
    with pytest.raises(ValueError):
        check_equal({"count": 2}, {"count": 3})


def test_small_run_replays_and_detects_modified_timing_summary(tmp_path, missing_cgroup_limits):
    from thermo_lab.sampling_portability import run_study
    from thermo_lab.sampling_time_to_accuracy import make_request, replay

    request = make_request()
    request["targets"] = request["targets"][:1]
    request["budgets"] = [4, 16]
    request["trials"] = 4
    request["timing_repeats"] = 2
    output = tmp_path / "study"
    run_study("sampling_time_to_accuracy", output, request)
    assert not (output / "completion.json").exists()
    assert replay(output)["cells_replayed"] == 10
    with pytest.raises(FileExistsError):
        run_study("sampling_time_to_accuracy", output, request)
    result = json.loads((output / "results.json").read_text())
    bad = copy.deepcopy(result)
    bad["cells"][0]["pipeline_median_seconds"] += 0.1
    (output / "results.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        replay(output)


def test_archived_timing_and_transfer_evidence_replays(tmp_path):
    report = Path(__file__).resolve().parents[2] / (
        "docs/experiment-reports/2026-10-02-sampling-time-to-accuracy"
    )
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    output = tmp_path / "sampling-time-to-accuracy"
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    subprocess.run(
        [
            sys.executable,
            "-m",
            "thermo_lab.sampling_time_to_accuracy",
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
        "sha256:59923617c0c84db837af98d7b9cc3bdee87ff6fc25982606b596b09562c485cf"
    )
    assert completion["cells_replayed"] == 330
    assert completion["decisions_replayed"] == 66
