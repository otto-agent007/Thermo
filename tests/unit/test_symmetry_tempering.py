"""Independent checks of the combined estimator and fresh evidence contract."""

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

from thermo_lab import sampling_time_to_accuracy as prior


def test_tempering_flip_uses_only_cold_samples_and_preserves_work():
    from thermo_lab import symmetry_tempering as study

    target = prior.graph_targets()[0]
    states = np.random.default_rng(91).integers(0, 2, (4, 17, 5, 12)).astype(bool)
    accepted = np.random.default_rng(92).integers(0, 2, (4, 16, 2)).astype(bool)
    actual = prior.estimates(target, "tempering-flip", states, 16)
    cold = states[:, :, -1:]
    explicit = prior.estimates(target, "independent", np.concatenate([cold, ~cold], axis=2), 16)
    changed_hot = states.copy()
    changed_hot[:, :, :-1] = ~changed_hot[:, :, :-1]
    for name in actual:
        np.testing.assert_allclose(actual[name], explicit[name], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(
            actual[name], prior.estimates(target, "tempering-flip", changed_hot, 16)[name]
        )
    exact = prior.reference(target)
    timings = [{"pipeline_seconds": 0.2, "sample_transfer_seconds": 0.1}]
    raw = study.make_cell(target, "tempering", states, accepted, 16, exact, timings, 0.05)
    flip = study.make_cell(target, "tempering-flip", states, accepted, 16, exact, timings, 0.05)
    assert flip["method"] == "tempering-flip"
    assert flip["augmentation"] == "analytic global flip"
    assert flip["spin_updates_per_trial"] == raw["spin_updates_per_trial"] == 960
    assert flip["swap_attempts_per_trial"] == raw["swap_attempts_per_trial"] == 32
    assert flip["retained_cold_states_per_trial"] == raw["retained_cold_states_per_trial"] == 12
    assert flip["exchange_acceptance_by_pair"] == raw["exchange_acceptance_by_pair"]
    assert flip["per_trial"]["edge_mae"] == raw["per_trial"]["edge_mae"]
    np.testing.assert_allclose(flip["per_trial"]["marginal_mae"], 0, atol=1e-14)
    with pytest.raises(ValueError, match="zero fields"):
        study.make_cell(
            prior.graph_targets()[1],
            "tempering-flip",
            states,
            accepted,
            16,
            exact,
            timings,
            0.05,
        )


def test_protocol_uses_fresh_zero_field_cubic_graphs_and_fixed_controls():
    from thermo_lab import symmetry_tempering as study

    request = study.make_request()
    assert {t["seed"] for t in request["targets"]} == {300, 301, 302}
    assert len(request["targets"]) == 6
    assert request["root_seed"] == 20261005
    assert request["budgets"] == [16, 64, 256, 1024, 4096]
    assert request["trials"] == 16
    assert request["threshold"] == 0.05
    for target in request["targets"]:
        assert not np.any(target["fields"])
        assert len(target["edges"]) == 3 * target["n"] // 2
        np.testing.assert_array_equal(np.bincount(np.array(target["edges"]).ravel()), 3)
        assert study.methods(target) == [
            "long-flip",
            "independent-flip",
            "tempering",
            "tempering-flip",
        ]


def test_small_run_replays_both_tempering_arms_and_rejects_changed_work(
    tmp_path, missing_cgroup_limits
):
    from thermo_lab import symmetry_tempering as study
    from thermo_lab.sampling_portability import run_study

    request = study.make_request()
    request["targets"] = [prior.graph_targets()[0]]
    request["budgets"] = [4, 16]
    request["trials"] = 4
    request["timing_repeats"] = 2
    output = tmp_path / "study"
    run_study("symmetry_tempering", output, request)
    assert not (output / "completion.json").exists()
    complete = study.replay(output)
    assert complete["status"] == "exploratory_symmetry_tempering_complete"
    assert complete["cells_replayed"] == 8
    assert complete["decisions_replayed"] == 4
    with pytest.raises(FileExistsError):
        run_study("symmetry_tempering", output, request)
    result = json.loads((output / "results.json").read_text())
    bad = copy.deepcopy(result)
    next(c for c in bad["cells"] if c["method"] == "tempering-flip")["swap_attempts_per_trial"] = 0
    (output / "results.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="swap_attempts_per_trial"):
        study.replay(output)


def test_archived_symmetry_tempering_evidence_replays(tmp_path):
    report = Path(__file__).resolve().parents[2] / (
        "docs/experiment-reports/2026-10-02-symmetry-tempering"
    )
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    output = tmp_path / "symmetry-tempering"
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    subprocess.run(
        [
            sys.executable,
            "-m",
            "thermo_lab.symmetry_tempering",
            "--output-dir",
            str(output),
            "--replay",
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
    assert complete["status"] == "exploratory_symmetry_tempering_complete"
    assert complete["cells_replayed"] == 120
    assert complete["decisions_replayed"] == 24
