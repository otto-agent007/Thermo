"""Checks for the exchange-interval sampler and the Z1 pricing of exchanges."""

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

from thermo_lab import exchange_cost_projection as study
from thermo_lab import fixed_budget_sampling as base
from thermo_lab import sampling_time_to_accuracy as prior
from thermo_lab.hardware.z1 import Z1HardwareProfile

REPORT = Path(__file__).resolve().parents[2] / "docs/experiment-reports"


def _inputs(target, trials=4, index=0, root=7):
    fields, matrix = (jnp.asarray(x) for x in base.model(target))
    initial, keys = base.key_inputs(index, target["n"], trials, root)
    return initial, keys, fields, matrix


@pytest.mark.parametrize("method", base.METHODS)
def test_interval_one_reproduces_the_archived_sampler_bit_for_bit(method):
    target = prior.graph_targets()[0]
    initial, keys, fields, matrix = _inputs(target)
    interval = 1 if method == "tempering" else 0
    mine = study.compile_interval_sampler(method, target["n"], 32, interval)
    theirs = base.compile_sampler(method, target["n"], 32)
    a_states, a_flags = map(np.asarray, mine(keys, initial, fields, matrix))
    b_states, b_flags = map(np.asarray, theirs(keys, initial, fields, matrix))
    np.testing.assert_array_equal(a_states, b_states)
    np.testing.assert_array_equal(a_flags, b_flags)


def test_exchange_interval_gates_attempts_and_keeps_the_sampling_stream():
    target = prior.graph_targets()[0]
    initial, keys, fields, matrix = _inputs(target)
    every = study.compile_interval_sampler("tempering", target["n"], 64, 1)
    sparse = study.compile_interval_sampler("tempering", target["n"], 64, 16)
    rare = study.compile_interval_sampler("tempering", target["n"], 64, 64)
    s1, f1 = map(np.asarray, every(keys, initial, fields, matrix))
    s16, f16 = map(np.asarray, sparse(keys, initial, fields, matrix))
    s64, f64 = map(np.asarray, rare(keys, initial, fields, matrix))
    due = (np.arange(64) + 1) % 16 == 0
    assert not f16[:, ~due].any()
    assert f16[:, due].sum() > 0
    assert f64[:, :63].sum() == 0 and f64.shape == (4, 64, 2)
    # Until the first exchange is due the sampling stream does not depend on the
    # interval: k=16 and k=64 agree on the initial state and the first 15 sweeps,
    # and diverge from k=1, which has already exchanged.
    np.testing.assert_array_equal(s16[:, :16], s64[:, :16])
    assert not np.array_equal(s1[:, :16], s16[:, :16])
    assert f1.shape == f16.shape == (4, 64, 2)
    with pytest.raises(ValueError, match="interval"):
        study.compile_interval_sampler("tempering", target["n"], 64, 3)
    with pytest.raises(ValueError, match="exchange interval"):
        study.compile_interval_sampler("long", target["n"], 64, 4)


def test_pricing_follows_the_sealed_profile_constants():
    profile = Z1HardwareProfile()
    target = prior.graph_targets()[0]
    n = target["n"]
    accepted = np.zeros((3, 64, 2), dtype=bool)
    accepted[0, 15, :] = True  # one trial accepts both pairs at the first exchange
    priced = study.price_cell(target, "tempering", 16, 64, accepted)
    assert priced["exchange_attempts_per_trial"] == 8
    assert priced["host_round_trips_per_trial"] == 4
    assert priced["elapsed_complete_sweeps"] == 64
    assert priced["gibbs_node_updates"] == 5 * n * 64
    published = priced["energy"]["published"]
    knob = priced["energy"]["beta_knob"]
    reads = 8 * 2 * n * profile.node_read_energy_j
    updates = 5 * n * 64 * profile.gibbs_node_update_energy_j
    writes_trial0 = 2 * 2 * n * profile.node_full_sram_write_energy_j
    assert published["mean_read_energy_j"] == pytest.approx(reads)
    assert knob["mean_read_energy_j"] == pytest.approx(reads)
    assert published["mean_sampling_energy_j"] == pytest.approx(updates)
    assert published["mean_write_energy_j"] == pytest.approx(writes_trial0 / 3)
    assert knob["mean_write_energy_j"] == 0
    assert published["max_total_energy_j"] == pytest.approx(updates + reads + writes_trial0)
    assert published["min_total_energy_j"] == pytest.approx(updates + reads)
    ordinary = study.price_cell(target, "independent", 0, 64, np.zeros((3, 64, 2), bool))
    assert ordinary["energy"]["published"]["mean_total_energy_j"] == pytest.approx(updates)
    assert ordinary["host_round_trips_per_trial"] == 0
    long = study.price_cell(target, "long", 0, 64, np.zeros((3, 320, 2), bool))
    assert long["elapsed_complete_sweeps"] == 320
    assert long["physical_pbits_used"] == n
    with pytest.raises(ValueError, match="more accepted"):
        study.price_cell(target, "tempering", 64, 64, np.ones((3, 64, 2), bool))


def test_stage_a_prices_the_archive_and_reproduces_its_decisions():
    manifest, request, results, traces = study.load_archive()
    result = study.stage_a(manifest, request, results, traces)
    assert len(result["cells"]) == 120
    assert len(result["decisions"]) == 24
    assert result["archive_sha256"] == manifest["sha256"]
    archived = {(d["target"], d["method"]): d for d in results["decisions"]}
    for d in result["decisions"]:
        method = d["arm"].replace("-k1", "")
        assert d["status"] == archived[d["target"], method]["status"]
    # A qualifying tempering-flip arm is three orders of magnitude more expensive
    # than its cheapest ordinary baseline under the published convention.
    ratios = [
        c["published"]["tempering"]["tempering-flip-k1"].get("energy_ratio_vs_cheapest_ordinary")
        for c in result["comparisons"]
    ]
    ratios = [r for r in ratios if r is not None]
    assert ratios and min(ratios) > 1000
    # ... and still faster in elapsed sweeps on every target it qualifies on.
    time_ratios = [
        c["published"]["tempering"]["tempering-flip-k1"].get("sweep_time_ratio_vs_fastest_ordinary")
        for c in result["comparisons"]
    ]
    assert all(r <= 1 for r in time_ratios if r is not None)


def test_small_run_replays_and_rejects_changed_pricing(tmp_path):
    request = study.make_request()
    request["targets"] = [prior.graph_targets()[0]]
    request["budgets"] = [256]
    request["trials"] = 4
    output = tmp_path / "study"
    study.run_study(output, request)
    assert not (output / "completion.json").exists()
    complete = study.replay(output)
    assert complete["status"] == "exchange_cost_projection_complete"
    assert complete["cells_replayed"] == 7
    assert complete["decisions_replayed"] == 7
    assert complete["archived_cells_priced"] == 120
    with pytest.raises(FileExistsError):
        study.run_study(output, request)
    result = json.loads((output / "results.json").read_text())
    bad = copy.deepcopy(result)
    cell = next(c for c in bad["cells"] if c["arm"] == "tempering-flip-k4")
    cell["z1"]["energy"]["published"]["mean_write_energy_j"] = 0.0
    (output / "results.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="mean_write_energy_j"):
        study.replay(output)


def test_archived_exchange_cost_evidence_replays(tmp_path):
    report = REPORT / "2026-10-07-exchange-cost-projection"
    if not report.exists():
        pytest.skip("study not yet archived")
    manifest = json.loads((report / "manifest.json").read_text())
    archive_path = report / manifest["archive"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive_path) as archive:
        archive.extractall(tmp_path, filter="data")
    output = tmp_path / "exchange-cost-projection"
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    subprocess.run(
        [
            sys.executable,
            "-m",
            "thermo_lab.exchange_cost_projection",
            "--output-dir",
            str(output),
            "--replay",
        ],
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "false"},
        capture_output=True,
        text=True,
    )
    complete = json.loads((output / "completion.json").read_text())
    assert complete["status"] == "exchange_cost_projection_complete"
    assert complete["archived_cells_priced"] == 120
    assert complete["archived_decisions_priced"] == 24
    assert complete["cells_replayed"] == 210
    assert complete["decisions_replayed"] == 42


def test_jax_platform_is_cpu():
    assert jax.default_backend() == "cpu"
