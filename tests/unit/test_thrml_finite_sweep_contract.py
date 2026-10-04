"""E0 / A1: THRML finite-sweep contract, exact side and archive replay."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import thrml_finite_sweep_contract as study
from thermo_lab.exact import enumerate_ising

REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiment-reports"
    / "2026-10-03-thrml-finite-sweep-contract"
)


@pytest.fixture(scope="module")
def request_small() -> dict:
    return study.study_request(chains=2_000)


@pytest.fixture(scope="module")
def references(request_small: dict) -> dict:
    return study.exact_references(request_small)


def test_cells_cover_every_arm_order_init_and_sweep(request_small: dict) -> None:
    ids = [c["id"] for c in study.cells(request_small)]
    assert len(ids) == 64
    assert len(set(ids)) == 64
    assert "free/forward/all_minus/K0" in ids
    assert "free/reversed/hinton/K30" in ids
    assert "clamped+/forward/hinton/K1" in ids
    assert "clamped-/forward/hinton/K30" in ids


def test_sweep_kernels_are_stochastic_and_fix_the_boltzmann_law(request_small: dict) -> None:
    model = study._model(request_small)
    exact = enumerate_ising(model)
    for order in request_small["orders"]:
        kernel = study.sweep_kernel(model, exact.states, request_small["blocks"], order)
        assert kernel.shape == (32, 32)
        np.testing.assert_allclose(kernel.sum(axis=1), 1.0, atol=1e-12)
        assert study.total_variation(exact.probabilities @ kernel, exact.probabilities) < 1e-12
    # the forward and reversed sweeps are different kernels
    forward = study.sweep_kernel(model, exact.states, request_small["blocks"], "forward")
    reverse = study.sweep_kernel(model, exact.states, request_small["blocks"], "reversed")
    assert np.abs(forward - reverse).max() > 1e-3


def test_block_kernel_rejects_a_dependent_block(request_small: dict) -> None:
    model = study._model(request_small)
    exact = enumerate_ising(model)
    with pytest.raises(ValueError, match="independent set"):
        study.block_kernel(model, exact.states, [0, 1])


def test_single_site_conditional_is_sigmoid_2_beta_h(request_small: dict) -> None:
    """Energy sign, temperature and encoding: P(s_i=+1|rest) = sigmoid(2 beta h_i)."""
    model = study._model(request_small)
    exact = enumerate_ising(model)
    kernel = study.block_kernel(model, exact.states, [0])
    x = 0  # all spins -1; neighbour of site 0 is site 1 with s_1 = -1
    h = model.biases[0] + model.weights[0] * -1.0
    expected = 1.0 / (1.0 + np.exp(-2.0 * model.beta * h))
    y = int(
        np.flatnonzero((exact.states[:, 0] == 1) & np.all(exact.states[:, 1:] == -1, axis=1))[0]
    )
    assert kernel[x, y] == pytest.approx(expected, abs=1e-12)
    assert kernel[x, x] == pytest.approx(1.0 - expected, abs=1e-12)


def test_references_converge_to_stationary_and_controls_separate(references: dict) -> None:
    for order in ("forward", "reversed"):
        assert references["cells"][f"free/{order}/all_minus/K30"]["tv_to_stationary"] < 1e-9
        assert references["cells"][f"free/{order}/all_minus/K1"]["tv_to_stationary"] > 0.1
    for control in references["controls"]:
        if control["cell"].endswith("all_minus/K1"):
            assert control["exact_separation"] > 0.1, control
    assert references["encoding_check"]["exact_separation"] > 0.04


def test_hinton_initial_law_matches_thrml_source_reading(request_small: dict) -> None:
    """THRML 0.1.4 hinton_init: P(S_i=1) = sigmoid(beta * bias_i), independent sites."""
    model = study._model(request_small)
    exact = enumerate_ising(model)
    p0 = study.initial_distribution(request_small, exact.states, "hinton")
    beta, biases = model.beta, np.array(model.biases)
    site = 1.0 / (1.0 + np.exp(-beta * biases))
    marginal = np.array([p0[exact.states[:, i] == 1].sum() for i in range(5)])
    np.testing.assert_allclose(marginal, site, atol=1e-12)
    assert p0.sum() == pytest.approx(1.0)


def test_clamped_reference_lives_on_consistent_states(references: dict) -> None:
    for arm, value in (("clamped+", 1), ("clamped-", -1)):
        for k in (0, 1, 30):
            dist = np.array(references["cells"][f"{arm}/forward/hinton/K{k}"]["distribution"])
            states = enumerate_ising(study._model(study.study_request(10))).states
            assert dist[states[:, 0] != value].sum() == 0.0
            assert dist.sum() == pytest.approx(1.0)


def test_tolerance_shrinks_with_chains(references: dict) -> None:
    p = np.array(references["cells"]["free/forward/uniform/K1"]["distribution"])
    spec = {"quantile": 0.999, "draws": 500, "numpy_seed": 1}
    assert study.tolerance(p, 1_000, spec, 0) > study.tolerance(p, 100_000, spec, 0)


@pytest.mark.slow
def test_one_thrml_cell_matches_exact_reference_at_small_n(request_small: dict) -> None:
    refs = study.exact_references(request_small)
    cell = next(c for c in study.cells(request_small) if c["id"] == "free/forward/all_minus/K1")
    out = study.run_cell(request_small, cell)
    empirical = np.array(out["histogram"]) / request_small["chains_per_cell"]
    ref = refs["cells"][cell["id"]]
    assert study.total_variation(empirical, np.array(ref["distribution"])) <= ref["tolerance"]


def test_archived_report_replays(tmp_path: Path) -> None:
    archive = REPORT / "study.json.gz"
    if not archive.exists():
        pytest.skip("archive not recorded yet")
    record = study.replay_archive(archive, tmp_path / "replay")
    completion = json.loads((tmp_path / "replay" / "completion.json").read_text())
    published = json.loads((REPORT / "completion.json").read_text())
    assert completion["status"] == "thrml_finite_sweep_contract_complete"
    assert completion["cells"] == 64
    assert completion["controls_gate_passed"] is True
    assert completion["result_digest"] == published["result_digest"]
    assert completion["archive_sha256"] == published["archive_sha256"]
    assert record["evidence"]["thrml_cells"].startswith("software_simulation")
    assert record["evidence"]["exact_references"].startswith("exact_reference")


def test_replay_rejects_a_tampered_histogram(tmp_path: Path) -> None:
    archive = REPORT / "study.json.gz"
    if not archive.exists():
        pytest.skip("archive not recorded yet")
    record = json.loads(gzip.decompress(archive.read_bytes()))
    record["histograms"]["free/forward/all_minus/K1"][0] += 1
    tampered = tmp_path / "tampered.json.gz"
    tampered.write_bytes(gzip.compress(json.dumps(record).encode()))
    with pytest.raises(ValueError, match="result digest"):
        study.replay_archive(tampered, tmp_path / "out")
