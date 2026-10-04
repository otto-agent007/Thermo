"""E0 stage B: THRML execution of an M5a kernel's inner sweep; exact side and replay."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization_core as core
from thermo_lab import thrml_m5a_kernel_inner_sweep as study

REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiment-reports"
    / "2026-10-03-thrml-m5a-kernel-inner-sweep"
)


@pytest.fixture(scope="module")
def request_small() -> dict:
    return study.study_request(chains=2_000)


@pytest.fixture(scope="module")
def kernel(request_small: dict) -> dict:
    return study.load_kernel(request_small)


def test_request_pins_the_m5c_primary_arm_and_evidence_classes(request_small: dict) -> None:
    k = request_small["kernel"]
    assert (k["reading"], k["method"], k["cap"]) == ("B", "variational", 1.0)
    assert k["seed"] == 0 and k["site"] == 1
    assert request_small["sweeps"] == [1, 2, 4]
    assert request_small["thrml_model"]["numeric_dtype"] == "float32"
    assert request_small["thrml_model"]["block_order"] == "hidden_first"
    assert [c["id"] for c in study.cells(request_small)] == [
        "y0-/K1",
        "y0-/K2",
        "y0-/K4",
        "y0+/K1",
        "y0+/K2",
        "y0+/K4",
    ]


def test_kernel_is_authenticated_from_the_m5b_archive(kernel: dict) -> None:
    structure, vector = kernel["structure"], kernel["vector"]
    assert structure["site"] == 1
    assert len(structure["blanket"]) == 10
    assert len(structure["triples"]) == 7
    assert vector.shape == (m5a.parameter_count(structure),)
    assert vector.dtype == np.float64


def test_joint_sweep_kernel_is_stochastic_and_fixes_the_boltzmann_law(kernel: dict) -> None:
    structure, vector = kernel["structure"], kernel["vector"]
    inputs = m5a.blanket_inputs(structure)[:3]
    boltzmann = study._stationary_joint(vector, structure, inputs)
    for order in study.ORDERS:
        for x, law in zip(inputs, boltzmann, strict=True):
            t = study.joint_sweep_kernel(vector, structure, x, order)
            assert t.shape == (256, 256)
            np.testing.assert_allclose(t.sum(axis=1), 1.0, atol=1e-12)
            assert 0.5 * np.abs(law @ t - law).sum() < 1e-12
    hidden = study.joint_sweep_kernel(vector, structure, inputs[0], "hidden_first")
    output = study.joint_sweep_kernel(vector, structure, inputs[0], "output_first")
    assert np.abs(hidden - output).max() > 1e-3


def test_joint_enumeration_reproduces_the_archived_inner_k_law(kernel: dict) -> None:
    """The study-local 256-state joint marginalizes to the hash-bound powered_rates law."""
    structure, vector = kernel["structure"], kernel["vector"]
    inputs = m5a.blanket_inputs(structure)
    inner = core.inner_kernel(vector, structure)
    laws, residual = study._joint_laws(vector, structure, inputs, "hidden_first", 1.0, 4)
    assert residual < 1e-12
    y_plus = study.joint_states(7)[:, 7] > 0
    for incoming in study.INCOMING:
        for k in (1, 2, 4):
            archived = study.output_law(inner, k, incoming)
            np.testing.assert_allclose(
                laws[(incoming, k)][:, y_plus].sum(axis=1), archived, atol=1e-12
            )


def test_first_hidden_block_forgets_the_hidden_start(kernel: dict) -> None:
    structure, vector = kernel["structure"], kernel["vector"]
    x = m5a.blanket_inputs(structure)[5]
    t = study.joint_sweep_kernel(vector, structure, x, "hidden_first")
    states = study.joint_states(7)
    rows = np.flatnonzero(states[:, 7] > 0)  # every (w, y=+1) start
    np.testing.assert_allclose(t[rows], np.broadcast_to(t[rows[0]], t[rows].shape), atol=1e-12)


def test_joint_index_inverts_joint_states() -> None:
    states = study.joint_states(7)
    np.testing.assert_array_equal(study.joint_index(states), np.arange(256))


def test_tolerance_shrinks_with_chains() -> None:
    law = np.full((4, 256), 1.0 / 256)
    spec = {"quantile": 0.999, "draws_output": 200, "draws_joint": 50, "numpy_seed": 1}
    assert study._joint_tolerance(law, 1_000, spec, 0) > study._joint_tolerance(
        law, 100_000, spec, 0
    )
    out = law[:, :2].sum(axis=1)
    assert study._output_tolerance(out, 1_000, spec, 0) > study._output_tolerance(
        out, 100_000, spec, 0
    )


def test_archived_report_replays(tmp_path: Path) -> None:
    archive = REPORT / "study.json.gz"
    if not archive.exists():
        pytest.skip("archive not recorded yet")
    record = study.replay_archive(archive, tmp_path / "replay", light=True)
    completion = json.loads((tmp_path / "replay" / "completion.json").read_text())
    published = json.loads((REPORT / "completion.json").read_text())
    assert completion["status"] == "thrml_m5a_kernel_inner_sweep_complete"
    assert completion["cells"] == 6
    assert completion["inputs_per_cell"] == 1024
    assert completion["controls_gate_passed"] is True
    assert completion["result_digest"] == published["result_digest"]
    assert completion["archive_sha256"] == published["archive_sha256"]
    for key in ("output_cells_passed", "joint_cells_passed", "control_rejections"):
        assert completion[key] == published[key]
    assert record["evidence"]["thrml_cells"].startswith("software_simulation")
    assert record["evidence"]["exact_references"].startswith("exact_reference")


def test_replay_rejects_a_tampered_histogram(tmp_path: Path) -> None:
    archive = REPORT / "study.json.gz"
    if not archive.exists():
        pytest.skip("archive not recorded yet")
    record = json.loads(gzip.decompress(archive.read_bytes()))
    record["histograms"]["y0-/K1"][0][0] += 1
    tampered = tmp_path / "tampered.json.gz"
    tampered.write_bytes(gzip.compress(json.dumps(record).encode()))
    with pytest.raises(ValueError, match="result digest"):
        study.replay_archive(tampered, tmp_path / "out", light=True)


@pytest.mark.slow
def test_one_thrml_cell_matches_the_exact_law_at_small_n(request_small: dict, kernel: dict) -> None:
    request = study.study_request(chains=512)
    refs = study.exact_references(
        request,
        tolerances={
            c["id"]: {"tolerance_output": 0.1, "tolerance_joint": 0.2} for c in study.cells(request)
        },
    )
    cell = study.cells(request)[0]  # y0-/K1
    out = study.run_cell(request, cell, kernel)
    evaluation = study.evaluate(
        request, refs, {c["id"]: out["histograms"] for c in study.cells(request)}
    )
    result = evaluation["cells"][cell["id"]]
    assert result["output_max_deviation"] < 0.1
    assert result["joint_max_tv"] < 0.2
