"""Pinned uniform control and every context-weighted update replay before publication."""

import copy
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256

CONTROL = Path(__file__).resolve().parents[2] / (
    "docs/experiment-reports/2026-09-16-local-conservation-tradeoff/study.json"
)


@pytest.fixture(scope="module")
def audit():
    from thermo_lab.context_conservation_audit import build_audit

    return build_audit(json.loads(CONTROL.read_text()))


def test_full_matched_grid_replays_with_original_control_unchanged(audit):
    from thermo_lab.context_conservation_audit import CONTROL_DIGEST, validate_audit

    assert validate_audit(json.loads(json.dumps(audit))) == audit
    assert canonical_sha256(audit["control"]) == CONTROL_DIGEST
    assert audit["schema_version"] == "1.1.0"
    assert audit["request"]["identity_version"] == "context_weighted_conservation.v2"
    assert len(audit["cells"]) == 3
    assert len(audit["evaluations"]) == 8
    assert len(audit["comparisons"]) == 6
    assert "contexts" not in audit["request"]  # derived coefficients belong to the result
    for cell in audit["cells"]:
        assert len(cell["fits"]) == 37
        assert len(cell["measurements"]["survival"]) == 500
        for fit in cell["fits"]:
            assert fit["objective"] <= fit["attempts"][0]["initial_objective"]
            assert np.max(np.abs(fit["parameters"])) <= 2
            assert [attempt["updates"] for attempt in fit["attempts"]] == [100, 100]
    # Independent reconstruction of each selection objective from its reported visible law.
    contexts = np.array([p["context_weights"] for p in audit["contexts"]["profiles"]])
    target = np.array([t["conditional"] for t in audit["control"]["request"]["targets"]])
    particles = np.array([0, 1, 1, 2])
    invalid = particles[:, None] != particles[None, :]
    for cell in audit["cells"]:
        visible = np.array(cell["measurements"]["visible_tables"])
        objective = np.sum(
            contexts[:, :, None] * ((visible - target) ** 2 + cell["penalty"] * visible * invalid),
            axis=(1, 2),
        )
        np.testing.assert_allclose(
            [f["objective"] for f in cell["fits"]], objective, atol=1e-14, rtol=0
        )


def test_repaired_digest_cannot_authorize_modified_context_profile(audit):
    from thermo_lab.context_conservation_audit import audit_digest, validate_audit

    changed = copy.deepcopy(audit)
    weights = changed["contexts"]["profiles"][0]["context_weights"]
    weights[0] -= 0.001
    weights[1] += 0.001
    changed["result_digest"] = audit_digest(changed)
    with pytest.raises(ValueError, match="numerical replay"):
        validate_audit(changed)


def test_new_weighted_results_still_require_bitwise_replay(audit):
    from thermo_lab.context_conservation_audit import audit_digest, validate_audit

    changed = copy.deepcopy(audit)
    parameters = changed["cells"][0]["parameters"][0]
    parameters[0] = float(np.nextafter(parameters[0], np.inf))
    changed["result_digest"] = audit_digest(changed)
    with pytest.raises(ValueError, match="numerical replay"):
        validate_audit(changed)


@pytest.mark.slow
@pytest.mark.skipif(platform.machine() != "x86_64", reason="x86 SIMD dispatch regression")
def test_pinned_control_replays_with_baseline_numpy_cpu_instructions():
    script = """
import json
from pathlib import Path
from thermo_lab.hashing import canonical_sha256
from thermo_lab.pinned_control_replay import CONTROL_DIGEST, replay_pinned_control
control = json.loads(Path(__import__('sys').argv[1]).read_text())
checked = replay_pinned_control(control)
assert canonical_sha256(checked) == CONTROL_DIGEST
"""
    env = {
        **os.environ,
        "NPY_DISABLE_CPU_FEATURES": "X86_V3,X86_V4,AVX512_ICL,AVX512_SPR",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script, str(CONTROL)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mutation", ["policy", "control", "comparison"])
def test_reporting_rejects_changed_inputs_or_results(audit, mutation):
    from thermo_lab.context_conservation_audit import audit_digest, render_report

    changed = copy.deepcopy(audit)
    if mutation == "policy":
        changed["request"]["joint_screen"] = "survival_only"
        changed["request_hash"] = canonical_sha256(changed["request"])
        changed["result_digest"] = audit_digest(changed)
    elif mutation == "control":
        changed["control"]["cells"][2]["parameters"][0][0] = 0.0
    else:
        changed["comparisons"][0]["survival_difference"] += 0.1
    with pytest.raises(ValueError):
        render_report(changed)


def test_existing_destination_is_not_overwritten(tmp_path):
    from thermo_lab.context_conservation_audit import run_study

    marker = tmp_path / "completion.json"
    marker.write_text("earlier evidence")
    with pytest.raises(FileExistsError):
        run_study(CONTROL, tmp_path)
    assert marker.read_text() == "earlier evidence"


def test_failed_replay_cannot_publish_completion(audit, tmp_path, monkeypatch):
    from thermo_lab import context_conservation_audit as study

    def fail(_audit):
        raise ValueError("injected replay failure")

    monkeypatch.setattr(study, "build_audit", lambda control: audit)
    monkeypatch.setattr(study, "render_report", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected replay"):
        study.run_study(CONTROL, output)
    assert (output / "study.json").is_file()
    assert not (output / "completion.json").exists()
