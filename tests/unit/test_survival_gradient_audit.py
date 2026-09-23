"""Source authentication and persisted replay must reject edited evidence."""

import importlib

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec("thermo_lab.survival_gradient_audit") is not None, (
        "audit missing"
    )
    return importlib.import_module("thermo_lab.survival_gradient_audit")


def fixture():
    p = np.zeros((1, 9)).tolist()
    q = np.array(p)
    q[0, 0] = 0.001
    return {
        "request": {
            "seed": 0,
            "horizon": 1,
            "inputs": {
                "initial_parameters": p,
                "occurrence_target_indices": [0, 0],
                "occurrence_site_indices": [[0, 1], [1, 2]],
            },
        },
        "steps": [{"update": {"updated_parameters": q.tolist()}}],
        "result_digest": "fixture-only",
    }


def test_recorded_displacement_and_prediction_are_reconstructible():
    m = api()
    f = fixture()
    r = m.analyze_fit(f, site_count=3)
    before, after = r["checkpoints"]
    step = r["updates"][0]
    gradient = np.array(before["gradient"])
    delta = np.array(f["steps"][0]["update"]["updated_parameters"])
    assert step["predicted_log_change"] == pytest.approx(float(np.sum(gradient * delta)))
    assert step["actual_log_change"] == pytest.approx(
        after["log_survival"] - before["log_survival"]
    )
    assert sum(step["occurrence_directional_contributions"]) == pytest.approx(
        step["predicted_log_change"]
    )
    assert sum(step["group_directional_contributions"]) == pytest.approx(
        step["predicted_log_change"]
    )
    assert step["displacement_norm"] == pytest.approx(0.001)


def test_untrusted_archive_bytes_rejected(tmp_path):
    m = api()
    (tmp_path / "study.json").write_text("{}")
    with pytest.raises(ValueError, match="archive"):
        m.load_archive(tmp_path)


def test_rehashed_changed_result_rejected():
    m = api()
    f = fixture()
    r = m.analyze_fit(f, site_count=3)
    r["updates"][0]["actual_log_change"] += 0.1
    with pytest.raises(ValueError, match="replay"):
        m.validate_fit_analysis(r, f, site_count=3)


def test_json_numeric_type_substitution_rejected():
    m = api()
    f = fixture()
    r = m.analyze_fit(f, site_count=3)
    r["seed"] = False
    with pytest.raises(ValueError, match="replay"):
        m.validate_fit_analysis(r, f, site_count=3)


def test_preflight_certifies_every_law_before_archive_analysis():
    m = api()
    assert hasattr(m, "gradient_preflight"), "preflight missing"
    rows = m.gradient_preflight()
    assert {r["horizon"] for r in rows} == {1, 2, 4, 8, 16, 30, "equilibrium"}
    assert all(r["maximum_absolute_error"] < 1e-7 for r in rows)


def test_report_rejects_truncated_evidence_before_rendering():
    with pytest.raises(ValueError, match="replay"):
        api().render_report({"fits": []})


def test_failed_source_load_never_writes_completion(tmp_path):
    m = api()
    with pytest.raises(ValueError, match="archive"):
        m.run_audit(tmp_path / "output", tmp_path / "missing")
    assert not (tmp_path / "output" / "completion.json").exists()
    with pytest.raises(FileExistsError):
        m.run_audit(tmp_path / "output", tmp_path / "missing")
