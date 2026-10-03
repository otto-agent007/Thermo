"""Portability must preserve numerical evidence and ordinary filesystem errors."""

import json
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["cpu.max", "memory.max"])
@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_optional_metadata_reports_unavailable_only_for_cgroup_reads(monkeypatch, name, error):
    from thermo_lab.sampling_portability import _MetadataPath

    def unreadable(path, *args, **kwargs):
        raise error(str(path))

    monkeypatch.setattr(Path, "read_text", unreadable)
    assert _MetadataPath(f"/sys/fs/cgroup/{name}").read_text() == "unavailable"
    with pytest.raises(error):
        _MetadataPath("request.json").read_text()


def test_metadata_preserves_readable_values(monkeypatch):
    from thermo_lab.sampling_portability import _MetadataPath

    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "400000 100000\n")
    assert _MetadataPath("/sys/fs/cgroup/cpu.max").read_text() == "400000 100000\n"


def test_metric_comparison_accepts_roundoff_and_reports_its_size():
    from thermo_lab.sampling_portability import compare_metrics

    difference = compare_metrics({"float": [0.5 + 1e-14], "count": 4}, {"float": [0.5], "count": 4})
    assert 0 < difference < 2e-12


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (0.5 + 1e-8, 0.5),
        (float("nan"), 0.5),
        (0.5, float("nan")),
        (float("inf"), float("inf")),
        (1.0, 1),
        (True, 1),
        ("wrong", "right"),
        ([0.5, 0.5], [0.5]),
        ({"other": 0.5}, {"metric": 0.5}),
    ],
)
def test_metric_comparison_rejects_changes_and_nonfinite_values(actual, expected):
    from thermo_lab.sampling_portability import compare_metrics

    with pytest.raises(ValueError):
        compare_metrics(actual, expected)


def test_portable_replay_rejects_changed_request_before_reading_traces(tmp_path):
    from thermo_lab.sampling_portability import replay_fixed_budget

    (tmp_path / "request.json").write_text(json.dumps({"sources": {}}))
    (tmp_path / "results.json").write_text(json.dumps({"request_digest": "wrong"}))
    with pytest.raises(ValueError, match="request digest"):
        replay_fixed_budget(tmp_path)
    assert not (tmp_path / "portable-completion.json").exists()


@pytest.mark.parametrize("corruption", ["metric", "work_count", "trace", "source"])
def test_portable_archive_rejects_corruption(tmp_path, corruption):
    from thermo_lab.sampling_portability import replay_fixed_budget

    report = Path(__file__).resolve().parents[2] / (
        "docs/experiment-reports/2026-10-01-fixed-budget-sampling"
    )
    with tarfile.open(report / "evidence.tar.gz") as archive:
        archive.extractall(tmp_path, filter="data")
    out = tmp_path / "fixed-budget-sampling"
    original_completion = (out / "completion.json").read_bytes()
    if corruption in ("metric", "work_count"):
        path = out / "results.json"
        result = json.loads(path.read_text())
        if corruption == "metric":
            result["cells"][0]["means"]["joint_tv"] += 1e-6
        else:
            result["cells"][0]["spin_updates_per_trial"] += 1
        path.write_text(json.dumps(result))
        error = "value differs"
    else:
        path = (
            out / "traces.npz"
            if corruption == "trace"
            else out / "source/src/thermo_lab/fixed_budget_sampling.py"
        )
        path.write_bytes(path.read_bytes() + b"corruption")
        error = "hash mismatch" if corruption == "trace" else "source mismatch"
    with pytest.raises(ValueError, match=error):
        replay_fixed_budget(out)
    assert not (out / "portable-completion.json").exists()
    assert (out / "completion.json").read_bytes() == original_completion
