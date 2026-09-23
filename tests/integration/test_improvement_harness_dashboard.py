"""Dashboard catalog execution and provenance-bound screenshot retention."""

import subprocess

import pytest

from thermo_lab.improvement_harness import dashboard
from thermo_lab.improvement_harness.checks import run_checks
from thermo_lab.improvement_harness.plan import Plan


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    baseline, candidate, records = (
        tmp_path / name for name in ("baseline", "candidate", "records")
    )
    for worktree in (baseline, candidate):
        (worktree / "dashboard/test-results").mkdir(parents=True)
        for name in dashboard.SCREENSHOTS:
            (worktree / "dashboard/test-results" / name).write_bytes(b"stale")
    plan = Plan(
        schema_version=1,
        track="dashboard",
        objective="Readable layout",
        baseline_commit="a" * 40,
        allowed_paths=("dashboard/src/",),
        primary_metric="checks",
        direction="pass",
        threshold=1,
        max_candidates=2,
        wall_seconds=1800,
    )
    return plan, baseline, candidate, records


@pytest.mark.parametrize("mode", ["passed", "failed", "missing", "unavailable"])
def test_only_fresh_passing_screenshots_are_retained(evaluation, monkeypatch, mode):
    plan, baseline, candidate, records = evaluation

    def runner(argv, **kwargs):
        if argv[-1] == "test:browser":
            if mode != "missing":
                for name in dashboard.SCREENSHOTS:
                    (kwargs["cwd"] / "test-results" / name).write_bytes(b"fresh")
            if mode == "failed":
                return subprocess.CompletedProcess(argv, 1, stdout=b"assertion", stderr=b"")
            if mode == "unavailable":
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    stdout=b"",
                    stderr=b"Executable doesn't exist. Please run npx playwright install",
                )
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(
        dashboard,
        "run_checks",
        lambda track, cwd, seconds, *, record_dir: run_checks(
            track, cwd, seconds, record_dir=record_dir, runner=runner
        ),
    )
    result = dashboard.evaluate_dashboard(plan, baseline, candidate, record_dir=records)
    assert result["visual_evidence"] == ("available" if mode == "passed" else "unavailable")
    retained = list((records / "candidate/artifacts").glob("*.png"))
    assert len(retained) == (3 if mode == "passed" else 0)
    assert result["research_outcome"] == "not_applicable"
    assert "aesthetic_score" not in result
    if mode == "unavailable":
        assert result["execution"] == "unavailable"


def test_record_destination_must_be_outside_worktrees(evaluation):
    plan, baseline, candidate, _ = evaluation
    with pytest.raises(ValueError, match="outside"):
        dashboard.evaluate_dashboard(plan, baseline, candidate, record_dir=candidate / "evidence")


def test_symlink_observation_destination_is_rejected(evaluation, monkeypatch, tmp_path):
    plan, baseline, candidate, records = evaluation
    records.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (records / "baseline").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        dashboard.evaluate_dashboard(plan, baseline, candidate, record_dir=records)
    assert list(outside.iterdir()) == []
