"""CLI orchestration against real isolated Git trees and fake external commands."""

import hashlib
import subprocess

import pytest

from thermo_lab.improvement_harness import cli
from thermo_lab.improvement_harness.checks import run_checks
from thermo_lab.improvement_harness.plan import load_plan
from thermo_lab.improvement_harness.store import read_candidate


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "dashboard/src").mkdir(parents=True)
    (repo / "dashboard/src/title.txt").write_text("before\n")
    (repo / ".gitignore").write_text(".worktrees/\ndashboard/test-results/\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Test", "-c", "user.email=a@b.c", "commit", "-qm", "base")
    monkeypatch.chdir(repo)
    plan_file = tmp_path / "plan.json"
    assert (
        cli.main(
            [
                "init-plan",
                "--track",
                "dashboard",
                "--objective",
                "Clear title",
                "--output",
                str(plan_file),
            ]
        )
        == 0
    )
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[-1] == "test:browser":
            directory = kwargs["cwd"] / "test-results"
            directory.mkdir(exist_ok=True)
            for name in ("overview.png", "mobile.png", "experiments.png"):
                (directory / name).write_bytes(b"\x89PNG\r\n\x1a\n")
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok", stderr=b"")

    def checks(track, cwd, seconds, *, record_dir):
        return run_checks(track, cwd, seconds, record_dir=record_dir, runner=runner)

    monkeypatch.setattr(cli, "run_checks", checks)
    monkeypatch.setattr(cli.dashboard, "run_checks", checks)

    def proposer(worktree, prompt, seconds, *, baseline):
        assert baseline == git(repo, "rev-parse", "HEAD")
        path = worktree / "dashboard/src/title.txt"
        path.write_text(path.read_text() + "after\n")
        return "Make the title clearer."

    monkeypatch.setattr(cli, "run_codex", proposer)
    return repo, plan_file, tmp_path / "output", calls


def candidates(output):
    return sorted(path for path in output.iterdir() if (path / "request.json").is_file())


def snapshot(path):
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def test_init_pins_clean_full_head_and_refuses_overwrite(setup):
    repo, plan_file, _, _ = setup
    assert load_plan(plan_file).baseline_commit == git(repo, "rev-parse", "HEAD")
    assert (
        cli.main(
            [
                "init-plan",
                "--track",
                "dashboard",
                "--objective",
                "Other",
                "--output",
                str(plan_file),
            ]
        )
        == 1
    )
    (repo / "dashboard/src/title.txt").write_text("dirty")
    assert (
        cli.main(
            [
                "init-plan",
                "--track",
                "dashboard",
                "--objective",
                "Other",
                "--output",
                str(plan_file.parent / "dirty.json"),
            ]
        )
        == 1
    )


def test_two_runs_preserve_candidate_and_baseline_and_budget(setup, capsys):
    repo, plan_file, output, calls = setup
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 0
    first = candidates(output)[0]
    original = snapshot(first)
    for name in (
        "recommendation.md",
        "patch.diff",
        "request.json",
        "result.json",
        "report.md",
        "plan.json",
    ):
        assert (first / name).is_file()
    result = read_candidate(output, first.name)["result"]
    assert result["execution"] == "complete"
    assert result["verification"] == "passed"
    assert (
        result["patch_digest"]
        == "sha256:" + hashlib.sha256((first / "patch.diff").read_bytes()).hexdigest()
    )
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 0
    assert len(calls) == 15  # baseline once, two candidates
    assert snapshot(first) == original
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 1
    assert len(calls) == 15
    assert git(repo, "status", "--porcelain") == ""
    assert cli.main(["show", first.name, "--output-dir", str(output)]) == 0
    assert len(calls) == 15
    assert "Clear title" in capsys.readouterr().out
    assert (
        cli.main(
            [
                "review",
                first.name,
                "--output-dir",
                str(output),
                "--decision",
                "accepted",
                "--note",
                "Owner reviewed",
            ]
        )
        == 0
    )
    assert {k: v for k, v in snapshot(first).items() if not k.startswith("review-")} == original


def test_manual_patch_requires_recommendation_before_candidate(setup, tmp_path):
    _, plan_file, output, calls = setup
    patch = tmp_path / "manual.diff"
    patch.write_text("invalid")
    assert (
        cli.main(["run", str(plan_file), "--output-dir", str(output), "--manual-patch", str(patch)])
        == 1
    )
    assert not output.exists()
    assert not calls


def test_parent_patch_and_development_only_feedback(setup, monkeypatch):
    _, plan_file, output, _ = setup
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 0
    parent = candidates(output)[0]
    original = snapshot(parent)

    def proposer(worktree, prompt, seconds, *, baseline):
        path = worktree / "dashboard/src/title.txt"
        assert path.read_text() == "before\nafter\n"
        assert "Development checks" in prompt
        assert "Owner reviewed" not in prompt
        path.write_text("child\n")
        return "A cumulative child."

    monkeypatch.setattr(cli, "run_codex", proposer)
    assert (
        cli.main(["run", str(plan_file), "--output-dir", str(output), "--parent", parent.name]) == 0
    )
    child = next(path for path in candidates(output) if path != parent)
    assert "-before" in (child / "patch.diff").read_text()
    assert snapshot(parent) == original


def test_failure_preserves_diagnostics(setup, monkeypatch):
    _, plan_file, output, _ = setup

    def unavailable(*args, **kwargs):
        raise RuntimeError("proposer unavailable: missing CLI")

    monkeypatch.setattr(cli, "run_codex", unavailable)
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 1
    candidate = candidates(output)[0]
    result = read_candidate(output, candidate.name)["result"]
    assert result["execution"] == "unavailable"
    assert result["verification"] == "inconclusive"
    assert "missing CLI" in (candidate / "report.md").read_text()


def test_heldout_reserved_before_evaluation_and_blocks_same_plan(setup, monkeypatch):
    _, plan_file, output, _ = setup
    plan = load_plan(plan_file).model_copy(update={"track": "research", "heldout_role": "final"})
    # Keep the fake source hook within this fixture's declared allowlist; the real
    # scientific evaluator separately enforces the exact research preset.
    monkeypatch.setattr(cli, "_validate_preset", lambda plan: None)
    events = []
    real_claim = cli.claim_heldout

    def claim(*args):
        events.append("claim")
        return real_claim(*args)

    def evaluate(*args, **kwargs):
        events.append("evaluate")
        assert events == ["claim", "evaluate"]
        return {
            "execution": "complete",
            "verification": "passed",
            "research_outcome": "inconclusive",
            "evidence": "exact_reference",
        }

    monkeypatch.setattr(cli, "claim_heldout", claim)
    monkeypatch.setattr(cli, "evaluate_three_site", evaluate)
    assert cli.run_candidate(plan, output) == 0
    parent = candidates(output)[0]
    assert cli.run_candidate(plan, output, parent=parent.name) == 1
    assert cli.run_candidate(plan, output) == 1
    assert len(candidates(output)) == 1
    assert events == ["claim", "evaluate"]


def test_time_budget_is_shared_by_candidates(setup, monkeypatch):
    _, plan_file, output, calls = setup
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 0
    monkeypatch.setattr(cli, "_remaining_budget", lambda *args: 0)
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 1
    assert len(candidates(output)) == 1
    assert len(calls) == 10


def test_manual_intake_and_parent_tampering(setup, tmp_path):
    repo, plan_file, output, _ = setup
    path = repo / "dashboard/src/title.txt"
    path.write_text("manual\n")
    patch = tmp_path / "manual.diff"
    patch.write_text(git(repo, "diff") + "\n")
    path.write_text("before\n")
    recommendation = tmp_path / "recommendation.md"
    recommendation.write_text("Manual title change.")
    assert (
        cli.main(
            [
                "run",
                str(plan_file),
                "--output-dir",
                str(output),
                "--manual-patch",
                str(patch),
                "--recommendation-file",
                str(recommendation),
            ]
        )
        == 0
    )
    parent = candidates(output)[0]
    assert read_candidate(output, parent.name)["result"]["proposer"] == {"identity": "manual"}
    (parent / "patch.diff").write_text("tampered")
    assert (
        cli.main(["run", str(plan_file), "--output-dir", str(output), "--parent", parent.name]) == 1
    )
    assert len(candidates(output)) == 1


def test_oversized_manual_recommendation_is_inspectable_failure(setup, tmp_path):
    repo, plan_file, output, calls = setup
    path = repo / "dashboard/src/title.txt"
    path.write_text("manual\n")
    patch = tmp_path / "manual.diff"
    patch.write_text(git(repo, "diff") + "\n")
    path.write_text("before\n")
    recommendation = tmp_path / "recommendation.md"
    recommendation.write_text("x" * 100_001)
    assert (
        cli.main(
            [
                "run",
                str(plan_file),
                "--output-dir",
                str(output),
                "--manual-patch",
                str(patch),
                "--recommendation-file",
                str(recommendation),
            ]
        )
        == 1
    )
    candidate = candidates(output)[0]
    result = read_candidate(output, candidate.name)["result"]
    assert result["execution"] == "failed"
    assert "recommendation" in result["diagnostic"]
    assert result.get("recommendation") is None
    assert not (candidate / "recommendation.md").exists()
    assert not calls


def test_symlink_manual_recommendation_is_inspectable_failure(setup, tmp_path):
    repo, plan_file, output, calls = setup
    path = repo / "dashboard/src/title.txt"
    path.write_text("manual\n")
    patch = tmp_path / "manual.diff"
    patch.write_text(git(repo, "diff") + "\n")
    path.write_text("before\n")
    target = tmp_path / "real-recommendation.md"
    target.write_text("Useful title change")
    recommendation = tmp_path / "recommendation.md"
    recommendation.symlink_to(target)
    assert (
        cli.main(
            [
                "run",
                str(plan_file),
                "--output-dir",
                str(output),
                "--manual-patch",
                str(patch),
                "--recommendation-file",
                str(recommendation),
            ]
        )
        == 1
    )
    candidate = candidates(output)[0]
    result = read_candidate(output, candidate.name)["result"]
    assert result["execution"] == "failed"
    assert "regular file" in result["diagnostic"]
    assert not calls


def test_oversized_codex_message_is_inspectable_failure(setup, monkeypatch):
    from thermo_lab.improvement_harness import proposer

    _, plan_file, output, calls = setup

    def fake_runner(argv, **kwargs):
        from pathlib import Path

        Path(argv[argv.index("--output-last-message") + 1]).write_text("x" * 100_001)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(proposer, "_bounded_run", fake_runner)
    monkeypatch.setattr(cli, "run_codex", proposer.run_codex)
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 1
    candidate = candidates(output)[0]
    result = read_candidate(output, candidate.name)["result"]
    assert result["execution"] == "failed"
    assert "recommendation is too large" in result["diagnostic"]
    assert result.get("recommendation") is None
    assert not (candidate / "recommendation.md").exists()
    assert not calls


def test_oversized_result_is_reduced_to_visible_failure(setup, monkeypatch):
    _, plan_file, output, _ = setup
    original = cli.dashboard.evaluate_dashboard

    def oversized(*args, **kwargs):
        result = original(*args, **kwargs)
        result["unexpected_output"] = "x" * 1_000_000
        return result

    monkeypatch.setattr(cli.dashboard, "evaluate_dashboard", oversized)
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 1
    candidate = candidates(output)[0]
    result = read_candidate(output, candidate.name)["result"]
    assert result["execution"] == "failed"
    assert result["diagnostic"] == "result is too large for dashboard review"
    assert (candidate / "result.json").stat().st_size <= 1_000_000


def test_proposal_overrun_stops_before_evaluation(setup, monkeypatch):
    _, plan_file, output, calls = setup
    original = cli.run_codex
    clock = [0.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 1801.0
        return result

    monkeypatch.setattr(cli, "run_codex", delayed)
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 1
    result = read_candidate(output, candidates(output)[0].name)["result"]
    assert result["execution"] == "timed_out"
    assert not calls
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 1
    assert len(candidates(output)) == 1


def test_broken_baseline_never_certifies_green_candidate(setup, monkeypatch):
    _, plan_file, output, _ = setup
    original = cli.dashboard.run_checks
    observations = [0]

    def checks(*args, **kwargs):
        result = original(*args, **kwargs)
        observations[0] += 1
        if observations[0] == 1:
            result[1] = result[1].model_copy(
                update={"execution": "failed", "verification": "failed", "exit_status": 1}
            )
        return result

    monkeypatch.setattr(cli.dashboard, "run_checks", checks)
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 1
    result = read_candidate(output, candidates(output)[0].name)["result"]
    assert result["verification"] == "inconclusive"
    assert result["execution"] == "failed"


def test_output_symlink_cannot_redirect_writes(setup, tmp_path):
    _, plan_file, output, calls = setup
    outside = tmp_path / "outside"
    outside.mkdir()
    output.symlink_to(outside, target_is_directory=True)
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 1
    assert list(outside.iterdir()) == []
    assert not calls


@pytest.mark.parametrize(
    "artifact",
    ["patch.diff", "plan.json", "candidate/artifacts/mobile.png", "candidate/checks/unit.log"],
)
def test_show_rejects_tampered_evidence_without_commands(setup, artifact):
    _, plan_file, output, calls = setup
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 0
    candidate = candidates(output)[0]
    (candidate / artifact).write_bytes(b"changed")
    assert cli.main(["show", candidate.name, "--output-dir", str(output)]) == 1
    assert len(calls) == 10


def test_review_rejects_symlink_output_root(setup, tmp_path):
    _, plan_file, output, _ = setup
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 0
    candidate = candidates(output)[0]
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)
    original = snapshot(candidate)
    assert (
        cli.main(
            [
                "review",
                candidate.name,
                "--output-dir",
                str(alias),
                "--decision",
                "accepted",
                "--note",
                "Owner review",
            ]
        )
        == 1
    )
    assert snapshot(candidate) == original


@pytest.mark.parametrize(
    "damage", ["missing_log", "changed_screenshot", "symlink_logs", "symlink_screenshots"]
)
def test_show_and_parent_reject_damaged_baseline_evidence(setup, tmp_path, damage):
    _, plan_file, output, calls = setup
    assert cli.main(["run", str(plan_file), "--output-dir", str(output)]) == 0
    candidate = candidates(output)[0]
    result = read_candidate(output, candidate.name)["result"]
    baseline = (output / result["baseline_record"]).parent
    if damage == "missing_log":
        (baseline / "checks/unit.log").unlink()
    elif damage == "changed_screenshot":
        (baseline / "artifacts/mobile.png").write_bytes(b"altered baseline")
    else:
        name = "checks" if damage == "symlink_logs" else "artifacts"
        outside = tmp_path / f"outside-{name}"
        (baseline / name).rename(outside)
        (baseline / name).symlink_to(outside, target_is_directory=True)
    assert cli.main(["show", candidate.name, "--output-dir", str(output)]) == 1
    assert (
        cli.main(["run", str(plan_file), "--output-dir", str(output), "--parent", candidate.name])
        == 1
    )
    assert len(candidates(output)) == 1
    assert len(calls) == 10
