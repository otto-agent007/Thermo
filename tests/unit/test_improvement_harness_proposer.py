"""The proposer is an isolated external process with checked output."""

import subprocess
from pathlib import Path

import pytest

from thermo_lab.improvement_harness.proposer import build_proposer_prompt, run_codex


def test_prompt_includes_frozen_scope_and_checks() -> None:
    prompt = build_proposer_prompt(
        "Improve dashboard navigation", ("dashboard/src/",), ("unit tests", "browser checks")
    )
    assert "Objective: Improve dashboard navigation" in prompt
    assert "Allowed paths: dashboard/src/" in prompt
    assert "Fixed checks: unit tests, browser checks" in prompt


def test_codex_returns_recommendation_from_output_file(tmp_path: Path, monkeypatch) -> None:
    worktree = tmp_path / "candidate"
    worktree.mkdir()
    monkeypatch.setenv("SECRET_TEST_TOKEN", "must-not-be-inherited")
    calls = []

    def fake_run(argv, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, b"a" * 40 + b"\n", b"")
        calls.append((argv, kwargs))
        Path(argv[argv.index("--output-last-message") + 1]).write_text("Useful recommendation\n")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("thermo_lab.improvement_harness.proposer.subprocess.run", fake_run)
    assert (
        run_codex(
            worktree,
            "Objective: improve. Paths: dashboard/src. Checks: test.",
            10,
            baseline="a" * 40,
        )
        == "Useful recommendation\n"
    )
    argv, kwargs = calls[0]
    assert argv[:7] == [
        "codex",
        "exec",
        "-C",
        str(worktree),
        "-s",
        "workspace-write",
        "--output-last-message",
    ]
    assert argv[-1].startswith("Objective:")
    assert "--ignore-user-config" in argv
    assert "--strict-config" in argv
    assert "sandbox_workspace_write.network_access=false" in argv
    assert "sandbox_workspace_write.writable_roots=[]" in argv
    assert kwargs["timeout"] == 10
    assert kwargs["shell"] is False
    assert "SECRET_TEST_TOKEN" not in kwargs["env"]


@pytest.mark.parametrize("returncode,write_file", [(1, True), (0, False)])
def test_codex_fails_without_successful_recommendation(
    tmp_path: Path, monkeypatch, returncode: int, write_file: bool
) -> None:
    def fake_run(argv, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, b"a" * 40 + b"\n", b"")
        if write_file:
            Path(argv[argv.index("--output-last-message") + 1]).write_text("text")
        return subprocess.CompletedProcess(argv, returncode, "", "failure")

    monkeypatch.setattr("thermo_lab.improvement_harness.proposer.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="recommendation"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10)


def test_codex_missing_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    def missing(*args, **kwargs):
        if args[0][0] == "git":
            return subprocess.CompletedProcess(args[0], 0, b"a" * 40 + b"\n", b"")
        raise FileNotFoundError("codex")

    monkeypatch.setattr("thermo_lab.improvement_harness.proposer.subprocess.run", missing)
    with pytest.raises(RuntimeError, match="unavailable"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10)


def test_codex_timeout_is_failed(tmp_path: Path, monkeypatch) -> None:
    def timed_out(argv, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, b"a" * 40 + b"\n", b"")
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr("thermo_lab.improvement_harness.proposer.subprocess.run", timed_out)
    with pytest.raises(RuntimeError, match="timed out"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 1)


def test_codex_rejects_changed_head_after_generation(tmp_path: Path, monkeypatch) -> None:
    heads = iter((b"a" * 40 + b"\n", b"b" * 40 + b"\n"))

    def fake_run(argv, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, next(heads), b"")
        Path(argv[argv.index("--output-last-message") + 1]).write_text("recommendation")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("thermo_lab.improvement_harness.proposer.subprocess.run", fake_run)
    with pytest.raises(ValueError, match="baseline mismatch"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10)


def test_codex_rejects_wrong_frozen_baseline_before_launch(tmp_path: Path, monkeypatch) -> None:
    codex_called = False

    def fake_run(argv, **kwargs):
        nonlocal codex_called
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, b"a" * 40 + b"\n", b"")
        codex_called = True
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("thermo_lab.improvement_harness.proposer.subprocess.run", fake_run)
    with pytest.raises(ValueError, match="baseline mismatch"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10, baseline="b" * 40)
    assert not codex_called
