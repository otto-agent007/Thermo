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
    from thermo_lab.improvement_harness import proposer

    worktree = tmp_path / "candidate"
    worktree.mkdir()
    monkeypatch.setenv("SECRET_TEST_TOKEN", "must-not-be-inherited")
    calls = []

    def fake_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        Path(argv[argv.index("--output-last-message") + 1]).write_text("Useful recommendation\n")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(proposer, "_head", lambda worktree: "a" * 40)
    monkeypatch.setattr(proposer, "_bounded_run", fake_runner)
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
    from thermo_lab.improvement_harness import proposer

    def fake_runner(argv, **kwargs):
        if write_file:
            Path(argv[argv.index("--output-last-message") + 1]).write_text("text")
        return subprocess.CompletedProcess(argv, returncode, b"", b"error: failure")

    monkeypatch.setattr(proposer, "_head", lambda worktree: "a" * 40)
    monkeypatch.setattr(proposer, "_bounded_run", fake_runner)
    with pytest.raises(RuntimeError, match="recommendation"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10, baseline="a" * 40)


def test_codex_missing_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    from thermo_lab.improvement_harness import proposer

    def missing(*args, **kwargs):
        raise FileNotFoundError("codex")

    monkeypatch.setattr(proposer, "_head", lambda worktree: "a" * 40)
    monkeypatch.setattr(proposer, "_bounded_run", missing)
    with pytest.raises(RuntimeError, match="unavailable"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10, baseline="a" * 40)


def test_codex_timeout_is_failed(tmp_path: Path, monkeypatch) -> None:
    from thermo_lab.improvement_harness import proposer

    def timed_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(proposer, "_head", lambda worktree: "a" * 40)
    monkeypatch.setattr(proposer, "_bounded_run", timed_out)
    with pytest.raises(RuntimeError, match="timed out"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 1, baseline="a" * 40)


def test_codex_rejects_changed_head_after_generation(tmp_path: Path, monkeypatch) -> None:
    from thermo_lab.improvement_harness import proposer

    heads = iter(("a" * 40, "b" * 40))

    def fake_runner(argv, **kwargs):
        Path(argv[argv.index("--output-last-message") + 1]).write_text("recommendation")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(proposer, "_head", lambda worktree: next(heads))
    monkeypatch.setattr(proposer, "_bounded_run", fake_runner)
    with pytest.raises(ValueError, match="baseline mismatch"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10, baseline="a" * 40)


def test_codex_rejects_wrong_frozen_baseline_before_launch(tmp_path: Path, monkeypatch) -> None:
    from thermo_lab.improvement_harness import proposer

    codex_called = False

    def fake_runner(argv, **kwargs):
        nonlocal codex_called
        codex_called = True
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(proposer, "_head", lambda worktree: "a" * 40)
    monkeypatch.setattr(proposer, "_bounded_run", fake_runner)
    with pytest.raises(ValueError, match="baseline mismatch"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10, baseline="b" * 40)
    assert not codex_called


def test_codex_requires_frozen_baseline(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="baseline"):
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10)


def test_codex_failure_uses_bounded_runner_and_safe_local_detail(
    tmp_path: Path, monkeypatch
) -> None:
    from thermo_lab.improvement_harness import proposer

    monkeypatch.setattr(proposer, "_head", lambda worktree: "a" * 40)
    seen = []

    def bounded(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            7,
            b"x" * 65536,
            b"error: invalid configuration OPENAI_API_KEY=leakvalue\n"
            b"error: Authorization: Bearer topsecret\n",
        )

    def no_unbounded_codex(argv, **kwargs):
        raise AssertionError("Codex must use the bounded subprocess runner")

    monkeypatch.setattr(proposer, "_bounded_run", bounded, raising=False)
    monkeypatch.setattr(proposer.subprocess, "run", no_unbounded_codex)
    with pytest.raises(RuntimeError) as failure:
        run_codex(tmp_path, "Objective: x. Paths: y. Checks: z.", 10, baseline="a" * 40)
    message = str(failure.value)
    assert "exit 7" in message
    assert "invalid configuration" in message
    assert "leakvalue" not in message
    assert "topsecret" not in message
    assert len(message) < 700
    assert seen[0][1]["env"].keys() <= {
        "PATH",
        "HOME",
        "CODEX_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "TMPDIR",
    }
    assert seen[0][1]["timeout"] == 10


def test_codex_rejects_oversized_recommendation(tmp_path: Path, monkeypatch) -> None:
    from thermo_lab.improvement_harness import proposer

    monkeypatch.setattr(proposer, "_head", lambda worktree: "a" * 40)

    def fake_runner(argv, **kwargs):
        Path(argv[argv.index("--output-last-message") + 1]).write_text("x" * 100_001)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(proposer, "_bounded_run", fake_runner)
    with pytest.raises(ValueError, match="recommendation.*too large"):
        run_codex(tmp_path, "Objective: x", 10, baseline="a" * 40)


def test_codex_failure_redacts_credential_bearing_detail(tmp_path: Path, monkeypatch) -> None:
    from thermo_lab.improvement_harness import proposer

    monkeypatch.setattr(proposer, "_head", lambda worktree: "a" * 40)

    def fake_runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            7,
            b"",
            b'error: {"token":"fake-secret-value"}\n'
            b"error: Authorization: Basic dXNlcjpwYXNz\n"
            b'error: invalid configuration OPENAI_API_KEY="fake secret value"\n',
        )

    monkeypatch.setattr(proposer, "_bounded_run", fake_runner)
    with pytest.raises(RuntimeError) as failure:
        run_codex(tmp_path, "Objective: x", 10, baseline="a" * 40)
    message = str(failure.value)
    assert "fake-secret-value" not in message
    assert "dXNlcjpwYXNz" not in message
    assert "fake secret value" not in message
    assert "invalid configuration" in message
