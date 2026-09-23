"""Bounded local Codex proposer adapter."""

import os
import re
import subprocess
import tempfile
from pathlib import Path

from thermo_lab.improvement_harness.checks import _bounded_run
from thermo_lab.improvement_harness.limits import read_recommendation

_LOCAL_DIAGNOSTIC_CHARS = 512


def _safe_stderr_tail(stderr: bytes | str | None) -> str:
    """Keep a short local error hint while dropping logs and obvious credentials."""
    if not stderr:
        return ""
    raw = stderr if isinstance(stderr, bytes) else stderr.encode("utf-8", errors="replace")
    tail = raw[-2048:].decode("utf-8", errors="replace")
    tail = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", tail)
    lines = [
        line.strip()
        for line in tail.splitlines()
        if re.match(r"(?i)^\s*(?:error|fatal|warning|caused by):", line)
    ]
    # Truncate each credential-bearing line at the key: values may be quoted,
    # contain spaces, or appear in JSON, so token-by-token replacement is unsafe.
    credential = re.compile(
        r"(?i)\b(?:authorization|[a-z0-9_]*(?:api[_-]?key|token|password|secret))"
        r"[\"']?\s*[:=]"
    )
    detail = " | ".join(
        line[: match.start()].rstrip() + " [redacted]"
        if (match := credential.search(line))
        else line
        for line in lines
    )
    detail = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", detail)
    return detail[-_LOCAL_DIAGNOSTIC_CHARS:]


def _failure_message(exit_status: int, stderr: bytes | str | None) -> str:
    message = f"proposer did not produce a recommendation (exit {exit_status})"
    detail = _safe_stderr_tail(stderr)
    return f"{message}: {detail}" if detail else message


def _head(worktree: Path) -> str:
    return (
        subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            shell=False,
        )
        .stdout.decode("ascii")
        .strip()
    )


def build_proposer_prompt(
    objective: str, allowed_paths: tuple[str, ...], checks: tuple[str, ...]
) -> str:
    """Keep the request's objective, edit scope, and fixed checks visible."""
    return (
        "Propose one bounded change in this detached candidate worktree.\n"
        f"Objective: {objective}\n"
        f"Allowed paths: {', '.join(allowed_paths)}\n"
        f"Fixed checks: {', '.join(checks)}\n"
        "Edit only allowed paths. Do not execute commands outside the fixed checks. "
        "Finish with a recommendation explaining the patch and any remaining risk.\n"
    )


def run_codex(worktree: Path, prompt: str, seconds: int, *, baseline: str) -> str:
    """Run the local proposer and return its recommendation text."""
    worktree = Path(worktree).resolve(strict=True)
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    current_head = _head(worktree)
    if current_head != baseline.lower():
        raise ValueError("baseline mismatch before proposal generation")
    allowed_env = {
        name: value
        for name in ("PATH", "HOME", "CODEX_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "TMPDIR")
        if (value := os.environ.get(name)) is not None
    }
    # Output lives outside the candidate so a proposer cannot edit its own recommendation file.
    with tempfile.TemporaryDirectory(prefix="thermo-proposer-") as directory:
        message_path = Path(directory) / "recommendation.txt"
        argv = [
            "codex",
            "exec",
            "-C",
            str(worktree),
            "-s",
            "workspace-write",
            "--output-last-message",
            str(message_path),
            "--ignore-user-config",
            "--strict-config",
            "-c",
            "sandbox_workspace_write.network_access=false",
            "-c",
            "sandbox_workspace_write.writable_roots=[]",
            prompt,
        ]
        try:
            completed = _bounded_run(
                argv,
                cwd=worktree,
                env=allowed_env,
                timeout=seconds,
                capture_output=True,
                shell=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("proposer unavailable: Codex CLI is missing") from error
        except subprocess.TimeoutExpired as error:
            detail = _safe_stderr_tail(error.stderr)
            message = "proposer timed out"
            raise RuntimeError(f"{message}: {detail}" if detail else message) from error
        if _head(worktree) != baseline.lower():
            raise ValueError("baseline mismatch after proposal generation")
        if completed.returncode != 0 or not message_path.is_file():
            raise RuntimeError(_failure_message(completed.returncode, completed.stderr))
        recommendation = read_recommendation(message_path)
        if not recommendation.strip():
            raise RuntimeError(_failure_message(completed.returncode, completed.stderr))
        return recommendation
