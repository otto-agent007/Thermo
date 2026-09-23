"""Bounded local Codex proposer adapter."""

import os
import subprocess
import tempfile
from pathlib import Path


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
            completed = subprocess.run(
                argv,
                cwd=worktree,
                env=allowed_env,
                timeout=seconds,
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("proposer unavailable: Codex CLI is missing") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("proposer timed out") from error
        if _head(worktree) != baseline.lower():
            raise ValueError("baseline mismatch after proposal generation")
        if completed.returncode != 0 or not message_path.is_file():
            raise RuntimeError("proposer did not produce a recommendation")
        recommendation = message_path.read_text(encoding="utf-8")
        if not recommendation.strip():
            raise RuntimeError("proposer did not produce a recommendation")
        return recommendation
