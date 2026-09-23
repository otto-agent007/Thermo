"""Isolated candidate worktrees and checked Git patch intake."""

import os
import re
import subprocess
import uuid
from pathlib import Path


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], shell=False, check=True, capture_output=True
    ).stdout


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").decode("ascii").strip()


def create_candidate_worktree(repo: Path, baseline: str, candidate_id: str) -> Path:
    """Create a detached candidate only when the source is the frozen baseline."""
    repo = Path(repo).resolve()
    if re.fullmatch(r"[0-9a-fA-F]{40}", baseline) is None or _head(repo) != baseline.lower():
        raise ValueError("baseline mismatch")
    baseline = baseline.lower()
    if _git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude).worktrees/",
    ):
        raise ValueError("baseline worktree is dirty")
    try:
        parsed = uuid.UUID(candidate_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("candidate ID must be a canonical UUID") from error
    if str(parsed) != candidate_id:
        raise ValueError("candidate ID must be a canonical UUID")
    worktree = repo / ".worktrees" / f"harness-{candidate_id}"
    if worktree.parent.is_symlink():
        raise ValueError("worktree directory cannot be a symlink")
    if worktree.exists() or worktree.is_symlink():
        raise ValueError("candidate worktree already exists")
    worktree.parent.mkdir(exist_ok=True)
    _git(repo, "worktree", "add", "--detach", str(worktree), baseline)
    if _head(worktree) != baseline:
        raise ValueError("baseline mismatch after worktree creation")
    return worktree


def parse_status_z(raw: bytes) -> list[str]:
    """Parse porcelain v1 -z, including both sides of rename/copy records."""
    if not raw:
        return []
    records = raw.split(b"\0")
    if records[-1] != b"":
        raise ValueError("incomplete Git status record")
    paths: list[str] = []
    index = 0
    while index < len(records) - 1:
        record = records[index]
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError("invalid Git status record")
        paths.append(os.fsdecode(record[3:]))
        if b"R" in record[:2] or b"C" in record[:2]:
            index += 1
            if index >= len(records) - 1:
                raise ValueError("incomplete Git rename/copy record")
            paths.append(os.fsdecode(records[index]))
        index += 1
    return paths


def is_allowed_path(worktree: Path, path: str, allowed_paths: tuple[str, ...]) -> bool:
    """Require a relative, non-symlink path within an allowed prefix."""
    if not path or path.startswith("/") or "\\" in path or "\0" in path:
        return False
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    if not any(
        path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")
        for prefix in allowed_paths
    ):
        return False
    root = Path(worktree)
    if root.is_symlink():
        return False
    try:
        resolved_root = root.resolve(strict=True)
        candidate = root
        for part in parts:
            candidate /= part
            if candidate.is_symlink():
                return False
        candidate.resolve(strict=False).relative_to(resolved_root)
    except (OSError, ValueError):
        return False
    return True


def _was_symlink(worktree: Path, path: str) -> bool:
    """Catch removed or replaced symlinks that no longer exist on disk."""
    return any(
        record.startswith(b"120000 ")
        for command in (("ls-files", "-s", "-z"), ("ls-tree", "-z", "HEAD"))
        for record in _git(worktree, *command, "--", path).split(b"\0")
    )


def export_checked_patch(worktree: Path, allowed_paths: tuple[str, ...]) -> bytes:
    """Stage only checked changed paths, then export a nonempty binary patch."""
    worktree = Path(worktree)
    status = _git(worktree, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    paths = parse_status_z(status)
    if any(
        not is_allowed_path(worktree, path, allowed_paths) or _was_symlink(worktree, path)
        for path in paths
    ):
        raise ValueError("outside allowed paths")
    if paths:
        _git(worktree, "add", "--", *paths)
    staged = parse_status_z(
        _git(worktree, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    )
    if any(
        not is_allowed_path(worktree, path, allowed_paths) or _was_symlink(worktree, path)
        for path in staged
    ):
        raise ValueError("outside allowed paths")
    patch = _git(worktree, "diff", "--cached", "--binary", "--no-ext-diff")
    if not patch:
        raise ValueError("no diff to export")
    return patch


def import_manual_patch(worktree: Path, patch: Path) -> None:
    """Check patch applicability before mutating the candidate worktree."""
    worktree = Path(worktree)
    patch = Path(patch).resolve(strict=True)
    _git(worktree, "apply", "--check", str(patch))
    _git(worktree, "apply", str(patch))
