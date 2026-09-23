"""Isolated candidate worktrees and checked Git patch intake."""

import os
import re
import subprocess
import tempfile
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


def _ignored_paths(worktree: Path) -> list[str]:
    raw = _git(worktree, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
    return [os.fsdecode(path) for path in raw.split(b"\0") if path]


def export_checked_patch(worktree: Path, allowed_paths: tuple[str, ...]) -> bytes:
    """Stage only checked changed paths, then export a nonempty binary patch."""
    worktree = Path(worktree)
    status = _git(worktree, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    paths = parse_status_z(status)
    all_paths = paths + _ignored_paths(worktree)
    if any(
        not is_allowed_path(worktree, path, allowed_paths) or _was_symlink(worktree, path)
        for path in all_paths
    ):
        raise ValueError("outside allowed paths")
    if all_paths:
        _git(worktree, "add", "-f", "--", *all_paths)
    staged = parse_status_z(
        _git(worktree, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    ) + _ignored_paths(worktree)
    if any(
        not is_allowed_path(worktree, path, allowed_paths) or _was_symlink(worktree, path)
        for path in staged
    ):
        raise ValueError("outside allowed paths")
    patch = _git(worktree, "diff", "--cached", "--binary", "--no-ext-diff")
    if not patch:
        raise ValueError("no diff to export")
    return patch


def _manual_patch_paths(patch_bytes: bytes, numstat: bytes) -> list[str]:
    """Accept ordinary Git diffs and reject ambiguous rename/copy headers."""
    header_paths: list[str] = []
    for line in patch_bytes.splitlines():
        if not line.startswith(b"diff --git "):
            continue
        match = re.fullmatch(rb"diff --git a/([^\s]+) b/([^\s]+)", line)
        if match is None or match.group(1) != match.group(2):
            raise ValueError("outside allowed paths: ambiguous patch path")
        header_paths.append(os.fsdecode(match.group(1)))
    if re.search(rb"(?m)^(?:new file mode|new mode) 120000$", patch_bytes):
        raise ValueError("outside allowed paths: symlink patch")
    records = numstat.split(b"\0")
    if not numstat or records[-1] != b"":
        raise ValueError("invalid patch path list")
    stat_paths: list[str] = []
    for record in records[:-1]:
        fields = record.split(b"\t", 2)
        if len(fields) != 3 or not fields[2]:
            raise ValueError("outside allowed paths: ambiguous patch path")
        stat_paths.append(os.fsdecode(fields[2]))
    if not header_paths or set(header_paths) != set(stat_paths):
        raise ValueError("outside allowed paths: patch path mismatch")
    return header_paths


def import_manual_patch(worktree: Path, patch: Path, allowed_paths: tuple[str, ...]) -> None:
    """Check applicability and every patch path before mutating the worktree."""
    worktree = Path(worktree)
    patch_bytes = Path(patch).read_bytes()
    with tempfile.TemporaryDirectory(prefix="thermo-manual-patch-") as directory:
        checked_patch = Path(directory) / "candidate.patch"
        checked_patch.write_bytes(patch_bytes)
        _git(worktree, "apply", "--check", str(checked_patch))
        numstat = _git(worktree, "apply", "--numstat", "-z", str(checked_patch))
        paths = _manual_patch_paths(patch_bytes, numstat)
        if any(
            not is_allowed_path(worktree, path, allowed_paths) or _was_symlink(worktree, path)
            for path in paths
        ):
            raise ValueError("outside allowed paths")
        _git(worktree, "apply", str(checked_patch))
