"""Optional bounded GitHub snapshot transport; never enables a worker or accepts evidence.

Run this module only after an operator configures a dedicated clean evidence
checkout and authorizes its exact evidence/<job_id> branch. A successful push
does not establish that ChatGPT Work can retrieve the immutable commit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.research_cycle import worker
from thermo_lab.research_cycle.contracts import PHASES, Request

MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_FILES = 104
GIT_TIMEOUT = 120
ORIGINS = frozenset(
    {
        "https://github.com/otto-agent007/Thermo.git",
        "https://github.com/otto-agent007/Thermo",
        "git@github.com:otto-agent007/Thermo.git",
        "ssh://git@github.com/otto-agent007/Thermo.git",
    }
)


def _safe_path(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink paths are not accepted")
    return path


def _member(root: Path, name: str) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
        raise ValueError("snapshot path escapes the job")
    result = _safe_path(root / relative)
    if not result.is_relative_to(root):
        raise ValueError("snapshot path escapes the job")
    return result


def _git(repo: Path, *arguments: str, local_transport=False, binary=False) -> str | bytes:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"})
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgSign=false",
            "-c",
            "push.followTags=false",
            "-c",
            "remote.origin.mirror=false",
            "-c",
            f"protocol.file.allow={'always' if local_transport else 'never'}",
            *arguments,
        ],
        text=not binary,
        capture_output=True,
        timeout=GIT_TIMEOUT,
        env=environment,
        check=False,
    )
    if result.returncode:
        # Never reset, clean or erase an unpublished local commit after failure.
        raise ValueError(f"git {arguments[0]} failed; checkout retained for explicit recovery")
    return result.stdout if binary else result.stdout.strip()


def _checkout(repo: Path, branch: str):
    if not (repo / ".git").is_dir() or (repo / ".git").is_symlink():
        raise ValueError("evidence checkout must be a dedicated ordinary Git clone")
    if Path(_git(repo, "rev-parse", "--show-toplevel")) != repo:
        raise ValueError("evidence path must be the checkout root")
    if _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD") != branch:
        raise ValueError("evidence checkout is on the wrong branch")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("evidence checkout must be clean; existing changes are retained")
    for arguments in (
        ("remote", "get-url", "--all", "origin"),
        ("remote", "get-url", "--push", "--all", "origin"),
    ):
        if _git(repo, *arguments) not in ORIGINS:
            raise ValueError("origin must be exactly the authorized Thermo GitHub repository")


def _history(repo: Path, branch: str, request: Request):
    """Do not smuggle unrelated local commits through an otherwise clean clone."""
    _git(repo, "merge-base", "--is-ancestor", request.source_sha, "HEAD")
    known_remote = _git(
        repo, "for-each-ref", "--format=%(objectname)", f"refs/remotes/origin/{branch}"
    )
    base = known_remote or request.source_sha
    _git(repo, "merge-base", "--is-ancestor", base, "HEAD")
    revision_range = f"{base}..HEAD"
    if _git(repo, "rev-list", "--min-parents=2", revision_range):
        raise ValueError("local evidence history must not contain merge commits")
    commits = _git(repo, "rev-list", revision_range).splitlines()
    if len(commits) > 100:
        raise ValueError("too many unpublished commits; explicit history review required")
    prefix = f"cycles/{request.job_id}/".encode()
    for commit in commits:
        paths = _git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "-z", commit, binary=True
        ).split(b"\0")
        if any(path and not path.startswith(prefix) for path in paths):
            raise ValueError("local commits touch paths outside the designated cycle")


def _refresh(repo: Path, branch: str, transport: str, *, local_transport: bool):
    remote_ref = f"refs/heads/{branch}"
    listing = _git(
        repo, "ls-remote", "--heads", "--", transport, remote_ref, local_transport=local_transport
    )
    if not listing:
        return
    if not re.fullmatch(rf"[a-f0-9]{{40}}\s+{re.escape(remote_ref)}", listing):
        raise ValueError("unexpected remote branch identity")
    tracking = f"refs/remotes/origin/{branch}"
    _git(
        repo,
        "fetch",
        "--no-tags",
        "--no-recurse-submodules",
        "--",
        transport,
        f"{remote_ref}:{tracking}",
        local_transport=local_transport,
    )
    remote = _git(repo, "rev-parse", tracking)
    head = _git(repo, "rev-parse", "HEAD")
    common = _git(repo, "merge-base", head, remote)
    if common == head:
        # Fast-forward to an existing remote commit, without creating a merge
        # commit, resolving conflicts or rewriting local history.
        _git(repo, "merge", "--ff-only", "--no-edit", "--no-stat", remote)
    elif common != remote:
        raise ValueError("local and remote evidence history diverged; explicit recovery required")


@contextmanager
def _publisher_lock(repo: Path):
    descriptor = os.open(
        repo / ".git/thermo-publisher.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another publisher owns this evidence checkout") from exc
        yield
    finally:
        os.close(descriptor)


def _read_files(job: Path, names: set[str]) -> dict[str, bytes]:
    if len(names) > MAX_FILES:
        raise ValueError("snapshot file count exceeds its bound")
    files = {}
    total = 0
    for name in sorted(names):
        path = _member(job, name)
        if not path.is_file():
            raise ValueError(f"missing snapshot file: {name}")
        size = path.stat().st_size
        if total + size > MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot size exceeds its bound")
        with path.open("rb") as stream:
            content = stream.read(MAX_SNAPSHOT_BYTES - total + 1)
        total += len(content)
        if total > MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot size exceeds its bound")
        files[name] = content
    return files


def _check_observation(files: dict[str, bytes], job_id: str) -> Request:
    request = Request.from_dict(json.loads(files["request.json"]))
    status = json.loads(files["status.json"])
    if (
        request.job_id != job_id
        or not isinstance(status, dict)
        or status.get("job_id") != job_id
        or status.get("source_sha") != request.source_sha
        or status.get("request_digest") != request.digest
        or status.get("phase") not in PHASES
    ):
        raise ValueError("status and request identities differ")
    return request


def _snapshot(job: Path) -> tuple[dict[str, bytes], str]:
    if (job / "artifact_manifest.json").exists():
        with worker.worker_lock(job.parent):
            manifest = worker.verify_evidence(job)
            names = set(manifest["files"]) | {
                "request.json",
                "status.json",
                "journal.json",
                "artifact_manifest.json",
            }
            return _read_files(job, names), "evidence"
    # Request is immutable and status is atomically replaced. This bounded
    # observation deliberately does not contend with the running worker.
    return _read_files(job, {"request.json", "status.json"}), "observation"


def publish_snapshot(state_root, job_id, evidence_repo, branch, *, _test_transport=None) -> dict:
    """Publish one observation/evidence snapshot, without changing worker state.

    The private transport argument is only for isolated bare-repository tests;
    it is deliberately absent from the public module CLI.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", job_id):
        raise ValueError("invalid job_id")
    if branch != f"evidence/{job_id}":
        raise ValueError("only the exact evidence/<job_id> branch may be published")
    state, repo = _safe_path(Path(state_root)), _safe_path(Path(evidence_repo))
    if state.is_relative_to(repo) or repo.is_relative_to(state):
        raise ValueError("evidence checkout and worker state must be separate")
    job = _safe_path(state / job_id)
    _checkout(repo, branch)
    with _publisher_lock(repo):
        _checkout(repo, branch)
        files, kind = _snapshot(job)
        request = _check_observation(files, job_id)
        _history(repo, branch, request)
        transport = (
            str(Path(_test_transport).resolve()) if _test_transport is not None else "origin"
        )
        _refresh(repo, branch, transport, local_transport=_test_transport is not None)
        _checkout(repo, branch)
        destination = _safe_path(repo / "cycles" / job_id)
        if (destination / "request.json").exists():
            previous = Request.from_dict(worker.read_json(destination / "request.json"))
            if previous.digest != request.digest:
                raise ValueError("published job already has different immutable inputs")
        if kind == "observation" and (destination / "artifact_manifest.json").exists():
            raise ValueError("refusing to replace completed evidence with an observation")
        # Stage a complete snapshot first; verify the copied bytes, not merely
        # the source files. A Git commit makes the external snapshot atomic.
        with tempfile.TemporaryDirectory(prefix="thermo-snapshot-", dir=repo / ".git") as temporary:
            staged = Path(temporary)
            for name, content in files.items():
                target = _member(staged, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            manifest = worker.verify_evidence(staged) if kind == "evidence" else None
            for name in files:
                _member(destination, name)
            for name, content in files.items():
                target = _member(destination, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            if kind == "evidence":
                worker.verify_evidence(destination)
        relative = f"cycles/{job_id}"
        _git(repo, "add", "--", relative)
        changed = bool(_git(repo, "diff", "--cached", "--name-only", "--", relative))
        if changed:
            _git(repo, "commit", "-m", f"Record {job_id} {kind} snapshot", "--", relative)
        commit = _git(repo, "rev-parse", "HEAD")
        for name, content in files.items():
            if _git(repo, "show", f"{commit}:{relative}/{name}", binary=True) != content:
                raise ValueError(
                    "Git blob differs from verified snapshot; checkout retained for recovery"
                )
        _git(
            repo,
            "push",
            "--no-follow-tags",
            "--recurse-submodules=no",
            "--",
            transport,
            f"HEAD:refs/heads/{branch}",
            local_transport=_test_transport is not None,
        )
        return {
            "schema_version": 1,
            "job_id": job_id,
            "kind": kind,
            "changed": changed,
            "branch": branch,
            "commit_sha": commit,
            "source_sha": request.source_sha,
            "request_digest": request.digest,
            "evidence_digest": canonical_sha256(manifest) if manifest else None,
            "connected_access": "not_verified",
            "uri": f"https://github.com/otto-agent007/Thermo/tree/{commit}/{relative}",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--evidence-repo", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args(argv)
    try:
        print(
            canonical_json(
                publish_snapshot(args.state_root, args.job_id, args.evidence_repo, args.branch)
            )
        )
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"research-cycle-publisher: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
