"""Bounded three-site comparisons and append-only held-out role reservations."""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from thermo_lab.improvement_harness.plan import Plan
from thermo_lab.improvement_harness.workspace import _git, _head, is_allowed_path, parse_status_z

_HOOK = "src/thermo_lab/research_candidates/three_site.py"
_OUTPUT_BYTES = 65536


def claim_heldout(root: Path, plan_digest: str, role: str, candidate_id: str) -> None:
    """Reserve before execution; a failed run also consumes its held-out role."""
    if re.fullmatch(r"sha256:[0-9a-f]{64}", plan_digest) is None:
        raise ValueError("invalid plan digest")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", role) is None:
        raise ValueError("invalid held-out role")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be nonempty")
    directory = Path(root)
    for part in ("heldout", plan_digest.removeprefix("sha256:")):
        if directory.is_symlink():
            raise ValueError("held-out directory cannot be a symlink")
        directory /= part
        if directory.is_symlink():
            raise ValueError("held-out directory cannot be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
    try:
        with (directory / f"{role}.json").open("x", encoding="utf-8") as file:
            json.dump({"candidate_id": candidate_id, "role": role}, file)
    except FileExistsError as error:
        raise ValueError("held-out role already used under this plan") from error


def _check_sources(plan: Plan, baseline: Path, candidate: Path) -> None:
    if _head(baseline) != plan.baseline_commit.lower():
        raise ValueError("baseline mismatch")
    if _git(baseline, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("baseline worktree is dirty")
    if _head(candidate) != plan.baseline_commit.lower():
        raise ValueError("candidate baseline mismatch")
    paths = parse_status_z(
        _git(candidate, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    )
    if any(path != _HOOK or not is_allowed_path(candidate, path, (_HOOK,)) for path in paths):
        raise ValueError("candidate changed protected files outside allowed paths")
    # Ignored source and sourceless bytecode can shadow tracked modules even when
    # Git status is clean. Only ordinary caches redirected by pycache_prefix are safe.
    for worktree in (baseline, candidate):
        ignored = _git(
            worktree, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--", "src"
        )
        for raw_path in ignored.split(b"\0"):
            path = Path(os.fsdecode(raw_path))
            redirected_cache = path.parent.name == "__pycache__" and re.fullmatch(
                rf".+\.{re.escape(sys.implementation.cache_tag)}(?:\.opt-[12])?\.pyc",
                path.name,
            )
            if path.suffix in {".py", ".pyc"} and not redirected_cache:
                raise ValueError("protected source includes ignored Python files or bytecode")


def _run_module(worktree: Path, module: str, payload: str, deadline: float) -> dict:
    """Select worktree source explicitly, excluding caller Python import settings."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(module, 0)
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONPATH=str(worktree / "src"), JAX_PLATFORMS="cpu")
    with (
        tempfile.TemporaryDirectory(prefix="thermo-fixture-cache-") as cache,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        # -B suppresses writes but still reads existing .pyc files. A fresh cache
        # prefix also prevents ignored/stale bytecode from overriding pinned source.
        argv = [sys.executable, "-B", "-P", "-s", "-X", f"pycache_prefix={cache}", "-m", module]
        process = subprocess.run(
            argv,
            cwd=worktree,
            env=env,
            input=payload.encode(),
            stdout=stdout,
            stderr=stderr,
            timeout=min(remaining, 300),
            check=False,
        )
        stdout.seek(0)
        output = stdout.read(_OUTPUT_BYTES + 1)
        if process.returncode:
            stderr.seek(0, os.SEEK_END)
            stderr.seek(max(0, stderr.tell() - 4096))
            message = stderr.read().decode(errors="replace")
            raise ValueError(f"fixture subprocess failed ({process.returncode}): {message}")
    if len(output) > _OUTPUT_BYTES:
        raise ValueError("fixture JSON output exceeds limit")
    try:
        result = json.loads(output)
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid fixture JSON output") from error
    if not isinstance(result, dict):
        raise ValueError("fixture JSON output must be an object")
    return result


def evaluate_three_site(
    plan: Plan, baseline_worktree: Path, candidate_worktree: Path, *, seconds: float | None = None
) -> dict:
    """Evaluate candidate parameters through the pinned baseline exact reference.

    This is a development fixture comparison. Callers must reserve any declared
    held-out role with ``claim_heldout`` before invoking a held-out evaluation.
    Repository checks remain a separate verification dimension in the CLI runner.
    """
    if (
        plan.track != "research"
        or plan.allowed_paths != (_HOOK,)
        or plan.primary_metric != "exact_objective_delta"
        or plan.direction != "lower"
        or plan.threshold > 0
    ):
        raise ValueError("plan does not match the bounded research preset")
    if seconds is not None and seconds <= 0:
        raise TimeoutError("plan wall time exhausted")
    baseline = Path(baseline_worktree).resolve()
    candidate = Path(candidate_worktree).resolve()
    _check_sources(plan, baseline, candidate)
    deadline = time.monotonic() + min(
        plan.wall_seconds, seconds if seconds is not None else plan.wall_seconds, 1800
    )
    parameters = _run_module(
        candidate, "thermo_lab.improvement_harness.fixture_candidate", "", deadline
    )
    # A hook may write files while running; check again before trusting baseline imports.
    _check_sources(plan, baseline, candidate)
    result = _run_module(
        baseline,
        "thermo_lab.improvement_harness.fixture_reference",
        json.dumps(parameters, allow_nan=False),
        deadline,
    )
    _check_sources(plan, baseline, candidate)
    delta = result["delta"]
    outcome = "improved" if delta < plan.threshold else "regressed" if delta > 0 else "inconclusive"
    return {
        **result,
        "parameters": parameters["parameters"],
        "baseline_commit": plan.baseline_commit,
        "primary_metric": plan.primary_metric,
        "scope": "three-site fixture only",
        "execution": "complete",
        "verification": "passed",
        "research_outcome": outcome,
    }
