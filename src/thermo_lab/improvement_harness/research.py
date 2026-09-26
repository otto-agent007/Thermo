"""Bounded three-site comparisons and append-only held-out role reservations."""

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from thermo_lab.improvement_harness import sandbox
from thermo_lab.improvement_harness.limits import MAX_PATCH_BYTES, read_bounded_regular_file
from thermo_lab.improvement_harness.plan import Plan
from thermo_lab.improvement_harness.workspace import _git, _head, is_allowed_path, parse_status_z

_HOOK = "src/thermo_lab/research_candidates/three_site.py"
_DRIVER = "src/thermo_lab/improvement_harness/fixture_candidate.py"
_OUTPUT_BYTES = 65536


def claim_heldout(root: Path, plan_digest: str, role: str, candidate_id: str) -> None:
    """Reserve before execution; a failed run also consumes its held-out role.

    No current preset has held-out data: the research preset rejects plans
    that declare a role until an evaluator with genuinely held-out inputs exists.
    """
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


def _run(argv: list[str], cwd: Path, env: dict, payload: str, deadline: float) -> dict:
    """Run one bounded JSON subprocess and kill its whole session afterwards."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(argv, 0)
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            process.communicate(input=payload.encode(), timeout=min(remaining, 300))
        finally:
            # The hook may spawn children that outlive its direct process. Kill
            # the whole isolated session before trusting its output or worktree.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait()
        stdout.seek(0)
        output = stdout.read(_OUTPUT_BYTES + 1)
        if process.returncode:
            stderr.seek(0, os.SEEK_END)
            stderr.seek(max(0, stderr.tell() - 4096))
            message = stderr.read().decode(errors="replace")
            if message.startswith("bwrap:"):
                raise sandbox.SandboxUnavailable(f"sandbox unavailable: {message.strip()}")
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


def _run_module(worktree: Path, module: str, payload: str, deadline: float) -> dict:
    """Run trusted baseline source, excluding caller Python import settings."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONPATH=str(worktree / "src"), JAX_PLATFORMS="cpu")
    with tempfile.TemporaryDirectory(prefix="thermo-fixture-cache-") as cache:
        # -B suppresses writes but still reads existing .pyc files. A fresh cache
        # prefix also prevents ignored/stale bytecode from overriding pinned source.
        argv = [sys.executable, "-B", "-P", "-s", "-X", f"pycache_prefix={cache}", "-m", module]
        return _run(argv, worktree, env, payload, deadline)


def _stage_hook(baseline: Path, candidate: Path, stage: Path) -> None:
    """Copy only the candidate hook and the baseline's protected driver."""
    packages = ("thermo_lab", "thermo_lab/research_candidates", "thermo_lab/improvement_harness")
    for package in packages:
        (stage / package).mkdir(parents=True, exist_ok=True)
        (stage / package / "__init__.py").write_text("")
    for source, relative in ((candidate / _HOOK, _HOOK), (baseline / _DRIVER, _DRIVER)):
        payload = read_bounded_regular_file(source, MAX_PATCH_BYTES, "hook source")
        (stage / relative.removeprefix("src/")).write_bytes(payload)


def _run_hook(baseline: Path, candidate: Path, inputs: dict, deadline: float) -> dict:
    """Run the hook in the strict sandbox: standard library only, no network or repository."""
    with tempfile.TemporaryDirectory(prefix="thermo-hook-") as directory:
        stage = Path(directory)
        _stage_hook(baseline, candidate, stage)
        argv = sandbox.hook_command(stage, "thermo_lab.improvement_harness.fixture_candidate")
        return _run(argv, stage, {}, json.dumps(inputs, allow_nan=False), deadline)


def _checked_parameters(output: dict, cap: float) -> list[float]:
    values = output.get("parameters")
    if set(output) != {"parameters"} or not isinstance(values, list) or len(values) != 9:
        raise ValueError("hook must return exactly nine values")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("hook values must be finite numbers")
        if not math.isfinite(value):
            raise ValueError("hook values must be finite numbers")
        if abs(value) > cap:
            raise ValueError("hook values must stay inside the parameter cap")
    return values


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
    reference = "thermo_lab.improvement_harness.fixture_reference"
    inputs = _run_module(baseline, reference, json.dumps({"request": "inputs"}), deadline)
    first = _run_hook(baseline, candidate, inputs, deadline)
    if _run_hook(baseline, candidate, inputs, deadline) != first:
        raise ValueError("hook output must be deterministic")
    parameters = {"parameters": _checked_parameters(first, inputs["cap"])}
    _check_sources(plan, baseline, candidate)
    result = _run_module(baseline, reference, json.dumps(parameters, allow_nan=False), deadline)
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
