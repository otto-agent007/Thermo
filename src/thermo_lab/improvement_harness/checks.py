"""Fixed, versioned checks for bounded candidate and baseline evaluation."""

import hashlib
import re
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from thermo_lab.improvement_harness.plan import Plan

CATALOG_VERSION = 1
_LOG_BYTES = 65536
_TAIL_CHARACTERS = 4096
_COMMAND_CAP_SECONDS = 300

# name, argv, cwd relative to the supplied repository worktree
CATALOG: dict[str, tuple[tuple[str, tuple[str, ...], str], ...]] = {
    "dashboard": (
        ("dependencies", ("npm", "ci"), "dashboard"),
        ("unit", ("npm", "test"), "dashboard"),
        ("typecheck", ("npm", "run", "typecheck"), "dashboard"),
        ("build", ("npm", "run", "build"), "dashboard"),
        ("browser", ("npm", "run", "test:browser"), "dashboard"),
    ),
    "research": (
        ("dependencies", ("uv", "sync", "--frozen"), "."),
        (
            "exact_fixture",
            (
                "uv",
                "run",
                "pytest",
                "tests/unit/test_trajectory_reinforce_refinement.py::test_exact_objective_evaluation_ties_updated_parameters_across_both_occurrences",
                "-q",
            ),
            ".",
        ),
        (
            "focused_tests",
            ("uv", "run", "pytest", "tests/unit/test_trajectory_reinforce_refinement.py", "-q"),
            ".",
        ),
        ("ruff", ("uv", "run", "ruff", "check", "src/thermo_lab", "tests/unit"), "."),
    ),
}


class CheckResult(BaseModel):
    """One command observation, with only a bounded log tail in JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_version: Literal[1]
    name: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    exit_status: int | None
    duration_seconds: float = Field(ge=0, allow_inf_nan=False)
    log_sha256: str
    log_tail: str
    log_path: str
    execution: Literal["complete", "failed", "timed_out", "unavailable"]
    verification: Literal["passed", "failed", "inconclusive"]

    @field_validator("log_sha256")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ValueError("invalid log digest")
        return value

    @field_validator("log_tail")
    @classmethod
    def bounded_tail(cls, value: str) -> str:
        if len(value) > _TAIL_CHARACTERS:
            raise ValueError("log tail too long")
        return value

    @field_validator("log_path")
    @classmethod
    def safe_log_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or not value.startswith("checks/") or ".." in path.parts:
            raise ValueError("log path must stay under checks/")
        return value

    @model_validator(mode="after")
    def coherent_status(self) -> "CheckResult":
        if self.execution == "complete" and (
            self.exit_status != 0 or self.verification != "passed"
        ):
            raise ValueError("complete check must pass with exit status zero")
        if self.execution == "failed" and (
            self.exit_status is None or self.exit_status == 0 or self.verification != "failed"
        ):
            raise ValueError("failed check must have nonzero exit status")
        if self.execution in {"timed_out", "unavailable"} and self.verification != "inconclusive":
            raise ValueError("incomplete check must be inconclusive")
        return self


def _bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value if isinstance(value, bytes) else value.encode("utf-8", errors="replace")


def _missing_browser(stderr: bytes, stdout: bytes) -> bool:
    message = (stderr + stdout).decode("utf-8", errors="replace").lower()
    return "executable doesn't exist" in message and "playwright install" in message


def _missing_dependency(stderr: bytes, stdout: bytes) -> bool:
    message = (stderr + stdout).decode("utf-8", errors="replace").lower()
    return any(
        marker in message
        for marker in (
            "failed to download",
            "could not resolve",
            "cannot find module",
            "module not found",
            "enoent",
            "not found in the registry",
        )
    )


def run_checks(
    track: Literal["research", "dashboard"],
    cwd: Path,
    seconds: int,
    *,
    runner=subprocess.run,
    record_dir: Path | None = None,
) -> list[CheckResult]:
    """Run only catalogued argv, using one total deadline and local log files."""
    if track not in CATALOG:
        raise ValueError(f"unknown track: {track}")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    cwd = Path(cwd)
    record_dir = Path(record_dir) if record_dir is not None else cwd / "results" / "harness"
    checks_dir = record_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    if checks_dir.is_symlink():
        raise ValueError("checks directory cannot be a symlink")
    deadline = time.monotonic() + seconds
    results: list[CheckResult] = []
    for name, argv, relative_cwd in CATALOG[track]:
        command_cwd = cwd if relative_cwd == "." else cwd / relative_cwd
        remaining_seconds = min(_COMMAND_CAP_SECONDS, deadline - time.monotonic())
        started = time.monotonic()
        stdout = stderr = b""
        exit_status: int | None = None
        if remaining_seconds <= 0:
            execution, verification = "timed_out", "inconclusive"
        else:
            try:
                completed = runner(
                    list(argv),
                    cwd=command_cwd,
                    timeout=remaining_seconds,
                    shell=False,
                    capture_output=True,
                )
                exit_status = completed.returncode
                stdout, stderr = _bytes(completed.stdout), _bytes(completed.stderr)
                if exit_status == 0:
                    execution, verification = "complete", "passed"
                elif name == "browser" and _missing_browser(stderr, stdout):
                    execution, verification = "unavailable", "inconclusive"
                elif name == "dependencies" and _missing_dependency(stderr, stdout):
                    execution, verification = "unavailable", "inconclusive"
                else:
                    execution, verification = "failed", "failed"
            except subprocess.TimeoutExpired as error:
                stdout, stderr = _bytes(error.stdout), _bytes(error.stderr)
                execution, verification = "timed_out", "inconclusive"
            except FileNotFoundError as error:
                stderr = str(error).encode("utf-8", errors="replace")
                execution, verification = "unavailable", "inconclusive"
        duration = time.monotonic() - started
        payload = b"stdout:\n" + stdout + b"\nstderr:\n" + stderr
        if len(payload) > _LOG_BYTES:
            marker = b"[earlier output truncated]\n"
            payload = marker + payload[-(_LOG_BYTES - len(marker)) :]
        log_path = f"checks/{name}.log"
        (record_dir / log_path).write_bytes(payload)
        results.append(
            CheckResult(
                catalog_version=CATALOG_VERSION,
                name=name,
                argv=argv,
                exit_status=exit_status,
                duration_seconds=duration,
                log_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                log_tail=payload.decode("utf-8", errors="replace")[-_TAIL_CHARACTERS:],
                log_path=log_path,
                execution=execution,
                verification=verification,
            )
        )
        if name == "dependencies" and verification != "passed":
            break
    return results


def compare_baseline(
    plan: Plan, baseline: list[CheckResult], candidate: list[CheckResult]
) -> Literal["passed", "failed", "inconclusive"]:
    """Compare complete matching checks; a broken baseline cannot certify a candidate."""
    if plan.track not in CATALOG:
        raise ValueError("unknown track")
    expected = [(name, argv) for name, argv, _ in CATALOG[plan.track]]
    baseline_signature = [(item.name, item.argv) for item in baseline]
    candidate_signature = [(item.name, item.argv) for item in candidate]
    if baseline_signature != expected or candidate_signature != expected:
        return "inconclusive"
    if any(item.verification != "passed" for item in baseline):
        return "inconclusive"
    if any(item.verification == "inconclusive" for item in candidate):
        return "inconclusive"
    if any(item.verification == "failed" for item in candidate):
        return "failed"
    return "passed"
