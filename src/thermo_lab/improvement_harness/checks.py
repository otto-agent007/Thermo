"""Fixed, versioned checks for bounded candidate and baseline evaluation."""

import hashlib
import os
import re
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from thermo_lab.improvement_harness import sandbox
from thermo_lab.improvement_harness.plan import Plan

CATALOG_VERSION = 2
_LOG_BYTES = 65536
_TAIL_CHARACTERS = 4096
_COMMAND_CAP_SECONDS = 300

# The dependency step installs only protected, lockfile-pinned packages and needs
# the network. Every later step can execute candidate code, so it runs in the
# check sandbox with no network, a read-only host and a hidden $HOME.
_DEPENDENCY_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TERM")

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
            "candidate_hook",
            ("uv", "run", "pytest", "tests/unit/test_research_candidate_hook.py", "-q"),
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

    catalog_version: Literal[1, 2]
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


def _missing_dependency(tool: str, stderr: bytes) -> bool:
    """Recognize infrastructure errors without masking a bad pin or install script."""
    message = stderr.decode("utf-8", errors="replace").lower()
    if tool == "npm":
        codes = re.findall(r"(?m)^npm\s+(?:err!|error)\s+code\s+([a-z_0-9]+)\b", message)
        return bool(codes) and all(code in {"enotcached", "eai_again"} for code in codes)
    if tool == "uv":
        if (
            re.search(r"\b404\b", message)
            or "lockfile needs to be updated" in message
            or "failed to build" in message
            or "build backend" in message
        ):
            return False
        if "no cached distribution available" in message:
            return True
        has_network_cause = any(
            cause in message
            for cause in (
                "dns error",
                "failed to lookup address",
                "temporary failure in name resolution",
                "network is unreachable",
            )
        )
        has_fetch_context = any(
            context in message for context in ("failed to download", "failed to fetch")
        )
        return has_network_cause and has_fetch_context
    return False


def _append_tail(buffer: bytearray, chunk: bytes) -> None:
    """Retain only the newest bytes while continuing to drain the pipe."""
    if len(chunk) >= _LOG_BYTES:
        buffer[:] = chunk[-_LOG_BYTES:]
    else:
        buffer.extend(chunk)
        if len(buffer) > _LOG_BYTES:
            del buffer[: len(buffer) - _LOG_BYTES]


def _bounded_run(
    argv: list[str],
    *,
    cwd: Path,
    timeout: float,
    shell: bool,
    capture_output: bool,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Drain both process pipes without holding more than two log caps in memory."""
    if shell or not capture_output:
        raise ValueError("bounded runner requires shell=False and captured output")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout = bytearray()
    stderr = bytearray()
    with selectors.DefaultSelector() as selector:
        assert process.stdout is not None and process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, stdout)
        selector.register(process.stderr, selectors.EVENT_READ, stderr)
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                ready = selector.select(remaining)
                if not ready:
                    raise subprocess.TimeoutExpired(argv, timeout)
                for key, _ in ready:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        _append_tail(key.data, chunk)
                    else:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            returncode = process.wait(timeout=remaining)
            return subprocess.CompletedProcess(argv, returncode, bytes(stdout), bytes(stderr))
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise subprocess.TimeoutExpired(
                argv, timeout, output=bytes(stdout), stderr=bytes(stderr)
            ) from error
        finally:
            process.stdout.close()
            process.stderr.close()
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def run_checks(
    track: Literal["research", "dashboard"],
    cwd: Path,
    seconds: int,
    *,
    record_dir: Path,
    runner=_bounded_run,
) -> list[CheckResult]:
    """Run only catalogued argv into one explicit observation directory."""
    if track not in CATALOG:
        raise ValueError(f"unknown track: {track}")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    cwd = Path(cwd)
    record_dir = Path(record_dir)
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
                if name == "dependencies":
                    command, env = (
                        list(argv),
                        {key: os.environ[key] for key in _DEPENDENCY_ENV if key in os.environ},
                    )
                else:
                    command = sandbox.check_command(list(argv), worktree=cwd, cwd=command_cwd)
                    env = None
                completed = runner(
                    command,
                    cwd=command_cwd,
                    timeout=remaining_seconds,
                    shell=False,
                    capture_output=True,
                    env=env,
                )
                exit_status = completed.returncode
                stdout, stderr = _bytes(completed.stdout), _bytes(completed.stderr)
                if exit_status == 0:
                    execution, verification = "complete", "passed"
                elif name != "dependencies" and stderr.startswith(b"bwrap:"):
                    # The sandbox itself failed to start; never blame the candidate.
                    execution, verification = "unavailable", "inconclusive"
                elif name == "browser" and _missing_browser(stderr, stdout):
                    execution, verification = "unavailable", "inconclusive"
                elif name == "dependencies" and _missing_dependency(argv[0], stderr):
                    execution, verification = "unavailable", "inconclusive"
                else:
                    execution, verification = "failed", "failed"
            except subprocess.TimeoutExpired as error:
                stdout, stderr = _bytes(error.stdout), _bytes(error.stderr)
                execution, verification = "timed_out", "inconclusive"
            except (FileNotFoundError, sandbox.SandboxUnavailable) as error:
                stderr = str(error).encode("utf-8", errors="replace")
                execution, verification = "unavailable", "inconclusive"
        duration = time.monotonic() - started
        payload = b"stdout:\n" + stdout + b"\nstderr:\n" + stderr
        if len(payload) > _LOG_BYTES:
            marker = b"[earlier output truncated]\n"
            payload = marker + payload[-(_LOG_BYTES - len(marker)) :]
        log_path = f"checks/{name}.log"
        with (record_dir / log_path).open("xb") as log_file:
            log_file.write(payload)
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
