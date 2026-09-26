"""Linux-only, bounded M5a execution with a durable, single-host work ledger.

The state directory is an operator-controlled durable directory outside the
checkout. Do not share it between hosts. Its advisory lock is global to the
worker and inherited by every command. Recorded process groups additionally
prevent recovery when any orphaned descendants could still be running.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.research_cycle.contracts import PHASES, PROTOCOL, Request

LOG_LIMIT = 1024 * 1024
ARTIFACT_LIMIT = 64 * 1024 * 1024
OUTPUT_FILES = (
    "study.json.gz",
    "generation-provenance.json",
    "provenance.json",
    "summary.md",
    "completion.json",
)
CAPS = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0, 10.0)


def read_json(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe file: {path.name}")
    if path.stat().st_size > ARTIFACT_LIMIT:
        raise ValueError(f"oversized file: {path.name}")
    try:
        return json.loads(
            path.read_text(), parse_constant=_invalid_json, object_pairs_hook=_unique_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path.name}") from exc


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _invalid_json(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def write_json(path: Path, value, *, immutable=False):
    """Persist atomically and fsync; immutable entries may never be replaced."""
    text = canonical_json(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable:
        # Link the complete fsynced temporary file atomically without overwrite.
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()
    else:
        if path.is_symlink():
            raise ValueError("refusing symlink state file")
        atomic_write_text(path, text)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe artifact: {path.name}")
    if path.stat().st_size > ARTIFACT_LIMIT:
        raise ValueError(f"oversized artifact: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def worker_lock(state_root: Path):
    state_root = Path(state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(state_root / ".worker.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("worker lock already owned; no duplicate execution allowed") from exc
        yield fd
    finally:
        # Do not LOCK_UN: children share the open-file description. close keeps
        # the advisory lock held until the final inherited descriptor closes.
        os.close(fd)


def process_identity(pid: int) -> dict:
    return {
        "pid": pid,
        "pgid": os.getpgid(pid),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "start_ticks": None,
    }


def _group_alive(owner: dict) -> bool:
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if owner["boot_id"] != current_boot:
        return False
    pgid = owner["pgid"]
    if type(pgid) is not int or pgid <= 1:
        raise ValueError("invalid process ownership")
    try:
        os.killpg(pgid, 0)
        return True  # Including PID reuse or unreaped members: fail closed.
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def execution_environment(repo: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": str(Path(repo) / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "UV_NO_CONFIG": "1",
    }


def _terminate_group(pgid: int):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    # A bounded grace period applies to the entire group, not just its leader.
    time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_process(
    argv,
    *,
    cwd: Path,
    environment: dict,
    timeout: float,
    stage_dir: Path,
    lock_fd: int,
    on_tick=None,
) -> dict:
    """Bound a command group; public job execution only supplies fixed argv."""
    if timeout <= 0:
        raise ValueError("execution budget exhausted")
    stage_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = {
        "argv": list(map(str, argv)),
        "timeout_seconds": timeout,
        "cwd": str(cwd),
        "deadline_monotonic": started + timeout,
    }
    write_json(stage_dir / "command.json", config, immutable=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-P",
            str(Path(__file__).with_name("supervisor.py")),
            str(stage_dir / "command.json"),
        ],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        pass_fds=(lock_fd,),
        start_new_session=True,
    )
    timed_out = False
    written = 0
    next_tick = started
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        with (stage_dir / "stdout.log").open("xb") as log:
            while process.poll() is None:
                now = time.monotonic()
                if now - started >= timeout + 1:
                    timed_out = True
                    _terminate_group(process.pid)
                    break
                for key, _ in selector.select(min(0.1, max(0, timeout + 1 - (now - started)))):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        retained = chunk[: max(0, LOG_LIMIT - written)]
                        log.write(retained)
                        written += len(retained)
                    else:
                        selector.unregister(key.fileobj)
                if on_tick is not None and now >= next_tick:
                    on_tick(now - started)
                    next_tick = now + 1
            # No command can leave an unbounded background process after exit.
            _terminate_group(process.pid)
            process.wait(timeout=5)
            for key, _ in selector.select(0):
                chunk = os.read(key.fileobj.fileno(), 65536)
                retained = chunk[: max(0, LOG_LIMIT - written)]
                log.write(retained)
                written += len(retained)
            log.flush()
            os.fsync(log.fileno())
    finally:
        if process.poll() is None:
            _terminate_group(process.pid)
            process.wait(timeout=5)
        process.stdout.close()
        selector.close()
    guardian = (
        read_json(stage_dir / "guardian-result.json")
        if (stage_dir / "guardian-result.json").exists()
        else {}
    )
    result = {
        "exit_status": guardian.get("exit_status", process.returncode),
        "timed_out": timed_out or guardian.get("timed_out", False),
        "elapsed_seconds": time.monotonic() - started,
        "log_bytes": written,
        "group_exited": guardian.get("group_exited", False),
    }
    write_json(stage_dir / "result.json", result, immutable=True)
    return result


def _git(repo: Path, *args) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "GIT_CONFIG_NOSYSTEM": "1"},
            timeout=10,
        ).strip()
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError("cannot authenticate repository") from exc


def validate_checkout(repo: Path, request: Request | None = None):
    repo = Path(repo).resolve()
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("repository must be its checkout root")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("a clean pinned checkout is required")
    sha = _git(repo, "rev-parse", "HEAD")
    if not re.fullmatch("[a-f0-9]{40}", sha):
        raise ValueError("source must be a full Git SHA")
    for name in (PROTOCOL, "uv.lock", "src/thermo_lab/meta_ebm_cap_baseline.py"):
        path = repo / name
        if not path.resolve().is_relative_to(repo) or path.is_symlink():
            raise ValueError("source paths may not escape checkout")
    if request:
        if sha != request.source_sha:
            raise ValueError("source SHA differs from reviewed request")
        if file_hash(repo / PROTOCOL) != request.protocol_sha256:
            raise ValueError("protocol digest differs from reviewed request")
        if file_hash(repo / "uv.lock") != request.lock_sha256:
            raise ValueError("lockfile digest differs from reviewed request")
    return sha


def identity_argv(repo: Path, destination: Path) -> list[str]:
    code = (
        "from pathlib import Path; from thermo_lab.meta_ebm_cap_baseline import study_request; "
        "from thermo_lab.hashing import canonical_json,canonical_sha256; "
        "import sys; Path(sys.argv[1]).write_text(canonical_json("
        "{'scientific_request_digest':canonical_sha256(study_request())}))"
    )
    return [str(repo / ".venv/bin/python"), "-P", "-c", code, str(destination)]


def _scientific_identity(repo: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="thermo-identity-") as temporary:
        root = Path(temporary)
        with worker_lock(root) as fd:
            result = run_process(
                identity_argv(repo, root / "identity.json"),
                cwd=repo,
                environment=execution_environment(repo),
                timeout=30,
                stage_dir=root / "identity",
                lock_fd=fd,
            )
        if result["exit_status"] != 0:
            raise ValueError("scientific identity check failed; prepare frozen dependencies first")
        return read_json(root / "identity.json")["scientific_request_digest"]


def prepare_request(repo, job_id, question, *, workers=2, budget_seconds=21600, max_attempts=2):
    repo = Path(repo).resolve()
    sha = validate_checkout(repo)
    value = {
        "schema_version": 1,
        "job_id": job_id,
        "source_sha": sha,
        "protocol_sha256": file_hash(repo / PROTOCOL),
        "lock_sha256": file_hash(repo / "uv.lock"),
        "scientific_request_digest": _scientific_identity(repo),
        "workers": workers,
        "budget_seconds": budget_seconds,
        "max_attempts": max_attempts,
        "question": question,
    }
    validate_checkout(repo)
    return Request.from_dict(value)


def _status(job: Path, journal: dict, phase: str, message: str) -> dict:
    request = Request.from_dict(read_json(job / "request.json"))
    if phase not in PHASES:
        raise ValueError("invalid phase")
    status = {
        "schema_version": 1,
        "job_id": request.job_id,
        "source_sha": request.source_sha,
        "request_digest": request.digest,
        "phase": phase,
        "attempts": len(journal["attempts"]),
        "elapsed_seconds": min(request.budget_seconds, journal["elapsed_seconds"]),
        "heartbeat_at": datetime.now(UTC).isoformat() if phase != "queued" else None,
        "evidence_digest": journal.get("evidence_digest"),
        "review_status": "pending",
        "message": message,
    }
    write_json(job / "status.json", status)
    return status


def initialize_job(state_root: Path, request: Request) -> Path:
    job = Path(state_root) / request.job_id
    if job.is_symlink():
        raise ValueError("unsafe job directory")
    job.mkdir(parents=True, exist_ok=True)
    request_path = job / "request.json"
    if request_path.exists():
        existing = Request.from_dict(read_json(request_path))
        if existing.digest != request.digest:
            raise ValueError("job ID already has immutable, different inputs")
    else:
        write_json(request_path, request.to_dict(), immutable=True)
    if not (job / "journal.json").exists():
        journal = {"schema_version": 1, "attempts": [], "elapsed_seconds": 0.0, "active": None}
        write_json(job / "journal.json", journal)
        _status(job, journal, "queued", "Reviewed request staged; execution has not started.")
    return job


def dependency_argv() -> list[str]:
    executable = shutil.which("uv")
    if executable is None:
        raise ValueError("uv is unavailable; frozen dependencies cannot be verified")
    return [executable, "sync", "--frozen", "--no-dev", "--no-config"]


def study_argv(repo: Path, output: Path, workers: int, replay_from: Path | None) -> list[str]:
    argv = [
        str(repo / ".venv/bin/python"),
        "-P",
        "-m",
        "thermo_lab.meta_ebm_cap_baseline",
        "--output-dir",
        str(output),
        "--workers",
        str(workers),
    ]
    if replay_from is not None:
        argv.extend(["--replay-from", str(replay_from)])
    return argv


def _read_archive(output: Path, scientific_digest: str) -> dict:
    path = output / "study.json.gz"
    file_hash(path)
    try:
        with gzip.open(path, "rb") as stream:
            raw = stream.read(ARTIFACT_LIMIT + 1)
        if len(raw) > ARTIFACT_LIMIT:
            raise ValueError("archive expands beyond its limit")
        record = json.loads(raw, parse_constant=_invalid_json)
        if record["request_digest"] != scientific_digest:
            raise ValueError("scientific request digest differs")
        if canonical_sha256(record["request"]) != scientific_digest:
            raise ValueError("scientific request hash mismatch")
        body = {key: record[key] for key in ("targets", "fits", "chains", "integrity")}
        if canonical_sha256(body) != record["result_digest"]:
            raise ValueError("result hash mismatch")
        expected_targets = {(r, s) for r in ("A", "B") for s in range(5)}
        if (
            len(record["targets"]) != 10
            or {(t["reading"], t["seed"]) for t in record["targets"]} != expected_targets
        ):
            raise ValueError("incomplete target grid")
        expected_chains = {
            (r, s, c, m)
            for r, s in expected_targets
            for c in CAPS
            for m in ("constructive", "variational")
        }
        if (
            len(record["chains"]) != 180
            or {(c["reading"], c["seed"], c["cap"], c["method"]) for c in record["chains"]}
            != expected_chains
        ):
            raise ValueError("incomplete chain grid")
        if record["integrity"] != {"passed": True, "failures": []}:
            raise ValueError("scientific integrity failed")
        return record
    except (OSError, EOFError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid or incomplete generation archive") from exc


def authenticate_generation(output: Path, scientific_digest: str) -> dict:
    record = _read_archive(output, scientific_digest)
    generation = read_json(output / "generation-provenance.json")
    if (
        generation.get("status") != "available"
        or generation.get("request_digest") != scientific_digest
        or generation.get("result_digest") != record["result_digest"]
        or generation.get("archive_sha256") != file_hash(output / "study.json.gz")
        or not isinstance(generation.get("runtime"), dict)
    ):
        raise ValueError("generation provenance is missing or mismatched")
    return record


def validate_output(output: Path, scientific_digest: str) -> dict:
    for name in OUTPUT_FILES:
        file_hash(output / name)
    record = authenticate_generation(output, scientific_digest)
    completion = read_json(output / "completion.json")
    expected = {
        "status": "meta_ebm_cap_baseline_complete",
        "targets": 10,
        "caps": 9,
        "methods": 2,
        "chains": 180,
        "samples": 0,
        "integrity": True,
        "replayed": True,
        "request_digest": scientific_digest,
        "result_digest": record["result_digest"],
        "provenance_digest": canonical_sha256(read_json(output / "provenance.json")),
    }
    if completion != expected or any(
        type(completion[k]) is not type(v) for k, v in expected.items()
    ):
        raise ValueError("completion is corrupt, incomplete or lacks mandatory full replay")
    provenance = read_json(output / "provenance.json")
    if provenance.get("generation") != read_json(output / "generation-provenance.json"):
        raise ValueError("original generation provenance differs")
    return completion


def _artifact_path(root: Path, name: str) -> Path:
    path = root / name
    if (
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("artifact path escapes destination")
    if path.is_symlink():
        raise ValueError("symlink artifacts are not accepted")
    return path


def verify_evidence(job: Path) -> dict:
    job = Path(job)
    manifest = read_json(job / "artifact_manifest.json")
    projection = read_json(
        job / ("journal.json" if (job / "journal.json").exists() else "status.json")
    )
    if canonical_sha256(manifest) != projection.get("evidence_digest"):
        raise ValueError("manifest digest differs from durable evidence identity")
    request = Request.from_dict(read_json(job / "request.json"))
    if (
        manifest.get("request_digest") != request.digest
        or manifest.get("source_sha") != request.source_sha
        or manifest.get("scientific_request_digest") != request.scientific_request_digest
    ):
        raise ValueError("manifest identity differs from reviewed request")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > 100:
        raise ValueError("missing manifest-bound artifacts")
    for name, digest in files.items():
        if file_hash(_artifact_path(job, name)) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    output = _artifact_path(job, manifest["output_dir"])
    required = {str((output / name).relative_to(job)) for name in OUTPUT_FILES}
    if not required.issubset(files) or "request.json" not in files:
        raise ValueError("manifest missing mandatory artifacts")
    validate_output(output, request.scientific_request_digest)
    if manifest.get("replay_coverage") != {"targets": 10, "chains": 180, "full": True}:
        raise ValueError("manifest lacks full replay coverage")
    return manifest


def _recover(job: Path, journal: dict, request: Request):
    active = journal["active"]
    if not active:
        return
    stage = _artifact_path(job, active["stage_dir"])
    if (stage / "owner.json").exists() and _group_alive(read_json(stage / "owner.json")):
        raise ValueError("prior process ownership remains live or uncertain")
    # Without an owner file, the bootstrap never executed study code; the free
    # global lock establishes that it cannot still be registering ownership.
    recovery_path = stage.parent / f"{stage.name}-recovery.json"
    if recovery_path.exists():
        recovery = read_json(recovery_path)
        if recovery.get("active") != active:
            raise ValueError("immutable recovery identity differs")
    else:
        charged = max(0.0, time.time() - active["started_at"])
        if active.get("boot_id"):
            current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if active["boot_id"] == current_boot:
                charged = max(charged, time.monotonic() - active["started_monotonic"])
            else:
                # A restart across a host reboot cannot prove the unobserved
                # active interval; consume the remaining allowance, fail closed.
                charged = request.budget_seconds - journal["elapsed_seconds"]
        recovery = {
            "active": active,
            "charged_seconds": charged,
            "elapsed_after": min(request.budget_seconds, journal["elapsed_seconds"] + charged),
            "reason": "interrupted; conservative wall-clock charge",
        }
        write_json(recovery_path, recovery, immutable=True)
    journal["elapsed_seconds"] = recovery["elapsed_after"]
    journal["active"] = None
    write_json(job / "journal.json", journal)


def _stage(job, journal, request, repo, lock_fd, stage, argv, phase):
    remaining = request.budget_seconds - journal["elapsed_seconds"]
    if remaining <= 0:
        raise ValueError("cumulative execution budget exhausted")
    journal["active"] = {
        "started_at": time.time(),
        "started_monotonic": time.monotonic(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "stage_dir": str(stage.relative_to(job)),
    }
    write_json(job / "journal.json", journal)
    baseline = journal["elapsed_seconds"]

    def heartbeat(elapsed):
        current = {**journal, "elapsed_seconds": baseline + elapsed}
        actual_phase = phase
        if phase == "running" and (stage.parent / "output/study.json.gz").exists():
            actual_phase = "verifying"
        _status(
            job, current, actual_phase, "Bounded CPU execution; scientific review remains pending."
        )

    result = run_process(
        argv,
        cwd=repo,
        environment=execution_environment(repo),
        timeout=remaining,
        stage_dir=stage,
        lock_fd=lock_fd,
        on_tick=heartbeat,
    )
    journal["elapsed_seconds"] = min(request.budget_seconds, baseline + result["elapsed_seconds"])
    journal["active"] = (
        None
        if result["group_exited"]
        else {
            "started_at": time.time(),
            "started_monotonic": time.monotonic(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "stage_dir": str(stage.relative_to(job)),
        }
    )
    write_json(job / "journal.json", journal)
    if not result["group_exited"]:
        raise ValueError("process group ownership remains uncertain after termination")
    return result


def _manifest(job, journal, request, output):
    files = {"request.json": file_hash(job / "request.json")}
    commands = []
    for attempt in journal["attempts"]:
        directory = job / attempt["directory"]
        for stage in sorted(directory.iterdir()):
            if stage.is_dir() and (stage / "command.json").exists():
                command = read_json(stage / "command.json")
                result = (
                    read_json(stage / "result.json") if (stage / "result.json").exists() else None
                )
                commands.append({"attempt_id": attempt["attempt_id"], **command, "result": result})
                for name in ("command.json", "result.json", "stdout.log", "owner.json"):
                    if (stage / name).exists():
                        files[str((stage / name).relative_to(job))] = file_hash(stage / name)
    for name in OUTPUT_FILES:
        files[str((output / name).relative_to(job))] = file_hash(output / name)
    value = {
        "schema_version": 1,
        "job_id": request.job_id,
        "source_sha": request.source_sha,
        "request_digest": request.digest,
        "scientific_request_digest": request.scientific_request_digest,
        "attempt_id": journal["attempts"][-1]["attempt_id"],
        "output_dir": str(output.relative_to(job)),
        "files": files,
        "commands": commands,
        "replay_coverage": {"targets": 10, "chains": 180, "full": True},
        "environment": read_json(output / "provenance.json"),
        "elapsed_seconds": journal["elapsed_seconds"],
    }
    if (job / "artifact_manifest.json").exists():
        if read_json(job / "artifact_manifest.json") != value:
            raise ValueError("immutable manifest differs from reconstructed evidence")
    else:
        write_json(job / "artifact_manifest.json", value, immutable=True)
    journal["evidence_digest"] = canonical_sha256(value)
    write_json(job / "journal.json", journal)
    verify_evidence(job)


def _check_foreign_owners(state_root: Path, current_job: Path, request: Request):
    for other in state_root.iterdir():
        if other == current_job or not other.is_dir() or not (other / "journal.json").exists():
            continue
        active = read_json(other / "journal.json").get("active")
        if active:
            stage = _artifact_path(other, active["stage_dir"])
            if (stage / "owner.json").exists() and _group_alive(read_json(stage / "owner.json")):
                raise ValueError("another job's process ownership remains live or uncertain")
        if (other / "artifact_manifest.json").exists():
            previous = Request.from_dict(read_json(other / "request.json"))
            if (
                previous.source_sha == request.source_sha
                and previous.scientific_request_digest == request.scientific_request_digest
            ):
                verify_evidence(other)
                raise ValueError(
                    f"Matching completed evidence exists under job {previous.job_id}; reuse it."
                )


def _run_attempt(request_path, repo, state_root) -> dict:
    """Execute at most one fresh attempt of a reviewed job; never automatically accept it."""
    request = Request.from_dict(read_json(Path(request_path)))
    repo, state_root = Path(repo).resolve(), Path(state_root).resolve()
    if state_root.is_relative_to(repo):
        raise ValueError("durable state must be outside the source checkout")
    with worker_lock(state_root) as lock_fd:
        job = initialize_job(state_root, request)
        journal = read_json(job / "journal.json")
        try:
            _check_foreign_owners(state_root, job, request)
            _recover(job, journal, request)
            if (job / "artifact_manifest.json").exists():
                if not journal.get("evidence_digest"):
                    output = job / journal["attempts"][-1]["directory"] / "output"
                    validate_output(output, request.scientific_request_digest)
                    _manifest(job, journal, request, output)
                verify_evidence(job)
                status = read_json(job / "status.json")
                if status["phase"] not in {"verifying", "awaiting_review"} or not status.get(
                    "evidence_digest"
                ):
                    return _status(
                        job,
                        journal,
                        "verifying",
                        "Full local evidence recovered; mirror verification is required.",
                    )
                return status
            if journal.get("integrity_failure"):
                return _status(
                    job,
                    journal,
                    "failed",
                    "Integrity failure requires review; refitting is refused.",
                )
            validate_checkout(repo, request)
            if journal["elapsed_seconds"] >= request.budget_seconds:
                return _status(job, journal, "failed", "Cumulative execution budget exhausted.")
            if len(journal["attempts"]) >= request.max_attempts:
                return _status(job, journal, "failed", "Maximum execution attempts exhausted.")
            replay_from = None
            for previous in reversed(journal["attempts"]):
                candidate = job / previous["directory"] / "output"
                if (candidate / "study.json.gz").exists():
                    try:
                        authenticate_generation(candidate, request.scientific_request_digest)
                    except ValueError:
                        journal["integrity_failure"] = True
                        write_json(job / "journal.json", journal)
                        raise
                    replay_from = candidate
                    break
            number = len(journal["attempts"]) + 1
            attempt = {
                "attempt_id": f"attempt-{number:04d}",
                "directory": f"attempts/attempt-{number:04d}",
                "mode": "replay" if replay_from else "generate",
            }
            directory = job / attempt["directory"]
            if directory.exists():
                # No subprocess can launch before this reservation reaches the
                # journal. Only that exact, otherwise empty reservation is safe.
                names = {path.name for path in directory.iterdir()}
                if names not in (set(), {"attempt.json"}):
                    raise ValueError("unrecorded attempt directory has uncertain ownership")
                if names and read_json(directory / "attempt.json") != attempt:
                    raise ValueError("immutable attempt reservation differs")
            else:
                directory.mkdir(parents=True, exist_ok=False)
            if not (directory / "attempt.json").exists():
                write_json(directory / "attempt.json", attempt, immutable=True)
            journal["attempts"].append(attempt)
            write_json(job / "journal.json", journal)
            for name, argv in (
                ("sync", dependency_argv()),
                ("identity", identity_argv(repo, directory / "scientific-identity.json")),
            ):
                result = _stage(
                    job, journal, request, repo, lock_fd, directory / name, argv, "running"
                )
                if result["exit_status"] != 0:
                    return _status(
                        job,
                        journal,
                        "failed",
                        f"Infrastructure stage {name} failed; remaining limits still apply.",
                    )
            identity = read_json(directory / "scientific-identity.json")
            if identity.get("scientific_request_digest") != request.scientific_request_digest:
                raise ValueError("scientific request digest differs from reviewed source")
            validate_checkout(repo, request)
            output = directory / "output"
            result = _stage(
                job,
                journal,
                request,
                repo,
                lock_fd,
                directory / "study",
                study_argv(repo, output, request.workers, replay_from),
                "verifying" if replay_from else "running",
            )
            if result["exit_status"] != 0:
                if not result["timed_out"] and result["exit_status"] > 0:
                    journal["integrity_failure"] = True
                    write_json(job / "journal.json", journal)
                return _status(
                    job,
                    journal,
                    "failed",
                    "Study did not complete; no evidence accepted. Inspect bounded diagnostics.",
                )
            try:
                validate_output(output, request.scientific_request_digest)
                validate_checkout(repo, request)
                _manifest(job, journal, request, output)
            except ValueError:
                journal["integrity_failure"] = True
                write_json(job / "journal.json", journal)
                raise
            return _status(
                job,
                journal,
                "verifying",
                "Full local replay passed; a verified second copy and "
                "connected-access attestation are required.",
            )
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            write_json(job / "last-error.json", {"type": type(exc).__name__, "detail": str(exc)})
            message = (
                str(exc)[:240]
                if isinstance(exc, ValueError)
                else ("Worker infrastructure error; inspect private diagnostics.")
            )
            return _status(job, journal, "blocked", message)


def run(request_path, repo, state_root) -> dict:
    """Run the bounded job, including its one allowed infrastructure retry."""
    request = Request.from_dict(read_json(Path(request_path)))
    while True:
        status = _run_attempt(request_path, repo, state_root)
        journal = read_json(Path(state_root) / request.job_id / "journal.json")
        if (
            status["phase"] != "failed"
            or journal.get("integrity_failure")
            or journal["active"]
            or len(journal["attempts"]) >= request.max_attempts
            or journal["elapsed_seconds"] >= request.budget_seconds
        ):
            return status


def inspect(state_root, job_id) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", job_id):
        raise ValueError("invalid job_id")
    return read_json(Path(state_root) / job_id / "status.json")


def verify_mirror(job: Path, mirror: Path, *, connected_uri=None, verified_by=None) -> dict:
    job, mirror = Path(job).resolve(), Path(mirror).resolve()
    if mirror == job or mirror.is_relative_to(job) or job.is_relative_to(mirror):
        raise ValueError("second copy must be a separate directory")
    if bool(connected_uri) != bool(verified_by):
        raise ValueError("connected URI and operator identity must be supplied together")
    if connected_uri and (
        not re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/tree/[a-f0-9]{40}/[A-Za-z0-9_./-]+",
            connected_uri,
        )
        or ".." in connected_uri.split("/")
        or not isinstance(verified_by, str)
        or not verified_by.strip()
        or len(verified_by) > 200
    ):
        raise ValueError(
            "connected destination must be a GitHub tree at a full commit and explicit path"
        )
    with worker_lock(job.parent):
        manifest = verify_evidence(job)
        copied = verify_evidence(mirror)
        if canonical_sha256(manifest) != canonical_sha256(copied):
            raise ValueError("second-copy manifest hash differs")
        acknowledgement = {
            "schema_version": 1,
            "evidence_digest": canonical_sha256(manifest),
            "local_second_copy": str(mirror),
            "local_files_rehashed": True,
            "connected_access": "operator_attested" if connected_uri else "not_verified",
            "connected_uri": connected_uri,
            "verified_by": verified_by,
            "remote_fetched_by_worker": False,
            "verified_at": datetime.now(UTC).isoformat(),
        }
        write_json(job / "mirror.json", acknowledgement)
        journal = read_json(job / "journal.json")
        return _status(
            job,
            journal,
            "awaiting_review" if connected_uri else "verifying",
            "Local copies verified; operator attests connected access. Independent review pending."
            if connected_uri
            else "Local second copy verified; connected access requires operator attestation.",
        )
