"""Append-only local candidate records with byte-level integrity checks."""

import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from thermo_lab.improvement_harness.plan import Plan, plan_digest

_LOCK_WAIT_SECONDS = 5.0


class Candidate(BaseModel):
    """Frozen request identity, without observed execution data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    id: str
    parent_id: str | None = None
    track: Literal["dashboard", "research"]
    objective: str = Field(min_length=1)
    baseline_commit: str
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    plan_digest: str

    @field_validator("id", "parent_id")
    @classmethod
    def canonical_uuid(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                parsed = uuid.UUID(value)
            except (ValueError, AttributeError) as error:
                raise ValueError("candidate ID must be a UUID") from error
            if str(parsed) != value:
                raise ValueError("candidate ID must be a canonical UUID")
        return value

    @field_validator("baseline_commit")
    @classmethod
    def full_commit(cls, value: str) -> str:
        return Plan.full_commit(value)

    @field_validator("allowed_paths")
    @classmethod
    def relative_prefixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return Plan.relative_prefixes(value)

    @field_validator("plan_digest")
    @classmethod
    def valid_plan_digest(cls, value: str) -> str:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ValueError("invalid plan digest")
        return value


class Review(BaseModel):
    """One append-only owner decision with an observed, timezone-aware time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    candidate_id: str
    decision: Literal["proposed", "accepted", "rejected"]
    note: str = Field(min_length=1)
    observed_at: AwareDatetime

    @field_validator("candidate_id")
    @classmethod
    def canonical_uuid(cls, value: str) -> str:
        return Candidate.canonical_uuid(value)

    @field_validator("observed_at", mode="before")
    @classmethod
    def observed_time_is_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("observed_at must be an ISO 8601 string")
        return value


def _candidate_dir(root: Path, candidate_id: str) -> Path:
    try:
        parsed = uuid.UUID(candidate_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("candidate_id must be a UUID") from error
    if str(parsed) != candidate_id:
        raise ValueError("candidate_id must be a canonical UUID")
    directory = Path(root) / candidate_id
    if directory.is_symlink():
        raise ValueError("candidate directory cannot be a symlink")
    return directory


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _write_record(directory: Path, name: str, value: dict[str, Any]) -> None:
    payload = _json_bytes(value)
    with (directory / f"{name}.json").open("xb") as file:
        file.write(payload)
    with (directory / f"{name}.sha256").open("x", encoding="ascii") as file:
        file.write(hashlib.sha256(payload).hexdigest() + "\n")


def _read_record(directory: Path, name: str) -> dict[str, Any]:
    payload = (directory / f"{name}.json").read_bytes()
    try:
        digest = (directory / f"{name}.sha256").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"missing or unreadable {name} digest") from error
    if (
        re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or hashlib.sha256(payload).hexdigest() != digest
    ):
        raise ValueError(f"{name} digest mismatch")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _read_request(directory: Path) -> dict[str, Any]:
    request = _read_record(directory, "request")
    Candidate.model_validate(request)
    if request["id"] != directory.name:
        raise ValueError("invalid candidate request identity")
    return request


def _validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != 1:
        raise ValueError("unsupported result schema_version")
    if result.get("execution") not in {"complete", "failed", "timed_out", "unavailable"}:
        raise ValueError("invalid execution status")
    if result.get("verification") not in {"passed", "failed", "inconclusive"}:
        raise ValueError("invalid verification status")


@contextmanager
def _plan_lock(root: Path, digest: str) -> Iterator[None]:
    """Serialize a plan's budget check and creation across local processes."""
    locks_dir = root / ".locks"
    locks_dir.mkdir(exist_ok=True)
    if locks_dir.is_symlink():
        raise ValueError("plan lock directory cannot be a symlink")
    lock_path = locks_dir / f"{digest.removeprefix('sha256:')}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+b") as file:
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for plan candidate lock") from None
                time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def create_candidate(root: Path, plan: Plan, parent_id: str | None = None) -> Candidate:
    """Create a new request, respecting the frozen plan's candidate budget."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    digest = plan_digest(plan)
    with _plan_lock(root, digest):
        if parent_id is not None:
            parent = read_candidate(root, parent_id)["request"]
            if parent["plan_digest"] != digest:
                raise ValueError("parent candidate belongs to another plan")
        count = 0
        for directory in root.iterdir():
            if not directory.is_dir():
                continue
            try:
                candidate_dir = _candidate_dir(root, directory.name)
            except ValueError:
                continue
            request = _read_request(candidate_dir)
            if request.get("plan_digest") == digest:
                count += 1
        if count >= plan.max_candidates:
            raise ValueError("max_candidates reached for this plan")
        candidate = Candidate(
            schema_version=1,
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            track=plan.track,
            objective=plan.objective,
            baseline_commit=plan.baseline_commit,
            allowed_paths=plan.allowed_paths,
            plan_digest=digest,
        )
        directory = _candidate_dir(root, candidate.id)
        directory.mkdir()
        _write_record(directory, "request", candidate.model_dump(mode="json", exclude_none=True))
        return candidate


def record_result(root: Path, candidate_id: str, result: dict[str, Any]) -> None:
    """Write the sole result for a candidate; never replace an observation."""
    directory = _candidate_dir(root, candidate_id)
    _read_request(directory)
    _validate_result(result)
    _write_record(directory, "result", result)


def append_review(root: Path, candidate_id: str, decision: str, note: str) -> Path:
    """Append an owner decision without modifying request or result bytes."""
    directory = _candidate_dir(root, candidate_id)
    _read_request(directory)
    if decision not in {"proposed", "accepted", "rejected"}:
        raise ValueError("invalid review decision")
    review = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "decision": decision,
        "note": note,
        "observed_at": datetime.now(UTC).isoformat(),
    }
    Review.model_validate(review)
    payload = _json_bytes(review)
    for number in range(1, 10000):
        path = directory / f"review-{number:04d}.json"
        try:
            with path.open("xb") as file:
                file.write(payload)
            return path
        except FileExistsError:
            continue
    raise ValueError("review sequence exhausted")


def read_candidate(root: Path, candidate_id: str) -> dict[str, Any]:
    """Read only records whose request/result bytes match their sidecars."""
    directory = _candidate_dir(root, candidate_id)
    request = _read_request(directory)
    has_result = (directory / "result.json").exists()
    has_digest = (directory / "result.sha256").exists()
    if has_result != has_digest:
        raise ValueError("missing result digest or result record")
    result = _read_record(directory, "result") if has_result else None
    if result is not None:
        _validate_result(result)
    reviews = []
    for path in sorted(directory.glob("review-*.json")):
        review = json.loads(path.read_bytes())
        Review.model_validate(review)
        if review["candidate_id"] != candidate_id:
            raise ValueError("invalid review identity")
        reviews.append(review)
    return {"request": request, "result": result, "reviews": reviews}
