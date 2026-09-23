"""Append-only local candidate records with byte-level integrity checks."""

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from thermo_lab.improvement_harness.plan import Plan, plan_digest


class Candidate(BaseModel):
    """Frozen request identity, without observed execution data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    id: str
    parent_id: str | None = None
    track: Literal["dashboard", "research"]
    objective: str
    baseline_commit: str
    allowed_paths: tuple[str, ...]
    plan_digest: str


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


def _validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != 1:
        raise ValueError("unsupported result schema_version")
    if result.get("execution") not in {"complete", "failed", "timed_out", "unavailable"}:
        raise ValueError("invalid execution status")
    if result.get("verification") not in {"passed", "failed", "inconclusive"}:
        raise ValueError("invalid verification status")


def create_candidate(root: Path, plan: Plan, parent_id: str | None = None) -> Candidate:
    """Create a new request, respecting the frozen plan's candidate budget."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    digest = plan_digest(plan)
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
        request = _read_record(candidate_dir, "request")
        if request.get("plan_digest") == digest:
            count += 1
    if count >= plan.max_candidates:
        raise ValueError("max_candidates reached for this plan")
    candidate = Candidate(
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
    _read_record(directory, "request")
    _validate_result(result)
    _write_record(directory, "result", result)


def append_review(root: Path, candidate_id: str, decision: str, note: str) -> Path:
    """Append an owner decision without modifying request or result bytes."""
    directory = _candidate_dir(root, candidate_id)
    _read_record(directory, "request")
    if decision not in {"proposed", "accepted", "rejected"}:
        raise ValueError("invalid review decision")
    review = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "decision": decision,
        "note": note,
        "observed_at": datetime.now(UTC).isoformat(),
    }
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
    request = _read_record(directory, "request")
    if request.get("id") != candidate_id or request.get("schema_version") != 1:
        raise ValueError("invalid candidate request identity or version")
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
        if review.get("candidate_id") != candidate_id or review.get("schema_version") != 1:
            raise ValueError("invalid review identity or version")
        reviews.append(review)
    return {"request": request, "result": result, "reviews": reviews}
