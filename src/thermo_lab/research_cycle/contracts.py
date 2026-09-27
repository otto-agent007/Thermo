"""Strict immutable execution requests, separate from scientific identity."""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from thermo_lab.hashing import canonical_sha256

PROTOCOL = "docs/experiments/meta-ebm-cap-baseline.md"
MAX_SECONDS = 21600
PHASES = frozenset(
    {"queued", "running", "verifying", "awaiting_review", "recorded", "blocked", "failed"}
)


@dataclasses.dataclass(frozen=True)
class Request:
    schema_version: int
    job_id: str
    source_sha: str
    protocol_sha256: str
    lock_sha256: str
    scientific_request_digest: str
    workers: int
    budget_seconds: int
    max_attempts: int
    question: str

    @classmethod
    def from_dict(cls, value: Any) -> Request:
        fields = {field.name for field in dataclasses.fields(cls)}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("request fields must exactly match schema v1")
        for key, maximum in (
            ("schema_version", 1),
            ("workers", 2),
            ("budget_seconds", MAX_SECONDS),
            ("max_attempts", 2),
        ):
            if type(value[key]) is not int or not 1 <= value[key] <= maximum:
                raise ValueError(f"{key} must be an integer between 1 and {maximum}")
        for key, pattern in (
            ("job_id", r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"),
            ("source_sha", r"[a-f0-9]{40}"),
            ("protocol_sha256", r"[a-f0-9]{64}"),
            ("lock_sha256", r"[a-f0-9]{64}"),
            ("scientific_request_digest", r"sha256:[a-f0-9]{64}"),
        ):
            if not isinstance(value[key], str) or not re.fullmatch(pattern, value[key]):
                raise ValueError(f"invalid {key}")
        question = value["question"]
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("question must contain between 1 and 4000 characters")
        return cls(**value)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())
