"""Validated, immutable inputs for one improvement-harness iteration."""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from thermo_lab.improvement_harness.limits import MAX_OBJECTIVE_BYTES


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    track: Literal["dashboard", "research"]
    objective: str = Field(min_length=1, max_length=MAX_OBJECTIVE_BYTES)
    baseline_commit: str
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    direction: Literal["pass", "lower", "higher"]
    threshold: float = Field(allow_inf_nan=False)
    max_candidates: int = Field(ge=1)
    wall_seconds: int = Field(gt=0)
    heldout_role: str | None = None

    @field_validator("objective")
    @classmethod
    def bounded_objective(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_OBJECTIVE_BYTES:
            raise ValueError("objective is too large for dashboard review")
        return value

    @field_validator("heldout_role")
    @classmethod
    def valid_heldout_role(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None:
            raise ValueError("invalid held-out role")
        return value

    @field_validator("baseline_commit")
    @classmethod
    def full_commit(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
            raise ValueError("baseline_commit must be a full 40-hex commit")
        return value

    @field_validator("allowed_paths")
    @classmethod
    def relative_prefixes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        for value in values:
            path = PurePosixPath(value)
            if (
                not value
                or value.startswith("/")
                or "\\" in value
                or any(part in ("", ".", "..") for part in value.rstrip("/").split("/"))
                or str(path) == "."
            ):
                raise ValueError(f"allowed path must be a relative prefix: {value!r}")
            normalized = value.rstrip("/")
            if normalized in seen:
                raise ValueError(f"duplicate allowed path: {value!r}")
            seen.add(normalized)
        return values


def load_plan(path: Path) -> Plan:
    """Load a declared plan; runtime observations are never part of it."""
    return Plan.model_validate_json(path.read_text(encoding="utf-8"))


def plan_digest(plan: Plan) -> str:
    raw = json.dumps(
        plan.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()
