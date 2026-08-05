"""Canonical JSON normalization and hashing."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


def to_json_value(value: Any) -> Any:
    """Convert supported Python/JAX/NumPy values into strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floating-point values are not valid experiment data")
        return value
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_json_value(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return to_json_value(value.model_dump(mode="python", by_alias=True))
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
            normalized[key] = to_json_value(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return to_json_value(value.tolist())
    if hasattr(value, "item"):
        return to_json_value(value.item())
    raise TypeError(f"Unsupported experiment value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a value with deterministic ordering and no insignificant whitespace."""

    return json.dumps(
        to_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    """Return a self-describing SHA-256 digest of canonical JSON input."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
