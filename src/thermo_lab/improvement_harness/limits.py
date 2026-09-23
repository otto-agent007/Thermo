"""Artifact limits shared by local producers and the read-only dashboard."""

import errno
import os
import stat
from pathlib import Path

MAX_RECOMMENDATION_BYTES = 100_000
MAX_PATCH_BYTES = 1_000_000
MAX_RESULT_BYTES = 1_000_000
MAX_OBJECTIVE_BYTES = 8_000
MAX_SCREENSHOT_BYTES = 8_000_000
MAX_REVIEW_NOTE_BYTES = 100_000


def read_bounded_regular_file(path: Path, max_bytes: int, label: str) -> bytes:
    """Read at most the dashboard limit from one stable, non-symlink file."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"{label} must be a regular file") from error
        raise
    with os.fdopen(descriptor, "rb") as file:
        before = os.fstat(file.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_size > max_bytes:
            raise ValueError(f"{label} is too large")
        payload = file.read(max_bytes + 1)
        after = os.fstat(file.fileno())
        if len(payload) > max_bytes:
            raise ValueError(f"{label} is too large")
        if (
            len(payload) != before.st_size
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ValueError(f"{label} changed while reading")
        return payload


def read_recommendation(path: Path) -> str:
    """A byte cap also bounds JavaScript UTF-16 string length below 100,000."""
    payload = read_bounded_regular_file(path, MAX_RECOMMENDATION_BYTES, "recommendation")
    return payload.decode("utf-8")
