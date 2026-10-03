"""Keep the default research test suite on its reproducible CPU baseline."""

import os
from pathlib import Path

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"


@pytest.fixture
def missing_cgroup_limits(monkeypatch):
    """Reproduce hosts without the two container metadata files."""
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max"):
            raise FileNotFoundError(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
