"""Runtime and pinned-release provenance collection."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from thermo_lab.records import PackageProvenance, RuntimeProvenance


@dataclass(frozen=True)
class PinnedRelease:
    version: str
    source_repository: str
    source_commit: str
    wheel_sha256: str


_PINNED_RELEASES: dict[str, PinnedRelease] = {
    "thrml": PinnedRelease(
        version="0.1.4",
        source_repository="https://github.com/extropic-ai/thrml",
        source_commit="9c4e6fbb800f5e5c627122e668ff1b158ef3782b",
        wheel_sha256="6e2f38cecb562589d230ca063b5fcb5d2a6533201e37bb70c1f2dac4a63a0858",
    ),
    "extro-torx": PinnedRelease(
        version="0.0.1",
        source_repository="https://github.com/extropic-ai/torx",
        source_commit="769d2f90abdfda14798fceb521143f4b99d370da",
        wheel_sha256="e51d6efe0a8bc62fb4b2b417d5e4ac8190e3fb22c9d14d9342c207afdc64a23c",
    ),
}


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def find_repository_root(start: Path) -> Path | None:
    """Walk upward from ``start`` to the checkout holding ``pyproject.toml`` and ``.git``."""

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / ".git").exists():
            return candidate
    return None


def _git_metadata(repository_root: Path | None) -> tuple[str | None, bool | None]:
    if repository_root is None:
        return None, None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def collect_runtime_provenance(repository_root: Path | None = None) -> RuntimeProvenance:
    """Capture software, device, and source state used for a run."""

    # Imported here so this module stays importable before the CLI selects the
    # default JAX platform.
    import jax

    package_names = ("thermo-lab", "thrml", "extro-torx", "jax", "jaxlib", "equinox")
    packages = []
    for name in package_names:
        release = _PINNED_RELEASES.get(name)
        installed_version = _version(name)
        release_matches = release is not None and installed_version == release.version
        packages.append(
            PackageProvenance(
                distribution=name,
                version=installed_version,
                release_source_repository=(release.source_repository if release_matches else None),
                release_source_commit=release.source_commit if release_matches else None,
                expected_wheel_sha256=release.wheel_sha256 if release_matches else None,
                artifact_verification=(
                    "expected_hash_enforced_by_uv_lock_not_runtime_reverified"
                    if release_matches
                    else "unverified_or_not_a_pinned_release"
                ),
            )
        )

    commit, dirty = _git_metadata(repository_root)
    devices = tuple(f"{device.platform}:{device.device_kind}" for device in jax.devices())
    return RuntimeProvenance(
        python_version=platform.python_version(),
        platform=platform.platform(),
        jax_version=_version("jax"),
        jaxlib_version=_version("jaxlib"),
        jax_backend=jax.default_backend(),
        jax_devices=devices,
        git_commit=commit,
        git_dirty=dirty,
        jax_enable_x64=bool(jax.config.jax_enable_x64),
        packages=tuple(packages),
    )
