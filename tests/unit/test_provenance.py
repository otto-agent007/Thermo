import importlib.metadata
import tomllib
from pathlib import Path

import pytest

from thermo_lab.provenance import collect_runtime_provenance, find_repository_root
from thermo_lab.release_pins import PINNED_RELEASES


def test_repository_root_is_found_from_a_nested_directory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "configs" / "experiments"
    nested.mkdir(parents=True)

    assert find_repository_root(nested) == tmp_path
    assert find_repository_root(tmp_path) == tmp_path


def test_repository_root_accepts_a_worktree_git_file(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")

    assert find_repository_root(tmp_path / "src") == tmp_path


def test_repository_root_requires_both_markers(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert find_repository_root(tmp_path) is None


def _locked_release(name: str) -> tuple[str, str]:
    """Return (version, wheel sha256) for ``name`` from the committed uv.lock."""

    root = find_repository_root(Path(__file__).resolve())
    assert root is not None
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    for package in lock["package"]:
        if package["name"] == name:
            wheels = package["wheels"]
            assert len(wheels) == 1, f"{name} has {len(wheels)} wheels in uv.lock"
            return package["version"], wheels[0]["hash"].removeprefix("sha256:")
    raise AssertionError(f"{name} is not in uv.lock")


@pytest.mark.parametrize("name", sorted(PINNED_RELEASES))
def test_pinned_release_matches_uv_lock(name: str) -> None:
    """A Dependabot bump that moves uv.lock must also move the provenance pin.

    Otherwise every new run record silently reports the upstream package as
    ``unverified_or_not_a_pinned_release``.
    """

    version, wheel_sha256 = _locked_release(name)
    pinned = PINNED_RELEASES[name]
    assert pinned.version == version
    assert pinned.wheel_sha256 == wheel_sha256


@pytest.mark.parametrize("name", sorted(PINNED_RELEASES))
def test_installed_release_is_the_pinned_one(name: str) -> None:
    assert importlib.metadata.version(name) == PINNED_RELEASES[name].version


def test_runtime_provenance_marks_pinned_releases_verified() -> None:
    provenance = collect_runtime_provenance()
    by_name = {package.distribution: package for package in provenance.packages}
    for name, release in PINNED_RELEASES.items():
        package = by_name[name]
        assert package.version == release.version
        assert package.release_source_commit == release.source_commit
        assert package.artifact_verification == (
            "expected_hash_enforced_by_uv_lock_not_runtime_reverified"
        )
