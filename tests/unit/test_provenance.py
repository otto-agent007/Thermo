from pathlib import Path

from thermo_lab.provenance import find_repository_root


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
