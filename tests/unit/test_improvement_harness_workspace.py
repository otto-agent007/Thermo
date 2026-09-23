"""Git boundary tests for candidate patch intake."""

import subprocess
from pathlib import Path

import pytest

from thermo_lab.improvement_harness.workspace import (
    create_candidate_worktree,
    export_checked_patch,
    import_manual_patch,
    is_allowed_path,
    parse_status_z,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "dashboard" / "src").mkdir(parents=True)
    (root / "dashboard" / "src" / "page.ts").write_text("before\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    return root


def test_create_candidate_worktree_is_detached_at_baseline(repo: Path) -> None:
    baseline = git(repo, "rev-parse", "HEAD")
    worktree = create_candidate_worktree(repo, baseline, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    assert worktree == repo / ".worktrees" / "harness-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    assert git(worktree, "rev-parse", "HEAD") == baseline
    assert (
        subprocess.run(
            ["git", "-C", str(worktree), "symbolic-ref", "-q", "HEAD"], capture_output=True
        ).returncode
        == 1
    )


def test_create_candidate_rejects_baseline_mismatch(repo: Path) -> None:
    with pytest.raises(ValueError, match="baseline mismatch"):
        create_candidate_worktree(repo, "0" * 40, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    assert not (repo / ".worktrees").exists()


def test_create_candidate_accepts_uppercase_baseline(repo: Path) -> None:
    baseline = git(repo, "rev-parse", "HEAD")
    worktree = create_candidate_worktree(
        repo, baseline.upper(), "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    )
    assert git(worktree, "rev-parse", "HEAD") == baseline


def test_can_create_second_candidate_in_unignored_worktree_directory(repo: Path) -> None:
    baseline = git(repo, "rev-parse", "HEAD")
    create_candidate_worktree(repo, baseline, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    second = create_candidate_worktree(repo, baseline, "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    assert git(second, "rev-parse", "HEAD") == baseline


def test_status_parser_includes_both_rename_and_copy_paths() -> None:
    raw = (
        b"R  dashboard/src/new.ts\0dashboard/src/old.ts\0"
        b"C  dashboard/src/copy.ts\0docs/secret\0?? dashboard/src/new.json\0"
    )
    assert parse_status_z(raw) == [
        "dashboard/src/new.ts",
        "dashboard/src/old.ts",
        "dashboard/src/copy.ts",
        "docs/secret",
        "dashboard/src/new.json",
    ]


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside", "dashboard/src/../../outside"])
def test_path_check_rejects_absolute_and_traversal(repo: Path, path: str) -> None:
    assert not is_allowed_path(repo, path, ("dashboard/src/",))


def test_path_check_rejects_symlink_target_and_symlink_parent(repo: Path) -> None:
    (repo / "dashboard" / "src" / "link").symlink_to("/etc/passwd")
    assert not is_allowed_path(repo, "dashboard/src/link", ("dashboard/src/",))
    assert not is_allowed_path(repo, "dashboard/src/link/child", ("dashboard/src/",))


def test_export_includes_tracked_and_allowed_untracked_files(repo: Path) -> None:
    (repo / "dashboard" / "src" / "page.ts").write_text("after\n")
    (repo / "dashboard" / "src" / "new.ts").write_text("new\n")
    patch = export_checked_patch(repo, ("dashboard/src/",))
    assert b"+after" in patch
    assert b"+new" in patch
    assert b"new file mode" in patch


def test_untracked_protected_file_is_rejected(repo: Path) -> None:
    target = repo / "docs" / "experiment-reports" / "new.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    with pytest.raises(ValueError, match="outside allowed paths"):
        export_checked_patch(repo, ("dashboard/src/",))
    assert git(repo, "diff", "--cached", "--name-only") == ""


def test_ignored_protected_file_is_rejected_with_allowed_edit(repo: Path) -> None:
    (repo / ".gitignore").write_text("results/\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-m", "ignore results")
    (repo / "dashboard" / "src" / "page.ts").write_text("after\n")
    protected = repo / "results" / "harness" / "forged.json"
    protected.parent.mkdir(parents=True)
    protected.write_text("{}")
    with pytest.raises(ValueError, match="outside allowed paths"):
        export_checked_patch(repo, ("dashboard/src/",))
    assert git(repo, "diff", "--cached", "--name-only") == ""


def test_ignored_protected_file_added_during_staging_is_rejected(repo: Path, monkeypatch) -> None:
    from thermo_lab.improvement_harness import workspace

    (repo / ".gitignore").write_text("results/\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-m", "ignore results")
    (repo / "dashboard" / "src" / "page.ts").write_text("after\n")
    real_git = workspace._git

    def git_with_late_ignored_file(worktree, *args):
        output = real_git(worktree, *args)
        if args[:2] == ("add", "-f"):
            protected = repo / "results" / "harness" / "late.json"
            protected.parent.mkdir(parents=True)
            protected.write_text("{}")
        return output

    monkeypatch.setattr(workspace, "_git", git_with_late_ignored_file)
    with pytest.raises(ValueError, match="outside allowed paths"):
        export_checked_patch(repo, ("dashboard/src/",))


def test_rename_from_protected_path_is_rejected(repo: Path) -> None:
    protected = repo / "docs" / "experiment-reports" / "source.json"
    protected.parent.mkdir(parents=True)
    protected.write_text("{}")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "protected fixture")
    git(repo, "mv", "docs/experiment-reports/source.json", "dashboard/src/source.json")
    with pytest.raises(ValueError, match="outside allowed paths"):
        export_checked_patch(repo, ("dashboard/src/",))


def test_export_rejects_symlink_even_inside_allowed_prefix(repo: Path) -> None:
    (repo / "dashboard" / "src" / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="outside allowed paths"):
        export_checked_patch(repo, ("dashboard/src/",))


def test_export_rejects_deletion_of_tracked_symlink(repo: Path) -> None:
    link = repo / "dashboard" / "src" / "link"
    link.symlink_to("/etc/passwd")
    git(repo, "add", "dashboard/src/link")
    git(repo, "commit", "-m", "symlink fixture")
    link.unlink()
    with pytest.raises(ValueError, match="outside allowed paths"):
        export_checked_patch(repo, ("dashboard/src/",))


def test_export_rejects_empty_patch(repo: Path) -> None:
    with pytest.raises(ValueError, match="no diff"):
        export_checked_patch(repo, ("dashboard/src/",))


def test_export_rejects_patch_larger_than_dashboard_limit(repo: Path) -> None:
    (repo / "dashboard" / "src" / "large.txt").write_text("x" * 1_000_000)
    with pytest.raises(ValueError, match="patch.*too large"):
        export_checked_patch(repo, ("dashboard/src/",))


def test_manual_patch_is_checked_before_apply(repo: Path, tmp_path: Path) -> None:
    patch = tmp_path / "bad.patch"
    patch.write_text("not a patch\n")
    with pytest.raises(subprocess.CalledProcessError):
        import_manual_patch(repo, patch, ("dashboard/src/",))
    assert (repo / "dashboard" / "src" / "page.ts").read_text() == "before\n"


def test_manual_patch_rejects_oversized_and_nonregular_input(repo: Path, tmp_path: Path) -> None:
    patch = tmp_path / "large.diff"
    patch.write_bytes(b"x" * 1_000_001)
    with pytest.raises(ValueError, match="patch.*too large"):
        import_manual_patch(repo, patch, ("dashboard/src/",))
    link = tmp_path / "link.diff"
    link.symlink_to(patch)
    with pytest.raises(ValueError, match="regular file"):
        import_manual_patch(repo, link, ("dashboard/src/",))
    assert (repo / "dashboard" / "src" / "page.ts").read_text() == "before\n"


def test_manual_patch_applies_after_check(repo: Path, tmp_path: Path) -> None:
    target = repo / "dashboard" / "src" / "page.ts"
    target.write_text("after\n")
    patch = tmp_path / "good.patch"
    patch.write_text(git(repo, "diff") + "\n")
    git(repo, "checkout", "--", "dashboard/src/page.ts")
    import_manual_patch(repo, patch, ("dashboard/src/",))
    assert target.read_text() == "after\n"


def test_manual_patch_rejects_forbidden_path_before_apply(repo: Path, tmp_path: Path) -> None:
    forbidden = repo / "docs" / "experiment-reports" / "forged.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("before\n")
    git(repo, "add", "docs/experiment-reports/forged.json")
    git(repo, "commit", "-m", "protected fixture")
    forbidden.write_text("after\n")
    patch = tmp_path / "forbidden.patch"
    patch.write_text(git(repo, "diff") + "\n")
    git(repo, "checkout", "--", "docs/experiment-reports/forged.json")
    with pytest.raises(ValueError, match="outside allowed paths"):
        import_manual_patch(repo, patch, ("dashboard/src/",))
    assert forbidden.read_text() == "before\n"


def test_manual_patch_rejects_rename_from_forbidden_path(repo: Path, tmp_path: Path) -> None:
    forbidden = repo / "docs" / "experiment-reports" / "source.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("before\n")
    git(repo, "add", "docs/experiment-reports/source.json")
    git(repo, "commit", "-m", "protected fixture")
    git(repo, "mv", "docs/experiment-reports/source.json", "dashboard/src/source.json")
    patch = tmp_path / "rename.patch"
    patch.write_text(git(repo, "diff", "--cached", "--binary") + "\n")
    git(repo, "reset", "--hard", "HEAD")
    with pytest.raises(ValueError, match="outside allowed paths"):
        import_manual_patch(repo, patch, ("dashboard/src/",))
    assert forbidden.read_text() == "before\n"
    assert not (repo / "dashboard" / "src" / "source.json").exists()
