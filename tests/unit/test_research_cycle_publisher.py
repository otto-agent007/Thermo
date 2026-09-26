"""Publisher transport checks use disposable local Git repositories only."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from thermo_lab.hashing import canonical_sha256
from thermo_lab.research_cycle import worker
from thermo_lab.research_cycle.contracts import Request


def git(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


@pytest.fixture
def context(tmp_path):
    repo = tmp_path / "evidence"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "evidence/test-m5a")
    git(repo, "config", "user.name", "Disposable fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "commit", "--allow-empty", "-qm", "Fixture only")
    git(repo, "remote", "add", "origin", "https://github.com/otto-agent007/Thermo.git")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    request = Request.from_dict(
        {
            "schema_version": 1,
            "job_id": "test-m5a",
            "source_sha": git(repo, "rev-parse", "HEAD"),
            "protocol_sha256": "b" * 64,
            "lock_sha256": "c" * 64,
            "scientific_request_digest": canonical_sha256({"fixture": "NOT SCIENTIFIC EVIDENCE"}),
            "workers": 2,
            "budget_seconds": 10,
            "max_attempts": 2,
            "question": "Fixture only",
        }
    )
    state = tmp_path / "state"
    job = worker.initialize_job(state, request)
    return state, job, repo, remote


def publish(context):
    from thermo_lab.research_cycle import publisher

    state, _, repo, remote = context
    return publisher.publish_snapshot(
        state,
        "test-m5a",
        repo,
        "evidence/test-m5a",
        _test_transport=remote,
    )


def complete(context):
    _, job, _, _ = context
    output = job / "attempts/attempt-0001/output"
    output.parent.mkdir(parents=True)
    fixture = Path(__file__).parents[1] / "fixtures/research_cycle/study.py"
    subprocess.run([sys.executable, str(fixture), "--output-dir", str(output)], check=True)
    request = Request.from_dict(worker.read_json(job / "request.json"))
    journal = worker.read_json(job / "journal.json")
    journal["attempts"] = [{"attempt_id": "attempt-0001", "directory": "attempts/attempt-0001"}]
    worker._manifest(job, journal, request, output)
    worker._status(job, journal, "verifying", "Fixture only")
    return output


def test_active_observation_publishes_without_worker_lock_and_noop_is_stable(context):
    state, job, repo, remote = context
    with worker.worker_lock(state):
        first = publish(context)
    assert first["kind"] == "observation"
    assert first["connected_access"] == "not_verified"
    assert first["commit_sha"] == git(remote, "rev-parse", "refs/heads/evidence/test-m5a")
    assert git(repo, "status", "--porcelain") == ""
    published = repo / "cycles/test-m5a"
    assert {p.name for p in published.iterdir()} == {"request.json", "status.json"}
    assert json.loads((published / "status.json").read_text()) == worker.read_json(
        job / "status.json"
    )
    second = publish(context)
    assert second["changed"] is False
    assert second["commit_sha"] == first["commit_sha"]


def test_completed_evidence_is_verified_and_only_bound_files_are_published(context):
    _, job, repo, _ = context
    complete(context)
    (job / "secret.txt").write_text("never copied")
    before = (job / "status.json").read_bytes()
    result = publish(context)
    assert result["kind"] == "evidence"
    copied = repo / "cycles/test-m5a"
    assert worker.verify_evidence(copied) == worker.verify_evidence(job)
    assert not (copied / "secret.txt").exists()
    assert (copied / "journal.json").exists()
    assert (job / "status.json").read_bytes() == before


@pytest.mark.parametrize("failure", ["branch", "dirty", "origin", "symlink", "escape"])
def test_refuses_unsafe_checkout_or_job_path(context, failure):
    from thermo_lab.research_cycle import publisher

    state, job, repo, remote = context
    job_id = "test-m5a"
    if failure == "branch":
        git(repo, "checkout", "-qb", "main")
    elif failure == "dirty":
        (repo / "unrelated.txt").write_text("retain this")
    elif failure == "origin":
        git(repo, "remote", "set-url", "origin", "https://github.com/other/repo.git")
    elif failure == "symlink":
        (repo / "cycles").symlink_to(state, target_is_directory=True)
        git(repo, "add", "cycles")
        git(repo, "commit", "-qm", "Unsafe fixture path")
    else:
        job_id = "../test-m5a"
    head = git(repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError):
        publisher.publish_snapshot(state, job_id, repo, "evidence/test-m5a", _test_transport=remote)
    assert git(repo, "rev-parse", "HEAD") == head
    assert job.exists()


def test_invalid_completed_evidence_never_changes_checkout(context):
    _, _, repo, _ = context
    output = complete(context)
    (output / "summary.md").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        publish(context)
    assert not (repo / "cycles").exists()


def test_oversized_observation_is_refused(context, monkeypatch):
    from thermo_lab.research_cycle import publisher

    monkeypatch.setattr(publisher, "MAX_SNAPSHOT_BYTES", 10)
    with pytest.raises(ValueError, match="size"):
        publish(context)


def test_completed_snapshot_requires_worker_lock(context):
    state, _, _, _ = context
    complete(context)
    with worker.worker_lock(state):
        with pytest.raises(ValueError, match="lock"):
            publish(context)


def test_push_failure_preserves_local_commit_for_explicit_retry(context):
    _, _, repo, remote = context
    original = git(repo, "rev-parse", "HEAD")
    git(remote, "config", "core.bare", "false")
    git(remote, "symbolic-ref", "HEAD", "refs/heads/evidence/test-m5a")
    git(remote, "config", "receive.denyCurrentBranch", "refuse")
    with pytest.raises(ValueError, match="push failed.*retained"):
        publish(context)
    saved = git(repo, "rev-parse", "HEAD")
    assert saved != original
    assert (repo / "cycles/test-m5a/request.json").exists()
    git(remote, "config", "core.bare", "true")
    result = publish(context)
    assert result["changed"] is False
    assert result["commit_sha"] == saved


def test_git_attributes_cannot_change_evidence_bytes_silently(context):
    _, _, repo, remote = context
    (repo / ".git/info/attributes").write_text("*.json filter=corrupt\n")
    git(repo, "config", "filter.corrupt.clean", "sed s/queued/corrupted/g")
    with pytest.raises(ValueError, match="Git blob"):
        publish(context)
    assert git(remote, "for-each-ref", "refs/heads") == ""


def test_clean_checkout_cannot_push_unrelated_local_commits(context):
    _, _, repo, remote = context
    unrelated = repo / "unrelated.txt"
    unrelated.write_text("Must not publish this local commit")
    git(repo, "add", "unrelated.txt")
    git(repo, "commit", "-qm", "Unrelated fixture change")
    git(repo, "rm", "unrelated.txt")
    git(repo, "commit", "-qm", "Revert unrelated fixture change")
    assert git(repo, "status", "--porcelain") == ""
    with pytest.raises(ValueError, match="outside the designated cycle"):
        publish(context)
    assert git(remote, "for-each-ref", "refs/heads") == ""


def remote_review(context):
    _, _, _, remote = context
    reviewer = remote.parent / "reviewer"
    subprocess.run(
        ["git", "clone", "-q", "--branch", "evidence/test-m5a", str(remote), str(reviewer)],
        check=True,
    )
    git(reviewer, "config", "user.name", "Independent fixture reviewer")
    git(reviewer, "config", "user.email", "reviewer@example.invalid")
    (reviewer / "reviews").mkdir()
    (reviewer / "reviews/test-m5a.md").write_text("Fixture review only")
    git(reviewer, "add", "reviews")
    git(reviewer, "commit", "-qm", "Independent fixture review")
    git(reviewer, "push", "origin", "HEAD:refs/heads/evidence/test-m5a")


def test_remote_only_review_is_fast_forwarded_without_deleting_review(context):
    _, _, repo, remote = context
    publish(context)
    remote_review(context)
    result = publish(context)
    assert result["changed"] is False
    assert (repo / "reviews/test-m5a.md").read_text() == "Fixture review only"
    assert result["commit_sha"] == git(remote, "rev-parse", "refs/heads/evidence/test-m5a")


def test_divergent_review_branch_requires_explicit_recovery(context):
    _, _, repo, remote = context
    publish(context)
    (repo / "cycles/test-m5a/note.md").write_text("Local-only note")
    git(repo, "add", "cycles")
    git(repo, "commit", "-qm", "Local fixture note")
    local_head = git(repo, "rev-parse", "HEAD")
    remote_review(context)
    remote_head = git(remote, "rev-parse", "refs/heads/evidence/test-m5a")
    with pytest.raises(ValueError, match="diverged"):
        publish(context)
    assert git(repo, "rev-parse", "HEAD") == local_head
    assert git(remote, "rev-parse", "refs/heads/evidence/test-m5a") == remote_head


def test_cli_has_no_local_transport_override():
    from thermo_lab.research_cycle import publisher

    with pytest.raises(SystemExit) as raised:
        publisher.main(
            [
                "--state-root",
                "/tmp/state",
                "--job-id",
                "test-m5a",
                "--evidence-repo",
                "/tmp/evidence",
                "--branch",
                "evidence/test-m5a",
                "--test-transport",
                "/tmp/not-allowed",
            ]
        )
    assert raised.value.code == 2
