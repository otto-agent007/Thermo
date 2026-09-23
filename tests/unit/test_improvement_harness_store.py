import fcntl
import hashlib
import json
import multiprocessing
import threading
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import thermo_lab.improvement_harness.store as store_module
from thermo_lab.improvement_harness.plan import Plan
from thermo_lab.improvement_harness.store import (
    append_review,
    create_candidate,
    read_candidate,
    record_result,
)


@pytest.fixture
def plan():
    return Plan(
        schema_version=1,
        track="dashboard",
        objective="labels",
        baseline_commit="a" * 40,
        allowed_paths=("dashboard/src/",),
        primary_metric="checks",
        direction="pass",
        threshold=1,
        max_candidates=1,
        wall_seconds=120,
    )


def test_candidate_and_result_are_written_once_with_exact_byte_digests(tmp_path, plan):
    candidate = create_candidate(tmp_path, plan)
    uuid.UUID(candidate.id)
    directory = tmp_path / candidate.id
    request = (directory / "request.json").read_bytes()
    assert (directory / "request.sha256").read_text().strip() == hashlib.sha256(request).hexdigest()
    assert json.loads(request)["plan_digest"].startswith("sha256:")
    result = {"schema_version": 1, "execution": "complete", "verification": "passed"}
    record_result(tmp_path, candidate.id, result)
    original = (directory / "result.json").read_bytes()
    assert (directory / "result.sha256").read_text().strip() == hashlib.sha256(original).hexdigest()
    with pytest.raises(FileExistsError):
        record_result(
            tmp_path,
            candidate.id,
            {"schema_version": 1, "execution": "failed", "verification": "failed"},
        )
    assert (directory / "result.json").read_bytes() == original


def test_reviews_append_without_changing_result(tmp_path, plan):
    candidate = create_candidate(tmp_path, plan)
    record_result(
        tmp_path,
        candidate.id,
        {"schema_version": 1, "execution": "complete", "verification": "passed"},
    )
    result_before = (tmp_path / candidate.id / "result.json").read_bytes()
    first = append_review(tmp_path, candidate.id, "proposed", "Inspect screenshot")
    second = append_review(tmp_path, candidate.id, "rejected", "Labels obscure plot")
    assert first.name == "review-0001.json"
    assert second.name == "review-0002.json"
    assert json.loads(first.read_text())["candidate_id"] == candidate.id
    assert json.loads(second.read_text())["decision"] == "rejected"
    assert json.loads(second.read_text())["observed_at"]
    assert (tmp_path / candidate.id / "result.json").read_bytes() == result_before
    assert len(read_candidate(tmp_path, candidate.id)["reviews"]) == 2


@pytest.mark.parametrize("note", ["x" * 100_001, "é" * 50_001], ids=["ascii", "utf8"])
def test_oversized_review_note_is_rejected_before_append(tmp_path, plan, note):
    candidate = create_candidate(tmp_path, plan)
    with pytest.raises(ValueError):
        append_review(tmp_path, candidate.id, "accepted", note)
    assert list((tmp_path / candidate.id).glob("review-*.json")) == []


def test_candidate_limit_applies_to_same_plan_digest(tmp_path, plan):
    create_candidate(tmp_path, plan)
    with pytest.raises(ValueError, match="max_candidates"):
        create_candidate(tmp_path, plan)
    other = plan.model_copy(update={"objective": "other"})
    create_candidate(tmp_path, other)


@pytest.mark.parametrize("file", ["request.json", "result.json"])
def test_changed_record_fails_to_load(tmp_path, plan, file):
    candidate = create_candidate(tmp_path, plan)
    record_result(
        tmp_path,
        candidate.id,
        {"schema_version": 1, "execution": "complete", "verification": "passed"},
    )
    path = tmp_path / candidate.id / file
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        read_candidate(tmp_path, candidate.id)


@pytest.mark.parametrize("file", ["request.sha256", "result.sha256"])
def test_missing_or_unreadable_digest_fails_to_load(tmp_path, plan, file):
    candidate = create_candidate(tmp_path, plan)
    record_result(
        tmp_path,
        candidate.id,
        {"schema_version": 1, "execution": "complete", "verification": "passed"},
    )
    (tmp_path / candidate.id / file).write_text("not-a-digest\n")
    with pytest.raises(ValueError, match="digest"):
        read_candidate(tmp_path, candidate.id)


def test_result_requires_known_candidate(tmp_path):
    with pytest.raises(FileNotFoundError):
        record_result(
            tmp_path,
            str(uuid.uuid4()),
            {"schema_version": 1, "execution": "complete", "verification": "passed"},
        )


def test_read_rejects_orphaned_result_digest(tmp_path, plan):
    candidate = create_candidate(tmp_path, plan)
    (tmp_path / candidate.id / "result.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="result digest"):
        read_candidate(tmp_path, candidate.id)


def test_read_rejects_unsupported_result_version_even_with_matching_digest(tmp_path, plan):
    candidate = create_candidate(tmp_path, plan)
    directory = tmp_path / candidate.id
    payload = b'{"schema_version":2,"execution":"complete","verification":"passed"}\n'
    (directory / "result.json").write_bytes(payload)
    (directory / "result.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n")
    with pytest.raises(ValueError, match="result schema_version"):
        read_candidate(tmp_path, candidate.id)


def _rewrite_request_with_matching_digest(directory, changes):
    request = json.loads((directory / "request.json").read_text())
    request.update(changes)
    for name, value in changes.items():
        if value is None:
            request.pop(name)
    payload = (json.dumps(request, sort_keys=True) + "\n").encode()
    (directory / "request.json").write_bytes(payload)
    (directory / "request.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n")


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": None},
        {"track": "unapproved"},
        {"baseline_commit": None},
        {"baseline_commit": "main"},
        {"allowed_paths": ["/etc/"]},
        {"plan_digest": "sha256:bad"},
    ],
)
def test_read_rejects_invalid_request_even_with_matching_digest(tmp_path, plan, changes):
    candidate = create_candidate(tmp_path, plan)
    _rewrite_request_with_matching_digest(tmp_path / candidate.id, changes)
    with pytest.raises(ValueError):
        read_candidate(tmp_path, candidate.id)


def test_invalid_request_cannot_be_used_as_parent(tmp_path, plan):
    plan = plan.model_copy(update={"max_candidates": 2})
    candidate = create_candidate(tmp_path, plan)
    _rewrite_request_with_matching_digest(tmp_path / candidate.id, {"baseline_commit": None})
    with pytest.raises(ValueError):
        create_candidate(tmp_path, plan, parent_id=candidate.id)


@pytest.mark.parametrize(
    "changes",
    [
        {"decision": "approved"},
        {"decision": None},
        {"note": None},
        {"note": ""},
        {"observed_at": None},
        {"observed_at": "2026-09-23T10:00:00"},
    ],
)
def test_read_rejects_invalid_review_even_with_valid_request(tmp_path, plan, changes):
    candidate = create_candidate(tmp_path, plan)
    path = append_review(tmp_path, candidate.id, "proposed", "Inspect")
    review = json.loads(path.read_text())
    review.update(changes)
    for name, value in changes.items():
        if value is None:
            review.pop(name)
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        read_candidate(tmp_path, candidate.id)


def _simultaneous_candidate_worker(root, plan_data, scan_barrier, outcomes):
    """Align two real filesystem scans to expose a missing cross-process lock."""
    original_iterdir = Path.iterdir
    root = Path(root)

    def aligned_iterdir(path):
        entries = list(original_iterdir(path))
        if path == root:
            try:
                scan_barrier.wait(timeout=2)
            except threading.BrokenBarrierError:
                pass  # A lock correctly allows only one process into this scan.
        return iter(entries)

    with patch.object(Path, "iterdir", aligned_iterdir):
        try:
            create_candidate(root, Plan.model_validate(plan_data))
        except ValueError as error:
            outcomes.put("limited" if "max_candidates" in str(error) else repr(error))
        else:
            outcomes.put("created")


def test_candidate_budget_is_atomic_across_processes(tmp_path, plan):
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_simultaneous_candidate_worker,
            args=(str(tmp_path), plan.model_dump(mode="json"), barrier, outcomes),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
        assert [process.exitcode for process in processes] == [0, 0]
        assert sorted(outcomes.get(timeout=1) for _ in processes) == ["created", "limited"]
        assert len([path for path in tmp_path.iterdir() if path.name[0] != "."]) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)


def _hold_lock(lock_path, started, release):
    with Path(lock_path).open("r+b") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        started.set()
        release.wait(timeout=5)


def test_plan_lock_wait_is_bounded_and_timeout_consumes_no_candidate(tmp_path, plan, monkeypatch):
    plan = plan.model_copy(update={"max_candidates": 2})
    create_candidate(tmp_path, plan)
    lock_path = next((tmp_path / ".locks").iterdir())
    context = multiprocessing.get_context("fork")
    started = context.Event()
    release = context.Event()
    holder = context.Process(target=_hold_lock, args=(str(lock_path), started, release))
    holder.start()
    try:
        assert started.wait(timeout=2)
        monkeypatch.setattr(store_module, "_LOCK_WAIT_SECONDS", 0.1)
        with pytest.raises(TimeoutError, match="plan candidate lock"):
            create_candidate(tmp_path, plan)
        create_candidate(tmp_path, plan.model_copy(update={"objective": "unrelated"}))
    finally:
        release.set()
        holder.join(timeout=2)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=1)
    assert holder.exitcode == 0
    create_candidate(tmp_path, plan)
