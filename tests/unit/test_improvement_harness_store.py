import hashlib
import json
import uuid

import pytest

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
