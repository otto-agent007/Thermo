"""Cheap lifecycle checks: real Linux subprocesses, locks and disposable git repos."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from thermo_lab.hashing import canonical_json, canonical_sha256


def api():
    from thermo_lab.research_cycle import contracts, worker

    return contracts, worker


def request_values():
    return {
        "schema_version": 1,
        "job_id": "m5a-2026-09-26",
        "source_sha": "a" * 40,
        "protocol_sha256": "b" * 64,
        "lock_sha256": "c" * 64,
        "scientific_request_digest": "sha256:" + "d" * 64,
        "workers": 2,
        "budget_seconds": 21600,
        "max_attempts": 2,
        "question": "Does the fixed M5a cap grid pass its stated gates?",
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("job_id", "../escape"),
        ("source_sha", "main"),
        ("protocol_sha256", "z" * 64),
        ("scientific_request_digest", "d" * 64),
        ("workers", True),
        ("workers", 3),
        ("budget_seconds", 21601),
        ("budget_seconds", 1.5),
        ("max_attempts", 3),
        ("question", ""),
        ("command", "rm -rf ."),
    ],
)
def test_requests_reject_unsafe_or_unbounded_values(key, value):
    contracts, _ = api()
    values = request_values()
    values[key] = value
    with pytest.raises(ValueError):
        contracts.Request.from_dict(values)


def test_request_is_immutable_and_hashes_only_its_declared_inputs():
    contracts, _ = api()
    request = contracts.Request.from_dict(request_values())
    assert request.digest == canonical_sha256(request_values())
    with pytest.raises((AttributeError, TypeError)):
        request.workers = 3


def test_duplicate_worker_contends_on_real_advisory_lock(tmp_path):
    _, worker = api()
    with worker.worker_lock(tmp_path):
        code = (
            "from pathlib import Path; "
            "from thermo_lab.research_cycle.worker import worker_lock; "
            f"worker_lock(Path({str(tmp_path)!r})).__enter__()"
        )
        child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert child.returncode != 0
    assert "already owned" in child.stderr
    with worker.worker_lock(tmp_path):
        pass


def test_process_deadline_kills_grandchildren_and_bounds_logs(tmp_path):
    _, worker = api()
    pid_file = tmp_path / "grandchild.pid"
    heartbeat_file = tmp_path / "grandchild.heartbeat"
    grandchild = (
        "import time; from pathlib import Path; p=Path("
        + repr(str(heartbeat_file))
        + '); exec("while True: p.write_text(str(time.time())); time.sleep(.02)")'
    )
    command = [
        sys.executable,
        "-c",
        "import subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        f"open({str(pid_file)!r},'w').write(str(p.pid)); "
        "sys.stdout.write('X'*200000); sys.stdout.flush(); time.sleep(60)",
    ]
    with worker.worker_lock(tmp_path) as lock_fd:
        result = worker.run_process(
            command,
            cwd=tmp_path,
            environment=worker.execution_environment(tmp_path),
            timeout=0.5,
            stage_dir=tmp_path / "stage",
            lock_fd=lock_fd,
        )
    assert result["timed_out"] is True
    assert result["elapsed_seconds"] < 5
    assert (tmp_path / "stage/stdout.log").stat().st_size <= worker.LOG_LIMIT
    assert int(pid_file.read_text()) > 1
    heartbeat = heartbeat_file.read_text()
    time.sleep(0.1)
    assert heartbeat_file.read_text() == heartbeat


def test_child_retains_lock_after_worker_parent_is_killed(tmp_path):
    _, worker = api()
    script = tmp_path / "parent.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from thermo_lab.research_cycle.worker import "
        "worker_lock,run_process,execution_environment\n"
        f"root=Path({str(tmp_path)!r})\n"
        "with worker_lock(root) as fd:\n"
        " run_process([sys.executable,'-c','import time; time.sleep(60)'],cwd=root,"
        "environment=execution_environment(root),timeout=60,stage_dir=root/'stage',lock_fd=fd)\n"
    )
    parent = subprocess.Popen([sys.executable, str(script)])
    owner_file = tmp_path / "stage/owner.json"
    try:
        deadline = time.monotonic() + 5
        while not owner_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert owner_file.exists()
        parent.kill()
        parent.wait(timeout=5)
        with pytest.raises(ValueError, match="already owned"):
            with worker.worker_lock(tmp_path):
                pass
    finally:
        if owner_file.exists():
            os.killpg(json.loads(owner_file.read_text())["pgid"], signal.SIGKILL)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)


def test_same_job_id_cannot_change_request(tmp_path):
    contracts, worker = api()
    request = contracts.Request.from_dict(request_values())
    worker.initialize_job(tmp_path, request)
    changed = request_values() | {"question": "A different question"}
    with pytest.raises(ValueError, match="immutable"):
        worker.initialize_job(tmp_path, contracts.Request.from_dict(changed))


def test_status_exports_exact_schema_and_never_claims_review(tmp_path):
    contracts, worker = api()
    job = worker.initialize_job(tmp_path, contracts.Request.from_dict(request_values()))
    status = json.loads((job / "status.json").read_text())
    assert set(status) == {
        "schema_version",
        "job_id",
        "source_sha",
        "request_digest",
        "phase",
        "attempts",
        "elapsed_seconds",
        "heartbeat_at",
        "evidence_digest",
        "review_status",
        "message",
    }
    assert status["phase"] == "queued"
    assert status["review_status"] == "pending"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "Test fixture")
    (repo / "docs/experiments").mkdir(parents=True)
    (repo / "docs/experiments/meta-ebm-cap-baseline.md").write_text("fixture protocol\n")
    (repo / "uv.lock").write_text("fixture lock\n")
    (repo / ".gitignore").write_text(".venv/\n__pycache__/\n")
    package = repo / "src/thermo_lab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    fixture = Path(__file__).parents[1] / "fixtures/research_cycle/study.py"
    (package / "meta_ebm_cap_baseline.py").write_text(fixture.read_text())
    (package / "hashing.py").write_text(
        "import json,hashlib\n"
        "def canonical_json(x): return json.dumps(x,sort_keys=True,separators=(',',':'))\n"
        "def canonical_sha256(x): return 'sha256:'+"
        "hashlib.sha256(canonical_json(x).encode()).hexdigest()\n"
    )
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Disposable test fixture, never scientific evidence")
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    return repo


def make_request(repo, destination, **kwargs):
    _, worker = api()
    request = worker.prepare_request(repo, "test-m5a", "Fixture lifecycle only", **kwargs)
    destination.write_text(canonical_json(request.to_dict()))
    return request


def test_checkout_refuses_dirty_wrong_source_and_nonexternal_state(repository, tmp_path):
    _, worker = api()
    request = make_request(repository, tmp_path / "request.json")
    worker.validate_checkout(repository, request)
    (repository / "dirty.txt").write_text("dirty")
    with pytest.raises(ValueError, match="clean"):
        worker.validate_checkout(repository, request)
    (repository / "dirty.txt").unlink()
    contracts, _ = api()
    changed = contracts.Request.from_dict(request.to_dict() | {"source_sha": "a" * 40})
    with pytest.raises(ValueError, match="source"):
        worker.validate_checkout(repository, changed)
    with pytest.raises(ValueError, match="outside"):
        worker.run(tmp_path / "request.json", repository, repository / "state")


def test_lifecycle_requires_real_evidence_and_separate_access_attestation(
    repository, tmp_path, monkeypatch
):
    _, worker = api()
    request_path = tmp_path / "request.json"
    make_request(repository, request_path, budget_seconds=20)
    monkeypatch.setattr(worker, "dependency_argv", lambda: [sys.executable, "-c", "pass"])
    state = tmp_path / "state"
    status = worker.run(request_path, repository, state)
    assert status["phase"] == "verifying"
    assert status["attempts"] == 1
    job = state / "test-m5a"
    manifest = worker.verify_evidence(job)
    assert manifest["replay_coverage"] == {"targets": 10, "chains": 180, "full": True}
    assert worker.run(request_path, repository, state)["attempts"] == 1
    import shutil

    mirror = tmp_path / "mirror"
    shutil.copytree(job, mirror)
    assert worker.verify_mirror(job, mirror)["phase"] == "verifying"
    with pytest.raises(ValueError, match="GitHub"):
        worker.verify_mirror(job, mirror, connected_uri="https://example.org", verified_by="owner")
    assert (
        worker.verify_mirror(
            job,
            mirror,
            connected_uri="https://github.com/example/research/tree/"
            + "a" * 40
            + "/evidence/test-m5a",
            verified_by="operator:test",
        )["phase"]
        == "awaiting_review"
    )
    assert json.loads((job / "mirror.json").read_text())["connected_access"] == "operator_attested"
    archive_path = next(job.glob("attempts/*/output/study.json.gz"))
    archive_path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash"):
        worker.verify_evidence(job)


def test_missing_completion_and_unreplayed_or_partial_evidence_are_refused(tmp_path):
    _, worker = api()
    with pytest.raises(ValueError, match="missing"):
        worker.validate_output(tmp_path, "sha256:" + "a" * 64)
    (tmp_path / "completion.json").write_text(json.dumps({"replayed": False}))
    with pytest.raises(ValueError):
        worker.validate_output(tmp_path, "sha256:" + "a" * 64)


def test_recovery_charges_interrupted_time_and_refuses_exhausted_budget(
    repository, tmp_path, monkeypatch
):
    _, worker = api()
    request_path = tmp_path / "request.json"
    request = make_request(repository, request_path, budget_seconds=1)
    state = tmp_path / "state"
    job = worker.initialize_job(state, request)
    journal = json.loads((job / "journal.json").read_text())
    journal["active"] = {"started_at": time.time() - 10, "stage_dir": "attempts/attempt-0001/run"}
    worker.write_json(job / "journal.json", journal)
    monkeypatch.setattr(worker, "dependency_argv", lambda: pytest.fail("must not launch"))
    status = worker.run(request_path, repository, state)
    assert status["phase"] == "failed"
    assert status["elapsed_seconds"] == 1


def test_recovery_blocks_live_process_group_even_with_free_lock(repository, tmp_path):
    _, worker = api()
    request_path = tmp_path / "request.json"
    request = make_request(repository, request_path)
    state = tmp_path / "state"
    job = worker.initialize_job(state, request)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        stage = job / "attempts/attempt-0001/run"
        stage.mkdir(parents=True)
        worker.write_json(stage / "owner.json", worker.process_identity(child.pid))
        journal = json.loads((job / "journal.json").read_text())
        journal["active"] = {
            "started_at": time.time() - 5,
            "stage_dir": "attempts/attempt-0001/run",
        }
        worker.write_json(job / "journal.json", journal)
        status = worker.run(request_path, repository, state)
        assert status["phase"] == "blocked"
        assert "ownership" in status["message"]
    finally:
        child.kill()
        child.wait(timeout=5)


def test_manifest_metadata_tampering_is_rejected(repository, tmp_path, monkeypatch):
    _, worker = api()
    request_path = tmp_path / "request.json"
    make_request(repository, request_path, budget_seconds=20)
    monkeypatch.setattr(worker, "dependency_argv", lambda: [sys.executable, "-c", "pass"])
    state = tmp_path / "state"
    worker.run(request_path, repository, state)
    job = state / "test-m5a"
    manifest = json.loads((job / "artifact_manifest.json").read_text())
    manifest["environment"] = {"invented": "provenance"}
    (job / "artifact_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest digest"):
        worker.verify_evidence(job)


def test_interrupted_generation_recovers_via_replay_without_changing_original_archive(
    repository, tmp_path, monkeypatch
):
    _, worker = api()
    request_path = tmp_path / "request.json"
    request = make_request(repository, request_path, budget_seconds=20)
    monkeypatch.setattr(worker, "dependency_argv", lambda: [sys.executable, "-c", "pass"])
    state = tmp_path / "state"
    job = worker.initialize_job(state, request)
    first = job / "attempts/attempt-0001"
    first.mkdir(parents=True)
    first_output = first / "output"
    subprocess.run(
        worker.study_argv(repository, first_output, 1, None),
        cwd=repository,
        env=worker.execution_environment(repository),
        check=True,
    )
    original_archive = (first_output / "study.json.gz").read_bytes()
    original_generation = (first_output / "generation-provenance.json").read_bytes()
    (first_output / "completion.json").unlink()
    journal = json.loads((job / "journal.json").read_text())
    journal["attempts"] = [
        {"attempt_id": "attempt-0001", "directory": "attempts/attempt-0001", "mode": "generate"}
    ]
    journal["active"] = {"started_at": time.time() - 2, "stage_dir": "attempts/attempt-0001/study"}
    worker.write_json(job / "journal.json", journal)
    result = worker.run(request_path, repository, state)
    assert result["phase"] == "verifying"
    assert result["attempts"] == 2
    assert result["elapsed_seconds"] >= 2
    second = job / "attempts/attempt-0002"
    assert json.loads((second / "attempt.json").read_text())["mode"] == "replay"
    assert (second / "output/study.json.gz").read_bytes() == original_archive
    assert (second / "output/generation-provenance.json").read_bytes() == original_generation
    assert not (first_output / "completion.json").exists()


def test_invalid_saved_generation_blocks_refitting(repository, tmp_path, monkeypatch):
    _, worker = api()
    request_path = tmp_path / "request.json"
    request = make_request(repository, request_path, budget_seconds=20)
    state = tmp_path / "state"
    job = worker.initialize_job(state, request)
    output = job / "attempts/attempt-0001/output"
    output.mkdir(parents=True)
    (output / "study.json.gz").write_bytes(b"not authentic evidence")
    journal = json.loads((job / "journal.json").read_text())
    journal["attempts"] = [
        {"attempt_id": "attempt-0001", "directory": "attempts/attempt-0001", "mode": "generate"}
    ]
    worker.write_json(job / "journal.json", journal)
    monkeypatch.setattr(worker, "dependency_argv", lambda: pytest.fail("must not refit"))
    result = worker.run(request_path, repository, state)
    assert result["phase"] == "blocked"
    assert result["attempts"] == 1
    assert worker.run(request_path, repository, state)["phase"] == "failed"


def test_max_attempts_cannot_be_reset_by_reinvocation(repository, tmp_path, monkeypatch):
    _, worker = api()
    request_path = tmp_path / "request.json"
    make_request(repository, request_path, budget_seconds=20, max_attempts=2)
    state = tmp_path / "state"
    monkeypatch.setattr(
        worker, "dependency_argv", lambda: [sys.executable, "-c", "raise SystemExit(7)"]
    )
    assert worker.run(request_path, repository, state)["attempts"] == 2
    assert worker.run(request_path, repository, state)["attempts"] == 2
    final = worker.run(request_path, repository, state)
    assert final["phase"] == "failed"
    assert final["attempts"] == 2
    assert "Maximum" in final["message"]


def test_request_json_duplicate_keys_are_rejected(tmp_path):
    _, worker = api()
    path = tmp_path / "duplicate.json"
    path.write_text('{"job_id":"approved","job_id":"changed"}')
    with pytest.raises(ValueError, match="duplicate"):
        worker.read_json(path)


def test_other_job_cannot_start_while_old_orphan_group_is_live(repository, tmp_path, monkeypatch):
    contracts, worker = api()
    request_path = tmp_path / "request.json"
    request = make_request(repository, request_path)
    state = tmp_path / "state"
    old_job = worker.initialize_job(state, request)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        stage = old_job / "attempts/attempt-0001/study"
        stage.mkdir(parents=True)
        worker.write_json(stage / "owner.json", worker.process_identity(child.pid))
        journal = json.loads((old_job / "journal.json").read_text())
        journal["active"] = {"started_at": time.time(), "stage_dir": "attempts/attempt-0001/study"}
        worker.write_json(old_job / "journal.json", journal)
        new_request = contracts.Request.from_dict(request.to_dict() | {"job_id": "second-job"})
        request_path.write_text(canonical_json(new_request.to_dict()))
        monkeypatch.setattr(
            worker, "dependency_argv", lambda: pytest.fail("orphan group must block all jobs")
        )
        result = worker.run(request_path, repository, state)
        assert result["phase"] == "blocked"
        assert result["attempts"] == 0
        assert "ownership" in result["message"]
    finally:
        child.kill()
        child.wait(timeout=5)


def test_ignored_cwd_package_cannot_shadow_pinned_source(repository, tmp_path):
    import py_compile

    _, worker = api()
    expected = worker._scientific_identity(repository)
    shadow = repository / "thermo_lab"
    shadow.mkdir()
    payload = tmp_path / "shadow.py"
    payload.write_text("raise RuntimeError('ignored cwd code executed')\n")
    py_compile.compile(str(payload), cfile=str(shadow / "__init__.pyc"))
    (repository / ".git/info/exclude").write_text("/thermo_lab/\n")
    worker.validate_checkout(repository)
    assert worker._scientific_identity(repository) == expected


def test_guardian_enforces_deadline_after_worker_parent_dies(tmp_path):
    _, worker = api()
    heartbeat = tmp_path / "heartbeat"
    script = tmp_path / "parent.py"
    command = (
        "import time; from pathlib import Path; p=Path("
        + repr(str(heartbeat))
        + "); exec('while True: p.write_text(str(time.time())); time.sleep(.02)')"
    )
    script.write_text(
        "from pathlib import Path\nimport sys\n"
        "from thermo_lab.research_cycle.worker import "
        "worker_lock,run_process,execution_environment\n"
        f"root=Path({str(tmp_path)!r})\n"
        "with worker_lock(root) as fd:\n"
        f" run_process([sys.executable,'-c',{command!r}],cwd=root,"
        "environment=execution_environment(root),timeout=.4,stage_dir=root/'stage',lock_fd=fd)\n"
    )
    parent = subprocess.Popen([sys.executable, str(script)])
    owner = tmp_path / "stage/owner.json"
    try:
        deadline = time.monotonic() + 5
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert heartbeat.exists()
        parent.kill()
        parent.wait(timeout=5)
        time.sleep(0.7)
        stopped = heartbeat.read_text()
        time.sleep(0.15)
        assert heartbeat.read_text() == stopped
        with worker.worker_lock(tmp_path):
            pass
    finally:
        if owner.exists():
            try:
                os.killpg(json.loads(owner.read_text())["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)


def test_infrastructure_retry_runs_automatically_with_remaining_budget(
    repository, tmp_path, monkeypatch
):
    _, worker = api()
    request_path = tmp_path / "request.json"
    make_request(repository, request_path, budget_seconds=20)
    calls = 0

    def sync_command():
        nonlocal calls
        calls += 1
        return [sys.executable, "-c", "raise SystemExit(7)" if calls == 1 else "pass"]

    monkeypatch.setattr(worker, "dependency_argv", sync_command)
    status = worker.run(request_path, repository, tmp_path / "state")
    assert status["phase"] == "verifying"
    assert status["attempts"] == 2
    assert status["elapsed_seconds"] > 0


def test_manifest_publication_recovers_after_atomic_journal_write_interruption(
    repository, tmp_path, monkeypatch
):
    _, worker = api()
    request_path = tmp_path / "request.json"
    make_request(repository, request_path, budget_seconds=20)
    monkeypatch.setattr(worker, "dependency_argv", lambda: [sys.executable, "-c", "pass"])
    original = worker.write_json

    def interrupt(path, value, **kwargs):
        if path.name == "journal.json" and value.get("evidence_digest"):
            raise SystemExit("simulated crash after manifest, before journal")
        return original(path, value, **kwargs)

    monkeypatch.setattr(worker, "write_json", interrupt)
    state = tmp_path / "state"
    with pytest.raises(SystemExit):
        worker.run(request_path, repository, state)
    monkeypatch.setattr(worker, "write_json", original)
    status = worker.run(request_path, repository, state)
    assert status["phase"] == "verifying"
    assert status["attempts"] == 1
    worker.verify_evidence(state / "test-m5a")


def test_interruption_during_recovery_is_idempotent(repository, tmp_path, monkeypatch):
    _, worker = api()
    request_path = tmp_path / "request.json"
    request = make_request(repository, request_path, budget_seconds=1)
    state = tmp_path / "state"
    job = worker.initialize_job(state, request)
    journal = json.loads((job / "journal.json").read_text())
    journal["active"] = {"started_at": time.time() - 2, "stage_dir": "attempts/attempt-0001/study"}
    worker.write_json(job / "journal.json", journal)
    original = worker.write_json

    def interrupt(path, value, **kwargs):
        if path.name == "journal.json" and value["active"] is None:
            raise SystemExit("simulated crash during recovery")
        return original(path, value, **kwargs)

    monkeypatch.setattr(worker, "write_json", interrupt)
    with pytest.raises(SystemExit):
        worker.run(request_path, repository, state)
    monkeypatch.setattr(worker, "write_json", original)
    status = worker.run(request_path, repository, state)
    assert status["phase"] == "failed"
    assert status["elapsed_seconds"] == 1
    assert "budget exhausted" in status["message"]


def test_guardian_deadline_before_child_creates_process_group(tmp_path):
    config = tmp_path / "command.json"
    config.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
                "deadline_monotonic": time.monotonic() + 0.1,
            }
        )
    )
    code = (
        "import os,time; from thermo_lab.research_cycle import supervisor; "
        "original=os.setsid; os.setsid=lambda:(time.sleep(.3),original()); "
        f"supervisor.main({str(config)!r})"
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        assert process.wait(timeout=2) == 0
        assert json.loads((tmp_path / "guardian-result.json").read_text())["timed_out"] is True
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        if (tmp_path / "owner.json").exists():
            try:
                os.killpg(json.loads((tmp_path / "owner.json").read_text())["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("recorded", [False, True])
def test_interrupted_attempt_reservation_can_be_reconciled(
    repository, tmp_path, monkeypatch, recorded
):
    _, worker = api()
    request_path = tmp_path / "request.json"
    request = make_request(repository, request_path, budget_seconds=20)
    state = tmp_path / "state"
    job = worker.initialize_job(state, request)
    reserved = job / "attempts/attempt-0001"
    reserved.mkdir(parents=True)
    if recorded:
        worker.write_json(
            reserved / "attempt.json",
            {
                "attempt_id": "attempt-0001",
                "directory": "attempts/attempt-0001",
                "mode": "generate",
            },
            immutable=True,
        )
    monkeypatch.setattr(worker, "dependency_argv", lambda: [sys.executable, "-c", "pass"])
    status = worker.run(request_path, repository, state)
    assert status["phase"] == "verifying"
    assert status["attempts"] == 1


def test_interrupted_budget_charge_survives_wall_clock_rollback(repository, tmp_path):
    _, worker = api()
    request_path = tmp_path / "request.json"
    request = make_request(repository, request_path, budget_seconds=1)
    state = tmp_path / "state"
    job = worker.initialize_job(state, request)
    journal = json.loads((job / "journal.json").read_text())
    journal["active"] = {
        "started_at": time.time() + 3600,
        "started_monotonic": time.monotonic() - 2,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "stage_dir": "attempts/attempt-0001/study",
    }
    worker.write_json(job / "journal.json", journal)
    worker._recover(job, journal, request)
    assert journal["elapsed_seconds"] == 1


def test_complete_files_with_skipped_replay_are_refused(repository, tmp_path):
    _, worker = api()
    request = make_request(repository, tmp_path / "request.json")
    output = tmp_path / "output"
    subprocess.run(
        worker.study_argv(repository, output, 1, None),
        cwd=repository,
        env=worker.execution_environment(repository),
        check=True,
    )
    worker.validate_output(output, request.scientific_request_digest)
    completion = json.loads((output / "completion.json").read_text())
    completion["replayed"] = False
    (output / "completion.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="mandatory full replay"):
        worker.validate_output(output, request.scientific_request_digest)


def test_matching_completed_science_is_not_refit_under_new_job_id(
    repository, tmp_path, monkeypatch
):
    contracts, worker = api()
    request_path = tmp_path / "request.json"
    request = make_request(repository, request_path, budget_seconds=20)
    monkeypatch.setattr(worker, "dependency_argv", lambda: [sys.executable, "-c", "pass"])
    state = tmp_path / "state"
    assert worker.run(request_path, repository, state)["phase"] == "verifying"
    second = contracts.Request.from_dict(request.to_dict() | {"job_id": "duplicate-job"})
    request_path.write_text(canonical_json(second.to_dict()))
    monkeypatch.setattr(
        worker, "dependency_argv", lambda: pytest.fail("must reuse matching evidence")
    )
    status = worker.run(request_path, repository, state)
    assert status["phase"] == "blocked"
    assert status["attempts"] == 0
    assert "Matching completed evidence" in status["message"]
