import hashlib
import subprocess

import pytest

from thermo_lab.improvement_harness.checks import compare_baseline, run_checks
from thermo_lab.improvement_harness.plan import Plan
from thermo_lab.improvement_harness.store import create_candidate, record_result


def _plan():
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


def _passing_results(tmp_path):
    def passing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    return run_checks("dashboard", tmp_path, 120, runner=passing)


def test_passing_catalog_commands_use_fixed_argv_cwd_and_bounded_logs(tmp_path):
    calls = []

    def passing(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b"a" * 100000, stderr=b"warn\n")

    checks = run_checks("dashboard", tmp_path, 120, runner=passing)
    assert [check.argv for check in checks] == [
        ("npm", "ci"),
        ("npm", "test"),
        ("npm", "run", "typecheck"),
        ("npm", "run", "build"),
        ("npm", "run", "test:browser"),
    ]
    assert all(call[1]["cwd"] == tmp_path / "dashboard" for call in calls)
    assert all(call[1]["shell"] is False and call[1]["capture_output"] is True for call in calls)
    assert all(0 < call[1]["timeout"] <= 120 for call in calls)
    assert all(check.execution == "complete" and check.verification == "passed" for check in checks)
    assert all(check.catalog_version == 1 for check in checks)
    assert len(checks[0].log_tail) <= 4096
    raw = (tmp_path / "results" / "harness" / checks[0].log_path).read_bytes()
    assert len(raw) <= 65536
    assert raw.startswith(b"[earlier output truncated]\n")
    assert checks[0].log_sha256 == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_nonzero_exit_fails_verification(tmp_path):
    def failing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, stdout=b"", stderr=b"assertion failed")

    checks = run_checks("dashboard", tmp_path, 120, runner=failing)
    assert checks[0].exit_status == 2
    assert checks[0].execution == "failed"
    assert checks[0].verification == "failed"


def test_timeout_is_not_a_pass(tmp_path):
    def timed_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1, output=b"partial")

    result = run_checks("dashboard", tmp_path, 1, runner=timed_out)[0]
    assert result.execution == "timed_out"
    assert result.verification == "inconclusive"


def test_missing_executable_is_unavailable(tmp_path):
    def missing(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    result = run_checks("research", tmp_path, 120, runner=missing)[0]
    assert result.execution == "unavailable"
    assert result.verification == "inconclusive"


def test_missing_dependency_in_preflight_is_unavailable(tmp_path):
    def missing_package(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"Failed to download thrml")

    checks = run_checks("research", tmp_path, 120, runner=missing_package)
    assert len(checks) == 1
    assert checks[0].execution == "unavailable"
    assert checks[0].verification == "inconclusive"


def test_missing_playwright_browser_binary_is_unavailable(tmp_path):
    def runner(argv, **kwargs):
        if argv[-1] == "test:browser":
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout=b"",
                stderr=b"Executable doesn't exist. Please run npx playwright install",
            )
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    checks = run_checks("dashboard", tmp_path, 120, runner=runner)
    assert checks[-1].execution == "unavailable"
    assert checks[-1].verification == "inconclusive"


def test_unknown_track_cannot_select_a_command(tmp_path):
    with pytest.raises(ValueError, match="unknown track"):
        run_checks("custom", tmp_path, 120, runner=lambda *args, **kwargs: None)


def test_preexisting_baseline_failure_cannot_count_as_improvement(tmp_path):
    baseline = _passing_results(tmp_path)
    candidate = _passing_results(tmp_path)
    baseline[1] = baseline[1].model_copy(
        update={"exit_status": 1, "execution": "failed", "verification": "failed"}
    )
    assert compare_baseline(_plan(), baseline, candidate) == "inconclusive"


def test_candidate_failure_and_missing_check_statuses(tmp_path):
    baseline = _passing_results(tmp_path)
    candidate = _passing_results(tmp_path)
    candidate[1] = candidate[1].model_copy(
        update={"exit_status": 1, "execution": "failed", "verification": "failed"}
    )
    assert compare_baseline(_plan(), baseline, candidate) == "failed"
    assert compare_baseline(_plan(), baseline, []) == "inconclusive"
    assert compare_baseline(_plan(), baseline, _passing_results(tmp_path)) == "passed"


def test_store_rejects_malformed_typed_check_even_with_valid_top_level_status(tmp_path):
    candidate = create_candidate(tmp_path, _plan())
    malformed = _passing_results(tmp_path)[0].model_dump(mode="json")
    malformed["verification"] = "passed"
    malformed["execution"] = "timed_out"
    with pytest.raises(ValueError):
        record_result(
            tmp_path,
            candidate.id,
            {
                "schema_version": 1,
                "execution": "timed_out",
                "verification": "inconclusive",
                "checks": [malformed],
            },
        )
