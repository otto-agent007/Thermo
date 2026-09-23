"""Fixed dashboard checks and screenshots for owner visual review."""

import hashlib
import time
from pathlib import Path

from thermo_lab.improvement_harness.checks import CheckResult, compare_baseline, run_checks
from thermo_lab.improvement_harness.plan import Plan

SCREENSHOTS = ("overview.png", "mobile.png", "experiments.png")


def execution_status(checks: list[CheckResult]) -> str:
    for status in ("timed_out", "unavailable", "failed"):
        if any(check.execution == status for check in checks):
            return status
    return "complete" if checks else "unavailable"


def observe_dashboard(worktree: Path, record_dir: Path, seconds: float) -> dict:
    """Retain only fresh evidence from a successful browser check."""
    record_dir = Path(record_dir).absolute()
    if record_dir.resolve().is_relative_to(worktree.resolve()):
        raise ValueError("record_dir must be outside worktree")
    for part in (record_dir, *record_dir.parents):
        if part.is_symlink():
            raise ValueError("record destination cannot contain symlinks")
    if seconds <= 0:
        raise TimeoutError("plan wall time exhausted")
    source_dir = worktree / "dashboard/test-results"
    for path in (worktree / "dashboard", source_dir):
        if path.is_symlink():
            raise ValueError("screenshot source cannot be a symlink")
    for name in SCREENSHOTS:
        source = source_dir / name
        if source.is_symlink():
            raise ValueError("screenshot source cannot be a symlink")
        source.unlink(missing_ok=True)
    checks = run_checks("dashboard", worktree, seconds, record_dir=record_dir)
    browser_passed = any(
        check.name == "browser" and check.verification == "passed" for check in checks
    )
    artifacts = {}
    artifact_dir = record_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    if artifact_dir.is_symlink():
        raise ValueError("artifact directory cannot be a symlink")
    if browser_passed:
        for name in SCREENSHOTS:
            source = source_dir / name
            if (
                (worktree / "dashboard").is_symlink()
                or source_dir.is_symlink()
                or source.is_symlink()
                or not source.is_file()
            ):
                continue
            payload = source.read_bytes()
            with (artifact_dir / name).open("xb") as file:
                file.write(payload)
            artifacts[f"artifacts/{name}"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    return {
        "checks": [check.model_dump(mode="json") for check in checks],
        "visual_evidence": "available" if len(artifacts) == len(SCREENSHOTS) else "unavailable",
        "artifacts": artifacts,
    }


def evaluate_dashboard(
    plan: Plan,
    baseline_worktree: Path,
    candidate_worktree: Path,
    *,
    record_dir: Path,
    baseline_checks: list[CheckResult] | None = None,
    seconds: float | None = None,
) -> dict:
    """Run the fixed catalog; passing checks still require owner visual review.

    ``record_dir`` must be external to both worktrees. Supplied baseline checks
    are the CLI's authenticated, once-per-plan observation, never assumed green.
    """
    if plan.track != "dashboard":
        raise ValueError("dashboard evaluator requires dashboard plan")
    record_dir = Path(record_dir)
    resolved = record_dir.resolve()
    for worktree in (baseline_worktree, candidate_worktree):
        if resolved.is_relative_to(Path(worktree).resolve()):
            raise ValueError("record_dir must be outside both worktrees")
    if record_dir.is_symlink():
        raise ValueError("record_dir cannot be a symlink")
    deadline = time.monotonic() + min(
        plan.wall_seconds, seconds if seconds is not None else plan.wall_seconds
    )
    baseline = None
    if baseline_checks is None:
        baseline = observe_dashboard(
            baseline_worktree, record_dir / "baseline", deadline - time.monotonic()
        )
        baseline_checks = [CheckResult.model_validate(check) for check in baseline["checks"]]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("plan wall time exhausted")
    candidate = observe_dashboard(candidate_worktree, record_dir / "candidate", remaining)
    checks = [CheckResult.model_validate(check) for check in candidate["checks"]]
    verification = compare_baseline(plan, baseline_checks, checks)
    if (
        candidate["visual_evidence"] != "available"
        or (baseline is not None and baseline["visual_evidence"] != "available")
    ) and verification == "passed":
        verification = "inconclusive"
    return {
        **candidate,
        "execution": execution_status(baseline_checks + checks),
        "verification": verification,
        "research_outcome": "not_applicable",
        "visual_review": "required",
        "baseline_observation": baseline,
    }
