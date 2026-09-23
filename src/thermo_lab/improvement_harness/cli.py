"""Local, review-only candidate generation and immutable evaluation records."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from thermo_lab.improvement_harness import dashboard
from thermo_lab.improvement_harness.checks import CATALOG, CheckResult, compare_baseline, run_checks
from thermo_lab.improvement_harness.limits import MAX_RESULT_BYTES, read_recommendation
from thermo_lab.improvement_harness.plan import Plan, load_plan, plan_digest
from thermo_lab.improvement_harness.proposer import build_proposer_prompt, run_codex
from thermo_lab.improvement_harness.research import claim_heldout, evaluate_three_site
from thermo_lab.improvement_harness.store import (
    _json_bytes,
    _plan_lock,
    _read_record,
    _write_record,
    append_review,
    create_candidate,
    read_candidate,
    record_result,
)
from thermo_lab.improvement_harness.workspace import (
    _git,
    _head,
    create_candidate_worktree,
    export_checked_patch,
    import_manual_patch,
    is_allowed_path,
    parse_status_z,
)

_HOOK = "src/thermo_lab/research_candidates/three_site.py"


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _safe_directory(path: Path) -> Path:
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("output paths cannot contain symlinks")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_root(root: Path) -> None:
    path = Path(root).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("record paths cannot contain symlinks")


def _evidence_bytes(root: Path, relative: str) -> bytes:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence path escapes record root")
    target = root / path
    _check_root(target)
    return target.read_bytes()


def _read_candidate_evidence(root: Path, candidate_id: str) -> dict:
    """Show or reuse only evidence that still matches the saved observation."""
    root = Path(root)
    _check_root(root)
    record = read_candidate(root, candidate_id)
    result = record["result"]
    if result is None:
        return record
    directory = root / candidate_id
    for name, digest in result.get("artifacts", {}).items():
        if _digest(_evidence_bytes(directory, name)) != digest:
            raise ValueError(f"candidate artifact digest mismatch: {name}")
    if result.get("patch_digest"):
        if _digest(_evidence_bytes(directory, "patch.diff")) != result["patch_digest"]:
            raise ValueError("parent patch digest mismatch")
    for check in result.get("checks", []):
        item = CheckResult.model_validate(check)
        relative = f"{result['check_record_dir']}/{item.log_path}"
        if _digest(_evidence_bytes(directory, relative)) != item.log_sha256:
            raise ValueError("candidate log digest mismatch")
    if result.get("baseline_record"):
        expected = (
            Path(".plans")
            / record["request"]["plan_digest"].removeprefix("sha256:")
            / "baseline"
            / "result.json"
        )
        if result["baseline_record"] != expected.as_posix():
            raise ValueError("baseline record path does not match candidate plan")
        if (
            _digest(_evidence_bytes(root, result["baseline_record"]))
            != result["baseline_record_digest"]
        ):
            raise ValueError("baseline record digest mismatch")
        _read_baseline(
            root / expected.parent,
            record["request"]["plan_digest"],
            record["request"]["baseline_commit"],
        )
    return record


def _source_repo() -> Path:
    return Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())


def _clean_source(repo: Path, baseline: str | None = None) -> str:
    head = _head(repo)
    if baseline is not None and head != baseline.lower():
        raise ValueError("baseline mismatch")
    if _git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude).worktrees/",
    ):
        raise ValueError("source worktree is dirty")
    return head


def _validate_preset(plan: Plan) -> None:
    if plan.track == "dashboard":
        valid = (
            plan.allowed_paths == ("dashboard/src/",)
            and plan.primary_metric == "checks"
            and plan.direction == "pass"
            and plan.threshold == 1
            and plan.heldout_role is None
        )
    else:
        valid = (
            plan.allowed_paths == (_HOOK,)
            and plan.primary_metric == "exact_objective_delta"
            and plan.direction == "lower"
            and plan.threshold <= 0
        )
    if not valid:
        raise ValueError("plan does not match the fixed track preset")


def write_plan_from_preset(track: str, objective: str, output: Path) -> int:
    baseline = _clean_source(_source_repo())
    plan = Plan(
        schema_version=1,
        track=track,
        objective=objective,
        baseline_commit=baseline,
        allowed_paths=("dashboard/src/",) if track == "dashboard" else (_HOOK,),
        primary_metric="checks" if track == "dashboard" else "exact_objective_delta",
        direction="pass" if track == "dashboard" else "lower",
        threshold=1 if track == "dashboard" else 0,
        max_candidates=2,
        wall_seconds=1800,
    )
    output = Path(output)
    _safe_directory(output.parent)
    with output.open("x", encoding="utf-8") as file:
        file.write(plan.model_dump_json(indent=2) + "\n")
    return 0


def _remaining_budget(plan_dir: Path, plan: Plan) -> float:
    """Charge active time across runs; an interrupted run consumes elapsed time."""
    consumed = 0.0
    for start_path in plan_dir.glob("run-*.json"):
        start = json.loads(start_path.read_bytes())
        finish_path = start_path.with_suffix(".finished")
        if finish_path.exists():
            consumed += float(finish_path.read_text())
        else:
            consumed += max(0, time.time() - start["started_epoch"])
    return plan.wall_seconds - consumed


def _freeze_plan(directory: Path, plan: Plan) -> None:
    path = directory / "plan.json"
    if path.exists():
        if load_plan(path) != plan:
            raise ValueError("frozen plan mismatch")
    else:
        with path.open("x", encoding="utf-8") as file:
            file.write(plan.model_dump_json(indent=2) + "\n")


def _parent(root: Path, parent: str | None, plan: Plan) -> dict | None:
    if parent is None:
        return None
    record = _read_candidate_evidence(root, parent)
    if record["request"]["plan_digest"] != plan_digest(plan):
        raise ValueError("parent candidate belongs to another plan")
    result = record["result"]
    if result is None or not result.get("patch_digest"):
        raise ValueError("parent has no validated patch")
    patch_path = root / parent / "patch.diff"
    if patch_path.is_symlink() or _digest(patch_path.read_bytes()) != result["patch_digest"]:
        raise ValueError("parent patch digest mismatch")
    if plan.track == "research" and result.get("heldout_role"):
        raise ValueError("parent has visible held-out feedback; use a new plan and fresh role")
    return record


def _assert_patch(worktree: Path, plan: Plan, patch: bytes) -> None:
    if _head(worktree) != plan.baseline_commit.lower():
        raise ValueError("candidate baseline changed during evaluation")
    paths = parse_status_z(
        _git(worktree, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    )
    if any(not is_allowed_path(worktree, path, plan.allowed_paths) for path in paths):
        raise ValueError("evaluation changed protected paths")
    if _git(worktree, "diff", "HEAD", "--binary", "--no-ext-diff") != patch:
        raise ValueError("candidate patch changed during evaluation")
    if _git(worktree, "ls-files", "--others", "--exclude-standard", "-z"):
        raise ValueError("evaluation created untracked source")


def _read_baseline(directory: Path, digest: str, baseline_commit: str) -> dict:
    """Authenticate the complete baseline evidence graph, including path parents."""
    _check_root(directory)
    for name in ("result.json", "result.sha256"):
        _check_root(directory / name)
    result = _read_record(directory, "result")
    if result["plan_digest"] != digest or result["baseline_commit"] != baseline_commit:
        raise ValueError("baseline plan or source mismatch")
    for check in result["checks"]:
        item = CheckResult.model_validate(check)
        if _digest(_evidence_bytes(directory, item.log_path)) != item.log_sha256:
            raise ValueError("baseline log digest mismatch")
    for name, digest in result.get("artifacts", {}).items():
        if _digest(_evidence_bytes(directory, name)) != digest:
            raise ValueError("baseline artifact digest mismatch")
    return result


def _baseline(plan: Plan, worktree: Path, directory: Path, remaining) -> dict:
    if (directory / "result.json").exists():
        return _read_baseline(directory, plan_digest(plan), plan.baseline_commit)
    # An interrupted baseline is never silently rerun under the same plan.
    directory.mkdir()
    if plan.track == "dashboard":
        observation = dashboard.observe_dashboard(worktree, directory, remaining())
    else:
        checks = run_checks(plan.track, worktree, remaining(), record_dir=directory)
        observation = {"checks": [check.model_dump(mode="json") for check in checks]}
    _clean_source(worktree, plan.baseline_commit)
    result = {
        **observation,
        "plan_digest": plan_digest(plan),
        "baseline_commit": plan.baseline_commit,
        "observed_at": datetime.now(UTC).isoformat(),
    }
    _write_record(directory, "result", result)
    return result


def render_candidate(record: dict) -> str:
    request, result = record["request"], record["result"]
    lines = [
        f"# Candidate {request['id']}",
        "",
        request["objective"],
        "",
        f"Track: {request['track']}",
        f"Baseline: {request['baseline_commit']}",
        f"Plan: {request['plan_digest']}",
    ]
    if result:
        lines += [
            f"Execution: {result['execution']}",
            f"Verification: {result['verification']}",
            f"Research outcome: {result.get('research_outcome', 'not_applicable')}",
            "",
            result.get("recommendation", "Recommendation unavailable."),
            "",
            f"Patch: patch.diff ({result.get('patch_digest', 'unavailable')})",
            f"Baseline observation: {result.get('baseline_record', 'unavailable')}",
        ]
        if result.get("diagnostic"):
            lines += ["", result["diagnostic"]]
        if request["track"] == "dashboard":
            lines += [
                f"Visual evidence: {result.get('visual_evidence', 'unavailable')}; "
                "owner review required."
            ]
        else:
            lines += ["Scope: three-site fixture only; no hardware evidence."]
        lines += ["", "Baseline checks:"]
        lines += [
            f"- {check['name']}: {check['execution']} / {check['verification']}"
            for check in result.get("baseline_checks", [])
        ]
        lines += ["", "Candidate checks:"]
        lines += [
            f"- {check['name']}: {check['execution']} / {check['verification']}"
            for check in result.get("checks", [])
        ]
    reviews = record.get("reviews", [])
    lines += ["", f"Review: {reviews[-1]['decision'] if reviews else 'proposed'}"]
    return "\n".join(lines) + "\n"


def run_candidate(
    plan: Plan,
    output_dir: Path,
    manual_patch: Path | None = None,
    recommendation_file: Path | None = None,
    parent: str | None = None,
) -> int:
    """Generate one observation; failures consume budget and remain reviewable."""
    try:
        if (manual_patch is None) != (recommendation_file is None):
            raise ValueError("manual mode requires both --manual-patch and --recommendation-file")
        _validate_preset(plan)
        repo = _source_repo()
        _clean_source(repo, plan.baseline_commit)
        root = _safe_directory(output_dir)
        plan_dir = _safe_directory(root / ".plans" / plan_digest(plan).removeprefix("sha256:"))
        # Different lock directory from create_candidate's count-and-create lock.
        with _plan_lock(plan_dir, plan_digest(plan)):
            _freeze_plan(plan_dir, plan)
            parent_record = _parent(root, parent, plan)
            heldout_dir = root / "heldout" / plan_digest(plan).removeprefix("sha256:")
            if plan.track == "research" and heldout_dir.exists() and any(heldout_dir.iterdir()):
                raise ValueError("held-out role already exposed under this plan; use a new plan")
            seconds = _remaining_budget(plan_dir, plan)
            if seconds <= 0:
                raise TimeoutError("plan wall time exhausted")
            candidate = create_candidate(root, plan, parent)
            directory = root / candidate.id
            started = time.monotonic()
            deadline = started + seconds
            run_path = plan_dir / f"run-{candidate.id}.json"
            with run_path.open("x", encoding="utf-8") as file:
                json.dump({"started_epoch": time.time()}, file)

            def remaining():
                value = deadline - time.monotonic()
                if value <= 0:
                    raise TimeoutError("plan wall time exhausted")
                return value

            result = {
                "schema_version": 1,
                "execution": "failed",
                "verification": "inconclusive",
                "research_outcome": "not_applicable"
                if plan.track == "dashboard"
                else "inconclusive",
                "checks": [],
                "started_at": datetime.now(UTC).isoformat(),
            }
            try:
                _freeze_plan(directory, plan)
                baseline_worktree = create_candidate_worktree(
                    repo, plan.baseline_commit, str(uuid.uuid4())
                )
                candidate_worktree = create_candidate_worktree(
                    repo, plan.baseline_commit, candidate.id
                )
                result["source_state"] = {
                    "baseline_worktree": str(baseline_worktree),
                    "candidate_worktree": str(candidate_worktree),
                    "baseline_commit": plan.baseline_commit,
                }
                if parent_record:
                    import_manual_patch(
                        candidate_worktree, root / parent / "patch.diff", plan.allowed_paths
                    )
                if manual_patch is not None:
                    recommendation = read_recommendation(Path(recommendation_file))
                    if not recommendation.strip():
                        raise ValueError("manual recommendation is empty")
                    import_manual_patch(candidate_worktree, manual_patch, plan.allowed_paths)
                    result["proposer"] = {"identity": "manual"}
                else:
                    prompt = build_proposer_prompt(
                        plan.objective,
                        plan.allowed_paths,
                        tuple(" ".join(argv) for _, argv, _ in CATALOG[plan.track]),
                    )
                    prompt += "\nFrozen plan: " + plan.model_dump_json() + "\n"
                    if parent_record:
                        feedback = [
                            {
                                key: check[key]
                                for key in ("name", "execution", "verification", "log_tail")
                            }
                            for check in parent_record["result"].get("checks", [])
                        ]
                        prompt += "\nDevelopment checks: " + json.dumps(feedback) + "\n"
                    recommendation = run_codex(
                        candidate_worktree, prompt, remaining(), baseline=plan.baseline_commit
                    )
                    result["proposer"] = {"identity": "codex", "version": "unavailable"}
                remaining()
                with (directory / "recommendation.md").open("x", encoding="utf-8") as file:
                    file.write(recommendation)
                result["recommendation"] = recommendation
                patch = export_checked_patch(candidate_worktree, plan.allowed_paths)
                with (directory / "patch.diff").open("xb") as file:
                    file.write(patch)
                result["patch_digest"] = _digest(patch)
                baseline_dir = plan_dir / "baseline"
                observation = _baseline(plan, baseline_worktree, baseline_dir, remaining)
                result["baseline_record"] = str(baseline_dir.relative_to(root) / "result.json")
                result["baseline_record_digest"] = _digest(
                    (baseline_dir / "result.json").read_bytes()
                )
                baseline_checks = [
                    CheckResult.model_validate(check) for check in observation["checks"]
                ]
                result["baseline_checks"] = [
                    {
                        "name": check.name,
                        "execution": check.execution,
                        "verification": check.verification,
                    }
                    for check in baseline_checks
                ]
                if plan.track == "dashboard":
                    result.update(
                        dashboard.evaluate_dashboard(
                            plan,
                            baseline_worktree,
                            candidate_worktree,
                            record_dir=directory,
                            baseline_checks=baseline_checks,
                            seconds=remaining(),
                        )
                    )
                    result["baseline_visual_evidence"] = observation["visual_evidence"]
                    if (
                        observation["visual_evidence"] != "available"
                        and result["verification"] == "passed"
                    ):
                        result["verification"] = "inconclusive"
                    result["check_record_dir"] = "candidate"
                    result["artifacts"] = {
                        f"candidate/{name}": digest for name, digest in result["artifacts"].items()
                    }
                else:
                    if plan.heldout_role:
                        claim_heldout(root, plan_digest(plan), plan.heldout_role, candidate.id)
                        result["heldout_role"] = plan.heldout_role
                    science = evaluate_three_site(
                        plan, baseline_worktree, candidate_worktree, seconds=remaining()
                    )
                    result["science"] = science
                    result["research_outcome"] = science["research_outcome"]
                    checks = run_checks(
                        plan.track,
                        candidate_worktree,
                        remaining(),
                        record_dir=directory / "candidate",
                    )
                    result["checks"] = [check.model_dump(mode="json") for check in checks]
                    result["check_record_dir"] = "candidate"
                    result["execution"] = dashboard.execution_status(baseline_checks + checks)
                    result["verification"] = compare_baseline(plan, baseline_checks, checks)
                    if science["verification"] != "passed":
                        result["verification"] = "inconclusive"
                _assert_patch(candidate_worktree, plan, patch)
                _clean_source(baseline_worktree, plan.baseline_commit)
                _clean_source(repo, plan.baseline_commit)
                remaining()
            except Exception as error:
                message = str(error)
                status = (
                    "timed_out"
                    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired))
                    or "timed out" in message
                    else "unavailable"
                    if isinstance(error, FileNotFoundError) or "unavailable" in message
                    else "failed"
                )
                if plan.track == "research":
                    result["research_outcome"] = "inconclusive"
                result.update(
                    execution=status,
                    verification="inconclusive" if status != "failed" else "failed",
                    diagnostic=f"{type(error).__name__}: {message}"[-4096:],
                )
            finally:
                elapsed = time.monotonic() - started
                with run_path.with_suffix(".finished").open("x", encoding="ascii") as file:
                    file.write(str(elapsed))
            artifacts = result.setdefault("artifacts", {})
            for name in ("plan.json", "recommendation.md", "patch.diff"):
                artifact = directory / name
                if artifact.is_file():
                    artifacts[name] = _digest(artifact.read_bytes())
            result.update(finished_at=datetime.now(UTC).isoformat(), duration_seconds=elapsed)
            if len(_json_bytes(result)) > MAX_RESULT_BYTES:
                result = {
                    "schema_version": 1,
                    "execution": "failed",
                    "verification": "failed",
                    "research_outcome": "inconclusive"
                    if plan.track == "research"
                    else "not_applicable",
                    "checks": [],
                    "diagnostic": "result is too large for dashboard review",
                    "started_at": result["started_at"],
                    "finished_at": result["finished_at"],
                    "duration_seconds": elapsed,
                }
            record_result(root, candidate.id, result)
            with (directory / "report.md").open("x", encoding="utf-8") as file:
                file.write(render_candidate(read_candidate(root, candidate.id)))
            print(candidate.id)
            return 0 if result["execution"] == "complete" else 1
    except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
        print(f"thermo-harness: {error}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermo-harness")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-plan")
    init.add_argument("--track", required=True, choices=("research", "dashboard"))
    init.add_argument("--objective", required=True)
    init.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("plan", type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--manual-patch", type=Path)
    run.add_argument("--recommendation-file", type=Path)
    run.add_argument("--parent")
    for command in ("show", "review"):
        subparser = commands.add_parser(command)
        subparser.add_argument("candidate")
        subparser.add_argument("--output-dir", required=True, type=Path)
        if command == "review":
            subparser.add_argument("--decision", required=True, choices=("accepted", "rejected"))
            subparser.add_argument("--note", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-plan":
            return write_plan_from_preset(args.track, args.objective, args.output)
        if args.command == "run":
            return run_candidate(
                load_plan(args.plan),
                args.output_dir,
                args.manual_patch,
                args.recommendation_file,
                args.parent,
            )
        if args.command == "show":
            print(
                render_candidate(_read_candidate_evidence(args.output_dir, args.candidate)), end=""
            )
        else:
            _check_root(args.output_dir)
            append_review(args.output_dir, args.candidate, args.decision, args.note)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"thermo-harness: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
