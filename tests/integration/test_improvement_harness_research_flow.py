"""Real candidate subprocesses and pinned baseline objective evaluation."""

import importlib
import py_compile
import shutil
import struct
import subprocess
import time
from pathlib import Path

import pytest

from thermo_lab.improvement_harness.plan import Plan

HOOK = "src/thermo_lab/research_candidates/three_site.py"


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def worktrees(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    source = Path(__file__).resolve().parents[2] / "src"
    shutil.copytree(source, baseline / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (baseline / ".gitignore").write_text("__pycache__/\n*.py[cod]\n")
    git(baseline, "init", "-q")
    git(baseline, "add", ".")
    git(
        baseline,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-qm",
        "base",
    )
    candidate = tmp_path / "candidate"
    git(baseline, "worktree", "add", "--detach", str(candidate), "HEAD")
    plan = Plan(
        schema_version=1,
        track="research",
        objective="bounded fixture comparison",
        baseline_commit=git(baseline, "rev-parse", "HEAD"),
        allowed_paths=(HOOK,),
        primary_metric="exact_objective_delta",
        direction="lower",
        threshold=0,
        max_candidates=2,
        wall_seconds=1800,
    )
    return plan, baseline, candidate


def evaluate(worktrees):
    research = importlib.import_module("thermo_lab.improvement_harness.research")
    return research.evaluate_three_site(*worktrees)


def test_timed_out_fixture_cannot_leave_a_child_to_write_after_return(tmp_path):
    research = importlib.import_module("thermo_lab.improvement_harness.research")
    source = tmp_path / "src"
    source.mkdir()
    spawned = tmp_path / "child-spawned"
    sentinel = tmp_path / "late-write"
    # The fixture must spawn its child well inside the deadline, even on a slow
    # CI runner with a cold bytecode cache. The child outsleeps the whole
    # deadline, so it can only write after _run_module has returned.
    deadline_seconds = 2.0
    child_delay = deadline_seconds + 0.5
    child_code = (
        f"import time; from pathlib import Path; time.sleep({child_delay}); "
        f"Path({str(sentinel)!r}).touch()"
    )
    (source / "timeout_probe.py").write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"Path({str(spawned)!r}).touch()\n"
        "time.sleep(60)\n"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        research._run_module(tmp_path, "timeout_probe", "", time.monotonic() + deadline_seconds)

    assert spawned.exists(), "the fixture must have spawned its child before timing out"
    # The child started before return, so a surviving child writes within child_delay.
    time.sleep(child_delay + 0.5)
    assert not sentinel.exists(), "a timed-out fixture left a child running"


def test_default_candidate_is_observed_zero_not_improvement(worktrees):
    result = evaluate(worktrees)
    assert result["before"] == pytest.approx(0.5246570826850282, abs=1e-15)
    assert result["delta"] == 0
    assert result["research_outcome"] == "inconclusive"
    assert result["execution"] == "complete"
    assert result["verification"] == "passed"
    assert result["scope"] == "three-site fixture only"
    assert result["evidence"] == "exact_reference"
    assert git(worktrees[1], "status", "--porcelain") == ""


@pytest.mark.parametrize(
    ("expression", "outcome"),
    [
        (
            "tuple(v - 0.25 * g for v, g in zip(fixture.model_parameters.values, gradient))",
            "improved",
        ),
        (
            "tuple(v + 0.25 * g for v, g in zip(fixture.model_parameters.values, gradient))",
            "regressed",
        ),
    ],
)
def test_valid_scientific_result_is_separate_from_verification(worktrees, expression, outcome):
    hook = worktrees[2] / HOOK
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(
        "from thermo_lab.trajectory_reinforce import build_exact_reference\n"
        "def propose_parameters(fixture):\n"
        "    gradient = build_exact_reference(fixture).score.shared\n"
        f"    return {expression}\n"
    )
    result = evaluate(worktrees)
    assert result["research_outcome"] == outcome
    assert (result["delta"] < 0) == (outcome == "improved")
    assert result["execution"] == "complete"
    assert result["verification"] == "passed"


def test_candidate_cannot_change_reference_even_with_broader_plan(worktrees):
    plan, baseline, candidate = worktrees
    (candidate / "src/thermo_lab/trajectory_reinforce_refinement.py").write_text(
        "raise RuntimeError\n"
    )
    plan = plan.model_copy(update={"allowed_paths": ("src/thermo_lab/",)})
    with pytest.raises(ValueError, match="allowed|protected|preset"):
        evaluate((plan, baseline, candidate))


def test_candidate_reference_edit_is_rejected_before_execution(worktrees):
    (worktrees[2] / "src/thermo_lab/trajectory_reinforce_refinement.py").write_text(
        "raise RuntimeError\n"
    )
    with pytest.raises(ValueError, match="allowed|protected"):
        evaluate(worktrees)


@pytest.mark.parametrize("expression", ["(0.0,) * 8", "(float('nan'),) + (0.0,) * 8", "(3.0,) * 9"])
def test_invalid_hook_output_is_rejected_by_frozen_reference(worktrees, expression):
    hook = worktrees[2] / HOOK
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(f"def propose_parameters(fixture):\n    return {expression}\n")
    with pytest.raises(ValueError, match="nine|finite|cap|JSON"):
        evaluate(worktrees)


def test_dirty_or_mismatched_baseline_is_rejected(worktrees):
    plan, baseline, candidate = worktrees
    with pytest.raises(ValueError, match="baseline"):
        evaluate((plan.model_copy(update={"baseline_commit": "a" * 40}), baseline, candidate))
    (baseline / "src/thermo_lab/trajectory_reinforce_refinement.py").write_text(
        "raise RuntimeError\n"
    )
    with pytest.raises(ValueError, match="baseline"):
        evaluate(worktrees)


def test_baseline_scoring_ignores_untracked_bytecode_cache(worktrees, tmp_path):
    source = worktrees[1] / "src/thermo_lab/improvement_harness/fixture_reference.py"
    replacement = tmp_path / "replacement.py"
    replacement.write_text(
        'print(\'{"before": 1, "after": 0, "delta": -1, "evidence": "exact_reference"}\')\n'
    )
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(exist_ok=True)
    py_compile.compile(str(replacement), cfile=str(cache), doraise=True)
    # Simulate a stale or forged cache whose timestamp and size match the clean source.
    data = cache.read_bytes()
    cache.write_bytes(
        data[:8]
        + struct.pack("<II", int(source.stat().st_mtime), source.stat().st_size)
        + data[16:]
    )
    assert git(worktrees[1], "status", "--porcelain") == ""
    result = evaluate(worktrees)
    assert result["delta"] == 0
    assert result["research_outcome"] == "inconclusive"


def test_stricter_predeclared_threshold_is_required_for_improvement(worktrees):
    plan, baseline, candidate = worktrees
    (candidate / HOOK).write_text(
        "from thermo_lab.trajectory_reinforce import build_exact_reference\n"
        "def propose_parameters(fixture):\n"
        "    gradient = build_exact_reference(fixture).score.shared\n"
        "    return tuple(v - 0.25 * g for v, g in "
        "zip(fixture.model_parameters.values, gradient))\n"
    )
    result = evaluate((plan.model_copy(update={"threshold": -1.0}), baseline, candidate))
    assert result["delta"] < 0
    assert result["research_outcome"] == "inconclusive"


def test_candidate_runtime_reference_edit_is_rejected(worktrees):
    (worktrees[2] / HOOK).write_text(
        "from pathlib import Path\n"
        "def propose_parameters(fixture):\n"
        "    Path('src/thermo_lab/trajectory_reinforce_refinement.py').write_text('')\n"
        "    return fixture.model_parameters.values\n"
    )
    with pytest.raises(ValueError, match="protected"):
        evaluate(worktrees)


@pytest.mark.parametrize(
    ("worktree_index", "module", "payload"),
    [
        (
            1,
            "fixture_reference",
            '{"before":1,"after":0,"delta":-1,"evidence":"exact_reference"}',
        ),
        (
            2,
            "fixture_candidate",
            '{"parameters":[0.25,-0.35,0.20,0.45,-0.30,-0.40,0.25,0.30,-0.20]}',
        ),
    ],
)
def test_ignored_sourceless_package_is_rejected_before_execution(
    worktrees, tmp_path, worktree_index, module, payload
):
    worktree = worktrees[worktree_index]
    package = worktree / "src/thermo_lab/improvement_harness" / module
    package.mkdir()
    replacement = tmp_path / "replacement.py"
    replacement.write_text("")
    py_compile.compile(str(replacement), cfile=str(package / "__init__.pyc"), doraise=True)
    marker = tmp_path / "forged-module-executed"
    replacement.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nprint({payload!r})\n"
    )
    py_compile.compile(str(replacement), cfile=str(package / "__main__.pyc"), doraise=True)
    assert git(worktree, "status", "--porcelain") == ""

    with pytest.raises(ValueError, match="protected.*ignored"):
        evaluate(worktrees)

    assert not marker.exists()


def test_explicit_remaining_budget_preserves_frozen_plan(worktrees, monkeypatch):
    research = importlib.import_module("thermo_lab.improvement_harness.research")
    plan, baseline, candidate = worktrees
    original = plan.model_dump_json()
    deadlines = []
    monkeypatch.setattr(research.time, "monotonic", lambda: 100.0)

    def module(worktree, name, payload, deadline):
        deadlines.append(deadline)
        if name.endswith("fixture_candidate"):
            return {"parameters": [0.0] * 9}
        return {"delta": 0.0, "evidence": "exact_reference"}

    monkeypatch.setattr(research, "_run_module", module)
    research.evaluate_three_site(plan, baseline, candidate, seconds=0.25)
    assert deadlines == [100.25, 100.25]
    assert plan.model_dump_json() == original
    with pytest.raises(TimeoutError, match="wall time"):
        research.evaluate_three_site(plan, baseline, candidate, seconds=0)


def test_cli_manual_research_uses_real_exact_comparison(worktrees, tmp_path, monkeypatch):
    from thermo_lab.improvement_harness import cli
    from thermo_lab.improvement_harness.checks import run_checks
    from thermo_lab.improvement_harness.store import read_candidate

    plan, baseline, candidate = worktrees
    monkeypatch.chdir(baseline)
    hook = candidate / HOOK
    hook.write_text(
        "from thermo_lab.trajectory_reinforce import build_exact_reference\n"
        "def propose_parameters(fixture):\n"
        "    gradient = build_exact_reference(fixture).score.shared\n"
        "    return tuple(v - 0.25 * g for v, g in "
        "zip(fixture.model_parameters.values, gradient))\n"
    )
    patch = tmp_path / "manual.diff"
    patch.write_text(git(candidate, "diff", "--binary") + "\n")
    recommendation = tmp_path / "recommendation.md"
    recommendation.write_text("One exact-fixture gradient step; no broader scientific claim.")

    def checks(track, cwd, seconds, *, record_dir):
        return run_checks(
            track,
            cwd,
            seconds,
            record_dir=record_dir,
            runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, b"fake gate", b""),
        )

    monkeypatch.setattr(cli, "run_checks", checks)
    output = tmp_path / "records"
    assert cli.run_candidate(plan, output, patch, recommendation) == 0
    directory = next(path for path in output.iterdir() if (path / "request.json").exists())
    result = read_candidate(output, directory.name)["result"]
    assert result["science"]["before"] == pytest.approx(0.5246570826850282, abs=1e-15)
    assert result["science"]["delta"] < 0
    assert result["science"]["evidence"] == "exact_reference"
    assert result["research_outcome"] == "improved"
    assert result["verification"] == "passed"
    assert cli.main(["show", directory.name, "--output-dir", str(output)]) == 0
