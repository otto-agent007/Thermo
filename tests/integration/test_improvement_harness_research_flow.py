"""Real candidate subprocesses and pinned baseline objective evaluation."""

import importlib
import py_compile
import shutil
import struct
import subprocess
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
    (baseline / ".gitignore").write_text("__pycache__/\n")
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
