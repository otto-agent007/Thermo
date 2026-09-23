# Fixture objective-by-step comparison implementation plan

> **For agentic workers:** Use superpowers:executing-plans for Native execution.

**Goal:** Run the approved bounded six-arm exact comparison with replayable evidence.
**Architecture:** A bounded exact fixture evaluator, a deterministic budget-matched
runner, and a validating report/CLI. Reuse endpoint Jacobians without changing
historical scientific modules.
**Tech Stack:** Existing NumPy, Python and pytest; no dependency changes.
**Spec:** docs/experiments/fixture-objective-step-comparison.md

## Global constraints

K4; beta one; bounds [-2,2]; existing shared nine-parameter fixture; 25 rounds;
eight evaluations per round; 201 evaluations per arm; no sampled roles; no
held-out checkpoint selection. Completion follows strict persisted replay.

## Review focus

- Invalid paths that return to one particle cannot regain killed mass.
- Hidden-state summation must precede visible-path KL.
- Equal evaluator budgets must include rejected proposals and initialization.
- Projection and no-descent rounds must be replayed, not silently skipped.
- Forged trial records with repaired hashes must fail numerical reconstruction.

## Task 1: Exact evaluator

Create src/thermo_lab/fixture_objectives.py and tests/unit/test_fixture_objectives.py.
Interface: evaluate(parameters) returns metrics and gradients for all three
objectives. `parameters` is exactly nine finite real float64 values in [-2,2].

- [x] Write tests which fail on absent evaluator, finite-difference all nine
  coordinates of all three objectives, compare occupancy to existing terminal_law
  and survival to survival_gradient. Pin path-KL >= joint-terminal >= -log S.
- [x] Implement 16-path visible enumeration, two-occurrence product derivatives,
  full terminal and killed terminal mass, and target-to-model path KL.
- [x] Reject bool/string/nonfinite/wrong-shape/out-of-bound parameters.
- [x] Run `uv run pytest tests/unit/test_fixture_objectives.py -q`; expect all pass.
- [x] Commit exact evaluator and tests.

## Task 2: Budget-matched runner and strict release

Create src/thermo_lab/fixture_objective_study.py and
tests/unit/test_fixture_objective_study.py. Consume evaluate(parameters).
Produce build_study(), validate_study(record), render_report(record), run_study(path).

- [x] Test absent runner, arm ordering, 201 evaluations per arm, Armijo selection,
  fixed sequential updates, parameter projection and no-descent behavior.
- [x] Implement protocol literally; retain all metrics/parameters/trials; run
  finite-difference preflight before all six arms. No configurable study budget.
- [x] Test rehashed mutations to selections, metrics, request and counts fail;
  test CLI fresh destination, reload and completion-last behavior.
- [x] Run both focused files; expect all pass. Commit runner/tests.

## Task 3: Generate, review, verify and record

- [x] Generate fresh evidence via `uv run python -m thermo_lab.fixture_objective_study
  --output-dir results/fixture-objective-study` and require six arms, 1206 calls,
  zero samples, strict replay and completion.
- [x] Obtain one independent whole-branch scientific/code review; fix blockers
  with regression tests. Record any limitations, not just favorable results.
- [x] Run required repository gates on a stable committed checkout. Archive the
  compressed study, summary, provenance, reviews and verification; add README
  and roadmap references without changing historical claims.

## Execution ruling

The user approved the proposed comparison and previously selected Native execution.
Proceed without another approval loop for routine numeric protocol choices; the
fixed protocol above makes those choices reviewable before any results exist.
Publication remains blocked pending explicit permission from the earlier push
rejection. No publication or merge is attempted as part of this implementation.

Ruling: the two path-aware objectives coincide on the required fixture. Preserve
the approved six-arm layout but report only two distinct objectives. Cost if
wrong: overstating the diagnostic's ability to separate path and endpoint
fidelity. An explicit algebraic-equivalence regression test prevents that claim.

## Completed outcome

Implementation commits `482ece7` and `386338d` complete the evaluator, runner,
CI gate and equivalence disclosure. Independent review found no blockers.
All 2,094 regression tests and 25 experiment gates passed on clean `386338d`.
The complete evidence is archived under
`docs/experiment-reports/2026-09-21-fixture-objective-step-comparison/`.
The two-operation result improves survival but does not meet full-program
quality requirements; reverse-context coverage and objective equivalence are
explicit limitations. Work is preserved locally; publication remains pending.
