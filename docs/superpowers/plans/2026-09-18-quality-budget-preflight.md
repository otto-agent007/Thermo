# M4G decision preflight implementation plan

**Goal:** Complete the request, randomness, accounting and statistical-decision
preflight that precedes the M4G training runner.

**Architecture:** Add separate request/accounting, acceptance and preflight
modules. Reuse the pinned M4B archive adapter and canonical hashing. The command
authenticates inputs, runs deterministic mathematical checks and replays its
persisted evidence before writing completion. It cannot fit models or consume
the held-out evaluation stream.

**Tech stack:** Existing Python 3.11, NumPy, SciPy and Pydantic dependencies.

**Spec:** `docs/experiments/task-quality-inference-budget.md`, protocol commit
`ae2ecac3be45501d52ac60113ee6ccc71e6adc0c`.

## Constraints and boundary

Preserve N=32768, seeds 0/1/2, K=1/2/4/8/16/30, five updates, alpha=0.05/1560,
all fixed quality limits, complete grid accounting and distinct evidence labels.
No model training, final evaluation, scientific outcome or full M4G completion
is claimed by this component preflight. The matched runner, law-specific
training validation and full-study release remain the next implementation.

## Tasks

- [ ] Request and accounting: create `quality_budget_protocol.py` and unit tests.
  Test immutable identity, rejected changes, 71 roles per seed, all 213 distinct,
  21 fit/60 cell manifests, local references and equilibrium costs. Interface:
  `QualityBudgetProtocol`, `role_schedule(seed)`, `fit_manifest()`,
  `evaluation_manifest()`, `cost_ledger()`. Watch missing imports fail, then
  implement and verify `pytest tests/unit/test_quality_budget_protocol.py`.
- [ ] Acceptance: create `quality_budget_acceptance.py` and tests. Interface:
  `binomial_intervals(counts, n)`, `occupancy_loss_bounds(intervals, target)`,
  `classify_bounds(loss, leakage, survival, hop_mae, asymmetry_mae)`,
  `classify_cell(...)`, `classify_method(...)`, `budget_bracket(statuses)` and
  `compare_budgets(finite_statuses, equilibrium_statuses)`. Test independent
  binomial inversion, invalid counts, exact equality/nextafter, correlated
  rectangle coverage and nonmonotone classifications. Run red then green.
- [ ] Deterministic preflight and persistence: create `quality_budget_preflight.py`
  with `build_preflight`, `validate_preflight`, `write_preflight` and module CLI.
  Check the complete declared coverage grid, source pins and common initial
  matrix. Save all requested/derived evidence, strict replay and a report; write
  completion last. Reject rehashed evidence tampering and occupied destinations
  in integration tests. Test before implementation.
- [ ] Independent statistical and implementation review; address findings.
  Record component scope and evidence in README, roadmap and a dated report.
- [ ] Preserve existing gates, add the preflight CI job, run required repository
  gates and build/package checks. Commit and publish a stacked PR against #31
  under existing automatic-push authorization; keep merge to the user.

## Review checklist

Coverage is a binomial theorem plus a union bound, not proved by a finite grid.
Infinity is serialized as null with explicit bracket status. Exact thresholds
have no tolerance. Numerical validation tolerance never changes acceptance.
Independent reference draws count as local endpoint work, not full trajectories.
Counts must be integral and consistent with the terminal histogram. No acceptance
API may silently pool seeds or use a first pass as a resolved minimum.
