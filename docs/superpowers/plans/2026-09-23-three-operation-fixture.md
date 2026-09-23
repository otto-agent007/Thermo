# Three-Operation Return Fixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and archive a checked exact three-operation comparison that separates path KL from valid-terminal divergence and measures reverse-hop fidelity.

**Architecture:** Add a new path evaluator and study runner alongside the immutable two-operation modules. Reuse only the frozen conditional kernel, checked initial parameters and established canonical hashing; independently reconstruct every persisted result before declaring completion.

**Tech Stack:** Python 3.11, NumPy, the existing Torx/THRML-independent finite-sweep kernel, pytest, Ruff, uv and GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-23-three-operation-fixture-design.md`

## Global Constraints

- Three visible sites, initial state 100, occurrences ((0,1),(1,2),(0,1)), 64 complete visible paths.
- Reuse the exact existing PAsymSwap conditional, nine shared initial parameters, beta 1, K4, uniform hidden reset, hidden-then-output sweeps, float64 and bounds [-2,2].
- This is `exact_reference`; samples = 0. Keep all old M4G and two-operation requests, digests and archives unchanged.
- Three objectives: squared full-terminal occupancy, target-to-model complete-path KL, and valid-terminal divergence.
- Six arms plus unchanged baseline. Each arm gets 201 complete evaluations: 25 rounds × eight trials plus initialization. Fixed step 0.01; backtracking steps 2^-i for i=0..7 with Armijo 1e-4.
- Preserve all trials and final checkpoints. Numerical identities are diagnostics, not physical or full-program certificates.
- Write completion last after strict persisted replay. Record runtime provenance outside scientific request identity.

## Review Focus

- An invalid model path may later return to one particle: full terminal mass includes it, uninterrupted survival excludes it (Task 1 test).
- Multiple valid paths can end at the same state: sum their valid mass before computing the terminal objective (Task 1 test).
- A reverse 01 input occurs on the third edge: visit-weighted diagnostics must count it, without confusing it with 10 (Task 1 test).
- A projected proposal may have zero displacement or no Armijo descent: retain the round start and evaluate all eight proposals (Task 2 test).
- A rehashed record with altered selection, metric, schedule or numeric type must fail independent replay (Task 2 test).

## File map

- `src/thermo_lab/return_fixture_objectives.py`: new exact path enumeration, shared gradients, terminal and fidelity diagnostics.
- `tests/unit/test_return_fixture_objectives.py`: independent 64-path reference, path merging, survival, reverse inputs, gradients and invalid input tests.
- `src/thermo_lab/return_fixture_study.py`: versioned request, preflight, equal-budget arms, replay, report, CLI and completion.
- `tests/unit/test_return_fixture_study.py`: exact budget/selection, record tampering and completion-order tests.
- `docs/experiments/three-operation-return-fixture.md`: frozen user-facing protocol and interpretation limits.
- `.github/workflows/return-fixture-study.yml`: dedicated reproducible CI gate.
- `AGENTS.md`, `README.md`, `docs/roadmap.md`: command and evidence classification.
- `docs/experiment-reports/2026-09-23-three-operation-return-fixture/`: verified archive and research conclusion after the runner is complete.

### Task 1: Exact return-path evaluator

**Files:** Create `src/thermo_lab/return_fixture_objectives.py`; test `tests/unit/test_return_fixture_objectives.py`.

**Interfaces:** Consume `build_checked_fixture()`, `KernelParameters`, `finite_sweep_joint_law()`. Produce `evaluate(parameters: object) -> dict` with old `objectives`, `gradients`, `metrics` keys plus metrics `path_objective_gap`, `target_visited_rows` and `reverse_hop_01`. Export `OCCURRENCES=((0,1),(1,2),(0,1))` and `OBJECTIVES=("occupancy","trajectory_kl","valid_terminal")`.

- [ ] **Step 1: Write the failing path tests.** Use a reference loop over `product(range(4), repeat=3)` that mutates a three-bit state at each edge, multiplies the independently obtained marginal 4×4 target/model laws, and records full terminal mass and first-failure killed mass. Assert 64 enumerated paths, probability sums 1, and survival agrees with the independent `survival_gradient(..., [(0,1),(1,2),(0,1)], horizon=4, site_count=3)` reference. Assert two positive target histories finish at 100; a path that fails then returns to one particle never contributes to uninterrupted survival. Assert third-edge parent 01 has positive target visitation and label it separately from 10.

  ```python
  outputs = tuple(product(range(4), repeat=3))
  assert len(outputs) == 64
  result = evaluate(build_checked_fixture().model_parameters.values)
  reference = survival_gradient(
      np.asarray([build_checked_fixture().model_parameters.values]),
      [0, 0, 0], [(0, 1), (1, 2), (0, 1)], horizon=4, site_count=3,
  )
  assert result["metrics"]["survival"] == pytest.approx(reference["survival"], abs=1e-12)
  assert result["metrics"]["target_visited_rows"][2]["01"] > 0
  ```
- [ ] **Step 2: Run** `uv run pytest -q tests/unit/test_return_fixture_objectives.py`; expect import failure.
- [ ] **Step 3: Implement the evaluator.** Compute visible `v=joint.probabilities.reshape(4,2,4).sum(1)` and `dv=joint.jacobian.reshape(4,2,4,9).sum(1)`. For each path, update the three-site state and multiply `v[parent, output]`; differentiate with the product rule, accumulating all three occurrence derivatives into the same 9-vector. Accumulate full terminal, first-failure valid terminal and logical path distributions separately. Sum positive-target `P log(P/Q)` and `-P dQ/Q`. Compute `sum_x q_x log(q_x/m_x)` and `-sum_x q_x dm_x/m_x`. Compute occupancy residual from the full terminal law. Return exact survival, leakage, occurrence KL, unconditional 10/01 hop and asymmetry MAE, target-visitation-weighted row errors, path-objective gap and diagnostics. Reject malformed, nonfinite, non-float64-compatible or out-of-bounds nine-parameter input using the existing `_readonly_float64` guard.

  ```python
  # Inside one path's edge loop, after reading parent/output from the current state:
  next_derivative = derivative * v[parent, output] + probability * dv[parent, output]
  probability *= v[parent, output]
  derivative = next_derivative
  state[left], state[right] = divmod(output, 2)
  valid_so_far &= sum(state) == 1
  # After the third edge, always add probability to full_terminal[endpoint].
  # Add it to valid_terminal[endpoint] only when valid_so_far is true.
  ```
- [ ] **Step 4: Extend independent tests.** At initial parameters and `[0.15,-0.22,0.31,-0.11,0.27,-0.14,0.18,-0.26,0.09]`, central-difference all 27 shared-gradient components with step 1e-6 and absolute tolerance 1e-7. Independently compute conditional path divergence per endpoint and assert `trajectory_kl - valid_terminal` equals its target-weighted sum within 1e-12. Check `trajectory_kl >= valid_terminal >= -log(survival)` and bounds rejection for wrong shape, boolean, strings, NaN, infinity and 2.01.
- [ ] **Step 5: Run** `uv run pytest -q tests/unit/test_return_fixture_objectives.py` and `uv run ruff check src/thermo_lab/return_fixture_objectives.py tests/unit/test_return_fixture_objectives.py`; require zero failures. Commit evaluator and tests.

### Task 2: Replayable six-arm exact study

**Files:** Create `src/thermo_lab/return_fixture_study.py`; test `tests/unit/test_return_fixture_study.py`.

**Interfaces:** Consume Task 1 `evaluate`, `OBJECTIVES`, `OCCURRENCES`; produce `study_request() -> dict`, `gradient_preflight() -> dict`, `run_arm(objective: str, step_rule: str) -> dict`, `build_study() -> dict`, `validate_study(record: dict) -> None`, `render_report(record: dict) -> str`, `run_study(output_dir: Path) -> dict` and `main() -> None`.

- [ ] **Step 1: Write failing study tests.** Assert request schema `return_fixture_objective_step.v1`, three occurrences, six distinct arms, 25 rounds with eight trials, 201 evaluations per arm, 1,206 arm and 19 preflight evaluations, samples 0, and a final baseline retained. In each fixed arm independently replay 200 projected updates. In each backtracking arm compute the first Armijo-eligible index from all eight stored evaluations, or `None` if none passes. Monkeypatch a flat gradient and assert all 201 evaluations occur while all rounds stay at initial parameters.

  ```python
  study = build_study()
  assert study["request"]["occurrences"] == [[0, 1], [1, 2], [0, 1]]
  assert study["accounting"] == {
      "arm_evaluations": 1206, "preflight_evaluations": 19,
      "generation_evaluations": 1225, "samples": 0,
  }
  assert all(len(arm["rounds"]) == 25 and arm["evaluations"] == 201
             for arm in study["arms"])
  ```
- [ ] **Step 2: Run** `uv run pytest -q tests/unit/test_return_fixture_study.py`; expect import failure.
- [ ] **Step 3: Implement the checked runner.** Copy the old study's behavioral contract into a separate new module, replacing the evaluator and request identity; bind implementation hashes for both new modules and dependency code, the exact schedule, target, parameters, all six arms, trial policy, and accounting. On each backtracking round evaluate all eight projected candidates from the same start and accept the first passing slope/Armijo check. Record each round's gradient norm and count components clipped by projection in the trial rows. Bind canonical request and result digests. `validate_study()` compares the entire persisted record against a newly built study, not merely its hashes. Render path gap and separate survival, hop, asymmetry, occupancy and reverse-row metrics without selecting a winner.

  ```python
  for index in range(8):
      step = 2.0 ** -index
      raw = start - step * gradient
      proposal = np.clip(raw, -2.0, 2.0)
      result = evaluate(proposal)  # evaluate even after a candidate is selected
      slope = float(gradient @ (proposal - start))
      if selected is None and slope < 0 and result["objectives"][objective] <= value + 1e-4 * slope:
          selected = index
      trials.append({"step": step, "parameters": proposal.tolist(),
                     "clipped_components": int(np.count_nonzero(raw != proposal)),
                     "evaluation": result})
  ```
- [ ] **Step 4: Test tampering and write order.** Deep-copy a real record, alter one of selection, objective value, budget, schedule or numeric type, repair its digest with `canonical_sha256`, and require `validate_study()` to reject each. In a fresh `tmp_path`, intercept report rendering to assert `study.json` exists while `completion.json` does not; on replay/report failure require no completion. Reusing an output directory must raise `FileExistsError`.
- [ ] **Step 5: Run** `uv run pytest -q tests/unit/test_return_fixture_study.py` and `uv run ruff check src/thermo_lab/return_fixture_study.py tests/unit/test_return_fixture_study.py`; require zero failures. Commit runner and tests.

### Task 3: Dedicated gate and protocol documentation

**Files:** Create `docs/experiments/three-operation-return-fixture.md`, `.github/workflows/return-fixture-study.yml`; modify `AGENTS.md`, `README.md`, `docs/roadmap.md`.

**Interfaces:** CI executes `uv run pytest tests/unit/test_return_fixture_objectives.py tests/unit/test_return_fixture_study.py`, then `uv run python -m thermo_lab.return_fixture_study --output-dir results/return-fixture-study` and checks `completion.json` status `return_fixture_study_complete`, arms 6, samples 0.

- [ ] **Step 1: Write the protocol** with the frozen edges, 64-path enumeration, objectives, equality of evaluator budgets, metrics, first-failure survival, exact-reference label and interpretation limits from the approved spec. Put the new CLI command in `AGENTS.md`; add concise README and roadmap entries as a *planned* experiment until Task 4 records results.
- [ ] **Step 2: Write the workflow** using the same `setup-python@v5`, `setup-uv@v6`, `uv sync --frozen`, CPU environment and artifact upload pattern as `.github/workflows/fixture-objective-study.yml`; replace test/module/status names exactly as in Interfaces.

  ```yaml
  - run: uv sync --frozen
  - run: uv run pytest tests/unit/test_return_fixture_objectives.py tests/unit/test_return_fixture_study.py
  - run: uv run python -m thermo_lab.return_fixture_study --output-dir results/return-fixture-study
  ```
- [ ] **Step 3: Run** `uv lock --check --offline`, `uv run ruff format --check .`, `uv run ruff check .` and the new CLI with a fresh `results/return-fixture-study` path. Inspect `completion.json` and `summary.md`; rerun record validation after JSON reload and check that the archived two-operation record remains unchanged. Commit protocol, gate and docs only after successful commands.

### Task 4: Verified evidence and scientific conclusion

**Files:** Create `docs/experiment-reports/2026-09-23-three-operation-return-fixture/{study.json.gz,summary.md,review.md,verification.md,archive-metadata.json,provenance.json,completion.json}`; modify `docs/roadmap.md`, `README.md`.

**Interfaces:** Consume Task 2 `run_study()` and `validate_study()`; archive the complete checked request/result and report with a SHA-256 of decompressed canonical bytes, exact source commit and verification command outcomes.

- [ ] **Step 1: Run the frozen study** with a fresh results directory and recompute `validate_study(json.loads(study.json))` from the persisted file. Compress canonical `study.json` losslessly; record byte count, gzip/file SHA-256 and source commit. Preserve runtime provenance separately from request identity.

  ```bash
  uv run python -m thermo_lab.return_fixture_study --output-dir results/return-fixture-study
  uv run python -c 'import json; from pathlib import Path; from thermo_lab.return_fixture_study import validate_study; validate_study(json.loads(Path("results/return-fixture-study/study.json").read_text()))'
  ```
- [ ] **Step 2: Review every arm** against the original protocol, including all six values, path-gap identity, reverse-hop visitation, survival and fidelity; separately check the CI workflow and tamper tests. Document objective-scale and single-fixture limitations. A negative or inconclusive comparison is a valid result.
- [ ] **Step 3: Run** `uv run pytest`, `uv run ruff format --check .`, `uv run ruff check .`, `uv lock --check --offline`, the new CLI from a fresh directory and strict replay of the archived gzip. Update roadmap/README only with conclusions supported by those outputs; mark the archive complete last. Commit evidence and documentation, then review `git diff --check` and the final branch tree before publishing.

## Execution ordering

Tasks 1 and 2 are sequential because the runner consumes the evaluator. Task 3 consumes the runner and Task 4 consumes its frozen output. Do not merge the design or implementation over an unmerged PR #38; rebase/reconcile with its merged `main` state at execution time. Do not amend historical M4G records or claim a hardware/sampling-budget result.
