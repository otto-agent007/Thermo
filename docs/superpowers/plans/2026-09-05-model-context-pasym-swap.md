# One-Pass Mean-Field Model-Context PAsymSwap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checked, one-pass mean-field model-context PAsymSwap compiler study without changing independent or exact target-context behavior.

**Architecture:** Rebuild target-context artifacts from their checked TOML, derive a 500-occurrence first-moment trace from their exact local conditionals, pool 37 profiles, and compile a third artifact variant. A dedicated result contract, backend, aggregate branch, and report render all diagnostic findings without claiming a full composed rollout.

**Tech Stack:** Python 3.11, NumPy float64, SciPy L-BFGS-B, Pydantic 2, JAX/THRML 0.1.4, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-05-model-context-pasym-swap-design.md`

## Global Constraints

- Use experiment ID `thrml.model_context_pasym_swap_compilation.v1`, CPU release seeds `0,1,2`, and the exact policy literals in the approved spec.
- Rebuild upstream target-context artifacts; never accept a result directory as input.
- Use float64, fixture order, and `math.fsum`; do not clip, smooth, threshold, or renormalize.
- Model contexts are `software_simulation` approximation evidence, not exact full-joint or hardware evidence.
- Require 500 occurrences, 37 sorted profiles, multiplicities `26 × 10`, `9 × 20`, and `2 × 30`.
- Compile once only: warm start from target-context winner, then zero, positive, antithetic-negative.
- Gate only own-model-profile KL, local K30, optimizer validity, and model-artifact THRML agreement. Make target-profile changes and expected-particle drift required, non-gating evidence.

---

### Task 1: Canonical mean-field model trace

**Files:**
- Create: `src/thermo_lab/model_context.py`
- Create: `tests/unit/test_model_context.py`

**Interfaces:**
- Produces frozen `ModelContextOccurrence`, `ModelContextTrace`, and `PooledModelContextProfile`.
- Produces `derive_model_context_trace(fixture, target_artifacts, *, initial_occupancy)` and `pool_model_context_profiles(trace)`.

- [ ] **Step 1: Write failing trace tests**

```python
def test_model_trace_uses_factorized_endpoint_means() -> None:
    trace = derive_model_context_trace(fixture, artifacts, initial_occupancy=(1.0,) + (0.0,) * 24)
    assert trace.occurrences[0].context_weights == (0.0, 0.0, 1.0, 0.0)


def test_model_trace_uses_artifact_conditional_not_paper_target() -> None:
    assert (
        altered_trace.occurrences[1].context_weights != paper_trace.occurrences[1].context_weights
    )
```

Also test disjoint and overlapping updates, invalid tables/means, no mutation, canonical order, trace hash stability, 500 occurrences, 37 profiles, and the checked multiplicity histogram.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/unit/test_model_context.py -q`

Expected: import failure for `thermo_lab.model_context`.

- [ ] **Step 3: Implement the exact first-moment update**

```python
context = ((1.0 - qi) * (1.0 - qj), (1.0 - qi) * qj, qi * (1.0 - qj), qi * qj)
next_i = math.fsum(context[row] * (conditional[row][2] + conditional[row][3]) for row in range(4))
next_j = math.fsum(context[row] * (conditional[row][1] + conditional[row][3]) for row in range(4))
```

Validate each probability and record endpoint means, upstream artifact hash, and expected-occupancy diagnostics. Pool sorted target hashes with `math.fsum`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest tests/unit/test_model_context.py -q`

```bash
git add src/thermo_lab/model_context.py tests/unit/test_model_context.py
git commit -m "feat: derive mean-field model contexts"
```

### Task 2: Checked model-context configuration and dispatch

**Files:**
- Create: `configs/experiments/thrml-model-context-pasym-swap.toml`
- Create: `src/thermo_lab/experiments/model_context_pasym_swap.py`
- Modify: `src/thermo_lab/config.py`, `src/thermo_lab/schemas.py`, `src/thermo_lab/experiments/__init__.py`
- Create: `tests/unit/test_model_context_pasym_swap_schemas.py`

**Interfaces:**
- Produces the model-context constants, strict run schema, validator, non-seed hash, and experiment spec factory.

- [ ] **Step 1: Write failing checked-input tests**

Assert the exact experiment ID and policy literals, shared target-context model/sampler settings, seed-independent non-seed hash, and rejection of every altered policy literal, threshold, float encoding, unknown field, or invalid seed.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/unit/test_model_context_pasym_swap_schemas.py -q`

Expected: missing configuration/schema/factory failure.

- [ ] **Step 3: Implement TOML, schema, and dispatch**

Keep existing schemas closed; branch supported-experiment validation explicitly. Set context source to `mean_field_model_pre_gate`, trace policy to `one_pass_first_moment_factorization`, upstream policy to `rebuild_checked_target_context_artifacts`, and warm-start policy to `paired_target_context_artifact_then_three_fixed_restarts`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest tests/unit/test_model_context_pasym_swap_schemas.py tests/unit/test_target_context_pasym_swap_schemas.py -q`

```bash
git add configs/experiments/thrml-model-context-pasym-swap.toml src/thermo_lab/config.py src/thermo_lab/schemas.py src/thermo_lab/experiments tests/unit/test_model_context_pasym_swap_schemas.py
git commit -m "feat: add checked model-context experiment"
```

### Task 3: Model-context compiler and result contract

**Files:**
- Create: `src/thermo_lab/model_context_compiler.py`, `src/thermo_lab/model_context_pasym_swap_results.py`
- Create: `tests/unit/test_model_context_compiler.py`, `tests/unit/test_model_context_pasym_swap_results.py`

**Interfaces:**
- Produces `ModelContextCompiledKernelArtifact`, `compile_model_context(...)`, deterministic three-way result models, and deep persisted-record validation.

- [ ] **Step 1: Write failing tests**

Test four starts and exact roles, upstream warm-start equality, optimizer endpoint checks, lexicographic ties, identity scope excluding timing, altered trace/profile/hash/conditional rejection, own-model-profile KL gates, exact K30 gates, and visible non-gating target degradation.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/unit/test_model_context_compiler.py tests/unit/test_model_context_pasym_swap_results.py -q`

Expected: missing modules.

- [ ] **Step 3: Implement compilation and pure validation**

Reuse `loss_and_gradient`, `project_gradient`, target-context compiler settings, and exact local evaluators. Hash scientific inputs only. Recompute both target-profile and model-profile KL/TV, occurrence-weighted reductions, trace-drift summaries, artifact identities, and acceptance without rerunning SciPy or THRML.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest tests/unit/test_model_context_compiler.py tests/unit/test_model_context_pasym_swap_results.py -q`

```bash
git add src/thermo_lab/model_context_compiler.py src/thermo_lab/model_context_pasym_swap_results.py tests/unit/test_model_context_compiler.py tests/unit/test_model_context_pasym_swap_results.py
git commit -m "feat: compile and validate model-context artifacts"
```

### Task 4: Dedicated backend, runner, aggregate, and report

**Files:**
- Create: `src/thermo_lab/backends/thrml_model_context_pasym_swap.py`, `src/thermo_lab/model_context_pasym_swap_reporting.py`
- Modify: `src/thermo_lab/backends/__init__.py`, `src/thermo_lab/runner.py`, `src/thermo_lab/aggregate.py`, `src/thermo_lab/reporting.py`, `src/thermo_lab/record_schemas.py`
- Create: `tests/integration/test_thrml_model_context_pasym_swap_backend.py`, `tests/integration/test_model_context_pasym_swap_runner.py`
- Modify: `tests/unit/test_aggregation.py`

- [ ] **Step 1: Write failing lifecycle/backend tests**

Assert fresh runs rebuild upstream artifacts, caches are deeply revalidated, only model artifacts are sampled with separate deterministic keys, 4096-chain K30 empirical residuals are gated, seed failure cannot fabricate output, report validates before atomic write, aggregate marker publishes last, and hostile persisted text renders inertly.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/integration/test_thrml_model_context_pasym_swap_runner.py tests/integration/test_thrml_model_context_pasym_swap_backend.py -q`

Expected: dispatch/backend/renderer failures.

- [ ] **Step 3: Implement dedicated branches**

Build the upstream target-context lineage in-memory, derive exactly one model trace, compile exactly one model variant, and use shared THRML sampler primitives. Report target-fit, model-fit, shift, occupancy drift, empirical evidence, and the no-full-rollout boundary in separate sections. Preserve independent and target-context paths unchanged.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest tests/integration/test_thrml_model_context_pasym_swap_runner.py tests/integration/test_thrml_model_context_pasym_swap_backend.py tests/unit/test_aggregation.py -q`

```bash
git add src/thermo_lab/backends src/thermo_lab/{runner.py,aggregate.py,reporting.py,record_schemas.py,model_context_pasym_swap_reporting.py} tests
git commit -m "feat: run and report model-context evidence"
```

### Task 5: Documentation, CI, and release verification

**Files:**
- Modify: `README.md`, `AGENTS.md`, `docs/roadmap.md`, `docs/experiments/biased-random-walk.md`, `.github/workflows/ci.yml`
- Modify: `tests/unit/test_checked_configs.py`, `tests/integration/test_cli.py`

- [ ] **Step 1: Write failing docs/package/CLI tests**

Assert package membership and CLI support for the new TOML; assert docs call the trace a one-pass mean-field diagnostic and keep full rollout and REINFORCE deferred.

- [ ] **Step 2: Implement docs and CI**

Add the model-context command after target-context CI, retain its 20-minute CPU timeout, and verify wheel/sdist membership for both target-context TOMLs.

- [ ] **Step 3: Run final verification**

```bash
uv sync --frozen
uv lock --check --offline
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run thermo-lab smoke
uv run thermo-lab run configs/experiments/thrml-model-context-pasym-swap.toml --seeds 0,1,2 --output-dir results/model-context-pasym-swap
uv build
git diff --check
```

Verify both built archives contain `configs/experiments/thrml-model-context-pasym-swap.toml`, request final review, then prepare a PR without merging or pushing unless authorized.

- [ ] **Step 4: Commit**

```bash
git add README.md AGENTS.md docs .github tests
git commit -m "docs: publish model-context compiler study"
```
