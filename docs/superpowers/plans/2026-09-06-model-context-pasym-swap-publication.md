# Model-context PAsymSwap publication integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the checked model-context PAsymSwap study as a standard, reload-validated THRML experiment with a canonical report.

**Architecture:** Add a strict backend-independent `ModelContextPAsymSwapSummary` that joins existing exact profile evidence to sampler integer-count evidence. Route the already registered checked config to a standard `ExecutionResult`, explicitly aggregate only the empirical residual across seeds, and render only a fully validated persisted record through a dedicated reporting module.

**Tech Stack:** Python 3.11, Pydantic v2, JAX, THRML 0.1.4, NumPy, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-06-model-context-pasym-swap-publication-design.md`

## Global Constraints

- Preserve the PAsymSwap fixture, compiler schedule, K=30 horizon, 4 inputs, 4096 chains/input, and 30 THRML sweeps.
- Keep this `thrml_local` / `software_simulation` evidence; never make Z1, TSU, or physical-hardware claims.
- Derive empirical conditionals and TV residuals only from integer counts.
- Render reports from persisted evidence only; never compile, sample, or rebuild a fixture while reporting.
- Treat seeded full-study executions, rather than profiles, inputs, or chains, as replications.
- Time only synchronized sampling work; exclude loading, reconstruction, exact evaluation, compilation, warm launch, and persistence.

---

## File structure

| File | Responsibility |
|---|---|
| `src/thermo_lab/model_context_pasym_swap_results.py` | Typed summary, builder, and deep validator. |
| `src/thermo_lab/backends/thrml_model_context_pasym_swap.py` | Standard execution, provenance, and sampling timing. |
| `src/thermo_lab/backends/__init__.py` | Public backend export used by runner dispatch. |
| `src/thermo_lab/runner.py` | Backend selection. |
| `src/thermo_lab/aggregate.py` | Model-context compatibility identity and sampled-scalar allowlist. |
| `src/thermo_lab/model_context_pasym_swap_reporting.py` | Complete persisted-record validation and Markdown section. |
| `src/thermo_lab/reporting.py` | Generic report orchestration and cross-seed completeness text. |
| `tests/unit/test_model_context_pasym_swap_results.py` | Summary and tamper validation. |
| `tests/integration/test_thrml_model_context_pasym_swap_backend.py` | Standard backend record contract. |
| `tests/integration/test_model_context_pasym_swap_runner.py` | Runner persistence and report round trip. |

### Task 1: Strict publication-summary contract

**Files:**
- Modify: `src/thermo_lab/model_context_pasym_swap_results.py`
- Test: `tests/unit/test_model_context_pasym_swap_results.py`

**Interfaces:**
- Consumes: `ModelContextProfileResult`, `ModelContextScheduleAcceptance`, exact K=30 conditional tables, integer count tables, and `SampledK30Evaluation`; it does not import backend dataclasses.
- Produces: `ModelContextPAsymSwapSummary`, `build_model_context_pasym_swap_summary`, and `validate_model_context_pasym_swap_summary(value)`.

- [ ] **Step 1: Write failing round-trip and count-tamper tests**

```python
def test_model_context_publication_summary_recomputes_empirical_evidence() -> None:
    exact, sampled, run = _evidence_fixture()
    summary = build_model_context_pasym_swap_summary(
        request_hash="sha256:" + "0" * 64,
        profile_results=exact.profile_results,
        schedule_acceptance=exact.acceptance,
        profile_samples=_normalized_samples(sampled),
        thrml_k30_tv_tolerance=run.thrml_k30_tv_tolerance,
    )
    assert validate_model_context_pasym_swap_summary(summary.model_dump(mode="json")) == summary
    assert len(summary.profile_samples) == 37


def test_model_context_publication_summary_rejects_changed_count() -> None:
    payload = _publication_summary_fixture().model_dump(mode="json")
    payload["profile_samples"][0]["sampled_k30"]["counts"][0][0] -= 1
    with pytest.raises(ValueError, match="counts|conditional|residual|digest"):
        validate_model_context_pasym_swap_summary(payload)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_model_context_pasym_swap_results.py -k publication_summary -v`

Expected: FAIL because the summary builder and validator do not exist.

- [ ] **Step 3: Implement the immutable summary and builder**

```python
class ModelContextPAsymSwapSummary(_StrictFrozenResultModel):
    request_hash: str
    model_trace_hash: str
    profile_results: tuple
    profile_samples: tuple
    schedule_acceptance: ModelContextScheduleAcceptance
    maximum_empirical_k30_residual: StrictFloat
    thrml_k30_tv_tolerance: StrictFloat
    exact_acceptance_passed: StrictBool
    empirical_acceptance_passed: StrictBool
    acceptance_passed: StrictBool
    deterministic_result_hash: str
    summary_hash: str


def build_model_context_pasym_swap_summary(
    *,
    request_hash: str,
    profile_results: Sequence[ModelContextProfileResult],
    schedule_acceptance: ModelContextScheduleAcceptance,
    profile_samples: Sequence[ModelContextProfileSampleResult],
    thrml_k30_tv_tolerance: float,
) -> ModelContextPAsymSwapSummary:
    return _build_checked_model_context_summary(
        request_hash,
        profile_results,
        profile_samples,
        schedule_acceptance,
        thrml_k30_tv_tolerance,
    )


def validate_model_context_pasym_swap_summary(value: object) -> ModelContextPAsymSwapSummary:
    parsed = ModelContextPAsymSwapSummary.model_validate(_json_tuple(value))
    return _recompute_checked_model_context_summary(parsed)
```

`ModelContextProfileSampleResult` retains target/profile/artifact hashes, the exact K=30
reference conditional, and the typed `SampledK30Evaluation`. The validator uses existing
exact validators, requires canonical one-to-one 37-profile ordering, checks that the
reference conditional has the persisted exact equilibrium residual, recomputes every
empirical evaluation from counts, recomputes maxima and flags, then matches every
regenerated value and both hashes. `deterministic_result_hash` covers the request,
exact profile/schedule evidence, model artifact identities, and exact K=30 references;
`summary_hash` additionally covers the seeded counts and sampled results.

- [ ] **Step 4: Add all remaining contradiction tests**

```python
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: p["profile_samples"][0].__setitem__(
                "model_context_artifact_hash", "sha256:" + "f" * 64
            ),
            "artifact",
        ),
        (lambda p: p.__setitem__("maximum_empirical_k30_residual", 0.0), "residual"),
        (lambda p: p.__setitem__("empirical_acceptance_passed", False), "acceptance"),
        (lambda p: p.__setitem__("thrml_k30_tv_tolerance", 0.01), "tolerance"),
    ],
)
def test_model_context_publication_summary_rejects_tampering(mutate, message) -> None:
    payload = _publication_summary_fixture().model_dump(mode="json")
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        validate_model_context_pasym_swap_summary(payload)
```

- [ ] **Step 5: Run the unit module**

Run: `uv run pytest tests/unit/test_model_context_pasym_swap_results.py -v`

Expected: PASS, including existing exact-evidence tests.

- [ ] **Step 6: Commit the evidence contract**

```bash
git add src/thermo_lab/model_context_pasym_swap_results.py tests/unit/test_model_context_pasym_swap_results.py
git commit -m "feat: persist model-context PAsymSwap evidence"
```

### Task 2: Standard backend execution and dispatch

**Files:**
- Modify: `src/thermo_lab/backends/__init__.py`
- Modify: `src/thermo_lab/runner.py`
- Modify: `src/thermo_lab/backends/thrml_model_context_pasym_swap.py`
- Modify: `tests/integration/test_thrml_model_context_pasym_swap_backend.py`
- Create: `tests/integration/test_model_context_pasym_swap_runner.py`

**Interfaces:**
- Consumes: Task 1 summary builder and existing `ExperimentBackend` / `ExecutionResult`.
- Produces: `ThrmlModelContextPAsymSwapBackend.run(spec) -> RunRecord` and `.execute(spec) -> ExecutionResult`.

- [ ] **Step 1: Write failing dispatch and standard-record tests**

```python
def test_backend_dispatches_model_context_exact_id() -> None:
    configured = load_experiment_config(CONFIG)
    assert isinstance(_backend(configured, ROOT), ThrmlModelContextPAsymSwapBackend)


def test_model_context_backend_emits_one_standard_run_record() -> None:
    result = ThrmlModelContextPAsymSwapBackend().execute(model_context_pasym_swap_spec(seed=0))
    summary = validate_model_context_pasym_swap_summary(
        result.record.metrics["model_context_pasym_swap_summary"].value
    )
    assert result.record.evidence_class is EvidenceClass.SOFTWARE_SIMULATION
    assert summary.acceptance_passed
    assert result.record.timing.synchronized
```

- [ ] **Step 2: Run the focused tests to verify failure**

Run: `uv run pytest tests/integration/test_model_context_pasym_swap_runner.py tests/integration/test_thrml_model_context_pasym_swap_backend.py -k 'dispatches_model_context or emits_one_standard' -v`

Expected: FAIL because runner dispatch/export and the standard backend protocol are absent.

- [ ] **Step 3: Export and route the already registered experiment**

```python
if config.experiment_id == MODEL_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
    return ThrmlModelContextPAsymSwapBackend(repository_root)
```

Import and export `ThrmlModelContextPAsymSwapBackend` in `backends.__init__` and use the
existing experiment constant. The config registry and strict request validation already
exist and must remain unchanged.

- [ ] **Step 4: Implement standard execution with one reconstruction**

```python
def run(self, spec: ExperimentSpec) -> RunRecord:
    return self.execute(spec).record


def execute(self, spec: ExperimentSpec) -> ExecutionResult:
    model, run, request_hash, _ = self.checked_request(spec)
    prepared = self.prepare(spec)
    exact = self._evaluate_prepared(model, run, prepared)
    sampled, timing = self._sample_prepared(spec, model, run, prepared)
    summary = build_model_context_pasym_swap_summary(
        request_hash=request_hash,
        profile_results=exact.profile_results,
        schedule_acceptance=exact.acceptance,
        profile_samples=self._publication_samples(sampled),
        thrml_k30_tv_tolerance=run.thrml_k30_tv_tolerance,
    )
    record = build_run_record(
        backend_id=self.backend_id,
        evidence_class=self.evidence_class,
        spec=spec,
        provenance=collect_runtime_provenance(self.repository_root),
        timing=timing,
        metrics={
            "model_context_pasym_swap_summary": MetricObservation(
                value=summary,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="bounded model-context PAsymSwap exact-plus-THRML study",
            ),
            "maximum_empirical_k30_residual": MetricObservation(
                value=summary.maximum_empirical_k30_residual,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                method="independently seeded 4096-chain THRML cross-check",
            ),
        },
    )
    return ExecutionResult.build(record)
```

Factor current APIs only as needed so one `execute` rebuilds the lineage exactly once.
Add the exact K=30 reference conditional to each in-memory profile sample. Use
`collect_runtime_provenance`, `RUN_TIMING_SOURCE`, `RunTiming`, and `MetricObservation`.
Keep direct `evaluate(spec)` and `sample(spec)` APIs; their internal prepared helpers avoid
duplicate compilation in standard execution.

- [ ] **Step 5: Assert the timing boundary and run focused tests**

```python
assert "148 keyed 4096-chain" in result.record.timing.timing_method
assert "excludes compilation" in result.record.timing.timing_method
assert result.record.timing.execution_seconds >= 0.0
```

Run: `uv run pytest tests/integration/test_model_context_pasym_swap_runner.py tests/integration/test_thrml_model_context_pasym_swap_backend.py -k 'dispatches_model_context or standard_run_record or cross_checks' -v`

Expected: PASS.

- [ ] **Step 6: Commit registered execution**

```bash
git add src/thermo_lab/backends/__init__.py src/thermo_lab/runner.py src/thermo_lab/backends/thrml_model_context_pasym_swap.py tests/integration/test_thrml_model_context_pasym_swap_backend.py tests/integration/test_model_context_pasym_swap_runner.py
git commit -m "feat: run model-context PAsymSwap through THRML"
```

### Task 3: Aggregate contract and reload-only canonical report

**Files:**
- Modify: `src/thermo_lab/aggregate.py`
- Create: `src/thermo_lab/model_context_pasym_swap_reporting.py`
- Modify: `src/thermo_lab/reporting.py`
- Modify: `tests/integration/test_model_context_pasym_swap_runner.py`

**Interfaces:**
- Consumes: persisted metric `model_context_pasym_swap_summary` and Task 1 validator.
- Produces: `validate_persisted_model_context_pasym_swap_record(record)`,
  `render_model_context_pasym_swap_section(record)`, deterministic cross-seed identity,
  and an aggregate containing only `maximum_empirical_k30_residual`.

- [ ] **Step 1: Write failing aggregate and runner/report tests**

```python
def test_model_context_runner_writes_reload_validated_evidence_report(tmp_path: Path) -> None:
    run_experiment(experiment_config_path("thrml-model-context-pasym-swap.toml"), tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Model-context PAsymSwap study" in report
    assert "37 profiles / 500 occurrences" in report
    assert "148 keyed 4096-chain" in report
    assert "not independent replications" in report
    assert "not a physical Z1 or TSU hardware measurement" in report


def test_model_context_aggregate_exposes_only_seeded_empirical_residual(
    completed_model_context_run: CompletedModelContextRun,
) -> None:
    aggregate = completed_model_context_run.aggregate
    assert set(aggregate.metric_aggregates) == {"maximum_empirical_k30_residual"}
    assert aggregate.omitted_metrics["model_context_pasym_swap_summary"] == (
        "nested model-context evidence is retained only in per-run records"
    )
    assert "timing.execution_seconds" in aggregate.omitted_metrics
```

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run pytest tests/integration/test_model_context_pasym_swap_runner.py::test_model_context_runner_writes_reload_validated_evidence_report -v`

Expected: FAIL because the dedicated section is absent.

- [ ] **Step 3: Implement persisted-summary loading and rendering**

```python
def validate_persisted_model_context_pasym_swap_record(
    record: RunRecord,
) -> tuple[ModelContextPAsymSwapSummary, PAsymSwapModelConfig, ModelContextCompilerRunConfig]:
    observation = record.metrics["model_context_pasym_swap_summary"]
    if observation.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("model-context PAsymSwap summary must be software-simulation evidence")
    model = PAsymSwapModelConfig.model_validate(to_json_value(record.spec.model_parameters))
    run = ModelContextCompilerRunConfig.model_validate(to_json_value(record.spec.run_parameters))
    validate_model_context_pasym_swap_request(model, run, record.spec.seed)
    summary = validate_model_context_pasym_swap_summary(observation.value)
    if summary.request_hash != model_context_pasym_swap_non_seed_config_hash(model, run):
        raise ValueError("model-context request hash differs from persisted checked inputs")
    if (
        record.metrics["maximum_empirical_k30_residual"].value
        != summary.maximum_empirical_k30_residual
    ):
        raise ValueError("model-context scalar residual differs from structured summary")
    return summary, model, run


def render_model_context_pasym_swap_section(record: RunRecord) -> list[str]:
    summary, model, run = validate_persisted_model_context_pasym_swap_record(record)
    return _render_model_context_profile_table(summary)
```

Before the helper above, validate exact experiment/sample identity, THRML backend and pinned
version, software evidence, synchronized timing, timing source/unit/method, and the exact
two-metric set. Render from the returned checked values only: identity and 37/500 scope; per-profile multiplicity,
KL improvement, exact and empirical K=30 residuals, pass state; acceptance gates;
sampling interpretation; and scientific caveats. Select it by exact experiment ID
without changing weighted-graph behavior.

In `aggregate.py`, mirror the target-context pattern: deep-validate each record, use
`summary.deterministic_result_hash` as compatibility identity, canonicalize the timing
compile/reuse suffix, allowlist `maximum_empirical_k30_residual`, and omit the structured
summary plus both timing scalars with explicit reasons.

- [ ] **Step 4: Write persisted-artifact tamper tests**

```python
def test_model_context_report_rejects_changed_persisted_count(tmp_path: Path) -> None:
    aggregate, record = _persisted_model_context_artifacts(tmp_path)
    payload = record.model_dump(mode="json")
    payload["metrics"]["model_context_pasym_swap_summary"]["value"]["profile_samples"][0][
        "sampled_k30"
    ]["counts"][0][0] -= 1
    with pytest.raises(ValueError, match="counts|conditional|residual|digest"):
        render_report(aggregate, (RunRecord.model_validate(payload),))
```

```python
@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("profile_samples", 0, "model_context_artifact_hash"), "sha256:" + "f" * 64, "artifact"),
        (("empirical_acceptance_passed",), False, "acceptance"),
    ],
)
def test_model_context_report_rejects_other_persisted_contradictions(
    tmp_path: Path, path: Sequence[object], value: object, message: str
) -> None:
    aggregate, record = _persisted_model_context_artifacts(tmp_path)
    payload = record.model_dump(mode="json")
    target = payload["metrics"]["model_context_pasym_swap_summary"]["value"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        render_report(aggregate, (RunRecord.model_validate(payload),))
```

- [ ] **Step 5: Run the integration module**

Run: `uv run pytest tests/integration/test_model_context_pasym_swap_runner.py -v`

Expected: PASS, including report round-trip and all tamper rejections.

- [ ] **Step 6: Commit the report layer**

```bash
git add src/thermo_lab/aggregate.py src/thermo_lab/model_context_pasym_swap_reporting.py src/thermo_lab/reporting.py tests/integration/test_model_context_pasym_swap_runner.py
git commit -m "feat: report model-context PAsymSwap evidence"
```

### Task 4: Whole-slice verification and review readiness

**Files:**
- Modify if test results require it: only files named in Tasks 1–3.

**Interfaces:**
- Consumes: the registered runner, strict summary, and report section.
- Produces: a clean pushed branch whose remote tree matches the verified local tree.

- [ ] **Step 1: Run the focused publication suite**

```bash
uv run pytest tests/unit/test_model_context_pasym_swap_results.py tests/integration/test_thrml_model_context_pasym_swap_backend.py tests/integration/test_model_context_pasym_swap_runner.py -v
```

Expected: PASS.

- [ ] **Step 2: Run format, lint, build, and whitespace gates**

```bash
uv run ruff format --check .
uv run ruff check .
uv build
git diff --check origin/feat/model-context-pasym-swap-sampling..HEAD
```

Expected: every command exits 0.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest`

Expected: PASS with no failures.

- [ ] **Step 4: Inspect final changes and commit verification fixes if needed**

```bash
git status --short
git diff --check origin/feat/model-context-pasym-swap-sampling..HEAD
git log --oneline origin/feat/model-context-pasym-swap-sampling..HEAD
```

If a verification fix is needed, add only touched Task 1–3 files and commit:
`git commit -m "fix: harden model-context publication evidence"`.

- [ ] **Step 5: Push and verify remote content**

```bash
git push -u origin feat/model-context-pasym-swap-publication
git fetch origin feat/model-context-pasym-swap-publication
git diff --exit-code HEAD origin/feat/model-context-pasym-swap-publication
```

Expected: the published remote tree matches the verified local tree exactly.
