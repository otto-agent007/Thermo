# Thermo Improvement Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reviewable research and dashboard improvement proposals, each with a draft patch and a reproducible comparison against a fixed baseline.

**Architecture:** A local Python CLI owns immutable plans, isolated Git worktrees, a candidate ledger, and fixed evaluators. One research evaluator scores a bounded three-site candidate through an unchanged exact reference; one dashboard evaluator runs the existing Node and browser checks. A read-only dashboard route displays validated candidate summaries without exporting local drafts into a public snapshot.

**Tech Stack:** Python 3.11, Pydantic, pytest, Git, optional local Codex CLI, Node 22.12+, React/Vite, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-23-self-improving-harness-design.md`

## Global Constraints

- Keep import-package code under `src/thermo_lab`; the Python project requires `>=3.11,<3.12`.
- Keep completed M4G records, pins, release validators and acceptance decisions byte-identical. A fixture trial never becomes an M4G claim.
- Use `exact_reference` only for the frozen exact evaluator and `software_simulation` for sampled results; never label a local run as hardware evidence.
- The user starts iterations locally. No scheduler, remote job control, automatic merge, deploy, or scientific publication.
- A candidate includes a recommendation and patch. An owner decision is stored separately and never inferred from a metric.
- Fix an evaluation plan before candidate generation. Use catalogued commands as argument arrays, not agent-supplied shell text.
- Keep local drafts under ignored `results/harness/`; the dashboard reads only allowlisted records, and static export omits unreviewed drafts.
- Follow the complete local gates in `AGENTS.md` before claiming implementation is ready; dashboard checks are additional.

## Review Focus

These are the five risky inputs to exercise in the owning tasks:

1. A baseline commit changes or has a dirty worktree between planning and scoring: reject the run (Tasks 1 and 3).
2. A patch contains an untracked protected file or symlink outside the allowed tree: reject it before evaluation (Task 2).
3. A second candidate tries to reuse a held-out role after the result is visible: reject it under that plan (Task 4).
4. A check times out or Playwright is missing: report `timed_out` or `unavailable`, never `passed` (Task 3).
5. A malformed, large or escaping local report reaches the dashboard adapter: omit it with an explicit issue and serve no path outside `results/harness` (Task 6).

## File map and interfaces

| File | Responsibility |
| --- | --- |
| `src/thermo_lab/improvement_harness/plan.py` | Frozen plan schema, canonical hash and input validation |
| `src/thermo_lab/improvement_harness/store.py` | Append-only request, observation and owner-review records |
| `src/thermo_lab/improvement_harness/workspace.py` | Candidate worktree creation, allowed-path checks and patch export |
| `src/thermo_lab/improvement_harness/proposer.py` | Local Codex CLI adapter and manual-patch intake |
| `src/thermo_lab/improvement_harness/checks.py` | Fixed command catalog, bounded process execution and statuses |
| `src/thermo_lab/improvement_harness/research.py` | Candidate vector extraction and frozen exact three-site scoring |
| `src/thermo_lab/improvement_harness/dashboard.py` | Dashboard build/browser evaluation and screenshot paths |
| `src/thermo_lab/improvement_harness/cli.py` | `thermo-harness run`, `review`, and `show` commands |
| `dashboard/server/proposals.ts` | Bounded, symlink-safe local report reader |
| `dashboard/server/plugin.ts` | Serve a validated patch as read-only text |
| `dashboard/src/features/Proposals.tsx` | Read-only proposal list and detail view |

`Plan` has `schema_version`, `track`, `objective`, `baseline_commit`, `allowed_paths`, `primary_metric`, `direction`, `threshold`, `max_candidates`, `wall_seconds`, and optional `heldout_role`. `load_plan(Path) -> Plan` rejects unknown fields and computes `plan_digest(plan) -> str` as `sha256:<hex>` from sorted canonical requested inputs. `create_candidate(root, plan, parent_id=None) -> Candidate` writes `request.json` once. `record_result(root, candidate_id, result) -> None` writes `result.json` once. `append_review(root, candidate_id, decision, note) -> Path` creates a new review file without changing earlier bytes. `run_checks(track, cwd, seconds, *, runner=subprocess.run) -> list[CheckResult]` accepts only catalogued commands. All JSON files have a version and explicit status dimensions.

## Task 1: Frozen plan and candidate ledger

**Files:** Create `src/thermo_lab/improvement_harness/__init__.py`, `plan.py`, `store.py`, `tests/unit/test_improvement_harness_plan.py`, `tests/unit/test_improvement_harness_store.py`. Runnable plans are generated under ignored `results/harness/plans/` after the implementation baseline is committed.

**Interfaces:** Produces `Plan`, `load_plan`, `plan_digest`, `create_candidate`, `record_result`, `append_review`, and `read_candidate`; later tasks consume these exact names. Candidate IDs are generated UUIDs, not timestamps or metric values. The plan digest excludes observed runtime data.

- [ ] **Step 1: Write failing plan tests.** Cover unknown fields, duplicate/absolute allowed paths, a non-commit baseline, `max_candidates < 1`, `wall_seconds <= 0`, and identical digests when only the JSON key order differs. Task 5 checks Git cleanliness when the plan is initialized and run.

```python
def test_plan_digest_ignores_json_key_order(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"schema_version":1,"track":"dashboard","objective":"labels","baseline_commit":"' + "a" * 40 + '","allowed_paths":["dashboard/src/"],"primary_metric":"checks","direction":"pass","threshold":1,"max_candidates":1,"wall_seconds":120}')
    b.write_text('{"track":"dashboard","schema_version":1,"objective":"labels","baseline_commit":"' + "a" * 40 + '","allowed_paths":["dashboard/src/"],"primary_metric":"checks","direction":"pass","threshold":1,"max_candidates":1,"wall_seconds":120}')
    assert plan_digest(load_plan(a)) == plan_digest(load_plan(b))
```

- [ ] **Step 2: Run `uv run pytest tests/unit/test_improvement_harness_plan.py -q`.** Expect import or assertion failures before implementation.
- [ ] **Step 3: Implement frozen Pydantic models and canonical hashing.** Use `ConfigDict(extra="forbid", frozen=True)` and the following canonical bytes; validate a full 40-hex SHA and relative allowlisted prefixes. Keep Git cleanliness checking in the CLI before creation.

```python
def plan_digest(plan: Plan) -> str:
    raw = json.dumps(plan.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()
```
- [ ] **Step 4: Write failing store tests.** Verify one `request.json` and one `result.json` per candidate, a second write fails, review records append without rewriting results, a second candidate is rejected when `max_candidates=1`, and unreadable or mismatched digests fail to load.

```python
candidate = create_candidate(tmp_path, plan)
record_result(tmp_path, candidate.id, {"schema_version": 1, "execution": "complete", "verification": "passed"})
with pytest.raises(FileExistsError):
    record_result(tmp_path, candidate.id, {"schema_version": 1, "execution": "failed", "verification": "failed"})
```

- [ ] **Step 5: Implement the store using exclusive file creation (`open(..., "x")`) and SHA-256 verification on read.** Write `request.sha256` and `result.sha256` sidecars beside their JSON files, with the digest of the exact bytes. Keep review entries as `review-0001.json`, `review-0002.json`; each contains candidate ID, decision, note and observed time. Reject creating more than `plan.max_candidates` for the same plan digest. Never mutate request/result files.

```python
with (candidate_dir / "request.json").open("x", encoding="utf-8") as file:
    file.write(request_json)
(candidate_dir / "request.sha256").write_text(hashlib.sha256(request_json.encode()).hexdigest() + "\n")
```
- [ ] **Step 6: Run both focused tests, `uv run ruff check src/thermo_lab/improvement_harness tests/unit/test_improvement_harness_plan.py tests/unit/test_improvement_harness_store.py`, and commit the focused change.**

## Task 2: Candidate isolation and patch intake

**Files:** Create `workspace.py`, `proposer.py`, `tests/unit/test_improvement_harness_workspace.py`, `tests/unit/test_improvement_harness_proposer.py`.

**Interfaces:** `create_candidate_worktree(repo: Path, baseline: str, candidate_id: str) -> Path` creates `.worktrees/harness-<id>` detached at `baseline`. `parse_status_z(raw: bytes) -> list[str]` returns both rename/copy paths. `is_allowed_path(worktree: Path, path: str, allowed_paths: tuple[str, ...]) -> bool` checks lexical and resolved paths. `export_checked_patch(worktree: Path, allowed_paths: tuple[str, ...]) -> bytes` validates every staged and untracked path, then exports a binary Git patch. `run_codex(worktree: Path, prompt: str, seconds: int) -> str` returns the recommendation text; `import_manual_patch(worktree: Path, patch: Path) -> None` uses `git apply --check` before applying. Manual intake also requires a recommendation file at the CLI boundary.

- [ ] **Step 1: Write failing tests in temporary Git repositories.** Test an ordinary allowed edit, an untracked file under `docs/experiment-reports/`, a symlink to `/etc/passwd`, an absolute path and `../` traversal, a patch with no diff, and a baseline mismatch. Verify a rejected patch never starts an evaluator.

```python
def test_untracked_protected_file_is_rejected(tmp_path):
    worktree = tmp_path / "repo"
    worktree.mkdir()
    subprocess.run(["git", "init", str(worktree)], check=True)
    subprocess.run(["git", "-C", str(worktree), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "base"], check=True)
    (worktree / "docs/experiment-reports/new.json").parent.mkdir(parents=True)
    (worktree / "docs/experiment-reports/new.json").write_text("{}")
    with pytest.raises(ValueError, match="outside allowed paths"):
        export_checked_patch(worktree, ("dashboard/src/",))
```

- [ ] **Step 2: Run the two focused pytest files.** Expect failures before implementation.
- [ ] **Step 3: Implement worktree and patch checks with `subprocess.run([...], shell=False, check=True)`.** Parse `git status --porcelain=v1 -z`, check both sides of renames, resolve every candidate path under the worktree, reject symlinks, stage only validated files with `git add -- <paths>`, and export `git diff --cached --binary`. Check `git rev-parse HEAD` against the frozen SHA before and after proposal generation.

```python
changed = subprocess.run(["git", "-C", str(worktree), "status", "--porcelain=v1", "-z"], check=True, capture_output=True).stdout
paths = parse_status_z(changed)  # Return both paths for rename/copy records.
if any(not is_allowed_path(worktree, path, allowed_paths) for path in paths):
    raise ValueError("outside allowed paths")
```
- [ ] **Step 4: Implement the proposer adapter and fake it in tests.** Invoke `codex exec -C <worktree> -s workspace-write --output-last-message <path> <prompt>` with an allowlisted environment, a timeout, and a fixed prompt containing objective/paths/checks. Treat missing Codex, nonzero exit, timeout or absent recommendation as `unavailable`/`failed`; never invoke a shell to interpret model output. Manual intake is the credential-free fallback.

```python
argv = ["codex", "exec", "-C", str(worktree), "-s", "workspace-write", "--output-last-message", str(message_path), prompt]
completed = subprocess.run(argv, cwd=worktree, env=allowed_env, timeout=seconds, check=False, capture_output=True, text=True)
if completed.returncode != 0 or not message_path.is_file():
    raise RuntimeError("proposer did not produce a recommendation")
```
- [ ] **Step 5: Run focused pytest and Ruff checks, then commit.** No live Codex call belongs in unit tests.

## Task 3: Fixed checks and baseline comparison

**Files:** Create `checks.py`, `tests/unit/test_improvement_harness_checks.py`. Modify `store.py` only to add typed result validation needed here.

**Interfaces:** `CheckResult` records name, argv, exit status, duration, bounded log hash, execution and verification. `run_checks(track: Literal["research", "dashboard"], cwd: Path, seconds: int) -> list[CheckResult]` gets argv from a versioned catalog. `compare_baseline(plan: Plan, baseline: list[CheckResult], candidate: list[CheckResult]) -> str` returns `passed`, `failed` or `inconclusive` and never treats baseline failure as candidate improvement.

- [ ] **Step 1: Write failing tests for a passing command, nonzero exit, `TimeoutExpired`, missing executable, preexisting baseline failure, and an unknown track name that could select an arbitrary command.** Use a fake process runner so tests need no Node or browser.

```python
def test_timeout_is_not_a_pass(tmp_path):
    def timed_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)
    result = run_checks("dashboard", tmp_path, 1, runner=timed_out)[0]
    assert result.execution == "timed_out"
    assert result.verification == "inconclusive"
```

- [ ] **Step 2: Run `uv run pytest tests/unit/test_improvement_harness_checks.py -q`; confirm the tests fail.**
- [ ] **Step 3: Implement a constant catalog.** Dashboard: `npm test`, `npm run typecheck`, `npm run build`, `npm run test:browser` in `dashboard/`. Research: exact fixture check plus focused pytest/Ruff in the repository. Preflight installs use the pinned `uv sync --frozen` and `npm ci` commands and are reported as checks; missing dependencies are `unavailable`. Each command has a fixed cwd and per-command cap bounded by the plan's total wall budget. Capture stdout/stderr to files under the candidate record, hash them, and expose only a bounded tail in JSON.

```python
CATALOG = {"dashboard": (("npm", "test"), ("npm", "run", "typecheck"), ("npm", "run", "build"), ("npm", "run", "test:browser")), "research": (("uv", "run", "pytest", "tests/unit/test_trajectory_reinforce_refinement.py", "-q"),)}
completed = runner(argv, cwd=command_cwd, timeout=remaining_seconds, shell=False, capture_output=True)
```
- [ ] **Step 4: Run the focused tests and Ruff checks, then commit.** Include a test that a missing Playwright binary is `unavailable`, not a passing skip.

## Task 4: Bounded research evaluator and held-out lock

**Files:** Create `research.py`, `fixture_candidate.py`, `fixture_reference.py`, `tests/unit/test_improvement_harness_research.py`, and `tests/integration/test_improvement_harness_research.py`. Create `src/thermo_lab/research_candidates/three_site.py` with the stable `propose_parameters(fixture) -> tuple[float, ...]` hook and no changed scientific policy.

**Interfaces:** `evaluate_three_site(plan: Plan, baseline_worktree: Path, candidate_worktree: Path) -> dict` extracts the candidate's nine-value vector, then scores baseline and candidate through the **baseline worktree's** `evaluate_exact_shared_objective`. `claim_heldout(root: Path, plan_digest: str, role: str, candidate_id: str) -> None` atomically reserves a role before a held-out run.

- [ ] **Step 1: Write failing tests for a valid vector, eight values, NaN, out-of-cap values, candidate changes to an exact-reference module, and two candidates claiming the same held-out role.** Include a negative outcome where the exact objective increases but execution/verification still pass.

```python
claim_heldout(tmp_path, "sha256:" + "a" * 64, "seed-7", "candidate-one")
with pytest.raises(ValueError, match="already used"):
    claim_heldout(tmp_path, "sha256:" + "a" * 64, "seed-7", "candidate-two")
```

- [ ] **Step 2: Run focused unit and integration tests; confirm failures.**
- [ ] **Step 3: Implement candidate extraction and frozen scoring.** Run `python -m thermo_lab.improvement_harness.fixture_candidate` in the candidate worktree to serialize a vector. Run `python -m thermo_lab.improvement_harness.fixture_reference` in the clean baseline worktree with that vector on stdin. The reference imports `build_checked_fixture` and `evaluate_exact_shared_objective`, computes both objectives, verifies finite bounded values, and returns `objective_after - objective_before`. Lower is better only if the predeclared threshold is met. Keep `exact_reference` on those values and label the result `three-site fixture only`.

```python
# fixture_candidate.py; this module is protected, while the imported hook is editable.
fixture = build_checked_fixture()
values = propose_parameters(fixture)
print(json.dumps({"parameters": values}))

# fixture_reference.py, executed from the frozen baseline worktree.
fixture = build_checked_fixture()
candidate = tuple(json.load(sys.stdin)["parameters"])
before = evaluate_exact_shared_objective(fixture=fixture, shared_parameters=fixture.model_parameters.values)
after = evaluate_exact_shared_objective(fixture=fixture, shared_parameters=candidate)
print(json.dumps({"before": before.objective, "after": after.objective, "delta": after.objective - before.objective, "evidence": "exact_reference"}))
```

The research plan preset fixes `track="research"`, `allowed_paths=["src/thermo_lab/research_candidates/three_site.py"]`, `primary_metric="exact_objective_delta"`, `direction="lower"`, `threshold=0`, `max_candidates=2`, and `wall_seconds=1800`. Task 5's `init-plan` command fills the current full baseline SHA before candidate generation. The initial hook returns `fixture.model_parameters.values`, so the unchanged baseline is a real observed zero-delta comparison. Do not claim the example is improved when the delta is zero.
- [ ] **Step 4: Implement held-out lock with exclusive creation under a digest-keyed directory.** Validate role names and reject retries under the same plan after a result; a new versioned plan needs fresh roles. No fitting or M4G release command is added.

```python
role_path = root / "heldout" / digest.removeprefix("sha256:") / f"{role}.json"
role_path.parent.mkdir(parents=True, exist_ok=True)
with role_path.open("x", encoding="utf-8") as file:
    json.dump({"candidate_id": candidate_id, "role": role}, file)
```
- [ ] **Step 5: Run focused tests, Ruff, and `uv run thermo-lab run configs/experiments/numpy-trajectory-reinforce-pasym-swap-one-step.toml --seeds 0,1,2 --output-dir results/harness-fixture-check` in a fresh output directory; then commit.** Record the output path and the exact scope of the check.

## Task 5: Dashboard evaluator and CLI orchestration

**Files:** Create `dashboard.py`, `cli.py`, `tests/unit/test_improvement_harness_cli.py`, and `tests/integration/test_improvement_harness_dashboard.py`; modify `pyproject.toml` to add `thermo-harness = "thermo_lab.improvement_harness.cli:main"`.

**Interfaces:** `evaluate_dashboard(plan: Plan, baseline_worktree: Path, candidate_worktree: Path) -> dict` runs Task 3's catalog and retains desktop/mobile screenshots. CLI commands: `thermo-harness init-plan --track research|dashboard --objective TEXT --output FILE`, `thermo-harness run PLAN --output-dir DIR [--manual-patch FILE --recommendation-file FILE] [--parent CANDIDATE]`, `thermo-harness show CANDIDATE --output-dir DIR`, and `thermo-harness review CANDIDATE --output-dir DIR --decision accepted|rejected --note TEXT`. `run_candidate`, `write_plan_from_preset`, and `render_candidate` are local functions in `cli.py`.

- [ ] **Step 1: Write failing CLI tests with a fake proposer and command runner.** Verify `init-plan` pins the full clean HEAD SHA before any proposal, a dry candidate creates recommendation, `patch.diff`, request/result JSON and a readable report; manual mode rejects a patch without `--recommendation-file`; `--parent` applies a validated parent patch, passes only development feedback to the proposer, and keeps the parent unchanged; a parent with visible held-out feedback cannot spawn another research candidate under the same plan; a second run leaves the first unchanged; `show` does not execute commands; `review` only appends owner decision; no command writes outside the output/worktree roots.

```python
assert main(["show", candidate_id, "--output-dir", str(tmp_path)]) == 0
assert (tmp_path / candidate_id / "patch.diff").is_file()
assert (tmp_path / candidate_id / "result.json").is_file()
```

- [ ] **Step 2: Run focused unit/integration tests and confirm failure.**
- [ ] **Step 3: Implement CLI flow.** Resolve the Git root and full baseline SHA, require a clean source, freeze the plan, create baseline/candidate worktrees, run proposer or manual intake, validate/export patch, evaluate the fixed track, and write one result/report. For `--parent`, apply the validated parent patch to a fresh worktree at the same baseline before generation, pass its development checks to the proposer, and export a cumulative new patch; reject reuse of exposed research held-out feedback. Record baseline checks once per plan digest and compare each candidate against that saved baseline; hash plans and patches. Enforce fixed candidate count and total wall time, and claim an optional held-out role before evaluation. On exceptions, preserve diagnostics with failed/unavailable status. Do not auto-review, commit, push or merge a candidate. Add a `--max-candidates` override only if it can reduce the predeclared limit, never raise it.

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-plan":
        return write_plan_from_preset(args.track, args.objective, args.output)
    if args.command == "run":
        return run_candidate(load_plan(args.plan), args.output_dir, args.manual_patch, args.recommendation_file, args.parent)
    if args.command == "show":
        print(render_candidate(read_candidate(args.output_dir, args.candidate)))
        return 0
    append_review(args.output_dir, args.candidate, args.decision, args.note)
    return 0
```

The dashboard plan preset fixes `track="dashboard"`, `allowed_paths=["dashboard/src/"]`, `primary_metric="checks"`, `direction="pass"`, `threshold=1`, `max_candidates=2`, and `wall_seconds=1800`; `init-plan` supplies the current full baseline SHA. Tests and generated files stay outside the edit allowlist; evaluators produce them.
- [ ] **Step 4: Implement dashboard screenshot capture from the existing Playwright suite.** Copy `dashboard/test-results/overview.png`, `mobile.png`, and `experiments.png` into the candidate artifact directory only if their producing checks pass; missing files make visual evidence `unavailable`. Do not assign an automated aesthetic score.

```python
for name in ("overview.png", "mobile.png", "experiments.png"):
    source = candidate_worktree / "dashboard" / "test-results" / name
    if not source.is_file():
        visual_evidence = "unavailable"
        continue
    shutil.copyfile(source, artifact_dir / name)
```
- [ ] **Step 5: Run focused tests, Ruff and dashboard `npm test`, `npm run typecheck`, `npm run build`, `npm run test:browser` in the candidate branch; then commit.** Browser absence is reported plainly rather than skipped as passing.

## Task 6: Read-only proposal display

**Files:** Create `dashboard/server/proposals.ts`, `dashboard/src/features/Proposals.tsx`, `dashboard/tests/proposals.test.ts`; modify `dashboard/server/service.ts`, `dashboard/server/plugin.ts`, `dashboard/server/export.ts`, `dashboard/shared/model.ts`, `dashboard/src/App.tsx`, `dashboard/src/styles.css`, `dashboard/tests/browser.spec.ts`, and `dashboard/README.md`.

**Interfaces:** `readProposals(repoRoot: string): Promise<{ items: ProposalSummary[]; issues: string[] }>` reads only validated local records under `results/harness/`; `GET /data/proposals.json` returns the compact summary, `GET /data/proposals/<uuid>/patch.diff` returns validated patch text, and `GET /data/proposals/<uuid>/artifacts/<name>.png` serves only `overview`, `mobile`, or `experiments`. `ProposalSummary` includes ID, track, objective, baseline, execution, verification, research outcome, review, recommendation, baseline/candidate value summaries, patch URL and available screenshot URLs. Static export writes an empty `data/proposals.json` unless reviewed records have been deliberately curated for export.

- [ ] **Step 1: Write failing server tests.** Exercise valid summary, missing result, changed digest, oversized JSON, symlinked candidate directory, path traversal, unsupported screenshot name, `POST` returning 405, and static export excluding an unreviewed local draft.

```ts
assert.equal((await getPayload(root, "/data/proposals.json", "POST")).status, 405);
assert.deepEqual((await readProposals(emptyRoot)).items, []);
```

- [ ] **Step 2: Run `npm test` from `dashboard/`; confirm focused new tests fail.**
- [ ] **Step 3: Implement the bounded adapter and route.** Reuse `readBounded` and realpath-under-root checks from existing server code; cap item count and each file size; parse only the known record version; return an issue for rejected records. Add no mutation route or arbitrary file-serving endpoint. Validate UUIDs and the patch digest before serving `GET/HEAD /data/proposals/<uuid>/patch.diff` as `text/x-diff`; serve only the three named PNG artifacts after size/hash checks. Reject other methods. Update `plugin.ts` to send text and image bytes with the right content type rather than JSON, and `export.ts` to write an empty `data/proposals.json`. Never expose an absolute workspace path.

```ts
if (path === "/data/proposals.json") {
  return { status: 200, body: await readProposals(root) };
}
const match = /^\/data\/proposals\/([0-9a-f-]{36})\/patch\.diff$/.exec(path);
if (match) return await readValidatedPatch(root, match[1]);
const image = /^\/data\/proposals\/([0-9a-f-]{36})\/artifacts\/(overview|mobile|experiments)\.png$/.exec(path);
if (image) return await readValidatedImage(root, image[1], image[2]);
```
- [ ] **Step 4: Write failing browser tests for a proposal list, keyboard navigation, recommendation/patch links, a failed refresh notice, and mobile no-overflow.** Keep a fixture record under the Playwright temporary repo root so the test does not rely on a developer's local runs.
- [ ] **Step 5: Implement the Proposals view and navigation.** Show execution, verification, research result and owner review as separate labels. Explain that a three-site result is a bounded exact trial. Keep the project dashboard's existing read-only and evidence labels.

```tsx
<section aria-labelledby="proposals-title">
  <h2 id="proposals-title">Improvement proposals</h2>
  {items.map((item) => <article key={item.id}>
    <h3>{item.objective}</h3>
    <p>{item.recommendation}</p>
    <p>Execution: {item.execution} · Verification: {item.verification} · Review: {item.review}</p>
    <p>Baseline: {item.baselineSummary} · Candidate: {item.candidateSummary}</p>
    <a href={item.patchUrl}>View draft patch</a>
    {item.screenshotUrls.map((url) => <a key={url} href={url}>View screenshot</a>)}
  </article>)}
</section>
```
- [ ] **Step 6: Run `npm test`, `npm run typecheck`, `npm run build`, `npm run test:browser`, review desktop/mobile screenshots, and commit.**

## Task 7: End-to-end demonstration and repository gates

**Files:** Modify `README.md` with a short local usage link; create `docs/improvement-harness.md` for commands and status semantics; create `docs/experiment-reports/2026-09-23-improvement-harness-demo/summary.md` only for reviewed demonstration evidence, without calling it a scientific release.

**Interfaces:** Two retained local candidate directories under `results/harness/` and a dashboard view that reads their summaries. The report lists baseline/candidate SHAs, plan and patch hashes, exact commands, status dimensions, screenshot availability and known limits.

- [ ] **Step 1: Generate one dashboard plan and one bounded research plan with `thermo-harness init-plan`, then execute one manual-patch candidate per track through `thermo-harness run`.** Retain both patches and reports. If live Codex invocation is available, run a separately budgeted local candidate and report its actual outcome; credential-free tests must still pass without it.
- [ ] **Step 2: Verify the dashboard displays both candidates in Chromium at desktop and mobile sizes.** Inspect recommendation, patch link, statuses and fixture scope. Keep screenshots in the local report and use a dated summary only if the owner curates it.
- [ ] **Step 3: Run dashboard checks and the full `AGENTS.md` local gates in a clean, stable worktree.** Capture exit codes and durations; do not edit Git state while provenance-sensitive commands execute. Compare hashes of completed M4G archive files and protected validators to the baseline commit.
- [ ] **Step 4: Review the complete diff and report exact coverage.** Check that no credentials, absolute local paths, raw traces, or unreviewed draft payloads enter tracked files or static export. Record failed/unavailable checks and any scientific claims that remain unproven.
- [ ] **Step 5: Commit the documentation/demo summary separately.** Do not push, open a PR, merge or publish without the normal owner handoff decision.

## Execution order and completion condition

Tasks 1–3 establish the shared contract, isolation and fixed checks. Task 4 adds bounded research scoring; Task 5 wires the CLI and dashboard evaluator; Task 6 exposes read-only reports; Task 7 validates the complete loop. Review each task's diff and test result before starting the next. The work is complete only when both tracks yield inspectable recommendation-plus-patch reports, the dashboard presents them accurately, all required gates pass or any exceptions are reported, and no research release or automatic publication is implied.
