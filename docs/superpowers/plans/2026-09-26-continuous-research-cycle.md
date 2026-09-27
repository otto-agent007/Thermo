# Continuous Research Cycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Prepare a trustworthy M5a execution/review cycle that can run on one durable CPU host and be supervised from ChatGPT Work.

**Architecture:** Keep the frozen scientific runner separate from a small Linux worker package. Requests bind source and science; a single durable journal owns execution and evidence. Work reads exported status and independently reviews accessible evidence. Deployment is a distinct final capability check.

**Tech Stack:** Python 3.11, existing uv lock, NumPy/SciPy, standard-library Linux process and file locking, existing React/Vite dashboard, GitHub and Work scheduling.

**Spec:** `docs/superpowers/specs/2026-09-26-continuous-research-cycle-design.md`

## Global Constraints

- CPU only; no new dependencies or paid services.
- One active study, at most two workers, one BLAS thread per worker.
- Six cumulative active wall-clock hours, including replay and retries; at most two attempts.
- Full M5a protocol: ten targets, nine caps, two methods, 180 chains, zero samples.
- Scientific failures remain valid outcomes; integrity errors block acceptance.
- Preserve all existing hash-bound source files and archived evidence.
- No automatic merge, live dashboard publication, or paid provisioning.
- Work state must be reconstructed from GitHub/evidence, not a retained scratch folder.
- No access to the user's always-on Ubuntu machine is established in this session. Do not call the transient execution workspace an activated worker.

## Review Focus

1. Duplicate triggers or stale heartbeats must never produce concurrent executions (Task 3).
2. Partial/corrupt evidence or a skipped replay must never look complete (Tasks 1, 3).
3. Retry/restart must retain consumed budget and immutable inputs (Task 3).
4. A scientific negative, unavailable reviewer, or unpublished dashboard must keep its true status (Tasks 4, 5).
5. Mismatched source/environment, path traversal, or arbitrary commands must be refused before execution (Tasks 3, 4).

## Task 1: Correct M5a and make replay mandatory

Files: `docs/research/meta_ebm_probe.py`, `docs/research/2026-09-26-meta-ebm-target-reading.md`, `src/thermo_lab/meta_ebm_cap_baseline.py`, `tests/unit/test_meta_ebm_cap_baseline.py`.

Interfaces: preserve the existing scientific request and evaluator. `run_study(output_dir, *, replay=True, workers=None)` must reject `replay=False` before creating output. Add a safe CLI replay path for an already persisted archive without fitting; record original-generation provenance separately from replay provenance.

- [x] Reproduce the probe discrepancy against direct hidden-spin enumeration, including nonzero hidden biases and cap-boundary parameters.
- [x] Add regression tests that fail for the incorrect probe formula and skipped-replay completion.
- [x] Make the probe import-safe with a main guard, correct both NumPy/JAX formulas, and withdraw unsupported old cap-table numbers rather than invent replacements.
- [x] Pin the canonical M5a request digest; test altered request, result digest, and replay failures.
- [x] Add replay-only recovery with fresh output destination and atomic evidence writes; test that it cannot refit or fabricate missing original provenance.
- [x] Run focused tests and Ruff, preserving the existing seven numerical checks. No full refitting is necessary for regression tests.

## Task 2: Unify instructions and document operational contracts

Files: `AGENTS.md`, `CLAUDE.md`, `docs/continuous-research.md`, `docs/release-gates.md`, `docs/studies.md`.

Interfaces: all agents read common operational/research rules from AGENTS.md; CLAUDE.md imports them. The runbook distinguishes implemented software, an actual study, and activation.

- [x] Move common assumption checks, M4 closure, Git discipline, provenance and evidence-size guidance from CLAUDE.md into AGENTS.md without duplicating conflicting rules.
- [x] Keep CPU as the default and the first-cycle requirement; cloud GPU work needs a separately scoped execution path.
- [x] Add the M5a gate entry and document honest evidence status. Do not mark an archive or result present until generated and validated.
- [x] Describe source/evidence handoff, existing ruleset activation, and the existing operator-run live publication boundary.

## Task 3: Implement the single-host worker

Files: create `src/thermo_lab/research_cycle/{__init__,contracts,worker,supervisor,cli}.py`, `tests/unit/test_research_cycle.py`, and a cheap subprocess fixture under `tests/fixtures/research_cycle/` if needed.

Interfaces: `python -m thermo_lab.research_cycle.cli` exposes request preparation/validation, bounded execution, inspection, and evidence verification. Public worker entry accepts a reviewed request path, clean repository path, and external durable state root. It must never accept arbitrary executable command text. The only production study adapter is M5a.

Request schema v1 binds `job_id`, `source_sha`, `protocol_sha256`, `lock_sha256`, `scientific_request_digest`, `workers`, `budget_seconds`, `max_attempts`, and question text. Validate strict types, full SHA, safe job IDs, maxima 2/21600/2, and immutable canonical identity. Execution metadata is not inserted into the scientific request.

Export `status.json` schema v1 with `job_id`, `source_sha`, `request_digest`, `phase`, `attempts`, `elapsed_seconds`, `heartbeat_at`, `evidence_digest`, `review_status`, and `message`. Phases are queued/running/verifying/awaiting_review/recorded/blocked/failed. Review status starts `pending`; an execution alone never sets `recorded`.

- [x] Write failing tests for request refusal, same-ID changed inputs, duplicate lock contention, dirty/wrong checkout, timeout, exhausted budget, corrupt completion, and missing artifacts.
- [x] Validate a clean pinned checkout and lockfile/protocol/scientific hashes before running any study code. Use fixed argv and a restricted execution environment; run frozen dependency sync only from approved source.
- [x] Implement a Linux advisory lock shared with the child lifetime, atomic journal updates, bounded logs, heartbeat and whole-process-group termination. If ownership is uncertain, block; never reclaim based only on heartbeat age.
- [x] Charge cumulative active time conservatively after interruptions, with at most one infrastructure retry and a fresh attempt directory. Integrity failures do not trigger scientific refitting.
- [x] Support replay-only recovery from an authenticated completed generation archive using Task 1; otherwise restart generation within remaining budget.
- [x] Validate full M5a completion and artifact hashes; persist a wrapper manifest plus status outside the checkout. The local stage stays verifying until a separately accessible copy is verified.
- [x] Add an explicit mirror verification operation for a supplied second-copy directory. It must rehash all manifest-bound files; it does not claim a local second copy is GitHub-accessible. Record destination provenance and require operator/Work verification of the connected destination at activation.
- [x] Exercise the lifecycle and recovery with cheap actual subprocesses and temporary git repositories. Do not run full M5a to test process failure paths.

## Task 4: Evidence review and host activation package

Files: `docs/continuous-research.md`, `docs/automation-prompts/thermo-cycle-supervision.md`, example request preparation instructions; optional `tools/research-cycle/` service examples using documented placeholders.

Interfaces: the supervision prompt consumes the pinned request, source, status, manifest and evidence at an explicitly supplied GitHub branch/path. Review records bind author identity, source/evidence digests, scope, findings, and decision. A manual review is labeled manual.

- [x] Write a durable prompt that reconstructs state, deduplicates by job/evidence identity, reports only meaningful changes, and leaves unavailable reviews pending.
- [x] Provide concrete installation/start/stop/recovery commands for the existing Ubuntu host. Default service is disabled; require memory/disk/restart and dependency checks before enabling.
- [x] Supply an opt-in snapshot publisher for a designated `evidence/<job_id>` branch, with isolated Git tests. Publish active status observations and complete manifest-bound evidence; never mark a push as independent remote verification. Keep its service disabled until branch authority and host access are verified.
- [x] Specify allowed writes only to the designated evidence branch and draft archival PR. No comments/messages to other people, main writes, merges or publication.
- [ ] Validate a replayed evidence handoff with a separate reviewer context; deliberately altered evidence must be rejected. Test the prompt's read path against the real GitHub connector before scheduling.
- [ ] Activate hourly supervision only when the actual evidence destination and worker status are accessible and the prompt has passed a dry run. Otherwise leave a concrete deployment blocker; do not create a task against nonexistent paths.

## Task 5: Stage dashboard visibility

Files: existing dashboard server/shared model/export files, `dashboard/src/features/Research.tsx` or a small cycle component, and focused dashboard tests.

Interfaces: consume the worker's schema-v1 `status.json` via an explicitly configured local cycle root. Absence means no active cycle, never a fabricated run. Published snapshots carry a bounded sanitized status with snapshot time; local paths, secrets and raw logs stay out.

- [x] Add failing tests for absent state, malformed status, pending review, stale heartbeat, and negative scientific outcome versus execution success.
- [x] Show phase, source SHA, attempt/budget usage, last observation, evidence/review status, and actionable blocker text using the existing UI patterns.
- [x] Test the exporter and UI; preserve current study rendering and distinguish staged/snapshot status from live worker telemetry.
- [x] Run dashboard tests, typecheck and build.
- [ ] Run the browser check on a host with Chromium available; local installation failed. Do not deploy the site.

## Task 6: Integrate, review, and prepare the first real run

- [x] Run the changed-area Python and dashboard gates, lock consistency, Ruff, and package checks. Report existing unrelated failures by name.
- [x] Independent whole-change review against the approved spec; fix important findings and verify them.
- [x] Check GitHub for newer M5a code/evidence before launching any real study, so Claude's work is not duplicated.
- [ ] Measure bounded host resource suitability, then prepare a request pinned to corrected committed code. Keep the full grid and exact replay unchanged.
- [ ] Run/record the real study only on an authorized durable host that passed the activation checks. Add archive-backed replay tests and the report PR after evidence exists.
- [x] If that host is inaccessible, deliver tested code and exact activation instructions, identifying the missing access. Do not claim the approved unattended-cycle acceptance criterion is met.

## Execution decision

The owner approved the concrete design and requested continued progress. Execute reversible implementation work in the existing dedicated feature checkout with independent agents/review; reserve actual host/service activation for verified capabilities. No additional design approval is inferred for paid services, merges, or new science. The first implementation can be reviewed and used even if host activation remains blocked.

## Implementation verification and activation boundary

The implementation includes the M5a correction/replay guards, shared guidance,
Linux worker with an independent deadline guardian, optional Git publisher,
and dashboard observations with separate evidence-bound review records.
Independent M5a, worker, dashboard and handoff reviews found concrete issues
that were fixed before integration. Worker recovery is conservative across a
host reboot: unobserved active work spends the remaining allowance.

No durable host is connected, no production request has run, no service/timer
is enabled, and no scheduled Work reviewer exists. Real-host resource/restart
checks, connected evidence/review dry runs, the full M5a archive and report, main
ruleset activation and operator publication remain deployment/acceptance gates.
The local browser gate was blocked by a missing Chromium runtime and invalid
installation downloads; tests do not stand in for a rendered-layout pass.

Final local verification: 1,536 tests passed in the CI `unit-rest` partition
(non-slow unit tests excluding the two separately partitioned aggregation and
target-context-results modules). This includes 34 M5a, 39 worker and 16 publisher
tests. Dashboard: 109 tests, typecheck and build passed. Frozen lock check, Ruff
format/lint, CPU smoke, wheel/source build and service-template syntax checks
passed. No full scientific study or full CI-matrix pass is claimed locally.
