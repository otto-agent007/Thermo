# Thermo continuous research: first unattended cycle

Status: approved by owner on 2026-09-26; no scheduler or worker is enabled.

Baseline: `3d5b0aca89fcb25d54c2bb21f5e63d67b0d78f35` (PR #52).

## Goal and first deliverable

Thermo should keep making scientific progress between human sessions. ChatGPT
Work is a primary research and development environment alongside Claude and
Codex. GitHub and persisted evidence carry the handoff between them.

The first deliverable is one bounded, unattended M5a execution and review cycle:
an approved request produces the full recorded study, mandatory replay, an
independent review, and a staged dashboard entry with its actual outcome. Completion
does not require a favorable scientific result. This cycle establishes reliable
execution; it does not yet autonomously invent and approve new experiments.

The longer-term loop is question -> cheap assumption check -> frozen experiment
-> execution -> independent review -> recorded finding -> next justified question.
Its success measure is useful, reproducible findings, including rejected
hypotheses. Keeping a processor busy is not a scientific objective.

## What the merge provides

PR #52 supplies a frozen M5a protocol, NumPy/SciPy CPU runner, exploratory probe,
research note, and seven component tests. The study has ten targets and 180
compiled chains. Topology, finite kernel thermalization, THRML sampling, and
hardware costs are later stages.

Main CI passed for this baseline. The merge contains no completed M5a archive or
report, and the protocol's promised archive replay tests are not implemented.
M5a is therefore implemented in part, not a completed scientific result.

Before the first recorded run:

1. Correct the exploratory probe's hidden-spin contribution and qualify or
   replace its cap table. The production runner uses the correct log-cosh
   expression; the exploratory formula changes the effective capped family.
2. Require replay before any completion marker can declare a study complete.
   Currently `run_study(replay=False)` can write a complete status.
3. Add focused coverage for the request identity, replay rejection of altered
   evidence, and incomplete-run behavior. Archive-backed tests follow the first
   valid run; do not fabricate an archive to satisfy the tests.
4. Put shared research and handoff rules in AGENTS.md, retain CLAUDE.md as its
   entrypoint, and activate the existing main ruleset. Keep the M4 conservation
   line closed. These are reviewed follow-up changes, not changes made here.

The execution request must pin the corrected commit, not this baseline. Any
scientific protocol change requires its dated amendment before new fitting.

## Minimal architecture

| Component | Responsibility |
| --- | --- |
| ChatGPT Work | Design, implementation, analysis, independent review, and periodic supervision using GitHub and saved artifacts. |
| Claude / Codex | Implement or review an assigned task on an isolated branch; publish a handoff identifying the exact code and evidence examined. |
| GitHub | Reviewed job requests, source history, task status, CI, and links to recorded evidence. |
| One Linux CPU worker | Execute allowlisted, approved requests from pinned source; own the job lock, runtime limits, execution journal, and output persistence. |
| Existing dashboard | Prepare a tested view of request, phase, last heartbeat, evidence, review state, and scientific outcome; live publication stays operator-run. |

The proposed first worker is the existing Ubuntu machine if it can remain on.
Deployment must first verify available memory, disk, network, and restart
behavior. No new paid compute or GPU service is part of this slice. A cloud
worker can replace the host later without changing the scientific request.

Use one worker process and its durable local journal/lock initially. GitHub task
labels are a readable projection, not an atomic job lock. Do not introduce a
distributed queue, multiple execution hosts, or automatic model routing yet.

The current improvement harness stays separate: it evaluates bounded candidate
patches on its existing CPU presets. Its research hook is not an M5a scheduler.
Never disable its sandbox to make it run on a different host.

## Request, identity, and authority

A reviewed job request binds a job ID, hypothesis/question, full source SHA,
protocol and lockfile digests, scientific request digest, allowlisted command,
artifact destination, and resource limits. Execution settings remain outside
the scientific request identity. Use the existing hashing, persistence, and
provenance helpers rather than another study-specific framework.

V1 executes the frozen full M5a grid without adding seeds, changing caps,
choosing favorable cells, or substituting a different evaluator. It must first
check whether a completed matching run already exists, including work from
Claude, to avoid rerunning work only because another chat cannot see it.

The worker runs only reviewed requests against approved source. PR events and
issue text cannot become arbitrary shell commands or change the evaluator.
Candidate implementation, trusted execution, and scientific review remain
distinct responsibilities. Owner approval covers the bounded job, not every
ordinary execution step; it does not authorize paid services or auto-merges.

## Execution and restart behavior

The default proposed limits are one active study, at most two CPU processes,
single-threaded BLAS per process, six cumulative active wall-clock hours across
generation and replay attempts, and at most one infrastructure retry. Before
activation, a bounded resource probe
must establish whether the full run and mandatory replay fit these limits.
If they do not, revise the execution budget explicitly; do not silently extend
it or narrow the scientific grid. Count all attempts in the job's work ledger;
a retry receives only the remaining allowance. Existing model access must also
be verified before scheduling reviews. Use available plan allowances, stop on
quota/access failures, and never enable API billing or purchase credits as an
automatic recovery step. New paid service spend is zero for this slice; that
does not claim the existing machine or subscriptions have no operating cost.

At start, acquire the local job lock and record an attempt ID and heartbeat.
The worker uses a clean checkout at the pinned SHA and `uv sync --frozen`.
Every attempt has its own output directory; it never overwrites an earlier
attempt or accepts a completion marker without checking the evidence.

Current M5a does not checkpoint partial fitting. V1 may restart an interrupted
generation once from the beginning, recording the consumed work. If a complete
study archive was safely persisted, recovery can validate and replay that
archive without refitting. Implement this recovery path explicitly; it is not
a capability of the current CLI. Validate identity and integrity before reuse.

Persist the compressed study and an atomic artifact manifest before publishing
completion. The manifest binds the source SHA, scientific request digest,
environment/provenance, job/attempt IDs, exact command, exit status, bounded
logs, file hashes, and replay coverage. Keep a
durable copy outside the worker's temporary checkout, then verify a second
copy accessible to Work before marking execution ready for review. The first
cycle can use a bounded compressed archive on an evidence branch in the repo,
following the existing report conventions; oversized outputs require a
reviewed external artifact destination before launch. Keep accepted scientific
archives in versioned storage without automatic expiry. Retain failed-attempt
diagnostics for at least 30 days; any shorter provider retention must be
resolved by export before that provider's artifacts expire.

An expired heartbeat alone is not permission to start a duplicate process.
The single worker first establishes that the prior process has exited and the
job lock is free. If it cannot establish ownership safely, mark the job blocked.

## State, supervision, and review

Keep execution, evidence validity, review, and scientific outcome separate.
For example, a fully replayed negative result is a successful execution.

| State | Transition condition |
| --- | --- |
| queued | Approved request exists and no matching completed job exists. |
| running | Worker owns the lock; bounded attempt and heartbeat recorded. |
| verifying | Persisted outputs undergo the complete mandatory replay. |
| awaiting_review | Evidence is valid and accessible, with its manifest. |
| recorded | Independent review is resolved and the report is accepted. |
| blocked / failed | Missing capability, exhausted infrastructure budget, or invalid evidence is recorded with its cause. |

Propose an hourly Work supervision task, running in a separate scheduled
session from the implementation author. Test its read-only state inspection
and evidence-review paths before activation. Each run reads durable state,
checks progress and new evidence, and performs the next authorized review step.
The proposed write scope, to be authorized at activation, is the job's status
and review records on its designated evidence branch and a draft archival PR.
It excludes main writes, merging, deployment, and messages to other people.
Repeated supervision runs reuse the job ID and reviewed evidence digest so
they cannot duplicate reviews or PRs. It does not refit a study simply
because a previous scheduled session ended. It reports meaningful changes and
blockers, not repetitive hourly messages. No task is created by this design.

The reviewer must be a separate session from the implementation author. It
examines the pinned protocol, evaluator, baseline, all cells, replay evidence,
limitations, and any disagreement. A green CI run alone is not scientific
acceptance. Record reviewer tool/model identity when available and the exact
source/evidence digests reviewed; do not claim that two models reviewed a run
unless both actually did. If a reviewer is unavailable, remain in
`awaiting_review` and report the blocker. Claude can use the same handoff
manually until a tested Claude invocation path exists.

The dashboard's staged build may show an awaiting-review result with that label.
Run its existing export, build, and browser checks. Updating the deployed site
remains an authenticated operator publication step under the current CI/CD
runbook; it is not implied by staged visibility or green delivery checks.
Accepted research reports and merges retain owner control in V1. If a research
line fails twice for the same reason, pause that line for the owner's decision;
other approved jobs may continue. An empty queue waits without inventing work.

## Acceptance evidence and subsequent scope

V1 is demonstrated when one corrected M5a request completes without manual
execution intervention, all ten targets and 180 chains are retained, full
replay succeeds, the tested scheduled Work reviewer records an independent
review, and the staged dashboard accurately shows evidence and acceptance
status. A manual Claude review remains useful but does not demonstrate the
unattended-review acceptance criterion. An archival/report PR must include
the missing M5a request-pinning and archive-replay coverage described by its
protocol and pass CI. Owner review/merge and live dashboard publication are
the final intentional handoffs; they are not
unattended execution intervention. No new discovery is presumed.

Use a cheap fixture to verify duplicate-trigger handling, interruption/restart,
missing artifact handling, and refusal to mark unreplayed work complete. Do not
rerun the full study for each infrastructure failure-path test.

After that cycle, extend to a small approved queue and measured throughput.
GPU execution, new M5 scientific stages, automatic hypothesis selection, and
provider adapters each require a justified follow-up scope. Preserve held-out
evaluation when a future study uses adaptive selection; the current three-site
harness does not supply genuine held-out evidence.

## References

- [M5a protocol](../../experiments/meta-ebm-cap-baseline.md)
- [Current improvement harness](../../improvement-harness.md)
- [CI and delivery](../../ci-cd.md)
- [Official Work scheduled-task documentation](https://learn.chatgpt.com/docs/automations)
  (checked 2026-09-26): web tasks can use connected tools but do not retain a
  local checkout between runs. Schedules and available tools are capability
  checks at activation, not assumptions about an always-running chat.
