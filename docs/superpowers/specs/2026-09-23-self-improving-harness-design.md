# Thermo improvement harness

Status: design approved in chat on 2026-09-23; this document awaits owner review before implementation planning.

## Purpose and boundary

The owner wants a repeatable way to improve both Thermo research experiments and the project dashboard. Each iteration should produce a written recommendation, an inspectable draft patch, and evidence showing how that patch behaved against a fixed baseline. The harness proposes changes for review. It does not adopt a candidate, publish a scientific claim, merge a branch, deploy a dashboard, or start an unapproved full study.

The first release is a locally triggered workflow for one owner. It has two tracks sharing one candidate ledger and report format. Research trials are limited to a predeclared small fixture with an exact reference; dashboard trials use the existing React/Vite checks and browser flow. The dashboard remains a read-only viewer of candidate reports. A scheduled service, remote job control, multi-user permissions, and automatic publication are later work.

Success means the owner can request one candidate in either track, inspect the exact patch and recommendation, see the baseline and checks that were run, distinguish a failed check from a negative scientific result, and repeat an iteration without losing the earlier candidate. One demonstration candidate per track must complete this flow; a passing harness trial alone is not a research release.

## Approaches considered

| Approach | Benefit | Cost or limit |
| --- | --- | --- |
| Shared local candidate ledger with separate research and dashboard evaluators (chosen) | One review workflow and evidence vocabulary; track-specific gates stay separate | Requires a small CLI and dashboard adapter |
| Two independent improvement scripts | Easy to start each track | Duplicates provenance, review and failure semantics |
| Autonomous execution console | Could schedule and publish work | Adds credentials, lifecycle, scientific leakage and approval risks beyond review-first scope |

## Candidate lifecycle

1. The owner starts a local iteration with a track, objective, baseline commit, allowed edit paths, and evaluation plan. The plan is saved before a candidate is generated. It fixes the primary acceptance checks, comparison target, seed or fixture roles, command budget and time limits. Checks come from a versioned track command catalog, not commands supplied by the proposer.
2. The proposer receives that plan and relevant repository context. The initial adapter invokes an installed local Codex CLI in a new candidate worktree. It writes a recommendation and draft patch there. A manually supplied patch can use the same evaluator when the adapter is unavailable. The harness checks the proposer output and rejects edits outside the declared paths. Generated commands run with workspace-only write access, an allowlisted environment and restricted network access; the adapter fails if those limits cannot be enforced. It cannot change the pinned archive or its validators.
3. The evaluator records the patch digest and runs the track's fixed checks in the candidate worktree. It captures command, exit status, duration, bounded log, source revision and resulting artifacts. A baseline check is recorded once for the same plan so a candidate is compared with an observed baseline, not an assumed green state.
4. A report records the recommendation, patch, plan, baseline, check results and conclusion. The owner reviews it. A later draft may name a previous candidate as its parent, but all drafts remain isolated and reviewable; passing checks do not promote or merge them.

Each iteration has a maximum candidate count and wall-time budget set before it starts. A failed or inconclusive check can inform another development candidate. For scientific trials, a held-out evaluation is run at most once for the selected candidate under the frozen plan; its outcome cannot be used for another candidate under that same plan. Continuing after a held-out result requires a newly versioned protocol and fresh held-out roles. This prevents the loop from turning evaluation data into a hidden search budget.

## Record contract and states

An immutable candidate record contains a version, candidate and parent IDs, track, objective, baseline commit, plan and patch digests, allowed paths, proposer identity/version when available, source state, commands, bounded logs or log hashes, artifact hashes and timestamps. Requested inputs and plan identity exclude timestamps, machine details, outcomes and measured durations. Observed runtime provenance stays in the result. A later owner decision is a separate append-only review record linked to that candidate; it cannot rewrite evaluation evidence. Reports and patch files under ignored `results/harness/<candidate-id>/` are local working evidence; a separately reviewed record can be curated into the repository with its patch and source links. Re-running the same request creates a new observation, not a rewritten result.

Use separate status dimensions:

- Execution: `complete`, `failed`, `timed_out`, or `unavailable`.
- Verification: `passed`, `failed`, or `inconclusive`.
- Research outcome: `improved`, `regressed`, `inconclusive`, or `not_applicable`, each qualified by its declared fixture and evidence class.
- Review: `proposed`, `accepted`, or `rejected` by the owner in the separate review record. The harness never sets `accepted` from a metric.

Missing, changed or unsupported evidence is unavailable or inconclusive, never a zero value or a passing result. A scientific trial can execute and verify correctly yet find no improvement. A dashboard candidate can pass automated checks while still needing visual review. Numeric claims link to the exact source and distinguish `exact_reference` from `software_simulation`; no harness result is hardware evidence.

## Research evaluator

The first research flow uses an existing bounded three-site fixture and exact enumeration. The plan names one primary quantity, direction and threshold; fixed comparison parameters; seed policy if sampling is involved; and the tests that verify request and record integrity. Candidate changes may alter fixture-level experimental logic or a diagnostic, but cannot mutate completed M4G study inputs, archived results, pins, release validators or acceptance decisions. The evaluator checks exact-reference agreement, conservation and source bindings, then runs relevant Python tests and repository gates for the touched scientific paths.

Any sampled result remains `software_simulation`. Multiple states from one chain are not independent replications; confidence intervals require genuinely independent seeded runs and declared coverage. The harness may recommend a larger study, but a new study needs its own predeclared protocol, independent review, release gates and final evidence review. A fixture-level improvement is labeled as such in both the report and dashboard.

## Dashboard evaluator and viewer

The dashboard track fixes the target user flow and acceptance checks before generation. It runs unit/interaction tests, typecheck, production build and Playwright browser checks in the candidate worktree. It captures desktop and mobile screenshots for owner review, including accessibility or layout issues the automated checks do not settle. Pixel difference or an agent's aesthetic judgment is supporting feedback, not an automatic acceptance gate. A visual change with no functional regression may still be rejected by the owner.

The dashboard adds a read-only proposals view from an allowlisted local `results/harness` adapter. It shows track, objective, status dimensions, baseline, concise recommendation, before/after evidence and links to the patch and report. It does not execute proposals through a browser route. Local unreviewed drafts are not exported into a published static snapshot. A snapshot may include only explicitly curated reviewed records and must identify its export time and source revision. Paths are resolved under configured roots; user-supplied path traversal and symlink escapes are rejected.

## Failure handling and verification

The runner fails closed on a dirty or moving baseline, invalid plan, changed evidence pin, unsupported record version, edit outside allowed paths, missing dependency, timed-out command or incomplete artifact. It preserves the candidate and diagnostics for review without reporting success. Bounded logs avoid embedding secrets or large traces in browser payloads. A stopped run remains distinct from a scientific failure.

Implementation acceptance requires tests for plan immutability, canonical IDs, patch/path validation, baseline mismatch, held-out reuse rejection, tampered or missing evidence, status propagation, and dashboard read-only behavior. An end-to-end local demonstration must create and evaluate one dashboard draft and one bounded research draft, then show both reports in the dashboard with working patch/source links. Run the existing dashboard checks and the required local gates in `AGENTS.md` on the implementation branch. Record exact commands and any checks not run. Confirm that scientific source archives and the completed M4G evidence retain their hashes.

The demonstration may use deliberately small candidate changes; it cannot claim a general scientific or visual improvement merely to exercise the workflow. The owner reviews the final draft patches and evidence before deciding whether either change belongs in Thermo.
