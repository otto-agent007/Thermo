# Research loop

*Started October 8, 2026. Replaces the parked
[improvement harness](improvement-harness.md) as the self-improvement plan.*

The research loop runs Thermo studies unattended between owner sessions. It
takes the next row from the owner-curated [research queue](research-queue.md),
probes it, drafts a frozen protocol, asks the owner on Discord, then runs,
replays and writes up the study as a PR. It never chooses its own topics,
never merges and never accepts its own results: the owner approves each
protocol by a Discord reply and accepts each result by merging its PR.

It is built from existing pieces: a Hermes cron job, the `thermo` kanban
board, the THERMES Discord channel and the autosave runners. A
director agent reviews direction above the rows, after each accepted study,
and proposes queue changes the owner may adopt; it is advice, never control. There is no new
daemon (compare the rejected PR #57).

## Pieces

| Piece | What it is | Model |
| --- | --- | --- |
| Queue | `docs/research-queue.md` on `main`, edited only by the owner (or on the owner's request) | none |
| Gatekeeper | `loop-tick.py` in the THERMES Hermes profile, run every 5 minutes by a no-agent cron job. It reads owner commands from Discord by exact match, reconciles kanban cards and PRs, and creates the next card. | none |
| Proposer and executor | The `thermes` profile, working `LOOP P-NNNN` kanban cards in git worktrees | Opus 5.5 |
| Reviewer | The `thermes-review` profile in the kanban review lane, comment-only | Fable 5.1, high effort |
| CI fixer | The `thermes-ci` profile, given a card only when CI fails (see below) | Haiku 5.5 |
| Director | The `thermes-director` profile, run after each loop PR merges and weekly as a floor; writes a direction review as a docs-only PR and proposes queue changes; never edits the queue, approves, runs or comments on study PRs (see below) | Fable 5.1, high effort |
| Owner channel | #thermes on Discord | none |

The gatekeeper, the card templates and the guard live in the THERMES profile
(`~/.hermes/profiles/thermes/loop/` and `scripts/loop-tick.py`), next to the
other THERMES scheduled scripts, not in this repository.

## One row, start to finish

1. **Proposal.** When nothing is waiting for the owner, the gatekeeper gives
   the top open, unclaimed row an id (`P-0001`, ...) and a propose card. The
   proposer lists the study's fixed choices and what each could bound, then
   runs a probe of at most 30 CPU-minutes, written to
   `docs/research/<date>-<row>-probe.md` with its script and JSON and labelled
   exploration. If the probe answers the question, it opens a docs-only PR
   and stops. Otherwise it writes `docs/experiments/<row>.md` (fixed choices
   first, pre-registered expectations, arms, budgets, decision rule,
   persistence plan, archive size estimate, gate) and opens a draft PR on
   `research/<row>`. No runner code yet.
2. **Approval.** The gatekeeper posts the proposal to #thermes. The owner
   replies to that post with `approve`, `revise <notes>` or `reject <why>`.
   An approval binds to the PR's head commit at the time it was announced; if
   the PR changed since, the gatekeeper refuses and re-announces.
3. **Run.** One study runs at a time. The executor checks that the protocol
   is unchanged since the approved commit, writes the study-local runner and
   its tests (bitwise equality against any sampler it derives from, a
   small-run replay test, single-thread BLAS pinned), passes the full
   `pytest tests/unit -m "not slow"` gate, pushes the runner, and then runs
   from a clean tree with autosave when over 30 minutes. It replays all
   evidence, then writes the report, a figure, a gzipped archive of at most
   4 MB, and the studies, release-gate, AGENTS.md index, roadmap, CLAUDE.md
   and lessons entries, and marks the queue row done. Negative results are
   written as negatives.
4. **Second opinion.** The executor hands the card to the review lane. The
   reviewer reruns the replay and unit tests, checks the protocol diff,
   hash-bound files, numbers, evidence labels and archive size, and posts its
   verdict as a PR comment. It may send the work back once.
5. **Acceptance.** The gatekeeper posts the results and the reviewer's
   verdict. The owner reviews and merges the PR, or closes it. Merging is
   the only acceptance.

The next proposal is drafted while a finished study waits for the owner.

## Director

*Added 2026-10-08 with charter amendment 1.* The proposer works inside one
row: probe, fixed choices, protocol. The director works above the rows: did
the last accepted result change what the lab should ask next, and is the
queue still pointed at the charter's primary question? Those are different
jobs and get different agents, because an agent that drafts protocols has
every reason to find its own direction sound. The director is the second
opinion on direction in the way the reviewer is the second opinion on a
study.

**Trigger.** The gatekeeper creates a `DIRECT D-NNNN` card when a loop PR
merges (it already watches merges), and once a week on a fixed day if no
merge happened, so a quiet week still gets a review. Never per tick, and
never while a previous director card is open.

**Inputs, all read from `main`.** `PROJECT_CHARTER.md` including its
amendments; `docs/knowledge/lessons.md`; `docs/roadmap.md`; the findings of
the last three merged studies and any probe notes under `docs/research/`
since the previous review; `docs/research-queue.md`; the source cards under
`docs/knowledge/sources/`; and the previous direction review. The source
cards are not optional: the failure the amendment records was rediscovering
the literature, and the only defence is an agent told to ask "is this
already known?" against the lab's own reading list before endorsing a row.

**Rubric.** The review answers these, in this order, each with a citation
(a findings file, a lessons row, a source card, a queue row):

1. For each result since the last review: did it produce a number someone
   will design against (a precision requirement, a noise tolerance, a sweep
   or I/O budget, a task quality at a projected cost, an upstream contract),
   or was it a probe with a study's process? Was it run under the hardware
   constraints, and which did it relax? Did it state its resource
   accounting? Did it miss a pre-registered expectation, and which queue
   rows assumed the opposite?
2. For each queue row: what decision changes depending on its answer; what
   is the cheapest probe that would make the full study unnecessary; is it
   already answered in a source card; is it a third variant of something
   that has failed twice.
3. For the lab: can it state the amendment's four-week test today (one
   task's precision and noise tolerance; sweeps, reads and writes per
   useful sample; projected energy at a stated quality against a
   conventional baseline)? If not, which row closes the largest gap?

**Output.** One file, `docs/research/<date>-direction-review.md`, opened as
a docs-only PR labelled `loop-direction`, with: a verdict per result, a
proposed queue diff written as rows in the queue's own format (add, drop,
reorder, hold or open; at most one *new* row per review, each with the
decision it informs), and one named thing to drop. The gatekeeper posts the
PR to #thermes. The owner merges it or closes it; merging accepts the
review as a record, not the queue change. The queue stays owner-edited: the
owner applies the diff by hand, or replies `adopt D-NNNN` and the gatekeeper
opens the queue edit as a separate PR for the owner to merge.

**What it may not do.** Edit `docs/research-queue.md`; approve, revise or
reject a protocol; start, stop or touch a run; comment on an open study PR
(the review lane does that); add more than one new row per review; propose
a row the charter lists as out of bounds; or propose a third variant after
two consecutive failures of the same kind, where its only permitted output
is a question to the owner about the premise.

**Keeping it honest.** A separate profile and model from the proposer,
comment-only like the reviewer. Every verdict cites evidence, so a lazy
review is visibly lazy. Every review names something to drop; a director
that only adds rows is a backlog generator. Reviews are advice: the control
is the owner reading one page a week. If direction reviews start being
merged unread, the lab has an automated strategist, which is the failure the
amendment describes with a human one.

**Cost.** One card per merge or week, no probes, no runs. Flag a review
over $10 in drafting the way proposals are flagged.

## CI failures

The gatekeeper reads CI itself on every tick, at no model cost: the checks of
each loop PR that has no worker on its branch, and the latest `Thermo CI` run
on main. It pulls the failing test ids from the job log. A failure at a new
commit gets a `thermes-ci` card, which reads the log and either:

- fixes a failure the PR itself caused, touching only the PR's own files and
  never its protocol, a hash-bound source or an archive, and pushes;
- reruns the failed jobs once if nothing in the code is involved; or
- reports that the failure is main's and does not touch the PR. For a failure
  on main it opens a `ci:` PR with label `loop-ci` only for a mechanical fix,
  and otherwise posts a diagnosis.

A failure diagnosed as main's is remembered for 48 hours, so the same failing
test on later commits does not start another card. Each PR gets at most three
fixer cards. A run does not start while a fixer holds its branch. The
gatekeeper checks a pushed fix touched only the PR's files, and flags a fix
PR that edits a hash-bound file. Main failures, diagnoses and flagged fixes
mention the owner; routine fixes do not. Results posts include the PR's CI
state.

## Owner commands in #thermes

Reply to the gatekeeper's post, or name the proposal (`approve P-0003`).
Only messages from the Discord allowlist count; the THERMES chat agent is
told to leave these to the gatekeeper.

| Command | Effect |
| --- | --- |
| `approve` | Approves the announced protocol; its run starts when no other study is running |
| `revise <notes>` | Sends the draft back to the proposer with your notes |
| `answer <text>` | Answers a proposer's question and restarts the draft |
| `reject <why>` | Closes the draft PR; the row is skipped until its queue text changes |
| `pause loop` / `resume loop` | Stops or restarts new proposals and runs; running work continues |
| `loop status` | Lists active proposals and the week's spend |
| `adopt D-NNNN` | Opens the queue edit proposed by that direction review as a PR for the owner to merge |

Your words stay local: rejections and revision notes are kept in the
THERMES profile and given to the proposer, never posted to GitHub.

## Limits

- At most one proposal waiting for the owner, one study running, and two
  studies not yet accepted.
- At most one direction review open at a time, and at most one new queue row
  proposed per review.
- Probe at most 30 CPU-minutes; a production run over 8 CPU-hours is flagged
  for the owner. Run cards get a runtime cap of 2 × estimate + 3 hours
  (4 to 24 hours).
- No hard spend cap (owner, October 8, 2026). The gatekeeper alerts on
  Discord instead, using Hermes' list-price estimates (the Claude plan is not
  billed per token): each time 7-day loop spend crosses a multiple of $100,
  when a proposal costs over $15 in drafting or $50 in its run (and again at
  each doubling), and when a card runs well past its expected time. A cap can
  be restored with `weekly_budget_usd` in the loop config.
- Out of bounds: the conservation line, reading A arms, a third adaptive
  policy, the parked harness, the frustrated-grid/annealing track (recorded
  in PR #98; its follow-ups are notes, not queue rows, under charter
  amendment 1), hardware, network
  services, new dependencies, and anything CLAUDE.md says waits for the
  owner.

## Guardrails

The owner chose a soft guard on October 8, 2026. THERMES uses the owner's
GitHub login and `main` is not branch-protected, so these stop mistakes, not
a determined bypass:

- A `gh` wrapper on the THERMES and THERMES-REVIEW `PATH` refuses PR merges,
  approvals and API calls that merge or move `main`.
- A `pre-push` hook in the owner's checkout and the THERMES clone refuses
  pushes to `main`, tag pushes, branch deletions and force-pushes when
  `THERMES_LOOP_GUARD=1`, which only THERMES shells set. It does nothing for
  anyone else.
- The executor and reviewer check the protocol diff against the approved
  commit; the gatekeeper flags proposal PRs that touch files outside
  `docs/experiments/`, `docs/research/` and the queue.
- The gatekeeper flags a direction-review PR that touches anything outside
  `docs/research/`, and a director card that comments on a study PR.

## Operating

```bash
# status, or a dry run that prints what the next tick would do
python3 ~/.hermes/profiles/thermes/scripts/loop-tick.py --status
python3 ~/.hermes/profiles/thermes/scripts/loop-tick.py --dry-run
# the cron job (owner runs this once)
thermes cron create '*/5 * * * *' --name thermo-loop --script loop-tick.py --no-agent --deliver discord
# a stuck card, after fixing its cause
thermes kanban --board thermo unblock <card>
```

State is in `~/.hermes/profiles/thermes/state/loop.json`, with an event log in
`loop-log.jsonl` beside it. Pausing the cron job (`thermes cron pause <job id>`,
id from `thermes cron list`) stops the loop entirely; `pause loop` on Discord
stops new work but keeps reporting.
