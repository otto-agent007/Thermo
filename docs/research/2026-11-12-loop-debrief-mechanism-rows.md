# Loop debrief and design audit: the three mechanism rows

2026-11-12 (local; GitHub times below are UTC, 2026-11-13). Written in an
owner session with Claude at the owner's request, following
[Loop Debrief](../../.agents/skills/loop-debrief/SKILL.md) and then
[Loop Doctor](../../.agents/skills/loop-doctor/SKILL.md).
**Advice, not a decision, and a repository design audit only.** The Hermes
profile (`loop-tick.py`, card templates, `loop.json`, the event log) was not
inspected, so scheduler health is unknown and nothing here proves what the
profile loads. One evening's run supports findings about that run, not a
recurring pattern.

## What was observed

Evidence is GitHub PR metadata and bodies read on 2026-11-13 between 03:39Z
and 06:05Z, and the files those PRs carry.

| Time (UTC) | Event | Locator |
| --- | --- | --- |
| 03:39 | Queue PR opens three rows, with a proposal note | PR #155 |
| 03:42 | Owner merges it, about four minutes later, CI still running | PR #155 |
| 04:08 | P-0011 `margin-budget-copies` probe PR | PR #156 |
| 04:29 | Owner merges P-0011 | PR #156 |
| 04:34 | P-0012 `aggregator-read-gain` probe PR | PR #157, open |
| 04:40 | D-0010 direction review, with a premise question on `directed-relay-latent` | PR #158, open |
| 05:22 | P-0013 `directed-relay-latent` probe PR | PR #159, open |
| 05:54 | Source cards for the three rows | PR #160, open |

- **Scheduler health:** unknown. Three proposals and one review appeared in
  under two hours, which shows cards were created and worked; nothing more is
  claimed.
- **Operational outcome:** three probes completed inside the 30 CPU-minute
  ceiling and each answered its row with a docs-only PR, as
  [research-loop.md](../research-loop.md) step 1 allows. Two lost or dropped
  work on the way (below).
- **Scientific disposition:** one positive with a missed strict prediction
  (P-0011), one positive only under relaxed placement and caps (P-0012), one
  negative (P-0013). None is an integrity failure and none calls for a retry.

## Diagnosis

### Loop design (three findings, each with its evidence)

1. **An owed source card never comes due when a probe answers the row.**
   All three rows made a card a precondition "before any protocol". All three
   closed by probe, so no protocol followed and no card was written (PR #156,
   #157 and #159 bodies, "Not done"; D-0010, "Evidence obligations"). The
   cards written afterwards (PR #160) show the aggregator's ceiling is the
   published π/2 one-bit loss. The reading that would have changed the row
   happened after the row was spent.
2. **A direction review's question does not hold the row it names.** D-0010
   asked the owner whether `directed-relay-latent` was a requirement or a
   third construction, and advised against spending the probe in the second
   case (PR #158, 04:40Z). The probe PR for that row opened 42 minutes later
   (PR #159, 05:22Z) with no owner answer on record. The proposer followed
   the row as written: its checkpoint was "ask if `aggregator-read-gain`
   stops", and that row did not stop. D-0010 also named cheaper desk checks
   for both open rows after one probe was drafted and before the other; no
   step delivers that advice to the proposer.
3. **A probe can lose its work, and trims its grid to fit the ceiling.**
   P-0011's first run crashed in post-processing after 1,146 CPU-seconds with
   nothing saved, and its starts were cut from 16 to 6 per seed (D-0010,
   "Operational outcome"). P-0013 had two stopped runs and one crash, reused
   logged cells, and dropped its chromatic r = 2 finite-relay cells (PR #159
   body). The autosave contract applies only above 30 minutes, which no
   probe reaches by construction.

### Execution choices in the owner session (not loop defects)

- **A precondition was written as a note, not a status.**
  `directed-relay-latent` went in `open` with "runs after
  `aggregator-read-gain` and depends on it" in its notes (PR #155). The queue
  has a status for that: `held`.
- **The proposal note was not checked before rows cited it.** It claimed the
  margin law was consistent with M4I's 89.61 percent; P-0011 withdrew that
  (different kernel family). PR #155 merged in about four minutes.
- **A row's comparison assumed a failure that was not first reproduced.**
  `directed-relay-latent` compared relays against tied copies "where tied
  copies at finite tie do not" reach 0.01. On the reduced instance tied
  copies reached 0.0012 (PR #159), so the comparison could not be made.
- **A fixed choice that bounds the metric was missing from the row.** The
  update schedule (two-colour against sequential) decided P-0013's answer and
  was not among the row's declared choices.

### Not a process finding

All three rows were mechanism questions with no task. The four-week
evaluation's un-park condition (a task with digital cost headroom, native
placement, a user and a tolerance) is unmet, and D-0010 says so. A faster or
tidier loop would not have changed that. Whether a row must name a digital
baseline and its cost before it opens is a direction decision for the owner
and is not proposed here as a rule.

## Minimal amendments

The companion edits in this PR are the repository half. Each is the smallest
change the evidence supports.

| Finding | Amendment | Where | Offline check that tells new from old |
| --- | --- | --- | --- |
| Design 1 | A row's owed card is due before its probe result PR opens, at abstract level if need be and labelled so | `research-loop.md` step 1; queue header | A probe PR for a row that owes a card either adds the card or is flagged |
| Design 3 | A probe saves each finished cell as it finishes, and times one cell before fixing its grid | `research-loop.md` step 1 | Kill a probe mid-run; finished cells are on disk |
| Session: comparison | A probe reproduces the failure or baseline its row compares against before any main arm, and stops if it cannot | `research-loop.md` step 1 | The probe note's first result is the control |
| Session: note as status | A row waiting on an answer or on another row is `held` | queue header | No `open` row has "ask the owner" or "runs after" in its notes |
| Session: unchecked note | A queue PR that cites a new proposal note waits for CI and one second read | queue header | The queue PR shows a review comment before merge |

Not amended: thresholds, the 30 CPU-minute ceiling, approval binding, the
review lane, the director's limits. Design 2 has no repository half; it is
below.

## Spec for the Hermes profile (not applied; the profile was not inspected)

These need the gatekeeper or the card templates, which live outside this
repository. Inspect the profile before wiring any of them.

1. **Hold on an open question.** While a `loop-direction` PR is open and its
   body or file puts a question to the owner that names a queue row, the
   gatekeeper does not create a propose card for that row. It resumes when
   the owner answers on Discord, or the review PR is merged or closed. The
   idle post should say which row is waiting on which review. Check: replay
   the 04:40Z state with `--dry-run`; the next tick must not card
   `directed-relay-latent`.
2. **Deliver the director's cheapest check.** When a direction review names a
   cheaper check for an open, unclaimed row, the propose card for that row
   quotes it, and the proposer either runs it first or says why not. Check:
   the P-0013 propose card would have carried D-0010's desk calculation.
3. **Card-template lines for the probe rules above.** The propose template
   should carry the three step-1 rules (card before result PR, control first,
   save per cell) verbatim, because the repository text is not proof the
   profile loads it.
4. **Optional:** `docs/knowledge/lessons.md` conflicts whenever two result
   PRs are open (PR #157 and #159 both note it). Either the CI fixer keeps
   resolving it, as it did on #157, or the register moves to one file per
   lesson. No change is proposed here.

## Handoff

- **Inspected:** main at 34fe372; PRs #155 to #160 as of 06:05Z; no host
  evidence.
- **Outcome:** loop operationally sound on this run; three design weaknesses
  with evidence, four session-side mistakes.
- **Missing evidence:** `loop.json`, the event log, the propose and direct
  card texts, and whether the proposer saw D-0010 before starting P-0013.
- **Next actor:** the owner, to merge or close this PR and to decide whether
  the profile items are worth wiring.
- **Authorizes nothing:** no run, no queue row, no scheduler change.
