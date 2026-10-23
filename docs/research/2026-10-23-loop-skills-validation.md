# Doctor and Debrief integration: offline validation

2026-10-23. Documentation and skill integration only; no scientific run,
scheduler activation or external-profile change. Base: `539d1aac`.

## Scope and source

Use Debrief to explain observed runs, then Doctor only for supported design
amendments. Route requests through the existing research-loop review/handoff
(`docs/research-loop.md`, Doctor and Debrief). Two local skills plus one
shared diagnostic contract are sufficient; a new daemon, executable
classifier or runtime state schema would require implementation evidence
from the external Hermes profile.

Adapted from Forward Future Loopy at
[`75966cbd572a4185064971c9fe5e9c52e8f8456d`](https://github.com/Forward-Future/loopy/tree/75966cbd572a4185064971c9fe5e9c52e8f8456d/skills/loopy):
`references/audit.md` and `references/debrief.md`, with routing from `SKILL.md`.
The MIT notice is retained in `.agents/skills/LOOPY-LICENSE.txt`.
These are unpublished Thermo adaptations, not an installation of full Loopy
or a claim to match the latest published catalog.

## Method and baseline

Read-only fresh-agent application tests used stipulated scenarios, without
network access, file mutation or experiments. They test instruction behavior,
not scheduler execution. The baseline inspected the pre-change repository:
it respected scientific stops but explicitly found no Doctor/Debrief contract,
called Doctor "operational diagnosis", and did not identify the no-feedback
case as a one-shot workflow. This motivates explicit routing, cause separation
and outcome/handoff fields; it does not demonstrate a live Hermes defect.

## Offline cases

| Case | Required behavior |
| --- | --- |
| Sound bounded loop; no numeric retry cap | Doctor: Ready, no repair; existing no-progress stop is sufficient. |
| Feedback cannot change next action | Doctor: Not actually a loop; recommend one-shot work. |
| Prompt explicitly permits cached-row launch after failed refresh | Doctor: Repair needed; replace only that fallback with blocked/no launch pending fresh state. |
| Completed negative and no open rows despite demand for 24/7 discovery | Debrief: process can have worked; scientific stop remains, no open row, new topic requires owner selection. |
| Last study old; no host state/log | Inconclusive about scheduler health; no inferred crash or missed schedule. |
| API error; actual fallback unknown | Environment/tool evidence, blocked freshness; no speculative prompt rewrite. |
| Head changed for runner code; approved protocol unchanged; worker unknown | Do not invalidate approval from head inequality alone or launch a duplicate; obtain worker/handoff evidence. |
| One attempt ignored a correct fresh-state instruction | Execution issue, not a recurring pattern or proven design defect. |

Doctor application checks passed for the first three cases. Pair application
checks passed for the remaining five: no unjustified repair, restart or
reapproval; missing live evidence remained unknown. Review identified one
output ambiguity: Doctor's upstream verdict set had no missing-target option.
The Thermo adaptation adds `Inconclusive` so an uninspected design cannot be
labeled Ready. The focused recheck passed: unavailable fallback design yields
Inconclusive; an inspected sound repository design yields Ready while live
scheduler health remains unknown.

Structural checks passed: minimal skill frontmatter, every local link target
in the two skills, the loop documentation and this note (rechecked after the
revision below), and `git diff --check`.
Protected sources, frozen protocols, charter, queue and workflows are unchanged.
No production code was changed; no scientific test suite or replay was run for
this documentation-only change. These finite application checks do not prove
universal agent compliance or runtime liveness.

## Follow-through boundary

Before runtime wiring, inspect the existing Hermes script, templates, cron
status and state/event records read-only. Check whether it loads repository
instructions and already records freshness, outcomes and handoffs. Inspect
status/dry-run side effects before invoking them. Any later implementation
should use offline fixtures for refresh failure, repeated ticks, stale cards,
approval chronology and an intentionally empty queue, with no external sends
or experiment launches. The actual profile is absent from this workspace;
live health and adoption remain unverified. Publication and runtime changes
remain separate decisions.

## Revision after review

A host-side review found that the gatekeeper already reports an idle queue
(`Loop idle: no open, unclaimed rows in docs/research-queue.md.`, posted once)
and that `loop-tick.py --status` lists waiting items and active proposals.
The draft's six-value outcome table was a second vocabulary that no code
emits, so it was replaced by a pointer to those reports. The AGENTS.md
paragraph was dropped, since the research-loop section already routes
requests and AGENTS.md is loaded into every session. Neither Claude Code
(`.claude/skills/`) nor the thermes profile (`skills.external_dirs: []`)
discovers `.agents/skills/`; the skills are reached through the
research-loop links. The October 23 cloud inventory (four CPU equivalents
on an AMD EPYC 9V74, 16 GiB enforced, no GPU, Python 3.12 without the
project environment) moved to the Codex cloud column of
`docs/environments.md` for planning future cloud runs. The host
inventory is verified: the local box is `zad1`, an i7-7700K, as recorded
there.
