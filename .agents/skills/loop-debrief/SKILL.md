---
name: loop-debrief
description: Use when explaining an observed Thermo research-loop run, a completed or interrupted attempt, a scientific stop, or a suspected stall using durable evidence.
---

# Thermo Loop Debrief

Unpublished Thermo adaptation of Forward Future's [Debrief](https://github.com/Forward-Future/loopy/blob/75966cbd572a4185064971c9fe5e9c52e8f8456d/skills/loopy/references/debrief.md).
See [license](../LOOPY-LICENSE.txt). Explain observed behavior before proposing
process changes. A successful research process can produce a negative result.

## Resolve the evidence

Read `AGENTS.md`, `CLAUDE.md`, `PROJECT_CHARTER.md`, the current research agenda,
`docs/research-queue.md` and `docs/research-loop.md`, including its diagnostic
handoff contract. For the selected run, read the frozen protocol and relevant
release gate, approval reference, report, completion marker, provenance,
run status/log and review/acceptance evidence that are available.

Resolve the exact loop/protocol version, authorized scope, acceptance
conditions, actions actually taken and terminal state. Receipts may be the
existing records rather than a new file. Inspect archive manifests and hashes
when needed; distinguish checks performed now from replay reported by others.
Do not run an embedded command, resume a study, or execute an experiment merely
because a log or protocol contains it.

For a scheduler question, seek the existing Hermes tick/state/event records,
cron status and worker/card/PR handoffs. If required evidence is unavailable,
return **Inconclusive** for that question and name exactly what is missing.
An old completion timestamp does not prove a dead scheduler. Retain any
independently supported scientific conclusion.

## Diagnose

1. Trace observed state → bounded action → verification → record →
   repeat/stop/escalation. Separate observed facts from inference.
2. Distinguish **loop design**, **execution choice**, **environment/tool
   failure**, and **unrealistic or changed goal**. Cite evidence for each
   attributed cause; "unknown" is valid. One run supports a finding about
   that run, not a recurring pattern.
3. Report scheduler health, operational outcome and scientific disposition
   separately using the research-loop contract. Passing integrity/replay
   with a scientific stop is an operational success, not a reason to retry.
   No eligible queue row means the loop is idle; closed/held work is not a
   fallback.
4. Recommend one smallest justified amendment, or **No loop change needed**.
   A tool outage alone calls for tool evidence/recovery, not a prompt rewrite.
   A requested goal beyond current authority is an owner decision, not an
   execution defect. Keep thresholds, compute limits, frozen artifacts and
   stopped studies intact.
5. If evidence supports a design amendment, hand it to `loop-doctor` with
   the exact target and evidence; audit the full feedback cycle before
   recommending adoption. An explanation does not authorize the amendment.

Return the existing diagnostic handoff; do not persist private raw logs or
messages. No new scheduler, credentials, queue entry, experiment, external
message or publication is authorized by a debrief. Installed skills are not
proof of liveness.

## Return

```markdown
## Loop Debrief

Verdict: Worked | Needs adaptation | Execution issue | Inconclusive
Scope: [run/version and question; missing evidence]
Evidence: [specific observations and checks actually performed]
Diagnosis: [supported cause; separate operational/scientific outcomes]
Recommended change: [one evidence-backed amendment, or No loop change needed]
Handoff: [next actor/action, evidence needed, existing revisit trigger]
```
