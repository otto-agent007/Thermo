# Doctor and Debrief integration: offline validation

2026-10-23. Documentation and skill integration only; no scientific run,
scheduler activation or external-profile change. Base: `539d1aac`.

## Scope and source

Use Debrief to explain observed runs, then Doctor only for supported design
amendments. Route requests through `AGENTS.md` and the existing research-loop
review/handoff. Two local skills plus one shared diagnostic contract are
sufficient; a new daemon, executable classifier or runtime state schema would
require implementation evidence from the external Hermes profile.

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
| Completed negative and no open rows despite demand for 24/7 discovery | Debrief: process can have worked; scientific stop remains, queue exhausted, new topic requires owner selection. |
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

Structural checks passed: minimal skill frontmatter, all 37 local link targets
in the two skills, AGENTS and loop documentation, and `git diff --check`.
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

The owner reports an older Intel i7 (possibly i7-7700K), 32 GB RAM and a
GTX 1050 Ti on the Thermo host. This inventory is not host-verified. Verify it
before recommending concurrency or GPU settings; this cloud workspace's
resources are not evidence about that host. No compute-setting change follows
from this skills integration.

## Cloud executor inventory (2026-10-23)

Read-only resource inspection found five affinity-visible logical CPUs but a
`cpu.max` quota of `400000 100000`: four CPU equivalents, not five guaranteed
cores. The reported model is AMD EPYC 9V74. `memory.max` is 16 GiB; the
`/proc/meminfo` view is about 17.6 GiB and is not the enforced allocation.
Available space at inspection was about 29.8 GiB in `/workspace` and 8.8 GiB
in `/tmp`. No NVIDIA device nodes or `nvidia-smi` were available; GPU execution
is not established.

The default Python is 3.12.14 and uv is 0.12.19. Thermo requires Python 3.11;
there is no project virtual environment and no JAX/JAXlib or pytest in the
default interpreter. The lockfile pins JAX/JAXlib 0.10.2, but that is not an
installed/runtime-verified backend. Hermes/THERMES are absent. The default uv
cache is read-only; future setup must use a writable cache location. No
dependencies were installed and no JAX smoke or scientific run was performed.

This suggests a possible CPU host for bounded work after compatible frozen
setup and an approved question, not a continuously available worker. Session
lifetime, sustained performance and GPU availability remain unverified. Before
an approved run, reconcile the existing row/card with the other host to prevent
duplicate execution; record this host's actual provenance and limits, calibrate
within the existing budget, and follow the study's atomic checkpoint/resume and
replay contract. Keep CPU measurements separate from hardware cost projections.
No open study is selected by this inventory; denoising remains closed and the
tested adsorption allocation held.
