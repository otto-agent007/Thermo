# Execution environments

*Draft, October 2, 2026 (local and in-session cloud figures measured October 1;
published cloud figures checked October 2). Where Thermo work runs, what each place can hold,
and which gates fit where. Figures here are operational observations, not
evidence; a runner records the limits it actually saw in its own
`run-status.json` (see the autosave contract in
[experiment-runner.md](experiment-runner.md#autosave-and-resume-contract)).*

## Why this exists

The M5b calibration in
[2026-09-28-m5b-runtime.md](research/2026-09-28-m5b-runtime.md) was measured
on a cloud session with 8 CPUs and an 8 GiB cgroup cap. The cloud session
available on October 1 has 4 cores and 32 GiB. A worker count or memory
budget copied from one gate into the other environment is wrong in both
directions. Check this table, then check the live limits, before starting a
study that runs longer than a few minutes.

## Environments

| | Local workstation | Codex cloud (OpenAI) |
| --- | --- | --- |
| CPU | Intel i7-7700K, 4 cores / 8 threads, no cgroup CPU limit | Quota of 4 cores' CPU time; 5 logical CPUs visible |
| Memory | 31 GiB, 2 GiB swap | 16 GiB published for the plan; the October 1 instance reported 32 GiB; no swap |
| GPU | GTX 1050 Ti (4 GiB); **not used**, owner declined GPU work | None |
| Storage | Repo on `/mnt/2TBHDD` (about 1.6 TB free); root disk about 13 GB free | About 32 GiB workspace, 29 GiB free; `/tmp` and `/dev/shm` about 17 GiB each and counted against RAM |
| Open files / stack | 1,048,576 / 16 MiB | 16,384 per process / 8 MiB per thread |
| Network | Open | Restricted; package-manager destinations allowed, other downloads may be blocked (a Chromium download was) |
| Persistence | Everything persists | Processes end with the agent's turn; the workspace survives between turns (see the checkpoint rule) |
| Time limits | None | No per-turn maximum is published; observed single-turn cutoff: **not yet measured** (fill in after the probe below, with the date) |
| Sandboxing | `bwrap` 0.9 installed; `thermo-harness` runs | Commands already run in a nested sandbox; `thermo-harness` needs approved execution outside it |
| Toolchain | Python 3.11 via `uv`, Node 26, Codex CLI, Hermes Agent | `uv sync --frozen` on setup; Node only if the dashboard is in scope |

## Rules that follow

- **Thread settings.** Always set `JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1` and give the runner its parallelism through its worker
  flags. On the cloud box keep the **sum** of concurrent processes at four,
  including secondary flags such as M5a's `--fit-workers` and M5c's
  `--placement-workers`; the gate commands were sized for an 8-CPU session.
  The cloud commands are spelled out in the fit table. On the local box,
  three to four workers leave headroom for the owner.
- **Memory.** Budget for the published 16 GiB on the cloud box, not the
  32 GiB one instance happened to report, and there is no swap to absorb a
  spike. M5b peaked at 0.89 GiB per worker and 4.56 GiB for the session; the
  M5a gate's eight fit workers peaked at 6.33 GiB. Both fit in 16 GiB. Keep
  temporary arrays out of `/tmp` on the cloud box; they count as RAM.
- **Checkpoints.** This is the one authoritative statement of the cloud
  lifecycle: processes end when the agent's turn ends; the workspace is kept
  (container cache 12 h, saved VM state 7 days) and resumes on the next
  message. So on the cloud box write every output directory under the
  workspace (`results/` in the repo), never `/tmp`, and treat every study as
  interruptible. Only **M5a and M5b** have an autosave layer and a `--resume`
  flag. M5c, M4H and M4I do not: an interrupted run starts over in a fresh
  directory, and M4H's evaluator is hash-bound, so autosave cannot be added
  to it. Any new study over about 30 minutes must follow the autosave
  contract.
- **Evidence leaves by PR.** The cloud box has no path to the local
  dashboard. Archives come back as gzipped files in a PR; keep them bounded
  per CLAUDE.md, and ask the owner before anything over a few megabytes.
- **Browser checks stay local.** Playwright needs a Chromium download the
  cloud network blocks. Dashboard unit tests, typecheck and export can run
  anywhere; `npm run test:browser` runs on the local box or in GitHub CI.

## Which gates fit where

Durations are the ones recorded in AGENTS.md and the gates; the cloud box
has half the cores of the 8-CPU session that produced most of them, so
expect roughly 1.5–2× on multi-worker studies there.

| Work | Recorded cost | Local | Codex cloud |
| --- | --- | --- | --- |
| `pytest tests/unit -m "not slow"`, Ruff, lock check | Minutes | Yes | Yes |
| Base `thermo-lab run` gates (AGENTS.md list) | Minutes each | Yes | Yes |
| Archive replays (M4H, M4I, M5a, M5b, M5c replay-only tests) | Minutes | Yes | Yes |
| Knowledge-base checks (`thermo_lab.knowledge_base`, `--upstream`) | Seconds; `--upstream` needs arXiv, PyPI and GitHub | Yes | Structure only unless network allows |
| M4H raised-cap screen | About 25 min on 8 CPUs | Yes | Risky: no autosave, 40–50 min projected, hash-bound; replay the archive instead |
| M5c topology | 17.5 min on 8 CPUs | Yes | Single turn only, no autosave; expect 30–40 min with `--workers 2 --placement-workers 2`; placement solves can differ between machines |
| M5a cap baseline | About 60 min on 8 CPUs | Yes | As `--resume` turns: `--workers 2 --fit-workers 4`, same directory each turn |
| M5b thermalization | 2–3 h with three workers on 8 CPUs | Yes, overnight | As `--resume` turns with `--workers 3`, once the single-turn cutoff below is measured and each turn is sized under it |
| `thermo-harness` candidates | Minutes to 30 min | Yes | No (nested sandbox) |
| Dashboard browser tests, snapshot publication | Minutes | Yes | No |

## What OpenAI publishes about the cloud box

Checked October 2, 2026 against the Codex docs (developers.openai.com now
redirects to learn.chatgpt.com) and the public issue tracker.

- **VM size by plan:** Plus gets 2 vCPUs, 8 GiB memory and 8 GiB disk; Pro,
  Business, Enterprise and Edu get 4 vCPUs, 16 GiB and 32 GiB. The instance
  measured on October 1 reported 32 GiB of memory, so the published memory
  figure is a floor, not a promise. Plan on 16 GiB.
- **Container cache:** state is cached for up to 12 hours after the setup
  script completes, and invalidated when the setup script, maintenance
  script, environment variables or secrets change.
- **Saved state:** a task's VM state is recoverable for up to seven days
  after its last turn, so a follow-up message resumes the workspace instead
  of rebuilding it.
- **Setup script:** times out at 10 minutes (raised from 5). `uv sync
  --frozen` fits; a dashboard `npm ci` plus Playwright does not, and the
  browser download is blocked anyway.
- **Network:** blocked during the agent phase by default except package
  managers and GitHub hosts; setup scripts have internet; unrestricted access
  is a per-environment switch.
- **Not published:** any maximum turn duration or idle timeout. OpenAI's
  launch guidance was that most tasks take one to thirty minutes. Issue
  reports from May to September 2026 describe containers ending as soon as a
  reply is sent, long tasks ending after a progress update, and the agent
  reporting that no background process is running on the next turn.

Sources: [cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environments),
[legacy cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environment),
[setup-script timeout thread](https://community.openai.com/t/codex-changelog-indicates-setup-script-timeout-is-now-10-minutes-but-mine-timing-out-at-5/1278352),
[openai/codex#24084](https://github.com/openai/codex/issues/24084),
[openai/codex#43882](https://github.com/openai/codex/issues/43882),
[openai/codex#45264](https://github.com/openai/codex/issues/45264).

## What that means for studies

- **One turn is the unit of execution.** A study must finish, or reach a
  durable checkpoint, before the agent replies (see the checkpoint rule).
- **Chunk by turn, resume by message.** For M5a and M5b: start the gate
  command, let it autosave, and send the same command plus `--resume` as the
  next message, in the same output directory. Size each turn under the
  observed single-turn cutoff once the probe below has produced one.
- **Results must land in the workspace or a PR.** Anything written to `/tmp`
  or left in a process is lost. Archives come back gzipped and bounded.

## The one probe still worth running

The maximum length of a single turn is the only limit left unmeasured. Give
a fresh cloud task this prompt and read the last printed heartbeat when it
ends:

```text
Create results/env-probe/ and run this in the FOREGROUND, without
backgrounding it, until it exits or is killed:

  for i in $(seq 1 720); do
    date -u +%FT%TZ | tee -a results/env-probe/wallclock.log
    sleep 60
  done

Print every heartbeat so it appears in the transcript. When the loop ends,
report the first and last timestamps and whether all 720 ran. Do not modify
other files and do not open a PR.
```

Record the observed cutoff and date in the "Time limits" row of the
environments table. Then confirm autosave on the cloud file system once with
a runner that has it: start M5a with `--workers 2 --fit-workers 4`, let the
turn end after about ten minutes, and send the same command plus `--resume`.
The run log must show the saved units reused, not regenerated.

## Agent roles

Which model does which job is an operating choice, not evidence, and it
changes as models do. The current practice:

| Role | Where it runs | Notes |
| --- | --- | --- |
| Protocol design, synthesis notes, decisions | Strongest reasoning tier available in Claude Code or Codex | One vendor drafts, the other reviews; keep that split |
| Runner and test implementation | Mid tier, same session as the protocol | Must leave hash-bound evaluators untouched |
| Independent review of a PR | The vendor that did not write it | Review findings go in the PR, not in evidence |
| Study execution, replays, gate runs | No model; a script or a Hermes cron/kanban task | Owner approves anything that becomes evidence |
| Knowledge-base upkeep (`--stale-after`, `--upstream`, link checks) | No model, scheduled | Opens an issue or note when something changed upstream |

Hermes Agent on the local box has cron, a kanban board and a Discord gateway
and is the intended scheduler; it needs a Thermo-specific profile before use
because its default identity file belongs to another project.
