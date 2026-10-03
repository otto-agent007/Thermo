# Execution environments

*Draft, October 3, 2026 (local and in-session cloud figures measured October 1;
published cloud figures checked October 3). Where Thermo work runs, what each place can hold,
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
| Storage | Repo on `/mnt/2TBHDD` (about 1.6 TB free); root disk about 13 GB free | October 1 instance: about 32 GiB workspace, 29 GiB free; `/tmp` and `/dev/shm` about 17 GiB each and RAM-backed; check mounts in each session |
| Open files / stack | 1,048,576 / 16 MiB | 16,384 per process / 8 MiB per thread |
| Network | Open | Restricted; package-manager destinations allowed, other downloads may be blocked (a Chromium download was) |
| Persistence | Local files persist | Saved task VM state is recoverable for up to seven days by default; do not depend on background processes surviving a reply (see the checkpoint rule) |
| Time limits | None | No per-turn maximum is published; observed single-turn cutoff: **not yet measured** (fill in after the probe below, with the date) |
| Sandboxing | `bwrap` 0.9 installed; `thermo-harness` runs | Commands already run in a nested sandbox; `thermo-harness` needs approved execution outside it |
| Toolchain | Python 3.11 via `uv`, Node 26, Codex CLI, Hermes Agent | `uv sync --frozen` on setup; Node only if the dashboard is in scope |

## Rules that follow

- **Thread settings.** For CPU gates set `JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1` and give the runner its parallelism through its worker
  flags. On the cloud box keep simultaneously active workers at four or
  fewer. Consult each runner: secondary flags such as M5a's `--fit-workers`
  and M5c's `--placement-workers` can govern separate phases and need not
  be added together. The gate commands were sized for an 8-CPU session.
  The cloud commands are spelled out in the fit table. On the local box,
  three to four workers leave headroom for the owner.
- **Memory.** Budget for the published 16 GiB on the cloud box, not the
  32 GiB one instance happened to report, and there is no swap to absorb a
  spike. M5b peaked at 0.89 GiB per worker and 4.56 GiB for the session; the
  M5a four-worker dense matrix probe peaked at 6.33 GiB total cgroup memory
  (see its [dated protocol amendment](experiments/meta-ebm-cap-baseline.md#september-27-2026--durable-checkpointing-and-four-worker-matrix-profile)).
  Those measured peaks fit in 16 GiB; they do not bound every target set.
  Keep temporary arrays out of `/tmp` when its mount is RAM-backed.
- **Checkpoints.** Use durable files rather than assuming a background
  process survives a reply. The docs describe saved task VM recovery for up
  to seven days; legacy setup-container caching for up to 12 hours is a
  separate mechanism, not a process-lifetime guarantee. Write every output
  directory under the workspace (`results/` in the repo), never `/tmp`, and
  treat every study as interruptible. Of these studies, only **M5a and M5b**
  have an autosave layer and a `--resume`
  flag. M5c, M4H and M4I do not: an interrupted run starts over in a fresh
  directory, and M4H's evaluator is hash-bound, so autosave cannot be added
  to it. Any new study over about 30 minutes must follow the autosave
  contract.
- **Evidence leaves by PR.** The cloud box has no path to the local
  dashboard. Archives come back as gzipped files in a PR; keep them bounded
  per CLAUDE.md, and ask the owner before anything over a few megabytes.
- **Browser checks need a configured browser.** The October 1 cloud session
  blocked the Chromium download. Prefer the local box or GitHub CI for
  `npm run test:browser`; a cloud session can run the CLI tests if Chromium
  and its system dependencies are installed and downloads are permitted.
  Dashboard unit tests, typecheck and export do not need a browser.

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
| M5c topology | 17.5 min on 8 CPUs | Yes, subject to one-seed timing | Replay the archive by default. A reduced-worker estimate of 30–40 min exceeds the frozen protocol's no-autosave condition: calibrate below 30 min or amend the gate to add autosave before launching. Placement solves can differ between machines |
| M5a cap baseline | About 60 min on 8 CPUs | Yes | As `--resume` turns: `--workers 2 --fit-workers 4`, same directory each turn |
| M5b thermalization | 2–3 h with three workers on 8 CPUs | Yes, overnight | As `--resume` turns with `--workers 3`, once the single-turn cutoff below is measured and each turn is sized under it |
| `thermo-harness` candidates | Minutes to 30 min | Yes | No (nested sandbox) |
| Dashboard browser tests | Minutes | Yes | Only with Chromium and system dependencies available; download access depends on configuration |
| Snapshot publication | Minutes | Yes | Export can run here; publication needs the intended destination and owner approval |

## What OpenAI publishes about the cloud box

Checked October 3, 2026 against the Codex docs (developers.openai.com now
redirects to learn.chatgpt.com) and the public issue tracker.

- **VM size by plan:** Plus gets 2 vCPUs, 8 GiB memory and 8 GiB disk; Pro,
  Business, Enterprise and Edu get 4 vCPUs, 16 GiB and 32 GiB. The instance
  measured on October 1 reported 32 GiB of memory; that observation does not
  change the published default. Plan on 16 GiB for the four-vCPU plans and
  check the live limits.
- **Legacy setup-container cache:** state is cached for up to 12 hours after the setup
  script completes, and invalidated when the setup script, maintenance
  script, environment variables or secrets change.
- **Saved state:** a task's VM state is recoverable for up to seven days
  after its last turn, so a follow-up message resumes the workspace instead
  of rebuilding it.
- **Setup timing:** the linked legacy timeout thread discusses a change
  from five to ten minutes. It is not a guarantee for every current setup
  path. Measure `uv sync --frozen`, `npm ci` and browser installation in the
  intended environment rather than assuming they fit a fixed budget.
- **Network:** the current docs describe configurable internet access,
  including a package-manager preset and additional allowed domains; legacy
  docs describe agent-phase internet being off by default. Check the actual
  configuration and all download hosts before installing dependencies.
- **Not published:** any maximum turn duration or idle timeout. OpenAI's
  launch guidance was that most tasks take one to thirty minutes. Issue
  reports describe interruption or lost background processes on particular
  execution paths. Some concern the Windows desktop app rather than cloud
  VMs. They motivate conservative checkpoints, not a universal rule that all
  processes end at every reply.

Sources: [cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environments),
[legacy cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environment),
[setup-script timeout thread](https://community.openai.com/t/codex-changelog-indicates-setup-script-timeout-is-now-10-minutes-but-mine-timing-out-at-5/1278352),
[openai/codex#24084](https://github.com/openai/codex/issues/24084),
[openai/codex#43882](https://github.com/openai/codex/issues/43882),
[openai/codex#45264](https://github.com/openai/codex/issues/45264).

## What that means for studies

- **Checkpoint before yielding.** Plan for a study to finish, or reach a
  durable checkpoint, before the agent replies (see the checkpoint rule);
  process survival across replies has not been established for each runtime.
- **Chunk by turn, resume by message.** For M5a and M5b: start the gate
  command, let it autosave, and send the same command plus `--resume` as the
  next message, in the same output directory. Size each turn under the
  observed single-turn cutoff once the probe below has produced one.
- **Results must land in the workspace or a PR.** Do not rely on `/tmp` or
  in-memory state for recovery. Archives come back gzipped and bounded.

## The one probe still worth running

The maximum length of a single turn remains unmeasured. Give
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
