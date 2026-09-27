# Running the first continuous research cycle

The first cycle executes the frozen M5a study on one durable Linux CPU host,
then hands its evidence to an independent reviewer in ChatGPT Work. The worker
does not choose new experiments. A negative result can be a successful cycle.

Implementation and deployment are separate. No production worker, scheduled
review, completed M5a archive, or live dashboard update is established by this
change. The implementation workspace is transient and cannot be the 24/7 host.

## What runs where

| Location | Responsibility |
| --- | --- |
| Ubuntu host | Pinned checkout, frozen environment, single bounded worker, durable journal and evidence. |
| GitHub | Source history, approved request, second evidence copy, independent review and archival PR. |
| ChatGPT Work | Design and coding sessions; separate hourly supervision session after activation. |
| Claude / Codex | Assigned implementation or independent review using AGENTS.md and the same evidence. |
| Dashboard | Sanitized snapshot of the cycle; publication follows the existing operator runbook. |

The approved scope is CPU only, no new paid services, one active study, at most
two CPU workers, single-threaded BLAS, at most two attempts and six cumulative
active hours including replay. Requests may lower those limits. Generation
has no partial-fit checkpoint; a restart starts fresh unless a complete
authenticated archive is available for replay-only recovery.

## Prepare the durable host

Use the existing Ubuntu machine only if it can remain powered on. Before
launch, verify available RAM and disk, outbound GitHub access, power/suspend
settings, and persistence of the state directory across service restart and
reboot. Run a bounded resource probe to establish whether the full grid and
replay fit the approved budget. A short probe is exploration, not study evidence.
Do not shrink the grid or silently extend the budget if it does not fit.

Use a clean checkout of the reviewed implementation commit. The exact source
SHA is fixed at request preparation; pulling or editing that checkout afterward
invalidates it. Check GitHub for matching evidence or an active Claude run first.
The examples assume `~/Thermo` and a durable home directory; change the paths if
the host uses another persistent volume.

```bash
cd "$HOME/Thermo"
git status --porcelain
git rev-parse HEAD
uv sync --frozen
uv lock --check --offline
uv run pytest tests/unit/test_meta_ebm_cap_baseline.py tests/unit/test_research_cycle.py -q
mkdir -p "$HOME/thermo-research-state" "$HOME/thermo-research-requests"
```

The checkout must have no tracked or untracked changes other than ignored
runtime files. Keep requests, journals and output outside it. Prepare the
request from this checkout, inspect its fixed identity, and retain it as the
approved job specification:

```bash
uv run python -m thermo_lab.research_cycle.cli prepare \
  --repo "$HOME/Thermo" --job-id m5a-first-cycle \
  --question 'How does the frozen coupling cap constrain M5a conditional fidelity?' \
  --workers 2 --budget-seconds 21600 --max-attempts 2 \
  --output "$HOME/thermo-research-requests/m5a-first-cycle.json"
uv run python -m thermo_lab.research_cycle.cli validate \
  --repo "$HOME/Thermo" \
  --request "$HOME/thermo-research-requests/m5a-first-cycle.json"
```

The request binds the full source SHA, protocol and lockfile hashes, canonical
scientific request digest, job ID, question and resource limits. It accepts no
shell commands or substitute evaluator. Changing the approved science requires
the protocol's amendment process; changing a job request requires a new ID.

## Execute and recover

For unattended operation, copy the example
[`thermo-research-cycle.service`](../tools/research-cycle/thermo-research-cycle.service)
to `~/.config/systemd/user/`, edit its paths, then inspect it with
`systemd-analyze --user verify ~/.config/systemd/user/thermo-research-cycle.service`.
Check process-death and journal recovery with the cheap test fixture, then
check the host's service manager without starting a study:

```bash
uv run pytest tests/unit/test_research_cycle.py -k 'duplicate_worker or child_retains or recovery or deadline' -q
systemd-run --user --unit thermo-cycle-smoke --collect \
  --property=Type=exec --property=KillMode=control-group /usr/bin/sleep 30
systemctl --user status thermo-cycle-smoke.service
systemctl --user stop thermo-cycle-smoke.service
```

These checks exercise fixture recovery and service control separately; they do
not claim a full production cycle or production reboot test. User services need
lingering enabled if they must survive logout; verify that with the host owner.
Verify durable state survives a host reboot before launching the study.
Enable the service only after the host checks pass. **This starts the full
scientific study** and consumes the approved job budget:

```bash
systemctl --user daemon-reload
systemctl --user enable --now thermo-research-cycle.service
systemctl --user status thermo-research-cycle.service
journalctl --user -u thermo-research-cycle.service --no-pager -n 40
```

For an attended full-study launch instead of the service, use:

```bash
uv run python -m thermo_lab.research_cycle.cli run \
  --repo "$HOME/Thermo" \
  --request "$HOME/thermo-research-requests/m5a-first-cycle.json" \
  --state-root "$HOME/thermo-research-state"
```

Stop with `systemctl --user stop thermo-research-cycle.service`. The service
terminates its whole process group. It does not repeatedly restart exhausted
jobs. Its boot activation and a deliberate `systemctl --user start` invoke the
same durable recovery logic. Never delete a journal or unlock a job to obtain
more runtime. Never start another worker from a stale heartbeat alone.
On the same boot, recovery charges both persisted wall-clock and monotonic
elapsed time conservatively. If the host rebooted while work was active, the
worker cannot prove its consumed time and exhausts the remaining budget for
operator review. It does not grant a fresh six hours after power loss.

```bash
uv run python -m thermo_lab.research_cycle.cli inspect \
  --state-root "$HOME/thermo-research-state" --job-id m5a-first-cycle
uv run python -m thermo_lab.research_cycle.cli verify \
  --state-root "$HOME/thermo-research-state" --job-id m5a-first-cycle
```

Successful execution remains `verifying` until the verified second copy is
accessible to the reviewer. A completion marker alone is insufficient. Full
evidence must retain ten targets, all 180 chains, zero samples, mandatory replay,
the source/request identities, generation and replay provenance, command,
bounded diagnostics and manifest hashes. Infrastructure or integrity failures
do not become negative scientific findings.

## Transfer evidence and request review

Keep the state root on durable storage, preserve earlier attempts, and retain
failed-attempt diagnostics for at least 30 days. Accepted evidence has no
automatic expiry. Copy the completed job into a separate durable evidence
checkout; inspect its size and contents before publishing. Exclude credentials,
unrelated host files and raw unbounded logs. Follow the existing compressed
archive convention; an oversized archive requires a separately reviewed
destination before launch.

The optional publisher copies bounded status observations and completed
manifest-bound evidence to a dedicated Git checkout. It accepts only the
Thermo GitHub origin and the exact branch `evidence/<job_id>`. Configure that
checkout after the evidence-branch write scope is authorized at activation:

```bash
git clone https://github.com/otto-agent007/Thermo.git "$HOME/thermo-evidence"
git -C "$HOME/thermo-evidence" fetch origin FULL_REVIEWED_SOURCE_SHA
git -C "$HOME/thermo-evidence" switch -c evidence/m5a-first-cycle FULL_REVIEWED_SOURCE_SHA
```

Replace `FULL_REVIEWED_SOURCE_SHA` with the exact value in the approved request.
The source commit must be an ancestor of this evidence branch. After its known
remote tip (or the source commit on the first push), local history may touch
only `cycles/m5a-first-cycle` and contain no merge commits. Git identity and
credentials must already work on the host; the publisher never sets them up.
Once the worker has created a job status, this command **commits and pushes**:

```bash
uv run python -m thermo_lab.research_cycle.publisher \
  --state-root "$HOME/thermo-research-state" --job-id m5a-first-cycle \
  --evidence-repo "$HOME/thermo-evidence" --branch evidence/m5a-first-cycle
```

It rejects unsafe paths, dirty checkouts, changed job identity, invalid evidence
and snapshots larger than 64 MiB. It checks the committed Git bytes before
pushing, preserves failures for explicit recovery, and never force-pushes.
Active observations contain only the immutable request and atomic status;
completed snapshots include the manifest, bounded journal and manifest files.
Unchanged snapshots do not create commits. The command prints the immutable
commit URI; a successful push does not claim that Work can retrieve it.

After a successful host/connector dry run, copy and inspect the optional
[`thermo-research-publish.service`](../tools/research-cycle/thermo-research-publish.service)
and [`thermo-research-publish.timer`](../tools/research-cycle/thermo-research-publish.timer)
examples. Enabling the timer starts publication every ten minutes. Keep it
disabled until the designated branch authority and host setup are verified.
Stop it with `systemctl --user disable --now thermo-research-publish.timer`.

`mirror` checks the supplied second-copy directory against the manifest. It
does not upload files or prove that a remote URL contains those bytes:

```bash
uv run python -m thermo_lab.research_cycle.cli mirror \
  --state-root "$HOME/thermo-research-state" --job-id m5a-first-cycle \
  --directory "$HOME/thermo-evidence/cycles/m5a-first-cycle"
```

In a separate Work session, fetch the actual committed files through GitHub,
verify their digests and request identity, and record that check. Only then run
`mirror` with `--connected-uri` set to the immutable
`https://github.com/otto-agent007/Thermo/tree/FULL_40_CHARACTER_SHA/PATH`
and `--verified-by` naming the actual verifier. This is an explicit
**connected-access attestation**, not automatic remote verification. A local
copy or invented URL is insufficient. Publish the resulting sanitized status
as a subsequent snapshot, retaining the evidence digest. Record the immutable
evidence commit and connected-access verification in the independent review
record and reviewer configuration; local `mirror.json` is not published.

The execution worker does not push GitHub updates. Until the optional host
publisher is configured and tested for the designated branch, Work can review uploaded
evidence but cannot monitor the running host. The hourly reviewer must label
GitHub status as the last published observation. Transport of active status
and completed artifacts is an activation requirement for a fully unattended
cycle; an operator upload demonstrates only the manual handoff. Test that
publisher with the cheap fixture and verify its remote bytes before scheduling.

Use [`thermo-cycle-supervision.md`](automation-prompts/thermo-cycle-supervision.md)
for the separate reviewer session. First test read-only inspection of the real
request, status, manifest and archive; also demonstrate refusal of an altered
copy with a cheap fixture. Record the exact SHA/digests reviewed and actual
reviewer identity. Do not claim a Claude review until Claude has performed it.

Hourly supervision is enabled only after the snapshot publisher, these reads
and review checks succeed and the designated branch write scope is authorized.
Each scheduled session
reconstructs state from connected GitHub evidence. It does not require a chat
checkout to survive between runs. Access or model quota failures leave review
pending; they never trigger paid API provisioning. No automation has been
created by this implementation.

The supervisor writes `reviews/<evidence-digest-hex>.json` using the exact schema
in its prompt. Worker-owned status and evidence files stay read-only to the
reviewer. The publisher preserves remote review additions when it can safely
fast-forward; divergent local/remote changes stop publication for explicit
recovery. The worker never marks scientific evidence `recorded`. Owner acceptance
and an independent review belong in the archival PR and its durable review record.
No automatic merge, messages to other people, or live publication are enabled.
An empty queue waits for the next approved question.

## Dashboard and repository activation

Set `THERMO_CYCLE_ROOT` to the individual job directory containing `status.json`,
for example `~/thermo-research-state/m5a-first-cycle`, when starting or exporting
the dashboard. With no configured directory, the dashboard shows no configured
cycle. A configured but missing or malformed snapshot is unavailable. A snapshot
is not live worker telemetry, and an awaiting-review phase is not acceptance.
For the independent review to appear, point `THERMO_CYCLE_ROOT` to the published
evidence checkout's `cycles/<job_id>` directory after its remote review record
has been retrieved. The dashboard binds that separate record to the source,
request and evidence digests; it does not change the worker's execution phase.
An approved independent review still requires the owner's final acceptance.

Run dashboard tests, typecheck, export/build and browser checks before following
the live publication procedure in [ci-cd.md](ci-cd.md). Import and verify
`.github/main-ruleset.json` using the owner setup in that same runbook; the file
alone does not protect main.

The first cycle is demonstrated only after an unattended real study, complete
replay, independently scheduled review and honest staged dashboard evidence.
The full archive, replay regression coverage and scientific report are then
added to an archival PR. Software tests alone do not meet that acceptance gate.
