# Local improvement harness

`thermo-harness` evaluates bounded draft patches for owner review. It does not
adopt changes, commit candidates, push, merge, publish, or release research.
Start from a clean committed checkout with Python dependencies installed using
`uv sync --frozen`. Dashboard candidates also require Node, npm dependencies,
and the Chromium version expected by the pinned Playwright package; see the
[dashboard setup](../dashboard/README.md).

## Freeze a plan and evaluate a patch

Plans pin the full current Git commit before proposal generation. Store local
inputs under ignored `results/`, so preparing a plan does not dirty its source.

```bash
uv run thermo-harness init-plan \
  --track dashboard \
  --objective "Improve proposal recommendation readability" \
  --output results/harness-inputs/dashboard-plan.json
uv run thermo-harness run results/harness-inputs/dashboard-plan.json \
  --output-dir results/harness \
  --manual-patch results/harness-inputs/dashboard.diff \
  --recommendation-file results/harness-inputs/dashboard-recommendation.md
```

Prepare the patch in a separate checkout at the plan's baseline, using
`git diff --binary`; provide a nonempty recommendation describing the change
and remaining limits. Recommendations must be at most 100,000 UTF-8 bytes and
patches at most 1,000,000 bytes, matching the dashboard's read limits. Oversized
inputs produce a failed candidate record for inspection. Manual intake requires
both files. The source checkout must still be clean at the frozen commit when
`run` starts. The harness creates
separate baseline and candidate worktrees, checks the patch allowlist, retains
observations and prints the candidate ID. Neither passing checks nor a zero
process exit constitutes acceptance.
Plan objectives are limited to 8,000 UTF-8 bytes so the dashboard can read the
frozen request. Held-out role names must be at most 128 safe ASCII characters.

For a bounded research candidate, use `--track research` with `init-plan` and
supply its patch and recommendation to `run` in the same way. The presets are:

| Track | Editable scope | Comparison |
| --- | --- | --- |
| Dashboard | `dashboard/src/` | Dependency install, unit tests, typecheck, static build and browser checks; owner visual review remains required. |
| Research | `src/thermo_lab/research_candidates/three_site.py` | Nine finite parameters within the existing fixture's bounds, scored against the frozen three-site exact objective, plus dependency, fixture, focused-test and Ruff checks. |

Research code implements `propose_parameters(fixture) -> tuple[float, ...]`.
The initial hook returns unchanged baseline parameters: its observed delta is
zero and its outcome is inconclusive. A strictly negative delta passes the
preset improvement threshold; a positive delta is regressed. This is only a
three-site `exact_reference` trial, with no full-program, inference-budget,
convergence, simulation-speed, physical-device or scientific-release claim.
The completed M4G archives, validators and release policy remain protected.

Each generated preset starts with a limit of two candidates and 1,800 seconds
of cumulative active execution per plan. A retry consumes another slot; failed
runs remain inspectable. Before its first run, review the plan JSON and reduce its budget
if needed; there is no CLI budget-override flag. Once frozen, the plan cannot
be edited. An optional research held-out role is reserved once; exposing it prevents more
research candidates under that plan, including children. Changing a plan must
not be used to treat exposed held-out evidence as fresh evidence.

Omit both manual-input options to use the installed local Codex proposer. It
must support the adapter's strict configuration and restricted sandbox flags;
credential-free manual mode and tests do not require it. Missing tools,
unsupported invocation, timeout or absent recommendation can leave an
incomplete or failed record. The adapter does not report model token usage or
financial cost. `run --parent CANDIDATE_ID` creates a fresh child with the
validated cumulative parent patch and development-check feedback; it never
modifies the parent record.

## Inspect evidence and record owner decisions

```bash
uv run thermo-harness show CANDIDATE_ID --output-dir results/harness
```

Each candidate directory retains its frozen plan, request, recommendation,
`patch.diff`, result and readable `report.md` when those stages are reached.
The result binds plan, patch and evidence hashes, command arguments, bounded
logs, durations and the baseline observation. Failed early attempts may lack
a patch or recommendation. Local records can contain diagnostics and absolute
worktree paths; keep them out of tracked files and published snapshots. A
candidate's Git HEAD stays at the baseline; its patch digest identifies the
uncommitted proposed change.

| Status dimension | Meaning |
| --- | --- |
| Execution | Whether execution completed, failed, timed out or was unavailable. |
| Verification | Whether the fixed checks and required evidence passed, failed or remained inconclusive relative to the observed baseline. A failing baseline cannot establish improvement. |
| Research outcome | Improved, regressed or inconclusive for the bounded exact objective; not applicable for dashboard candidates. This can differ from verification. |
| Owner review | Proposed until the owner explicitly accepts or rejects it. A review appends a decision; it does not apply the patch. |

Only after the owner has chosen a decision, record it with:

```bash
uv run thermo-harness review CANDIDATE_ID --output-dir results/harness \
  --decision accepted --note "Owner reviewed the evidence and patch"
```

Use `rejected` for a rejection. Review never substitutes for separate research
release requirements or authorizes automatic publication. Review notes are
limited to 100,000 UTF-8 bytes.

## Read-only dashboard

Start the [local dashboard](../dashboard/README.md) and open **Proposals**. It
reads validated records under `results/harness/`, shows all four status
dimensions, and links to the draft patch, a derived evidence report and any
validated dashboard screenshots. The report includes baseline source and
plan/patch/observation digests; raw logs and local worktree paths are not served.
Screenshots are retained as visual evidence only when each is a PNG of at most
8,000,000 bytes; missing, invalid or larger files leave that observation
unavailable for visual verification.
A GitHub baseline link can resolve only after that commit is available remotely;
inspect an unpushed baseline locally with `git show BASELINE_SHA`.
Invalid or incomplete evidence is surfaced explicitly. No browser endpoint can
start a job, accept a candidate or write a review.

The browser suite uses an isolated fixture server on strict port 5174. The
normal preview on 5173 can stay open. A static build exports an empty proposal
list and excludes local drafts and artifacts. Keep demonstrations local until
the owner curates evidence for a dated report; a demonstration is not a
scientific release.

Repository-wide gates and scientific evidence requirements remain in
[AGENTS.md](../AGENTS.md). Harness checks cover their declared local scope and
do not replace those gates.
