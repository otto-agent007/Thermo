# Thermo project dashboard

Status: approved by the user on 2026-09-20. Implementation planning follows; no application code has been implemented.

## Purpose and success criteria

The user wants a UI to see what is happening in the Thermo project. The dashboard
should answer, within one screen: where the project stands, what the latest
experiment taught us, what is currently observable, and which work is next.
It should make the evidence easy to explore without requiring the user to read
large JSON files, terminal logs or every pull request.

Assumptions for this first version: one project owner; read-only inspection;
existing repository files are authoritative; local use can refresh from the
workspace; a published view can show a clearly dated snapshot. Execution controls,
accounts and remote live-process infrastructure are outside this first version.
These assumptions are reviewable choices, not claims of existing capabilities.

Success means the user can open the dashboard, understand the M4G conservation
finding, compare budgets/seeds, follow progress through milestones, and open the
underlying evidence. Negative scientific findings must remain distinct from
failed execution or missing verification.

## Approaches considered

| Approach | Benefit | Trade-off |
| --- | --- | --- |
| Evidence-backed dashboard with local refresh and exportable snapshots (recommended) | Interactive results and clear project status from existing records; no research engine changes required | A hosted snapshot updates when republished, not whenever a remote process changes |
| Static visual report only | Smallest maintenance burden and easy sharing | Does not provide workspace progress or rich experiment navigation |
| Remote execution/control console | Could eventually launch jobs and track them centrally | Adds authentication, job lifecycle, durable storage and cancellation semantics beyond the requested visibility |

## Product design

A restrained research-workbench layout: light neutral background, dark text,
teal for navigation and selected data, amber for an unmet scientific target,
and red reserved for actual integrity/execution errors. Use a compact left
navigation rail, generous readable plot areas and tables. Desktop uses a main
workspace plus a compact context column; mobile stacks content with reachable
filters. Charts, text and controls remain native interactive UI.

Navigation has four destinations:

### Overview

- A plain-language project state and the most recent completed study.
- Separate labels for execution, verification and scientific outcome.
- For M4G: 21 completed fits, 105 updates, 60 evaluated cells; joint quality unmet.
- A clear finding: at K30, occupancy loss and local fidelity pass while
  conservation fails. Show leakage, uninterrupted survival and their thresholds.
- A concise milestone sequence from M1 through M4G, with M5 shown as next on the
  existing roadmap. Research follow-ups are proposals, not started jobs.
- An activity panel with source timestamps and observed checkpoint counts when
  local result files are present. Unavailable activity is explicit.
- Links to the report, protocol, review evidence, repository and pull request.

### Experiments

- An experiment list with milestone, question, evidence class, execution state,
  scientific conclusion and source link.
- A detailed M4G view with horizon, member and seed filters.
- Five metric views: population loss, terminal leakage, uninterrupted survival,
  hop MAE and asymmetry MAE. Show literal acceptance thresholds and the recorded
  simultaneous bounds where applicable.
- Loss estimates and paired jackknife diagnostics are labeled descriptive;
  acceptance uses the stored decision and its actual gate quantities.
- A 60-row cell table, searchable/filterable, with a detail drawer containing
  particle histogram, per-operation survival and source/result identity.
- An operation-cost view that counts the entire finite training grid; oracle
  finite-sweep cost stays unassigned. CPU wall time is separately labeled.
- CSV download exports the currently selected rows with member, seed and horizon.
  A link also exposes the full recorded CSV.

### Roadmap and activity

- Milestone rows summarize question, result and linked evidence.
- Proposed, recorded execution, completed and unknown states are distinct.
- Local progress derives from persisted checkpoints, execution and completion
  records. A complete set of checkpoint files does not establish a live process.
- Only a final completion record establishes release completion; execution.json
  can remain a historical pending-release record after final completion exists.
- A local refresh control and automatic polling show when data was last read.
  The hosted snapshot labels its export time and source commit.
- No invented ETA, heartbeat, live agent status, GitHub CI result or progress
  animation. GitHub status is a link unless a dated authenticated snapshot has
  actually been collected. Active ChatGPT agent messages are not a data source.

### Research

- Readable questions, findings and source links from the conservation review.
- Clearly separate reported experiment findings, mathematical deductions and
  unimplemented proposals.
- Show the recommended exact survival-gradient audit and prospective joint
  sampler pilot, including what each would test and what it would not establish.
- State software capability and hardware availability according to the cited
  evidence; no hardware-performance claims from software measurements.

## Architecture and data flow

Place the application under dashboard/ within the existing Git repository.
Use React and Vite for the frontend, with a small Node-side adapter/exporter.
Keep this UI code and its tests outside src/thermo_lab and the scientific tests
folder. The UI must consume research evidence without changing the validated
numerical engine or the code-bound M4G study identity.

The adapter owns a curated project catalog linking milestone identifiers to
committed summaries and supported evidence schemas. It extracts compact view
models from actual records rather than duplicating numeric results by hand.
Narrative summaries have source links; source timestamps and commit identities
are included in the export. The 15 MB M4G study is parsed on the adapter side;
initial browser payloads contain only the summaries needed by the selected view.
Large details are lazy-loaded or split into per-study/per-cell payloads.

Local development serves the compact read-only API through the dev server and
polls it periodically. It may inspect explicitly configured result directories
under the repository, including saved fit checkpoints. Published builds use the
same adapter to export static data and expose a dated snapshot. No browser
credentials or unrestricted filesystem paths are accepted. Paths are defined by
the catalog/adapter, not by user-supplied traversal strings.

Do not rerun experiments or invoke expensive numerical replay merely to render
a page. Check supported record shapes and identity bindings needed for display.
Show recorded validation as recorded validation, not newly performed replay.
A missing, stale or unsupported record yields a visible unavailable/stale state.
Malformed data must not silently become a zero metric or a passing result.

Suggested component boundaries: application shell, project overview, milestone
list, experiment selector, metric chart, result table, cell detail, activity
panel, research reader, shared evidence labels, and adapter/view-model modules.
Keep the main application component as composition rather than one large file.

## Interaction and accessibility

Filters update charts and the table consistently; a reset restores all seeds
and supported members. Oracle cells are separate from finite K axes. Plot hover
and keyboard-accessible detail selection reveal exact stored values and bounds.
Threshold direction is explicit. Seed series are not pooled into a new estimate.

Use semantic landmarks and tables, visible focus, sufficient contrast, keyboard
navigation and reduced-motion support. Color is never the only status indicator.
Small screens must retain chart labels and access to full cell detail without
horizontal page overflow. There is no sign-in or mutating experiment control in
this version.

## Implementation and verification acceptance

1. Build/export the dashboard from the current repository evidence.
2. Check adapter counts and selected numeric values against the M4G records:
   21 fits, 105 updates, 60 cells, 18 pairs, all 60 failed quality classifications.
3. Check that release-complete and scientific-failure states display together;
   missing runtime files must not be shown as running or successful.
4. Check filters, threshold views, histogram/survival details, downloads and links.
5. Exercise missing/malformed evidence and an incomplete local checkpoint set.
6. Verify the rendered UI in a browser at desktop and mobile sizes, including
   keyboard interaction and visual review against the chosen concept.
7. Confirm the scientific implementation digest and archived study files are
   unchanged by UI work. Follow applicable repository verification gates.
8. Provide a usable preview plus documented local start/export commands, then
   publish the reviewed changes through the existing PR workflow.

No external site or hosting project is created by this design document. After
approval, implementation planning will make file-level tasks and select the
available preview/hosting path. A hosted snapshot must retain its refresh limits
in the UI rather than imply remote live monitoring.
