# Thermo research dashboard

Read-only project visibility: overview, experiment filters and charts, source-linked roadmap, local checkpoint observations, and research proposals. The scientific engine and archived study are unchanged.

Open the [private hosted dashboard](https://thermo-research-workspace.otto-agent007.chatgpt.site).
The current view includes the five September 21–23 follow-up studies, with
the complete-row fidelity pilot first. The study selector also retains the
original 60-cell M4G explorer. Each recent study links to its pinned report,
completion and independent review and retains its original gate.

## Run locally

Use Node 22.12+ (Node 24 used for development).

```bash
cd dashboard
npm ci
npm run dev
```

Open the address printed by Vite. Local mode reads the allowlisted archive and observes `results/m4g-task-quality-study/checkpoints`. It refreshes every 15 seconds while the tab is visible. Persisted checkpoints are not proof of a running process. Final release completion comes from the committed archive.

After installing dashboard dependencies, `npm run dev` also works from the
repository root. Use that root for a restricted preview so the dashboard can
read the sibling `docs/experiment-reports` directory; serving only the
dashboard subtree correctly withholds inaccessible evidence.

## Build a snapshot

```bash
npm run build
npm run preview
```

Deploy `dist/` as static files. Snapshots contain their export timestamp and the pinned evidence commit. Refresh re-reads the published snapshot; changes appear only after another export/deployment. Cell details are separate files; the initial payload does not include the 15MB raw study. Relative URLs support deployment under a path prefix.

The snapshot contains source links to GitHub; those links retain the repository's existing access requirements. No credentials, absolute workspace paths, or raw runtime provenance are included in browser payloads.

## Evidence policy

The supported M4G archive is bound to SHA-256 byte hashes in `server/archive-pins.json`, with source revision in `server/catalog.ts`. The adapter verifies all six pinned files before showing a completed release or its metrics. Missing, modified, unsupported or partial evidence is unavailable, never zero or passing. This is authentication of previously reviewed bytes, **not a new scientific replay**. The browser says verification is recorded.

To support a future archive, review its schema, decision semantics, final release, provenance and independent reviews before adding its adapter and pins. Do not automatically regenerate hashes to bless changed scientific results. Curated older milestone summaries and research items link to the dated roadmap/review rather than pretending every historical schema is supported.

Recent studies use a separate report adapter: `server/recent-pins.json`
authenticates the reviewed summary, completion and independent review bytes
at commit `918f67f62b9478276bc23d9e4a9a805937c58cf4`. Tables are extracted
from those reports, retaining every published arm or saved fit. A missing or
changed report withholds that study's findings and metrics independently.
This is report authentication, not a new numerical replay. The M4G pins and
scientific source files are unchanged.

Publishing a GitHub commit does not update the hosted static snapshot. After
adding a reviewed study, run the checks and snapshot build, then publish the
result to the existing Thermo Research Workspace Site. The browser's Refresh
button reloads the current publication; it cannot discover new repository
experiments by itself.

Unsafe integer seed identifiers are preserved as decimal strings. Downloaded CSV retains them, although spreadsheet applications may need that column imported as text. Loss U-statistics and paired jackknife intervals are descriptive; stored acceptance decisions and simultaneous gate bounds remain separate. Oracle cells have no finite sweep cost. CPU generation time is not energy or hardware latency.

## Verification

```bash
npm test
npm run typecheck
npm run build
npx playwright install chromium
npm run test:browser
```

Tests cover real React filter/keyboard interaction in a DOM harness, real archive extraction, exact seed identities, missing/changed evidence, partial checkpoints bound to expected archived fit slots, symlink escape, read-only routes, static export, filters and CSV. Browser tests cover desktop/mobile navigation, metric selection, details, downloads and failed refresh. They require an environment that can start the Vite server and run Chromium.

Scientific regression gates remain defined in the repository's `AGENTS.md`; this dashboard does not alter those gates or dependency pins.
