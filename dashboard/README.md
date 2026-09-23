# Thermo research dashboard

Read-only project visibility: overview, experiment filters and charts, source-linked roadmap, local checkpoint observations, and research proposals. The scientific engine and archived study are unchanged.

## Run locally

Use Node 22.12+ (Node 24 used for development).

```bash
cd dashboard
npm ci
npm run dev
```

Open the address printed by Vite. Local mode reads the allowlisted archive and observes `results/m4g-task-quality-study/checkpoints`. It refreshes every 15 seconds while the tab is visible. Persisted checkpoints are not proof of a running process. Final release completion comes from the committed archive.

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

The **Proposals** view reads local, digest-checked harness records from
`results/harness/`. Execution, verification, research outcome and owner review
are separate; a research result is a bounded three-site exact trial. Draft
patches and the three named screenshots are read-only links. Invalid or missing
evidence is reported as unavailable; logs and local source paths are omitted.
The adapter caps proposals at 100, JSON/patch files at 1 MB, PNGs at 8 MB, and
rejects symlinks and paths outside the repository. No browser mutation or job
execution endpoint exists. Static builds export an empty proposal list; local
drafts and their artifacts are never copied to the published snapshot.

Browser tests use temporary proposal records and an isolated server on strict
port 5174, preserving the development preview on 5173. Screenshots are written
to ignored `test-results/`, including `proposals-desktop.png` and
`proposals-mobile.png` for manual layout review.

A displayed verification pass requires the complete fixed check catalog,
coherent command outcomes, an authenticated matching baseline, and the track's
required evidence. Incomplete observations remain reviewable as inconclusive.
**Download evidence report** provides a derived Markdown report with the full
baseline source link, plan/patch/observation digests, validated check statuses
and bounded exact research values when available. This read-only report is
capped at 32 KiB and never serves the runner's raw report, logs or diagnostics.
