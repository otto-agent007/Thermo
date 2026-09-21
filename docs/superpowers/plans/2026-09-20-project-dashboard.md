# Thermo Project Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the project owner an interactive, evidence-backed view of progress, scientific findings, experiments and proposed next work.

**Architecture:** A React application in `dashboard/` consumes compact view models from a read-only Node adapter. Vite middleware serves local refreshes; the same adapter exports dated static snapshots with separate cell-detail files. Existing research records remain authoritative and unchanged.

**Tech Stack:** React, TypeScript, Vite, Node's test runner, Playwright browser tests, lossless JSON parsing for scientific identifiers; npm lockfile. Use the installed Node 24 runtime, check official package compatibility when installing, and lock the actual resolved dependencies.

**Spec:** `docs/superpowers/specs/2026-09-20-project-dashboard-design.md` (approved by user, 2026-09-20).

## Global Constraints

- Place the application under dashboard/ within the existing Git repository.
- Keep this UI code and its tests outside src/thermo_lab and the scientific tests folder.
- Negative scientific findings must remain distinct from failed execution or missing verification.
- Do not rerun experiments or invoke expensive numerical replay merely to render a page.
- A missing, stale or unsupported record yields a visible unavailable/stale state.
- Malformed data must not silently become a zero metric or a passing result.
- Oracle cells are separate from finite K axes.
- Seed series are not pooled into a new estimate.
- Color is never the only status indicator.
- There is no sign-in or mutating experiment control in this version.
- Preserve archived bytes, scientific Python files, experiment configs, pyproject.toml and uv.lock.
- No invented heartbeat, agent state, ETA, CI outcome, hardware measurements or cost advantage.
- Local activity is an observation of persisted files; static deployments show snapshot time and source commit.

## Review Focus

1. Historical execution says pending but a bound final completion exists: show completed execution and separately show scientific failure (Task 1).
2. Missing, truncated, unsupported or mismatched evidence: visible unavailable/stale state, never default success or zero (Tasks 1–2).
3. Scientific IDs exceed JavaScript's safe integer range: preserve exact decimal strings through parsing and export (Task 1).
4. Filters mix finite/oracle cells, empty matches, and rapid selection changes: consistent table/chart/export and no stale detail response (Task 4).
5. Snapshot hosting and narrow screens: relative data URLs, honest refresh labels, keyboard access and no page overflow (Tasks 2–5).

## File map and contracts

All paths below are relative to the repository root. Do not add scientific import-package code.

| Files | Responsibility |
| --- | --- |
| `dashboard/package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `.gitignore` | Isolated frontend build, dependencies and scripts |
| `dashboard/shared/model.ts` | Shared serializable contracts |
| `dashboard/server/catalog.ts` | Allowlisted milestones, evidence files, research narrative and sources |
| `dashboard/server/evidence.ts` | Lossless parsing, supported-shape/binding checks, compact M4G summaries |
| `dashboard/server/activity.ts` | Saved checkpoint observations, never process inference |
| `dashboard/server/service.ts`, `plugin.ts`, `export.ts` | Read-only routes, local Vite integration, static export |
| `dashboard/src/main.tsx`, `App.tsx`, `styles.css` | Shell, navigation and responsive visual system |
| `dashboard/src/data.ts`, `useProject.ts` | Local/snapshot loading and refresh lifecycle |
| `dashboard/src/components/Status.tsx`, `SourceLink.tsx` | Accessible state labels and evidence links |
| `dashboard/src/features/Overview.tsx`, `Roadmap.tsx`, `Research.tsx` | Project state, milestone/activity view, sourced findings/proposals |
| `dashboard/src/features/Experiments.tsx`, `MetricChart.tsx`, `CellTable.tsx`, `CellDetail.tsx`, `CostView.tsx`, `filters.ts` | Evidence exploration, exact values and CSV |
| `dashboard/tests/evidence.test.ts`, `service.test.ts`, `filters.test.ts`, `browser.spec.ts`, `playwright.config.ts` | Adapter, route, filter and browser checks |
| `dashboard/README.md`, `README.md`, `.github/workflows/dashboard.yml` | Operation instructions, discoverability and frontend CI |

Shared contract (declare in `dashboard/shared/model.ts`):

```ts
export type Availability = 'available' | 'unavailable' | 'stale';
export type Member = 'finite' | 'equilibrium' | 'frozen';
export type Horizon = 1 | 2 | 4 | 8 | 16 | 30 | 'equilibrium';
export type Source = { label: string; url: string; recordedAt: string | null };
export type Status = { state: string; label: string; source: Source | null };
export type CellSummary = {
  id: string; seed: number; evaluationSeed: string; member: Member;
  horizon: Horizon; decision: string; sampleCount: number;
  loss: number; lossBounds: [number, number];
  leakage: number; leakageInterval: [number, number];
  survival: number; hopMae: number; asymmetryMae: number;
  sampledEvidence: string; exactEvidence: string;
};
export type CellDetail = CellSummary & {
  histogram: number[]; survivalTrace: { operation: number; probability: number }[];
  identities: Record<string, string>; sources: Source[];
};
export type Activity = {
  availability: Availability; observedAt: string; latestFileAt: string | null;
  checkpointCount: number | null; expectedCheckpoints: number;
  label: string; issues: string[];
};
export type Milestone = {
  id: string; question: string; conclusion: string; state: string;
  evidenceClasses: string[]; sources: Source[];
};
export type ResearchItem = {
  id: string; kind: 'finding' | 'deduction' | 'proposal';
  title: string; text: string; scope: string; sources: Source[];
};
export type StudySummary = {
  counts: { fits: number; updates: number; cells: number; pairs: number };
  cells: CellSummary[];
  thresholds: { loss: number; leakage: number; survival: number; hopMae: number; asymmetryMae: number };
  costs: { label: string; value: number | null; unit: string; scope: string; evidence: string; source: Source }[];
  pairedDiagnostics: { label: string; estimate: number; interval: [number, number]; conclusion: string }[];
  sources: Source[];
};
export type ProjectSnapshot = {
  schemaVersion: 1; mode: 'local' | 'snapshot'; generatedAt: string;
  sourceCommit: string | null; availability: Availability; issues: string[];
  execution: Status; verification: Status; science: Status;
  study: StudySummary | null; activity: Activity;
  milestones: Milestone[]; research: ResearchItem[];
};
export type Filters = { seeds: number[]; members: Member[]; horizons: Horizon[]; query: string };
```

## Task 1: A trustworthy evidence adapter

**Files:** Create shared/model.ts, server/catalog.ts, server/evidence.ts, tests/evidence.test.ts and package/build configuration from the map.

**Interfaces:** `loadEvidence(repoRoot: string): Promise<{ snapshot: ProjectSnapshot; details: Map<string, CellDetail> }>`; `parseEvidence(text: string): unknown`. Consume the committed M4G archive and curated source paths, not arbitrary request paths.

- [ ] Read the approved spec and frontend-app-builder skill. Record a baseline SHA-256 manifest of scientific source/config/test/archive files in scratch for the final unchanged check. Check official React/Vite and lossless parser documentation, install compatible pinned dependencies, and commit the npm lockfile. Add scripts `test` (tsx with node --test), `typecheck` (tsc --noEmit), `dev`, `build` (typecheck, export, Vite build), `export`, `preview`, and `test:browser`.
- [ ] Write archive-backed adapter tests, plus temporary-directory fixtures copied from the archive and mutated only in the temporary directory:

```ts
const { snapshot } = await loadEvidence(repoRoot);
assert.deepEqual(snapshot.study?.counts, { fits: 21, updates: 105, cells: 60, pairs: 18 });
assert.equal(snapshot.study?.cells.filter(c => c.decision === 'fail').length, 60);
assert.equal(snapshot.execution.state, 'complete');
assert.equal(snapshot.science.state, 'quality_failure');
assert.equal(snapshot.study?.cells.find(c => c.id === 'seed-0/finite/1')?.evaluationSeed, '6717865023900054950');
assert.equal(snapshot.study?.cells.find(c => c.id === 'seed-0/finite/1')?.leakage, 27280 / 32768);
```

Add cases: absent study; truncated JSON; unsupported schema; completion result digest mismatching study; missing review; final completion supersedes historical execution pending state. In failure cases assert unavailable/stale plus an issue string, and no release-complete claim. Compare survival/hop/asymmetry directly with parsed archive values. Require all numeric UI metrics finite; identifiers stay decimal strings.
- [ ] Run `cd dashboard && npm test -- tests/evidence.test.ts`; expect missing adapter failure before implementation.
- [ ] Implement lossless decoding: integer tokens outside safe range become decimal strings, ordinary counts and metrics become numbers. Validate supported field types and required identity links before mapping. Never call Python replay. Extract the 60 cells and 18 paired diagnostics; select final stored survival entry, preserve interval semantics, and keep threshold values bound to the frozen protocol. Mapping core:

```ts
const cells = study.evaluations.flatMap(e => e.cells);
const summary = cells.map(c => ({
  id: c.cell_id, seed: c.seed, evaluationSeed: String(c.evaluation_seed),
  member: c.member, horizon: c.horizon, decision: c.decision.status,
  sampleCount: c.sample_count, loss: c.population_loss_estimate,
  lossBounds: c.decision.loss_bounds, leakage: c.leakage_count / c.sample_count,
  leakageInterval: c.decision.leakage_interval,
  survival: c.exact_metrics.survival.at(-1)!.survival_probability,
  hopMae: c.exact_metrics.hop_mae, asymmetryMae: c.exact_metrics.asymmetry_mae,
  sampledEvidence: c.evidence_class, exactEvidence: c.exact_metrics.evidence_class,
}));
```

Use validated internal types before this mapping. Validate required completion/review/gate/provenance bindings without claiming fresh numerical verification. If the binding scheme cannot be faithfully implemented in Node, report the unverified binding explicitly and withhold the verified badge. Source links point at a pinned Git commit when known. Populate older milestones from their curated reports without pretending their schemas match M4G. Read proposals from the dated conservation review; label M5 according to the roadmap.
- [ ] Run the adapter tests and typecheck; expect pass. Commit `feat: adapt project evidence for dashboard`.

## Task 2: Local observations and portable snapshots

**Files:** Create server/activity.ts, service.ts, plugin.ts, export.ts, tests/service.test.ts; connect vite.config.ts.

**Interfaces:** `observeActivity(repoRoot: string): Promise<Activity>`; `getPayload(repoRoot: string, route: string): Promise<{status: number; body: unknown}>`; `exportSnapshot(repoRoot: string, destination: string): Promise<void>`. Routes are `/data/project.json` and `/data/cells/<encoded-cell-id>.json`; only exact known cell IDs are resolved.

- [ ] Write tests for missing results (unavailable/count null), 3 valid checkpoints (count 3, no running claim), malformed checkpoint (issue shown), 21 checkpoints (still no heartbeat), and stable local/archive precedence. Exercise GET/HEAD, rejected POST, unknown cell IDs, encoded traversal, and symlink escape from allowlisted directories.

```ts
assert.equal((await getPayload(repoRoot, '/data/cells/..%2F..%2Fsecrets.json')).status, 404);
await exportSnapshot(repoRoot, tempDir);
const exported = JSON.parse(await readFile(join(tempDir, 'data/project.json'), 'utf8'));
assert.equal(exported.mode, 'snapshot');
assert.ok(exported.generatedAt);
assert.ok(!JSON.stringify(exported).includes(repoRoot));
```

- [ ] Run `cd dashboard && npm test -- tests/service.test.ts`; expect missing service failure.
- [ ] Implement allowlisted paths with realpath containment, bounded file sizes, before/after stat checks and one retry for a concurrent write. Stable malformed records return issues; they do not replace last valid records with fabricated progress. Resolve all cell IDs via the adapter Map. Cache parsed archive data by file metadata, invalidating on change; coalesce concurrent reads. Export compact project JSON and individual details under `dashboard/public/data/`, excluding the original 15MB study. Export all data into a staging directory and replace only after success. The Vite plugin handles read-only local requests; static builds use identical payload contracts.

```ts
const payload = await loadEvidence(repoRoot);
const snapshot = { ...payload.snapshot, mode: 'snapshot' as const };
await writeFile(join(staging, 'project.json'), JSON.stringify(snapshot));
for (const [id, detail] of payload.details) {
  await writeFile(join(staging, 'cells', `${encodeURIComponent(id)}.json`), JSON.stringify(detail));
}
```

- [ ] Run service and adapter tests, then `npm run export`; expect 1 compact project payload and 60 detail payloads, with no credentials or machine paths. Commit `feat: serve and export dashboard evidence`.

## Task 3: Project overview, roadmap and research

**Files:** Create src/main.tsx, App.tsx, styles.css, data.ts, useProject.ts, shared components and Overview/Roadmap/Research features; initialize browser.spec.ts and playwright.config.ts.

**Interfaces:** `loadProject(signal?: AbortSignal): Promise<ProjectSnapshot>`; `loadCell(id: string, signal?: AbortSignal): Promise<CellDetail>`; `useProject(): {data: ProjectSnapshot | null; error: string | null; refreshing: boolean; refresh: () => void}`. View components consume snapshot props; App owns navigation.

- [ ] Follow frontend-app-builder's visual concept workflow, using the approved restrained light/teal research-workbench direction. Keep design artifacts separate from data and record the comparison checklist. Implement actual HTML/SVG controls, not a screenshot UI.
- [ ] Write browser assertions for separate completed/recorded-verification/quality-unmet labels; all four navigation destinations; research proposal labels and primary sources; unknown runtime state; visible snapshot date and commit. Test a failed refresh retains the last view with an explicit stale notice.

```ts
await page.goto('/');
await expect(page.getByRole('navigation')).toBeVisible();
await expect(page.getByText('Joint quality unmet', { exact: true })).toBeVisible();
await page.getByRole('link', { name: 'Research', exact: true }).click();
await expect(page.getByText('Proposed', { exact: true }).first()).toBeVisible();
```

- [ ] Run `cd dashboard && npm run test:browser`; expect missing shell failure.
- [ ] Implement semantic header/nav/main, responsive context column and readable source labels. Use relative `import.meta.env.BASE_URL` data URLs so snapshots work under a path prefix. Poll local mode every 15 seconds only while visible; abort on unmount and reject out-of-order refresh results. Snapshot mode has no polling or implied remote updates. Manual refresh re-reads the current mode's source and displays its actual timestamp. Structure shell composition:

```tsx
<main id="main-content">
  {view === 'overview' && <Overview snapshot={data} />}
  {view === 'experiments' && <Experiments snapshot={data} />}
  {view === 'roadmap' && <Roadmap snapshot={data} />}
  {view === 'research' && <Research snapshot={data} />}
</main>
```

Until Task 4 lands, omit the unfinished Experiments route from committed runnable shell and add it with that task; no dummy data or placeholder metrics. Overview derives ranges/counts from cells. Show leakage <=5%, survival >=95%, loss <=0.0625, hop/asymmetry <=0.01 and their distinct meanings. Source-link narrative claims; distinguish mathematical deductions from experiment findings. GitHub status remains a link.
- [ ] Run browser checks for implemented destinations and `npm run build`. Commit `feat: show project findings and roadmap`.

## Task 4: Interactive scientific evidence exploration

**Files:** Create features/Experiments.tsx, MetricChart.tsx, CellTable.tsx, CellDetail.tsx, CostView.tsx, filters.ts; add tests/filters.test.ts; extend browser.spec.ts and App route.

**Interfaces:** `filterCells(cells: CellSummary[], filters: Filters): CellSummary[]`; `toCsv(cells: CellSummary[]): string`; `MetricChart({cells, metric})` with metric union `'loss' | 'leakage' | 'survival' | 'hopMae' | 'asymmetryMae'`; `CellDetail({id, onClose})` loads via loadCell.

- [ ] Write filter/export tests with actual adapter rows, empty queries, no matches, all supported horizons and members. Use explicit oracle panels and no seed aggregation. Pin CSV identity preservation and escaping.

```ts
const selected = filterCells(cells, { seeds: [0], members: ['finite'], horizons: [30], query: '' });
assert.equal(selected.length, 1);
assert.equal(selected[0].id, 'seed-0/finite/30');
assert.match(toCsv(selected), /seed,member,horizon/);
assert.equal(filterCells(cells, { seeds: [], members: [], horizons: [], query: 'no-such-cell' }).length, 0);
```

Define empty selection arrays to mean all values; Reset restores these arrays and clears query. Escape quotes/newlines in CSV fields and sanitize formula-leading text; numeric scientific fields remain numeric.
- [ ] Run `npm test -- tests/filters.test.ts`; expect missing module failure.
- [ ] Implement one filtered array shared by plots, table, counts and downloads. Use native SVG plots with labeled axes, threshold lines, stored intervals and seed-specific series. Plot oracle cells in a separate categorical region. Accessible table values provide exact numbers and keyboard detail buttons. Detail drawer returns focus to its opener, closes with Escape, and aborts stale requests. Display particle histogram, 500-operation survival trace and exact identity strings. Empty matches show an explicit empty state.

```tsx
const rows = filterCells(snapshot.study.cells, filters);
<MetricChart cells={rows} metric={metric} />
<CellTable cells={rows} onSelect={setSelectedId} />
<button onClick={() => downloadCsv(toCsv(rows))}>Download selected CSV</button>
```

Define `downloadCsv(text: string): void` in filters.ts using Blob/object URL and revoke the URL after download. Present the full recorded CSV as a separate source link. CostView uses the entire six-fit finite grid, preserves null oracle finite-sweep cost, shows operation-count units/scope, and separates sourced CPU elapsed time from hardware costs. Pair diagnostics retain descriptive jackknife wording; gate badges use stored acceptance status.
- [ ] Extend browser tests: select seed0/finite/K30 and assert one row, switch five metric tabs, open/close details by keyboard, download one-row CSV, reset to60, produce empty search, and rapidly switch cell IDs while intercepting delayed responses. Assert the late response cannot overwrite the latest selection. Run unit/browser tests and build. Commit `feat: explore study cells and quality thresholds`.

## Task 5: Preview, verification and delivery

**Files:** Create dashboard/README.md, .github/workflows/dashboard.yml; modify root README.md with dashboard link; extend browser tests. Record verification results in dashboard/README.md only if they are stable operating instructions; put run-specific evidence in the PR description.

**Interfaces:** documented commands `cd dashboard && npm ci && npm run dev -- --host 0.0.0.0`; `npm run build`; `npm run preview -- --host 0.0.0.0`. `dashboard/dist/` is a self-contained dated snapshot.

- [ ] Add Playwright desktop (1440x1000), mobile (390x844), keyboard and reduced-motion checks. Verify root and non-root hosting paths, export-mode no polling, read errors, chart/table consistency and source URLs.

```ts
await page.setViewportSize({ width: 390, height: 844 });
await page.goto('/');
expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
await page.keyboard.press('Tab');
await expect(page.locator(':focus')).toBeVisible();
```

- [ ] Run `npm ci`, `npm test`, `npm run typecheck`, `npm run build`, `npm run test:browser` from dashboard. Inspect screenshots using view_image and compare to the concept; record and fix material fidelity/accessibility gaps. Check browser console and failed requests. Add a separate Node dashboard CI workflow executing install, unit tests, typecheck and build without modifying scientific CI gates.
- [ ] Compare baseline scientific hashes byte-for-byte, including archive records, configs, Python source/tests and lockfile. Follow the repository's required local gates and preserve their exact arguments; consult AGENTS.md for the full current list. Do not confuse previously recorded scientific validation with newly run UI checks. If environmental limitations block a gate, name the command and limitation in delivery rather than claiming a pass. Do not run numerical replay merely for preview or polling.
- [ ] Write documented local commands, the source catalog policy, mode/refresh limitations, and how to export new data. Read Sites building/hosting skills when preparing the preview; use an available supported preview path for the actual static build. Do not claim a live hosted data service. Verify the provided preview URL in a browser before delivery. If hosting is unavailable, provide the runnable local dashboard and built snapshot with explicit access limitations.
- [ ] Obtain the final independent review required by the chosen execution skill, address findings and rerun only affected checks. Commit `docs: document and verify project dashboard`, publish the branch and create a draft PR against the appropriate base. Preserve the M4G dependency: if PR35 remains open, use its branch as the dashboard PR base and explain the stacking; if merged, use updated main. Do not merge. Return the usable preview, PR, checks and any material limitations.

## Self-review and handoff

- Spec coverage: Task1 owns evidence/provenance; Task2 local/static activity; Task3 overview/roadmap/research; Task4 all exploration and costs; Task5 accessibility, preview and delivery.
- Five review-focus failures have explicit owning tests. The unsafe-integer issue is grounded in the actual archived evaluation seed 6717865023900054950.
- Interfaces consistently share model.ts and the same local/static payloads. Raw traces are split from initial payloads. The scientific engine is untouched.
- Recommend Native execution: these five tasks share one adapter/view-model contract and the UI is read-only; implementing together reduces interface handoff overhead. One independent whole-branch review follows implementation.
- This plan awaits the user's review and choice of Native or Subagent-driven execution before product code is written.

## Native execution outcome — 2026-09-20

Tasks 1–4 are implemented; Task 5 produced the private dated snapshot and stacked draft PR36. See `dashboard/VERIFICATION.md` for actual checks and limitations rather than treating the original proposed commands as execution evidence.

Implementation rulings: authenticate the six approved files with reviewed byte hashes; retain exact large identities as strings; use portable `~` detail filenames; accept local checkpoints only when they match supported archived fit slots. The authored browser suite runs successfully in CI because the supported cloud preview was blocked. It covers desktop/mobile navigation and the principal experiment interaction; non-root hosting, exhaustive motion/accessibility and delayed-response browser scenarios are not claimed as completed. Code cancellation and static-mode behavior are implemented. Generated concept dates and activity counts were replaced by evidence; counts remain an open band per the written spec.

Independent review corrections, actual screenshot comparison and scientific baseline preservation are recorded in the verification report. PR36 remains stacked on PR35 until that dependency lands. No merge was performed.
