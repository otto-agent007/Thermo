# Dashboard verification record — 2026-09-20

## Scope

Read-only UI added outside scientific source, configs and tests. Recorded M4G evidence is unchanged. Numeric values come from the approved byte-pinned archive. Browser payloads are compact and use decimal strings for large seed identities. All 333 tracked files in the pre-implementation scientific baseline retain their SHA-256 hashes; source, tests, configs, archived reports, Python metadata and lockfile have no Git diff against the M4G base.

## Automated dashboard checks

- 15 unit, SSR and DOM tests pass: archive counts, all 60 unmet decisions, seed precision, changed/missing evidence, unsafe paths, activity observations, static export, filters and CSV, unavailable-state propagation and preserved keyboard navigation state.
- Clean-checkout `npm ci`, tests, TypeScript typecheck, snapshot export and Vite production build pass.
- Four Chromium browser tests pass: navigation, mobile overflow/focus, all five metrics with filters/detail/CSV/reset/empty state, and failed refresh retaining previous data.
- [Final rendered-code Dashboard CI](https://github.com/otto-agent007/Thermo/actions/runs/35494427278) verifies commit `46ee3b7cf2fea4101c7d02f32e7802f4ba5e0307`. Screenshots are in its `dashboard-browser-evidence` artifact (14-day retention).

## Visual fidelity ledger

The generated overview concept was compared with actual Chromium desktop (1506px wide), mobile (390px wide) and experiment screenshots. There are no raster assets in the application.

| Point | Comparison and decision |
| --- | --- |
| Copy | Main finding and destination labels preserved. Invented dates/counts replaced by actual records; M5 retains its actual roadmap title. Headline spacing found in screenshots was corrected and rechecked. |
| Layout | Sidebar, main findings and context column retained. Mobile stacks content; navigation now wraps with all destinations visible. Header spans the content area and the context column starts beside the status band. |
| Palette | Rendered cool gray background, white surfaces, dark text, teal data and amber unmet-quality state match the direction. |
| Typography | System sans, bold hierarchy and muted metadata render legibly. More compact sizing than the concept is intentional for a data workspace. |
| Containers | Open count band follows the written spec instead of the concept's four cards. Real source links replace the concept's invented dated evidence row. |
| Icons | Recorded verification uses a document, unknown state a question; neither implies a newly passed scientific audit. |
| Interactions | Browser run confirms filter/metric/detail/export flow, keyboard Escape, navigation and stale-data notice. |

CI's clean checkout correctly shows local activity unavailable. The published snapshot contains 21 validated persisted checkpoints observed at export time. Neither is a live process heartbeat.

## Independent review and corrections

The whole-branch independent review found no Critical issues. Both Important findings were corrected with failing-then-passing tests: missing archive state propagates to M4G on Roadmap, and local checkpoints require the supported schema plus exact parsed equality to the approved fit slot. The keyboard skip link was also corrected because it reset the active view; a React DOM test verifies preserved filters and focus. Browser CI subsequently exposed an exact accessible-label mismatch; explicit labels fixed it.

## Hosting and limits

[Private dashboard](https://thermo-research-workspace.otto-agent007.chatgpt.site) is a dated static snapshot, published successfully through Sites. Local mode refreshes local observations; hosted mode changes only after another export/deploy. Publication does not rerun scientific replay.

The cloud browser could not reach the supported local preview (`ERR_BLOCKED_BY_CLIENT`); actual browser checks and screenshots were completed in GitHub CI instead. The deployed owner-authenticated URL was not separately browser-tested. No pixel-identical fidelity, exhaustive accessibility audit, cross-browser coverage, or non-root hosting browser verification is claimed.

## Repository regression gates

The existing 32-command local gate runner completed. 29 commands exited successfully on the first run, including frozen dependency checks, Python lint/format/build/packaging, the second integration partition and the other experiment/audit commands. Three executions were affected by changing the working checkout while seed records were being generated: the composed finite-Gibbs command, one aggregation unit test and four composed-runner fixture setups. Their numerical identity fields agreed; their Git commit/dirty provenance differed, so the existing integrity guard correctly rejected aggregation.

With the original checkout frozen clean at `4bddc2c97d49982293130c8eb33b0d4311afbe93`, all affected checks passed unchanged: the three-seed composed command exited 0, the aggregation test passed (1/1), and the complete composed-runner file passed (11/11). The initial unit partition had 1602 passes plus that one failure; the first integration partition had 189 passes plus those four fixture errors. No scientific checks or implementations were weakened. Subsequent UI layout work stayed in a separate worktree; scientific tracked files remain byte-identical.

GitHub's broader scientific CI is separate from these local results and must finish before merging. PR36 is kept as a draft and depends on PR35.
