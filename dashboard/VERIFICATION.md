# Dashboard verification record — 2026-09-20

## Scope

Read-only UI added outside scientific source, configs and tests. Recorded M4G evidence is unchanged. Numeric values come from the approved byte-pinned archive. Browser payloads are compact and use decimal strings for large seed identities.

## Automated checks

- Unit/SSR tests: archive counts, all60 unmet decisions, seed precision, changed/missing evidence, unsafe paths, activity observations, static export, filters and CSV, unavailable-state headline.
- TypeScript typecheck and Vite production build pass.
- Clean-checkout unit execution catches accidental dependency on ignored local results. Activity fixtures now come from committed study fits.
- Repository regression gates are recorded separately in the PR once complete.

## Browser limitation

Browser verification is **not complete**. The local development server encountered restricted network interfaces. The supervised preview reported healthy, but the cloud browser returned `ERR_BLOCKED_BY_CLIENT` for its supported address. A fallback Chromium download timed out and was stopped. No screenshot or successful interaction run is claimed. The authored Playwright suite runs in the Dashboard CI job.

## Visual fidelity ledger

Concept: generated full overview, `exec-cd631729-8d23-411c-82f1-92ceecc4d7ec.png` in the session's generated images. There are no raster assets in the application. Browser comparison remains pending.

| Point | Implementation decision | Verification |
| --- | --- | --- |
| Copy | Preserve dashboard destinations and finding; replace invented dates/counts with records; M5 retains its actual roadmap title | Source/SSR checked; visual pending |
| Layout | Sidebar, overview main column, context column; mobile stacks | CSS reviewed; viewport pending |
| Palette | Cool gray, white, dark text, teal data, amber unmet quality | CSS tokens checked; rendered comparison pending |
| Typography | Explicit heading, body, control and metadata styles; system sans | Source checked; computed rendering pending |
| Containers | Open metric-count band per written spec instead of concept's four cards | Intentional deviation; visual pending |
| Icons | Document for recorded verification, question for unknown; no success check for missing evidence | Source checked; visual pending |
| Interactions | Shared filtered rows, exact detail drawer, CSV, abortable requests | Data tests pass; browser suite pending |

No claim of pixel fidelity or completed visual approval is made. The private snapshot can be used for review while browser CI remains the merge gate.

## Independent review and corrections

The independent whole-branch review found no Critical issues. Both Important findings were corrected with failing-then-passing tests: missing archive state now propagates to M4G on Roadmap, and local checkpoint observations require the supported schema plus exact parsed equality to the approved fit slot. A keyboard shortcut issue was elevated and corrected because it reset the active view; a React DOM interaction test now verifies preserved filters and focus.

Final dashboard suite: 15 passing tests, including clean-checkout fixtures and DOM interaction. Typecheck and production build pass. A DOM harness does not establish layout, pixel fidelity or actual-browser modal behavior.
