# Verification record

Published source commit `624ab9b846dcb000f829c154b7c3dcf81db8aa2d`
has the exact source tree used to generate the archived study. The request
binds the frozen schedule and implementation hashes. `archive-metadata.json`
binds both compressed and raw archive bytes, plus the request and result
digests; runtime timing provenance is recorded separately.

- The full-row unit suite first failed on missing implementation, then
  passed after implementation. It independently checked all 16 conditional
  errors and every gradient component at two vectors, strict survival and
  max-entry qualification, all trials and Armijo choices, weight-zero
  reproduction of the prior committed 200-proposal trace, and rejection of
  a forged rehashed proposal. A final report regression checked that both
  Markdown tables have matching 10- and 17-column headers, separators and
  rows. The final focused full-row suite passed 7 tests.
- The combined new and previous return-fixture unit suites passed 31 tests
  before the report-only column-count fix. `uv lock --check --offline`,
  `uv run ruff format --check .`, `uv run ruff check .`, `uv build`, and
  `git diff --check` passed. The final source change only corrected report
  formatting; the final full-row suite and lint passed after it.
- The gradient preflight checked 18 components at two points against
  centered differences; its maximum absolute error was 5.69e-9, below
  the predeclared 1e-7 tolerance. Generation and complete persisted
  numerical replay each made 842 evaluator calls (804 arm and 38 preflight
  calls), two K4 laws per call. The completion file was written last.
- Two fresh CLI runs from the frozen source produced identical compressed
  `study.json.gz` bytes and identical `summary.md` bytes. The archived
  completion has four arms, zero samples, and the request/result digests
  recorded in `archive-metadata.json`.

The dedicated GitHub workflow reruns the focused suite and whole study on
the PR. Broad CI and that workflow must complete before the draft is
considered merge ready.
