# Verification record

Published source commit `efd1f5aeb20dfd6490c6e3dc2ac9e155a4bd3aac` binds the new
pilot implementation; the request binds exact implementation file hashes and
the fixed arm schedule. `archive-metadata.json` binds compressed and raw JSON
bytes and the request/result digests. Runtime provenance is separate.

- The first new test run failed because the pilot module was absent. After
  implementation, the new pilot suite passed 5 tests; the combined new and
  prior return-fixture suite passed 25 tests. The weight-zero regression was
  then strengthened to compare all 200 proposal traces against the committed
  prior archive. The first dedicated CI run exposed tiny cross-machine
  float64 rounding differences in an exact-equality assertion. The corrected
  test checks identical structure, types and choices while allowing 1e-12
  absolute numeric differences; it passed locally and awaits CI rerun.
- Both pilot gradient preflight points checked every shared parameter
  component against central differences at tolerance 1e-7. Exact path and
  terminal laws, killed survival, and the previous fixture's own 20 focused
  tests remained covered by the combined suite.
- `uv lock --check --offline`, `uv run ruff format --check .`, `uv run ruff
  check .`, `uv build`, and `git diff --check` passed before archiving. The
  wheel and sdist were rebuilt with the pilot module.
- The CLI generated all four arms and reloaded compressed `study.json.gz`
  for complete numerical replay before reporting and writing completion.
  Generation and replay each made 842 exact evaluator calls, with two K4
  law calculations inside each call. No samples were drawn.
- A second fresh run matched the committed compressed archive byte for byte
  and regenerated the report byte for byte. Archive SHA-256 and byte counts,
  request/result digests and completion were checked independently before
  a further complete numerical replay from the committed compressed JSON.

The full repository suite is a slower collection of existing integration and
quality-budget tests. Its dedicated GitHub CI run on the new PR must finish
before the branch is considered merge ready. The previous return-fixture PR's
four GitHub workflows, including main CI, passed at its published commit.
