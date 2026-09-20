# Verification — September 20, 2026

The new runner was implemented from merged PR #33 (`391ed68`). Local
implementation `940cfc21f05488ed169010b41aa566b8a5583458` and published implementation
`c56a51abfba27a3d3b0921f106dc32454187ac21` have the identical complete Git tree
`6f53d780ffb099a6cd0cb3091a5815bfc53c7e53`. The GitHub app creates its own commit
metadata. No source changed during that publication.

The published evidence was freshly generated and replayed from the clean
published implementation checkout. Its provenance reports that commit and
`git_dirty=false`; its complete result equals the earlier verification-run
artifact. Result identity:
`sha256:a5af9d0e0153e336d791f04b738294b342dd9bbe7aa7039ab69cc1661be809ce`.

## Tests and repository gates

All 25 verification command groups returned zero: three exhaustive test
partitions, including slow tests, and all 22 required experiment gates.
[Commands, exit codes and elapsed times](command-results.json) and captured
[logs](verification-logs/) are retained, with trailing blank lines normalized.

| Test partition | Coverage | Evidence |
| --- | ---: | --- |
| Unit and upstream | 1,567 | pytest summary: 1,567 passed |
| Integration A | 187 | pytest summary: 187 passed |
| Integration B | 245 collected | complete command returned zero |
| Final runner unit rerun | 108 | tool output: 108 passed in 31.16 s |

The complete-suite partitions started before the final minor error-handling
fix and 18 additional review tests. The final 108-test runner rerun includes
those 18 and the original 90 runner unit tests. Together these passing runs
cover the final **2,017-test collection**. The initial two new modules also
passed 98 tests in 110.42 s, and the independent implementation reviewer
repeated them successfully (98 tests in 116.43 s).

The integration-B captured stdout contains only initial progress, without a
final pytest summary. Its zero process exit and complete invocation are
preserved in the command results; 245 is the independently collected count,
not a count transcribed from its stdout. Focused-run summaries above were
captured in tool responses rather than standalone stdout files.

The 22 experiment gates include every prior AGENTS.md gate and the new
training-runner component. No slow tests, historical gate, tolerance, source
pin or protocol requirement was removed. Environment: JAX CPU,
OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, PYTHONHASHSEED=0. The M4F gate also
uses the prescribed baseline NumPy CPU-feature path.

Additional passing checks:

- `uv sync --frozen` and `uv lock --check --offline`.
- Repository-wide Ruff formatting, lint, and Git whitespace checks.
- Wheel and source-distribution builds; both contain all ten experiment
  configurations and both new modules.
- Exact persisted preflight/report replay using final source.
- The new CI job's embedded completion/evidence/report check, executed locally.
- Fresh generation from the clean published implementation and equality with
  the original complete result.

## Independent reviews

Implementation and scientific reviewers independently inspected the runner,
preflight, tests, frozen design, archive binding and inherited samplers. No
critical or important issue was found. Two minor improvements were made and
re-reviewed: malformed request values now raise ValueError, and second-moment
forgery tests distinguish one-ULP exact-digest binding from numerical rejection
after repairing both source and result hashes. The malformed-input regression
was reproduced failing before its fix. Both reviewers cleared the final changes.

Reviews also inspected the generated report's scope. This is a training-engine
component: 21 diagnostic fixture fits, 105 fixture updates, zero production
fits and zero held-out cells. It does not certify the missing evaluator or
integrated full-study preflight. Hosted PR CI is a separate, subsequent check.
