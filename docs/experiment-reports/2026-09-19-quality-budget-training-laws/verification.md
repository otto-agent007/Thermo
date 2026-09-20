# Verification and publication recovery

## Original verification — September 19

The complete verification run passed **1,901 tests**: 1,477 unit/upstream,
155 in integration batch A and 269 in integration batch B, including all slow
tests and all 53 new tests. All **21 experiment gates** passed. The 24 commands
partitioned the full test suite and executed every required experiment gate.

Frozen sync, offline lock checks, Ruff lint/format, whitespace checks, wheel
and source-distribution builds passed. Both distributions contained all ten
configs and both new modules. The new CI evidence-check shell block passed.
Separate statistical and implementation reviews completed, review fixes were
rechecked, and generated evidence passed independent complete replay review.

All commands ran against clean implementation commit
`4a50ded37737c1d7cb009d41614513cab362672b`, Git tree
`0b332d64affd4afff3c5844c74992ac217fb9e20`. The original final documentation
commit was `cc7f2ca606cb2b974ca241da968b038d63bcb321`. Publication was blocked
pending explicit approval.

## Recovery and fresh verification — September 20

After approval, the prior scratch checkout was unavailable. The preserved
preflight JSON, report, completion, provenance and summary survived in the
active workspace. The source, tests, CI configuration and documentation were
recovered from the recorded edits. **The complete implementation Git tree is
byte-for-byte identical to the originally verified tree above.** The recovered
implementation commit is `661c582240be18e60772132e973f49acaee1dde4`; commit
metadata differs, and original evidence provenance is retained unchanged.

Fresh checks on the recovered implementation passed:

- **98 affected tests** in 27.06 seconds, covering both decision and training-law
  components; no failed or skipped tests.
- Complete deterministic replay of the preserved preflight and exact report
  text equality. Result identity remains
  `sha256:07adfe6964346e9962301288a417381b1b13f962a5ea6a222c9e95cd322196dd`.
- Frozen dependency sync, offline lock check, Ruff lint/format and whitespace.
- Source-tree identity against the recorded original verified implementation.

The original full-suite counts above are the September 19 run, not a claim of
rerunning all 1,901 tests on September 20. Original raw verification logs and
the final documentation commit were not recoverable; their recorded command
outcomes survive in the conversation. This verification document was rebuilt
explicitly, rather than presenting reconstructed logs as original artifacts.
Hosted CI will independently verify the published branch.

The new gate records exact and sampled evidence separately and makes no
full-program quality, sample-savings, convergence or hardware claim. Its strict
canonical persisted replay is a current-runtime integrity policy, not a
promise of bitwise portability across numerical runtimes. Timings describe
local software execution, not physical-device performance.
