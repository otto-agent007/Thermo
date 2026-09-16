# Verification

Verified September 16, 2026. The final implementation and report code were
checked at clean commit `f9b86ce2081c0b54e022829754081dd344a07d13`.
The later verification-document commit changes no executable code or tests.

## Complete test coverage

The entire test tree passed in two non-overlapping runs, including slow tests:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  uv run pytest tests/unit tests/upstream_regressions
# 1,336 passed in 697.92 seconds

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  uv run pytest tests/integration
# 384 passed in 1,280.54 seconds
```

Total: **1,720 passed**, including all 17 new diagnostic tests. No test
categories, scientific thresholds, seeds or sample budgets were weakened.
The new tests check exhaustive tiny-state agreement, hidden-bit marginalization,
exit versus return, flux accounting, existing-sampler replay, malformed tables,
authenticated-source mutation, rehashed evidence forgery, report validation,
destination preservation, and completion suppression on validation failure.

An earlier broad run was interrupted after an existing aggregation gate rejected
mixed Git revision/dirty-state provenance: report commits had been made during
its multi-seed execution. The full test tree and affected composed-program gate
were rerun with a fixed clean commit. This required no application or assertion
change. BLAS/OMP thread limits bound host threading; they do not change requested
scientific budgets.

## Experiment and package gates

All 16 experiment commands completed successfully:

- Cross-library smoke and all ten checked experiment configurations.
- Full three-seed M2 audit (21 horizon cells).
- M3 exact gradient checks at six horizons.
- Full three-seed M4 study, five updates at K4, checkpoint five.
- Full three-seed M4B comparison, five updates per arm, K4 evaluation, and
  `matched_hardware_cost=false`.
- New conservation diagnostic: all three sources, both horizons, 32,768
  trajectories per cell, 500 operations, no updates, and complete reload/replay.

Completion markers require complete releases. M2 and M4 reproduced their
recorded result digests exactly. Scientific outcomes remain non-gating.

Frozen dependency synchronization, offline lock verification, full Ruff format
and lint checks, whitespace checks, wheel and source-distribution builds all
passed. Both package formats contain byte-identical copies of all ten checked
configurations and both new diagnostic modules.

Independent code and evidence review found no remaining blockers. The review
checked the recurrence and endpoint sampler, archive/replay boundaries, first-exit
creation/destruction, exact and sampled distinctions, reported figures, identical
initial parameter matrices, and sampling replication limits. Its request/result
hash wording concern was resolved transparently in the protocol. No code,
parameter, stream or result change was needed for that clarification.

This is local verification. No remote CI result is claimed for this branch.
