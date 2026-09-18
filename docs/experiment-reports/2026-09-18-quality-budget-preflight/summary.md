# M4G decision-component preflight

The fixed M4G acceptance and accounting machinery is implemented and validated.
The command performs **zero model fits and zero held-out evaluation cells**.
It emits `component_preflight_complete` with `full_m4g_ready=false`.

## What is now checked

- The unchanged September 17 protocol and all three archived M1 source pins;
  the sources share one initial matrix.
- Twenty-one planned fits, sixty planned evaluation cells, 213 distinct random
  roles, and a complete cost ledger that counts all six finite fits per seed.
- Simultaneous binomial intervals: 55 coverage-grid checks, 31 independent
  binomial-tail inversions, and literal float64 threshold boundaries.
- A 4,096-batch joint binary fixture, correlated count-to-decision examples,
  729 horizon-status patterns and 784 distinct budget-bracket pairs.
- Strict reconstruction of persisted evidence, rejection of rehashed changes,
  and completion written only after successful report replay.

The smallest checked marginal coverage was 0.999968168150, above the declared
nominal 1 - 0.05/1560. The finite grid is numerical validation, not a proof of
uniform coverage. The guarantee comes from binomial Clopper-Pearson coverage
and a union bound, conditional on frozen fits and independent trajectory rows.

The finite grid requires 4,423,680,000 training endpoint draws versus 737,280,000
for equilibrium training: per-fit budgets match, but total finite-grid training
uses six times as many draws. Lower K would reduce modeled Gibbs sweeps; it
would not reduce independent trajectory counts. Equilibrium oracle work is
counted, with undefined finite sweep cost.

## Independent review

Separate statistical and implementation reviewers examined the component.
The statistical review identified NumPy float32 threshold/order promotion and
missing ratio intervals for inconclusive or equal-budget comparisons. Failing
regression tests reproduced the defects before correction. Literal float64
comparisons now reject reversed bounds, and all comparisons with two finite
certified budgets retain ratio intervals regardless of the savings decision.

The implementation review corrected the preflight timing scope and added
NumPy/SciPy versions to runtime provenance. Both reviewers rechecked the fixes
and reported no remaining component blockers. Review does not certify the
future matched runner or the full M4G experiment.

## Next lab task

Implement the matched seven-law training runner, validate its law-specific
occupancy/gradient roles on the bounded fixture at every K, retain the M3
contract, and complete runner integrity preflight before new fitting. Then run
all 21 fits and 60 cells under the unchanged stopping/selection policy. Task
quality and inference savings remain unanswered by this component milestone.

[Generated report](report.md) · [Reproducible evidence](preflight.json) ·
[Verification](verification.md) · [Frozen protocol](../../experiments/task-quality-inference-budget.md)
