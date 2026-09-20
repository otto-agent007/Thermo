# M4G seven-law training validation

All six finite training laws (K = 1, 2, 4, 8, 16, 30) and the equilibrium
training law pass the existing three-site, two-operation fixture checks.
This completes a training-component prerequisite. **The full M4G runner and
21-fit/60-cell study remain pending. No task-quality or inference savings
result is established.**

## Evidence

- All six M3 exact-gradient contracts and the equilibrium reference remain intact.
- Maximum exact-gradient error: 1.44e-14; maximum finite-difference error: 3.73e-9.
- Independent replay matches occupancy counts and gradient sums/sum-squares
  exactly for all 21 diagnostic role pairs.
- All 42 diagnostic seeds are distinct and disjoint from the 213 study roles.
- Three pinned archives authenticate and retain their common initialization.
  Validation itself uses the existing bounded fixture, not the full 25-site program.
- Zero fits, parameter updates or held-out study evaluation cells executed.
  Completion retains `full_m4g_ready=false`.

At K1, substituting equilibrium occupancy into the finite-law gradient produces
a maximum gradient error of 0.0210331 on the fixture. The corresponding
wrong-score and missing-occurrence controls produce errors of 0.0341505 and
0.457863. These controls explain why both training roles must use the declared
law and why shared occurrences must be summed. They are validation results,
not improvements in full-program task quality.

## Review

Separate statistical and implementation reviews checked the code and evidence.
Review identified missing common bounds on equilibrium inputs and malformed
array-valued law names. Failing regression tests reproduced both; the fixes
now apply identical caps/dimension limits before every law and require exact
law-name types. Re-review found no remaining blockers. An unused analytical
second-moment field was removed to avoid confusing fixed-reward moments with
the random occupancy-derived coefficient. Realized sum-square replay remains.
The generated evidence and report passed independent complete replay review.

## Repository reconciliation

PR #31 landed in `main` before PR #32 was merged into its feature branch.
Consequently `main` lacked PR #32's decision preflight. This branch combines
`main` at `db68991` with the actual PR #32 merge `0125845`, preserving both
histories. The reconciliation tree is exactly the published PR #32 tree
`c1caa86277ef36b1d76db4e4a989e56df0ec6520`. The frozen M4G protocol is unchanged.

## Next lab task

Integrate the validated sampler into the seven-law, five-update full-program
runner. Bind every step to the archived initialization, fixed production role
schedule, law-specific tables, gradient moments, projection and selected fifth
checkpoint. Complete full-runner integrity preflight and independent review
before fitting. Then execute the entire predeclared 21-fit/60-cell grid,
including negative cells, and record the quality-versus-budget decision.

[Generated report](report.md) · [Replayable evidence](preflight.json) ·
[Verification](verification.md) · [Frozen protocol](../../experiments/task-quality-inference-budget.md)
