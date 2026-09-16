# M4F: bounded asymmetry preservation at K4

Predeclared September 16, 2026, before any M4F fitting. Test one added loss term
on the existing context-weighted penalty-1 cell. This completes the bounded
asymmetry question before M4G; a negative result does not authorize more tuning.

## Pinned reference and single intervention

Use the complete committed context-weighted-conservation/study.json artifact:
`sha256:d01e83fe9a9fb790910ddf964448aa19a98bb506cd547d3204d3b7e11139d816`.
Its recorded result is
`sha256:22be67a731ebdb55f19736da0a1b95ebbaf2df86410bf11c25021712cd428222`.
Authenticate this entire artifact, its embedded uniform-control pin, and all
three original M1 source records. They share one initial parameter matrix;
source seeds do not represent independent fits.

For each of the 37 groups, let P be the visible K4 law, T the logical target,
and w the archived exact logical parent-context weights. Keep the existing
F_w + C_w objective (conservation penalty lambda = 1). Add

A = ((P[01,10] - P[10,01]) - (T[01,10] - T[10,01]))^2

with coefficient mu = 1, so J = F_w + C_w + A. Unit coefficient is chosen in
advance as one unscaled squared-error term, not from an outcome sweep. A is
unweighted across the two directions to target the existing uniformly averaged
group asymmetry diagnostic. It is not multiplied by context probabilities or
group multiplicity. Differentiate both directions through the actual finite
sampler. Minimized squared error and reported mean absolute error are distinct.

## Fixed optimization budget

Retain beta 1, float64, K = 4 complete hidden-then-output Gibbs sweeps, uniform
free-state reset, nine parameters per group and bounds [-2,2]. Use the same
archived-initial and zero starts, exactly 100 projected updates from each, and
unchanged step 1/(1+lambda) = 1/2. Select the minimum J among archived start,
archived endpoint, zero start and zero endpoint, with ties in that order.

The new arm performs 37 x 2 x 100 = 7,400 group updates. Replay the original
weighted penalty-1 arm's 7,400 updates as a matched control. Do not rerun or
refit the unused penalty-0/10 or uniform grids within this new study. Full
validation repeats the consumed control and new-arm work; report replay as
verification overhead. No samples, extra starts, warm starts, coefficient grid,
learning-rate adjustment, checkpoint search, or early stopping.

## Numerical integrity and reference semantics

Reconstruct the original uniform request from the authenticated sources and
check it exactly. Rebuild logical contexts and the consumed logical-reference,
frozen-initial and weighted penalty-1 cells. Compare every consumed numeric
output at absolute tolerance 1e-12, relative tolerance zero; preserve scalar
types, structure, discrete choices and input policies exactly. Only the cells'
derived table digests and the contexts' derived trace/profile hashes are exempt
from cross-runtime equality. The complete artifact pin still authenticates
those original hashes. Keep the archived contexts and controls as the actual
inputs and comparison reference. This is a new study-specific compatibility
check, not a change to the historical schema-1.0 validator.

Bind reference identity, coefficient, formulas, update policy, replay tolerance,
comparison rules and stopping rule in the request. Persist the complete pinned
reference, new fits and parameters, all metrics and decisions; reconstruct them
on reload. New results require strict complete replay, including derived context
checks and all comparisons. Write completion only after successful report replay,
and refuse existing output directories. Preserve historical evidence and gates.

## Evaluation, success and stopping rule

Evaluate four cells: logical reference, frozen initialization, archived weighted
penalty 1, and the new asymmetry arm. Retain all-parent fidelity/TV/failure,
conditional and unconditional hop errors, asymmetry MAE, all 500 killed-survival
rows, occurrence-weighted logical-context F/C, mean empty-edge failure and mean
hop probability. Add per-group signed asymmetry errors and mean squared error.
Report candidate-minus-reference comparisons against the weighted control and
frozen initialization, regardless of outcome.

The predeclared primary joint screen requires all three:

1. final uninterrupted survival >= archived weighted penalty-1 survival;
2. asymmetry MAE <= frozen-initial asymmetry MAE;
3. unconditional hop MAE <= archived weighted penalty-1 hop MAE.

The pinned values define these thresholds (approximately 0.06149585 survival,
0.02903801 asymmetry MAE and 0.02485197 hop MAE). Use their full stored precision
and literal comparisons, with no scientific tolerance. Also report signed
margins and each component separately. This is descriptive and non-gating; the
CI gate checks integrity, not a favorable outcome. Endpoint residuals are
optimizer diagnostics, not convergence certificates.

Stop after this one fit. Update the roadmap with the outcome and proceed to the
M4G quality-versus-inference-budget protocol decision even if this screen fails.
Do not claim preserved full-program task quality, fewer inference samples,
convergence, optimal capacity, or hardware advantage. This study uses exact
finite-sampler calculations, not a limited sampled training batch.
