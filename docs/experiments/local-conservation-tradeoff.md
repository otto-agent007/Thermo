# Bounded local conservation–fidelity trade-off

Predeclared September 16, 2026, after PR #26. Evaluate whether a fixed local
search can improve conservation without erasing PAsymSwap's directed hopping.
This is an attained trade-off study, not a representational lower bound.

## Inputs and exact objective

Authenticate the three original M1 records embedded in the committed M4 archive.
Require identical initial parameters and schedules, then optimize their shared
37-group matrix once. The source seeds are not independent fits. Rebuild the
canonical PAsymSwap targets in their sorted hash order. Keep the existing nine
parameters, caps [-2,2], beta 1, float64, uniform free-state reset, and four
hidden-then-output Gibbs sweeps. No sampled training or full-program updates.

For each group let P be the exact visible K4 conditional, T its logical target,
and M[a,b] indicate a change in particle count. Use uniform weight 1/4 for all
four parent contexts, including 00 and 11:

- Fidelity F = sum((P-T)^2)/4 (squared conditional error).
- Conservation C = sum(P*M)/4 (mean probability of changing particle count).
- Objective J = F + lambda*C, for lambda in [0,1,10].

Differentiate P using the checked finite-sweep Jacobian, including all four
sweeps. No equilibrium gradients, context reweighting, rejection, or repair.

## Fixed optimization budget and selection

For every group and penalty independently, start from (a) its authenticated
archived initial parameters and (b) nine zeros. Perform exactly 100 projected
gradient updates with step size 1/(1+lambda): theta <- clip(theta-step*grad J,
-2,2). No early stopping, line search, adaptive budget, warm start across
penalties, or additional restarts. Evaluate each start and its update-100
endpoint. Select the lowest exact objective among those four candidates,
breaking ties by archived start, archived endpoint, zero start, zero endpoint.
Thus the declared search cannot return an objective worse than the archived
start, but it can fail to find a better solution. Total: 22,200 group updates.
Record both endpoint attempts, objectives, gradients, and selected candidates;
report a projected-gradient residual without asserting convergence.

## Evaluation and stopping

Report every penalty and the frozen baseline, plus the exact conserving logical
reference. Report per-group/per-parent conservation failure and total variation,
mean and maximum row TV, unconditional hop-probability MAE, hop MAE conditional
on conserving the single particle, and directed-hop asymmetry MAE. Conditional
hop fidelity never substitutes for the unconditional error or conservation.
Use uniform group/context averages; retain group details rather than treating
37 related targets as independent replicates.

Recompose every selected matrix through the unchanged 500-operation schedule
using the exact killed one-particle recurrence. Report its final uninterrupted
survival and first-exit curve. This is not a full-state terminal distribution,
terminal occupancy fidelity, or a hardware experiment. The logical reference
must conserve throughout but need not be representable by the capped kernel.
No Monte Carlo samples, confidence intervals, statistical acceptance threshold,
or winner selection across penalties. Stop after this grid regardless of outcome.

## Integrity and release

Bind source identities, requested parameters/targets/schedule, optimizer policy,
and evaluation definitions in the request hash; bind derived tables, selected
parameters and measurements in the result digest. Embed authenticated sources.
On reload reconstruct the request and replay all updates, selection and exact
measurements before reporting. Write completion last into a fresh directory.
Preserve runtime provenance and evidence. Cross-environment floating-point
changes must fail exact replay rather than silently authorize altered evidence.

Test analytic gradients with central finite differences at interior points;
check penalty masking, clipping, fixed update counts, deterministic tie-breaking,
selection, logical conservation, and tamper rejection. Outcomes are non-gating.
Preserve all existing gates and provenance boundaries. The search and its
finite budget cannot prove a global optimum, a cap-induced impossibility,
benefit of a new architecture, or a physical device advantage.
