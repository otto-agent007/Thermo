# Three-operation return fixture and objective comparison

Status: design for user review, September 23, 2026. No three-operation
experiment has been executed or interpreted.

## Purpose and success criteria

The completed two-operation exact fixture gave 95.8% survival under the
path-aware objective with backtracking, but its two path objectives were
identical and its schedule never exercised a reverse-hop input. The next
experiment should separate the objectives and expose that input before any
larger program or sampling-budget claim. The user's goal is an auditable,
bounded comparison that shows what the additional operation changes.

Success requires (1) a checked three-operation fixture with at least two
positive-probability valid histories reaching the same terminal state, (2)
independently checked exact values and nine-component shared gradients for
occupancy, target-to-model path KL and valid-terminal divergence, and (3) a
predeclared equal-evaluator-budget comparison that reports all arms and their
fidelity/survival trade-offs. A gap between objective values is a diagnostic;
it does not itself establish that either trained arm has better task quality.

## Approaches considered

| Approach | What it answers | Limitation |
| --- | --- | --- |
| Append edge (0,1) to the existing fixture (recommended) | Exercises parent 01 on the return step and merges distinct valid histories at endpoints, with the same kernel and three sites | Still one small deterministic circuit |
| Change initial state or reorder the existing two edges | May cover a missing context | May still leave one valid history per terminal state, obscuring the objective distinction |
| Move directly to the 500-operation M4G schedule | Tests the actual long task | Changes numerical scale and optimization difficulty before the structural question is isolated |

## Frozen circuit and evidence class

Introduce a **new** fixture and versioned study contract. Leave
`build_checked_fixture()`, the two-operation evaluator, its result digest and
archive unchanged. Start at visible state 100. Apply edges (0,1), (1,2),
(0,1), in that order. Use the existing PAsymSwap target conditional for each
edge, the same nine shared model parameters at every occurrence, beta 1,
uniform hidden reset, hidden-then-output finite Gibbs sweeps, K=4, float64,
and componentwise parameter bounds [-2,2]. Do not borrow M4G study random
roles or fitted checkpoints. This is `exact_reference`, not sampled training
or a physical-device result.

Enumerate all 4^3=64 visible output sequences, including model paths that
violate count conservation. Continue their transitions to obtain the full
terminal law. For survival, kill probability mass at its **first** invalid
state; a later return to one particle does not revive it. The logical target
has one particle at every operation and assigns zero mass to invalid paths.
The final (0,1) occurrence exposes parent 01 when a particle occupies site 1.
Both a stay-at-site-0 history and a leave-and-return history can end at site 0.

For every path with target mass P(path)>0, use the marginal visible model
probability Q(path) after hidden-spin marginalization. Minimize:

- Squared terminal occupancy residual over all model paths, matching the old
  fixture's occupancy definition.
- D(P||Q) over complete visible paths, with contributions and gradients from
  all three occurrences accumulated into the nine **shared** parameters.
- J=sum_x q(x) log(q(x)/m(x)), where q is the target terminal distribution
  and m(x) is the unnormalized mass of valid model paths ending at x.

Record the identity D(P||Q) = J + expected target conditional path divergence
given terminal state. Check nonnegativity numerically and retain the actual
gap, including if it is zero; do not assert strict separation by construction.
Also check D(P||Q) >= -log(survival) as a numerical sufficient-bound
diagnostic, with no interval-certified threshold claim.

## Comparison and measurements

Use the existing six-arm design: three named objectives crossed with fixed
projected steps and bounded backtracking, plus one unchanged baseline. Each
arm gets 201 **complete** evaluator calls: the initial evaluation and 25
rounds of eight. Fixed uses step 0.01 for each of 200 updates. Backtracking
evaluates all eight steps [1, 1/2, ..., 1/128] from the round's starting
gradient, accepts the first Armijo candidate with negative slope and coefficient
1e-4, or keeps the starting point. Record all proposals, including rejected
ones, without early stopping or tuned hyperparameters. Select the final state
after round 25 and retain every arm regardless of result. Evaluator calls,
not accepted updates, samples, time or hardware operations, are matched.

At baseline and for every final arm report all three objectives, their
difference, terminal occupancy error, exact uninterrupted survival, terminal
leakage, unconditional local hop and asymmetry MAE, and errors for the target
visited input rows (with their visitation weights). Show the 01 reverse-hop
row explicitly alongside 10. Report gradient norms and projection activity
so objective scale and clipping remain visible. Do not rank an arm on survival
alone: fidelity remains a separate measurement. The old two-operation result
may be displayed as context but is not a matched comparison with this circuit.

## Verification and persistence

Use a separate evaluator and study module or small shared path primitive;
do not change the old study's output schema or code-bound request. Bind the
new circuit, target, initial parameters, implementation hashes, objective
definitions, optimizer settings, complete evaluation budget and evidence label
in a new request version. Store every trial, final metric, gradient preflight,
accounting and separate runtime provenance. Write completion last after strict
persisted replay has regenerated the result and report.

Tests must independently enumerate the 64 paths, reconstruct terminal and
killed laws, demonstrate positive target mass on two merged histories, and
exercise parent 01 at the return step. Compare analytic shared gradients for
all three objectives with central differences at initialization and at a
nontrivial interior point. Check path-KL decomposition, valid-terminal masses,
and bound identities against an independent reference. Reject tampering with
the schedule, shared derivatives, trial selection, budget or archived record
even when a digest is recomputed. Run the repository's relevant unit gates,
format/lint and the new CLI gate before reporting a finding.

## Limits and next decision

One exact three-site fixture cannot demonstrate 500-operation fidelity,
representational capacity, convergence, inference-sample savings or hardware
advantage. These results may motivate a separately specified fidelity-constrained
pilot or a richer-kernel screen. Neither follows automatically from survival or
from a path-objective gap. M5 remains a separate roadmap milestone.
