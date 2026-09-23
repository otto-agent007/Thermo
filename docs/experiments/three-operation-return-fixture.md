# Exact three-operation return fixture

This protocol was frozen before generation. It follows the approved
[design](../superpowers/specs/2026-09-23-three-operation-fixture-design.md).
The existing two-operation study and completed M4G production study are
unchanged. This is an `exact_reference` CPU diagnostic with zero samples,
not a full-program or physical-hardware result.

Start in state 100. Apply edges (0,1), (1,2), (0,1) with the same target
PAsymSwap conditional and shared nine model parameters at each occurrence.
Use beta 1, K4 finite-Gibbs hidden-then-output sweeps, uniform hidden reset,
float64 and parameter bounds [-2,2]. Enumerate all 64 visible output
sequences, including invalid model paths that continue after failure for the
full terminal law. Uninterrupted survival kills each path at its first state
with particle count other than one and never revives it.

Minimize squared full-terminal occupancy error, complete-path target-to-model
KL, or valid-terminal divergence `sum_x q_x log(q_x/m_x)`; the latter uses
unnormalized surviving terminal mass. Exactly differentiate all three,
accumulating every occurrence into the same nine shared parameters. Record
the path objective difference as target-weighted conditional path KL given
terminal state. Report exact survival, terminal leakage, all-row hop and
asymmetry MAE, target-visited hop error and reverse 01 input behavior.
The numerical inequality `path KL >= valid-terminal divergence >= -log S`
is a diagnostic, not an interval-certified gate.

Use all six objective/step-rule arms and an unchanged baseline. Each arm
gets 201 complete evaluator calls: initialization then 25 rounds with eight
evaluations per round. Fixed makes eight successive projected steps of size
0.01 per round. Backtracking tests every projected step in
`[1,1/2,1/4,1/8,1/16,1/32,1/64,1/128]` from the round start, selecting the
first with negative gradient-dot-displacement and
`f_new <= f_old + 1e-4 * gradient-dot-displacement`. All proposals are
evaluated and retained after selection too. An unsuccessful round keeps its
start. All final arms are retained without hyperparameter selection.
Evaluator calls, not accepted updates or hardware work, are matched.

Preflight all 27 analytic shared-gradient components against central
differences with step 1e-6 and absolute tolerance 1e-7. Bind the versioned
request to the circuit, objective definitions, cost policy and implementation
hashes. Store every candidate, selection, metric and digest. Strict complete
numerical replay must validate a persisted record before rendering a report.
Write `completion.json` last. Runtime provenance is outside request identity.

Interpret survival together with local movement and asymmetry errors; a
conserving nonmoving kernel would miss the target dynamics. This single
deterministic three-operation fixture cannot establish convergence,
representational capacity, full M4G quality, inference savings or hardware
benefit. The two-operation results have a different circuit and are contextual,
not a matched comparator.
