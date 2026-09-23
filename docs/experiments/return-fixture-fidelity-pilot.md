# Three-operation fidelity penalty pilot

Freeze this protocol before running the pilot. This is one deterministic
`exact_reference` three-site circuit, not M4G production fitting or an
inference sampling experiment. Preserve the prior fixture, its shared nine
parameters and bounds [-2, 2], K4 law, initial state 100, and operations
(0,1), (1,2), (0,1). Retain the full terminal distribution and uninterrupted
first-failure survival from exact enumeration of all 64 visible paths.

Compare four arms at penalty weights 0, 1, 10 and 100, with the same initial
parameters and 201 complete evaluator calls per arm. Weight zero must
reproduce the archived path-KL/backtracking result numerically. Each arm
minimizes complete-path target-to-model KL plus its fixed weight times the sum
of squared absolute errors in the logical forward 10→01 and reverse 01→10
probabilities. Differentiate the local K4 visible law through the same shared
nine parameters. Do not choose or adjust weights after seeing results.

For each arm, take 25 rounds with eight evaluated projected proposals per
round, steps [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128], accepting the first
with negative gradient-dot-displacement and objective no larger than the
start plus 0.0001 times that displacement. Evaluate and retain all proposals,
including later and rejected ones. Preserve an unsuccessful round's start.
Selection uses the declared objective only. Keep all final arms and report
every evaluated proposal, parameter, objective, gradient, and metric.

Pilot decision: a checkpoint qualifies only when exact uninterrupted survival
is at least 0.95 and *each* logical forward and reverse hop probability has
absolute error at most 0.01. Report the two errors, all-row hop and asymmetry
MAE, row 00/01/10/11 four-output MAE, visited-row MAE, KL, and survival for
every final arm. Treat the thresholds as diagnostic for this fixture, not the
full M4G acceptance rule. A failure is a result at this budget; it is not a
capacity or convergence proof.

Check nine penalty-gradient components against central differences at the
initial vector and a second interior vector. Bind the request to implementation
hashes, weights, schedule, tolerances, bounds, and cost policy. Complete
persisted numerical replay before rendering a summary, write completion last,
and record runtime provenance separately. Report samples as zero. Do not infer
500-operation quality, inference-sweep savings, or hardware benefit.
