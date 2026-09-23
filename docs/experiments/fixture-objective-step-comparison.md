# Exact fixture objective-by-step comparison

User approved the three-objective, two-step-rule comparison on September 21.
This protocol is fixed before generating results. Native execution is preserved.

Use the existing `build_checked_fixture()` three-site/two-operation circuit,
its target and initial nine shared parameters. Fix K=4, beta=1, float64,
uniform reset, hidden-then-output sweeps and bounds [-2,2]. No production
checkpoint, M4G training/evaluation seed or sampled observation is consumed.

Compare three minimized objectives: squared terminal occupancy error;
target-to-model visible trajectory KL; and joint valid-terminal divergence
sum q_i log(q_i/m_i), where m_i is unnormalized surviving terminal mass.
Exactly enumerate all 16 visible two-operation paths and differentiate their
probabilities, summing both shared-parameter occurrences. Model paths keep
executing after count failure for full terminal metrics; killed mass never
returns. Marginalize hidden outputs before defining visible trajectory KL.

Two step rules each receive 201 complete exact objective/gradient evaluations,
including initialization, split into 25 rounds of eight evaluations:

- Fixed: eight successive projected gradient steps of size 0.01 per round.
- Backtracking: evaluate all eight projected proposals at steps
  [1,1/2,1/4,1/8,1/16,1/32,1/64,1/128] from the round's starting gradient.
  Select the first with negative gradient-dot-displacement and Armijo decrease
  f_new <= f_old + 1e-4 * gradient-dot-displacement. No passing proposal means
  retain the current parameters. All proposals are recorded and evaluated,
  including those after the selected proposal. No short-circuiting or caching.

Thus fixed takes 200 updates while backtracking accepts at most 25. The matched
resource is complete evaluator calls, not update count, wall time, sampled
trajectories or hardware operations. Every call computes the same three
objectives/gradients and metrics. Gradient magnitude depends on objective units;
cross-objective differences do not isolate geometry from objective scaling.

Select each arm's final state after round 25 in advance; retain all six arms,
all trials, selected indices, zero-update rounds and the unchanged baseline.
Report full occupancy error, trajectory KL, joint valid-terminal divergence,
survival, terminal leakage, local unconditional hop/asymmetry MAE, displacement
and exact evaluator counts. No optimization of hyperparameters or early stop.

Validate all three shared gradients by independent central differences (step
1e-6, absolute tolerance 1e-7) at initialization before fitting. Validate path
KL decomposition and killed survival against existing independent propagation.
The inequality D(P||Q) >= -log S provides a sufficient survival bound exp(-D).
Label the float64 threshold comparison with -log(0.95) a numerical diagnostic,
not an interval-certified proof or satisfaction of the full M4G quality gate.

Persist canonical request/result digests, runtime provenance separately, and
all trials. Strictly reconstruct the complete experiment on reload before
rendering the report or writing completion last. Tampering with objectives,
selections, budgets or identities must fail, even after digest repair.

This is an exact-reference optimization diagnostic on one fixture and one
horizon. It cannot establish generalization, model capacity, convergence,
inference savings or hardware advantage. No automatic winner selection or
full-program training follows. Preserve all earlier protocols and results.

Pre-run structural clarification: this fixture has exactly one valid path per
valid terminal state. Trajectory KL and joint valid-terminal divergence are
identical, including their gradients. Retain both named arms as planned, but
do not treat them as distinct treatments or independent replication. Their
equivalence is tested. Distinguishing them requires a later circuit with multiple
valid histories per endpoint. No extra operation is added to this experiment.
