# Matched-training-budget comparison (M4B)

Predeclared September 12, 2026, before implementation and new outcome generation.

## Question

Does training for K=4 produce lower terminal population occupancy loss at K=4
than equilibrium-directed training, when update and trajectory budgets match?
This is a comparison of training laws, not equal hardware cost or a claim of
reduced inference sampling requirements. Both occupancy and gradient roles use
their arm's law. It does not isolate a score substitution on a fixed law.

## Frozen protocol

- Use the three original M1 source records embedded in the committed M4 seed
  artifacts at main commit e6ce897. Validate full source lineage; start both
  arms from the actual initial parameters, never M1's updated parameters.
- Keep the 25-site, 500-occurrence schedule, 37 shared nine-parameter groups,
  exact target occupancy, beta 1, float64, and bounds [-2,2].
- Arms: `finite_k4` uses the checked finite endpoint probabilities and scores;
  `equilibrium` uses the existing exact equilibrium endpoint sampler and
  equilibrium sufficient-statistic main-minus-reference estimator.
- Each arm performs five projected updates, learning rate 0.01. Every update
  uses independent occupancy and gradient batches of 32,768 trajectories.
  References are independent same-parent draws and never propagate. All shared
  occurrences are summed before gradient second-moment reduction.
- New randomness: SeedSequence([0x4D3442, source_seed]).spawn(21), with the first
  ten children alternating occupancy/gradient for finite_k4, the next ten for
  equilibrium, and the last child exclusively for final evaluation. Seeds
  0,1,2 are independent study replications. Do not reuse M4 evaluation streams.
- Freeze the fifth update before evaluation. Evaluate three paired comparisons
  at K=4 with 32,768 trajectories per member: initial/finite, initial/equilibrium,
  equilibrium/finite. Reuse the same final stream across these comparisons so
  each member's terminal counts agree exactly. These comparisons are correlated.
- Primary contrast: finite minus equilibrium unbiased order-two occupancy loss.
  Negative favors finite training. Secondary: each arm versus initial, particle
  leakage, particle-count histograms, and projection activity. No outcome gates,
  hyperparameter tuning, checkpoint selection, or convergence claims.
- Preserve paired jackknife approximate normal 95% intervals, conditional on
  the frozen trained pair. Retain known small-sample/near-zero undercoverage;
  report descriptive pointwise intervals, not simultaneous or formal acceptance
  claims. Do not average intervals or treat paired trajectories as training-run
  replications. Three-seed means are descriptive only.

## Accounting and evidence

Each arm uses 5 * 32,768 * 500 * 3 = 245,760,000 logical endpoint draws in
training (occupancy, gradient main, gradient reference). Each explicit paired
evaluation uses 32,768 * 500 * 2 = 32,768,000 endpoint draws. Three pairs use
98,304,000 draws; common streams do not eliminate repeated execution.
Finite training represents 983,040,000 complete-sweep equivalents per arm;
the equilibrium oracle has no declared finite sweep count. At K=4, three free
pbits update per sweep. All are algorithmic equivalents, not executed Gibbs
updates or device energy. Report simulator timing separately and include table
construction in execution; exclude source validation, replay, reporting, and I/O.
No equal wall-clock, energy, or hardware-cost assertion follows from matching
draw counts. Retain setup, host, read/write and embedding costs as unevaluated.

Archive-import clarification before the completed study: authenticate each entire
canonical RunRecord against fixed hashes of the published M4-embedded sources.
Bind that hash into the M4B request and preserve all historical values and
provenance. This authenticates the input artifact; it does not freshly validate
or re-execute the unused M1 update or jackknife statistics. Independently check
the source specification, parameter bounds/digest, fixture schedule and exact
target. For independent target reconstruction only, use absolute tolerance
1e-14 (zero relative tolerance) for cross-CPU BLAS rounding; preserve the exact
archived target and reference for all M4B computation and request identity.
The full archive hash still rejects any mutation, however small. Recompiling historical parameters would change the intended starting
point, and rechecking unused derived scalars introduces host-dependent rounding.
Generic M1 validators remain unchanged. New M4B results still undergo full
numerical replay; only their historical platform label may differ on reload,
with Python, package and execution identities still checked.

All sampled results are software_simulation; exact local laws and tiny gradient
references are exact_reference. Preserve every projected update, role seed,
bounded source moments, input identities, and final joined moments. Reload must
replay the experiment and verify stored gradients with the existing M4 tolerance
(rtol 1e-12, atol 1e-10), while hashing exact stored moments. Bind request hashes
only to inputs, and result digests to complete evidence. Publish completion last.

## Completion and follow-up

Completion requires all three source seeds, checked CLI execution, adversarial
rehashed evidence tests, independent tiny-arm checks against existing samplers,
all repository gates, and a committed report with bounded replayable JSON.
Negative or inconclusive results complete the experiment successfully.
The existing M1–M4 contracts and results remain unchanged. A later, separately
predeclared study may vary inference samples or investigate Z1T activations;
this comparison alone cannot establish an inference-sample saving.
