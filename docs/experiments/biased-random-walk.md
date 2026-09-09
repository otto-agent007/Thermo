# Biased random-walk reproduction specification

## Question

How do small conditional-kernel errors accumulate through a long stochastic
program, and how much do context matching and trajectory-level refinement
recover?

## Required variants

1. analytic continuous-time reference (`exact_reference`) where available;
2. separately labeled high-accuracy numerical reference when no analytic result exists;
3. discretized Torx target circuit;
4. independently compiled thermodynamic kernels;
5. context-matched kernels;
6. trajectory-refined kernels;
7. each compiled variant at multiple finite Gibbs horizons.

## Current independently compiled-kernel boundary

The completed independent variant is an atomic method-level reconstruction of
the Thermalizers paper's PAsymSwap gate. The paper's separate 5 by 5 periodic
fixture supplies 500 gate occurrences; Thermo deduplicates those target
channels and independently compiles each one as a declared five-spin,
two-color `K_(3,2)` thermodynamic kernel. Exact equilibrium and finite-horizon
conditionals are evaluated for the frozen kernels, with a THRML sampled K = 30
cross-check.

This is distinct from the five-node Torx weighted-graph baseline, which
reproduces the Torx paper's continuous/discretized graph-walk reference. The
independent variant does not use target/model trajectory context, does not
compose all 500 occurrences into a 25-site execution, and makes no official
Thermalizers-compatibility or hardware claim. Its approved `[-2, 2]`
field/coupling cap is a Thermo checked-input revision; the target,
uniform-context target-to-model KL objective, finite-horizon reset semantics,
and fixed acceptance gates are unchanged.

## Error decomposition

```text
target modeling error
time-discretization error
per-kernel compilation residual
context-distribution mismatch
finite-thermalization residual
trajectory drift
task/readout error
```

## Exact target-context matching

The completed target-context variant propagates the exact one-particle target
marginal in canonical macrostep, layer, and edge order. Immediately before an
oriented gate on sites `(i, j)`, its input-word distribution is

```text
mu(00) = sum of probabilities at the other 23 sites
mu(01) = q_j
mu(10) = q_i
mu(11) = 0
```

The zero is exact one-particle support. Profiles are not clipped, smoothed, or
renormalized. After recording each profile, both endpoint probabilities are
updated from the same pre-gate marginal; normalization and nonnegativity are
checked after every occurrence. This produces 500 ordered occurrence profiles.

Occurrences that share a target-channel hash also share compiler parameters.
Thermo therefore pools their profiles by an equal mean over occurrences for
that hash. The 500 occurrences reduce to 37 target-hash profiles, whose
multiplicities sum to 500. Profile support remains exact and unsmoothed,
including zero support for input word `11`.

Each pooled profile compares a paired uniform baseline with a target-context
artifact using target-to-model conditional KL and row-wise TV weighted by that
profile. Schedule-level KL and TV weight each profile by its occurrence
multiplicity and divide by 500; an equal average over the 37 hashes is not the
schedule metric. Improvement is therefore qualified as improvement under the
exact target input distribution, rather than a claim of general accuracy.

The report also evaluates all four input rows separately. Uniform-weighted
all-context degradation and diagnostics for exactly zero-support rows are
separate, required, non-gating assessments and appear beside the paired
improvement. They are exact evaluations of frozen software-derived artifacts,
not sampling noise and not acceptance gates for target-context accuracy.

The paper fixture, analytic channels, propagated marginal, occurrence
profiles, and pooled profiles are `exact_reference` evidence for the declared
target process. Exact equilibrium and finite-horizon conditionals are
`exact_reference` for each frozen five-spin software-derived model. Learned
artifacts, optimizer observations, and the seeded THRML cross-check remain
`software_simulation`: for every target-context artifact and input context,
4,096 chains estimate the conditional after `K = 30` complete two-color Gibbs
sweeps. No result is a calibrated projection or physical-hardware measurement.

This target-context study does not itself evaluate model-context matching or
the later program-level stages.

## One-pass mean-field model-context matching

The completed model-context diagnostic starts from the declared single-particle
site means and walks the same 500 occurrences in canonical order. At each
occurrence it uses the frozen target-context artifact's exact equilibrium
conditional, together with the current endpoint means under a first-moment
factorization, to update only those two endpoint means. It does not propagate a
25-site joint distribution or preserve correlations. The full deterministic
trace is hashed, including every upstream artifact identity and the expected
occupancy before and after each update.

As in the target-context study, occurrences are pooled by target-channel hash
using an equal mean over occurrences. The 500 profiles reduce to 37 groups.
For each group, Thermo recompiles one model-context artifact under its pooled
model profile, warm-starting from the paired target-context artifact before the
three checked fixed restarts. The persisted comparison retains the uniform,
target-context, and model-context variants.

The gate asks whether the model-context artifact improves occurrence-weighted
conditional KL relative to its paired target-context artifact under the model
profile, without per-profile regression beyond the declared tolerance. It also
requires valid optimizer endpoints and bounded exact `K = 30` residuals. A
separate seeded THRML cross-check uses 4,096 chains for each of four inputs on
all 37 model-context kernels. Only the maximum empirical `K = 30` residual is
eligible for cross-seed statistics; profiles, inputs, and chains are not
replications.

This is one feedback pass and a local frozen-kernel diagnostic. It does not
claim fixed-point convergence or improvement of the composed target program.
The full finite-Gibbs-horizon composed-program comparison is a separate,
completed evaluation of frozen artifacts; the separate one-step equilibrium
refinement below updates the full 25-site program. This study also does not evaluate
official Thermalizers, hosted simulation, physical Z1, or TSU hardware.

## Exact trajectory-level estimator contract

The checked estimator study is a deliberately bounded prerequisite for the
refinement variants below. It composes two overlapping occurrences of one
shared five-spin `K_(3,2)` kernel on three visible sites. Exact enumeration
checks the trajectory-score gradient against an independently derived
expected-reference identity, untied occurrence finite differences, and a tied
shared-parameter finite difference. The exact terminal law also determines the
particle-number leakage and signed mass drift.

Each release seed is one independent batch of 65,536 augmented trajectories.
Every sample contains two propagated main exact-categorical draws and one
independent, same-parent, non-propagated reference draw per occurrence. The
reported shared-gradient standard error is reconstructed from occurrence
second moments and cross-products, so it retains the within-trajectory
covariance created by summing the two shared-parameter contributions. The
sampled comparison is non-gating; only deterministic exact identities define
acceptance, and independently seeded batches are the replication units.

The command is:

```bash
uv run thermo-lab run \
  configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/trajectory-reinforce-estimator
```

This contract does not update the shared parameters, test optimization or
trajectory improvement, run the 25-site 500-occurrence fixture, or establish
unbiasedness for a finite-Gibbs sampler. Its seeded categorical draws are
NumPy `software_simulation`, not THRML, official Thermalizers, hosted
simulation, or physical Z1/TSU evidence. The full finite-Gibbs-horizon
composed-program comparison and one-step full-program refinement are separate
evaluations below.

## Bounded one-step trajectory refinement

The checked refinement keeps the estimator's three-site, two-occurrence
circuit and ties the same nine shared parameters at both occurrences. For each
release seed, its covariance-aware sampled shared gradient defines exactly one
controlled update:

\[
\phi_{\mathrm{raw}} = \phi_0 - 0.25\,\widehat{\nabla J}, \qquad
\phi_1 = \operatorname{clip}(\phi_{\mathrm{raw}}, -2, 2).
\]

Exact enumeration evaluates the declared squared occupancy objective at
\(\phi_0\) and \(\phi_1\). The artifact records both vectors, cap flags, the
exact objective change, and whether the strict-decrease acceptance rule passes,
while retaining the trajectory-score, expected-reference, and finite-difference
gradient checks.

```bash
uv run thermo-lab run \
  configs/experiments/numpy-trajectory-reinforce-pasym-swap-one-step.toml \
  --seeds 0,1,2 \
  --output-dir results/trajectory-reinforce-one-step
```

This is a bounded one-update experiment with exact-categorical transitions and
seed-derived NumPy `software_simulation` parameters. It is not iterative
optimization, the 25-site 500-occurrence program, a finite-Gibbs-horizon
comparison, official Thermalizers, hosted simulation, or hardware evidence.

## Full composed finite-Gibbs evaluation

The completed composed comparison executes all 500 canonical PAsymSwap
occurrences on 25 sites for the frozen `independent`, `target_context`, and
`model_context` artifact families. Each family is evaluated at `equilibrium`
and finite Gibbs horizons `K = 1, 2, 4, 8, 16, 30`, producing 21 paired
family/horizon cells. Finite transitions use the declared uniform reset and
complete hidden-before-outputs sweeps; the experiment does not run a live
THRML chain.

Seeds `0`, `1`, and `2` are the independent cross-run replication units. Each
seed contains 32,768 complete trajectories per cell as within-batch samples.
Sampling uses NumPy `Generator(PCG64)` and one float64 uniform vector per
occurrence, reused across every family/horizon cell, so paired differences use
common random numbers. The common-random-number-correlated cells and their
within-batch trajectories are not extra independent replications. Checkpoints
are occurrence zero and every 50 occurrences through 500. Exact target
checkpoints and the reconstructed frozen local conditional tables are
`exact_reference`; sampled rollouts, paired comparisons, and timing are NumPy
`software_simulation`.

The report records depth-wise occupancy half-L1 error, probability leakage from
the one-particle sector, particle-number and path/invariant diagnostics, final
marginal and observable error, local conditional residuals, and paired
error/leakage comparisons. In the audited release, all three integrity rows
were accepted: 693 checkpoint summaries and 147 aggregate scalars were
reconciled. The checked request, frozen bundle, and exact target-checkpoint
identities are respectively
`sha256:51f554b4b1e7e747618ec75de5e4aa0a1dda8a93e29967bcd84d2116510f28a4`,
`sha256:d21bedf938b533b515dfc7d497c1668897a5cea663185e343a3306ba45d981bd`, and
`sha256:376b0d20a289d849326b710e023827cb6260bc081f953e374a0e127b65fc3a47`.
Across the three seeds, the final paired error/leakage comparison has 89
improved, 37 worsened, and 0 unchanged outcomes. These are descriptive,
non-gating mixed outcomes, not independent tests or an optimization acceptance
criterion.

Reproduce the checked release command with:

```bash
uv run thermo-lab run \
  configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml \
  --seeds 0,1,2 \
  --output-dir results/composed-pasym-swap-finite-gibbs
```

This completes an evaluation of frozen artifacts, not iterative optimization.
The separate one-step refinement below updates all 25-site parameter groups. It makes
no claim about official Thermalizers, hosted simulation, physical Z1/TSU
hardware, or live THRML sampling.

## One-step full-program equilibrium refinement

The checked command is:

```bash
uv run thermo-lab run \
  configs/experiments/numpy-composed-pasym-swap-trajectory-refinement-one-step.toml \
  --seeds 0,1,2 \
  --output-dir results/composed-trajectory-refinement-one-step
```

Start from the audited model-context lineage and share nine parameters within
each of 37 target-hash groups over all 500 occurrences. For each release seed,
derive independent occupancy, gradient, and evaluation seeds using
`SeedSequence(seed).spawn(3)`. Estimate terminal occupancy from 32,768
trajectories and use `2 * (estimated_occupancy - exact_target_occupancy)` as
the reward coefficient for a separate 32,768-trajectory gradient batch.
Every occurrence draws an independent reference from the same main-path
parent; only the main output propagates. Sum scores within each shared group
before taking trajectory moments, preserving occurrence covariance.

Apply exactly one gradient-descent step at learning rate 0.01 and project
every parameter onto [-2, 2]. A third held-out batch of 32,768 trajectories
evaluates before/after with common random numbers per occurrence. The
declared objective is the sum of squared terminal occupancy errors. The
exact target is `exact_reference`; sampled model objectives are
`software_simulation` with finite-batch bias from squaring empirical means.
Improvement is descriptive and non-gating.

The September 9 release completed all requested seeds with accepted integrity:

| Seed | Objective before | Objective after | Before minus after | Parameters changed by projection |
|---|---|---|---|---|
| 0 | 0.05168472917704707 | 0.05117100474698251 | 0.0005137244300645605 | 58 |
| 1 | 0.047790044264095374 | 0.04726481925853825 | 0.0005252250055571214 | 56 |
| 2 | 0.049938136558384465 | 0.04937294805831944 | 0.0005651885000650253 | 50 |

All updated parameters satisfy the bounds. These small sampled decreases do
not establish iterative convergence. Iterative and finite-Gibbs parameter refinement remain deferred.

Request identity:
`sha256:ce594c55034305bc25eb58f6b847a83e33dc027c69f0d8290d620522de20eec4`.
Validated summary identities, in seed order:

- `sha256:c8c0d6269c0c2c8ee3fadca35edb726494fb8fe0e788e43360c43f9dc83a0577`
- `sha256:c1abc247fe1c69d1c539d8357833c0026a6b15cd98dfca7baee600e638b5cbde`
- `sha256:0ed279cc868668cd2d7ceba07cbd250ac1ad3e93b135a36ddbf85ea03c6971d2`

Reload validation reconstructs every source digest, reward, gradient mean,
projected update, and objective from bounded counts/moments and trusted
lineage. It binds seeds, sample counts, schedule, parameter constraints,
metadata, and standalone scalar copies before aggregation/reporting. Exact
three-site score/reference/finite-difference checks and the full-program
gradient primitive's three-site oracle comparison remain in the test gates.
No official Thermalizers, hosted simulator, or physical hardware claim is made.

## Metrics

- conditional KL and total variation by input context;
- occupancy half-L1 error by program depth;
- probability leaking outside the one-particle state space;
- path/invariant violations;
- final marginal and observable error;
- hidden p-bits and Gibbs sweeps per kernel;
- reads, writes, clamps, reflashes, and host round trips;
- logical and physical graph sizes;
- evidence class for every metric.

## Acceptance

- Small kernels match exact conditional distributions within predeclared bounds.
- The report exposes local and composed errors separately.
- Context matching is trained on recorded target/model contexts, not silently on
  a uniform distribution.
- No projected Z1 cost is labeled as a physical measurement.
