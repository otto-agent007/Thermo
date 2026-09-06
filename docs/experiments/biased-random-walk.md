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
Trajectory-level REINFORCE refinement and the full finite-Gibbs-horizon
composed-program comparison across all 500 occurrences on 25 sites remain
deferred. The study also does not evaluate official Thermalizers, hosted
simulation, physical Z1, or TSU hardware.

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
