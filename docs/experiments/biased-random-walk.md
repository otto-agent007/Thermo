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
