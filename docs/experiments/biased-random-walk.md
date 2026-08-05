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
