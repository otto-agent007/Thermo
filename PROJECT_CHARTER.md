# Thermo project charter

## Project identity

Thermo is a reproducible research laboratory for designing stochastic programs,
implementing native thermodynamic models, lowering algorithms toward
TSU-compatible kernels, and evaluating them across exact references, software
sampling, calibrated hardware projections, and physical thermodynamic hardware.

The project uses Extropic's stack where it is publicly available:

```text
Torx ──→ Torx exact/sampled software or Extropic simulator API
  │
  └──→ Thermalizers lowering ──→ THRML kernels
                                     ↑
Native THRML models ─────────────────┘
                                     │
                                     └──→ local software, Extropic API, future Z1
```

Only Torx and THRML are installable public dependencies today. The remaining
layers are research boundaries, specifications, or future integrations.

## Goals

1. Build stochastic algorithms in both high-level Torx and hardware-near THRML.
2. Validate small systems exactly before trusting approximate or composed runs.
3. Measure kernel, trajectory, and task-level error separately.
4. Study finite thermalization, block scheduling, sparse connectivity, and
   logical-to-physical expansion.
5. Design algorithms around the cost of sampling, clamping, reading, writing,
   and host round trips rather than a clock-rate headline.
6. Produce auditable experiment records with pinned dependencies, source state,
   device provenance, timing semantics, sample definitions, and evidence labels.

## Research tracks

### A. Native THRML algorithms

- Ising, Potts, and discrete energy-based models
- Block and chromatic Gibbs schedules
- Hidden-spin representations and coupling constraints
- Associative memory, denoising, Max-Cut, and graph optimization
- Learned temperature and update schedules

### B. Torx stochastic programs

- Random walks and diffusion processes
- Sequential Bayesian inference and state-space models
- Chemical reaction networks and stochastic graph algorithms
- Hybrid discrete-continuous programs
- Sampling-based control and planning

### C. Compilation and hardware co-design

- Kernel decomposition and reusable stochastic primitives
- Variational compilation, context matching, and trajectory refinement
- Finite-thermalization and connectivity-aware training
- Hidden-p-bit allocation and sparse placement
- I/O-aware stochastic-program transformations

Track C remains interface and experiment-design work until Extropic publishes
the promised Thermalizers implementation or grants access to a supported API.

## Scientific boundaries

- A GPU benchmark cannot establish Z1 energy, power, or latency.
- The Z1 Appendix-B operation model is a calibrated projection, not a chip
  measurement and not a complete system-energy model.
- Advertised update frequency is not an independent-sample rate.
- A recorded Gibbs state is not automatically an effective independent sample.
- Local conditional accuracy does not guarantee low trajectory or task error.
- Unpublished Z1 grid dimensions and core boundaries will not be guessed.

## Experiment requirements

Each meaningful run records at least:

```text
schema and experiment identifiers
backend and evidence class
random seed and key policy
canonical model and run hashes
package versions and upstream source commits
project git commit and dirty state
Python, JAX, jaxlib, platform, backend, and device
declared model dtype and JAX x64 configuration
warmup, sample definition, and schedule
synchronized compilation and execution timing
metrics with claim-level evidence
```

For small models, exact enumeration is a release gate. Larger studies require a
suitable conventional baseline and multiple seeds.

## Milestone 1 — Cross-layer stochastic-kernel benchmark

The milestone begins with one complete, CPU-reproducible vertical slice:

1. Run a Torx circuit through the exact state-vector simulator.
2. Run a matching small-model workflow through THRML locally.
3. Validate THRML output against exact Ising enumeration.
4. Emit strict JSON records that cannot self-label software as hardware.
5. Demonstrate the versioned Z1 cost projection without presenting it as a
   physical measurement.
6. Pass frozen-install, formatting, lint, test, and smoke gates in CI.

Later increments add compiled thermodynamic kernels, topology-constrained
simulation, the Extropic simulator API, and physical Z1 only when those paths
are genuinely available.
