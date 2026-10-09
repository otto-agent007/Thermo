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

## Amendments

### Amendment 1 (2026-10-08, draft for owner approval): constraints and tasks before samplers

**Status.** Proposed by the October 8 review of the native-sampling line.
It takes effect when the owner merges it. Nothing in it changes the research
rules, evidence classes, replay requirements or pinned-source discipline
above; it changes which questions the lab asks first and how much process a
question earns.

#### Why

Between October 1 and October 8 the lab recorded ten native-sampling
studies: fixed-budget sampling, time to accuracy, symmetry plus tempering,
changing evidence, conditional estimation, two Potts stages, the exchange
cost projection, planar Ising scaling and planar annealing. Each is sound,
replayable and labelled, and together they say: tempering helps on trapping
targets and its exchange acceptance collapses with size unless the ladder
scales; symmetry estimators are free at zero field; annealing on frustrated
2D grids approaches the ground state slowly and parallel restarts give a
best-of; an equilibrium sampler reports its equilibrium energy, not a ground
state; exact 2D ground states are tractable. Those are results from the MCMC
and spin-glass literature of the 1990s. The lab built an expensive instrument
and used it to confirm them on random instances.

Two things explain the drift, and they are related.

1. **The hardware's constraints were not in the questions.** What separates
   a TSU from a GPU running the same Gibbs sampler is a fixed sparse topology
   (the 16-offset lattice), low coupling and bias precision, per-site noise
   and temperature non-uniformity, cheap updates against expensive reads,
   writes and clamps, chromatic block schedules and no global replica
   exchange. Almost every sampler study assumed the opposite: free exchange,
   free temperature schedules, float32 couplings, arbitrary graphs. The one
   study that priced a constraint (the exchange cost projection) overturned
   the "default to tempering" recommendation on energy at once. The useful
   results come from the constraints, not from the choice of sampler.
2. **There was no task.** Every target was synthetic (random grids, random
   cubic graphs, random stored patterns) and every threshold was round
   (0.05 total variation, 1e-3 per spin). Goals 3 and 5 above ask for
   task-level error and for designs built around the cost of sampling,
   clamping, reading, writing and host round trips. The lab has measured
   kernel and trajectory error only, and its cost accounting is one line
   per cell. Without a task there is no way to say whether a 5e-3 gap is a
   failure or irrelevant.

The fixed cost of a recorded study (frozen protocol, runner, replay test,
gate, four document updates, archive) is high enough that it biases the lab
toward questions that are safe to answer over questions that matter, and a
scheduled research loop will amplify that bias unless the queue is curated
against it.

#### What changes

**The primary question.** The lab's standing question becomes: *what can a
fixed-topology, low-precision, I/O-expensive p-bit array do usefully, at
what task quality and at what projected cost, compared with a conventional
machine doing the same task?* "Which sampler is best on a random instance"
is no longer a primary question; it is asked only when a task raises it.

**Priority order of work**, replacing the open-ended track list for the next
cycle:

1. **A hardware-constraint layer** (`thermo_lab.constraints`, or a name the
   implementer prefers): one module that takes any pairwise model and applies
   the chip's rules as declared, hashed inputs: placement on the 16-offset
   lattice (reuse `meta_ebm_topology`'s offsets and placement), coupling and
   bias quantization to *n* bits (reuse M5b's `round_parameters` codebook),
   per-site beta jitter, chromatic block Gibbs only, no replica exchange,
   and separate counts of updates, reads, writes and clamp changes feeding
   the existing `Z1HardwareProfile`. It is a transformation plus an
   accounting record, not a new runner or daemon. Exact references and
   pricing stay as they are. Every study after it runs under the layer by
   default and says which constraints it relaxed.
2. **One task the hardware is meant for.** Extropic's stated application is
   denoising by chained discrete energy-based models
   ([source card](docs/knowledge/sources/extropic-dtm-hardware.md)). The
   first task-level benchmark is a small denoising model on a small binary
   image set (a binarized digit or clothing set at 8 x 8 to 16 x 16, or a
   synthetic set with an exact likelihood where one exists), trained in
   software and run under the constraint layer, with task metrics
   (reconstruction error, held-out log-likelihood where tractable, sample
   quality) reported against bits, jitter, sweeps and I/O budget. Its
   outputs are specification numbers: the precision and noise tolerance the
   task needs, the sweeps per sample, the reads and writes per sample.
   Associative-memory recall is the second task, and the queue row
   `am-coupling-bits` is its first constraint question; it runs before any
   further sampler comparison.
3. **A conventional baseline that answers the question.** The same task on
   CPU or GPU with the same model, and a small conventional model of similar
   quality, so a claim reads "at this task quality, projected energy A
   against measured B" with every assumption attached. A projected number
   without this baseline is not a result.
4. **The Thermalizers and meta-EBM line (track C) stays small.** It
   reconstructs an unpublished specification from a paper appendix; keep it
   to contract checks (E0-style) and placement, and do not build task work
   on it until Extropic publishes the implementation.
5. **Sampler benchmarks on random instances are demoted** to probes. The
   open follow-ups of the planar annealing study (schedule shape, a planar
   matching solver for L = 32, an N-scaled cold ladder) stay recorded as
   notes and are not scheduled.

**Evidence tiers.** Two tiers, chosen before any code is written:

- A *probe* runs in minutes, lives under `docs/research/` with its script
  and a dated note, is labelled exploration and is never cited as evidence.
  It is the default for any question whose answer would not change a design
  decision. Most sampler questions are probes.
- A *recorded study* earns the full treatment (frozen protocol, runner,
  replay, gate, report, roadmap row) only when its result is a number
  someone will design against: a precision requirement, a noise tolerance,
  a sweep or I/O budget, a task quality at a projected cost, or a contract
  between an upstream library and an exact reference. A study that cannot
  name that number in its protocol's first paragraph is a probe.

**Standing rules added to the research rules above:**

8. Run every model under the declared hardware constraints by default, and
   state which constraints a study relaxed and why. A result that needs a
   feature the hardware does not have (free replica exchange, a free
   per-replica temperature schedule, float32 couplings) says so in its
   headline.
9. State the resource accounting with every comparison: elapsed sweeps,
   p-bit updates, projected energy or host round trips. A method that wins
   on one and loses on another is reported on both.
10. Where a task exists, report task-level error alongside kernel and
    trajectory error, and report it per unit of projected cost against the
    conventional baseline.

#### Proposed research queue (for the owner to adopt into `docs/research-queue.md`)

The loop takes rows in order; this is the order the amendment asks for.

| Row | Question | Metric |
| --- | --- | --- |
| `constraint-layer` | Does the constraint layer reproduce the archived M5c placement, M5b rounding and E0 contract bitwise, and does it price updates, reads, writes and clamps separately on the existing Z1 profile? | Bitwise agreement with the three archives; one priced example per constraint |
| `am-coupling-bits` | (existing row) How does associative-memory recall fall with coupling precision and per-site beta noise? | Recall against bits and jitter, with stage A's exact references |
| `dtm-small-image` | At what task quality does a small chained-EBM denoiser run under the constraint layer, as a function of bits, jitter, sweeps and I/O, and what does it cost against CPU Gibbs and a small conventional model of the same quality? | Held-out reconstruction error and log-likelihood where tractable, per projected joule and per measured CPU second |
| `planar-16-offset-ferro` | (existing row, moved down) Do the planar scaling allocations hold on the hardware-shaped graph? | Which arms qualify, at what budget |
| `e0-remaining-kernels` | (existing row, unchanged) | |

#### How to tell whether this is working

After four weeks under the amendment the lab should be able to state, with
evidence classes attached: the coupling bits and beta-noise tolerance one
task needs; the sweeps, reads and writes per useful sample for that task on
the constrained machine; and the projected energy of that task at a stated
quality against a conventional baseline. If it cannot, the amendment has
not fixed the problem and the owner should revisit it rather than add
process.

#### What does not change

Evidence classes and their separation, exact references as release gates
for small models, hashed canonical inputs, pinned 0.x upstreams, replay
before reporting, the autosave contract for long runs, the lessons
register, and the rule that two consecutive failures of the same kind stop
and ask the owner.
