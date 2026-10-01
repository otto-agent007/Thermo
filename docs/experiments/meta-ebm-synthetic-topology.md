# M5c: degree repair and placement on a synthetic offset lattice

*Status: **draft protocol for owner review**, September 30, 2026. Nothing
here is frozen until the owner approves it. It records the choices settled by
the M5c explorations: the
[input model and repair probe](../research/2026-09-28-m5c-input-model-probe.md),
the [bounded degree refit](../research/2026-09-29-m5c-degree-refit-probe.md)
and the [synthetic-patch placement probe](../research/2026-09-30-m5c-placement-probe.md).
Those notes are exploration; this study regenerates their fits and
placements as recorded evidence and does not cite their endpoints.*

## Question and scope

Can the primary M5 kernels be compiled onto the published local connection
rule without approximation, and what does that cost? Specifically:

1. How much accuracy does the minimal degree repair cost, measured on the
   exact outer chain at finite inner budgets?
2. How many clamped input copies, physical p-bits, clamp writes and lattice
   sites does an exact placement need, against the parity lower bound?

All results are CPU float64 `exact_reference` calculations of declared
models, with zero generated samples. Placement counts are algorithmic counts
on a **synthetic lattice**: the published local rule on an open square, not a
physical-chip graph, core map or placement API. No latency, energy,
reprogramming or hardware claim belongs in M5c.

## Sources

Pin the compressed M5b archive SHA-256
`4e5592572f078f1d36ff89928d145be62588ed5fb8c2adc842a403f278c9c674`, request
digest `sha256:36836ed0c15023f2261c98c930783065ab5a662c760d617231928c6a6399bb8d`
and result digest
`sha256:bf4771cfa760e1b099e7726490cef240d144493458f7f2c69d59386d224d7c0b`.
It holds the unchanged M5a parameter vectors and the M5b finite-K reference
metrics. Authenticate its five pinned implementation hashes
(`hashing.py`, `persistence.py`, `meta_ebm_cap_baseline.py`,
`meta_ebm_thermalization.py`, `meta_ebm_thermalization_core.py`) and import
from them; never edit them. New behavior goes in a study-local module.

The lattice rule is from *Thermalizing Stochastic Programs*,
arXiv:2608.01615v2 (revised August 13, 2026), §II.2.1: offsets `(1,0)`,
`(2,1)`, `(2,3)`, `(4,1)` and their four 90° rotations, 16 neighbors at an
interior node, bipartite by checkerboard color.

## Fixed grid

| Choice | Frozen choice |
|---|---|
| Arm | Reading B, variational, cap 1, all five seeds, 12 sites: 60 kernels. This is M5b's primary arm and the only arm refit or placed. |
| Input model | Clamped input copies, as approved on September 28 |
| Degree repair | J-only mask on each kernel with output degree above 16 (11 kernels), then a bounded refit. The other 49 vectors are used bitwise unchanged. |
| Methods compared | Original (M5b replay), unfitted J-only prune (control), J-only refit |
| Inner K | 4 (primary), 32, and the exact-marginal limit |
| Outer chain | M5b's sweep, sites 0–11 in order, uniform start over 4,096 states, t = 0–30 |
| Placement | Minimum-copy exact integer program per kernel; greedy resident packing per seed |
| Inference | Descriptive; every seed reported with min/median/max across seeds |

That is 15 original replays (five seeds, three K values), 30 new outer-chain
cells (J-only prune and refit, five seeds, three K values), 11 refits and 60
placements.
Additional arms, caps, masks, K values or placement models require a dated
amendment.

## Degree repair

For each kernel whose count of nonzero `J` plus nonzero `beta` exceeds 16,
zero exactly `degree - 16` nonzero `J` coefficients with the smallest
absolute archived value, ties by `J` index. Never mask `beta`; keep the hidden
count, cap and M5a uniform-input KL objective.

Refit with L-BFGS-B and M5a's analytic gradient: bounds `(0, 0)` on masked
coordinates and `(-1, 1)` elsewhere, M5a's `ftol` and `gtol`, **`maxiter`
20,000 and `maxfun` 200,000**. Starts are M5a's eight starts for that
kernel, masked, plus the masked archived vector: nine per kernel, 99 in all.
Select the minimum endpoint objective, ties by start index. Record every
start's initial and final objective, iterations, evaluations, success flag
and termination message; convergence is reported, not required. Outer metrics
never select starts. The unfitted J-only prune is the masked archived vector
without refitting.

Reference bars, fixed before the exploration's fit: at K = 4, five-seed median
stationary target bias and median target TV30 below **8.3e-3** (M5b's 8-bit
rounding bias for this arm) and below **5e-4** (its 12-bit value). They are
scientific reference points, not integrity gates.

## Placement

Placement depends only on each kernel's nonzero pattern, read from the
**recorded** refit and unchanged vectors.

- **Free p-bits:** output `y` and each hidden spin take one site each. No
  chains, routing nodes or split outputs. Each nonzero `beta` must be a
  lattice edge, so hidden spins sit on neighbors of `y`.
- **Copies:** each clamped copy site holds one blanket input. A nonzero `J`
  input gets one copy on a free neighbor of `y`, filled in offset order.
  `A` copies take the other color. For each nonzero `A[a, i]` exactly one
  adjacent copy carries the coefficient; every other lattice coupler is zero.
  Copies are shared only within a kernel.
- **Objective:** minimize copies with SciPy `milp` (HiGHS) on the unbounded
  lattice, `y` at the origin, time limit **600 s** per kernel. In the
  exploration all 60 kernels were proven optimal within 200 s at a longer
  limit, so 600 s leaves about 3× margin. Assert that
  all hidden rows share one input support, so only the set of hidden offsets
  matters. Record status, incumbent, dual bound and gap. A time-limited
  incumbent is a valid placement with a solver-reported bound, not a proven
  minimum.
- **Resident patch:** per seed, pack all 12 kernels into an open square of
  side `L`, disjoint site sets, no cross-kernel couplers, only the updated
  kernel sampled. Allow 90° rotations and translations. Greedy first-fit:
  largest footprint first (ties by site), anchors row-major, rotations 0–3,
  `L` from `ceil(sqrt(total sites))` upward. `L` is a heuristic upper bound.
- **Time-multiplexed alternative:** report the largest single-kernel bounding
  box. Reprogramming cost is not estimated.

## Required measurements

Per refit kernel: the start records above, selected objective, output
degree, local mixing (M5b's lambda summary and the K for TV ≤ 1e-3 and 1e-6).

Per outer-chain cell: stationary TV bias against the target, TV to target and
to the method's own stationary law at every t = 0–30, stationarity residual,
and work per outer sweep: `K · 72` spin redraws, 72 reset writes, 12 readout
bits and clamp writes. Report clamp writes both at the parity bound and at
the placed copy count.

Per placed kernel: blanket size, hidden count, output degree, `J` and `A`
copies, parity-bound copies, physical p-bits, footprint, bounding box, solver
status and bound. Per seed: free and physical p-bits, copies against the
parity bound, `L`, utilization and the largest single-kernel box.

Topology violations must be zero and there is no unresolved placement
residual by construction; the record states both explicitly. Energy and
modeled time are **excluded**: no calibrated clamp-write, read or sampling
cost exists for this lattice.

## Integrity and replay

1. Authenticate the archive and the five pinned implementation hashes.
   Replay all 15 original chains at the three K values against M5b's
   archived metrics (absolute tolerance 1e-12).
2. Check every mask, the `(0, 0)` bounds on masked coordinates, output degree
   ≤ 16, nonzero retained `beta`, unchanged hidden counts and the 49 unchanged
   vectors bitwise.
3. Check each refit kernel's marginalized conditional against brute-force
   hidden enumeration to 1e-12.
4. **Placement:** rebuild every placed kernel as a site-level coupling map.
   Every coupled pair must be a lattice edge of opposite color, every
   coefficient must sit on exactly one edge, and all sites must be distinct.
   Enumerate the free p-bits for every input row; the placed conditional must
   match the kernel's marginalized conditional to 1e-12. Repeat after
   packing, in patch coordinates, with the finite patch's neighbor sets.
5. Check the outer stationary residual (≤ 1e-10) and row sums.
6. **Persisted replay:** from the stored request, parameters and layouts,
   recompute every outer metric and rerun the placement checks before
   writing `completion.json`. Replay never refits and never re-solves an
   integer program; solver status and bounds are replayed as recorded.

A small fixture (one kernel's placement check and one outer chain) joins the
existing `unit-rest` tests. A unit test pins the request and replays a bounded
subset of the archived record; the full study stays a local gate. No new
workflow or preflight layer.

## Runtime

From the explorations on this 8-CPU host: refits about 10 minutes with three
workers; placements about 36 minutes of solver time in total, 19 of them
in the six hardest kernels at 180–200 s each; outer chains and replay
a few minutes. Run refits on three workers and placements on up to six,
one BLAS thread each; the placement problems are small in memory. The
expected total is **under 30 minutes**, so no autosave layer is planned. Measure one seed before the full
run; if the estimate exceeds 30 minutes, amend the gate to add the standard
autosave and resume contract before launching.

Store a bounded gzip record (parameters, layouts, metrics and summaries),
summary, provenance and `completion.json`. Completion requires 15 replayed
originals, 11 refits, 30 new outer cells, 60 verified placements, five
packed patches, zero samples, zero topology violations and full persisted
replay.

The entry point will be `python -m thermo_lab.meta_ebm_topology`, output
`results/meta-ebm-topology`, marker `meta_ebm_topology_complete`. The release
gate section is added with the runner.

## Decisions for review

1. **Primary arm only.** Reading A variational cap 3 shares the dense
   structure and would likely cost similarly; placing it is an amendment.
2. **Methods:** original, J-only prune and refit. The historical mixed prune
   and the capped chain stay in the exploration record.
3. **Solver limit: 600 s per kernel** (raised from 120 s on September 30).
   At 120 s six kernels stopped 2 copies above their bound. Re-solved at a
   900 s limit, all six were proven optimal in 182–200 s with the same
   copy counts, so the bound was the weak part, not the layouts.
4. **No autosave**, conditional on the one-seed timing staying under the
   30-minute threshold with placements on six workers.

## After M5c

M5c closes the M5 milestone's topology stage. Sparse-input training and
cross-kernel copy sharing are named levers on copy cost, not planned work;
either needs owner approval. The next milestone should come from the charter
tracks not yet started.
