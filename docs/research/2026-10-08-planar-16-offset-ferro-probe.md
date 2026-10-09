# Planar 16-offset ferro: probe for proposal P-0001 (exploration)

Exploration on 2026-10-08 for queue row `planar-16-offset-ferro`. Script:
[`planar_16_offset_probe.py`](planar_16_offset_probe.py); raw numbers:
[`2026-10-08-planar-16-offset-ferro-probe.json`](2026-10-08-planar-16-offset-ferro-probe.json).
**Not recorded evidence**: no archive, no replay, no gate; one coupling seed
(4500) and one graph seed (7) per family and size, 16 trials, fresh JAX root
20261012. It exists to find the fixed choice that bounds the metric and to
size the production run.

Sweeps are `software_simulation` (CPU, JAX float32). References are
`exact_reference` (Kac-Ward, float64). Nothing ran on hardware.

## Question

Do the planar Ising scaling allocations of
[PR #92](../experiment-reports/2026-10-07-planar-ising-scaling/findings.md)
hold on a planar subgraph of the Z1 16-offset lattice? In #92, past 64 spins
only tempering qualified on ferro grids (at 256 sweeps for L = 16 and 1024 for
L = 32), and five cold chains and the long chain never did within 4096 sweeps.

## The graph has to be chosen

The 16-offset rule ((1,0), (2,1), (2,3), (4,1) and rotations,
`Z1HardwareProfile.interior_offsets`) gives degree 16 and is not planar, so
"the planar subgraph" is not unique. Two facts fix what any choice can be:

- Every offset has odd parity, so every subgraph is bipartite under the
  (x + y) checkerboard and the two-colour sweep of #92 stays exact.
- A planar bipartite graph has at most 2N - 4 edges, so mean degree stays
  below 4. The square grid (the (1,0) offsets alone, 2N - 2L edges) is already
  within 2L of that bound. No planar subgraph is denser than the grid in any
  useful sense; what the hardware shape can change is *which* edges, not how
  many.

The probe builds maximal straight-line planar subgraphs on an L x L patch by
greedy insertion: candidate 16-offset edges in a seeded order, each kept if
its segment crosses no kept segment.

| Graph, L = 32 | edges | mean degree | isolated sites | (0,1) | (1,2) | (2,3) | (1,4) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `grid` (control, #92's graph) | 1984 | 3.88 | 0 | 1984 | 0 | 0 | 0 |
| `greedy-random` (seeded order) | 1806 | 3.53 | 13 | 1259 | 346 | 98 | 103 |
| `greedy-long` (longest offsets first) | 1982 | 3.87 | 0 | 1149 | 252 | 1 | 580 |

`greedy-long` keeps nearly the grid's edge count with 42 percent long edges.
`greedy-random` loses 9 percent of edges and leaves 13 sites isolated.

## Checks

- The study-local Kac-Ward on an explicit embedding equals the imported
  `planar_ising_scaling.kac_ward` bitwise on 3 x 3 and 8 x 8 grids, and
  matches brute force on 24 small patches of both greedy families (3 to 5
  non-grid edges each, ferro beta 4 and mixed beta 1) to 1.4e-14 in ln Z and
  4.2e-15 per edge.
- Finite-difference precision of the weakest two correlations: at most
  2.3e-9 on every probe target.
- The general neighbour-list sampler reproduces the study sampler's window
  sums and exchange flags bitwise on an 8 x 8 grid (five-replica tempering,
  256 sweeps).
- On a 4 x 4 greedy patch, independent and tempering5-k1 reach brute force
  within 7e-6 mean edge error at 65,536 sweeps.

## Result

Qualifying budget (mean trial edge MAE <= 0.05, sustained), one seed:

| Target | long | independent | t5-k1 | t5-k4 | t9-k1 | t9-k4 |
| --- | --- | --- | --- | --- | --- | --- |
| L8 grid | 1024 | 4096 | 64 | 64 | 64 | 64 |
| L8 greedy-random | 4096 | 4096 | 64 | 64 | 64 | 64 |
| L8 greedy-long | 1024 | - | 64 | 64 | 64 | 64 |
| L16 grid | - | - | 1024 | 1024 | 256 | 256 |
| L16 greedy-random | - | - | 256 | 256 | 256 | 256 |
| L16 greedy-long | - | - | 256 | 1024 | 256 | 256 |
| L32 grid | - | - | 1024 | 4096 | 1024 | 1024 |
| L32 greedy-random | - | - | **4096** | **4096** | 1024 | 1024 |
| L32 greedy-long | - | - | **4096** | **4096** | 1024 | 1024 |

Edge MAE at 4096 sweeps, L = 32: grid 0.0156 / 0.0278 / 0.0048 / 0.0055
(t5-k1, t5-k4, t9-k1, t9-k4); greedy-random 0.041 / 0.045 / 0.019 / 0.025;
greedy-long 0.036 / 0.044 / 0.021 / 0.021. Long and independent sit at 0.07
to 0.10 on every L = 32 graph.

Five-replica cold-pair acceptance at L = 32 and 4096 sweeps: grid 0.015,
greedy-random 0.002 to 0.005, greedy-long 0.003 to 0.008.

1. **The ordering holds.** At L = 16 and 32, on both hardware-shaped
   families, only tempering qualifies within 4096 sweeps; long and five cold
   chains never do, as in #92.
2. **The nine-replica budgets hold.** 256 sweeps at L = 16 and 1024 at L = 32
   on every graph.
3. **The thin five-replica ladder slips one budget step at L = 32.** On both
   greedy families it qualifies only at 4096, at 0.036 to 0.045 against the
   0.05 threshold, because its cold-pair acceptance drops three- to
   seven-fold relative to the grid. With one seed and a fourfold budget grid
   this is the arm whose verdict a recorded run could flip.
4. **L = 8 is not a clean copy of #92.** On `greedy-long` five cold chains
   miss 0.05 at 4096 (0.073); in #92 they qualified on all three L = 8 grids.
   One seed; noted, not gating.

## Fixed choices and what they bound

- **Thin five-replica ladder** (named in the row). Bounds the five-replica
  arms at L = 32: they qualify at the last tested budget, near the threshold.
  This is the choice that sets the metric there.
- **Budget cap of 4096 sweeps** (inherited). Censors the five-replica arms at
  L = 32 on the hardware-shaped graphs; a recorded run needs 16384 so that a
  slip reads as "16384", not "not reached".
- **Planar subgraph selection.** Bounds what "hardware-shaped" can mean:
  mean degree below 4 for every planar choice, so the study tests long planar
  edges, not the chip's degree 16. Exact references for the non-planar
  lattice are out of reach of Kac-Ward; this is a limit of the row's target
  family, not of the probe.
- **beta = 4, weights {1..5}/5, ferro gauge** (inherited). Ordered regime.
  `greedy-random` has weaker correlations (min |corr| 0.66 against 0.98 on
  the grid at L = 32); not a bound.
- **Sizes L <= 32**, open patch, interior rule only. The 2E x 2E Kac-Ward
  matrix takes about 25 s at L = 32; larger patches cost minutes each.
- **Threshold 0.05, 16 trials, quarter burn-in, k in {1, 4}** (inherited).
  Not bounds.

## Sizing

Per L = 32 target, all six arms to 4096 sweeps: about 220 s wall including
compilation (long chain 110 s, the rest 12 to 28 s each) plus a 25 s
reference. To 16384 sweeps: about 900 s. L = 16 about 260 s and L = 8 about
80 s at 16384. For 18 targets (two graph families, three seeds, three sizes):
about 2.1 h wall in one process, about 4 CPU-hours (XLA used about two cores
per process here).

Archive: the #92 ferro window sums took 1.65 MB deflated for nine targets at
five budgets, 0.44 MB per L = 32 target. At 16384 sweeps an int16 window sum
overflows (independent and long windows hold 61,440 products), so a recorded
run must store counts as uint16, and per-sweep exchange flags would be 4x
larger than in #92; per-budget accepted counts are enough for pricing and
acceptance. With those two changes 18 targets come to about 3.5 MB.

## Probe cost

About 16 minutes wall over three invocations (68 s for L = 8, 7 minutes for
L = 16 and the first L = 32 target before the shell timed out and killed the
process, 8 minutes for the remaining L = 32 targets on resume). Only the
first invocation was timed for CPU (139 s user for 68 s wall); at that ratio
the probe used about 30 CPU-minutes, at the allowance, about 6 of them in the
killed invocation. The script was reformatted by `ruff format` after the run;
no logic changed.
