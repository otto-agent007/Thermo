# Planar 16-offset ferro: #92's allocations on a planar subgraph of the Z1 lattice

Recorded 2026-10-08 from `evidence.tar.gz` (SHA-256 in `manifest.json`).
Protocol: [`docs/experiments/planar-16-offset-ferro.md`](../../experiments/planar-16-offset-ferro.md)
(proposal P-0001, approved 2026-10-08, unchanged since commit 46f3261).
Probe: [`2026-10-08-planar-16-offset-ferro-probe.md`](../../research/2026-10-08-planar-16-offset-ferro-probe.md).
Replay: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false uv run python -m thermo_lab.planar_16_offset_ferro --output-dir <extracted> --replay`.

**Evidence classes.** Sweeps are `software_simulation` (CPU, JAX float32,
two-colour block Gibbs). References are `exact_reference`: the Kac-Ward
determinant on an explicit straight-line embedding, float64. Z1 energies and
sweep times are `calibrated_projection` from the sealed `Z1HardwareProfile`,
which excludes host latency. Nothing ran on hardware. The graph is a synthetic
planar patch of the published 16-offset rule, not the physical chip map
(`exact_physical_graph_available = false`).

## Question

Do the planar Ising scaling allocations of
[PR #92](../2026-10-07-planar-ising-scaling/findings.md) hold on a planar
subgraph of the Z1 16-offset lattice? The 16-offset rule is degree 16 and not
planar. The study uses `greedy-long`, a maximal straight-line planar subgraph
(longest offsets first, ties in a seeded order). At L = 32 it keeps 1976 to
1984 edges, about 42 percent of them long ((1,2) and (1,4), a few (2,3)). It
is compared with the paired open grid under the same coupling seeds (4510 to
4512). Zero field, ferro, beta = 4, six arms at equal elapsed sweeps, budgets
64 to 16384, 16 trials, threshold 0.05 mean edge MAE sustained.

## Result

**#92 holds on the 16-offset subgraph.** Both clauses of the pre-registered
rule pass:

- (a) At L = 16 and 32 on `greedy-long`, the long chain and five cold
  chains qualify on no target within 4096 sweeps, and every tempering arm
  qualifies on 3/3 targets.
- (b) Both nine-replica arms are `holds` at L = 16 and 32. They qualify at
  256 and 1024 sweeps on every grid and every `greedy-long` target.

**Pre-registered expectation 3 failed.** The probe predicted a one-step
five-replica slip at L = 32. It did not reproduce: `tempering5-k1` is
`holds` (1024 on all six targets) and `tempering5-k4` is `mixed`, with one
slip and one improvement against the grid. The thin ladder still degrades on
`greedy-long`, but at L = 32 the effect sits inside one fourfold budget step:

- Its cold-pair acceptance falls more than tenfold. At 4096 sweeps it is 0.003
  against 0.035 to 0.049 on the grid, or 11 to 14 accepted swaps per trial
  against 142 to 200.
- Its seed-mean error at the 16384 horizon is about five times the grid's
  (0.016 to 0.019 against 0.003 to 0.004).
- `tempering5-k1` qualifies at 1024 with no margin. Mean MAE there is 0.038
  to 0.050 on `greedy-long` (seed 4512: 0.0497) and 0.034 to 0.049 on the
  grid.

A rerun with other seeds could flip it, as the probe's seed did.

## Qualifying budgets (grid / greedy-long, seeds 4510, 4511, 4512)

`-` = not reached by 16384.

| Arm, L | 8 | 16 | 32 | class L16 / L32 |
| --- | --- | --- | --- | --- |
| long | 4096/4096, 1024/4096, 4096/256 | 16384/16384, 16384/-, 4096/- | -/-, 16384/-, -/- | slips / holds |
| independent | 16384/-, 4096/16384, 4096/4096 | all - | all - | holds / holds |
| tempering5-k1 | 64 everywhere | 256/256, 1024/256, 256/256 | 1024 everywhere | holds / holds |
| tempering5-k4 | 256/64, 64/64, 64/64 | 1024/1024, 256/256, 256/256 | 1024/1024, 4096/1024, 1024/4096 | holds / mixed |
| tempering9-k1 | 64 everywhere | 256 everywhere | 1024 everywhere | holds / holds |
| tempering9-k4 | 64 everywhere | 256 everywhere | 1024 everywhere | holds / holds |

Rows for long, independent and L = 8 are descriptive. They do not gate.

- The long chain at L = 16 qualifies at 16384 on 3/3 grids but on only 1/3
  `greedy-long` targets. The long edges do not shorten ordinary-chain mixing
  here; if anything, they lengthen it.
- At L = 8, five cold chains fail to reach 0.05 on one `greedy-long` target
  (seed 4510, 0.098 at 4096 and above 0.05 at 16384). On the grid they reach
  it on 3/3. Expectation 4 is confirmed, on one of three seeds.
- Against #92's archived grids (seeds 400 to 402, horizon 4096), the paired
  grids here give the same nine-replica budgets (64, 256, 1024) and the same
  modal five-replica budgets. For `tempering5-k4` at L = 32, #92 had 4096 on
  2/3 seeds; here it is 4096 on 1/3.

## Exchange acceptance (mean per pair slot, T = 4096)

| | five-replica ladder | nine-replica ladder |
| --- | --- | --- |
| L16 grid (k = 1) | 0.002 to 0.004, 0.24 to 0.31 | - |
| L16 greedy-long (k = 1) | 0.002 to 0.003, 0.09 to 0.13 | - |
| L32 grid (k = 1) | 0.000, 0.035 to 0.049 | 0.011 to 0.013, 0.000, 0.009 to 0.011, 0.20 to 0.33 |
| L32 greedy-long (k = 1) | 0.000, 0.003 | 0.011, 0.000, 0.005 to 0.006, 0.056 to 0.088 |

The cold-end pair loses most of its acceptance on `greedy-long` at every
size. This is the same mechanism the probe found. At this seed set, the
qualification metric absorbs it at L = 32 within one budget step.

## Projected Z1 cost (secondary, `calibrated_projection`)

At L = 32 the qualifying tempering cells cost 3.6 to 53.6 uJ per trial under
the published convention. `greedy-long` comes out slightly cheaper than the
grid at the same budget (for example `tempering5-k4` at 1024: 3.6 to 3.7 uJ
against 4.2 to 4.5 uJ). The reason is that it accepts fewer swaps, so the
model prices fewer writes. #92 already warned about this: pricing exchanges
per accepted swap makes a broken ladder look economical. The cost model
excludes host latency.

## Checks (before any sampling)

- **Kac-Ward against the imported #92 function.** On 3 x 3 and 8 x 8 grids
  the study-local Kac-Ward is bitwise equal to the imported one: matrix, ln Z
  and every edge.
- **Kac-Ward against brute force.** On 24 patches of up to 16 sites with 2
  to 5 non-grid edges each (ferro at beta 4, mixed at beta 1), the largest
  error is 2.8e-14 in ln Z and 4.6e-15 per edge.
- **Finite differences on every target.** The precision of the two weakest
  correlations is at most 7.2e-9, against a limit of 1e-4. On grids with
  L <= 16, the transfer-matrix ln Z agrees to a relative 1e-8.
- **Stationarity.** One colour-A-then-colour-B sweep is exactly stationary on
  2 x 5 and 3 x 3 `greedy-long` patches (residual 2.4e-17).
- **Sampler equivalence.** The neighbour-list sampler reproduces
  `planar_ising_scaling.compile_sampler` bitwise on an 8 x 8 grid for all six
  arms: snapshot sums and exchange flags.
- **Empirical check.** On a 4 x 4 `greedy-long` patch at 65,536 sweeps,
  `tempering5-k1` comes within 1.5e-4 mean edge error of brute force, against
  a tolerance of 0.01.
- **Graph rebuild.** Every graph rebuilds from its seed on replay.

## Run

- Runner commit 3963ecd, clean tree (`git_dirty=false` in the generating
  attempt's provenance).
- One attempt, no resume. Local i7-7700K, CPU only, two workers.
- Times: checks and references 3 min; 108 units in 56 min; replay 3 min. The
  total is 62 min wall, against the protocol's estimate of about 1.1 h.
- `counts.npz` is 3.3 MB (uint16 counts) and the archive is 3,807,124 bytes
  gzipped.

## What this changes

The planar-family result of #92 carries over to a planar arrangement of the
chip's long edges, on ferro targets at beta = 4. Tempering is required past
64 spins, and the nine-replica ladder keeps #92's budgets. The thin
five-replica ladder is fragile on hardware-shaped graphs. It loses more than
a factor of ten in cold-end acceptance at L = 32 and qualifies there with no
margin, so #92's advice stands: do not reuse it above 64 spins. None of this
says anything about the non-planar degree-16 lattice, frustrated targets or
the physical chip.
