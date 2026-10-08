# Planar Ising scaling: exploratory protocol

Frozen on 2026-10-07 before production sampling. Every recorded sampling study
sits inside exact enumeration (at most 16 free spins), where a chain crosses
the whole state space in a handful of sweeps. This protocol takes the same
allocation question to 64, 256 and 1024 spins with an exact reference, so the
sampler is measured where sampling is the only practical route to the answer.

## Fixed choices and what they bound

- **Graphs.** Open L x L square grids, L = 8, 16, 32, zero field. Planarity is
  what makes the reference exact; the grid is bipartite, so the two-colour
  block-Gibbs sweep of the Z1 Appendix-B model is exact here. The grid is not
  the Z1 16-offset lattice; embedding is not modelled.
- **Couplings.** Edge weights w in {1..5}/5 as in the archived studies. Two
  variants per size and seed: `ferro` (all couplings -w/5, the archived sign
  convention, gauge-equivalent to a ferromagnet on a bipartite graph) and
  `mixed` (random sign), a +-J-type spin glass. Seeds 400, 401, 402 fold the
  size into the generator. Eighteen targets. No target selection after seeing
  errors.
- **Temperature.** Cold beta = 4, as archived. With |K| >= 0.8 on every edge
  this is deep in the ordered (ferro) or frozen (mixed) regime. The 5-minute
  probe before freezing showed that on `L32-mixed` no arm comes within a factor
  of three of the threshold at 4096 sweeps; that negative is pre-registered
  here and does not gate. The `ferro` line already separates the arms at L = 32.
- **Ladders.** The archived five-replica ladder [0.25, 0.5, 1, 2, 4] is kept
  unchanged, and one denser nine-replica geometric ladder from 0.25 to 4 is
  added. Neither is scaled with system size, so exchange acceptance is expected
  to collapse at L = 32 (the probe showed about zero for most pairs). That
  collapse is a result of the study, not a tuning target; an N-scaled ladder is
  a follow-up for the owner, not a third arm added here.
- **Exchange interval.** 1 and 4 sweeps (the exchange cost projection found
  k = 4 free at n <= 16).
- **Arms, at equal elapsed sweeps.** `long` (one chain, 5T sweeps),
  `independent` (five cold chains, T sweeps), `tempering5-k1`, `tempering5-k4`,
  `tempering9-k1`, `tempering9-k4` (T sweeps each). All non-long arms share
  wall-clock on a parallel array; `long` is the serial baseline.
- **Metrics.** Edge correlations are invariant under the global flip, so the
  archived analytic symmetry estimator is the identity here and is not an arm.
  Per trial: mean absolute edge-correlation error over all edges, the maximum
  edge error, and the error in the coupling-weighted energy per spin. The
  four-spin joint histogram of the archived studies has no exact reference at
  L = 32 and is dropped at every size.
- **Qualification.** Mean trial edge MAE <= 0.05, sustained at all later
  tested budgets, as archived. Budgets T = 16, 64, 256, 1024, 4096, 16 trials,
  quarter burn-in, fresh JAX root 20261009, one initialization shared by all
  arms per trial.
- **Cost model.** Every cell is priced in the sealed Z1 profile under the
  `published` and `beta_knob` conventions of the exchange cost projection,
  with R x N physical p-bits for R replicas.

## Exact reference

Kac-Ward determinant for the planar zero-field Ising model: ln Z and every edge
correlation (as the derivative of ln Z with respect to that edge's coupling)
from one complex matrix of size 2E. Before any sampling, the reference must
match brute-force enumeration to 1e-9 on 2 x 2, 3 x 3 and 4 x 4 grids of both
variants, and ln Z must match an independent transfer-matrix computation at
L = 8 and L = 16 to a relative 1e-8. The correlations need the inverse's
diagonal, which is ill-conditioned at beta = 4 on frustrated grids, so every
reference also measures its two weakest correlations against central
differences of ln Z (no inverse) and records the largest error; it must be
below 1e-4 (it is about 1e-9 on ferro grids and 1e-5 on mixed grids at
L = 32, three orders below the threshold). The 32 x 32 reference takes about
40 seconds. These references are `exact_reference` with that recorded
precision. Replay is bitwise under the gate's single-thread BLAS settings;
other thread counts change the ill-conditioned inverse at the 1e-5 level.

## Kernel checks

Before any sampling, the colour-A-then-colour-B sweep must preserve the
Boltzmann distribution exactly on a 2 x 3 grid (stationarity residual
<= 1e-12), and the compiled five-replica tempering sampler must come within
0.01 mean absolute edge error of brute force on a 4 x 4 mixed-sign grid after
65,536 sweeps (it reaches about 0.005; the error falls roughly as the inverse
square root of the horizon). The exchange acceptance uses the archived sign
convention and detailed-balance check.

## Persisted evidence and replay

The sampler keeps exact int32 prefix sums of the retained replicas' edge
products inside the scan and emits them at the window boundaries of each
budget, so no trace is stored. Persist, per target and arm: the exact int16
window sums (budgets x trials x edges) and the packed exchange flags. Replay
authenticates sources, request and the sums, recomputes every reference,
kernel check and the empirical check, then recomputes estimates, errors,
pricing, qualification decisions and summaries with tolerance 2e-12 relative.
It does not regenerate sweeps. Write completion last. The archive is expected
to be a few megabytes gzipped; report its size in the findings.

Sweeps are `software_simulation`; references `exact_reference`; energies and
sweep times `calibrated_projection` with the profile's exclusions. Expected
execution is 15 to 25 minutes on two CPU cores, under the checkpoint threshold;
restart an interrupted run in a fresh directory.
