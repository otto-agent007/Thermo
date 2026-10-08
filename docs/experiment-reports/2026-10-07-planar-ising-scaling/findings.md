# Planar Ising scaling: sampling allocations at 64 to 1024 spins

Recorded 2026-10-07 from `evidence.tar.gz` (SHA-256 in `manifest.json`).
Protocol: [`docs/experiments/planar-ising-scaling.md`](../../experiments/planar-ising-scaling.md).
Replay: `uv run python -m thermo_lab.planar_ising_scaling --output-dir <extracted> --replay`.

**Evidence classes.** Sweeps are `software_simulation` on CPU (JAX float32,
two-colour block Gibbs). References are `exact_reference`: the Kac-Ward
determinant for the planar zero-field Ising model, checked against brute
force on 2 x 2 to 4 x 4 grids (max error 7e-15 in ln Z, 2e-14 per edge) and
against an independent transfer matrix at L = 8 and 16 (relative 1e-10). The
edge correlations need the inverse of a 2E x 2E matrix that is ill-conditioned
at beta = 4 on frustrated grids, so every reference measures its two weakest
correlations against central differences of ln Z: the recorded error is at
most 2.4e-9 on ferro grids, 6.3e-7 on mixed grids at L = 16 and 1.3e-5 on
mixed grids at L = 32, three to seven orders below the 0.05 threshold.
Energies and sweep times are `calibrated_projection` from the sealed
`Z1HardwareProfile`, which excludes host latency and energy. Nothing ran on
hardware.

## Question

Every recorded sampling comparison so far sits inside exact enumeration
(at most 16 free spins). Does the "symmetry plus tempering" recommendation
hold where sampling is the only route to the answer: 64, 256 and 1024 spins
on planar grids, at the archived cold beta = 4, with the archived five-replica
ladder and the exchange-every-four-sweeps default from the exchange cost
projection? Arms run at equal elapsed sweeps, the parallel-hardware
comparison; edge correlations are the metric, so the analytic flip estimator
is the identity and is not an arm.

## Result in one paragraph

**Past 64 spins, tempering is the only arm that qualifies.** On the ferro
grids the five independent cold chains and the single long chain qualify at
L = 8 (3/3 each, at 1024 to 4096 sweeps) and never at L = 16 or 32 within
4096 sweeps; every tempering arm qualifies at all three sizes, at 64 sweeps
(L = 8), 256 (L = 16) and 1024 (L = 32). On the mixed-sign grids only
tempering qualifies at L = 8 (12/12 arm-targets, ordinary arms 0/6), and no
arm qualifies at L = 16 or 32: a pre-registered negative. The archived
ladder's exchange acceptance collapses with size (cold pair 0.38, 0.24,
0.018 on ferro; 0.095, 0.003, 0.001 on mixed), yet the five-replica arm still
qualifies at L = 32 ferro with about 85 accepted swaps in 4096 sweeps.
Exchanging every four sweeps is not free at this scale with the thin ladder
(two of three L = 32 ferro seeds move from 1024 to 4096 sweeps); the
nine-replica ladder keeps 1024 at k = 4. Where an ordinary arm also
qualifies (L = 8 ferro), tempering at k = 4 is 14x the published energy for a
64x sweep-time gain; elsewhere the energy ratio is undefined because the
ordinary arms never reach the threshold.

## Qualification by size and variant

Count of qualifying (arm, target) pairs out of three seeds:

| Arm | L8 ferro | L8 mixed | L16 ferro | L16 mixed | L32 ferro | L32 mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long | 3 | 0 | 0 | 0 | 0 | 0 |
| independent | 3 | 0 | 0 | 0 | 0 | 0 |
| tempering5-k1 | 3 | 3 | 3 | 0 | 3 | 0 |
| tempering5-k4 | 3 | 3 | 3 | 0 | 3 | 0 |
| tempering9-k1 | 3 | 3 | 3 | 0 | 3 | 0 |
| tempering9-k4 | 3 | 3 | 3 | 0 | 3 | 0 |

Mean edge-correlation MAE at T = 4096, averaged over seeds (threshold 0.05):

| | independent | long | tempering5-k1 | tempering5-k4 | tempering9-k1 | tempering9-k4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L8 ferro | 0.041 | 0.013 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| L8 mixed | 0.135 | 0.140 | 0.022 | 0.023 | 0.012 | 0.016 |
| L16 ferro | 0.080 | 0.062 | 0.0001 | 0.0006 | 0.0001 | 0.0001 |
| L16 mixed | 0.201 | 0.204 | 0.123 | 0.132 | 0.078 | 0.089 |
| L32 ferro | 0.096 | 0.076 | 0.018 | 0.024 | 0.008 | 0.009 |
| L32 mixed | 0.205 | 0.224 | 0.208 | 0.215 | 0.165 | 0.167 |

The ordinary arms get *worse* with size on the ferro grids: five cold chains
each freeze into a domain-wall configuration within the first sweeps and
leave it only slowly at beta = 4. The long chain is slightly better at every
size because its five-fold horizon gives domain walls time to drift. Their
errors do fall with budget, but only by about a factor of 1.4 per fourfold
budget (independent, L = 16: 0.244, 0.189, 0.148, 0.111, 0.080 at T = 16 to
4096), much slower than the inverse square root; reaching 0.05 at L = 16
would take on the order of a million sweeps, and more at L = 32.

## Qualifying budgets and projected cost

Ferro targets, qualifying arms only; elapsed sweeps at the 50 MHz assumed
maximum clock and per-trial energy under the published convention:

| Target | long | independent | t5-k1 | t5-k4 | t9-k1 | t9-k4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L8-ferro-s400 | 1024 (2.3 nJ) | 4096 (9.3 nJ) | 64 (451 nJ) | 256 (557 nJ) | 64 (2285 nJ) | 64 (620 nJ) |
| L8-ferro-s401 | 4096 (9.3 nJ) | 4096 (9.3 nJ) | 64 (524 nJ) | 64 (134 nJ) | 64 (2308 nJ) | 64 (643 nJ) |
| L8-ferro-s402 | 1024 (2.3 nJ) | 4096 (9.3 nJ) | 64 (455 nJ) | 64 (129 nJ) | 64 (2288 nJ) | 64 (629 nJ) |
| L16-ferro-s400 | - | - | 256 (2.9 uJ) | 256 (0.84 uJ) | 256 (13.7 uJ) | 256 (4.3 uJ) |
| L16-ferro-s401 | - | - | 256 (1.4 uJ) | 256 (0.54 uJ) | 256 (12.0 uJ) | 256 (3.7 uJ) |
| L16-ferro-s402 | - | - | 256 (1.6 uJ) | 1024 (2.7 uJ) | 256 (12.8 uJ) | 256 (3.8 uJ) |
| L32-ferro-s400 | - | - | 1024 (10.7 uJ) | 4096 (13.5 uJ) | 1024 (52.7 uJ) | 1024 (17.0 uJ) |
| L32-ferro-s401 | - | - | 1024 (10.3 uJ) | 1024 (4.0 uJ) | 1024 (54.9 uJ) | 1024 (17.5 uJ) |
| L32-ferro-s402 | - | - | 1024 (10.9 uJ) | 4096 (13.5 uJ) | 1024 (51.1 uJ) | 1024 (17.1 uJ) |

L = 8 mixed, where only tempering qualifies: t5-k1 at 4096/1024/1024 sweeps,
t5-k4 at 4096/4096/1024, t9-k1 at 1024/256/256, t9-k4 at 4096/1024/1024
(seeds 400/401/402). The nine-replica ladder is the fastest arm on every
mixed target and the five-replica k = 4 arm the cheapest on two of three.

Under the hypothetical beta-knob convention the L = 32 ferro five-replica arm
at 1024 sweeps costs 7.1 uJ against 10.7 uJ published: reads dominate when
few swaps are accepted, so the knob buys less at scale than it did at n = 16.

## Exchange acceptance collapses with size

Mean acceptance per pair slot at T = 4096 (slots alternate between the two
pair sets):

| | five-replica ladder | nine-replica ladder |
| --- | --- | --- |
| L8 ferro | 0.10, 0.38 | 0.53, 0.19, 0.34, 0.85 |
| L16 ferro | 0.003, 0.24 | 0.18, 0.008, 0.13, 0.71 |
| L32 ferro | 0.000, 0.018 | 0.013, 0.000, 0.010, 0.25 |
| L8 mixed | 0.15, 0.10 | 0.56, 0.32, 0.25, 0.48 |
| L16 mixed | 0.007, 0.003 | 0.23, 0.04, 0.03, 0.14 |
| L32 mixed | 0.000, 0.001 | 0.021, 0.000, 0.000, 0.012 |

Both ladders span 0.25 to 4 with fixed ratios, so the energy difference
between neighbours grows with N and the acceptance falls roughly as
exp(-c sqrt(N)); by L = 32 the ladder is effectively broken in the middle.
The five-replica arm still qualifies on L = 32 ferro because the rare
accepted swap at the cold end (about 85 per trial over 4096 sweeps) is enough
to move a frozen cold replica off its domain wall, and the ferro ground state
is easy to reach once it does. On the mixed grids no such rescue exists at
L = 16 or 32.

A side effect worth noting for the cost model: the published-convention
energy ratio of tempering to independent at a fixed 4096 sweeps falls from
4,293x (L = 8) to 3,009x (L = 16) to 371x (L = 32), not because tempering gets
cheaper but because it accepts fewer swaps. Pricing exchanges by accepted
swaps makes a broken ladder look economical.

## What this changes

1. The recorded comparisons at n <= 16 under-state tempering's advantage. At
   256 and 1024 spins on ferro grids, five cold chains and a long chain
   never reach the threshold; tempering reaches it at 256 to 1024 sweeps.
   On these targets the exchange cost projection's "energy versus time"
   trade-off becomes "tempering or nothing": the Z1 write cost buys the only
   qualifying answer.
2. The archived ladder does not scale. Acceptance at L = 32 is 0.0 to 0.02
   on the pairs that matter. An N-scaled ladder (spacing proportional to
   1/sqrt(N), or a measured-acceptance ladder) is the obvious next arm and
   waits for the owner. With it, the k = 4 recommendation should be
   re-tested; with the thin ladder, k = 4 cost two of three L = 32 seeds a
   budget step.
3. The mixed-sign grids at beta = 4 are out of reach for every arm at 256
   spins and above within 4096 sweeps. This was pre-registered and is the
   expected behaviour of a 2D spin glass at |K| >= 0.8; it is a statement
   about the temperature and budget, not about the samplers. A warmer target
   or a longer horizon is a different study.
4. The exact reference is reusable: Kac-Ward handles any planar zero-field
   graph, so the next step toward the chip's own lattice is a planar subgraph
   of the 16-offset rule, not a return to enumeration.

## Fixed choices that bound these numbers

- Cold beta = 4 and the two ladders are inherited unchanged; the ladder is
  the choice that bounds the mixed-grid result and the L = 32 acceptance.
- Edge correlations are the only metric. Four-spin joints have no exact
  reference at L = 32. Edge correlations are flip-invariant, so this study
  cannot see the symmetry estimator's benefit; it compares samplers, not
  estimators.
- The grid is planar and bipartite but is not the Z1 lattice; no embedding
  overhead is modelled. One logical p-bit per physical p-bit, R x N for R
  replicas.
- The budgets stop at 4096 sweeps. Independent and long errors on ferro
  grids fall by about 1.4x per fourfold budget, so a horizon of order a
  million sweeps would likely qualify them at L = 16; the comparison at a
  fixed budget is the one hardware makes. On mixed grids the nine-replica
  tempering error falls by about 1.5x per fourfold budget at L = 32 (0.37 to
  0.17), so a longer horizon might reach the threshold there too.
- Three seeds per size and variant from one generator. Qualification counts
  are 0/3 or 3/3 everywhere, which is a strong separation but not an
  interval.

## Integrity

Replay: `status=planar_ising_scaling_complete`, `targets=18`,
`cells_replayed=540`, `decisions_replayed=108`, `references_recomputed=18`,
all three checks passed. Replay recomputes every Kac-Ward reference with its
finite-difference check (about four minutes, dominated by the six 32 x 32
grids), the brute-force and transfer-matrix cross-checks, the kernel
stationarity check and the empirical 4 x 4 check, then every estimate from
the exact int16 window sums. It does not regenerate sweeps. Replay is
bitwise under the gate's single-thread BLAS settings; other thread counts
move the ill-conditioned inverse at the 1e-5 level and the 2e-12 comparison
then fails, which is why the gate and the unit test pin the threads.
Generation took 1,293 s on two CPU cores from a clean tree
(`git_dirty=false`). The evidence archive is 6.3 MB, dominated by the window
sums (5.8 MB as a deflated npz); this is the largest sampling archive in the
repository and the owner should confirm it is acceptable.
