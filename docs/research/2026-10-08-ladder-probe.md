# Scaled ladders and mixed-grid reachability at 1024 spins (exploration)

Exploration on 2026-10-08, after the
[planar Ising scaling study](../experiment-reports/2026-10-07-planar-ising-scaling/findings.md).
Script: [`ladder_probe.py`](ladder_probe.py); raw numbers:
[`2026-10-08-ladder-probe.json`](2026-10-08-ladder-probe.json). One ferro and
one mixed 32 x 32 target (`L32-ferro-s400`, `L32-mixed-s400`), the study's
exact Kac-Ward references and two-colour sampler unchanged, 16 trials, a
fresh JAX root (20261010). **Not recorded evidence**: no archive, no replay,
no gate; one target per variant and one seed. It exists to decide what to
freeze next, not to be cited as a result.

## Question A: does a denser ladder recover exchange acceptance, and does the cold error follow?

Geometric ladders from beta 0.25 to 4 with R replicas, exchange every sweep,
T = 4096. "min/median" are over pair slots at the final budget.

| R | ratio | ferro min / median acc | ferro edge MAE | mixed min / median acc | mixed edge MAE | p-bits |
| ---: | ---: | --- | ---: | --- | ---: | ---: |
| 5 | 2.000 | 0.000 / 0.008 | 0.0173 | 0.000 / 0.000 | 0.196 | 5,120 |
| 9 | 1.414 | 0.000 / 0.012 | 0.0038 | 0.000 / 0.006 | 0.168 | 9,216 |
| 13 | 1.260 | 0.000 / 0.067 | 0.0043 | 0.002 / 0.026 | 0.136 | 13,312 |
| 17 | 1.189 | 0.002 / 0.173 | 0.0026 | 0.017 / 0.092 | 0.130 | 17,408 |
| 25 | 1.122 | 0.027 / 0.334 | 0.0014 | 0.101 / 0.253 | 0.115 | 25,600 |
| 33 | 1.091 | 0.072 / 0.486 | 0.0002 | 0.222 / 0.390 | 0.117 | 33,792 |

Acceptance recovers to the usual 0.2 to 0.4 range only at R = 25 to 33, a
neighbour ratio of about 1.1, that is Delta beta / beta of about 0.1 at
N = 1024. That is 5 to 7 times the archived ladder's replica count and
25 to 33 x 1024 physical p-bits for one 1024-spin problem.

What the recovered acceptance buys differs by variant:

- **Ferro.** R = 5 and 9 already qualify (0.017, 0.004 below the 0.05
  threshold) with a ladder whose middle is dead; a working ladder takes the
  error to 0.0002. The archived five-replica result on ferro grids was a
  rescue by rare cold-end swaps, as the study said, and a ladder is not what
  qualification needed there.
- **Mixed.** The error falls from 0.196 to 0.136 by R = 13 and then
  flattens: 0.130, 0.115, 0.117. Turning the ladder from broken to working
  (min acceptance 0.00 to 0.22) buys about 40 percent and stops. At 4096
  sweeps the frustrated grid is a cold-replica mixing problem that exchanges
  do not solve.

## Question B: does anything reach the threshold on the mixed grid?

`L32-mixed-s400`, 100,000 sweeps (the long chain runs 5 x 20,000), threshold
0.05 on mean edge-correlation MAE. Energy per spin is the coupling-weighted
mean edge product, q = sum_e J_e <s_i s_j> / N, against the exact
beta = 4 value 0.8985.

| Arm | edge MAE at 4096 | edge MAE at 100k | q per spin at 100k | wall |
| --- | ---: | ---: | ---: | ---: |
| nine-replica ladder, k = 1 | 0.165 | 0.077 | 0.8960 | 490 s |
| five cold chains | 0.205 | 0.163 | 0.8840 | 288 s |
| single long chain | 0.224 | 0.203 | - | 64 s |
| annealed single chain (0.5 to 4 over 75k sweeps, then measure) | - | 0.141 | 0.8935 | 64 s |

Nothing qualifies. The nine-replica ladder is the only arm whose error keeps
falling with budget: 0.165 to 0.077 for a 24x longer horizon, roughly
T^-0.24, so reaching 0.05 would take on the order of 500k sweeps on this
target, with a broken ladder (R = 9). The cold chains are flat; 100k sweeps of
cold Gibbs on a frustrated grid at |K| >= 0.8 do not move them.

The annealed chain is the informative row. Its edge correlations are as wrong
as a cold chain's (0.141), but its energy per spin is within 0.6 percent of
the exact thermal average at beta = 4 after 100k sweeps. A single anneal
finds *a* low-energy state; the exact correlations average over the many
near-degenerate states a frustrated grid has, and one chain cannot see them.
The nine-replica ladder gets to 0.8960 (0.3 percent low) with nine times
the p-bits and eight times the wall time; five cold chains stop at 0.8840
(1.6 percent low). On energy, one annealed chain is between the two and
closest to tempering per p-bit.

## What this changes

1. **An N-scaled ladder is not the next study.** It needs 25+ replicas at
   1024 spins to work, and where it works (ferro) the thin ladder already
   qualified; where qualification is missing (mixed) a working ladder does
   not supply it. Freezing a 25-replica ladder protocol would spend a day
   measuring a known result. The planar study's "waits for the owner" line
   on the scaled ladder should be read as closed by this probe unless the
   owner wants the recorded version.
2. **The frustrated grid at beta = 4 is an optimization target, not a
   sampling target, at any budget this lab runs.** Edge correlations need
   about 500k sweeps with the best arm; energy per spin is reached by a
   100k-sweep anneal to within 0.6 percent. If the hardware question is
   "what can a p-bit array do with frustrated problems", the metric is
   ground-state energy (exact for planar graphs via the same Kac-Ward
   machinery, taken to large beta) and the arms are annealing schedules and
   restarts, not equilibrium samplers. That is a different protocol and the
   lessons file's "two adaptive policies failed" note does not apply to it.
3. **The hardware-shaped version of the planar study (planar subgraph of the
   16-offset lattice) can go ahead on ferro-type targets with the thin
   ladder**, since that is the regime where the recorded result stands. It
   should not include the mixed variant at beta = 4 unless it adopts the
   energy metric above.
4. Pricing: 25 x 1024 p-bits per problem for a working ladder is about 10
   percent of the chip for one 1024-spin instance. Either tempering on
   hardware means few, large problems, or the exchange step has to be made
   cheap enough to run on a thin ladder at high frequency, which is the
   `beta_knob` question from the exchange cost projection.

## Fixed choices in this probe

Geometric ladders only (an acceptance-targeted ladder would place replicas
where the energy variance is, probably needing fewer at the cold end);
hot end fixed at beta 0.25; one anneal schedule (geometric, 75 percent ramp,
measure the last quarter) without restarts; one seed and one target per
variant. The 100k-sweep arms were run once; no error bars.
