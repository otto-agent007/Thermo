# Planar annealing: exploratory protocol

Frozen on 2026-10-08 before production sampling. The planar Ising scaling
study and the October 8 probe showed that frustrated grids at beta = 4 are out
of reach for every equilibrium sampler at this lab's budgets, while a single
annealed chain reaches the exact energy per spin within 0.6 percent in 100k
sweeps. This protocol asks the optimization question directly: at equal
elapsed sweeps on a parallel p-bit array, how close to the exact ground-state
energy do annealing, restarts and tempering get on frustrated planar grids,
and what does each cost in the Z1 model?

## Fixed choices and what they bound

- **Targets.** The mixed-sign open grids of the scaling study at L = 8 and 16
  (seeds 400, 401, 402, same generator and coupling scale) plus L = 24 from
  the same generator: 64, 256 and 576 spins, nine targets. L = 32 is excluded
  because its exact ground state needs a planar matching solver this
  repository does not have; the probe's L = 32 result stays exploratory.
- **Reference.** The exact ground-state energy per spin, E0 = max over
  configurations of sum_e J_e s_i s_j / N, from a max-plus transfer matrix over
  2^L window states (new study-local code, checked against brute force at
  L <= 4 and against the Kac-Ward and log-sum-exp transfer matrices at
  beta = 4 on L <= 16). The thermal energy q(beta) at beta = 8 and 16 is also
  recorded from the same transfer matrix: at beta = 16 it sits within 2e-4 of
  E0 on every size checked, so beta = 16 is a ground-state search and beta = 8
  a search that stops about 1e-3 short. The transfer matrix is
  `exact_reference`; at L = 24 it takes about 15 minutes per quantity, so CI
  replays the archived L = 24 references by hash and recomputes L <= 16.
- **Metric.** Energy gap per spin, E0 - q_hat, where q_hat is the chain's
  coupling-weighted mean edge product over the final quarter of its horizon
  (the hold phase). Primary: mean gap over 16 trials; also the best trial
  (the number a restart policy would report), the fraction of trials within
  1e-3 and 1e-4 of E0, and the fraction that reach E0 exactly. Gaps are
  relative to E0, not to q(beta), so a chain that equilibrates at beta = 8
  shows the thermal 1e-3 as a gap; that is the point.
- **Arms, at equal elapsed sweeps T.** All single-replica arms use the
  scaling study's two-colour kernel with a per-sweep beta schedule.
  - `anneal8`, `anneal16`: one chain, beta rises geometrically from 0.5 to
    the endpoint over the first three quarters of T, then holds.
  - `restart8x4`, `restart16x4`: four chains in parallel, each annealed over
    T exactly as the single-chain arm; the reported energy is the best
    chain's. Same elapsed sweeps, four times the p-bits and energy: the
    hardware question is whether parallel restarts buy a better best-of.
  - `cold16`: one chain at beta = 16 from a random start (the floor).
  - `tempering9`: the scaling study's nine-replica ladder, cold replica at
    beta = 4, exchange every sweep; the only equilibrium sampler that moved
    in the probe. Its gap includes the thermal offset of beta = 4 by design.
  Budgets T = 256, 1024, 4096, 16384, 65536; 16 trials; fresh JAX root
  20261011; one initialization per trial shared across arms (the restart arm
  uses four folded keys).
- **Schedule.** One shape (geometric, 75 percent ramp, 25 percent hold). No
  adaptive or feedback schedule, per the lessons register. The 0.5 start is
  above the 2D critical coupling for every edge weight used.
- **Cost model.** Each cell priced in the sealed Z1 profile under both
  exchange conventions of the exchange cost projection; the annealing arms
  have no exchanges, so their cost is updates only, and the beta schedule is
  assumed free (a per-replica temperature control, exactly the `beta_knob`
  question). Physical p-bits are N per chain.

## Pre-registered expectations from the probe

From the 100k-sweep probe on `L32-mixed-s400` at beta = 4: a single anneal
reached a gap of about 5e-3, nine-replica tempering 2.5e-3, five cold chains
1.5e-2. Expect at T = 65536: `anneal16` below 5e-3 on L <= 24, `cold16`
above 1e-2, `tempering9` in between with nine times the p-bits. Whether
`restart16x4` beats `anneal16` at equal sweeps is the open question. A
negative on all arms (nothing within 1e-3 at any budget) is a result.

## Integrity, scope and artifacts

Before sampling, the max-plus transfer matrix must match brute force on
2 x 2 to 4 x 4 grids of both variants, and the log-sum-exp transfer matrix
must match `planar_ising_scaling.transfer_matrix_log_z` at beta = 4 on L = 8
and 16 to a relative 1e-12. The kernel is the scaling study's; its exact
stationarity check is reused at fixed beta.

Persist per cell the per-trial energies (16 floats), nothing else from the
chain; the exchange flags of the tempering arm, packed. Replay authenticates
sources and archived references, recomputes L <= 16 references, every gap,
fraction, decision and pricing from the persisted energies with the inherited
2e-12 tolerance. It does not regenerate sweeps. Write completion last. The
archive is small (kilobytes of energies plus flags).

Sweeps are `software_simulation`; references `exact_reference`; energies and
times `calibrated_projection`. Expected execution: about 2.5 hours on two
CPU cores, of which the three L = 24 references are 1.5 hours (a ground
state takes 14 minutes and a thermal energy 16 minutes; the thermal energy
is an exact one-sweep derivative, not a finite difference); the run autosaves
per target (`--resume`), since it exceeds the 30-minute threshold. Replay
with `--light` verifies the L = 24 references by digest and recomputes the
rest in about a minute; a full replay recomputes all nine.
