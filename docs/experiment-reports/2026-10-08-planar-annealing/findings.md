# Planar annealing: ground-state search on frustrated grids at 64 to 576 spins

Recorded 2026-10-08 from `evidence.tar.gz` (SHA-256 in `manifest.json`).
Protocol: [`docs/experiments/planar-annealing.md`](../../experiments/planar-annealing.md).
Replay: `uv run python -m thermo_lab.planar_annealing --output-dir <extracted> --replay --light`
(about 15 s; drop `--light` to recompute the three L = 24 references too,
about 80 minutes).

**Evidence classes.** Sweeps are `software_simulation` on CPU (JAX float32,
the scaling study's two-colour block Gibbs kernel with a per-sweep beta
schedule). References are `exact_reference`: a max-plus transfer matrix for
the ground-state energy and a one-sweep exact derivative of the log-sum-exp
transfer matrix for the thermal energy, both over 2^L window states, checked
against brute force on 2 x 2 to 4 x 4 grids (max error 2.2e-16 in the ground
state, 2.8e-14 in ln Z, 1.1e-15 in the thermal energy) and against the
scaling study's transfer matrix at beta = 4 on L = 8 and 16 (relative
1.4e-16). Energies and sweep times are `calibrated_projection` from the
sealed `Z1HardwareProfile`, which excludes host latency and energy. Nothing
ran on hardware.

## Question

The scaling study and the October 8 probe showed that frustrated planar grids
at beta = 4 are out of reach for every equilibrium sampler at this lab's
budgets, while a long annealed chain gets close to the exact energy per spin.
So ask the optimization question: at equal elapsed sweeps on a parallel p-bit
array, how close to the exact ground-state energy do a single anneal, four
parallel restarts, a cold chain and the nine-replica tempering ladder get on
mixed-sign open grids of 64, 256 and 576 spins, and what does each cost in
the Z1 model?

## Result in one paragraph

**Four parallel restarts are the best arm at every size and budget, but the
advantage shrinks with size, and past 64 spins nothing gets within 1e-3 per
spin of the exact ground state.** At 64 spins both restart arms reach a mean
gap of 2.3e-4 to 2.7e-4 at 65,536 sweeps, qualify at 1e-3 on 3/3 seeds
(from 1,024 to 65,536 sweeps) and put 96% to 98% of trials within 1e-3;
`restart16x4` reaches the exact ground state in 42% to 73% of trials from
256 sweeps on, and the mean-gap ratio of the best single anneal to the best
restart arm is 4x at 256 sweeps and 23x at 65,536. At 256 spins the ratio is
1.4x to 2.2x, no arm qualifies, and the best arm (`restart8x4`) puts 42% of
trials within 1e-3 at 65,536 sweeps with a mean gap of 3.3e-3. At 576 spins
the ratio is 1.3x to 1.7x, the best mean gap is 4.4e-3, and one trial in 48
gets within 1e-3. The annealing endpoint does not matter: `anneal8` and
`anneal16` are within 10% of each other everywhere, so the gap is trapping
in the ramp, not the thermal offset at the endpoint. The nine-replica ladder
at cold beta = 4 flattens at its own thermal offset (6.2e-3 at 64 spins and
6.7e-3 at 256, where a post-hoc transfer-matrix q(4) sits 6.3e-3 and 6.8e-3
below E0) and is no better than one annealed chain at 576 spins (8.1e-3
against 7.5e-3), at 6,900x to 72,000x the projected energy under the
published exchange convention. The cold chain never moves
below 8e-2. The pre-registered negative holds at 256 and 576 spins.

## Mean gap to the exact ground state

Mean over 16 trials and 3 seeds of E0 minus the best chain's hold-phase
energy per spin; the figure shows the same numbers.

| Arm | 64 spins, T = 256 | 65,536 | 256 spins, T = 256 | 65,536 | 576 spins, T = 256 | 65,536 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cold16` | 1.09e-1 | 8.02e-2 | 1.19e-1 | 8.71e-2 | 1.14e-1 | 8.52e-2 |
| `tempering9` | 8.86e-3 | 6.24e-3 | 2.71e-2 | 6.69e-3 | 2.92e-2 | 8.06e-3 |
| `anneal8` | 2.14e-2 | 5.51e-3 | 3.13e-2 | 8.27e-3 | 2.83e-2 | 7.87e-3 |
| `anneal16` | 2.77e-2 | 5.49e-3 | 3.32e-2 | 7.45e-3 | 3.00e-2 | 7.50e-3 |
| `restart8x4` | 5.22e-3 | 2.34e-4 | 2.24e-2 | 3.33e-3 | 2.19e-2 | 4.39e-3 |
| `restart16x4` | 8.87e-3 | 2.65e-4 | 2.18e-2 | 3.54e-3 | 2.38e-2 | 4.77e-3 |

Fraction of trials within 1e-3 of E0 at T = 65,536: 64 spins 0.98 / 0.96
(restart8x4 / restart16x4), 0.58 / 0.54 (anneal8 / anneal16), 0.02 (cold),
0 (tempering); 256 spins 0.42 / 0.21, 0.15 / 0.17, 0, 0; 576 spins 0.02 /
0, 0, 0, 0. Qualifying (arm, target) pairs at tolerance 1e-3, out of three
seeds per size: `restart8x4` 3 / 0 / 0 and `restart16x4` 3 / 0 / 0 at
64 / 256 / 576 spins; every other arm 0 everywhere. At 1e-4: `restart16x4`
2, `restart8x4` 1, all at 64 spins and 16,384 to 65,536 sweeps.

The "exact ground state" fraction is a hold-length statistic, not a search
statistic: at beta = 16 the equilibrium energy sits 1.3e-5 to 1.4e-4 below
E0 (the recorded q(16)), so a chain that has found the ground state still
records a nonzero gap whenever an excitation appears during its hold, and
longer holds record fewer exact trials (restart16x4 at 64 spins: 0.73 at
1,024 sweeps, 0.60 at 65,536). The 1e-4 fraction is the robust measure.

## Equal sweeps against equal updates

The restart arms use four times the p-bits and four times the projected
energy of a single anneal at the same elapsed sweeps (at 576 spins and
65,536 sweeps: 1.07e-6 J against 2.68e-7 J published, 2,304 against 576
p-bits, 1.31e-3 s either way at the assumed 50 MHz sweep rate). Compared at
equal p-bit updates instead, a single anneal four times longer against four
restarts:

| Spins | Restart T = 256 vs anneal 1,024 | 1,024 vs 4,096 | 4,096 vs 16,384 | 16,384 vs 65,536 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 5.2e-3 vs 1.6e-2 | 1.6e-3 vs 9.0e-3 | 9.2e-4 vs 8.3e-3 | 9.2e-4 vs 5.5e-3 |
| 256 | 2.2e-2 vs 2.1e-2 | 1.3e-2 vs 1.4e-2 | 9.1e-3 vs 1.0e-2 | 4.9e-3 vs 7.5e-3 |
| 576 | 2.2e-2 vs 1.9e-2 | 1.3e-2 vs 1.3e-2 | 8.9e-3 vs 9.2e-3 | 6.3e-3 vs 7.5e-3 |

At 64 spins the restarts win on both accountings by 3x to 9x. At 256 and
576 spins they are a wash on energy up to 16,384 sweeps and win by 1.2x to
1.5x only at the largest budget: on a budget-bound array, parallel restarts
buy sweep time, not energy, once the grid is bigger than the ground-state
basin a single anneal can find.

## Tempering is the wrong tool for this question

The nine-replica ladder (cold beta = 4, exchange every sweep, the only
equilibrium arm that moved in the probe) is included as the sampling-side
baseline. Its gap includes the beta = 4 thermal offset by design. A post-hoc
transfer-matrix q(4) (exploration, not archived; `thermal_energy` at
beta = 4) puts that offset at 6.3e-3 (mean of the three 64-spin targets)
and 6.8e-3 (256 spins): the ladder reaches it by 1,024 sweeps at 64 spins
(6.2e-3) and by 65,536 sweeps at 256 spins (6.7e-3), so there it is an
equilibrium sampler reporting its equilibrium energy, which is the wrong
quantity. At 576 spins its cold-pair exchange acceptance has collapsed
(0.003 on the second slot and 0.001 on the third, against 0.32 and 0.25 at
64 spins and 0.043 and 0.023 at 256) and it tracks a single anneal. Under the published convention it
costs 72,000x (64 spins), 20,000x (256) and 6,900x (576) the energy of one
anneal at 65,536 sweeps; under the hypothetical `beta_knob` convention
1,900x at every size. The annealing arms assume a free per-replica beta
schedule, exactly the hardware feature the exchange cost projection asked
for; an anneal without it would pay a full write per temperature step.

## Pre-registered expectations

- `anneal16` below 5e-3 at T = 65,536 on L <= 24: **not met.** 5.49e-3 at
  64 spins, 7.45e-3 at 256, 7.50e-3 at 576. The probe's 5e-3 was a 100k-sweep
  single chain at 1,024 spins and beta = 4; the geometric 0.5 to 16 ramp
  over 49,152 sweeps does not match it.
- `cold16` above 1e-2: met (8.0e-2 to 8.7e-2).
- `tempering9` between them: met against the cold chain at every size; it is
  slightly worse than the best single anneal at 64 and 576 spins and slightly
  better at 256.
- Whether `restart16x4` beats `anneal16` at equal sweeps: yes at every size
  and budget, by 4x to 23x at 64 spins and by 1.3x to 2.2x at 256 and 576.
- A negative on all arms past 64 spins (nothing within 1e-3 mean gap) is the
  recorded result at 256 and 576 spins.

## What this changes

- **Frustrated planar grids at beta 8 to 16 are an optimization target that
  this lab's samplers do not solve past 64 spins** within 65,536 sweeps: the
  best arm sits 3e-3 to 5e-3 per spin above the exact ground state at 256
  and 576 spins, and the gap is set by trapping during the ramp, not by the
  endpoint temperature. Any claim that annealing on a p-bit array "finds
  ground states" of frustrated 2D instances needs a size and a tolerance
  attached.
- **Parallel restarts are the default optimization arm**, with the
  accounting stated: at equal elapsed sweeps they are the best arm
  everywhere; at equal p-bit updates they win only at 64 spins or at the
  largest budget.
- **Do not use the archived nine-replica ladder as an optimization baseline
  above 64 spins**, and do not read its plateau as a sampler limit: it is
  the beta = 4 thermal offset at 64 spins and collapsed exchange acceptance
  above.
- **Open follow-ups, for the owner:** (1) a slower or two-stage schedule
  (the ramp fraction and shape were fixed choices, below); (2) a planar
  matching solver for exact ground states at L = 32 and beyond, so the trend
  from 1.7x can be followed to 1,024 spins; (3) an N-scaled ladder with a
  low-temperature cold replica as a true tempering-as-optimizer arm. None of
  these is a third adaptive policy.

## Fixed choices that bound these numbers

- **One schedule shape** (geometric 0.5 to the endpoint over 75% of T, hold
  25%). The restart and anneal arms share it, so the comparison between them
  is fair, but the absolute gaps are a statement about this schedule. A
  slower ramp or a hold at an intermediate beta could move every annealing
  number; that is the first thing a follow-up should vary.
- **Budget grid floor and ceiling.** The 65,536-sweep ceiling is where the
  256- and 576-spin curves are still falling (about a factor 1.5 per
  fourfold budget), so the negative is a budget statement at this ceiling.
- **Hold-phase averaging.** Reporting the hold-phase mean rather than the
  minimum energy visited under-reports a search that found E0 and left it;
  the 1e-4 fraction and the recorded q(16) bound that effect at 1.4e-4.
- **Sizes 8, 16 and 24 only.** The exact ground state is a 2^L transfer
  matrix (15 to 28 minutes per quantity at L = 24 on one core); L = 32 needs
  a planar matching solver this repository does not have.
- **The ladder is the archived one**, not an N-scaled ladder, by design
  (the probe showed a scaled ladder plateaus too).

## Integrity

`completion.json`: `planar_annealing_complete`, 9 targets, 270 cells and 54
decisions replayed, 6 references recomputed and 3 verified by digest
(`--light`), reference checks and the kernel's exact stationarity check
passed. Replay authenticates the request digest, the seven pinned sources,
the packed exchange flags and the archived references, then recomputes
every gap, fraction, pricing, decision and summary from the persisted
per-trial energies (16 floats per cell) with the inherited 2e-12 relative
tolerance; it does not regenerate sweeps. The run took about 2.5 hours on
two CPU cores (64 spins 3 minutes per target, 256 spins 6 minutes, 576
spins 37 minutes, of which 27 to 28 minutes are the two exact references),
autosaving one unit per target; the workspace was reclaimed twice during
the 576-spin phase and the run resumed from the saved units without
resampling. The archive is 2.9 MB, almost all of it the packed exchange
flags (`flags.npz`, 2.7 MB).
