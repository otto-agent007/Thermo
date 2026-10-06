# Symmetry plus tempering carries over to cold Potts targets, not to moderate ones

At the cold temperature (beta = 16), tempering combined with the
label-permutation estimator qualifies on **6/6** fresh antiferromagnetic
three-state Potts targets. Each symmetry-aware ordinary Gibbs baseline
qualifies on **5/6**. Tempering plus symmetry needs a **4x to 16x smaller
budget** on three targets (4.2x to 15.0x less warm CPU time), is the only arm
that qualifies on a fourth, ties on a fifth, and loses on the sixth (the
bipartite graph) by 4x in budget and about 2.3x in time. That
reproduces the October Ising pattern
([findings](../2026-10-02-symmetry-tempering/findings.md): 6/6 against 5/6,
4.05x to 12.33x faster on four targets).

At the moderate temperature (beta = 8) it does not carry over. The
symmetry-aware ordinary baselines reach the threshold at a **4x smaller
budget on four of six targets**; the two others are one tie and one 4x win
for tempering. Here no sampler is trapped, and tempering pays for keeping only
its cold replica: it retains a fifth as many states for the same redraws, and
its symmetrized error sits at or near the independent-sample floor for that
count.

The [frozen protocol](../../experiments/potts-symmetry-tempering.md) fixed
six fresh graphs (seeds 400 to 405) at two temperatures, three samplers, two
estimators, five budgets and 16 trials before any cell ran. All 180 sampler
cells, 360 estimator cells and 12 primary decisions completed and replayed.
Sampling is `software_simulation` (THRML 0.1.4 categorical block Gibbs, CPU,
float32); references are `exact_reference` (float64 enumeration of 3^12
states). Nothing here is hardware evidence.

## Every primary decision

Selected budget = smallest T at which BOTH mean trial joint TV and mean trial
edge-agreement MAE are at most 0.05, at that T and every larger tested T. NR =
not reached by T = 16384. All arms pay 5 T n label redraws per trial;
tempering also pays 2T exchange attempts.

| Target | Ground-state mass | long+sym | independent+sym | tempering+plain | tempering+sym | Outcome | Budget ratio | Time ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g400, beta 16 | 0.913 | 16384 | 16384 | 16384 | **4096** | tempering+sym smaller | 4.00 | 4.21 |
| g401, beta 16 | 0.962 | **256** | 4096 | 4096 | 1024 | baseline smaller | 0.25 | 0.44 |
| g402, beta 16 | 0.954 | NR | NR | 16384 | **1024** | only tempering+sym | n/a | n/a |
| g403, beta 16 | 0.943 | 16384 | 16384 | 16384 | **1024** | tempering+sym smaller | 16.00 | 14.79 |
| g404, beta 16 | 0.872 | 4096 | 4096 | 16384 | 4096 | equal | 1.00 | 0.94 |
| g405, beta 16 | 0.942 | 16384 | 16384 | 16384 | **1024** | tempering+sym smaller | 16.00 | 15.01 |
| g400, beta 8 | 0.582 | 1024 | **1024** | 16384 | 4096 | baseline smaller | 0.25 | 0.27 |
| g401, beta 8 | 0.797 | 256 | **256** | 16384 | 1024 | baseline smaller | 0.25 | 0.22 |
| g402, beta 8 | 0.780 | 4096 | 4096 | 16384 | **1024** | tempering+sym smaller | 4.00 | 3.78 |
| g403, beta 8 | 0.684 | 1024 | **1024** | 16384 | 4096 | baseline smaller | 0.25 | 0.24 |
| g404, beta 8 | 0.451 | 256 | **256** | 16384 | 1024 | baseline smaller | 0.25 | 0.24 |
| g405, beta 8 | 0.691 | 1024 | 1024 | 16384 | 1024 | equal | 1.00 | 0.95 |

Budget ratio is the best qualifying symmetry-aware baseline's budget over
tempering+sym's. Time ratio is the same comparison in median warm CPU seconds
at the selected budgets; where the two baselines tie in budget, the
comparison uses `independent` (name order, fixed in the runner before the
run). Ground-state mass is the exact probability of the lowest-energy
colourings. The budget grid is fourfold, so these ratios only locate the
crossings coarsely.

On g402 at beta 16, the strongest ordinary baseline still has joint TV
0.095 (independent+sym) and 0.200 (long+sym) at T = 16384, against 0.014 for
tempering+sym. That target traps ordinary chains in the way the probe found
for its seed-900 graph.

## Where the gain comes from

- **Symmetry is necessary for ordinary chains.** Without it the long and
  independent samplers qualify on 7/12 targets each; with it, 11/12. At zero
  field every label permutation of a state is equally likely, and averaging
  the joint over the six permutations removes the imbalance between colour
  classes that a trapped chain carries, without new draws. Edge agreement
  does not change under permutation, and it never decided anything here: its
  largest mean error over every cell is 0.021.
- **Tempering is necessary on the hard targets.** On g402, g403 and g405 at
  beta 16 the symmetry-aware ordinary chains stay well above their noise
  floor at T = 16384, while tempering+sym is near its own. Exchange
  acceptance across the ladder is 0.37 to 0.83.
- **Tempering alone is held back by its sample count.** Unsymmetrized
  tempering qualifies on 12/12, but on 11 of them only at T = 16384. At
  T = 4096 its error is still 0.047 to 0.102, about 1.3x to 2.9x its
  cold-replica noise floor. The probe's warning was right: on the October
  grid (T <= 4096) unsymmetrized tempering would have qualified on 1/12.
- **The moderate-temperature loss is the retained-sample cost.** At beta 8,
  tempering+sym is within 1.0x to 2.0x of its independent-sample floor at
  T = 4096, so its chains mix; the ordinary samplers keep five times as many
  states per trial for the same redraws.

At T = 16384, all 16 tempering+sym trials pass individually on every target.

## What the timings include

Each warm measurement covers THRML block sweeps, exchanges, device-to-host
transfer and the joint and edge-agreement estimation on host, as the median of
three repeats after one warm-up with identical keys. Compilation is recorded
separately and excluded. Every timed execution at a smaller budget reproduced
the corresponding prefix of the T = 16384 run exactly (180 of 180 prefix
checks). On this CPU a five-replica sweep costs about the same as one
replica's: per T, `independent` takes 0.95x and `long` (5T single-replica
sweeps) 1.78x the time of `tempering`. That is CPU vectorization across
replicas, and it is why the g401 beta 16 loss is smaller in time than in
budget. It is not a hardware claim.

Timings are descriptive and vary between runs. An earlier complete run with
the identical result digest (see Integrity) gave time ratios of 0.96 on g401
at beta 16 and 8.61 on g405 at beta 16, against 0.44 and 15.01 here; the other
nine with a ratio agree within 12%. Budget ratios, which decide the outcomes, do not
depend on timing.

## Integrity

- The preflight passed before production. The replica layout round-trips
  exactly, and one sweep of a five-replica THRML program at betas 1, 2, 4, 8
  and 16 reproduces, replica by replica, the stage A exact one-sweep law on
  the stage A graph with delta couplings (TV 0.0028 to 0.0254 against
  tolerances 0.0059 to 0.0282 at 100,000 chains).
- The first full run's provenance was `git_dirty` only because this
  module's test file, written while the study ran, was untracked; the runner
  and protocol matched the commit. After the test was committed, a re-run
  from the clean tree reproduced the identical result digest, and the
  archive here is the clean run.
- After the run, replay was changed to compare the recomputed evaluation with
  the archived one numerically instead of re-hashing it, because the joint
  estimates pass through BLAS and a simulated older CPU
  (`OPENBLAS_CORETYPE=Prescott`) moved them in the last bits. The archive
  and its digests are unchanged; replay passes on this host, Prescott and
  Haswell.

## What this does and does not settle

On zero-field antiferromagnetic Potts targets of this size, the rule from the
Ising studies holds when ordinary chains are trapped: combine tempering with
the symmetry estimator. When they are not, symmetry-aware ordinary Gibbs
reaches the threshold sooner, because it retains every replica. A practical
policy therefore has to detect trapping before choosing, as the changing
evidence study also found for restart policies. That is a design question for
later work, not a result here.

Not settled: graphs larger than 12 sites or other families, field-bearing
Potts targets (where label symmetry fails), q > 3, retaining hot replicas by
reweighting, other ladders, any colouring or optimization result, and any
hardware speed, energy or latency. Six graphs from one generator are not a
population.

## Files

- `study.json.gz` (0.31 MiB, SHA-256
  `e2327dc834f0a01c6ee5720d71236e823e7feae8933322417c6a40926613279c`): request, preflight, exact references, noise floors,
  per-trial joint and edge-agreement counts for every sampler cell and budget,
  exchange counts, evaluation, raw timings and the result digest. No
  trajectories.
- `summary.md`: the decision table and per-target error tables rendered from
  the archive.
- `completion.json`: 12 targets, 180 sampler cells, 360 estimator cells,
  12 decisions, qualification counts per arm, preflight and prefix checks
  passed, replayed.
- `provenance.json`, `run.log`: runtime (JAX 0.10.2 CPU, x64 off, Python
  3.11.15, commit 0bd6421, clean tree) and per-target progress. The run took
  817 s including replay.
