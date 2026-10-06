# Potts stage B: label symmetry and tempering on fresh antiferromagnetic Potts targets

**Status: draft, 2026-10-06, awaiting the owner's go-ahead to freeze.** No
study cell has run. The exploratory probe
(`docs/research/potts_tempering_probe.py`, graph seeds 900 and 901, which this
study does not reuse) checked that tempering runs natively in THRML and
located a cold temperature where these targets stop being easy. Its output is
exploration, not evidence. Before freezing, only structural properties of the
fresh graphs (3-colourability) and independent-sample noise floors were
computed. Change any value after freezing only through a new protocol
version.

## Question

On zero-field antiferromagnetic three-state Potts targets, does the October
Ising result carry over? That result was that tempering combined with an
analytic symmetry estimator reaches a fixed accuracy at a smaller budget than
the strongest symmetry-aware ordinary Gibbs baselines (6/6 against 5/6
qualifying, 4.05x to 12.33x faster on four targets;
[findings](../experiment-reports/2026-10-02-symmetry-tempering/findings.md)).
For Potts the symmetry is the group of all 3! = 6 label permutations rather
than one global flip. The study also asks how much of any gain comes from the
symmetry estimator, how much from tempering, and how the answer changes
between a moderate and a cold temperature.

All sampling runs through THRML 0.1.4's categorical Gibbs sampler, whose
per-sweep law Potts stage A checked against the exact kernel
([findings](../experiment-reports/2026-10-06-thrml-potts-contract/findings.md)).
Tempering is one THRML program over five disjoint replica copies of the graph,
with replica exchanges between sweeps. That is how replicas would be laid out
spatially on a sampler array, but nothing here is hardware evidence.

## Fixed choices

Every value below goes into the hashed request. For each, the last column
says whether it could cap or trivialize the accuracy metric.

| Choice | Value | Could it bound the result? |
| --- | --- | --- |
| Graphs | n = 12 weighted cubic graphs (ring plus a disjoint perfect matching), seeds 400 to 405, the generator of the October studies; integer weights 1 to 5 | Six graphs from one family; not a broad population. All six are 3-colourable (seed 401 is bipartite). |
| Model | q = 3, zero field, score `sum_e K_e delta(c_a, c_b)` with K_e = -w_e / 5, P(c) proportional to exp(beta * score) | Zero field is what makes label symmetry valid. |
| Cold temperatures | beta = 8 and beta = 16, every graph at both (12 targets) | In the probe, beta <= 8 was limited by sampling noise in every arm, and beta = 16 trapped ordinary chains on one of two n = 12 graphs. beta = 8 is the expected easy regime and beta = 16 the hard one. Too few hard targets is a possible outcome, not a reason to change seeds. |
| Blocks | A proper colouring found by deterministic backtracking (colours tried in order 0, 1, 2; empty blocks dropped), updated in colour order; one sweep updates every block once | Stage A checked 2- and 3-block categorical schedules. |
| Samplers | `long`: one chain, 5T sweeps. `independent`: five chains at the cold beta, T sweeps. `tempering`: five replicas at beta x (1/16, 1/8, 1/4, 1/2, 1), T sweeps, two exchange attempts after each sweep (pairs (0,1),(2,3) on even sweeps, (1,2),(3,4) on odd), Metropolis acceptance min(1, exp((beta_i - beta_j)(score_j - score_i))) | Every trial pays 5 T n label redraws; tempering also pays 2T exchange attempts and keeps only the cold replica, so it retains a fifth as many states. The ladder repeats the Ising studies' ratios. |
| Estimators | `plain` histograms, and `sym`: the 4-site histogram averaged over all six label permutations. Symmetry adds no redraws. | Edge agreement is permutation-invariant, so symmetry cannot change it. |
| Budgets | T in {64, 256, 1024, 4096, 16384}; the first quarter of each run is discarded; each budget is a prefix of the T = 16384 trace | **The independent-sample noise floor caps the metric.** At T = 4096 the expected 4-site joint TV of iid samples is 0.034 to 0.049 for the cold replica alone, at the 0.05 threshold. T = 16384 is added so unsymmetrized tempering is not capped by noise alone (floor 0.017 to 0.024). |
| Trials | 16 independent trials per target and sampler; each trial's key is folded from root 20261006, target and trial, split into initialization (uniform labels) and sampling keys. Initial states are shared across samplers; the sampler index is folded into the sampling key. | |
| Metrics | Per trial: TV of the joint law of sites 0 to 3 (81 bins), and the mean absolute error of P(c_a = c_b) over edges. Exact references by float64 enumeration of all 3^12 = 531,441 states | The probe found edge-agreement errors of 0.004 or less everywhere, so the joint TV is expected to decide. |
| Qualification | The October rule unchanged: BOTH the mean trial joint TV AND the mean trial edge MAE are at most 0.05 at the selected budget and every larger tested budget | Censored at T = 16384 when not reached. |

## Comparisons

- **Primary.** Per target, `tempering + sym` against the faster-qualifying of
  `long + sym` and `independent + sym`, in budget T. All arms pay the same
  redraws, and tempering also pays 2T exchanges. The study reports
  qualification counts and, where both sides qualify, the ratio of budgets.
- **Decomposition.** `tempering + plain`, `long + plain` and
  `independent + plain` against their `sym` versions, so the symmetry and
  tempering effects can be separated.
- **By regime.** Each comparison is reported separately for beta = 8 and
  beta = 16. Results are not pooled across targets, and nothing is selected
  after seeing errors.
- **Noise floor.** Beside every cell, the expected joint TV of the same
  number of iid draws from the exact law (fixed NumPy seed, plain and
  symmetrized), so a miss can be read as mixing or as sample count.
- **Wall time (secondary, descriptive).** Warm CPU time of each sampler at
  each budget, as the median of three repeats after one warm-up execution with
  identical keys. Compilation is recorded separately. The timed pipeline
  includes sweeps, exchanges, transfer and histogram estimation; it excludes
  enumeration, scoring and persistence. The timed executions must reproduce
  the prefix histograms exactly. Budget ratios are the primary comparison
  because all arms share the redraw count.

## Integrity checks before production

- The replica layout round-trips labels exactly, and one sweep of a
  five-replica program at five different betas reproduces, for each replica,
  the stage A exact one-sweep law at that beta on a 729-state fixture.
- The exchange step's acceptance probability matches the closed form on a
  table of hand-computed cases, including the sign convention.
- The symmetry estimator leaves an S3-invariant law unchanged and maps any
  histogram to an S3-invariant one.

## Outputs and completion

One runner (`thermo_lab.potts_symmetry_tempering`), one replay test and one
report. The archive keeps bounded summaries only: per trial and budget, the
81-bin joint counts, the per-edge agreement counts, the retained-state count
and exchange acceptances, plus timings. It keeps no trajectories. Replay
recomputes the exact references, noise floors, errors, qualification
decisions and budget ratios from the archived counts and checks the result
digest. Completion requires `status=potts_symmetry_tempering_complete`,
12 targets, 180 sampler cells (12 targets x 3 samplers x 5 budgets), 360
estimator cells, 12 primary decisions, `prefix_checks_passed=true` and
`replayed=true`. A negative or mixed scientific outcome is not an integrity
failure.

**Expected cost.** The probe ran 16 trials of every sampler at T = 1024 in
about 1 s each, compilation included, so the full grid with timing repeats
should take minutes on CPU, under the 30-minute autosave threshold. Restart in
a fresh directory. CI replays the archive through the unit test and does not
resample.

## Not claimed

Any hardware speed, energy or latency; behaviour on graphs beyond n = 12 or
other families; field-bearing Potts targets, where label symmetry does not
hold; q > 3; any optimization (colouring) result. Sampling is
`software_simulation` and enumeration is `exact_reference`.
