# Z1 acceptance kit: protocol (draft, proposal P-0015, round 1)

Redrafted on 2026-11-17 against the queue row `z1-acceptance-kit` as amended
on main (PRs #176 and #179). Not frozen until the owner approves it; no runner
exists yet. Probes (exploration, not evidence):

- round 0: [`2026-11-15-z1-acceptance-kit-probe.md`](../research/2026-11-15-z1-acceptance-kit-probe.md)
  (reproduce-first check, beta = 4 counts, defect admission on patches);
- review: `docs/research/z1_acceptance_kit_review_probe.py` and its JSON
  (read correlation at beta 4, warmer temperatures);
- round 1: [`2026-11-17-z1-acceptance-kit-probe-r1.md`](../research/2026-11-17-z1-acceptance-kit-probe-r1.md)
  (isolated-pair and single-site tests, temperature choice, pass-rule
  arithmetic).

## The number this study produces

For each of the row's four defects, **the reads and programming writes of
the cheapest exact test that detects it at 5 percent false alarm and 5
percent miss**, and **the smallest defect of that kind detectable within
10,000 reads**. Two families of test are compared on the same footing:

- **simple tests**: isolated pairs (one matching of the patch, one coupler
  per pair, pooled over pairs where the defect is shared) and single sites
  (one bias, no couplers), with closed-form exact references;
- **collective tests**: Kac-Ward statistics on the nine P-0001
  `greedy-long` graphs at zero field and one chosen inverse temperature.

A collective test stays in the kit only where it needs fewer reads or fewer
programming writes than the simple test, or detects something the simple
test cannot. The protocol says which, per defect (table below). Every
retained test's operating characteristic is then checked by sampling.
Means, shifts and read counts are `exact_reference`. Read correlation and
operating characteristics are `software_simulation`. It is a metrology row:
no chip, no claim about any chip, no digital baseline to price.

## Question

Which exact-answer tests on Z1-native graphs can accept or reject a sampled
array, and how many reads does each need to detect a stated defect at stated
error rates?

- **User and decision.** A person qualifying a sampled array decides which
  test to run per defect and how many reads and programmings to spend.
- **Mechanism.** Each declared defect is a fault of one coupler or one site.
  On an isolated pair or site the two-colour heat bath draws every read
  afresh, so reads are exactly independent and the count test has exact
  binomial error rates. A collective graph pools many edges per read but
  carries mixing correlation and a start-state question.
- **Falsifier.** A retained test's sampled false alarm or miss exceeds the
  nominal 5 percent by more than the multiplicity-corrected tolerance below.

## Hardware constraints (rule 8)

Applied: the two-colour schedule (colour A, then colour B, every sweep);
16-offset placement (the planar subgraph and its matchings use lattice
edges only, and every edge joins the two colours); zero field on the
collective graphs; coupling rounding under the constraint layer's codebook;
no replica exchange.

Relaxed or assumed, and why:

1. The ideal model has float couplings; it is the reference.
2. Single-site tests program a site bias. The row lifts the zero-field
   restriction for this test only. The bias is assumed representable at the
   declared value; a bias-scale error is indistinguishable from a gain error
   in this test (both enter as the product beta_i h), so a failed site test
   detects "gain times bias off nominal", not which of the two.
3. Simple tests choose their own program (beta, coupling value, bias) inside
   the declared ranges: beta in [0.5, 4], |J| at most the cap 1. The
   collective tests use the archived couplings at the chosen beta.
4. Planarity restricts the chip's degree-16 rule; the exact collective
   reference exists only on planar graphs. On the full degree-16 lattice a
   matching scan needs 16 programmings, a planar cover at least three.

## Resource accounting (rule 9)

Every test reports elapsed sweeps (burn-in included), full-array reads,
node reads, programming writes (one per coupling and bias configuration)
and state writes (one per chain start). Simple tests need no state write:
an isolated pair or site forgets its start after one sweep (checked by
enumeration). No joules are projected; this is not a task row.

## Defects and their kernels

All defects use the two-colour heat bath, P(s_i = +1) = sigmoid(2 beta g_i
h_i), with ideal gain g_i = 1.

- **Rounding.** Every coupling replaced by `quantize_array(J, 1.0, bits)`
  (the `full` cap). Computed at 4, 5, 6, 7, 8, 9, 10, 11 and 12 bits. 5 and
  9 bits are exact on the weights {1..5}/5 and carry no defect.
- **Missing coupling.** One coupler reads 0 whatever is programmed.
- **Stuck p-bit.** One site never changes value. Two behaviours, both
  covered by the single-site test: it ignores writes (reads a constant), or
  it accepts a write and never updates. An isolated unbiased ideal site
  resamples a fair coin every sweep from any start, so both behaviours read
  a constant where the ideal site does not. No random start is needed, and
  none is relied on.
- **Static beta variation.** One site's gain is 1 + delta, delta = ±0.1 or
  ±0.3 (the extremes of the declared spread). Only the single-site test has
  an exact reference. The collective graph has none (the stationary law with
  site gains is not an Ising law; round-0 probe finding 7), and the pair
  product sees only the colour-B site's gain (enumerated check).

## Chosen temperature for the collective tests

Rule (amendment item 1), fixed in the round-1 probe before its sampling:
among betas evaluated on all nine graphs whose random-start bias passes
(|z| ≤ 3.5 for energy and both rounding LLRs on every graph, after 512
burn-in sweeps), take the beta minimising the largest ratio, over graphs and
over the 6- and 8-bit rounding LLR tests, of sweeps per test (512 + reads ×
scatter ratio) to that graph's best over its own grid.

**Chosen: beta = 0.6.** Both betas run on all nine graphs (0.5 and 0.6)
passed the bias check (largest |z| 2.5). The largest regret was 1.19 at 0.6
and 1.40 at 0.5. That margin is close to the sampling error of the scatter
ratios (relative standard error 9 to 13 percent), so 0.5 would serve almost
as well; the rule, not a judgement, picks 0.6. Colder betas lost where they
were run: at 0.75 the 8-bit LLR scatter ratio is 1.96 to 6.42 at 64 and 256
spins (3.5 to 9.8 at 1,024 in the review probe), and at 2.0 the
random-start bias fails at 64 spins (z = −4.8 and −4.5). A second-seed
check at 0.6, two consecutive test windows per graph, gave 8-bit LLR
scatter ratios of 1.25 to 1.99 in both windows and no chain-level outlier
beyond what 128 to 256 normal chain means produce (largest |z|/sqrt(r)
3.66). Rare slow states below about one chain in 256 are not excluded by
the probe. New exact references at
beta = 0.6 are computed for all nine graphs. Beta = 4 is used only for
the reproduce-first check of the 18 P-0001 references.

Read correlation (amendment item 2) is the scatter ratio: the variance
across independent chains of the mean of a window of reads at the test
length, times the window length, over the exact per-read variance. It is
measured on every graph, for each retained statistic, at that statistic's
own test length. It is not a short-window estimate.

## Which test each defect gets (from the round-1 probe)

Reads are for independent reads at 5 percent false alarm and 5 percent
miss. Simple-test reads are exactly independent. Collective counts are
multiplied by the measured scatter ratio before use (n*).

| Defect | Simple test: reads, programming writes | Collective test at beta 0.6 | In the kit |
| --- | --- | --- | --- |
| Stuck p-bit (either behaviour) | Unbiased isolated site: 6 at a known site; 12, 14, 16 scanning 64, 256, 1,024 sites. 1 programming, 0 state writes | Same counts, but every read is a fresh chain: a state write, 512 burn-in sweeps and a fair-coin start each | Simple only |
| Gain error ±10 percent | Biased site at the best beta h: 2,239 (−10) and 2,751 (+10) at a known site; 6,278 and 7,878 scanning 1,024 sites. 1 programming | No exact reference | Simple only |
| Gain error ±30 percent | 199 and 368 at a known site; 543 and 1,077 scanning 1,024 sites | No exact reference | Simple only |
| Missing coupling, location known | Pair at \|J\| = 1, beta 4: 5 reads, 1 programming | Own edge: 35 to 928 reads (64 and 256 spins) | Simple only |
| Missing coupling, location unknown | 8 reads per matching, 7 to 9 matchings: 56 to 72 reads, 7 to 9 programmings | All edges at once: 1,906 to 2,444 reads, 1 programming (64 and 256 spins; 1,024 left to the runner) | Both: simple for reads, collective for writes |
| Rounding, 8 bits | Matching at weight 0.2, beta 4: 238, 955, 3,788 reads at 1,024, 256, 64 spins. 1 programming | LLR: 3,738 to 3,785 at 1,024 spins (scatter 1.35 to 1.61), 15,288 to 16,317 at 256, 73,881 to 83,235 at 64 | Simple only |

So the collective Kac-Ward test stays for one case only: scanning for a
missing coupling at an unknown location with one programming write instead
of 7 to 9, at about 30 times the reads. It sees nothing declared that the
simple tests miss. What only a collective test could see is the update
process (stale neighbour reads, colour order). The row does not declare
those defects, and this protocol does not add them.

## Pre-registered expectations (from the probes)

1. All 18 P-0001 references reproduce bitwise at beta = 4.
2. Simple-test counts match the round-1 probe JSON exactly (they are closed
   form): stuck p-bit 6 reads at a known site and 12, 14, 16 scanning all
   sites; missing coupling 5 reads at a known coupler and 8 reads per
   matching (7 to 9 programmings) scanning; 10 percent gain error 2,239 and
   2,751 reads at a known site; 8-bit rounding 238 reads at 1,024 spins.
3. The smallest defect detectable within 10,000 reads, over every depth
   computed (4, 6, 7, 8, 10, 11 and 12 bits; 5 and 9 carry no defect):
   - rounding: 8 bits at every size with the pooled pair test. 10, 11 and 12
     bits need 12,572, 15,509 and 61,981 reads at 1,024 spins. The
     collective LLR at beta 0.6 reaches 8 bits at 1,024 spins, 7 bits at
     256 and 6 bits at 64 in independent reads (the 7-bit scatter ratio is
     not yet measured);
   - gain error: a positive delta of 5.1 percent at a known site, and 7.6,
     8.3 and 8.8 percent scanning 64, 256 and 1,024 sites;
   - missing coupling: every coupler (pair test);
   - stuck p-bit: both behaviours, at any site.
4. Every sampled check passes its operating-characteristic rule, and every
   collective target passes its attribution check at beta = 0.6.

## Exact anchors and evidence classes

- Simple tests: closed-form two-state laws. P(pair product = sign J) =
  sigmoid(2 beta |J|); P(site = +1) = sigmoid(2 beta g h); a free unbiased
  site is a fair coin per read. Independence of consecutive reads under the
  two-colour sweep is checked by enumerating the two-site chain, to 1e-12.
  Counts use exact binomial error rates; pooled pairs are independent
  trials because the pairs are disconnected.
- Collective tests: the Kac-Ward determinant on the explicit embedding,
  imported unchanged from `planar_16_offset_ferro`. A missing edge's
  correlation is d ln Z / dK_e at 0 by central difference; all edges by
  rank-2 update, checked against direct evaluation on 20 declared edges per
  target to 1e-9. Per-read variances are second differences of ln Z,
  checked against brute force on a 4 x 4 patch to a relative 1e-6.

Means, shifts, per-read variances and independent-read counts are
`exact_reference`; normal-approximation counts are labelled as such.
Scatter ratios, sampled means and operating characteristics are
`software_simulation` (NumPy float64, CPU). Nothing runs on hardware.

## Arms and sampled checks

- **Simple tests** (run on the same two-colour kernel code, with all other
  couplers off): stuck p-bit at a known site and scanning 1,024 sites; gain
  error ±10 and ±30 percent at a known site, at the probe's best beta h;
  missing coupling at a known coupler and one scanned matching at 1,024
  spins; rounding at 4, 6, 7 and 8 bits pooled over a 1,024-spin matching at
  the probe's best (beta, weight). 12 checks, 1,000 replicates each.
- **Collective missing coupling** at beta = 0.6, as the scan over every
  edge with Bonferroni false alarm 0.05 / E: per target, three declared
  edges among those the scan detects within 10,000 reads (cheapest, median
  and dearest by n*, where n* uses the scatter ratio of that edge's own
  product). 27 checks, 1,000 replicates, ideal chains shared across the
  three edges of a target. The ideal side measures the family-wise false
  alarm of the whole scan.
- **Control**: the ideal model, with false alarm measured at every check's
  read count and threshold.
- **Excluded**: collective tests for rounding, stuck p-bit and gain error
  (the simple test needs fewer reads and no more writes, or the collective
  test has no exact reference).

The predicted count n* for a collective check is the independent-read count
times that statistic's scatter ratio on the ideal model at that length,
rounded up. Simple checks use the exact count (scatter ratio 1 by
construction; measured anyway and reported).

## Primary decision rule (fixed in advance)

Number of sampled checks m = 39, each tested on two sides. Per side the
tolerance is Bonferroni at 0.05 / (2m), so the chance that any correctly
calibrated check fails by luck is at most 5 percent over the whole study
(amendment item 5). Every check has 1,000 replicates, and a check fails if
the ideal sampler rejects in more than 74 of them or the defective sampler
in fewer than 926.
The family-wise false-failure bound is 3.2 percent. These thresholds still
flag a test whose true false alarm is 10 percent with probability 0.997,
and one at 8 percent with probability 0.74. At 400 replicates the same
multiplicity would give thresholds of 35 and 365 and only 0.77 power at 10
percent, which is why 1,000 are used. The round-0 rule (31 and 369 of 400
per check, about 54 checks) had a 30 percent chance of at least one false
failure on the false-alarm side alone and 52 percent on both sides.

**Attribution check** per collective target, before its checks count: the
ideal sampler's mean of each tested statistic, pooled over its replicates,
must have an upper confidence bound on |bias| (at 0.05 / 9) no larger than
0.05 times the smallest admitted shift of that statistic on that target. A
bias of that size moves a 5 percent false alarm to at most 6.9 percent at
n*. This replaces the round-0 "within 3 standard errors, no margin" rule.
The pooled ideal replicates give a standard error of about 0.01 sqrt(r)
times the shift (r the scatter ratio), so the bound is reachable for r up
to about 4 when the bias is near zero; a larger r leaves the target
unattributable for lack of precision, and the report says which cause.
A target that fails is unattributable; its checks are reported and do not
count. Device error (exact shift) and mixing error (ideal sampled mean minus
exact, with its standard error) are reported apart for every target.

Row verdict, one of three sentences:

- **The kit holds**: every collective target is attributable and every
  check passes.
- **The kit holds except for named checks**: each failure named with its
  rejection counts.
- **The collective test is not attributable at size L**: a target of that
  size fails attribution. The simple tests are unaffected.

## What one recorded sample means

One read is the full array after one complete two-colour sweep. A simple
replicate test uses n consecutive reads after one sweep from an arbitrary
start (no burn-in needed; the enumerated check shows the first full sweep
already gives the stationary law). A collective replicate test uses n
consecutive reads from one chain after a random start and 512 burn-in
sweeps. Replicates, not reads, are the replication unit.

## Seeds

Graph and coupling seeds 4510, 4511, 4512 are target identities. Sampler
root 20261118 (fresh; the probes used 20261115, 20261117 and the review's
seeds), folded with arm, target, model and replicate.

## Budgets and CPU estimate

Scaled from round-1 timings (42.7 µs per chain-sweep at 1,024 spins; Kac-Ward
13.6 s with correlations, 2.8 s for ln Z alone).

- Exact part, about 20 CPU-minutes: references at beta 4 and 0.6;
  every missing edge by rank-2 update with direct checks; simple-test
  tables.
- Simple sampled checks: under 2 CPU-minutes.
- Collective sampled checks: about 45 CPU-minutes, mostly the 1,024-spin
  targets (1,000 replicates × (512 + n*) sweeps per model, n* about 2,000
  to 4,000, one ideal and three defect models per target).
- Replay: about 10 minutes, no sweeps.

**Total about 1.3 CPU-hours**, about 30 min wall with three workers. Peak
memory about 0.3 GiB per worker.

## Autosave and resume

Over 30 minutes, so it follows the autosave contract in
`docs/experiment-runner.md`. The work unit is one (arm, target, model)
group. Units are written atomically under `units/` with a digest index.
`--resume` re-authenticates the request and sources, verifies every
completed unit and runs only the missing ones. Interruption test: kill after
at least one sampled unit, resume, require bitwise equality with an
uninterrupted run of the same small request.

Resume command:
`uv run python -m thermo_lab.z1_acceptance_kit --output-dir results/z1-acceptance-kit --resume`.

## Persistence and archive size

Request, graph edge lists, references, simple-test tables, collective exact
tables (every missing edge's c0, c1 and counts), per sampled unit the
replicate statistics (float32), rejection counts and scatter ratios, and
the decisions. No sweep traces. Estimated under 1 MB gzipped; above 4 MB the
executor does not commit and asks the owner.

## Gate

`uv run python -m thermo_lab.z1_acceptance_kit --output-dir results/z1-acceptance-kit`.
`completion.json` is written last, after a full replay, and must show
`status=z1_acceptance_kit_complete`, `targets=9`, `references_reproduced=18`,
`chosen_beta=0.6`, `simple_checks=12`, `collective_checks=27`,
`missing_edges_exact` equal to the summed edge count of the nine targets,
and `attribution_checks_passed`, `reference_checks_passed`,
`patch_checks_passed`, `enumeration_checks_passed` and `replayed`, all
`true`. CPU only, single-thread BLAS. CI replays the archive only.

## Stop rules

- A reference, rank-2, finite-difference, patch or enumeration check fails:
  stop before sampling and report. No tolerance is loosened.
- A unit fails to verify on resume: rerun only that unit; twice, stop.
- Wall time above 2x the estimate: checkpoint and report.
- Archive above 4 MB: do not commit; ask the owner.
- No defect, edge, statistic, threshold, temperature or seed changes after
  seeing results.

## Advance, stop or unresolved

- **Advance:** the kit holds. The per-defect table becomes the reference
  for testing a sampled array: simple tests for the defects they win, the
  collective test where it is retained.
- **Stop:** the collective test is not attributable at a size. Its budgets
  at that size are withdrawn; the simple tests stand.
- **Unresolved:** named checks fail; those counts need empirical
  calibration before use.

## What a negative result means

A failed check means the exact count does not deliver the stated error
rates for that test on this simulated sampler. For a simple test that would
point at the runner or the kernel code, since its law is closed form. For a
collective test it is a statement about mixing at beta = 0.6 on these
planar graphs. None of it is about any chip, frustrated targets, the full
degree-16 lattice, or update-process defects (stale reads, colour order),
which only a collective test could see and which the row does not declare.

## Evidence obligations

From `loop/evidence-obligations.json`, reconciled against the current main
snapshot: EO-001, EO-002, EO-004 do not apply (rounding is a defect here,
not a lever). EO-003 (adsorption read-allocation lesson) is adjacent and
stays open; closure evidence missing. EO-005 (amendment 1 evaluation) is
open and unaffected. Source card `extropic-thermalizers` exists and supports
nothing here beyond the kernel convention.
