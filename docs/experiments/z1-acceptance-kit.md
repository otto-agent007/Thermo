# Z1 acceptance kit: protocol (draft, proposal P-0015, round 3)

Redrafted on 2026-11-17 against the queue row `z1-acceptance-kit` as amended
on main (PRs #176 and #179), revised in round 2 after the round-1 review on
PR #174, and in round 3 after the round-2 review (Student-t scan threshold,
attribution limit 0.065, draw-averaged pooled gain count). Not frozen until
the owner approves it; no runner exists yet.
Probes (exploration, not evidence):

- round 0: [`2026-11-15-z1-acceptance-kit-probe.md`](../research/2026-11-15-z1-acceptance-kit-probe.md)
  (reproduce-first check, beta = 4 counts, defect admission on patches);
- review: `docs/research/z1_acceptance_kit_review_probe.py` and its JSON
  (read correlation at beta 4, warmer temperatures);
- round 1: [`2026-11-17-z1-acceptance-kit-probe-r1.md`](../research/2026-11-17-z1-acceptance-kit-probe-r1.md)
  (isolated-pair and single-site tests, temperature choice, pass-rule
  arithmetic);
- round 2: [`2026-11-17-z1-acceptance-kit-probe-r2.md`](../research/2026-11-17-z1-acceptance-kit-probe-r2.md)
  (pooled gain-spread test, sign-agnostic outlier test, attribution margin
  for the Bonferroni scan, temperature on the scan statistic with separate
  calibration chains);
- round 3: [`2026-11-17-z1-acceptance-kit-probe-r3.md`](../research/2026-11-17-z1-acceptance-kit-probe-r3.md)
  (Student-t scan threshold, scan family-wise false alarm on the six 64-
  and 256-spin graphs, draw-averaged pooled gain count and threshold).

## The number this study produces

For each of the row's four defects, **the reads and programming writes of
the cheapest exact test that detects it at 5 percent false alarm and 5
percent miss**, and **the smallest defect of that kind detectable within
10,000 reads**. Two families of test are compared by reads and programming
writes:

- **simple tests**: isolated pairs (one matching of the patch, one coupler
  per pair, pooled over pairs where the defect is shared) and isolated sites
  (one common bias, no couplers, pooled over sites where the defect is
  array-wide), with closed-form exact references;
- **collective tests**: Kac-Ward statistics on the nine P-0001
  `greedy-long` graphs at zero field and one inverse temperature chosen by
  the rule below.

The footing is not symmetric. Simple tests choose their own program (beta,
coupling value, bias) inside the declared ranges. The collective test is
held to the archived couplings, and its weakest couplers (|J| = 0.2) set its
read count. Every collective-versus-simple ratio in this protocol is
specific to these couplings.

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
- **Mechanism.** Three declared defects are faults of one coupler or one
  site; static beta variation is a spread over every site, testable one
  site at a time but pooled across sites per read.
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
2. Site tests program a site bias. The row lifts the zero-field
   restriction for this test only. The bias is assumed representable at the
   declared value. A bias-scale error is indistinguishable from a gain error
   in this test (both enter as the product beta_i h): a failed site test
   detects "gain times bias off nominal", and a failed pooled spread test
   detects "gain times bias varies across sites", not which of the two.
   The pair test has the same confound: it sees gain times coupling, so
   8-bit rounding at weight 0.2 (a 1.6 percent coupling shift) is
   indistinguishable from a 1.6 percent colour-B gain or temperature offset
   on that pair. A failed rounding test names the product, not the cause.
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
- **Static beta variation (primary, as the row defines it).** Every site's
  gain is drawn once per array, g_i ~ uniform(1 − j, 1 + j), j = 0.1 or
  0.3, the round-0 construction (`rng.uniform(1 - jitter, 1 + jitter, n)`).
  The defect is the spread law, not one array: counts, thresholds and
  sampled checks are averaged over gain draws, each replicate drawing its
  own array (64, 256 or 1,024 sites) from the declared draw seed roots.
  Test: all sites isolated, one common bias, every read pooled over all
  sites. The statistic is Pearson's homogeneity X² over sites, sum_i (k_i −
  n p̂)² / (n p̂ (1 − p̂)), with p̂ the pooled up fraction, rejected above a
  threshold set by Monte Carlo under the ideal model at the same count
  (below). It answers "do the sites' gains differ", and it is blind by
  construction to a common gain or temperature offset, which moves every
  site alike.
- **Static beta variation (secondary).** One outlying site at gain 1 ± δ,
  δ = 0.1 or 0.3, sign unknown. Test: that site alone, one bias for both
  signs, exact two-sided binomial at 2.5 percent per tail, 5 percent miss
  required at both 1 + δ and 1 − δ. Reported at a known site and scanning
  every site (Bonferroni over sites). It does not measure the array's
  spread: under a uniform ±10 percent spread almost no site sits at the
  extreme.

Neither gain test has a collective counterpart. The collective graph has no
exact reference (the stationary law with site gains is not an Ising law;
round-0 probe finding 7), and the pair product sees only the colour-B
site's gain (enumerated check).

## Chosen temperature for the collective tests

Round 1 chose beta 0.6 on the 6- and 8-bit rounding LLRs, which are no
longer in the kit, without measuring the scan's own read correlation. Round
2 replaces that choice with a rule on the retained statistic, run by the
runner as a calibration stage before any tested chain exists.

**The scan statistic.** For edge e, the mean edge product s_i s_j over the
scan's n reads, tested one-sided against the exact ideal correlation c0_e
(a missing coupling moves it towards c1_e, the exact correlation with that
coupling set to zero), at false alarm 0.05 / E per edge, E the target's edge
count. One scan uses one read count n_scan for all edges of a target. Edge
e's standardized statistic is (sign_e (c0_e − mean_e)) / se_e with se_e =
sqrt(r_e (1 − c0_e²) / n_scan), and it is rejected above t_α, the
Student-t quantile with 127 degrees of freedom at upper tail 0.05 / E.

**Why a t quantile.** se_e uses a scatter ratio r_e estimated from 128
calibration chains (about 12.5 percent relative noise in the variance), so
the standardized statistic is approximately t with 127 degrees of freedom,
not normal. Round 2 used the normal quantile z_α = Φ̄⁻¹(0.05 / E). At a
0.05 / E tail the ratio noise then raises an unbiased sampler's
family-wise false alarm to about 7 to 8 percent (round-2 review; round-3
probe: 7.1 percent at L8 s4510 on 4,000 replicates, table below). t_α is
3.40, 3.82 and 4.20 at E = 112, 480 and 1,984, against 3.32, 3.71 and 4.05.
The scatter ratios are not re-estimated with more chains; the t quantile is
the whole correction. The miss quantile stays the normal z_β = 1.645: the
5 percent miss side is far less sensitive to the ratio noise than the
0.05 / E tail.

**Calibration chains.** Per target and candidate beta: 128 chains from a
random start, 512 burn-in sweeps, then a window of exactly 8,192 reads, on a
calibration seed root disjoint from every tested chain. For every edge the
scatter ratio r_e is the variance across chains of the window mean, times
8,192, over the exact per-read variance 1 − c0_e². The per-edge count is
n*_e = ceil(n_ind_e × r_e), where n_ind_e = ((t_α sqrt(1 − c0_e²) + z_β
sqrt(1 − c1_e²)) / |c0_e − c1_e|)² is the independent-read count at
0.05 / E and 5 percent miss with the t quantile on the false-alarm side.
The target's scan length is n_scan = max_e n*_e. If n_scan exceeds 8,192 at
the chosen beta, the scatter ratios are measured once more on 128 fresh
calibration chains at a window of 16,384 reads (still 127 degrees of
freedom), and n_scan is recomputed from those. The tested chains then use
n_scan and the calibration r_e unchanged. A target whose n_scan exceeds
10,000 is reported as "the scan does not fit the 10,000-read limit at this
size", a scientific outcome, not a failure of the kit or of the run; its
sampled checks still run at n_scan. No 1,024-spin graph has been
previewed, and this outcome is plausible there (256-spin graphs already
need 6,285 to 7,811 reads with the t quantile).

**Rule.** Candidates beta ∈ {0.5, 0.6, 0.75}. The row bounds the choice to
[0.5, 4]; 1.0 is dropped because the probe found its scan needs 29,000 to
8 million reads on every 64- and 256-spin graph. A candidate is admitted
when, on every target, the calibration chains' energy bias and the mean over
edges of the standardized edge-product bias both have |z| ≤ 3.5. Among
admitted candidates, take the beta minimising the largest ratio over the
nine targets of scan sweeps (512 + n_scan) to that target's best over the
grid. The chosen beta is recorded in `completion.json`; it is not fixed in
advance, and every collective exact reference, sampled check and count is
computed at it.

**Probe preview (64 and 256 spins only; exploration).** The round-2 probe
applied this rule to the six smaller graphs at a 4,096-read window over
{0.5, 0.6, 0.75, 1.0}, with exact c0_e and c1_e for every edge. It picks
**beta 0.5**: largest regret 1.07, against 1.59 at 0.6, 9.1 at 0.75 and
over 1,000 at 1.0. At 0.5 the scan needs 2,541 to 3,062 independent reads,
median per-edge scatter ratio 1.14 to 1.20 and largest 2.2 to 2.9, so
n_scan = 3,671 to 5,554 at 64 spins and 6,034 to 7,499 at 256 with the
normal quantile. Recomputed with the t quantile on the same chains (round-3
probe, the round-2 values reproduced bitwise first), n_scan = 3,791 to
5,734 at 64 spins and 6,285 to 7,811 at 256 (2,624 to 3,189 independent),
3.2 and 4.2 percent longer. At 0.6 it
needs 3,411 to 7,093 and 8,449 to 12,027: one 256-spin graph exceeds the
10,000-read limit. Round 1's choice of 0.6 does not survive on the
retained statistic. The optimum sits at the row's lower bound; a warmer
beta might do better, but the row does not allow it, and this protocol does
not change the row. Beta 0.5 stays the lower bound in round 3, including at
1,024 spins. All candidates passed the bias check except 1.0
(largest |z| 3.7). The 1,024-spin graphs were not run: their all-edge c1
needs the runner's rank-2 update. The runner applies the rule on all nine
graphs. Because n_scan exceeded 4,096 on every previewed graph at 0.5, the
runner's first calibration window is 8,192 reads, not 4,096; the
recalibration step above then applies only if n_scan exceeds 8,192.

Read correlation (amendment item 2) is therefore measured on separate
chains, on the scan statistic itself, at a stated window. The round-2 probe
ran a second, independent chain set at the same window as a stand-in for the
tested chains. At 0.5 its n_scan was within 6 percent of the calibration
value on all six graphs. Per-edge scatter ratios differed by 17 to 18
percent rms in log, which is the sampling error expected from 128 chains,
so the edge-to-edge spread of r_e is mostly noise. Taking the maximum of
noisy n*_e over edges inflates n_scan. That errs towards more reads, and
it is kept. Beta = 4 is used only for the reproduce-first check of the 18
P-0001 references.

**Scan false alarm under an unbiased sampler (round-3 probe).** The
protocol's procedure at beta 0.5 on the six previewed graphs: 128 fresh
calibration chains (512 burn-in, 8,192-read window), then fresh tested
ideal chains at n_scan, family-wise false alarm of the whole scan
(`software_simulation`, NumPy float64, exact Kac-Ward c0 and c1). The
normal scan uses the first n_scan(normal) reads of the same tested chains.

| Graph | Replicates | n_scan, normal / t | False alarm, normal z_α | False alarm, t(127) |
| --- | --- | --- | --- | --- |
| L8 s4510 | 4,000 | 6,221 / 6,423 | 7.1 % | 5.5 % |
| L8 s4511 | 4,000 | 3,627 / 3,745 | 6.8 % | 5.0 % |
| L8 s4512 | 4,000 | 4,587 / 4,736 | 6.2 % | 4.9 % |
| L16 s4510 | 2,000 | 5,720 / 5,958 | 7.3 % | 4.9 % |
| L16 s4511 | 2,000 | 7,280 / 7,582 | 7.7 % | 5.2 % |
| L16 s4512 | 2,000 | 6,447 / 6,713 | 7.7 % | 5.1 % |

Binomial standard errors are 0.34 to 0.41 points at 4,000 replicates and
0.48 to 0.59 at 2,000. The t threshold brings every graph to 4.9 to 5.5
percent. Against the 74-of-1,000 check, a scan whose true false alarm is
5.5 percent fails with probability 0.005; at 7.7 percent it would fail
with probability about 0.6. This calibration set's n_scan differs from
round 2's on the same graphs by up to 19 percent (6,221 against 5,232 at
L8 s4510, normal), the noise of a maximum over edges noted above. 1,024
spins was not simulated.

## Which test each defect gets (from the round-1 to round-3 probes)

Reads are for independent reads at 5 percent false alarm and 5 percent
miss. Simple-test reads are exactly independent. Collective counts are
multiplied by the measured scatter ratio before use (n*).

| Defect | Simple test: reads, programming writes | Collective test | In the kit |
| --- | --- | --- | --- |
| Stuck p-bit (either behaviour) | Unbiased isolated site: 6 at a known site; 12, 14, 16 scanning 64, 256, 1,024 sites. 1 programming, 0 state writes | Same counts, but every read is a fresh chain: a state write, 512 burn-in sweeps and a fair-coin start each | Simple only |
| Gain spread, uniform ±10 percent (row definition) | All sites at one bias, pooled homogeneity X², count and threshold by draw-averaged Monte Carlo: 521, 229, 107 reads at 64, 256, 1,024 sites (beta h 1.15 to 1.2; round-3 probe). 1 programming | No exact reference | Simple only |
| Gain spread, uniform ±30 percent | 59, 26, 13 reads at 64, 256, 1,024 sites | No exact reference | Simple only |
| Single outlying site at 1 ± 10 percent, sign unknown (secondary) | One bias, two-sided exact: 3,368 at a known site; 6,456, 7,420, 8,377 scanning 64, 256, 1,024 sites. 1 programming | No exact reference | Simple only |
| Single outlying site at 1 ± 30 percent (secondary) | 469 at a known site; 893, 1,026, 1,160 scanning | No exact reference | Simple only |
| Missing coupling, location known | Pair at \|J\| = 1, beta 4: 5 reads, 1 programming | Own edge at beta 0.6: 35 to 928 independent reads (round 1, 64 and 256 spins) | Simple only |
| Missing coupling, location unknown | 8 reads per matching, 7 to 9 matchings: 56 to 72 reads, 7 to 9 programmings | Scan over all edges, 1 programming. Preview at beta 0.5 with the t(127) threshold: 3,791 to 5,734 reads at 64 spins and 6,285 to 7,811 at 256 with the measured scatter (2,624 to 3,189 independent); 1,024 left to the runner, and it may not fit the 10,000-read limit. Round 2's all-edge figure at beta 0.6 is 1,873 to 2,418 independent reads over all edges (normal approximation; the review's exact-binomial range is 1,906 to 2,464) | Both: simple for reads, collective for writes |
| Rounding, 8 bits | Matching at weight 0.2, beta 4: 238, 955, 3,788 reads at 1,024, 256, 64 spins. 1 programming | LLR at beta 0.6 (round 1): 3,738 to 3,785 at 1,024 spins (scatter 1.35 to 1.61), 15,288 to 16,317 at 256, 73,881 to 83,235 at 64 | Simple only |

Pooled gain-spread counts were first set by the noncentral chi-square
approximation on the exact per-site laws (round 2: 503, 226, 107 and 55,
25, 12). The runner instead sets each count averaged over gain draws: the
smallest n whose power, by 20,000-replicate Monte Carlo of the exact
per-site binomial laws with a fresh uniform draw per replicate, is at least
95 percent, with the rejection threshold at that n taken from the same
Monte Carlo under the ideal model (the 95th-percentile order statistic of
20,000 ideal replicates, false alarm at most 5 percent), not from the
chi-square table. Both are labelled "Monte Carlo on exact laws", not
`exact_reference`. The round-3 probe ran this procedure on a probe seed:

| Spread | Sites | Approximation | Draw-averaged Monte Carlo count | Power over individual draws, 5th percentile / minimum of 200 | Round 2's fixed draw |
| --- | --- | --- | --- | --- | --- |
| ±10 % | 64 | 503 | 521 | 0.89 / 0.80 | 0.97 |
| ±10 % | 256 | 226 | 229 | 0.93 / 0.90 | 0.95 |
| ±10 % | 1,024 | 107 | 107 | 0.93 / 0.92 | 0.96 |
| ±30 % | 64 | 55 | 59 | 0.89 / 0.82 | 0.97 |
| ±30 % | 256 | 25 | 26 | 0.92 / 0.89 | 0.95 |
| ±30 % | 1,024 | 12 | 13 | 0.94 / 0.93 | 0.98 |

The count is a property of the spread law, not of one array: a particular
64-site array can be detected with power as low as 0.80 at the averaged
count, and a fixed-draw count (round 2) would be specific to that draw. The
Monte Carlo threshold differs from the chi-square percentile by under 0.3
percent; at 13 reads on 1,024 sites the chi-square threshold's false alarm
is 4.5 percent, so the Monte Carlo threshold is slightly less
conservative there. The smallest uniform spread detectable within 10,000
reads is about ±2.2, ±1.5 and ±1.0 percent at 64, 256 and 1,024 sites
(approximation; the runner reports the Monte Carlo value).

So the collective Kac-Ward test stays for one case only: scanning for a
missing coupling at an unknown location with one programming write instead
of 7 to 9, at about 50 to 140 times the reads of the matching scan on these
couplings (64 and 256 spins, beta 0.5 preview, t threshold). It sees nothing declared that the simple tests miss. What only a
collective test could see is the update process (stale neighbour reads,
colour order). The row does not declare those defects, and this protocol
does not add them.

## Pre-registered expectations (from the probes)

1. All 18 P-0001 references reproduce bitwise at beta = 4.
2. Simple-test counts match the round-1 and round-2 probe JSON exactly
   where closed form: stuck p-bit 6 reads at a known site and 12, 14, 16
   scanning all sites; missing coupling 5 reads at a known coupler and 8
   reads per matching (7 to 9 programmings) scanning; 8-bit rounding 238
   reads at 1,024 spins; outlying site at ±10 percent 3,368 reads at a known
   site (two-sided exact). Pooled gain-spread counts and thresholds are set
   by draw-averaged Monte Carlo and are expected within about 10 percent of
   the round-3 probe's 521, 229, 107 (±10 percent) and 59, 26, 13 (±30
   percent).
3. The smallest defect detectable within 10,000 reads, over every depth
   computed (4, 6, 7, 8, 10, 11 and 12 bits; 5 and 9 carry no defect):
   - rounding: 8 bits at every size with the pooled pair test. 10, 11 and 12
     bits need 12,572, 15,509 and 61,981 reads at 1,024 spins. The
     collective LLR is not in the kit (round-1 figures at beta 0.6 for
     reference only);
   - gain spread (pooled): about ±2.2, ±1.5 and ±1.0 percent uniform spread
     at 64, 256 and 1,024 sites;
   - outlying site (secondary): reported at a known site and scanning;
   - missing coupling: every coupler (pair test);
   - stuck p-bit: both behaviours, at any site.
4. The rule picks beta 0.5. With the t(127) threshold every 64- and
   256-spin scan sits near its 5 percent family-wise false alarm (probe:
   4.9 to 5.5 percent). The 1,024-spin scans are unpreviewed; any whose
   n_scan exceeds 10,000 is reported as not fitting the limit, an outcome,
   not a failure. Every sampled check passes its
   operating-characteristic rule, and every collective target passes its
   attribution check.

## Exact anchors and evidence classes

- Simple tests: closed-form two-state laws. P(pair product = sign J) =
  sigmoid(2 beta |J|); P(site = +1) = sigmoid(2 beta g h); a free unbiased
  site is a fair coin per read. Independence of consecutive reads under the
  two-colour sweep is checked by enumerating the two-site chain, to 1e-12.
  Counts use exact binomial error rates; pooled pairs are independent
  trials because the pairs are disconnected. Pooled site reads are
  independent across sites for the same reason; the pooled X² count is a
  Monte Carlo evaluation of exact per-site laws, labelled as such.
- Collective tests: the Kac-Ward determinant on the explicit embedding,
  imported unchanged from `planar_16_offset_ferro`. A missing edge's
  correlation c1_e is d ln Z / dK_e at K_e = 0 by central difference; all
  edges at every candidate beta by rank-2 update, checked against direct
  evaluation on 20 declared edges per target and beta to 1e-9. The scan's
  per-read variance is 1 − c², exact for a ±1 product.

Means, shifts, per-read variances and independent-read counts are
`exact_reference`; normal-approximation counts are labelled as such.
Scatter ratios, sampled means and operating characteristics are
`software_simulation` (NumPy float64, CPU). Nothing runs on hardware.

## Arms and sampled checks

- **Simple tests** (run on the same two-colour kernel code, with all other
  couplers off): stuck p-bit at a known site and scanning 1,024 sites; gain
  spread j = 0.1 and 0.3, pooled over 1,024 sites with a fresh gain draw
  per replicate, at the best beta h; outlying site at 1 + δ and 1 − δ (two defect models, one
  bias, two-sided test) for δ = 0.1 and 0.3 at a known site; missing
  coupling at a known coupler and one scanned matching at 1,024 spins;
  rounding at 4, 6, 7 and 8 bits pooled over a 1,024-spin matching at the
  probe's best (beta, weight). 14 checks, 1,000 replicates each.
- **Collective missing coupling** at the chosen beta, as the scan over every
  edge with Bonferroni false alarm 0.05 / E and the calibration n_scan: per
  target, three declared edges (cheapest, median and dearest by n*_e). 27
  checks, 1,000 replicates, ideal chains shared across the three edges of a
  target, on a seed root disjoint from the calibration chains. The ideal
  side measures the family-wise false alarm of the whole scan.
- **Control**: the ideal model, with false alarm measured at every check's
  read count and threshold.
- **Excluded**: collective tests for rounding, stuck p-bit and gain error
  (the simple test needs fewer reads and no more writes, or the collective
  test has no exact reference).

The count for a collective check is n_scan from the calibration chains.
Simple checks use the exact (or, for the pooled spread test, Monte Carlo)
count; their scatter ratio is 1 by construction, measured anyway and
reported.

## Primary decision rule (fixed in advance)

Number of sampled checks m = 41, each tested on two sides. Per side the
tolerance is Bonferroni at 0.05 / (2m), so the chance that any correctly
calibrated check fails by luck is at most 5 percent over the whole study
(amendment item 5). Every check has 1,000 replicates, and a check fails if
the ideal sampler rejects in more than 74 of them or the defective sampler
in fewer than 926.
The family-wise false-failure bound is 3.3 percent. These thresholds still
flag a test whose true false alarm is 10 percent with probability 0.997,
and one at 8 percent with probability 0.74. At 400 replicates the same
multiplicity would give thresholds of 35 and 365 and only 0.77 power at 10
percent, which is why 1,000 are used. The round-0 rule (31 and 369 of 400
per check, about 54 checks) had a 30 percent chance of at least one false
failure on the false-alarm side alone and 52 percent on both sides.

**Attribution check** per collective target, before its checks count. It is
sized for the Bonferroni scan, not for a single 5 percent test. Round 1's
margin (|bias| at most 5 percent of the shift) would let a bias of one sign
across edges, which under-mixing from a random start plausibly produces and
which pushes towards "missing", raise the scan's family-wise false alarm to
12, 14 and 16 percent at E = 112, 480 and 1,984.

- **Unit: per edge.** Each edge's bias is measured in that edge's own
  standard error at the common scan length, se_e = sqrt(r_e (1 − c0_e²) /
  n_scan), with r_e from the calibration chains. The admitted shift is
  therefore per edge, never pooled over the target: the dearest edge sets
  n_scan and has shift (t_α + 1.645) se_e, t_α the t(127) quantile at
  0.05 / E as in the scan; every other edge has a larger shift in its own
  units, so the same margin is a smaller fraction of its shift.
- **Estimates.** From the tested ideal replicates (R = 1,000), b_e = (pooled
  ideal mean of edge e − c0_e) / se_e, signed so that positive points
  towards "missing". Coherent part κ = mean over edges of b_e. Heterogeneous
  part ρ² = mean over edges of (b_e − κ)², minus the mean sampling variance
  of b_e. Upper confidence bounds κ_U and ρ_U, each at 0.025 / 9, by a
  bootstrap over replicates (2,000 resamples, fixed seed), so edge
  correlation is carried by the replicate unit.
- **Rule.** The target is attributable if E × T̄₁₂₇((t_α − κ_U) / sqrt(1 +
  ρ_U²)) ≤ 0.065, T̄₁₂₇ the t(127) upper tail: the union bound on the scan's
  family-wise false alarm under those biases, in the same t units as the
  scan. With ρ = 0 this admits κ up to 0.079, 0.073 and 0.068 standard
  errors at E = 112, 480 and 1,984: 1.6, 1.3 and 1.2 percent of the
  dearest edge's shift. A random-sign spread of ρ = 0.1 alone costs 0.3 to
  0.4 points.
- **Why 0.065, not round 2's 0.06.** The union bound gives exactly 0.05 at
  κ = ρ = 0, but the scan's measured family-wise false alarm under an
  unbiased sampler with the t threshold is 5.2 to 5.8 percent at 64 spins
  on L8 s4511 (round-2 review, three calibration sets) and 4.9 to 5.5
  percent on the six 64- and 256-spin graphs (round-3 probe, table above).
  The tail of a ±1
  product at 0.05 / E is not exactly normal, so the scan sits slightly
  above nominal even with accurate ratios. That excess is not a bias the
  rule should spend: at a 0.06 limit it would leave 0.2 to 1.1 points for
  κ and ρ together. 0.065 keeps 0.7 to 1.6 points of room above the
  measured unbiased level. The limit bounds attributable bias; it does not
  change the scan's own false-alarm target of 0.05, and the sampled
  operating-characteristic check (74 of 1,000) still applies to every
  scan.
- **Precision.** In the round-2 calibration chains at beta 0.5, the standard
  deviation over chains of the edge-averaged standardized deviation was
  about 0.2 at 64 spins and 0.1 at 256 (independent-read units). With R =
  1,000 that gives SE(κ) of about 0.006 and 0.003, so κ_U ≈ |κ| + 0.018 and
  |κ| + 0.009. With the 0.065 limit the check is reachable if the true
  coherent bias is below about 0.06 standard errors at 64 spins and 0.064
  at 256 (round 2: about 0.03). The calibration chains found no coherent
  bias (|z| ≤ 2.1 on all six graphs) but cannot show it is that small. A
  margin of 1.2 to 1.6 percent of the shift is still close to what 1,000
  replicates resolve. A target that fails is reported as
  unattributable "for bias" (κ̂ itself exceeds the admitted value) or "for
  precision" (only the bound does).

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

Graph and coupling seeds 4510, 4511, 4512 are target identities. Gain-spread
draws (the defective arrays) use draw seed roots 4520, 4521, 4522 for 64,
256 and 1,024 sites, for both j, folded with the replicate index. The
unfolded draw from each root (round 2's fixed defective array) is reported
separately with its own power, labelled draw-specific. Calibration chains use root 20261121, the pooled
Monte Carlo count and threshold root 20261122, and tested chains root
20261118; all are fresh (the probes used 20261115, 20261117, 20261119,
20261120, 20261123, 20261124, 20261125 and the review's seeds) and each is
folded with arm, target, beta, model and replicate.

## Budgets and CPU estimate

Scaled from round-1 and round-2 timings (42.7 µs per chain-sweep at 1,024
spins, about 7 µs at 256; Kac-Ward 13.6 s with correlations and 2.8 s for
ln Z alone at 1,024 spins, 0.054 s at 256), assuming n_scan of 8,000 to
10,500 at 1,024 spins (extrapolated, not measured; the t quantile adds
about 5 percent at E ≈ 1,980). A larger n_scan there is a reportable
outcome, and it lengthens the 1,024-spin collective checks in proportion;
the wall-time stop rule below applies.

- Exact part, about 35 CPU-minutes: references at beta 4 and at the three
  candidates; every missing edge by rank-2 update at each candidate, with
  20 direct checks per target and beta (about 2 CPU-minutes each at 1,024
  spins); simple-test tables and pooled Monte Carlo counts.
- Calibration chains: about 12 CPU-minutes (3 candidates × 9 targets × 128
  chains × 8,704 sweeps), plus one 16,384-read recalibration where needed.
- Simple sampled checks: under 5 CPU-minutes.
- Collective sampled checks: about 1.9 CPU-hours, mostly the 1,024-spin
  targets (1,000 replicates × (512 + n_scan) sweeps per model, one ideal
  and three defect models per target; about 30 CPU-minutes per 1,024-spin
  target at n_scan 10,500).
- Replay: about 10 minutes, no sweeps.

**Total about 3.1 CPU-hours**, about 1 h 15 min wall with three workers.
Peak memory about 1 GiB per worker: the round-2 probe peaked at 0.31 GiB
accumulating per-edge window sums at 256 spins, and the 1,024-spin rank-2
update holds a dense inverse of the Kac-Ward matrix (about 0.25 GiB complex
at 2E ≈ 3,970) plus work arrays. Round 1's 1.6 GiB peak came from storing
full statistic traces, which the runner does not do.

## Autosave and resume

Over 30 minutes, so it follows the autosave contract in
`docs/experiment-runner.md`. The work unit is one (arm, target, model)
group, and one (target, beta) calibration set. Units are written atomically under `units/` with a digest index.
`--resume` re-authenticates the request and sources, verifies every
completed unit and runs only the missing ones. Interruption test: kill after
at least one sampled unit, resume, require bitwise equality with an
uninterrupted run of the same small request.

Resume command:
`uv run python -m thermo_lab.z1_acceptance_kit --output-dir results/z1-acceptance-kit --resume`.

## Persistence and archive size

Request, graph edge lists, gain draws, references, simple-test tables,
collective exact tables (every missing edge's c0, c1 and counts at each
candidate beta), per calibration set the per-edge scatter ratios (float32)
and bias summaries, per sampled unit the replicate statistics (float32),
rejection counts and scatter ratios, the attribution bootstrap summaries,
and the decisions. No sweep traces. Estimated under 2 MB gzipped (per-edge
collective replicate means are kept only as their bootstrap summaries);
above 4 MB the executor does not commit and asks the owner.

## Gate

`uv run python -m thermo_lab.z1_acceptance_kit --output-dir results/z1-acceptance-kit`.
`completion.json` is written last, after a full replay, and must show
`status=z1_acceptance_kit_complete`, `targets=9`, `references_reproduced=18`,
`chosen_beta` (the rule's pick, with `candidate_betas=[0.5, 0.6, 0.75]`),
`scan_quantile="student_t_127"`, `attribution_limit=0.065`,
`gain_spread_count="draw_averaged_monte_carlo"`,
`calibration_seed_root_disjoint=true`, `simple_checks=14`,
`collective_checks=27`, `missing_edges_exact` equal to three times the
summed edge count of the nine targets (one per candidate beta), and
`attribution_checks_passed`, `reference_checks_passed`,
`patch_checks_passed`, `enumeration_checks_passed` and `replayed`, all
`true`. CPU only, single-thread BLAS. CI replays the archive only.

## Stop rules

- A reference, rank-2, finite-difference, patch or enumeration check fails:
  stop before sampling and report. No tolerance is loosened.
- A unit fails to verify on resume: rerun only that unit; twice, stop.
- Wall time above 2x the estimate: checkpoint and report.
- Archive above 4 MB: do not commit; ask the owner.
- No defect, edge, statistic, threshold, candidate grid, selection rule or
  seed changes after seeing results. The temperature is whatever the
  declared rule picks on the calibration chains.

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
collective test it is a statement about mixing at the chosen beta on these
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
