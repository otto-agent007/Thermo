# Z1 acceptance kit (P-0015): findings

**Verdict: the kit holds except for named checks.** The frozen rule fails
the three collective checks on `L16-greedy-long-s4510`: its ideal sampler
rejected the whole scan in 76 of 1,000 replicates against a limit of 74
(defect 1,000, 1,000 and 968). Under the protocol that target's scan count
is **unresolved**: it needs empirical calibration before use. The arm the
rule lands on did not resolve there, so the operating point to use for a
missing coupling at an unknown location is the **simple matching scan**
(8 reads per matching, 7 to 9 programmings, 56 to 72 reads; check 3 and
955 of 1,000). The collective scan's single programming write is the only
thing it offers over that test.

Two assumptions change how this verdict reads (rule 8 items 2 and 4,
listed in full below). The 7 to 9 programmings hold on the planar
subgraphs; on the full degree-16 lattice a matching scan needs 16. Site and
pair tests detect gain times bias or gain times coupling, not the cause.
One owner-accepted protocol deviation applies to the rank-2 check (below);
no reported number depends on it.

Protocol: [`docs/experiments/z1-acceptance-kit.md`](../../experiments/z1-acceptance-kit.md)
at dc32142 (approved 2026-11-17, PR #174). Runner:
`src/thermo_lab/z1_acceptance_kit.py` at ba3f4f3. Queue row
`z1-acceptance-kit` (a metrology row, not a task row: no chip, no digital
baseline to price, no joule).

## What ran

- Exact part: the 18 P-0001 Kac-Ward references at beta 4 (all bitwise
  equal to the archive); for the nine `greedy-long` targets at beta 0.5,
  0.6 and 0.75, c0 and the missing-edge correlation c1 on every edge
  (23,136 values) by a rank-2 update of the Kac-Ward inverse; on 20 declared
  edges per target and beta, the derivative evaluated by LU on the rebuilt
  matrix (largest difference 4.4e-16, tolerance 1e-9) and the central
  difference (largest difference 2.32e-9, tolerance 1e-4). Applying 1e-9 to
  the LU derivative and 1e-4 to the central difference is a protocol
  deviation, accepted by the owner on 2026-11-17 (note below); brute
  force on 18 small 16-offset patches (1.6e-15); the enumerated two-site
  chain (2.2e-16).
- Simple-test tables in closed form, and the pooled gain-spread count and
  threshold by 20,000-replicate draw-averaged Monte Carlo.
- Calibration: 128 chains per target and candidate beta, 512 burn-in
  sweeps, 8,192-read window, root 20261121; recalibration at 16,384 reads
  on the three 1,024-spin targets.
- Sampled checks: 1,000 replicates per model, root 20261118; ideal plus
  three defect models per collective target; ideal plus defect for each of
  the 14 simple checks.
- Wall time 2 h 13 min 49 s with three workers, 5.7 CPU-h, peak 1.6 GB
  resident per process. The protocol estimated 1 h 15 min and 3.1 CPU-h;
  the 1,024-spin kernel cost about 66 µs per chain-sweep alone and 90 to
  120 µs with three workers, against the 42.7 µs assumed. The 2x
  wall-time stop (2 h 30 min) did not fire.

## Temperature

All three candidates passed the bias check (largest |z| 1.63, 2.27 and
1.69). Largest regret over the nine targets: 1.11 at beta 0.5, 2.26 at 0.6
and 72.3 at 0.75. **The rule picks beta 0.5**, the row's lower bound, as
the probes predicted. At 0.75 the largest scatter ratios reach 10 to 220
and the 1,024-spin scans would need 115,000 to 807,000 reads.

## Missing coupling: collective scan against the matching scan

At beta 0.5 the scan needs 2,624 to 4,013 independent reads
(`exact_reference`) and, with the measured scatter (median ratio 1.16 to
1.23, largest 2.1 to 3.3, `software_simulation`), n_scan of 3,658 to 5,969
at 64 spins, 6,316 to 7,219 at 256 and 9,765 to 11,677 at 1,024. n_scan is
therefore an exact count times a simulated ratio. Per target it is 65 to
182 times that graph's matching-scan reads (65.3 on L8 s4511, 182.5 on L32
s4512), for one programming write instead of 7 to 9. Counting the scan's
512 burn-in sweeps, it is 74 to 190 times the matching scan's elapsed
sweeps, and it needs one state write where the matching scan needs none
(rule 9 table below). Two 1,024-spin scans do not fit the
10,000-read limit at this size (s4510 at 10,494, s4512 at 11,677). The
protocol names that a scientific outcome, and their sampled checks ran at
n_scan and passed.

Ideal family-wise false alarm was 4.4 to 7.6 percent (binomial standard
error about 0.7 points). The round-3 probe measured 4.9 to 5.5 percent on
the six smaller graphs with a different calibration set. L16 s4510 at 7.6
percent and L16 s4512 at 6.7 percent sit above that range.

Attribution passed on every target. The coherent bias bound κ_U is at most
0.025 standard errors, the heterogeneous bound ρ_U at most 0.043, and the
union bound 0.051 to 0.055 against the 0.065 limit. On L16 s4510 the bound
is 0.053, so the excess false alarm is not a bias in the scan's centre.
Post-hoc diagnostic, not a frozen statistic: its 76 false alarms fall on 63
distinct edges, led by edges whose calibration ratio sits below the median
(0.73, 0.66 against 1.16). That fits scatter ratios underestimated by the
128-chain calibration noise on some edges, which the t(127) quantile
corrects only on average. This run does not test that explanation.

Device error and mixing error are reported apart for every declared edge:
exact shifts of 0.095 to 0.461 against ideal sampled-mean errors of at most
0.0013 (L32 s4512 dearest, 2.6 standard errors; every other edge 1.4 or
fewer).

## Simple tests

All 14 checks pass (ideal 0 to 62, defect 937 to 1,000 of 1,000). On the
ideal side 13 scatter ratios are 0.91 to 1.07, as expected for exactly
independent reads; the fourteenth, the known-coupler ideal, is 0.00
because at |J| = 1 and beta 4 its 5-read count did not vary across
replicates (0 rejections). On the defect side the ratios are 0.68 to 6.01.
The ratio is the variance across replicates of the pooled mean, scaled by
reads and pooled units, over 1 minus the squared sample mean. It is 1 only
when every pooled read follows one law independently. A defect breaks that:
pooled pairs or sites with unequal laws, a gain array drawn per replicate,
or reads that never change. 6.01 is the stuck known site: its 6 reads in a
replicate are identical, so the ratio sits near 6. Closed-form counts are in
[summary](summary.md). Draw-averaged pooled gain-spread counts (Monte Carlo
on exact laws, not `exact_reference`): 520, 229 and 108 reads (±10 %) and
59, 27 and 13 (±30 %) at 64, 256 and 1,024 sites. Their Monte Carlo
thresholds are 0.1 to 0.5 percent below the chi-square percentile. The
unfolded draws (round 2's fixed arrays, draw-specific) have power 0.944 to
0.975 at those counts.

## Pre-registered expectations

1. All 18 references reproduce bitwise: **held**.
2. Closed-form counts (6; 12, 14, 16; 5; 8 per matching with 7 to 9
   programmings; 238; 3,368) match exactly: **held**. Pooled counts within
   about 10 percent of 521, 229, 107 and 59, 26, 13: **held** (520, 229,
   108; 59, 27, 13).
3. Smallest defects within 10,000 reads: rounding 8 bits at every size
   (10, 11 and 12 bits need 12,572, 15,509 and 61,981 reads at 1,024
   spins): **held**. Gain spread about ±2.2, ±1.5 and ±1.0 percent: **held**
   (±2.29, ±1.52, ±1.03; Monte Carlo, 20 bisection steps). Missing coupling
   at every coupler and stuck p-bit in both behaviours: **held**.
4. The rule picks beta 0.5: **held**. Every 64- and 256-spin scan near 5
   percent family-wise false alarm: **not held** for L16 s4510 (7.6 %) and
   only loosely for L16 s4512 (6.7 %). Some 1,024-spin scans may exceed
   10,000 reads: **two did**. Every sampled check passes: **not held**,
   three fail (one target). Every collective target attributable: **held**.

## Notes on the implementation

- **Protocol deviation in the rank-2 check, accepted by the owner on
  2026-11-17.** The protocol checks the rank-2 c1 values "against direct
  evaluation on 20 declared edges per target and beta to 1e-9" and stops
  before sampling if a rank-2 or finite-difference check fails. Every probe
  computed c1 by the h = 1e-4 central difference. Against it the 1e-9 check
  fails in 13 of 27 (target, beta) tables: max 2.32e-9 (L32 s4512, beta
  0.75); at the chosen beta 0.5 one table, L32 s4512 at 1.02e-9. Before the
  run the executor read "direct evaluation" as the derivative evaluated by
  LU on the rebuilt Kac-Ward matrix and recorded that reading on the card.
  The runner checked rank-2 against it at 1e-9 (max 4.4e-16) and held the
  central difference to 1e-4, the P-0001 reference tolerance. The stop
  rule as written would have stopped the run before sampling. The gap is
  the central difference's own h² truncation (reviewer check: 4.98e-9,
  1.25e-9 and 3.60e-10 at h = 2e-4, 1e-4 and 5e-5), and no reported number
  depends on the check. The reviewer escalated it; the owner accepted the
  reading as a deviation, with no rerun.
- A collective defect replicate counts as detected when the whole scan
  rejects (any edge above t_α). Rejections at the missing edge itself were
  at most 4 fewer per check and change no decision.
- Site tests program bias 0.5 and beta = 2 x so that beta h = x. The stuck
  defect starts from a random state and never updates; it reads a constant,
  as an ignored write would.
- Pair checks use the maximum matching of `L32-greedy-long-s4510` with all
  other couplers off: 511 pairs for the pooled rounding test (the smallest
  1,024-spin matching) and the full matching for the scan.
- Gain arrays use the declared draw roots 4520 to 4522 folded with the
  replicate index: indices 0 to 19,999 for the Monte Carlo count and 20,000
  onwards for the sampled checks.
- Every chain and replicate has its own generator, so each replicate
  reproduces alone (tested). The kernel is bitwise equal to the round-2
  probe sampler under that probe's draw order (tested). It also reproduces
  the round-3 probe's n_scan of 5,402 on L8 s4510 (tested).

- **Provenance `git_dirty`.** The run-start record in `provenance.json`
  (`attempts[0]`, 2026-11-17T07:07:00Z) has `git_dirty=false` at ba3f4f3,
  and the 10 source digests in `request.json` equal the ba3f4f3 blobs
  (rechecked for this report). The assembly record (`runtime`, written at
  01:10:43 PST) has `git_dirty=true`. The runner sets it from a non-empty
  `git status --porcelain`. What was dirty: while the run was going, the
  executor wrote `docs/experiment-reports/2026-11-17-z1-acceptance-kit/plot.py`
  at 23:58:42 PST. That is recorded in the executor's session log, and the
  report directory's filesystem birth time is the same second. No removal
  of it before assembly appears in the log, so it was untracked at
  assembly. The session log shows no other write in the
  worktree before 01:10:43; the edits to `docs/studies.md` and
  `docs/release-gates.md` came after assembly. No file under `src/`
  or `tests/` changed after ba3f4f3.

## Relaxed or assumed constraints (protocol rule 8)

1. The ideal model has float couplings; it is the reference.
2. Site tests program a site bias, assumed representable at the declared
   value; the row lifts the zero-field restriction for this test only. A
   bias-scale error is indistinguishable from a gain error: a failed site
   test detects gain times bias off nominal, and a failed pooled spread test
   detects gain times bias varying across sites, not which factor. The pair
   test has the same confound with gain times coupling: 8-bit rounding at
   weight 0.2 (a 1.6 percent coupling shift) is indistinguishable from a
   1.6 percent colour-B gain or temperature offset on that pair.
3. Simple tests choose their own program (beta in [0.5, 4], |J| at most 1,
   a bias) inside the declared ranges. The collective scan uses the
   archived couplings at the chosen beta.
4. Planarity restricts the chip's degree-16 rule; the exact collective
   reference exists only on planar graphs. The 7 to 9 matching programmings
   reported here are for these planar subgraphs. On the full degree-16
   lattice a matching scan needs 16 programmings, a planar cover at least
   three.

## Resources per test (protocol rule 9)

Algorithmic counts, not device operations. They come from archived counts
and the protocol's constants; nothing was measured for this table. A read
is the full array after one two-colour sweep. Simple tests start from any
state and need no burn-in or state write (enumerated check), so their
elapsed sweeps equal their reads. The collective scan is one chain: one
state write (random start), 512 burn-in sweeps, then n_scan reads. Node
reads count the node values the statistic uses: reads times the sites, or
pair endpoints, read.

| Test | Elapsed sweeps | Full-array reads | Node reads | Programming writes | State writes |
| --- | --- | --- | --- | --- | --- |
| Stuck p-bit, known site | 6 | 6 | 6 | 1 | 0 |
| Stuck p-bit, scanning 64 / 256 / 1,024 sites | 12 / 14 / 16 | 12 / 14 / 16 | 768 / 3,584 / 16,384 | 1 | 0 |
| Gain spread ±10 %, 64 / 256 / 1,024 sites | 520 / 229 / 108 | same | 33,280 / 58,624 / 110,592 | 1 | 0 |
| Gain spread ±30 %, 64 / 256 / 1,024 sites | 59 / 27 / 13 | same | 3,776 / 6,912 / 13,312 | 1 | 0 |
| Outlier ±10 %, known site; scanning 64 / 256 / 1,024 | 3,368; 6,456 / 7,420 / 8,377 | same | 3,368; 413,184 / 1,899,520 / 8,578,048 | 1 | 0 |
| Outlier ±30 %, known site; scanning 64 / 256 / 1,024 | 469; 893 / 1,026 / 1,160 | same | 469; 57,152 / 262,656 / 1,187,840 | 1 | 0 |
| Missing coupling, known coupler | 5 | 5 | 10 | 1 | 0 |
| Missing coupling, matching scan (8 reads per matching) | 56 to 72 | 56 to 72 | 16 × edges: 1,792 / 7,584 to 7,680 / 31,616 to 31,744 | 7 to 9 | 0 |
| Rounding 8 bits, pooled matching at 1,024 / 256 / 64 spins | 238 / 955 / 3,788 | same | 243,236 / 242,570 / 242,432 | 1 | 0 |

Collective scan against the same graph's matching scan:

| Target | Scan sweeps (512 + n_scan) | Scan reads | Scan node reads | Matching sweeps = reads | Matching node reads | Ratio: reads / sweeps / node reads |
| --- | --- | --- | --- | --- | --- | --- |
| L8 s4510 | 5,965 | 5,453 | 348,992 | 64 | 1,792 | 85.2 / 93.2 / 195 |
| L8 s4511 | 4,170 | 3,658 | 234,112 | 56 | 1,792 | 65.3 / 74.5 / 131 |
| L8 s4512 | 6,481 | 5,969 | 382,016 | 56 | 1,792 | 106.6 / 115.7 / 213 |
| L16 s4510 | 7,127 | 6,615 | 1,693,440 | 64 | 7,680 | 103.4 / 111.4 / 220.5 |
| L16 s4511 | 7,731 | 7,219 | 1,848,064 | 64 | 7,680 | 112.8 / 120.8 / 241 |
| L16 s4512 | 6,828 | 6,316 | 1,616,896 | 72 | 7,584 | 87.7 / 94.8 / 213 |
| L32 s4510 | 11,006 | 10,494 | 10,745,856 | 64 | 31,744 | 164.0 / 172.0 / 339 |
| L32 s4511 | 10,277 | 9,765 | 9,999,360 | 72 | 31,616 | 135.6 / 142.7 / 316 |
| L32 s4512 | 12,189 | 11,677 | 11,957,248 | 64 | 31,712 | 182.5 / 190.5 / 377 |

The scan uses 1 programming write and 1 state write; the matching scan uses
7 to 9 programming writes and none. Scan rows inherit n_scan's labels: an
exact independent-read count times a `software_simulation` scatter ratio.

## What this does not show

Nothing here is measured on hardware or says anything about a chip. The
collective result is specific to these archived couplings (weakest |J| =
0.2), zero field, planar subgraphs and beta 0.5. It does not cover
frustrated targets, the full degree-16 lattice, or update-process defects
(stale reads, colour order), which the row does not declare.

## Archive

`evidence.tar.gz` (858 kB) holds the request, checks, simple tables, the
results record, exact c0 and c1 at every candidate beta, the calibration
ratios at the chosen beta, per-replicate scan statistics and rejection
flags, provenance and completion. The per-edge replicate sums (71 MB) stay
out of the repository; the full replay used them. CI replays the archive
with `uv run pytest tests/unit/test_z1_acceptance_kit.py -k committed`.
