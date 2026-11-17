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
  edges per target and beta, a direct evaluation on the rebuilt matrix
  (largest difference 4.4e-16, tolerance 1e-9) and the central difference
  (largest difference 2.3e-9, tolerance 1e-4; see the note below); brute
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

At beta 0.5 the scan needs 2,624 to 4,013 independent reads and, with the
measured scatter (median ratio 1.16 to 1.23, largest 2.1 to 3.3), n_scan of
3,658 to 5,969 at 64 spins, 6,316 to 7,219 at 256 and 9,765 to 11,677 at
1,024. That is 57 to 182 times the matching scan's 56 to 72 reads, for one
programming write instead of 7 to 9. Two 1,024-spin scans do not fit the
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

All 14 checks pass (ideal 0 to 62, defect 937 to 1,000 of 1,000). Their
scatter ratios are 0.91 to 1.07, as expected for exactly independent reads.
Closed-form counts are in [summary](summary.md). Draw-averaged pooled
gain-spread counts: 520, 229 and 108 reads (±10 %) and 59, 27 and 13
(±30 %) at 64, 256 and 1,024 sites. Their Monte Carlo thresholds are 0.1 to
0.5 percent below the chi-square percentile. The unfolded draws (round 2's
fixed arrays, draw-specific) have power 0.944 to 0.975 at those counts.

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

- **"Direct evaluation" of c1** (protocol, exact anchors). Before the run,
  the h = 1e-4 central difference was measured to differ from the exact
  derivative by up to 1.4e-9 at beta 0.6 and 0.75. That is its own h²
  truncation; the residual is about 1e-10. The runner therefore checks the
  rank-2 values against the derivative evaluated directly on the rebuilt
  matrix (LU, tolerance 1e-9). It keeps the central difference as the
  separate finite-difference check at the P-0001 reference tolerance
  (1e-4). This was recorded on the card before the run; no threshold
  changed.
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
