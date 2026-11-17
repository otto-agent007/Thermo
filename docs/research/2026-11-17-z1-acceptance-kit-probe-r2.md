# Z1 acceptance kit: round-2 probe (proposal P-0015)

**Exploration, not evidence.** Round-2 probe for queue row
`z1-acceptance-kit`, answering the round-1 review on PR #174. Run on
2026-11-16 (local time) on the i7-7700K, CPU only, single-thread BLAS.
Script: [`z1_acceptance_kit_probe_r2.py`](z1_acceptance_kit_probe_r2.py).
Output: [`2026-11-17-z1-acceptance-kit-probe-r2.json`](2026-11-17-z1-acceptance-kit-probe-r2.json).
CPU: 12.5 min user time for the recorded run, plus 0.3 min for a smoke run
of parts G and A: about 13 CPU-minutes against the 30-minute ceiling. Peak
RSS 0.31 GiB. Protocol:
[`docs/experiments/z1-acceptance-kit.md`](../experiments/z1-acceptance-kit.md).

Evidence classes. Per-site and per-pair laws, Kac-Ward c0 and c1, and
independent-read counts are `exact_reference`; pooled spread counts use the
noncentral chi-square approximation on exact laws; scan counts use the
normal approximation on exact moments. Scatter ratios, biases and the
pooled-test simulation are `software_simulation` (NumPy float64). Nothing
ran on hardware.

## Findings

1. **Gain spread, as the row defines it, is cheap to detect pooled.** All
   sites isolated at one bias, every site's gain uniform in [1 − j, 1 + j],
   Pearson homogeneity X² over sites per read set. At j = 0.1 it needs 503,
   226 and 107 reads at 64, 256 and 1,024 sites (beta h 1.15 to 1.2); at
   j = 0.3, 55, 25 and 12. Reads fall as one over the square root of the
   site count. The smallest spread detectable within 10,000 reads is about
   ±2.2, ±1.5 and ±1.0 percent. The statistic ignores a common gain or
   temperature offset by construction.
2. **The approximation is slightly optimistic.** Simulating 2,000
   replicates at each count, with a fresh gain draw per replicate, gave
   false alarm 3.7 to 5.5 percent and power 92.6 to 94.7 percent. The
   protocol therefore sets the runner's count by Monte Carlo on each fixed
   declared draw.
3. **A single outlying site, sign-agnostic, one bias: about 3,400 reads.**
   Exact two-sided binomial at 2.5 percent per tail with 5 percent miss at
   both 1 + δ and 1 − δ: 3,368 reads at a known site for δ = 0.1 (beta h
   1.15) and 469 for δ = 0.3 (beta h 1.0). Scanning 64, 256 and 1,024 sites:
   6,456, 7,420, 8,377 and 893, 1,026, 1,160. The review estimated about
   3,340 and 447. This test does not measure the array's spread.
4. **Attribution margin for the Bonferroni scan.** With per-edge false
   alarm 0.05 / E, a coherent bias of 5 percent of the shift gives a
   family-wise false alarm of 11.8, 13.9 and 16.3 percent at E = 112, 480
   and 1,984 (union bound). Holding it at 6 percent allows 0.051, 0.046 and
   0.043 per-edge standard errors at the scan length: 1.0, 0.87 and 0.75
   percent of the dearest edge's shift. A random-sign rms bias of 0.1
   standard errors alone gives 5.3 to 5.4 percent; 0.2 gives 6.3 to 7.0.
5. **Temperature on the scan statistic picks beta 0.5, not 0.6.** On the
   six 64- and 256-spin graphs, with exact c0 and c1 for every edge and
   per-edge scatter ratios from 128 calibration chains (512 burn-in, 4,096
   reads), the scan sweeps (512 + max over edges of independent reads ×
   scatter) give a largest regret of 1.07 at 0.5, 1.59 at 0.6, 9.1 at 0.75
   and over 1,000 at 1.0. At 0.5: 2,541 to 3,062 independent reads, median
   scatter 1.14 to 1.20, largest 2.2 to 2.9, n_scan 3,671 to 5,554 (64
   spins) and 6,034 to 7,499 (256). At 0.6: n_scan 3,411 to 7,093 and 8,449
   to 12,027. One 256-spin graph exceeds 10,000 at 0.6. The rule picks the
   row's lower bound; nothing below 0.5 was run.
6. **Calibration transfers to independent chains.** A second chain set on
   a disjoint seed gave n_scan within 6 percent of the calibration value at
   0.5 on all six graphs. Per-edge scatter ratios differed by 17 to 18
   percent rms in log, the sampling error of 128 chains. The edge-averaged
   standardized bias of the calibration chains had |z| ≤ 2.1 at 0.5 and 0.6
   (3.7 at 1.0).
7. **All-edge independent counts at 0.6** are 1,873 to 2,418 (normal
   approximation), against the review's exact-binomial 1,906 to 2,464.

## Not done

- No 1,024-spin graph in part B: all-edge c1 by finite differences would
  take about 3 CPU-hours per graph and beta. The runner uses a rank-2
  update.
- No beta below 0.5: the row bounds the choice to [0.5, 4].
- The attribution precision is estimated from calibration chains, not from
  1,000 tested replicates.
