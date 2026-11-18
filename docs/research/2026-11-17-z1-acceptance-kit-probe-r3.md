# Z1 acceptance kit: round-3 probe (proposal P-0015)

**Exploration, not evidence.** Round-3 probe for queue row
`z1-acceptance-kit`, answering the round-2 review on PR #174. Run on
2026-11-16 (local time) on the i7-7700K, CPU only, single-thread BLAS.
Script: [`z1_acceptance_kit_probe_r3.py`](z1_acceptance_kit_probe_r3.py).
Output: [`2026-11-17-z1-acceptance-kit-probe-r3.json`](2026-11-17-z1-acceptance-kit-probe-r3.json)
(two invocations, `--parts t,r` and `--parts g,f`, merged). CPU: 2.8 + 16.9
= 19.7 CPU-minutes against the 30-minute ceiling. Peak RSS 0.73 GiB.
Protocol: [`docs/experiments/z1-acceptance-kit.md`](../experiments/z1-acceptance-kit.md).

Evidence classes. t and normal quantiles, union-bound arithmetic, Kac-Ward
c0 and c1 are exact. Scatter ratios, scan false alarms and the pooled gain
Monte Carlo are `software_simulation` (NumPy float64). Nothing ran on
hardware.

## Findings

1. **t(127) quantiles.** At 0.05 / E: 3.40, 3.82 and 4.20 at E = 112, 480
   and 1,984, against normal 3.32, 3.71 and 4.05. Scan length factor with
   equal variances: 1.033, 1.042, 1.051.
2. **Round-2 scan lengths reproduce bitwise and grow 3.2 to 4.2 percent.**
   Rerunning round 2's beta 0.5 calibration chains (same seeds, 4,096-read
   window) reproduced all six n_scan values exactly. With t: 3,791 to
   5,734 at 64 spins, 6,285 to 7,811 at 256.
3. **The t threshold fixes the scan's false alarm.** Protocol procedure at
   beta 0.5 (128 fresh calibration chains, 8,192-read window, fresh tested
   chains at n_scan): family-wise false alarm 6.2 to 7.7 percent with the
   normal quantile, 4.9 to 5.5 percent with t(127), on all six 64- and
   256-spin graphs (4,000 replicates at 64 spins, 2,000 at 256). Matches the
   review's 7.0 to 7.9 and 5.2 to 5.8 percent.
4. **Attribution limit 0.065.** With the t tail, the union bound admits a
   coherent bias κ of 0.079, 0.073 and 0.068 standard errors at E = 112,
   480 and 1,984 (1.6, 1.3, 1.2 percent of the dearest edge's shift),
   against 0.055, 0.050, 0.047 at 0.06.
5. **Draw-averaged pooled gain counts.** Power averaged over a fresh uniform
   draw per replicate, threshold from 20,000 ideal replicates at each n:
   521, 229, 107 reads at ±10 percent and 59, 26, 13 at ±30 percent (64,
   256, 1,024 sites), against round 2's approximation 503, 226, 107 and 55,
   25, 12. Over 200 individual draws the power at those counts has 5th
   percentile 0.89 to 0.94 and minimum 0.80 to 0.93; round 2's fixed draws
   sit at 0.95 to 0.98. The Monte Carlo threshold is within 0.3 percent of
   the chi-square percentile.

## Not done

- No 1,024-spin scan length or false alarm: all-edge c1 there needs the
  runner's rank-2 update.
- No beta below 0.5: the row bounds the choice to [0.5, 4], and no row
  change is proposed this round.
- No recalibration with more chains: the t quantile is the whole
  correction.
