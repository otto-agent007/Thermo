# Z1 acceptance kit: round-1 probe (proposal P-0015)

**Exploration, not evidence.** Round-1 probe for queue row
`z1-acceptance-kit`, after the row was amended on main (PRs #176 and #179)
and the review comment on PR #174. Run on 2026-11-16 (local time) on the
i7-7700K, CPU only, single-thread BLAS. Script:
[`z1_acceptance_kit_probe_r1.py`](z1_acceptance_kit_probe_r1.py). Output:
[`2026-11-17-z1-acceptance-kit-probe-r1.json`](2026-11-17-z1-acceptance-kit-probe-r1.json).
CPU: 22.9 min user time for the recorded run, plus 0.4 min for a smoke run
of the closed-form parts: about 23 CPU-minutes against the 30-minute
ceiling. Peak RSS 1.6 GiB. Protocol:
[`docs/experiments/z1-acceptance-kit.md`](../experiments/z1-acceptance-kit.md).

Evidence classes. Simple-test laws and read counts are `exact_reference`
(closed-form two-state laws, exact binomial error rates, and an enumerated
check of the two-site sweep). Kac-Ward means, per-read variances and
independent-read counts are `exact_reference`; LLR counts use the normal
approximation on exact moments. Scatter ratios and biases are
`software_simulation` (NumPy float64 two-colour heat bath, one seed per cell
plus a second seed at the chosen beta). Nothing ran on hardware. At 1,024
spins, the exact LLR counts at beta 0.5 are reused from the review probe
JSON; beta 0.75 is not resampled there.

## Findings

1. **Isolated pairs and single sites: exactly independent reads.**
   Enumerating the two-site two-colour sweep, consecutive pair products are
   independent to 1.1e-16, and P(product = sign J) = sigmoid(2 g_B beta |J|)
   to 1.1e-16, where g_B is the colour-B site's gain. The colour-A gain
   does not enter. So a pair test needs no start state, no burn-in and no
   correlation correction, and it cannot see a colour-A gain error.
2. **Stuck p-bit: the single-site test covers both behaviours.** An
   unbiased isolated site is a fresh fair coin every sweep from any start.
   A site that ignores writes, or accepts a write and never updates, reads a
   constant. Reads: 5 (known site and value), 6 (known site), 12, 14 and 16
   scanning 64, 256 and 1,024 sites. Miss probability 0. One programming,
   no state writes. The round-0 collective version needed one chain, one
   state write and a fair-coin start per read.
3. **Gain error: single sites detect it; nothing else has an exact
   reference.** At 2 beta h = 1, a +10 percent gain needs 5,664 reads at a
   known site and +30 percent needs 675. The review estimated about 5,700
   and 660. At the best beta h (1.05 to 1.35), the counts are 2,239 to 2,751
   (±10 percent) and 199 to 368 (±30 percent). Scanning 1,024 sites they
   are 6,278 to 7,878 and 543 to 1,077. Smallest positive delta within
   10,000 reads: 5.1 percent at a known site, 8.8 percent scanning 1,024
   sites. A bias-scale error is indistinguishable from a gain error in
   this test.
4. **Missing coupling: pairs need the fewest reads, the collective scan the
   fewest writes.** Programmed at |J| = 1 and beta 4, a pair needs 5 reads at
   a known coupler. At beta J = 1 it needs 16 reads; the review estimated
   about 13. Scanning every coupler needs 8 reads per matching. Covering the
   planar subgraph takes 7 to 9 matchings (its maximum degree, by König's
   theorem; every edge joins the two colours). That is 56 to 72 reads and
   7 to 9 programmings. The collective own-edge scan at beta 0.6 detects
   every edge computed (all 112 at 64 spins and a stride of 48 to 53 at 256
   spins), in 1,906 to 2,444 independent reads with one programming. At a
   known location it needs 35 to 928 reads.
5. **Rounding: the pooled pair test beats the collective LLR at every
   depth.** Programming a whole matching at the worst-rounded weight (0.2 or
   0.4), at the best beta, 8-bit rounding needs 238, 955 and 3,788 reads at
   1,024, 256 and 64 spins. The collective LLR needs 3,738 to 3,785 reads
   at 1,024 spins and beta 0.6, before its scatter ratio of 1.35 to 1.61.
   With either test, 10, 11 and 12 bits are out of reach within 10,000
   reads; 5 and 9 bits carry no defect on this codebook. The smallest
   detectable rounding is 8 bits at every size for the pair test. For the
   collective LLR it is 8 bits at 1,024 spins, 7 at 256 and 6 at 64.
6. **Temperature for the collective tests: beta 0.6, by the declared
   rule.** The rule minimises the largest regret in sweeps per test (512
   burn-in sweeps + reads × scatter ratio). Both candidates passed the
   random-start bias check on all nine graphs (largest |z| 2.5). Largest
   regret: 1.19 at 0.6 and 1.40 at 0.5. That gap is near the scatter
   ratios' sampling error (relative SE 9 to 13 percent). Where colder betas
   were run they lost. At 0.75 the 8-bit LLR scatter ratio is 1.96 to 6.42
   at 64 and 256 spins. At 1.0 it reaches 40.9 on L16 s4511. At 2.0 the
   random-start energy bias fails at 64 spins (z = −4.8, −4.5).
7. **Read correlation measured on every graph at the test length.** At
   beta 0.6 the scatter ratio of the 6- and 8-bit LLRs is 1.26 to 2.03. On a
   second seed, over two consecutive windows, the 8-bit LLR gives 1.25 to
   1.99 with no drift between windows. The energy statistic is more
   correlated: 3.2 to 9.1. Chain-level outliers stay within what 128 to 256
   normal means produce (largest |z|/sqrt(r) 3.66). Slow states rarer than
   about one chain in 256 would not show.
8. **Pass rule.** With 39 checks, 1,000 replicates and Bonferroni per side,
   a check fails above 74 ideal rejections or below 926 defective ones. The
   family-wise false-failure bound is 3.2 percent, and the power against a
   true 10 percent false alarm is 0.997. The round-0 rule had a 30 percent
   chance of at least one false failure on the false-alarm side, and 52
   percent counting both sides. A mixing bias of 5 percent of the shift
   moves a 5 percent false alarm to 6.9 percent; 10 percent of the shift
   moves it to 9.4 percent.

## What this changes

The collective Kac-Ward test is the cheaper instrument for one declared
case only: scanning for a missing coupling with one programming instead of
7 to 9. For the other cases the simple tests need fewer reads and no more
writes, or the collective test has no exact reference. Most of the row's
answer is closed form. The remaining sampled work is the collective scan
and a check of the runner and kernel on the simple tests.

## What this does not show

- No missing-edge table at 1,024 spins and beta 0.6; the runner computes it.
- Scatter ratios for single edge products and for the 7-bit LLR were not
  measured.
- Simple tests assume the declared programming range (beta up to 4, |J| up
  to 1, a representable site bias). Nothing is said about update-process
  defects, frustrated targets, the full degree-16 lattice or any chip.
