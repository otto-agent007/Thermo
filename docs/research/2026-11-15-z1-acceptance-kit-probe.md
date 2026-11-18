# Z1 acceptance kit: probe (proposal P-0015)

**Exploration, not evidence.** One probe for queue row `z1-acceptance-kit`,
run on 2026-11-15 on the local i7-7700K, CPU only, single-thread BLAS.
Script: [`z1_acceptance_kit_probe.py`](z1_acceptance_kit_probe.py). Output:
[`2026-11-15-z1-acceptance-kit-probe.json`](2026-11-15-z1-acceptance-kit-probe.json).
CPU: 16.9 min for the recorded run plus about 5 min of smoke runs on
reduced inputs, about 22 CPU-minutes against the 30-minute ceiling. Peak
RSS 1.5 GiB. The protocol drafted from it is
[`docs/experiments/z1-acceptance-kit.md`](../experiments/z1-acceptance-kit.md).

Evidence classes. Means, shifts and per-read variances are
`exact_reference` (Kac-Ward on the explicit embedding, float64; variances by
a central second difference of ln Z, checked against brute force on a 4 x 4
patch to a relative 6.4e-8). Read counts for the energy and
log-likelihood-ratio (LLR) statistics apply the normal approximation to those
exact moments. Single-edge and stuck-spin counts use exact binomial laws.
The sampler runs are `software_simulation` (NumPy float64 two-colour heat
bath). Nothing ran on hardware. The graphs are synthetic planar patches of
the published 16-offset rule, not the chip map.

## What one read is

One read is a full-array snapshot after one complete colour-A-then-colour-B
sweep. "Independent reads" means equilibrium snapshots with no correlation.
The sampler part measures how far consecutive one-sweep reads are from
that.

## Findings

1. **Reproduce first: passed.** All 18 Kac-Ward references of
   `planar-16-offset-ferro` (P-0001) rebuild from their seeds and equal the
   committed archive bitwise: ln Z and every edge correlation, maximum
   difference 0.0.
2. **The archived start bounds the metric, not beta.** Ordinary two-colour
   Gibbs from the archived random start leaves frozen domain walls in 42,
   89 and 100 percent of chains at 64, 256 and 1,024 spins. Energy-per-spin
   bias in the second half of 16,384 sweeps is -0.043, -0.056 and -0.072,
   with integrated autocorrelation over 1,700 sweeps. That bias is 230 times
   the largest admitted shift at 1,024 spins (4-bit rounding, -3.1e-4), so
   from that start the ideal sampler fails every test and no failure can be
   attributed. From a flip-symmetric ordered start (half the chains in each
   Néel state, both ground states), the same kernel has bias consistent with
   zero at every size: -4.9e-7 ± 8.1e-7 at 1,024 spins. Its integrated
   autocorrelation is 1.12 sweeps for energy and 1.05 for the 8-bit LLR. It
   also reproduces the exact shifted mean of 4- and 8-bit rounded couplings
   within its standard error. With that start, one-sweep reads are nearly
   independent, and mixing error sits two orders of magnitude below device
   error.
3. **Beta = 4 bounds detection power.** It is the only temperature in the
   archive, and ordinary Gibbs qualified there only at 64 spins (and on one
   256-spin target). Deep in the ordered phase, edge correlations exceed
   0.98 on most edges, so each read carries little information. Reads for
   independent equilibrium reads at 5 percent false alarm and 5 percent miss,
   best statistic (LLR), are:

   | Defect | 64 spins | 256 spins | 1,024 spins |
   | --- | --- | --- | --- |
   | 4-bit rounding | 1,535 to 2,626 | 54 to 140 | 34 to 46 |
   | 6-bit rounding | 19,512 to 43,334 | 4,401 to 16,492 | 1,889 to 3,021 |
   | 8-bit rounding | 733,721 to 1,180,893 | 26,420 to 85,086 | 17,366 to 28,514 |

   Energy per spin needs 1.1 to 2.9 times as many reads as the LLR. The best
   single edge needs 1.8 to 14 times as many when its location is known, and
   2.1 to 3.0 times more again when every edge is scanned. Global statistics gain roughly in proportion
   to N. The smallest rounding defect detectable within 10,000 reads is 4
   bits at 64 spins, 6 bits on two of three targets at 256 spins and 6 bits
   on all three at 1,024. No 8-bit defect is detectable within 10,000 reads
   at any size.
4. **Rounding is not monotone in bits on this alphabet.** The weights are
   {1, ..., 5}/5 and the `full` cap is 1. At 5 and 9 bits, 2^(b-1) - 1 is
   15 or 255, both multiples of 5, so every coupling is represented exactly.
   The defect vanishes (maximum coupling error 0.0), and those bit counts
   have no test. The row's 4, 6 and 8 bits all round (maximum errors 0.057,
   0.013, 0.003). "Smallest detectable bits" must therefore be read per
   codebook, not as a threshold.
5. **One missing coupling: detectability is a property of the edge.** With
   the edge's own product (the LLR for this defect), required reads range
   from 7 to over 10^9. At 64 spins, 41 to 46 percent of edges are
   detectable within 10,000 reads with the location known (median 20,000 to
   37,000 reads). At 256 spins, a 48-edge stride sample gives 38 to 40
   percent. Edges into low-degree boundary sites are easiest: removing a
   bridge sends the correlation from -0.92 to exactly 0, detectable in 7 to
   8 reads. Edges whose endpoints are locked by parallel paths are
   effectively invisible: removing a middle (1,4) edge at 1,024 spins moves
   its correlation by 3e-9. Scanning all edges with a Bonferroni-corrected
   false alarm costs 1.6 to 2.6 times the known-location count.
6. **One stuck p-bit: admitted, with an odd statistic and an exact
   reference.** On an enumerated 12-spin patch, the stationary law of the
   two-colour kernel with one site pinned equals the conditional Boltzmann
   law to 1.5e-12. Every flip-even expectation is unchanged to 1.9e-12, as
   the owner's note states. The odd identity E[s_j | s_k = +1] = <s_j s_k>
   holds to 1.2e-12. A sharper exact fact makes the test independent of
   mixing. Zero field and a flip-symmetric start give the ideal kernel a
   flip-symmetric law at every finite time, so a site's value across
   independent chains is exactly Bernoulli(1/2). A stuck site reads its
   stuck value every time. Required reads, with miss probability 0:
   - 5, at a known site and a known stuck value;
   - 6, at a known site;
   - 12, 14 and 16, scanning all 64, 256 and 1,024 sites.

   Each read here is a separate chain, so each costs one initialization
   write. Within one continuing chain this test does not exist, because the
   ideal chain's magnetization never reverses at beta = 4.
7. **Static per-site beta variation: excluded from exact claims.** The
   stationary law of the two-colour sweep with per-site gains is not a
   pairwise Ising law on an enumerated 12-spin patch. A least-squares fit of
   ln p onto every field and pair term leaves an RMS residual of 4.3e-3
   (gains ±10 percent) and 1.5e-2 (±30 percent) at beta = 1. The ideal
   kernel's residual is 1.2e-8. So no Kac-Ward or other symmetric-coupling
   reference exists. The patch's exact law can be enumerated, but 64 spins
   cannot. The enumerated shifts are large next to the rounding shifts:
   maximum edge shift 0.030 at ±10 percent and beta = 1, and 0.010 at
   beta = 4. A test would probably see the defect, but the row allows no
   exact detection claim for it.

## Fixed choices and whether each bounds the metric

| Fixed choice | Value in the probe | Bounds the metric? |
| --- | --- | --- |
| Start state | archived random start against flip-symmetric Néel | **Yes, for attribution.** Random start: no test attributable at 256 or 1,024 spins. Néel: mixing bias at or below 1e-6 per spin |
| Inverse temperature | beta = 4 (the only archived value) | **Yes, for power.** 8-bit rounding is out of reach within 10,000 reads; a lower beta was not probed (the row fixes it) |
| Graph | `greedy-long`, seeds 4510 to 4512, L = 8, 16, 32 | Sets missing-edge detectability through local degree; global statistics scale with N |
| Couplings and cap | {1..5}/5, `full` cap 1 under M5b `round_parameters` | Yes, through the codebook: 5 and 9 bits are exact on this alphabet |
| Statistics | energy per spin, single edges, LLR | LLR is the most powerful test for a simple pair of hypotheses; energy costs at most 2.9 times as many reads |
| Error rates | 5 percent false alarm, 5 percent miss, one-sided | Set by the row |
| Read spacing | one sweep | No, once the Néel start is used: tau_int is at most 1.29 |
| Read limit | 10,000 | Set by the row; it defines "smallest detectable" |
| Sampler | NumPy heat bath, 64 chains, 16,384 sweeps, one seed per size | Probe-only; enough to see bias at 1e-6 |

## What this does not show

- No sampled operating characteristic: the probe did not run the tests
  themselves on samplers and count false alarms and misses. The protocol
  does.
- Missing couplings at 1,024 spins: two declared edges per target, not all
  edges. At 256 spins, a 48-edge stride sample.
- Normal-approximation counts are not exact error rates. The protocol checks
  them by simulation.
- No temperature other than 4, no frustrated targets, and nothing about the
  non-planar degree-16 lattice or a chip.
- The beta-variation shifts are exact only on the 12-spin patch.
