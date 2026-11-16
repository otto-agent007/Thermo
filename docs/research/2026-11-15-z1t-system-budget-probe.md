# Z1T system budget: reproduction probe (P-0014)

2026-11-15 (card created 2026-11-16 UTC). **Exploration, not evidence.** The
probe re-derives figures Extropic published as projections. It runs nothing
on hardware and draws no samples. Every Extropic figure quoted here is their
`calibrated_projection`. Script: [`z1t_system_budget_probe.py`](z1t_system_budget_probe.py).
Output: [`2026-11-15-z1t-system-budget-probe.json`](2026-11-15-z1t-system-budget-probe.json).
It is pure arithmetic and runs in under a second of CPU.

Row: `z1t-system-budget`. The row has a precondition: reproduce the post's
per-token total and its split between co-processor and sampling from the
constants the post states. If that fails, name the missing constant and stop.
**The Z1 side reproduces. The FPGA side does not.** Following the row's rule,
this probe stops there and computes no budget.

## Sources read

- The post, <https://extropic.ai/writing/z1t/>, fetched as raw HTML on
  2026-11-16, together with its two figure sources (`assets/z1-xpu-trace.html`,
  `assets/energy-bars.html`) and the six `journey/cards`. Source card:
  [extropic-z1t](../knowledge/sources/extropic-z1t.md), claims C3 and C5 to C8.
- The released model code, `research/z1t/components.py` in
  extropic-ai/sparse-transformers (local checkout from the noise runs). For an
  AFT-conv block it gives qkv width 2D + H, MLP hidden width 4D, two DyT norms
  per block and one final DyT. With D = 12,288 this matches the row's unit
  counts (qkv 24,580, out 12,288, p1 49,152, p2 12,288).

## What reproduces

For the post's configuration: L = 4, D = 512, H = 4, kernel 4, fan-in 4,
N = 32.

| Post figure | Re-derived | How |
| --- | --- | --- |
| Z1 sampling 8.74 nJ | 8.739 nJ | 21,008 sampled outputs × 32 samples × 1.3e-14 J |
| 25 sequential sampling layers | 25 | DyT1, qkv, out, DyT2, p1, p2 per block, plus the final DyT |
| Z1 sampling time 16.0 µs | 16.0 µs | 25 × 32 / 50 MHz |
| Latency 58.8 µs | 58.77 µs | parts below |
| H100 40.9 / 8.17 / 4.09 µJ | 40.89 / 8.18 / 4.09 µJ | 2 × 11.55M FLOPs × 0.177 pJ / MFU |

- The 21,008 outputs are 4 × (512 + 1,028 + 512 + 512 + 2,048 + 512) + 512.
  Only one reading of a "sample" matches 8.74 nJ: one draw of each sampled
  output p-bit per sample step, with every DyT sampled on Z1, the final one
  included. Leaving out the DyTs gives 6.82 nJ. Leaving out only the final DyT
  gives 8.53 nJ. Counting the 4 dy4p p-bits per value gives 34.96 nJ.
  Counting the 16 input p-bits of each unit gives 148.6 nJ. So the post's
  sampling energy prices output draws only. The dy4p input p-bits are not
  priced as samples, and no write or read is priced on the Z1 side.
- Latency: 38 × 1 µs orchestration, 16.0 µs sampling, 3.2 µs reads and
  1.28 µs writes add up to 58.77 µs once the post's 0.13 and 0.16 µs are
  included. The reads match 5 steps × 512 values × 32 bits exactly, at
  25.6 Gbit/s. The writes match 8 steps × 512 values × 8 bits. Both bit
  counts are my inference and the post states neither. The reads cover
  512 values per step, although a block's qkv site has 1,028 outputs.
- H100: the 50 percent point re-derives to 8.177 µJ, and the post prints
  8.17. That is a rounding difference only.

## What does not reproduce

**FPGA work, 285.78 nJ.** The post gives three FPGA prices: 0.2 pJ per
matrix-multiply op, 3.0 pJ per scalar op and 1.5 W static. It gives no FPGA
operation count of any kind. The probe priced 2,048 plausible accountings at
the stated constants. The choices it varied were:

- the averaging scope: all outputs, the 5 × 512 read-back values, qkv + p2 +
  final DyT, or none;
- averaging ops per value: N, N − 1, N + 1 or 1;
- the price class of the averaging and of the convolution;
- the pool variant: conv taps only, conv plus elementwise ops, elementwise
  only, or full-context recompute;
- one or two ops per multiply-accumulate;
- residual adds included or not;
- the positional add included or not.

None of the 2,048 lands within the post's rounding (±0.005 nJ). The nearest
is 285.84 nJ, 0.06 nJ away. It is one coincidence among many and is not a
reproduction. 285.78 / 3.0 pJ comes to exactly 95,260 scalar ops. A third of
all two-decimal figures divide evenly like this, so it identifies nothing.
The page's own trace figure says in its source: "Totals exact … per-stage
pacing and split are illustrative". The figure spreads the FPGA total evenly
over nine stages, so the post's materials contain no breakdown.

**Static power.** At 1.5 W, 58.8 µs of static power is 88.2 µJ. That is 309
times the stated FPGA total, so the 285.78 nJ leaves static power out. The
post does not say this.

**Readout, "approximately 136.4 µJ" with the logit layer.** The readout is
512 × 50,257 = 25.7M multiply-accumulates. At the stated prices that is
5.15 µJ (one op per MAC), 10.29 µJ (two ops at 0.2 pJ) or 77.2 µJ (one
scalar op). None equals the 136.1 µJ the readout adds. Two readings would
close the gap:

- 5.29 pJ per MAC with no static power;
- static power over a readout time of 83.9 to 87.3 µs in total.

The post states neither constant.

## The missing constants

1. **The FPGA's per-token operation counts by price class.** That is, how
   many 0.2 pJ matrix-multiply ops and how many 3.0 pJ scalar ops make up the
   285.78 nJ. It includes how the 32-sample averages, the AFT-conv pool, the
   residual adds and the positional encoding are counted.
2. **Whether static power is in either total, and over what time.** It is
   not in 285.78 nJ. For the 136.4 µJ readout figure it is unresolved.
3. **The readout's operation count and time**, or its energy per MAC,
   needed to reproduce 136.4 µJ.

The Z1 side needs none of these. Its single-reading match gives the row a
firm anchor for the sampling count: one priced sample per sampled output per
sample step.

## Fixed choices the study would make (step 1)

| Choice | Bounds the metric? |
| --- | --- |
| Width: 512 (post) or 12,288 (released) | Yes: all counts scale with D. The readout is D × 50,257 MACs. |
| Samples per unit N: 32 to 2,048 | Yes: samples scale linearly with N. Sampling time does too if samples are serial. |
| Samples in time vs. in parallel p-bits | Yes: it alone sets p-bit and chip count. The post assumes serial samples (C6) and does not model chips (C3). |
| Input encoding (dy4p, 4 p-bits per value) | Yes: it sets the input p-bits per unit (16) and the bits written per value. The post prices none of them. |
| Placement: shared vs. copied input p-bits for values with fan-out above 1 | Yes: it sets the p-bit count, so the chip count, by up to the mean fan-out. The post leaves it open. |
| Which values cross to the co-processor and at what precision | Yes: it sets the read bits and off-chip ops. The post's latency reads 512 values per step, while qkv has 2D + H. |
| Where averaging runs (FPGA or on-chip) | Yes, for off-chip ops and read bits. |
| Gated near-zero units (105,442 of 393,232 in Z1T-0) | It changes counts only in the separate hypothetical budget. Quality equivalence is not established. |
| Per-MAC off-chip price 0.1 / 1 / 10 pJ | It sets the off-chip joule band. Fixed by the row. |
| Sealed Z1 profile (update 7.09 fJ, read 1.692 pJ, write 153.6 pJ) vs. the post's 1.3e-14 J per sample | It sets the on-chip joule band. Both are fixed constants. |

At D = 12,288 the readout's D × V multiply-accumulates are a fixed property
of the model, and the post itself says the readout dominates once included
(C3). The probe does not price them, because the row says to stop before
budgeting.

## Proposal frame (for the owner's decision)

- **Design decision informed.** Whether an outside reader can check the
  post's system-level figures. It also gives the counts (p-bits, chips,
  reads, writes, off-chip MACs) that the post's headline ratio leaves out.
- **Mechanism and falsifier.** It counts a published design's operations at
  published constants. The falsifier is the reproduction gate itself, and it
  failed on the FPGA side.
- **Constrained realization and baseline.** Not a task row. No digital
  baseline is claimed and no headroom is asserted.
- **Cheapest decisive check.** This probe. Once the owner decides, the counts
  are closed-form arithmetic. They would most likely close by a docs-only
  probe note, not a recorded study, in well under a CPU-minute.
- **Outcomes.** Advance if the owner accepts a reproduction that covers only
  the Z1 side. Hold if the FPGA counts are needed first. Stop if the
  unreproduced split makes the row moot.

## Evidence obligations

`evidence-obligations.json` (version 1, checked 2026-11-11) lists EO-001 to
EO-005. None covers Z1T, the extropic-z1t card or this row, so no applicable
open obligation exists. The extropic-z1t card's 2026-11-15 re-read was an
automated summary. This probe read the raw page and the figure sources. The
card would gain one fact from them: the trace figure calls its FPGA split
illustrative. The card is not edited here.

## What this probe did not do

- It computed no p-bit, chip, sample, read, write or MAC budget at any N or
  width.
- It made no joule projection beyond re-deriving the post's own figures.
- It did not touch the dead-unit counts or the loss-versus-N inputs.
- It used no GPU and no sampling.
