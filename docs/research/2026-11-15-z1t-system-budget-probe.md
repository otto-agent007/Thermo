# Z1T system budget: reproduction and per-token budget (P-0014)

2026-11-15 (cards created 2026-11-16 UTC). **Exploration, not evidence.** The
probe re-derives figures Extropic published as projections and counts the
operations of a published design. It runs nothing on hardware and draws no
samples. Every joule here is `calibrated_projection`; every count is exact
arithmetic on a stated design and a stated assumption. The loss figures quoted
in the last section are `software_simulation`. Script:
[`z1t_system_budget_probe.py`](z1t_system_budget_probe.py). Output:
[`2026-11-15-z1t-system-budget-probe.json`](2026-11-15-z1t-system-budget-probe.json)
(150 KB). It is pure arithmetic and runs in under a second of CPU.

Row: `z1t-system-budget`. The row has a precondition: reproduce the post's
per-token total and its split between co-processor and sampling from the
constants the post states, or name the missing constant and stop. Part 1 is
the first round's reproduction: **the Z1 side reproduces; the FPGA side does
not.** The owner then chose (2026-11-16, option 1) to proceed on the partial
reproduction: anchor the Z1 side on the reproduced 8.74 nJ, price off-chip work
only at the row's 0.1, 1 and 10 pJ per multiply-accumulate (MAC), and quote the
post's 285.78 nJ FPGA figure as unreproduced and unused. Part 2 is that budget.

## Answer in brief

- **Counts.** At the post's N = 32, the post's 512-wide model needs 96,784 to
  301,840 p-bits (1 to 2 chips) and the released 12,288-wide Z1T-0 needs
  2.32M to 7.24M (9 to 27 chips), with samples drawn in series as the post
  assumes. Holding the post's 16 µs sampling time at N = 2,048 takes 23 to 72
  chips for the 512-wide model and 552 to 1,719 for Z1T-0.
- **The Z1 side the post prices is the smallest part.** The post's 8.74 nJ is
  output samples only. At the post's own inferred read count (5 steps × 512
  values × 32 bits), reads at the sealed read constant cost 138.6 nJ, 16
  times the sampling energy. Clamp loads add 27.7 nJ to 43.1 µJ depending on
  placement and write constant.
- **The dense readout dominates the total at the row's MAC prices.** It is
  25.8M MACs per token at D = 512 (2.58, 25.8 or 258 µJ) and 617.6M at
  D = 12,288 (61.8, 618 or 6,176 µJ). The post's 136.4 µJ readout figure
  implies 5.29 pJ per MAC, inside the row's band.
- **N scales reads, samples and averaging, not p-bits or writes,** when samples
  are serial. At N = 2,048 the Z1-side band for Z1T-0 is 0.22 to 2.97 mJ per
  token before the readout, and serial sampling takes 1,024 µs.

## Part 1: reproduction (first round)

### Sources read

- The post, <https://extropic.ai/writing/z1t/>, fetched as raw HTML on
  2026-11-16, together with its two figure sources (`assets/z1-xpu-trace.html`,
  `assets/energy-bars.html`) and the six `journey/cards`. Source card:
  [extropic-z1t](../knowledge/sources/extropic-z1t.md), claims C3 and C5 to C8.
- The released model code, `research/z1t/components.py` in
  extropic-ai/sparse-transformers (local checkout at commit `8c0f7e6`, from the
  noise runs). For an AFT-conv block it gives qkv width 2D + H, MLP hidden
  width 4D, two DyT norms per block, one final DyT and a dense D × V logit
  head exempt from the tanh wrapping. With D = 12,288 this matches the row's
  unit counts (qkv 24,580, out 12,288, p1 49,152, p2 12,288).
- The sealed profile, `thermo_lab.hardware.z1.Z1HardwareProfile`
  (`z1-thermalizers-v1-2026-08`): 7.09 fJ per p-bit Gibbs update, 1.692 pJ per
  node read, 153.6 pJ per node full SRAM write, 269,568 p-bits per chip.

### What reproduces

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

### What does not reproduce

**FPGA work, 285.78 nJ.** The post gives three FPGA prices: 0.2 pJ per
matrix-multiply op, 3.0 pJ per scalar op and 1.5 W static. It gives no FPGA
operation count of any kind. The probe priced 2,048 plausible accountings
(averaging scope, ops per averaged value, price class of averaging and
convolution, pool variant, ops per MAC, residual and positional adds) at the
stated constants. None lands within the post's rounding (±0.005 nJ); the
nearest is 285.84 nJ, one coincidence among many. The page's own trace figure
says in its source: "Totals exact … per-stage pacing and split are
illustrative".

**Static power.** At 1.5 W, 58.8 µs of static power is 88.2 µJ, 309 times the
stated FPGA total, so the 285.78 nJ leaves static power out. The post does not
say this.

**Readout, "approximately 136.4 µJ".** 512 × 50,257 = 25.7M MACs. At the
stated prices that is 5.15 µJ (one op per MAC), 10.29 µJ (two ops at 0.2 pJ)
or 77.2 µJ (one scalar op). None equals the 136.1 µJ the readout adds. It
closes at 5.29 pJ per MAC with no static power, or with static power over a
readout time of 83.9 to 87.3 µs. The post states neither.

Missing constants, as reported to the owner: the FPGA's per-token op counts by
price class; whether static power is in either total and over what time; the
readout's op count and time. The budget below needs none of them.

## Part 2: the budget

### The design priced

One token, decoded with cached pool state, through the released code's
structure. A **sampled unit** is one tanh output computed on Z1 as the mean of
N one-bit draws of its output p-bit. Each unit reads its inputs from dy4p
registers: four ±1 p-bits per input value (C8), fan-in 4 for the projections,
fan-in 1 for a DyT. The FPGA (co-processor) averages, runs the AFT-conv pool,
the MLP activation, residual and positional adds, the DyT affine, and the
dense logit readout. Two unit sets are given: **with DyT** (the set that
reproduces 8.74 nJ; 21,008 units at D = 512, 503,824 at D = 12,288) and
**projections only** (the row's unit counts; 16,400 and 393,232).

| Structure per token | D = 512, with DyT | D = 12,288, with DyT | D = 12,288, projections only |
| --- | --- | --- | --- |
| Sampled units (output p-bits) | 21,008 | 503,824 | 393,232 |
| Sequential sampling layers | 25 | 25 | 16 |
| dy4p input p-bits, shared registers | 75,776 | 1,818,624 | 1,376,256 |
| dy4p input p-bits, one copy per consumer | 280,832 | 6,734,080 | 6,291,712 |
| Readout MACs (D × V + V bias adds) | 25,781,841 | 617,608,273 | 617,608,273 |
| Co-processor ops not scaling with N | 38,032 | 909,456 | 798,864 |
| Sparse body MACs (units × fan-in), reference | 70,208 | 1,683,520 | 1,572,928 |

### Open assumptions, each kept as a band

| Assumption the post leaves open | Low end | High end |
| --- | --- | --- |
| Placement of input p-bits | one shared dy4p register per input value (ignores the degree-16 limit: p1 inputs have mean fan-out 16) | a private copy of all inputs for every unit |
| Input encoding | clamped code, loaded once per token, not sampled (the post's pricing) | free biased p-bits, loaded once, sampled every step |
| Which values cross to the co-processor | the post's inferred 5 steps × D values × N bits | every sampled output, every step (U × N bits) |
| Node writes per token | the post's inferred 8 steps × D values × 4 p-bits | one load per input p-bit (copied placement) |
| Clamp-load price | node read constant, 1.692 pJ | node full SRAM write, 153.6 pJ |
| Per-sample price | sealed update, 7.09 fJ | the post's sample, 13 fJ |
| How samples are drawn (sets chips) | in series in time (C6) | N/32 replicas to hold 16 µs, or N replicas (fully parallel) |

The 8-step write and 5-step read counts are lower bounds borrowed from the
post's latency; they are not a feasible schedule, since a qkv site alone has
2D + H outputs and every site needs its inputs loaded. A re-clamp of the
input code at every sample step (dithering by writes) is excluded: the post's
8 write steps per token rule it out, and it would multiply writes by N.

### Counts against N (with DyT; full model)

p-bits and writes do not depend on N when samples are serial.

| N | Samples, 512 | Read bits, 512 | Samples, 12,288 | Read bits, 12,288 | Co-processor ops, 12,288 | Serial sampling time |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 672,256 | 81,920 – 672,256 | 16.1M | 1.97M – 16.1M | 2.94M – 17.5M | 16 µs |
| 128 | 2.69M | 0.33M – 2.69M | 64.5M | 7.86M – 64.5M | 8.84M – 65.9M | 64 µs |
| 512 | 10.8M | 1.31M – 10.8M | 258M | 31.5M – 258M | 32.4M – 259M | 256 µs |
| 2,048 | 43.0M | 5.24M – 43.0M | 1.03G | 126M – 1.03G | 127M – 1.03G | 1,024 µs |

| N | Chips, 512: serial / N/32 replicas / N replicas | Chips, 12,288: serial / N/32 replicas / N replicas |
| --- | --- | --- |
| 32 | 1–2 / 1–2 / 12–36 | 9–27 / 9–27 / 276–860 |
| 128 | 1–2 / 2–5 / 46–144 | 9–27 / 35–108 / 1,103–3,437 |
| 512 | 1–2 / 6–18 / 184–574 | 9–27 / 138–430 / 4,412–13,748 |
| 2,048 | 1–2 / 23–72 / 736–2,294 | 9–27 / 552–1,719 / 17,645–54,989 |

Chips are ceil(p-bits / 269,568) at 100 percent packing, so they are lower
bounds for each placement. Node writes per token: 16,384 to 280,832 at
D = 512 and 393,216 to 6,734,080 at D = 12,288; with N/32 replicas each
replica needs its own clamps, so writes scale by N/32.

### Projected joules per token (`calibrated_projection`)

Z1 side: samples + reads + clamp loads, every open assumption at its low or
high end together. Off-chip: readout and co-processor ops at one MAC price
each.

| Config, N | Z1 side | Readout at 0.1 / 1 / 10 pJ | Total at 1 pJ |
| --- | --- | --- | --- |
| 512, N = 32 | 0.171 – 44.4 µJ | 2.58 / 25.8 / 258 µJ | 26.1 – 70.9 µJ |
| 512, N = 2,048 | 9.20 – 124 µJ | same | 40.3 – 193 µJ |
| 12,288, N = 32 | 4.11 – 1,065 µJ | 61.8 / 618 / 6,176 µJ | 0.625 – 1.70 mJ |
| 12,288, N = 128 | 14.4 – 1,156 µJ | same | 0.641 – 1.84 mJ |
| 12,288, N = 512 | 55.7 – 1,519 µJ | same | 0.706 – 2.40 mJ |
| 12,288, N = 2,048 | 221 – 2,973 µJ | same | 0.965 – 4.62 mJ |

Breakdown for the post's own point (D = 512, N = 32): sampling 4.77 nJ
(sealed) or 8.74 nJ (post); reads 139 nJ to 1.14 µJ; clamp loads 27.7 nJ (post
write count at the read constant) to 43.1 µJ (copied placement at the write
constant); free input p-bits would add 117 nJ of sampling; co-processor
0.12 to 0.73 µJ at 1 pJ. The low end of the Z1 band is set by reads; the high
end by copied placement at the write constant. At N = 2,048 reads dominate
both ends for Z1T-0 (0.21 to 1.75 mJ).

Every table cell at every N in the grid {32, 64, 128, 256, 512, 1,024, 2,048},
for every variant and each MAC price, is in the JSON.

**Included:** output p-bit draws; input p-bit draws (high end only); node
reads of every crossing bit; one clamp load per input p-bit per token (per
replica when replicated); co-processor averaging (N adds and one scale per
averaged value), AFT-conv decode (D·4 + H·4 taps and 6D + 5H elementwise ops
per block), MLP activation, residual and positional adds, DyT affine; the
dense readout D × V MACs and V bias adds. Every off-chip op, transcendental or
not, is priced as one MAC.

**Excluded:** FPGA static power (at the post's 1.5 W it is 88 µJ over the
post's 58.8 µs and at least 1.5 mJ over N = 2,048's serial sampling time);
link, SerDes and chip-to-chip energy (the sealed profile prices idealized node
access, excluding the host); readout and embedding weight traffic from
memory; softmax and token selection; one-time coupling programming (p-bits ×
153.6 pJ per model load: 15 to 46 µJ at D = 512, 0.36 to 1.11 mJ at
D = 12,288); placement and packing overhead; orchestration; analog error and
calibration; clocking of idle layers. If idle layers' p-bits are clocked
every cycle, updates rise by the number of sequential layers (25 times):
218 nJ at D = 512, N = 32, and 0.34 mJ for Z1T-0 at N = 2,048.

### Hypothetical gated-unit budget (Z1T-0 only)

Dropping the 105,442 near-zero projection units (exploration counts, threshold
1e-6; not shown to be exact zeros, and gating is not shown to be
noise-equivalent) removes their output p-bits, samples, reads and copied
inputs. Shared input registers stay. With DyT: units 398,382 (−21 percent),
p-bits 2.22M to 5.45M (9 to 21 chips serial), samples at N = 2,048 816M, Z1
band at N = 2,048 0.219 to 2.30 mJ. The low end barely moves because it is
set by the post-step read count; the high end falls 23 percent. No quality
claim attaches to this budget.

### Quality at each N (input, not a result of this probe)

Loss of the released model with output sampling noise only
(`software_simulation`, [first results note](2026-11-15-z1t-noise-finetune-results.md)):
10.746 nats at N = 32, 9.184 at 128, 6.901 at 512, 4.916 at 2,048, against
4.002 noise-free. To be replaced by `z1t-sample-budget-map` when it posts. At
the post's N = 32 the released model is 6.7 nats from its noise-free loss; at
2,048 it is still 0.91 nats away. The budget at N = 2,048, not N = 32, is the
nearer one to the post's quality, and its serial sampling time is 64 times
the post's 16 µs.

## Proposal frame

- **Design decision informed.** What an outside reader should expect a Z1T
  token to cost in p-bits, chips, I/O and off-chip work, and which of the
  post's figures carry that cost. Not a task row; no digital headroom is
  claimed.
- **Intended model, representation, update process, estimator.** Intended:
  Z1T-0's tanh-linear network. Representation: one output p-bit per unit with
  dy4p input registers. Update process: N serial (or replicated) Gibbs draws
  of each output p-bit, layer by layer. Estimator: the sample mean, computed
  off-chip. The quality cost of that estimator is the loss table above.
- **Mechanism and falsifier.** Counting published operations at published
  constants. The reproduction gate is the falsifier; the Z1 side passed and the
  FPGA split failed (part 1).
- **Strongest practical digital baseline.** Not part of this row. For
  reference only: the sparse body is 70,208 MACs at D = 512 and 1.68M at
  D = 12,288. The low end of the Z1-side band equals the sparse body at
  2.44 pJ per MAC at N = 32 and 131 pJ per MAC at N = 2,048, for both widths.
  These are break-even MAC prices for the body alone, which exclude the
  digital side's memory traffic; they are not an advantage claim either way.
- **Counterfactual feature.** An on-chip counter per output that reads
  ceil(log2(N + 1)) bits instead of N would cut read bits for Z1T-0 at
  N = 2,048 from 1.03G to 6.05M. Z1 is not known to have one; it pays off on
  reads alone if a counter increment costs less than about one node read
  (1.69 pJ).
- **Cheapest decisive check.** This probe; the counts are closed-form.
- **Outcome.** Answered as a docs-only probe. No protocol and no production
  run follow.

## Fixed choices and what each bounds (step 1)

| Choice | Bounds the metric? |
| --- | --- |
| Width: 512 (post) or 12,288 (released) | Yes: all counts scale with D, the readout as D × V. |
| Samples per unit N: 32 to 2,048 | Yes: samples, read bits and averaging scale with N; p-bits and writes do not when serial. |
| Samples in time vs. replicated p-bits | Yes: sets chips and replica clamp loads. |
| Input encoding (dy4p clamped vs. free) | Yes: adds up to 16 input draws per unit per step. |
| Placement: shared vs. copied input p-bits | Yes: sets p-bits, chips and writes, up to 3.7 times. |
| Which values cross and at what width | Yes: sets the low end of the Z1 band at every N. |
| Clamp-load price (read vs. write constant) | Yes: sets the high end of the Z1 band. |
| Off-chip MAC price 0.1 / 1 / 10 pJ | Yes: sets the readout, which dominates every total at 1 and 10 pJ. |
| Gated near-zero units | Only in the separate hypothetical budget. |

## Evidence obligations

`evidence-obligations.json` (version 1, checked 2026-11-11) lists EO-001 to
EO-005. None covers Z1T, the extropic-z1t card or this row, so no applicable
open obligation exists; EO-005 (2026-11-11 evaluation) is open and unaffected.
The extropic-z1t card's 2026-11-15 re-read was an automated summary. This
probe read the raw page and the figure sources. The card would gain two facts:
the trace figure calls its FPGA split illustrative, and the 8.74 nJ prices
output draws only. The card is not edited here.

## What this probe did not do

- It drew no samples, used no GPU and ran no model.
- It did not reproduce or use the post's 285.78 nJ FPGA figure.
- It did not place Z1T on the degree-16 lattice; chips are packing lower
  bounds and the shared placement ignores the degree limit.
- It priced no static power, link energy or weight traffic (listed above).
- It made no quality claim for gated units or for any N beyond the quoted
  loss figures.
