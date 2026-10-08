# Exchange cost projection: tempering priced in the Z1 Appendix-B model

Recorded 2026-10-07 from `evidence.tar.gz` (SHA-256 in `manifest.json`).
Protocol: [`docs/experiments/exchange-cost-projection.md`](../../experiments/exchange-cost-projection.md).
Replay: `uv run python -m thermo_lab.exchange_cost_projection --output-dir <extracted> --replay`.

**Evidence classes.** Sampling traces are `software_simulation` on CPU
(JAX, float32). Exact references are `exact_reference` (float64 enumeration).
Every energy and time figure below is a `calibrated_projection` from the sealed
`Z1HardwareProfile` (`z1-thermalizers-v1-2026-08`), which excludes host energy,
host latency, I/O latency, idle and board power. No sweep ran on hardware.

## Question

The October 2 symmetry-plus-tempering study and Potts stage B recommend
tempering with the analytic flip estimator because it qualifies at smaller
sweep budgets. On the Z1 cost model a replica exchange is host I/O, not a
redraw: a node read is 1.692 pJ and a full node SRAM write is 153.6 pJ against
7.09 fJ per Gibbs update, so one node write is about 21,665 updates. Does the
recommendation survive that pricing, and how does attempting exchanges less
often move the trade-off?

## Result in one paragraph

**Under the published cost model, tempering never wins on energy at any tested
exchange interval; it wins on elapsed sweeps.** Exchanging every sweep (the
archived arm) costs 475x to 7,459x the energy of the cheapest qualifying
symmetry-aware ordinary baseline across the five targets where one exists,
almost all of it SRAM writes of accepted swaps. Attempting exchanges every 4
sweeps keeps the same qualifying budget on five of six targets and cuts the
energy ratio to 117x to 1,886x; it is on the (sweep-time, energy) frontier on
five of six targets. Intervals of 16 and above lose the sweep-time advantage
(qualification moves to T = 4096 or is not reached) and at best reach 31x
(k = 256). With a hypothetical per-replica temperature control that makes an
accepted swap free of node writes, the k = 4 arm costs 3.0x to 48.7x the
ordinary baseline while keeping its 4x to 16x sweep-time advantage where it
has one; that
control is the hardware feature that would make tempering worth its exchanges.
On n16-g302 tempering is dominated under every convention. On n12-g301 only
tempering qualifies at all, so the ratios are undefined there.

## Stage A: the archived study, re-priced (zero new samples)

All 120 archived cells were priced from the archived exchange flags and all 24
archived qualification decisions were reproduced exactly. Ratios are the
tempering-flip arm at its qualifying budget against the cheapest qualifying
ordinary baseline (energy) and the fastest (sweep time).

| Target | T* | Energy ratio, published | Energy ratio, beta knob | Sweep-time ratio |
| --- | ---: | ---: | ---: | ---: |
| n12-g300 | 256 | 2,311x | 48.0x | 0.250 |
| n12-g301 | 1024 | no ordinary arm qualifies | - | - |
| n12-g302 | 64 | 2,257x | 48.0x | 0.062 |
| n16-g300 | 1024 | 1,895x | 48.0x | 0.250 |
| n16-g301 | 256 | 1,889x | 48.0x | 0.062 |
| n16-g302 | 4096 | 7,482x | 191.9x | 1.000 |

The n16-g302 cell at T = 4096 shows where the energy goes: 2.3 nJ of sampling,
444 nJ of energy reads, 16,884 nJ of state writes per trial. The beta-knob
ratio is 48x wherever T* is the same for both arms because reads scale with
attempts, which scale with T, and the ordinary arm pays only updates.

## Stage B: fresh seeds, exchange interval swept

Fresh root 20261008, same six graphs, 16 trials, budgets 16 to 4096, 210 cells
and 42 decisions, all replayed. The k = 1 arm is a fresh-seed replication of
the archived tempering-flip arm.

**Replication.** Tempering-flip at k = 1 qualifies on 6/6 targets again; both
ordinary arms qualify on 5/6 (n12-g301 is again unreached by every ordinary
arm). The k = 1 qualifying budgets match the archive on all six targets (256, 1024,
64, 1024, 256 and 4096 for n12-g300, n12-g301, n12-g302, n16-g300, n16-g301 and
n16-g302).

**Accuracy per interval at T = 4096, mean over six targets.** Acceptance is
flat at 0.47 to 0.48 for every k, as expected. Error rises with k:

| Arm | mean joint TV | mean edge MAE | trial pass fraction |
| --- | ---: | ---: | ---: |
| tempering-flip-k1 | 0.0130 | 0.0121 | 0.96 |
| tempering-flip-k4 | 0.0181 | 0.0165 | 0.93 |
| tempering-flip-k16 | 0.0278 | 0.0277 | 0.77 |
| tempering-flip-k64 | 0.0397 | 0.0380 | 0.65 |
| tempering-flip-k256 | 0.0503 | 0.0518 | 0.58 |
| independent-flip | 0.0332 | 0.0330 | - |
| long-flip | 0.0340 | 0.0352 | - |

k = 4 is almost free in accuracy. By k = 16 the cold replica mixes no better
than five independent cold chains, and at k = 256 it is worse.

**Qualifying budgets and ratios by interval** (energy against the cheapest
qualifying ordinary baseline; sweep time against the fastest):

| Target | Arm | T* | Published | Beta knob | Sweep time |
| --- | --- | ---: | ---: | ---: | ---: |
| n12-g300 | k1 | 256 | 584x | 12.0x | 0.062 |
| | k4 | 256 | 149x | 3.0x | 0.062 |
| | k16 / k64 / k256 | 4096 | 590x / 148x / 38.8x | 12.9x / 4.0x / 1.7x | 1.000 |
| n12-g301 | k1 | 1024 | undefined | undefined | undefined |
| | k4 | 4096 | undefined | undefined | undefined |
| | k16 / k64 / k256 | not reached | | | |
| n12-g302 | k1 | 64 | 2,229x | 48.0x | 0.062 |
| | k4 | 64 | 573x | 12.2x | 0.062 |
| | k16 / k64 / k256 | 256 | 620x / 158x / 41.9x | 12.9x / 4.0x / 1.7x | 0.250 |
| n16-g300 | k1 | 1024 | 1,896x | 48.0x | 0.250 |
| | k4 | 1024 | 479x | 12.2x | 0.250 |
| | k16 / k64 | 4096 | 476x / 124x | 12.9x / 4.0x | 1.000 |
| | k256 | not reached | | | |
| n16-g301 | k1 | 256 | 475x | 12.0x | 0.062 |
| | k4 | 256 | 117x | 3.0x | 0.062 |
| | k16 / k64 / k256 | 4096 | 474x / 120x / 31.5x | 12.9x / 4.0x / 1.7x | 1.000 |
| n16-g302 | k1 / k4 / k16 | 4096 | 7,459x / 1,886x / 467x | 192x / 48.7x / 12.9x | 1.000 |
| | k64 / k256 | not reached | | | |

**Frontier** (qualifying arms not dominated on sweep time and published
energy): n12-g300 {independent-flip, k4}; n12-g301 {k1, k4}; n12-g302
{long-flip, independent-flip, k4, k256}; n16-g300 {independent-flip, k4};
n16-g301 {independent-flip, k4}; n16-g302 {independent-flip}. k = 1 is never
on the frontier where an ordinary arm qualifies: k = 4 reaches the same budget
at a quarter of the exchange cost.

## What this changes

1. The standing recommendation "default to tempering plus symmetry at budgets
   of 1024 and above" (Potts stage C) is a sweep-budget recommendation. In
   the published Z1 model it is an energy recommendation against tempering by
   two to four orders of magnitude. Which one matters depends on whether the
   target application is time- or energy-bound; the recorded studies did not
   say, and the cost model cannot price the host latency that an exchange adds
   to time.
2. Exchange every sweep is wasteful even where tempering is right: k = 4 gives
   the same qualifying budgets on 5/6 targets at a quarter of the I/O. Future
   tempering arms should default to k = 4 unless a target shows otherwise.
3. A per-replica temperature control at zero node-write cost is the concrete
   hardware feature that reconciles the two views: with it, k = 4 costs 3x to
   12x the ordinary energy on the four targets where it keeps a 4x to 16x
   sweep-time gain. Without it, every
   accepted exchange is a full rewrite of two replicas.
4. The cheapest arm on the (time, energy) frontier is independent-flip on five
   of six targets. The one target where only tempering qualifies (n12-g301)
   is the case for tempering; it exists, but it is one graph in six.

## Fixed choices that bound these numbers

- The write-to-update energy ratio (21,665x) is a published constant. With
  acceptance near 0.5 and two pairs per attempt, each exchange step costs on
  the order of 2n writes, so the energy ratio at k = 1 is of order
  2 x 21,665 / (5n) sweeps of sampling per sweep. No interval in the tested
  range reaches parity; k = 256 reaches 31x and gives up the time advantage.
- A sequential systematic sweep is counted as one complete sweep of n updates.
  The Appendix-B sweep is two-colour and the graphs are not bipartite, so this
  is an update-count equivalence only.
- Placement is one logical to one physical p-bit with no embedding overhead,
  5n physical p-bits for five replicas. Embedding costs would raise the
  ordinary arms' energy as much as tempering's.
- Six small zero-field cubic graphs, n = 12 and 16, from one generator. The
  conservation and AM studies showed that mixing conclusions at this size do
  not always transfer; the energy ratios here are driven by constants, not by
  mixing, and should transfer, but the qualifying budgets that set the time
  ratios may not.
- Energy reads price an attempt at 2n reads per pair. An on-chip energy
  register would remove them (not modelled); it would not touch the writes.

## Integrity

Both stages replay: `status=exchange_cost_projection_complete`,
`archived_cells_priced=120`, `archived_decisions_priced=24`,
`cells_replayed=210`, `decisions_replayed=42`. The stage A archive is
authenticated by its manifest SHA-256. All sources, including the archived
sampler and estimator, are pinned by hash; the study-local sampler reproduces
the archived sampler bit for bit at k = 1 (`tests/unit/test_exchange_cost_projection.py`).
Generation took 161 s on two CPU cores; replay about 12 s. Provenance records
Torx 0.0.2 as a pinned release (this branch carries the pin fix from PR #90).
