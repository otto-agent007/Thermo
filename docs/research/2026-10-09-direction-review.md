# Direction review D-0001

2026-10-09. Main commit d88410ddad80d85cc21981b362c529b1247ea638. THERMES-DIRECTOR.
**Advice, not a decision.** The owner edits the queue; this file proposes.

Covers the two loop results accepted since the loop started (P-0001
`planar-16-offset-ferro`, PR #95; P-0002 `am-coupling-bits`, PR #99), weighs
the proposal in flight (P-0003 `e0-remaining-kernels`, PR #103, awaiting the
owner), and reads the queue against charter amendment 1
(`PROJECT_CHARTER.md`, "Amendment 1"). No previous review.

## The change I would make first

Drop `e0-remaining-kernels` and reject P-0003. It is 7.5 to 8.5 CPU-hours
for a fourth THRML-matches-the-exact-law contract after three have passed
154 of 154 cells (`docs/knowledge/lessons.md`, rows 2026-10-03 "E0 stage A",
"E0 stage B", 2026-10-06 "Potts stage A"). No row downstream designs against
its answer, and amendment 1 (priority 4) says the meta-EBM line stays small
until Extropic publishes Thermalizers. The queue's own note calls it "low
marginal information; idle-time work" (`docs/research-queue.md`, row
`e0-remaining-kernels`). The loop has no idle time: every study it runs costs
owner review time, and that is the scarce resource the amendment names.

## Verdicts

**P-0001 `planar-16-offset-ferro`: holds, low value.** A probe-grade
question that received a study's process. The probe already answered it
(`docs/research/2026-10-08-planar-16-offset-ferro-probe.md`, Result 1 and
2: the ordering and the nine-replica budgets hold on both hardware-shaped
families); the recorded run confirmed those and corrected the probe's
third point (the thin-ladder slip did not reproduce,
`docs/experiment-reports/2026-10-08-planar-16-offset-ferro/findings.md`,
"Pre-registered expectation 3 failed"). Not a process fault: it was
approved before amendment 1 merged. It should be the last of its kind.

**P-0002 `am-coupling-bits`: positive, the first specification number.**
Six bits keep exact equilibrium recall within 0.01 when the codebook step
equals J; 10 and 12 bits under the other caps; static ±10% beta jitter is
negligible (`docs/experiment-reports/2026-10-08-am-coupling-bits/findings.md`).
This is exactly what amendment 1's recorded-study tier asks for. Two caveats
below limit how far it carries: the design was not placed on the chip's
graph, and the cap rule that made 6 bits work is specific to a model whose
couplings all share one magnitude.

## Rubric 1: the accepted results

### P-0001 planar 16-offset ferro

- **Design number?** No. "Which tempering arm qualifies at which sweep
  budget on a random ferro instance" is a sampler allocation, not a
  precision, noise, I/O or task-quality number. The one hardware-relevant
  observation, cold-pair acceptance falling more than tenfold on the long
  edges (findings, "Exchange acceptance"), concerns replica exchange, which
  the chip does not have (amendment 1, "Why", item 1) and which never wins
  on projected energy
  (`docs/experiment-reports/2026-10-07-exchange-cost-projection/findings.md`).
- **Hardware constraints.** Relaxed: free replica exchange and a free
  temperature ladder (all qualifying arms are tempering); float32 couplings;
  no beta jitter; and the graph is a maximal planar subgraph with mean degree
  below 4, not the degree-16 lattice (probe, "Fixed choices": "the study
  tests long planar edges, not the chip's degree 16"). Kept: a two-colour
  chromatic schedule. The findings say this plainly ("None of this says
  anything about the non-planar degree-16 lattice").
- **Resource accounting.** Yes: sweeps, p-bits, and Z1 energy per trial
  under the published convention, with the warning that a broken ladder
  prices cheap (findings, "Projected Z1 cost").
- **Missed expectation.** Expectation 3 (five-replica slip at L = 32). No
  queue row assumed either outcome. Lesson already recorded: a one-seed probe
  cannot size an effect inside one fourfold budget step (lessons, row
  2026-10-08 "Planar 16-offset ferro").

### P-0002 am coupling bits and beta jitter

- **Design number?** Yes: b* = 6 bits (step = J), 10 (`split`), 12
  (`full`); ±10% static per-site gain jitter negligible, ±30% within 0.003
  (findings, table and "Beta jitter"). Stated in units of field/J, which
  grows like sqrt(P) at N = 24 (findings, "Evidence classes and limits").
- **Hardware constraints.** Kept: quantization through the M5b codebook,
  per-site beta jitter, two-block chromatic Gibbs, no exchange. Relaxed:
  **placement.** The `bias` design is a complete bipartite graph: each hidden
  unit couples to all N = 24 visible spins and each visible spin to all P
  (up to 128) hidden units (`docs/experiments/am-coupling-bits.md`, "Fixed
  choices"), against the lattice's degree 16
  (`docs/knowledge/sources/extropic-z1t.md`, C1). M5c placed over-degree
  kernels by copying *clamped inputs* with no chains
  (`docs/experiment-reports/2026-09-30-meta-ebm-topology/findings.md`); the
  AM hidden units are free spins, so placing them needs chains, and chains
  change the law. The 6-bit number is therefore a spec for an unplaceable
  graph until `constraint-layer` says what placement costs it. Also relaxed:
  the ratio of field range to coupling range the chip's programming path
  allows is unpublished; the design needs fields up to 28 J (findings).
- **Resource accounting.** CPU seconds and sweep budgets only. No updates,
  reads, writes or projected energy per recalled pattern; the protocol
  excludes speed and energy ("Not claimed"). Rule 9 is half met; the
  `am-io-budget` row exists to close that half.
- **Missed expectations.** `split` needed 10 bits against at most 7, and
  `full` 12 against 8 (findings, table). Rows that assumed the opposite:
  `constraint-layer`'s default "step = J" is consistent with the result.
  `dtm-small-image`'s arms "at 6 and 8 bits (step = J)" are not: step = J
  is a cap rule, and it kept couplings exact only because every coupling in
  the AM design has the same magnitude (protocol, "Codebook cap"). A trained
  denoiser has heterogeneous couplings, so "step = J" has no referent there
  and the bit grid the row inherits has no basis yet. The dtm probe must
  choose and justify a cap rule before any bit count is pre-registered.

### Pattern across both

Both probes mis-sized a pre-registered effect: one overcalled a slip, the
other undercalled two bit budgets by 3 to 4 bits. The lessons register
already carries this twice (rows 2026-10-08 "Planar 16-offset ferro" and
2026-10-07 "Associative memory A2"). It is a process pattern, not a failed
line; the two-consecutive-failures rule does not apply to any queue line
today. No premise question for the owner.

## Rubric 2: the queue rows

**`constraint-layer` (open).** Decision it informs: whether every later study
runs under the chip's rules by default (amendment 1, rule 8), and whether
updates, reads, writes and clamps get separate prices
(`src/thermo_lab/hardware/z1.py`, `Z1HardwareProfile`). Cheapest probe: none
replaces it, because it is code; but the recorded-study treatment is
unnecessary. The amendment calls it "a transformation plus an accounting
record, not a new runner" (priority 1). Unit tests that pin bitwise agreement
with the three archives plus one dated note under `docs/research/` satisfy
the row's metric; a frozen protocol, archive and gate add nothing a designer
uses. Source cards: `extropic-z1t.md` C1 supplies the topology and clock;
`extropic-thermalizers.md` describes compilation but is not public code.
Not answered in the literature. Not a third variant. **One text fix needed**:
the row's "bitwise agreement with ... the `am-coupling-bits` quantized
recall at 6 bits" can only hold with placement relaxed (degree 24 and 128
against 16, above), and the row should say so, or the proposer will either
fail the contract or quietly skip the AM check. The row's "default: step = J"
should also be marked as valid only for single-magnitude models; a general
cap rule has to be declared in the hashed request. Both fixes are in the
proposed queue below.

**`dtm-small-image` (held).** Decision: whether the lab's first task-level
claim ("at this quality, projected energy A against measured B") can be made
at all, and at what bits and jitter. This is amendment 1's priority 2 and the
row that would satisfy all three clauses of the four-week test for the
hardware's stated application. Cheapest probe: train the smallest chained
EBM on a synthetic set with an exact likelihood, on CPU, and compare held-out
reconstruction at float32 against 6 and 8 bits under a declared cap rule; if
quality collapses at 8 bits, the row's arms change before any protocol.
Source cards: `extropic-dtm-hardware.md` C2 asserts GPU parity at about
10,000x less energy on a simple image benchmark, under Extropic's hardware
model; `extropic-torx-paper.md` C7 gives a binary-MNIST denoising reference
point (bit-error rate 0.114 at 28 x 28, 10 steps);
`wang-deng-online-learning.md` C3 claims lower coupling precision suffices
for online-trained continuous models. None answers the row: they are the
claims the row checks against the lab's own cost model and a conventional
baseline. The card's caution applies: record the paper version and
cost-model assumptions before comparing. Not a third variant. Keep held;
fix the bit-grid wording when it opens.

**`am-io-budget` (held).** Decision: whether reads, writes and clamp changes
dominate the projected energy of one recalled pattern, which decides whether
the AM task is I/O-bound or update-bound on the chip (Z1T's own projection
has the FPGA at over 95% of energy, `extropic-z1t.md`, cautions). Cheapest
probe: this row may need no new sampling. Reads, writes and clamps per recall
are deterministic given stage A's schedule (cue clamp, K two-block sweeps,
readout), and the sweeps-to-recall per cell are archived
(`docs/experiment-reports/2026-10-08-am-coupling-bits/`,
`docs/experiment-reports/2026-10-07-am-binary-emulation/`). Price the
archived runs through the layer, as the exchange cost projection priced
archived cells, and add one measured CPU-Gibbs timing on the same model for
rule 10's baseline. That is a day, not a study. Source cards: none address
I/O per recall. Not a third variant. **Missing piece**: the row has no
conventional baseline, which amendment 1 (priority 3, rule 10) says makes a
projected number "not a result". Added below.

**`e0-remaining-kernels` (open, P-0003 awaiting owner).** Decision: none
that any row needs. If THRML mismatched a kernel, the consequence would land
on M5-derived task work, which amendment 1 (priority 4) forbids until the
upstream implementation is public. Already answered, three times, by the
lab's own records (lessons rows cited above) and once by the probe's exact
side (31,232 inputs match the archived law to 4e-15; P-0003 body). Cheapest
alternative: nothing, or a K = 1 pass over the 16 fast kernels as a probe if
the owner wants coverage of the controls that lack power at K = 4. Not a
third variant. **Drop.**

**Done rows** (`am-coupling-bits`, `planar-16-offset-ferro`): unchanged.

## Rubric 3: can the lab state the four-week test today?

No. Clause by clause, for the one task that has numbers (associative-memory
recall, stage A's `bias` design):

1. *Precision and noise tolerance.* Yes, with a caveat: 6 bits at step = J
   and ±10% static jitter negligible (`exact_reference` and
   `software_simulation`, am-coupling-bits findings), for a design not yet
   placed on the chip's graph.
2. *Sweeps, reads and writes per useful sample.* Sweeps: partly, from the
   archived K-to-recall cells. Reads and writes: no number anywhere.
3. *Projected energy at a stated quality against a conventional baseline.*
   No number for any task. The planar studies price sweeps but have no task
   and no CPU baseline at matched quality.

The row that closes the largest gap at the lowest cost is `am-io-budget`
with a CPU baseline added, run right after `constraint-layer`: it turns one
task from one clause into three, mostly from archived evidence.
`dtm-small-image` closes the same gap for the hardware's stated task and
should follow, but its probe has to size training first and it is weeks,
not days. The four-week clock started on 2026-10-08.

## The one thing to drop

`e0-remaining-kernels`, with P-0003 rejected. Reasons above.

## Proposed queue

Changes, each in one sentence:

- **Drop** `e0-remaining-kernels` (status `dropped`, text unchanged): a
  fourth THRML contract, 8 CPU-hours, informs no row.
- **Reorder** `am-io-budget` above `dtm-small-image`: it completes the
  four-week statement for one task from mostly archived evidence; both stay
  `held` and the owner opens `am-io-budget` when `constraint-layer` merges.
- **Edit (text only)** `constraint-layer`, Notes column: state that the AM
  recall check runs with placement relaxed because the `bias` design has
  degree 24 and up to 128 against the lattice's 16, and that step = J is a
  default only for single-magnitude models.
- **Edit (text only)** `am-io-budget`, Metric and Notes columns: add the
  measured CPU-Gibbs baseline rule 10 requires, and say the row prices
  archived runs first.
- **No new row.** The placement question the AM result raises belongs inside
  `constraint-layer`, and the denoiser cap-rule question belongs inside the
  `dtm-small-image` probe.

`adopt D-0001` replaces the table with this block verbatim. The owner who
wants the drop and reorder without the two text edits should apply them by
hand instead.

```queue
| `am-coupling-bits` | done | How does associative-memory recall fall with coupling precision and per-site beta noise? | Stage A's bias-binary design with 4-, 6- and 8-bit couplings from the M5b codebook and ±10% per-site beta jitter | Recall against coupling bits and beta jitter, using stage A's exact references | Produces a spec number: the coupling bits needed to keep recall within tolerance. |
| `planar-16-offset-ferro` | done | Do the planar Ising scaling allocations (PR #92) hold on the hardware-shaped graph? | Zero-field ferromagnetic Ising on the planar subgraph of the 16-offset lattice; the thin five-replica ladder; the exact Kac-Ward reference imported from `planar_ising_scaling` | Which arms qualify, and at what sweep budget, against the exact reference | Known-good regime that confirms #92 on the chip's graph. Ferro targets only; the frustrated-grid track is reserved for the owner's annealing study. |
| `constraint-layer` | open | Does one study-local constraint layer reproduce the archived placement, rounding and finite-sweep contracts bitwise, and price updates, reads, writes and clamp changes separately? | A `thermo_lab.constraints` module that places any pairwise model on the 16-offset lattice (M5c's offsets and placement, or the planar subgraph used by `planar-16-offset-ferro`), quantizes couplings and biases with M5b's `round_parameters` codebook under a declared cap (default: step = J, the `coupling` cap `am-coupling-bits` found sufficient at 6 bits), applies static per-site beta jitter, runs chromatic block Gibbs only with no replica exchange, and counts updates, node reads, full SRAM writes and clamp changes into the existing `Z1HardwareProfile` | Bitwise agreement with the M5c placement archive, the M5b rounding archive, the `am-coupling-bits` quantized recall at 6 bits and the E0 stage B inner-K law through the layer; one priced example per constraint with the relaxed-constraint flag recorded in the hashed request | A contract study, not a sampler study: short, exact, CPU. Every later row runs through the layer by default and states which constraints it relaxed (charter amendment 1, rule 8). Reuse only; no edits to hash-bound sources. The `am-coupling-bits` check runs with placement relaxed and says so: the `bias` design is complete bipartite (hidden degree 24, visible degree up to 128) against the lattice's 16, and its free hidden units cannot be copied the way M5c copied clamped inputs; report what placing it would cost. "Step = J" is a default only for models whose couplings share one magnitude; the hashed request declares the cap rule for every model (D-0001). |
| `am-io-budget` | held | How many node reads, full SRAM writes and clamp changes does one recalled pattern cost in the bias associative-memory design under the constraint layer, and how does that I/O compare with its update count? | Stage A's `bias` design at 6-bit couplings with step = J (the `am-coupling-bits` spec), cue write and readout counted through `constraint-layer`, Z1 pricing | Reads, writes, clamps and updates per recalled pattern at stage A's recall tolerance; the I/O share of projected energy; and the measured CPU seconds of Gibbs on the same model at the same recall, so the row reads "projected A against measured B" (charter amendment 1, rule 10) | Held; the I/O half of the associative-memory spec, now that the precision half is recorded. Open it when `constraint-layer` is done, before `dtm-small-image`: it completes the four-week test for one task (D-0001). Price the archived stage A and `am-coupling-bits` runs first, as the exchange cost projection priced archived cells; new sampling only if the archived budgets do not cover the question. The CPU baseline is part of the row, not a separate row. |
| `dtm-small-image` | held | At what task quality does a small chained-EBM denoiser run under the constraint layer, as a function of coupling bits, beta jitter, sweeps per step and reads and writes per sample, and what does it cost against CPU Gibbs on the same model and a small conventional model of the same quality? | A two- to four-step discrete denoising model (the architecture in the DTM source card) on a small binary image set: 8 x 8 to 16 x 16 binarized digits or clothing, or a synthetic set with an exact likelihood; trained in software, inference under `constraint-layer` at 6 and 8 bits (step = J) and 0 and 10 percent jitter; baselines inside the row | Held-out reconstruction error and log-likelihood where tractable, per projected joule and per measured CPU second, against the baselines; the bits, jitter, sweeps and I/O at which task quality stays within tolerance | Held until `constraint-layer` is done; the owner flips it to open. The lab's first task-level benchmark (amendment priority 2). Baselines are part of the row (rule 10), not a separate row. The probe must size training cost and say whether a synthetic exact-likelihood set or a real binarized set is the first target; if both fit in budget, synthetic first. |
| `e0-remaining-kernels` | dropped | Does THRML match the archived exact inner-K law for every compiled M5a kernel, as E0 stage B showed for one? | The other 59 compiled M5a kernels (reading B) at K = 1, 2 and 4, with stage B's contract, tolerances and controls | Cells passing on output rate and joint, per kernel; control rejections | Pure replication. Stage B took about 11 CPU-minutes for one kernel, so all 59 is roughly 11 CPU-hours before parallelism; the probe sizes it, and a run over 8 CPU-hours needs the owner's say-so. Last in the order: a contract worth having, low marginal information; idle-time work. |
```

## Evidence classes

Nothing in this review is new evidence. Every number is quoted from the
cited findings, lessons rows, probes or source cards with its original
label. No hardware claim.
