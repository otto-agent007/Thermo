# Direction review D-0003

2026-10-15. Main commit 5258a1c8254dd483d50e9af15ad6ea27081d997d. THERMES-DIRECTOR.
**Advice, not a decision.** The owner edits the queue; this file proposes.

Covers the one loop result accepted since D-0002 (P-0006 `dtm-small-image`,
PR #118), nothing in flight, and reads the queue against charter amendment 1
(`PROJECT_CHARTER.md`, "Amendment 1"). Previous review: D-0002
(`docs/research/2026-10-14-direction-review.md`, PR #108, merged; its queue
changes landed in PR #110 and PR #117).

## The change I would make first

Open one row, `dtm-baseline-headroom`, and make its probe decide whether the
denoising line continues at all. The queue at 5258a1c has no open row: four
`done`, two `dropped`. The loop is idle until the owner adds one.

P-0006 is the lab's first complete four-week statement for a placeable model
(`docs/experiment-reports/2026-10-14-dtm-small-image/findings.md`, "The
number"), and the statement is a negative: a 114-parameter logistic
denoiser is ahead of the lattice EBM in every seed at about 1,000 times less
CPU (findings, "Cost against conventional machines"). The reason is visible
in the same table. The exact Bayes floor is 0.067 and the logistic sits at
0.071, so a 7 x 7 receptive field already captures nearly everything the
posterior has to say on this family. A denoiser cannot beat a local model
on a task whose posterior is local. That is a fixed choice (the data
family) bounding the metric (the EBM's room to win), the pattern CLAUDE.md
warns about ("Check assumptions before building process"). The probe
scanned the family for headroom between noisy input and Bayes
(`docs/research/2026-10-14-dtm-small-image-probe.md`, finding 5) but not for
headroom between a small local model and Bayes, and that second gap is the
one the charter's question needs. Before the lab trains another denoiser,
it should know whether any small binary family at L <= 16 has such a gap.
If none does, the denoising task at this size cannot show an advantage and
the line should stop, not vary.

## Verdicts

**P-0006 `dtm-small-image`: positive as a study, negative as a task.** It
delivered the numbers amendment 1 asked for, under placement with nothing
relaxed, with both sides of the accounting and a conventional baseline
inside the row (findings, "Hardware constraints", "Cost against
conventional machines"; `docs/knowledge/lessons.md`, row 2026-10-14). The
task result is that the hardware's stated application, at L = 8 on an
exact Ising-mixture family, is done better and about 1,000x cheaper in CPU
by a 114-parameter digital model. The findings and summary say so plainly
and do not soften it. Two facts from it should shape every later task row:

1. **The projected joule is one unpublished constant.** At the designer's
   point (T = 2, 4 bits, K = 64) the sample costs 2.01e-8 J with clamp loads
   priced as full SRAM writes and 6.1e-10 J with them priced as node reads
   (findings, "The number"). That is a 33x band set by the sealed profile's
   rule that a clamp-state change is a write because "the paper does not
   publish a cheaper clamp-only constant" (`docs/z1-hardware-model.md`,
   "Refined Appendix-B operation model"). Bits never enter the price and K
   moves it under 3 percent (probe, finding 1; findings, "Energy is I/O").
   The number a designer can use is therefore the count, 2 T L^2 = 256 reads
   and writes per sample, not the joule. Every "per projected joule" claim
   the lab makes should carry the band until Extropic publishes the
   constant, and no row should hinge on a factor smaller than 33.
2. **The chip's free start is a trap.** Carrying free-node state between
   images (the only start that costs no write) left one of three T = 1
   models 0.018 above its own K = 1024 reference at 256 sweeps, and that
   reference was itself 0.020 behind a random start (findings, "What the
   result means"). T = 2 escapes within 64 sweeps in every seed at twice the
   I/O. The lessons row records the check (carried against random start,
   per seed); any chained-sampler row inherits it.

One process fault, not yet in the lessons register: decision rule 6
(`docs/experiments/dtm-small-image.md`, "Primary decision rule") coded the
operating point as "the cheaper T when T = 2 does not help", and the
inconclusive steps verdict sent it to T = 1, the point where K\*, b\* and the
3-bit control did not resolve. The archive's `completion.json` therefore
carries an operating point the findings tell the reader to ignore. A rule
that defaults to the cheaper arm should require that arm to have resolved.
Worth a lessons row.

## Rubric 1: the accepted result

### P-0006 DTM small-image denoiser

- **Design number?** Yes, the first set for a placeable model, from the
  T = 2 column (findings, "The number"): 4 bits under the `split` cap; 10
  percent static beta jitter negligible; K = 64 sweeps per step, T = 2;
  128 reads, 128 writes, 0 clamp changes and 24,576 active updates per
  sample; 16 draws per estimate; error 0.075 against Bayes 0.067, logistic
  0.071 and noisy input 0.199; 2.01e-8 J per sample (band to 6.1e-10 J),
  3.2e-7 J per estimate. Labels: `exact_reference` for the floor,
  `software_simulation` for every denoiser and the baseline,
  `calibrated_projection` for every joule. At T = 1 nothing resolves
  (K\* > 256, b\* > 8 at zero jitter, 3-bit control undetectable in
  seed 1), and that is reported as a result, not hidden.
- **Hardware constraints.** Applied: placement on the 16-offset lattice
  with nothing relaxed (maximum degree 16, the first recorded study whose
  model passes `require_placement` unrelaxed), M5b codebook
  quantization under `split`, static per-site jitter, two-colour chromatic
  Gibbs, no exchange, I/O counted per sample on the host-per-step path.
  Relaxed, flagged in the hashed request: quantization on the float arm
  and jitter on zero-jitter arms, both controls. Outside the layer:
  training in float64 software, stated (findings, "Integrity"; protocol,
  "Hardware constraints applied and relaxed"). Rule 8 met.
- **Resource accounting.** Yes, on both sides: updates, reads, writes and
  clamp changes with the joule band, and `time.process_time` per estimate
  for THRML, NumPy Gibbs, the exact posterior and the logistic (findings,
  "Cost against conventional machines"). The logistic's energy is not
  priced, so the comparison is projected joules against measured CPU
  seconds, which is what amendment 1 priority 3 asks for and no more. Rules
  9 and 10 met.
- **Missed expectations.** Four of six (findings, "Pre-registered
  expectations"): 1 (float within 0.007 of Bayes) missed at T = 1; 2 (4 bits
  misses) wrong at T = 2, where 4 bits passes; 4 (K\* = 256 at T = 1)
  missed, K\* > 256; 5 (T = 2 no better) not returned. All four trace to
  one seed trapped under continuation at T = 1, which the round-1 probe
  saw as +0.012 at K = 64 on one seed
  (`docs/research/2026-10-14-dtm-small-image-probe-r1.md`, finding 3) and
  sized as a sweep-count effect. A one-seed probe cannot size a
  seed-dependent trap; the lab already has that lesson twice (lessons,
  rows 2026-10-08 "Planar 16-offset ferro" and 2026-10-07 "Associative
  memory A2"). No queue row assumed the opposite: the queue has no open
  row. The follow-up THERMES names (a continuation reset rule,
  `docs/research/2026-10-14-dtm-small-image-next.md`) assumes T = 1
  continuation is worth rescuing; see "The one thing to drop".

### Pattern

The task line has one negative, not two. The adaptive-policy line stays
at two (lessons, "What Thermo has learned so far") and stays out of
bounds. No premise question is forced on the owner by the rule. One
question is still worth asking, because it decides what the four-week
test's third clause can mean: is a projected joule that is 97 percent one
unpublished clamp-write constant a number the lab should report as a
headline, or should task rows report reads and writes per sample as the
headline and the joule only as a band? The answer changes how the next
row's metric is written.

## Rubric 2: the queue rows

The queue at 5258a1c has six rows: `am-coupling-bits`,
`planar-16-offset-ferro`, `constraint-layer`, `dtm-small-image` (done) and
`am-io-budget`, `e0-remaining-kernels` (dropped). None is open.

**Done rows.** `am-coupling-bits` gave the 6-bit spec for an unplaceable
design (D-0002, "Rubric 3"); `planar-16-offset-ferro` was the last
sampler-allocation study (D-0001, "Verdicts"); `constraint-layer` is the
layer every row now runs through; `dtm-small-image` is reviewed above. No
decision depends on re-running any of them. Unchanged.

**Dropped rows.** `am-io-budget` and `e0-remaining-kernels` were dropped by
the owner in PR #117 on D-0002's and D-0001's reasons. Nothing in P-0006
reopens either: the AM `bias` design is still unplaceable, and the
remaining E0 kernels still inform no row. Unchanged.

**Proposed new row `dtm-baseline-headroom` (open).** The question: is there
a small binary denoising family at L <= 16, with an exact posterior, on
which the best small local digital model sits well above the Bayes floor,
and does the P-0006 lattice denoiser close that gap under the constraint
layer at its recorded operating point?

- *Decision it informs.* Whether the denoising task (amendment 1,
  priority 2) continues on this hardware model, or the lab records the
  line as "no advantage at L <= 16 over a small digital model" and moves to
  the second task. Today that decision would be made on one family chosen
  for noisy-to-Bayes headroom, which is the wrong gap.
- *Cheapest probe that makes the study unnecessary.* The exact scan alone,
  with no EBM: for each candidate family compute the Bayes MPM error by
  transfer matrix (1.7 ms per image at L = 8, 0.37 s per component at
  L = 16; probe, finding 2) and fit the 114-parameter logistic (seconds).
  Candidates that break locality: higher flip noise (q of 0.3 to 0.4, where
  the template must be inferred from the whole image), more templates, and
  template fields that differ only at long range. If no family at L <= 16
  gives logistic minus Bayes of at least 0.03 (about ten times P-0006's
  resolution of 0.002 to 0.004), the probe answers the row with a docs-only
  PR and the line stops. If one does, the study trains the P-0006 model on
  it (three seeds, about 80 CPU-s per layer) and runs only the recorded
  operating point (T = 2, 4 bits, 10 percent jitter, K = 64, with the
  float K = 1024 reference and a random-start control), so it is a fraction
  of P-0006's 4.8 CPU-hours.
- *Already in a source card?* No. `extropic-dtm-hardware.md` C2 claims GPU
  parity at 10,000x less energy on "a simple image benchmark" against a
  GPU, not against a small digital model of equal error, and it is a
  projection. `extropic-torx-paper.md` C7 reports bit-error rate 0.114 to
  0.113 on 28 x 28 binary MNIST with no baseline of equal size. Neither
  says whether a lattice EBM has room over a local model on any family.
  The lab has no source card on classical MRF image denoising (Geman and
  Geman 1984; Besag 1986), which is where "a local prior gives a nearly
  local posterior" comes from; the probe should add one before the
  protocol, as amendment 1 asks, and it is reading, not a row.
- *Already known inside the lab?* The exact posterior and Bayes floor are
  tractable for nearest-neighbour grids of width at most 24
  (`docs/experiment-reports/2026-10-08-planar-annealing/findings.md`;
  P-0006 preflight). Independent Gibbs is the sampler for denoising
  posteriors (lessons, 2026-10-02). The carried-state check is required
  per seed (lessons, 2026-10-14). The row inherits all three.
- *Third variant?* No. It is the first follow-up on a line with one
  negative. If its study also finds the logistic at least as good, that is
  two consecutive negatives of the same kind and the next review's only
  output on denoising is a premise question for the owner.

## Rubric 3: can the lab state the four-week test today?

Yes, for one task, with the evidence classes attached, 27 days before the
deadline (the test is dated from the layer's merge, 2026-10-14, so
2026-11-11; `docs/experiments/dtm-small-image.md`, opening paragraph). For
the lattice-native denoiser on the exact Ising-mixture family at L = 8:

1. *Precision and noise tolerance.* 4 bits under the `split` cap; static 10
   percent beta jitter negligible (`software_simulation` against
   `exact_reference`; findings, "The number"). Resolved at T = 2 only.
2. *Sweeps, reads and writes per useful sample.* 64 sweeps per step at
   T = 2; 128 reads and 128 writes per sample, 0 clamp changes, 24,576
   active updates; 16 draws per estimate (algorithmic counts).
3. *Projected energy at a stated quality against a conventional baseline.*
   At pixel error 0.075 (Bayes 0.067), 2.01e-8 J per sample under the
   sealed profile, 6.1e-10 J with clamp loads at the read constant
   (`calibrated_projection`), against a logistic model at error 0.071 and
   5.2e-5 CPU-s per estimate (measured software). The baseline's energy is
   not priced; the charter asked for "projected A against measured B" and
   that is what the lab has.

The amendment asked whether the lab could state these numbers, not whether
they would favour the hardware. It can, and they do not. The caveats are
the 33x band in clause 3 and the fact that the family is one where a local
model is near-optimal. The row that closes the largest remaining gap is the
one proposed above: it tests whether clause 3 can ever read in the
hardware's favour on this task at this size. If it cannot, the owner should
revisit the task choice, as the amendment says, rather than add rows.

## The one thing to drop

The T = 1 continuation reset-rule follow-up (findings, "Follow-ups", first
item; `2026-10-14-dtm-small-image-next.md`, candidate rows). It is a
sampler start-rule question, which amendment 1 (priority 5) makes a probe
at most. Its best case, T = 1 at about 1.0e-8 J against T = 2 at 2.0e-8 J,
is a factor of 2 inside the 33x band of the constant that sets both
numbers, and the quality it would rescue is already available at T = 2 in
every seed. The owner should not open a row for it. The real binarized set
(second follow-up) stays a candidate for after the headroom row, because a
real set has no exact floor and cannot answer the headroom question first.

## Proposed queue

Changes, each in one sentence:

- **Add** `dtm-baseline-headroom` (`open`), placed after `dtm-small-image`
  as the only open row: it decides whether the denoising line continues,
  and its probe can close it in a docs-only PR without a study.
- **No other change.** Done rows stay done; the two dropped rows stay
  dropped, as the owner set them in PR #117.

```queue
| `am-coupling-bits` | done | How does associative-memory recall fall with coupling precision and per-site beta noise? | Stage A's bias-binary design with 4-, 6- and 8-bit couplings from the M5b codebook and ±10% per-site beta jitter | Recall against coupling bits and beta jitter, using stage A's exact references | Produces a spec number: the coupling bits needed to keep recall within tolerance. |
| `planar-16-offset-ferro` | done | Do the planar Ising scaling allocations (PR #92) hold on the hardware-shaped graph? | Zero-field ferromagnetic Ising on the planar subgraph of the 16-offset lattice; the thin five-replica ladder; the exact Kac-Ward reference imported from `planar_ising_scaling` | Which arms qualify, and at what sweep budget, against the exact reference | Known-good regime that confirms #92 on the chip's graph. Ferro targets only; the frustrated-grid track is reserved for the owner's annealing study. |
| `constraint-layer` | done | Does one study-local constraint layer reproduce the archived placement, rounding and finite-sweep contracts bitwise, and price updates, reads, writes and clamp changes separately? | A `thermo_lab.constraints` module that places any pairwise model on the 16-offset lattice (M5c's offsets and placement, or the planar subgraph used by `planar-16-offset-ferro`), quantizes couplings and biases with M5b's `round_parameters` codebook under a declared cap (default: step = J, the `coupling` cap `am-coupling-bits` found sufficient at 6 bits), applies static per-site beta jitter, runs chromatic block Gibbs only with no replica exchange, and counts updates, node reads, full SRAM writes and clamp changes into the existing `Z1HardwareProfile` | Bitwise agreement with the M5c placement archive, the M5b rounding archive, the `am-coupling-bits` quantized recall at 6 bits and the E0 stage B inner-K law through the layer; one priced example per constraint with the relaxed-constraint flag recorded in the hashed request | A contract study, not a sampler study: short, exact, CPU. Every later row runs through the layer by default and states which constraints it relaxed (charter amendment 1, rule 8). Reuse only; no edits to hash-bound sources. Built as engineering: [plan and As built](research/2026-10-13-constraint-layer-plan.md). |
| `dtm-small-image` | done | At what task quality does a small chained-EBM denoiser run under the constraint layer, as a function of coupling bits, beta jitter, sweeps per step and reads and writes per sample, and what does it cost against CPU Gibbs on the same model and a small conventional model of the same quality? | A two- to four-step discrete denoising model (the architecture in the DTM source card) defined on the 16-offset lattice from the start (sites are lattice sites, edges are lattice offsets, two colours) so that it passes `require_placement` with nothing relaxed; on a small binary image set: 8 x 8 to 16 x 16 binarized digits or clothing, or a synthetic set with an exact likelihood; trained in software, inference under `constraint-layer` at 4, 6 and 8 bits under a cap rule the probe declares (`absolute` or `split`; `coupling` is undefined for trained weights) and 0 and 10 percent jitter; baselines inside the row | Held-out reconstruction error and log-likelihood where tractable, per projected joule and per measured CPU second, against the baselines; the bits, jitter, sweeps and reads and writes per sample (under the declared I/O path) at which task quality stays within tolerance | Opened by D-0002 now that `constraint-layer` is done. The lab's first task-level benchmark (amendment priority 2). Baselines are part of the row (rule 10), not a separate row. The probe settles these fixed choices before any protocol: (1) the I/O path per sample (per-step host readback and rewrite, or on-chip chaining), because the sealed profile prices a clamp change as a full SRAM write at about 21,665 updates and I/O was 98 to 99.996 percent of energy in every example the layer priced, so this choice, not sweeps or bits, sets the projected joule; (2) which graph the exact reference covers (a nearest-neighbour grid of width at most 24 has a transfer-matrix likelihood; the long offsets do not); (3) whether a synthetic exact-likelihood set or a real binarized set is first, synthetic first if both fit; and it sizes training cost. No tempering arms: the layer has no exchange, and independent Gibbs won on denoising posteriors (lessons, 2026-10-02). Declare everything the plan's "What a later row must declare" lists. |
| `dtm-baseline-headroom` | open | Is there a small binary denoising family at L of at most 16, with an exact posterior, on which the best small local digital model sits well above the Bayes floor, and does the P-0006 lattice denoiser close that gap under the constraint layer at its recorded operating point? | The P-0006 generator (mixture of nearest-neighbour Ising grids with template fields, exact transfer-matrix posterior) with the family varied to break locality: flip noise q of 0.3 to 0.4, more templates, long-range template differences; L = 8 first, L = 16 only if the gap appears there; the P-0006 lattice-native model and training recipe, three seeds, under `constraint-layer` at the recorded operating point (T = 2, 4 bits under `split`, 10 percent jitter, K = 64, host-per-step I/O, continuation with the float K = 1024 reference and a random-start control) | Primary: the exact gap of the 114-parameter 7 x 7 logistic above the Bayes MPM floor per family; a family qualifies at 0.03 or more. Secondary, on a qualifying family only: the DTM's error against the logistic and against Bayes in every seed, with reads, writes and updates per sample and the joule reported as the band (profile write constant and clamp loads at the read constant) | Proposed by D-0003. Decides whether the denoising task (amendment 1, priority 2) continues at this size or stops as "no advantage over a small digital model"; today that decision rests on one family whose posterior is local (P-0006: Bayes 0.067, logistic 0.071). The probe is the exact scan alone, no EBM, under 30 CPU-minutes; if no family qualifies, the probe closes the row with a docs-only PR. Add a source card on classical MRF image denoising (Geman and Geman 1984; Besag 1986) before any protocol. Inherits the per-seed carried-state check (lessons, 2026-10-14). A second logistic-at-least-as-good result is two consecutive negatives and goes to the owner as a premise question, not a third variant. |
| `am-io-budget` | dropped | How many node reads, full SRAM writes and clamp changes does one recalled pattern cost in the bias associative-memory design under the constraint layer, and how does that I/O compare with its update count? | Stage A's `bias` design at 6-bit couplings with step = J (the `am-coupling-bits` spec), cue write and readout counted through `constraint-layer`, Z1 pricing | Reads, writes, clamps and updates per recalled pattern at stage A's recall tolerance; the I/O share of projected energy | Dropped by D-0002. The layer's priced example answers the question (12 reads, 32 writes, 5,120 updates per recall at P = 8; I/O 99.3 percent of modelled energy; the same arithmetic holds across stage A's cells), and the `bias` design cannot be placed (hidden degree 24, visible up to 128 against 16), so a projected joule for it prices a machine that does not exist. A placeable memory is a different design; read the diluted-memory literature and add a source card before opening a row for it. |
| `e0-remaining-kernels` | dropped | Does THRML match the archived exact inner-K law for every compiled M5a kernel, as E0 stage B showed for one? | The other 59 compiled M5a kernels (reading B) at K = 1, 2 and 4, with stage B's contract, tolerances and controls | Cells passing on output rate and joint, per kernel; control rejections | Pure replication. Stage B took about 11 CPU-minutes for one kernel, so all 59 is roughly 11 CPU-hours before parallelism; the probe sizes it, and a run over 8 CPU-hours needs the owner's say-so. Last in the order: a contract worth having, low marginal information; idle-time work. Dropped by D-0001 (owner, 2026-10-10: later); the change never reached the queue and the row was re-proposed as P-0005. |
```

## Evidence classes

Nothing in this review is new evidence. The 33x band is the ratio of two
figures the findings report side by side under the sealed profile; the
per-sample counts are the protocol's own arithmetic. Every other number is
quoted from the cited findings, protocol, probes, lessons rows or source
cards with its original label. No hardware claim.
