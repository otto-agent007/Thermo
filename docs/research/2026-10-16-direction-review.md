# Direction review D-0004

2026-10-16. Main commit 7d351724ada07e7efc9f35bf5c26cd9d1053724a. THERMES-DIRECTOR.
**Advice, not a decision.** The owner edits the queue; this file proposes.

Covers the one loop result accepted since D-0003 (P-0007
`dtm-baseline-headroom`, PR #122), nothing in flight, and reads the queue
against charter amendment 1 (`PROJECT_CHARTER.md`, "Amendment 1").
Previous review: D-0003 (`docs/research/2026-10-15-direction-review.md`,
PR #120, merged; its row landed in PR #121).

## The change I would make first

Stop treating the trained P-0006 denoiser as the hardware side of the
denoising task, and ask the owner for a stop rule before the line continues.

The denoising line now has two consecutive negatives of the same kind: a
trained lattice DTM at P-0006's recipe loses to a small digital model, first
on a local family (P-0006, `docs/experiment-reports/2026-10-14-dtm-small-image/findings.md`,
"Cost against conventional machines") and now on three families built to
have digital headroom (P-0007, `docs/experiment-reports/2026-10-16-dtm-baseline-headroom/findings.md`,
"The number"). Under the loop's rule the director's only output on this line
is a premise question. The owner has already answered one ("continue",
`CLAUDE.md`, "DTM baseline headroom", 51d27ed) and named the next question:
whether a hand-built model of the exact posterior, with its global latent
spread over the lattice, reaches Bayes under the constraint layer before any
training. That reframing is sound, and the question below is about what
ends it, not whether to ask it.

## Verdicts

**P-0007 `dtm-baseline-headroom`: negative, and the more useful of the two
negatives.** Stage 1 found what D-0003 asked for: three exact L = 8 families
where no digital model under X = 320,826 MACs per image reaches Bayes + 0.01
(findings, "Stage 1"; on `smooth16` none in the ladder does at any size).
Stage 2 then showed the P-0006 model does not use the headroom: 0.078 to
0.147 above Bayes at its operating point, and the 3,136-MAC 7 x 7 logistic
matches it in all nine seeds (findings, "Stage 2"). Two facts separate this
from P-0006's negative:

1. **The model is the limit, not the constraints.** The float K = 1024
   reference (nothing quantized, no jitter) is 0.25 to 0.39, and the logistic
   matches it too in every seed (findings, "The model, not only the point, is
   the limit"). So raising bits or sweeps would not have changed the verdict.
   The DTM's latents are local; these posteriors hinge on which template
   generated the image, a global latent the model has no unit for.
2. **Continuation failed in nine of nine seeds.** A fresh uniform start beat
   the carried state by 0.008 to 0.107 with every interval past -0.01
   (findings, "Rule 6 fails in every seed"). P-0006 saw this on one of three
   seeds at T = 1; on families with a global latent it is the rule. Any row
   that carries free-node state between images must price an alternative
   start. The study priced one: a random start is 7.9e-8 J per sample against
   2.0e-8 under continuation, because 384 free-node writes are full SRAM
   writes (`evidence.tar.gz`, `study.json.gz`,
   `evaluation.pricing.random_start_per_sample`).

The study did what its protocol said, reported the negative in its first
line, kept the three evidence classes apart, and stated rule 8 (after the
owner's one-sentence fix at 51d27ed). The reviewer replayed the archive in
full. Nothing to soften.

Two process faults, both already in the lessons register (row 2026-10-16):
the step-cost calibration returned negative per-step costs from
compile-dominated short fits and underestimated by 2.7x, so the 10 CPU-hour
stop rule protected nothing; and 1.4 of the 6 CPU-hours went to training and
running the expensive arm after the cheapest rung already matched its float
reference.

## Rubric 1: the accepted result

### P-0007 DTM baseline headroom

- **Design number?** Not the one the protocol named. The break-even pJ per
  MAC on a qualifying family needed a qualifying family, and there is none
  (findings, "The number"). What it did produce is a negative a designer can
  use: the P-0006 recipe and operating point do not transfer off the local
  family (summary, first paragraph), and three archived families with their
  exact floors and a fitted 270-fit ladder now exist as a fixed target set
  (`t4`: Bayes 0.216, cheapest reaching rung 1.74e6 MACs; `smooth8`: 0.219,
  3.43e6; `smooth16`: 0.230, none). The archived ladder is the design input
  the next row should reuse, not refit.
- **Hardware constraints.** Operating-point arm relaxed nothing: placement on
  the 16-offset lattice, 4-bit `split` codebook, 10 percent static jitter,
  two-colour Gibbs, no exchange, host-per-step I/O (findings, "Evidence
  classes and claims"). Float K = 1024 arm relaxed `quantization` and
  `beta_jitter` with reasons in the hashed request. Training and the digital
  ladder are float software outside the layer, stated (protocol, "Hardware
  constraints applied and relaxed"). Rule 8 met.
- **Resource accounting.** Yes, both sides projected under the queue's
  standing rules: 128 reads, 128 writes, 0 clamp changes, 24,576 updates per
  sample; joule band 3.2e-7 to 9.7e-9 J per 16-draw estimate; the logistic at
  3,136 MACs priced at 0.1, 1 and 10 pJ per MAC; CPU seconds for every arm
  beside them (findings, "Cost, both sides projected", "CPU seconds"). Rules
  9 and 10 met. This is the first comparison in the lab where both sides are
  projected joules.
- **Missed expectations.** One of five (findings, "Pre-registered
  expectations"): the DTM was expected 0.02 to 0.06 above Bayes on the
  candidates and landed 0.078 to 0.147. Expectation 5 (no family qualifies)
  held, so the protocol's authors expected the negative. The queue at
  7d35172 has no open row, so no row assumed the opposite. One recorded
  assumption does: the owner's continue note says 99 percent of the DTM's
  joule is I/O "which does not grow with latent nodes or sweeps"
  (`CLAUDE.md`, "DTM baseline headroom"). That holds under continuation
  only. Under a random start, free-node writes grow with every latent node
  (384 at three free nodes per pixel, 7.9e-8 J per sample, archive pricing),
  and continuation is the start that failed nine of nine. The cost case for
  continuing rests on a carried state working for a model with a global
  latent, which no result yet shows.

### Pattern

Two consecutive negatives of one kind on the denoising line: trained lattice
DTM at P-0006's recipe against a digital ladder on exact synthetic families.
The adaptive-policy line stays at two and out of bounds (lessons, "What
Thermo has learned so far"). The conservation line stays closed.

**Premise question for the owner.** The continue decision names a new model
but no end. What result on the hand-built row stops the denoising line?
Two candidates, either of which the probe can return in under an hour:
(a) the placed, quantized exact-posterior model's own equilibrium error
(by enumeration over hidden configurations, no sampler) is above Bayes +
0.01 at 8 bits, which says the lattice at the chip's precision cannot hold
the posterior; or (b) it holds it, but no priced start reaches Bayes + 0.01
within K = 1024 sweeps per step, which says chromatic Gibbs cannot switch a
replicated global latent in budget. If neither stops it, a third negative
after training would be the fourth of its kind and the line should have
been stopped here. The same answer should say whether the real binarized
set opens on a positive only.

## Rubric 2: the queue rows

The queue at 7d35172 has seven rows: `am-coupling-bits`,
`planar-16-offset-ferro`, `constraint-layer`, `dtm-small-image`,
`dtm-baseline-headroom` (done) and `am-io-budget`, `e0-remaining-kernels`
(dropped). None is open; the loop is idle until the owner adds one.

**Done rows.** No decision depends on re-running any of them. Two now
matter as inputs to the next row: `am-coupling-bits` gave the bit budget
for a design whose hidden units are one per stored pattern with a negative
bias, 6 bits with fields encoded in units of J (lessons, 2026-10-08), and
`dtm-baseline-headroom` gives the target family, its exact floor and its
priced digital ladder. Unchanged.

**Dropped rows.** `am-io-budget` was dropped because the AM `bias` design
has hidden degree 24 and visible degree up to 128 against the lattice's 16
(queue row notes). That same obstacle is the engineering content of the
next row: a global latent coupled to 64 pixels must be split into tied
copies to be placed. The drop stands; the lesson transfers. `e0-remaining-kernels`
still informs no row. Unchanged.

**Proposed new row `dtm-exact-posterior-lattice` (open).** This is the
owner's decision of 2026-10-16 written in queue form, not a director
variant on the failed line. It changes the question, not the recipe: from
"can a trained local-latent DTM beat a digital model" to "can the
hardware-shaped graph hold and sample the exact posterior of a family where
the cheapest matching digital model costs 1.74e6 MACs". No training.

- *Decision it informs.* Whether the lattice, at the chip's precision and
  under chromatic Gibbs, can represent and sample a four-way global latent
  within a sweep budget. A positive gives the four-week test's three
  clauses for the first family where the digital side is costly. A negative
  of kind (a) or (b) above is a stop reason with a mechanism attached,
  which neither P-0006 nor P-0007 gave.
- *Cheapest probe that makes the study unnecessary.* Two steps, both
  exact or NumPy, no THRML and no training, under 30 CPU-minutes on `t4`:
  (1) build the model from the mixture posterior (one hidden unit per
  template; field on pixel i from template c is h_c(i); the noise
  likelihood is the pixel-clamp coupling P-0006 already uses) and compute
  its exact equilibrium pixel error by summing the transfer matrix over the
  16 hidden configurations, at float and through the M5b codebook at 4, 6
  and 8 bits; (2) if that is within 0.01 of Bayes, run NumPy chromatic Gibbs
  on the unplaced model from carried and random starts at K in {64, 256,
  1024} and record the smallest K within Bayes + 0.01. If (1) fails at 8 bits
  or (2) fails at 1024, the probe closes the row with a docs-only PR. Only
  if both pass does the study place the latent as tied copies on the
  16-offset lattice, run THRML under the layer and price starts.
- *Already in a source card?* The structure is: a mixture of MRFs with
  template fields is a dense associative memory's posterior, with templates
  as stored patterns, and the lab's own stage A found the pairwise
  emulation of that categorical latent: binary hidden units with a negative
  bias, within 0.01 of the categorical equilibrium at range 14 to 26
  (`docs/experiment-reports/2026-10-07-am-binary-emulation/findings.md`;
  `docs/knowledge/sources/krotov-hopfield-large-memory.md` C1). The same
  study recorded the two risks: one-hot inhibition needs range above 128
  and domain-wall chains do not mix, and a cue-informed start is still open
  (lessons, 2026-10-07). Geman and Geman C1 says the posterior given the
  component is an MRF, so the exact enumeration above is sound
  (`docs/knowledge/sources/geman-gibbs-restoration.md`). Placement of a
  high-degree node as tied copies has an exact precedent at 1.5x the parity
  bound (`docs/experiment-reports/2026-09-30-meta-ebm-topology/findings.md`),
  with no mixing measured. No card answers whether a replicated global latent
  switches modes under chromatic Gibbs at 4 to 8 bits in 64 to 1024 sweeps.
  That is the row's content and it is not known.
- *Already known inside the lab?* The precision side is partly known: a
  tied-copy latent at a quantized tie coupling is the conservation line's
  structure (copies that must agree, a cap that leaks), and the lab's rule
  from it is to list the fixed choice that bounds the metric before running
  (`CLAUDE.md`, "Check assumptions before building process"). The probe
  does that in step (1). Continuation's failure is known nine of nine, so
  the start rule is a priced arm inside the row, which is where D-0003 said
  it belongs.
- *Third variant?* No. The two failures were trained models with local
  latents; this row trains nothing and tests representability and mixing
  of the exact model first. It shares one risk with them (a carried state
  stuck on the wrong template), and the probe measures that risk directly
  instead of assuming it away. The owner's answer to the premise question
  above should still come before the protocol.

## Rubric 3: can the lab state the four-week test today?

For one task on one family, yes, and the statement got worse. The test
falls due on 2026-11-11 (`CLAUDE.md`, first bullet). For the P-0006
denoiser on the local L = 8 family:

1. *Precision and noise tolerance.* 4 bits under `split`, 10 percent static
   jitter negligible (P-0006 findings, "The number"). P-0007 adds the bound:
   this holds on the local family only; off it the model's own error, not
   its precision, is the limit (P-0007 findings, "The model, not only the
   point, is the limit").
2. *Sweeps, reads and writes per useful sample.* 64 sweeps per step at
   T = 2 under continuation; 128 reads, 128 writes, 0 clamp changes. P-0007
   adds: continuation is not a usable start on any family with a global
   latent, and a random start costs 384 more writes per sample.
3. *Projected energy at a stated quality against a conventional baseline.*
   Both sides are now projected joules. At error 0.075 (Bayes 0.067) the
   DTM is 2.0e-8 J per sample at the write constant and 6.1e-10 J at the
   read end; the logistic at error 0.071 is 3.1e-10, 3.1e-9 or 3.1e-8 J per
   image at 0.1, 1 or 10 pJ per MAC (P-0007 findings, "Cost, both sides
   projected"). Per 16-draw estimate the DTM is 3.2e-7 or 9.7e-9 J. The
   digital model wins at every point of the two bands except the corner of
   10 pJ per MAC with clamp loads at the read constant, and there it is
   still the more accurate. Stated plainly: the lab can state clause 3 and
   it reads against the hardware model on its only task.

The row that closes the largest gap is the one proposed: `t4` is the first
archived family where the cheapest digital model within Bayes + 0.01 costs
1.74e6 MACs, so clause 3 could read in the hardware's favour by more than
the 33x band. The arithmetic is tight, and the row should carry it: against
that net at 1 pJ per MAC (1.74e-6 J per image), the DTM's estimate costs
3.2e-7 J under continuation (5.4x cheaper) but 1.26e-6 J under a random
start (1.4x, inside any reasonable uncertainty on the MAC constant), and
at 0.1 pJ per MAC the random-start DTM loses 7x. A positive on this row at
the random-start cost is a tie, not a win. The win needs a carried state
that works, which is exactly what the probe's step (2) measures.

## The one thing to drop

The trained P-0006 denoiser (recipe, operating point and its T = 1
continuation follow-up) as the hardware side of any further denoising row.
Two studies say the same thing about it: a 3,136-MAC logistic matches it
on every family tested, including at its float K = 1024 reference
(P-0007 findings, "Stage 2"), and its recorded operating point resolves in
no seed off the local family (summary, first paragraph). It should not be
retrained on a new family, a real image set or at L = 16. Concretely, the
next row's DTM arm is the hand-built model or nothing, and its digital
side is the archived P-0007 ladder for the family, not a refit (4.63 of
P-0007's 6 CPU-hours were ladder fits; findings, "CPU seconds"). The real
binarized set stays a candidate behind this row, as the owner said; a third
ladder-wins result on a set with no exact floor could not tell model from
task apart.

## Proposed queue

Changes, each in one sentence:

- **Add** `dtm-exact-posterior-lattice` (`open`), placed after
  `dtm-baseline-headroom` as the only open row: the owner's named next
  question in queue form, probe-first, with the two stop conditions above
  written into the row so the owner can confirm or change them.
- **No other change.** Done rows stay done; the two dropped rows stay
  dropped as the owner set them in PR #117.

```queue
| `am-coupling-bits` | done | How does associative-memory recall fall with coupling precision and per-site beta noise? | Stage A's bias-binary design with 4-, 6- and 8-bit couplings from the M5b codebook and ±10% per-site beta jitter | Recall against coupling bits and beta jitter, using stage A's exact references | Produces a spec number: the coupling bits needed to keep recall within tolerance. |
| `planar-16-offset-ferro` | done | Do the planar Ising scaling allocations (PR #92) hold on the hardware-shaped graph? | Zero-field ferromagnetic Ising on the planar subgraph of the 16-offset lattice; the thin five-replica ladder; the exact Kac-Ward reference imported from `planar_ising_scaling` | Which arms qualify, and at what sweep budget, against the exact reference | Known-good regime that confirms #92 on the chip's graph. Ferro targets only; the frustrated-grid track is reserved for the owner's annealing study. |
| `constraint-layer` | done | Does one study-local constraint layer reproduce the archived placement, rounding and finite-sweep contracts bitwise, and price updates, reads, writes and clamp changes separately? | A `thermo_lab.constraints` module that places any pairwise model on the 16-offset lattice (M5c's offsets and placement, or the planar subgraph used by `planar-16-offset-ferro`), quantizes couplings and biases with M5b's `round_parameters` codebook under a declared cap (default: step = J, the `coupling` cap `am-coupling-bits` found sufficient at 6 bits), applies static per-site beta jitter, runs chromatic block Gibbs only with no replica exchange, and counts updates, node reads, full SRAM writes and clamp changes into the existing `Z1HardwareProfile` | Bitwise agreement with the M5c placement archive, the M5b rounding archive, the `am-coupling-bits` quantized recall at 6 bits and the E0 stage B inner-K law through the layer; one priced example per constraint with the relaxed-constraint flag recorded in the hashed request | A contract study, not a sampler study: short, exact, CPU. Every later row runs through the layer by default and states which constraints it relaxed (charter amendment 1, rule 8). Reuse only; no edits to hash-bound sources. Built as engineering: [plan and As built](research/2026-10-13-constraint-layer-plan.md). |
| `dtm-small-image` | done | At what task quality does a small chained-EBM denoiser run under the constraint layer, as a function of coupling bits, beta jitter, sweeps per step and reads and writes per sample, and what does it cost against CPU Gibbs on the same model and a small conventional model of the same quality? | A two- to four-step discrete denoising model (the architecture in the DTM source card) defined on the 16-offset lattice from the start (sites are lattice sites, edges are lattice offsets, two colours) so that it passes `require_placement` with nothing relaxed; on a small binary image set: 8 x 8 to 16 x 16 binarized digits or clothing, or a synthetic set with an exact likelihood; trained in software, inference under `constraint-layer` at 4, 6 and 8 bits under a cap rule the probe declares (`absolute` or `split`; `coupling` is undefined for trained weights) and 0 and 10 percent jitter; baselines inside the row | Held-out reconstruction error and log-likelihood where tractable, per projected joule and per measured CPU second, against the baselines; the bits, jitter, sweeps and reads and writes per sample (under the declared I/O path) at which task quality stays within tolerance | Opened by D-0002 now that `constraint-layer` is done. The lab's first task-level benchmark (amendment priority 2). Baselines are part of the row (rule 10), not a separate row. The probe settles these fixed choices before any protocol: (1) the I/O path per sample (per-step host readback and rewrite, or on-chip chaining), because the sealed profile prices a clamp change as a full SRAM write at about 21,665 updates and I/O was 98 to 99.996 percent of energy in every example the layer priced, so this choice, not sweeps or bits, sets the projected joule; (2) which graph the exact reference covers (a nearest-neighbour grid of width at most 24 has a transfer-matrix likelihood; the long offsets do not); (3) whether a synthetic exact-likelihood set or a real binarized set is first, synthetic first if both fit; and it sizes training cost. No tempering arms: the layer has no exchange, and independent Gibbs won on denoising posteriors (lessons, 2026-10-02). Declare everything the plan's "What a later row must declare" lists. |
| `dtm-baseline-headroom` | done | Is there a small binary denoising family at L of at most 16, with an exact posterior, on which the smallest conventional model within 0.01 of the P-0006 lattice denoiser's error is costly, and does the denoiser beat that model's projected cost at its recorded operating point under the constraint layer? | The P-0006 generator (mixture of nearest-neighbour Ising grids with template fields, exact transfer-matrix posterior) with the family varied to break locality: flip noise q of 0.3 to 0.4, more templates, long-range template differences; L = 8 first, L = 16 only if the gap appears there; a declared digital ladder (logistic on 7 x 7, 11 x 11 and full-image windows; one- and two-layer convolutional nets at a few widths; each fits in seconds); the P-0006 lattice-native model and training recipe, three seeds, under `constraint-layer` at the recorded operating point (T = 2, 4 bits under `split`, 10 percent jitter, K = 64, host-per-step I/O, continuation with the float K = 1024 reference and a random-start control) | Primary, from the probe's exact scan (no EBM): per family, the Bayes MPM floor and each ladder model's error, parameter count and MACs per image. The DTM cannot beat the floor, so a family is a candidate only if the smallest ladder model within 0.01 of the floor costs at least X or does not exist (X and its unit declared in the protocol). Study, on candidate families only: per seed, the DTM's error against Bayes and against the baseline, which is the smallest ladder model within 0.01 of the DTM's error, with that model's size and cost; a family qualifies when the baseline costs at least X. Both sides in projected joules: the DTM's reads, writes and updates per sample with the joule as the band (profile write constant and clamp loads at the read constant), and the baseline's MACs per image at 0.1, 1 and 10 pJ per MAC (`calibrated_projection`, constant stated) | Proposed by D-0003 and amended by the owner on #120: a baseline ladder instead of a fixed 7 x 7 logistic, so the row measures EBMs rather than receptive field, and the digital side priced per MAC. Decides whether the denoising task (amendment 1, priority 2) continues at this size or stops as "no advantage over a small digital model"; today that decision rests on one family whose posterior is local (P-0006: Bayes 0.067, logistic 0.071). P-0006's quality result is a positive (the DTM recovers 94 percent of the noisy-to-Bayes headroom) and its negative is cost; cite the two apart. The probe is the exact scan alone, no EBM, under 30 CPU-minutes; if no family is a candidate, the probe closes the row with a docs-only PR and the line stops at L of at most 16 on exact synthetic families. A real binarized set (no exact floor, but a global posterior: digit identity) is then the natural next test, not a line-ending result; it needs a download, so the owner opens it. Add a source card on classical MRF image denoising (Geman and Geman 1984; Besag 1986) before any protocol. Inherits the per-seed carried-state check (lessons, 2026-10-14). A second ladder-at-least-as-good result is two consecutive negatives and goes to the owner as a premise question, not a third variant. |
| `dtm-exact-posterior-lattice` | open | Can a hand-built lattice model of the exact posterior of a family with digital headroom, its template latent encoded as binary hidden units and spread over the 16-offset lattice as tied copies, reach Bayes + 0.01 under the constraint layer with no training, and at what sweeps, reads, writes and free-node writes per sample from a priced start? | P-0007's `t4_h05_q035` (four templates, J 0.35, h0 0.5, q 0.35; exact transfer-matrix posterior, Bayes 0.216; archived ladder: 7 x 7 logistic at 3,136 MACs, cheapest rung within Bayes + 0.01 is `cnn2_w32_k5` at 1.74e6 MACs), `smooth8_h05_q035` only if `t4` passes; the model built from the mixture posterior, not trained: one hidden unit per template with a negative bias (stage A's bias design) or exact one-hot, the pixel-clamp coupling from the noise likelihood as in P-0006, each hidden unit replicated into lattice copies tied ferromagnetically so every edge is a Z1 offset and degree stays at most 16 (M5c-style copies); under `constraint-layer` at 4, 6 and 8 bits `split` and 0 and 10 percent jitter; starts priced per sample: continuation, random, and a reset every N images; no training seeds, sampler seeds are the replication unit | Exact first: the placed, quantized model's own equilibrium pixel error against the Bayes MPM floor by enumeration over hidden configurations times the transfer matrix (float and 4, 6, 8 bits), which gives the tie-coupling and field precision the latent needs. Then sampled: pixel error at K in {64, 256, 1024} per step from each priced start; headline reads, writes and free-node writes per sample at the smallest K within Bayes + 0.01, joule as the band (profile write constant and clamp loads at the read constant); baseline: the archived P-0007 ladder for the family priced per MAC at 0.1, 1 and 10 pJ (`calibrated_projection`, no refit) | Proposed by D-0004 as the owner's decision of 2026-10-16 on the premise question (continue; a hand-built model of the exact posterior before any training; the real binarized set waits for its answer). Informs whether the lattice at the chip's precision can hold and sample a global latent in budget. Two stop conditions for the owner to confirm: (a) the quantized model's exact equilibrium error is above Bayes + 0.01 at 8 bits, or (b) no priced start reaches Bayes + 0.01 within K = 1024; either closes the denoising line with a mechanism named. Probe first, under 30 CPU-minutes, enumeration and NumPy Gibbs on the unplaced model only, no THRML and no training; it closes the row with a docs-only PR if (a) or (b) fails. Known risks: continuation lost to a random start in 9 of 9 P-0007 seeds, so continuation's I/O advantage (no free-node writes) is measured, not assumed, and a random start costs 384 more writes per sample (7.9e-8 J against 2.0e-8); one-hot inhibition is range-hungry and domain-wall chains do not mix (AM A, lessons 2026-10-07); a cue-informed start is the open AM variant. Inherits the per-seed carried-state check (lessons, 2026-10-14). Cost arithmetic to carry: against the 1.74e6-MAC net at 1 pJ per MAC the DTM estimate is 5.4x cheaper under continuation and 1.4x under a random start, so a positive at random-start cost is a tie, not a win. Not a third variant of the trained-DTM line: it trains nothing and tests representability and mixing of the exact model first. |
| `am-io-budget` | dropped | How many node reads, full SRAM writes and clamp changes does one recalled pattern cost in the bias associative-memory design under the constraint layer, and how does that I/O compare with its update count? | Stage A's `bias` design at 6-bit couplings with step = J (the `am-coupling-bits` spec), cue write and readout counted through `constraint-layer`, Z1 pricing | Reads, writes, clamps and updates per recalled pattern at stage A's recall tolerance; the I/O share of projected energy | Dropped by D-0002. The layer's priced example answers the question (12 reads, 32 writes, 5,120 updates per recall at P = 8; I/O 99.3 percent of modelled energy; the same arithmetic holds across stage A's cells), and the `bias` design cannot be placed (hidden degree 24, visible up to 128 against 16), so a projected joule for it prices a machine that does not exist. A placeable memory is a different design; read the diluted-memory literature and add a source card before opening a row for it. |
| `e0-remaining-kernels` | dropped | Does THRML match the archived exact inner-K law for every compiled M5a kernel, as E0 stage B showed for one? | The other 59 compiled M5a kernels (reading B) at K = 1, 2 and 4, with stage B's contract, tolerances and controls | Cells passing on output rate and joint, per kernel; control rejections | Pure replication. Stage B took about 11 CPU-minutes for one kernel, so all 59 is roughly 11 CPU-hours before parallelism; the probe sizes it, and a run over 8 CPU-hours needs the owner's say-so. Last in the order: a contract worth having, low marginal information; idle-time work. Dropped by D-0001 (owner, 2026-10-10: later); the change never reached the queue and the row was re-proposed as P-0005. |
```

## Evidence classes

Nothing in this review is new evidence. The random-start pricing (7.9e-8 J
per sample, 384 free-node writes) is read from P-0007's archived
`study.json.gz` under the same `calibrated_projection` label; the per-MAC
figures are the findings' own. The 1.4x and 5.4x ratios are arithmetic on
those numbers. The reading of the mixture posterior as an associative
memory's structure is this review's, not a source card's claim. No
hardware claim.
