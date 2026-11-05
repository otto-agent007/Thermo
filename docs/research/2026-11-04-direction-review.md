# Direction review D-0008

2026-11-04. Main commit beff3283afcbf6ba6ce3688d9d3116c079c51161. THERMES-DIRECTOR.
Label: advice, not a decision. The owner edits the queue; this file proposes.

Trigger: one loop result accepted since D-0007
(`docs/research/2026-11-02-direction-review.md`, PR #141, merged): P-0009 on
row `adsorption-precision-requirement`, PR #135, merged 2026-11-04 with its
report under
`docs/experiment-reports/2026-11-02-adsorption-precision-requirement/`.
PR #140 (adoption of D-0006) merged the same day and kept
`native-adsorption-eligibility` at `held` with the drop recorded in its
notes. Nothing is in flight. The queue at beff328 has no open row. Read
against charter amendments 1 and 2 (`PROJECT_CHARTER.md`), the agenda
(`docs/research/2026-10-20-lab-research-agenda.md`), the queue, the lessons
register and all 26 source cards.

## The change I would make first

Open one exact-only row, `rounding-rule-transfer`, before anyone re-reads an
archived bit requirement under class error diffusion or reruns the
adsorption pore with it. P-0009 read the rule on one pore whose rounding
errors were all field errors in a few large classes of equal nominal field
(findings, "Findings" 1; couplings were exact at every bit count). Amendment
2 says a general claim needs an independent case beyond the one used to
develop the explanation ("How a question earns work"). The lab already owns
the right case: P-0008's archived placed tie-limit model fails at 4 bits by
0.0752 and passes at 6 by 0.0025 under nearest rounding on `split` caps
(`docs/experiment-reports/2026-10-17-dtm-exact-posterior-lattice/findings.md`,
"Exact stages"). That is an exact point estimate on a model with large
coefficient classes of a different kind (couplings), so it is a clean
falsifier. The row is a probe-sized question (minutes, enumeration and
transfer matrix, no sampling) with one design consequence: the default
rounding rule of `thermo_lab.constraints.quantize`, which rounds to nearest
(`src/thermo_lab/constraints/quantize.py`), and under which every archived
bit requirement was read.

The thing to drop is **a powered sampled rerun of the adsorption pore** as a
candidate next row. D-0006 left that door open ("sampling adsorption again
is a new row"). The arithmetic below closes it: with the probe's own
accounting, a rerun powered to certify 0.01 reaches parity with the pore's
exact digital baseline only at the top of the queue's MAC band and never
meets the memo's 3x target. Reasons under "The one thing to drop".

## Verdicts

**P-0009 `adsorption-precision-requirement`: positive, exact, one pore; a
precision requirement with an upper bound where its headline number
should be.** Under class error diffusion on the archived family caps the
held 3×3×16 lattice gas holds loading and capacity within 0.01 and pair
occupancy within 0.02 at all three frozen chemical potentials from 4 field
bits; under nearest rounding it needs 9 (passes at 6 and 7, fails at 8 by
the archived 0.0111, passes from 9); per-point caps need 8. All four rules
also pass six fresh chemical potentials at their own b_stable (findings,
"Design numbers"; `completion.json` `recommended_rule=diffuse_class_family`,
`recommended_b_stable=4`). Three things to read with it:

1. **4 is the grid floor.** The protocol tested 4 to 12 bits, so "needs 4"
   means "at most 4" (findings, first paragraph). A designer wants the
   count, not a bound; the pore at 2 and 3 bits is a one-minute exact probe.
2. **The margin away from the frozen points is thinner than the headline.**
   On the secondary grid at 4 bits the worst loading error is 0.0071
   against 0.01 and pair 0.0089 against 0.02 (findings, "Findings" 3); the
   0.85 point exceeds the archived family cap and clips 13 to 14 wall sites
   (findings, "Findings" 4). A family cap set from three points is itself a
   design parameter: any operating point past loading 0.8 is a different
   model and a reprogramming.
3. **The result is the rule, not the pore**, as D-0006 and D-0007 predicted
   would be the useful reading. The mechanism is first-order and plain:
   nearest rounding sends all 128 wall fields the same way and reaches 91
   percent of the linear bound; diffusing the residual within a class keeps
   the class mean within step/(2 × class size) of nominal (findings,
   "Findings" 1; lessons 2026-11-02). That mechanism needs large classes of
   equal nominal coefficient. It says nothing yet about a model whose
   coefficients are distinct per site.

Is it already known? Yes, the technique is. Error diffusion is the 1976
halftoning rule, sigma-delta noise shaping is its signal-processing form,
and per-channel bias correction is standard in post-training weight
quantization. No source card covers any of them; D-0006 and D-0007 both
asked for one before the report froze, and the report merged without it
(findings cite none). What is the lab's own is the quantification on this
model: the 9-bit linear bound, nearest at 91 percent of it, and b_stable as
the design number because nearest passes at 6 and 7 and fails at 8. The
card is still owed, as is the lessons entry D-0006 asked for on the
eligibility probe's unpowered read allocation (`docs/knowledge/lessons.md`
has one adsorption entry, 2026-11-02, on rounding only).

**PR #140: process, correct**, as D-0007 said. Nothing further.

## Rubric 1: P-0009

- **Design number or probe with a study's process?** Both. It is a
  precision requirement (amendment 1 tier 2) and it is the 2026-10-23 probe
  with the frozen points unmasked plus one held-out grid: 11.7 s wall, 5.8 s
  of it replay (findings, "Protocol"), against an 862-line runner, a
  234-line test file, a 216-line protocol and the gate, studies, roadmap and
  lessons updates. That is the fixed-cost bias amendment 1 names ("Why").
  The owner selected it knowing that (row notes, 2026-10-23); it is not a
  template. The held-out discipline was the study's value and the proposer
  kept it.
- **Constraints.** Applied: codebook quantization with clipping, archived
  placement (irrelevant to an exact result). Relaxed by the row's design:
  jitter (zero) and sampling (none). Error diffusion is a host compile step
  and needs no hardware feature (protocol, "Hardware constraints"). Rule 8
  satisfied by statement.
- **Accounting.** Stated as not applicable: not a task row, no joule, CPU
  time reported and not gating (protocol, "Resource accounting"). Rule 9
  satisfied by declaration; the number costs nothing to use.
- **Expectations.** Three of six missed (findings, "Findings" 5): nearest
  b\* 6 not 9 (b_stable 9 as expected), class diffusion 4 not 5, site
  diffusion 4 not 5 or 6. All three miss in the same direction: the probe's
  six development points were harder than the frozen three, which is the
  right way round for a held-out design. No queue row assumed the opposite;
  the only other precision rows are `done` and were read under nearest,
  which is the point of the row proposed below.

## Rubric 2: the queue rows

Twelve rows after this review: ten `done` or `dropped`, unchanged since
D-0005 and covered one line each in D-0007 ("Rubric 2"); nothing depends on
re-running them. The two that matter:

### `native-adsorption-eligibility` (held)

- **Decision.** Whether sampled adsorption on this pore continues under
  this allocation. No (probe, "Research decision"; D-0006, "The change I
  would make first"). P-0009 does not unhold it (findings, "Not claimed").
- **Cheapest probe.** Ran. The exact half decided alone.
- **Source card.** `docs/knowledge/sources/schneider-local-adsorption.md`
  gives the model class; `native-adsorption-baselines.md` the digital
  comparators. Neither answers power or precision.
- **Third variant.** No. The line has one sampled negative and one exact
  positive. The owner kept the row `held` on #140; I do not re-propose
  `dropped`, because the loop's rule returns it to `held` anyway and the
  note already records the verdict. The candidate it tempts, a powered
  rerun, is the drop below.

### `adsorption-precision-requirement` (done)

- **Decision.** Narrow, answered: this pore's field bits per rule. Wide,
  open: whether class diffusion is a general lever on the layer's precision
  requirements (D-0006, "Rubric 2"). P-0009 could not answer the wide one
  by construction; its secondary grid is the same pore.
- **Cheapest probe for the wide question.** Not the am-coupling-bits
  re-read D-0006 named. That design's `split` requirement of 10 bits is an
  interval-width result at P = 128 with an 8-bit cue (mean +0.0013,
  interval −0.0152 to +0.0184 over 16 pattern sets;
  `docs/experiment-reports/2026-10-08-am-coupling-bits/findings.md`), not a
  mean shift, and a rule that cancels mean shift cannot be expected to move
  it; its visible fields are also distinct integer multiples of J on 24
  sites, so its classes are small. P-0008's placed tie-limit model is the
  case: exact, archived, 0.0752 at 4 bits, 0.0025 at 6, 0.0003 at 8 under
  `split`, with coefficient classes of a different kind (112 identical grid
  couplings, 64 identical clamp couplings, per-unit copy biases) that round
  coherently at 4 bits if the mechanism is right. That is the row below.
- **Source card.** None; owed (above).
- **Third variant.** No.

## Rubric 3: can the lab state the four-week test today?

Due 2026-11-11, seven days out. Yes, with one clause updated:

1. *Precision and noise tolerance.* 4 bits under `split` on the local
   denoising family (P-0006); 6 bits with fields in J for the bias memory
   (am-coupling-bits); 8 or 6 bits for a tied-copy latent at the tie limit,
   4 fails (P-0008); static 10 percent beta jitter negligible in four
   studies. New: at most 4 field bits under class error diffusion and 9
   under nearest rounding for a 144-site lattice gas, exact, zero jitter
   (P-0009). The only precision figure Extropic has published for a model
   on Z1 is 4-bit weights (`docs/knowledge/sources/extropic-z1t.md`, C2;
   Z1's own coupling precision is on no card). Under nearest rounding the
   pore could not be programmed at that figure; under class diffusion it
   can. That is the first lab number to land on the stack's stated
   precision, and it is why the transfer question is worth one row.
2. *Sweeps, reads and writes per useful sample.* Unchanged from D-0007:
   K = 64, 64 reads, 64 writes on the local family (P-0006); no useful
   sample on any family with a global latent (P-0007, P-0008); the
   adsorption pore priced a trajectory that certified nothing.
3. *Projected energy at stated quality against a conventional baseline.*
   Unchanged: cost-competitive above 1 pJ per MAC, 0.002 short on quality,
   no task wins on both (D-0005, "Owner decision").

Largest gap: clause 3, for any task with a quality pass. No row closes it,
and the row proposed here does not either; it is exact and prices nothing.
The honest 2026-11-11 statement is D-0007's: the amendment produced every
number it asked for, under the constraints, with accounting, and no task
the lab has run wins on cost and quality together. P-0009 adds that the
representation line produces transferable-looking rules at near-zero
compute, which is an argument for keeping that line cheap, not for
expecting it to close clause 3. The consequential owner decision is still
the one D-0007 named: continue representation requirements without an
application, or park until a task with demonstrated digital headroom and a
resident model is named. The row below is compatible with either; it is the
last cheap question in the representation line before that choice.

## Rubric 4: representation, architecture, useful computation

- **Established (exact, one model).** On the pore, the bit requirement is
  set by the sign pattern of the rounding error against site
  susceptibility, not by the step: the step bound is 9 bits for every rule,
  nearest meets it, class diffusion beats it by at least five bits
  (findings, "Findings" 1).
- **Hypothesis.** That the same holds on any model whose coefficients fall
  in large classes of equal nominal value, couplings included; and that it
  fails on models with distinct per-site coefficients. Untested; the row
  tests the first half on P-0008's archive and the probe counts class sizes
  on the AM archive for the second.
- **User and decision.** The constraint layer's author (default rule), and
  anyone reading an archived bit requirement.
- **Hardware requirement.** None. Rounding is host-side. If class diffusion
  transfers, the requirement it removes is bits, which is the one
  architecture lever the stack has already set at 4.

## The one thing to drop

**A powered sampled rerun of the adsorption pore under the new rule**, as a
candidate row. This review's arithmetic from the eligibility probe's own
accounting (`docs/research/2026-10-20-native-adsorption-eligibility-probe.md`,
"Reads, writes and conventional baseline" and "Follow-up diagnosis"),
labelled analysis:

1. The probe's allocation cost 6.4e-8 J per trajectory active-only (4.2e-9
   updates, 1.56e-8 reads, 4.42e-8 writes), six trajectories per chemical
   potential, and its ideal control could not certify 0.01: the iid
   three-group half-width was 0.0162 at 384 midpoint reads.
2. Powering it to 0.01 under the optimistic iid assumption scales reads and
   post-burn-in updates by (0.0162/0.01)² = 2.6: about 168 snapshots per
   trajectory, 9.4e-8 J per trajectory, 5.7e-7 J per chemical potential.
   Host reductions and the 1,134-slot footprint (6.2x the active energy in
   the probe's second scenario) are excluded.
3. The exact digital answer per chemical potential is 11 ms and a partial
   73,728 MACs: 7.4e-9, 7.4e-8 and 7.4e-7 J at 0.1, 1 and 10 pJ per MAC,
   with the omitted operations uncounted.
4. The array is therefore 77x, 7.7x and 0.8x the digital projection across
   the band, for a 0.01-certified estimate against an exact one. The memo's
   own success target, 3x lower energy at matched accuracy
   (`docs/research/2026-10-20-next-task-recommendation.md`, "Go/no-go
   rules"), is met nowhere in the band. The partial MAC count understates
   the digital side and the excluded host and footprint costs understate
   the array side; neither correction reaches 3x at 10 pJ.

P-0009 removed the precision objection to this pore. It did not touch the
cost objection, which was decided by the exact baseline before any sampling
and which the agenda's rule (identify cost headroom before optimizing a
sampler) treats as the screen. A larger pore or a validated material
workload is a different question with its own selection (amendment 2,
"Commitments"); this review does not propose one.

## Proposed queue

One change: add `rounding-rule-transfer`, open, at the end. No other row is
touched; `native-adsorption-eligibility` stays `held` as the owner left it
on #140.

Decision the new row informs: whether `thermo_lab.constraints.quantize`
keeps nearest rounding as its default, and whether the archived bit
requirements (am-coupling-bits 6/10/12 by cap, P-0006 4, P-0008 8/6) are
re-read under class diffusion. Falsifier: P-0008's placed tie-limit model
still misses Bayes + 0.01 at 4 bits under class diffusion, which scopes
P-0009 to field-dominated models with few classes. It is not a denoising
study and cannot reopen that line: it re-reads an archived exact
computation, samples, trains and prices nothing, and P-0008's stop fired on
finite tie strength at 8 bits, where precision was not binding (lessons
2026-10-17). Not a task row, so the standing per-sample I/O and MAC rules
do not apply. A source card on error diffusion and quantization bias
correction is owed before its protocol.

```queue
| `native-adsorption-eligibility` | held | Does the directly represented finite-pore model retain loading and working-capacity accuracy at 8 bits and ±10% static beta variation within one fixed read/update allocation? | One open 3×3×16 grand-canonical lattice gas with local pair attraction and the literal positive lateral-wall field; no global latent or copied variable | Loading and capacity error at most 0.01, pair-occupation error at most 0.02, with uncertainty and start-agreement checks; reads, writes and baseline cost at qualifying accuracy | Owner selected the adsorption recommendation on 2026-10-20. The bounded exploratory probe found exact 8-bit middle-loading error 0.011078, above 0.01; sampled intervals remain unresolved, including the ideal control. Hold this allocation: no recorded study, extra budget or cap tuning. See docs/research/2026-10-20-native-adsorption-eligibility-probe.md and the three-family recommendation. No loop proposal ID assigned. Dropped by D-0006: the exact 8-bit miss and the unpowered read allocation mean no outcome can unhold this allocation; sampling adsorption again is a new row. |
| `am-coupling-bits` | done | How does associative-memory recall fall with coupling precision and per-site beta noise? | Stage A's bias-binary design with 4-, 6- and 8-bit couplings from the M5b codebook and ±10% per-site beta jitter | Recall against coupling bits and beta jitter, using stage A's exact references | Produces a spec number: the coupling bits needed to keep recall within tolerance. |
| `planar-16-offset-ferro` | done | Do the planar Ising scaling allocations (PR #92) hold on the hardware-shaped graph? | Zero-field ferromagnetic Ising on the planar subgraph of the 16-offset lattice; the thin five-replica ladder; the exact Kac-Ward reference imported from `planar_ising_scaling` | Which arms qualify, and at what sweep budget, against the exact reference | Known-good regime that confirms #92 on the chip's graph. Ferro targets only; the frustrated-grid track is reserved for the owner's annealing study. |
| `constraint-layer` | done | Does one study-local constraint layer reproduce the archived placement, rounding and finite-sweep contracts bitwise, and price updates, reads, writes and clamp changes separately? | A `thermo_lab.constraints` module that places any pairwise model on the 16-offset lattice (M5c's offsets and placement, or the planar subgraph used by `planar-16-offset-ferro`), quantizes couplings and biases with M5b's `round_parameters` codebook under a declared cap (default: step = J, the `coupling` cap `am-coupling-bits` found sufficient at 6 bits), applies static per-site beta jitter, runs chromatic block Gibbs only with no replica exchange, and counts updates, node reads, full SRAM writes and clamp changes into the existing `Z1HardwareProfile` | Bitwise agreement with the M5c placement archive, the M5b rounding archive, the `am-coupling-bits` quantized recall at 6 bits and the E0 stage B inner-K law through the layer; one priced example per constraint with the relaxed-constraint flag recorded in the hashed request | A contract study, not a sampler study: short, exact, CPU. Every later row runs through the layer by default and states which constraints it relaxed (charter amendment 1, rule 8). Reuse only; no edits to hash-bound sources. Built as engineering: [plan and As built](research/2026-10-13-constraint-layer-plan.md). |
| `dtm-small-image` | done | At what task quality does a small chained-EBM denoiser run under the constraint layer, as a function of coupling bits, beta jitter, sweeps per step and reads and writes per sample, and what does it cost against CPU Gibbs on the same model and a small conventional model of the same quality? | A two- to four-step discrete denoising model (the architecture in the DTM source card) defined on the 16-offset lattice from the start (sites are lattice sites, edges are lattice offsets, two colours) so that it passes `require_placement` with nothing relaxed; on a small binary image set: 8 x 8 to 16 x 16 binarized digits or clothing, or a synthetic set with an exact likelihood; trained in software, inference under `constraint-layer` at 4, 6 and 8 bits under a cap rule the probe declares (`absolute` or `split`; `coupling` is undefined for trained weights) and 0 and 10 percent jitter; baselines inside the row | Held-out reconstruction error and log-likelihood where tractable, per projected joule and per measured CPU second, against the baselines; the bits, jitter, sweeps and reads and writes per sample (under the declared I/O path) at which task quality stays within tolerance | Opened by D-0002 now that `constraint-layer` is done. The lab's first task-level benchmark (amendment priority 2). Baselines are part of the row (rule 10), not a separate row. The probe settles these fixed choices before any protocol: (1) the I/O path per sample (per-step host readback and rewrite, or on-chip chaining), because the sealed profile prices a clamp change as a full SRAM write at about 21,665 updates and I/O was 98 to 99.996 percent of energy in every example the layer priced, so this choice, not sweeps or bits, sets the projected joule; (2) which graph the exact reference covers (a nearest-neighbour grid of width at most 24 has a transfer-matrix likelihood; the long offsets do not); (3) whether a synthetic exact-likelihood set or a real binarized set is first, synthetic first if both fit; and it sizes training cost. No tempering arms: the layer has no exchange, and independent Gibbs won on denoising posteriors (lessons, 2026-10-02). Declare everything the plan's "What a later row must declare" lists. |
| `dtm-baseline-headroom` | done | Is there a small binary denoising family at L of at most 16, with an exact posterior, on which the smallest conventional model within 0.01 of the P-0006 lattice denoiser's error is costly, and does the denoiser beat that model's projected cost at its recorded operating point under the constraint layer? | The P-0006 generator (mixture of nearest-neighbour Ising grids with template fields, exact transfer-matrix posterior) with the family varied to break locality: flip noise q of 0.3 to 0.4, more templates, long-range template differences; L = 8 first, L = 16 only if the gap appears there; a declared digital ladder (logistic on 7 x 7, 11 x 11 and full-image windows; one- and two-layer convolutional nets at a few widths; each fits in seconds); the P-0006 lattice-native model and training recipe, three seeds, under `constraint-layer` at the recorded operating point (T = 2, 4 bits under `split`, 10 percent jitter, K = 64, host-per-step I/O, continuation with the float K = 1024 reference and a random-start control) | Primary, from the probe's exact scan (no EBM): per family, the Bayes MPM floor and each ladder model's error, parameter count and MACs per image. The DTM cannot beat the floor, so a family is a candidate only if the smallest ladder model within 0.01 of the floor costs at least X or does not exist (X and its unit declared in the protocol). Study, on candidate families only: per seed, the DTM's error against Bayes and against the baseline, which is the smallest ladder model within 0.01 of the DTM's error, with that model's size and cost; a family qualifies when the baseline costs at least X. Both sides in projected joules: the DTM's reads, writes and updates per sample with the joule as the band (profile write constant and clamp loads at the read constant), and the baseline's MACs per image at 0.1, 1 and 10 pJ per MAC (`calibrated_projection`, constant stated) | Proposed by D-0003 and amended by the owner on #120: a baseline ladder instead of a fixed 7 x 7 logistic, so the row measures EBMs rather than receptive field, and the digital side priced per MAC. Decides whether the denoising task (amendment 1, priority 2) continues at this size or stops as "no advantage over a small digital model"; today that decision rests on one family whose posterior is local (P-0006: Bayes 0.067, logistic 0.071). P-0006's quality result is a positive (the DTM recovers 94 percent of the noisy-to-Bayes headroom) and its negative is cost; cite the two apart. The probe is the exact scan alone, no EBM, under 30 CPU-minutes; if no family is a candidate, the probe closes the row with a docs-only PR and the line stops at L of at most 16 on exact synthetic families. A real binarized set (no exact floor, but a global posterior: digit identity) is then the natural next test, not a line-ending result; it needs a download, so the owner opens it. Add a source card on classical MRF image denoising (Geman and Geman 1984; Besag 1986) before any protocol. Inherits the per-seed carried-state check (lessons, 2026-10-14). A second ladder-at-least-as-good result is two consecutive negatives and goes to the owner as a premise question, not a third variant. |
| `dtm-global-latent` | dropped | Can a model placed on the 16-offset lattice carry a global latent well enough to come within 0.01 of Bayes on P-0007's candidate families under the constraint layer, and if not, which constraint stops it? | P-0007's candidate families at L = 8 with their exact transfer-matrix posteriors: `smooth8_h05_q035` and `smooth16_h05_q035` first (the ladder's first rung within Bayes + 0.01 costs 3.4e6 MACs or more), `t4_h05_q035` second (its margin over X is 0.0005). The model is hand-built, not trained: the generator's own nearest-neighbour couplings, template fields and clamp, with the template identity as a latent whose reach to all 64 pixels is spread over the lattice (tied copies, a coarse layer of block latents, or both; one-hot with inhibition or stage A's bias-binary encoding; the probe chooses), placed with `require_placement`. Operating arms run under `constraint-layer` with nothing relaxed: 4, 6 and 8 bits under a declared cap rule, 10 percent jitter, host-per-step I/O, with sweeps per sample and copy coupling varied. Reference arms state what they relax: float, long chains for equilibrium, and placement relaxed to one hub node of declared degree | Per family and arm, the pixel error of P-0007's 16-draw majority estimate against Bayes (exact), with reads, writes, clamp changes and updates per sample. The design number is the smallest sweeps, bits and copy coupling at which the hand-built model is within 0.01 of Bayes under the constraints, or, if placement must be relaxed, the hub degree it needs. Joules as a band per the standing rules, against the P-0007 ladder's MACs at 0.1, 1 and 10 pJ per MAC | Dropped 2026-10-16 before any proposal: superseded by `dtm-exact-posterior-lattice` (D-0004, adopted by the owner with amendments). Owner, 2026-10-16, on P-0007's premise question: continue. P-0007's float K = 1024 reference was as poor as its operating point, so the model class (local latents only) bounds the metric; this row tests that fixed choice before any training (lessons, 2026-09-24). The exact posterior is one categorical latent coupled to every pixel, plus nearest-neighbour couplings and the clamp, so only the latent's reach exceeds degree 16. Known risks from the AM track: the bias design needed degree 24 and up, one-hot inhibition is range-hungry, and domain-wall chains did not mix. Each outcome names a number. If the model is within 0.01 under the constraints, a training row follows and prices break-even pJ per MAC. If it gets there only at equilibrium, the number is sweeps against copy coupling. If it needs a hub, the number is the hub degree, a hardware feature. If it never gets there, global-latent tasks have a structural stop on this array, and the denoising line ends at L of at most 16 on exact families. No training in this row. The probe stays under 30 CPU-minutes; calibrate runtime on full-length runs, not short ones (P-0007 underestimated by 2.7x). A real binarized set waits for this answer; it needs a download the owner opens. |
| `dtm-exact-posterior-lattice` | done | Can a hand-built lattice model of the exact posterior of a family with digital headroom, its template latent encoded as binary hidden units and spread over the 16-offset lattice as tied copies, reach Bayes + 0.01 under the constraint layer with no training, and at what sweeps, reads, writes and free-node writes per sample from a priced start? | P-0007's `t4_h05_q035` (four templates, J 0.35, h0 0.5, q 0.35; exact transfer-matrix posterior, Bayes 0.216; archived ladder: 7 x 7 logistic at 3,136 MACs, cheapest rung within Bayes + 0.01 is `cnn2_w32_k5` at 1.74e6 MACs), `smooth8_h05_q035` whenever `t4` passes (a) and (b), because the cost verdict comes from it; the model built from the mixture posterior, not trained: one hidden unit per template with a negative bias (stage A's bias design) or exact one-hot, the pixel-clamp coupling from the noise likelihood as in P-0006, each hidden unit replicated into lattice copies tied ferromagnetically so every edge is a Z1 offset and degree stays at most 16 (M5c-style copies); under `constraint-layer` at 4, 6 and 8 bits `split` and 0 and 10 percent jitter; starts priced per sample: continuation, random, and a reset every N images; no training seeds, sampler seeds are the replication unit | Exact first: the placed, quantized model's own equilibrium pixel error against the Bayes MPM floor by enumeration over hidden configurations times the transfer matrix (float and 4, 6, 8 bits), which gives the tie-coupling and field precision the latent needs. Then sampled: pixel error at K in {64, 256, 1024} per step from each priced start; headline reads, writes and free-node writes per sample at the smallest K within Bayes + 0.01, joule as the band (profile write constant and clamp loads at the read constant); baseline: the archived P-0007 ladder for the family priced per MAC at 0.1, 1 and 10 pJ (`calibrated_projection`, no refit) | Proposed by D-0004 as the owner's decision of 2026-10-16 on the premise question (continue; a hand-built model of the exact posterior before any training) and amended by the owner on #124: stops (a) and (b) confirmed, cost stop (c) added, cost verdict from `smooth8`. Informs whether the lattice at the chip's precision can hold and sample a global latent in budget. Stop conditions (owner, 2026-10-16): (a) the quantized model's exact equilibrium error is above Bayes + 0.01 at 8 bits; (b) no priced start reaches Bayes + 0.01 within K = 1024; (c) on `smooth8_h05_q035`, the cheapest priced start that reaches Bayes + 0.01 within K = 1024 costs, per 16-draw estimate at the profile's write constant, at least as much as the archived ladder's cheapest rung within Bayes + 0.01 (`cnn2_w32_k5_gpool`, 3.43e6 MACs) at 1 pJ per MAC. (a) and (b) are judged on `t4` and then on `smooth8`; (c) on `smooth8`. Any one closes the denoising line at L of at most 16 with its mechanism named, and the real binarized set opens only on a positive. Calibrate runtime on full-length runs, not short compile-dominated ones (P-0007 underestimated by 2.7x). Probe first, under 30 CPU-minutes, enumeration and NumPy Gibbs on the unplaced model only, no THRML and no training; it closes the row with a docs-only PR if (a) or (b) fails. Known risks: continuation lost to a random start in 9 of 9 P-0007 seeds, so continuation's I/O advantage (no free-node writes) is measured, not assumed, and a random start costs 384 more writes per sample (7.9e-8 J against 2.0e-8); one-hot inhibition is range-hungry and domain-wall chains do not mix (AM A, lessons 2026-10-07); a cue-informed start is the open AM variant. Inherits the per-seed carried-state check (lessons, 2026-10-14). Cost arithmetic to carry: `t4` tests representability and mixing, not cost. Its 243,200-MAC net misses Bayes + 0.01 by only 0.0005, so its real digital cost is probably near 2.4e5 MACs (2.4e-7 J at 1 pJ per MAC), where even P-0007's continuation estimate (3.2e-7 J at the write constant) does not win. The cost verdict comes from `smooth8`, whose 243,200-MAC net misses by 0.0078: against 3.43e6 MACs at 1 pJ per MAC, P-0007's estimate would be 10.7x cheaper under continuation and 2.7x under a random start. The hand-built model's own reads, writes and free-node writes replace P-0007's in that arithmetic, and a pass on (c) is against the archived rung, not the digital minimum (the ladder has nothing between 243,200 and 1.74e6 MACs); say so. Not a third variant of the trained-DTM line: it trains nothing and tests representability and mixing of the exact model first. |
| `am-io-budget` | dropped | How many node reads, full SRAM writes and clamp changes does one recalled pattern cost in the bias associative-memory design under the constraint layer, and how does that I/O compare with its update count? | Stage A's `bias` design at 6-bit couplings with step = J (the `am-coupling-bits` spec), cue write and readout counted through `constraint-layer`, Z1 pricing | Reads, writes, clamps and updates per recalled pattern at stage A's recall tolerance; the I/O share of projected energy | Dropped by D-0002. The layer's priced example answers the question (12 reads, 32 writes, 5,120 updates per recall at P = 8; I/O 99.3 percent of modelled energy; the same arithmetic holds across stage A's cells), and the `bias` design cannot be placed (hidden degree 24, visible up to 128 against 16), so a projected joule for it prices a machine that does not exist. A placeable memory is a different design; read the diluted-memory literature and add a source card before opening a row for it. |
| `e0-remaining-kernels` | dropped | Does THRML match the archived exact inner-K law for every compiled M5a kernel, as E0 stage B showed for one? | The other 59 compiled M5a kernels (reading B) at K = 1, 2 and 4, with stage B's contract, tolerances and controls | Cells passing on output rate and joint, per kernel; control rejections | Pure replication. Stage B took about 11 CPU-minutes for one kernel, so all 59 is roughly 11 CPU-hours before parallelism; the probe sizes it, and a run over 8 CPU-hours needs the owner's say-so. Last in the order: a contract worth having, low marginal information; idle-time work. Dropped by D-0001 (owner, 2026-10-10: later); the change never reached the queue and the row was re-proposed as P-0005. |
| `adsorption-precision-requirement` | done | How many coefficient bits, and which rounding rule, keep the exact equilibrium observables of the held adsorption model within tolerance at every frozen chemical potential? | The frozen `native-adsorption-eligibility` model only: open 3×3×16 lattice gas, beta = epsilon = 1, literal wall field phi = 3, its three frozen chemical potentials (target loadings 0.2, 0.5, 0.8) and its archived spin map and placement; bits 4 to 12; rounding rules declared in the protocol before any result: nearest on the probe's family-wide split caps (the archived rule, reproduced first as a control), a balanced or error-diffused rule on the same caps, and per-point caps; no jitter, no sampling | Exact (`exact_reference`, transfer matrix) loading error per chemical potential, working-capacity error and aggregate pair-occupation error against full precision; the design number is the smallest bit count, per rule, at which loading and capacity errors are at most 0.01 and pair error at most 0.02 at all three points, or none within 12 bits | Owner selected 2026-10-23 (option 2 of THERMES's next-direction note). A representation requirement, not an application or advantage claim. Exact only: no sampling, no jitter and no cost comparison. The finite-budget sampling question and the held row stay as they are. Report non-monotone error in bits (the probe's 6-bit arm beat 8-bit at the middle point) as found, without smoothing. A rule chosen after seeing errors on these three points is development, so the protocol must say how the rule is fixed before results, and must check any rule that passes on chemical potentials held out from its choice. P-0008's 4/6/8-bit tie-limit gaps (0.0752, 0.0025, 0.0003) are cited for comparison, not rerun; its miss was finite tie strength, not precision. A pass does not unhold `native-adsorption-eligibility` and does not authorize sampling: the owner decides. Probe under 30 CPU-minutes; the transfer reference is 512 states per slice, so the full grid should fit in it. |
| `rounding-rule-transfer` | open | Does class error diffusion move an archived exact bit requirement on a model it was not developed on, and does its mechanism (coherent rounding across a large class of equal nominal coefficients) predict where it helps? | P-0008's archived placed tie-limit model of `t4_h05_q035`'s posterior (bias-design template units as tied copies on the carried layout at w*, the 1,000 archived test images, exact enumeration over hidden configurations times the transfer matrix) with its archived `split` caps; bits 4, 5, 6 and 8; two rules fixed before any result: nearest (the archived rule, reproduced first as a control at 4, 6 and 8 bits) and P-0009's `diffuse_class_family` extended to every coefficient class, couplings included, with the class rule stated in the protocol; no sampling, no jitter, no training, no new layout | Exact (`exact_reference`) MPM pixel error against the Bayes floor per rule and bit count, with class sizes and the coherent rounding pattern at 4 bits reported; the design number is the smallest bit count at which the placed tie-limit model is within Bayes + 0.01 under class diffusion, against the archived 6 under nearest (4 bits fails at 0.0752), or no change | Proposed by D-0008 as the independent case amendment 2 requires before class diffusion is called a general lever: P-0009 read the rule on one pore whose rounding errors were all field errors in a few large classes. Informs one decision: whether `thermo_lab.constraints.quantize` keeps nearest rounding as its default and whether the archived bit requirements (am-coupling-bits 6/10/12 by cap, P-0006 4, P-0008 8/6) are re-read under the new rule. Pre-registered prediction from the mechanism: the 4-bit miss comes from coefficients that round coherently in large classes (the probe reports which), so class diffusion should pass at 4 or 5 bits; the falsifier is a 4-bit gap still above 0.01, which scopes P-0009 to field-dominated models with few classes. Replaces the am-coupling-bits transfer check D-0006 named, whose `split` requirement is an interval-width result that a mean-cancelling rule cannot be expected to move; the probe counts that archive's class sizes instead. Not a denoising study and cannot reopen that line: it re-reads an archived exact computation, samples, trains and prices nothing, and P-0008's stop fired on finite tie strength at 8 bits where precision was not binding. Not a task row, so the per-sample I/O and MAC rules do not apply. Probe under 30 CPU-minutes, exact only, hash-checked imports from the P-0008 archive, no edits to hash-bound sources; if it answers the question it closes the row with a docs-only PR. A source card on error diffusion and quantization bias correction is owed before any protocol. The proposer may add the adsorption pore at 2 and 3 bits as exploration, because P-0009's 4 is its grid floor. |
```

## Evidence classes

Nothing in this review is new evidence. P-0009's numbers are
`exact_reference` from a recorded study cited by path; P-0008's and
am-coupling-bits' from recorded studies cited by path; the adsorption
allocation numbers from an exploration note. The powered-rerun arithmetic
is this review's calculation from the sealed profile's published constants
and the probe's counts, labelled analysis; it is a break-even requirement,
not a measurement. The mechanism prediction for P-0008's model is this
review's reading of its protocol's coefficient structure, to be checked by
the row's probe. No hardware claim.
