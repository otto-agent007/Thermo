# Direction review D-0006

2026-10-26. Main commit e341b9116f1b5a5665a7bc70f0fe7552a908303f. THERMES-DIRECTOR.
Label: advice, not a decision. The owner edits the queue; this file proposes.

Weekly floor. No loop result was accepted since D-0005
(`docs/research/2026-10-18-direction-review.md`, PR #128). In flight and
weighed, not touched: P-0009 on row `adsorption-precision-requirement`
(PR #135, draft, awaiting the owner). Read against charter amendments 1 and
2 (`PROJECT_CHARTER.md`), the agenda
(`docs/research/2026-10-20-lab-research-agenda.md`) and the five notes
under `docs/research/` dated after 2026-10-18.

## The change I would make first

Move `native-adsorption-eligibility` from `held` to `dropped`. No result
can unhold it. Its 8-bit nearest-rounding arm misses the loading tolerance
exactly, by construction, at the middle chemical potential (0.011078 against
0.01; `docs/research/2026-10-20-native-adsorption-eligibility-probe.md`,
"Exact representation result"), and its sampled allocation cannot certify
0.01 even with independent draws: the exact variance gives a per-trajectory
loading standard error of 0.0092 at 64 reads and a three-group t half-width
of about 0.016 (same note, "Follow-up diagnosis"). The row's own text forbids
extra budget or cap tuning, and the open row says a pass "does not unhold"
it. A held row that no outcome can open is a dropped row; its notes stay.
If the owner wants adsorption sampled again, that is a new row with a
powered read budget and whichever rounding rule P-0009 recommends, and it
needs its own selection (amendment 2, "Commitments"). Nothing else changes:
no new row, and P-0009 proceeds at the owner's pace.

## Verdicts

No accepted result. Verdicts on what moved since D-0005:

**Adsorption eligibility probe (2026-10-20, exploration): negative on both
halves, and the sampled half was unpowered before it ran.** Exact 8-bit
loading error 0.011078 at the middle point (tolerance 0.01); the ideal
control's intervals fail too (largest loading bound 0.039). The second audit
found the cause of the exact miss is coherent rounding: all 128 wall-site
fields round upward, and linear response to that pattern predicts 99.68
percent of the shift. The sampled miss is statistical, not mixing: every
ideal point estimate is inside tolerance, and the design could not have
certified 0.01 at 64 reads times three groups (probe, "Follow-up
diagnosis"). The probe is honest about both. What it does not say in its
headline: the allocation failed a power calculation that the exact variance
(Var(N) = 112.3) made available before any sampling. That is the lesson
2026-10-06 (Potts stage B design: check the noise floor before fixing a
budget grid) applied again, and it has no row in
`docs/knowledge/lessons.md` yet. The note exists, so a lessons entry is
owed.

**P-0009 probe (2026-10-23, exploration, in PR #135): a positive on
development points, correctly held out from the frozen ones.** Nearest
rounding on the family cap first passes at 9 bits; error diffusion within
field classes passes at 5; nearest on per-point caps at 7 (probe,
"Development result"). The linear-response bound says any rule with per-site
error at most step/2 passes at 9, so the rule, not the step, sets the
budget. The proposer chose class diffusion after seeing development errors
and says so; the frozen points and a disjoint six-point grid test it. This
is the lab's first representation result whose mechanism (balance the
rounding residual within sites that share a nominal field) plausibly
transfers beyond its model. It is not yet a result.

**Thinned-draw probe (2026-10-18, exploration, PR #129): recorded under
D-0005's owner decision.** One clamp load serves 64 draws; about 1.05e-7 J
per estimate at write pricing; gap 0.0122, still 0.002 outside tolerance.
Cost moved, quality did not. Nothing new since D-0005.

**Agenda and amendment 2 (2026-10-20): process, no evidence.** They broaden
the primary question, keep the constraint layer, the evidence tiers and the
2026-11-11 test, and add no row. The loop-skills note (2026-10-23) is
documentation only.

## Rubric 1: the last three recorded studies, briefly

All three are in the closed denoising line and were covered in full by
D-0003 to D-0005. One line each, with what still holds:

- **P-0006 `dtm-small-image`** (`docs/experiment-reports/2026-10-14-dtm-small-image/findings.md`):
  design numbers yes (4 bits under `split`, K = 64, 64 reads and 64 writes
  per sample, 2.0e-8 J at the write constant, jitter negligible); nothing
  relaxed on the operating arm; accounting stated; quality positive, cost
  negative against a 114-parameter logistic (lessons 2026-10-14).
- **P-0007 `dtm-baseline-headroom`** (`.../2026-10-16-dtm-baseline-headroom/findings.md`):
  a cost number on both sides (DTM 2.0e-8 J against 3.1e-10 to 3.1e-8 J per
  image for the logistic across the MAC band); exact scan found headroom
  for the digital ladder but the lattice denoiser did not take it;
  continuation lost to a random start in 9 of 9 seeds; the 2.7x runtime
  miss is in lessons 2026-10-16.
- **P-0008 `dtm-exact-posterior-lattice`** (`.../2026-10-17-dtm-exact-posterior-lattice/findings.md`):
  one exact design number (tied-copy latent holds at 8 and 6 bits, fails at
  4 at the tie limit); stop (b) fired, mechanism finite tie strength; two of
  six expectations missed; no operating point, so no cost at quality.

No open row assumed the opposite of any of these; the queue at e341b91 has
one open row and it is outside denoising.

## Rubric 2: the queue rows

The queue at e341b91 has eleven rows: nine `done` or `dropped` and unchanged
since D-0005 (nothing depends on re-running them; D-0005, "Rubric 2";
`am-coupling-bits` remains the lab's cleanest spec number, 6 bits with
fields in units of J), one `held`, one `open`.

### `native-adsorption-eligibility` (held)

- **Decision.** Whether adsorption on this pore continues as a task. The
  probe answered it: not at this allocation (probe, "Research decision").
- **Cheapest probe.** It ran (10 CPU-seconds). The exact half could have been
  decided alone, and the sampled half should have been sized on the exact
  variance first.
- **Source card.** `docs/knowledge/sources/schneider-local-adsorption.md`
  gives the model class and its repulsive-wall sign; no card answers the
  precision or power question.
- **Third variant.** No. One failure in this line.
- **Verdict.** Drop (above). The status change is the whole proposal.

### `adsorption-precision-requirement` (open; P-0009 awaiting owner)

- **Decision.** Two. Narrow: the field bit count this pore needs per rounding
  rule (a precision requirement, tier 2 of amendment 1; amendment 2 priority
  3). Wide, and the one worth the process: whether a compile-time rounding
  rule is a general lever on the constraint layer's precision requirements.
  `thermo_lab.constraints.quantize` rounds to nearest, and every recorded
  precision number the lab owns was read under that rule (am-coupling-bits
  6/10/12 bits by cap, lessons 2026-10-08; P-0006 4 bits; P-0008 8/6 bits,
  4 fails). If class-balanced rounding moves those numbers, the layer's
  default and every future bit requirement change. If it only moves this
  pore, the row's number designs nothing anyone will build.
- **Cheapest probe.** Already run; the remaining 108 plus 216 exact cells
  cost under a CPU-minute (PR #135 body). The recorded study is a replay of
  the probe with the frozen points unmasked and a second held-out grid. Its
  entire value is the held-out discipline, which the proposer kept. Say
  plainly: this is the smallest recorded study the lab has proposed, and its
  fixed cost (protocol, runner, replay, gate, four documents, archive) is
  the thing amendment 1 warns about. The owner selected it knowing that
  (row notes, 2026-10-23); I would not reverse that, but I would hold the
  report to the probe's length.
- **Source card.** None answers it, and none covers it. Error diffusion
  (dithering) and per-class quantization bias correction are standard in
  signal processing and in neural-network weight quantization; the lab has
  no card for either, so the protocol names a rule it reinvented. Before the
  protocol freezes, one source card on quantization bias correction or
  stochastic rounding would let the report say what is known and what is
  this model's. That is advice to the owner, not a protocol revision.
- **Third variant.** No. It asks a different question (exact representation)
  from the eligibility probe (sampled eligibility), and the line has one
  failure.
- **Two cautions for the owner's reading of PR #135, from the probe's own
  numbers.** (1) The per-point-cap rule assumes the host reprograms the
  field scale per chemical potential, which the protocol states; that is a
  write per point and the row prices nothing, so its number is weaker than
  the family-cap numbers. (2) Expectation 2 is already public (the control's
  8-bit miss is archived), so the study's only unknown outcomes are rules 2
  to 4 on the frozen points and the held-out grid. If class diffusion
  passes at 5 bits on both, the result is the rule, not the pore.

## Rubric 3: can the lab state the four-week test today?

Due 2026-11-11. Yes, for one task, and the answer is the one the owner
recorded on 2026-10-18 (D-0005, "Owner decision"):

1. *Precision and noise tolerance.* 4 bits under `split` on the local
   denoising family (P-0006); 6 bits with fields in J for the bias memory
   (am-coupling-bits); 8 or 6 bits for a tied-copy latent at the tie limit
   (P-0008); static 10 percent beta jitter negligible in four studies. If
   P-0009 is accepted and its development numbers hold on the frozen
   points, add: 9 bits under nearest rounding and 5 under class-balanced
   rounding for a 144-site lattice gas, exact, no jitter.
2. *Sweeps, reads and writes per useful sample.* K = 64, 64 reads, 64
   writes, 0 clamp changes on the local family (P-0006). On every family
   with a global latent there is no useful sample (P-0007, P-0008). The
   adsorption probe adds a cost per trajectory (9,216 reads, 288 writes,
   589,824 updates; 6.4e-8 J active-only) but no useful sample at
   tolerance, because the allocation could not certify one.
3. *Projected energy at stated quality against a conventional baseline.*
   Owner's wording stands: cost-competitive above 1 pJ/MAC, 0.002 short on
   quality, no task wins on both. The adsorption pore does not change this:
   its exact digital baseline is 11 ms and a partial 73,728 MACs per point
   (7.4e-8 J at 1 pJ/MAC) for an exact answer, against 1.15e-6 J for an
   18-trajectory array evaluation that resolved nothing (probe, "Reads,
   writes and conventional baseline").

Largest gap: clause 3 for any task with a quality pass. No row closes it.
The open row excludes sampling and cost by design, so it cannot. The honest
statement on 2026-11-11 is unchanged from D-0005: the amendment produced the
numbers it asked for, and no task the lab has run wins on both cost and
quality.

## Rubric 4: representation, architecture, useful computation

- **Established (exploration, exact).** On a directly represented local
  model, 8-bit nearest rounding of a three-valued field pattern shifts a
  global observable by 1.1 percentage points through coherent, same-sign site errors
  amplified by susceptibility (adsorption probe audit). The small-parameter-
  error-implies-small-observable-error assumption is false at this
  precision. Intended model: the full-precision lattice gas. Constrained
  representation: the rounded model, exact. Update process and estimator:
  not separated by this result, because the sampled half was unpowered.
- **Challenged.** The eligibility memo's premise that a native local
  representation makes precision a non-issue
  (`docs/research/2026-10-20-next-task-recommendation.md`, "Why adsorption
  ranks first"). It did not; the sign structure of the rounding did.
- **Unsupported generalization to avoid.** "5 bits suffices with error
  diffusion" is a claim about one pore with three field classes, read on six
  development points. It becomes a claim about the rule only after the
  frozen and held-out grids (P-0009) and then a second, unrelated archived
  model.
- **User and decision.** The constraint layer's author, choosing the
  default rounding rule; secondarily, anyone reading an old bit requirement.
  Strongest digital baseline for the pore: the exact transfer matrix; no
  array result on this pore can beat an exact 11 ms answer on quality, so
  the pore is a representation vehicle only.
- **Hardware requirement.** None follows. Rounding is a host compile step
  (PR #135 protocol, "Outside the layer"); it relaxes nothing and costs no
  feature.
- **Next question, if P-0009 passes.** Does class-balanced rounding change
  an archived bit requirement on a model it was not developed on? Cheapest
  decisive check: a probe, minutes, exact only, re-reading
  `am-coupling-bits`'s archived exact recall at 4 to 12 bits under `split`
  caps with the new rule (that design needed 10 bits under nearest with
  split caps, lessons 2026-10-08). Falsifier: the split-cap requirement
  stays at 10. A pass moves the layer's default and is a general lever; a
  fail scopes P-0009 to models with few field classes. This is not a row
  yet and should not be one until P-0009 is accepted; amendment 2 says
  general claims need independent cases, and this is the independent case.
  It is not a denoising rerun and does not touch the closed line.

## The one thing to drop

`native-adsorption-eligibility`: `held` to `dropped`. Reasons in "The
change I would make first". It is the only thing in the queue whose status
promises a future that the arithmetic has removed.

## Proposed queue

One change: `native-adsorption-eligibility` moves from `held` to `dropped`,
with one sentence added to its notes naming this review. No row added, no
reorder, no other row touched. Decision the change informs: whether the
adsorption line needs a new, powered row (yes, if the owner wants it
sampled) or can be revisited through the held row (no).

```queue
| `native-adsorption-eligibility` | dropped | Does the directly represented finite-pore model retain loading and working-capacity accuracy at 8 bits and ±10% static beta variation within one fixed read/update allocation? | One open 3×3×16 grand-canonical lattice gas with local pair attraction and the literal positive lateral-wall field; no global latent or copied variable | Loading and capacity error at most 0.01, pair-occupation error at most 0.02, with uncertainty and start-agreement checks; reads, writes and baseline cost at qualifying accuracy | Owner selected the adsorption recommendation on 2026-10-20. The bounded exploratory probe found exact 8-bit middle-loading error 0.011078, above 0.01; sampled intervals remain unresolved, including the ideal control. Hold this allocation: no recorded study, extra budget or cap tuning. See docs/research/2026-10-20-native-adsorption-eligibility-probe.md and the three-family recommendation. No loop proposal ID assigned. Dropped by D-0006: the exact 8-bit miss and the unpowered read allocation mean no outcome can unhold this allocation; sampling adsorption again is a new row. |
| `am-coupling-bits` | done | How does associative-memory recall fall with coupling precision and per-site beta noise? | Stage A's bias-binary design with 4-, 6- and 8-bit couplings from the M5b codebook and ±10% per-site beta jitter | Recall against coupling bits and beta jitter, using stage A's exact references | Produces a spec number: the coupling bits needed to keep recall within tolerance. |
| `planar-16-offset-ferro` | done | Do the planar Ising scaling allocations (PR #92) hold on the hardware-shaped graph? | Zero-field ferromagnetic Ising on the planar subgraph of the 16-offset lattice; the thin five-replica ladder; the exact Kac-Ward reference imported from `planar_ising_scaling` | Which arms qualify, and at what sweep budget, against the exact reference | Known-good regime that confirms #92 on the chip's graph. Ferro targets only; the frustrated-grid track is reserved for the owner's annealing study. |
| `constraint-layer` | done | Does one study-local constraint layer reproduce the archived placement, rounding and finite-sweep contracts bitwise, and price updates, reads, writes and clamp changes separately? | A `thermo_lab.constraints` module that places any pairwise model on the 16-offset lattice (M5c's offsets and placement, or the planar subgraph used by `planar-16-offset-ferro`), quantizes couplings and biases with M5b's `round_parameters` codebook under a declared cap (default: step = J, the `coupling` cap `am-coupling-bits` found sufficient at 6 bits), applies static per-site beta jitter, runs chromatic block Gibbs only with no replica exchange, and counts updates, node reads, full SRAM writes and clamp changes into the existing `Z1HardwareProfile` | Bitwise agreement with the M5c placement archive, the M5b rounding archive, the `am-coupling-bits` quantized recall at 6 bits and the E0 stage B inner-K law through the layer; one priced example per constraint with the relaxed-constraint flag recorded in the hashed request | A contract study, not a sampler study: short, exact, CPU. Every later row runs through the layer by default and states which constraints it relaxed (charter amendment 1, rule 8). Reuse only; no edits to hash-bound sources. Built as engineering: [plan and As built](research/2026-10-13-constraint-layer-plan.md). |
| `dtm-small-image` | done | At what task quality does a small chained-EBM denoiser run under the constraint layer, as a function of coupling bits, beta jitter, sweeps per step and reads and writes per sample, and what does it cost against CPU Gibbs on the same model and a small conventional model of the same quality? | A two- to four-step discrete denoising model (the architecture in the DTM source card) defined on the 16-offset lattice from the start (sites are lattice sites, edges are lattice offsets, two colours) so that it passes `require_placement` with nothing relaxed; on a small binary image set: 8 x 8 to 16 x 16 binarized digits or clothing, or a synthetic set with an exact likelihood; trained in software, inference under `constraint-layer` at 4, 6 and 8 bits under a cap rule the probe declares (`absolute` or `split`; `coupling` is undefined for trained weights) and 0 and 10 percent jitter; baselines inside the row | Held-out reconstruction error and log-likelihood where tractable, per projected joule and per measured CPU second, against the baselines; the bits, jitter, sweeps and reads and writes per sample (under the declared I/O path) at which task quality stays within tolerance | Opened by D-0002 now that `constraint-layer` is done. The lab's first task-level benchmark (amendment priority 2). Baselines are part of the row (rule 10), not a separate row. The probe settles these fixed choices before any protocol: (1) the I/O path per sample (per-step host readback and rewrite, or on-chip chaining), because the sealed profile prices a clamp change as a full SRAM write at about 21,665 updates and I/O was 98 to 99.996 percent of energy in every example the layer priced, so this choice, not sweeps or bits, sets the projected joule; (2) which graph the exact reference covers (a nearest-neighbour grid of width at most 24 has a transfer-matrix likelihood; the long offsets do not); (3) whether a synthetic exact-likelihood set or a real binarized set is first, synthetic first if both fit; and it sizes training cost. No tempering arms: the layer has no exchange, and independent Gibbs won on denoising posteriors (lessons, 2026-10-02). Declare everything the plan's "What a later row must declare" lists. |
| `dtm-baseline-headroom` | done | Is there a small binary denoising family at L of at most 16, with an exact posterior, on which the smallest conventional model within 0.01 of the P-0006 lattice denoiser's error is costly, and does the denoiser beat that model's projected cost at its recorded operating point under the constraint layer? | The P-0006 generator (mixture of nearest-neighbour Ising grids with template fields, exact transfer-matrix posterior) with the family varied to break locality: flip noise q of 0.3 to 0.4, more templates, long-range template differences; L = 8 first, L = 16 only if the gap appears there; a declared digital ladder (logistic on 7 x 7, 11 x 11 and full-image windows; one- and two-layer convolutional nets at a few widths; each fits in seconds); the P-0006 lattice-native model and training recipe, three seeds, under `constraint-layer` at the recorded operating point (T = 2, 4 bits under `split`, 10 percent jitter, K = 64, host-per-step I/O, continuation with the float K = 1024 reference and a random-start control) | Primary, from the probe's exact scan (no EBM): per family, the Bayes MPM floor and each ladder model's error, parameter count and MACs per image. The DTM cannot beat the floor, so a family is a candidate only if the smallest ladder model within 0.01 of the floor costs at least X or does not exist (X and its unit declared in the protocol). Study, on candidate families only: per seed, the DTM's error against Bayes and against the baseline, which is the smallest ladder model within 0.01 of the DTM's error, with that model's size and cost; a family qualifies when the baseline costs at least X. Both sides in projected joules: the DTM's reads, writes and updates per sample with the joule as the band (profile write constant and clamp loads at the read constant), and the baseline's MACs per image at 0.1, 1 and 10 pJ per MAC (`calibrated_projection`, constant stated) | Proposed by D-0003 and amended by the owner on #120: a baseline ladder instead of a fixed 7 x 7 logistic, so the row measures EBMs rather than receptive field, and the digital side priced per MAC. Decides whether the denoising task (amendment 1, priority 2) continues at this size or stops as "no advantage over a small digital model"; today that decision rests on one family whose posterior is local (P-0006: Bayes 0.067, logistic 0.071). P-0006's quality result is a positive (the DTM recovers 94 percent of the noisy-to-Bayes headroom) and its negative is cost; cite the two apart. The probe is the exact scan alone, no EBM, under 30 CPU-minutes; if no family is a candidate, the probe closes the row with a docs-only PR and the line stops at L of at most 16 on exact synthetic families. A real binarized set (no exact floor, but a global posterior: digit identity) is then the natural next test, not a line-ending result; it needs a download, so the owner opens it. Add a source card on classical MRF image denoising (Geman and Geman 1984; Besag 1986) before any protocol. Inherits the per-seed carried-state check (lessons, 2026-10-14). A second ladder-at-least-as-good result is two consecutive negatives and goes to the owner as a premise question, not a third variant. |
| `dtm-global-latent` | dropped | Can a model placed on the 16-offset lattice carry a global latent well enough to come within 0.01 of Bayes on P-0007's candidate families under the constraint layer, and if not, which constraint stops it? | P-0007's candidate families at L = 8 with their exact transfer-matrix posteriors: `smooth8_h05_q035` and `smooth16_h05_q035` first (the ladder's first rung within Bayes + 0.01 costs 3.4e6 MACs or more), `t4_h05_q035` second (its margin over X is 0.0005). The model is hand-built, not trained: the generator's own nearest-neighbour couplings, template fields and clamp, with the template identity as a latent whose reach to all 64 pixels is spread over the lattice (tied copies, a coarse layer of block latents, or both; one-hot with inhibition or stage A's bias-binary encoding; the probe chooses), placed with `require_placement`. Operating arms run under `constraint-layer` with nothing relaxed: 4, 6 and 8 bits under a declared cap rule, 10 percent jitter, host-per-step I/O, with sweeps per sample and copy coupling varied. Reference arms state what they relax: float, long chains for equilibrium, and placement relaxed to one hub node of declared degree | Per family and arm, the pixel error of P-0007's 16-draw majority estimate against Bayes (exact), with reads, writes, clamp changes and updates per sample. The design number is the smallest sweeps, bits and copy coupling at which the hand-built model is within 0.01 of Bayes under the constraints, or, if placement must be relaxed, the hub degree it needs. Joules as a band per the standing rules, against the P-0007 ladder's MACs at 0.1, 1 and 10 pJ per MAC | Dropped 2026-10-16 before any proposal: superseded by `dtm-exact-posterior-lattice` (D-0004, adopted by the owner with amendments). Owner, 2026-10-16, on P-0007's premise question: continue. P-0007's float K = 1024 reference was as poor as its operating point, so the model class (local latents only) bounds the metric; this row tests that fixed choice before any training (lessons, 2026-09-24). The exact posterior is one categorical latent coupled to every pixel, plus nearest-neighbour couplings and the clamp, so only the latent's reach exceeds degree 16. Known risks from the AM track: the bias design needed degree 24 and up, one-hot inhibition is range-hungry, and domain-wall chains did not mix. Each outcome names a number. If the model is within 0.01 under the constraints, a training row follows and prices break-even pJ per MAC. If it gets there only at equilibrium, the number is sweeps against copy coupling. If it needs a hub, the number is the hub degree, a hardware feature. If it never gets there, global-latent tasks have a structural stop on this array, and the denoising line ends at L of at most 16 on exact families. No training in this row. The probe stays under 30 CPU-minutes; calibrate runtime on full-length runs, not short ones (P-0007 underestimated by 2.7x). A real binarized set waits for this answer; it needs a download the owner opens. |
| `dtm-exact-posterior-lattice` | done | Can a hand-built lattice model of the exact posterior of a family with digital headroom, its template latent encoded as binary hidden units and spread over the 16-offset lattice as tied copies, reach Bayes + 0.01 under the constraint layer with no training, and at what sweeps, reads, writes and free-node writes per sample from a priced start? | P-0007's `t4_h05_q035` (four templates, J 0.35, h0 0.5, q 0.35; exact transfer-matrix posterior, Bayes 0.216; archived ladder: 7 x 7 logistic at 3,136 MACs, cheapest rung within Bayes + 0.01 is `cnn2_w32_k5` at 1.74e6 MACs), `smooth8_h05_q035` whenever `t4` passes (a) and (b), because the cost verdict comes from it; the model built from the mixture posterior, not trained: one hidden unit per template with a negative bias (stage A's bias design) or exact one-hot, the pixel-clamp coupling from the noise likelihood as in P-0006, each hidden unit replicated into lattice copies tied ferromagnetically so every edge is a Z1 offset and degree stays at most 16 (M5c-style copies); under `constraint-layer` at 4, 6 and 8 bits `split` and 0 and 10 percent jitter; starts priced per sample: continuation, random, and a reset every N images; no training seeds, sampler seeds are the replication unit | Exact first: the placed, quantized model's own equilibrium pixel error against the Bayes MPM floor by enumeration over hidden configurations times the transfer matrix (float and 4, 6, 8 bits), which gives the tie-coupling and field precision the latent needs. Then sampled: pixel error at K in {64, 256, 1024} per step from each priced start; headline reads, writes and free-node writes per sample at the smallest K within Bayes + 0.01, joule as the band (profile write constant and clamp loads at the read constant); baseline: the archived P-0007 ladder for the family priced per MAC at 0.1, 1 and 10 pJ (`calibrated_projection`, no refit) | Proposed by D-0004 as the owner's decision of 2026-10-16 on the premise question (continue; a hand-built model of the exact posterior before any training) and amended by the owner on #124: stops (a) and (b) confirmed, cost stop (c) added, cost verdict from `smooth8`. Informs whether the lattice at the chip's precision can hold and sample a global latent in budget. Stop conditions (owner, 2026-10-16): (a) the quantized model's exact equilibrium error is above Bayes + 0.01 at 8 bits; (b) no priced start reaches Bayes + 0.01 within K = 1024; (c) on `smooth8_h05_q035`, the cheapest priced start that reaches Bayes + 0.01 within K = 1024 costs, per 16-draw estimate at the profile's write constant, at least as much as the archived ladder's cheapest rung within Bayes + 0.01 (`cnn2_w32_k5_gpool`, 3.43e6 MACs) at 1 pJ per MAC. (a) and (b) are judged on `t4` and then on `smooth8`; (c) on `smooth8`. Any one closes the denoising line at L of at most 16 with its mechanism named, and the real binarized set opens only on a positive. Calibrate runtime on full-length runs, not short compile-dominated ones (P-0007 underestimated by 2.7x). Probe first, under 30 CPU-minutes, enumeration and NumPy Gibbs on the unplaced model only, no THRML and no training; it closes the row with a docs-only PR if (a) or (b) fails. Known risks: continuation lost to a random start in 9 of 9 P-0007 seeds, so continuation's I/O advantage (no free-node writes) is measured, not assumed, and a random start costs 384 more writes per sample (7.9e-8 J against 2.0e-8); one-hot inhibition is range-hungry and domain-wall chains do not mix (AM A, lessons 2026-10-07); a cue-informed start is the open AM variant. Inherits the per-seed carried-state check (lessons, 2026-10-14). Cost arithmetic to carry: `t4` tests representability and mixing, not cost. Its 243,200-MAC net misses Bayes + 0.01 by only 0.0005, so its real digital cost is probably near 2.4e5 MACs (2.4e-7 J at 1 pJ per MAC), where even P-0007's continuation estimate (3.2e-7 J at the write constant) does not win. The cost verdict comes from `smooth8`, whose 243,200-MAC net misses by 0.0078: against 3.43e6 MACs at 1 pJ per MAC, P-0007's estimate would be 10.7x cheaper under continuation and 2.7x under a random start. The hand-built model's own reads, writes and free-node writes replace P-0007's in that arithmetic, and a pass on (c) is against the archived rung, not the digital minimum (the ladder has nothing between 243,200 and 1.74e6 MACs); say so. Not a third variant of the trained-DTM line: it trains nothing and tests representability and mixing of the exact model first. |
| `am-io-budget` | dropped | How many node reads, full SRAM writes and clamp changes does one recalled pattern cost in the bias associative-memory design under the constraint layer, and how does that I/O compare with its update count? | Stage A's `bias` design at 6-bit couplings with step = J (the `am-coupling-bits` spec), cue write and readout counted through `constraint-layer`, Z1 pricing | Reads, writes, clamps and updates per recalled pattern at stage A's recall tolerance; the I/O share of projected energy | Dropped by D-0002. The layer's priced example answers the question (12 reads, 32 writes, 5,120 updates per recall at P = 8; I/O 99.3 percent of modelled energy; the same arithmetic holds across stage A's cells), and the `bias` design cannot be placed (hidden degree 24, visible up to 128 against 16), so a projected joule for it prices a machine that does not exist. A placeable memory is a different design; read the diluted-memory literature and add a source card before opening a row for it. |
| `e0-remaining-kernels` | dropped | Does THRML match the archived exact inner-K law for every compiled M5a kernel, as E0 stage B showed for one? | The other 59 compiled M5a kernels (reading B) at K = 1, 2 and 4, with stage B's contract, tolerances and controls | Cells passing on output rate and joint, per kernel; control rejections | Pure replication. Stage B took about 11 CPU-minutes for one kernel, so all 59 is roughly 11 CPU-hours before parallelism; the probe sizes it, and a run over 8 CPU-hours needs the owner's say-so. Last in the order: a contract worth having, low marginal information; idle-time work. Dropped by D-0001 (owner, 2026-10-10: later); the change never reached the queue and the row was re-proposed as P-0005. |
| `adsorption-precision-requirement` | open | How many coefficient bits, and which rounding rule, keep the exact equilibrium observables of the held adsorption model within tolerance at every frozen chemical potential? | The frozen `native-adsorption-eligibility` model only: open 3×3×16 lattice gas, beta = epsilon = 1, literal wall field phi = 3, its three frozen chemical potentials (target loadings 0.2, 0.5, 0.8) and its archived spin map and placement; bits 4 to 12; rounding rules declared in the protocol before any result: nearest on the probe's family-wide split caps (the archived rule, reproduced first as a control), a balanced or error-diffused rule on the same caps, and per-point caps; no jitter, no sampling | Exact (`exact_reference`, transfer matrix) loading error per chemical potential, working-capacity error and aggregate pair-occupation error against full precision; the design number is the smallest bit count, per rule, at which loading and capacity errors are at most 0.01 and pair error at most 0.02 at all three points, or none within 12 bits | Owner selected 2026-10-23 (option 2 of THERMES's next-direction note). A representation requirement, not an application or advantage claim. Exact only: no sampling, no jitter and no cost comparison. The finite-budget sampling question and the held row stay as they are. Report non-monotone error in bits (the probe's 6-bit arm beat 8-bit at the middle point) as found, without smoothing. A rule chosen after seeing errors on these three points is development, so the protocol must say how the rule is fixed before results, and must check any rule that passes on chemical potentials held out from its choice. P-0008's 4/6/8-bit tie-limit gaps (0.0752, 0.0025, 0.0003) are cited for comparison, not rerun; its miss was finite tie strength, not precision. A pass does not unhold `native-adsorption-eligibility` and does not authorize sampling: the owner decides. Probe under 30 CPU-minutes; the transfer reference is 512 states per slice, so the full grid should fit in it. |
```

## Evidence classes

Nothing in this review is new evidence. The adsorption numbers are from an
exploration note with exact calculations and software samples; the P-0009
numbers are from a probe in an open draft PR and are exploration; the
denoising numbers are from recorded studies cited by path. The reading of
class-balanced rounding as a general lever is this review's hypothesis, not
a claim. No hardware claim.
