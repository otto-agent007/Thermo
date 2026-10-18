# Direction review D-0005

2026-10-18. Main commit 1d78bae75ffeeb6171e0b7ed28510d30b07a519f. THERMES-DIRECTOR.
**Advice, not a decision.** The owner edits the queue; this file proposes.

Covers the one loop result accepted since D-0004 (P-0008
`dtm-exact-posterior-lattice`, PR #126), nothing in flight, and reads the
queue against charter amendment 1 (`PROJECT_CHARTER.md`, "Amendment 1").
Previous review: D-0004 (`docs/research/2026-10-16-direction-review.md`,
PR #124, merged; its row landed in PR #125 with the owner's amendments).

## The change I would make first

Leave the queue empty and write the four-week test's answer down now, before
its due date of 2026-11-11 (`CLAUDE.md`, "Where the project stands", first
bullet). On every task the lab has run, clause 3 reads against the hardware,
and the P-0008 pricing shows why no L = 8 task could have changed that under
the current I/O premise: 64 clamp loads per draw at the sealed profile's write
constant cost 6.3e-7 J per 64-draw estimate before a single sweep (arithmetic
on `docs/experiment-reports/2026-10-17-dtm-exact-posterior-lattice/findings.md`,
"Resource accounting": 64 writes per sample, 64 samples per estimate, write
constant 1.536e-10 J), which is above the 2.4e-7 J a 243,200-MAC digital net
costs on `t4` at 1 pJ per MAC (`docs/research-queue.md`, row
`dtm-exact-posterior-lattice`, notes). The denoising line is closed by the
owner's own stop rule, its last two results fail the same way, and the
charter's second task (dense associative memory) needs the same construction
that just failed. Under the loop's rule my only output for this line is the
premise question in "Pattern" below. No new row.

## Verdict

**P-0008 `dtm-exact-posterior-lattice`: negative, by the frozen rule, at the
resolution limit, and it would not have mattered.** Stop (b) fired on `t4`:
no priced start on the operating arm reached Bayes + 0.01 within K = 1024
(best: random start, pooled gap 0.0166). The oracle pair agreed at K = 4096
on 0.0117, so rule 7 names finite-w representation (findings, "Rule 7: the
mechanism"; `completion.json`, `verdicts.t4_h05_q035`). Three facts a reader
needs beside the verdict:

1. **The exact side is a positive and a design number.** The placed,
   quantized tie-limit model holds the posterior at 8 bits (MPM gap 0.0003)
   and at 6 bits (0.0025), and fails at 4 bits (0.0752), all
   `exact_reference` on 1,000 test images (findings, "Exact stages"). That is
   the precision requirement for a four-way global latent spread as tied
   copies on the 16-offset lattice, at infinite tie strength.
2. **The miss is 0.0017 wide and sits inside both pair intervals** (0.008 to
   0.014 and 0.009 to 0.015; findings, "Caveats"). The protocol's own power
   analysis puts a true gap of +0.012 at a 0.17 chance of reaching. The rule
   gated on the point estimate, as frozen, and the report says so. Nothing to
   soften: by the rule the line is closed. But a re-run with more images is
   exactly the third variant the loop forbids, and item 3 says it would buy
   nothing.
3. **A pass on `t4` could not have given clause 3.** `t4` was never the cost
   family (row notes; protocol, "Fixed choices"). Its continuation estimate
   costs 7.24e-7 J at the write constant against 2.4e-7 J for the digital net
   that misses by 0.0005, and 1.02e-7 J only if clamp loads are priced at the
   read constant, a constant Extropic has not published (queue header,
   standing rule 1). The cost verdict was reserved for `smooth8`, and the
   round-3 probe found the bias-design encoding on `smooth8` sits at +0.0102
   to +0.0124 at 8 bits before any sampling
   (`docs/research/2026-10-17-dtm-exact-posterior-lattice-layout-probe.md`,
   finding 5; protocol expectation 6). So the study's possible outcomes were
   "negative on `t4`" or "stage E stop on `smooth8`", and the proposer wrote
   both down in advance. The 4.56 CPU-hours bought a mechanism name and the
   precision number, not a cost number.

Process: the study did what its protocol said, kept the three evidence
classes apart, said at every sampled number that NumPy Gibbs is not the
lowered THRML program, and replayed in full; the reviewer re-ran the replay
and spot-checked 30 numbers (PR #126, second opinion). Runtime calibration
underestimated by 1.6x after P-0007's 2.7x, both already in
`docs/knowledge/lessons.md` (rows 2026-10-17, 2026-10-16). The probe overran
its budget twice over (59 CPU-minutes against 30, round-2 note). Four proposal
rounds for a 2.6-hour study is the fixed cost amendment 1 warns about.

## Rubric 1: the accepted result

### P-0008 DTM exact-posterior lattice

- **Design number?** One, exact: 8 and 6 bits hold a placed tied-copy latent
  at the tie limit, 4 bits does not (findings, "Exact stages"). The sweep
  and I/O numbers (64 reads; 64, 101.8 or 253 writes; 193,536 updates per
  sample at K = 1024) attach to no quality, because no cell reached and
  `operating_point` is null (`completion.json`). The negative carries a
  mechanism with a number: at w = 1.75 the finite-tie equilibrium is 0.0117
  above Bayes, and larger ties mix worse (development gap 0.0197 at 1.75 to
  0.039 at 3.0; findings, "Caveats"). That is a two-sided trade-off a
  designer can use: too weak and the copies disagree, too strong and under
  `split` the tie takes the coupling cap and the chains slow.
- **Hardware constraints.** Operating arm relaxed nothing: placement with
  `require_placement`, 8-bit `split`, 10 percent static jitter, two-colour
  Gibbs, no exchange, host-per-step I/O (summary, "Constraints (rule 8)").
  Relaxed and stated: `beta_jitter` on the jitter-0 and 6-bit arms,
  `quantization` and `beta_jitter` on the float reference, the start on the
  oracle cell, finite w on the exact stage (tie limit, by construction).
  The sampler is NumPy, not the layer's THRML lowering, stated everywhere.
  Rule 8 met.
- **Resource accounting.** Yes: counts per start as the headline, joule as
  the band (7.24e-7 to 1.02e-7 J per estimate under continuation), the
  archived `cnn2_w32_k5` rung at 0.1, 1 and 10 pJ per MAC, CPU seconds
  beside (findings, "Resource accounting"). Rule 9 met; rule 10 is not one
  ratio because there is no operating point (reviewer caveat, PR #126).
- **Missed expectations.** Two of six (findings, "Pre-registered
  expectations"): the operating arm landed at +0.017 to +0.021 against an
  expected +0.004 to +0.015, and the oracle pair agreed where disagreement
  (mixing) was expected. The second miss is the result: the probes ran the
  oracle start only to K = 1024, where it still held the latent (round-3
  note, finding 6), so "mixing" was the expected mechanism and
  "representation" the actual one. Which rows assumed the opposite: the
  dropped `dtm-global-latent` row listed "if it gets there only at
  equilibrium, the number is sweeps against copy coupling" as an outcome,
  which assumed a finite-w equilibrium within tolerance exists; P-0008 says
  it does not at any tie value in the grid. The owner's amendment to the
  row, "the real binarized set opens only on a positive", now closes that
  set. No open row assumed anything; the queue at 1d78bae has none.

## Pattern, and the premise question

The denoising line has three recorded negatives (P-0006 on cost,
`docs/experiment-reports/2026-10-14-dtm-small-image/findings.md`, "Cost
against conventional machines"; P-0007,
`docs/experiment-reports/2026-10-16-dtm-baseline-headroom/findings.md`,
"The model, not only the point, is the limit"; P-0008 above). The last two
fail the same way: the placeable model's own equilibrium misses the
posterior, and neither the sampler, the bits nor the jitter is the cause
(P-0007: the float K = 1024 reference is as poor as the operating point;
P-0008: the float K = 4096 reference misses at 0.0124 and the oracle pair
agrees). Under the loop's rule (`docs/research-loop.md`, "Director", "What
it may not do") I propose no third variant, and the owner's stop rules have
already closed the line (row notes, "Any one closes the denoising line at L
of at most 16 with its mechanism named").

Two things make this a premise question rather than a row question.

1. **Every task the charter names needs a global latent on a degree-16
   lattice, and the only placement is tied copies.** Amendment 1 names
   denoising first and associative-memory recall second. A posterior with
   digital headroom needs a global latent (Besag,
   `docs/knowledge/sources/besag-dirty-pictures.md`, C1 and "Relevance";
   confirmed by P-0007 stage 1), and the dense memory's hidden unit couples
   to every visible bit (`docs/knowledge/sources/krotov-hopfield-large-memory.md`,
   C1; the AM `bias` design has degree 24 and up, `am-io-budget` row notes).
   Spread as tied copies, that latent misrepresents at an expressible tie
   strength (P-0008) or does not mix (AM A, lessons 2026-10-07: domain-wall
   chains). A placeable associative memory is therefore the same
   construction with the same two failure modes, and a row for it would be
   the third variant.
2. **The cost verdict is set by the I/O premise, not by any model.** Under
   the sealed profile a clamp load is a full SRAM write, about 21,665 updates
   (`docs/experiment-reports/2026-10-07-exchange-cost-projection/findings.md`),
   and every DTM study prices one clamp load per pixel per draw. At L = 8
   and 64 draws that is 6.3e-7 J per estimate at the write constant with
   zero sweeps, 87 percent of P-0008's continuation estimate (arithmetic on
   the findings' counts). The band's low end (clamp loads at the read
   constant) is 7x lower and rests on a constant Extropic has not published
   (queue header). So on any L = 8 task the hardware wins only if the digital
   side costs more than about 6e5 MACs at 1 pJ per MAC, or if the clamp
   constant or the I/O path changes. Neither is a sampler or model question.

**Premise question for the owner (also in this card's metadata).** The
denoising line is closed under your stop rules, its last two results fail
the same way, and the charter's second task needs the construction that
failed. Which premise, if any, do you want tested next?

- (1) None: state the four-week test's answer on 2026-11-11 as "clause 3
  reads against the hardware on every task run, and the write-constant clamp
  floor says it would on any L = 8 task whose digital cost is under about
  6e5 MACs", and choose the next task outside denoising and dense memory.
  This is the charter's own instruction when the test fails ("revisit it
  rather than add process"), and it is what I would do.
- (2) The I/O premise: whether one clamp load may serve many draws (thinned
  draws from one chain, or any on-chip path the profile allows). That is
  the only lever in the lab's hands that moves clause 3 by more than the
  band, up to 64x on writes at 64 draws, and it is a pricing and estimator
  question answerable on P-0008's archived model in NumPy, not a study.
  Whether the sealed profile permits it is your reading of the source, not
  mine.
- (3) The tied-copy premise: before any further tied-copy design, a source
  card on minor-embedding chain strength (the penalty trade-off
  `docs/knowledge/sources/doucet-qubo-encoding.md` C1 already names for
  one-hot penalties: too weak leaves infeasible states, too strong
  compresses the energy scale) and one exact probe of finite-w leakage
  against w and bits. The lab reproduced that trade-off at 8 bits without a
  card for it, which is the failure amendment 1 records.

I recommend (1). If you choose (2) or (3), say what result ends it.

## Rubric 2: the queue rows

The queue at 1d78bae has eight rows, none open: `am-coupling-bits`,
`planar-16-offset-ferro`, `constraint-layer`, `dtm-small-image`,
`dtm-baseline-headroom`, `dtm-exact-posterior-lattice` (done);
`dtm-global-latent`, `am-io-budget`, `e0-remaining-kernels` (dropped). The
loop is idle until the owner adds a row.

**Done rows.** No decision depends on re-running any of them.
`dtm-exact-posterior-lattice`: the decision it informed is made (the line
closes); the cheapest probe was run, three rounds of it; no source card
answered it, and the nearest cards (Chancellor C2,
`docs/knowledge/sources/chancellor-domain-wall.md`; Doucet C1) predicted the
shape but not the number; not a third variant when proposed, but its result
makes the next one so. `am-coupling-bits` remains the lab's cleanest spec
number (6 bits with fields in units of J; jitter negligible, lessons
2026-10-08). `constraint-layer` is the shared code every row used. The three
DTM rows are the closed line. Unchanged.

**Dropped rows.** `dtm-global-latent`: stays dropped; P-0008 answered its
question ("which constraint stops it": tie strength under `split`'s shared
coupling cap, with mixing worsening as the tie rises), so re-opening it would
re-ask a settled question. `am-io-budget`: stays dropped; its note already
says a placeable memory is a different design that needs the diluted-memory
literature first, and item 1 above says that design is the construction
that just failed. `e0-remaining-kernels`: informs no row. Unchanged.

**New row.** None. The one row I could write (a placeable associative
memory, or a `smooth8` re-run under a changed estimator or tolerance) is a
third variant of a line that has failed twice the same way.

## Rubric 3: can the lab state the four-week test today?

Due 2026-11-11. For one task, yes, and every clause now has a bound on it:

1. *Precision and noise tolerance.* The strongest numbers the lab owns.
   Static 10 percent beta jitter changed nothing in four studies
   (`am-coupling-bits`, P-0006, P-0007, P-0008). Precision: 4 bits under
   `split` on the local denoising family (P-0006); 6 bits with fields in J
   for the bias memory (`am-coupling-bits`); 8 or 6 bits for a tied-copy
   latent at the tie limit, 4 fails (P-0008). All `exact_reference` or
   checked against one.
2. *Sweeps, reads and writes per useful sample.* K = 64 per step, 64 reads,
   64 writes, 0 clamp changes on the local family under continuation
   (P-0006). On any family with a global latent there is no useful sample:
   continuation lost in 9 of 9 seeds (P-0007) and in every cell at every K
   (P-0008, expectation 4), and no start reached at all (P-0008).
3. *Projected energy at a stated quality against a conventional baseline.*
   Both sides projected on the local family: the DTM at error 0.075 costs
   2.0e-8 J per sample at the write constant; a 114-parameter logistic at
   0.071 costs 3.1e-10 to 3.1e-8 J per image across the per-MAC band
   (P-0007 findings, "Cost, both sides projected"). The digital model wins
   at every band point but one corner, and is more accurate there. On `t4`
   there is no operating point to price, and the clamp floor above says
   the hardware could not have won at the write constant.

The row that closes the largest gap: none open, and the gap is not a row's.
It is the clamp constant and the I/O path (item 2 of "Pattern"), which only
Extropic's publication or the owner's reading of the profile can move. The
honest statement on 2026-11-11 is that the amendment produced the numbers
it asked for and they read against the hardware on the task Extropic names.

## The one thing to drop

`smooth8_h05_q035` as the cost-verdict family, and stop (c) with it. It is
the only archived family whose digital rung is costly enough to lose
(3.43e6 MACs), and the hand-built encoding's own exact 64-draw gap at 8
bits is +0.0102 to +0.0124 on both layouts before any sampling (round-3
probe, finding 5; P-0008 protocol, expectation 6, "Fixed choices: Hidden
encoding"). No re-run of P-0008 on it, with more images, another tie value
or another layout, can produce the cost number, because a perfect sampler
of that model already misses. Concretely: do not re-run P-0008 on `smooth8`,
do not widen its tolerance or draw count to make (c) decidable, and do not
cite its 3.43e6-MAC rung as headroom the hardware could claim. If the owner
chooses premise (2), price it on `t4`'s archived model, where the exact side
passed.

## Proposed queue

No queue change.

## Evidence classes

Nothing in this review is new evidence. The 6.3e-7 J clamp floor is
arithmetic on P-0008's archived counts (64 writes per sample, 64 samples per
estimate) and the sealed profile's write constant (1.536e-10 J, as the
round-1 probe JSON records it), under the same `calibrated_projection`
label; the 87 percent share and the 6e5-MAC break-even follow from it at
1 pJ per MAC. The reading of the dense memory's hidden unit and the mixture
posterior's template latent as the same placement problem is this review's,
not a source card's claim. No hardware claim.
