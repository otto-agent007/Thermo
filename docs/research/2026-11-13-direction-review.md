# Direction review D-0010

2026-11-13. Main commit 34fe37210d8e0e946fda0977376e99bf0f754657. THERMES-DIRECTOR.
Label: advice, not a decision. The owner edits the queue; this file proposes.

Trigger: one loop result accepted since D-0009
(`docs/research/2026-11-08-direction-review.md`, PR #147, merged): P-0011 on
row `margin-budget-copies`, PR #156, merged 2026-11-13 as a docs-only probe,
note `docs/research/2026-11-13-margin-budget-copies-probe.md`, lessons row
2026-11-13. In flight: P-0012 on `aggregator-read-gain`, drafting; not
touched here, weighed below. Since D-0009 main also took the four-week
evaluation (`docs/research/2026-11-11-four-week-evaluation.md`, PR #153), the
domain-wall probe (PR #151), four source cards (PR #150), the scope audit
(PR #154) and the owner's three mechanism rows (PR #155,
`docs/research/2026-11-12-mechanism-proposals.md`). Read against charter
amendments 1 and 2, the agenda, the queue at 34fe372 (fifteen rows, two
open), the lessons register, the roadmap, the last three merged studies
(P-0009, P-0008, P-0007) and all 31 source cards.

## The change I would make first

No queue change. Two things instead:

1. **Drop any further margin-budget variant on the PAsymSwap kernel
   family** (majority-amplifying hidden units, the output-output family M4I
   used, grouped-kernel survival, a finite-K theorem, tightening the
   factor-2 frontier at r >= 2). Reasons under "The one thing to drop".
2. **Put a premise question to the owner on `directed-relay-latent` before
   its probe is spent**, not a third construction. The global-latent line
   on `t4_h05_q035` has failed the same way twice (P-0007 trained denoiser,
   P-0008 tied copies: both miss Bayes + 0.01 at every priced operating
   point; `docs/experiment-reports/2026-10-16-dtm-baseline-headroom/summary.md`,
   `docs/experiment-reports/2026-10-17-dtm-exact-posterior-lattice/summary.md`),
   D-0005 asked the premise question, and the owner closed the line
   (`PROJECT_CHARTER.md`, amendment 2, "Commitments"). The row's
   pre-registered prediction is that a third construction reaches 0.01 on
   the same instance. The question is at the end of Rubric 2.

P-0011 is positive, cheap, correctly tiered, and it changes nothing about
what the lab should ask next: it explains a closed result, its design number
belongs to a compiler nobody runs, and the row itself says a pass authorizes
nothing (queue row `margin-budget-copies`, notes).

## Verdicts

**P-0011 `margin-budget-copies`: positive on the row's question, exact, a
probe with a probe's process, and it answers the row.** With r clamped
copies per input bit and every coupling capped at c, the best leakage
product found sits between 0.63 and 2.44 times e^(-4cr) in all 24 cells
(caps 1 to 4, r in {1, 2, 4}, equilibrium and K = 4); the 10-times
falsifier did not fire (probe, "Finding"; queue row, pre-registered
prediction). The useful part is not the table but the proof: at equilibrium
the floor L00 * L01 >= (1 - L00)(1 - L01) e^(-4cr) is a theorem for this
family, because the hidden spin does not see the inputs, so flipping an
input tilts the output law by a factor that does not depend on the hidden
state (probe, "The equilibrium floor, closed analytically"). The
cap-leakage note of 2026-09-23 had this as an observed frontier with no
proof (`docs/research/2026-09-23-cap-leakage-analysis.md`, "Limits";
lessons 2026-09-23). That is a genuine upgrade of a recorded limitation, at
27 CPU-minutes. Four things to read with it:

1. **The strict prediction missed, and the probe says so.** The row asked
   for the product "within optimizer resolution" of e^(-4cr); at r >= 2 the
   found frontier sits a stable factor 2 above the floor while the two
   search seeds agree to 0.03 decades (probe, result table). So the floor
   is tight at r = 1 and not shown tight for copies. The probe labels this
   a prefactor of the found frontier, not search noise, and leaves the true
   minimum between the two. Correct, and nobody should chase it (drop
   below).
2. **It does not explain M4I.** The proposal note claimed consistency with
   M4I's 89.61 percent at cap 4 (`2026-11-12-mechanism-proposals.md`,
   proposal 3, "Consequence if it holds"). The probe corrects this: M4I's
   kernel has an output-output coupling, a different family, and "this
   probe does not explain M4I" (probe, "Design number"). The lessons row
   has the scope right. The conservation line's closure rests on M4I as
   before (lessons 2026-09-25), neither strengthened nor weakened.
3. **The hidden arm answered a narrower question than the proposal
   posed.** The proposal hoped input-coupled hidden units could "amplify
   margin by majority"; the probe tested one input-hidden coupling and
   found it buys margin additively, like one more copy, and "tested no
   majority amplification" (probe, "Hidden arm"). That leaves the
   amplification idea open as a hypothesis, which is where it should stay.
4. **The owed source card is still unwritten.** The row made a card on
   noisy-gate reliability bounds (von Neumann 1956; Evans and Schulman) a
   precondition for any protocol; no protocol followed, so the trigger
   never fired, and the probe records "no literature search was done and
   no novelty is claimed" (probe, "Mechanism, model and limits"). The tilt
   argument is elementary and may well be in that literature in another
   dress. A lessons row now states a "theorem" with novelty unchecked. That
   is the rediscovery gap amendment 1 names, bounded this time to one
   paragraph of algebra.

Is it already known? Not on any card. The nearest are the Thermalizers
card's cap analysis (`extropic-thermalizers.md`, C1, reproduced on the
PAsymSwap program) and the cap note. Nothing in the 31 cards covers bounded-
weight gate reliability; the card owed by the row would.

**Operational outcome, separate from the science.** The first run crashed
in post-processing after 1,146 CPU-seconds and saved nothing; the fix
(save before post-processing) and a cut from 16 to 6 starts per seed to stay
under the 30 CPU-minute ceiling are commit 6a670f3 (probe, "What ran"). The
second run and the control passed; total about 27 CPU-minutes, inside the
ceiling. A 19-minute probe losing its work is a small process miss that
the autosave contract covers only above 30 minutes; the fix landed and no
lesson row is needed. Scheduler health: observed only as PR #156 merged
04:29:24Z and this card created 04:30:55Z, with a P-0012 card drafting;
`loop.json` was not inspected, so nothing more is claimed.

**PRs #150 to #155: process, correct.** The four new cards
(`extropic-codon-optimization.md`, `normal-cn101.md`,
`extropic-cv-blueprint.md`, `aadit-million-pbits.md`) and the scope audit
closed debts D-0006 to D-0009 named. The codon card's reflash arithmetic
(160x falls to about 2.5x once writes are priced) is the sharpest thing on
main since D-0009 and is labelled analysis, as it should be.

## Rubric 1: P-0011

- **Design number or probe with a study's process?** A probe with a
  probe's process: a committed script with its fixed choices in the
  docstring, a JSON with its hash, a note, a lessons row and a docs-only
  PR; no protocol, runner, gate or roadmap row (probe, "What ran";
  PR #156). The number it produced, c * r >= 4.29 by the bound (about
  4.46 for the kernels found) for 95 percent uniform-visitation survival
  over 500 operations, is a design rule for a compiled gate of this
  family. Nobody is compiling one: the conservation line is closed
  (lessons 2026-09-25), the Thermalizers line is parked at contract checks
  (charter amendment 1, priority 4; amendment 2, "Architecture"), and the
  row says a pass authorizes no compiler, pilot or sampled study. It is a
  bounded representation result with no present consumer, which is what
  the row asked for.
- **Constraints.** Exact only; no placement question arises for a
  five-spin kernel. The cap is a Thermo convention, not a published Z1
  limit (cap note, "Limits"); each clamped copy is counted as a full write
  under the sealed profile, as the row required (probe, "Cost of copies").
  Nothing relaxed beyond the row's design; rule 8 satisfied by statement.
- **Accounting.** Stated: 441 CPU-seconds for the run, 1,146 for the
  crashed run, 39 for a timing check; copies priced at 153.6 pJ per clamp
  change (`calibrated_projection`); the cap-versus-copies trade is left
  unpriced because Z1 publishes no cap or precision cost. Rule 9 satisfied.
- **Expectations.** The strict "within optimizer resolution" missed at
  r >= 2 by a factor 2 (inside the falsifier); the proposal's "about 4"
  was 4.29; the proposal's M4I consistency claim was withdrawn by the
  probe. No queue row assumed the opposite of anything here: the two open
  rows do not depend on this result, and the row's text said so.

## Rubric 2: the queue rows

Fifteen rows: nine `done`, three `dropped`, one `held`, two `open`. The
thirteen closed rows are as D-0007 to D-0009 covered them; P-0011 touched
none of them. The two open rows were added by the owner on 2026-11-12 and
are new to the director.

### `aggregator-read-gain` (open; P-0012 drafting)

- **Decision.** How to read a scalar observable off a resident model: one
  one-way aggregator p-bit per 16-site block against sparse, thinned site
  reads. A pass (gain >= 3) says a read-out design for a resident model
  should aggregate on chip. The only resident model in the archive is the
  held pore. On it, the eligibility probe's allocation was 9,216 reads,
  288 writes and 589,824 updates per trajectory (four-week evaluation,
  clause 2), which under the sealed profile is about 15.6 nJ of reads,
  44.2 nJ of writes and 4.2 nJ of updates (this review's arithmetic on
  `docs/z1-hardware-model.md`, analysis). A gain of 3 cuts that total by
  about 16 percent. A certified 0.01 needs about 3.4 times the reads
  (lessons 2026-10-20: per-trajectory standard error 0.0092 against the
  0.005 a 0.01 half-width needs, independence assumed), so at that
  allocation a gain of 3 is worth about 1.5x on the pore's projected
  joule. That is the size of the decision: real, modest, and on a held
  allocation whose digital denominator is unknown
  (`native-adsorption-baselines.md`, "Our baseline requirements").
  Nothing here unholds `native-adsorption-eligibility`; the row says so.
- **Cheapest probe.** The row is already probe-first and exact. Cheaper
  still, before any aggregator enters the transfer matrix: for a sharp
  threshold on a block sum the Gaussian limit gives gain about
  (2/pi) k / (1 + (k - 1) rho), with rho the mean pairwise occupation
  correlation inside the block (this review's analysis). At k = 16 a gain
  of 3 needs rho below about 0.16. P-0009's archive has the pore's exact
  pair-occupation observables (`2026-11-02-adsorption-precision-requirement/findings.md`),
  so the falsifier can be sized at a desk in an hour. One structural point
  the row should carry: an aggregator's 16 listeners must sit within the
  16-offset reach, so its block is spatially contiguous and the correlation
  penalty is forced by placement, while the baseline's sites are free to be
  spaced beyond the correlation length. That asymmetry is correct and is
  the test.
- **Source card.** None covers it. The independent-site ceiling (2/pi) k
  is the textbook one-bit quantizer loss for estimating a Gaussian mean,
  and the aggregator is one-bit stochastic computing of a block sum; the
  CN101 card already records that a stochastic bitstream's precision cost
  scales as epsilon^-2 (`normal-cn101.md`, "Method"). The row owes a card
  on dithered one-bit quantization and stochastic computing "before any
  protocol". If the probe answers the row, no protocol follows and the
  card is never written, as just happened on `margin-budget-copies`. I
  advise the owner to make the trigger "before the row closes".
- **Third variant?** No. Read-out design has no prior result in the lab.
  One-way couplings are relaxed (rule 8) and the row flags them; the
  hardware model says directional endpoint couplings are supported and
  publishes no cost (`docs/z1-hardware-model.md`), so the result stays
  apart from the supported machine. Falsifier: gain under 3 with
  correlations included, pre-registered. Low value until a resident task
  exists; cheap; honest about both.

### `directed-relay-latent` (open; runs after `aggregator-read-gain`)

- **Decision.** Whether one-way relay trees are a hardware requirement
  worth naming: the sweeps and extra sites a global binary latent needs
  to reach posterior error 0.01 on a reduced `t4` instance, against tied
  copies on the same instance. If it passes, the lab can state "a global
  latent on this lattice needs one-way couplings at cost X"; if it fails
  on lossy fan-in, binary relays cannot carry a four-template logit and no
  coupling feature fixes that. Either answer is a bounded architecture
  statement of the kind amendment 2 priority 3 permits.
- **Cheapest probe.** Cheaper than enumerating the non-reversible chain:
  compute, on the reduced instance, the exact logit the latent needs
  (from the posterior) against the effective field a chain of binary
  relays can deliver at 8 bits in their near-linear regime. If the relay
  field's resolution cannot separate the templates at the needed margin,
  the row's own falsifier fires before any chain is built. That is a
  half-day desk calculation on P-0008's archived posterior.
- **Source card.** None. The row owes asynchronous or stale Gibbs bias
  bounds; the Aadit card's communication-versus-update ratio
  (`aadit-million-pbits.md`, C2) is the nearest thing on file and is about
  inter-device staleness, not a relay tree.
- **Third variant? Yes, by the rubric's test.** The last two results in the
  global-latent line on `t4_h05_q035` failed the same way: P-0007's
  trained denoiser landed 0.078 to 0.147 above Bayes where the digital
  ladder was costly, and P-0008's hand-built tied-copy latent missed
  Bayes + 0.01 at every priced start with the oracle pair agreeing on
  0.0117 (summaries cited above). D-0005 asked the premise question; the
  owner closed the denoising line with no further premise test, and
  amendment 2 keeps that closure. The row says it does not reopen the
  line, and its framing (a counterfactual coupling feature, reported as a
  requirement) is the one amendment 2 allows. But its pre-registered
  prediction is "the directed construction reaches error 0.01 where tied
  copies at finite tie do not": a third construction aimed at the gate two
  constructions missed, on the same instance. Under the two-failures rule
  my only output for this line is a question, so:

  **Question to the owner.** Is `directed-relay-latent` asking for a
  hardware requirement (what one-way fan-in and fan-out would cost to hold
  a global latent, reported as a requirement whether or not 0.01 is
  reached), or for a construction that reaches 0.01 on `t4`? If the
  latter, it is the third try at a closed premise and I advise against
  spending the probe. If the former, the row's success condition should
  be the requirement, not the pass, and the falsifier (lossy fan-in at
  8 bits) is the result most likely to come back.

The row already waits on `aggregator-read-gain` and tells the proposer to
ask the owner if that row stops. The question above fits that checkpoint
and needs no queue edit.

### Is the queue pointed at the primary question?

At the architecture half of it, with the task half empty. Amendment 2's
question has two parts, a useful capability more economically than the best
digital method, and the representations and architectures that make it
possible (`PROJECT_CHARTER.md`, "Scientific mission"). The two open rows
address the second part under a relaxed feature whose Z1 cost is
unpublished. Neither names a task, a user or a digital baseline; the
proposal note says so in its first section. The agenda's named gap,
application validity together with digital cost headroom
(`2026-10-20-lab-research-agenda.md`, "Immediate disposition"), is touched
by no row in the queue and cannot be closed by any mechanism probe. That
is a statement of fact about the queue, not an objection to the rows: the
owner read the four-week evaluation's parking recommendation and chose
three cheap mechanism rows instead, which is the owner's call to make.

## Rubric 3: can the lab state the four-week test today?

The test fell due on 2026-11-11 and was reported on its original terms
(`docs/research/2026-11-11-four-week-evaluation.md`; lessons 2026-11-11).
P-0011 is not a task row and changed none of its numbers. Restated:

1. *Precision and noise tolerance.* 4 bits under `split` with 10 percent
   static jitter on the local L = 8 denoiser (P-0006); 6 bits with fields
   in J for the bias memory; 5 bits for the placed tie-limit latent under
   either rounding rule (P-0010 correcting P-0008's grid); exactly 4 field
   bits under class diffusion and 9 under nearest for the 144-site pore
   (P-0009, P-0010). Jitter isolated in three studies, static per-site
   gain only (evaluation, clause 1).
2. *Sweeps, reads and writes per useful sample.* 64 sweeps per step, 128
   reads and 128 clamp loads per two-step sample on the local family; no
   useful sample on any family with a global latent; the pore priced a
   trajectory that certified nothing (evaluation, clause 2).
3. *Projected energy at stated quality against a conventional baseline.*
   **Negative.** 3.2e-7 J per 16-draw estimate at error 0.075 (9.7e-9 J at
   the clamp-read end) against a 3,136-MAC logistic at 0.071 for 3.1e-10
   to 3.1e-8 J; break-even only above 3.1 pJ per MAC at the favourable end
   of the band, and less accurate at any price (evaluation, clause 3).

Largest gap: clause 3 for any task with a quality pass. No row closes it;
the two open rows cannot by construction. The evaluation's recommendation
was to park until a task passes the headroom screen; the owner's response
was three mechanism rows. The honest status is: the test was met in form,
its content is negative for the hardware, the loop is running on mechanism
questions at probe cost, and the un-park condition the evaluation named (a
task with digital headroom, native placement, a user and a tolerance) is
unmet and unaddressed.

Missing evidence, unchanged from the evaluation: no hardware measurement;
no published clamp-only constant (the denoiser's cost moves 33x across
that band); no lower bound on digital cost; no training under the
constraints; thinned draws from one clamp load untested. P-0011 adds one
dependence of the same kind: the copies-versus-cap trade cannot be priced
because Z1's coefficient range and precision cost are on no card (D-0009,
"Rubric 4"; probe, "Cost of copies"). Every lever the lab has priced since
October, reflashes (codon card), ramp stages (domain-wall probe), clamp
loads (P-0006, P-0007), tempering exchanges (exchange cost projection) and
now gate copies, lands on the write constant. The aggregator row is the
first that targets reads instead; on the pore that is the smaller term.

## Rubric 4: representation, architecture, useful computation

**P-0011.**

- *Intended model:* a gate mapping 00 to 00 and 01, 10 to a one-particle
  word, with bounded coefficients. *Constrained representation:* r clamped
  spins per input bit, exactly equivalent to one input with coupling range
  rc; coefficient cap c. *Actual update process:* equilibrium (proved),
  K = 4 uniform-reset sweeps (checked numerically only). *Estimator:* none;
  every number is an exact conditional (probe, "Mechanism, model and
  limits").
- *Established, bounded.* In a kernel whose hidden unit is blind to the
  inputs, the discrimination margin is additive in coupling range along
  the input-to-output path, and the hidden unit contributes nothing to it.
  Precision (cap, with c = delta (2^(n-1) - 1) at n bits) and fan-in
  (copies, each a write) are therefore exchangeable currencies for margin,
  with a hard leakage floor set by their product. Scope: this family,
  single kernel, uniform visitation, hop fidelity ignored.
- *Challenged.* The cap note's status of the e^(-4c) law as "an observed
  minimum, not a globally certified bound" (lessons 2026-09-23): at r = 1
  it is now a tight theorem at equilibrium. Also the proposal's hope that
  hidden units could amplify margin cheaply: one input-coupled hidden unit
  is worth one copy, no more (probe, "Hidden arm").
- *Unsupported generalizations to avoid.* "Gate leakage floors are
  e^(-4cr)" for any family (M4I's output-output coupling moves the
  frontier; the probe does not cover it); "copies are cheap" (each clamp
  change of each copy is a 153.6 pJ write, the dominant constant in every
  pricing the lab has done); "the floor explains M4I's 89.61 percent" (it
  does not, different family).
- *User and decision.* A compiler author choosing between a larger cap and
  more input copies for a bounded-coupling gate. None is active: the
  conservation line is closed and the Thermalizers implementation is
  unpublished (`extropic-thermalizers.md`, code note).
- *Strongest digital baseline.* Not applicable; a digital gate does not
  leak, and the row claims no advantage (probe, "Digital baseline and
  headroom").
- *Hardware requirement.* A conditional one: a 500-operation program in
  this gate family at 95 percent survival needs coefficient range times
  fan-in of at least 4.3 logit units per clamped input bit. Whether Z1
  meets it depends on its coupling range and precision, which are
  unpublished; it is a question for a source card, not a row.
- *Counterfactuals.* None relaxed beyond the cap convention. No advantage
  on the supported machine is claimed or implied.

**Domain-wall probe (owner-requested, 2026-11-11), briefly, since it is the
only other new result.** It scopes a stage A negative: domain-wall chains
mix under a ramped penalty at 5 to 7 spins and not at 31 or 127, and each
ramp stage is a reflash that costs about 1,000 times the sweeps for P = 8
(probe, "Resource accounting"). The representation claim is bounded to
chain length and schedule; the bias design stays ahead in 30 of 32
cell-budgets. It reopens nothing and corrects the scope of lessons
2026-10-07, which the register now carries. The codon card's working
domain-wall runs (at most 5 spins) fit it.

**For the two open rows,** falsifiers and cheapest checks are in Rubric 2:
gain under 3 with correlations included, sized first from P-0009's pair
statistics; an exact relay-field resolution short of the template margin
at 8 bits, sized first on P-0008's archived posterior. Both are relaxed-
feature rows (one-way couplings) and establish nothing on the supported
machine; both should report a break-even allowance for the feature rather
than an advantage, as the agenda instructs ("How to choose the next
question", counterfactual paragraph). No new row is proposed, so no new
falsifier is owed.

## The one thing to drop

**Any further margin-budget variant on the PAsymSwap kernel family**: a
majority-amplification arm with several input-coupled hidden units, the
output-output family of M4I, a grouped-kernel or context-weighted survival
bound, a K = 4 theorem, pricing copies against caps, or a search to close
the factor 2 at r >= 2. Three reasons:

1. **No consumer.** The conservation line is closed (lessons 2026-09-25;
   CLAUDE.md, "Where the project stands"), the compiler that would use a
   margin rule is parked until Extropic publishes its implementation
   (amendment 1, priority 4), and the row's own notes say a pass authorizes
   no compiler, pilot or sampled study. A second margin result would be a
   design rule for a program nobody compiles. "Explains a closed result" is
   a one-time licence, not a programme.
2. **The only hardware consequence is unpriceable today.** Copies trade
   writes for cap; whether that is a good trade on Z1 depends on a cap or
   precision cost that is on no card (probe, "Cost of copies"; D-0009,
   "Rubric 4"). Until a coupling-range figure is published, every further
   theorem lands on the same unknown.
3. **Novelty is unchecked, and a third pass would be the amendment's
   failure in miniature.** The owed von Neumann and Evans-Schulman card was
   never written because no protocol followed (probe, "Novelty
   unchecked"). Bounded-weight gate reliability is a 70-year-old
   literature. If the owner wants one more thing done in this line, it is
   that card, not another cell.

This drop is a boundary, not a row status: no queue row carries these
variants. It is written down so the next proposal note does not grow them.

## Evidence obligations

Read from `/home/user1/.hermes/profiles/thermes/loop/evidence-obligations.json`
(checked 2026-11-11T08:54Z at 682884a), reconciled against main at 34fe372.
The index is dated administration, not proof; this review does not edit it.

| ID | Status in index | Reconciliation at 34fe372 |
| --- | --- | --- |
| EO-001, error-diffusion card | `closed_scoped` | Unchanged. Card on main at abstract depth; `error-diffusion-quantization-bias.md`, "Limits". |
| EO-002, full-source check before stronger reliance | `conditional_open` | Not triggered. P-0011 and the domain-wall probe do not rely on that card. Stays conditional. |
| EO-003, unpowered read-allocation lesson | `open` | **Closure evidence now on main:** lessons entry 2026-10-20 ("Size a sampled allocation against the exact variance before running it", standard error 0.0092, t half-width about 0.016), added by a7197c3 in PR #154 on 2026-11-13. Exploration label and held status retained. Closable. |
| EO-004, independent rounding-transfer case | `closed_negative` | Unchanged; P-0010. |
| EO-005, amendment 1 four-week evaluation | `open` (D-0009 awaiting owner at check time) | **Closure evidence now on main:** `docs/research/2026-11-11-four-week-evaluation.md` (PR #153), lessons row 2026-11-11, D-0009 merged as PR #147. Reported on the original terms with negatives and gaps. Closable. The owner's direction decision in response (three mechanism rows) is recorded in the queue, not in the evaluation. |

Candidate debts not in the index, for the owner to add or decline:

- Source card on noisy-gate reliability bounds, owed by
  `margin-budget-copies` "before any protocol"; the row closed by probe,
  so the trigger never fired while a lessons row asserts the law. Missing.
- Source card on dithered one-bit quantization and stochastic computing,
  owed by `aggregator-read-gain` on the same trigger; P-0012 is drafting
  and the same loophole applies. Missing.
- Source card on asynchronous or stale Gibbs bias bounds, owed by
  `directed-relay-latent`. Missing; moot if the premise question closes
  the row.

## Proposed queue

No queue change.

Both open rows were selected by the owner on 2026-11-12; P-0011 does not
bear on them; the premise question on `directed-relay-latent` goes to the
owner through the checkpoint the row already contains, and a hold would be
more than the two-failures rule lets me propose. No new row: the queue's
gap is a task with digital headroom, which no mechanism row supplies, and
the agenda forbids filling the queue to keep the loop running. An empty
queue after P-0012 is a valid state.

## Evidence classes

Nothing in this review is new evidence. P-0011's numbers are exploration
with `exact_reference` values, cited from its note; the pore's read, write
and update counts are from the eligibility probe as restated in the
four-week evaluation; the joule arithmetic on them, the Gaussian gain
estimate and the 1.5x figure are this review's analysis under the sealed
profile and stated assumptions, labelled as such; P-0007, P-0008 and P-0009
figures are from recorded studies cited by path. No hardware claim.
