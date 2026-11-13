# Direction review D-0011

2026-11-13. Main commit 0fdd07ab88410d4d8a2b56c5ac7e9dd6ff050354. THERMES-DIRECTOR.
Label: advice, not a decision. The owner edits the queue; this file proposes.

Trigger: one loop result accepted since D-0010
(`docs/research/2026-11-13-direction-review.md`, PR #158, merged): P-0012 on
row `aggregator-read-gain`, PR #157, merged 2026-11-13 22:31Z as a docs-only
probe, note `docs/research/2026-11-13-aggregator-read-gain-probe.md`, lessons
row 2026-11-13. Since D-0010 main also took the three source cards the
mechanism rows owed (PR #160: `one-bit-mean-estimation.md`,
`asynchronous-gibbs-bias.md`, `noisy-gate-reliability.md`). In flight, not
touched, weighed below: P-0013 on `directed-relay-latent` (draft PR #159,
answered, negative) and the owner's loop debrief (PR #161). Read against
charter amendments 1 and 2, the agenda, the queue at 0fdd07a (fifteen rows,
one open), the lessons register, the roadmap, the last three merged studies
(P-0009, P-0008, P-0007) and all 34 source cards. No note under
`docs/research/` on main is dated after 2026-11-13.

## The change I would make first

No queue change, and nothing to open. Two things instead:

1. **Read P-0012's "pass" as a fail of the trigger it was meant to fire.**
   The proposal note held a second case in reserve, susceptibility on the
   planar ferromagnet, "only on a pass"
   (`docs/research/2026-11-12-mechanism-proposals.md`, proposal 1, "Cheapest
   decisive check"). The gain of 5.2 holds only with one-way couplings, a
   placement the archived map cannot give (k is at most 5 there), a bias
   range four times the pore's cap and a host-adapted threshold (probe,
   "Finding"; "Constraints applied and relaxed"). On the supported machine
   the answer is no. The second case, and every other aggregator follow-up,
   is the thing to drop (below).
2. **Put the global-latent premise to the owner again, now with P-0013 in
   hand.** D-0010 asked it; the probe ran 42 minutes later with no answer on
   record (PR #161, finding 2). P-0013 is the line's third same-way negative
   on `t4_h05_q035` (P-0007, P-0008, P-0013). The question is at the end of
   Rubric 2 and in this card's metadata. No fourth construction, however
   framed.

After P-0013 merges the queue is empty. That is a valid state (card; agenda,
"Immediate disposition"). The un-park condition the four-week evaluation
named, a task with digital headroom, native placement, a user and a
tolerance, is still unmet, and no mechanism row can meet it.

## Verdicts

**P-0012 `aggregator-read-gain`: negative on the supported machine, positive
only as a conditional hardware requirement; exact; a probe with a probe's
process; it answers the row.** On the frozen pore an aggregator over a
contiguous 16-site z-channel at w = 0.5 with a host-adapted half-rate
threshold needs 5.2 to 6.0 times fewer reads than round-robin site reads for
loading error 0.01 (one asymptotic standard error), at float, 4 and 6 bits
and ±10 percent aggregator gain, with within-block correlations costing 22 to
39 percent of the independent-site gain (probe, "Results"). Four things
decide it, and each is a relaxation (probe, "The relaxed constraints decide
the result"; lessons 2026-11-13):

1. **Placement.** On the archived map a free slot sees at most 5 pore sites;
   the gain there is 1.1 to 2.7, under 3 at every point.
2. **Caps.** The centring bias needs |b| up to 7.95 against the pore's field
   cap of 1.99; on the pore's caps the gain is 2.93 at low gain and the
   worst-block jitter check breaks.
3. **Threshold rule.** A fixed bias gives about 1 away from its point; a
   round-robin bank of three gives 2.2 to 2.6. The threshold, not the
   coupling, is the design.
4. **One-way couplings.** Symmetric coupling moves loading by 0.011 to 0.139,
   above tolerance, so the stop case the row named is real.

Four things to read with it:

- **It was already known, and the card says so.** The (2/π)·k ceiling is
  Kipnis and Duchi's one-bit loss (C1), the adaptive threshold is their C2,
  aggregating m samples before the bit is Lau and Scarlett's C3: "None of the
  three is new" (`docs/knowledge/sources/one-bit-mean-estimation.md`,
  "Relevance to Thermo"). The card was written after the row was spent
  (PR #160 opened 05:54Z; PR #157 opened 04:34Z). The probe's contribution is
  the constrained correction, and that correction is the negative: on the
  supported placement and caps the published ceiling is not reachable. This
  is the rediscovery amendment 1 records, bounded to about ten CPU-minutes.
- **The strict prediction held only in the relaxed arm, and not everywhere
  there.** "Gain of at least 3 at all three chemical potentials" (row,
  pre-registered) holds at w = 0.5 and 1; the falsifier fires at w = 0.25
  with 10 percent low gain (2.94 at loading 0.2), and at w = 1 the declared
  worst-block bias check fails (0.0116) while the pooled bias is under 0.001
  (post-hoc). The probe reports both without smoothing. Correct.
- **Reads are the minor term on the only resident model.** At the
  eligibility probe's allocation the pore costs about 15.6 nJ of reads, 44.2
  nJ of writes and 4.2 nJ of updates per trajectory (D-0010's arithmetic on
  `docs/z1-hardware-model.md`, analysis). A 5.2x read gain saves about 12.6
  nJ of 64, a fifth, and costs 45 adaptation writes (6.9 nJ) per chemical
  potential, repaid after 3 to 5 estimates (probe, "Writes against reads").
  Even on the hypothetical machine this is a modest saving on a held
  allocation with no user and no certified estimate.
- **The one number on the supported machine is not the aggregator's.** The
  baseline column says round-robin single-site reads, thinned to
  independence, need 1,038 / 2,083 / 1,454 reads for a 0.01 standard error
  at loadings 0.2 / 0.5 / 0.8 (probe, "Reads to error 0.01"). The eligibility
  allocation read the whole array 64 times, 9,216 node reads, for a standard
  error of 0.0092 (lessons 2026-10-20). One full-array read carries the
  information of about 46 independent site reads at the middle point, since
  Var(N) is 3.1 times the independent value there (probe, "Fixed choices",
  model row; this review's arithmetic, analysis). So sparse thinned site
  reads cut the pore's read count three- to fourfold at equal precision,
  with the thinning updates uncounted (cheap, 7 fJ each) and the pore still
  held.
  It changes no decision, because writes dominate and nothing certifies 0.01
  either way. It is the kind of read budget rule 9 asks for, and the probe
  did not headline it.

Is it already known? Yes, the mechanism; see above. Nothing on any card
covers the constrained case (contiguous block, logistic dither, capped bias,
a slot that sees 5 sites), which is what the probe measured and what came
out negative.

**Operational outcome, separate from the science.** One process, 386
CPU-seconds, peak RSS 256 MiB; a post-hoc script of 198 CPU-seconds kept
apart and labelled; about 10 CPU-minutes in all, inside the ceiling. The
script was committed before it ran with its fixed choices in the docstring
(`41fbd34`), the JSON is hashed and gzipped after review, and the PR was
revised to lead with the constrained answer (`238355e`). No crash, no lost
work, no cut grid. A clean probe. Scheduler health: observed only as PR #157
merged 22:31:14Z and this card created 22:35:43Z; `loop.json` was not
inspected. Scientific disposition: negative for the supported machine,
positive as a requirement list for a machine that is not published.

**P-0013 `directed-relay-latent` (in flight, draft PR #159; weighed, not
reviewed).** Negative. Under the two-colour schedule the directed
construction's stationary latent error is 0.065 to 0.079 even with ideal
noiseless relays, in every feasible colouring; the binding limit is
staleness in the four-hop relay loop, not the lossy fan-in the row's
falsifier named. Under a sequential order the Z1 does not have, a thermometer
fan-in reaches 0.0052 at 8 bits. And tied copies at tie 2 reach 0.0012 on the
reduced instance, so the comparison the row was built on (tied copies fail
at finite tie) was not reproduced there (PR #159 body; probe note on the
branch). The pre-registered prediction fails on both halves. The
`asynchronous-gibbs-bias.md` card, written the same night, already said
Dobrushin's condition "should not be expected to hold" for a global latent
and that the paper's bias and mixing warnings were the applicable ones; the
probe found that bias. The debrief (PR #161) records that the row's missing
fixed choice was the update schedule. Nothing here is accepted evidence
until the owner merges it.

**PR #160, source cards: process, correct and late.** The three cards close
the debts D-0010 listed, at abstract depth and labelled so. All three say the
same thing in different words: the mechanism each row tested is published,
and the lab's contribution can only be the constrained number.

## Rubric 1: P-0012

- **Design number or probe with a study's process?** A probe with a probe's
  process (script, hashed JSON, note, lessons row, docs-only PR; no
  protocol, runner, gate or roadmap row). Its number is a conditional
  requirement, not a budget: one-way fan-in of at least 16 model sites per
  node, w of at least 0.5, bias range about 8 in the pore's field units
  against a cap of 1.99, and residency of at least 3 to 5 estimates per
  operating point (probe, "Requirement read off the relaxed arm"). Nobody
  designs against it: the hardware model prices no one-way coupling
  (`docs/z1-hardware-model.md`, "Directional endpoint couplings are
  supported", no cost), and the pore is held.
- **Constraints.** Applied: the frozen model and map, the 4- and 6-bit
  codebook, aggregator gain jitter, priced programming and adaptation
  writes. Relaxed and flagged (rule 8): one-way couplings, aggregator
  placement (k = 16 is not placeable), aggregator caps, jitter on pore sites
  (untested), read thinning (assumed independent, updates uncounted). The
  constrained arms are reported beside the relaxed ones. Rule 8 satisfied.
- **Accounting.** Stated: CPU-seconds, writes per chemical potential per
  rule, the write-to-read ratio 90.8 under the sealed profile, break-even in
  estimates, and the one-way couplings and extra nodes left unpriced because
  nothing publishes them. Rule 9 satisfied.
- **Expectations.** Held in the relaxed arm at w = 0.5 and 1; the falsifier
  fires at w = 0.25 with low gain; the constrained realization fails
  everywhere. Which rows assumed the opposite: `directed-relay-latent`
  assumed "fan-in by aggregators in their near-linear regime" and one
  aggregator per 16-site block. P-0012's placement check (k ≤ 5) and
  P-0013's own fan-in table (near-linear misses at 0.075 to 0.143) show
  both premises wrong. No done row depended on P-0012.

## Rubric 2: the queue rows

Fifteen rows at 0fdd07a: ten `done`, three `dropped`, one `held`, one
`open`. P-0012 touched none of the fourteen closed rows. Its note and the
row both say it does not unhold `native-adsorption-eligibility` (probe,
"What this probe does not establish"; row notes). The D-0006 drop of that
row stands on the structural exact 8-bit miss and the unpowered read
allocation; P-0012 adds a reason, reads are the minor term even on the
hypothetical machine.

### `directed-relay-latent` (open; P-0013 drafted, PR #159)

- **Decision.** Whether "an update schedule other than two colours" is a
  hardware requirement worth naming for a global latent carried by one-way
  relays. P-0013 answers: on the reduced instance, yes, the schedule is the
  binding fixed choice; and the requirement has no consumer, because the
  tied-copy construction already reaches on that instance and the full-size
  miss (P-0008) was never reproduced in the reduced setting. Extra phases or
  frozen-pixel clamp writes would each be a full write under the sealed
  profile (probe note, "Design consequence").
- **Cheapest probe.** Spent. D-0010's desk check (compare the needed logit
  against the relay field's 8-bit resolution) would have missed the actual
  cause, staleness; the probe's ideal-relay arm found it. The cheapest check
  that would have changed the row before it ran was the debrief's: reproduce
  the failure the row compares against on its own instance first (PR #161,
  "Control first"). Tied copies at 0.0012 would have stopped the row at the
  control.
- **Source card.** `asynchronous-gibbs-bias.md` (C1, C2: stale Gibbs is
  biased and can mix exponentially slower; C4 does not apply to a
  high-influence latent). Written after the probe ran. It would have
  predicted the sign of the result, not its size.
- **Third variant? Yes, and it ran.** The global-latent line on
  `t4_h05_q035` has now missed its gate three times under the supported
  schedule: a trained local-latent denoiser 0.078 to 0.147 above Bayes
  (`docs/experiment-reports/2026-10-16-dtm-baseline-headroom/summary.md`), a
  hand-built tied-copy latent at 0.0117 against 0.01
  (`docs/experiment-reports/2026-10-17-dtm-exact-posterior-lattice/summary.md`),
  and a directed relay latent at 0.059 to 0.079 (PR #159). Each miss has a
  different named mechanism (model class, finite tie, staleness); the kind
  of failure is the same. The owner closed the line at D-0005 and amendment
  2 keeps that closure (`PROJECT_CHARTER.md`, "Commitments"). Under the
  two-failures rule my only output is a question:

  **Question to the owner.** Does P-0013 close the global-latent line on
  `t4` for every construction, so that no fourth variant is proposed under
  any framing (a frozen-pixel or sequential-phase schedule, a 64-pixel
  staleness number, a weaker-coupling instance, relay placement for k = 16),
  with "a schedule other than two colours" recorded as a conditional
  requirement in the knowledge base and nothing more? Or does the owner want
  the schedule feature priced as a desk note (phases or clamp writes per
  sweep under the sealed profile), with no run? I advise the first. The
  second is a note, not a row.

### Is the queue pointed at the primary question?

At neither half now. The three mechanism rows the owner opened on
2026-11-12 were the architecture half under a relaxed feature; two of three
came back negative for the supported machine and the third explains a
closed result. None named a task, a user or a digital baseline
(`2026-11-12-mechanism-proposals.md`, "Where these sit"). After P-0013 the
queue holds nothing open, and the gap the agenda names, application
validity together with digital cost headroom, is where it was on
2026-10-20. The loop has done what the owner asked, cleanly and cheaply,
and the answers say the three bottlenecks it priced (reads, global latents,
gate margins) are not the thing standing between the hardware and a task.
The thing standing there is the absence of a task.

## Rubric 3: can the lab state the four-week test today?

Reported on its original terms on 2026-11-11
(`docs/research/2026-11-11-four-week-evaluation.md`; lessons 2026-11-11).
P-0012 is not a task row and changes no clause. Restated, with P-0012's
only addition in clause 2:

1. *Precision and noise tolerance.* 4 bits under `split` with 10 percent
   static jitter on the local L = 8 denoiser (P-0006); 6 bits with fields in
   J for the bias memory; 5 bits for the placed tie-limit latent under either
   rounding rule (P-0010); exactly 4 field bits under class diffusion and 9
   under nearest for the 144-site pore (P-0009, P-0010). Jitter isolated in
   three studies, static per-site gain only (evaluation, clause 1). P-0012
   adds ±10 percent static gain on an aggregator node: harmless at w = 0.5,
   fails the declared worst-block check at w = 1. Pore-site jitter is still
   untested.
2. *Sweeps, reads and writes per useful sample.* 64 sweeps per step, 128
   reads and 128 clamp loads per two-step sample on the local family; no
   useful sample on any family with a global latent; the pore priced a
   trajectory that certified nothing (evaluation, clause 2). P-0012 adds,
   for the pore, an asymptotic read count on a machine with one-way fan-in
   (188 / 393 / 250 aggregator reads per 0.01 standard error) and, on the
   supported machine, about 1,000 to 2,100 thinned site reads for the same
   precision, thinning updates uncounted. Neither is a certified estimate.
3. *Projected energy at stated quality against a conventional baseline.*
   **Negative.** 3.2e-7 J per 16-draw estimate at error 0.075 (9.7e-9 J at
   the clamp-read end) against a 3,136-MAC logistic at 0.071 for 3.1e-10 to
   3.1e-8 J; break-even only above 3.1 pJ per MAC at the favourable end of
   the band, and less accurate at any price (evaluation, clause 3).

Largest gap: clause 3 for any task with a quality pass. No row closes it
and no mechanism row can. Missing evidence, unchanged: no hardware
measurement; no published clamp-only constant (the denoiser's cost moves
33x across that band); no lower bound on digital cost; no training under
the constraints; thinned draws from one clamp load untested. P-0012 adds
one more unpriced feature to the list the lab has accumulated: the cost,
precision and placement of a one-way coupling. Every lever priced since
October lands on the write constant; the one row that targeted reads found
reads were the small term and the feature unplaceable.

## Rubric 4: representation, architecture, useful computation

**P-0012.**

- *Intended model:* the pore's Boltzmann law at each frozen chemical
  potential. *Constrained representation:* the archived 144-spin map plus
  aggregator p-bits as one-way listeners outside that law, with their own
  caps in the primary arm and the pore's caps in the secondary. *Actual
  update process:* chromatic Gibbs assumed at equilibrium; each aggregator
  drawn from its conditional on the current sites; reads thinned to
  independence by assumption, updates uncounted. *Estimator:* maximum
  likelihood for the chemical potential from read counts, mapped to loading
  (probe, "Design question, mechanism, falsifier").
- *Established, bounded.* On this pore, a one-bit aggregator over a
  contiguous 16-site block loses 22 to 39 percent of the published
  independent-site information to within-block correlation and still keeps
  5.2 to 6.0 site-reads of information per read, provided its threshold
  sits at the block median, which needs a bias range of about w·k, about
  8 at w = 0.5, and a host that adapts it. That is a representation limit of
  the kind amendment 2 priority 1 asks about: the field cap, not the
  coupling precision, is what the supported machine lacks (4 bits changes
  gains by at most 0.17). Scope: one pore, three chemical potentials, two
  block geometries, asymptotic counts.
- *Challenged.* The proposal's thesis that "coarse coefficients should be
  enough" (proposal 1, "Exploratory calculation"): true of w, false of the
  bias. And the lab's October premise that the pore's read allocation was
  the problem: P-0012 says a thinned site-read scheme needs about a quarter
  to a third of the eligibility probe's reads, so the allocation was
  wasteful as well as underpowered.
- *Unsupported generalizations to avoid.* "Aggregators save reads on Z1"
  (placement caps k at 5; caps clip the bias); "5.2 is a design number"
  (it is a requirement list for an unpublished feature); "the planar
  susceptibility case would pass" (not run, and its block correlations are
  the ferromagnet's, not the pore's); "the adapted threshold is free"
  (45 writes per point at 4 bits).
- *User and decision.* None active. The row's decision, whether to ask the
  hardware side for one-way fan-in as a read feature, has an answer: not
  on the strength of this pore, where reads are a fifth of the projected
  joule and the model is held with no user named
  (`native-adsorption-baselines.md`, "Our baseline requirements").
- *Strongest digital baseline.* The 512-state exact transfer, about 11 ms
  per point, exact (probe, "Strongest practical digital baseline"). The
  comparison is read paths inside one sampler and claims no advantage.
- *Hardware requirement.* Conditional and complete as stated: one-way
  fan-in ≥ 16, w ≥ 0.5, bias range about 8 field units, residency of 3 to
  5 estimates per point. Priceable only when Extropic publishes a one-way
  coupling's cost and placement rule.
- *Counterfactuals.* Four relaxed features, all labelled. Nothing
  established on the supported machine except the negative.

**P-0013, briefly, since it is in flight.** Intended model: the reduced
instance's exact posterior. Representation: six pixels, one latent, relays
with one-way couplings. Update process: the Z1 two-colour schedule, with a
sequential order as a relaxed diagnostic. Estimator: the latent's exact
marginal. Established, bounded: under two colours a four-hop one-way relay
loop is a stale Gibbs sampler whose stationary law misses by 0.065 or more
on a strongly coupled ambiguous latent, at any relay quality. Challenged:
the proposal's claim that one-way copies have "no domain walls, errors
accumulate with depth, not with mixing time" (true only under sequential
order), and the row's assumption that tied copies fail on the instance.
Hardware requirement: an update schedule other than two colours, unpriced.
Falsifier for anything further: none owed, because nothing further is
proposed. Not evidence until merged.

**For a next proposed question:** none is proposed, so no falsifier or
cheapest check is owed. If the owner names a task, the cheapest decisive
check is the evaluation's desk screen in its stated order (digital cost at
the array's size first, native placement second, user and tolerance third),
before any code (`2026-11-11-four-week-evaluation.md`, "Un-park when").

## The one thing to drop

**Every aggregator follow-up: the proposal note's second case (susceptibility
on the planar ferromagnet with the Kac-Ward reference), two-level aggregator
trees, a relay-tree placement that reaches k = 16, a sampled check of read
thinning, and a bank design beyond three members.** Three reasons:

1. **The trigger did not fire.** The second case was "only on a pass"
   (`2026-11-12-mechanism-proposals.md`, proposal 1). The pass is relaxed
   on placement, caps and coupling direction; on the supported machine the
   row failed its own falsifier (gain under 3). A pass that needs a machine
   the hardware model does not describe is not the pass the note meant.
2. **The mechanism is published and the constrained correction is now
   measured.** Kipnis and Duchi give the ceiling, the adaptive threshold
   and the aggregation gain (`one-bit-mean-estimation.md`, C1 to C3). The
   only thing the lab can add is the correction for a contiguous, capped,
   placed block, and on the one resident model in the archive that
   correction says no. A second instance would measure another correction
   with the same placement problem, since every block must sit within the
   16-offset reach.
3. **Reads are the small term.** On the pore writes are 44 nJ against 16 nJ
   of reads per trajectory (D-0010, analysis). A read feature cannot change
   a cost verdict that writes set, and the one task with a cost verdict,
   the L = 8 denoiser, is 99 percent clamp writes (evaluation, clause 3).

This drop is a boundary, not a row status: no queue row carries these
variants. It is written down so the next proposal note does not grow them.
D-0010's drop (further margin-budget variants on the PAsymSwap family)
stands.

## Evidence obligations

Read from `/home/user1/.hermes/profiles/thermes/loop/evidence-obligations.json`
(checked 2026-11-11T08:54Z at 682884a), reconciled against main at 0fdd07a.
The index is dated administration, not proof; this review does not edit it.

| ID | Status in index | Reconciliation at 0fdd07a |
| --- | --- | --- |
| EO-001, error-diffusion card | `closed_scoped` | Unchanged; `error-diffusion-quantization-bias.md`, "Limits". |
| EO-002, full-source check before stronger reliance | `conditional_open` | Not triggered. P-0012 does not rely on that card. |
| EO-003, unpowered read-allocation lesson | `open` | Closure evidence on main since PR #154: lessons entry 2026-10-20. P-0012's note says the same ("looks stale"). Closable. P-0012 does not reuse the 64-read allocation; its baseline is a different read scheme. |
| EO-004, independent rounding-transfer case | `closed_negative` | Unchanged; P-0010. |
| EO-005, amendment 1 four-week evaluation | `open` | Closure evidence on main: `2026-11-11-four-week-evaluation.md` (PR #153), lessons row 2026-11-11. Closable. Rubric 3 above restates it. |

Candidate debts D-0010 listed, now reconciled:

- Source cards on one-bit quantization, stale Gibbs bias and noisy-gate
  reliability: **on main** (PR #160, merged 22:01Z), abstract depth,
  labelled. Closed as scoped. Each was written after its row's probe ran;
  PR #161 proposes the rule that would have prevented that.

New candidate debts, for the owner to add or decline:

- P-0012's note on main still says the one-bit card "is not written"
  (probe, "What this probe does not establish"). It is. One line of docs
  hygiene, not a research debt.
- No card or note records the four-feature requirement list P-0012 read off
  the relaxed arm in one place with the other conditional requirements
  (clamp cost, zero-write replica temperatures, in-place ramps, margin
  budget, degree). A collected list exists in the owner's unpushed branch
  `docs/z1-feature-requirements` (seen in this worktree's prior HEAD, not
  on main, not reviewed here). If it merges, this debt closes.

## Proposed queue

No queue change.

The one open row, `directed-relay-latent`, is being closed by P-0013's own
PR, and a row changed on main since 0fdd07a keeps main's version, so any
edit here would conflict with it. The premise question goes to the owner
through this review. No new row: the queue's gap is a task with digital
headroom, which no row a director can write supplies, and the agenda
forbids filling the queue to keep the loop running. An empty queue after
P-0013 is the correct state until the owner names a task or parks the loop.

## Evidence classes

Nothing in this review is new evidence. P-0012's numbers are exploration
with `exact_reference` values, cited from its note; P-0013's are from an
unmerged draft and are weighed, not cited as accepted; the pore's read,
write and update counts are from the eligibility probe as restated in the
four-week evaluation; the joule arithmetic on them, the 46-reads
equivalence and the "about a third" figure are this review's analysis
under the sealed profile and stated assumptions, labelled as such; P-0007,
P-0008 and P-0009 figures are from recorded studies cited by path. No
hardware claim.
