# Direction review D-0012

2026-11-16. Main commit 3daa147b1111822b2f0bd6a6cf99e9bf23d99086. THERMES-DIRECTOR.
Label: advice, not a decision. The owner edits the queue; this file proposes.

Trigger: two loop results accepted since D-0011
(`docs/research/2026-11-13-direction-review-d-0011.md`, PR #162, merged
2026-11-16T19:49Z):

- P-0013 on row `directed-relay-latent`, PR #159, merged 2026-11-13T22:51Z,
  note `docs/research/2026-11-13-directed-relay-latent-probe.md`;
- P-0014 on row `z1t-system-budget`, PR #173, merged 2026-11-16T08:10Z, note
  `docs/research/2026-11-15-z1t-system-budget-probe.md`.

In flight, not touched, weighed below: P-0015 on `z1-acceptance-kit` (draft
PR #174, awaiting the owner), the owner-session amendment of that row (PR
#176) and the owner's answer to D-0011's premise question (PR #177). Read
against charter amendments 1 and 2, the agenda, the queue at 3daa147
(twenty-one rows: thirteen `done`, three `dropped`, four `held`, one `open`),
the lessons register, the roadmap, the last three merged studies (P-0009,
P-0008, P-0007), all 34 source cards and every note under `docs/research/`
dated after 2026-11-13. Most of those notes are owner-session work outside
the loop (the QEC record, the Z1T-0 noise runs and their closing note, the
Z1 feature list); they are read as context and get one paragraph each near
the end.

## The change I would make first

1. **Before P-0015 is redrafted, put the simplest exact test beside it:
   disconnected pairs and single sites.** The row's four defects (rounded
   couplings, one missing coupling, one stuck p-bit, per-site beta variation)
   are all defects of one coupler or one site. Each has a closed-form
   reference on an isolated pair or site, with no Kac-Ward, no start state
   and no mixing question. P-0015's own probe says the collective test
   cannot see about 60 percent of single missing couplings within 10,000
   reads and has no exact reference at all for beta variation (PR #174
   body; probe note on the branch, findings 5 and 7). Rough arithmetic for
   the pair test is in Rubric 2. It is this review's estimate and is the
   check proposed, not a result. No queue change for this row: PR #176
   proposes its amendment, and a row changed on main keeps main's version.
2. **Queue: amend `z1t-sample-budget-map` before the owner runs it.** P-0014
   prices a sample at 13 fJ and a read bit at 1.692 pJ, so "samples per
   token", the map's unit, is the smallest priced term. Report each grid
   point also in read bits and serial sampling time, and set the same
   network run digitally beside the result. No configuration is added.
3. **Queue: drop `z1t-sampled-head`** (below, "The one thing to drop").

No new row. After these, the queue holds one open row whose draft was not
approved and the owner-run Z1T rows on one released checkpoint. None names
a task with digital headroom, and each row's notes say so.

## Verdicts

**P-0013 `directed-relay-latent`: negative; exact; a probe with a probe's
process; it answers the row.** D-0011 weighed it in flight and nothing in
the merged note changes that reading. Under the two-colour schedule the
directed construction's stationary latent error is 0.065 to 0.079 with
ideal relays in every feasible colouring, and 0.059 at 8 bits with finite
relays. Staleness in the four-hop relay loop binds, not the lossy fan-in
the row's falsifier named. Under a sequential order the Z1 does not offer,
thermometer fan-in reaches 0.0052 at 8 bits with 8 extra sites. Tied copies
at tie 2 reach 0.0012 on the same instance (note, "Finding"; lessons
2026-11-13). Four things to read with it:

- **Both halves of the pre-registered prediction failed, and the falsifier
  fired with the wrong cause.** The row predicted the directed construction
  would reach 0.01 "where tied copies at finite tie do not", and named lossy
  fan-in if it missed (row notes). Tied copies reach; the directed
  construction misses with noiseless relays (note, "Results").
- **The row's comparison was never reproduced.** The reduced instance does
  not show P-0008's finite-tie miss, so the row compared against a failure
  that does not exist there. `docs/research-loop.md` now has the rule that
  would have stopped it at the control ("Control first", added 2026-11-12).
- **The range mixes converged and capped cells.** The 0.0651 cells converged
  in 343 to 380 sweeps. The 0.0708 colourings and both r = 2 cells (0.0793)
  stopped at the 400-sweep cap with their last change unlogged (note, "Fixed
  choices", convergence row). Agreement across colourings to 1e-9 shows the
  cells are the same chain, not that the chain converged. The negative
  stands on 0.0651 alone; quote that number, not the range.
- **It was predictable in sign from a card written the same night.**
  `docs/knowledge/sources/asynchronous-gibbs-bias.md` (C1, C2; "Those
  conditions point the wrong way for a global latent"). The card gives the
  sign, not the size, and it is at abstract depth.

The owner has answered D-0011's question: the global-latent line on `t4` is
closed for every construction, with "a schedule other than two colours"
kept as a conditional requirement and nothing more (PR #177, open, not on
main at 3daa147; weighed, not cited as accepted). I ask nothing further on
that line.

**P-0014 `z1t-system-budget`: an audit that answers the row; counts exact,
joules `calibrated_projection`; the reproduce-first gate half-failed and the
owner chose to proceed; negative for the published design as a system under
the sealed profile.** The post's 8.74 nJ of Z1 sampling reproduces as one
13 fJ draw per sampled output per step. Its 285.78 nJ FPGA figure does not
reproduce from its stated prices (0 of 2,048 accountings) and is unused.
At the post's own inferred read count, reads at the sealed constant cost
138.6 nJ, 16 times the sampling. The dense readout is 25.8M
multiply-accumulates (MACs) at width 512 and 617.6M at the released 12,288.
The released model needs 9 to 27 chips with serial samples (note, "Answer in
brief"; lessons 2026-11-16). Four things to read with it:

- **"The readout dominates every total" is wider than the table.** The
  queue row and the note's fixed-choices table say the readout dominates
  every total at 1 and 10 pJ per MAC. The note's own totals at 1 pJ put the
  readout at 0.618 of 0.625 mJ at the low end for Z1T-0 at N = 32, but at
  0.618 of 1.70 mJ at the high end, and at 0.618 of 4.62 mJ at the high end
  at N = 2,048 (note, "Projected joules per token"). At the high end the Z1
  side is larger than the readout at 1 pJ at every N. At N = 2,048 the
  co-processor's averaging (127M to 1.03G operations) is 0.2 to 1.7 times
  the readout at any MAC price. The supported statement: the readout is the
  largest term at the low end of the band. The lessons header warns that
  summaries are where scope gets lost.
- **The N = 32 budget prices a model that does not work.** The only loss
  under noise in the repository is the 12,288-wide checkpoint's: 10.746
  nats at N = 32 against 4.002 noise-free, and 4.916 at 2,048
  (`docs/research/2026-11-15-z1t-noise-retrofit-closing.md`, finding 1).
  The budget nearer a usable model is the N = 2,048 one, at 64 times the
  post's sampling time. The 512-wide configuration the post prices has no
  loss under noise anywhere in the repository.
- **The low end is a lower bound, not an operating point.** The 5 read
  steps and 8 write steps are borrowed from the post's latency and "are not
  a feasible schedule"; chips are packing bounds at 100 percent; the shared
  placement ignores the degree-16 limit (note, "Open assumptions"; "What
  this probe did not do").
- **The baseline the row left out is in its own last section.** The low end
  of the Z1-side band equals the same sparse body run digitally at 2.44 pJ
  per MAC at N = 32 and at 131 pJ per MAC at N = 2,048 (note, "Proposal
  frame"). The queue's digital band is 0.1 to 10 pJ. So at the sample count
  nearer a usable model, the Z1 side's best case is 13 times the top of the
  band, for a worse loss. That is the number an outside reader should take
  away, and the note does not lead with it. Rubric 3 uses it.

Is it already known? In kind, yes, twice over. That a rate-coded value needs
about ε⁻² events is on two cards
(`docs/knowledge/sources/one-bit-mean-estimation.md`, C4;
`docs/knowledge/sources/normal-cn101.md`, "Method", where a fabricated chip
ran 500,000 cycles per tile). That I/O sets the projected joule is the
lab's own repeated result (lessons 2026-10-14, 2026-10-07;
`docs/research/2026-11-13-z1-feature-requirements.md`, feature 1). The
counts for this design are new, and they are the contribution.

## Rubric 1

**P-0013.**

- **Design number or probe with a study's process?** A probe with a probe's
  process (script, logs, JSON, note, lessons row, docs-only PR). Its number
  is a conditional requirement, an update schedule other than two colours,
  recorded as feature 7 of the feature list. Nobody designs against it: the
  schedule is unpriced, and tied copies already reach on the instance.
- **Constraints.** Applied: the two-colour schedule in the primary arm, the
  8-bit `split` codebook. Relaxed and flagged (rule 8): one-way couplings,
  with the symmetric codebook assumed to apply to them; a sequential order
  as a diagnostic; no placement; a 2 × 3 instance with every pixel coupled
  to the latent. Rule 8 satisfied.
- **Accounting.** Stated: extra sites, programming writes, sweeps to reach
  from two starts, 27 CPU-minutes. One-way couplings are unpriced because
  nothing publishes them. Rule 9 satisfied.
- **Expectations.** Missed on both halves. Rows that assumed the opposite:
  none open. The row's own target text assumed "aggregators in their
  near-linear regime", which its table puts at 0.075 to 0.143.

**P-0014.**

- **Design number or probe with a study's process?** A probe with a probe's
  process (script, hashed JSON, note, lessons row; under a second of CPU).
  It produces numbers of the kind amendment 1 names, an I/O budget and a
  projected cost: p-bits, chips, samples, read bits, node writes and
  off-chip MACs per token against N, and the break-even MAC prices above.
  They are for an outside reader checking the post; the lab has no design
  that consumes them.
- **Constraints.** It prices the post's design as published and relaxes
  what the post leaves open, each kept as a band: placement (shared
  registers against copies), input encoding (clamped against sampled),
  which values cross, clamp price. It applies no quantization or jitter and
  makes no quality claim beyond the quoted output-noise losses. Stated
  (note, "Open assumptions"). Rule 8 satisfied in the note; the one-line
  summaries drop the band.
- **Accounting.** Stated in full, with included and excluded costs listed
  (static power, links, weight traffic, one-time programming, idle-layer
  clocking). Rule 9 satisfied.
- **Expectations.** The row's precondition was to reproduce the post's
  total and its split. The Z1 side passed and the FPGA split failed. The
  proposer stopped and reported, as the row required, and the owner chose
  option 1 on 2026-11-16 (note, opening). Rows that assumed the opposite of
  what it found:
  - `z1t-sample-budget-map` counts its budget in samples, the smallest
    priced term, and its notes say it "feeds" a row that is now done.
  - `z1t-full-constraints` applies "the input encoding as the post describes
    it"; P-0014 had to keep that encoding as a band because the post leaves
    it open.
  - `z1t-sampled-head` assumes the readout is the term to attack. P-0014
    confirms its size and shows it sits on both sides of the post's
    comparison.

## Rubric 2: the queue rows

### `z1-acceptance-kit` (open; P-0015 drafted, PR #174; row amendment PR #176)

- **Decision.** Which test a person with a chip would run to accept an
  array, and how many reads and writes it takes. The lab has no chip, and
  the row claims nothing about one. The number has a consumer only if
  someone outside takes it.
- **Cheapest probe that would make the study unnecessary.** A desk
  calculation of the disconnected test, per declared defect, in reads and
  programming writes:
  - *Missing coupling.* On an isolated pair the agreement probability is
    sigmoid(2βJ) with the coupler and 1/2 without. At βJ = 1 that is 0.88
    against 0.5, about 13 reads at 5 percent and 5 percent, for every edge
    of a matching at once. P-0015's collective statistic needs 7 to over
    10^9 reads by edge, and reaches about 40 percent of edges within 10,000.
  - *Stuck p-bit.* An isolated unbiased site is Bernoulli(1/2) on every
    read. P-0015's 5 to 16 reads are the same test, but each of its reads is
    a fresh chain and costs an initialization write, because the ordered
    array never flips at beta 4. An isolated site needs no re-initialization.
  - *Per-site beta variation.* An isolated site with bias h reads +1 with
    probability sigmoid(2β_i h), an exact reference. At 2βh = 1, a 10
    percent gain error moves it from 0.731 to 0.750, about 5,700 reads per
    site; 30 percent, about 660. Both are inside the row's 10,000 limit.
    P-0015 had to exclude this defect, because the collective law with site
    gains is not an Ising law (fit residual 4.3e-3).
  - *Rounded couplings.* Open. A collective statistic pools every edge and
    may need fewer reads for a uniform rounding; not estimated here.
  - *Writes.* A matching covers one sixteenth of a patch's couplers per
    programming; a maximal planar subgraph covers at most three eighths
    (3N of 8N). So pairs cost more programmings per coupler covered and the
    planar test fewer. The row headlines both reads and writes, so both go
    in the comparison.

  These figures are this review's normal-approximation arithmetic on
  two-spin formulas. They are not computed in any note. If they hold, the
  collective Kac-Ward test is the wrong instrument for three of the row's
  four defects.
- **What the collective test can see that a pair test cannot** is the
  update process: stale neighbour reads, a wrong colour order, updates that
  are not simultaneous. P-0013 is the lab's own example of a schedule
  defect that moves a stationary law by 0.065 with every coupling correct.
  None of these is in the row's defect list. Adding one is a change to the
  row, which is the owner's; I do not propose it as a row.
- **Source card.** The row names `extropic-thermalizers.md`, which supports
  only the kernel convention (PR #174 body). Three cards bear on it and are
  not cited: `extropic-torx-paper.md` (C9 and "Cautions": Extropic already
  calibrates single X0 p-bits, to about 5 percent), `asynchronous-gibbs-bias.md`
  (C1) and `aadit-million-pbits.md` (C3) for process defects. No card
  covers how many samples it takes to test an Ising model against a
  reference. That literature exists (for example Daskalakis, Dikkala and
  Kamath, "Testing Ising Models", arXiv:1612.03147; not read for this
  review, and nothing is claimed from it). A card is owed before the row
  closes (queue header, second standing rule).
- **Third variant?** No. Two cautions. This is the fourth study on planar
  Ising graphs (PR #92, P-0001, planar annealing), and PR #176 has the probe
  scan beta for the fewest sweeps per test on nine graphs. That scan is a
  sampler comparison on random instances; amendment 1 keeps those at probe
  level, which is where it is. And P-0015's draft sized its reads as
  independent; the review probe found the variance of a read mean up to 76
  times the independent value at beta 4 (PR #176 body). Lessons 2026-10-20
  already says to size a sampled allocation against the exact variance
  before running it; correlated reads make that size optimistic.

### `z1t-sample-budget-map` (held, owner-run)

- **Decision.** At which sample counts the released model is within 0.1 and
  0.5 nats of its noise-free loss, and whether splitting the count between
  internal and residual-writing sites saves anything. With P-0014's counts
  that becomes the first statement of reads and projected joules per token
  at a stated loss for a published design.
- **Cheapest probe.** For the uniform arm, the four points already on main.
  The excess over noise-free is 2.90 nats at N = 512 and 0.91 at 2,048
  (closing note, table), a factor of 3.2 per fourfold N. On that slope the
  uniform arm reaches about 0.3 at 8,192: inside 0.5 and outside 0.1
  (extrapolation, this review's). Only the allocation arm is new
  information.
- **Source card.** `one-bit-mean-estimation.md` (C4) and `normal-cn101.md`
  give the ε⁻² law the uniform arm will trace. The row cites neither.
- **Third variant?** No. It is evaluation only, and the closing note keeps
  it apart from the closed retrofit line ("What stays open").
- **Change.** Report in P-0014's units (Proposed queue).

### `z1t-full-constraints` (held on two conditions)

- **Decision.** Whether the released model survives its published
  constraints together. It can only move a loss that is already 6.7 nats
  off at the post's N and 0.91 off at 2,048. It changes a reading only if
  the map finds a usable budget first, which is its first hold condition.
- **Cheapest probe.** The map. If no tested budget meets 0.5 nats, this row
  has nothing to check.
- **Source card.** `extropic-z1t.md` (C8) describes dy4p; C9 lists what the
  post leaves unstated. P-0014 shows the encoding's semantics are open
  (clamped code or sampled p-bits). The row must pick one before the run
  and say so, as its target text already requires.
- **Third variant?** No.

### `z1t-sampled-head` (held, owner-run)

- **Decision.** None on the hardware side. See "The one thing to drop".
- **Cheapest probe.** A source card on hierarchical and adaptive softmax
  heads. The proposal note quotes "the published hierarchical-softmax
  results" and a text8 figure for a naive tree with no card behind either
  (`docs/research/2026-11-14-z1t-sampled-head-proposal.md`, "Exploration",
  "Reading rule"). The technique dates from Morin and Bengio (2005; not
  read for this review). The exploration number is already on main: +0.62
  nats at 246 times fewer MACs, "large" by the row's own rule.
- **Third variant?** No.

### `native-adsorption-eligibility` (held; D-0006's drop in its notes)

Unchanged. Neither result touches it.

### Closed rows

Thirteen `done` and three `dropped`. P-0013 closed `directed-relay-latent`;
P-0014 closed `z1t-system-budget`. `z1t-noise-recovery` (done, owner-run)
supplies the loss table P-0014 quotes. No other closed row depends on
either result.

### Lines with two or more same-way negatives

- **Global latent on `t4`** (P-0007, P-0008, P-0013): closed by the owner
  for every construction (PR #177, pending merge).
- **Z1T-0 noise retrofit** (three fine-tunes): closed by the owner on
  2026-11-15 with three named reopen conditions (closing note, "What would
  reopen it").
- **Rounding rules beyond nearest.** P-0010 (class diffusion does not
  transfer) and the owner-run QEC decision-aware rule (ties nearest at 3
  bits; `docs/research/2026-11-14-qec-research-record.md`, first table row)
  are two in a row. No variant is queued. A loss-aware rule on another
  model would be the third; it gets a premise question first, not a row.

No line has a third variant open, so this review puts no premise question.

### Is the queue pointed at the primary question?

At the architecture half, by audit, and at no task. The Z1 rows the owner
opened on 2026-11-15 price and test a published design. That is amendment
2's "use existing results and literature first" applied to Extropic's own
post, and P-0014 shows it is cheap and productive. It is not the un-park
condition the four-week evaluation named (digital headroom, native
placement, a user and a tolerance;
`docs/research/2026-11-11-four-week-evaluation.md`, "Un-park when").
P-0014 applied the first screen to Z1T and it fails: at the sample count
nearer a usable model, the same body run digitally is cheaper across the
whole MAC band. The owner-run QEC work screened a
second candidate family from the 2026-10-20 memo and found no decoder
advantage over strong conventional decoders and no mapping to the hardware
(QEC record, "Current answer"; "What remains unresolved", item 5).

**The decision I would put to the owner, outside the two-failures rule:**
who receives the Z1T audit and the acceptance kit? The four-week evaluation
said a hardware team that will design against the numbers makes continuing
worth its cost. If there is a channel, P-0014's list of missing constants
(FPGA operation counts, static power, readout time) is the first thing to
send, and the kit should be shaped by what that reader can run. If there is
none, both are numbers nobody designs against, and the evaluation's parking
advice stands unanswered.

## Rubric 3: can the lab state the four-week test today?

Reported on its original terms on 2026-11-11 (four-week evaluation; lessons
2026-11-11): the numbers exist for one task and are negative for the
hardware. P-0013 is not a task row and changes no clause. P-0014 and the
owner-run Z1T notes add a second, exploration-grade statement on a
published design. Restated, additions marked:

1. *Precision and noise tolerance.* 4 bits under `split` with 10 percent
   static jitter on the local L = 8 denoiser; 6 bits with fields in J for
   the bias memory; 5 bits for the placed tie-limit latent; 4 field bits
   under class diffusion and 9 under nearest for the pore (evaluation,
   clause 1). **Added:** a sampling-noise tolerance for Z1T-0, output noise
   only: unusable at 32 samples per unit (10.746 nats against 4.002), 0.91
   nats off at 2,048; a 2,000-update retrofit recovers 70 percent of the
   gap at 32 and costs 1.42 nats noise-free (closing note;
   `software_simulation`, exploration, one seed).
2. *Sweeps, reads and writes per useful sample.* 64 sweeps per step, 128
   reads and 128 clamp loads per two-step sample on the local family; no
   useful sample on any family with a global latent (evaluation, clause 2).
   **Added,** per Z1T-0 token: 16.1M samples, 1.97M to 16.1M read bits and
   0.39M to 6.73M node writes at N = 32; 1.03G samples and 126M to 1.03G
   read bits at N = 2,048 (P-0014, "Counts against N"). No token within 0.5
   nats of noise-free has been shown at any tested N, so there is still no
   "useful sample" to count on this task.
3. *Projected energy at stated quality against a conventional baseline.*
   **Negative** on the denoiser: 3.2e-7 J per 16-draw estimate at error
   0.075 against a 3,136-MAC logistic at 0.071 for 3.1e-10 to 3.1e-8 J
   (evaluation, clause 3). **Added, and negative again:** Z1T-0 at
   N = 2,048 and 4.916 nats projects to 0.22 to 2.97 mJ on the Z1 side,
   before averaging and the readout. The same sparse body run digitally at
   4.002 nats is 1.68M MACs, 0.17 to 17 µJ across the band; the readout is
   common to both. Break-even is 131 pJ per MAC at the low end (P-0014,
   "Proposal frame"). At N = 32 the break-even is 2.44 pJ, for a model at
   10.7 nats.

Two pieces of arithmetic on the post's own constants, analysis and not
evidence:

- P-0014 re-derives the post's GPU figure as two operations on each of
  11.55M body parameters ("What reproduces"); the readout is 25.7M
  weights. Priced alike on both sides, the readout is 69
  percent of the GPU's per-token arithmetic, so no body, however cheap,
  improves the per-token total by more than 1.45 times while the readout
  stays dense and off the chip.
- Against the H100 body at 10 percent MFU (40.9 µJ, source card C3), the
  Z1 side at width 512 is 0.171 to 44.4 µJ at N = 32 and 9.2 to 124 µJ at
  N = 2,048. The post's 139 times becomes 0.9 to 239 times at N = 32 and
  0.3 to 4.4 times at N = 2,048, before any co-processor work.

Largest gap: clause 3 for any task with a quality pass. No row closes it
with a positive. `z1t-sample-budget-map`, re-priced with P-0014's counts,
closes clauses 2 and 3 for Z1T-0 at a stated tolerance; expect a negative.
Missing evidence, unchanged: no hardware measurement; no clamp-only
constant; no lower bound on digital cost; no training under the
constraints. Added by P-0014: the FPGA's operation counts and static-power
treatment; whether the sealed read constant is the right price for the
post's 25.6 Gbit/s link; any loss under noise for the 512-wide model; any
loss with input encoding and placement applied.

## Rubric 4: representation, architecture, useful computation

**P-0014.**

- *Intended model:* Z1T-0's deterministic tanh-linear network, scored by
  next-token loss. *Constrained representation:* each real value as the
  mean of N one-bit draws of an output p-bit, with inputs in dy4p registers
  that use all 16 couplings of a node. *Actual update process:* N serial
  draws per layer over 25 sequential layers, inputs clamped or sampled
  (open). *Estimator:* the sample mean, formed off the chip.
- *Established, bounded.* Under the sealed profile a read bit costs 130
  times a sample (1.692 pJ against 13 fJ). A rate code that sends every
  sample bit across the boundary is therefore read-bound: at the post's own
  inferred crossing count reads are 16 times sampling, and at N = 2,048
  reads set both ends of the Z1 band. Scope: one published design, sealed
  constants, serial samples, counts only.
- *Challenged.* The post's "over 100x" as a system statement (source card
  C3 already notes it holds at one MFU point with the readout excluded).
  And the lab's own unit for the Z1T rows: samples.
- *Unsupported generalizations to avoid.* "A Z1T token costs X" as a point:
  the Z1 band is 260 times wide at N = 32. "The readout dominates" without
  saying which end of the band. "9 to 27 chips" as a count: it is a packing
  bound. "The post is wrong": a figure that cannot be re-derived from
  stated constants is a missing constant. Any quality statement for the
  512-wide model.
- *User and decision.* An outside reader of the post, and Extropic if a
  channel exists. The decision: do not read 8.74 nJ, or 294.52 nJ, as the
  cost of a token.
- *Strongest digital baseline.* Not in the row. It is the same sparse body
  on a digital MAC engine (70,208 MACs at width 512, 1.68M at 12,288), then
  the H100 at the post's constants. Both are used in Rubric 3. Neither
  counts digital memory traffic for sparse gathers, so the break-evens are
  one-sided.
- *Hardware requirement.* Numerical and conditional: an on-chip counter
  per output, read as ceil(log2(N + 1)) bits, pays off on reads alone if an
  increment costs less than about one node read, 1.69 pJ (note,
  "Counterfactual feature").
- *Counterfactuals.* The counter, gated near-zero units and replicated
  p-bits are all labelled hypothetical in the note. None is on the
  supported machine, and none is a row.

**P-0013.**

- *Intended model:* the reduced instance's exact posterior. *Constrained
  representation:* six pixels, one latent and relays joined by one-way
  couplings, 8-bit `split`. *Actual update process:* the two-colour
  schedule; sequential order as a relaxed diagnostic. *Estimator:* the
  latent's exact marginal under the enumerated stationary law.
- *Established, bounded.* Under two colours, a four-hop one-way relay loop
  is a stale sampler whose stationary law misses by 0.065 on a strongly
  coupled, ambiguous latent, at any relay quality. One instance.
- *Challenged.* The proposal's "errors accumulate with depth, not with
  mixing time", which holds only under a sequential order; and the row's
  premise that tied copies fail on the instance.
- *Unsupported generalizations to avoid.* A full-size staleness figure (not
  computed); "relays cannot carry a latent" (they do under a sequential
  order); "0.079" as a converged value.
- *Strongest baseline.* On the instance, the tied-copy construction on the
  supported machine: 0.0012 with one extra site. It wins.
- *Hardware requirement.* A schedule other than two colours, unpriced;
  extra phases or clamp writes would each be a full write under the sealed
  profile (note, "Design consequence").

**For a next proposed question:** none is proposed. For the advice on
`z1-acceptance-kit`, the falsifier is a declared defect for which the
disconnected test needs more reads and more programming writes than the
collective one. The cheapest decisive check is the closed-form binomial
calculation in Rubric 2: minutes, no Kac-Ward and no sampling.

## Owner-session notes since 2026-11-13, read as context

- **Z1T-0 noise retrofit** (`2026-11-15-z1t-noise-finetune-results.md`,
  `2026-11-15-z1t-noise-quality-followup-results.md`, closing note).
  Negative within its budget: three routes, partial recovery each, 1.4 to
  2.4 nats of noise-free loss. Pre-registered reading rules, a matched
  control and a pairing check; the first runner lost its intermediate
  checkpoints (lessons 2026-11-14). The finding that survives every caveat
  is the evaluation-only one: the released model at 32 samples per unit is
  at 10.7 nats.
- **QEC record** (`2026-11-14-qec-research-record.md`). Negative
  throughout: no decoder advantage over Tesseract, confidence gates
  rejected, no mapping to the hardware. Its two largest batches charged
  3,638 and 7,810 CPU-seconds. It adds no row and says so.
- **Z1 feature list** (`2026-11-13-z1-feature-requirements.md`). D-0011
  listed it as a missing collected note; it is on main, with rows 9 and 10
  for Z1T. P-0014's counter is not in it yet.
- **Loop debrief** (`2026-11-12-loop-debrief-mechanism-rows.md`). Its three
  probe rules are now in `docs/research-loop.md`. P-0014 followed "control
  first" (it stopped at the reproduction gate). Whether the Hermes card
  templates carry the rules was not inspected.

## The one thing to drop

**`z1t-sampled-head`.** Three reasons:

1. **Its answer changes no hardware or direction decision.** The row's own
   notes: hierarchical heads "are a known digital technique and are as
   cheap on a digital chip", nothing is priced, and every hardware
   constraint is relaxed, placement included, because a 12,288-wide logit
   does not fit a degree-16 node. The p-bit's part is one Bernoulli draw
   from a logit computed digitally.
2. **P-0014 shows where the readout sits.** It is on both sides of the
   post's comparison (source card `extropic-z1t.md`, C3). A cheaper head
   helps the GPU as much as Z1T. It matters to the lab only if the body
   beneath it has headroom, and P-0014's break-even says it does not at a
   usable sample count.
3. **It is the most expensive held row and the technique is published.**
   About 32 GB of cached states and one to three GPU-hours (proposal note,
   "Run steps"), to put a number on a 2005 method for one frozen
   checkpoint. The exploration already reads "large" by the row's rule.

Reopen condition, written into the row: the map, priced with P-0014's
counts, leaves the Z1 body cheaper than the same body run digitally at a
stated tolerance. D-0010's drop (margin-budget variants) and D-0011's
(aggregator follow-ups) stand. Also not to grow: a row for P-0014's
counter, gated units or replica placements. Each is one line in the
feature list.

## Evidence obligations

Read from `/home/user1/.hermes/profiles/thermes/loop/evidence-obligations.json`
(checked 2026-11-11T08:54Z at 682884a), reconciled against main at 3daa147.
The index is dated administration, not proof; this review does not edit it.

| ID | Source | Accountable role | Due trigger | Status in index | Reconciliation at 3daa147 |
| --- | --- | --- | --- | --- | --- |
| EO-001, error-diffusion card | D-0008 verdicts | proposer; reviewer checks scope | before a protocol using the rounding explanation | `closed_scoped` | Unchanged. Card on main at abstract depth. |
| EO-002, full-source check | the card's "Limits" | next proposer relying on it | before a stronger claim than the abstracts support | `conditional_open` | Not triggered by P-0013 or P-0014. Closure evidence: none, none due. |
| EO-003, read-allocation lesson | D-0008 verdicts | director names; docs author writes | next docs update, before a proposal reuses the allocation | `open` | Stale. Closure evidence on main: lessons 2026-10-20. PR #174's body still calls it missing. The lesson applies to P-0015: its draft sized reads as independent. |
| EO-004, independent rounding case | D-0008 | proposer; owner accepts | before generalizing P-0009 | `closed_negative` | Unchanged; P-0010. The QEC rule is a second negative. |
| EO-005, four-week evaluation | agenda, "November 11" | director supplies; owner decides | 2026-11-11 | `open` | Stale. Closure evidence on main: the four-week evaluation and lessons 2026-11-11. The owner's direction since (rows of 2026-11-12 and 2026-11-15) is recorded in queue notes only. |

Candidate debts for the owner to add or decline:

- **`extropic-z1t.md` lags P-0014.** The probe read the raw page; the
  card's last check was an automated summary. Two facts are missing: the
  trace figure's source calls its split "illustrative", and the 8.74 nJ
  prices output draws only (P-0014, "Evidence obligations"). Role: docs
  author. Trigger: before any note quotes C3's split. Evidence: missing.
- **`z1-acceptance-kit` owes a card** on sample counts for testing an Ising
  model, and should cite the three cards named in Rubric 2. Role: proposer.
  Trigger: before the row closes. Evidence: missing.
- **`z1t-sampled-head` owes a card** on hierarchical softmax if it is kept.
  Role: whoever runs it. Trigger: before the run. Evidence: missing.
- **Two notes on main say a card "is not written" that is**: P-0013's note
  ("Not done") and P-0012's. Docs hygiene.
- **The owner's answer to D-0011** is recorded only in PR #177. It closes
  when that PR merges.
- **A source card's depth is not validation.** `asynchronous-gibbs-bias.md`,
  `one-bit-mean-estimation.md` and `noisy-gate-reliability.md` are at
  abstract depth and say so. This review cites them for direction only.

## Operational outcome and scheduler health

Kept apart from the science above.

- **P-0013.** Operational success with losses: about 27 CPU-minutes over
  five invocations against a 30-minute ceiling; one tool timeout before
  anything was written, one crash on a same-colour loop, cells trimmed and
  reused from logs (note, "Controls and integrity"). Scientific
  disposition: negative. The "save as you go" rule postdates it.
- **P-0014.** Operational success: the probe stopped at its gate, the owner
  answered, the budget followed; closed-form, under a second.
  Scientific disposition: the row is answered; the system reading is
  negative under the sealed profile; one precondition is unmet and
  recorded.
- **Scheduler.** Observed only: this card was created at 19:50:38Z, 56
  seconds after PR #162 merged. D-0011 stayed open from 2026-11-13 to
  2026-11-16, so one director card covers two results, as the one-card
  limit intends. `loop.json`, the event log and the card templates were not
  inspected; scheduler health beyond that is unknown.

## Proposed queue

Two rows changed, none added, none moved.

- **Hold, with an amended note: `z1t-sample-budget-map`.** Report each grid
  point in read bits and serial sampling time as well as samples, and set
  the same network run digitally beside the result. It informs what a token
  costs in the terms P-0014 found binding. No configuration is added.
- **Drop: `z1t-sampled-head`.** Status `dropped`, with the reason and the
  reopen condition appended to its notes. It is already `held`; under the
  gatekeeper's rule it comes out `held` unless the owner adopts the drop.

`z1-acceptance-kit` is copied unchanged: PR #176 proposes its amendment, and
a row changed on main keeps main's version. The advice on it is in Rubric 2.

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
| `rounding-rule-transfer` | done | Does class error diffusion move an archived exact bit requirement on a model it was not developed on, and does its mechanism (coherent rounding across a large class of equal nominal coefficients) predict where it helps? | P-0008's archived placed tie-limit model of `t4_h05_q035`'s posterior (bias-design template units as tied copies on the carried layout at w*, the 1,000 archived test images, exact enumeration over hidden configurations times the transfer matrix) with its archived `split` caps; bits 4, 5, 6 and 8; two rules fixed before any result: nearest (the archived rule, reproduced first as a control at 4, 6 and 8 bits) and P-0009's `diffuse_class_family` extended to every coefficient class, couplings included, with the class rule stated in the protocol; no sampling, no jitter, no training, no new layout | Exact (`exact_reference`) MPM pixel error against the Bayes floor per rule and bit count, with class sizes and the coherent rounding pattern at 4 bits reported; the design number is the smallest bit count at which the placed tie-limit model is within Bayes + 0.01 under class diffusion, against the archived 6 under nearest (4 bits fails at 0.0752), or no change | Proposed by D-0008 as the independent case amendment 2 requires before class diffusion is called a general lever: P-0009 read the rule on one pore whose rounding errors were all field errors in a few large classes. Informs one decision: whether `thermo_lab.constraints.quantize` keeps nearest rounding as its default and whether the archived bit requirements (am-coupling-bits 6/10/12 by cap, P-0006 4, P-0008 8/6) are re-read under the new rule. Pre-registered prediction from the mechanism: the 4-bit miss comes from coefficients that round coherently in large classes (the probe reports which), so class diffusion should pass at 4 or 5 bits; the falsifier is a 4-bit gap still above 0.01, which scopes P-0009 to field-dominated models with few classes. Replaces the am-coupling-bits transfer check D-0006 named, whose `split` requirement is an interval-width result that a mean-cancelling rule cannot be expected to move; the probe counts that archive's class sizes instead. Not a denoising study and cannot reopen that line: it re-reads an archived exact computation, samples, trains and prices nothing, and P-0008's stop fired on finite tie strength at 8 bits where precision was not binding. Not a task row, so the per-sample I/O and MAC rules do not apply. Probe under 30 CPU-minutes, exact only, hash-checked imports from the P-0008 archive, no edits to hash-bound sources; if it answers the question it closes the row with a docs-only PR. A source card on error diffusion and quantization bias correction is owed before any protocol. The proposer may add the adsorption pore at 2 and 3 bits as exploration, because P-0009's 4 is its grid floor. Answered by the P-0010 probe: negative for transfer; both rules need 5 bits on P-0008's model ([probe note](research/2026-11-05-rounding-rule-transfer-probe.md)). |
| `margin-budget-copies` | done | Does the best leakage product L00·L01 of the five-spin PAsymSwap kernel family follow e^(−4cr) when each input bit has r clamped copies, as the fixed-hidden-state Lipschitz argument predicts? | The nine-parameter kernel of `docs/research/cap_probe.py`, extended so each input bit has r in {1, 2, 4} clamped copies with every per-copy coupling capped at c; caps 1 to 4; equilibrium and K = 4; one separate arm at r = 1 with a single input-hidden coupling; no output-output coupling | Exact (`exact_reference`) best product found per cap and r against e^(−4cr), with the multi-start search and its resolution declared first; the design number is the smallest c·r at which the uniform-visitation 500-operation survival bound reaches 95 percent (predicted c·r of about 4), or the statement that the law fails | Owner requested 2026-11-12; proposal 3 of the [proposal note](research/2026-11-12-mechanism-proposals.md). A zero-sample theory check that explains a closed result. It does not reopen the conservation line or the parked Thermalizers work, proposes no compiler, pilot or sampled study, and a pass authorizes none. Pre-registered prediction: product within optimizer resolution of e^(−4cr) at every cap and r; falsifier: a deviation of more than 10 times in either direction. Reproduce the archived r = 1 table (caps 2 to 4) first as a control. The proof sketch holds at a fixed hidden state; say whether the hidden-reweighting gap was closed analytically or only checked numerically. Local multi-start search gives upper bounds on the minimum, not certificates. Count each clamped copy as a full write under the sealed profile. Not a task row, so the per-sample I/O and MAC rules do not apply. A source card on noisy-gate reliability bounds (von Neumann 1956; Evans and Schulman) is owed before any protocol. Probe under 30 CPU-minutes, exact only, no edits to hash-bound sources; if it answers the question it closes the row with a docs-only PR. Answered by the P-0011 probe: the law holds (best product 0.63 to 2.44 times e^(−4cr) in all 24 cells; equilibrium floor proved for this family); 95 percent survival needs c·r ≥ 4.29 by the bound, about 4.46 for the kernels found ([probe note](research/2026-11-13-margin-budget-copies-probe.md)). |
| `aggregator-read-gain` | done | Does reading a one-way aggregator p-bit over k = 16 sites need at least 3 times fewer node reads than sparse thinned site reads to estimate loading to 0.01 on a resident model, once within-block correlations are included? | The frozen `native-adsorption-eligibility` pore (open 3×3×16 lattice gas, its three frozen chemical potentials, archived spin map) with aggregator p-bits added as one-way listeners in the exact transfer-matrix calculation; aggregator coupling w in {0.25, 0.5, 1}; a declared threshold rule (fixed bank or host-adapted bias, its writes priced); float, then 4 and 6 bits with and without 10 percent static beta jitter; no sampling | Exact (`exact_reference`) Fisher information per node read about loading, and the reads and writes needed to reach loading error 0.01 at each chemical potential, for aggregator reads against the baseline of site reads spaced beyond the correlation length and thinned in time; the design number is the read gain at k = 16 under the declared constraints | Owner requested 2026-11-12; proposal 1 of the [proposal note](research/2026-11-12-mechanism-proposals.md). One-way couplings are a relaxed constraint (rule 8): the hardware model says directional endpoint couplings are supported but publishes no cost, precision or placement rule, so the flag is recorded and the result is kept apart from the supported machine. Pre-registered prediction: gain of at least 3 at all three chemical potentials; falsifier: a gain under 3 with correlations included. The exploratory independent-site ceiling is 9.7 at w = 1 (`docs/research/aggregator_read_gain_probe.py`), and it falls to about 2 when the threshold is four sites off, so the threshold rule is part of the design and is fixed before results. If only symmetric couplings are assumed the aggregator back-acts and the model changes: report that case as a stop, not a variant. The baseline is sparse thinned site reads, not full-array reads. Not an advantage claim (the pore's digital reference takes about 11 ms per point) and it does not unhold `native-adsorption-eligibility` or authorize sampling. Reads and writes per estimate are the headline; no joule comparison. A source card on dithered one-bit quantization and stochastic computing is owed before any protocol. Probe under 30 CPU-minutes, exact only, hash-checked imports from the adsorption archive; if it answers the question it closes the row with a docs-only PR. Answered by the P-0012 probe: yes under the relaxed constraints (one-way couplings, k = 16, own caps), with a read gain of at least 5.2 at w = 0.5 and an adapted threshold; no on the archived placement (k ≤ 5, gain 1.1 to 2.7) ([probe note](research/2026-11-13-aggregator-read-gain-probe.md)). |
| `directed-relay-latent` | done | Can one global binary latent carried by one-way fan-in and fan-out relay trees reach a stationary posterior error within 0.01 of exact on a reduced instance, and at what sweeps and extra sites? | A reduced instance built from P-0008's `t4_h05_q035` construction: one template latent and a pixel patch small enough to enumerate the full non-reversible chain; fan-in by aggregators in their near-linear regime, fan-out by a one-way copy tree with r in {1, 2} redundant parents; float and 8 bits; NumPy enumeration only, no THRML, no training, no new layout on the full archive | Exact stationary law of the directed chain by enumeration: latent posterior error against the exact posterior of the reduced instance, sweeps to reach it from a priced start, and extra sites and programming writes; the design number is the relay footprint and sweep count at error 0.01, against the tied-copy construction on the same reduced instance | Owner requested 2026-11-12; proposal 2 of the [proposal note](research/2026-11-12-mechanism-proposals.md). Runs after `aggregator-read-gain` and depends on it: if that row stops on back-action or on a gain under 3, the proposer says so and asks the owner before spending the probe. Uses the P-0008 archive as a fixture for an architecture mechanism; it does not reopen the denoising line, and a pass opens no training or image row. One-way couplings are a relaxed constraint (rule 8), flagged as in `aggregator-read-gain`. The chain has no detailed balance: nothing is called a Boltzmann equilibrium, and the stationary law is enumerated, not assumed. Pre-registered prediction: the directed construction reaches error 0.01 where tied copies at finite tie do not; falsifier: an exact stationary error above 0.01 at 8 bits, which would name lossy fan-in (binary relays cannot sum linearly), not mixing, as the binding limit. The stale-Gibbs bias form in the proposal note is a hypothesis to check against the enumeration. Not a task row, so the per-sample I/O and MAC rules do not apply. A source card on asynchronous or stale Gibbs bias bounds is owed before any protocol. Probe under 30 CPU-minutes; if it answers the question it closes the row with a docs-only PR. Answered by the P-0013 probe: no under the two-colour schedule (stationary latent error 0.065 to 0.079 even with ideal relays, 0.059 at 8 bits; staleness binds, not lossy fan-in); with a sequential order, which the Z1 lacks, a thermometer fan-in reaches 0.0052 at 8 bits with 8 extra sites; tied copies reach 0.0012 on this reduced instance ([probe note](research/2026-11-13-directed-relay-latent-probe.md)). |
| `z1t-noise-recovery` | done | Does fine-tuning the sparse body of Extropic's released Z1T-0 with N = 32 p-bit sampling noise on every tanh-linear output bring its 32-sample loss back to its noise-free loss, beyond what the same fine-tuning without noise does? | The Z1T-0 checkpoint (4 layers, width 12,288, fan-in 4) with its embedding and output head frozen; two arms from the same checkpoint with the same data order (noise-trained, and a clean-trained control), 2,000 steps of 4 × 256 tokens on an OpenWebText sample; output-side sampling noise only, float weights, no placement, no dy4p input encoding | Next-token loss in nats (`software_simulation`) under exact 32-sample output means on 64 OpenWebText windows with three noise draws each, paired between arms; the share of the noise-induced loss increase removed (recovered at 0.9 or more, not recovered in this budget below 0.5); the same loss at 128, 512 and 2,048 samples | Owner requested 2026-11-14. Held because the owner runs it on the local GTX 1050 Ti outside the loop: an estimated 4 to 5 GPU-hours, past the probe ceiling, with packages that are not in `uv.lock`; it stays held until the owner posts `summary.json`, and THERMES does not take it. Not a task row: nothing is priced and every hardware constraint except output sampling noise is relaxed. Control first: step 0 of the script reproduces the collapse on its own windows (exploration: clean 3.78, 32-sample 10.52 on OpenWebText) before either arm trains. Source card: extropic-z1t (exists). Proposal, fixed choices and run steps: [note](research/2026-11-14-z1t-noise-finetune-proposal.md); script `docs/research/z1t_noise_finetune.py`. Answered by the owner's run of 2026-11-14 (2,000 updates per arm, one training seed, `software_simulation`, no hardware claim): partial recovery. At 32 samples the noise-trained arm scores 6.003 nats on OpenWebText against 10.521 for the clean-trained control and 10.746 before training (noisy minus control -4.518, SEM 0.049 over 64 windows), which recovers 70.3 percent of the gap to the noise-free loss (68.0 percent on WikiText-103; the control recovers 3.3 percent) and misses the 90 percent mark set beforehand. It costs noise-free quality: 4.002 to 5.422 nats, and at 2,048 samples the noise-trained arm is 0.666 nats behind the control. The noisy arm's intermediate weights were not kept (lessons, 2026-11-14), and resume is not bound to the input files. Results note: `docs/research/2026-11-15-z1t-noise-finetune-results.md` (PR #169). |
| `z1t-sampled-head` | dropped | On Extropic's released Z1T-0, how many nats does it cost to draw the first L bits of the next token as L yes/no samples, one Bernoulli draw each with a logit linear in the final hidden state, before any dense scoring? | The Z1T-0 checkpoint with its body frozen and noise-free; a balanced binary tree over the 50,257 output vectors (2-means bisection in 256 PCA dimensions), one trained 12,288-wide vector per node to depth 12, and the released rows for the choice inside each depth-L group, which is the dense model's own conditional; a 256-way softmax over the depth-8 groups as the reference for binary draws, a 256-way softmax over free k-means clusters as the reference for the balanced grouping, and a random-leaf tree as the control; 4,000 training windows of 256 tokens from an OpenWebText sample | Exact next-token loss in nats (`software_simulation`; every head is a normalised distribution, nothing is sampled in the evaluation) on 128 held-out OpenWebText windows, paired against the dense head, at L = 8 (246 times fewer multiply-accumulates per token) and L = 12 (2,070 times); small at 0.05 or less, moderate to 0.15, large above; the per-level excess over the dense model; the same on WikiText-103 | Proposed 2026-11-14 in an owner session after the owner asked for the follow-up runs to be set up; the owner's merge accepts the row. Held because the owner runs it on the local GTX 1050 Ti outside the loop, after `z1t-noise-recovery` frees the card, with packages that are not in `uv.lock` and about 32 GB of cached states and archived heads; it stays held until the owner posts `summary.json`, and THERMES does not take it. Not a task row: hierarchical output heads are a known digital technique and are as cheap on a digital chip, nothing is priced (multiply-accumulates are a count), and every hardware constraint is relaxed, including placement (a 12,288-wide logit does not fit a degree-16 node). Control first: the script's self-check that a tree with exact branch probabilities reproduces the dense loss, then stage 0 and the random-leaf tree on the same windows. Exploration (CPU, 51k training tokens, depth 10): +0.62 nats at L = 8 and +0.90 at L = 10, still data-limited. Source card: extropic-z1t (exists). Proposal, fixed choices, reading rule and run steps: [note](research/2026-11-14-z1t-sampled-head-proposal.md); scripts `docs/research/z1t_sampled_head.py` and, for reading `z1t-noise-recovery`, the evaluation-only `docs/research/z1t_site_probe.py`. Dropped by D-0012: P-0014 shows the dense readout sits on both sides of the post's comparison (617.6M multiply-accumulates per token at width 12,288, outside its ratio), a cheaper head is a published digital technique that helps a GPU as much, and at 2,048 samples per unit, where the released model is still 0.91 nats from its noise-free loss, the low end of the Z1 side equals the same sparse body run digitally only at 131 pJ per multiply-accumulate, so no hardware or direction decision changes with this row's answer. Reopen only if `z1t-sample-budget-map`, priced with P-0014's counts, leaves the Z1 body cheaper than the same body run digitally at a stated tolerance. A source card on hierarchical and adaptive softmax heads is owed before any run. |
| `z1t-system-budget` | done | Per generated token, how many Z1 p-bits and chips, p-bit samples, node writes, read bits and off-chip multiply-accumulates does Z1T need, as a function of samples per unit, for the post's 512-wide configuration and for the released 12,288-wide checkpoint? | Z1T as published (source card extropic-z1t): the post's energy and latency model, and the released Z1T-0 with its four blocks and its unit counts per block (qkv 24,580, out 12,288, p1 49,152, p2 12,288); the sealed Z1 profile's constants (source card extropic-thermalizers: per p-bit per Gibbs cycle, per p-bit read, per node write) and the post's own; samples per unit from 32, the count the post's latency estimate assumes (source card claim C6), to 2,048 | Counts first: p-bits, chips at 269,568 p-bits each, samples, reads and writes per token, and off-chip multiply-accumulates for the readout and the co-processor; then the projected joule as a band from the write constant to clamp loads at the read constant, `calibrated_projection`, with every included and excluded cost listed and the off-chip side at 0.1, 1 and 10 pJ per multiply-accumulate. No sampling | Opened 2026-11-15 at the owner's request, after the Z1T noise retrofit line closed. Not a task row and not a claim of digital headroom: it prices a published design with published constants, so that an outside reader can check the post's system-level figures. Reproduce first: the post's own per-token total and its split between co-processor and sampling, from the constants it states; if that cannot be reproduced, say which constant is missing and stop there. The post says chip count is not modelled and its ratio excludes the dense readout; the row supplies those counts. State every assumption the post leaves open (input encoding, placement, which values cross to the co-processor) as a band, not a point. Inputs already in the repository: loss of the released model against samples per unit (10.746 nats at 32, 9.184 at 128, 6.901 at 512, 4.916 at 2,048, against 4.002 noise-free; `software_simulation`, first results note), to be replaced by `z1t-sample-budget-map` when it posts; and 105,442 of Z1T-0's 393,232 sampled units have every incoming weight and bias below 1e-6 in magnitude (near-zero under that threshold, not demonstrated exact zeros) (exploration, [counts](research/2026-11-15-z1t-dead-units-exploration.json)), so give the full budget and a hypothetical gated-unit budget separately. Gating is not established as noise-equivalent: even an exactly zero-mean sampled unit has nonzero output variance; no savings claim assumes equivalent quality. Earlier owner-session arithmetic on readout and I/O cost is unrecorded and must be redone, not reused. Notes: [first results note](research/2026-11-15-z1t-noise-finetune-results.md); [follow-up results](research/2026-11-15-z1t-noise-quality-followup-results.md) (PR #171). Answered by the P-0014 probe (docs-only; counts exact, joules `calibrated_projection`): the post's Z1 sampling 8.74 nJ reproduces as output draws only; its 285.78 nJ FPGA figure does not and is unused (owner option 1, 2026-11-16). At N = 32 with serial samples the 512-wide model needs 1 to 2 chips and Z1T-0 9 to 27; reads at the post's own inferred count cost 16 times its sampling energy; the dense readout (25.8M and 617.6M MACs per token) dominates every total at 1 and 10 pJ per MAC. [Probe note](research/2026-11-15-z1t-system-budget-probe.md). |
| `z1-acceptance-kit` | open | Which exact-answer tests on Z1-native graphs can accept or reject a sampled array, and how many reads does each need to detect a stated defect at stated error rates? | Maximal planar subgraphs of the Z1 16-offset rule at 64, 256 and 1,024 spins (the P-0001 graphs), zero field, at inverse temperatures where ordinary two-colour Gibbs qualified in the archive; defects applied one at a time: static per-site beta variation of 10 and 30 percent, couplings rounded to 4, 6 and 8 bits, one missing coupling, one stuck p-bit | For each test statistic with an exact value under both the ideal and the defective model (energy per spin and per-edge correlations from Kac-Ward only where its assumptions hold), the reads needed to detect each defect at 5 percent false alarm and 5 percent miss with the sweeps between reads stated, and the smallest defect of each kind detectable within 10,000 reads; `exact_reference` for means and shifts, `software_simulation` only for the variance and autocorrelation of the sampled statistic | Opened 2026-11-15 at the owner's request. A metrology row, not a task row: its output is a reference a chip can be scored against, so there is no digital baseline to price and no claim about any chip; nothing is measured on hardware. Reproduce first: the 18 exact references of the planar 16-offset ferro study (P-0001). Before admitting a defect, define its update kernel and establish an applicable exact reference. Kac-Ward covers the zero-field planar Ising law with symmetric edge couplings, not generic site-dependent-temperature updates: those can violate detailed balance and need a separately justified stationary reference. Missing or symmetrically rounded couplings may retain its assumptions. A stuck spin or added field needs a justified conditional/field reference, not an assumed zero-field formula. Exclude unsupported defects from exact detection claims; do not silently label simulated means exact. A failed test must be attributable, so use only temperatures where the ideal sampler passes within the stated budget, and report device error and mixing error apart. Runs under the declared constraints (two-colour schedule, 16-offset placement on the planar subgraph); reads and writes per test are the headline. To check in the probe before any protocol: pinning one spin in an otherwise unchanged zero-field equilibrium Gibbs kernel leaves spin-flip-even expectations unchanged under its stationary conditional law; this is not a claim about arbitrary stuck-device dynamics or finite-time samples. That conditional-law case needs an odd statistic or a small field, and a justified exact reference decides whether it stays in the row. Earlier studies found 10 percent static beta variation negligible for task quality; this row asks whether it is detectable, which is a different question. Source card: extropic-thermalizers (exists). |
| `z1t-sample-budget-map` | held | On the released Z1T-0, what is the next-token loss at each pair of sample counts for the internal units (qkv, p1) and the residual-writing units (out, p2) from 128 to 8,192, and what is the smallest sample budget per token within 0.1 and within 0.5 nats of the noise-free loss? | The released checkpoint, evaluation only, output sampling noise only; `z1t_site_probe.py` without `--run`, with `--internal-Ns 128,512,2048,8192 --writing-Ns 128,512,2048,8192 --norms both`, 32 OpenWebText windows of 256 tokens and two noise draws | Loss in nats (`software_simulation`) at the 16 grid points with paired differences to noise-free; the smallest tested budget whose mean plus 2 SEM meets each tolerance as an exploratory screen (not a simultaneous confidence bound across 16 cells), for uniform counts and for any allocation; the ratio of the two as a candidate allocation saving, not a confirmed advantage. If no tested point meets a tolerance, say so | Proposed 2026-11-15 at the owner's request. Held because the owner runs it on the local GTX 1050 Ti outside the loop with packages that are not in `uv.lock`; THERMES does not take it. It trains nothing, so the 4 GB card limits only its run time (estimated one to two hours; the probe has run only at smoke scale on a CPU, including 8,192 samples). Not a task row. Reading set here, before the run: the two budgets and the allocation ratio are exploratory grid-selected estimates, with no interpolation. Before claiming a tolerance or allocation saving, freeze the selected allocation and uniform comparator, the confirmatory window/noise-seed split, sample size and uncertainty rule, and check them on independent held-out windows and fresh noise draws, accounting for corpus dependence and any multiple confirmatory claims. This additional check requires its own bounded owner approval and is not launched or added to the 16-cell run by this queue edit; an owner-session estimate of a 2.3 times saving from allocation is arithmetic from two earlier small measurements and is what the run checks. Amended 2026-11-15 at the owner's request, before any run: `--norms both` added. The probe left the model's DyT normalisation layers noise-free, but the post's 25 sequential sampling layers and its 8.74 nJ of sampling (21,008 outputs × 32 samples × 13 fJ at width 512) re-derive only when every DyT is counted as a sampled layer; the post does not say so in words. The uniform reference and all 16 grid points are therefore also run with each DyT tanh sampled at the count of the site it feeds (503,824 sampled units per token against 393,232), and each DyT layer alone at 32 samples. Both versions are read by the same rule and both pairs of budgets are quoted; the version with DyT layers sampled is the one to set beside the post. About twice the configurations, so the run-time estimate becomes two to four hours (still unmeasured on the GPU). Feeds `z1t-system-budget`. Script and notes: `docs/research/z1t_site_probe.py`, [first results note](research/2026-11-15-z1t-noise-finetune-results.md). Amended by D-0012 before any run, adding no configuration: P-0014 prices a sample at 13 fJ, the smallest term per token, and a read bit at 1.692 pJ, so for every grid point also report read bits per token under P-0014's two crossing assumptions (the post's inferred 5 steps of D values, saying which site's count those values take, and every sampled output) and the serial sampling time (the sum over the sequential sampling layers of that layer's count at 50 MHz), and name the smallest tested point per tolerance in each of the three units, which need not agree. The result note re-prices P-0014's per-token budget at the screened counts and sets beside it the same network run digitally without sampling (1.68M body multiply-accumulates at 0.1, 1 and 10 pJ, with the readout common to both); `calibrated_projection`, no advantage claim either way. |
| `z1t-full-constraints` | held | What is the released Z1T-0's loss with output sampling noise, 4-bit weights and the post's input encoding applied together, at the sample budgets found by `z1t-sample-budget-map`? | The released checkpoint, evaluation only; per-row-scaled and per-layer-scaled 4-bit weights; the input encoding as the post describes it, stated before the run; the same windows as the map | Loss in nats (`software_simulation`) for each constraint alone and all together, with paired differences, at the budgets screened as candidates for 0.1 and 0.5 nats in the map (not confirmed tolerances unless the separate confirmatory check passes) | Proposed 2026-11-15 at the owner's request. Held on two conditions: `z1t-sample-budget-map` has posted, and a script exists (the site probe models output noise only; weight rounding and the input encoding have to be added and checked against the upstream forward pass first). Owner-run on the local GPU, outside the loop. Exploration to reproduce first (CPU, 15 windows, unrecorded): 4-bit weights cost about 0.015 nats with a scale per output row and about 1.5 with one scale per layer, noise-free. Not a task row. |
```

## Evidence classes

Nothing in this review is new evidence. P-0013's numbers are exploration
with exact stationary laws, cited from its note. P-0014's counts are exact
arithmetic on a stated design and its joules are `calibrated_projection`,
cited from its note; the loss figures are `software_simulation`
exploration from the owner-run notes. P-0015's and PR #176's figures are
from unmerged work and are weighed, not cited as accepted. The 1.45 times
bound, the H100 ratios, the map extrapolation and the pair-test read counts
are this review's arithmetic under stated assumptions, labelled where they
appear. P-0007, P-0008 and P-0009 figures are from recorded studies cited
by path. No hardware claim.
