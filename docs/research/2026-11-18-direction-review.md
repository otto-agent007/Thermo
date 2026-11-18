# Direction review D-0013

2026-11-18. Main commit 9a0b03f777073958c6dc9a02e7526776b0bca87f. THERMES-DIRECTOR.
Label: advice, not a decision. The owner edits the queue; this file proposes.

Trigger: one loop result accepted since D-0012
(`docs/research/2026-11-16-direction-review.md`, PR #178, merged
2026-11-16T20:41Z):

- P-0015 on row `z1-acceptance-kit`, PR #174, merged 2026-11-18T03:53Z,
  report `docs/experiment-reports/2026-11-17-z1-acceptance-kit/`.

Nothing is in flight in the loop. One owner-session PR is open and is not
touched: #181, which indexes the kit in AGENTS.md, CLAUDE.md and the README.
Read against charter amendments 1 and 2, the agenda, the queue at 9a0b03f
(twenty-one rows: fourteen `done`, three `dropped`, four `held`, none
`open`), the lessons register, the roadmap, the last three merged studies
(P-0015, P-0009, P-0008), all 34 source cards, the three P-0015 probe notes
dated 2026-11-17, the `t4` closure note and the P-0015 protocol.

## The change I would make first

1. **No queue change and no new row.** P-0015 closed the only open row. The
   queue is empty of eligible work, and nothing P-0015 found names a task
   with digital headroom. An empty queue is the right state.
2. **Read the kit as its simple tests, and stop work on the collective
   Kac-Ward scan.** The report keeps the scan for one reason: one
   programming write against 7 to 9
   (`docs/experiment-reports/2026-11-17-z1-acceptance-kit/summary.md`,
   "Collective scan"). Priced with the lab's own sealed constants, that is
   not a saving. A full-array programming is worth about 91 full-array
   reads (153.6 pJ against 1.692 pJ per node, `docs/z1-hardware-model.md`).
   The scan saves 6 to 8 programmings and spends 3,600 to 11,600 more
   reads. The arithmetic is in the verdict below. It is this review's, and
   it is analysis, not evidence. So the rerun the findings suggest, a
   larger calibration set of about 1,500 chains, should not be run. It is
   the named drop.
3. **One decision for the owner, carried from D-0012 and not answered on
   main: who receives the kit?** Its numbers are for a person holding an
   array. The lab holds none. With a recipient, the next step is theirs:
   the accept limits and the write and read costs are what the kit lacks.
   With none, the four-week evaluation's parking advice stands
   (`docs/research/2026-11-11-four-week-evaluation.md`, "The decision, and
   a recommendation"). This is not a premise question under the
   two-failures rule.

## Verdict

**P-0015 `z1-acceptance-kit`: the kit holds except for named checks.
Positive for the simple tests. Negative for the collective scan. Mostly a
probe's answer given a study's process.** Isolated sites and pairs detect
every declared defect with exactly independent reads and closed-form error
rates. A stuck p-bit takes 6 reads at a known site. A ±10 percent gain
spread takes 108 reads at 1,024 sites. A missing coupling at an unknown
place takes 56 to 72 reads and 7 to 9 programmings. All 14 simple checks
pass. The collective scan at beta 0.5 needs 3,658 to 11,677 reads, 65 to
182 times the matching scan on the same graph. It fails its false-alarm
check on one of nine targets: 76 of 1,000 against a limit of 74
(`findings.md`, opening and "Missing coupling"; lessons 2026-11-17). Six
things to read with it:

- **The design numbers existed before the run.** Every simple count was in
  the round-1 to round-3 probes, closed form except the pooled gain
  counts, and pre-registered expectation 2 was that the run would match
  them. It did (`findings.md`, "Pre-registered expectations"). The round-1
  note already said "Most of the row's answer is closed form"
  (`docs/research/2026-11-17-z1-acceptance-kit-probe-r1.md`, "What this
  changes"). The recorded run added three things: the 1,024-spin scan
  lengths, sampled confirmation of the counts, and the calibration miss.
- **The scan loses on energy too, under the sealed profile.** The report
  projects no joule, by design. Pricing its own resource table gives:

  | Sites | Matching scan | Collective scan | Ratio | Break-even, node write per node read |
  | --- | --- | --- | --- | --- |
  | 64 | 0.072 to 0.082 µJ | 0.42 to 0.67 µJ | 5.8 to 9.3 | 605 to 990 |
  | 256 | 0.33 to 0.37 µJ | 2.83 to 3.22 µJ | 7.7 to 9.8 | 786 to 1,027 |
  | 1,024 | 1.31 to 1.47 µJ | 17.3 to 20.6 µJ | 11.8 to 15.7 | 1,217 to 1,664 |

  The sealed ratio is 90.8. Assumptions: one programming is one full SRAM
  write of every node in the patch; the scan's state write is one more;
  the matching scan reads pair endpoints only; updates at 7.09 fJ. Without
  the state write, and with the matching scan reading the whole array, the
  ratio is 5.4 to 15.0. Writes are 96 percent of the matching scan's
  joule and reads are 95 to 98 percent of the collective scan's. This is
  `calibrated_projection` arithmetic on archived counts (`findings.md`,
  "Resources per test"). It says nothing about time: the model's read,
  write and host-latency constants are incomplete
  (`docs/z1-hardware-model.md`).
- **The one failure is marginal, and the protocol had the warning.** The
  scan's false alarm depends on which calibration set was drawn. On L16
  s4510 it is 6.0 percent with the archived 128-chain ratios and 2.9
  percent with ratios taken from 2,000 fresh chains (`findings.md`,
  "Missing coupling"; exploration, one seed). The round-3 probe saw 4.9 to
  5.5 percent on six graphs with one set each. A user draws one set. At a
  true 6.0 percent a check fails the 74 rule with probability 0.03 (this
  review's binomial arithmetic). So the 76 is part luck, as the findings
  say, and the frozen verdict is still right. The protocol recorded that
  n_scan moved up to 19 percent between calibration sets and still chose
  "the t quantile is the whole correction"
  (`docs/experiments/z1-acceptance-kit.md`, "Why a t quantile"; "Scan
  false alarm under an unbiased sampler"). Lessons 2026-10-08 (P-0001)
  already says a one-seed probe can name the arm a study hinges on and
  cannot size the effect.
- **"Accept" is wider than the result.** The row asks which tests can
  "accept or reject a sampled array". Every test here rejects one exact
  ideal: gain 1 at every site, float couplings. No device is that. The
  only device calibration figure on a card is Extropic's own X0 p-bit,
  about 5 percent in absolute probability
  (`docs/knowledge/sources/extropic-torx-paper.md`, "Cautions"). The kit's
  ±10 percent outlier test resolves a shift of 0.017 to 0.021 in absolute
  probability at its bias (this review's arithmetic on the site law). A
  p-bit off by the X0 figure would read like a gain error of 20 to 37
  percent at the kit's biases, past its ±10 percent defect. An acceptance
  test needs a limit to accept inside. The kit has detection counts and no
  limit.
- **The counts assume a program range no card gives for Z1.** The 5-read
  missing-coupler test programs |J| = 1 at beta 4. At beta J = 1 it is 16
  reads (round-1 note, finding 4). Coefficient range and precision are
  unpublished (`docs/research/2026-11-13-z1-feature-requirements.md`,
  "Constants whose publication would re-price the archive"). Site tests
  assume a representable bias, and pair tests assume every other coupler
  can be set to exactly zero.
- **D-0012's estimates, checked.** That review estimated 13 reads for a
  missing coupling at beta J = 1; the exact count is 16. It estimated
  5,700 and 660 reads per site for 10 and 30 percent gain errors; exact,
  5,664 and 675. It left pooled rounding open; pairs win at every depth
  (round-1 note, findings 3 to 5). It said the collective test was the
  wrong instrument for three of four defects. On reads it is all four.

Is it already known? In kind, yes. The vendor already calibrates single
p-bits: its X0 experiment used "one calibrated X0 p-bit"
(`extropic-torx-paper.md`, C9; the card does not say how). The binomial
sample sizes are textbook.
That a threshold built on an estimated variance holds only on average over
calibration sets is textbook too, and it is on no card. The literature on
testing a sampled Ising model against a reference is on no card either
(D-0012, Rubric 2, named one paper, unread). The row closed without that
reading. Its notes name only `extropic-thermalizers.md`, which the protocol
says "supports nothing here beyond the kernel convention". The counts on
the Z1 planar subgraphs, and the scan-against-matching ratio, are the new
part.

## Rubric 1

**P-0015.**

- **Design number or probe with a study's process?** Both. The numbers are
  of a kind amendment 1 names, a read and write budget per test. They are
  for a user the lab does not have. The exact part was a probe's answer
  before the run. The study's process went to the collective
  arm: 5.7 CPU-hours, three redrafts of the protocol, four probes, and a
  PR of 26,339 added lines. The probes had already shown that arm to lose
  by 50 to 140 times on reads (protocol, "Which test each defect gets").
- **Constraints.** Applied: the two-colour schedule, 16-offset placement on
  the planar subgraph and its matchings, zero field on the collective
  graphs, codebook rounding, no exchange. Relaxed or assumed, and stated
  (rule 8): a float ideal; a representable site bias, with zero field
  lifted for site tests only; simple tests choosing their own beta,
  coupling and bias; planarity, so 7 to 9 programmings here and 16 on the
  full degree-16 lattice (`findings.md`, "Relaxed or assumed constraints").
  Rule 8 satisfied. Not listed: that unused couplers can be programmed to
  exactly zero.
- **Accounting.** Stated as counts: elapsed sweeps, full-array reads, node
  reads, programming writes and state writes per test (`findings.md`,
  "Resources per test"). Rule 9 satisfied in the letter. It reports a
  method that wins on one count and loses on four, and it does not weigh
  them. The weight is in the sealed profile and is the table above.
- **Expectations.** Expectation 4 missed in two places: L16 s4510 ran at
  7.6 percent and L16 s4512 at 6.7, against "near 5 percent", and three
  sampled checks failed where all were expected to pass. Two 1,024-spin
  scans exceed the 10,000-read limit, which the protocol had allowed for.
  Expectations 1 to 3 held. Rows that assumed the opposite: none. No row
  is open, and no held row uses the scan.

## Rubric 2: the queue rows

### `z1-acceptance-kit` (done; P-0015)

- **Decision.** Which test a person holding an array runs per defect, and
  what it costs in reads and programmings. Answered: the simple tests.
- **Cheapest probe that would have made the study unnecessary.** The
  round-1 and round-2 probes, 36 CPU-minutes together. They had the
  simple counts and the scan's loss on reads.
- **Source card.** None answers it. `extropic-torx-paper.md` (C9) shows
  the vendor already calibrates single p-bits. The row closed with no card
  on test sample sizes.
- **Third variant?** No. But this is the fourth study on planar Ising
  graphs (PR #92, P-0001, planar annealing, P-0015). A follow-up would be
  the fifth. See "The one thing to drop".

### `z1t-sample-budget-map` (held, owner-run)

- **Decision.** At which sample counts the released Z1T-0 is within 0.1
  and 0.5 nats of noise-free, and what that costs per token in reads and
  joules under P-0014's counts. No result is on main.
- **Cheapest probe.** The four uniform points already on main; only the
  allocation arm is new (D-0012, Rubric 2).
- **Source card.** `one-bit-mean-estimation.md` (C4) and `normal-cn101.md`
  give the ε⁻² law the uniform arm will trace.
- **Third variant?** No.
- **From P-0015.** One caution transfers. The row screens 16 cells by mean
  plus 2 SEM. P-0015 shows a threshold tuned on one draw of a noisy
  quantity can miss its nominal rate. The row's notes already call the
  screen exploratory and require a separate confirmation. Keep that.

### `z1t-full-constraints` (held on two conditions)

- **Decision.** Whether the released model survives its published
  constraints together. It matters only if the map finds a usable budget.
- **Cheapest probe.** The map.
- **Source card.** `extropic-z1t.md` (C8, C9): the input encoding is open,
  and the row must fix it before a run.
- **Third variant?** No. Unchanged.

### `z1t-sampled-head` (held; D-0012's drop is in its notes)

- **Decision.** None on the hardware side (D-0012, "The one thing to
  drop"). The owner kept it `held`. No new evidence since, so I do not
  propose the drop again.
- **Cheapest probe.** The source card on hierarchical softmax the row's
  notes now say is owed before any run.
- **Third variant?** No.

### `native-adsorption-eligibility` (held; D-0006's drop is in its notes)

Unchanged. P-0015 does not touch it.

### Closed rows

Thirteen other `done` rows and three `dropped`. P-0015 reproduced all 18
P-0001 references bitwise, so `planar-16-offset-ferro` stands as archived.
No closed row depends on P-0015.

### Lines with two or more same-way negatives

- Global latent on `t4`: closed by the owner for every construction
  (`docs/research/2026-11-16-t4-global-latent-closure.md`; lessons
  2026-11-16).
- Z1T-0 noise retrofit: closed by the owner on 2026-11-15.
- Rounding rules beyond nearest: two negatives, no variant queued.

No line has a third variant open. This review puts no premise question.

### Is the queue pointed at the primary question?

There is no open row. The four held rows are three owner-run evaluations
of one released checkpoint and one adsorption allocation with a drop in
its notes. None names a task with digital headroom, native placement, a
user and a tolerance, which is the un-park condition
(`docs/research/2026-11-11-four-week-evaluation.md`, "Un-park when"). The
kit served amendment 2's architecture half as metrology. It does not move
the task half.

## Rubric 3: can the lab state the four-week test today?

Reported on its original terms on 2026-11-11: the numbers exist for one
task and are negative for the hardware (four-week evaluation; lessons
2026-11-11). P-0015 is not a task row and changes no clause. Restated:

1. *Precision and noise tolerance.* 4 bits under `split` with 10 percent
   static jitter on the local L = 8 denoiser; 6 bits with fields in J for
   the bias memory; 5 bits for the placed tie-limit latent; 4 field bits
   under class diffusion and 9 under nearest for the pore (evaluation,
   clause 1). Unchanged.
2. *Sweeps, reads and writes per useful sample.* 64 sweeps per step, 128
   reads and 128 clamp loads per two-step sample on the local family. No
   useful sample on any family with a global latent, and none for a Z1T-0
   token (evaluation, clause 2; D-0012, Rubric 3). Unchanged.
3. *Projected energy at stated quality against a conventional baseline.*
   **Negative.** 3.2e-7 J per 16-draw estimate at error 0.075, against a
   3,136-MAC logistic at 0.071 for 3.1e-10 to 3.1e-8 J (evaluation, clause
   3). Negative again for Z1T-0 at 2,048 samples per unit (D-0012, Rubric
   3). Unchanged.

What P-0015 adds beside clause 1, and it is not a clause: the reads it
takes to check those tolerances on an array. The lab's tasks tolerate ±10
percent static gain spread. The kit detects that spread in 108 to 520
reads and one programming, and detects ±1.0 to ±2.3 percent within 10,000
reads (`summary.md`). The tasks need 4 to 6 coupling bits. On the kit's
own weights, rounding at 6 and 7 bits is detected in 46 and 60 reads on a
1,024-spin matching (`summary.md`, "Simple sampled checks"). So on gain
spread the kit is 4 to 10 times more sensitive than any recorded task
requires, and on rounding it resolves two bits finer. Its 10,000-read
operating points have no task behind them.

Largest gap: clause 3 for any task with a quality pass. No row closes it
with a positive. `z1t-sample-budget-map`, priced with P-0014's counts,
closes clauses 2 and 3 for Z1T-0 at a stated tolerance; expect a negative.

Missing evidence, unchanged: no hardware measurement; no clamp-only
constant; no lower bound on digital cost; no training under the
constraints. Added by P-0015: no accept limit for any defect; no read or
write time; no published coefficient range, so the simple counts are for
an assumed program range.

## Rubric 4: representation, architecture, useful computation

**P-0015.**

- *Intended model.* For the simple tests, two-state laws: a site reads +1
  with probability sigmoid(2 β g h), and a pair agrees with probability
  sigmoid(2 g_B β |J|). For the scan, the zero-field planar Ising law on
  the archived graphs at beta 0.5.
- *Constrained representation.* The same laws with one defect applied:
  couplings on the `full` codebook, one coupler at zero, one constant
  site, or a gain array. Placement is a planar subgraph of the 16-offset
  rule and its matchings.
- *Actual update process.* The two-colour heat bath in NumPy float64, one
  read per sweep. On an isolated site or pair each read is an exact fresh
  draw (enumerated to 1.1e-16). On the collective graph it is one chain
  after 512 burn-in sweeps, with scatter ratios of median 1.16 to 1.23 and
  up to 3.3.
- *Estimator.* Exact binomial counts; a pooled homogeneity X² with a
  Monte Carlo threshold; for the scan, a Bonferroni test of edge means
  standardized by scatter ratios from 128 calibration chains, at a t(127)
  quantile.
- *Established, bounded.* Every declared defect is local, and on a
  two-colour machine the local test is also the exact one: isolation
  removes mixing, start state and read correlation at once. Pooling
  disconnected units lowers the reads: as one over the unit count for a
  shared shift (rounding: 3,788, 955 and 238 reads at 64, 256 and 1,024
  spins) and as one over its square root for a spread (gain: 520, 229 and
  108). A collective graph adds correlation and a calibration
  burden and adds no power against a local defect. Scope: these four
  defect models, a simulated ideal kernel, planar subgraphs, the declared
  program range.
- *Challenged.* The row's premise that exact collective references on
  Z1-native graphs are the acceptance instrument. With it, the remaining
  use of the planar Kac-Ward line as a chip reference.
- *Unsupported generalizations to avoid.*
  - "The kit accepts an array": it rejects a point ideal.
  - "The scan saves writes" read as "the scan is cheaper".
  - Any n_scan as a design number: each rests on one 128-chain set.
  - "7 to 9 programmings" off the planar subgraph: 16 on the full lattice.
  - "It detects rounding" or "it detects gain error": it detects gain
    times coupling and gain times bias (`findings.md`, rule 8 item 2).
  - Anything about update-process defects, which the row did not declare.
- *User and design decision.* A person qualifying an array: which test,
  how many reads and programmings. No such person is named on main.
- *Strongest relevant baseline.* A metrology row prices no digital
  computation. The relevant baseline is the conventional test procedure,
  per-device calibration by independent draws. That is the simple test,
  and it won.
- *Numerical hardware requirement.* Two, both conditional. The collective
  scan pays on projected energy only if a node write costs more than 600
  to 1,660 node reads; the sealed profile says 90.8 (this review's
  arithmetic). And the 5-read coupler test needs beta |J| of 4; at 1 it
  is 16 reads.
- *Counterfactuals.* None claimed. No advantage is claimed for any
  machine.

**For a next proposed question:** none is proposed. For the advice to stop
the collective scan, the falsifier is either a published write cost above
the break-even in the table, or an owner row that declares an
update-process defect (stale neighbour reads, colour order), which only a
collective test could see. The cheapest decisive check is the arithmetic
above: minutes, no sampling.

## The one thing to drop

**The collective Kac-Ward scan as a kit component, and with it the
larger-calibration rerun.** The findings say "A rerun of this kit would
need a larger calibration set; the post-run check suggests about 1,500
chains" (`findings.md`, "What this does not show"). Three reasons not to:

1. **The scan sees no declared defect the simple tests miss** (`summary.md`,
   "Collective scan").
2. **Its single programming write is not a saving** at the sealed price
   ratio. The break-even ratio is 7 to 18 times the sealed one (table
   above).
3. **A better-calibrated scan is still the scan.** Better ratios fix the
   false alarm, not the read count. With perfectly independent reads the
   scan still needs 2,624 to 4,013 of them, 42 to 63 times the matching
   scan on the same graph (`summary.md`, "Collective scan by target").

Reopen condition: either falsifier in Rubric 4. Also not to grow, each for
the same lack of a recipient: a full degree-16 version, frustrated
targets, an accept-limit version of the simple tests. D-0010's, D-0011's
and D-0012's drops stand.

## Evidence obligations

Read from `/home/user1/.hermes/profiles/thermes/loop/evidence-obligations.json`
(checked 2026-11-11T08:54Z at 682884a), reconciled against main at 9a0b03f.
The index is dated administration, not proof; this review does not edit it.

| ID | Source | Accountable role | Due trigger | Status in index | Reconciliation at 9a0b03f |
| --- | --- | --- | --- | --- | --- |
| EO-001, error-diffusion card | D-0008 verdicts | proposer; reviewer checks scope | before a protocol using the rounding explanation | `closed_scoped` | Unchanged. Card on main at abstract depth. |
| EO-002, full-source check | the card's "Limits" | next proposer relying on it | before a stronger claim than the abstracts support | `conditional_open` | Not triggered: rounding is a defect in P-0015, not a lever. Closure evidence: none, none due. |
| EO-003, read-allocation lesson | D-0008 verdicts | director names; docs author writes | next docs update, before a proposal reuses the allocation | `open` | Stale. Closure evidence on main: lessons 2026-10-20. P-0015's frozen protocol still calls it missing. The lesson was half applied: reads were sized against measured scatter, and the calibration set's own variance was not. |
| EO-004, independent rounding case | D-0008 | proposer; owner accepts | before generalizing P-0009 | `closed_negative` | Unchanged; P-0010. |
| EO-005, four-week evaluation | agenda, "November 11" | director supplies; owner decides | 2026-11-11 | `open` | Stale. Closure evidence on main: the four-week evaluation and lessons 2026-11-11. Rubric 3 restates it. The owner's direction since is recorded in queue notes only. |

Candidate debts for the owner to add or decline:

- **The kit has no reading behind its test design.** No card covers sample
  sizes for testing a sampled Ising model, or thresholds built on
  estimated variances. D-0012 named this before the row closed; it was
  not adopted. Role: docs author. Trigger: before the kit is sent to
  anyone or extended. Evidence: missing.
- **`extropic-z1t.md` still lags P-0014** (D-0012). Last checked
  2026-11-15; the two facts are still absent. Evidence: missing.
- **`z1t-sampled-head` owes a hierarchical-softmax card**, now written
  into its notes. Trigger: before any run. Evidence: missing, not due.
- **Runtime calibration under worker contention is in the findings and
  not in the lessons register.** See below. Role: docs author.
- **P-0012's note still says a card "is not written" that is**
  (`docs/research/2026-11-13-aggregator-read-gain-probe.md`). Docs hygiene.
- **A source card's depth is not validation.** This review cites
  `extropic-torx-paper.md` at its stated depth (full v2 text, read
  2026-10-01) and no abstract-level card for a number.

Closed since D-0012: the owner's answer to D-0011 is on main (closure
note; lessons 2026-11-16).

## Operational outcome and scheduler health

Kept apart from the science above.

- **P-0015, operational outcome.** Success, with one recorded protocol
  deviation. Completion was written after a full replay (601 s); 38 of 41
  checks pass; the archive is 858 kB. Scientific disposition: the row is
  answered; the collective arm is negative.
- **The deviation.** The frozen stop rule read "A reference, rank-2,
  finite-difference, patch or enumeration check fails: stop before
  sampling and report. No tolerance is loosened." As written it would
  have stopped the run. The executor chose a reading of "direct
  evaluation" before the run and went on. The reviewer escalated, and the
  owner accepted it as a deviation on 2026-11-17 (`findings.md`, "Notes on
  the implementation"). No reported number depends on it. P-0014 met the
  same kind of gate by stopping and asking (D-0012). An ambiguous frozen
  check resolved by the executor leaves the protocol text unchanged, so
  the protocol-diff check cannot see it. The owner has ruled; I propose no
  repair.
- **Two registered lessons recurred.** The tree was dirty at assembly
  because a plot script was written during the run (lessons 2026-10-06,
  Potts stage B runs, item 1). Wall time was 1.8 times the estimate and
  CPU time 1.8 times, after 2.7 times on P-0007 and 1.6 times on P-0008
  (lessons 2026-10-16, 2026-10-17). The cause here is new: the kernel was
  timed alone at 42.7 µs and ran at 90 to 120 µs with three workers. The
  run ended 16 minutes short of its own 2x stop.
- **Scheduler.** Observed only: PR #174 merged at 03:53:20Z and this card
  was created at 03:55:46Z. The queue has no open row, so no experiment is
  eligible. `loop.json`, the event log and `loop-tick.py --status` were
  not inspected; scheduler health beyond that is unknown. One setup note:
  this card's worktree was created at 56fa0a9, the head of the open
  docs branch behind PR #181, two commits past the sync target. The
  card's `git reset --hard` is refused by this profile's terminal gate. I
  moved the branch to the target with `git reset --keep`; both commits
  remain on origin.
- **Handoff.** Inspected: main at 9a0b03f, 2026-11-18. Row
  `z1-acceptance-kit`, P-0015, PR #174, protocol approved at dc32142,
  runner at ba3f4f3. Next actor: the owner, on the decision in "The change
  I would make first", item 3. Trigger for revisiting: a named recipient,
  a published write or read constant, or a new owner row.

## Proposed queue

No queue change.

## Evidence classes

Nothing in this review is new evidence. P-0015's simple-test counts and
collective means are `exact_reference`; its pooled gain counts are Monte
Carlo on exact laws; n_scan is an exact count times a `software_simulation`
scatter ratio; its operating characteristics are `software_simulation`. All
are cited from its report. The post-run fresh-chain figures are exploration,
quoted from the findings. The projected joules, the break-even ratios, the
binomial tail at 6.0 percent and the absolute-probability shifts are this
review's arithmetic under stated assumptions, labelled where they appear.
The X0 calibration figure is `asserted` on its source card. Four-week
figures are from recorded studies cited through the evaluation. No hardware
claim.
