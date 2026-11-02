# Direction review D-0007

2026-11-02. Main commit 5f21edff254e2f7245af33f17621d9e00a1390d2. THERMES-DIRECTOR.
Label: advice, not a decision. The owner edits the queue; this file proposes.

Weekly floor. No loop result was accepted since D-0006
(`docs/research/2026-10-26-direction-review.md`, PR #136, merged). In
flight and weighed, not touched: P-0009 on row
`adsorption-precision-requirement` (PR #135, draft since 2026-10-23, no
comment or review on it, head unchanged), and PR #140, the gatekeeper's
adoption of D-0006 into the queue (opened 2026-11-02; it keeps
`native-adsorption-eligibility` at `held` and records the drop in the
notes, as the loop's rule says). Nothing under `docs/research/` is dated
after 2026-10-26. Read against charter amendments 1 and 2
(`PROJECT_CHARTER.md`), the agenda
(`docs/research/2026-10-20-lab-research-agenda.md`), the queue at the
commit above, and all 26 source cards.

## The change I would make first

Decide P-0009 before 2026-11-11, either way. It is the only item that can
change what the four-week evaluation says, and its remaining cost is under
a CPU-minute (PR #135 body; probe, "What this sizes"). Approved, it adds
one exact representation number to clause 1 of the test. Rejected, the
evaluation reports the adsorption line as one negative probe and no
selected study. Undecided, it reports a draft, which is the worst of the
three for a review whose purpose is to say whether the amendment worked.
That is an owner decision; I do not recommend which way.

The thing to drop this week is not a queue row. It is **surface-code
decoding as a standing candidate task** (rank 2 in
`docs/research/2026-10-20-next-task-recommendation.md`, "Three candidate
families"). Reasons under "The one thing to drop". No new row: the agenda
says not to fill an empty queue to keep the loop running
(`docs/research/2026-10-20-lab-research-agenda.md`, "Immediate
disposition"), and the next question depends on P-0009's outcome.

## Verdicts

No accepted result. On what moved since D-0006: nothing scientific.

**PR #140 (queue adoption of D-0006): process, correct.** The gatekeeper
kept the row `held` and appended the drop to its notes. That is the rule
in this card's own text ("rows you set to `dropped` come out `held` unless
the owner says otherwise"). My verdict from D-0006 stands: no outcome can
unhold the row (exact 8-bit miss 0.011078; 64-read allocation cannot
certify 0.01; `docs/research/2026-10-20-native-adsorption-eligibility-probe.md`,
"Follow-up diagnosis"). If the owner agrees, say so on #140 and the status
becomes `dropped`; if not, `held` with the note is harmless.

**P-0009 (PR #135): unchanged for ten days.** The probe's development
result (class diffusion passes at 5 bits, nearest at 9, per-point caps at
7; `docs/research/2026-10-23-adsorption-precision-requirement-probe.md`,
"Development result") and the protocol's pre-registered expectations are
as D-0006 read them. Two things D-0006 asked for are still absent on main:
a lessons row for the unpowered read allocation (the 2026-10-06 Potts
stage B design lesson applied again; `docs/knowledge/lessons.md` has no
adsorption entry) and a source card for error diffusion or quantization
bias correction, so that P-0009's report can say what is known. Neither
blocks the decision. Both are owed before the report.

**Dependabot merges (#137, #138): no bearing on direction.**

## Rubric 1: the last three recorded studies, briefly

Unchanged from D-0006, "Rubric 1". All three are in the closed denoising
line and the owner's closure stands (amendment 2, "Commitments").

- **P-0006 `dtm-small-image`**
  (`docs/experiment-reports/2026-10-14-dtm-small-image/findings.md`):
  design numbers (4 bits under `split`, K = 64, 64 reads and 64 writes per
  sample, 2.0e-8 J per sample, I/O 99.1 percent, jitter negligible);
  nothing relaxed on the operating arm; accounting stated; quality
  positive, cost negative against a 114-parameter logistic (lessons
  2026-10-14).
- **P-0007 `dtm-baseline-headroom`**
  (`docs/experiment-reports/2026-10-16-dtm-baseline-headroom/findings.md`):
  negative; a 3,136-MAC logistic (3.1e-10 to 3.1e-8 J per image across the
  MAC band) matches the DTM (3.2e-7 J to 9.7e-9 J per estimate) in all nine
  seeds; continuation lost to a random start everywhere (lessons
  2026-10-16).
- **P-0008 `dtm-exact-posterior-lattice`**
  (`docs/experiment-reports/2026-10-17-dtm-exact-posterior-lattice/findings.md`):
  negative by the frozen rule; one exact design number (tie-limit latent
  holds at 8 and 6 bits, fails at 4); mechanism finite tie strength, gap
  0.0117 against 0.01 (lessons 2026-10-17).

No open row assumes the opposite of any of these. The open row is exact
only and outside denoising.

## Rubric 2: the queue rows

Eleven rows at 5f21edf. Nine are `done` or `dropped` and unchanged since
D-0005; nothing depends on re-running them (D-0005 and D-0006, "Rubric 2").
One line each, so the record is complete:

- `am-coupling-bits` (done): the lab's cleanest spec number, 6 bits with
  fields in units of J (lessons 2026-10-08); its design is not placeable
  (lessons 2026-10-13), so it prices no machine.
- `planar-16-offset-ferro` (done): sampler result on a known-good regime;
  demoted class under amendment 1 (lessons 2026-10-08).
- `constraint-layer` (done): engineering; every later row runs through it.
- `dtm-small-image`, `dtm-baseline-headroom`, `dtm-exact-posterior-lattice`
  (done): covered above; line closed.
- `dtm-global-latent` (dropped): superseded by P-0008.
- `am-io-budget` (dropped): answered by the layer's priced example; a
  placeable memory needs a source card first (row notes).
- `e0-remaining-kernels` (dropped): pure replication, owner said later.

### `native-adsorption-eligibility` (held; drop pending in PR #140)

- **Decision.** Whether sampled adsorption on this pore continues under
  this allocation. Answered: no (probe, "Research decision").
- **Cheapest probe.** Ran. The exact half decided alone.
- **Source card.** `docs/knowledge/sources/schneider-local-adsorption.md`
  gives the model class; none answers precision or power.
- **Third variant.** No; one failure in the line.
- **Verdict.** Drop, as D-0006 said. Nothing new.

### `adsorption-precision-requirement` (open; P-0009 awaiting owner)

- **Decision.** Narrow: this pore's field bits per rounding rule (a
  precision requirement, amendment 1 tier 2). Wide: whether a compile-time
  rounding rule is a general lever on the constraint layer's precision
  requirements, since `thermo_lab.constraints.quantize` rounds to nearest
  and every archived bit requirement was read under that rule
  (am-coupling-bits 6/10/12 by cap; P-0006 4; P-0008 8/6, fails 4). The
  wide decision is the one worth the process; the narrow one designs
  nothing anyone will build, because the pore is a representation vehicle
  only (D-0006, "Rubric 2").
- **Cheapest probe.** Ran (4.7 CPU-s). The recorded study is the probe
  with the frozen points unmasked plus a disjoint six-point grid; its value
  is the held-out discipline, which the proposer kept (protocol, "Arms and
  controls").
- **Source card.** None covers it. Error diffusion and per-class bias
  correction are standard in signal processing and weight quantization; the
  lab has no card, so the protocol names a rule it reinvented. One card
  before the report freezes. Advice to the owner, not a protocol change.
- **Third variant.** No. It asks the exact representation question; the
  line has one failure (sampled eligibility).
- **Expectation 2 is already public** (the control's 8-bit miss is
  archived), so the study's unknown outcomes are rules 2 to 4 on the frozen
  points and the held-out grid. If class diffusion passes at 5 on both, the
  result is the rule, not the pore; the next check is then the transfer
  probe D-0006 named (am-coupling-bits' archived exact recall under `split`
  caps re-read with class diffusion; falsifier: the requirement stays at
  10 bits). That is a probe, minutes, exact only, and not a row until
  P-0009 is accepted.

## Rubric 3: can the lab state the four-week test today?

Due 2026-11-11, nine days out. Yes, and the answer is the one D-0005
recorded with the owner's wording. Nothing accepted since then changes a
number:

1. *Precision and noise tolerance.* 4 bits under `split` on the local
   denoising family (P-0006); 6 bits with fields in J for the bias memory
   (am-coupling-bits); 8 or 6 bits for a tied-copy latent at the tie limit,
   4 fails (P-0008); static 10 percent beta jitter negligible in four
   studies. P-0009, if accepted and its development numbers hold, adds: 9
   bits under nearest and 5 under class-balanced rounding for a 144-site
   lattice gas, exact, no jitter.
2. *Sweeps, reads and writes per useful sample.* K = 64, 64 reads, 64
   writes, 0 clamp changes on the local family (P-0006). No useful sample
   on any family with a global latent (P-0007, P-0008). The adsorption probe
   priced a trajectory (9,216 reads, 288 writes, 589,824 updates; 6.4e-8 J
   active-only) with no useful sample at tolerance (probe, "Reads, writes
   and conventional baseline").
3. *Projected energy at stated quality against a conventional baseline.*
   Cost-competitive above 1 pJ per MAC, 0.002 short on quality, no task
   wins on both (D-0005, "Owner decision"). Unchanged.

Largest gap: clause 3 for any task with a quality pass. No row closes it;
the open row excludes sampling and cost by design. The honest 2026-11-11
statement is: the amendment produced every number it asked for, under the
constraints, with accounting, and no task the lab has run wins on cost and
quality together. The charter's instruction for that outcome is to revisit
the amendment rather than add process ("How to tell whether this is
working"). The agenda adds: name one consequential owner decision,
including parking. My reading of what that decision is: whether the lab
continues on representation requirements without an application
(P-0009's wide question, then the transfer probe) or parks until a task
with demonstrated digital cost headroom and a resident model is named.
Both are defensible; the second is cheaper. I would not put a third task
screen into the queue to avoid the choice.

## Rubric 4: representation, architecture, useful computation

Nothing was established or challenged this week. The standing bounded
claims are D-0006's: coherent same-sign rounding amplified by
susceptibility shifts a global observable by 1.1 points at 8 bits on a
directly represented local model (adsorption audit), so small parameter
error does not imply small observable error at that precision; and
class-balanced rounding is a hypothesis, read on six development points of
one pore, with its independent case not yet run (P-0009 frozen points and
held-out grid, then an unrelated archived model). Intended user: the
constraint layer's author choosing the default rounding rule. Strongest
digital baseline on the pore: the exact transfer matrix, 11 ms per point.
No hardware requirement follows; rounding is a host compile step.

## The one thing to drop

**Surface-code decoding, as a standing candidate task** in
`docs/research/2026-10-20-next-task-recommendation.md` ("Why decoding is
second"). The memo ranks it second and calls it the most immediately
standardized benchmark; that makes it the most tempting next row after the
adsorption line, and it should not be one. Three cited reasons:

1. Its own table says every query reprograms the model: "syndrome-dependent
   signs must be programmed each query" and "sector scoring needs repeated
   configuration reads and host computation" (memo, "Three candidate
   families", row 2). Under the sealed profile a full node write is 1.536e-10 J
   and a node read 1.692e-12 J (adsorption probe, "Reads, writes and
   conventional baseline"). Per data qubit per query, programming alone
   costs what 15, 154 or 1,536 MACs cost at 10, 1 or 0.1 pJ per MAC, before
   a single read and before the host compares two sectors. The constraint
   layer's priced examples put I/O at 98 to 99.996 percent of projected
   energy (`dtm-small-image` row notes), and P-0006 measured 99.1 percent
   at its operating point (P-0006 summary). A task that writes the whole
   model per query is the worst
   case for an I/O-expensive machine. That is a break-even requirement in
   the agenda's sense, not an advantage claim: a digital decoder that
   needs fewer operations per qubit than those figures wins before the
   sampler starts, and the memo itself says the practical comparator is
   highly optimized and exact maximum-likelihood decoding is polynomial in
   the regime with the cleanest mapping.
2. The memo already names "significant risk that digital specialization
   and input programming remove all headroom." Under the agenda's rule
   (identify cost headroom before optimizing a sampler), that risk is the
   screen, and the arithmetic above fails it with no probe.
3. No source card exists for any decoder, the sector-decision rule or the
   code family. Opening a row would mean reading that literature first; the
   failure amendment 1 records is rediscovering it.

Dropping it narrows the 2026-11-11 selection to the two candidates the
memo's own filter leaves (adsorption, with P-0009 pending, and the
channel-assignment restriction, which the memo already half-rejects).
This is advice about a memo's ranking, not an edit to it.

## Proposed queue

No queue change.

The change D-0006 proposed is in PR #140, and P-0009 is the owner's
decision. Adding a row now would be filling the queue; the next row, if
any, follows from P-0009's outcome and the 2026-11-11 decision.

## Evidence classes

Nothing in this review is new evidence. The P-0009 numbers are from a probe
in an open draft PR and are exploration; the adsorption numbers are from an
exploration note with exact calculations and software samples; the
denoising numbers are from recorded studies cited by path. The decoding
arithmetic is this review's calculation from the sealed profile's published
constants and is a break-even requirement, labelled analysis. No hardware
claim.
