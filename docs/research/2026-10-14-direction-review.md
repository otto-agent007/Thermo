# Direction review D-0002

2026-10-14. Main commit 6c9e5f444ff93d72640e6b80c5b8c4dba5bef825. THERMES-DIRECTOR.
**Advice, not a decision.** The owner edits the queue; this file proposes.

Covers the one loop result accepted since D-0001 (P-0004 `constraint-layer`,
PR #107, engineering), weighs the card in flight (P-0005
`e0-remaining-kernels`, stopped as a question, no PR), and reads the queue
against charter amendment 1 (`PROJECT_CHARTER.md`, "Amendment 1"). Previous
review: D-0001 (`docs/research/2026-10-09-direction-review.md`, PR #105,
merged 2026-10-10).

## The change I would make first

Open `dtm-small-image` now, with its target family rewritten so the model is
designed on the 16-offset lattice from the start and its probe settles the
I/O path before anything is trained. The hold condition ("until
`constraint-layer` is done", `docs/research-queue.md`, row `dtm-small-image`)
is met as of PR #107. It is the only row that can give the four-week test
all three clauses for a model the chip can hold: P-0004 showed that both
models the lab has specification numbers for cannot be placed (the AM `bias`
design at degree 24 and up to 128, the E0 kernel at degree 17, against 16;
`docs/research/2026-10-13-constraint-layer-plan.md`, "Fixed choices"). A
denoiser whose sites are lattice sites is placeable by construction. The
four-week clock started 2026-10-08 and has 22 days left; a row that is
weeks of work has to start this week or the amendment's own test fails on
schedule.

Two housekeeping facts shape the rest. First, D-0001's queue block never
reached `main`: the queue at 6c9e5f4 still has `e0-remaining-kernels` open,
`am-io-budget` below `dtm-small-image`, and the `constraint-layer` row
without the D-0001 text fixes (the owner applied those by answering P-0004
instead: "E0 and AM checked with placement relaxed and flagged", plan,
opening paragraph). The merge-applies-review mechanism arrived with PR #106
on 2026-10-12, after D-0001 merged. Second, because the row stayed open, the
gatekeeper gave `e0-remaining-kernels` a second proposal (P-0005) three
days after the owner rejected the first (P-0003, "because we are doing it
later. Approve D-0001"). P-0005 stopped itself as a question. That row has
now consumed two proposal cards and produced nothing, which is what a drop
is for.

## Verdicts

**P-0004 `constraint-layer`: positive, done as engineering, no design
number, and it was not meant to produce one.** `thermo_lab.constraints`
reproduces M5b rounding (120/120), M5c placement (60/60), the planar graphs
(18/18), the AM 6-bit recall (96/96 exact, 2/2 sampled) and the E0 stage B
histograms (6/6 cells), each with its relaxed constraints in the hashed
request (`docs/research/2026-10-13-constraint-layer-plan.md`, "Archives and
samplers reproduced bit for bit"; `docs/knowledge/lessons.md`, row
2026-10-13). D-0001 asked for exactly this treatment (unit tests plus a
dated note, no study process) and the owner chose it. What the build taught
that no row had assumed:

1. **Two of the lab's three constrained models are unplaceable.** The AM
   `bias` design needs a chain embedding of 28 to 400 p-bits for 20 to 144
   logical spins, and the chain changes the law; the E0 kernel has output
   degree 17. The placed M5c refit is a different law from E0's, up to 0.31,
   0.40 and 0.38 at K = 1, 2, 4 (probe,
   `docs/research/2026-10-10-constraint-layer-probe.md`, items 3 and 4; plan,
   "As built", slow-run results). The 6-bit AM spec number from P-0002 is
   therefore a number for a graph the chip cannot hold. This was a caveat in
   D-0001; it is now a measured fact.
2. **I/O dominates projected energy in every priced example**: 99.996% for
   one M5c outer sweep (272 of 344 writes are clamp changes), 99.3% for one
   AM recall at P = 8, 98.0% for a 1024-spin planar chain read every 4 sweeps
   (probe, item 5; the plan pins the counts). The cause is the sealed
   profile, not the models: a clamp change is priced as a full SRAM write at
   153.6 pJ, 21,665 Gibbs updates
   (`docs/z1-hardware-model.md`, "Clamped nodes ... treated as a write";
   `docs/experiment-reports/2026-10-07-exchange-cost-projection/findings.md`),
   and a node read is 239 updates. Under this model, the "projected joule"
   of any task at the lab's sizes is reads and writes per sample times a
   published constant; sweeps and bits barely move it. The Z1T source card
   already warned of this shape (the FPGA at over 95% of Z1T's projected
   energy, `docs/knowledge/sources/extropic-z1t.md`, cautions). Every row
   that promises "per projected joule" inherits it.
3. **"Bitwise" is a per-host claim.** CI's runner matched the AM exact
   recall only to 1e-12 and E0's float32 histograms only within frozen
   tolerances (5 of 6 cells differed); the tests now check each study's own
   replay rule off the archive host (plan, "As built", deviation 7; lessons
   row 2026-10-13). Later rows should pre-register tolerances, not bitwise
   equality, for anything that passes through BLAS or XLA.

## Rubric 1: the accepted result

### P-0004 constraint layer

- **Design number?** No. It is a transformation plus an accounting record
  (amendment 1, priority 1). The two structural facts above (unplaceability
  of the AM and E0 models; I/O share of 98% to 99.996%) are what later rows
  design around, and they came from the probe for 22 CPU-minutes.
- **Hardware constraints.** It is the layer. In its own checks it relaxed
  placement (AM, E0), quantization and jitter (M5c, planar) and quantization
  for E0 (float64 cast to float32), each flagged in the hashed request
  (plan, "Hardware constraints applied and relaxed"). `no_replica_exchange`
  cannot be relaxed at all (plan, `constraints/gibbs.py`). Rule 8 is met in
  the strongest form the lab has yet produced: a relaxation changes the
  digest.
- **Resource accounting.** Yes: one priced example per constraint with
  updates, reads, full SRAM writes and clamp changes as separate fields,
  labelled `calibrated_projection` of algorithmic counts (plan, "Pricing").
  The `update_rule` choice (`whole_array` against `active_kernel`) changes
  the priced numbers and is declared, not defaulted.
- **Missed expectation.** The probe and plan expected bitwise agreement
  everywhere; the build found it holds only on the archive host (above).
  Rows that assumed the opposite: `constraint-layer`'s own metric ("bitwise
  agreement ... through the layer") was met for E0 and AM only with
  placement relaxed, as D-0001 predicted. `dtm-small-image` assumes arms
  "at 6 and 8 bits (step = J)"; the layer's `coupling` rule rejects mixed
  magnitudes outright (plan, `constraints/quantize.py`, `cap_for`), so those
  arms cannot be requested for a trained denoiser. `am-io-budget` assumes
  the `bias` design can be "counted through `constraint-layer`" on the chip;
  it can only be counted with placement relaxed.

### Pattern

D-0001 noted two probes that mis-sized a pre-registered effect; P-0004's
probe over-claimed bitwise portability. All three are process, not a failed
line; the lessons register already carries each. No queue line has failed
twice in a row. No premise question for the owner.

## Rubric 2: the queue rows

**`dtm-small-image` (held; propose open).** Decision: whether the lab can
make its first task-level claim, "at this quality, projected energy A
against measured B", for a model the chip can hold, and at what bits,
jitter, sweeps and I/O. This is amendment 1's priority 2 and the only row
that can satisfy the four-week test for a placeable model. Cheapest probe
that shrinks the study: price the candidate I/O paths through
`constraints.price` before training anything. The arithmetic is immediate
from the sealed constants: a 16 x 16 denoiser with four steps, each step a
host rewrite of 256 clamps and a 256-pixel readback, is about 1.6e-7 J per
sample with I/O at 99.9% whether K is 16 or 256 sweeps per step. (That is
my arithmetic through the merged layer, not evidence; the row records its
own.) So the projected-joule axis of the row is decided by one fixed choice,
the I/O path per sample (per-step host readback and rewrite, or chaining
steps on chip), and the sweep and bit axes decide quality only. The probe
should name that choice first, per the lab's own rule (CLAUDE.md, "Check
assumptions before building process"). Three more fixed choices bound the
metric: the model must be lattice-native (sites on the 16-offset lattice,
edges on its offsets, two colours), or the row repeats P-0004's AM result;
the cap rule must be declared (`absolute` or `split`; AM found `split`
needs 10 bits and `full` 12 for a model with fields up to 28 J,
`docs/experiment-reports/2026-10-08-am-coupling-bits/findings.md`); and the
"log-likelihood where tractable" metric is tractable for a nearest-neighbour
grid of width at most 24 by transfer matrix (the method of
`docs/experiment-reports/2026-10-08-planar-annealing/findings.md`, with
fields added) but not for the long offsets (1,2) and (1,4), so the probe
must say which graph the exact reference covers. Source cards:
`extropic-dtm-hardware.md` C2 (GPU parity at about 10,000x less energy, a
projection under Extropic's model) and `extropic-torx-paper.md` C7 (binary
MNIST at 28 x 28, 10 steps, bit-error rate 0.114) are the claims the row
checks against the lab's own cost model; `extropic-z1t.md` warns the I/O
side dominates; `wang-deng-online-learning.md` C3 says online-trained
continuous models need less coupling precision. None answers the row.
Already known inside the lab: independent Gibbs beat tempering on every
denoising posterior (lessons, row 2026-10-02 "Sampling time to accuracy"),
and the layer has no exchange, so the row needs no tempering arm. Not a
third variant. **Open it**, with the text changes below.

**`am-io-budget` (held; propose dropped).** Decision it was meant to
inform: whether AM recall is I/O-bound or update-bound on the chip. P-0004
answered it with one priced example (12 reads, 32 writes, 5,120 updates per
recall at P = 8; I/O 99.3% of modelled energy; plan, "Pricing"), and the
same arithmetic over all six stage A cells at 256 to 4,096 sweeps gives I/O
shares of 85% to 99% (my arithmetic through the layer, not evidence). The
counts are deterministic given stage A's schedule, as D-0001 said; no run
changes them. What the row still promised, a projected joule per recalled
pattern against a measured CPU second, would price a design P-0004 showed
cannot be placed without a chain embedding that changes the law (probe,
item 3). A projected number for an unplaceable model is not a result under
amendment 1 (priority 3). The AM line's real next question is a placeable
memory (hidden degree at most 16), and that is 1980s literature on diluted
associative memories before it is a Thermo row; the lab has no source card
for it (`docs/knowledge/sources/` has the dense-memory cards only). Not a
third variant. **Drop**, and read before reopening.

**`e0-remaining-kernels` (open on main; D-0001 drop never applied).**
D-0001's reasons stand unchanged (`docs/research/2026-10-09-direction-review.md`,
"The change I would make first"): a fourth THRML contract, about 8
CPU-hours, informing no row, with the meta-EBM line held small by
amendment 1 (priority 4). The owner accepted that on 2026-10-10 ("doing it
later"), which the gatekeeper's default (dropped rows come out held) matches.
Because the queue never changed, P-0005 was created on the same text and
stopped as a question (card t_4439c307, no PR). The row should leave `open`
so this does not happen a third time; P-0005's first option does the same
thing. Carried in the block below as `dropped`; the owner's "later" makes
it `held` by default.

**Done rows** (`am-coupling-bits`, `planar-16-offset-ferro`,
`constraint-layer`): unchanged. The D-0001 text fixes to `constraint-layer`
are moot now that the row is done and the plan records the relaxations.

## Rubric 3: can the lab state the four-week test today?

No. For associative-memory recall, the one task with numbers:

1. *Precision and noise tolerance.* 6 bits at step = J; static ±10% jitter
   negligible (`exact_reference` and `software_simulation`,
   am-coupling-bits findings). For a design the chip cannot hold (P-0004).
2. *Sweeps, reads and writes per useful sample.* Now stateable from the
   archive and the layer's arithmetic: 12 reads and 32 writes per recall at
   P = 8, sweeps from the archived budget cells. Same caveat.
3. *Projected energy at a stated quality against a conventional baseline.*
   No number for any task.

For the hardware's stated task (denoising) there is nothing yet. The row
that closes the largest gap is `dtm-small-image`, because it is the only
one whose model is placeable by construction and whose three clauses are
all inside the row. It is weeks of work; with 22 days left on the
amendment's clock the honest forecast is that the test is not met by
2026-11-05 for a placeable model. The amendment says to revisit rather than
add process if that happens. The owner may prefer to date the clock from
the layer's merge (2026-10-14); either way, say which.

## The one thing to drop

`am-io-budget`. P-0004 answered its I/O question by arithmetic and showed
the design it would price cannot be placed. The `e0-remaining-kernels` drop
is D-0001's, already accepted by the owner and carried here only because the
queue never changed.

## Proposed queue

Changes, each in one sentence:

- **Open** `dtm-small-image` (`held` to `open`): the hold condition is met
  and it is the row that closes the largest gap.
- **Edit (text)** `dtm-small-image`, Target family, Metric and Notes: the
  model is lattice-native so it passes `require_placement` with nothing
  relaxed; bits run under a declared `absolute` or `split` cap rule, not
  "step = J", which is undefined for trained weights; the probe settles the
  I/O path per sample first, because under the sealed profile it sets the
  projected joule; the exact reference's graph is named; no tempering arms.
- **Drop** `am-io-budget` (status `dropped`, text kept, note added): its
  I/O half is answered by P-0004's pricing and its design is unplaceable.
- **Drop** `e0-remaining-kernels` (status `dropped`, text kept, note
  added): D-0001's drop, accepted by the owner on 2026-10-10 and never
  applied; P-0005 is the symptom.
- **No reorder** beyond what the status changes imply; done rows stay in
  their place.
- **No new row.** The placeable-memory question needs a source card before
  it needs a row; the denoiser's design questions belong inside the
  `dtm-small-image` probe.

```queue
| `am-coupling-bits` | done | How does associative-memory recall fall with coupling precision and per-site beta noise? | Stage A's bias-binary design with 4-, 6- and 8-bit couplings from the M5b codebook and ±10% per-site beta jitter | Recall against coupling bits and beta jitter, using stage A's exact references | Produces a spec number: the coupling bits needed to keep recall within tolerance. |
| `planar-16-offset-ferro` | done | Do the planar Ising scaling allocations (PR #92) hold on the hardware-shaped graph? | Zero-field ferromagnetic Ising on the planar subgraph of the 16-offset lattice; the thin five-replica ladder; the exact Kac-Ward reference imported from `planar_ising_scaling` | Which arms qualify, and at what sweep budget, against the exact reference | Known-good regime that confirms #92 on the chip's graph. Ferro targets only; the frustrated-grid track is reserved for the owner's annealing study. |
| `constraint-layer` | done | Does one study-local constraint layer reproduce the archived placement, rounding and finite-sweep contracts bitwise, and price updates, reads, writes and clamp changes separately? | A `thermo_lab.constraints` module that places any pairwise model on the 16-offset lattice (M5c's offsets and placement, or the planar subgraph used by `planar-16-offset-ferro`), quantizes couplings and biases with M5b's `round_parameters` codebook under a declared cap (default: step = J, the `coupling` cap `am-coupling-bits` found sufficient at 6 bits), applies static per-site beta jitter, runs chromatic block Gibbs only with no replica exchange, and counts updates, node reads, full SRAM writes and clamp changes into the existing `Z1HardwareProfile` | Bitwise agreement with the M5c placement archive, the M5b rounding archive, the `am-coupling-bits` quantized recall at 6 bits and the E0 stage B inner-K law through the layer; one priced example per constraint with the relaxed-constraint flag recorded in the hashed request | A contract study, not a sampler study: short, exact, CPU. Every later row runs through the layer by default and states which constraints it relaxed (charter amendment 1, rule 8). Reuse only; no edits to hash-bound sources. Built as engineering: [plan and As built](research/2026-10-13-constraint-layer-plan.md). |
| `dtm-small-image` | open | At what task quality does a small chained-EBM denoiser run under the constraint layer, as a function of coupling bits, beta jitter, sweeps per step and reads and writes per sample, and what does it cost against CPU Gibbs on the same model and a small conventional model of the same quality? | A two- to four-step discrete denoising model (the architecture in the DTM source card) defined on the 16-offset lattice from the start (sites are lattice sites, edges are lattice offsets, two colours) so that it passes `require_placement` with nothing relaxed; on a small binary image set: 8 x 8 to 16 x 16 binarized digits or clothing, or a synthetic set with an exact likelihood; trained in software, inference under `constraint-layer` at 4, 6 and 8 bits under a cap rule the probe declares (`absolute` or `split`; `coupling` is undefined for trained weights) and 0 and 10 percent jitter; baselines inside the row | Held-out reconstruction error and log-likelihood where tractable, per projected joule and per measured CPU second, against the baselines; the bits, jitter, sweeps and reads and writes per sample (under the declared I/O path) at which task quality stays within tolerance | Opened by D-0002 now that `constraint-layer` is done. The lab's first task-level benchmark (amendment priority 2). Baselines are part of the row (rule 10), not a separate row. The probe settles these fixed choices before any protocol: (1) the I/O path per sample (per-step host readback and rewrite, or on-chip chaining), because the sealed profile prices a clamp change as a full SRAM write at about 21,665 updates and I/O was 98 to 99.996 percent of energy in every example the layer priced, so this choice, not sweeps or bits, sets the projected joule; (2) which graph the exact reference covers (a nearest-neighbour grid of width at most 24 has a transfer-matrix likelihood; the long offsets do not); (3) whether a synthetic exact-likelihood set or a real binarized set is first, synthetic first if both fit; and it sizes training cost. No tempering arms: the layer has no exchange, and independent Gibbs won on denoising posteriors (lessons, 2026-10-02). Declare everything the plan's "What a later row must declare" lists. |
| `am-io-budget` | dropped | How many node reads, full SRAM writes and clamp changes does one recalled pattern cost in the bias associative-memory design under the constraint layer, and how does that I/O compare with its update count? | Stage A's `bias` design at 6-bit couplings with step = J (the `am-coupling-bits` spec), cue write and readout counted through `constraint-layer`, Z1 pricing | Reads, writes, clamps and updates per recalled pattern at stage A's recall tolerance; the I/O share of projected energy | Dropped by D-0002. The layer's priced example answers the question (12 reads, 32 writes, 5,120 updates per recall at P = 8; I/O 99.3 percent of modelled energy; the same arithmetic holds across stage A's cells), and the `bias` design cannot be placed (hidden degree 24, visible up to 128 against 16), so a projected joule for it prices a machine that does not exist. A placeable memory is a different design; read the diluted-memory literature and add a source card before opening a row for it. |
| `e0-remaining-kernels` | dropped | Does THRML match the archived exact inner-K law for every compiled M5a kernel, as E0 stage B showed for one? | The other 59 compiled M5a kernels (reading B) at K = 1, 2 and 4, with stage B's contract, tolerances and controls | Cells passing on output rate and joint, per kernel; control rejections | Pure replication. Stage B took about 11 CPU-minutes for one kernel, so all 59 is roughly 11 CPU-hours before parallelism; the probe sizes it, and a run over 8 CPU-hours needs the owner's say-so. Last in the order: a contract worth having, low marginal information; idle-time work. Dropped by D-0001 (owner, 2026-10-10: later); the change never reached the queue and the row was re-proposed as P-0005. |
```

## Evidence classes

Nothing in this review is new evidence. The I/O arithmetic quoted for the
denoiser and the six AM cells was computed through the merged
`thermo_lab.constraints.price` with the sealed profile's constants; it is a
`calibrated_projection` of hypothetical counts and is cited only to show
which fixed choice sets the number. Every other figure is quoted from the
cited findings, plan, probe, lessons rows or source cards with its original
label. No hardware claim.
