# Direction review D-0009

2026-11-08. Main commit 682884a4943b7d8d24f6972df5d0a7480e56e16d. THERMES-DIRECTOR.
Label: advice, not a decision. The owner edits the queue; this file proposes.

Trigger: one loop result accepted since D-0008
(`docs/research/2026-11-04-direction-review.md`, PR #144, merged): P-0010 on
row `rounding-rule-transfer`, PR #146, merged 2026-11-05 as a docs-only
probe, note `docs/research/2026-11-05-rounding-rule-transfer-probe.md`,
lessons row 2026-11-05, source card
`docs/knowledge/sources/error-diffusion-quantization-bias.md`. PR #145
adopted D-0008's queue change the same day. Nothing is in flight. The queue
at 682884a has twelve rows and no open row. Read against charter
amendments 1 and 2 (`PROJECT_CHARTER.md`), the agenda
(`docs/research/2026-10-20-lab-research-agenda.md`), the queue, the lessons
register, the last three merged studies (P-0008, P-0009 and P-0007 by date)
and all 27 source cards.

## The change I would make first

No new row. Put the 2026-11-11 decision to the owner on the premise, not on
a next variant: does the lab keep producing exact representation
requirements with no task attached (P-0009, P-0010), or park the loop until
a task with demonstrated digital cost headroom and a resident model is
named? D-0007 and D-0008 both named this as the consequential decision
(`2026-11-02-direction-review.md`, "Rubric 3";
`2026-11-04-direction-review.md`, "Rubric 3"). P-0010 is the last cheap
question in the representation line, and it closed negative. The agenda's
own instruction for an empty queue is not to fill it to keep the loop
running (`2026-10-20-lab-research-agenda.md`, "Immediate disposition").

The thing to drop is **any further rounding or cap variant on P-0008's
placed tie-limit model**: per-kind caps, loss-aware rounding (AdaRound or
GPTQ style) or a spatial class rule. Reasons under "The one thing to drop".

## Verdicts

**P-0010 `rounding-rule-transfer`: negative for transfer, exact, correctly
tiered, and it answers its row.** On P-0008's archived placed tie-limit model
class error diffusion cuts the 4-bit MPM gap from 0.0752 to 0.0150, still
above 0.01 with the per-image bootstrap interval (0.0111 to 0.0191) above
it too; both rules first pass at 5 bits (nearest 0.0029, diffusion 0.0041)
and pass at 6 and 8 (probe, "Result"). The pre-registered falsifier fired
(queue row `rounding-rule-transfer`, notes). Three things to read with it:

1. **The mechanism prediction was half right, and the half that failed is
   the general lesson.** Every large coefficient class did round coherently
   under nearest, and diffusion brought every class sum within half a step,
   as the row predicted. That fixed the copy couplings (one-kind ablation
   0.0452 to 0.0028) and copy biases (0.0092 to -0.0001) because they act
   through sums. It left the 112 grid couplings at 0.0118 alone, because an
   Ising grid responds to each edge's J, not to the class mean; diffused,
   the grid takes two levels, 0.25 and 0.5, against J = 0.35 (probe,
   "Mechanism"). Class diffusion helps where the observable is near-linear
   in the class sum and nowhere else. That scopes P-0009's rule to
   field-dominated models (lessons 2026-11-05), and it was predictable from
   the source card: bias correction fixes a mean output shift, not
   per-weight error (`error-diffusion-quantization-bias.md`, C2 and
   "Relevance", second bullet). The lab paid six CPU-minutes to confirm it,
   which is the right price.
2. **The 4-bit miss on this model is a cap problem, not a rounding
   problem.** The archived `split` rule has one coupling cap, set by the tie
   w* = 1.75, so the 4-bit coupling step is 0.25 and J = 0.35 sits at 1.4
   steps with no codebook level within 0.1 (probe, "Fixed choices", Caps).
   A 5x dynamic range between the strongest and the weakest coupling costs
   this model more than two bits. That is a representation observation
   worth one sentence in the lessons register, which the 2026-11-05 entry
   already has.
3. **The archived "6 under nearest" was a grid artefact.** P-0008 never ran
   5 bits; both rules pass there. P-0010's own grid has a hole at 7 (bits 4,
   5, 6, 8, as the row specified), so its "5 is stable" rests on three
   points. The gaps at 6 and 8 are 0.0003 to 0.0025, far below 0.01, so a
   7-bit failure is unlikely, but P-0009 saw nearest pass at 6 and 7 and
   fail at 8 on the pore (P-0009 findings, "Design numbers"). D-0008 wrote
   that grid; the one-minute 7-bit cell is the fix, if anyone cares about
   this model's count, and nobody should (below).

Is it already known? The technique and its limit are (source card, C1 to
C4 and "Limits"). What is the lab's own is the exact quantification on a
model with mixed coefficient kinds: diffusion removes sum-coupled error and
not edge-coupled error, and the cap rule sets the floor. The card now
exists, as D-0006 to D-0008 asked; the probe cites it. The lessons entry
D-0006 asked for on the eligibility probe's unpowered read allocation is
still absent (`docs/knowledge/lessons.md`'s only adsorption entry,
2026-11-02, is on rounding).

**PR #145: process, correct.** Nothing further.

## Rubric 1: P-0010

- **Design number or probe with a study's process?** A probe with a
  probe's process: 307 CPU-s plus a one-minute post-hoc spread check, a
  committed script with its fixed choices in the docstring, a note, a
  lessons row and a docs-only PR; no protocol, runner, gate or roadmap row
  (probe, "What ran", "Not done"). That is amendment 1's tier 1 used as
  written, for the first time since the amendment on a loop row. The number
  it produced, 5 bits under either rule, is `exact_reference` on a tie-limit
  model with no sampled operating point (P-0008 findings, "Sampled stage";
  stop (b) fired on finite tie strength). Nobody designs against it. The
  decision it informs is answered: `thermo_lab.constraints.quantize` keeps
  nearest rounding as its default, and no archived bit requirement needs
  re-reading except P-0008's, which is 5 not 6 (probe, "Reading" 3).
- **Constraints.** Applied: archived placement and `split` codebook with
  clipping. Relaxed by the row's design: jitter (none) and sampling (none).
  Rounding is a host compile step and needs no hardware feature. Rule 8
  satisfied by statement (probe, first paragraph).
- **Accounting.** Stated: 307 CPU-s, one process, about 10 s per exact row;
  not a task row, so no joule and no MAC count. Rule 9 satisfied by
  declaration.
- **Expectations.** The row's pre-registered prediction (class diffusion
  passes at 4 or 5 bits) missed at 4 and held at 5; the falsifier fired.
  No queue row assumed the opposite: the only rows that read a bit
  requirement are `done`, and their numbers stand. The source card's
  prediction of where diffusion fails held exactly.

## Rubric 2: the queue rows

Twelve rows, none open: eight `done`, three `dropped`, one `held`. Nothing
depends on re-running any of them. The three worth a line:

### `rounding-rule-transfer` (done)

- **Decision.** Answered: nearest stays the default; class diffusion is a
  per-model compile option for models whose large classes act through sums,
  checked exactly before use (probe, "Reading" 3).
- **Cheapest probe.** Ran, and it was the whole row.
- **Source card.** `error-diffusion-quantization-bias.md` now covers the
  technique and predicted the negative.
- **Third variant.** This is the line's second result (P-0009 positive on
  the pore, P-0010 negative for transfer), so no two-failure rule fires. The
  natural third variant, a loss-aware or per-kind-cap rule on the same model,
  is the drop below.

### `adsorption-precision-requirement` (done)

- P-0010's exploration closed D-0008's first caveat: the pore needs exactly
  4 field bits under class diffusion, not "at most 4"; 2 and 3 fail (probe,
  "Adsorption pore at 2 and 3 bits"). The `completion.json` and findings
  still say "at most 4" by construction; the note is the correction. No
  action; a reader of the findings should be pointed at the note.

### `native-adsorption-eligibility` (held)

- Unchanged from D-0008: held, no outcome unholds it, the powered rerun was
  D-0008's drop. P-0010 touched nothing here.

The other nine rows are as D-0007 covered them (`2026-11-02-direction-review.md`,
"Rubric 2").

## Rubric 3: can the lab state the four-week test today?

Due 2026-11-11, three days out. Yes, with clause 1 updated by one line:

1. *Precision and noise tolerance.* 4 bits under `split` on the local
   denoising family (P-0006); 6 bits with fields in J for the bias memory
   (am-coupling-bits); 5 bits, not 6, for the placed tie-limit latent under
   nearest or class diffusion, 4 fails under both (P-0010, correcting
   P-0008's grid); exactly 4 field bits under class diffusion and 9 under
   nearest for the 144-site lattice gas, exact, zero jitter (P-0009,
   P-0010); static 10 percent beta jitter negligible in four studies. The
   only precision figure Extropic has published for a model on Z1 is 4-bit
   weights (`extropic-z1t.md`, C2). Of the lab's models, the pore reaches it
   with a compile-time rule and the local denoiser reaches it natively; the
   tied-copy latent and the bias memory do not.
2. *Sweeps, reads and writes per useful sample.* Unchanged from D-0007:
   K = 64, 64 reads, 64 writes on the local family (P-0006); no useful
   sample on any family with a global latent (P-0007, P-0008); the
   adsorption pore priced a trajectory that certified nothing.
3. *Projected energy at stated quality against a conventional baseline.*
   Unchanged: cost-competitive above 1 pJ per MAC, 0.002 short on quality,
   no task wins on both (D-0005, "Owner decision").

Largest gap: clause 3, for any task with a quality pass. No row closes it;
there is no row. The honest 2026-11-11 statement is D-0007's, with one
addition: the amendment produced every number it asked for, under the
constraints, with accounting; no task the lab has run wins on cost and
quality together; and the representation line, run cheaply as the
amendment intended, has now produced one transferable-looking rule and
shown its scope in two probes for under ten CPU-minutes. That is an
argument for keeping any representation work at probe cost, not for
expecting it to close clause 3.

## Rubric 4: representation, architecture, useful computation

- **Established (exact, two models).** A compile-time rounding rule moves a
  bit requirement only where the observable is near-linear in a class sum:
  five bits on the pore (fields on 128 wall sites), zero on the tie-limit
  model's grid couplings (P-0009 findings, "Findings" 1; P-0010, "Mechanism").
- **Challenged.** The reading that precision requirements are set by the
  rounding rule. On the tie-limit model the floor is set by the cap rule:
  one coupling scale shared between a tie of 1.75 and a grid coupling of
  0.35 (P-0010, "Fixed choices", Caps). Intended model, constrained
  representation and estimator are separated cleanly here because the
  study is exact; nothing about the actual update process is tested.
- **Unsupported generalization to avoid.** "Class diffusion is a general
  lever" (now refuted) and its mirror, "rounding rules do not matter" (the
  pore says they do by five bits).
- **User and decision.** The constraint layer's author (default rule,
  answered) and anyone writing a codebook for a model with mixed coupling
  scales (declare per-kind caps or budget the dynamic range in bits).
- **Strongest digital baseline.** Not applicable; no task.
- **Hardware requirement.** A conditional one, not a result: if Z1 exposes
  one coupling scale per core or chip, a model that mixes a strong tie with
  weak couplings spends log2 of their ratio in bits before rounding starts
  (about 2.3 bits at 5x). Z1's coupling precision and scale granularity are
  on no card (`extropic-z1t.md` gives weight bits for a model, not the
  chip). That is a question for a source card, not a row, and only if a
  model anyone will run has that structure.
- **Next question, falsifier, cheapest check.** None proposed. The
  premise question above goes to the owner instead.

## The one thing to drop

**Any further rounding or cap variant on P-0008's placed tie-limit model**:
per-kind caps (P-0010, "Reading" 4), loss-aware rounding (source card,
"Relevance", third bullet: "needs its own row") or a spatial or per-kind
class rule (P-0010, "Fixed choices", Class definition). Three reasons:

1. The model has no operating point to design for. P-0008's stop (b) fired
   on finite tie strength at 8 bits, where precision was not binding; the
   float K = 4096 reference missed by the same 0.012 (P-0008 findings,
   "Rule 7"). Lowering its exact tie-limit requirement from 5 bits to 4
   changes nothing a sampler would run.
2. The owner closed the denoising line with no further premise test, and
   amendment 2 keeps that closure ("Commitments"). A row on this model's
   codebook is a denoising row with the task removed.
3. The gain is bounded and known: the grid class alone leaves 0.0118 at 4
   bits (P-0010, ablation table), so any rule that fixes it must place J
   within about 0.1 of 0.35 on a 0.25 step, which only a second coupling
   scale can do. That is reason 1 again, as a cap.

This also drops the am-coupling-bits re-read as a candidate: P-0010 counted
its classes and found one large coherent class whose effect enters through
each hidden unit's own log cosh, not a sum, and its 10-bit `split` figure is
an interval-width result (P-0010, "am-coupling-bits"; D-0008, "Rubric 2").

## Proposed queue

No queue change.

The queue has no open row and the four-week evaluation is three days out.
A new row now would fill the queue to keep the loop running, which the
agenda forbids. The owner's 2026-11-11 decision (continue representation
requirements at probe cost without a task, or park until a task with
digital headroom and a resident model is named) decides whether a row
follows, and which kind.

## Evidence classes

Nothing in this review is new evidence. P-0010's numbers are exploration
with `exact_reference` values, cited from its note; P-0008's and P-0009's
are from recorded studies cited by path; the dynamic-range remark is this
review's arithmetic on P-0010's reported caps, labelled analysis. No
hardware claim.
