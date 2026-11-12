# Amendment 1 four-week evaluation

2026-11-11. Main commit 8ca12ef5a3f8bba9913f49884a5d160bb2393341. THERMES.
**Evaluation, not evidence.** Every number is restated from the cited report
or note, with that source's evidence class. Nothing was run for this note.
The recommendation at the end is advice; the owner decides.

The test is charter amendment 1's
([charter](../../PROJECT_CHARTER.md), "How to tell whether this is working").
Amendment 2 kept its date and terms ("Commitments and evaluation") and asked
for a separate knowledge summary and one named decision. This note gives all
three.

## Outcome

- **Read literally, the test passes.** For one task, the lab can state the
  bits and jitter tolerance, the sweeps, reads and writes per useful
  estimate, and the projected energy at a stated quality against a
  conventional baseline.
- **What those numbers say is negative for the hardware.** At the stated
  quality the conventional model is more accurate, and cheaper everywhere
  except one corner of the pricing band. No task the lab has run wins on
  quality and cost together.
- **Recommendation: park the loop** until a task is named that shows digital
  cost headroom first. Reasons and the un-park condition are at the end.

The first two points are different kinds of outcome. The test asks whether
the lab can state the numbers, not whether the hardware wins. Direction
reviews D-0005 to D-0009 reached the same two-part answer
([D-0009](2026-11-08-direction-review.md), "Rubric 3"). A stricter reading
is available: on every family where the digital side is costly, clauses 2
and 3 have no qualifying operating point to state. Which reading counts is
the owner's call. The numbers are the same under both.

## Clause 1: coupling bits and beta-noise tolerance

| Model and scope | Bits needed | Static ±10% per-site beta jitter | Evidence class | Source |
| --- | --- | --- | --- | --- |
| **Local denoiser** (the task): L = 8, two steps, trained weights, `split` caps | 4; 3 misses | negligible | `software_simulation` | [P-0006](../experiment-reports/2026-10-14-dtm-small-image/findings.md) |
| Bias associative memory: N = 24, P up to 128; not placeable (hidden degree 24) | 6 with step = J; 10 under `split`; 12 under `full` | negligible; ±30% within 0.003 | `exact_reference` (bits), `software_simulation` (jitter) | [am-coupling-bits](../experiment-reports/2026-10-08-am-coupling-bits/findings.md) |
| Tied-copy latent for `t4`, at the tie limit, `split` caps | 5 under nearest or class diffusion; 4 fails | no effect in the sampled 8-bit arm | `exact_reference`; the 5-bit cell is exploration | [P-0008](../experiment-reports/2026-10-17-dtm-exact-posterior-lattice/findings.md), [P-0010](2026-11-05-rounding-rule-transfer-probe.md) |
| 144-site pore, fields only | 9 nearest; 4 class diffusion; 8 per-point caps | not tested | `exact_reference`; "exactly 4" is exploration | [P-0009](../experiment-reports/2026-11-02-adsorption-precision-requirement/findings.md), P-0010 |

Jitter was isolated in three studies, not the four the direction reviews
cite: P-0007 has no jitter-free arm at its operating point. Only static
per-site gain jitter was tested. Per-sweep noise and coupling-value noise
were not.

## Clause 2: sweeps, reads and writes per useful sample

| Model | Counts | Source |
| --- | --- | --- |
| **Local denoiser**, 64 sweeps per step, state carried between images | Per sample: 128 reads, 128 clamp loads priced as full writes, 24,576 updates, no free-node writes. Per 16-draw estimate, the unit that has error 0.075: 2,048 reads, 2,048 writes, 393,216 updates | P-0006; [P-0007](../experiment-reports/2026-10-16-dtm-baseline-headroom/findings.md), "Cost, both sides projected" |
| Families with a global latent (`t4`, `smooth8`, `smooth16`) | No useful sample. The trained denoiser lands 0.078 to 0.147 above Bayes, and carried state lost to a random start in 9 of 9 seeds. On `t4` no start of the hand-built model reaches Bayes + 0.01 within 1,024 sweeps | P-0007, P-0008 |
| 144-site pore | 9,216 reads, 288 writes, 589,824 updates per trajectory, which certified nothing: the ideal control was unresolved too (exploration) | [eligibility probe](2026-10-20-native-adsorption-eligibility-probe.md) |

The direction reviews quote "64 reads, 64 writes" for the denoiser. That is
per step. A two-step sample is 128 of each.

## Clause 3: projected energy at a stated quality against a baseline

Local denoising family, pixel error per image. Errors are P-0006's. Counts
and prices are P-0007's, for the same operating point and the same logistic.
Every joule is a `calibrated_projection`.

| Method | Error | Projected joules per image |
| --- | --- | --- |
| Lattice denoiser, 16-draw estimate, 4 bits, 10% jitter | 0.075 | 3.2e-7 at the sealed profile's write constant; 9.7e-9 with clamp loads at the read constant |
| 7 × 7 logistic, 114 parameters, 3,136 multiply-accumulates (MACs) | 0.071 | 3.1e-10, 3.1e-9 and 3.1e-8 at 0.1, 1 and 10 pJ per MAC |
| Bayes floor (`exact_reference`) | 0.067 | |

**Negative.** The lattice denoiser breaks even at 102 pJ per MAC at the write
end of the band, or 3.1 pJ per MAC at the clamp-read end. Inside the queue's
0.1 to 10 pJ band it is cheaper only if clamp loads cost a read and a MAC
costs more than 3.1 pJ. The logistic is more accurate at any price. D-0005
set one sample (2.0e-8 J) against the logistic. The 0.075 error belongs to
the 16-draw estimate, so the estimate is the matched unit.

Where the digital side is costly, there is no operating point to price. On
`t4` the cheapest ladder rung within Bayes + 0.01 is 1.74e6 MACs (a
243,200-MAC net misses by 0.0005). P-0008's stop (b) fired there: the
finite-tie equilibrium sits 0.0117 above Bayes against a 0.01 tolerance, a
margin inside both pair intervals. The
[thinned-draw probe](2026-10-18-dtm-exact-posterior-lattice-thinned-probe.md)
(exploration) lowered the projected cost to 1.05e-7 J per 64-draw estimate
and left the gap at 0.012. The owner's wording of 2026-10-18 stands:
cost-competitive above 1 pJ per MAC, 0.002 short on quality, no task wins on
both. On the pore, the exact digital answer took about 11 ms per point
(software timing, exploration) and no array estimate qualified.

## Missing evidence

- No hardware measurement. Every joule is a projection under the sealed
  profile, which excludes host energy, I/O latency, idle and board power.
- No published clamp-only constant. The denoiser's projected cost moves 33
  times across that band.
- No lower bound on digital cost. The ladder is not the cheapest possible
  model, and MAC pricing excludes activations and memory traffic.
- P-0008's sampler is NumPy Gibbs, not the lowered THRML program. Training
  was never under the constraints.
- The thinned-draw cost assumes one clamp load can be held and read 64
  times. The probe did not test that. It is the owner's reading of the
  profile, recorded on 2026-10-18.

## What the lab can support

Bounded claims only. None is a hardware claim or an advantage claim.

| Claim | Scope | Evidence class |
| --- | --- | --- |
| A local pairwise model keeps its float quality at 4 bits with 10% static jitter | One L = 8 family with a local posterior | `software_simulation` |
| Where a codebook places its cap sets the bit budget | Bias memory (6, 10 or 12 bits); tie-limit model, where one coupling cap shared by a tie of 1.75 and J = 0.35 costs the 4-bit pass | `exact_reference`; the second is exploration |
| A rounding rule moves a bit requirement only where the observable is close to linear in a class sum | Two models: the pore (9 to 4 bits) and the tie-limit model (no change) | `exact_reference`; the second is exploration |
| A global latent spread as tied copies holds the posterior at the tie limit but misses at the selected finite tie (w = 1.75), and stronger ties mix worse | One family, one construction, ties 1.5 to 3.0, margin 0.0017 | `exact_reference` and `software_simulation` |
| Headroom for a costly digital model is not headroom for the lattice model | Three L = 8 families; a 3,136-MAC logistic came within 0.01 of the local-latent denoiser or beat it in 9 of 9 seeds | `software_simulation` against `exact_reference` |
| Projected energy is I/O whenever each draw needs a host load | 99.1% on the denoiser; an accepted [replica exchange](../experiment-reports/2026-10-07-exchange-cost-projection/findings.md) or a schedule stage is a full write | `calibrated_projection`; the schedule-stage figure is analysis |
| Domain-wall encoding mixes under a ramped penalty on 5- and 7-spin chains, not on 31 or 127 | N = 24 memory; the bias design stays ahead | exploration ([probe](2026-11-11-domainwall-anneal-probe.md)) |

Still hypotheses, each needing an independent case:

- The amendment 2 working hypothesis itself. No tested case supports it and
  none refutes it as a family.
- Class diffusion as a general compile option (two models).
- Thinned draws from one clamp load (one model, one tie value).
- That another construction could hold a global latent in budget. P-0008 is
  one construction, not a theorem.
- Any crossover with size. Nothing above L = 8, 144 sites or N = 24 was run.

## The decision, and a recommendation

The one consequential decision: keep producing representation requirements
at probe cost with no task attached, or park the loop until a task with
demonstrated digital cost headroom and a resident model is named.

**I recommend parking.** Four reasons:

1. The open gap is clause 3's content, and a representation probe cannot
   close it. Such probes are exact and price nothing, by design.
2. The representation line has no remaining question with a design consumer.
   P-0010 answered the last one: the constraint layer keeps nearest rounding.
   Z1's coupling precision and scale granularity are on no source card
   (D-0009, "Rubric 4"), so nobody can yet design against a finer bit count.
3. In the cases with a costed digital baseline, the lattice model held its
   target only where the digital method was already cheap (a 114-parameter
   logistic; an exact transfer calculation). Where the digital method was
   costly, the lattice model missed tolerance. That is an observation on
   small models, not a law. It does say the next question has to start from
   headroom.
4. The largest cost uncertainty is the unpublished clamp constant. The lab
   cannot resolve it by running anything.

The case for continuing is cost. P-0009's rule and P-0010's scoping of it
took under ten CPU-minutes between them, and the pore result is a usable
compile rule. The cost of continuing is owner and director attention, not
compute. I still recommend parking, because a cheap answer with no consumer
changes no decision.

Parked means the queue stays empty and no row is proposed. The scheduled
gates and archive replays are not part of the loop and are unaffected.
Probes the owner asks for stay available, as with today's domain-wall probe.
Parking closes no family and no archive. The director's weekly review is a
loop function ([research loop](../research-loop.md), "Director"). Whether it
keeps firing while parked is a scheduler setting, and the owner's.

Un-park when a named task passes a desk screen, in this order, before any
code. The screen is the agenda's existing list
([agenda](2026-10-20-lab-research-agenda.md), "How to choose the next
question"), not a new stage:

1. The strongest practical digital method is costed at the size the array
   can hold, including exact structure-aware methods, and leaves headroom.
2. The model places natively, with no tied copies or global latent, and its
   coefficient ratio fits the cap at a stated bit count.
3. A user, an observable and a justified tolerance are named.

The [codon-optimization card](../knowledge/sources/extropic-codon-optimization.md)
shows what the screen does to a published task. Once reflashes are priced
under the sealed profile, the claimed margin over the idealized GPU bound
falls from 160 times to about 2.5 (the card's arithmetic, analysis). That is
an illustration, not a proposed task.

What would change this recommendation: a hardware team that will design
against bit counts makes continuing worth its cost. Published clamp, core
I/O or coupling-precision constants justify re-pricing the archives, with no
new runs.

On the amendment itself: every loop study since 2026-10-14 ran under
declared constraints and named a task or a design number, and every task row
carried a conventional baseline. The drift it described, sampler comparisons
on random instances, has not recurred.
Amendment 2 was already the owner's revisit of the task ordering. This note
proposes no further process.

## Not done

No run, no queue edit, no change to any frozen protocol, threshold or
archive. No new task is proposed. The two unselected families in the
[October 20 memo](2026-10-20-next-task-recommendation.md) stay unselected;
each carries a stated digital-headroom objection.
