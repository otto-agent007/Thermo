# QEC research record: November 13–14, 2026

**Retrospective documentation of owner-authorized work, not a new protocol or
permission to run.** This consolidates the QEC work performed outside the live
THERMES checkout. It does not retrospectively promote probes into recorded
production studies, accept a merge, open a scheduler row, or establish a hardware
advantage. The owner asked to preserve this work in Thermo on November 14.

The foundational algorithm goal supplements the representation/architecture
agenda. This sequence asks whether logical-class probability estimation and
selective extra computation improve error correction over strong conventional
decoders. No thermodynamic device was used. Original plans, outcomes, failed
methods, negative findings and execution repairs are retained rather than only
successful examples.

## Current answer

- The tested decision-aware rounding rule **did not beat nearest rounding** on
  stable coefficient precision: both required 3 bits on the tiny held-out family.
- Correlated matching improved on ordinary matching in the d5 benchmark at 1%
  and 2% noise, but our probabilistic/list refinements have **not established an
  advantage over Tesseract** or a useful computational-cost advantage.
- AIS could be confidently wrong while reporting high effective sample size.
  Finding important explanations and estimating their probability accurately
  are separate problems. Later candidate coverage did not rescue a validated
  confidence gate.
- The latest d5 close-case selector failed its intended role. Of six list-selected
  cases, five were demonstrably not close and one remained numerically unresolved.
- Outward-rounded high precision worked on the new tiny numerical fixtures. It
  **has not been ported to the d5 sliced tensor backend** and does not resolve its
  three numerically unresolved cases.

## Model and evidence distinctions

The initial d3 studies had perfect syndrome measurements. The X-only studies
enumerated 512 physical error patterns; the full-Pauli study enumerated 262,144
patterns over 256 syndromes and four logical classes. These are finite numerical
`exact_reference` calculations, not real-device observations.

The d5 benchmark and its saved-shot follow-ups used **phenomenological noise**:
independent data depolarization immediately before each of five rounds and
independent ancilla-record flips, with ideal gates, resets and final data readout.
The three data/readout rates were 0.5%, 1% and 2%. Separate X-memory and Z-memory
experiments measure separate logical observables, **not joint logical-Pauli
failure**. Earlier conversational shorthand calling this circuit-level noise
was incorrect. No gate-level circuit-noise comparison was performed here.

Sampled-shot decisions are `software_simulation`. A model's most probable logical
class can still be wrong on a realized shot; the best possible decoder does not
make every stochastic outcome predictable. Sliced references use the saved
Bernoulli detector-error model, retaining correlated separator mechanisms as
single events; their finite-precision certificates are conditional on documented
arithmetic assumptions. They are not unconditional physical truth.

## Research sequence and what each step changed

| Work | Scope and result | Consequence / limitation |
| --- | --- | --- |
| Decision-aware rounding | 6 development and 6 held-out d3 X-sector noise instances, 2–8 bits, nearest/diffusion/decision/expected-stochastic rules. Decision and nearest both need 3 stable bits. At 2 bits decision is worse on 5 held instances and equal on 1. | Negative under the one-bit-improvement rule. No practical precision, latency or energy advantage; do not retune against those held cases. [Original report](qec-work-2026-11-14/reports/rounding.txt). |
| d3 probabilistic refinement probe | 2,304 development cells. Baseline is already exact on 5/6 fixtures; remaining absolute logical-risk headroom is only 0.0001281503. Larger AIS batches recover the small correction more reliably; small batches can harm. | Mechanism-only probe; no held-out efficacy study. Stop the full comparison on that tiny family for inadequate headroom. [Report](qec-work-2026-11-14/reports/d3-refinement.txt). |
| d3 full-Pauli headroom | Exhaustive four-logical-class risks at 2%, 5%, 8% symmetric depolarizing noise. Better matching is ordinary, not correlated, on all three cases. Oracle headroom is about 15.3%, 13.4%, 11.6% relative; the absolute floor passes at 5% and 8%. | Larger oracle opportunity, not a practical refiner result. The gate admitted preparation of d5, not automatic execution. [Report](qec-work-2026-11-14/reports/d3-full-pauli.txt), [audit](qec-work-2026-11-14/reports/d3-full-pauli-audit.txt). |
| d5 baseline | 300,000 shots, 50,000 per memory/rate cell. Correlated matching improves at 1% and 2% in both memories under paired simultaneous intervals. At 0.5% differences are unresolved. | This reverses the tiny-code baseline ordering. No stronger reference was included in this first baseline; optimal-decoder headroom remained unknown. [Report](qec-work-2026-11-14/reports/d5-baseline.txt). |
| d5 refiner pilot | 192 saved cases: Tesseract and belief matching 0 observed errors, correlated matching 1, refinement 2; refinement has 0 repairs and 1 harm relative to correlated matching and falls back on 161/192. | Negative pilot, not a population ranking. A validated strong comparison should have accompanied the baseline sooner. |
| AIS harmful-case diagnosis | A valid known configuration alone has at least 1.32e14 times the mass of AIS's purported whole-class estimate. High ESS did not detect failure. | Missing relevant probability, not evidence that the valid configuration is impossible. No established normalization bug; concentration and correlated particles were implicated. |
| Multimode proposal prototype | Same previously examined case improved; 3/8 mass estimates were still vetoed by known-mass checks. | Development on a reused case, not held-out efficacy or a reliable uncertainty gate. |
| Exact-fixture calibration | 12 synthetic fibers, 768 runs grouped into 192 four-replicate summaries. No candidate heuristic confidence gate met its frozen acceptance rule. | Reject the gate. High visited mass and high ESS do not certify an accurate estimate. |
| Seed-plus-residual estimator | Same reused exact fixtures: MSE improved on 8/12 and worsened on 4/12; inaccurate group estimates fell from 50/192 to 37/192. | Known seed mass is included structurally, but unknown residual coverage/variance remain concerns. Not fresh held-out evidence. |
| One-case sliced reference | A bounded untruncated signed contraction orders the examined classes under its numerical assumptions. Four candidates capture about 91.12–98.93% of winning-class mass; losing-class relative coverage remains unresolved. | Establishes tractability of a selected reference, not robust decoding or a general exact certificate. |
| Six-case candidate coverage | Completed across six settings after separately authorized recovery and retry. Sixteen candidates capture at least 98.8% of winning-class mass; strong decoders agree on these examples. | Feasibility/coverage, no performance advantage. Preserve interrupted partials and charge authorized retries separately; see execution notes below. |
| Seven-case targeted diagnostic | 6 random controls plus 1 decoder disagreement after a blind screen. Candidate lists agree with reference-MAP 7/7 but observed outcomes 6/7; 0 repairs and 1 harm versus correlated/belief matching. Tesseract makes identical choices. | MAP agreement is not realized correctness. Close-case screen produced no eligible cases due a software guard bug; one selected disagreement cannot estimate population benefit. |
| Held-out screen and 14 references | 3,072 fresh shots screened; 192 random panel shots plus 4 enriched disagreements. All methods make 1 error on random192. In selected4, list/Tesseract make1, belief matching4, correlated2. Fourteen references yield11 secure winners and3 unresolved; five of6 list-close choices are unambiguous. | No list advantage over Tesseract. List repairs3 BM errors but introduces1 harm versus correlated matching in the enriched group. Fresh shots include repeated syndrome patterns; selection and fallbacks preclude a population claim. |
| Tiny precision/screen diagnostic | 10 stress models plus6 development/6 held-out generic parity-event models;976 mass and450 posterior enclosures independently checked. Rare-syndrome fixtures require64 or128 decimal digits. On48 held-out syndromes, top4 flags both truly close syndromes plus1false positive. | Arithmetic mechanism passes tiny fixtures, not d5. Top4 probability-weighted precision is only34.4%; exact top-k surrogates are not Tesseract/BP or a practical solver. [Report](qec-work-2026-11-14/reports/precision.txt), [audit](qec-work-2026-11-14/reports/precision-audit.txt). |

### Original middle-study records

- [d5 pilot](qec-work-2026-11-14/source-records/qec-extension/PILOT_REPORT.md.txt)
- [AIS harm diagnosis](qec-work-2026-11-14/source-records/qec-diagnostics/REPORT.md.txt)
- [multimode prototype](qec-work-2026-11-14/source-records/qec-prototype/REPORT.md.txt)
- [exact calibration](qec-work-2026-11-14/source-records/qec-calibration/REPORT.md.txt)
- [residual estimator](qec-work-2026-11-14/source-records/qec-residual/REPORT.md.txt)
- [one-case sliced reference](qec-work-2026-11-14/source-records/qec-reference/FULL_REFERENCE_REPORT.md.txt)
- [six-case recovery](qec-work-2026-11-14/source-records/qec-six-case-recovery/RECOVERY_OUTCOME.md.txt)
- [separately authorized sixth retry](qec-work-2026-11-14/source-records/qec-six-case-recovery/RETRY_OUTCOME.md.txt)
- [targeted seven](qec-work-2026-11-14/source-records/qec-targeted-study/FINAL_REPORT.md.txt)
- [held-out fourteen](qec-work-2026-11-14/source-records/qec-close-screen/FINAL_REPORT.md.txt)

Protocols, reviews, bounded summaries and recovery tests accompany these reports
in the same source-record directories. Their historical status messages do not
supersede the completed reports and later authorized retry outcomes.

## Important failed assumptions and corrections

### Confidence and selection

The first close-case screen incorrectly rejected infinite BP scores associated
with legitimately forced fault variables even when the logical-score entries
were finite. The corrected guard fixes that implementation defect; it does not
make the BP score a reliable closeness measure. The replacement truncated-list
score also misranked the subsequent d5 examples. Do not equate finite values,
small top-k margins, high ESS, or agreement between candidate estimates with
verified full-class probabilities.

The 192-case held-out panel used fallback on 58 cases. The four enriched
cases were selected for decoder disagreement. The random and enriched groups
must remain separate; repairs/harms and timeouts/fallbacks belong in denominators.
The 14-reference subset is not an iid sample for estimating decoder error rates.

### Precision and scientific interpretation

The signed contraction can suffer severe cancellation. A numerical interval must
enclose every conversion and operation, and posterior normalization needs a
strictly positive syndrome-mass lower bound. An unknown result is preferable to
a false certificate. Narrow posterior width and a certified winning class are
different goals: in the tiny near-tie fixture, width passes at 16 digits but the
winner only resolves at 32; a rare competing-explanation fixture needs 64 digits
for its winner despite a narrow interval at 32.

The tiny implementation uses a **full Fourier sum**, not a port of the existing
sliced tensor contraction. Its fixed 16/32/64/128-digit sweep does not measure
adaptive-precision runtime. Exact rational probabilities describe specified
inputs, not a perfectly calibrated physical device. Higher precision fixes
neither missing modes nor noise-model mismatch.

### Execution and accounting

A historical long-lived process stopped without an exit receipt. The evidence
did not establish an out-of-memory or CPU-limit cause, so do not call the event
an OOM. The execution design initially lacked reliable supervision/resumption.
It was replaced by short bounded jobs, source/input authentication, persistent
CPU ledgers, leases/locks, parent-death handling, exit receipts and checkpoint
reuse. Fault injections found and corrected lease/publication/retry defects.

The original interrupted partial remains evidence of an incomplete attempt.
Later four-case recovery and the sixth-case retry had separate owner approvals.
The combined six-case charges, including that later retry, total **3,638.384 CPU
seconds**, not a claim that everything fit the earlier 3,600-second allowance.
The held-out14 study completed 748 batch receipts without interruption, charging
**7,810.044 CPU seconds** across screening, candidate lists and references. The
latest tiny diagnostic charged **24.356 CPU seconds**, including conservative
runner accounting and completed-run replay; its measured worker cost is much
smaller. These are workflow costs, not per-shot tail latency or device energy.

## Reproducibility and coverage

[Evidence inventory and replay instructions](qec-work-2026-11-14/README.md)
identify original archive names, source reports/protocols, compressed provenance
and retained limitations. The original tiny-diagnostic archive is bundled
unchanged, with a portable independent numerical replay test. Other studies'
complete original archives remain the authoritative reproduction sources;
compact imported records do not substitute for missing traces or environments.
No archived evaluator was edited or retroactively rehashed to make a check pass.

The three earlier planning documents are retained as historical plans:
[rounding](qec-work-2026-11-14/plans/2026-11-13-qec-decision-aware-rounding-plan.txt),
[selective refinement](qec-work-2026-11-14/plans/2026-11-13-qec-selective-probabilistic-refinement-plan.txt),
and [observable-specific sampler reliability](qec-work-2026-11-14/plans/2026-11-13-observable-specific-sampler-reliability-plan.txt).
Their original “held/not run” labels describe planning time, not the subsequent
QEC outcomes above. The separate observable-specific Ising study remains planning
only. Stale draft queue/charter text was not copied over the current live queue.

## What remains unresolved

1. Portability and cost of reliable interval arithmetic in the actual d5 tensor
   backend. The three old unresolved cases were not rerun.
2. A held-out close-case or expected-benefit gate that generalizes beyond the
   generator/cases on which it was developed.
3. Practical accuracy/cost superiority over tuned matching, Tesseract and other
   relevant baselines on sufficiently powered, identically modeled tests.
4. Robustness to device-model mismatch, drift or true circuit-level faults.
5. Any mapping to thermodynamic hardware or useful latency/energy advantage.

No larger efficacy study, new benchmark family, new hardware, or autonomous queue
entry follows from this documentation import. Future work needs a separately
bounded question; this record preserves the negative findings so they are not
repeated unknowingly.
