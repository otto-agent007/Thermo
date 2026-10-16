# DTM baseline headroom: summary

**Outcome: negative. The ladder wins all three candidate families, in every
seed. The operating point did not resolve in any of the nine seeds.** Under
rule 6 every seed verdict is marked `operating_point_unresolved`, and no
rule falls back to another arm. A reader should not use P-0006's point
(4 bits, 10 percent jitter, K = 64, T = 2, continuation) on these families.
No operating point is resolved here, so there is none to recommend. The
verdict is the same at the float K = 1024 reference (secondary check, below):
the cheapest rung, a 7 x 7 logistic of 3,136 MACs, matches the DTM in all
nine seeds. This is the line's second ladder-at-least-as-good result. Under
the protocol it goes to the owner as a premise question, not a third variant.

- **Headroom exists (stage 1).** Three of six L = 8 families are candidates.
  No rung below X = 320,826 MACs per image reaches Bayes + 0.01 there:
  - `t4_h05_q035`: Bayes 0.216; reached first by 1.74e6 MACs.
  - `smooth8_h05_q035`: Bayes 0.219; reached first by 3.43e6 MACs.
  - `smooth16_h05_q035`: Bayes 0.230; no rung reaches it.
  - Quadrant parity and corner-key are not candidates: the 8-wide pooling
    net (243,200 MACs) reaches Bayes + 0.01 on both.
  - The anchor control passed: the 7 x 7 logistic is 0.004 above Bayes.
- **The DTM does not use the headroom (stage 2).** At the operating point
  it is 0.078 to 0.147 above Bayes, which is 0.04 to 0.12 worse than the
  7 x 7 logistic. On the two smooth families it is near or worse than the
  noisy input (0.350) in five of six seeds.
- **Why unresolved.** In all nine seeds the continuation control is not
  within 0.01 of the random-start control. The random start is better by
  0.008 to 0.107, and every paired interval extends past -0.01. In eight
  seeds the interval of (operating point - float K = 1024) also reaches
  past +0.01. Only `smooth16` seed 2 passes that check.
- **Reads and writes per sample** (the headline under the queue's standing
  rules): 128 node reads, 128 full writes, 24,576 updates. A 16-sample
  estimate is 2,048 reads, 2,048 writes and 393,216 updates.
- **Projected joule per estimate** (`calibrated_projection`, sealed Z1
  profile): the band runs from 3.2e-7 J at the write constant down to
  9.7e-9 J with clamp loads at the read constant.
- **The baseline, priced per MAC** (`calibrated_projection`): the 7 x 7
  logistic at 3,136 MACs costs 3.1e-10, 3.1e-9 or 3.1e-8 J per image at 0.1,
  1 or 10 pJ per MAC. The DTM would break even only above 3.1 pJ per MAC
  (clamp-read end) or 102 pJ per MAC (write end), and even then it is less
  accurate.
- **CPU seconds per estimate** (software on this box): THRML operating
  point 0.020, NumPy Gibbs 0.012, exact posterior 0.002 to 0.007, ladder
  rungs 7e-7 to 1.4e-4.
- **Integrity.** `dtm_baseline_headroom_complete`. The run had 6 families,
  270 ladder fits, 3 candidates and 36 DTM cells. Preflight, placement,
  recomputed references, the anchor control, the prefix check and the full
  replay all passed.
- **Run.** One attempt, no resume. It ran 2 h 09 min wall on 3 workers
  (08:54 to 11:04 UTC) and used just over 6 CPU-hours. Commit a394916, clean
  tree. The calibration projected 2.24 CPU-hours, so it underestimated
  by 2.7x (see findings).

Evidence classes:

- Exact reference: the mixtures, the posteriors and the Bayes floors
  (`exact_reference`, float64).
- The ladder (JAX float32), the DTM runs and the NumPy controls:
  `software_simulation`, CPU.
- Joules on both sides: `calibrated_projection`.

There is no hardware claim.

Details: [findings.md](findings.md). Figure: `error-versus-macs.png`
(regenerate with `plot.py`).
