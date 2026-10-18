# DTM exact-posterior lattice: summary

**Outcome: negative. Stop (b) fires on `t4_h05_q035`, so there is no
operating point and `smooth8_h05_q035` did not run.** No priced start
reaches Bayes + 0.01 within K = 1024 on the operating arm (8 bits `split`,
10 percent jitter). The frozen rule 7 names the mechanism **finite-w
representation**. The oracle pair agrees at K = 4096 (d = -0.0009) on a
common gap g\* = 0.0117, which is above 0.01. A reader has no operating
point to use. The lowest-gap priced cell is random start at K = 1024, with
a pooled gap of 0.0166. It does not reach and is not a recommendation. The
denoising line closes at L ≤ 16 for the tested construction only: hand-built
bias-design template units as tied copies on the index-4 pixel basis, with
relay paths on the 16-offset lattice, 8-bit `split`, and NumPy two-colour
Gibbs (not the lowered THRML program). No other layout was tested.

- **The exact model holds the posterior (stop (a) does not fire).** At
  w\* = 1.75, the placed 8-bit tie-limit model has an exact MPM gap of
  0.0003 on 1,000 test images. Its 64-draw gap is 0.0031, against a 0.0028
  floor for the true posterior. All six tie values pass admission on both
  layouts. Index 4 is carried (189 free nodes, max degree 9).
- **Sampling does not reach it.** These are pooled 64-draw gaps on the
  operating arm, over 3 seeds and 1,500 images:

  | Start | K = 64 | K = 256 | K = 1024 |
  | --- | --- | --- | --- |
  | random | 0.035 | 0.026 | 0.017 |
  | reset (every 5) | 0.047 | 0.036 | 0.020 |
  | continuation | 0.052 | 0.041 | 0.021 |

  Jitter 0 at 8 bits is no better (0.016 at random K = 1024). Neither is the
  float K = 4096 reference (0.012).
- **Rule 7.** Precision or jitter is not the mechanism: the jitter-0 arm
  never reaches where the operating arm misses, and the float reference
  does not reach either. The oracle pair at K = 4096 gives g_o = 0.0112 and
  g_r = 0.0121, so d = -0.0009 (95 percent interval -0.0029 to 0.0010) and
  the pair agrees. g\* = 0.0117 > 0.01 gives **finite-w representation**.
  The margin is small. Both pair cells' pooled intervals (0.008 to 0.014 and
  0.009 to 0.015) include 0.01, and the rule gates on the point estimate.
- **Reads and writes per sample** (the standing-rule headline, at K = 1024,
  189 free nodes): 64 reads in every start. Writes are 64 (continuation),
  101.8 (reset, 37.8 free-node writes) or 253 (random, 189 free-node
  writes). Every start does 193,536 updates.
- **Projected joule per 64-draw estimate** (`calibrated_projection`):
  - continuation: 7.2e-7 J at the write constant, 1.0e-7 J with clamp
    loads at the read constant;
  - reset: 1.1e-6 J to 4.7e-7 J;
  - random: 2.6e-6 J to 2.0e-6 J.
- **The baseline, priced per MAC** (`calibrated_projection`): the archived
  `cnn2_w32_k5` (1.74e6 MACs) costs 1.7e-7, 1.7e-6 or 1.7e-5 J per image at
  0.1, 1 or 10 pJ per MAC. On `t4` this baseline is not a cost verdict.
  Stop (c) is defined on `smooth8`, which did not run, so (c) is undecided.
- **Integrity.** The status is `dtm_exact_posterior_lattice_complete`, with
  `families_run=1`, `exact_rows=20` and `sampled_cells=96` (6 development
  and 90 stage S). Preflight, placement, recomputed references, the prefix
  check and the full replay all passed.
- **Run.** One attempt, no resume. Generation took 1 h 33 min wall on 3
  workers and replay took 3 min. CPU time was 4.56 CPU-hours, against a
  calibration projection of 2.82 for `t4` (1.6x under) and a protocol
  estimate of 2.6. The run was from commit b8aba65 with a clean tree.

Evidence classes:

- Exact reference (`exact_reference`, float64): Bayes, the tie-limit and
  conditioned equilibria, and the 64-draw floors.
- Software simulation (`software_simulation`, NumPy on CPU): every Gibbs
  cell. None comes from the lowered THRML program.
- Calibrated projection (`calibrated_projection`): every joule, on both
  sides.
- Constraints (rule 8): nothing is relaxed on the operating arm. The
  jitter-0 and 6-bit arms relax `beta_jitter`. The float reference relaxes
  `quantization` and `beta_jitter`.

There is no hardware claim.

Details: [findings.md](findings.md). Figure: `gap-versus-sweeps.png`
(regenerate with `plot.py`).
