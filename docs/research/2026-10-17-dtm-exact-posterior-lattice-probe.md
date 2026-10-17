# dtm-exact-posterior-lattice probe, round 2 (P-0008)

Date: 2026-10-17. Label: exploration, not evidence. This round follows the
owner's answer to round 1: an estimate is a 64-draw majority on both
families, the tolerance stays Bayes + 0.01, and stop (c) is priced per
64-draw estimate. Round 1 is
[`2026-10-16-dtm-exact-posterior-lattice-probe.md`](2026-10-16-dtm-exact-posterior-lattice-probe.md).

Scripts and output (all import round 1's functions unchanged):

- [`dtm_exact_posterior_lattice_scan2.py`](dtm_exact_posterior_lattice_scan2.py)
  ->
  [`2026-10-17-dtm-exact-posterior-lattice-scan2.json`](2026-10-17-dtm-exact-posterior-lattice-scan2.json):
  `smooth8` bias-design b0 scan beyond round 1's edge, and one-hot scored at
  8 bits.
- [`dtm_exact_posterior_lattice_scan3.py`](dtm_exact_posterior_lattice_scan3.py)
  ->
  [`2026-10-17-dtm-exact-posterior-lattice-scan3.json`](2026-10-17-dtm-exact-posterior-lattice-scan3.json):
  `smooth8` bias design at its chosen b0 on round 1's test set, at float and
  8, 6 and 4 bits.
- [`dtm_exact_posterior_lattice_placed.py`](dtm_exact_posterior_lattice_placed.py)
  ->
  [`2026-10-17-dtm-exact-posterior-lattice-placed.json`](2026-10-17-dtm-exact-posterior-lattice-placed.json):
  the first **placed** model, with each template unit spread over the lattice
  as tied copies, sampled with NumPy two-colour Gibbs.

CPU only, single-thread BLAS, Python 3.11, probe seed 20261017 (round 1's).
CPU time: 433 s, 228 s and 626 s recorded, plus about 420 s for a first
scan-2 run that the tool timed out and killed before it wrote output. That
is about 28 CPU-minutes, so the two rounds together used about 59, twice
the 30-minute probe budget. The overrun went on the two things round 1
left open: the `smooth8` stop (a) scan and the placed model. After the
recorded run, `dtm_exact_posterior_lattice_placed.py` received lint-only
edits (closure binding, one unused variable, formatting); the layout check
reproduces.

Evidence classes: the Bayes posterior, every model's exact equilibrium and
every n-draw floor are `exact_reference` (float64). Gibbs errors are
`software_simulation` (NumPy, CPU). Joules are `calibrated_projection`
(sealed Z1 profile constants). Nothing here is a hardware measurement.

## Findings

1. **`smooth8` passes stop (a) at 8 bits, narrowly.** The bias design,
   centered, needs b0 = −15. Round 1 stopped its scan at −10, where the
   development gap was +0.112. Development gaps: +0.0249 at −17, +0.0087 at
   −16, **+0.0040 at −15**, +0.0078 at −14, +0.0270 at −13. On round 1's
   500 test images:

   | `smooth8`, bias, b0 = −15 | float | 8 bits | 6 bits | 4 bits |
   | --- | --- | --- | --- | --- |
   | MPM gap to Bayes (SE) | +0.0072 (0.0019) | **+0.0084** (0.0020) | +0.0377 | +0.0768 |
   | Exact 64-draw majority gap | +0.0101 | **+0.0106** | +0.0398 | +0.0786 |

   - The 8-bit margin under stop (a) is 0.0016, about one standard error.
   - 6 bits fails. Its `split` field cap is 8.27, so a 6-bit field step is
     0.27, and the hidden biases near −15 lose the precision the latent
     needs. On `t4` (b0 = −5, field cap 3.55) 6 bits passed.
   - One-hot stays range-bound. At λ = 16 it is +0.0009 in float and
     +0.0221 at 8 bits (coupling cap 8, field cap 48.8); at λ = 24, +0.0000
     and +0.0616 (development set). Round 1 found the same at λ = 32.
   - Placement changes the codebook. Spreading a unit's bias over its 27 to
     34 copies drops the `split` field cap from 8.27 to 0.5 (the pixel
     biases set it) on both families in finding 4. The 6-bit failure above
     is a property of the unplaced codebook and may not hold placed. The
     probe did not compute the placed codebook's collapsed-tie equilibrium.
2. **On `smooth8` the 64-draw estimate of the best 8-bit model sits above
   Bayes + 0.01 at equilibrium.** The exact 64-draw majority of that model
   is +0.0106 (float +0.0101). The true posterior's own 64-draw floor is
   +0.0047, so the bias design's representation error (+0.0084 MPM) plus
   estimator variance uses the whole tolerance. A perfect sampler of this
   model would sit on the stop (b) line. With a standard error near 0.002
   the outcome is not fixed, but stop (b) on `smooth8` is expected to fire
   with the bias design as the mechanism, not the lattice.
3. **A placed layout exists with pixel degree 9 (`t4`) and 13 (`smooth8`).**
   - Pixel (i, j) sits at site i·(2, 1) + j·(−1, 2). All four grid offsets
     are Z1 offsets, so the 8 x 8 Ising grid is native.
   - That pixel sublattice has index 5. One of its four other cosets holds
     the clamps. Each of the other three can hold up to three template
     units.
   - A site in those cosets touches exactly three pixels. Those triangles
     tile the pixel grid under an index-3 sublattice. So each unit is 27 or
     34 copies, every pixel couples to exactly one copy of every unit, and
     no copy has more than 3 pixel edges.
   - Copies of one unit tie along one axis only, so a unit is three or four
     parallel chains of copies. Joining the chains needs relay paths through
     empty sites outside the image. This probe joined chain ends with one
     direct tie per pair instead: 2 or 3 edges per unit with `placement`
     relaxed, colour-checked so two-colour Gibbs stays valid.
   - Free nodes: 64 + 115 = 179 on `t4`, 64 + 237 = 301 on `smooth8`, before
     relay paths.
   - Other pixel bases built from two Z1 offsets exist. The index-4 basis
     (−4, −1), (0, −1) gives 4 pixels per site in three cosets, so two
     cosets for units (fewer, larger copies). It was not tested. Index 7 to
     13 bases leave cosets with 0 or 1 pixel per site.
4. **Tied copies sit 0.016 or more above the unplaced floor on `t4` at 8
   bits.** Each copy carries
   its unit's pixel couplings and 1/n of the unit bias, so infinite ties
   reproduce the unplaced model. Gap of the 64-draw majority to Bayes, 200
   test images (10 streams of 20), one sampler seed, 95 percent stream
   bootstrap:

   | Tie w (coupling cap) | random K = 256 | random K = 1024 | continue K = 256 | continue K = 1024 |
   | --- | --- | --- | --- | --- |
   | 1.0 (1.0) | +0.030 | +0.030 | +0.027 | +0.032 |
   | 2.0 (2.0) | +0.026 | **+0.018** [0.005, 0.030] | +0.036 | +0.024 |
   | 4.0 (4.0) | +0.036 | +0.039 | +0.040 | +0.037 |

   - The unplaced 8-bit model's exact 64-draw floor on the same images is
     +0.0017. Round 1's unplaced Gibbs was at its floor by K = 256.
   - w = 1 is too weak to hold a unit's copies together. w = 4 is worse, and
     at 8 bits under `split` it also sets the coupling cap: the step becomes
     0.031, coarse against J = 0.35 and pixel couplings near 0.1 to 0.25.
   - At w = 2 the error still falls from K = 256 to 1024, so the copy chains
     have not mixed by 1024 sweeps. The best interval straddles +0.01.
   - Continuation is no better than a random start at any w.
5. **`smooth8` placed, 8 bits, K = 1024, b0 = −15:** random start +0.034
   (w = 2) and +0.081 (w = 4). Continuation +0.057 and +0.122. Against an
   unplaced 64-draw floor of +0.0113 on these 200 images.

## Cost (calibrated_projection, sealed profile, placed node counts)

Per sample on the host-per-step path: 64 pixel reads, 64 clamp loads at the
write constant, free-node writes by start, and 179 or 301 node updates per
sweep. Relay-path nodes are not yet counted, so these are lower bounds.

- **`smooth8` stop (c) against 3.43e-6 J (3.43e6 MACs at 1 pJ):**
  - random start: 3.60e-6 to 3.73e-6 J per 64-draw estimate at K = 64 to
    1024, **1.05x to 1.09x the rung, so a random start fails (c) at
    every K**. (Corrected in round 3 of the proposal: this line first
    printed the inverse ratio, 0.92x to 0.95x. The verdict was right.)
  - reset every 5 images: 1.24e-6 to 1.37e-6 J, 2.5x to 2.8x cheaper;
  - continuation: 6.4e-7 to 7.8e-7 J, 4.4x to 5.3x cheaper.
  - The owner's round-1 estimate (1.8x for a random start) used about 190
    free nodes. The placed `smooth8` model has 301.
- **`t4`:** random start 2.4e-6 to 2.5e-6 J, continuation 6.4e-7 to
  7.2e-7 J per 64-draw estimate. `t4` does not decide cost; its likely
  digital cost is near 2.4e5 MACs.
- With clamp loads at the read constant the random start falls to 2.98e-6
  to 3.11e-6 J on `smooth8`, 0.87x to 0.91x the rung.
- (c) is judged against the archived rung, not the digital minimum. The
  ladder has nothing between 243,200 and 1.74e6 MACs.

## Fixed choices that could bound the metric (update to round 1)

- **Draws per estimate.** Now 64 (owner). It no longer bounds `t4` (floor
  +0.0018). On `smooth8` it leaves +0.0053 of the tolerance for the model.
- **Hidden encoding.** Binding on `smooth8`. The bias design's own 64-draw
  equilibrium is +0.0106; one-hot is exact in float and fails at 8 bits on
  range.
- **Tie coupling w and its precision.** Binding on `t4` in this probe. Too
  weak decouples copies; too strong takes the `split` coupling cap and
  slows the copy chains.
- **Copy geometry.** Each unit is three or four one-dimensional chains, so
  its latent flips by moving a domain wall along a chain. AM A found that
  domain-wall chains do not mix (lessons, 2026-10-07). This is the likely
  mechanism behind finding 4.
- **Relay paths.** Relaxed in this probe. A placed study must route them on
  the lattice and count their nodes.
- **Start.** Continuation was no better than random on placed `t4`, and on
  `smooth8` only continuation or reset can pass (c).
- **Sweeps K.** Not saturated at 1024 on placed `t4` at w = 2.

## What this means for the row

- The probe does not answer the question outright. On `t4`, every placed
  cell missed Bayes + 0.01, but the best interval straddles it, with one
  seed, 200 images and three tie values.
- A study can settle `t4` stops (a) and (b) for the placed model with fresh
  seeds, more images and a tie scan, and name the mechanism if (b) fires.
- On `smooth8`, the exact side already puts the bias design on the stop (b)
  line, and a random start fails (c). Only continuation or a periodic reset
  can pass (c), and continuation was the trapped start in both families
  here.
- Sizing for the protocol: placed NumPy Gibbs costs 2.05e-4 CPU-s per image
  per sweep at 64 chains on `t4` and 3.5e-4 on `smooth8`, measured on full
  K = 1024 runs.
