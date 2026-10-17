# DTM exact-posterior lattice: exploratory protocol (draft, proposal P-0008)

Drafted on 2026-10-17 for queue row `dtm-exact-posterior-lattice`, revised
the same day after the owner's review of PR #126 (round 2 of the proposal). It
is not frozen until the owner approves it, and no runner exists yet. The
probes are
[round 1](../research/2026-10-16-dtm-exact-posterior-lattice-probe.md)
(unplaced model, exact and NumPy Gibbs),
[round 2](../research/2026-10-17-dtm-exact-posterior-lattice-probe.md)
(64-draw estimate, `smooth8` exact scan, first placed model) and
[round 3](../research/2026-10-17-dtm-exact-posterior-lattice-layout-probe.md)
(index-4 layout check, relay paths placed with nothing relaxed, development
scan, oracle start). All are exploration. The owner set the estimate size on
round 1: a 64-draw majority on both families, tolerance Bayes + 0.01, stop
(c) priced per 64-draw estimate. Tolerance and draws are the same on both
families.

## The number this study produces

For a template latent spread over the 16-offset lattice as tied copies, at
8 bits under `split` with 10 percent jitter, on P-0007's `t4_h05_q035`:

- the **tie coupling w\*** and **coupling and field precision** at which the
  placed, quantized model holds the exact posterior within Bayes + 0.01; and
- the **smallest sweeps per sample K\*** and the **reads, writes and
  free-node writes per sample** at which a priced start reaches Bayes + 0.01
  with a 64-draw estimate, or, if none does within K = 1024, the gap at
  K = 1024 and the constraint that holds it there.

On `smooth8_h05_q035`, if it runs, the same numbers plus the projected joule
per 64-draw estimate against the archived rung `cnn2_w32_k5_gpool` (3.43e6
MACs) at 1 pJ per MAC, which is stop (c).

A designer reads it as: "a global latent costs this many tied copies at this
tie strength and precision, and this many sweeps and writes per estimate,
on the lattice", or "tied copies cannot carry it within 1024 sweeps, because
of X". On `smooth8` it may instead read "the bias-design encoding itself
misses Bayes + 0.01 at 8 bits, before any sampling" (stage E stop, below).

## Question

Can a hand-built lattice model of the exact posterior of a family with
digital headroom, its template latent encoded as binary hidden units and
spread over the 16-offset lattice as tied copies, reach Bayes + 0.01 under
the constraint layer with no training, and at what sweeps, reads, writes
and free-node writes per sample from a priced start?

## Hardware constraints applied and relaxed (rule 8)

- **Applied on the operating arm.** Placement on the 16-offset lattice with
  nothing relaxed (`require_placement` passes, maximum degree 16, every edge
  a Z1 offset, two colours); `split` quantization at 8 bits; static
  10 percent per-site beta jitter; two-colour chromatic Gibbs with no
  exchange; host-per-step I/O with the start priced per sample.
- **Relaxed, with reasons in the hashed request.**
  - `quantization` and `beta_jitter` on the float K = 4096 reference arm,
    which is the convergence reference for the placed model.
  - `beta_jitter` on the 8-bit jitter-0 arm and the 6-bit arm, which locate
    the precision and jitter limits.
  - The exact stage collapses each unit's ties (the w → ∞ limit), because a
    finite-w placed model has no tractable exact equilibrium (each unit is
    27 to 34 copies plus relay nodes, coupled to 3 or 4 pixels per copy).
    The collapsed model keeps the placed, quantized couplings and per-copy
    biases. The report names it as the tie limit everywhere it appears.
  - The start, on the oracle-start reference arm only (copies and relays set
    to the generating template, pixels to the noisy image). The arm is
    unpriced and never enters stop (b) or (c); it only separates finite-w
    representation from mixing.
- **Outside the layer.** Nothing is trained. Model parameters are computed
  from the generator in float64 and then quantized.

## Resource accounting (rule 9) and the queue's standing rules

- **Lattice side.** Reads, writes, free-node writes and active updates per
  sample from `thermo_lab.constraints.count`, priced by `price` under the
  sealed Z1 profile, on the host-per-step path:
  - 64 pixel reads; 64 clamp loads, counted as full SRAM writes;
  - free-node writes: 0 for continuation, all free nodes for a random
    start, all free nodes every fifth image for a reset;
  - updates: free nodes times K.
  The headline is the counts. The joule is a band: the profile's write
  constant at the top, clamp loads at the read constant (1.692 pJ, outside
  the profile) at the bottom. One estimate is a 64-draw majority, so the
  estimate's cost is 64 times the sample's.
- **Baseline.** The archived P-0007 ladder for the family, priced per MAC at
  0.1, 1 and 10 pJ (`calibrated_projection`, constant stated, no refit):
  `cnn2_w32_k5` (1.74e6 MACs) on `t4`, `cnn2_w32_k5_gpool` (3.43e6 MACs) on
  `smooth8`. One baseline evaluation is one estimate.
- **CPU seconds** (`time.process_time`) are reported beside the projections
  and do not gate.
- Both standing rules of `docs/research-queue.md` are met: counts per sample
  are the headline, the joule is a band, and the baseline is priced per MAC
  at three constants.
- (c) compares against the archived rung, not the digital minimum. The
  ladder has nothing between 243,200 and 1.74e6 MACs, and the `t4` 243,200-MAC
  net misses Bayes + 0.01 by only 0.0005. The report says so.

## Fixed choices and what they bound

- **Families (L = 8, exact transfer-matrix posterior).** `t4_h05_q035`
  first. `smooth8_h05_q035` runs only if `t4` passes stops (a) and (b), as
  the row says. Fields are imported from `thermo_lab.dtm_baseline_headroom`
  unchanged.
  - *Bounds:* `t4` tests representability and mixing, not cost (its likely
    digital cost is near 2.4e5 MACs). Cost is decided on `smooth8` only.
- **Hidden encoding.** Stage A's bias design, centered: one binary unit per
  template, pixel coupling (h_c − mean h)/2 in spin form, unit bias b0 minus
  the centered log-partition. b0 = −5 on `t4` and −15 on `smooth8`, from
  the probes' development scans; not re-tuned.
  - *Bounds:* representation. On `smooth8` the placed tie-limit model's
    exact 64-draw equilibrium is +0.0102 to +0.0124 at 8 bits across the w
    grid and both layouts (round 3, 200 images), on the (b) line before any
    sampling. The
    stage E stop below decides this exactly instead of leaving it to
    sampling noise. One-hot is not run: it is exact in float and fails at 8
    bits on range on both families (round 1, round 2).
- **Layout: two candidate pixel bases, one carried.** Both are checked by
  `require_placement` with nothing relaxed; round 3 placed both on both
  families.
  - *Index 5.* Pixel (i, j) at site i·(2, 1) + j·(−1, 2). Clamps on coset
    (0, 1); units on three other cosets, each copy touching at most 3
    pixels, copy classes from a 3-tiling of the grid.
  - *Index 4.* Pixel (i, j) at site i·(−4, −1) + j·(0, −1). Clamps on coset
    (−3, 2); units on the two other cosets, each copy touching at most 4
    pixels, copy classes from a 4-tiling (one coset's tile tiles only with
    period 2, which the layout code handles). Classes filled coset by
    coset.
  - Both bases make the four grid offsets Z1 offsets. Every pixel couples to
    exactly one copy of every unit. Copies of one class that sit a Z1
    offset apart are tied directly; the class's remaining components are
    joined by **relay paths** routed over empty sites by breadth-first
    search on Z1 offsets (relay nodes: free, zero bias, tie w on every relay
    edge). Pixel degree 4 + 1 + C: 9 on `t4`, 13 on `smooth8`.
  - Free nodes, relays included (round 3): `t4` 197 (index 5) and 189
    (index 4); `smooth8` 339 and 332. Border copies are partial, so the
    index-4 basis does not halve the copies: 27 to 32 per unit against 27
    to 34. It saves 8 and 7 free nodes, not about 110.
  - **Carried layout (rule, fixed now).** Stage E computes the tie-limit
    gap on both layouts. The study carries the layout with fewer free nodes
    among those that pass stop (a), index 4 on equal counts; if neither
    passes, (a) fires. Round 3 predicts index 4 on both families. Stage D
    scans w on the carried layout.
  - *Bounds:* mixing. A unit's latent flips by moving domain walls through
    a sparse, mostly one-dimensional copy graph, which AM A found does not
    mix (lessons, 2026-10-07). Only these two bases are tested. Index-7 to
    13 bases leave cosets with 0 or 1 pixel per site (round 2).
- **Copy parameters.** Each copy carries its unit's pixel couplings
  unchanged and 1/n of the unit bias, so the w → ∞ limit is the unplaced
  model.
- **Tie coupling w.** Chosen per family on a development set from
  {1.5, 2, 2.5, 3} at 8 bits, jitter 0, random start, K = 1024, one
  development sampler seed, on the carried layout: the w with the smallest
  development gap. Round 2 found w = 1 too weak and w = 4 worse than 2 on
  `t4`; round 3 (index 4, `t4`, 200 images) found +0.011, +0.004, +0.016
  and +0.024 for w = 1.5, 2, 2.5, 3.
  - *Bounds:* everything. Under `split` the ties share the coupling cap, so
    at w ≥ 0.375 the tie sets the coupling step (w/127 at 8 bits) and coarsens
    J = 0.35 and the pixel couplings. This is a property of `split`, and
    the report states it as such.
- **Bits and jitter.** Operating arm 8 bits, 10 percent jitter. Secondary:
  8 bits jitter 0, 6 bits jitter 0. 4 bits is exact-only (round 1: +0.033 on
  `t4`; round 2: +0.077 on `smooth8`, unplaced codebook).
- **Starts.** Continuation, random, reset every 5 images (N = 5 from round
  1). Streams of 20 images. Plus an unpriced oracle start on the reference
  arm (below).
- **Sweeps.** K in {64, 256, 1024} per sample, as the row says. A float
  K = 4096 random-start reference locates the placed model's own
  equilibrium. Round 2 found `t4` still falling between 256 and 1024 at
  w = 2.
- **Estimate.** 64-draw pixelwise majority, ties to +1 (owner). Exact floor
  of the true posterior's own 64-draw majority: +0.0018 on `t4`, +0.0047 on
  `smooth8`.
- **Test set.** 500 fresh images per sampler seed (25 streams of 20), 1,000
  for the exact stage. Per-seed standard error on a gap about 0.003 to
  0.005; pooled over three seeds about 0.002 to 0.003.
- **Sampler engine.** NumPy two-colour Gibbs over the constraint layer's
  quantized, jittered `PairwiseModel`. It is **not the lowered THRML
  program**: the layer's default lowering (`thrml_lowering`) is not run, and
  the report says so wherever a sampled number appears. THRML is not a
  constraint; the engine is checked against the layer's exact chromatic
  kernels (preflight 3), which is what licenses it.

## Pre-registered expectations (from the probes)

1. `t4` passes stop (a) on both layouts: the tie-limit placed model is
   within 0.01 of Bayes at 8 bits (round 3: MPM gap −0.0006 to +0.0009,
   64-draw gap +0.0015 to +0.0027 across the w grid, 200 images). Index 4
   is carried (189 free nodes against 197).
2. `t4` stop (b) is undecided by the probes. Round 3's best index-4 cell
   (w = 2, random start, K = 1024, one seed, 200 images) is +0.0043; the
   other three tie values are +0.011 to +0.024; index 5 at w = 2 is +0.0125.
   Round 2's index-5 cells were +0.018 to +0.040. Expect the operating arm
   to sit between +0.004 and +0.015, near the threshold, so the pooled rule
   and its power (below) matter.
3. If (b) fires on `t4`, the mechanism is mixing: the oracle start holds
   (round 3, K = 1024, one seed, 200 images: −0.053 on index 4 and −0.041
   on index 5 against Bayes, while the conditioned target is −0.068, so the
   chains partly relax but keep most of the latent), and the gap falls with
   K beyond 1024.
4. Continuation is no better than a random start in any seed.
5. Jitter changes nothing (round 1: +0.0096 against +0.0098).
6. `smooth8` does not run. If it does, the stage E stop fires: the tie-limit
   model's 64-draw gap is above 0.01 at every w (round 3, index 4: +0.0102
   to +0.0121; index 5: +0.0105 to +0.0124; 200 images), and the mechanism
   is bias-design representation. Had it passed, a random start would fail
   (c) at every K (index 4 with relays: 332 free nodes, 3.91e-6 to 4.05e-6
   J per estimate at K = 64 to 1024, 1.14x to 1.18x the 3.43e-6 J rung).

A miss on any of these is a result.

## Exact anchor and evidence classes

- `exact_reference`: the mixture samples, the Bayes posterior marginals and
  MPM error, the tie-limit placed model's equilibrium (float64, enumeration
  over 2^C hidden states times the transfer matrix), the same model and the
  true posterior conditioned on the generating hidden state (the oracle
  arm's latent-retention diagnostic), and every n-draw floor.
- `software_simulation`: every NumPy Gibbs cell (CPU), including the oracle
  arm. NumPy Gibbs is not the lowered THRML program.
- `calibrated_projection`: every joule, on both sides.

## Arms and controls

Per family (`smooth8` only if `t4` passes (a) and (b)):

**Stage E, exact (gating stop (a), the layout choice, and on `smooth8` stop
(b)).** The tie-limit placed model on both layouts at each w in the grid at
8 bits, and on the carried layout at float, 6 and 4 bits at the tie value
with the smallest 8-bit tie-limit MPM gap (fixed before sampling, so stage E
does not wait for w\*), on 1,000 test images: MPM error and exact 64-draw
majority error against Bayes. Also at the unplaced codebook, for comparison
with the probes. Stage E runs before any sampling.

**Stage D, development.** On the carried layout, the tie values that stage E
admits x 300 development images, one development sampler seed.

**Stage S, sampled.** On the carried layout at w\*, for each sampler seed
s in {0, 1, 2}:

- operating arm: 8 bits, jitter 0.1;
- jitter control: 8 bits, jitter 0;
- precision control: 6 bits, jitter 0;

each from three starts at K in {64, 256, 1024}: 27 cells per seed, 81 per
family. Plus two references per seed:

- float K = 4096 random-start reference (1 cell);
- **oracle-start reference** (3 cells): operating arm (8 bits, jitter 0.1)
  at K = 1024, every copy and relay node of unit c set to +1 if c generated
  the image and −1 otherwise, pixels set to the noisy image. One cell per
  seed on each of three disjoint thirds of the seed's 500 images (167, 167,
  166). Start relaxed, unpriced, outside (b) and (c).

Controls: Bayes; noisy input; the unplaced model's exact 64-draw floor; the
true posterior's 64-draw floor; the float K = 4096 reference; the
conditioned exact targets for the oracle arm's latent retention.

## Primary decision rule (thresholds fixed now)

Replication unit: the sampler seed. "Gap" is the 64-draw majority's pixel
error minus the Bayes MPM error on the same images.

1. **Stop (a).** On a layout and tie value, the tie-limit placed model's
   exact MPM gap at 8 bits is above 0.01 (point estimate, 1,000 images).
   A tie value that fails is dropped from stage D. If every tie value fails
   on both layouts, (a) fires: stage S does not run on that family, and the
   mechanism is precision or the bias design, whichever the float row
   separates.
2. **Stage E stop (b), `smooth8` only.** On the carried layout, a tie value
   whose tie-limit model's exact 64-draw majority gap at 8 bits on the
   1,000 fresh test images is above 0.01 is dropped from stage D. If every
   tie value is dropped, stop (b) fires at stage E: stage S does not run on
   `smooth8`, and the mechanism is **bias-design representation**, not the
   lattice. A perfect sampler of that model would already miss, so no
   sampled cell could reach. Tolerance 0.01 and 64 draws, as on `t4`. The
   stop is checked per tie value and fires only if every value misses, so
   picking among tie values on the test images can only make the negative
   harder to reach, never easier.
3. **Reaching.** A cell (bits, jitter, start, K) reaches if (i) its gap
   pooled over the three seeds (mean of the three per-seed gaps, 1,500
   images) is at most 0.01, and (ii) every per-seed gap is at most 0.015
   (consistency bound). The 95 percent stream bootstrap, pooled and per
   seed, is reported beside each.
   - *Power* (normal approximation, three independent seeds, Monte Carlo of
     the rule; per-seed standard error 0.004, mid-range of 0.003 to 0.005):
     a cell whose true gap is +0.006 reaches with probability 0.94 (0.86 to
     0.99 across the range), +0.008 with 0.77 (0.68 to 0.87), +0.009 with
     0.63, +0.010 with 0.46, +0.012 with 0.17 (0.12 to 0.20). The old
     all-three-seeds rule had 0.33 at +0.008. Near the threshold the rule is
     a coin flip by design; a "(b) fires" on a cell with a true gap at
     +0.008 still happens about one time in four, and the report states
     the pooled gap and interval so a reader can see it.
4. **Carried-state check** (lessons, 2026-10-14). A continuation or reset
   cell counts as reaching only if its pooled gap is within 0.01 of the
   random-start cell at the same K (paired stream bootstrap upper bound,
   pooled over seeds).
5. **Stop (b).** No start reaches on the operating arm (8 bits, jitter 0.1)
   at any K ≤ 1024, or the stage E stop fired.
6. **Operating point.** If (b) does not fire: K\* is the smallest K at which
   some start reaches on the operating arm; at K\*, the start with the
   fewest writes per sample. Its counts are the headline. Because the
   point is chosen only among reaching cells, the recorded point is resolved
   by construction (lessons, 2026-10-15).
7. **Mechanism when (b) fires after sampling**, named from fixed
   comparisons, in this order:
   - *precision or jitter*, if the 8-bit jitter-0 arm or the float
     reference reaches where the operating arm does not;
   - the **oracle reference** separates the rest. It *holds* if its pooled
     gap to Bayes is at most 0.01 and every per-seed gap at most 0.015 (the
     reaching rule applied to the oracle arm);
   - *mixing within budget*, if the oracle arm holds: started in the right
     mode, the finite-w placed model keeps a posterior good enough to reach,
     so the random-start miss is the chain not finding the mode in 1024
     sweeps. The float K = 4096 reference is reported beside it;
   - *finite-w representation*, if the oracle arm does not hold: started in
     the right mode, the copies still lose the latent at w\*, so tied copies
     at a tie strength the codebook can express do not represent a global
     latent, whatever the mixing.
   - Caveat, stated in the report: the oracle start carries the generating
     template, which the noisy image does not. The exact tie-limit model
     conditioned on it sits far below Bayes (round 3: −0.068 on `t4`, −0.102
     on `smooth8`, 64-draw gap, 200 images), so a chain that never leaves
     the start mode also "holds". The oracle cell's distance from that
     conditioned target (its *latent retention*) is reported as a
     diagnostic: near the target means the latent never moved in 1024
     sweeps; near Bayes means the chain relaxed to the model's own
     posterior. Neither gates.
   Copy agreement (fraction of a unit's copies equal to its majority) at the
   last sweep is reported per cell as a diagnostic and does not gate.
8. **Stop (c), `smooth8` only.** The operating point's joule per 64-draw
   estimate at the profile's write constant is at least 3.43e-6 J (the
   rung at 1 pJ per MAC). The band's low end and the 0.1 and 10 pJ
   constants are reported beside it.

Any stop closes the denoising line at L ≤ 16 **for this construction**:
hand-built bias-design template units, spread as tied copies (index-4 or
index-5 pixel basis, whichever stage E carries, with relay paths on the
16-offset lattice), 8-bit `split` codebook, NumPy two-colour Gibbs. The
closing sentence names that geometry and says no other layout was tested.
The real binarized set opens only on a positive (row).

## What one recorded sample means

One chain's 64 pixel states after K sweeps of two-colour Gibbs on the placed
model with the noisy image clamped, starting from the previous image's
final state in the stream (continuation), a uniform state (random), a
uniform state every fifth image (reset), or, on the unpriced reference arm
only, the oracle state (every copy and relay of the generating unit +1, of
the others −1, pixels at the noisy image). An estimate is the pixelwise
majority of 64 independent chains, ties to +1. Bayes: the exact marginal
posterior mode per pixel.

## Seeds

All fresh; none appears in an archive or a probe (probe seed 20261017;
P-0007 seeds 20261201 to 20261210; P-0006 seeds 20261101 to 20261108).

- Test images and noise: `default_rng([20261301, f])`. The generating
  component of each image is recorded from the same stream, for the oracle
  arm.
- Development images and noise: `default_rng([20261302, f])`.
- Development sampler: `default_rng([20261303, f, w_index])`.
- Stage S chains: `default_rng([20261304, f, s, cell])`.
- Float reference chains: `default_rng([20261305, f, s])`.
- Jitter words: (20261306, f, s).
- Bootstrap: `default_rng([20261307, f, s])`.
- Oracle reference chains: `default_rng([20261308, f, s, third])`.

## Budgets and CPU estimate

Measured in round 3 on full K = 1024 runs with relay nodes placed (200
images, 64 chains): 2.28e-4 CPU-s per image per sweep on index-4 `t4` (189
free nodes), 4.76e-4 on index-4 `smooth8` (332). Tie-limit exact rows: about
17 CPU-s per 200 images on `smooth8`, under 2 on `t4`.

- **`t4`:** stage S 27 cell-groups x 500 images x 1,344 sweeps, 1.15
  CPU-hours; float reference 3 x 500 x 4,096, 0.4; oracle reference 1,500
  images x 1,024, 0.1; development 0.1; exact stage (two layouts) and
  preflight 0.1. **About 1.8 CPU-hours.**
- **`smooth8`, stage E only** (expected if it runs): 15 exact rows on
  1,000 images, **about 0.4 CPU-hours.**
- **`smooth8`, full** (only if the stage E stop does not fire): stage S 2.4,
  float reference 0.8, oracle 0.2, development 0.15, exact 0.4. About 4.0
  CPU-hours.
- **Total: 1.8 CPU-hours if `t4` stops, 2.2 if `smooth8` stops at stage E,
  5.8 if both run in full.** P-0007 ran 2.7x over a short-run calibration,
  so the runner calibrates on full-length cells and stops if the projection
  exceeds 8 CPU-hours. About 0.7, 0.9 or 2.2 hours wall on three workers.

## Preflight (gating, probe seed)

1. Layout, on both bases: every unit covers each pixel exactly once, sites
   are distinct, relay paths use only Z1 offsets, and `require_placement`
   passes with nothing relaxed on the operating arm's model (round 3: both
   bases, both families).
2. Tie limit: summing each unit's quantized per-copy biases and taking its
   copies' common pixel couplings reproduces the collapsed model used by
   stage E, to 1e-12.
3. Sampler: the NumPy engine against the constraint layer's
   `exact_chromatic_kernels` on a sub-model of at most 12 free nodes (a
   2 x 2 pixel patch with one unit's copies and a relay), K in {1, 2, 4},
   400,000 chains, max |z| < 5. This licenses the NumPy engine; it does not
   make it the lowered THRML program.
4. Pricing: `count` and `price` reproduce a hand count of reads, writes,
   free-node writes and updates per sample for each start, and both ends of
   the band.
5. Oracle: the unplaced one-hot restriction reproduces the Bayes posterior
   to 1e-12 (round 1: 6.9e-15).
6. Oracle start: the oracle state's copy and relay values match the
   recorded generating component for every image.

## Autosave and resume

The run may exceed 30 minutes, so the runner follows the
[autosave contract](../experiment-runner.md#autosave-and-resume-contract).
Work units are atomic: one exact stage per family, one development cell,
one stage S cell (family, arm, start, K, seed), one reference cell. The
`smooth8` exact stage is created only after the `t4` verdict is written and
digested, and its sampled units only after its stage E verdict. Resume is
the same command with `--resume` and the same output directory. A unit
test interrupts a small run and checks that the resumed record is
identical.

## Persistence and archive size

`study.json.gz`: the hashed request (family field arrays by SHA-256, both
layouts with relay paths, the carried-layout rule, constraint requests with
digests and relaxed reasons, w grid, seeds); preflight; per-image
wrong-pixel counts (uint8) for every cell, Bayes and the noisy input;
per-image generating components; per-cell copy agreement; exact-stage
marginal summaries; pricing; CPU timings, kept out of the result digest;
the evaluation. At most 97 cells per family x 500 images, under 0.3 MB
gzipped.

Replay recomputes Bayes errors, the exact stage, the conditioned targets
and the floors at 1e-12 relative, recomputes the carried layout, reaching,
the operating point, mechanism, bootstraps, prices and verdicts from the
archived counts, and re-verifies placement of both layouts. A prefix check
reruns one stage S cell and one development cell and requires identical
counts. CI replays without the prefix check (`--light`).

## Gate

`completion.json`, written last by the runner, must show:

- `status=dtm_exact_posterior_lattice_complete`;
- `families_run` (1 or 2; a family runs when its stage E runs);
  `exact_rows` equal to 15 times `families_run` (two layouts x four tie
  values at 8 bits; float, 6 and 4 bits at the stage E tie value; float,
  8, 6 and 4 bits unplaced);
- `sampled_cells` equal to the development cells admitted by stage E plus
  93 for each family that reaches stage S (81 stage S, 3 float reference,
  9 oracle reference), with the per-family split listed;
- `preflight_passed`, `placement_passed`, `references_recomputed`,
  `prefix_check_passed`, `replayed` all `true`.

It also carries the carried layout, w\*, the stop verdicts (a), (b) (and
whether (b) fired at stage E), (c) per family, the named mechanism, and the
operating point with its counts and joule band. These are scientific
results, not integrity checks.

## Stop rules

- Preflight fails: stop before production. Do not tune tolerances.
- Calibration projects over 8 CPU-hours: stop and report.
- Stop (a) or (b) fires on `t4`: `smooth8` does not run.
- The stage E stop fires on `smooth8`: its stage D and S do not run.
- Two consecutive failed runs with the same error: stop and report.
- A replay mismatch is an integrity failure.

## What a negative result means

- **(a) fires.** The lattice's precision under `split`, with ties sharing
  the coupling cap, cannot hold the posterior even with infinite ties. The
  number is the precision gap.
- **(b) fires at stage E (`smooth8`).** The hand-built bias-design encoding
  misses Bayes + 0.01 with a perfect sampler at 8 bits. The mechanism is
  bias-design representation, not the lattice. The number is the
  tie-limit 64-draw gap. Cost (c) is not decided.
- **(b) fires with "mixing within budget".** The placed model at w\* holds
  the posterior from the right start, but random, continued and reset
  chains do not reach it in 1024 sweeps. The number is the gap at K = 1024
  and the float K = 4096 reference's gap.
- **(b) fires with "finite-w representation".** Even from the generating
  template, tied copies at a tie strength the codebook can express lose the
  latent. The number is the oracle arm's gap.
- **(c) fires on `smooth8`.** The lattice holds and samples the posterior but
  costs at least as much per estimate as the archived rung at 1 pJ per MAC.
- **Any of these** closes the denoising line at L ≤ 16 for the tested
  construction only: bias-design template units as tied copies on the
  carried pixel basis (index 4 or 5) with relay paths on the 16-offset
  lattice, 8-bit `split`, NumPy two-colour Gibbs (not the lowered THRML
  program). The closing sentence names that geometry. None is an integrity
  failure.

## Not claimed

No Z1 measurement, latency or device noise. No energy outside the sealed
profile's counted operations (host transfer and control excluded; on the
digital side activations, memory traffic and the bias map excluded). No
lower bound on digital cost. No result from the lowered THRML program. No
trained model, no L = 16 result, no real image set, and no claim about
layouts other than the two pixel bases tested.
