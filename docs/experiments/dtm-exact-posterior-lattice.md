# DTM exact-posterior lattice: exploratory protocol (draft, proposal P-0008)

Drafted on 2026-10-17 for queue row `dtm-exact-posterior-lattice`. It is not
frozen until the owner approves it, and no runner exists yet. The probes are
[round 1](../research/2026-10-16-dtm-exact-posterior-lattice-probe.md)
(unplaced model, exact and NumPy Gibbs) and
[round 2](../research/2026-10-17-dtm-exact-posterior-lattice-probe.md)
(64-draw estimate, `smooth8` exact scan, first placed model). Both are
exploration. The owner set the estimate size on round 1: a 64-draw majority
on both families, tolerance Bayes + 0.01, stop (c) priced per 64-draw
estimate.

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
of X".

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
    finite-w placed model has no tractable exact equilibrium (each unit is a
    chain of 27 to 34 copies coupled to 3 pixels each). The collapsed model
    keeps the placed, quantized couplings and per-copy biases. The report
    names it as the tie limit everywhere it appears.
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
  - *Bounds:* representation. On `smooth8` the bias design's exact 64-draw
    equilibrium is +0.0106 at 8 bits (round 2), on the (b) line. One-hot is
    not run: it is exact in float and fails at 8 bits on range on both
    families (round 1, round 2).
- **Layout.** Pixel (i, j) at site i·(2, 1) + j·(−1, 2), so the four grid
  offsets are Z1 offsets. Clamps on coset (0, 1). Units on cosets (1, 0),
  (−1, 0), (0, −1), up to three per coset, each copy class an index-3
  translate so every pixel couples to exactly one copy of every unit and a
  copy has at most 3 pixel edges. Copies tie along one axis by a Z1 offset,
  so a unit is three or four parallel chains (27 or 34 copies). Chains are
  joined by **relay paths** routed outside the image over empty sites by
  breadth-first search on Z1 offsets (relay nodes: free, zero bias, tie w on
  every relay edge). Pixel degree 4 + 1 + C: 9 on `t4`, 13 on `smooth8`.
  - *Bounds:* mixing. A unit's latent flips by moving domain walls along
    one-dimensional chains, which AM A found do not mix (lessons,
    2026-10-07). The index-4 basis (−4, −1), (0, −1) would give 4 pixels per
    copy and fewer copies; it is not tested, and the report says so.
- **Copy parameters.** Each copy carries its unit's pixel couplings
  unchanged and 1/n of the unit bias, so the w → ∞ limit is the unplaced
  model.
- **Tie coupling w.** Chosen per family on a development set from
  {1.5, 2, 2.5, 3} at 8 bits, jitter 0, random start, K = 1024, one
  development sampler seed. Round 2 found w = 1 too weak and w = 4 worse
  than 2 on `t4`.
  - *Bounds:* everything. Under `split` the ties share the coupling cap, so
    at w ≥ 0.375 the tie sets the coupling step (w/127 at 8 bits) and coarsens
    J = 0.35 and the pixel couplings. This is a property of `split`, and
    the report states it as such.
- **Bits and jitter.** Operating arm 8 bits, 10 percent jitter. Secondary:
  8 bits jitter 0, 6 bits jitter 0. 4 bits is exact-only (round 1: +0.033 on
  `t4`; round 2: +0.077 on `smooth8`, unplaced codebook).
- **Starts.** Continuation, random, reset every 5 images (N = 5 from round
  1). Streams of 20 images.
- **Sweeps.** K in {64, 256, 1024} per sample, as the row says. A float
  K = 4096 random-start reference locates the placed model's own
  equilibrium. Round 2 found `t4` still falling between 256 and 1024 at
  w = 2.
- **Estimate.** 64-draw pixelwise majority, ties to +1 (owner). Exact floor
  of the true posterior's own 64-draw majority: +0.0018 on `t4`, +0.0047 on
  `smooth8`.
- **Test set.** 500 fresh images per sampler seed (25 streams of 20), 1,000
  for the exact stage. Per-seed standard error on a gap about 0.003 to 0.005.
- **Sampler engine.** NumPy two-colour Gibbs over the constraint layer's
  quantized, jittered `PairwiseModel`. No THRML: it is not a constraint, and
  the engine is checked against the layer's exact chromatic kernels.

## Pre-registered expectations (from the probes)

1. `t4` passes stop (a): the tie-limit placed model is within 0.01 of
   Bayes at 8 bits (unplaced: +0.0001; the placed codebook's coupling step
   is coarser, so expect up to +0.005).
2. `t4` fails stop (b). Every operating cell at K ≤ 1024 is +0.015 to
   +0.035 above Bayes, against an unplaced floor of +0.0018. The float
   K = 4096 reference lands closer but above +0.01.
3. The mechanism is the tied-copy chains: the per-unit copy agreement is
   below 0.9 at K = 1024, and the gap falls with K beyond 1024.
4. Continuation is no better than a random start in any seed.
5. Jitter changes nothing (round 1: +0.0096 against +0.0098).
6. `smooth8` does not run. If it does, a random start fails (c) at every K
   (3.6e-6 to 3.7e-6 J per estimate against 3.43e-6 J, before relay nodes),
   and only continuation or reset can pass.

A miss on any of these is a result.

## Exact anchor and evidence classes

- `exact_reference`: the mixture samples, the Bayes posterior marginals and
  MPM error, the tie-limit placed model's equilibrium (float64, enumeration
  over 2^C hidden states times the transfer matrix), and every n-draw floor.
- `software_simulation`: every NumPy Gibbs cell (CPU).
- `calibrated_projection`: every joule, on both sides.

## Arms and controls

Per family (`smooth8` only if `t4` passes (a) and (b)):

**Stage E, exact (gating stop (a)).** The tie-limit placed model at float,
8, 6 and 4 bits, on the codebook set by w\*, on 1,000 test images: MPM error
and 64-draw majority error against Bayes. Also at the unplaced codebook, for
comparison with the probes.

**Stage D, development.** Four tie values x 300 development images, one
development sampler seed.

**Stage S, sampled.** At w\*, for each sampler seed s in {0, 1, 2}:

- operating arm: 8 bits, jitter 0.1;
- jitter control: 8 bits, jitter 0;
- precision control: 6 bits, jitter 0;

each from three starts at K in {64, 256, 1024}: 27 cells per seed, 81 per
family. Plus the float K = 4096 random-start reference per seed (3 cells).

Controls: Bayes; noisy input; the unplaced model's exact 64-draw floor; the
true posterior's 64-draw floor; the float K = 4096 reference.

## Primary decision rule (thresholds fixed now)

Replication unit: the sampler seed. "Gap" is the 64-draw majority's pixel
error minus the Bayes MPM error on the same images.

1. **Stop (a).** The tie-limit placed model's exact MPM gap at 8 bits is
   above 0.01 (point estimate, 1,000 images). If it fires, stage S does not
   run on that family and the mechanism is precision or the bias design,
   whichever the float row separates.
2. **Reaching.** A cell (bits, jitter, start, K) reaches if its gap is at
   most 0.01 in each of the three seeds (point estimates; the 95 percent
   stream bootstrap is reported beside each).
3. **Carried-state check** (lessons, 2026-10-14). A continuation or reset
   cell counts as reaching only if, in each seed, it is within 0.01 of the
   random-start cell at the same K (paired stream bootstrap upper bound).
4. **Stop (b).** No start reaches on the operating arm (8 bits, jitter 0.1)
   at any K ≤ 1024.
5. **Operating point.** If (b) does not fire: K\* is the smallest K at which
   some start reaches on the operating arm; at K\*, the start with the
   fewest writes per sample. Its counts are the headline. Because the
   point is chosen only among reaching cells, the recorded point is resolved
   by construction (lessons, 2026-10-15).
6. **Mechanism when (b) fires**, named from fixed comparisons:
   - *precision or jitter*, if the 8-bit jitter-0 arm or the float
     reference reaches where the operating arm does not;
   - *mixing within budget*, if the float K = 4096 reference is within 0.01
     in every seed but no K ≤ 1024 cell is;
   - *tied copies at finite w*, if the float K = 4096 reference also misses
     while the tie-limit exact model passes (a).
   Copy agreement (fraction of a unit's copies equal to its majority) at the
   last sweep is reported per cell as a diagnostic and does not gate.
7. **Stop (c), `smooth8` only.** The operating point's joule per 64-draw
   estimate at the profile's write constant is at least 3.43e-6 J (the
   rung at 1 pJ per MAC). The band's low end and the 0.1 and 10 pJ
   constants are reported beside it.

Any stop closes the denoising line at L ≤ 16 with its mechanism named, and
the real binarized set opens only on a positive (row).

## What one recorded sample means

One chain's 64 pixel states after K sweeps of two-colour Gibbs on the placed
model with the noisy image clamped, starting from the previous image's
final state in the stream (continuation), a uniform state (random), or a
uniform state every fifth image (reset). An estimate is the pixelwise
majority of 64 independent chains, ties to +1. Bayes: the exact marginal
posterior mode per pixel.

## Seeds

All fresh; none appears in an archive or a probe (probe seed 20261017;
P-0007 seeds 20261201 to 20261210; P-0006 seeds 20261101 to 20261108).

- Test images and noise: `default_rng([20261301, f])`.
- Development images and noise: `default_rng([20261302, f])`.
- Development sampler: `default_rng([20261303, f, w_index])`.
- Stage S chains: `default_rng([20261304, f, s, cell])`.
- Reference chains: `default_rng([20261305, f, s])`.
- Jitter words: (20261306, f, s).
- Bootstrap: `default_rng([20261307, f, s])`.

## Budgets and CPU estimate

Measured in round 2 on full K = 1024 runs (not compile-dominated):
2.05e-4 CPU-s per image per sweep for 64 chains on placed `t4`, 3.5e-4 on
`smooth8`. Relay nodes add an estimated 10 percent: 2.25e-4 and 3.85e-4.

- **`t4`:** stage S 27 cell-groups x 500 images x 1,344 sweeps, 1.1
  CPU-hours; reference 3 x 500 x 4,096, 0.4; development 0.1; exact stage
  and preflight 0.1. **About 1.7 CPU-hours.**
- **`smooth8` (only if `t4` passes):** stage S 1.9, reference 0.7,
  development 0.15, exact 0.15. About 2.9 CPU-hours.
- **Total: about 1.7 CPU-hours if `t4` stops, 4.6 if both run.** Round 2's
  figures are one-seed timings, and P-0007 ran 2.7x over a short-run
  calibration, so the runner calibrates on full-length cells and stops if
  the projection exceeds 8 CPU-hours. About 0.7 or 1.8 hours wall on three
  workers.

## Preflight (gating, probe seed)

1. Layout: every unit covers each pixel exactly once, sites are distinct,
   relay paths use only Z1 offsets, and `require_placement` passes with
   nothing relaxed on the operating arm's model.
2. Tie limit: summing each unit's quantized per-copy biases and taking its
   copies' common pixel couplings reproduces the collapsed model used by
   stage E, to 1e-12.
3. Sampler: the NumPy engine against the constraint layer's
   `exact_chromatic_kernels` on a sub-model of at most 12 free nodes (a
   2 x 2 pixel patch with one unit's copies and a relay), K in {1, 2, 4},
   400,000 chains, max |z| < 5.
4. Pricing: `count` and `price` reproduce a hand count of reads, writes,
   free-node writes and updates per sample for each start, and both ends of
   the band.
5. Oracle: the unplaced one-hot restriction reproduces the Bayes posterior
   to 1e-12 (round 1: 6.9e-15).

## Autosave and resume

The run may exceed 30 minutes, so the runner follows the
[autosave contract](../experiment-runner.md#autosave-and-resume-contract).
Work units are atomic: one exact stage per family, one development cell,
one stage S cell (family, arm, start, K, seed), one reference cell. The
`smooth8` units are created only after the `t4` verdict is written and
digested. Resume is the same command with `--resume` and the same output
directory. A unit test interrupts a small run and checks that the resumed
record is identical.

## Persistence and archive size

`study.json.gz`: the hashed request (family field arrays by SHA-256, layout
with relay paths, constraint requests with digests and relaxed reasons, w
grid, seeds); preflight; per-image wrong-pixel counts (uint8) for every
cell, Bayes and the noisy input; per-cell copy agreement; exact-stage
marginal summaries; pricing; CPU timings, kept out of the result digest;
the evaluation. At most 87 cells per family x 500 images, under 0.3 MB
gzipped.

Replay recomputes Bayes errors, the exact stage and the floors at 1e-12
relative, recomputes reaching, the operating point, mechanism, bootstraps,
prices and verdicts from the archived counts, and re-verifies placement. A
prefix check reruns one stage S cell and one development cell and requires
identical counts. CI replays without the prefix check (`--light`).

## Gate

`completion.json`, written last by the runner, must show:

- `status=dtm_exact_posterior_lattice_complete`;
- `families_run` (1 or 2), `exact_rows` equal to 8 times `families_run`
  (float, 8, 6, 4 bits at the placed and unplaced codebooks), and
  `sampled_cells` equal to 88 times `families_run` (4 development, 81
  stage S, 3 reference);
- `preflight_passed`, `placement_passed`, `references_recomputed`,
  `prefix_check_passed`, `replayed` all `true`.

It also carries w\*, the stop verdicts (a), (b), (c) per family, the named
mechanism, and the operating point with its counts and joule band. These
are scientific results, not integrity checks.

## Stop rules

- Preflight fails: stop before production. Do not tune tolerances.
- Calibration projects over 8 CPU-hours: stop and report.
- Stop (a) or (b) fires on `t4`: `smooth8` does not run.
- Two consecutive failed runs with the same error: stop and report.
- A replay mismatch is an integrity failure.

## What a negative result means

- **(a) fires.** The lattice's precision under `split`, with ties sharing
  the coupling cap, cannot hold the posterior even with infinite ties. The
  number is the precision gap.
- **(b) fires with "mixing within budget".** The placed model holds the
  posterior but tied-copy chains do not equilibrate in 1024 sweeps. The
  number is the gap at K = 1024 and the K at which the reference reaches.
- **(b) fires with "tied copies at finite w".** Tied copies at a tie strength
  the codebook can express do not represent a global latent. The number is
  the gap of the float reference.
- **(c) fires on `smooth8`.** The lattice holds and samples the posterior but
  costs at least as much per estimate as the archived rung at 1 pJ per MAC.
- **Any of these** closes the denoising line at L ≤ 16 with its mechanism
  named. None is an integrity failure.

## Not claimed

No Z1 measurement, latency or device noise. No energy outside the sealed
profile's counted operations (host transfer and control excluded; on the
digital side activations, memory traffic and the bias map excluded). No
lower bound on digital cost. No trained model, no L = 16 result, no real
image set, and no claim about layouts other than the one tested.
