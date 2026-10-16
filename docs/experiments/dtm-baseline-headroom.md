# DTM baseline headroom: exploratory protocol (draft, proposal P-0007)

Drafted on 2026-10-16 for queue row `dtm-baseline-headroom`. It is not
frozen until the owner approves it, and no runner exists yet. The probe is
[`2026-10-16-dtm-baseline-headroom-probe.md`](../research/2026-10-16-dtm-baseline-headroom-probe.md)
(exact scan, no EBM, one ladder seed, 1,000 test images per family). It is
exploration. Sources: [Geman and Geman 1984](../knowledge/sources/geman-gibbs-restoration.md),
[Besag 1986](../knowledge/sources/besag-dirty-pictures.md).

**What P-0006 found, kept apart.** Its quality result is a positive: the
lattice denoiser recovered 94 percent of the noisy-to-Bayes headroom (0.075
against Bayes 0.067 and noisy 0.199). Its task result is a negative: a
114-parameter logistic was at least as good in every seed. This study asks
whether the negative is a property of the family.

## The number this study produces

On each qualifying family at L = 8, the **break-even energy per
multiply-accumulate**: the digital cost per MAC at which the cheapest
conventional model that matches the lattice denoiser's error (within 0.01)
costs as much per estimate as the denoiser's projected joule. It is given
at both ends of the clamp-constant band.

A designer reads it as: "on this task, at this quality, the p-bit array is
cheaper than a digital denoiser only if the digital side spends more than
this many picojoules per MAC." If no family qualifies, the number is
"none at L = 8": a conventional model at least as good costs less than X on
every tested family.

## Question

Is there a small binary denoising family at L = 8 with an exact posterior on
which the smallest conventional model within 0.01 of the P-0006 lattice
denoiser's error is costly? On such a family, does the denoiser beat that
model's projected cost at its recorded operating point under the constraint
layer?

## Hardware constraints applied and relaxed (rule 8)

The DTM side is P-0006's model, run as P-0006 ran it:

- **Applied.** Placement on the 16-offset lattice with nothing relaxed (2 x 2
  site cell: pixel, clamp, latents A and B; maximum degree 16;
  `require_placement` must pass). M5b codebook at 4 bits under `split`.
  Static 10 percent per-site beta jitter. Two-colour chromatic Gibbs, no
  exchange. I/O counted per sample on the host-per-step path with
  continuation.
- **Relaxed, with reasons in the hashed request.** `quantization` and
  `beta_jitter` on the float K = 1024 reference arm, which is the convergence
  reference. Nothing is relaxed on the operating-point arm.
- **Outside the layer.** Training is float64 software, as in P-0006. The
  digital ladder is float32 software and is not a hardware model.

## Resource accounting (rule 9) and the queue's standing rules

- **DTM.** Reads, writes, clamp changes and active updates per sample from
  `thermo_lab.constraints.count`, priced by `price` under the sealed Z1
  profile. At T = 2, K = 64, L = 8 that is 128 reads, 128 writes, 0 clamp
  changes and 24,576 updates per sample. The headline is the count. The
  joule is a band: 2.01e-8 J per sample at the profile's write constant,
  6.1e-10 J with the 128 clamp loads at the read constant (1.692 pJ, outside
  the profile). One recorded estimate is a 16-draw majority, so 3.21e-7 J and
  9.7e-9 J per estimate.
- **Baseline.** MACs per image for each ladder rung (convolution and
  pooled-feature MACs; activations, the bias map and the sigmoid are not
  counted), priced at 0.1, 1 and 10 pJ per MAC, labelled
  `calibrated_projection` with the constant stated. One baseline evaluation
  is one estimate.
- **CPU seconds** (`time.process_time`) are reported for both sides beside
  the projections and do not gate.
- Both standing rules of `docs/research-queue.md` are met: reads and writes
  per sample are the headline and the joule is a band; the baseline is priced
  per MAC at the three constants.

## The cost threshold X

X = **3.2e5 MACs per image** (exactly the DTM's per-estimate joule at the
profile's write constant divided by 1 pJ per MAC, 320,826 MACs, recomputed by
the runner from the profile). It is the middle constant at the headline end
of the band. A model below X is cheaper than the DTM at 1 pJ per MAC even when
clamp loads are priced as full writes. The other five break-evens (970 to
3.2e6 MACs) are reported beside every verdict.

## Fixed choices and what they bound

- **Families (L = 8, exact transfer-matrix posterior).** Six, from the
  probe; fields are fixed arrays built from the probe's code and seed and
  archived with the request:
  1. `anchor_p0006`: P-0006's four templates, J 0.55, h0 0.2, q 0.2. Local
     control; the ladder must reach Bayes here or the ladder is broken.
  2. `t4_h05_q035`: the same four templates, J 0.35, h0 0.5, q 0.35.
  3. `smooth8_h05_q035`: eight smooth random templates, J 0.35, h0 0.5,
     q 0.35.
  4. `smooth16_h05_q035`: sixteen smooth random templates, same.
  5. `parity_J03_h06_q035`: quadrant parity (top-left sign a, bottom-right
     b, the off-diagonal quadrants a b), J 0.3, h0 0.6, q 0.35.
  6. `cornerkey_J03_h06_q035`: a smooth body whose sign is keyed by a 2 x 2
     corner, J 0.3, h0 0.6, q 0.35.
  - *Bounds:* whether any gap exists. The probe found that raising q or
    widening the window alone leaves no gap (full-image logistic within
    0.008 of Bayes at q 0.4), so those families are not repeated. Every
    gap the probe found came from component identity.
- **Digital ladder.** Nine rungs, all with a learned per-pixel bias map and
  sigmoid output: logistic on 7 x 7, 11 x 11 and 15 x 15 (full-image)
  windows; one hidden layer of 4 or 16 channels (5 x 5); two hidden layers
  of 8 or 32 channels (5 x 5); and two global-pooling rungs, the 8- and
  32-channel two-layer nets with each hidden layer's channel means fed back
  as extra inputs.
  - The pooling rungs are an addition to the row's list. The probe's 8-wide
    pooling net (243,200 MACs) reached quadrant parity within 0.0075 of
    Bayes where the 1.74e6-MAC plain net stayed 0.014 above. A ladder without
    them measures receptive field again, which the owner's amendment to the
    row set out to avoid.
  - *Bounds:* "smallest conventional model". A model outside the ladder
    could be cheaper. The report says so and does not claim a lower bound
    on digital cost.
- **Ladder training (the probe's binding choice).** Adam (0.9, 0.999,
  1e-8), batch 128, 6,000 steps, fresh noise every batch from a pool of
  20,000 exact samples. Learning rate chosen per (family, rung) from
  {0.003, 0.01, 0.03} by error on a separate 1,000-image validation set at
  ladder seed 0. Then three seeds at that rate. The rung's error on a family
  is the test error of the seed with the lowest validation error.
  - *Bounds:* every gap. The probe's single-rate, 2,000-step fits diverged
    on two families (gaps 0.093 and 0.222 for a rung that reached 0.0075
    elsewhere). Any rung whose best-seed validation error is above the noisy
    input's is recorded as `failed_to_train` and does not count as reaching.
- **DTM model and recipe.** P-0006's layout, T = 2, contrastive Adam
  (lr 0.03, batch 64, K_train 64, 1,600 steps per layer), 4,000 training
  images, imported from `thermo_lab.dtm_small_image` unchanged. Not tuned.
  - *Bounds:* the DTM's error. The recipe was tuned for a local family. On a
    family that needs component identity it may be worse than it could be.
    The report says so; the row asks about the recorded operating point.
- **Operating point.** 4 bits, `split`, 10 percent jitter, K = 64, T = 2,
  host-per-step, continuation (50 streams of 20 images, 16 chains per
  stream). P-0006's designer point, not re-derived here.
- **Image size.** L = 8 only. A qualifying family opens L = 16 as a
  follow-up for the owner; this study does not run it.
- **Test set.** 1,000 images per family with fresh noise; standard error
  about 0.002 to 0.003 on a gap, smaller on paired differences.

## Pre-registered expectations (from the probe)

1. On `anchor_p0006` the 7 x 7 logistic is within 0.01 of Bayes (probe
   +0.004). It is the control the ladder must pass.
2. On families 2 to 6, no rung below X (logistic windows, one-layer nets,
   8-wide two-layer net: at most 128,000 MACs) is within 0.01 of Bayes. The
   probe's gaps for those rungs were 0.017 to 0.13.
3. With tuned rates and three seeds, the 8-wide pooling rung (about 2.4e5
   MACs, below X) reaches Bayes + 0.01 on quadrant parity, and possibly on
   two more families. Expected candidates: two to four of families 2 to 6.
4. The DTM at the operating point sits 0.02 to 0.06 above Bayes on the
   candidates. This is a guess: no EBM was trained in the probe, and the
   P-0006 recipe has no global latent.
5. The baseline for the DTM's error is therefore cheap (a logistic window
   or a one-layer net, below X), and no family qualifies. That would be the
   line's second ladder-at-least-as-good result.

A miss on any of these is a result.

## Exact anchor and evidence classes

- `exact_reference`: the mixtures' samples, posterior marginals and Bayes
  MPM error (float64 transfer matrix, checked against brute force on 3 x 3
  and 4 x 4 grids to 1e-12).
- `software_simulation`: the THRML and NumPy DTM runs; the JAX ladder.
- `calibrated_projection`: every joule, on both sides.

## Arms and controls

**Stage 1, ladder (all six families).** 9 rungs x (3 rates at seed 0, then
seeds 1 and 2 at the chosen rate) = 45 fits per family, 270 fits.

**Stage 2, DTM (candidate families only, at most three).** Per candidate
and training seed s in {0, 1, 2}: one T = 2 model, then

- **operating point:** THRML, 4 bits, jitter 0.1, K = 64, continuation;
- **float reference:** THRML, float, K = 1024, continuation;
- **random-start control:** NumPy, float, K = 64, fresh uniform free-node
  start every step (3 L^2 T free-node writes counted);
- **continuation control:** NumPy, float, K = 64, continuation (the
  carried-state check, lessons 2026-10-14).

That is 6 THRML cells and 6 NumPy cells per family, 36 cells at most.

Controls: Bayes; noisy input; the anchor family's ladder (stage 1); a
**stop on dead models** (below).

## Candidate rule (stage 1, fixed now)

A rung **reaches** a target error e on a family if its best-seed test error
is at most e + 0.01 (point estimate; the paired bootstrap interval over
images is reported beside it). A family is a **candidate** if no rung with
MACs per image below X reaches the Bayes MPM error. If more than three
families are candidates, stage 2 runs on the three with the largest gap of
their best rung below X. The anchor family is never a candidate; if the
ladder fails to reach Bayes there, stage 2 does not run (stop rule).

## Primary decision rule (stage 2, thresholds fixed now)

Replication unit: the DTM training seed. Per seed s on candidate f:

1. **DTM error** e_s: 16-draw majority pixel error at the operating point.
2. **Baseline** b_s: the rung with the fewest MACs that reaches e_s (rule
   above). If none reaches it, b_s is "none in the ladder".
3. **Seed verdict.** The seed favours the DTM if b_s costs at least X or is
   "none". Otherwise the ladder wins the seed.
4. **Family qualifies** if all three seeds favour the DTM. The ladder wins
   the family if all three seeds go to the ladder. Otherwise the family is
   inconclusive.
5. **The number.** For each qualifying family and seed: break-even pJ per
   MAC = (DTM joule per estimate) / (MACs of b_s), at the write constant and
   with clamp loads at the read constant. If b_s is "none", the break-even is
   reported as "below (joule) / (MACs of the largest rung)", an upper bound.
6. **Operating point resolved.** In each seed, the paired interval of
   (operating point - float K = 1024) must lie below +0.01, and the
   continuation control must be within 0.01 of the random-start control
   (paired 95 percent stream bootstrap, as P-0006). A seed that fails either
   is reported as `operating_point_unresolved` and its verdict is marked so.
   No rule falls back to another arm: the recorded point is P-0006's, and an
   unresolved seed is a result about it (lessons, 2026-10-15).

Intervals: 95 percent paired bootstrap over the 50 streams within seed
(2,000 resamples) for DTM comparisons; over images for ladder-only
comparisons. Conditional log-loss is reported for the DTM and the Bayes
posterior beside every error and does not gate.

## What one recorded sample means

DTM: one chain's pixel state after two denoising steps of 64 sweeps from the
noisy image, continuing each layer region's free-node state from the
previous image of the stream (uniform state for a stream's first image). An
estimate is the pixelwise majority of 16 such samples (ties to +1).
Ladder: one forward pass on the noisy image, thresholded at 0.5. Bayes: the
exact marginal posterior mode per pixel.

## Seeds

All fresh; none appears in an archive or a probe (probe seed 20261016,
P-0006 seeds 20261101 to 20261108).

- Family fields: rebuilt from the probe's code at probe seed 20261016 and
  archived as arrays with their SHA-256 (they are inputs, not draws).
- Training pools and DTM training data: `default_rng([20261201, f, s])`.
- Validation images and noise: `default_rng([20261202, f])`.
- Test images and noise: `default_rng([20261203, f])`.
- Ladder batches and noise: `default_rng([20261204, f, rung, seed, rate])`;
  JAX init key `PRNGKey(20261205)` folded with (f, rung, seed).
- DTM training chains: `default_rng([20261206, f, s, layer])`.
- THRML chain keys: `fold_in(key(20261207), f, s, cell, image, layer)`.
- Jitter words: (20261208, f, s, layer).
- Stream states and NumPy chains: `default_rng([20261209, f, s, cell])`.
- Bootstrap: `default_rng([20261210, f, s])`.

## Budgets and CPU estimate

From the probe's timings (3,000-step fits: 5 to 80 CPU-s; exact posterior
about 7 CPU-s per 1,000 images) and P-0006's run (4.8 CPU-hours for 168
THRML and 36 NumPy cells, 1.4 times its calibration):

- **Ladder:** 45 fits per family at 6,000 steps, about 0.7 CPU-hours per
  family, 4.2 CPU-hours for six.
- **DTM:** per candidate, training 3 x 2 layers x 80 CPU-s (8 minutes), the
  K = 1024 float cells about 0.7 CPU-hours, the rest about 0.2. About 1.1
  CPU-hours per candidate, 3.3 for three.
- **Exact references, preflight, replay:** about 0.3 CPU-hours.
- **Total: about 7.8 CPU-hours at three candidates**, 4.5 at none. This is
  close to the 8 CPU-hour flag, and P-0006 overran its calibration by 1.4
  times, so the run may exceed 8. About 3 hours wall on 3 workers.
- The runner calibrates on the probe seed and stops if the projection
  exceeds 10 CPU-hours.

## Preflight (gating, probe seed)

1. Transfer matrix against brute force on 3 x 3 and 4 x 4 grids for every
   family's field pattern at that size, to 1e-12.
2. Placement: every DTM layer passes `require_placement`.
3. THRML against NumPy on one candidate-shaped model (10 images x 6,400
   chains, K = 64), uniform and carried start, max |z| < 5, as P-0006.
4. Ladder MAC and parameter counts equal a hand count for every rung at
   L = 8.
5. Pricing reproduces 128 reads, 128 writes and 24,576 updates per sample
   and both ends of the band exactly, and X = 320,826 MACs (rounded from the
   profile).

## Autosave and resume

The run exceeds 30 minutes, so the runner follows the
[autosave contract](../experiment-runner.md#autosave-and-resume-contract).
Work units are atomic: one exact reference per family, one ladder fit
(family, rung, seed, rate), one trained DTM (family, seed), one inference
cell. Stage 2 units are created only after stage 1 is complete and its
candidate list is written and digested. Resume is the same command with
`--resume` and the same output directory. A unit test interrupts a small run
and checks that the resumed record is identical.

## Persistence and archive size

`study.json.gz`: the hashed request (families' field arrays, ladder spec,
constraint requests with digests and relaxed reasons, seeds); preflight;
per-image wrong-pixel counts (uint8) for every ladder fit's test pass, Bayes
and noisy input; validation errors per fit; per-image DTM cell summaries as
P-0006; trained DTM weights; pricing; CPU timings, kept out of the result
digest; the evaluation. Ladder weights are not archived (up to 27,000
parameters x 270 fits); the replay re-runs a prefix instead. Expected size
under 1.5 MB gzipped.

Replay recomputes exact references and Bayes errors at 1e-12 relative,
recomputes candidacy, baselines, bootstraps, prices and verdicts from the
archived counts, and re-verifies placement. A prefix check refits one ladder
rung on one family and reruns one DTM cell and requires identical counts.
CI replays without the prefix check (`--light`).

## Gate

`completion.json`, written last by the runner, must show:

- `status=dtm_baseline_headroom_complete`;
- `families=6`, `ladder_fits=270`, `test_images_per_family=1000`;
- `candidates` (0 to 3) and `dtm_cells` equal to 12 times `candidates`;
- `preflight_passed`, `placement_passed`, `references_recomputed`,
  `anchor_control_passed`, `prefix_check_passed`, `replayed` all `true`.

It also carries X, the candidate list, the per-family verdicts and the
break-evens. These are scientific results, not integrity checks.

## Stop rules

- Preflight fails: stop before production. Do not tune tolerances.
- The ladder does not reach Bayes + 0.01 on `anchor_p0006`: stop after
  stage 1 and report the ladder as broken. No stage 2.
- Calibration projects over 10 CPU-hours: stop and report.
- On a candidate, the float DTM at K = 1024 is not below the noisy input in
  two seeds: stop that family and report the model as not a denoiser.
- Two consecutive failed runs with the same error: stop and report.
- A replay mismatch is an integrity failure.

## What a negative result means

- **No candidate family.** A cheap digital model (under X) reaches the Bayes
  floor on every family tested, including those that need component
  identity. The line stops at L = 8 on exact synthetic families. This is the
  second ladder-at-least-as-good result on the denoising task, so the next
  review takes it to the owner as a premise question, not a third variant.
  A real binarized set is the natural next test and needs a download the
  owner opens.
- **Candidates exist but no family qualifies.** The DTM's own error is far
  enough above Bayes that a cheap model matches it. The lattice-native
  model, not the task, is the limit at this recipe. The same premise rule
  applies.
- **A family qualifies only at the read-constant end of the band.** The
  advantage rests on the unpublished clamp constant and is reported that
  way.
- **The operating point does not resolve on a family.** P-0006's designer
  point does not transfer to a family with a global latent. That is a
  finding about the point.

None of these outcomes is an integrity failure.

## Not claimed

No Z1 measurement, latency or device noise. No energy outside the sealed
profile's counted operations (host transfer and control excluded on the DTM
side; activations, memory traffic and the bias map excluded on the digital
side). No lower bound on digital cost: a model outside the ladder could be
cheaper. No L = 16 result, no real image set, no training under
constraints, and no comparison with the DTM paper's numbers.
