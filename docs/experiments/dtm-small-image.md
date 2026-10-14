# DTM small-image denoiser under the constraint layer: exploratory protocol (draft, proposal P-0006)

Drafted on 2026-10-14 for queue row `dtm-small-image`. It is not frozen until
the owner approves it, and no runner exists yet. The probe is
[`docs/research/2026-10-14-dtm-small-image-probe.md`](../research/2026-10-14-dtm-small-image-probe.md)
(exploration: one seed, 200 test images, NumPy sampler).

## The number this study produces

The study produces the operating point of a lattice-native denoiser:

- **b\***, the smallest coupling bit width at which a lattice-native denoiser
  stays within 0.01 pixel error of its own float model;
- **K\***, the fewest sweeps per step that give the same;
- the **projected joule per denoised sample** at (b\*, K\*) under the
  declared I/O path;
- how far the denoiser sits from the exact Bayes floor and from a small
  conventional model.

A designer reads it as: "this task needs b\* bits, K\* sweeps and 2 T L^2
I/O operations per sample, which costs this much, and it gets this close to
optimal."

## Question

At what task quality does a small chained-EBM denoiser run under the
constraint layer? The study measures this as a function of:

- coupling bits;
- beta jitter;
- sweeps per step;
- reads and writes per sample.

It also asks what that costs against CPU Gibbs on the same model and against
a small conventional model of the same quality.

## Hardware constraints applied and relaxed (rule 8)

Applied:

- **Placement.** Every layer is built on the 16-offset lattice. Each pixel
  has a 2 x 2 site cell: pixel, clamp, latent A and latent B. Edges are Z1
  offsets on pixel-A and A-B pairs, plus pixel-own clamp. Maximum degree is
  16, and every layer must pass `require_placement` with nothing relaxed.
- **Two-colour chromatic Gibbs.** Colour is (u + v) mod 2.
- **No exchange.** There are no tempering arms. The layer has no exchange,
  and independent Gibbs won on denoising posteriors (lessons, 2026-10-02).
- **Quantization.** Couplings and biases use M5b's codebook at 3, 4, 6 and 8
  bits under the `split` cap rule.
- **Jitter.** Static per-site beta jitter is 0 or 10 percent.
- **I/O accounting.** I/O is counted per sample under the declared path.

Relaxed, each with its reason in the hashed request:

- `quantization` on the float arm, which is the reference the tolerance is
  measured against.
- `beta_jitter` on noise-free arms, which are the jitter control.

Training happens in software with float64 weights. Nothing is trained under
constraints, and the row says so.

## Resource accounting (rule 9)

- **Projected energy.** `thermo_lab.constraints.count` and `price` with the
  sealed Z1 profile and `update_rule = active_kernel`.
- **I/O path: host per step.** For each step t the host writes x^t into the
  layer's L^2 clamp nodes. The layer then runs K sweeps over its 3 L^2 free
  nodes, and the host reads the L^2 pixel nodes.
  - Per denoised sample this is T L^2 reads, T L^2 writes, 0 clamp-state
    changes and 3 L^2 K T active updates.
  - The probe priced on-chip chaining at 1.3 to 1.8 times more under the
    sealed profile, because every inter-layer handoff is two clamp-state
    changes per pixel, each a full SRAM write. Chaining is therefore not an
    arm.
  - The hypothetical free-copy bound is reported once, labelled outside the
    profile.
- **CPU time.** CPU time is `time.process_time` per recorded estimate, with
  steady state separated from the first call. It is measured for:
  - THRML through `thrml_lowering.final_states`;
  - the NumPy chromatic Gibbs on the same float model;
  - the exact posterior;
  - the logistic baseline.
- One **recorded estimate** is the pixelwise majority of 16 independent
  denoised samples. Its projected cost is 16 times the per-sample price.

Projected joules are `calibrated_projection`, CPU seconds are measurements
of software on this box, and neither is hardware evidence.

## Fixed choices and what they bound

- **Data family.** A four-component mixture (disk, vertical bar, horizontal
  bar, ring) of nearest-neighbour Ising grids, with J = 0.55, template field
  h0 = 0.2 and L = 8. Noise is symmetric flips at q = 0.2.
  - *Bounds:* the headroom. The probe scan gives noisy 0.201 against Bayes
    0.071 (headroom 0.131 +- 0.005) at this family. The probe family
    (J = 0.35, h0 = 0.35) had only 0.046, which compressed every comparison
    below noise.
  - The family is chosen from the probe's exact scan only. No denoiser was
    trained on it during the probe.
- **Exact reference.** The exact reference covers the data grid only. A
  site-by-site transfer matrix gives exact samples, likelihood and the
  flip-noise posterior.
  - *Bounds:* only the data side is exact. The denoiser's own likelihood is
    intractable (latents, long offsets) and is not reported. Quality is
    measured as held-out pixel error and conditional log-loss against the
    exact Bayes floor.
- **Synthetic set only.** No binarized digit or clothing set is on disk, and
  fetching one is an owner decision. A real set is a follow-up row.
- **Image size.** L = 8 (256 nodes per layer). L = 16 is priced but not
  trained.
  - *Bounds:* cost only, if the per-pixel task is local. The protocol makes
    no claim at L = 16.
- **Steps.** T in {1, 2}, with a per-step flip schedule that composes to q.
  - *Bounds:* I/O per sample scales with T. The probe saw no quality gain
    from T > 1, so T = 4 is dropped.
- **Training.** Contrastive learning with Adam (lr 0.03), batch 64,
  K_train = 64 (half burn-in), 1,600 steps per layer, and 4,000 training
  images.
  - *Bounds:* model quality. The probe found K_train 64 better than 16 by
    about 0.01 and 3x the steps no better. Training is not varied.
- **Cap rule.** `split`, which takes the model's own max |w| and max |b|.
  - *Bounds:* nothing beyond bits. `absolute` was equivalent from 4 bits and
    needs a per-model number.
- **Inference sweeps.** K in {16, 64, 256} per step, from a uniform random
  start each step (no write, by the path's definition).
- **Test set.** 1,000 held-out images with fresh noise.
  - *Bounds:* resolution. The per-arm standard error is about 0.002, and
    paired differences resolve better.
- **Conventional baseline.** A 7 x 7 patch logistic denoiser with position
  bias (114 parameters, 2,000 Adam steps).
  - *Bounds:* "same quality" means against this model. It matched the DTM
    in the probe.

## Pre-registered expectations (from the probe)

Expected values per denoised sample:

| Configuration | Projected energy | I/O share |
| --- | --- | --- |
| T = 1, K = 64 | 1.0e-8 J | 99.1 % |
| T = 2, K = 64 | 2.0e-8 J | 99.1 % |

These follow from the sealed profile and are recomputed, not measured.

Expected outcomes:

1. The float DTM majority error falls between the Bayes floor (about 0.07)
   and the noisy input (0.20), and well inside that gap.
2. 6 and 8 bits are within tolerance. 4 bits is about 0.01 worse and may
   miss. 3 bits misses.
3. 10 percent jitter is negligible.
4. K\* is 16 or 64.
5. T = 2 is no better than T = 1.
6. The logistic baseline matches the DTM within 0.01.

A miss on any of these is a result.

## Exact anchor and evidence classes

- `exact_reference`:
  - the mixture's likelihood;
  - exact training and test samples;
  - the exact posterior marginals;
  - the Bayes MPM error.
- `software_simulation`: the THRML and NumPy denoiser runs and the
  logistic baseline.
- `calibrated_projection`: every joule figure.

Labels stay separate in records and prose.

## Arms and controls

Per training seed s in {0, 1, 2} and T in {1, 2}, one trained model is
quantized into these arms:

- float (relaxed quantization, relaxed jitter);
- bits b in {3, 4, 6, 8} x jitter in {0, 0.1}.

Each arm runs through THRML at K in {16, 64, 256}. That gives 9 arms x
3 K x 2 T x 3 seeds = 162 THRML cells.

Controls:

- **CPU Gibbs.** The NumPy chromatic sampler on the float model at each K
  (18 cells). This is the "CPU Gibbs on the same model" cost baseline and a
  sampler cross-check.
- **Identity.** The noisy input itself (0.20).
- **Bayes.** Exact MPM from the posterior marginals.
- **Logistic.** One per training seed.
- **3-bit positive control.** It must show a detectable loss, or the
  tolerance is reported as not resolving.

## Primary decision rule (thresholds fixed now)

Replication unit: the training seed. Images are paired within a seed.
Intervals use a 95 percent paired bootstrap over the 1,000 test images,
2,000 resamples, within each seed.

1. **b\* (primary, per jitter and T, at K = 64).** b\* is the smallest
   tested width b such that, at b and every larger tested width, the upper
   bound of mean (quantized - float) majority error is at most 0.01 in
   all three seeds. If 8 bits fails, b\* is "> 8".
2. **K\* (per T, float arm).** K\* is the smallest K whose upper bound of
   (K - K256) error is at most 0.01 in all seeds.
3. **Jitter.** Jitter is negligible if, at K = 64, every (b, seed) interval
   for (jitter 0.1 - jitter 0) lies within +-0.01. It **matters** if any
   interval lies above +0.01.
4. **Steps.** T = 2 **helps** if (T2 - T1) at the float arm and K\* has an
   upper bound below 0. It **does not** if the lower bound is at least
   -0.01. Otherwise it is inconclusive.
5. **Baseline.** The DTM matches the logistic baseline if the interval of
   (DTM at b\*, K\* - logistic) lies within +-0.01. It is **worse** if the
   lower bound exceeds 0.01.
6. **Operating point.** The report gives these at (T, b\*, K\*) with the
   cheaper T when step 4 says T = 2 does not help:
   - projected joules per sample and per estimate;
   - CPU-s per estimate for THRML, NumPy Gibbs and logistic;
   - error relative to Bayes and to noisy input.

Conditional log-loss is reported beside every error, as the Jeffreys-smoothed
16-draw marginal against the exact posterior's log-loss. It does not gate.

## What one recorded sample means

One chain's pixel state after T denoising steps of K sweeps each, starting
from the noisy image. Each step starts its free nodes from a uniform random
state. The archive stores, per image and cell:

- the count of wrong pixels in the 16-draw majority;
- the mean wrong-pixel count over the draws;
- the conditional log-loss as float32.

It does not store states.

## Seeds

- Training data: `default_rng([20261101, s])`.
- Test images and noise: `default_rng(20261102)`.
- Training chains: `default_rng([20261103, s, T, layer])`.
- JAX root: 20261104.
- Jitter words: (20261105, s, T, layer).
- Logistic: `default_rng([20261106, s])`.

None of these is the probe seed (20261014), and none appears in any archive.
Unit tests use probe seeds only.

## Budgets and CPU estimate

- **Training.** 3 seeds x 3 layers (T1 plus T2's two) x about 82 CPU-s =
  about 12 CPU-minutes.
- **THRML.** Warm rates are 8.4 CPU-s per 6,400 chains at K = 64 and 28 at
  K = 256 (L = 8). Each cell runs 16,000 chains per layer. That gives about
  95 CPU-s per arm-layer over the three K values, and 9 arms x 3 layers x
  3 seeds = about 2.1 CPU-hours. JAX uses several threads per process, so
  CPU time exceeds wall time.
- **NumPy Gibbs, exact posterior and logistic.** Under 10 CPU-minutes.
- **Preflight, prefix check and full replay.** About 0.3 CPU-hours.
- **Total about 2.5 to 3 CPU-hours.** That is about 1 to 1.5 hours of wall
  time on 3 workers, within `docs/environments.md` limits for this box and
  under the 8 CPU-hour flag. The runner calibrates on probe seeds first and
  stops if the projection exceeds 5 CPU-hours.

## Preflight (gating)

The preflight runs on probe seeds and fails closed:

1. **Transfer matrix.** The transfer-matrix logZ and marginals equal brute
   force on 3 x 3 and 4 x 4 grids to 1e-12.
2. **Placement.** Every layer passes `require_placement`.
3. **Samplers.** THRML and NumPy float samplers agree on per-pixel
   up-frequency for 10 images x 6,400 chains at K = 64, with max |z| < 5.
4. **Lowering.** The quantized model and the lowered model agree, through
   `lower`, on weights.
5. **Pricing.** Pricing of one sample reproduces the counts above exactly.

## Autosave and resume

The run exceeds 30 minutes, so the runner follows the
[autosave contract](../experiment-runner.md#autosave-and-resume-contract).

- Work units are atomic: one trained model (seed, T), then one inference
  cell (seed, T, arm).
- A unit is reused only if the request digest and runner source hash match.
- Resume is the same command with `--resume` and the same output directory.
- A unit test interrupts a small run and checks that the resumed record is
  identical.

## Persistence and archive size

`study.json.gz` holds:

- the hashed request, with one `ConstraintRequest` and its digest per layer
  and arm, the relaxed constraints with reasons, bits, cap rule, jitter and
  jitter words, update rule and dtype float64;
- the preflight;
- trained weights (9 layers, about 9 kB each);
- per-image cell summaries;
- pricing;
- CPU timings, kept out of the result digest;
- the evaluation.

Expected size is under 1 MB gzipped.

Replay checks:

- it recomputes the exact references and Bayes errors and checks them
  numerically at 1e-12 relative;
- it recomputes pricing, bootstraps and verdicts from the archived counts;
- it re-verifies placement;
- a prefix check reruns one cell per T at K = 64 and requires identical
  counts.

CI replays the archive without the prefix check (`--light`).

## Gate

`completion.json`, written last by the runner, must show:

- `status=dtm_small_image_complete`;
- `seeds=3`;
- `trained_models=6`;
- `thrml_cells=162`;
- `numpy_cells=18`;
- `test_images=1000`;
- `preflight_passed=true`;
- `placement_passed=true`;
- `references_recomputed=true`;
- `prefix_check_passed=true`;
- `replayed=true`.

It also carries `b_star`, `k_star` and the verdicts. These are scientific
results, not integrity checks.

## Stop rules

- Preflight fails: stop before production and report. Do not tune
  tolerances.
- Calibration projects over 5 CPU-hours: stop and report.
- The float DTM at K = 256 is not below the noisy-input error in two seeds:
  stop after training and report. The trained model is not a denoiser, and
  no quantization result about it is meaningful.
- Two consecutive failed runs with the same error: stop and report.
- A replay mismatch is an integrity failure.

## What a negative result means

- **b\* is "> 8".** Trained lattice denoisers need more precision than the
  M5b codebook gives at this size, or `split` wastes range on a few large
  weights. Either is a spec finding.
- **Jitter matters.** Gain uniformity is a requirement for this task.
- **The DTM is worse than the logistic baseline.** At this size, a
  lattice-native EBM denoiser buys no quality over a 114-parameter
  conventional model. Its projected joule must then be weighed against a
  digital model of equal error, which this study prices only in CPU
  seconds.
- **T = 2 does not help.** The extra step is pure I/O cost.

None of these outcomes is an integrity failure.

## Not claimed

The study does not claim:

- any Z1 measurement, DAC behaviour or device noise;
- latency;
- energy outside the sealed profile's counted operations (host transfer,
  control and the digital baseline's energy are excluded);
- real image sets;
- L > 8 quality;
- training under constraints;
- on-chip chaining costs beyond the profile;
- comparison with the DTM paper's reported numbers, which use a different
  model, data and hardware assumptions.
