# DTM small-image denoiser under the constraint layer: exploratory protocol (draft, proposal P-0006)

Drafted on 2026-10-14 for queue row `dtm-small-image` and revised the same
day (round 1). It is not frozen until the owner approves it, and no runner
exists yet. The probes are
[`2026-10-14-dtm-small-image-probe.md`](../research/2026-10-14-dtm-small-image-probe.md)
(one seed, 200 test images, J = 0.35) and
[`2026-10-14-dtm-small-image-probe-r1.md`](../research/2026-10-14-dtm-small-image-probe-r1.md)
(one seed, 500 test images, this protocol's J = 0.55 family and step
definition). Both are exploration with a NumPy sampler.

**Read b\* and K\* as a cost, not a win.** In the round-1 probe the
114-parameter logistic baseline was at least as good as every float DTM
cell on this family: ahead by 0.002 to 0.004 at converged K, and by 0.016
at T = 1, K = 64. If expectation 6 holds, the headline is that a
lattice-native EBM denoiser at L = 8 on this family buys no quality over a
small conventional model. The projected joule is then priced against a
digital model of equal error, which this study measures only in CPU
seconds. The amendment's four-week test is dated from the constraint
layer's merge, 2026-10-14.

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
  - The free nodes are never written. Each step **continues** from the
    state the layer region's free nodes already hold. For a stream's first
    image that state is arbitrary. For every later image it is the end state
    the region reached on the previous image of the same stream.
  - Per denoised sample this is T L^2 reads, T L^2 writes, 0 clamp-state
    changes, 0 free-node writes and 3 L^2 K T active updates.
  - A fresh uniform start would cost one full SRAM write per free node,
    3 L^2 T more writes per sample. That is 3.9 times the joule under the
    profile. It is kept only as a NumPy control, with those writes counted.
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
- **Joule band.** Each clamp load writes one bit, but the profile prices it
  as a full 153.6 pJ SRAM node write because no cheaper constant is
  published. Every operating-point joule is reported twice:
  - at the profile's write constant (the headline);
  - with the T L^2 clamp loads priced at the node-read constant, 1.692 pJ.
    This figure is labelled outside the profile.
  - At T = 1, K = 256 the round-1 probe gives 1.03e-8 J and 5.7e-10 J.

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
  - The family was chosen from the probe's exact scan. The round-1 probe
    then trained the fixed recipe on it once (one seed, T = 1 and 2). That
    run gave noisy 0.201, Bayes 0.062 and a float DTM at 0.069 (T = 1,
    K = 1024). The family trains, so the dead-model stop rule is not what
    stands between this choice and the metric.
  - J = 0.55 is above the nearest-neighbour 2D critical coupling, so images
    are ordered blobs. That is why the floor is low. It is also why a
    carried-over free-node state is slow to leave (see the start choice).
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
  - *Bounds:* I/O per sample scales with T. Neither probe saw a quality
    gain from T > 1, so T = 4 is dropped.
- **Training.** Contrastive learning with Adam (lr 0.03), batch 64,
  K_train = 64 (half burn-in), 1,600 steps per layer, and 4,000 training
  images. The training chains start uniform; training is software and is
  not priced.
  - *Bounds:* model quality. The first probe found K_train 64 better than
    16 by about 0.01 and 3x the steps no better. The round-1 probe confirmed
    the recipe on this family at 80 CPU-s per layer. Training is not varied.
- **Cap rule.** `split`, which takes the model's own max |w| and max |b|.
  - *Bounds:* nothing beyond bits. `absolute` was equivalent from 4 bits and
    needs a per-model number.
- **Free-node start: continuation.** Test images are ordered into 50
  streams of 20, assigned in test-set order. Each stream has 16 chains, one
  per draw. Each layer region carries its free-node state from one image of
  the stream to the next.
  - The first image of a stream starts from a uniform state. This is a
    software stand-in for the chip's arbitrary state and is not written.
  - *Bounds:* K\*. In round 1, continuation needed K = 256 at T = 1 where a
    random start needed 64 (+0.012 +- 0.002 at K = 64). The carried state
    is the previous image's denoised blob, often of another component.
    Sweeps cost little under the profile (+3 % from K = 64 to 256), so K\*
    moves and the joule barely does.
  - Stream-first images are also reported separately, and the per-stream
    lag-1 error autocorrelation is recorded. It was -0.03 to -0.08 in
    round 1.
- **Inference sweeps.** K in {16, 64, 256} per step on every arm. The float
  arm also runs K = 1024 as the convergence reference.
- **Test set.** 1,000 held-out images with fresh noise.
  - *Bounds:* resolution. The per-arm standard error is about 0.002, and
    paired differences resolve better.
- **Conventional baseline.** A 7 x 7 patch logistic denoiser with position
  bias (114 parameters, 2,000 Adam steps).
  - *Bounds:* "same quality" means against this model. It matched the DTM
    in the first probe and was 0.002 to 0.004 ahead of it in round 1.

## Pre-registered expectations (from the round-1 probe)

Expected values per denoised sample (host path, continuation, 0 free-node
writes):

| Configuration | Projected energy | Clamp loads at read constant (outside profile) | I/O share |
| --- | --- | --- | --- |
| T = 1, K = 256 | 1.03e-8 J | 5.7e-10 J | 96.6 % |
| T = 2, K = 64 | 2.01e-8 J | 6.1e-10 J | 99.1 % |
| T = 1, K = 64, random start, writes counted (control) | 3.95e-8 J | 2.98e-8 J | 99.8 % |

These follow from the sealed profile and are recomputed, not measured.

Expected outcomes, all from one seed and 500 images on this family:

1. The float DTM at K = 1024 sits about 0.007 above the Bayes floor (about
   0.06) and far below the noisy input (0.20).
2. 6 and 8 bits are within tolerance. 4 bits misses (about +0.018). 3 bits
   misses (about +0.026).
3. 10 percent jitter costs about 0.002 and is negligible.
4. Under continuation K\* is 256 at T = 1 and 64 or 256 at T = 2. The
   random-start control reaches tolerance at K = 64.
5. T = 2 is no better than T = 1.
6. The DTM at (b\*, K\*) matches the logistic baseline within 0.01, with
   the logistic ahead by about 0.002 to 0.004. Rule 5 returns "matches".
   It does not return "better".

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

Every arm runs through THRML under continuation at K in {16, 64, 256}. The
float arm also runs K = 1024. That gives (9 x 3 + 1) cells x 2 T x 3 seeds
= 168 THRML cells.

Controls:

- **CPU Gibbs.** The NumPy chromatic sampler on the float model under
  continuation at K in {16, 64, 256} (18 cells). This is the "CPU Gibbs on
  the same model" cost baseline and a sampler cross-check.
- **Random start.** The NumPy sampler on the float model with a fresh
  uniform free-node start every step, at K in {16, 64, 256} (18 cells). It
  is priced with its 3 L^2 T free-node writes counted. It shows what
  continuation costs in sweeps and saves in writes. It does not gate.
- **Identity.** The noisy input itself (0.20).
- **Bayes.** Exact MPM from the posterior marginals.
- **Logistic.** One per training seed.
- **3-bit positive control.** It must show a detectable loss, or the
  tolerance is reported as not resolving.

## Primary decision rule (thresholds fixed now)

Replication unit: the training seed. Within a seed, images are paired
across cells, and every cell sees the same streams in the same order.
Images within a stream share a carried state, so intervals use a 95 percent
paired bootstrap over the 50 streams (2,000 resamples, within each seed).

1. **K\* (per T, float arm, continuation).** K\* is the smallest K in
   {16, 64, 256} whose upper bound of (K - K1024) error is at most 0.01 in
   all seeds. If 256 fails, K\* is "> 256", and rules 2 and 3 are read at
   K = 256 with that flag.
2. **b\* (primary, per jitter and T, at K = K\* of that T).** b\* is the
   smallest tested width b such that, at b and every larger tested width,
   the upper bound of mean (quantized - float) majority error is at most
   0.01 in all three seeds. If 8 bits fails, b\* is "> 8".
3. **Jitter.** Jitter is negligible if, at K\*, every (b, seed) interval
   for (jitter 0.1 - jitter 0) lies within +-0.01. It **matters** if any
   interval lies above +0.01.
4. **Steps.** T = 2 **helps** if (T2 - T1) at the float arm, each T at its
   own K\*, has an upper bound below 0. It **does not** if the lower bound
   is at least -0.01. Otherwise it is inconclusive.
5. **Baseline.** The DTM matches the logistic baseline if the interval of
   (DTM at b\*, K\* - logistic) lies within +-0.01. It is **worse** if the
   lower bound exceeds 0.01. Otherwise it is inconclusive.
6. **Operating point.** The report gives these at (T, b\*, K\*) with the
   cheaper T when step 4 says T = 2 does not help:
   - projected joules per sample and per estimate, as the band defined
     under resource accounting;
   - CPU-s per estimate for THRML, NumPy Gibbs and logistic;
   - error relative to Bayes and to noisy input.

Conditional log-loss is reported beside every error, as the Jeffreys-smoothed
16-draw marginal against the exact posterior's log-loss. It does not gate.
Stream-first-image errors and the lag-1 error autocorrelation along streams
are reported per cell and do not gate.

## What one recorded sample means

One chain's pixel state after T denoising steps of K sweeps each, starting
from the noisy image. Each step's free nodes continue from the state that
layer region held after the previous image of the same stream and chain.
For a stream's first image they start from a uniform state. Chain d of a
stream is draw d for every image in it. The archive stores, per image and
cell:

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
- Stream-first uniform states and NumPy control chains:
  `default_rng([20261107, s, T, cell])`.

None of these is a probe seed (20261014, or the words (20261014, 101) of
round 1), and none appears in any archive. Unit tests use probe seeds only.

## Budgets and CPU estimate

- **Training.** 3 seeds x 3 layers (T1 plus T2's two) x about 80 CPU-s =
  about 12 CPU-minutes. Round 1 measured 80 CPU-s per layer.
- **THRML.** Warm rates are 8.4 CPU-s per 6,400 chains at K = 64 and 28 at
  K = 256 (L = 8). Each cell processes 16,000 chain-images per layer.
  - All arms at K in {16, 64, 256}: about 95 CPU-s per arm-layer, and
    9 arms x 3 layers x 3 seeds = about 2.1 CPU-hours.
  - Float arm at K = 1024: about 280 CPU-s per layer-seed, 9 of them =
    about 0.7 CPU-hours.
  - Continuation processes a stream's 20 images in sequence, so each THRML
    call holds 800 chains, not 16,000. That per-call overhead is not yet
    measured through THRML; the calibration step measures it.
  - JAX uses several threads per process, so CPU time exceeds wall time.
- **NumPy Gibbs (continuation and random start), exact posterior and
  logistic.** Under 15 CPU-minutes.
- **Preflight, prefix check and full replay.** About 0.3 CPU-hours.
- **Total about 3.5 to 4 CPU-hours.** That is about 1.5 to 2 hours of wall
  time on 3 workers, within `docs/environments.md` limits for this box and
  under the 8 CPU-hour flag. The runner calibrates on probe seeds first,
  including the 800-chain call overhead, and stops if the projection
  exceeds 6 CPU-hours.

## Preflight (gating)

The preflight runs on probe seeds and fails closed:

1. **Transfer matrix.** The transfer-matrix logZ and marginals equal brute
   force on 3 x 3 and 4 x 4 grids to 1e-12.
2. **Placement.** Every layer passes `require_placement`.
3. **Samplers.** THRML and NumPy float samplers agree on per-pixel
   up-frequency for 10 images x 6,400 chains at K = 64, with max |z| < 5.
   The check runs once from a uniform start and once continuing from a
   fixed archived state, so the carried-state path through
   `thrml_lowering.final_states(..., initial=...)` is covered.
4. **Lowering.** The quantized model and the lowered model agree, through
   `lower`, on weights.
5. **Pricing.** Pricing of one sample reproduces the counts above exactly,
   for continuation (0 free-node writes) and for the random-start control
   (3 L^2 T free-node writes), in both halves of the joule band.

## Autosave and resume

The run exceeds 30 minutes, so the runner follows the
[autosave contract](../experiment-runner.md#autosave-and-resume-contract).

- Work units are atomic: one trained model (seed, T), then one inference
  cell (seed, T, arm, K). A cell's streams run inside one unit, so no
  carried state crosses a unit boundary.
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
- a prefix check reruns one cell per T at K = 256 and requires identical
  counts.

CI replays the archive without the prefix check (`--light`).

## Gate

`completion.json`, written last by the runner, must show:

- `status=dtm_small_image_complete`;
- `seeds=3`;
- `trained_models=6`;
- `thrml_cells=168`;
- `numpy_cells=36`;
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
- Calibration projects over 6 CPU-hours: stop and report.
- The float DTM at K = 1024 is not below the noisy-input error in two
  seeds: stop after training and report. The trained model is not a
  denoiser, and no quantization result about it is meaningful.
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
- **K\* is "> 256".** A carried free-node state on this ordered family
  needs more sweeps than the tested range. The joule barely moves with K,
  so this is a sweep-count and CPU finding, not an energy one.

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
