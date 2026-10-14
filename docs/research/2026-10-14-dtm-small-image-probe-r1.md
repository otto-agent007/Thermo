# dtm-small-image probe, round 1 (P-0006)

Date: 2026-10-14. Label: exploration, not evidence. Script:
[`dtm_small_image_probe_r1.py`](dtm_small_image_probe_r1.py). Output:
[`2026-10-14-dtm-small-image-probe-r1.json`](2026-10-14-dtm-small-image-probe-r1.json).
CPU only, single-thread BLAS, Python 3.11. Seed words (20261014, 101), probe
seeds only. One run took 471 CPU-s. The script was `ruff format`ted after
the run; formatting only, no logic change.

It answers two review questions before the protocol freezes:

1. How should the free nodes start each step? The draft started them from a
   fresh uniform state and counted no write for it, which the chip cannot do.
2. Does the fixed training recipe work on the J = 0.55, h0 = 0.2 family? The
   draft chose that family from the exact headroom scan alone.

Evidence classes:

- Bayes errors and exact samples are `exact_reference`.
- Denoiser and logistic errors are `software_simulation` in float64 NumPy,
  on one training seed and 500 test images.
- Joule figures are `calibrated_projection`s from the sealed Z1 profile.

Nothing here is a hardware measurement.

## Setup

- Family L = 8, J = 0.55, h0 = 0.2, q = 0.2. There are 4,000 training images
  and 500 test images.
- Training follows the draft recipe: K_train 64, 1,600 Adam steps per layer,
  batch 64, for T = 1 and T = 2.
- **Continuation** start: 500 test images form 25 streams of 20, and each
  stream has 16 chains. Each layer region keeps its free-node state from one
  image to the next. Only a stream's first image starts from a uniform
  (arbitrary, unwritten) state.
- **Random** start: the draft definition, a uniform start on every step.
- Differences are paired over images and use the majority-of-16 error.

## Findings

1. **The family trains, and its floor is reachable.** Noisy input is 0.201
   and Bayes MPM is 0.062 (SE 0.002). The float DTM reaches 0.0685 (T = 1)
   and 0.0726 (T = 2) at K = 1024 from random starts. The "not below noisy
   input" stop rule is nowhere near tripping. Training took 80 CPU-s per
   layer, as the draft assumed. Max |w| is 2.2 and max |b| is 1.9.
2. **The logistic baseline is at least as good as the DTM.** The
   114-parameter 7 x 7 logistic scores 0.0649 (SE 0.002). The DTM's
   converged cells are 0.002 to 0.004 worse, within 0.01. At T = 1,
   continuation, K = 64 the DTM is 0.016 +- 0.0025 worse. On this family
   the DTM buys no quality over the conventional model.
3. **Continuation mixes worse and moves K\*.** Errors are relative to the
   random-start K = 1024 cell of the same model:

   | T | K | Random start | Continuation |
   | --- | --- | --- | --- |
   | 1 | 16 | +0.020 +- 0.002 | +0.104 +- 0.009 |
   | 1 | 64 | +0.0005 +- 0.001 | +0.012 +- 0.002 |
   | 1 | 256 | -0.0006 +- 0.001 | -0.0016 +- 0.001 |
   | 2 | 16 | +0.003 +- 0.001 | +0.036 +- 0.003 |
   | 2 | 64 | -0.0001 +- 0.001 | +0.0045 +- 0.0013 |
   | 2 | 256 | +0.0001 +- 0.001 | +0.002 +- 0.001 |

   - The continued state is the previous image's denoised blob, often of
     another component, so it is a misleading warm start. At J = 0.55 it is
     slow to leave.
   - Under the 0.01 rule, K\* is 64 from random starts. Under continuation
     it is 256 at T = 1 and borderline 64 at T = 2.
   - The lag-1 autocorrelation of per-image error along a stream is -0.03
     to -0.08. The carried state leaves no positive error correlation
     between consecutive images.
4. **Continuation is still about 4x cheaper.** The joule is I/O-bound, so
   extra sweeps barely move it. Values are per denoised sample, host path:

   | Configuration | Writes | Profile | Clamp loads at read constant (outside profile) |
   | --- | --- | --- | --- |
   | T1, K64, random start, writes counted | 256 | 3.95e-8 J | 2.98e-8 J |
   | T1, K256, continuation | 64 | 1.03e-8 J | 5.65e-10 J |
   | T2, K64, continuation | 128 | 2.01e-8 J | 6.07e-10 J |

   Counting the random start's writes makes per-step writes 4 L^2 and
   multiplies the joule by 3.9. Continuation at K = 256 costs 3 % more than
   at K = 64.
5. **Bits and jitter, at T = 1, continuation, K = 64:**
   - 8 bits: -0.0001 +- 0.0012 against float.
   - 6 bits: +0.001 +- 0.0012.
   - 4 bits: +0.018 +- 0.003, which now misses clearly.
   - 3 bits: +0.026 +- 0.003.
   - 10 % jitter adds about 0.002 at 6 and 8 bits.
6. **T = 2 does not help.** At the float arm, continuation, K = 64, T2 - T1
   is -0.0036 +- 0.002. At converged K, T = 2 is 0.004 to 0.008 worse than
   T = 1.

## What changes in the protocol

- The step continues from each region's current free-node state. That costs
  no write and is what the hardware does. Random start is kept as a
  NumPy-only control with its writes counted.
- A float-arm K = 1024 reference is added, so K\* = 256 is testable.
- b\* and jitter are read at K\*, not at a fixed K = 64. Under
  continuation, K = 64 is not converged at T = 1.
- Bootstrap resamples streams, not images.
- Expectations are re-registered from this family.
