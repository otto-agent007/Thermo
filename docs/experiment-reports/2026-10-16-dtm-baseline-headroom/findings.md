# DTM baseline headroom: findings

**The frozen decision rule lands on an unresolved arm. The result is
negative.** On all three candidate families the ladder wins every seed. All
nine seed verdicts carry `operating_point_unresolved`, because P-0006's
operating point fails rule 6 in every seed. There is no fallback arm (rule
6). Do not use P-0006's point (4 bits, `split`, 10 percent jitter, K = 64,
T = 2, continuation) on these families. This study resolves no operating
point. The verdict does not depend on that, because the 3,136-MAC logistic
matches even the DTM's float K = 1024 reference in all nine seeds (below).

Proposal P-0007, queue row `dtm-baseline-headroom`. The protocol is
[`docs/experiments/dtm-baseline-headroom.md`](../../experiments/dtm-baseline-headroom.md)
(approved at 1c91073, PR #122). The runner is
`src/thermo_lab/dtm_baseline_headroom.py` at commit a394916. It imports
`thermo_lab.dtm_small_image`, the probe and `thermo_lab.constraints`
unchanged. The run is one attempt on 2026-10-16, 08:54:54 to 11:04:09 UTC:
local CPU, 3 workers each pinned to one core, single-thread BLAS, clean
tree, no resume.

## The number

The protocol's number is the break-even pJ per MAC on a qualifying family.
No family qualifies, so there is no number to design against. The
structure is:

| Family | Bayes MPM | Noisy | Smallest rung within Bayes + 0.01 | Candidate | DTM error (s0, s1, s2) | Baseline b_s | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `anchor_p0006` (control) | 0.069 | 0.203 | `logistic_7x7`, 3,136 MACs | no (control) | not run | | |
| `t4_h05_q035` | 0.216 | 0.351 | `cnn2_w32_k5`, 1.74e6 | **yes** | 0.315, 0.294, 0.295 | `logistic_7x7` in all three | ladder wins (unresolved) |
| `smooth8_h05_q035` | 0.219 | 0.350 | `cnn2_w32_k5_gpool`, 3.43e6 | **yes** | 0.358, 0.366, 0.349 | `logistic_7x7` in all three | ladder wins (unresolved) |
| `smooth16_h05_q035` | 0.230 | 0.350 | none in the ladder | **yes** | 0.370, 0.327, 0.377 | `logistic_7x7` in all three | ladder wins (unresolved) |
| `parity_J03_h06_q035` | 0.131 | 0.349 | `cnn2_w8_k5_gpool`, 243,200 | no | not run | | |
| `cornerkey_J03_h06_q035` | 0.133 | 0.350 | `cnn2_w8_k5_gpool`, 243,200 | no | not run | | |

X = 320,826 MACs per image (from the sealed profile, protocol). Errors are
pixel error on 1,000 exact test images per family. Ladder errors are the
best-validation seed at the tuned rate.

## Stage 1: the ladder and the floor

- **Anchor control passed.** On `anchor_p0006` the 7 x 7 logistic is
  0.0043 [0.0029, 0.0057] above Bayes. Every rung reaches the floor.
- **Headroom is real on three families.** Below X, the best rung on each
  family is the 8-wide pooling net (243,200 MACs):
  - `t4_h05_q035`: +0.0105 [0.0079, 0.0131] above Bayes.
  - `smooth8_h05_q035`: +0.0178 [0.0135, 0.0219].
  - `smooth16_h05_q035`: +0.0148 [0.0108, 0.0188].
  - No rung, even at 3.4e6 MACs, reaches `smooth16`; the best is
    +0.0147 at 1.74e6 MACs.
  - The t4 margin is small. Its best rung below X is 0.0105 above Bayes,
    which misses the 0.01 threshold by 0.0005, and its interval straddles
    0.01.
- **Global pooling settles parity and corner-key.** The 8-wide pooling net
  reaches both at +0.0084 and +0.0057. This confirms the probe's reason for
  adding the pooling rungs.
- **No rung failed to train.** Every best-seed validation error is below
  the noisy input. Learning rates were tuned per rung at seed 0. The large
  rungs' seeds still spread by up to 0.019 (`smooth8`, 32-wide pooling net).
  So "smallest rung that reaches" is a three-seed best, not a guarantee.

## Stage 2: the DTM on the candidates

All 9 models are P-0006's recipe unchanged (T = 2, 1,600 steps per layer,
4,000 training images). Per seed, at the operating point, with paired
95 percent stream-bootstrap intervals:

| Family, seed | DTM - Bayes | DTM - noisy | op - float K = 1024 | random - continuation | THRML op - NumPy continuation |
| --- | --- | --- | --- | --- | --- |
| t4, 0 | +0.099 [0.093, 0.105] | -0.036 | +0.064 [0.059, 0.069] | -0.011 [-0.017, -0.006] | +0.043 |
| t4, 1 | +0.078 [0.071, 0.085] | -0.057 | +0.034 [0.029, 0.040] | -0.008 [-0.013, -0.004] | +0.025 |
| t4, 2 | +0.079 [0.073, 0.086] | -0.055 | +0.040 [0.036, 0.046] | -0.012 [-0.016, -0.008] | +0.022 |
| smooth8, 0 | +0.139 [0.130, 0.148] | +0.008 | +0.006 [-0.004, 0.017] | -0.034 [-0.041, -0.028] | -0.006 |
| smooth8, 1 | +0.147 [0.136, 0.157] | +0.016 | +0.046 [0.040, 0.053] | -0.062 [-0.069, -0.055] | -0.011 |
| smooth8, 2 | +0.130 [0.121, 0.138] | -0.001 | +0.053 [0.047, 0.059] | -0.038 [-0.044, -0.030] | +0.004 |
| smooth16, 0 | +0.140 [0.130, 0.149] | +0.020 | +0.090 [0.082, 0.097] | -0.027 [-0.032, -0.021] | +0.049 |
| smooth16, 1 | +0.097 [0.090, 0.103] | -0.023 | +0.036 [0.029, 0.043] | -0.035 [-0.043, -0.027] | -0.002 |
| smooth16, 2 | +0.147 [0.137, 0.157] | +0.027 | -0.018 [-0.028, -0.007] | -0.107 [-0.119, -0.094] | -0.044 |

- **Rule 6 fails in every seed.** The carried-state check fails in all
  nine: a fresh uniform start beats continuation, and every interval
  crosses -0.01. Precision and sweeps also fail in eight. Only `smooth16`
  seed 2 has its operating point within 0.01 of the float reference, and
  that reference (0.395) is itself worse than the noisy input.
- **The model, not only the point, is the limit.** The float K = 1024
  reference is 0.25 to 0.39. In no seed is it more than 0.01 below the
  7 x 7 logistic (0.255 to 0.272), the first rung on each family.
  - Secondary check (not a gate; `plot.py` prints it): with the float
    reference's error in place of e_s, b_s is still `logistic_7x7` in all
    nine seeds.
  - So the ladder wins whether or not the operating point resolves.
- **On the smooth families the DTM barely denoises.** At the operating
  point it is worse than the noisy input in four of six seeds. The float
  stop rule did not fire: each family has at most one seed whose float
  K = 1024 error is above the noisy input (`smooth8` s0 at 0.352;
  `smooth16` s2 at 0.395). `float-reference-check.json` is empty.
- **Conditional log-loss** (does not gate): 0.60 to 0.71 nats per pixel at
  the operating point, against Bayes's 0.46 to 0.48.
- **Bounds.** The DTM recipe was tuned on a local family (protocol). This
  says the recorded recipe and point do not carry to families with a
  global latent. It does not say no lattice-native model could.

## Pre-registered expectations

1. The anchor's 7 x 7 logistic is within 0.01 of Bayes. **Held** (+0.004).
2. No rung up to 128,000 MACs reaches Bayes + 0.01 on families 2 to 6.
   **Held.** The closest are corner-key's 8-wide net at +0.0145 and t4's at
   +0.0136.
3. The 8-wide pooling rung reaches parity, making two to four candidates.
   **Held.** It reached parity and corner-key, which left three candidates.
4. The DTM is 0.02 to 0.06 above Bayes on the candidates. **Missed.** It is
   0.078 to 0.147 above.
5. The baseline is cheap and no family qualifies. **Held.** The baseline is
   the 3,136-MAC logistic in every seed.

## Cost, both sides projected (standing rules)

All joules are `calibrated_projection` under the sealed Z1 profile on the
DTM side, and at a stated pJ per MAC on the digital side.

- **DTM reads and writes per sample** (host per step, continuation, L = 8,
  T = 2, K = 64): 128 node reads, 128 full SRAM writes, 24,576 updates and
  no clamp state changes. Per 16-sample estimate that is 2,048 reads,
  2,048 writes and 393,216 updates.
- **DTM joule per estimate, as a band.** It is 3.21e-7 J at the profile's
  write constant (2.01e-8 J per sample, 99.1 percent I/O). It is 9.72e-9 J
  with clamp loads priced at the node-read constant, which lies outside the
  profile.
- **Digital baseline.** `logistic_7x7` is 3,136 MACs per image: 3.1e-10 J at
  0.1 pJ, 3.1e-9 J at 1 pJ and 3.1e-8 J at 10 pJ per MAC. Its activations,
  memory traffic and bias map are excluded.
- **Break-even.** The DTM costs the same as the baseline at 102 pJ per MAC
  (write end) or 3.1 pJ per MAC (clamp-read end). Even then the baseline
  is the more accurate.
- **X itself.** X is the DTM's write-end joule at 1 pJ per MAC. It falls to
  9,718 MACs at the clamp-read end. So at the clamp-read end, X would
  shrink by 33 times. That moves no candidate, because no rung below
  243,200 MACs reached Bayes on the three candidates.

## CPU seconds

Software timings on this box (`time.process_time`), outside the result
digest:

- Per estimate:
  - THRML operating point: 0.020 s.
  - THRML float K = 1024: 0.33 s.
  - NumPy Gibbs: 0.012 s.
  - Exact posterior: 0.0018 to 0.0069 s per image.
  - Ladder: 7e-7 s (logistic) to 1.4e-4 s (32-wide pooling net) per image.
- Totals:
  - Ladder fits: 4.63 CPU-hours.
  - DTM training: 0.44 CPU-hours.
  - Inference cells: 0.94 CPU-hours.
  - Exact references, pools and preflight make up the rest, just over
    6 CPU-hours in all.
- **Calibration underestimated by 2.7x.** It projected 2.24 CPU-hours,
  against the protocol's own estimate of 7.8.
  - Its two-point step-cost fit returned negative per-step costs for most
    rungs, because short fits are dominated by compile time.
  - The stop at 10 CPU-hours therefore had no real protection here.
  - The protocol's own estimate (7.8) was closer.

## Integrity

`completion.json` was written by the runner after the full replay
(133.5 s):

- `status=dtm_baseline_headroom_complete`
- `families=6`, `ladder_fits=270`, `test_images_per_family=1000`
- `candidates=3` and `dtm_cells=36`
- `preflight_passed`, `placement_passed`, `references_recomputed`,
  `anchor_control_passed`, `prefix_check_passed` and `replayed` are all
  `true`.

What the replay did:

- Recomputed the exact references at 1e-12.
- Recomputed candidacy, baselines, bootstraps, prices and verdicts from the
  counts.
- Re-verified placement.
- Refit one ladder rung and reran one DTM cell, both with identical counts.

The committed archive replays with `--light`, including under
`OPENBLAS_CORETYPE=Prescott`.

Determinism: each worker is pinned to one core. With that pin, the float32
ladder fits were bitwise identical across cores and under Prescott in the
pre-run tests.

Archive: `evidence.tar.gz`, 838,376 bytes. It holds the request, study,
provenance, run log, sources and status files. Work units are excluded;
`manifest.json` lists member hashes.

## What this means

This is the second ladder-at-least-as-good result on the denoising task,
after P-0006. The protocol's negative case "candidates exist but no family
qualifies" applies. Headroom for a costly digital model exists at L = 8.
The lattice-native model at P-0006's recipe and point is far from using
it, and a 3,136-MAC logistic beats it. The protocol routes this to the
owner as a premise question, not a third variant. The natural next tests
are those the protocol and queue name:

- a real binarized set, which needs a download the owner opens;
- a different lattice-native model with a global latent.

L = 16 is not opened: it was conditional on a qualifying family.

## Evidence classes and claims

- `exact_reference`: the mixture samples, posteriors and Bayes MPM errors
  (float64 transfer matrix).
- `software_simulation`: the JAX float32 ladder on CPU, the THRML DTM cells
  and the NumPy controls.
- `calibrated_projection`: every joule.

There is no Z1 measurement, latency or device noise. No energy is claimed
outside the profile's counted operations. There is no lower bound on
digital cost, because a model outside the ladder could be cheaper. There
is no L = 16 result and no real image set. There is no hardware claim.
