# DTM small-image denoiser under the constraint layer: findings

Proposal P-0006, queue row `dtm-small-image`. The protocol is
[`docs/experiments/dtm-small-image.md`](../../experiments/dtm-small-image.md)
(approved at 711ed4b, merged by PR #111). The runner is
`src/thermo_lab/dtm_small_image.py` at commit c39e91e. The run is one attempt
on 2026-10-14, 22:17 to 23:11 UTC, local CPU, 3 workers, single-thread BLAS
and a clean tree.

## The number

The protocol asked for (b\*, K\*), the projected joule at that point, and the
distance to Bayes and to a small conventional model.

| | T = 1 (continuation) | T = 2 (continuation) |
| --- | --- | --- |
| K\* (float within 0.01 of K = 1024, all seeds) | **> 256** (seed 1 +0.018 [0.010, 0.027] at 256) | **64** |
| b\*, jitter 0 | **> 8** (seed 1's 8-bit interval reaches +0.011) | **4** |
| b\*, jitter 0.1 | **4** | **4** |
| 3-bit control detectable (all seeds) | **no** (seed 1: -0.011) | yes (+0.009 to +0.035) |
| Jitter 0.1 - 0 | negligible | negligible |
| DTM at (b\*, K\*) - logistic | inconclusive (seed 1 +0.034 to +0.051) | **matches** (+0.003 to +0.006, every interval above 0) |

T = 2 vs T = 1, each at its own K\*: inconclusive. Per seed the differences
are -0.0005 [-0.002, 0.001], -0.045 [-0.054, -0.035] and -0.005 [-0.007,
-0.002]. Rule 6 therefore codes T = 1 as the operating point (1.03e-8 J per
sample, error 0.088). The designer's number is the T = 2 row:

- **4 bits, 10 percent jitter, K = 64, T = 2.** Error 0.075 against Bayes
  0.067, the logistic baseline 0.071 and the noisy input 0.199.
- **Per sample:** 2.01e-8 J at the profile's write constant. With the 128
  clamp loads at the node-read constant (outside the profile), 6.1e-10 J.
  I/O share is 99.1 percent.
- **Per 16-draw estimate:** 3.2e-7 J, or 9.7e-9 J with the clamp loads at the
  read constant.
- Those joules are `calibrated_projection` under the sealed Z1 profile and
  are not a measurement.

Intervals are 95 percent paired bootstraps over the 50 streams, within seed
(2,000 resamples). Errors are 16-draw majority pixel errors.

## Pre-registered expectations

1. **Float DTM about 0.007 above Bayes at K = 1024: held at T = 2, missed at
   T = 1.** At T = 2 the float model at K = 1024 is 0.072, 0.071 and 0.072 by
   seed (Bayes 0.067). At T = 1 it is 0.072, 0.099 and 0.076. Seed 1's T = 1
   model at K = 1024 is still 0.032 above Bayes, and 0.020 behind its own
   NumPy random-start run at K = 64 (0.079). The model is a denoiser (stop
   rule passed in every seed). Continuation does not reach its stationary
   regime within 1,024 sweeps from the previous image's state.
2. **6 and 8 bits within tolerance, 4 bits misses (+0.018), 3 bits misses:
   partly wrong.** At T = 2, 4 bits costs only +0.002 to +0.004 and passes.
   3 bits misses (+0.009 to +0.035). At T = 1, seed 1 makes quantization
   look *better* than float (3 bits -0.011, 4 bits -0.014): quantizing
   changes which trapped state the chains carry, not just the model.
3. **Jitter costs about 0.002: held.** Every (bit width, seed) interval is
   inside +-0.01, at both T.
4. **K\* = 256 at T = 1 and 64 or 256 at T = 2; random start reaches
   tolerance at 64: T = 2 held (64); T = 1 missed (> 256).** The
   random-start control at T = 1 is 0.073, 0.079 and 0.075 at K = 64, below
   continuation in every seed at K = 16 and 64. At T = 2 random start and
   continuation agree within 0.002 from K = 64.
5. **T = 2 no better than T = 1: not returned.** The verdict is
   inconclusive, because seed 1's T = 1 model is trapped. Seeds 0 and 2 agree
   with the expectation within 0.005.
6. **DTM matches logistic, logistic ahead by 0.002 to 0.004: held at T = 2.**
   The logistic is ahead by 0.003 to 0.006, and rule 5 says "matches". It
   never says "better". At T = 1 the verdict is inconclusive.

## What the result means

- **A lattice-native EBM denoiser at L = 8 buys no quality over a
  114-parameter logistic model on this family.** This is the negative the
  protocol's opening paragraph anticipated. The logistic is ahead in every
  seed, at every tested T, and it is not close to Bayes either (0.071 against
  0.067).
- **Precision is not the binding constraint. Mixing from a carried state
  is.** 4 bits with `split` caps is enough wherever the chain mixes. The
  coupling cap is the trained max |w|, 2.0 to 3.7 across models.
- **Carrying the free-node state between images (no free-node writes) is not
  safe at T = 1 on this ordered family.** In one of three seeds the float
  model is still 0.018 above its own K = 1024 reference at 256 sweeps. That
  reference is itself 0.020 behind a random start at 64 sweeps. A random
  start costs 3.9 times the joule at T = 1, K = 64. Continuation instead
  costs a seed-dependent sweep count that this range does not bound.
- **T = 2 is the robust choice here.** Each step removes half the noise, so
  each layer's posterior is less ordered. In every seed the carried
  state is left behind within 64 sweeps. It doubles the I/O and therefore
  the joule: 2.0e-8 against 1.0e-8 J per sample.
- **Energy is I/O.** 96.6 percent at T = 1, K = 256 and 99.1 percent at
  T = 2, K = 64. Going from 64 to 256 sweeps adds 2.6 percent to the joule.
  A random start (free nodes rewritten) is 3.9 times continuation at T = 1,
  K = 64. The pricing reproduced the protocol's table exactly: 1.03e-8,
  2.01e-8 and 3.95e-8 J.

## Cost against conventional machines

CPU seconds per 16-draw estimate are steady-state `time.process_time` on this
box. They are software measurements, not hardware, and they are kept out of
the result digest.

| Method | CPU-s per estimate | Error |
| --- | --- | --- |
| THRML, T = 2, 4-bit, K = 64 | 0.054 | 0.075 |
| THRML, T = 1, 4-bit, K = 256 | 0.111 | 0.088 |
| NumPy chromatic Gibbs, float, T = 2, K = 64 | 0.025 | 0.072 |
| Exact posterior (transfer matrix) | 0.0021 | 0.067 (MPM) |
| 7 x 7 patch logistic | 5.2e-5 | 0.071 |

The digital model of equal or better error costs about 1,000 times less CPU
than the Gibbs denoiser on this box. The study does not price the
logistic's energy, so no joule-for-joule comparison is claimed.

## Other recorded diagnostics (non-gating)

- **THRML against NumPy on the same float model, continuation:** every
  interval is within 0.007. The largest are in seed 1 at T = 1, where both
  samplers are trapped (+0.007 at K = 64, -0.006 at K = 256).
- **Stream-first images** (uniform start), float model at T = 1, K = 256:
  0.081, against 0.089 for the rest. That is the carried-state penalty
  again.
- **Lag-1 error autocorrelation along streams:** -0.06 to +0.06, near zero.
- **Conditional log-loss** (Jeffreys 16-draw marginal) at T = 2, 4 bits,
  K = 64: 0.214 nats against 0.181 for the exact posterior and 0.191 for the
  logistic.
- **Preflight** (probe seeds):
  - transfer matrix against brute force within 7.4e-15;
  - THRML against NumPy max |z| 3.31 (uniform start) and 3.13 (fixed carried
    state), against the limit of 5;
  - every arm lowered to float64 with weights equal and placement passing at
    maximum degree 16;
  - pricing reproduced exactly.
- **Calibration** projected 3.37 CPU-hours. The run used 4.8 CPU-hours of
  unit time, including JAX compiles and multi-threaded JAX overhead, under
  the 6-hour stop.

## Integrity

`completion.json` shows:

- `status=dtm_small_image_complete`, `seeds=3`, `trained_models=6`;
- `thrml_cells=168`, `numpy_cells=36`, `test_images=1000`;
- `preflight_passed`, `placement_passed`, `references_recomputed`,
  `prefix_check_passed` and `replayed` all true.

Replay (66 s) does the following:

- re-authenticates the sources and every constraint request digest;
- recomputes the exact references at 1e-12 relative;
- re-verifies placement and rounding for all 72 quantized layer-arm models
  and rebuilds all 81 constraint requests by digest;
- recomputes pricing, bootstraps, verdicts and the operating point from the
  archived per-image counts;
- reruns the seed-0 float K = 1024 cell at each T, with identical counts.

The archive (`evidence.tar.gz`, 1.7 MB) holds the request, the preflight,
the trained weights, the per-image counts and log-losses, the evaluation,
the timings and source copies.

Labels:

- `exact_reference`: the mixture likelihood, exact samples, the posterior
  and the Bayes error.
- `software_simulation`: the THRML and NumPy denoisers, training and the
  logistic baseline.
- `calibrated_projection`: every joule.

No hardware claim. Training was in software with float64 weights and was
not under constraints.

## Not claimed

The study claims none of the following:

- a Z1 measurement;
- latency;
- energy outside the sealed profile's counted operations;
- real image sets;
- L > 8;
- training under constraints;
- on-chip chaining;
- any comparison with the DTM paper's numbers.

## Follow-ups (proposals only; the owner decides)

- A free-node start rule between continuation and random. For example, a
  reset every N images, priced with its writes. Seed 1 shows continuation's
  sweep count is unbounded on ordered families.
- A real binarized image set (owner download decision).
- A stronger conventional baseline is not needed to read this result: the
  small one already wins.
