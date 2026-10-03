# Sampling time to accuracy and denoising transfer: exploratory protocol

Frozen on 2026-10-02 before sampling. Continue the approved CPU experiments:
does tempering's accuracy per redraw survive execution overhead and stronger
baselines, and does it help a different inference family?

## Fixed choices and possible structural shortcuts

- Weighted cubic graphs use sizes 12/16, fresh seeds 200/201/202, and the
  previous ring-plus-disjoint-matching generator with integer weights 1..5.
  Couplings are float32(-weight/5). Each graph has zero fields and weak fields
  drawn from {-0.15,+0.15}, SeedSequence([20261004,n,seed]), rounded float32.
  The cold inverse temperature is 4. There are 12 graph targets.
- Denoising uses a 4x4 nearest-neighbor ferromagnetic Ising prior with J=0.4,
  beta=1, and six posterior targets: exact prior draws at seeds 100/101/102,
  each corrupted with independent pixel flips at rates 0.2 and 0.35. Use
  SeedSequence([20261004,seed,41]); noise uniforms are shared within each pair.
  Posterior fields are observed_spin*0.5*log((1-noise)/noise), rounded float32.
  Denoising is model-correct and small; it does not establish real-image quality.
- Reuse the unchanged October 1 JAX systematic-Gibbs sampler and ladder
  [0.25,0.5,1,2,4]. Denoising parameters are divided by 4 exactly in float32,
  so the cold law is its beta-1 posterior and the ladder has the same relative
  temperatures. Do not adapt the ladder to either family's outcomes.
- Arms: one long cold chain, five independent cold chains, five tempered
  replicas. On zero-field graphs additionally evaluate long and independent
  chains with analytic global-flip augmentation: average the four-spin histogram
  with its reversed-bin histogram, set spin marginals to 0.5, retain pair
  correlations. This costs no extra redraws; estimator overhead is timed.
  It can fix global orientation but cannot resolve different cut basins.

## Sampling, accuracy and decision

Sixteen independently seeded trials per target; JAX root 20261004, fold target
index then trial, split initialization and sampling. All arms share five uniform
initial states; the long chain starts in slot 4. Sampling keys fold in method
index 0/1/2 as before. Symmetry arms reuse their underlying traces, not new trials.

T = 16,64,256,1024,4096 sweeps per replica for independent/tempering; 5T for
long. Total redraws per trial are 5*T*n in every arm, including the first-quarter
burn-in; tempering also attempts 2T exchanges. Retain the same end-of-sweep
observations as before. Budgets are prefixes, not independently seeded trials.

Primary qualification per target/method/budget requires BOTH mean trial
four-spin joint TV <= 0.05 AND mean trial edge-correlation MAE <= 0.05.
Report marginal MAE and fraction of individual trials satisfying both limits
as secondary metrics. These are estimator-error criteria, not guarantees for
every future chain or for the full joint law. Exact references enumerate up
to 16 spins in float64 on the sampler's float32 coefficients.

Time to accuracy is the measured warm 16-trial batch time at the earliest
tested budget that qualifies and whose larger tested budgets also qualify.
If there is none, report not reached within the grid, without inventing a time
or speedup. This sustained criterion avoids selecting an isolated lucky crossing.
Report all cells, all per-trial errors, qualifying target counts, and paired
time ratios only where both methods qualify. Do not pool targets to rescue a
failing target. No universal superiority or hardware claim.

## Timing scope and experimental controls

Compile each method/size/horizon executable once with dynamic coefficients;
record compilation separately. Prepare identical initialization and method
keys before measurement and record their cost separately. Inputs are resident
on the CPU device. Warm every executable and the host estimator before timing.

Measure five repeated executions of each target/arm/budget, rotating arm order
each repetition to reduce order effects. Repeated timings reuse identical keys
and are not extra statistical trials. Each timed pipeline includes Gibbs
updates, replica exchanges, device trace collection, synchronization, transfer
to host, and computing all joint/marginal/edge estimates (including augmentation).
It excludes compilation, initialization, exact enumeration, comparisons with
the oracle, plotting and persistence. Record pipeline and sampler-plus-transfer
times separately. The primary time is the median pipeline time; record every
repeat and their range. Timings are warm amortized batch costs, not single-chain
latency, cold-start service latency or equal hardware work. No other studies or
test suites run concurrently with measurement. Compilation costs remain visible.

For every horizon, compare sampled states and exchange flags to the corresponding
prefix of the largest-horizon trace. All timing repeats must reproduce the same
estimates. Persist packed maximum-horizon states and exchange flags separately;
symmetry arms reuse them. No warm/long-chain extra evidence is discarded.

## Validation, replay and budget

Use the unchanged exact/empirical Gibbs/exchange fixture and authenticate its
source. Add tests for analytic flip augmentation versus explicit complemented
states, rejection on nonzero fields, posterior scaling, baseline metric agreement,
sustained threshold selection and censoring, and archived evidence replay.
Replay recomputes all exact references, every estimate/error/count, every timing
median and threshold decision, and authenticates requested inputs, source code
and traces. It cannot reproduce historical wall-clock times. Compare numeric
derived values with fixed atol=2e-12 and rtol=0 for BLAS reduction portability;
hashes, identities, counts and threshold outcomes remain exact. No evaluator from
an earlier archive is edited.

Use a fresh directory, write the immutable request before sampling, then replay
all evidence before writing completion last. Label sample evidence
`software_simulation` and enumeration `exact_reference`. Freeze source, protocol
and lock hashes. Gzip archived evidence, retain the original sources, and report
all negative results. This finite grid is expected to take minutes on the current
four-CPU/16-GiB environment, below the 30-minute autosave threshold. No GPU, new
language, dependency or remote service is required.
