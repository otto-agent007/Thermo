# Fresh-seed symmetry plus tempering: exploratory protocol

Frozen on 2026-10-02 before production sampling. Test whether the exploratory
global-flip improvement from the previous study survives fresh graph seeds and
measured execution cost against the strongest available Gibbs baselines.

## Targets and arms

Use six zero-field weighted cubic graphs: sizes 12 and 16, each at seeds
300, 301 and 302. Generate the same ring plus disjoint perfect matching as the
previous studies, then draw integer edge weights 1..5 with the same generator.
Couplings are float32(-weight/5), fields are zero, and cold inverse temperature
is 4. Sizes reuse seeds; these six small cases are not six independent draws
from a broad application population. No target selection after seeing errors.

Four arms: long-flip, independent-flip, tempering, tempering-flip. The first two
are the strongest ordinary baselines from the previous study. Retain all three
underlying samplers' maximum traces, including every hot replica. Do not alter
the archived sampler or previous evaluator. Reuse its analytic estimator:
average the four-spin histogram with its complemented-bin histogram, set
marginals to 0.5 and retain edge correlations. Symmetry adds no redraws and is
valid only at zero fields. Both tempering arms share exactly the same draws.

Keep the ladder [0.25,0.5,1,2,4], 16 independent trials, budgets
T=16,64,256,1024,4096, quarter burn-in and systematic Gibbs updates. Long runs
5T sweeps; independent and tempering run T on five replicas. Every trial pays
5*T*n redraws, including burn-in; each tempering arm also pays 2T swap attempts
and retains only its cold replica. Use fresh JAX root 20261005, fold target then
trial, split initialization from sampling, and fold the original method index
0/1/2 into sampling keys. Initialization is shared across arms. Prefixes,
symmetry variants and timing repetitions are not extra statistical trials.

## Accuracy and timing

Use exactly the previous qualification rule: BOTH mean trial four-spin joint
TV and mean trial edge-correlation MAE <= 0.05 at the selected budget and every
larger tested budget. Enumerate the same float32 model in float64 for exact
references. Report every cell, per-trial errors, marginal errors, individual
trial pass fractions and every reached/not-reached decision. These projected
metrics do not certify the full distribution or every individual trial.

Primary comparison: tempering-flip versus the faster qualifying symmetry-aware
ordinary baseline per target. Also compare the paired unaugmented tempering
arm. Report qualification counts and paired time ratios only where both sides
qualify; leave failed thresholds censored at the largest tested budget. Do not
choose methods using pooled target errors or drop adverse outcomes. Ratios
against the best baseline are descriptive, not an independently tuned selector.

Use the previous warm CPU 16-trial batch pipeline: Gibbs updates, exchanges,
device traces, synchronization, transfer and joint/marginal/edge estimation,
including analytic symmetry. Time each arm separately, even when draws are
shared. Compile once per underlying method/size/horizon; record compilation
and initialization separately. Exclude those costs, exact enumeration, oracle
scoring, plotting and persistence from the warm batch timer. Warm sampler and
estimator before five repeats with rotated arm order, retaining every raw
timing. Repeats use identical keys. Report timing variability; small differences
are not persuasive evidence. No other test suite or study runs during timing.

## Integrity, scope and artifacts

Before production, verify analytic tempering augmentation against explicit cold
samples and their complements; verify that hot replicas cannot affect the
estimator. Check both tempering arms have identical redraw/swap/retained-sample
counts and exchange rates. Test a small run and replay using an earlier target,
including corruption detection. Reuse the exact/empirical Gibbs/exchange fixture.

Freeze requested inputs, source/protocol/lock hashes before sampling in a fresh
directory. The full grid has 120 cells and 24 decisions. Check every short
execution is a prefix of the corresponding maximum trace, and repeated timing
executions yield identical estimates. Persist packed states and exchange flags.
Replay authenticates all sources and traces and recomputes references, errors,
work counts, exchange rates, timing medians and qualification decisions, using
the inherited fixed atol=2e-12, rtol=0. Hashes and discrete values remain exact.
Replay cannot reproduce historical timings or regenerate trajectories.

Write completion last, archive evidence and source snapshots with SHA-256
manifests, and keep all prior archives immutable. Sampling evidence is
software_simulation; enumeration is exact_reference. No hardware speed/energy
claim follows. Expected execution is minutes on CPU, under the 30-minute
checkpoint threshold; interrupted generation restarts in a fresh directory.
No dependency upgrade, new language, GPU or remote service is needed.
