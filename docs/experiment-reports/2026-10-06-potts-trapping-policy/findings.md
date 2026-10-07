# A reference-free trapping policy does not beat always-tempering

**Negative result.** The predeclared policy (a 256-sweep pilot of five cold
chains, then a switch to tempering when the spread ratio exceeds 1) fails its
success test on 36 fresh Potts targets. Its total budget regret against the
per-target hindsight oracle is **8** grid steps. That is far below
always-independent (**24**), but above always-tempering (**3**). The policy
qualifies on 36/36 targets, as does always-tempering; always-independent
qualifies on 34/36. Within the tested budgets, **always use tempering plus
symmetry** is the better rule for these targets.

The detector itself works as a signal. Against the hindsight label
"always-independent fails at T = 4096", the spread ratio has a per-trial AUC
of 0.81, close to the probe's 0.85. It switches 80% of failing trials and 41%
of the others. It loses because each detection error costs more than the
fixed rule's errors do, and because the budget grid left little to gain (see
below).

The [frozen protocol](../../experiments/potts-trapping-policy.md) fixed the
threshold from theory before any cell ran and tested it once on fresh graph
seeds 600 to 611 at beta 8, 12 and 16, with 16 trials per target. All 576
policy trials completed and replayed. Sampling is `software_simulation` (THRML
0.1.4 categorical block Gibbs, CPU, float32); references are
`exact_reference` (float64 enumeration of 3^12 states). Budgets are redraw
counts; there is no wall-clock or hardware claim.

## Regret by temperature

Regret is the sum over targets of log4(arm budget / oracle budget), where the
oracle is the better of the two fixed arms on that target in hindsight. A
target an arm never qualifies on counts as 65536 (one step past the grid).

| beta | Policy | Always-independent | Always-tempering | Qualifying (policy / ind / temp) | Mean switch fraction |
| --- | --- | --- | --- | --- | --- |
| 8 | 1 | 0 | 3 | 12 / 12 / 12 | 0.21 |
| 12 | 4 | 8 | 0 | 12 / 12 / 12 | 0.49 |
| 16 | 3 | 16 | 0 | 12 / 10 / 12 | 0.79 |
| All | **8** | 24 | **3** | 36 / 34 / 36 | |

`summary.md` lists every target's selected budgets, regrets, median spread
ratio and switch fraction.

## Why the policy loses

- **The policy makes its decision per trial, but accuracy is judged on the
  mean over 16 trials.** On trapping targets, the trials it fails to switch
  stay stuck. That happened to 26 of the 133 trials that always-independent
  fails at T = 4096. On n12-g611-b16 the one unswitched trial has joint TV
  0.115 at T = 1024; on n12-g600-b16 the six average 0.087. A few such trials
  push a target's mean just over 0.05: the policy's mean at T = 1024 is 0.051
  to 0.069 on the targets it loses, against 0.024 to 0.050 for always-tempering.
- **Switched trials pay for the pilot.** At T = 1024 a switched trial gets
  768 tempering sweeps rather than 1024. Across the 287 switched trials its
  mean joint TV at T = 1024 is 0.047, against 0.042 for fresh tempering on the
  same trials.
- **The budget grid capped the possible gain. This is a limit of the design
  that the protocol did not flag.** The pilot forced the smallest budget to
  T = 1024, and always-tempering already qualifies at that floor on 30 of 36
  targets. The only targets where always-independent beats always-tempering
  are three at beta 8, each by one grid step. So the policy could save at most
  3 steps against always-tempering, while every detection error at beta 12
  and 16 costs a step. Stage B's large beta 8 advantage for ordinary chains
  (256 against 1024) lies below this grid. Recovering it would need budgets
  below T = 1024, which a 256-sweep pilot cannot serve.

The false-switch rate (41% of non-failing trials) is much higher than the
roughly 5% that independent chains would give. Many of those trials
eventually pass at T = 4096 but are still mixing slowly at 256 sweeps, so the
switch is not wrong in itself. At beta 8 it cost one step, on n12-g604-b8.

## What this does and does not settle

On zero-field antiferromagnetic three-state Potts targets of this size, at
budgets of 1024 sweeps and above, defaulting to tempering plus symmetry costs
at most one fourfold step on targets that mix, and avoids the large losses
ordinary chains take on targets that trap. The tested switching policy does
not improve on that default. This does not show that no adaptive policy could
help. A policy evaluated at smaller budgets, one that pools pilot samples, or
one that decides per target rather than per trial might. But this is the
second adaptive policy in a row that fails to beat a simple baseline (the
[changing-evidence](../2026-10-02-changing-evidence/findings.md) restart
policy was the first). Whether to pursue a third should be the owner's
decision.

Not settled: other graph sizes or families, field-bearing targets, other
thresholds or pilot lengths (none were tuned), and any hardware speed or
energy result.

## Disclosures

- The exploratory probe used graph seeds 910 to 915. The pipeline smoke,
  after freezing and before the full run, ran graph 600 at beta 8 and 16 with
  budgets up to 4096. Graph 600 is one of the study's 12 graphs. Nothing
  changed after the smoke.
- The runner, protocol and tests were committed before the run (`d86b518`),
  and the run's provenance records a clean tree.
- One unit test's expectation was corrected before the run. It had assumed
  the spread ratio of perfectly independent chains never exceeds 1; in fact it
  centres at 0.72 and exceeds 1 about 5% of the time. The policy and threshold
  were unchanged.

## Files

- `study.json.gz` (0.48 MiB, SHA-256
  `acdbc209bf9bcb6ca0e845847987fc9f2a36c459cacf5806ed93c02909d8bacb`): request, exact references, per-trial pilot
  signals, switch decisions and joint/edge-agreement counts for every arm and
  budget, integrity flags, evaluation and the result digest. No trajectories.
- `summary.md`: the regret, per-target and detection tables rendered from the
  archive.
- `completion.json`: 36 targets, 576 policy trials, `policy_succeeds=false`,
  regret per arm, qualifying counts, prefix and continuation checks passed,
  replayed.
- `provenance.json`, `run.log`: runtime (JAX 0.10.2 CPU, x64 off, Python
  3.11.15, commit d86b518, clean tree) and per-target progress. The run took
  597 s including replay.
