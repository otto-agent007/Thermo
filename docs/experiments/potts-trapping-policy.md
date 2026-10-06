# Potts stage C: a reference-free trapping detector chooses the sampler

**Status: draft, 2026-10-06, awaiting the owner's go-ahead to freeze.** No
study cell has run. The exploratory probe
(`docs/research/potts_trapping_probe.py`, graph seeds 910 to 915, which this
study does not reuse) compared five pilot signals on 18 targets; its output is
exploration, not evidence. The decision rule below is fixed from theory, not
fitted to that probe, and is tested once on fresh seeds with no development
split. Change any value after freezing only through a new protocol version.

## Question

Potts stage B
([findings](../experiment-reports/2026-10-06-potts-symmetry-tempering/findings.md))
found that tempering plus the label-symmetry estimator wins when ordinary
chains trap and loses, by about 4x in budget, when they mix, because it keeps
only its cold replica. It could tell the two cases apart only with exact
references. This study asks whether a fixed policy that decides from a short
pilot, with no reference, beats both fixed choices on fresh targets once the
pilot's cost is counted.

## The policy (fixed before any cell runs)

1. Run the trial's five independent cold chains for a pilot of P = 256
   sweeps.
2. On the pilot after its first quarter (sweeps 64 to 255), compute the
   **spread ratio**: the mean pairwise TV between the five chains'
   symmetrized joint histograms of sites 0 to 3, divided by the mean TV
   between the first and second halves of each chain's symmetrized
   histogram.
3. If the ratio exceeds **1**, switch: continue with five-replica tempering,
   started from the five pilot chains' final states, for T - P sweeps.
   Estimate from the cold replica after the first quarter of that
   continuation; pilot samples are discarded.
4. Otherwise continue the same five chains to T sweeps in total and estimate
   from sweeps T/4 to T - 1, as the independent sampler does. The pilot is the
   start of that run.

Every trial pays 5 T n label redraws at budget T, pilot included; a switched
trial also pays 2(T - P) exchange attempts. The policy decides once per
trial, independent of T.

**Why the threshold is 1.** If the chains mix, each chain is a draw from the
same law. Each half-chain holds half as many samples as a whole chain, so its
sampling noise is about sqrt(2) times larger, and the expected ratio is about
1/sqrt(2), roughly 0.71. A ratio above 1 means the chains disagree more than
halves of one chain do, which noise alone does not explain. In the probe,
mixing targets sat at 0.56 to 0.85 and targets where ordinary chains fell
behind at 1.33 to 2.85. The threshold was not fitted to those values.

## Fixed choices

| Choice | Value | Could it bound the result? |
| --- | --- | --- |
| Graphs | n = 12 weighted cubic graphs from the October generator, fresh seeds 600 to 611; every connected cubic graph other than K4 is 3-colourable (Brooks), so none is excluded | Twelve graphs from one family. |
| Temperatures | beta in {8, 12, 16}, so 36 targets | The probe and stage B suggest most beta 8 targets mix and some beta 12 and 16 targets trap. If few trap, the policy has little to gain; that is a possible result. |
| Model, samplers, estimator | Exactly stage B's: zero-field antiferromagnetic q = 3 Potts, THRML categorical block Gibbs, five-replica ladder beta x (1/16, ..., 1), label-permutation symmetrization, imported from `thermo_lab.potts_symmetry_tempering` | |
| Budgets | T in {1024, 4096, 16384} | The pilot is a quarter of T = 1024, so a switched trial gets only 768 tempering sweeps there; the policy pays for detection at the smallest budget. A fourfold grid only locates crossings coarsely. |
| Trials | 16 per target; root seed 20261007, folded with target and trial, split into init and sampling keys; all arms share each trial's initial labels | |
| Accuracy rule | Stage B's: mean trial joint TV and mean trial edge-agreement MAE both at most 0.05 at the selected budget and every larger one | Censored at T = 16384. |

## Arms and comparisons

Four arms per target, scored with the symmetry estimator: **policy**,
**always-independent** (five cold chains, T sweeps), **always-tempering**
(fresh five-replica tempering, T sweeps, no pilot) and the **per-target
oracle** (the better of the two fixed arms on that target, chosen with
hindsight, which no deployable rule can match).

- **Primary.** Total budget regret over the 36 targets against the oracle:
  the sum of log4(arm budget / oracle budget). A target an arm never
  qualifies on counts as one grid step beyond the largest budget (65536);
  that convention and the number of such targets are reported. The policy
  **succeeds** if its total regret is lower than both fixed arms' AND it
  qualifies on at least as many targets as the better fixed arm. Otherwise it
  fails, and the report says so.
- **Per regime.** The same quantities by beta, and every target's four
  selected budgets.
- **Detection.** Per target, the fraction of trials that switched. Against
  the hindsight label "always-independent fails the accuracy rule at
  T = 4096", the per-trial AUC of the spread ratio, as a check on the probe.
  The raw between-chain spread and the all-pairs agreement R-hat are reported
  beside it with no threshold.
- **Wall time** is not measured. Stage B found per-T warm cost of the
  independent and tempering samplers within 5% on this CPU, and every arm
  here is compared at matched redraws.

## Integrity

- An always-independent trial and an unswitched policy trial are the same
  run, so the runner computes them once and records that identity.
- A switched trial's continuation is re-executed once per target for a fixed
  trial and must reproduce its histograms exactly.
- Budgets are prefixes of one run per arm per trial, as in stage B.
- The archive keeps per-trial counts, signals and decisions only, no
  trajectories. Replay recomputes exact references, errors, qualification,
  regret and detection metrics from the counts, compares them with the
  archive numerically and checks the digest of the archived values.

## Outputs and completion

One runner (`thermo_lab.potts_trapping_policy`), one replay test and one
report. Completion requires `status=potts_trapping_policy_complete`,
36 targets, 576 policy trials, `prefix_checks_passed=true`,
`continuation_check_passed=true` and `replayed=true`. Whether the policy
succeeds or fails is a scientific result, not an integrity failure.

**Expected cost.** Three runs of up to 16384 sweeps per target at about
3 s each warm, so under 10 minutes on CPU. No autosave layer; restart in a
fresh directory. CI replays the archive and does not resample.

## Not claimed

Any wall-clock, hardware speed or energy result; other graph families or
sizes; field-bearing targets; that the threshold is optimal (it is fixed from
theory and not tuned); any other pilot length.
