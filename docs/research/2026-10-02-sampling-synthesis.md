# Sampling research synthesis: what changed and what to test next

This note connects the October 1–2 experiments, including negative results,
and records the resulting research decisions. The individual frozen protocols
and authenticated reports remain authoritative. These are small-model CPU
software experiments with exact references, not thermodynamic-hardware tests.

## Main conclusion

There is no single winning sampler across these workloads. Useful gains came
from matching the sampler and estimator to the target distribution, keeping
strong conventional baselines, and evaluating the actual inference or decision
task. More accurate probabilities per spin update did not always mean faster
execution, and lower probability error did not always mean better decisions.

| Tested setting | Best-supported working choice | Boundary |
| --- | --- | --- |
| Small weighted Max-Cut | Keep random-restart greedy as a strong baseline | No stochastic optimization advantage established |
| Zero-field frustrated ring-plus-matching graphs | Combine valid global-flip symmetry with tempering | Six fresh 12/16-spin targets; projected joint and edge accuracy only |
| Biased frustrated graphs | Keep tempering as a candidate | It qualified on all six targets; speed ratios exclude censored baselines |
| Small, correctly specified denoising posteriors | Independent Gibbs reference | Tempering did not repay its replica cost on the six cases |
| Related posterior queries | Retain sampler states, while checking history dependence | The tested failure-triggered reset policy did not reliably help |
| Weak-coupling marginal inference | Add conditional-probability averaging | Include its extra estimator work in timing |
| Strong-checkerboard inference | Preserve the difficult cases and improve exploration | Conditional averaging leaves most error unresolved |
| Twelve-spin exact inference | Include cached enumeration as a practical competitor | Exact-state enumeration scales exponentially |

These are working choices for the declared families, not a universal ranking.
The query streams use current evidence only; retained state is a computational
shortcut, not a temporal Bayesian prior.

## What each experiment established

1. **[Exploratory optimization and inference pilots](../experiment-reports/2026-10-01-exploratory-pilots/summary.md).**
   THRML, annealing and restart-greedy all reached the exact Max-Cut optimum
   in 96/96 largest-budget runs. Restart-greedy was stronger at a shorter
   budget. Cold chains could find good cuts while producing poor probabilities:
   994/1,344 saved spin series were constant. Small denoising posteriors were
   feasible, but these pilots established no runtime advantage.
2. **[Fixed total redraw budget](../experiment-reports/2026-10-01-fixed-budget-sampling/findings.md).**
   In 108 cells, tempering lowered mean four-spin joint-probability error
   3.11x on zero-field targets and 13.05x on weak-field targets against the
   better ordinary baseline at the largest budget. It paid for every replica
   and retained fewer cold states. Its timings overlapped other work and do
   not support speed claims.
3. **[Time to accuracy and transfer](../experiment-reports/2026-10-02-sampling-time-to-accuracy/findings.md).**
   All 330 cells and 66 decisions replayed. Tempering qualified on all six
   biased graphs; it was 3.12–44.61x faster on the four with a qualifying
   ordinary baseline. Independent Gibbs beat tempering on all six denoising
   cases, by 3.87–17.86x. Valid global-flip estimators removed much of the
   ordinary baselines' zero-field disadvantage. A post-hoc symmetry-plus-
   tempering analysis used saved states and motivated a prospective test.
4. **[Fresh-seed symmetry plus tempering](../experiment-reports/2026-10-02-symmetry-tempering/findings.md).**
   All 120 cells and 24 decisions replayed. The combination qualified on
   6/6 fresh targets, symmetry-aware ordinary baselines on 5/6, and plain
   tempering on 2/6. It was 4.05–12.33x faster than the best qualifying
   ordinary baseline on four targets; one comparison was close, and one
   baseline remained censored. Symmetry and tempering address different errors.
5. **[Changing evidence and causal restart policy](../experiment-reports/2026-10-02-changing-evidence/findings.md).**
   All 416 cells and 83,200 query estimates replayed, using eight independent
   development and eight held-out seeds. Retention helped initialization but
   increased forward/reverse disagreement at identical inputs. State-aware
   failure ranking improved (held-out AUC 0.810/0.920), yet its matched-work
   restart policy did not establish a dependable gain over retention. Exact
   enumeration was faster than every sampled held-out pipeline on these
   12-spin cases. A separate analytic iid reference showed that much weak-
   regime error was consistent with limited sample counts.
6. **[Conditional estimates on identical states](../experiment-reports/2026-10-02-conditional-estimation/findings.md).**
   Sixteen fresh seeds produced 96 shared trajectory cells, 192 estimator
   cells and 76,800 query estimates, all replayed. At 16 sweeps/query,
   conditional averaging lowered marginal MAE 26.1% for Gibbs and 29.9% for
   tempering, with median paired runtime overhead 26.3%/7.4%. Weak-regime
   errors fell 56–64%; strong-checkerboard errors only 4–5%. In the weak
   regimes, 16 conditional sweeps beat 64 empirical sweeps at roughly
   2.7–3.1x lower runtime. Gibbs alarm-decision regret increased despite
   lower probability error, so utility needs its own evaluation.

The time-to-accuracy thresholds concern mean trial four-spin joint TV and
edge-correlation MAE, sustained across all larger tested budgets. They do not
certify the full distribution or every trial. Speed comparisons are warm CPU
batches; each report states included and excluded costs. They are not energy
claims or independent-sample rates. Seeds, not correlated queries or timing
repeats, supply replication for uncertainty estimates.

## Public work and earlier Thermo results

The [knowledge base](../knowledge/README.md) separates paper claims, released
code, pinned dependencies and results Thermo reproduced. These experiments
used THRML 0.1.4, Torx 0.0.1 and the locked JAX/NumPy/SciPy environment. The
[Torx source card](../knowledge/sources/extropic-torx.md) records a newer release;
this PR does not silently upgrade scientific dependencies or claim to test it.
The new sampler comparisons are JAX algorithm experiments, not THRML throughput
measurements or official Thermalizers compatibility tests.

The public [Torx](../knowledge/sources/extropic-torx-paper.md) and
[Thermalizers](../knowledge/sources/extropic-thermalizers.md) material motivates
stochastic programming and lowering toward local thermodynamic kernels. It
does not establish that these particular methods are efficient on physical
hardware. [Z1T](../knowledge/sources/extropic-z1t.md), diffusion-hardware and
other hardware claims retain their source-card evidence status. CPU update
counts and timings cannot validate energy projections.

Earlier work already on `main` supplies a warning: the
[M4 cap analysis](2026-09-23-cap-leakage-analysis.md) identified a structural
limit after many conservation variants, and that line is closed. M5a–M5c
address exact small-kernel behavior, finite thermalization, precision and
synthetic placement; [complete execution costs remain open](../roadmap.md).
The native experiments broaden the charter without reopening M4.

Conditional expectations and symmetry averaging are established variance-
reduction ideas. The new learning is where they help these workloads, what
their implementations cost, and which errors remain. Nothing here shows that
another programming language or a GPU would resolve the observed mixing or
decision-quality limitations.

## Review corrections and reproducibility

Independent PR review authenticated **eight compressed archives and 114
members**, and recomputed the central accuracy ratios, qualification counts,
timing ratios and conditional-estimation outcomes. Two corrections appear
in the changing-evidence report:

- The frozen `initialized_spins` field counts applied reset-state writes.
  Batched policy initialization generates extra rows that are discarded.
  The [authenticated corrective analysis](../experiment-reports/2026-10-02-changing-evidence/initialization-accounting.json)
  records 136,320/190,080 generated spins versus 122,220/174,120 applied spins
  for the Gibbs/tempering policies. Query timings already include generation;
  accuracy and redraw/swap comparisons are unchanged.
- Coupling installation was outside that study's query timer and was not
  separately measured. The conditional-estimation follow-up measures it.
  Warm timing comparisons do not establish complete cold-start costs.

Frozen evaluators, protocols, raw results and archived lock snapshots remain
unchanged. Corrections are separate analyses and prose. The eight archives
total approximately 72.1 MiB compressed; the largest is 25.6 MiB. They retain
states, requests and source snapshots, not only charts. Replay authenticates
artifacts and recomputes scientific summaries; it does not reproduce historical
wall times or regenerate every trajectory.

## Next research decisions

The next proposed mechanism test is grouped spin updates for the remaining
strong-coupling errors. Hold conditional estimation fixed, use fresh seeds,
compare ordinary Gibbs and tempering, and retain exact small-case references.
Count group-update work and measure query costs; sweep counts alone are not
comparable across kernels. Grouped updates have not been run in this work.

The knowledge-base [E0 finite-sweep THRML contract](../knowledge/experiment-backlog.md#draft-protocol-e0-thrml-finite-sweep-contract-against-thermos-exact-kernel)
also remains open. The new JAX sampler's exact fixtures do not verify THRML's
initialization, block ordering, clamping or finite-K conventions. The native
experiments do not supersede that dependency contract.

For continuous learning, choose the cheapest experiment that distinguishes
plausible explanations; check fixed assumptions first; freeze inputs and
metrics before examining test outcomes; report negative results and utility;
and use each result to choose the next unresolved question. Test transfer and
scale periodically. Long runs should calibrate, save progress and resume,
rather than inherit an arbitrary five-minute cutoff. Keep the process
proportional to the scientific question.

## Verification for the PR

The branch is based on current `main` at `fbe8c3e`. That upstream revision adds
the knowledge base and updates pytest to 9.1.1, Ruff to 0.16.9 and setuptools
to 84.0.0; numerical sampling dependencies are unchanged. Archived scientific
lock snapshots still describe the original runs.

The earlier unsharded full-suite process ended without a completion summary
and is not counted as a pass. Fresh verification collected **2,472 tests** and
confirmed that the repository's eleven existing CI partitions cover every test
exactly once. All eleven partitions completed: **2,470 passed, 2 skipped,
0 failed**. The two existing dashboard tests skipped because their Playwright
dependency is unavailable. The suite ran with three workers, CPU-only,
single-thread BLAS and no outer time cutoff; all 28 slow-contract tests passed.

The frozen environment sync and offline lock check, Ruff lint and format
checks, smoke command, all ten base experiment CLI gates, and package build
passed. Both wheel and source distribution include the five new study modules
and all ten experiment configurations. The knowledge-base validator passed
14 source cards and 40 claims; all 225 local file links checked in changed
Markdown resolve.

The five new archived-study replay tests passed within the full suite. All four
original pilot replay commands also passed from fresh archive extractions.
Independent review resolved both accounting disclosures and found no remaining
blockers. No archived scientific evidence or hash-bound evaluator was changed
to make these checks pass.

The first GitHub CI run subsequently exposed two host assumptions: missing
cgroup metadata files and CPU-dependent floating-point last bits in the first
archive's bitwise replay. The [portability follow-up](sampling-portability.md)
adds an authenticated execution adapter and a separately labeled numerical
replay. The Haswell probe differs by at most 1.11e-16, with a 2e-12 audit
tolerance. It retains exact integrity/discrete checks and preserves every
original evaluator, archive and historical completion record. Regression
tests cover unavailable metadata and corrupted evidence.
All 47 targeted tests and all 1,600 tests in the previously failing CI group
pass locally with the fix. Build, lint, formatting and source-hash checks pass.
