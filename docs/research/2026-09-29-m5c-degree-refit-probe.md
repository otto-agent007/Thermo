# M5c bounded degree refit exploration

Status: **exploration**, `exact_reference`, CPU float64, zero samples. This
extends [the degree and repair probe](2026-09-28-m5c-input-model-probe.md)
after owner approval of the bounded training change. It is not a frozen M5c
protocol or a new study gate.

## Choices fixed before fitting

- Scope: only the **11 over-degree reading-B, variational, cap-1 kernels**
  from the authenticated M5b archive. The other 49 vectors stay bitwise
  identical. No additional arms, caps, hidden spins, or training objective.
- Mask: in each archived vector remove exactly `output_degree - 16` nonzero
  **J** edges with smallest absolute magnitude; ties use J index. Fix this mask
  before optimization. Every beta coordinate remains available, along with
  the original hidden count. This differs from the previous mixed J/beta
  pruning mask, which cuts beta at seed/site 0/1, 2/6, and 3/7.
- Fit: reuse M5a's uniform-input KL objective and analytic gradient without
  modifying any archive-bound module. L-BFGS-B uses `(0, 0)` for masked J
  coordinates and `(-1, 1)` for every other coordinate, with M5a's unchanged
  optimizer options. Start from M5a's eight starts, masked, plus the masked
  archived vector as start 8: **11 x 9 = 99 attempts**. Select minimum endpoint
  KL, ties by start index, and report convergence even when a termination is
  unsuccessful. Outer metrics do not select starts.
- Comparison: original, cap-1 two-node chain, original mixed-edge pruning,
  **unfitted J-only pruning**, and **J-only refit**. Only the same 11 sites
  change. Keep the original comparison to separate mask choice from fitting.
- Evaluation: the existing exact 4,096-state outer probe at **K=4, 32, and
  infinity**, uniform initial outer law, systematic site order, horizon 30.
  Report stationary target bias, target TV at step 30, distance to each
  method's own stationary law, inner mixing, redraws, clamp writes, reset
  writes, and readouts. Infinity is an oracle, without a finite work budget.
- Primary success bar is fixed at **K=4**, before seeing fitted results.
  Across all five seeds, **both median stationary target bias and median
  target TV30 must be below 8.3e-3** for the refit to be useful; both below
  **5e-4** meet the stronger, small-topology-cost benchmark. These are total
  target errors, not improvements relative to pruning. K=32 and infinity are
  diagnostics, not alternative primary endpoints. Report seed ranges too.

The two constants are rounded references to M5b's cap-1 reading-B variational
rounding bias: exact medians **0.008253446201539972** at 8 bits and
**0.0005000505740879214** at 12 bits. They describe M5b's mathematical codebook,
not verified chip precision. This probe evaluates unrounded parameters; it
does not assume that rounding and topology errors add or certify their joint
error.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 uv run python docs/research/m5c_refit_probe.py --workers 3 > /tmp/m5c-refit.jsonl
# budget sensitivity (see below); maxfun scales as 10 x maxiter, as in M5a
OPENBLAS_NUM_THREADS=1 uv run python docs/research/m5c_refit_probe.py --workers 3 --maxiter 20000 > /tmp/m5c-refit-20k.jsonl
```

The script authenticates the M5b archive and all five pinned implementation
hashes. It reuses the existing outer probe and replays all 15 original
baselines in this arm. JSON lines contain fixed scope, per-seed parameters
and convergence diagnostics, exact comparisons, and aggregate outcomes.
The selected parameters are exploratory output, not a new archived source.

## Results (2026-09-29)

The refit **passes both predeclared primary bars**. At K=4, five-seed median
stationary bias is **6.67150e-5** and median target TV30 is **6.67150e-5**.
Both are below 8.3e-3 and 5e-4; every individual seed is also below 5e-4.
The original fits remain more accurate, but the degree repair is below the
declared reference error budget. This does not certify combined rounding
and topology error.

J-only pruning without training has median target TV30 **0.0144047**, versus
**0.0173731** for historical mixed-edge pruning. The mask change alone does
not meet the useful bar. Refitting recovers the lost accuracy, beating both
pruning controls and cap-1 chaining in all five seeds at every evaluated K.

The following are separately computed five-seed medians, not paired ratios.

| K | Method | Stationary target bias | Target TV30 |
| ---: | --- | ---: | ---: |
| 4 | Original | 1.25208e-06 | 1.252e-06 |
| 4 | Chain, c=1 | 0.0503927 | 0.0503927 |
| 4 | Historical J/beta prune | 0.0173731 | 0.0173731 |
| 4 | J-only prune | 0.0144047 | 0.0144047 |
| 4 | J-only refit | 6.6715e-05 | 6.6715e-05 |
| 32 | Original | 1.26406e-06 | 1.26406e-06 |
| 32 | Chain, c=1 | 0.0200013 | 0.0200013 |
| 32 | Historical J/beta prune | 0.0181058 | 0.0181058 |
| 32 | J-only prune | 0.0145182 | 0.0145182 |
| 32 | J-only refit | 7.11867e-05 | 7.11867e-05 |
| ∞ | Original | 1.26406e-06 | 1.26406e-06 |
| ∞ | Chain, c=1 | 0.0199645 | 0.0199645 |
| ∞ | Historical J/beta prune | 0.0181058 | 0.0181058 |
| ∞ | J-only prune | 0.0145182 | 0.0145182 |
| ∞ | J-only refit | 7.11867e-05 | 7.11867e-05 |

The primary refit results by seed are:

| Seed | Stationary target bias, K=4 | Target TV30, K=4 |
| ---: | ---: | ---: |
| 0 | 0.000127380632 | 0.000127380659 |
| 1 | 6.67150454e-05 | 6.67150454e-05 |
| 2 | 2.00185168e-05 | 2.00184295e-05 |
| 3 | 7.19222092e-05 | 7.1922205e-05 |
| 4 | 6.50188877e-05 | 6.50196296e-05 |

The K=4 target-TV range is **2.00184e-5–1.27381e-4**. At K=32 it is
**2.25315e-5–1.28926e-4**, and at equilibrium **2.25375e-5–1.28885e-4**.
All three refit medians are below the stronger benchmark, although only
K=4 determines the declared outcome.

### Fit endpoints and convergence

All **99 starts** ran with the declared bounds and options. **59** terminated
with `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`; **40** reached
`STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`. Among the selected endpoints,
**4** have successful termination and **7** reached the 5,000-iteration limit.
These are bounded optimizer endpoints, with no global-optimum or full
convergence claim. Selection follows M5a: minimum endpoint objective regardless
of termination status. There was no extra optimization after inspecting
outer results.

Indices are zero-based. Start 0 is constructive, 1–7 are M5a random starts,
and 8 is the masked archived warm start; no warm start was selected.
All selected outputs have degree **16**, all masked J coefficients are exactly
zero, and every retained beta is nonzero. Hidden counts are unchanged.

| Seed/site | Masked input sites | Selected start | Iterations | Termination | Successful starts / 9 | Selected mean KL |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 0/1 | 6 | 6 | 5000 | Iteration limit | 7 | 1.50737e-07 |
| 0/10 | 6 | 6 | 5000 | Iteration limit | 5 | 3.17674e-09 |
| 1/8 | 11 | 6 | 2430 | Relative reduction | 7 | 6.86209e-14 |
| 1/9 | 2, 5 | 1 | 5000 | Iteration limit | 1 | 3.9124e-08 |
| 2/5 | 3, 6 | 7 | 5000 | Iteration limit | 5 | 2.42726e-09 |
| 2/6 | 5, 10 | 1 | 5000 | Iteration limit | 6 | 8.72412e-09 |
| 3/0 | 3 | 7 | 4122 | Relative reduction | 7 | 7.51837e-07 |
| 3/7 | 8, 10 | 2 | 4834 | Relative reduction | 7 | 7.25654e-14 |
| 4/5 | 8, 9 | 2 | 5000 | Iteration limit | 4 | 3.91532e-08 |
| 4/8 | 5 | 0 | 1061 | Relative reduction | 9 | 8.5836e-14 |
| 4/10 | 0, 4 | 0 | 5000 | Iteration limit | 1 | 1.99425e-08 |

The reproducer emits each start's initial/final objective, iterations, function
evaluations, success flag and termination text, plus each selected parameter
vector. It also emits all per-seed metrics and aggregate min/median/max.

### Mixing and work

Refitting changes local mixing. Across the 11 changed kernels, the worst
number of sweeps needed to put either initial output within TV 1e-3 of its
own equilibrium rises from **77 to 80**; at tolerance 1e-6 it rises from
**159 to 167**. Some sites slow more substantially (seed 0/site 10 goes from
15 to 60 sweeps at 1e-3), while others speed up. The unchanged 49 kernels
retain their archived dynamics. The finite-K outer results above directly
check whether these changes matter at the selected budget.

These are maxima over the changed kernels and all their inputs:

| Method | Maximum lambda | K for TV ≤ 1e-3 | K for TV ≤ 1e-6 | Worst finite/equilibrium TV, K=4 | K=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.918725 | 77 | 159 | 0.550468 | 0.043818 |
| J-only prune | 0.917403 | 76 | 156 | 0.545646 | 0.0436375 |
| J-only refit | 0.924006 | 80 | 167 | 0.543073 | 0.0462458 |

The refit outer law is already close to its own stationary law after 30
steps: median TV **1.41e-10**, maximum **6.57e-9**, at K=4. Low target error
at K=4 therefore does not depend on an unobserved long outer transient.

Original, historical prune, J-only prune and refit all retain **72K spin
redraws per outer sweep**: **288 at K=4** and **2,304 at K=32** (8,640 and
69,120 through t=30). Historical pruning retains its original accounting,
including the three dead hidden spins, so earlier results are not rewritten.
Chain uses 74K for seeds 0–3 and 75K for seed 4.

For the refit, per-sweep clamp writes by seed are **182, 197, 200, 193, 199**,
the same as the J-only pruning control, versus **184, 200, 204, 196, 204**
originally. Free-spin reset writes remain **72**, with **12 readout bits**,
per outer sweep. These are the existing clamped-copy/color counts, without
placement or routing overhead. Equilibrium has no finite redraw budget.

### Verification

- Completed all **11 fits, 99 starts and 75 exact outer comparisons** in
  **404.8 seconds (6.7 minutes)** with three CPU workers and one BLAS thread
  each. This is evaluator wall time, not device latency.
- Replayed all **15 original baselines** against M5b; maximum discrepancy
  **5.31e-16**. All **45 original/prune/chain metric and work records**
  reproduced the earlier full outer probe bitwise.
- Checked all start vectors, selected objectives, masks, cap bounds, degree
  counts and selection rules from the emitted output. The 49 unchanged
  vectors are checked bitwise in the probe.
- Independently enumerated local readout marginals for all 11 sites in
  original/J-only-prune/refit form; maximum discrepancy **1.45e-15**.
- Maximum outer stationary residual **3.08e-16**, row-sum error **5.56e-16**;
  chain local/control invariants also passed.
- All five M5b implementation hashes and the archive hash remain unchanged.
- The 74 selected M5a/M5b unit tests passed (two slow tests excluded);
  repository-wide Ruff formatting/lint and whitespace checks passed.

The successful bounded refit supports taking J-only degree constraints into
the placement exploration. No wider refit, cap sweep, or new training study
is needed to answer this probe's question. The iteration budget is revisited
below because most selected fits stopped at the limit.

## Budget sensitivity (2026-09-30)

This check was run **after** the primary result above; it does not replace
the declared K=4 outcome. The owner approved raising the refit budget for the
frozen M5c protocol on the basis of this check.

Seven of the 11 selected endpoints above stopped at M5a's 5,000-iteration
limit, against 12 of 1,080 selected fits in M5a itself. The masked problem
converges more slowly, so the budget was checked directly. Both budgets were
run on one machine with identical scope, starts, bounds and selection. Only
`maxiter` and `maxfun` (kept at 10 x `maxiter`, as in M5a) changed.

| Refit budget | Successful attempts / 99 | Selected successful / 11 | K=4 median bias | K=4 median TV30 | K=4 seed range (TV30) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 5,000 | 46 | 3 | 6.5366e-5 | 6.5366e-5 | 4.09e-5–1.45e-4 |
| 20,000 | 96 | 10 | 3.47332e-5 | 3.47328e-5 | 3.46e-6–7.19e-5 |

The 5,000-iteration row is this machine's rerun of the primary run. It passes
the same bars, but 8 of its 11 selected starts and its per-seed errors differ
from the 2026-09-29 run above. At 20,000, K=32 and equilibrium medians are
**4.52e-5**, and every seed stays below 5e-4 at every K.

| Seed/site | Selected start | Iterations | Termination | Successful starts / 9 | Selected mean KL |
| --- | ---: | ---: | --- | ---: | ---: |
| 0/1 | 3 | 6907 | Relative reduction | 9 | 3.41267e-13 |
| 0/10 | 8 | 9216 | Relative reduction | 9 | 1.8959e-08 |
| 1/8 | 6 | 3133 | Relative reduction | 9 | 4.23156e-14 |
| 1/9 | 8 | 20000 | Iteration limit | 6 | 1.2826e-08 |
| 2/5 | 5 | 17218 | Relative reduction | 9 | 9.68085e-12 |
| 2/6 | 7 | 7484 | Relative reduction | 9 | 1.07282e-09 |
| 3/0 | 7 | 7313 | Relative reduction | 9 | 7.51833e-07 |
| 3/7 | 5 | 9181 | Relative reduction | 9 | 5.58685e-14 |
| 4/5 | 3 | 7938 | Relative reduction | 9 | 6.09721e-14 |
| 4/8 | 8 | 377 | Relative reduction | 9 | 3.29947e-13 |
| 4/10 | 0 | 7032 | Relative reduction | 9 | 1.97868e-08 |

- Most selected fits need **6,900–9,200 iterations**. At 5,000 they are cut
  off mid-descent. The masked archived warm start is now selected at three
  sites.
- The error floor is set by the mask, not the budget. Seed 3 stays at
  **7.19e-5** because site 3/0 converges to the same objective, **7.52e-7**,
  at both budgets. Lowering it would need a different mask or more hidden
  spins, which is outside this probe's scope.
- Site 1/9 still reaches 20,000 iterations with objective 1.28e-8. It does
  not set the floor and no further budget is spent on it.
- Mixing is essentially unchanged. The worst K for TV ≤ 1e-3 is **79** and
  for TV ≤ 1e-6 is **165**, against 77 and 159 originally. Redraw, clamp,
  reset and readout counts do not depend on the budget.
- Summed per-seed fit time rose from about 635 to 1,129 seconds, with both
  runs sharing the machine. This is evaluator time, not device latency.

Converged endpoints are expected to depend less on the machine than truncated
ones, but this was not checked on a second machine.

**Decision:** the frozen M5c protocol uses **`maxiter` 20,000 and `maxfun`
200,000** for the degree-repair refit and keeps M5a's selection rule, which
reports but does not require convergence. M5a's archived fits and their
5,000-iteration budget are unchanged.

## Boundary

This refit only addresses output degree. A named synthetic lattice patch,
actual placement, and extra copy/routing costs remain necessary before
freezing M5c. If this approach is adopted, the frozen M5c protocol must
regenerate the fits as recorded evidence at the budget above; it must not
promote these exploratory endpoints. No hardware latency, energy, or
physical-placement claim follows.
