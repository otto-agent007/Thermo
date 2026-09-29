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
```

The script authenticates the M5b archive and all five pinned implementation
hashes. It reuses the existing outer probe and replays all 15 original
baselines in this arm. JSON lines contain fixed scope, per-seed parameters
and convergence diagnostics, exact comparisons, and aggregate outcomes.
The selected parameters are exploratory output, not a new archived source.

## Boundary

This refit only addresses output degree. A named synthetic lattice patch,
actual placement, and extra copy/routing costs remain necessary before
freezing M5c. If this approach is adopted, the frozen M5c protocol must
regenerate the fits as recorded evidence; it must not promote these
exploratory endpoints. No hardware latency, energy, or physical-placement
claim follows.
