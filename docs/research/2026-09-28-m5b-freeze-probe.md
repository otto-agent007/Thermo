# M5b freeze: exploratory optimizer continuation

September 28, 2026 UTC / September 27 owner discussion. **Exploration, not a
new recorded study or a replacement for M5a evidence.**

The owner requested a small check of whether extra optimization materially
changes the chains associated with the three largest non-converged selected
M5a site fits. The [script](m5b_optimizer_probe.py) and
[bounded diagnostic record](2026-09-28-m5b-optimizer-probe.json) preserve what
was tried. The M5a archive and source files were not modified.

## Declared probe

- Authenticate the existing M5a archive and its frozen request/source hashes.
- Rank the 13 non-converged selected site fits by their stored objective,
  descending, breaking ties by archive key and site. Continue only the top
  three endpoints; do not add random starts or select replacements afterward.
- Use the original L-BFGS-B objective, bounds and tolerances, raising maxiter
  from 5,000 to 20,000 and maxfun from 50,000 to 200,000.
- Replace that one site in its associated variational chain while holding
  every other site at its original M5a parameters. Re-evaluate the chain
  with the source-bound M5a exact evaluator and recorded mixing constants.
- Impose a ten-minute invocation budget, save each completed refit and chain
  result atomically, and retain logs and an exit marker. Repeating the same
  command resumes only when its request/source digest matches.

## Observation

| Reading / seed / cap / site (zero-based) | Objective before | Objective after | Chain TV bias before | Chain TV bias after | Bias change |
|---|---|---|---|---|---|
| A / 2 / 1.5 / 6 | 2.0119863e-5 | 2.0119749e-5 | 0.040266111 | 0.040264219 | -1.8918e-6 |
| B / 1 / 0.75 / 9 | 3.6102745e-6 | 3.6010147e-6 | 0.003730059 | 0.003730210 | +1.5169e-7 |
| A / 4 / 2 / 10 | 1.2904444e-6 | 1.2895272e-6 | 0.092714589 | 0.092711576 | -3.0124e-6 |

All three continuation runs terminated successfully, after 429, 2,950 and
4,464 additional iterations respectively. The whole probe, including three
exact chain evaluations, took 53.84 seconds on this CPU runtime with one BLAS
thread; its exit code was zero. These are execution observations, not a
physical-device performance measurement. The JSON contains parameters,
objectives, termination messages, stationarity residuals and timings.
The probe used Python 3.11.16, NumPy 2.4.6 and SciPy 1.17.1.

The largest absolute chain-bias change was 3.02e-6. This bounded probe supports
leaving M5a's parameters fixed for M5b rather than introducing an optimizer
milestone. It does not prove optimality or show that every local minimum has
been excluded. A better local objective did not always lower global bias.

## Mathematical choices carried into the protocol

- K counts a hidden block followed by an output block, inside one conditional
  update. The outer 12-site sweep remains a different unit.
- The output starts at its current visible value. With a full hidden-first
  redraw, the previous hidden state is forgotten, yielding an exact 2x2
  output transition for fixed blanket inputs.
- That local transition and all its positive integer powers retain the M5a
  marginalized conditional and local detailed balance. A shifted outer
  stationary law can arise from incompatibility of independently compiled
  conditionals; it is not evidence that local detailed balance was broken.
- The reported lambda and an adaptive large-K check expose slow inner
  convergence. Increasing caps need not improve finite-work accuracy.

## Reproduce the exploration

From a clean checkout with the locked environment:

```bash
uv run python docs/research/m5b_optimizer_probe.py --output results/m5b-optimizer-probe/probe.json
```

This is an exploratory command, not an M5b production or completion command.
The JSON pins its own script hash and original archive hash. The parameters
from this probe are never inputs to the proposed M5b production grid.

## Verification of this protocol change

The three persisted endpoints replay their objective values and satisfy their
caps; all three chain stationarity residuals are below 1e-10. The saved script
hash matches, resuming a completed copy reuses all fits/chains, and a changed
request digest is rejected. The existing archive-identity and all-fit-objective
replay tests passed (2 tests); Ruff, formatting and changed-file link checks
passed. No production evaluator changed, and the full repository suite was
not rerun for this protocol/exploration change.
