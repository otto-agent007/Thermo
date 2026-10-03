# Exploratory CPU pilots: optimization, mixing and inference

These are preserved exploratory results, not formal milestone releases.
The original scripts, requests, metrics, traces and completion records are
unchanged inside three gzip archives. `manifest.json` records archive and
member SHA-256 hashes. Each archive was extracted into a fresh directory and
all four original replay commands passed before this report was finalized.

## Findings that motivated the next experiment

1. **No optimization advantage was established.** On six small weighted
   degree-three graphs (12/16 vertices, seeds 0/1/2), THRML at beta 1 and
   simulated annealing both found the exact best cut in all 96 runs by 256
   sweeps. A stronger random-restart greedy baseline also reached 96/96,
   and reached 93/96 by 64 sweeps versus THRML's 87/96. These compare
   algorithmic updates; they do not establish equal runtime or energy.
2. **Good cuts did not imply accurate probabilities.** Reanalysis of the
   saved beta-4 THRML chains found 994/1,344 spin series constant over the
   retained 192 sweeps. Median across graphs of median-chain marginal MAE
   was 0.4751, against 0.0788 at beta 1 and 0.0299 at beta 0.25. Global
   spin-flip symmetry makes exact marginals 0.5; pooling opposite trapped
   chains can conceal this problem. Short-chain ESS is diagnostic only.
3. **Small posterior inference was feasible.** Six 4x4 Ising denoising
   cases gave mean pooled probability MAE 0.00334 (THRML), 0.00400
   (independent Gibbs), and 0.00279 (Metropolis) at 1,024 sweeps with
   16 chains. Exact posterior pixel decisions reduced errors from 26 to
   20 out of 96 pixels. Sampled decisions sometimes scored one pixel better
   on the realized truth by chance; that does not beat the Bayes rule.

The graph family, small sizes and three clean-image draws limit these
conclusions. All samplers are `software_simulation`; exhaustive references
are `exact_reference`. There are no physical-hardware measurements.

The resulting next question is whether longer chains, independent chains,
or replica exchange improve cold-target probability accuracy at a fixed
total spin-update budget. The [frozen follow-up protocol](../../experiments/fixed-budget-sampling.md)
uses fresh graph seeds and checks a joint distribution plus correlations.

## Archive contents and replay

| Archive | Contents |
| --- | --- |
| `maxcut-cpu-pilot-2026-10-01.tar.gz` | Original optimization pilot; exact cut optima, THRML, annealing and single-start greedy |
| `maxcut-restart-greedy-2026-10-01.tar.gz` | Random-restart comparison and authenticated copies of its source evidence |
| `inference-pilot-2026-10-01.tar.gz` | Saved-chain analysis, denoising experiment, figures and authenticated source evidence |

From the repository root, extract each archive into a fresh directory under
`results/`, for example `results/pilot-replay/`. Preserve the internal paths.
Use the frozen project environment (`uv sync --frozen`), then run:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
uv run --frozen python results/pilot-replay/maxcut-cpu-pilot/pilot.py --replay
uv run --frozen python results/pilot-replay/maxcut-restart-greedy/compare.py --replay
uv run --frozen python results/pilot-replay/inference-pilot/analyze.py --replay
uv run --frozen python results/pilot-replay/inference-pilot/denoise.py --replay
```

Replay recomputes saved scientific metrics and exact references, authenticates
the specified sources and traces, and rewrites completion records in the
extracted copy. It does not reproduce historical timings or THRML trajectories.
The original request/README files define each replay's complete scope and
generation command. Some archived READMEs describe the earlier untracked
workspace state; this report records their subsequent preservation.
