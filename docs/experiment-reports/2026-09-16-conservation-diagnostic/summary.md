# Conservation fails along almost every K4 path

September 16, 2026. The frozen initial model keeps exactly one particle throughout
all 500 operations with probability **0.00040392135 at K4 (0.0404%)** and
**0.0348647811 at equilibrium (3.49%)**. These are exact-reference probabilities
from a substochastic recurrence on the 25 one-particle states, with paths killed
on their first exit. No full-state approximation or Monte Carlo estimate is
used for these probabilities; numerical arithmetic is float64.

| Measurement | K4 | Equilibrium |
| --- | ---: | ---: |
| First-operation conservation failure | 10.36% | 3.87% |
| Operation by which half of paths have first failed | 52 | 106 |
| Operation by which 99% have first failed | 302 | Not reached within 500 |
| Exact probability of uninterrupted survival through operation 500 | 0.0404% | 3.49% |
| Sampled final leakage, across three streams | 77.04–77.14% | 58.87–59.61% |
| Sampled fraction ending valid after an earlier failure | 22.82–22.91% | 37.00–37.72% |

The approximately 77% terminal leakage in previous studies understates the
frequency of pathwise failure for this initialization. A final count of one
can result from later creation or destruction of particles. A recovered path
is not an uninterrupted faithful execution of the one-particle program.
The sampled percentages are software_simulation evidence from fresh streams,
not physical-hardware measurements or new evaluations of trained M4B models.

## What the diagnostic identifies

There is both an immediate local defect and accumulation through composition.
The first operation already breaks conservation with probability 10.36% at K4;
exact survival falls to 56.43% after 50 operations, 27.05% after 100, and 2.48%
after 250. This is not solely a late-program phenomenon.

At K4, unconditional first-exit probability divides into 0.690048362 for
particle destruction and 0.309547716 for creation. These sum to the overall
first-exit probability, 0.999596079. About 0.252540757 of probability mass first
fails through particle creation on an empty active edge (parent 00).
Equilibrium's first-exit masses are 0.431118866 for destruction and 0.534016353
for creation. Different surviving context distributions contribute to these
splits; the difference is not a controlled causal decomposition.

Every frozen group's empty-edge creation probability is positive. Across the
37 groups it ranges from about 0.0471% to 2.558% per use at either horizon.
These small local defects can recur hundreds of times. The equilibrium control has higher
pathwise survival than K4 but does not remove the conservation problem.

All three archived source records have the **same initial parameter matrix**
(digest `sha256:04d28612ef90031e303e48ec9c88d1e849b7fffafada006a86c89e42d9f37b71`).
Their exact curves therefore coincide. The three seeds replicate sampling from
this frozen model; they are not three independent parameter fits. We report no
confidence intervals or generalization across initializations.

## Research decision

Keep conservation and occupancy fidelity as separate measurements. Increasing
occupancy-loss training alone is not justified by M4B or this diagnostic.
The next bounded question is whether the existing local kernel, within the
same caps and K4 budget, can substantially reduce conservation error while
retaining the desired asymmetric transition probabilities. Compare that
trade-off against an exact particle-preserving logical reference before
committing to more full-program training or a different architecture.

This study does not optimize the kernel, prove the best achievable error under
the caps, establish that a conservation penalty will work, or quantify hardware
cost. M5's topology-aware target reproduction remains queued. A new intervention
requires its own predeclared objective, budget, evaluation and stopping rule.

## Evidence and reproducibility

The [protocol](../../experiments/conservation-leakage-diagnostic.md) was committed
before the study at `6d51984`. The full study was generated from clean implementation
commit `760ec2b373adb19868150c8c35a3600c27913937`; [provenance](provenance.json)
records the runtime. A subsequent documentation clarification puts derived
table digests in the result rather than the requested-input hash, matching the
implementation and contributor rules; no scientific input or output changed.

The [generated report](report.md), [completion](completion.json),
[requested inputs](protocol.json), and complete bounded artifacts for
[seed 0](seed-0000000000.json), [seed 1](seed-0000000001.json), and
[seed 2](seed-0000000002.json) are committed together. Each artifact embeds its
authenticated source and all 500 rows at both horizons. The report is regenerated
only after full numerical replay. No raw trajectories are retained.

```bash
uv run python -m thermo_lab.conservation_audit \
  --output-dir results/conservation-diagnostic
```

Use a fresh destination. The study evaluates 196,608 full trajectories across
three seeds and two horizons (98,304,000 logical endpoint draws), plus exact
propagation and replay. These are NumPy endpoint draws, not actual Gibbs updates
or measured device work. Exact replay requires the same computed numerical
tables; incompatible floating-point results fail rather than silently drift.

The [verification record](verification.md) documents all 1,720 passing tests,
16 successful experiment gates, package checks, and independent review.
