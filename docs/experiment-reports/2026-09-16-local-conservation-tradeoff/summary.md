# Better local averages, worse conserved execution

September 16, 2026. The predeclared local K4 search reduced mean conservation
failure from **26.96% to 3.55%** at penalty 10, but reduced the probability of
conserving one particle through all 500 operations from **4.039e-4 to 1.701e-8**.
Mean directed hopping fell from 8.97% to 0.0334%, against a logical target mean
of 5%. This bounded uniform-context objective did not deliver the intended
combination of conservation and asymmetric-transition fidelity.

All values below are exact-reference calculations for the evaluated parameters,
up to float64 arithmetic. There is no sampled evaluation, confidence interval,
independent-fit replication, or physical-hardware measurement.

| Evaluated cell | Mean local failure | Mean empty-edge failure (00) | Mean unconditional hop probability | Unconditional hop MAE | 500-operation uninterrupted survival |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logical reference | 0% | 0% | 5.000% | 0 | 1 |
| Frozen initial | 26.962% | 0.415% | 8.967% | 0.0398724 | 4.0392135e-4 |
| Penalty 0 | 4.257% | 2.809% | 0.982% | 0.0408266 | 3.8689664e-7 |
| Penalty 1 | 3.554% | 3.371% | 0.0363% | 0.0496366 | 3.1971546e-8 |
| Penalty 10 | 3.549% | 3.509% | 0.0334% | 0.0496660 | 1.7006944e-8 |

Local means weight all 37 groups and four parent contexts uniformly; hop means
and MAE weight the two single-particle directions uniformly. Survival instead
follows the actual schedule and evolving distribution of surviving paths.
The logical reference's stored final mass is 1.0000000000000002 from roundoff;
its transitions preserve particle count exactly.

## What changed, and why the average is insufficient

Every fitted cell improves mean row total variation and squared conditional
error substantially relative to initialization. Yet all three have worse
uninterrupted program survival. Even penalty 0, which optimizes only squared
conditional fidelity, raises empty-edge creation and worsens survival.
Median first exit moves from operation 52 initially to 24, 21, and 20 for
penalties 0, 1, and 10.

The uniform objective gives parent 11 one quarter of its weight, although
that context cannot occur before first exit from the one-particle sector.
It also permits better occupied-parent accuracy to offset worse empty-parent
accuracy in its aggregate. The exact recurrence exposes the program-level
consequence: unconditional first-exit mass through parent 00 rises from
0.252541 initially to 0.870469, 0.905716, and 0.909908 in the fitted cells.
These are descriptive contributions under each cell's own surviving path
distribution, not an isolated causal decomposition.

The penalty also suppresses intended hopping. Directed-hop asymmetry MAE rises
from 0.0290380 initially to 0.0494791, 0.0609995, and 0.0610391. Conditional-hop
MAE is 0.0401745, 0.0496224, and 0.0496536: these improve on the poor baseline
value of 0.0845250, but do not rescue the worsened unconditional hop errors.
Conditioning on a conserving output cannot substitute for measuring lost mass.

## Budget, selection, and limits

The [protocol](../../experiments/local-conservation-tradeoff.md) was committed
at `fadca77` before fitting. Three authenticated original M1 source records
share one initial parameter matrix, so it is optimized once. Each of 37 groups
receives exactly 100 projected-gradient updates from the archived start and
from zero for each penalty: 22,200 group updates total. Keep K4, beta 1,
float64 and caps [-2,2]. Each cell selects its lowest-objective candidate from
the two starts and their update-100 endpoints using the predeclared tie order.
No settings or additional runs were chosen after inspecting outcomes.

Penalty 0 selects 22 archived endpoints and 15 zero endpoints. Penalties 1 and
10 each select all 37 zero endpoints. Maximum endpoint projected-gradient
residuals are 0.003124, 0.005481, and 0.054565, respectively. These are diagnostics,
not convergence certificates. This short nonconvex search establishes attained
results only. It cannot prove optimal capacity, impossibility under the caps,
or an advantage of a different architecture.

The exact logical reference conserves perfectly but is not asserted to lie in
the capped parameter family. Killed-path survival is not terminal leakage or
terminal occupancy loss. This study does not re-evaluate the trained M4B arms.

## Research decision

Do not scale up this uniform-context penalty as a program-conservation remedy.
The next bounded question is whether the existing exact target-context profiles
can improve this K4 conservation–fidelity objective while retaining directed-hop
fidelity. Reuse the checked trace and pooling machinery in
`pasym_swap_context.py`; the new question concerns finite-K4 training and
conservation, not a new context-estimation framework. Predeclare its objective, budget, hop-fidelity measurements and
stopping rule, then compare with all cells here. Context reweighting is a
hypothesis to test, not an established fix. More full-program training and
architecture conclusions remain unsupported by these results.

## Evidence and reproduction

[Full artifact](study.json), [generated report](report.md),
[requested inputs](protocol.json), [completion](completion.json), and
[runtime provenance](provenance.json) were generated at clean implementation
commit `4d655119114a0ab7cb361825895df6fc31718912`.
The artifact embeds the three authenticated sources, every optimizer attempt,
selected parameters, exact local tables and errors, and all 500 first-exit rows
for every cell. Reporting replays all updates and measurements before completion.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  uv run python -m thermo_lab.conservation_tradeoff_audit \
  --output-dir results/local-conservation-tradeoff
```

Use a fresh destination. Exact replay requires compatible floating-point
results. The [verification record](verification.md) documents test, experiment,
package and independent-review evidence.
