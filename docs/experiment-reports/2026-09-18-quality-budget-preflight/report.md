# M4G decision-component preflight

**Component preflight passed. Full M4G runner preflight remains pending.** No model fitting or held-out task evaluation was performed.

Request: `sha256:0a8496a7ace6115e52a1688002335ae906c2cf2eb9285f1d468402b23fe843ae`

Result: `sha256:50a274ec1a34c443915fd39393ee5cd0684f28de4314e12490db2995edd5b898`

- Three pinned archives authenticated; one common initialization.
- 21 planned fits, 60 planned evaluation cells and 213 distinct role seeds.
- 55 coverage checks; all counts enumerated at N=1,2,8,32,32768.
- Minimum checked marginal coverage: 0.999968168150.
- 31 independent binomial-tail inversions; interval nesting and literal threshold boundaries checked.
- 4,096 joint binary batches; perfectly correlated interval family; 729 horizon patterns and 784 distinct bracket pairs.

| Planned work | Endpoint draws |
| --- | ---: |
| Finite training grid (18 fits) | 4,423,680,000 |
| Equilibrium training (3 fits) | 737,280,000 |
| Total training | 5,160,960,000 |
| All evaluation | 983,040,000 |
| Equilibrium diagnostic evaluation subset | 98,304,000 |

These are declared modeled operations, not physical measurements. Equilibrium sweep costs are null, not zero. Independent trajectory counts do not fall with K.

Each CP marginal covers with probability >=1-0.05/1560. The union bound covers all 1560 intervals with probability >=0.95, without cross-cell independence. On that event the rectangle encloses the population loss. Coverage is conditional on frozen fits and binomial within-cell sampling. A finite grid is a numerical check, not a uniform proof.

Next: implement the matched runner, validate both training laws at every K, and complete its integrity preflight before executing the fixed study. No task-quality result or inference saving follows from this component gate.
