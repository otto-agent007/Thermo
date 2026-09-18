# Verification

All commands ran against clean implementation commit
`261ca1f5a6b8125c78806700fb750912ef24e62b`; no source or documentation changed
while the full test suites and experiment gates were running. The subsequent
commit adds evidence and updates continuation documentation only.

- **1,848 tests passed**: 1,432 unit/upstream and 416 integration, including all slow tests.
- All 45 new component tests are included in that complete suite.
- All **20 experiment gates** passed; no scientific acceptance thresholds were relaxed.
- Frozen sync, offline lock check, Ruff lint/format and whitespace checks passed.
- Wheel and source distribution builds passed; both include all 10 configs and four new modules.
- The new CI shell evidence check executed successfully against generated output.
- Independent statistical and implementation review findings were fixed and rechecked.

| Command/gate | Result | Seconds |
| --- | --- | ---: |
| asymmetry-preservation | Passed | 57.62 |
| bounded-finite-sweep-refinement | Passed | 130.37 |
| composed-finite-gibbs | Passed | 175.79 |
| composed-trajectory-refinement-one-step | Passed | 64.76 |
| conservation-diagnostic | Passed | 33.09 |
| context-weighted-conservation | Passed | 167.78 |
| finite-sweep-gradient-contract | Passed | 5.64 |
| frozen-pair-finite-sweeps | Passed | 103.47 |
| independent-pasym-swap | Passed | 22.81 |
| local-conservation-tradeoff | Passed | 84.80 |
| matched-training-budget | Passed | 360.51 |
| model-context-pasym-swap | Passed | 46.22 |
| quality-budget-preflight | Passed | 3.30 |
| smoke | Passed | 2.87 |
| target-context-pasym-swap | Passed | 55.71 |
| tests-integration | Passed | 1814.69 |
| tests-unit-upstream | Passed | 693.65 |
| thrml-run | Passed | 5.49 |
| torx-run | Passed | 1.73 |
| trajectory-reinforce-estimator | Passed | 13.66 |
| trajectory-reinforce-one-step | Passed | 19.51 |
| weighted-graph-walk | Passed | 3.04 |

Times are observed local verification durations, including process startup and
concurrent execution on CPU. They are not isolated experiment benchmarks or
device performance claims. Test groups ran on disjoint CPU pairs; historical
gate pipelines used the remaining CPUs with BLAS/OMP threads fixed to one.
The M4F gate also exercised its declared baseline NumPy CPU path.

## Provenance and publication

The pinned protocol commit is `ae2ecac3be45501d52ac60113ee6ccc71e6adc0c`.
Publication uses the connected GitHub app because local Git transport lacks
credentials. Published commits preserve the exact local trees; commit metadata
differs. The recorded experiment provenance retains the original local identity.

| Local commit | Published commit | Shared tree |
| --- | --- | --- |
| `7cb8f934c60ff11cb526a9e345da23ec8ea25e07` | `69d349a876d02a2fcd5a2bc8c407f3d5da45962a` | `548536ec24beee1e212543b617ee850bbe913e1b` |
| `261ca1f5a6b8125c78806700fb750912ef24e62b` | `979be526a59b6620339e8552eb2b96e7257f2e57` | `97f79f4d683efbccbd1ad8b3d033c7026982c4de` |

[PR #32](https://github.com/otto-agent007/Thermo/pull/32) is stacked on protocol
[PR #31](https://github.com/otto-agent007/Thermo/pull/31). Remote CI status is
reported separately in the PR; local success is not a claim of remote completion.
The published implementation commit passed all 26 jobs in
[CI run 113](https://github.com/otto-agent007/Thermo/actions/runs/35305883757).
The subsequent documentation/evidence commit starts its own remote run.
