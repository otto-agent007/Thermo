# M4F: bounded asymmetry-preservation experiment

One shared initialization, K4, beta 1, float64, caps [-2,2]. One unit-weight squared directional-asymmetry term is added to F_w + C_w. The same two starts, step 1/2 and 100 updates give 7,400 new group updates and a matched 7,400-update weighted control. Replay is additional verification work. The three source seeds are not independent fits. No sampling or hardware evidence.

Averages of squared asymmetry error and absolute asymmetry error are distinct. All-parent metrics and exact killed survival remain visible. Survival excludes reentry and is not terminal particle conservation or full occupancy fidelity.

| Cell | Survival at 500 | Hop MAE | Asymmetry MAE | Asymmetry MSE | Target-weighted failure | Uniform failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logical_reference | 1 | 0 | 0 | 0 | 0 | 0 |
| frozen_initial | 0.00040392135 | 0.0398724355 | 0.0290380109 | 0.00159473577 | 0.0128132027 | 0.269615481 |
| target_context_1 | 0.0614958479 | 0.0248519703 | 0.031607622 | 0.00173340618 | 0.00761200835 | 0.264284633 |
| asymmetry_1 | 0.0611288246 | 0.0155785822 | 0.00352232527 | 6.90882748e-05 | 0.00765424285 | 0.266660595 |

| Cell | Target F | Uniform F | Mean TV | Max TV | Conditional hop MAE | Mean empty failure | Mean hop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logical_reference | 0 | 0 | 0 | 0 | 0 | 0 | 0.05 |
| frozen_initial | 0.0070939209 | 0.233468949 | 0.289500032 | 0.80052744 | 0.0845249602 | 0.00415358709 | 0.0896657658 |
| target_context_1 | 0.00101969769 | 0.230555572 | 0.272859584 | 0.798046753 | 0.0659204303 | 0.00500114178 | 0.0574957328 |
| asymmetry_1 | 0.00108960907 | 0.227312264 | 0.271938832 | 0.793713417 | 0.0543165436 | 0.00500067363 | 0.0536166815 |

## Predeclared primary screen

All three margins must be nonnegative, using literal comparisons. This descriptive screen is not a release gate or absolute task-quality certificate.

| Requirement | Margin (nonnegative passes) | Pass |
| --- | ---: | --- |
| Retain weighted-control survival | -0.000367023223 | False |
| Restore frozen-initial asymmetry | +0.0255156856 | True |
| Preserve weighted-control hopping | +0.00927338809 | True |

Joint screen: **False**.

## Signed comparisons

| Reference | Survival difference | Hop MAE difference | Asymmetry MAE difference |
| --- | ---: | ---: | ---: |
| target_context_1 | -0.000367023223 | -0.00927338809 | -0.0280852968 |
| frozen_initial | +0.0607249033 | -0.0242938533 | -0.0255156856 |

## Optimizer and evidence limits

Selections (archived start, archived endpoint, zero start, zero endpoint): [0, 33, 0, 4]. Maximum endpoint projected-gradient residual: 0.0129243028. Residuals do not certify convergence.

The complete historical reference remains hash-pinned. Consumed control cells and logical profiles are recomputed at absolute tolerance 1e-12, relative tolerance zero; unused grids are not rerun. Comparisons use unchanged archived values. All new results require strict complete replay before reporting and completion.

Stop after this fixed coefficient regardless of outcome; proceed to the M4G protocol decision without retuning. No optimal-capacity, generalization, inference-sample saving, or device advantage follows from this exact-reference study.

Request: `sha256:a0ec3eeb9ae54ef0e4d53be68bc81bc6c28ebd151c73006b15f00a304554cd7e`

Result: `sha256:8173af83fad19d6cc59275f4243a84b24fe2e4d5a89fd5a734af6e962ce56ce8`
