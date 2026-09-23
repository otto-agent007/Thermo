# Three-operation return fixture: complete local-row penalty

Exact-reference CPU study on one frozen K4 three-site circuit, with no samples. Every arm starts at the same parameters and evaluates 201 complete candidate states.

The declared objective is path KL plus weight times the sum of squared errors across all four outcomes of all four input rows. Arms retain all rejected proposals; selection uses only that objective.

| Weight | Survival | Path KL | Max error across 16 entries | Forward 10→01 model (target) | Reverse 01→10 model (target) | Hop MAE | Asymmetry MAE | Visited row MAE | Qualifies |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.934739588 | 0.115122732 | 0.416306875 | 0.000411512102 (0.00962887814) | 0.00469740014 (0.0903711219) | 0.0474455439 | 0.0764563557 | 0.0116764789 | False |
| 1 | 0.923363191 | 0.130556844 | 0.0874256547 | 0.000292808571 (0.00962887814) | 0.00294546713 (0.0903711219) | 0.0483808622 | 0.0780895852 | 0.0132106604 | False |
| 10 | 0.911408904 | 0.142716022 | 0.0888182292 | 0.000312041219 (0.00962887814) | 0.00155289266 (0.0903711219) | 0.0490675331 | 0.0795013923 | 0.0153190711 | False |
| 100 | 0.90660262 | 0.110451336 | 0.0850716883 | 0.00264463585 (0.00962887814) | 0.00529943356 (0.0903711219) | 0.0460279653 | 0.078087446 | 0.0161468275 | False |

Pilot qualification requires exact survival ≥ 0.95 and maximum absolute error ≤ 0.005 across all 16 entries. The unchanged initial survival is 0.118342345.
The forward target is 0.009628878; a zero forward hop fails this gate. Row 11 is included even though valid target paths never visit it.
Target input-row visitation by operation (00, 01, 10, 11): 1: 0, 0, 1, 0; 2: 0.990371122, 0, 0.00962887814, 0; 3: 9.27152942e-05, 0.00953616284, 0.990371122, 0.

| Weight | Row 00 max | Row 01 max | Row 10 max | Row 11 max | Row 00 TV | Row 01 TV | Row 10 TV | Row 11 TV | Row 00 MAE | Row 01 MAE | Row 10 MAE | Row 11 MAE | Final gradient norm | Accepted rounds | Clipped proposal components | Final parameters on bounds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0233691975 | 0.386779799 | 0.0168529904 | 0.416306875 | 0.0233691975 | 0.388260302 | 0.0216050073 | 0.416306875 | 0.0116845987 | 0.194130151 | 0.0108025037 | 0.208153438 | 0.0792519212 | 25 | 195 | 1 |
| 1 | 0.0242913275 | 0.0874256547 | 0.0178492184 | 0.0411020432 | 0.0242913275 | 0.0874256547 | 0.0271852879 | 0.0411020432 | 0.0121456637 | 0.0437128274 | 0.013592644 | 0.0205510216 | 0.0885481535 | 25 | 388 | 2 |
| 10 | 0.0253790371 | 0.0888182292 | 0.0236483745 | 0.0304815861 | 0.0253790371 | 0.0888182292 | 0.0329652114 | 0.0304815861 | 0.0126895185 | 0.0444091146 | 0.0164826057 | 0.0152407931 | 0.207130897 | 25 | 422 | 2 |
| 100 | 0.0279200671 | 0.0850716883 | 0.027223703 | 0.0286914845 | 0.0279200671 | 0.0850716883 | 0.0342079453 | 0.0286914845 | 0.0139600335 | 0.0425358442 | 0.0171039726 | 0.0143457423 | 1.37799138 | 25 | 548 | 2 |

These are all final arms; there is no after-the-fact weight selection. Complete row errors expose failures that visit-weighted averages can hide. This single deterministic fixture does not establish capacity limits, convergence, 500-operation quality, inference-sample savings, or hardware performance.

Complete persisted numerical replay precedes this report. Generation and replay each make 842 complete evaluator calls (two K4 law calculations per call).
