# Three-operation return fixture: forward/reverse fidelity penalty

Exact-reference CPU study on one frozen K4 three-site circuit, with no samples. Every arm starts at the same parameters and evaluates 201 complete candidate states.

The declared objective is path KL plus weight times the sum of squared forward 10→01 and reverse 01→10 probability errors. Arms retain all rejected proposals; selection uses only that objective.

| Weight | Survival | Path KL | Forward model (target) | Reverse model (target) | Forward error | Reverse error | Hop MAE | Asymmetry MAE | Visited row MAE | Qualifies |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.934739588 | 0.115122732 | 0.000411512102 (0.00962887814) | 0.00469740014 (0.0903711219) | -0.00921736603 | -0.0856737217 | 0.0474455439 | 0.0764563557 | 0.0116764789 | False |
| 1 | 0.934874364 | 0.115280583 | 0.000407005718 (0.00962887814) | 0.00580481759 (0.0903711219) | -0.00922187242 | -0.0845663043 | 0.0468940883 | 0.0753444319 | 0.011681547 | False |
| 10 | 0.935502362 | 0.112733586 | 0.000485453531 (0.00962887814) | 0.0907244575 (0.0903711219) | -0.00914342461 | 0.000353335682 | 0.00474838014 | 0.00949676029 | 0.0117548999 | False |
| 100 | 0.944750388 | 0.169677618 | 1.57868303e-05 (0.00962887814) | 0.0934606645 (0.0903711219) | -0.00961309131 | 0.00308954264 | 0.00635131697 | 0.0127026339 | 0.0102775338 | False |

Pilot qualification requires exact survival ≥ 0.95 and each signed hop error within ±0.01. The unchanged initial survival is 0.118342345.
The forward target is below 0.01, so even a zero forward hop would pass that absolute-error condition. Read the forward model probability and row errors alongside qualification; this pilot gate is too loose to certify faithful forward movement.

| Weight | Row 00 MAE | Row 01 MAE | Row 10 MAE | Row 11 MAE | Final gradient norm | Accepted rounds | Clipped proposal components |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0116845987 | 0.194130151 | 0.0108025037 | 0.208153438 | 0.0792519212 | 25 | 195 |
| 1 | 0.0116788327 | 0.203431734 | 0.0107686107 | 0.210442973 | 0.0780664136 | 25 | 195 |
| 10 | 0.0115679241 | 0.265678958 | 0.0106367657 | 0.280669001 | 0.0840742442 | 25 | 167 |
| 100 | 0.0118483864 | 0.294635418 | 0.00814375593 | 0.398411219 | 0.246354764 | 25 | 115 |

These are all final arms; there is no after-the-fact weight selection. Forward and reverse hop errors are reported individually to expose rare-row failures that visit-weighted averages can hide. This single deterministic fixture does not establish capacity limits, convergence, 500-operation quality, inference-sample savings, or hardware performance.

Complete persisted numerical replay precedes this report. Generation and replay each make 842 complete evaluator calls (two K4 law calculations per call).
