# Bounded local conservation–fidelity trade-off

Three authenticated source seeds share one matrix; this is one deterministic 37-group search, not three independent fits. K4, beta 1, float64, caps [-2,2]. Three penalties × 37 groups × two starts × 100 projected updates = 22,200 group updates. No sampled training or evaluation. All reported laws and measurements are exact_reference up to floating-point arithmetic; numerical optimization does not establish a global optimum or convergence.

All local means weight groups and parent contexts uniformly. Hop MAE averages the two single-particle directions. Conditional hop MAE conditions each direction on a conserving output; it must be read beside unconditional error and failure. Asymmetry MAE compares the difference between the two unconditional hop probabilities.

| Cell | Mean failure | Fidelity F | Mean row TV | Max row TV | Hop MAE | Conditional hop MAE | Asymmetry MAE | 500-operation survival |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logical_reference | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| frozen_initial | 0.269615481 | 0.233468949 | 0.289500032 | 0.80052744 | 0.0398724355 | 0.0845249602 | 0.0290380109 | 0.00040392135 |
| penalty_0 | 0.0425727227 | 0.00314968311 | 0.0446600893 | 0.0970456756 | 0.040826572 | 0.0401744847 | 0.0494791293 | 3.86896638e-07 |
| penalty_1 | 0.0355429898 | 0.0036525746 | 0.0462803061 | 0.0976475639 | 0.0496365634 | 0.0496224167 | 0.0609994931 | 3.19715461e-08 |
| penalty_10 | 0.0354864797 | 0.0037561222 | 0.0469194109 | 0.0977098007 | 0.0496660194 | 0.0496535891 | 0.0610390838 | 1.70069441e-08 |

## Optimizer diagnostics

Residual = ||theta - clip(theta - grad J, -2,2)||_infinity at each update-100 endpoint. It is not a convergence acceptance test. Candidate selection is predeclared per group and penalty; no penalty is declared a statistical winner.

| Penalty | Archived start selected | Archived endpoint | Zero start | Zero endpoint | Max endpoint residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 22 | 0 | 15 | 0.00312378172 |
| 1 | 0 | 0 | 0 | 37 | 0.0054805709 |
| 10 | 0 | 0 | 0 | 37 | 0.0545648006 |

The logical reference preserves particle count exactly but is not claimed to be representable by these capped parameters. Survival kills paths on their first exit; it is not terminal count-one probability or occupancy fidelity. The study does not evaluate equilibrium-trained controls, physical timing/energy, confidence intervals, or generalization across initializations. No architecture or parameter-cap impossibility result follows from a finite search.

The artifact retains per-group attempts, selected parameters, visible laws, per-parent failure and TV, directional errors, and all 500 first-exit rows. Reporting authenticates sources and replays every update and measurement. Exact replay requires compatible floating-point results.

Request: `sha256:ff351a369b8718527caad913723dc362892e0dea124d9a0744735b716fb22263`

Result: `sha256:04be52b0465cafbfd3281109e8606c570b8b837fce4494a5b12a117ea77517f8`
