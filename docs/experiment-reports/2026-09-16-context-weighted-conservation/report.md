# Matched K4 context-weighting comparison

One shared archived initialization; three source seeds are not independent fits. Exact logical contexts replace uniform parent weights with the same K4, beta 1, caps [-2,2], starts, 100-update policy and three penalties. Each arm has 22,200 group updates; replay is additional verification work. All metrics are exact_reference up to float64 arithmetic. No sampled or hardware measurements.

Target-context F/C average the fixed logical pre-gate contexts over 500 occurrences, using multiplicity/500. They do not estimate fitted-model contexts. Uniform means retain all four parents, including the zero-training-weight 11 context. Hop metrics average both single-particle directions over all groups.

| Cell | Target-weighted failure | Uniform failure | Mean 00 failure | Mean hop | Hop MAE | Asymmetry MAE | Final uninterrupted survival |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logical_reference | 0 | 0 | 0 | 0.05 | 0 | 0 | 1 |
| frozen_initial | 0.0128132027 | 0.269615481 | 0.00415358709 | 0.0896657658 | 0.0398724355 | 0.0290380109 | 0.00040392135 |
| penalty_0 | 0.0290891083 | 0.0425727227 | 0.0280885784 | 0.0098195363 | 0.040826572 | 0.0494791293 | 3.86896638e-07 |
| penalty_1 | 0.0339330511 | 0.0355429898 | 0.033707203 | 0.000363436576 | 0.0496365634 | 0.0609994931 | 3.19715461e-08 |
| penalty_10 | 0.0351472907 | 0.0354864797 | 0.0350909009 | 0.000333980576 | 0.0496660194 | 0.0610390838 | 1.70069441e-08 |
| target_context_0 | 0.0108451417 | 0.248058608 | 0.00792295372 | 0.0578259479 | 0.0214504056 | 0.0265003363 | 0.00828956296 |
| target_context_1 | 0.00761200835 | 0.264284633 | 0.00500114178 | 0.0574957328 | 0.0248519703 | 0.031607622 | 0.0614958479 |
| target_context_10 | 0.00739773554 | 0.283336895 | 0.00456127647 | 0.070587745 | 0.0312648837 | 0.0418768103 | 0.0828309092 |

| Cell | Target-weighted F | Uniform F | Mean row TV | Max row TV | Conditional hop MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| logical_reference | 0 | 0 | 0 | 0 | 0 |
| frozen_initial | 0.0070939209 | 0.233468949 | 0.289500032 | 0.80052744 | 0.0845249602 |
| penalty_0 | 0.00129257216 | 0.00314968311 | 0.0446600893 | 0.0970456756 | 0.0401744847 |
| penalty_1 | 0.0017621874 | 0.0036525746 | 0.0462803061 | 0.0976475639 | 0.0496224167 |
| penalty_10 | 0.00189134778 | 0.0037561222 | 0.0469194109 | 0.0977098007 | 0.0496535891 |
| target_context_0 | 0.000810328024 | 0.207143336 | 0.255381421 | 0.803388438 | 0.0544567034 |
| target_context_1 | 0.00101969769 | 0.230555572 | 0.272859584 | 0.798046753 | 0.0659204303 |
| target_context_10 | 0.00151166311 | 0.259935429 | 0.296801661 | 0.796571111 | 0.0836401437 |

## Descriptive joint screen

New minus reference. Passing requires strictly higher survival with neither unconditional hop MAE nor asymmetry MAE increasing. It is not a statistical test, absolute fidelity certificate, or release gate. No penalty is selected.

| Weighted cell | Reference | Survival difference | Hop MAE difference | Asymmetry MAE difference | Joint screen |
| --- | --- | ---: | ---: | ---: | --- |
| target_context_0 | penalty_0 | +0.00828917606 | -0.0193761663 | -0.022978793 | True |
| target_context_0 | frozen_initial | +0.00788564161 | -0.0184220299 | -0.00253767458 | True |
| target_context_1 | penalty_1 | +0.0614958159 | -0.0247845931 | -0.029391871 | True |
| target_context_1 | frozen_initial | +0.0610919265 | -0.0150204652 | +0.00256961116 | False |
| target_context_10 | penalty_10 | +0.0828308922 | -0.0184011357 | -0.0191622736 | True |
| target_context_10 | frozen_initial | +0.0824269878 | -0.00860755176 | +0.0128387994 | False |

## Optimizer diagnostics

| Penalty | Archived start | Archived endpoint | Zero start | Zero endpoint | Max endpoint projected-gradient residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 33 | 0 | 4 | 0.0042376515 |
| 1 | 0 | 31 | 0 | 6 | 0.012866468 |
| 10 | 0 | 34 | 0 | 3 | 0.11634915 |

Residuals are diagnostics, not convergence certificates. Exact killed survival forbids reentry and is not terminal count-one probability or occupancy fidelity. A finite nonconvex search cannot establish optimal capacity, hardware advantage, or a general benefit from context weighting. No confidence intervals are asserted.

The complete pinned uniform control, derived profiles, new fits, all-parent laws, survival curves, evaluations and comparisons are persisted. Reporting authenticates and replays both arms. Incompatible floating-point results fail exact replay.

Request: `sha256:cd76f2b367dc501e94df5852fd4f80e7782cd1be04984a7b7e3fe1a5f174d408`

Result: `sha256:22be67a731ebdb55f19736da0a1b95ebbaf2df86410bf11c25021712cd428222`
