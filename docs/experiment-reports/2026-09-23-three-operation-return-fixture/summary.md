# Exact three-operation return fixture objective comparison

Three sites, three operations, K4, shared nine parameters; exact_reference. No new samples or M4G fitting. All final arms are retained.

Each arm uses 201 complete evaluator calls. Fixed takes 200 projected updates; backtracking accepts at most 25. Calls, not update counts or hardware costs, are matched.

| Objective | Step rule | Occupancy loss | Path KL | Joint valid loss | Path gap | Survival | Leakage | Hop MAE | Asymmetry MAE | Visited row MAE | Visited hop MAE | Reverse 01 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unchanged | baseline | 0.717076422 | 2.53299972 | 2.52202016 | 0.0109795582 | 0.118342345 | 0.606026593 | 0.0704213425 | 0.140842685 | 0.273521881 | 0.0694772569 | 0.0189966476 |
| occupancy | fixed | 0.0484718303 | 0.670004117 | 0.669422347 | 0.000581769569 | 0.514429366 | 0.296324593 | 0.0322733404 | 0.0528263433 | 0.093628792 | 0.00611085376 | 0.0316846099 |
| occupancy | backtracking | 0.00151783466 | 0.385459042 | 0.381842623 | 0.00361641894 | 0.725023389 | 0.0909196078 | 0.0288705372 | 0.0388099741 | 0.0480931423 | 0.00964972117 | 0.0420955976 |
| trajectory_kl | fixed | 0.00617276502 | 0.235695681 | 0.231820829 | 0.00387485188 | 0.816536294 | 0.141997795 | 0.0475698422 | 0.0774757074 | 0.032791986 | 0.00919964599 | 0.00406342597 |
| trajectory_kl | backtracking | 0.000327123429 | 0.115122732 | 0.110423504 | 0.0046992271 | 0.934739588 | 0.0549885962 | 0.0474455439 | 0.0764563557 | 0.0116764789 | 0.00958018621 | 0.00469740014 |
| valid_terminal | fixed | 0.00607637755 | 0.23487039 | 0.23095481 | 0.00391557982 | 0.817550501 | 0.141216734 | 0.0476306735 | 0.0775608272 | 0.0325941063 | 0.00921832125 | 0.00396003478 |
| valid_terminal | backtracking | 0.000305467592 | 0.117720888 | 0.112703411 | 0.00501747649 | 0.935948232 | 0.0534999236 | 0.0475677332 | 0.0765696961 | 0.0115358696 | 0.00964624317 | 0.00451854058 |

Target visitation weights for every parent row (each operation sums to one):

| Operation | Row 00 | Row 01 | Row 10 | Row 11 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 1 | 0 |
| 2 | 0.990371122 | 0 | 0.00962887814 | 0 |
| 3 | 9.27152942e-05 | 0.00953616284 | 0.990371122 | 0 |

Local row MAE includes all four visible outcomes, including leakage. The forward 10 and reverse 01 columns show model hop probability (target in parentheses).

| Objective | Step | Row 00 MAE | Row 01 MAE | Row 10 MAE | Row 11 MAE | forward 10 | reverse 01 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unchanged | baseline | 0.376074198 | 0.11669389 | 0.223482469 | 0.399163048 | 0.0790970889 (0.00962887814) | 0.0189966476 (0.0903711219) |
| occupancy | fixed | 0.186060083 | 0.181772387 | 0.0474335904 | 0.372840061 | 0.0037687094 (0.00962887814) | 0.0316846099 (0.0903711219) |
| occupancy | backtracking | 0.121316556 | 0.269087558 | 0.0107768512 | 0.439379934 | 0.000163327943 (0.00962887814) | 0.0420955976 (0.0903711219) |
| trajectory_kl | fixed | 0.0477256271 | 0.135026944 | 0.0249089057 | 0.271459067 | 0.000796889602 (0.00962887814) | 0.00406342597 (0.0903711219) |
| trajectory_kl | backtracking | 0.0116845987 | 0.194130151 | 0.0108025037 | 0.208153438 | 0.000411512102 (0.00962887814) | 0.00469740014 (0.0903711219) |
| valid_terminal | fixed | 0.047632209 | 0.133747169 | 0.0246644519 | 0.2715038 | 0.000778618277 (0.00962887814) | 0.00396003478 (0.0903711219) |
| valid_terminal | backtracking | 0.0114487618 | 0.21542811 | 0.0106068333 | 0.216806678 | 0.000345992987 (0.00962887814) | 0.00451854058 (0.0903711219) |

Optimizer diagnostics: gradient norm at initialization and selected final state for each arm's objective; clipped components sum over all 200 evaluated proposals, including rejected proposals.

| Objective | Step | Initial gradient norm | Final gradient norm | clipped components |
| --- | --- | ---: | ---: | ---: |
| occupancy | fixed | 1.251669 | 0.175466845 | 0 |
| occupancy | backtracking | 1.251669 | 0.00831565125 | 0 |
| trajectory_kl | fixed | 3.7968321 | 0.269946231 | 0 |
| trajectory_kl | backtracking | 3.7968321 | 0.0792519212 | 195 |
| valid_terminal | fixed | 3.72876908 | 0.271883572 | 0 |
| valid_terminal | backtracking | 3.72876908 | 0.0809232778 | 189 |

The return operation exposes the reverse 01 parent and merges valid histories at the same terminal state. Path KL minus joint valid-terminal divergence is the target-weighted conditional history KL. Visited-row MAE includes all parent rows with target visitation; visited-hop MAE includes only 01 and 10. All-row hop/asymmetry metrics remain separate.

The bound exp(-path KL) is a numerical sufficient survival diagnostic, not an interval-certified proof or the full M4G quality gate. Objective scaling affects gradients; this fixture does not establish convergence, capacity, generalization, inference savings or hardware advantage.

Generation uses 1,206 arm evaluations plus 19 preflight evaluations; complete persisted replay repeats that work. All rejected candidates are retained in study.json.
