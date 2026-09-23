# Independent review and interpretation

An independent read-only code and evidence review found no defect in the
forward 10→01 and reverse 01→10 visible-law indices, their shared nine-parameter
derivatives, the 804 arm / 38 preflight evaluator-call accounting, or the
complete persisted replay. The 18 penalty-gradient components at two interior
points had maximum central-difference error 1.59e-9. The zero-weight runner
reproduces all 200 proposals and evaluations within 1e-12 absolute tolerance,
and makes identical choices in the saved three-operation path-KL/backtracking
arm. The strengthened regression test compares the full archived trace,
including structure and numeric types, not just final metrics.

The weight-10 arm restores the reverse logical hop to 0.09072446 versus its
0.09037112 target, from 0.00469740 at weight zero. Survival changes from
0.93473959 to 0.93550236. This does not restore the complete local row:
the reverse parent 01 produces invalid output 00 with probability 0.52062153
versus target zero, and retains output 01 with probability 0.37827096 versus
target 0.90962888. Its four-outcome row MAE increases from 0.19413015 to
0.26567896. Both the reverse hop and the other outcomes must be reported.

The forward 10→01 target is 0.00962888. Weight 10 produces 0.00048545 and
weight 100 produces 0.00001579. Because the target itself is below the
predeclared absolute-error tolerance 0.01, a zero forward hop would pass
that part of the pilot gate. This is a weakness of the gate for judging
forward fidelity; it was not changed after seeing results. All four arms
fail the pilot gate on survival (<0.95), so no qualifying result is hidden.

The weight-100 arm reaches the largest survival, 0.94475039, but its row 01
four-outcome MAE is 0.29463542. The single frozen starting point and short
budget do not establish an optimal weight, joint feasibility, representational
capacity, convergence, or full M4G task quality. A later test should freeze a
complete conditional-row fidelity criterion, or a scale-aware forward-hop
tolerance, before choosing and executing additional fitting arms.
