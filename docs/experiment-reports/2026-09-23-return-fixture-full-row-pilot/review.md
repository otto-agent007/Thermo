# Independent review and interpretation

An independent read-only review checked the frozen protocol, exact visible
conditional and shared nine-parameter gradient, all-entry qualification gate,
201-call backtracking schedule, persisted replay, and the report. The reviewer
compared the weight-zero arm with the previously committed return-fixture
archive: initial and final base evaluations, all 200 proposal base
evaluations and parameter vectors, and every selection matched. The reviewer
found no remaining important mathematical or selection defect. An initial
report omission of target visitation and mismatched Markdown table
separators were fixed before the source was frozen; focused tests check the
final table column counts.

At weight zero, the largest of 16 conditional errors is 0.41630688, on
unvisited input row 11; row 01 reaches 0.38677980. Weight 100 reduces
the all-entry maximum to 0.08507169, now on row 01. Its survival falls
from 0.93473959 to 0.90660262. Weight 1 reaches survival 0.92336319
and maximum error 0.08742565. No arm meets survival 0.95 or maximum-entry
error 0.005. All four final arms, their row diagnostics, and rejected
proposals remain in the archive.

The hop-only pilot's weight-10 reverse-hop recovery does not carry over:
the full-row weight-10 arm gives reverse 01→10 probability 0.00155289
against 0.09037112 target, and forward 10→01 probability 0.00031204
against 0.00962888 target. The target visits input row 01 on only
0.00953616 of third operations, while row 11 is never visited on valid
target paths. Equal row weighting exposes errors that path visitation can
hide, but this short run does not settle joint feasibility, convergence,
parameterization limits, or a larger M4G program.
