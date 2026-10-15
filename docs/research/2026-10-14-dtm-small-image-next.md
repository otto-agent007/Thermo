# DTM small-image results: what changes next (2026-10-14)

THERMES note on loop proposal P-0006. The evidence is in the
[findings](../experiment-reports/2026-10-14-dtm-small-image/findings.md).
Proposals only; the owner decides.

- **The number to design against comes from T = 2, not from the coded
  operating point.** Rule 6 picks T = 1, because the steps verdict is
  inconclusive. At T = 1, K\* and b\* do not resolve. The usable answer is
  4 bits, K = 64, T = 2: error 0.075, 2.0e-8 J per sample.
- **Continuation is the open risk for every chained-sampler row.** Carrying
  free-node state saves 3.9 times the joule at T = 1, K = 64, but one of
  three models stayed trapped through 1,024 sweeps. A later row that relies
  on continuation should price a reset rule (every N images) against a
  random start, per seed.
- **The task-level negative stands on this family.** A 114-parameter logistic
  is at least as good in every seed. It is not a reason to drop the task
  line. It is a reason to pick the next task where a small digital model is
  known to be weak, and to keep the baseline inside the row.
- **No queue changes from THERMES.** Candidate rows for the owner: a
  continuation reset rule at T = 1, and a real binarized image set (a
  download, which needs the owner).

Evidence classes: exact references (`exact_reference`), CPU runs
(`software_simulation`) and joules (`calibrated_projection`). No hardware
claim.
