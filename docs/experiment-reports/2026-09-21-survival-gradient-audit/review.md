# Independent audit review — 2026-09-21

Reviewed scientific range: `eabe49d..1f7ef9d`. No blocking scientific or code issues were found.

- All 20 focused tests passed.
- Scaled forward/backward gradients, killed transitions, shared-parameter reduction and projected displacement diagnostics were checked.
- Independent untied-occurrence finite differences across seven laws and nine parameters agreed within 2.81e-9 (tolerance 1e-7).
- Six source byte pins authenticated; compressed evidence matched its completion digest.
- Confirmed 21 fits, 126 checkpoints, 105 updates, 103 positive changes, all predicted/actual change signs and reported retention statistics.
- Conclusions distinguish attained improvements from convergence, capacity and task-quality claims.

The reviewer did not rerun all 126 checkpoints. The main executor separately completed strict full archived numerical replay and verified the completion digest in this continuation.

Deferred minor: retain the reviewer's independent untied-occurrence derivative check as a regression test. The existing shared-derivative tests, multi-group routing test and long-chain test pass; the independent check was performed during review.

## Continuation decisions

The saved implementation and results already existed at the start of this continuation; they were verified rather than regenerated unnecessarily. The initial interrupted repository-gate run is not claimed as complete.

The earlier integration timeout was resolved upstream by isolating quality-budget integration tests. The final PR36 CI run 35547714193 passed all test shards. No timeout increase or relaxed validation is needed.

PR36 merged into the former M4G study branch, not main. The focused publication branch remains based on main (`eabe49d`) and includes only audit changes. Broader verification runs in a separate checkout which also contains the merged dashboard; its scientific source and tests match the focused branch. The dashboard is excluded from this audit PR.
