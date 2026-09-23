# Audit exact survival gradients of all saved M4G updates

M4G passed marginal-fit thresholds at K30 but failed conservation. This retrospective audit asks whether its occupancy-training updates work against whole-path survival or merely achieve small gains.

## Changes

- Differentiate exact killed-path survival with scaled forward/backward messages under all seven training laws.
- Authenticate the six reviewed M4G release files, analyze all 21 fits / 126 checkpoints / 105 actual projected updates, retain per-occurrence and shared-group diagnostics, and strictly replay saved evidence before reporting.
- Preserve first-exit creation/destruction, displacement norms, cancellation ratios and linearization residuals.
- Archive complete compressed evidence, protocol, findings and independent review. No new fitting, sampling, held-out selection or hardware claim.

## Finding

103/105 updates improve survival and every fit improves from initialization. Predicted and actual signs agree for all 105 updates. K30 survival moves from 3.3211% to 3.3776–3.3803%, still far below 95%. This weakens a general objective-conflict explanation but does not prove that longer training, a different objective or a different sampler will succeed.

## Verification

Twenty focused tests, complete archived numerical replay and independent scientific/code review passed. All 2,072 tests and 23 historical experiment gates passed; all 32 repository commands returned zero. Exact scope is recorded in verification.md. The previous integration timeout is resolved in upstream CI run 35547714193; no validation was relaxed.

The optional independent untied-occurrence derivative regression test is deferred; the reviewer performed that check across every law and parameter with maximum error 2.81e-9.

## Publication

Target main; leave unmerged. The dashboard merged into the former study branch and is intentionally outside this focused audit diff. Publication is awaiting explicit approval after automatic review rejected the attempted push.
