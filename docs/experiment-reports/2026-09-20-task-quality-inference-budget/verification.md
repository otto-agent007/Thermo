# M4G verification

## Frozen implementation and independent reviews

The implementation digest is
`sha256:0eaf7344c1c27a7e5c3964610f5eee2fd38d629ef0c0b22655d817bb78e952f4`.
The protocol is unchanged. Before any production fitting, the complete integrated
preflight passed and separate statistical and implementation reviewers approved
its exact result digest
`sha256:7bf1e9fad9bcaca725fd27e8e8d18bbce7dfdc4dc2c33a680c524f22691a8863`.

The preflight includes the decision, seven-law training and five-update runner
components, then exercises all 60 fixture evaluation cells and 18 pairs. An
independent parent-partition sampler exactly reproduces terminal counts and
joined moments. Enumerated killed survival agrees within `5.55e-17`. Its
production fit/cell counts are zero. All 468 production and diagnostic roles
are distinct.

Production retained all 21 fits, 105 updates, 60 cells and 18 paired comparisons.
The generation command completed its full persisted numerical replay and exited
zero. The independent result reviews then approved the exact study result
`sha256:a9e7bf3ea2c5a37a28493a4df8287ad6d526b509994dce871db33f957533d423`.

Independent statistical checks covered 1,560 binomial intervals through a
separate exact-binomial route, all 30,000 survival endpoints through 25-by-25
transition matrices, all-group local metrics, and paired U-statistics/jackknife
SEs. Maximum discrepancy was below `5e-13`. Independent implementation checks
also sampled the complete seed-1 K4 finite/equilibrium pair with exactly matching
counts, histograms, joined moments and paired leakage. These numerical replay
tolerances never relax the literal scientific acceptance thresholds.

## Repository gates

All **32 required commands returned zero**: three exhaustive test partitions,
23 experiment gates, frozen dependency sync/lock checks, Ruff format/lint, build,
and wheel/sdist packaging checks. Actual commands, return codes and observed
seconds are retained in [gates.json](gates.json). The canonical gate plan binds
test partitions, configs, seeds and packaging code. Ruff checks were repeated
after the plotting script was added; their final invocations are recorded.

The independent complete collection contains **2,052 tests**, partitioned as:

| Partition | Collected tests | Process exit |
| --- | ---: | ---: |
| Unit and upstream regressions | 1,603 | 0 |
| Integration A | 193 | 0 |
| Integration B | 256 | 0 |

No slow-test exclusion or weakened gate was used. Retained stdout captures are
incomplete for some long-running partitions and lack their final pytest summary.
The counts above come from the separate complete collection; successful exits
come from the subprocess driver. See [collection](verification-logs/collection.log)
and the retained [unit](verification-logs/tests-unit-upstream.log),
[integration A](verification-logs/tests-integration-a.log) and
[integration B](verification-logs/tests-integration-b.log) captures.

Focused release/fixture tests also returned zero: 23 tests before the final
canonical-command addition, then all six release-contract tests after that
addition. The exhaustive suite subsequently exercised the final source/tests.
CI now includes a dedicated complete fixture preflight job and retains its
artifacts. Hosted CI status is reported separately from these local gates.

## Source and runtime identity

Production ran from the working tree based on
`daa76ed4829185c7baedd88a85863761dc112b26`; runtime provenance correctly records
the dirty state. The implementation digest was frozen and reviewed before fitting
and remained unchanged through the run and final release checks.

Published implementation commit
`d636601174977564400b65cb8fc8f5ef4240e641` has complete tree
`227d0898d3739c87e47bc6c615ca8baba5bf4b45`, exactly matching locally verified
implementation commit `182c351363284b76ce24be8ea5d45d26d7f69193`. The scientific
records retain their original observed runtime provenance.

The release finalizer performs another complete reconstruction from the saved
production evidence. Its first invocation did not leave a completion marker;
the execution service subsequently reported an unknown process ID, and its
retained log contains only the start line. Finalization was invoked again from
the unchanged persisted evidence and original roles. This is a validation retry,
not a new study generation or statistical replication. The original generation
and its persisted replay had already completed successfully. No training
restart, outcome-driven selection or additional production cell was added to
the scientific result. The retry completed successfully in 767.554 seconds and wrote the final
completion marker after validating the recorded reviews and all gates.

[Reproduction](reproduction.md) specifies timing inclusions/exclusions and
replay semantics. `execution.json` remains the historical pending-release marker;
`completion.json` binds that execution to the successful reviews and gate record.
