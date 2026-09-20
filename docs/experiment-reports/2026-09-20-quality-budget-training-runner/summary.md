# M4G training-runner component

The shared engine now performs exactly five law-specific projected updates at
K=1,2,4,8,16,30 or equilibrium. It authenticates the original M1 archives,
constructs the 21 production requests, retains every checkpoint and selects
update five. A complete ordered training bank must replay all 21 fits before
supplying evaluation parameters. The frozen M4G protocol is unchanged.

The component preflight completed **21 three-site fixture fits and 105 fixture
updates**, with independent realized-draw replay at every evolving checkpoint.
Maximum occupancy and gradient replay errors were both zero. All 210 diagnostic
roles are distinct from the 213 production roles and 42 earlier law-validation
roles. This exercises the shared engine; it does not execute the full program.

**Zero production fits and zero held-out study cells executed. Full M4G
readiness remains false.** Sampled fixture evidence is `software_simulation`;
local endpoint laws are `exact_reference`. No task-quality preservation,
inference savings, convergence, or hardware conclusion follows.

## Evidence contract

- Every request binds the frozen protocol, source lineage, archived initial
  parameters, target, schedule, training law, and all ten assigned role seeds.
- Every step binds its parameters and tables, independent occupancy counts,
  realized reward, shared-gradient sums and sum-squares, and projected update.
- Replay compares gradient moments at absolute tolerance 1e-10 and relative
  tolerance 1e-12, binds their exact stored values in source/result digests,
  and derives subsequent checkpoints from those stored moments.
- Counts, rewards, roles, tables, updates, checkpoint selection and request
  identity are reconstructed exactly. Supplied results are never cached.
- A separate parent-partition/searchsorted replay reconstructs sampled draws
  on the fixture at every updated parameter matrix.
- Reporting validates persisted evidence before completion is written.

Independent implementation and scientific reviews found no blocking issues.
The two minor suggestions were implemented and re-reviewed: malformed request
objects now raise a consistent validation error, and tests distinguish exact
moment-digest binding from numerical rejection of fully rehashed forgeries.

See [report](report.md), [complete evidence](preflight.json),
[completion](completion.json), [runtime provenance](provenance.json), and
[verification](verification.md).

## Next task

Implement the complete 60-cell held-out evaluator, paired joined terminal
moments, exact local metrics, acceptance decisions, and full-study persisted
replay. Complete the integrated preflight and independent review before running
any production fits. Then execute the unchanged 21-fit/60-cell design in full,
including all cells even if earlier quality screens fail. Record the M4G
scientific decision before M5.
