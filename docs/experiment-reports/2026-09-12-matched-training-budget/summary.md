# Matched training budgets: no demonstrated finite-K4 advantage

September 12, 2026. This is a NumPy software simulation, with exact local
probability tables as references. No physical hardware was used.

Five finite-K4 updates did not demonstrate lower K4 occupancy loss than five
equilibrium-directed updates from the same archived initial parameters. All
three primary point estimates slightly favor equilibrium training, and all
three approximate paired 95% intervals include zero. The descriptive mean
finite-minus-equilibrium loss difference is +0.000224159401. This is neither an
equivalence result nor evidence that equilibrium training is generally superior.

Both arms lower the point estimate of occupancy loss versus initialization in
all three seeds. The descriptive intervals exclude zero for two finite-arm
comparisons and all three equilibrium-arm comparisons. Those correlated
secondary comparisons do not establish a general training advantage.

Particle leakage remains approximately 0.767–0.772 in both trained arms: roughly
77% of final trajectories fail to contain exactly one particle. Improving
marginal occupancy loss is therefore insufficient to establish faithful
one-particle program execution. Projection clips 22–51 of 333 parameters per
update; this is a diagnostic, not proof that the cap causes the weak improvement.

See [the generated report](report.md) for every contrast, interval, leakage
measurement, projection count, work count, identity and simulator timing.
The [predeclared protocol](../../experiments/matched-training-budget.md) fixes
seeds, initial parameters, learning rate, five updates, and the final evaluation
stream. Three independent seeds are the replication units. The approximate
intervals are conditional, pointwise and descriptive, with known coverage
limitations; no across-seed confidence interval is asserted.

## What to investigate next

First ask whether the present parameterization and objective can sustain the
one-particle invariant through 500 operations at K4. A small, separately
predeclared diagnostic should locate where leakage enters and compare occupancy
loss with conservation-sensitive measurements. It should distinguish a local
representational limitation from cumulative propagation error before selecting
new training objectives or architectures. This study does not identify the
mechanism, and it does not justify increasing the number of updates by itself.

Any later claim about reduced inference sampling needs its own sample-budget
study at fixed task quality. Equal logical training draws here do not imply
equal Gibbs sweeps, wall-clock time, energy, or hardware cost. The equilibrium
arm is an oracle with no assigned finite-sweep cost.

## Reproducibility and archive boundary

The three seed JSON files retain the complete original M1 source record, every
new update, independent random roles, bounded training moments, and final
paired terminal moments. JSON whitespace may be normalized; canonical values
and their hashes are preserved. Completion is emitted only after reload and
full numerical replay succeeds and the report has been written.

Historical source records are authenticated against full canonical hashes of
the sources embedded in M4 at `e6ce897`. Their unused M1 update and jackknife
statistics are not re-executed. The importer independently checks the request,
initial parameter bounds/digest, fixture schedule, and target reconstruction.
Cross-CPU BLAS rounding in target reconstruction has absolute tolerance 1e-14
and zero relative tolerance; all new computation still uses the exact archived
target and reference. Any archived payload mutation fails its exact hash check.
Generic M1 numerical validators remain unchanged.

The persisted runs were generated from clean local commit
`795588ed677f366647d92df92c79af3ec2622295`, tree
`f710e93cd62cdfedcddd46c3cd6fb5896ef24da8`, published with the identical tree as
[commit 37b06b9](https://github.com/otto-agent007/Thermo/commit/37b06b9db67f09cf5dae163511661526ac8d20ee).
The subsequent archive reconstruction rounding fix changes no sampled inputs,
training computation, or result values. All three complete numerical comparison
payloads also match the earlier local run before the archive-import correction.
Simulator timings are host-specific and exclude replay as described in the
report; they are not physical latency or arm-to-arm cost comparisons.

## Verification

All 14 pre-existing local experiment gates passed, including smoke, the ten
checked configurations, and M2–M4. The unit/upstream suite passed 1,319 tests
at the preceding implementation snapshot. The archive-import correction passed
31 focused unit/integration tests, and the final rounding correction passed all
eight archive tests under the Haswell BLAS implementation. Formatting, lint,
lock consistency, wheel/sdist builds, and byte-for-byte inclusion of the ten
experiment configurations and three new modules passed. Independent code review
found no remaining blockers. Full PR CI is tracked on
[PR #25](https://github.com/otto-agent007/Thermo/pull/25).
