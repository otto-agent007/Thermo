# Model-context PAsymSwap publication integration

## Purpose and scope

Publish the checked `thrml.model_context_pasym_swap.v1` study through Thermo's
ordinary configuration, runner, persistence, aggregation, and Markdown-report
path.  The study already has a deterministic 37-profile compiler lineage,
exact three-way evidence, and a 37-kernel THRML empirical cross-check.  This
slice makes that evidence a reproducible, reload-validated experiment artifact.

This slice does not change the PAsymSwap fixture, optimizer schedules,
model-context derivation, sampler transition kernel, chain count, warmup
schedule, tolerances, or claims about physical hardware.  It is intentionally
an integration and evidence-publication slice.

## Architecture

The authoritative TOML config is already a strictly registered
`thrml_local` software-simulation experiment.  `runner._backend` now selects
`ThrmlModelContextPAsymSwapBackend`; the backend implements the standard
`run`/`execute` protocol and produces one `RunRecord` per seed.

The backend remains the sole owner of reconstruction.  One execution:

1. validates the exact checked request;
2. rebuilds the target-context ancestors and the 37 model-context artifacts;
3. derives exact profile/schedule evidence;
4. executes the keyed THRML cross-check for all four inputs of all 37 kernels;
5. builds a typed publication summary from those two evidence families;
6. records synchronized THRML execution timing and runtime provenance; and
7. persists only the normal `RunRecord` through the existing runner.

Compilation/optimizer work remains excluded from the THRML execution interval.
The interval begins after a single untimed, synchronized warm launch and ends
after every queued sampling launch is synchronized.  Its method text must say
that it covers 148 keyed 4096-chain, K=30 THRML launches and excludes
configuration loading, reconstruction, exact evaluation, and persistence.

## Persisted evidence contract

Add a strict `ModelContextPAsymSwapSummary` result model alongside the current
model-context result models.  It contains, in canonical 37-profile order:

- the checked request hash and the model-trace hash;
- all `ModelContextProfileResult` values;
- all empirical profile samples, each with target/profile/artifact hashes,
  the exact K=30 reference conditional, integer `4 x 4` counts, and the
  derived `SampledK30Evaluation`;
- schedule acceptance and empirical acceptance;
- the configured THRML K=30 TV tolerance and the maximum observed empirical
  residual; and
- a deterministic-result digest over the request, exact evidence, artifact
  identities, and exact K=30 references; plus a per-seed summary digest that
  also covers integer counts and sampled results.

The results module remains backend-independent.  Its builder accepts exact
profile results, schedule acceptance, and normalized sample-result tuples
rather than importing backend dataclasses.  The validator parses JSON
strictly, requires exactly 37 unique target/profile/artifact triples in the
same order as the exact evidence, verifies each exact K=30 reference against
the persisted equilibrium residual, recomputes every empirical conditional
and TV residual from integer counts, recomputes the maximum residual and both
acceptance flags, and verifies the digest.  It also reuses the existing exact
profile and schedule validators.  A record cannot claim a passed acceptance
when its recomputed values fail the checked tolerance.

`RunRecord.metrics` stores the full summary as a structured observation named
`model_context_pasym_swap_summary` and separately stores
`maximum_empirical_k30_residual` as the only cross-seed aggregate scalar.
Both use software-simulation evidence.  Exact content inside the summary is
not relabeled as hardware evidence; the existing THRML-local metric policy
already permits exact-reference comparisons within a software-simulation run.
The aggregate layer explicitly allowlists only the residual, retains the
summary per-run, omits timing as cache-sensitive, and rejects incompatible
deterministic-result digests across seeds while allowing per-seed summary
digests to differ.

## Reporting and reload validation

Reporting detects this experiment ID and delegates to a focused
`model_context_pasym_swap_reporting` module.  That module validates the
complete record boundary—request, backend, provenance, timing, metric set,
request hash, structured summary, and scalar residual—before generating
Markdown.  It must never invoke a compiler, sampler, or fixture reconstruction
while rendering.

The dedicated section includes:

- model-trace/request identity and the 37-profile / 500-occurrence scope;
- a per-profile table with multiplicity, model-profile KL improvement,
  exact model-context K=30 residual, empirical K=30 residual, and pass state;
- exact schedule acceptance and empirical maximum-residual acceptance;
- the 4096-chain, 30-sweep, four-input sampling interpretation; and
- clear wording that these are software-simulation cross-checks, not
  independent chains, physical Z1 evidence, or TSU hardware measurements.

The generic report keeps its provenance, scalar aggregate, failure, caveat,
and machine-readable-artifact sections.  Because a seed corresponds to one
full stochastic empirical execution, seeded runs—not profiles, inputs, or
chains—remain the only replication unit for any across-seed interval.

## Failure handling

Invalid config registration, unsupported request fields, stale artifact
hashes, non-canonical profile order, counts that do not sum to 4096, stale
derived conditionals/residuals, altered tolerances, failed acceptance flags,
or digest mismatches raise before persistence or report rendering.  The
runner's existing per-seed failure capture then creates a truthful incomplete
aggregate rather than silently publishing partial evidence.

## Tests and verification

Tests will be added before implementation for:

1. registered config and backend dispatch;
2. a normal one-seed runner execution that writes validated JSON schemas,
   run record, aggregate, and deterministic report;
3. persisted-summary round-trip plus representative tampering of an artifact
   hash, an integer count, a residual, an acceptance flag, and the configured
   tolerance;
4. report rendering from reloaded files and the required evidence-caveat and
   replication-language assertions; and
5. preservation of the existing strict request and sampler tests.

Final verification runs targeted tests first, then the full suite, Ruff format
and lint checks, package build, and `git diff --check` before commit and push.
