# One-Pass Mean-Field Model-Context PAsymSwap Design

## Decision

Implement one checked sibling experiment, `thrml.model_context_pasym_swap_compilation.v1`. It derives deterministic **mean-field model contexts** from the frozen target-context PAsymSwap artifacts that are rebuilt from the authoritative checked target-context configuration. It then compiles one model-context artifact per target hash.

This is a diagnostic context-matching slice. It is not an exact 25-site composed-program rollout, an iterative fixed-point procedure, trajectory-level REINFORCE, a hardware measurement, or a claim that model-context recompilation improves the target program.

## Why a Mean-Field Trace

The ideal target trace can propagate 25 one-site probabilities exactly because the target PAsymSwap preserves a single particle. A learned five-spin conditional kernel need not preserve that support: after one approximate gate it can assign probability to the `11` word and create correlations. Exact later input contexts would therefore require the full 25-site joint distribution, with up to `2^25` states. That is the separately deferred full composed-program study.

The present slice intentionally retains only one-site occupancies after every gate. Its contexts are deterministic, reproducible diagnostics of the frozen software-derived kernels, but the discarded correlations make the trace `software_simulation` evidence rather than `exact_reference` composed-program evidence.

## Scientific Boundary

The following remain unchanged and independently valid:

- the checked independent uniform-context experiment;
- the checked exact target-context experiment and its target trace;
- exact local conditional, equilibrium, and finite-horizon evaluations of frozen five-spin artifacts;
- seeded THRML cross-checks, which remain `software_simulation`.

The new experiment must state in records, reports, README, roadmap, and experiment documentation that it does **not** evaluate an exact full-joint 25-site rollout, real thermal hardware, official Thermalizers, hosted simulation, model-context fixed-point convergence, or REINFORCE refinement.

## Canonical Mean-Field Model Trace

Start from the checked one-particle occupancy vector: site `(0, 0)` has probability `1.0`; the other 24 canonical `(x, y)` sites have probability `0.0`. Use the existing 500-occurrence fixture order and the frozen target-context artifact paired with each occurrence's target hash.

For an occurrence on oriented edge `(i, j)`, let the current retained endpoint means be `q_i` and `q_j`. Record the model input distribution before updating the edge as:

```text
mu(00) = (1 - q_i) * (1 - q_j)
mu(01) = (1 - q_i) * q_j
mu(10) = q_i * (1 - q_j)
mu(11) = q_i * q_j
```

Use binary64 arithmetic and canonical tuple order `(00, 01, 10, 11)`. Each row must be finite and nonnegative and sum to one within `1e-12`; do not clip, smooth, threshold, or renormalize it.

Let `C[input][output]` be the exact equilibrium conditional table of the frozen target-context artifact for this target hash, in the same word order. Update **only** the two endpoint means:

```text
q_i' = sum_input mu(input) * (C[input][10] + C[input][11])
q_j' = sum_input mu(input) * (C[input][01] + C[input][11])
```

All other site means are retained. Every updated site mean must remain finite and in `[0, 1]` within `1e-12`; out-of-range values are validation failures, not values to repair. The total expected particle count is recorded after each occurrence as a bounded diagnostic. It is not constrained to remain one, because the learned conditional kernel is not assumed to conserve particle number.

Each occurrence record includes the canonical location metadata, target hash, upstream target-context artifact hash, four context weights, per-site means before and after the update, and total expected occupancy before and after. The trace hash includes every deterministic input and every occurrence record, but no optimizer timing, sampler observation, timestamp, or device metadata.

## Pooling

Pool the 500 recorded model contexts by the same target hash used by the fixture. For every hash, sort contributing occurrence indices, take the component-wise equal-occurrence mean with `math.fsum`, compute support exactly as `weight != 0.0`, and hash the explicit identity payload. The canonical fixture still produces 37 sorted profiles with multiplicities `26 × 10`, `9 × 20`, and `2 × 30`.

Model profiles are distinct from target profiles even if numerical context weights happen to agree. Their identity payload must include the model trace hash, the upstream artifact hash(es), `context_source = "mean_field_model_pre_gate"`, `context_reduction = "equal_occurrence_mean_by_target_hash"`, and `zero_support_policy = "exact_unsmoothed"`.

## Artifact Lineage and Compilation

The model-context backend must rebuild, in memory, the exact target trace, target profiles, uniform baselines, and target-context artifacts from the authoritative packaged target-context TOML. It must not accept a prior result directory or seed artifact as scientific input.

For each target hash, compile one `ModelContextCompiledKernelArtifact` against the pooled model-context weights and the paper target conditional. Its identity includes:

- target hash and model-profile hash;
- upstream target-context artifact hash;
- model-context weights;
- topology, role order, parameter order, dtype, beta, cap, and checked compiler settings;
- the selected parameter vector and the explicit four-start values.

It excludes optimizer diagnostics, wall-clock timings, and sampler observations. Its starts are, in this exact order: the paired target-context selected parameters, the checked zero start, the checked positive start, and the checked antithetic-negative start. Selection is minimum exact objective, then lexicographic parameter vector, among endpoints that satisfy the same SciPy-success, finite-observation, cap, and projected-gradient requirements already used by target-context compilation.

No second model trace or second model-context recompilation is permitted in this slice.

## Measurements and Acceptance

Persist three local variants for every profile: uniform baseline, target-context artifact, and model-context artifact. Recompute all persisted scalar fields from bounded source fields without rerunning SciPy or THRML during reload validation.

Required measurements are:

- local target-to-model KL and row-wise TV under the exact target profile;
- local target-to-model KL and row-wise TV under the pooled mean-field model profile;
- occurrence-weighted schedule reductions using `math.fsum(multiplicity * value) / 500` in sorted target-hash order;
- exact equilibrium and `K = 1, 2, 4, 8, 16, 30` finite-horizon residuals for each frozen local artifact;
- model-trace total-expected-occupancy summaries and extrema as non-gating diagnostics;
- target-context-to-model-context distribution-shift summaries, including per-profile TV.

Required gates are:

1. all schema, identity, trace, profile, artifact, and deep reload validations pass;
2. every model-context artifact has a valid selected optimizer endpoint;
3. each model-context artifact's KL under its own pooled model profile is no greater than its paired target-context artifact's KL under that same profile plus `1e-12`;
4. the occurrence-weighted model-profile KL improvement is at least `1e-8`;
5. both target-context and model-context artifacts have exact `K = 30` residual at most `0.05` and `K30 <= K1 + 1e-12`;
6. target-profile KL/TV changes, all-row changes, and any model-trace occupancy drift are persisted as required non-gating evidence.

The target-profile result may improve, match, or degrade. It must never be described as a program-level improvement. A target-profile degradation does not make the model-context diagnostic fail, but it must be prominent in the report.

Run one independent THRML `K = 30` cross-check for each checked seed `0,1,2`, using 4,096 chains per input context on the **model-context artifacts only**. Require target-only empirical residual at most `0.10`; report the cross-seed Student-t interval only for `maximum_empirical_k30_residual`. The mean-field trace, deterministic exact metrics, optimizer timings, and JAX timings have explicit aggregation omission reasons.

## Configuration, Records, and Reporting

Add a checked TOML at `configs/experiments/thrml-model-context-pasym-swap.toml` and a dedicated experiment factory. Its model, fixture schedule, horizons, sampler settings, optimizer settings, cap, initial occupancy, and release seeds are identical to the target-context experiment. The new checked literals are:

```text
experiment_id = "thrml.model_context_pasym_swap_compilation.v1"
context_source = "mean_field_model_pre_gate"
context_reduction = "equal_occurrence_mean_by_target_hash"
model_trace_policy = "one_pass_first_moment_factorization"
upstream_artifact_policy = "rebuild_checked_target_context_artifacts"
warm_start_policy = "paired_target_context_artifact_then_three_fixed_restarts"
```

Use a new result schema, aggregate schema branch, renderer, report heading, deterministic-result projection, and CLI dispatch. Persist enough bounded fields to recompute all gates but never raw 500-step unbounded diagnostic traces outside the specifically hashed trace payload. Publish model trace/profile identities and bounded extrema/summaries in ordinary run JSON.

Rendering remains safe for untrusted persisted strings: known legacy fast-path bytes may be preserved only where already defined, and all other values render inertly.

## Tests and CI

Tests must cover:

- hand-calculated first, disjoint, and overlapping mean-field updates;
- canonical order, 500 occurrences, 37 profiles, checked multiplicities, identity stability, and no mutation of inputs;
- malformed artifact/table/context rejection, including non-finite values and no silent clipping or renormalization;
- a proof that the trace uses the target-context artifact conditional, not the paper target channel;
- exact pooling algebra and occurrence-weighted schedule reduction;
- one-pass lineage: cached/reloaded model-context work must not rerun upstream compilation, while fresh runs rebuild it from checked inputs;
- compiler start order, optimization endpoint checks, artifact identity scope, and deterministic tie-breaking;
- deep reload rejection for altered upstream hashes, trace/profile content, conditionals, metrics, report text hazards, and stale pass flags;
- target-profile degradation being visible and non-gating;
- THRML key separation, model-artifact-only sampling, deterministic output for a seed, and empirical residual gates;
- runner behavior: per-seed failure handling, atomic report-before-aggregate publication, and no fabricated outputs;
- independent and target-context experiment regression snapshots remain unchanged;
- wheel and sdist membership for the new TOML.

CI retains its 20-minute CPU timeout and adds the checked model-context command after the existing independent and target-context commands. The full local gate set remains the project commands in `AGENTS.md`, extended with the new `--seeds 0,1,2` model-context run and archive membership checks for both target-context TOMLs.
