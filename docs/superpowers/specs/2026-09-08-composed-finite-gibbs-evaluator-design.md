# Full Composed PAsymSwap Finite-Gibbs Evaluator Design

## Status

Approved in chat on 2026-09-08. This design is stacked on the one-step
trajectory-refinement work proposed in PR #16. It defines the next Phase 2
slice: a full 25-site, 500-occurrence evaluation harness for the three existing
compiled-kernel families. It does not add 25-site parameter refinement.

## Goal

Measure how local compiled-kernel error and finite thermalization accumulate
through the complete biased-random-walk program. The experiment must execute
the canonical 5 by 5 periodic fixture for all 500 ordered PAsymSwap
occurrences, compare the independent, target-context, and model-context
artifact families, retain equilibrium and six finite Gibbs horizons, and
report whether context matching improves or worsens composed behavior.

Scientific performance is an outcome, not an acceptance assumption. A valid
experiment may report a regression.

## Approved Decisions

- Use vectorized sampled 25-site program trajectories instead of propagating a
  dense `2^25` joint distribution.
- Use precomputed exact local conditional tables for equilibrium and
  `K = 1, 2, 4, 8, 16, 30` complete Gibbs sweeps.
- Do not execute a live THRML chain at every program occurrence in this slice.
- Gate experiment integrity and reconstruction, not which artifact family wins.
- Use common random numbers across artifact families and horizons within each
  release seed.
- Use 32,768 complete program trajectories for each release seed `0, 1, 2`.
- Keep full 25-site trajectory refinement as a later slice built on this
  evaluator.

## Why This Is a Separate Experiment

The existing independent, target-context, and model-context studies are local
compiler or context diagnostics. Extending one of those records with a full
program rollout would mix artifact construction with a materially different
claim about accumulated trajectory behavior. Reading prior result JSON would
also make a checked experiment depend on mutable external files.

The composed evaluator is therefore a new versioned experiment. It rebuilds
the authoritative frozen artifacts from checked configuration inputs, binds
their identities into its own request and result digests, and evaluates them
through a dedicated backend and report section.

Proposed identity:

- experiment ID: `numpy.composed_pasym_swap_finite_gibbs.v1`;
- backend ID: `numpy_exact_categorical`;
- run evidence: `software_simulation`;
- release seeds: `0, 1, 2`.

The existing backend identifier is appropriate because each local transition
table is enumerated exactly and each program transition is sampled locally by
NumPy. A new backend class owns the composed execution without changing the
meaning of the identifier.

## Authoritative Inputs

The experiment configuration fixes:

- the paper source and 5 by 5 periodic fixture;
- coordinate, color, macrostep, layer, and edge ordering;
- the initial state with one particle at canonical site `(0, 0)`;
- all 500 occurrence identities and target-channel hashes;
- the checked independent, target-context, and model-context configuration
  identities;
- artifact-family order:
  `independent`, `target_context`, `model_context`;
- horizon order: `equilibrium`, `1`, `2`, `4`, `8`, `16`, `30`;
- 32,768 trajectories per release seed;
- NumPy `Generator(PCG64)`;
- the common-random-number stream policy;
- checkpoints after occurrence zero and each 50-occurrence macrostep;
- float64 probability calculations and uint8 program states.

The composed request hash includes every scientific field above. The seed is
excluded only from the non-seed identity used to establish cross-seed
compatibility.

## Deterministic Artifact Bundle

### Construction

One pure deterministic construction boundary rebuilds the three artifact
families from their checked TOML inputs. It reuses existing compiler and
context derivation primitives, but it does not invoke prior experiment runners,
load generated result files, or perform THRML cross-check sampling.

Within one runner invocation, the immutable artifact bundle is constructed
once and reused across release seeds. Each per-seed record carries the same
bundle digest.

### Required identities

For each of the 37 target-channel hashes, the bundle contains exactly one
artifact from each family. Every artifact retains:

- target-channel hash;
- parameter vector in canonical nine-parameter order;
- parameter cap and beta;
- optimizer/artifact digest from its originating checked contract;
- exact equilibrium conditional table;
- exact finite-horizon conditional tables for all six horizons.

The occurrence map has exactly 500 entries in canonical order. Every entry
binds its target channel to the corresponding artifact in all three families.
No artifact lookup may fall back to positional coincidence.

### Local finite-Gibbs semantics

For a clamped two-bit input, each finite-horizon table is reconstructed with
the existing checked semantics:

- uniform reset over the eight free hidden/output states;
- one complete sweep updates hidden first and outputs second;
- outputs are marginalized after exactly `K` complete sweeps;
- every operation resets its free-state distribution rather than carrying a
  hidden state between program occurrences.

Equilibrium uses the exact Gibbs marginal for the same frozen parameters.

All conditional rows must be finite, nonnegative, normalized within the
declared exact tolerance, and regenerated from parameters rather than trusted
from serialized derived values.

## Exact Target Reference

The analytic target remains in the one-particle sector. Reuse the checked
target propagation to obtain the exact 25-site occupancy at occurrence zero
and after each 50-occurrence macrostep. The target checkpoints must conserve
unit mass and agree with the canonical 500-occurrence trace already used by the
target-context experiment.

The target checkpoint sequence and its digest are `exact_reference` evidence.
It is not sampled and is identical across release seeds.

## Vectorized Composed Execution

### State layout

For each artifact-family/horizon cell, maintain one uint8 state matrix with
shape `(32768, 25)`. All rows start at the checked one-particle initial state.
The 21 cells are advanced together through the canonical occurrence sequence.

### One occurrence

For occurrence `t` on endpoint indices `(i, j)`:

1. Draw one float64 uniform vector `u_t` of length 32,768 from the seed's single
   checked PCG64 stream.
2. Reuse that exact vector for all three artifact families and all seven
   horizons.
3. For each cell, read the current `(state[i], state[j])` word for every
   trajectory.
4. Resolve the occurrence's target hash to the cell's artifact and select the
   corresponding conditional row.
5. Inverse-CDF sample one replacement word with `u_t` and update only columns
   `i` and `j`.
6. Reject any non-binary state or invalid output index immediately.

This is common-random-number coupling. The marginal distribution of every
family/horizon cell remains correct. Diverged cells may have different current
input words, but they still consume the uniform value assigned to the same
seed, trajectory, and occurrence.

The implementation generates one occurrence vector at a time. It does not
retain a 500 by 32,768 random matrix or any raw trajectory history.

### Stream identity

The stream policy is part of the checked run configuration. Reordering
families or horizons must not alter results because the uniform vector is
generated once outside cell iteration. Tests must prove this invariance.

## Source Counts and Checkpoints

At occurrence zero and after each macrostep, retain integer source counts for
every family/horizon cell:

- number of trajectories occupying each of the 25 sites;
- number with particle count zero;
- number with particle count one;
- number with particle count greater than one;
- sum of particle counts;
- sum of squared particle counts;
- number that have ever left the one-particle sector up to the checkpoint;
- for trajectories currently in the one-particle sector, the 25 location
  counts.

The three particle-sector counts must sum to 32,768. Site occupancy counts and
the particle-count sum must agree exactly. One-particle location counts must
sum to the one-particle-sector count. The ever-left count is monotone across
checkpoints and cannot be smaller than the current out-of-sector count.

Raw program states and per-occurrence trajectories are deliberately not
persisted. All reported scalar and vector metrics are rebuilt from these
bounded integer sources.

## Derived Measurements

For every family, horizon, checkpoint, and seed, derive:

- 25-site occupancy probabilities;
- occupancy half-L1 error versus the exact target checkpoint;
- maximum absolute site-occupancy error;
- particle-number leakage `P(N != 1)`;
- zero-, one-, and multiple-particle probabilities;
- expected particle count;
- signed mass drift `E[N] - 1`;
- particle-count variance;
- probability of having ever left the one-particle sector;
- current one-particle location distribution conditional on `N = 1`, when
  that sector has nonzero count;
- conditional one-particle-location half-L1 error versus the exact target.

If no sampled trajectory remains in the one-particle sector, the conditional
location metric is explicitly unavailable rather than silently set to zero.

The report separates local kernel residuals from composed program errors. A
small local equilibrium or finite-horizon residual must never be presented as
evidence of a small 500-occurrence endpoint error.

## Paired Comparisons

For every horizon and checkpoint, derive these paired metric differences:

- `target_context - independent`;
- `model_context - target_context`;
- `model_context - independent`.

For error and leakage metrics, a negative difference means the first-named
family improved upon the second. Reports state this sign convention beside
every comparison table.

Common random numbers reduce variance but do not turn trajectories into
independent release replications. Seeds `0, 1, 2` remain the cross-run
replication units. Report per-seed values and cross-seed mean, sample standard
deviation, median, extrema, and the repository's existing 95% interval policy.

No paired comparison is an acceptance gate.

## Persistence Contract

Add strict, frozen, finite, extra-forbid result models for:

- deterministic artifact-bundle identity;
- exact target checkpoints;
- source-count checkpoints;
- derived checkpoint metrics;
- one family/horizon trajectory;
- paired comparisons;
- one complete seeded composed-program summary.

Every nested source and derived field is digest-bound. Reload validation must:

- reconstruct the checked request and deterministic artifact bundle;
- verify all artifact, target, occurrence-map, and configuration identities;
- rederive every floating metric from persisted integer counts;
- reconcile all sector, occupancy, mass, and monotonicity identities;
- rederive all paired differences from the family/horizon results;
- reject malformed lowercase SHA-256 fields;
- reject stale standalone metrics and incompatible runtime provenance.

Reload validation does not resample 32,768 trajectories. Reproducibility is
proved by deterministic rerun tests and the checked RNG policy, while the
persisted contract validates the complete bounded source reduction.

## Top-Level Metrics and Aggregation

The run record contains the nested composed summary plus a bounded set of
standalone final-checkpoint scalars for each family/horizon combination:

- occupancy half-L1 error;
- maximum site-occupancy error;
- particle-number leakage;
- signed mass drift;
- ever-left-sector probability.

It also contains the final paired error/leakage differences and a boolean
integrity acceptance metric. Metric names are generated from a fixed declared
family/horizon/measurement product, and persisted validators require the exact
set.

These seed-derived scalars are eligible for aggregation. Deterministic bundle
identities, exact target values, nested summaries, booleans, and execution
timings are omitted from confidence intervals with explicit reasons.

Macrostep trajectories remain in the validated per-run summaries. The report
derives cross-seed macrostep tables directly from compatible run records so
that accumulated drift is visible rather than reduced to one endpoint.

## Reporting

The generated Markdown report includes:

1. source, fixture, artifact-family, horizon, batch, stream, and evidence
   conventions;
2. deterministic identities for the 500-occurrence schedule, 37 target
   channels, three artifact families, and exact target checkpoints;
3. local equilibrium and finite-horizon residual summaries;
4. macrostep occupancy-error trajectories for every family and horizon;
5. macrostep leakage and ever-left-sector trajectories;
6. final 25-site occupancy comparison tables;
7. final particle-sector, mass-drift, and conditional-location tables;
8. paired context-matching differences with the declared sign convention;
9. per-seed values and cross-seed intervals;
10. integrity acceptance and explicit non-gating scientific outcomes;
11. machine-readable artifact links and evidence caveats.

The headline must say whether each context transition improved or worsened the
declared final composed metrics. It must not collapse mixed results into a
single success label.

## Acceptance Contract

Acceptance requires only the following integrity conditions:

1. The authoritative fixture has exactly 25 sites and 500 canonical ordered
   occurrences.
2. Exactly 37 target hashes resolve one artifact in each of the three declared
   families.
3. Every equilibrium and finite-horizon conditional table regenerates from its
   frozen parameters and satisfies probability integrity.
4. The exact target checkpoints regenerate, remain in the one-particle sector,
   and conserve unit mass.
5. Exactly 32,768 trajectories contribute to every family/horizon/checkpoint
   cell for the selected release seed.
6. All sampled states, integer sources, sectors, occupancy counts, mass sums,
   and monotone path counts reconcile.
7. All derived metrics and paired differences regenerate from their sources.
8. Request, artifact, stream, result, runtime-provenance, and digest identities
   pass strict validation.
9. All requested release seeds complete and aggregate compatibly.

Acceptance does not require target-context or model-context artifacts to beat
another family at any horizon or checkpoint. A regression is a valid and
auditable scientific result.

## Evidence Classification

- The analytic target trace and checkpoint occupancies are `exact_reference`.
- Equilibrium and finite-`K` local conditionals are `exact_reference` for the
  declared frozen software-derived parameter vectors and reset semantics.
- Artifact optimization histories remain `software_simulation` evidence from
  their originating compiler contracts.
- All sampled 25-site trajectories, derived composed metrics, paired
  comparisons, and timings are `software_simulation`.

The report and documentation must explicitly state that this experiment is
not:

- live THRML execution of every program occurrence;
- an unbiased finite-Gibbs gradient result;
- 25-site trajectory-level parameter refinement;
- an implementation of unpublished or official Thermalizers;
- hosted Extropic simulation;
- calibrated projection or physical Z1/TSU hardware evidence.

## Failure Handling

Fail before persistence for:

- unchecked or coercively encoded configuration values;
- source configuration, artifact, target, or schedule hash mismatches;
- missing, duplicate, or positionally misbound target hashes;
- incorrect family or horizon order;
- nonfinite, negative, or non-normalized conditional rows;
- invalid endpoint indices, input words, output words, or sampled states;
- incomplete batches or checkpoints;
- integer count overflow or reconciliation failure;
- nonfinite derived metrics;
- stale standalone metrics;
- incompatible cross-seed records;
- malformed digests or runtime provenance.

Do not fail because an artifact family performs worse than another.

## Implementation Boundaries

Prefer focused modules:

- `composed_pasym_swap.py` for artifact-neutral vectorized execution and source
  reductions;
- `composed_pasym_swap_results.py` for strict persistence and derivation;
- `composed_pasym_swap_reporting.py` for record validation and domain report
  sections;
- a dedicated NumPy backend class for authoritative artifact reconstruction,
  timing, and run-record assembly;
- the existing config, runner, aggregate, and report dispatch boundaries for
  versioned integration.

If existing deterministic artifact construction is trapped inside backend
methods, extract only the pure reusable boundary required here. Do not combine
the existing experiment result contracts or introduce a general compiler
framework in this slice.

## Test Strategy

### Pure execution tests

- canonical endpoint updates affect only the two selected sites;
- input and output word ordering matches the checked convention;
- fixed uniforms select exact inverse-CDF outcomes at boundaries;
- identical conditional cells remain trajectory-identical under common random
  numbers;
- family/horizon iteration order does not alter stream results;
- different cells can diverge while retaining valid binary states;
- invalid shapes, probabilities, endpoints, seeds, batch sizes, and states are
  rejected;
- inputs are not mutated.

### Small exact comparisons

Use reduced two- and three-site circuits whose complete distributions can be
enumerated. For rational test conditionals chosen to align with a fixed grid of
uniform values, require exact source counts from the sampler. Separately verify
fixed-seed Monte Carlo behavior against exhaustively enumerated expectations
within predeclared statistical tolerances. These tests validate composition
rather than only individual conditional rows.

### Source and result tests

- checkpoint sector and occupancy reconciliation;
- mass mean/variance reconstruction;
- conditional one-particle location availability rules;
- ever-left-sector monotonicity;
- paired-difference sign and reconstruction;
- strict types, finite values, frozen models, unknown-field rejection;
- digest round trips and targeted tamper rejection for every source family;
- valid worsening outcomes persist and render without failing acceptance.

### Artifact integration tests

- 25 sites, 500 occurrences, 37 target hashes, three complete artifact maps;
- exact target checkpoint identity;
- parameter and conditional regeneration for all families and horizons;
- reduced-batch execution of the complete schedule;
- deterministic rerun for a fixed seed;
- different seed changes sampled results but not deterministic identities;
- common-random-number pairing survives report reload.

### Runner and report tests

- checked config loads, hashes, snapshots, packages, and dispatches correctly;
- a reduced test fixture covers all report sections;
- the release config runs seeds `0,1,2` with 32,768 trajectories each;
- aggregate metrics match per-run sources;
- incomplete and incompatible records fail safely;
- report language distinguishes exact local tables from sampled composed
  evidence and lists every deferred scope item.

### Final verification

- frozen dependency and lock checks;
- Ruff format and lint;
- complete pytest suite;
- all pre-existing checked experiment gates;
- the new three-seed composed-program command;
- wheel and source distribution membership;
- independent read-only code review;
- inspection of run JSON, aggregate JSON, and Markdown report for identities,
  evidence labels, non-gating outcomes, and machine-readable links.

## Documentation and Roadmap

Add the checked command and evidence boundary to `README.md`, `AGENTS.md`, CI,
and the biased-random-walk experiment specification. Mark the full
finite-Gibbs-horizon composed-program comparison complete only after the
release experiment passes all acceptance and packaging gates.

Leave full 25-site trajectory-level parameter refinement explicitly deferred.
The roadmap should split that work into its own unchecked item rather than
implying that the bounded PR #16 update trained the complete program.

## Completion Criterion

The slice is complete when a clean checkout can run one checked command and
produce reconstruction-validated records, an aggregate, and a report that show
how the independent, target-context, and model-context kernels behave across
the complete 25-site, 500-occurrence program at equilibrium and all six finite
Gibbs horizons, including any regressions, with every result carrying the
correct evidence class.
