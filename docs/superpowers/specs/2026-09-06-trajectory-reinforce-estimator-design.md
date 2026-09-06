# Exact Trajectory-Level REINFORCE Estimator Contract

## Decision

Implement one checked, exact-enumeration validation experiment for the
trajectory-level energy-based REINFORCE estimator. The experiment composes two
PAsymSwap thermodynamic kernels over three visible sites, reuses one shared
nine-parameter kernel at both occurrences, and compares three independently
formed gradients of a terminal-occupancy objective:

1. the exact trajectory score gradient;
2. the exact expectation of the one-reference REINFORCE estimator; and
3. a central finite-difference gradient of the enumerated scalar objective.

The first two must agree at binary64 roundoff scale, and the finite-difference
cross-check must agree at a separately declared numerical tolerance. A seeded
Monte Carlo execution of the same estimator supplies non-gating stochastic
evidence through the ordinary runner and report path.

This is an estimator-contract slice. It does not optimize parameters, claim
trajectory improvement, execute the 25-site 500-occurrence fixture, use finite
Gibbs thermalization, or reproduce the paper's random-walk result.

## Position in the Repository Study

The completed PAsymSwap sequence currently establishes:

- independently compiled local kernels under uniform input context;
- exact target-context recompilation;
- one-pass mean-field model-context recompilation; and
- strict publication contracts for the local exact and sampled evidence.

None of those stages differentiates a composed stochastic program. This slice
introduces that missing mathematical primitive while keeping every state space
small enough to enumerate. It intentionally leaves the roadmap item
"trajectory-level REINFORCE refinement" unchecked. The roadmap should instead
record a checked subordinate estimator-validation milestone before the still
unchecked refinement and full finite-Gibbs-horizon composed-program studies.

The intended follow-on sequence is:

1. exact estimator contract (this slice);
2. bounded seeded optimizer with training and held-out readouts;
3. full 25-site equilibrium-conditional rollout and refinement; and
4. finite-`K` composed-program evaluation with an explicitly appropriate
   gradient or a clearly bounded approximation claim.

## Research Basis and Thermo-Specific Boundary

The paper defines the trajectory objective as an expected scalar reward under
the full compiled trajectory law. For an energy-based factor, its gradient is
the reward times the factor sufficient statistics minus their conditional
model mean. The paper's linear-cost estimator replaces that conditional mean
with one independent reference sample drawn from the same parent input. The
reference is not propagated. Hidden spins participate in the sufficient
statistics even though the terminal reward does not read them.

The primary source is [Thermalizing Stochastic
Programs](https://arxiv.org/abs/2608.01615v2), especially Section III.4 and
Appendix I. The implementation and report identify that version explicitly.

For the biased random walk, the paper specializes the reward to the
linearization of squared terminal-occupancy error. If

```text
m(phi) = E_phi[f(z_L)]
D(phi) = ||m(phi) - t||^2
```

then the effective scalar reward at the current parameters is

```text
F_eff(z_L) = 2 * (m(phi) - t)^T f(z_L),
```

where the coefficient `m(phi)` is held fixed while applying the score
identity.

This validation fixture computes `m(phi)` by exact enumeration. It must not
replace it with a mean estimated from the same sampled trajectories. Multiplying
a same-batch sample mean by a same-batch score mean generally introduces a
finite-batch cross term. A production trainer must make its own explicit
choice, such as an exact coefficient where feasible or an independently
estimated/cross-fitted coefficient, and validate that choice separately.

Likewise, the paper's unbiased score derivation targets samples from the
conditional Boltzmann law. A `K`-sweep Gibbs draw is instead a draw from a
parameter-dependent finite-horizon transition law. Using an equilibrium score
with that draw is not automatically an unbiased gradient of either law. This
slice therefore samples the exactly enumerated conditional categorically and
makes no finite-thermalization gradient claim.

## Canonical Three-Site Microcircuit

Use visible site order `(site_0, site_1, site_2)` and the existing two-bit word
order `(00, 01, 10, 11)`. The initial visible state is exactly `(1, 0, 0)`.
Apply two sequential factors:

1. occurrence 0 updates `(site_0, site_1)` and leaves `site_2` unchanged;
2. occurrence 1 updates `(site_1, site_2)` and leaves `site_0` unchanged.

Both occurrences use the same target PAsymSwap channel and the same model
parameter vector. The target channel is the checked paper-fixture channel for
the canonical oriented edge `((0, 0), (1, 0))`, rebuilt with the existing
`paper_logit`, `hop_probability`, and `build_pasym_swap_conditional` functions.
Its oriented edge, probabilities, conditional table, and target hash are
included in the experiment identity. Applying one channel at two overlapping
locations makes the fixture a genuine ancestor chain while keeping the shared
target identity needed to exercise tied-parameter accumulation.

The model at each occurrence is the existing five-spin `K_(3,2)` kernel in role
order `(input_0, input_1, hidden_0, output_0, output_1)` and the repository's
canonical nine-parameter order. The checked fixture declares one asymmetric,
nonzero, strictly interior float64 parameter vector with every absolute value
below the cap `2.0`; it does not run SciPy or import a previously observed
optimizer endpoint. This makes two-sided finite differences valid and prevents
an upstream compiler change from changing the estimator oracle.

In canonical parameter order, the vector is:

```text
(0.25, -0.35, 0.20, 0.45, -0.30, -0.40, 0.25, 0.30, -0.20)
```

The same parameter object is used at both occurrences. The gradient with
respect to the shared vector is the ordered sum of the two factor-occurrence
contributions. Persist both contributions and the shared sum so an
implementation cannot accidentally overwrite one occurrence or treat the two
uses as independent trainable kernels.

## Exact Joint Conditional and Sufficient Statistics

For a clamped two-bit input `x`, enumerate all eight `(w, y)` pairs in canonical
hidden-major/output-major order. Their probability is

```text
p_phi(w, y | x) = exp(-beta * E_phi(x, w, y)) / Z_phi(x),
```

with checked `beta = 1.0`. Use the same stable log-normalization and spin
mapping as the existing exact thermodynamic-kernel module. The sampled object
is the joint `(w, y)`, not only the marginalized output.

For this linear Ising energy, define the nine-vector

```text
Phi_phi(x, w, y) = -d E_phi(x, w, y) / d phi.
```

At `beta = 1`, this is the existing vector of spin products in canonical
parameter order. The implementation must define one public, checked sufficient-
statistics function rather than duplicating the feature ordering in exact and
sampled code paths. Inputs, spins, shapes, beta, probabilities, and parameter
vectors reject booleans, non-finite values, invalid support, and non-canonical
dimensions; nothing is clipped or silently normalized.

For general test-only `beta`, the factor score is `beta` times the centered
`Phi` vector. All exact and sampled gradient paths must apply that factor even
though it evaluates to one in the checked publication fixture.

## Terminal Objective

Compute the target terminal occupancy `t` by composing the two exact target
PAsymSwap conditionals from the initial visible state. Compute the model
terminal occupancy `m(phi)` by composing the two exact joint model
conditionals, propagating only their visible outputs and marginalizing both
hidden spins. Use binary64 arithmetic and site features

```text
f(z_2) = (z_2[0], z_2[1], z_2[2]).
```

The scalar objective is

```text
D(phi) = sum_i (m_i(phi) - t_i)^2.
```

Record terminal target/model occupancies, total target/model occupancy,
particle-number leakage, and the scalar objective. Only the objective defines
the gradient. Leakage is a required diagnostic and is not silently projected
away by conditioning onto the one-particle sector.

## Three Gradient Oracles

### Exact trajectory score gradient

Enumerate the 64 main augmented trajectories: eight `(w, y)` outcomes at each
of two factors. For each occurrence `ell`, sum

```text
F_eff(z_2) * (
    Phi(x_ell, w_ell, y_ell)
    - E[Phi(x_ell, w', y') | x_ell]
)
```

under the exact main-trajectory probability. Retain the two occurrence
contributions and add them with `math.fsum` component by component to obtain
the shared-parameter gradient.

### Exact expected one-reference estimator

Independently enumerate one reference `(w'_ell, y'_ell)` from the same clamped
parent `x_ell` as each main occurrence. A reference output must never replace a
visible site or become an input to the next occurrence. The single augmented
sample estimator is

```text
g_hat_ell = F_eff(z_2) * (
    Phi(x_ell, w_ell, y_ell)
    - Phi(x_ell, w'_ell, y'_ell)
).
```

The complete joint enumeration has `8^4 = 4096` main/reference combinations.
Its exact expectation is reduced in canonical order with `math.fsum`. The two
occurrence vectors are again summed to form the shared-parameter gradient.

The exact release gate requires each occurrence contribution and the shared
sum to match the corresponding trajectory-score value within `1e-12` absolute
and relative tolerance.

### Central finite-difference gradient

Recompute the scalar objective at `phi_j + h` and `phi_j - h` for each of the
nine shared parameters, using a checked `h = 1e-6` and no score-function code.
The finite-difference vector is

```text
(D(phi + h e_j) - D(phi - h e_j)) / (2h).
```

The parameter vector must remain strictly inside the cap under both
perturbations. The finite-difference implementation may reuse probability and
objective primitives, but it must not call the score-gradient or reference-
estimator functions. Its configured comparison tolerance is a numerical
validation tolerance, not an exact-identity tolerance. The checked maximum
absolute tolerance is `1e-7`; the design fixture has been confirmed to remain
well inside the parameter cap under both perturbations and to produce a
nonzero objective and nonzero gradient components.

## Seeded Monte Carlo Cross-Check

The checked run uses an exact categorical sampler over each eight-state joint
conditional. One sample is one augmented trajectory consisting of:

- two propagated main factor draws; and
- one independent, same-parent, non-propagated reference draw at each factor.

The checked batch size is 65,536 augmented trajectories for each release seed
`0,1,2`. Use independently derived RNG streams for the two main draws and two
reference draws; never reuse a uniform variate across main/reference roles or
occurrences. The sampler must be CPU-only, credential-free, and independent of
THRML so the experiment tests the estimator rather than a finite-Gibbs
implementation.

For every occurrence and the shared sum, retain bounded sufficient aggregates:
sample count, component-wise sum, and component-wise sum of squares. Derive the
sample mean, unbiased component variance, standard error, absolute error from
the exact expected estimator, and maximum absolute error from those aggregates.
Do not persist per-trajectory samples or RNG state.

The stochastic comparison is deliberately non-gating. Its purpose is to show
the estimator's scale and variance and to detect gross sampling mistakes during
development. Exact enumeration, not a confidence interval that can fail by
chance, is the correctness gate. Across-seed intervals may be reported only for
a predeclared scalar such as maximum absolute shared-gradient error, with the
independently seeded run as the replication unit.

## Strict Result and Identity Contract

Add a dedicated strict frozen result family rather than extending the local
context-matching schemas. The deterministic reference contains:

- an identity version and experiment request hash;
- site, word, role, hidden/output enumeration, and parameter order;
- initial state and the two ordered occurrence locations;
- target edge, target probabilities, target conditional, and target hash;
- shared model parameters, beta, cap, finite-difference step, and tolerances;
- exact target/model terminal occupancy and leakage diagnostics;
- scalar objective and effective-reward coefficient;
- both occurrence gradients and shared gradient from all three oracles;
- component-wise comparison errors and exact acceptance; and
- a deterministic-result digest over all preceding scientific fields.

Each seeded sample result contains:

- the deterministic-result digest and seed;
- checked batch size and exact sample definition;
- occurrence and shared sufficient aggregates;
- recomputed means, variances, standard errors, and errors; and
- a per-seed digest that includes the source aggregates.

Validators parse strictly, require exact vector lengths and canonical order,
recompute all derived scalar/vector fields from the retained sources, verify
that the shared vectors are the component-wise sums of the occurrence vectors,
recompute both digests, and reject a persisted pass flag unless the exact gates
actually pass. NaN, infinity, booleans in numeric fields, negative counts,
incorrect sample totals, stale errors, altered hashes, duplicate occurrences,
and unknown fields fail validation.

The full structured result remains in each run record. Only the declared
sampled scalar is eligible for ordinary cross-seed aggregation; vector
gradients, exact deterministic content, objective values, and timings receive
explicit aggregation-omission reasons.

## Evidence Semantics and Reporting

The exact target composition, exact model enumeration, exact trajectory-score
gradient, and exact reference-estimator expectation are `exact_reference`
evidence for the declared three-site fixture and frozen software-defined
kernel. The finite-difference comparison is a deterministic numerical
validation, not an exact theorem. The categorical Monte Carlo execution and
all timing are `software_simulation`.

The report must state:

- the exact two-occurrence/three-site scope;
- that one shared kernel is used twice and occurrence gradients are summed;
- that each reference is independent, conditioned on the main occurrence's
  parent, and never propagated;
- the target/model terminal occupancies and leakage;
- the three shared gradient vectors and their maximum discrepancies;
- sampled estimate/standard-error summaries for each seed;
- that exact `m(phi)`, rather than a same-batch estimate, defines the effective
  reward coefficient;
- that no parameters were updated and no improvement was tested; and
- that the run used exact categorical software sampling, not THRML, finite
  Gibbs dynamics, official Thermalizers, hosted simulation, or physical Z1/TSU
  hardware.

Reports render only validated typed values and must remain inert under altered
persisted strings. Rendering may not rerun a sampler or optimizer. Pure
deterministic reference validation may rebuild the tiny checked fixture from
the parsed request, but its result must match the persisted deterministic
digest before any report is published.

## Configuration and Integration

Add one checked configuration under `configs/experiments/` with experiment ID
`numpy.trajectory_reinforce_pasym_swap_estimator.v1` and a new backend ID
`numpy_exact_categorical`. The backend emits `software_simulation` run evidence
and permits nested `exact_reference` metrics, matching the existing THRML
policy for exact comparisons inside a sampled run. Its hashed request declares
every fixture, objective, parameter, numeric, sampling, and tolerance choice.
Seed, runtime provenance, timings, and observed samples remain outside the
non-seed scientific request hash.

Use a dedicated experiment factory, backend, result module, and reporting
module. The runner dispatch and aggregate allowlist are explicit. The ordinary
CLI command accepts `--seeds 0,1,2`, persists one run record per seed, publishes
the report before the aggregate, and produces a truthful incomplete aggregate
if any seed fails.

The configuration must be included in wheel and sdist membership checks. The
default smoke path remains unchanged. CI should add the three-seed command only
after its measured CPU runtime fits comfortably inside the existing timeout;
otherwise the deterministic exact contract runs in CI and the larger sampled
cross-check remains an explicit release gate documented in `AGENTS.md`.

## Tests

Write tests before implementation for:

- the hand-checked joint eight-state conditional and sufficient-statistics
  ordering;
- exact normalization and hidden marginalization for all four inputs;
- canonical two-occurrence state propagation and untouched-site behavior;
- target terminal occupancy from a small direct calculation;
- exact model objective from independent brute-force trajectory enumeration;
- occurrence-local score gradients and their shared-parameter sum;
- the 4096-state one-reference expectation identity;
- a failure when a reference is drawn from a different parent or propagated;
- central finite differences that cannot call score-gradient code;
- beta handling, including a test that would fail if the score omits beta when
  a non-unit test-only beta is used;
- parameter-sharing accumulation versus an intentionally untied comparison;
- independent RNG streams, fixed-seed determinism, and no raw sample traces;
- sufficient-aggregate algebra and deep reload rejection after representative
  tampering with counts, sums, sums of squares, derived errors, hashes, vector
  ordering, pass flags, or tolerances;
- report generation from reloaded records without sampling or optimization;
- evidence labels, non-gating Monte Carlo language, and all excluded claims;
- per-seed failure behavior and atomic report/aggregate publication; and
- wheel/sdist inclusion plus regression coverage for every prior PAsymSwap
  experiment.

## Acceptance and Claim Boundary

The slice is merge-ready only when:

1. exact score and exact one-reference occurrence gradients agree within the
   declared `1e-12` tolerance;
2. their shared-parameter sums agree within the same tolerance;
3. the independent central finite-difference vector agrees within its declared
   numerical tolerance;
4. every strict identity, schema, digest, reload, reporting, packaging, Ruff,
   and test gate passes; and
5. the checked seeded command completes and reports its stochastic uncertainty
   without using it to manufacture a correctness claim.

Passing supports only this statement: for the declared bounded microcircuit,
Thermo's exact one-reference implementation has the same expected gradient as
the exact trajectory score and agrees with a numerical derivative of the
terminal-occupancy objective. It does not support a claim that REINFORCE
training improves PAsymSwap, that the 25-site program has been executed, that
finite Gibbs samples produce an unbiased estimator, or that any physical
hardware was evaluated.
