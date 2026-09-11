# Exact finite-sweep gradient contract (M3)

M2 measured transfer of an equilibrium-trained update to finite sweep budgets.
M3 checks the derivative of that finite execution law before it can be used
for training. It extends the existing three-site, two-operation exact fixture:
initial state `(1, 0, 0)`, occurrences `(0, 1)` then `(1, 2)`, the same target
channel, and the same nine shared parameters. No optimizer runs.

## Checked execution and objective

Each occurrence clamps its two visible inputs and independently resets its
hidden/output state uniformly over eight possibilities. One complete sweep
updates the hidden spin and then both output spins. The checked horizons are
`K = 1, 2, 4, 8, 16, 30`. Only the two output bits propagate to the next
occurrence. The objective remains squared terminal population occupancy error
against the exact target circuit. Every result is `exact_reference`.

The fixture retains beta `1.0`, parameter bounds `[-2, 2]`, finite-difference
step `1e-6`, exact-comparison tolerance `1e-12`, and finite-difference tolerance
`1e-7`. All perturbations stay strictly inside the parameter bounds. The
request identity binds the exact parameters, their order, target, geometry,
horizons, reset, sweep order, objective, derivative, and verification policies.

## Derivative and independent checks

For local transition matrix `T`, the endpoint distribution is `p_K = p_0 T^K`.
The reset is parameter-independent, so `d p_0 = 0`. The implementation carries
both the law and its nine derivatives through every complete sweep:

`d p_(k+1) = (d p_k) T + p_k (d T)`.

The transition derivative includes both hidden-conditional and
output-conditional derivatives. Endpoint scores are `d p_K / p_K`, rather
than the equilibrium sufficient-statistic score. Weighted score means must
be zero. Joint hidden/output laws and all derivative components are checked
against the existing kernel's transition matrix and central differences.

The composed objective has four checks:

1. exact expectation of terminal reward times endpoint trajectory scores;
2. a direct chain rule through the two visible output channels;
3. JAX autodiff of independently implemented Bernoulli spin probabilities and
   full eight-state visible transition matrices;
4. central differences through the existing finite-kernel implementation.

Autodiff explicitly uses CPU float64 inside a scoped x64 context and restores
the caller's precision configuration. Its implementation uses neither the
new derivative builder nor the sufficient-statistic score implementation.
Each occurrence is differentiated independently; the summed nine-component
shared gradient must also match a perturbation that changes both occurrences.

At `K=1`, substituting an equilibrium-form score centered under the finite
endpoint law and dropping the second occurrence are negative controls. The
checked circuit must detect both errors above `1e-7`. This demonstrates that
the comparisons can reject plausible incorrect gradients.

## Bounded persistence and reproduction

```bash
uv run thermo-lab check-finite-sweep-gradients \
  --output-dir results/finite-sweep-gradient-contract
```

The destination must be fresh. The command writes `audit.json`, its JSON
Schema, a Markdown report, and finally `completion.json`. The audit retains
all terminal probabilities and all occurrence-level gradient vectors for
the six horizons, with separate request and result digests and runtime
provenance. It stores no raw trajectory or sweep histories and publishes no
timing, power, or energy claim.

Reloading strictly validates types, geometry, ordering, policies, and hashes,
then reconstructs the numerical references from the frozen request. Exact
vectors use `1e-12` reconstruction tolerance and finite-difference vectors
use `1e-7`; derived discrepancies must independently pass their gates. These
numerical tolerances avoid requiring portable bitwise identity across CPU
implementations. Result digests still bind every persisted value exactly.
Reporting revalidates the complete artifact, including objects modified with
`model_copy`. A failed write or validation cannot publish completion.

Local enumeration remains eight states per kernel and eight visible states;
the two endpoint draws have 64 joint paths. Dynamic programming integrates
intermediate sweep histories, so computation does not enumerate exponentially
many paths as K grows. The public local derivative helper caps K at 30.

## Completion and next boundary

Completion requires all six horizons, all occurrence and shared-gradient
comparisons, both negative controls, artifact round trips and adversarial
checks, CPU CLI execution, and a checked report. Numerical agreement is a
correctness gate, not evidence of optimization improvement.

M3 does not train parameters, establish a full-program Monte Carlo estimator,
or demonstrate iterative convergence. M4 must first validate any sampled
finite-sweep estimator, then predeclare its training horizon, fixed step
budget, independent training roles, checkpoint rule, and untouched final
evaluation. M1 and M2 retain their original scientific interpretations.
