# Exact finite-sweep gradient contract (M3)

Recorded September 11, 2026 from clean code commit
[`c0579bf`](https://github.com/otto-agent007/Thermo/commit/c0579bff8d0f517cd1b30096b63217e706bef191).
The [complete bounded JSON evidence](2026-09-11-finite-sweep-gradient-contract.json)
includes the immutable request, all reference vectors, artifact digests, and
runtime provenance. The independent autodiff calculation explicitly scopes
float64 on CPU and restores the caller's JAX precision setting; the provenance
records that ambient setting after restoration.

All results are exact_reference in float64 for the existing three-site, two-operation circuit. No parameter update, Monte Carlo estimate, iterative learning, or hardware measurement is performed.

Each occurrence resets uniformly over eight free states, clamps its two inputs, and executes complete hidden-then-output sweeps. Only outputs propagate. The objective is squared terminal population occupancy error against the unchanged target circuit.

The matrix derivative includes every sweep and the zero derivative of the fixed reset. Its endpoint scores are compared with a visible-state chain rule, independent autodiff of Bernoulli spin updates, and central differences through the existing finite-kernel implementation. Both occurrence-local vectors and their shared nine-parameter sum are checked.

| Sweeps | Exact objective | Maximum exact gradient discrepancy | Maximum finite-difference discrepancy | Equilibrium-form score discrepancy | Accepted |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.5381106559202388 | 6.314393452555578e-16 | 1.6494194898797332e-10 | 0.034150509190385855 | True |
| 2 | 0.5258257274934166 | 8.396061623727746e-16 | 2.871180515562344e-10 | 0.00783339457795472 | True |
| 4 | 0.5246667606258139 | 1.817990202823694e-15 | 5.909643330248571e-10 | 0.00014516577100402023 | True |
| 8 | 0.5246570833813492 | 3.4139358007223564e-15 | 1.1019920531651906e-09 | 2.2017356743342376e-08 | True |
| 16 | 0.5246570826850269 | 7.382983113757291e-15 | 2.005574650798536e-09 | 1.5057399771478686e-15 | True |
| 30 | 0.5246570826850256 | 1.441902153231922e-14 | 3.7264195618114115e-09 | 2.2169766022983595e-15 | True |

Exact-reference tolerance is 1e-12. Central differences use step 1e-6 and tolerance 1e-7; all perturbations remain inside [-2, 2]. These tolerances test numerical agreement, not statistical significance.

At K=1 the checks detect both substituting the equilibrium-form sufficient-statistic score and dropping one shared-parameter occurrence. Failure of either negative control rejects publication.

The JSON retains all terminal probabilities and all occurrence-level gradient vectors. Reloading reconstructs the checked references with declared numerical tolerances and verifies exact artifact digests. No raw trajectory or exponentially growing sweep-path enumeration is stored.

Request: `sha256:2f0837fd0fa73a18d1af310d8081f0e8ceff0c236000a98b19657e9c398bafe3`
Result: `sha256:1b5d245ebe62b8997886d485fce5ea2c97ddb102c6fe6e45ebbed65607e5bac2`

M3 establishes a bounded finite-sweep gradient contract. It does not establish an unbiased large-program stochastic gradient estimator, optimization gains, or convergence. M4 must declare its training horizon, step budget, role separation, checkpoint policy, and untouched final evaluation.
