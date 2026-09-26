# M5a: exact meta-EBM cap baseline

*Status: **frozen September 26, 2026** after owner review (both readings, five
seeds, both compile methods). The exploratory basis is
[the September 26 research note](../research/2026-09-26-meta-ebm-target-reading.md).
Changes after this point require a dated amendment below.*

## Question

On exactly enumerable d = 12 three-body Ising targets, how does the stationary
bias of a compiled single-site Gibbs chain depend on the coupling cap J_max?
And how does it compare with the provable perturbation floor and with the
Thermalizers paper's plotted floor?

M5a is the first stage of the [topology-aware meta-EBM](topology-aware-meta-ebm.md)
milestone. Like the paper, it compiles kernels over a fully connected spin set,
so the cap is the only substrate constraint. Z1 topology, embedding, finite
kernel thermalization and execution costs are later M5 stages.

## Fixed choices and why

The paper leaves several choices open. The research note gives the evidence
for each decision.

| Choice | Decision |
|---|---|
| Instance | Predeclared PCG64 seeds 0–4. Every instance is reported, and none is selected to match the paper. |
| Energy reading | Both. **B** (primary): ½ and 1/3! applied to unordered sums, matching the paper's reported mixing. **A**: unit weight per unordered term, which gives slowly mixing targets. |
| Kernel family | The paper's single-hidden-layer kernel **plus hidden-spin biases**. Without them a kernel cannot represent a three-body term. |
| Compile method | Both of the paper's descriptions, fully specified: **constructive** (appendix, with clipping) and **variational** (main text, uniform inputs). |
| Cap | J_max ∈ {0.3, 0.5, 0.75, 1, 1.5, 2, 3, 6, 10}, as in the paper, applied to every coupling and field including hidden biases. |
| Sweep | Systematic single-site sweep in site order 0, 1, …, 11. One sweep is one macro-step. |
| Bound | η_sweep/(1 − ρ_D) is the only quantity called a bound. The paper's ε̄/(1 − SLEM) is reported as its comparison value. |

This gives 2 readings × 5 seeds = 10 targets. Each target has 9 caps × 2
compile methods = 18 compiled chains, for 180 chains in all.

## Targets

States are x ∈ {−1, +1}¹², with site n stored as bit n of the state index
(bit set means x_n = +1). For each seed s, one `numpy.random.Generator(PCG64(s))`
draws, in this order:

1. 18 pairs, via `rng.choice(P, 18, replace=False)` from the lexicographic list
   P of the 66 pairs;
2. 20 triples, via `rng.choice(T, 20, replace=False)` from the lexicographic
   list T of the 220 triples;
3. fields W₁ ∈ ℝ¹², pair weights W₂ ∈ ℝ¹⁸ and triple weights W₃ ∈ ℝ²⁰, each
   drawn with `rng.normal(0, 0.6, size)` in that order.

Reading A is E(x) = −Σₙ W₁ₙxₙ − Σ_pairs W₂ xₘxₙ − Σ_triples W₃ xₙxₘxₘ′.
Reading B is the same with W₂/2 and W₃/6. Both readings use the same draws.
The target is π(x) ∝ exp(−E(x)), computed exactly in float64 over all 4,096
states.

## Ideal chain

The exact conditional of site n is p(xₙ = +1 | x) = σ(2θₙ(x)), with θₙ the
half log-odds of E. The ideal sweep P is the dense 4,096 × 4,096 row-stochastic
matrix of the 12 kernels applied in order. For each target, M5a reports:

- **ρ_D**: the Dobrushin coefficient, max over state pairs of the total
  variation between their rows of P, computed exactly in float64;
- **SLEM**: the second-largest eigenvalue modulus of P.

## Compiled kernels

Site n's kernel has inputs x_B, where B is the blanket of sites sharing a pair
or triple with n. It has n_h = (number of triples containing n) hidden spins w
and output y, with energy

E(x_B, w, y) = −(Jᵀx_B + h) y − Σₐ (αₐᵀx_B + bₐ) wₐ − Σₐ βₐ wₐ y.

Its output conditional is evaluated in closed form (softplus features) and
checked against brute-force enumeration of all 2^{n_h} ≤ 256 hidden states.
Every parameter satisfies |p| ≤ J_max.

For a triple (n, m, m′), write u for its reading-adjusted weight and pairs of
n likewise. Each compile method then fixes the parameters as follows.

- **Constructive.** Set J_m to the pair weight of (n, m), and h to the field
  of n. Each triple gets one hidden spin: αₐ = (J_max/2)(e_m + e_m′),
  bₐ = J_max, βₐ = −4u. With this energy the spin adds
  βₐ + 2u·relu(−(x_m + x_m′)) to θ, up to terms exponentially small in J_max.
  The identity x_m x_m′ = 2 relu(−(x_m + x_m′)) + x_m + x_m′ − 1 then adds u
  to J_m and J_m′ and 3u to h (−u from the identity, −βₐ for the constant).
  Finally, every parameter is clipped to [−J_max, J_max]. A brute-force check
  on one triple gives a maximum error of 1.1 × 10⁻⁸ at J_max = 10 for u = 0.3.
  The paper's main-text formula uses the opposite sign for β (its appendix
  relabels β → −β); the energy above is authoritative.
- **Variational.** Minimize the mean KL(target ‖ kernel) over uniform blanket
  inputs. Use L-BFGS-B in float64 with box bounds [−J_max, J_max], exact
  analytic NumPy gradients (checked against centered differences in the tests)
  and at most 5,000 iterations. There are 8 starts: the constructive
  parameters, plus 7 uniform draws in [−min(J_max, 1), min(J_max, 1)] from
  `numpy.random.default_rng([5, reading index, seed, site, cap index])`.
  Keep the lowest final objective, ties broken by start order.

## Measurements

All measurements are `exact_reference` quantities of the declared compiled
models, computed by full enumeration. The variational parameters come from
software optimization; the metrics evaluated on them are exact. No sampling is
performed, so no "sample" is recorded.

| Symbol | Definition |
|---|---|
| ε_n, ε̄ | max over blanket inputs of the Bernoulli total variation between compiled and exact kernels; ε̄ = maxₙ ε_n |
| η_sweep | maxₓ ‖P̃(·\|x) − P(·\|x)‖_TV, from the dense compiled sweep P̃ |
| δ̃_t | ‖q̃_t − q_t‖_TV for t = 0, …, 30, from a common uniform start (δ̃₀ = 0) |
| π̃, bias | the stationary law of P̃ by dense linear solve; bias = ‖π̃ − π‖_TV |
| floor_D | η_sweep/(1 − ρ_D): the provable floor |
| floor_paper | ε̄/(1 − SLEM): the paper's plotted comparison value |
| settle | the first t with \|δ̃_t − δ̃₃₀\| ≤ 0.1 δ̃₃₀ |
| site error | mean and max over n of \|E_π̃ xₙ − E_π xₙ\| |
| KL | mean variational objective per site |

## Integrity requirements (gating)

These checks can only fail through an implementation error:

1. The ideal chain is exact: ‖πP − π‖₁ ≤ 10⁻¹², and every site kernel
   satisfies detailed balance with π to 10⁻¹³.
2. Every compiled kernel's closed-form conditional matches brute-force hidden
   enumeration to 10⁻¹², and every parameter lies within the cap.
3. The stationary solve holds: ‖π̃P̃ − π̃‖₁ ≤ 10⁻¹², with π̃ ≥ 0 summing to 1.
4. The proven bound holds: δ̃_t ≤ η_sweep(1 − ρ_D^t)/(1 − ρ_D) + 10⁻¹² for all
   t ≤ 30, and bias ≤ floor_D + 10⁻¹².
5. The complete record replays from its stored parameters and the target
   seeds before reporting, and `completion.json` is written last. Replay
   rebuilds targets, constructive kernels and every metric, and recomputes the
   selected variational objectives from the stored parameters. It does not
   refit. Replayed floats must match to relative 10⁻⁹ and absolute 10⁻¹⁰;
   integers, strings and structure must match exactly.

## Descriptive comparisons (non-gating)

These are reported for every target and cap, with no pass/fail:

- bias against floor_D and floor_paper, and the ratio of bias to floor_paper
  (the paper reports ≈ 0.6);
- whether bias rises monotonically as the cap tightens, and at which caps it
  does not;
- settle sweeps (the paper reports ≈ 3);
- constructive against variational residuals at each cap;
- reading A against reading B;
- the paper's quoted range: bias 0.46 at J_max = 0.3 down to 0.024 at 10, and
  mean site error ≈ 5 × 10⁻³ at J_max = 10. Our instances differ from the
  paper's, so only orders of magnitude and trends are comparable.

## Execution and records

Run with `uv run python -m thermo_lab.meta_ebm_cap_baseline --output-dir results/meta-ebm-cap-baseline`
and a fresh destination; `--workers` must not change the record. Work runs on
CPU in NumPy/SciPy float64; JAX is not used. Library versions are recorded as
runtime provenance, outside the request identity.

The canonical request binds the seeds, readings, caps, compile budgets,
tolerances and dtype. Outputs are one `study.json.gz` record (targets,
parameters, all metrics above), `summary.md`, runtime provenance and
`completion.json`. Dense matrices and δ̃ traces beyond t = 30 are not
persisted.

Following [CLAUDE.md](../../CLAUDE.md), CI runs a unit test that pins the
request and replays the archived metrics from the stored parameters, without
refitting. The Dobrushin and spectral scans cost about two minutes per target,
so the ordinary unit job replays them for none and the `slow` job replays one
target. The full local replay covers all ten. The full run stays a local gate.
The expected cost is about one to two CPU-hours; the report records the
measured time.

## Not claimed

M5a does not claim:

- a reproduction of the paper's specific instance or numbers;
- Z1-topology, embedding, finite-thermalization, latency or energy results;
- any hardware or official-Thermalizers compatibility evidence;
- optimality of either compile method.

Hidden-spin and parameter counts are logical counts, not device operations.

## Later M5 stages

Later stages extend M5a with:

- **Z1 connectivity:** placement on published offsets, and the per-site
  residual left when a spin must be both a pairwise neighbor and a three-body
  partner of the same site;
- **finite kernel thermalization:** K hidden-spin sweeps in place of exact
  marginalization;
- **cost accounting:** logical-to-physical p-bits, embedding chains, and
  reads and writes, following the parent specification;
- **THRML cross-check:** a sampled cross-check of the compiled kernels in
  THRML, labeled `software_simulation`.
