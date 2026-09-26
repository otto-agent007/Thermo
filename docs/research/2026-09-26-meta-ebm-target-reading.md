# Reading the Thermalizers meta-EBM target before M5

*Research note, September 26, 2026. Exploratory analysis to inform the M5a protocol; not recorded evidence. Probe script: [`meta_ebm_probe.py`](meta_ebm_probe.py). Source: the arXiv v2 LaTeX of the [Thermalizers paper](https://arxiv.org/abs/2608.01615v2), §"Meta-EBMs" and Appendix "Meta-EBM construction and methods".*

## Summary

The paper's d = 12 three-body meta-EBM demonstration leaves four choices
open, and one written formula cannot do what the text says. Each one changes
what an M5 reproduction would measure.

| # | Issue | Consequence | M5a decision |
|---|---|---|---|
| 1 | The instance is unpublished: only the recipe (18 random pairs, 20 random triples, weights N(0, 0.6²)) is given. | The paper's exact numbers cannot be reproduced; only the method and qualitative claims can. | Predeclared seeds, every instance reported. |
| 2 | The kernel energy in the appendix has no hidden-spin biases. | The written kernel cannot represent any three-body term, at any cap. | Include hidden biases, which Z1's native fields allow. |
| 3 | Energy prefactors are ambiguous: ½ and 1/3! over ordered tuples (unit weight per unordered term) or over unordered sums. | Unit weights give slowly mixing targets; the prefactor reading gives the paper's ρ₀ ≈ 0.28. | Run both readings, the prefactor reading as primary. |
| 4 | "ρ₀ ≈ 0.28" is an exact-diagonalization estimate, but the floor ε̄/(1−ρ₀) needs the Dobrushin coefficient, and a whole-sweep residual in place of the per-site ε̄. | On matching instances, Dobrushin is 0.77–0.93, and one sweep can inject up to 12ε̄, so the plotted "parameter-free bound" is not a guaranteed bound. | Report the paper's value; treat only η_sweep/(1−ρ_D) as a bound. |

A fifth, milder ambiguity: the main text says the kernels are compiled
*variationally* under uniform inputs, while the appendix says the numbers come
from a *constructive recipe with clipping*, whose hidden-spin parameters are
not specified. M5a uses the variational fit.

## 1. Hidden-spin biases are required for three-body terms

The appendix kernel energy is

E(x, w, y) = −(Jᵀx + h_y) y − Σₐ (αₐᵀx) wₐ − Σₐ βₐ wₐ y.

Write fₐ(s) = ½[softplus(−2(s − βₐ)) − softplus(−2(s + βₐ))] for the feature a
hidden spin contributes to the output half log-odds, with s = αₐᵀx. Using
softplus(z) = z + softplus(−z), one finds fₐ(−s) = 2βₐ − fₐ(s). So with no
hidden bias, θ(x) − θ(−x) carries every non-constant term and
½[θ(x) + θ(−x)] = h_y + Σₐ βₐ is constant. The kernel log-odds minus a
constant is an odd function of the inputs. The three-body contribution to a
site's log-odds is x_m x_m′, which is even. It cannot be represented, however
large the couplings.

Numerically (probe, four ±1 inputs): across 200 random bias-free kernels, the
even part of θ varies by at most 7×10⁻¹⁵. The best least-squares fit of x₀x₁
has mean squared error 1.000, the full variance of the target. Adding one
hidden bias bₐ (the term −bₐwₐ) makes the fit exact (MSE 8×10⁻¹²). An explicit
construction uses one hidden spin with α = c(e₀ + e₁), b = 2c and β = c. Its
feature equals (c/2) relu(−(x₀ + x₁)) up to terms exponentially small in c, and
x₀x₁ = 2 relu(−(x₀ + x₁)) + x₀ + x₁ − 1. The maximum error of this
construction is 1.8×10⁻², 1.8×10⁻⁵ and 4×10⁻¹⁰ at c = 2, 5 and 10.

The paper's reported small errors imply its implementation had biases. Z1's
native energy has a field hᵢ on every spin, so hidden biases are legitimate
and should count against the same cap.

## 2. The two energy readings give different targets

The paper writes E(x) = −Σ Wₙxₙ − ½Σ₍ₘ,ₙ₎ Wₘₙxₘxₙ − (1/3!)Σ₍ₙ,ₘ,ₘ′₎ W xₙxₘxₘ′,
with sums over ordered tuples and symmetric couplings. Taken literally, that
is a unit weight per unordered pair and triple (reading A). An implementation
that applies ½ and 1/6 to unordered sums (reading B) has pair and triple
weights 2× and 6× smaller. The main text's phrase "linear coefficients ½Wₘₙ"
fits reading B.

Mixing of the ideal systematic single-site sweep, per seeded instance
(recipe as in the paper, PCG64 seed shown; Dobrushin scan in float32):

| Reading | Seeds | Dobrushin ρ(P) | Second eigenvalue modulus |
|---|---|---|---|
| A (unit weights) | 0–8 | 0.9995–1.0000 | 0.896–0.989 |
| B (½, 1/6) | 0–3 | 0.774–0.929 | 0.283–0.397 |

Only reading B resembles the paper's ρ₀ ≈ 0.28. Seed 1 gives 0.2831.

## 3. The paper's floor uses the wrong coefficient

The paper defines ρ(P) as the Dobrushin coefficient and proves
δ̃ ≤ ε̄/(1 − ρ(P)). Its figure caption says ρ₀ ≈ 0.28 was "estimated using exact
diagonalization", which gives the second eigenvalue modulus. The worked value
0.5052/(1 − 0.28) = 0.70215 shows 0.28 was the value used. On reading-B
instances with a second eigenvalue near 0.28, the Dobrushin coefficient is
0.77–0.93. So the correct floor is 3–10× larger than the plotted one, and the
plotted line is a heuristic, not a bound. The measured-to-bound ratio "≈ 0.6"
is relative to that heuristic.

The residual is also taken at the wrong granularity. ε̄ is the worst
*per-site* kernel residual, but ρ₀ is the contraction of the whole 12-site
*sweep*. A single-site Gibbs kernel leaves every other site unchanged, so two
inputs that differ elsewhere have disjoint outputs, and its own Dobrushin
coefficient is 1. The recursion therefore only closes one sweep at a time.
One compiled sweep injects its sweep residual
η_sweep = maxₓ ‖P̃(·|x) − P(·|x)‖_TV, which the triangle inequality bounds by
Σₙ εₙ ≤ 12 ε̄, not by ε̄. The provable floor is η_sweep/(1 − ρ_D), and it holds
at every sweep: δ̃_t ≤ η_sweep (1 − ρ_D^t)/(1 − ρ_D) from a common start. On an
exactly enumerable target, η_sweep is exact to compute.

## 4. A first cap sweep (reading B, seed 1)

Variational fit (mean KL under uniform blanket inputs, L-BFGS-B, six starts),
hidden biases included, n_h = number of triples containing the site. The
bias is the total-variation error after 40 sweeps from a uniform start.

| Cap | ε̄ | ε̄/(1−ρ_D) | ε̄/(1−SLEM) | Bias | Sweeps to settle |
|---|---|---|---|---|---|
| 0.3 | 0.244 | 1.08 | 0.341 | 0.125 | 1 |
| 1 | 0.012 | 0.051 | 0.016 | 0.0017 | 2 |
| 2 | 0.0027 | 0.012 | 0.0038 | 0.0007 | 1 |
| 10 | 0.0020 | 0.009 | 0.0028 | 0.0005 | 1 |

The picture is qualitatively the paper's. The bias settles within a few sweeps,
stays below both floors, and grows as the cap tightens. The magnitudes differ
(the paper reports 0.46 at cap 0.3). Plausible causes are the instance, the
compile method, and the reading. Above cap 2 the residual stops improving
(ε̄ ≈ 0.002), which points to the six-start optimizer rather than the cap.

## Implications for M5

- A faithful reproduction is of the method and qualitative claims, on declared
  instances, under both readings.
- Kernels need hidden biases; the cap applies to them.
- Report Dobrushin and spectral coefficients separately, and call only the
  Dobrushin floor a bound.
- The fit budget needs enough starts, or a structured initialization, that
  large caps are not optimizer-limited.
