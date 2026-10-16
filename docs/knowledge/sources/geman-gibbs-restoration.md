# Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images

```toml
id = "geman-gibbs-restoration"
title = "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images"
kind = "paper"
status = "asserted"
last_checked = 2026-10-16

[source]
url = "https://doi.org/10.1109/TPAMI.1984.4767596"
version = "IEEE TPAMI 6(6), 721-741, November 1984"
released = 1984-11-30
authors = "Stuart Geman and Donald Geman"

[[claims]]
id = "C1"
status = "asserted"
text = "An image prior given as a Gibbs distribution over pixel (and line) states is a Markov random field, and for a range of degradations (blur, nonlinear deformation, additive or multiplicative noise) the posterior is again a Markov random field with structure like the prior's."

[[claims]]
id = "C2"
status = "asserted"
text = "Site-by-site stochastic relaxation (the Gibbs sampler) with a slowly decreasing temperature converges to the maximum a posteriori restoration, and gives good restorations of simple pictures at low signal-to-noise ratio."
```

## Claim

A local energy on pixels defines a Markov random field prior. Under
pixelwise noise the posterior keeps the prior's neighbourhood structure,
with the observation entering as a field. The paper introduces the Gibbs
sampler and anneals it toward the MAP restoration.

## Method

The prior is a Gibbs distribution with pairwise (and line-process) cliques.
The posterior adds the log-likelihood of the observed image as single-site
terms. Sites are updated in parallel within an independent set, which is the
chromatic block Gibbs the Z1 lattice runs. A logarithmic cooling schedule is
proved to reach the MAP set.

## Relevance to Thermo

- It is the source of the statement P-0006 and D-0003 rely on: a local prior
  under pixelwise noise gives a local posterior, so a small local digital
  model can come close to the Bayes floor. P-0006 found exactly that
  (Bayes 0.067, 7 x 7 logistic 0.071).
- The row `dtm-baseline-headroom` looks for families where the posterior is
  not local, which needs a global latent (component identity) on top of the
  MRF, not a stronger local coupling.
- Thermo's denoising studies report the marginal-posterior (MPM) estimate,
  not MAP. Besag (1986, [card](besag-dirty-pictures.md)) discusses the
  difference.

## Cautions

- The convergence result needs a logarithmic schedule that no finite-sweep
  study runs. Do not cite it for any fixed K.
- The paper's line processes make the model larger than a pairwise Ising
  grid; Thermo's families have no line process.
