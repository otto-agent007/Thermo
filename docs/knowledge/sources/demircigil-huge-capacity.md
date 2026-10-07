# On a model of associative memory with huge storage capacity

```toml
id = "demircigil-huge-capacity"
title = "On a model of associative memory with huge storage capacity"
kind = "paper"
status = "asserted"
last_checked = 2026-10-06

[source]
url = "https://arxiv.org/abs/1702.01929v2"
version = "v2"
released = 2017-06-30
authors = "Mete Demircigil, Judith Heusel, Matthias Löwe, Sven Upgang and Franck Vermet"
published = "https://doi.org/10.1007/s10955-017-1806-y"

[[claims]]
id = "C1"
status = "asserted"
text = "Theorem 1.3: with interaction F(x) = e^x and M = exp(alpha N) + 1 patterns drawn uniformly from {-1,+1}^N, for 0 < alpha < log(2)/2 and a corruption fraction rho in [0, 1/2), one update corrects a configuration taken uniformly from the Hamming sphere of radius rho N around any pattern with probability tending to 1, provided alpha < I(1 - 2 rho)/2, where I(x) = ((1+x) log(1+x) + (1-x) log(1-x))/2."

[[claims]]
id = "C2"
status = "asserted"
text = "Theorem 1.2: for the polynomial model with n-body couplings W = N^(1-n) sum_mu xi^mu (x) ... (x) xi^mu, up to alpha_n N^(n-1) patterns can be stored if small retrieval errors are tolerated, and c_n N^(n-1) / log N with c_n > 2 (2n-3)!! for a fixed pattern to be a fixed point with probability tending to 1."

[[claims]]
id = "C3"
status = "asserted"
text = "For the standard Hopfield model, replica computations suggest capacity M = alpha N with alpha < 0.138 if a small fraction of retrieval errors is allowed; rigorous results confirm linear capacity with smaller alpha (cited, not proved in this paper)."
```

## Claim

Taking Krotov and Hopfield's polynomial interaction to its limit, F(x) = eˣ,
gives a model whose storage capacity is exponential in the number of neurons,
while each stored pattern still corrects a finite fraction of flipped bits in
one update.

## Method

The paper proves its theorems with large-deviation estimates for random ±1
patterns. It also gives a corrected version of the polynomial-case result for a
closely related model with explicit n-body couplings. Published in the Journal
of Statistical Physics 168 (2017).

## Relevance to Thermo

- The energy −log Σ_μ exp(β ξ^μ · σ / √N) of Thermo's probe "dense" memory is
  the finite-temperature counterpart of this model. The probe saw perfect
  half-cue recall up to P = 4N at N = 32, which is consistent with, but no test
  of, the theorem.
- The theorem is about deterministic one-step dynamics. A sampler's recall
  also depends on temperature and on how many sweeps it gets.

## Cautions

- These are asymptotic statements (N → ∞) for i.i.d. uniform patterns. They
  say nothing quantitative about N = 16 or N = 32.
- Exponential capacity needs an exponential interaction, that is, many-body
  terms or hidden units. The paper does not address implementation cost.
