# Dense Associative Memory for Pattern Recognition

```toml
id = "krotov-hopfield-dense-memory"
title = "Dense Associative Memory for Pattern Recognition"
kind = "paper"
status = "asserted"
last_checked = 2026-10-06

[source]
url = "https://arxiv.org/abs/1606.01164v2"
version = "v2"
released = 2016-09-27
authors = "Dmitry Krotov and John J. Hopfield"

[[claims]]
id = "C1"
status = "asserted"
text = "The standard Hopfield model, energy E = -1/2 sum_ij sigma_i T_ij sigma_j with Hebbian T_ij = sum_mu xi_i^mu xi_j^mu, stores and reliably retrieves of the order of K_max ~ 0.14 N random binary memories (citing earlier statistical-physics work)."

[[claims]]
id = "C2"
status = "asserted"
text = "With energy E = -sum_mu F(xi^mu . sigma) and polynomial F(x) = x^n, a signal-to-noise estimate for random binary memories gives K_max = alpha_n N^(n-1) for a small per-bit error threshold (0.5%), and about N^(n-1) / (2 (2n-3)!! ln N) for error-free recall (equations 5 and 6). n = 2 recovers the standard model."
```

## Claim

Replacing the Hopfield model's quadratic energy with a sharper function of each
pattern's overlap, E = −Σ_μ F(ξ^μ · σ), lets a network of N binary neurons store
far more than N patterns. For F(x) = xⁿ the capacity estimate grows as Nⁿ⁻¹.

## Method

The paper uses energy-difference updates, flipping one ±1 unit at a time, and
estimates the signal and noise in the energy gap for random ±1 memories (section
2). It then relates the model to feedforward networks with one hidden layer and
rectified-polynomial activations, and tests them on XOR and MNIST. Published at
NeurIPS 2016 (Advances in Neural Information Processing Systems 29).

## Relevance to Thermo

- It is the reference point for the probe's "dense" construction. Thermo's
  categorical-hidden memory realizes the exponential limit of this family
  (see [the exponential-capacity card](demircigil-huge-capacity.md)) by
  marginalizing one multi-label hidden unit.
- The many-body energy needs either n-body couplings or hidden units, which is
  the hardware question Thermo's associative-memory study takes up.

## Cautions

- The capacity results are signal-to-noise estimates for zero-temperature
  dynamics started at a stored pattern. They are not statements about a
  finite-temperature sampler or a partial cue, which is what Thermo measures.
- The 0.14 N figure for the standard model is cited from earlier work, not
  derived in this paper.
