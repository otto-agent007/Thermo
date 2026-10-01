# Thermodynamic linear algebra

```toml
id = "aifer-thermodynamic-linear-algebra"
title = "Thermodynamic linear algebra"
kind = "paper"
status = "asserted"
last_checked = 2026-09-29

[source]
url = "https://arxiv.org/abs/2308.05660v2"
version = "v2"
released = 2024-06-10
note = "v1 posted 2023-08-10."
authors = "Aifer, Donatella, Gordon, Duffield, Ahle, Simpson, Crooks and Coles"
published = "https://www.nature.com/articles/s44335-024-00014-0"

[code]
card = "normal-thermox"
note = "The paper's algorithms are implemented in thermox."

[[claims]]
id = "C1"
status = "asserted"
text = "Linear systems, matrix inverses, determinants and Lyapunov equations can be solved by sampling the equilibrium distribution of coupled harmonic oscillators."

[[claims]]
id = "C2"
status = "asserted"
text = "Under the paper's hardware assumptions, the speedup over digital methods scales linearly in matrix dimension."
```

## Claim

Linear systems, matrix inverses, determinants and Lyapunov equations can be
solved by sampling the equilibrium distribution of coupled harmonic
oscillators. The paper argues for speedups that scale linearly in matrix
dimension relative to digital methods, under its hardware assumptions.

## Method

For a symmetric positive-definite A, an Ornstein–Uhlenbeck process with drift
−(Ax − b) has a Gaussian stationary distribution with mean A⁻¹b and covariance
proportional to A⁻¹. Time-averaging samples estimates the solution or the
inverse.

## Relevance to Thermo

- It is Normal Computing's stochastic-processing-unit line, a continuous
  counterpart to Extropic's discrete TSU.
- The OU process has a closed-form transition, so any sampled solve has an
  exact reference. It is a cheap, bounded warm-up (backlog item E4).

## Cautions

- The speedup is a scaling argument about idealized hardware. Error depends on
  conditioning, integration time and sample count, which a Thermo study would
  report as algorithmic counts, not device time.
