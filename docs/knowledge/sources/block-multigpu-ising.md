# Multi-GPU Accelerated Multi-Spin Monte Carlo Simulations of the 2D Ising Model

```toml
id = "block-multigpu-ising"
title = "Multi-GPU Accelerated Multi-Spin Monte Carlo Simulations of the 2D Ising Model"
kind = "paper"
status = "asserted"
last_checked = 2026-10-20

[source]
url = "https://arxiv.org/abs/1007.3726v1"
version = "v1"
published = "https://doi.org/10.1016/j.cpc.2010.05.005"
authors = "Benjamin Block, Peter Virnau and Tobias Preis"
note = "Abstract; Computer Physics Communications 181 (2010), 1549–1556."

[[claims]]
id = "C1"
status = "asserted"
text = "Conventional Ising Monte Carlo implementations use checkerboard parallelism, multispin coding and multiple GPUs."
```

This claim and its source metadata were separated from
[the adsorption application card](native-adsorption-baselines.md) to retain
one source per card under Thermo's knowledge-base schema.

The source establishes implementation techniques, not performance of
Thermo's inhomogeneous three-dimensional pore. Its historical GPU speed
ratios and hardware power must not be extrapolated to the adsorption screen.
Applicability and measured benefit require checking for the actual fields.
Thermo has not reproduced this paper's performance claims.
