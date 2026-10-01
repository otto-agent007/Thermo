# posteriors

```toml
id = "normal-posteriors"
title = "posteriors"
kind = "library"
status = "code_released"
last_checked = 2026-09-29

[source]
url = "https://github.com/normal-computing/posteriors"
version = "0.1.3"
released = 2026-09-07
runtime = "PyTorch (requires torch, torchopt, optree, tensordict)"
authors = "Normal Computing"
license = "Apache-2.0"
pypi = "posteriors"

[[claims]]
id = "C1"
status = "code_released"
text = "Scalable Bayesian posterior approximations in PyTorch, including stochastic-gradient MCMC samplers that are discretized Langevin dynamics."
```

## What it is

Normal Computing's library for uncertainty quantification with PyTorch:
scalable Bayesian posterior approximations, including stochastic-gradient
MCMC samplers that are discretized Langevin dynamics.

## Relevance to Thermo

- A reference for charter track B (Bayesian inference) and for how Langevin
  samplers are parameterized and tuned in practice.

## Cautions

- It is PyTorch. Thermo is JAX, and adding a second tensor framework to the
  lock is a large change for a reference implementation. Read its algorithms;
  don't add it as a dependency without an owner decision.
