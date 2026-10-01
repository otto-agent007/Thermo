# thermox

```toml
id = "normal-thermox"
title = "thermox"
kind = "library"
status = "code_released"
last_checked = 2026-09-29

[source]
url = "https://github.com/normal-computing/thermox"
version = "0.0.5"
released = 2025-03-19
runtime = "JAX (requires jax, jaxlib, fmmax)"
authors = "Normal Computing"
license = "Apache-2.0"
pypi = "thermox"

[[claims]]
id = "C1"
status = "code_released"
text = "Exact simulation of Ornstein-Uhlenbeck processes by diagonalization, with no discretization error (thermox.sample, thermox.log_prob)."

[[claims]]
id = "C2"
status = "code_released"
text = "thermox.linalg (solve, inv, expm, negexpm) implemented by simulating OU processes."
```

## What it is

A JAX library for exact simulation of Ornstein–Uhlenbeck processes by
diagonalization, with no discretization error. It exposes `thermox.sample`,
`thermox.log_prob`, and `thermox.linalg` (`solve`, `inv`, `expm`, `negexpm`)
implemented by simulating OU processes. It implements the algorithms of
[Thermodynamic linear algebra](aifer-thermodynamic-linear-algebra.md).

## Relevance to Thermo

- Same array stack as Thermo (JAX), so it could join without a second
  framework.
- Its exact OU sampler and `log_prob` could serve as an independent reference
  for the linear (J₄ = 0) case of the Langevin experiments in the backlog.

## Cautions

- 0.x with no release since March 2025. Under AGENTS.md rule 6 it must be
  pinned and its behavior preserved in contract tests before any result uses
  it. Check that its JAX requirement resolves against Thermo's lock first.
- A closed-form OU transition via a matrix exponential is short to write. Add
  the dependency only when a study needs more than that.
- "Exact" here means exact OU transition sampling. Estimates built from those
  samples are still `software_simulation`.
