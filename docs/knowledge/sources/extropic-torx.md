# Torx

```toml
id = "extropic-torx"
title = "Torx"
kind = "library"
status = "pinned_dependency"
last_checked = 2026-10-01

[source]
url = "https://github.com/extropic-ai/torx"
version = "0.0.1"
released = 2026-08-04
runtime = "JAX (requires jax, equinox, jaxtyping, ihoop)"
authors = "Extropic"
license = "Apache"
pypi = "extro-torx"

[dependency]
package = "extro-torx"
version = "0.0.1"

[[claims]]
id = "C1"
status = "pinned_dependency"
text = "High-level library for stochastic programs, with an exact state-vector simulator (used for references) and sampled execution (used for simulation)."
```

## What it is

Extropic's high-level library for stochastic programs, described in the
[Torx paper](extropic-torx-paper.md). Thermo uses its exact
state-vector simulator for references and its sampled execution for
simulation (see the charter's stack diagram).

## Use in Thermo

The two-gate and weighted-graph-walk experiments, and the Torx side of the
Milestone 1 cross-layer benchmark.

## Cautions

- 0.0.1 is an early release. Preserve its behavior in tests before upgrading.
- 0.0.2 was released on PyPI on September 30, 2026. Thermo still pins 0.0.1.
  Run `tests/upstream_regressions/test_torx_001_contracts.py` against 0.0.2
  before any upgrade.
- The Extropic simulator API and Thermalizers lowering are not public
  dependencies. Don't add placeholder integrations for them (AGENTS.md rule 7).
