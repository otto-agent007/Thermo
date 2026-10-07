# Torx

```toml
id = "extropic-torx"
title = "Torx"
kind = "library"
status = "pinned_dependency"
last_checked = 2026-10-07

[source]
url = "https://github.com/extropic-ai/torx"
version = "0.0.2"
released = 2026-09-30
runtime = "JAX (requires jax, equinox, jaxtyping, ihoop)"
authors = "Extropic"
license = "Apache"
pypi = "extro-torx"

[dependency]
package = "extro-torx"
version = "0.0.2"

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

Pinned at 0.0.2 since 2026-10-07 (0.0.1 before). 0.0.2 adds injectable
samplers (`torx.AbstractSampler`, default `JaxPRNGSampler`), takes discrete
dimensions from each gate, rejects adding circuits with `reps > 1`, and
computes generator-gate probabilities with `jax.nn.sigmoid`. Under 0.0.1 and
0.0.2, the two-gate (seeds 0–2), weighted-graph-walk and smoke runs gave
identical records apart from timings, and the upstream contract tests pass,
including a new one that pins the default sampler to `jax.random` draws.

## Cautions

- 0.0.1 is an early release. Preserve its behavior in tests before upgrading.
- 0.0.2 was released on PyPI and tagged on GitHub on September 30, 2026
  (tag `v0.0.2`, commit `6b74450`). Thermo still pins 0.0.1. The release
  notes list one code change, "Misc. Fixes" (torx PR #22: signature,
  fixes, PR builds), and three documentation changes (PRs #28, #29, #32);
  6 commits, 22 files. Checked October 3, 2026: the pinned contract test
  `tests/upstream_regressions/test_torx_001_contracts.py` passes against
  0.0.2 in an isolated environment (PSWAP two-gate float32 state-vector
  density, atol 1e-7). That test covers one circuit; it is not a full
  behavioral audit. A pin bump still needs the owner's decision and a
  `uv lock` update, and the Torx base gates in AGENTS.md rerun on the new
  version before it is recorded.
- The Extropic simulator API and Thermalizers lowering are not public
  dependencies. Don't add placeholder integrations for them (AGENTS.md rule 7).
