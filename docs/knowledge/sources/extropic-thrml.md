# THRML

```toml
id = "extropic-thrml"
title = "THRML"
kind = "library"
status = "pinned_dependency"
last_checked = 2026-09-29

[source]
url = "https://github.com/extropic-ai/thrml"
version = "0.1.4"
released = 2026-08-04
runtime = "JAX (requires equinox, jaxtyping)"
authors = "Extropic"
license = "Apache-2.0"
pypi = "thrml"

[dependency]
package = "thrml"
version = "0.1.4"

[[claims]]
id = "C1"
status = "pinned_dependency"
text = "Block-Gibbs sampling of discrete energy-based models in JAX. Four upstream contract tests guard the pinned behavior."

[[claims]]
id = "C2"
status = "asserted"
text = "Its block-Gibbs sampler follows the stated conventions at small sweep budgets. Thermo has compared it with exact references only near equilibrium (200 warm-up sweeps, K = 30); backlog item E0 checks small budgets."
```

## What it is

Extropic's JAX library for block-Gibbs sampling of discrete energy-based
models, written close to the TSU's execution model.

## Use in Thermo

The Ising-chain and PAsymSwap swap experiments run on it locally. Its runs
are `software_simulation`, never Z1 evidence. Four upstream contract tests
guard the pinned behavior (roadmap, Phase 0). Its outputs have been compared
with exact references only near equilibrium (200 warm-up sweeps in the smoke
test, K = 30 for PAsymSwap). Backlog item E0 checks agreement at small sweep
budgets.

## Cautions

- Pinned at 0.1.4. Upgrade only after the contract tests pass against the new
  version.
- Its `examples` extra pulls matplotlib, networkx, scikit-learn and Jupyter.
  Keep them out of the core install.
