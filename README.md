# Thermo

Thermo is a reproducible research laboratory for stochastic programs and
thermodynamic computing. It uses Extropic's
[Torx](https://github.com/extropic-ai/torx) and
[THRML](https://github.com/extropic-ai/thrml) libraries to:

- execute exact stochastic-circuit references;
- simulate TSU-oriented energy-based models locally;
- validate small sampled models against exact enumeration;
- develop algorithms without confusing GPU simulation, calibrated Z1
  projections, and measurements from physical thermodynamic hardware.

The first milestone is a cross-layer stochastic-kernel benchmark. The current
foundation implements two real execution paths—Torx state-vector execution and
local THRML Gibbs sampling—and a separately labeled Z1 cost-model projection.
Unreleased or inaccessible systems are documented in the roadmap rather than
represented by placeholder backends.

## Quick start

Python 3.11 and [uv](https://docs.astral.sh/uv/) are the supported baseline.

```bash
uv sync --frozen
uv run thermo-lab smoke --output-dir results/smoke
uv run thermo-lab run \
  configs/experiments/thrml-ising-chain.toml \
  --seeds 7,8,9,10 \
  --output-dir results/ising-chain
uv run thermo-lab run \
  configs/experiments/torx-weighted-graph-walk.toml \
  --output-dir results/weighted-graph-walk
uv run thermo-lab run \
  configs/experiments/thrml-independent-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/independent-pasym-swap
uv run thermo-lab run \
  configs/experiments/thrml-target-context-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/target-context-pasym-swap
uv run pytest
```

The smoke command writes validated JSON run records for:

1. a two-gate Torx circuit evaluated with the exact state-vector simulator;
2. a five-spin THRML Ising chain checked against exact enumeration.

Generated results are ignored by Git. Curated reports can be committed
deliberately under `docs/experiment-reports/`.

The `run` command strictly validates a checked TOML specification, selects its
explicit backend, and writes per-seed records, a compatibility-checked aggregate,
generated JSON Schemas, and a Markdown report. CPU execution is the default;
`--allow-accelerator` is an explicit opt-in. See the
[experiment runner guide](docs/experiment-runner.md) for the configuration and
statistical contracts.

The checked weighted graph-walk baseline implements the continuous-time Markov
chain semantics and deterministic Torx Euler-PSWAP state vectors described in
the
[approved weighted graph-walk design](docs/superpowers/specs/2026-08-12-weighted-graph-walk-baseline-design.md),
using the published fixture and gate order from the
[Torx paper](https://arxiv.org/pdf/2608.01612v1#page=10). Its seed zero is an
identity field, not a replication. Its compile and synchronized execution
timings are local `software_simulation` evidence; they are not hardware or
calibrated-projection measurements.

The checked independent PAsymSwap command is a separate, atomic method-level
reconstruction from the Thermalizers paper's 5 by 5 fixture. It independently
compiles each unique two-bit target channel into a declared five-spin,
two-color thermodynamic kernel, then checks exact equilibrium and finite
Gibbs-horizon behavior with a THRML sampled cross-check. It is distinct from
the five-node Torx weighted-graph paper baseline, and it does not compose the
500 gate occurrences into a 25-site program. The Thermo-selected `[-2, 2]`
field/coupling cap is the approved checked-input revision; the target,
objective, horizons, reset semantics, and acceptance gates are unchanged. This
is not an implementation of unpublished Thermalizers, official compatibility,
or hardware evidence.

The checked exact target-context PAsymSwap command propagates the declared
one-particle target marginal through all 500 occurrences, pools the resulting
contexts into 37 equal-occurrence target-hash profiles, and compares paired
uniform and target-context artifacts. Any reported improvement applies only
under that exact target input distribution. Exact target propagation and
evaluation are `exact_reference` evidence for the declared process and frozen
software-derived models; optimization, THRML sampling, and timings remain
`software_simulation`. This study did not evaluate model-context matching,
trajectory-level REINFORCE refinement, the complete compiled 25-site rollout,
official Thermalizers, hosted simulation, or Z1 hardware.

## Research contract

- [Project charter](PROJECT_CHARTER.md)
- [Evidence policy](docs/evidence-policy.md)
- [Z1 hardware model](docs/z1-hardware-model.md)
- [Roadmap](docs/roadmap.md)
- [Experiment runner](docs/experiment-runner.md)
- [August 2026 release intake](docs/release-intelligence/extropic-2026-08.md)

Every reported claim carries an evidence class. In particular, a THRML or Torx
run on a GPU remains software simulation; it is not a Z1 hardware measurement.

## Repository layout

```text
src/thermo_lab/       reusable experiment, backend, record, and cost-model code
tests/                unit, integration, statistical, and upstream contracts
configs/experiments/  checked-in machine-readable experiment specifications
docs/                 policies, release intelligence, and experiment designs
results/              generated local output (ignored)
```

## License

Thermo is licensed under the [Apache License 2.0](LICENSE), matching both THRML
and Torx.
