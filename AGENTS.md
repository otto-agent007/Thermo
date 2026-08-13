# Thermo contributor instructions

## Purpose

Thermo develops and evaluates stochastic algorithms with Torx and THRML while
preparing for future TSU hardware. Correct evidence labeling is a release gate.

## Non-negotiable research rules

1. Keep exact references, software simulations, calibrated projections, and
   physical-hardware measurements distinct in code, records, charts, and prose.
2. A CPU/GPU Torx or THRML result is `software_simulation` unless the algorithm
   is mathematically exact, in which case its result semantics may be
   `exact_reference`. It is never physical Z1 evidence.
3. Do not publish a latency, power, or energy number without its assumptions,
   units, included operations, excluded costs, evidence class, and source.
4. Synchronize JAX work before recording wall-clock time. Separate first-call
   compilation from steady-state execution.
5. Validate small probabilistic models against exact enumeration when feasible.
6. Pin upstream 0.x releases and preserve their public behavioral contracts in
   tests before upgrading.
7. Do not add placeholder integrations for unreleased Thermalizers, unavailable
   simulator APIs, or inaccessible physical hardware.

## Engineering rules

- Keep import-package code under `src/thermo_lab`; do not use `import thermo`,
  which collides with an existing package.
- Confine Torx and THRML API usage to backend/experiment adapters.
- Use immutable experiment inputs and separate observed run records.
- Hash only canonical requested inputs, never timestamps, device metadata,
  timings, or results.
- Declare model numeric dtype in the hashed input and cast backend parameters
  explicitly; record JAX x64 configuration as runtime provenance.
- Define precisely what one recorded "sample" means for every experiment.
- Use distinct JAX keys for initialization and sampling.
- Treat independently seeded runs, not correlated states within one chain, as
  the replication unit for confidence intervals.
- Keep raw diagnostic traces out of ordinary JSON run records; persist only
  bounded summaries unless a separately hashed trace artifact is specified.
- Keep exact enumerators deliberately bounded.
- Tests and the default smoke command must run on CPU without credentials,
  remote services, notebooks, or network access.
- Notebooks may visualize or call library code later; they must not become the
  sole source of an algorithm.

## Required local gates

```bash
uv sync --frozen
uv lock --check --offline
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run thermo-lab smoke --output-dir results/smoke
uv run thermo-lab run configs/experiments/torx-two-gate.toml --seeds 0,1,2 --output-dir results/torx-run
uv run thermo-lab run configs/experiments/thrml-ising-chain.toml --seeds 7,8,9,10 --output-dir results/thrml-run
uv run thermo-lab run configs/experiments/torx-weighted-graph-walk.toml --output-dir results/weighted-graph-walk
uv build
```
