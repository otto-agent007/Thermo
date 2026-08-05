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
uv run pytest
```

The smoke command writes validated JSON run records for:

1. a two-gate Torx circuit evaluated with the exact state-vector simulator;
2. a five-spin THRML Ising chain checked against exact enumeration.

Generated results are ignored by Git. Curated reports can be committed
deliberately under `docs/experiment-reports/`.

## Research contract

- [Project charter](PROJECT_CHARTER.md)
- [Evidence policy](docs/evidence-policy.md)
- [Z1 hardware model](docs/z1-hardware-model.md)
- [Roadmap](docs/roadmap.md)
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
