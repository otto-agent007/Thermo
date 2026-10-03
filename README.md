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

Every reported claim carries an evidence class. A THRML or Torx run on a CPU or
GPU is `software_simulation`, never a Z1 hardware measurement. Unreleased or
inaccessible systems are documented in the roadmap rather than represented by
placeholder backends.

## Status (October 2, 2026)

- **M1–M4** built and audited a compiled 25-site, 500-operation PAsymSwap
  program: population-objective audit, finite-sweep transfer, exact
  finite-sweep gradients, and five bounded K=4 updates.
- **M4B–M4I (conservation line, closed).** The compiled program loses its
  single particle. Only 0.04% of K=4 paths survive all 500 operations
  (M4C). No training, weighting or fixture variant reached the 95% survival
  target. The main cause was the ±2 field/coupling cap
  ([analysis](docs/research/2026-09-23-cap-leakage-analysis.md)). Raising the
  cap and adding one kernel feature (M4H, M4I) brought exact survival to 89.6%,
  still short of 95%.
- **M5:** the [topology-aware meta-EBM](docs/experiments/topology-aware-meta-ebm.md).
  The [M5a cap baseline](docs/experiment-reports/2026-09-27-meta-ebm-cap-baseline/summary.md)
  and [M5b thermalization/precision study](docs/experiment-reports/2026-09-28-meta-ebm-thermalization/findings.md)
  and [M5c synthetic-topology study](docs/experiment-reports/2026-09-30-meta-ebm-topology/findings.md)
  are recorded exact CPU references; complete hardware costs remain open.
- **Native sampling:** [time-to-accuracy experiments](docs/experiment-reports/2026-10-02-sampling-time-to-accuracy/findings.md)
  favor different methods for different targets. [Symmetry plus tempering](docs/experiment-reports/2026-10-02-symmetry-tempering/findings.md)
  transfers to six fresh zero-field graphs; ordinary Gibbs remains the working
  reference for the small denoising posteriors.
- **Changing evidence:** [the recorded study](docs/experiment-reports/2026-10-02-changing-evidence/findings.md)
  finds useful state retention and measurable history dependence. Failure
  prediction improves, but the tested restart policy does not reliably improve
  on retention. Exact enumeration wins the measured timings at this small size.
  The [conditional-estimation follow-up](docs/experiment-reports/2026-10-02-conditional-estimation/findings.md)
  reduces marginal error on identical states, especially at weak coupling;
  strong-coupling errors and decision-utility tradeoffs remain.

The [roadmap](docs/roadmap.md) has one row per milestone with links to every
recorded report. The [study guide](docs/studies.md) describes each experiment,
its command and its limitations.
The [October sampling synthesis](docs/research/2026-10-02-sampling-synthesis.md)
connects all six new reports, review corrections and the next research questions.

## Quick start

Python 3.11 and [uv](https://docs.astral.sh/uv/) are the supported baseline.
Everything runs on CPU without credentials or network access. The
improvement-harness tests also need Linux with bubblewrap
(`sudo apt-get install bubblewrap`), which sandboxes candidate code.

```bash
uv sync --frozen
uv run thermo-lab smoke --output-dir results/smoke
uv run pytest tests/unit -m "not slow"
```

The smoke command writes validated JSON run records for a two-gate Torx circuit
evaluated with the exact state-vector simulator, and a five-spin THRML Ising
chain checked against exact enumeration. `uv run thermo-lab run <config>` runs
any checked TOML under `configs/experiments/`. See the
[experiment runner guide](docs/experiment-runner.md). The full list of release
gates is in [AGENTS.md](AGENTS.md) and [docs/release-gates.md](docs/release-gates.md).

Generated results go to `results/`, which Git ignores. Curated reports are
committed deliberately under `docs/experiment-reports/`.

## Documentation

- [Project charter](PROJECT_CHARTER.md) and [roadmap](docs/roadmap.md)
- [Evidence policy](docs/evidence-policy.md) and [Z1 hardware model](docs/z1-hardware-model.md)
- [Study guide](docs/studies.md) and frozen protocols in [docs/experiments/](docs/experiments/)
- [Experiment reports](docs/experiment-reports/) and [research notes](docs/research/)
- [CI/CD runbook](docs/ci-cd.md) and [improvement harness](docs/improvement-harness.md)
- [Execution environments](docs/environments.md): local and Codex cloud limits, which gates fit where
- [August 2026 release intake](docs/release-intelligence/extropic-2026-08.md)
- [Research dashboard](dashboard/README.md)

## Repository layout

```text
src/thermo_lab/       reusable experiment, backend, record, and cost-model code
tests/                unit, integration, statistical, and upstream contracts
configs/experiments/  checked-in machine-readable experiment specifications
docs/                 policies, protocols, reports, and release intelligence
dashboard/            read-only research dashboard
results/              generated local output (ignored)
```

## License

Thermo is licensed under the [Apache License 2.0](LICENSE), matching both THRML
and Torx.
