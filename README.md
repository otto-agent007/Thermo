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

## Status (November 17, 2026)

The lab's question (charter amendment 2) is when physical stochastic
computation is worth using against the best practical digital method, which
representations allow it, and which hardware features would change the
answer. Work runs through an owner-curated
[research queue](docs/research-queue.md) and the [research loop](docs/research-loop.md).

- **No task has beaten digital yet.** The
  [four-week evaluation](docs/research/2026-11-11-four-week-evaluation.md)
  found the lab can state bits, sweeps, reads, writes and projected energy for
  a task, and that at the stated quality a conventional model is more accurate
  and, outside one corner of the pricing band, cheaper. Reading and writing
  the array, not sampling, dominates projected cost.
- **Closed lines:** conservation (M4B–M4I), small-image denoising, including
  every construction of the `t4` global latent
  ([closure](docs/research/2026-11-16-t4-global-latent-closure.md)), and the
  Z1T-0 sampling-noise retrofit
  ([closure](docs/research/2026-11-15-z1t-noise-retrofit-closing.md)).
- **Hardware features that would change an answer** are collected in
  [one note](docs/research/2026-11-13-z1-feature-requirements.md). If any of
  those constants is published, the archive can be re-priced without new runs.
- **Latest study, the Z1 acceptance kit (P-0015):**
  [findings](docs/experiment-reports/2026-11-17-z1-acceptance-kit/findings.md).
  Simple tests on isolated sites and pairs detect every declared chip defect
  cheaply. For a missing coupling, the matching scan needs 56 to 72 reads
  against thousands for the collective Kac-Ward scan. That scan failed its
  false-alarm limit on one target (76 against 74 of 1,000).
- **Queue:** no row is open to the loop. The remaining Z1T rows are held for
  owner-run GPU work.

Earlier work, all recorded with exact references or CPU simulation: the
PAsymSwap program (M1–M4), the topology-aware meta-EBM (M5), the THRML
finite-sweep contracts, sampling comparisons on Ising, Potts and planar
graphs, associative memory, exchange-cost pricing, and adsorption precision.
The [roadmap](docs/roadmap.md) has one row per milestone with a link to every
report, and the [study guide](docs/studies.md) gives each study's command and
limitations.

## Quick start

Python 3.11 and [uv](https://docs.astral.sh/uv/) are the supported baseline.
Everything runs on CPU without credentials or network access. The tests
for the parked improvement harness also need Linux with bubblewrap
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
- [Research queue](docs/research-queue.md), [research loop](docs/research-loop.md) and [research agenda](docs/research/2026-10-20-lab-research-agenda.md)
- [Experiment reports](docs/experiment-reports/) and [research notes](docs/research/)
- [Research knowledge base](docs/knowledge/README.md): source cards, concepts and research cautions
- [CI/CD runbook](docs/ci-cd.md) and the parked [improvement harness](docs/improvement-harness.md)
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
