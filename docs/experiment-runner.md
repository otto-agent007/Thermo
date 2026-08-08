# Experiment runner

Thermo's runner turns a checked TOML specification into reproducible per-seed
execution records, a separate aggregate, generated JSON Schemas, and a Markdown
report. Runtime Pydantic validation is authoritative; JSON Schema is a portable
description emitted from those same models.

## Configuration

Every file has one explicit implemented backend. Backend selection never uses
the filename.

```toml
schema_version = "1.0.0"
experiment_id = "thrml.ising_chain_exact_validation.v1"
backend = "thrml_local"
seed = 7
sample_definition = "One recorded full five-spin state after two complete ordered block-Gibbs sweeps; recorded-state count is not an effective-independent-sample count."

[model]
numeric_dtype = "float32"
# backend-specific, strictly typed model fields

[run]
# backend-specific, strictly typed execution fields
```

The loader uses Python's standard-library `tomllib`, rejects unknown fields,
unsupported schema versions, experiment/backend mismatches, and implicit
integer/float coercions. Validated inputs are deeply immutable. The normalized
snapshot is deterministic and never rewrites the source TOML.

## Commands

Use the checked seed:

```bash
uv run thermo-lab run configs/experiments/torx-two-gate.toml \
  --output-dir results/torx-two-gate
```

Run independent replications in the given deterministic order:

```bash
uv run thermo-lab run configs/experiments/thrml-ising-chain.toml \
  --seeds 7,8,9,10 \
  --output-dir results/ising-chain
```

`--seed N` and `--seeds A,B,C` are mutually exclusive. Seed lists must be
non-empty, unique, and non-negative. CPU is forced by default before importing
JAX backends; `--allow-accelerator` permits normal JAX device selection. A
completed output directory is protected unless `--overwrite` is explicit.

The legacy `thermo-lab smoke` command remains available for the two foundation
checks.

## Output structure

```text
results/ising-chain/
├── config.snapshot.toml
├── aggregate.json
├── report.md
├── schemas/
│   ├── run-record.schema.json
│   └── aggregate-record.schema.json
└── runs/
    ├── seed-0000000007.json
    ├── seed-0000000008.json
    ├── seed-0000000009.json
    └── seed-0000000010.json
```

Writes use a temporary sibling followed by atomic replacement where practical.
Each model is revalidated at persistence. Aggregate run paths are relative.
Multi-seed failures produce `partial` or `failed` aggregates; they never leave a
misleading `complete` aggregate.

## Hash semantics and compatibility

The model hash covers only canonical requested model inputs. Per-run
`run_config_hash` also covers experiment ID, seed, run configuration, and sample
definition. The aggregate's non-seed configuration hash excludes the seed so
independent replications share it. UUIDs, timestamps, device metadata,
provenance, timings, metrics, and results never enter input hashes.

Before aggregation, successful records must agree on experiment ID, backend,
evidence class, model hash, non-seed run configuration, sample definition,
package versions, numeric dtype, and JAX x64 setting. Incompatible records fail
loudly. Only scalar metrics present in every successful record are aggregated;
vectors remain inspectable per run.

## Statistical methods

### Within a chain

For each THRML spin coordinate and the optional magnetization trace, Thermo
computes lag-one autocorrelation, integrated autocorrelation time, and effective
sample size. It uses Geyer's initial-positive-sequence estimator: adjacent
positive autocorrelation-pair sums are monotonized and accumulated until the
first non-positive pair. The integrated autocorrelation time is bounded to
`[1, recorded_states]`, and ESS to `[0, recorded_states]`.

Constant traces are labeled `constant_series` with ESS zero because a stuck
coordinate provides no mixing evidence. Fewer than four recorded states are
labeled `insufficient_length` with no ESS. Recorded states are correlated chain
states, not independent samples. The record also states the number of complete
ordered Gibbs sweeps between recorded states.

### Between seeds

Each independently seeded run is one replication. Scalar metrics report count,
mean, sample standard deviation, median, minimum, maximum, and a two-sided 95%
Student-t interval. One successful run receives no manufactured interval and an
explicit reason. ESS intervals are truncated to their mathematical
`[0, recorded_states]` bounds. Vector metrics are not flattened.

## Reports and evidence boundaries

`report.md` is regenerated from persisted `aggregate.json` and validated run
records, not in-memory backend state. It includes hashes, seeds, provenance,
compile and synchronized steady-state timing summaries, exact comparisons,
THRML marginal/total-variation and ESS diagnostics when present, failures,
omissions, and relative artifact links.

Timing rows are labeled `software_simulation`, use seconds, preserve the
per-run synchronization method, and state their measured boundary. Compilation
and synchronized steady-state backend execution are separate; configuration
loading, the untimed warm launch, provenance collection, persistence,
aggregation, and reporting are explicitly excluded where applicable.

Torx exact state-vector output is `exact_reference`. THRML sampled CPU/GPU
output is `software_simulation`. Neither is a physical TSU or Z1 measurement.
Calibrated Z1 projections remain separate cost-model records and are never
relabeled by the runner.
