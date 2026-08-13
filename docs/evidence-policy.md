# Evidence policy

Thermo labels every result according to the evidence supporting the claim, not
the marketing category of the algorithm or the accelerator used to compute it.

## Evidence classes

| Class | Meaning | Example |
|---|---|---|
| `exact_reference` | Exact enumeration, analytic calculation, or exact state-vector semantics | Enumeration of all states in a five-spin Ising model |
| `software_simulation` | Approximate or sampled execution in software on CPU/GPU | THRML block-Gibbs samples on a GPU |
| `calibrated_projection` | A calculation from a cited, versioned hardware cost model | Z1 Appendix-B node-operation energy estimate |
| `physical_hardware` | Direct telemetry from identified physical thermodynamic hardware | Future Z1 run with device and measurement provenance |

Backend and evidence are separate. The Extropic GPU simulator, Torx sampled
execution, and THRML local execution remain software simulations. A state-vector
result may be an exact reference even though a CPU calculated it.

## Run-level and metric-level labels

A run has a default evidence class, while every metric has its own class. This
allows one THRML record to contain:

- sampled magnetization as `software_simulation`;
- an exact comparison value as `exact_reference`;
- a future Z1 energy estimate as `calibrated_projection` in a separate cost-model
  record.

The record validator enforces backend/evidence compatibility. No currently
implemented backend can emit `physical_hardware`.

## Timing

JAX dispatch is asynchronous. Wall-clock execution timing is valid only after
all measured output leaves have completed with `block_until_ready()`. Thermo
separates:

- lowering/compilation time;
- synchronized steady-state execution time.

The first-call duration is not silently presented as steady-state throughput.
Runner timing aggregates are labeled `software_simulation` even when the Torx
result semantics are `exact_reference`; local CPU/GPU wall-clock timing is not
an exact mathematical claim or a physical-hardware latency measurement. Reports
state units, measured operations, exclusions, synchronization method, and the
per-run `RunTiming` source.

## Sampling terminology

Every experiment defines “sample” explicitly. Reports distinguish, where
applicable:

```text
p-bit node updates within complete sweeps
color or block phases
complete Gibbs sweeps
recorded states
effective independent samples
```

A 50 MHz cycle or update statement does not imply 50 million independent
samples per second.

## Statistical evidence

Within one THRML chain, Thermo reports lag-one autocorrelation, integrated
autocorrelation time, and effective sample size for scalar coordinate traces.
These diagnostics use a conservative Geyer initial-positive sequence with
monotonized adjacent autocorrelation-pair sums. Constant and traces shorter
than four recorded states are marked explicitly rather than producing NaN or
misleading precision. ESS is always bounded by the recorded-state count.

Aggregates persist their statistical semantics. Under
`independent_seeded_replications`, one independently seeded execution is one
replication. Scalar metrics receive descriptive statistics and, with at least
two compatible replications, a two-sided 95% Student-t interval. A one-run
aggregate reports the interval as unavailable. Under `deterministic_identity`,
such as the weighted graph walk's seed-zero identity, confidence intervals are
not applicable and no Student-t method or independent-seed reason is assigned.
Vector states and deterministic variants are never flattened into fake
replications.

## Hardware claims

A calibrated Z1 projection must include:

- profile/version identifier, content hash, and primary-source references;
- modeled Gibbs node updates, node reads, and full local-SRAM node writes;
- logical/physical/participating p-bit counts plus clamp, reflash, and host-round-trip audits;
- affected full-SRAM node writes for every clamp change or coupling reflash;
- sampling-only critical-path complete sweeps and color-block phases;
- all energy constants and units;
- explicit excluded costs;
- `calibrated_projection` evidence.

At the Appendix-B assumed maximum 50 MHz clock, the reported sampling time is a
best-case lower bound, not a generic projected latency.

A physical-hardware result additionally requires the device identifier,
firmware/runtime version, measurement method, raw telemetry reference, and
system boundary. No configuration option may manufacture that evidence label.
