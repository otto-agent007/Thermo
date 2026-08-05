# Topology-aware meta-EBM experiment specification

## Question

What accuracy and resource cost remain when the Thermalizers paper's exactly
enumerable 12-spin, three-body Ising target is compiled into pairwise kernels
and then constrained by the published Z1 topology, finite thermalization, and
I/O model?

## Baselines

- exact enumeration of all 4,096 target states;
- software Gibbs using exact target conditionals (`software_simulation` output);
- Torx logical Gibbs program;
- compiled fully connected pairwise proxy;
- published-offset-constrained compiled model;
- local THRML samples;
- Extropic simulator and physical Z1 only when available.

## Parameter sweep

```text
hidden p-bits per kernel
coupling cap
complete two-color Gibbs sweeps per conditional update
logical block size
training context distribution
placement and embedding expansion
chain length and chain strength
readout interval
node reads and full-SRAM writes
```

## Required outputs

- single-site conditional error;
- stationary total variation against exact target;
- finite-sweep versus equilibrium residual;
- logical and physical p-bit count;
- topology violations and any unresolved placement residual;
- calibrated sampling/read/write energy with explicit exclusions;
- sampling-only modeled time, never total latency without I/O constants;
- result evidence class.

Only enumeration or an exact transition-matrix calculation is an
`exact_reference`; sampling from an exact conditional remains
`software_simulation`.

## Topology limitation

The public connection offsets do not define the full physical chip graph or
core boundaries. Until Extropic publishes or supplies that graph, a
“Z1-topology-constrained” result must identify the synthetic lattice region and
cannot claim a production-valid full-chip embedding.
