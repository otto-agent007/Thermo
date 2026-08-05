# Z1 hardware model

This document separates announced Z1 facts, the topology described in the first
Thermalizers paper, and its refined Appendix-B cost projection. These sources do
not form a complete public hardware specification.

Primary sources:

- [From One to One Billion](https://extropic.ai/writing/from-one-to-one-billion/)
- [Thermalizing Stochastic Programs](https://arxiv.org/abs/2608.01615)

## Published facts encoded by `Z1HardwareProfile`

| Field | Value | Interpretation |
|---|---:|---|
| p-bits | 269,568 | Announced physical count |
| cores | 8 | Announced core count |
| coupling parameters | 215,904 | Opaque announced count; not treated as an edge count |
| interior degree | 16 | Most interior nodes; boundaries have lower degree |
| graph colors | 2 | Supports alternating chromatic updates |
| connection rules | `(1,0)`, `(2,1)`, `(2,3)`, `(4,1)` plus rotations | Sixteen interior offsets; maximum length is sqrt(17) |

The public rules do not reveal the full physical grid dimensions, core boundary
layout, defects, routing constraints, or an exact chip graph. Thermo therefore
sets `physical_grid_shape=None` and `exact_physical_graph_available=False`.

Directional endpoint couplings are supported. A symmetric coupling matrix is
required when claiming a Boltzmann equilibrium and detailed balance.

## Two distinct rate statements

The announcement advertises a sampling rate greater than 50 MHz. Appendix B
uses 50 MHz as the assumed maximum complete-sweep rate for its updated cost model.
Thermo keeps these as separate typed statements; it does not collapse them into
one exact clock or independent-sample rate.

For this project, one modeled **complete sweep** means one full two-color Gibbs
iteration: update the first color block, then the second, with each participating
non-clamped p-bit updated at most once. The record exposes two color-block phases
per complete sweep. This definition prevents “block update,” “cycle,” and “sweep”
from being used interchangeably.

## Refined Appendix-B operation model

The encoded projection is:

\[
E = U(7.09\times10^{-15})
  + R(1.692\times10^{-12})
  + W(153.6\times10^{-12})\;\text{joules},
\]

where:

- \(U\) is the number of modeled p-bit-node updates within complete sweeps;
- \(R\) is the number of p-bit-node reads to the chip boundary;
- \(W\) is the number of full local-SRAM p-bit-node writes.

Clamped nodes are not counted as Gibbs updates. Changing clamp state is treated
as a write; the paper does not publish a cheaper clamp-only constant.

For \(C\) complete sweeps at the assumed maximum clock, the model reports a
best-case sampling-only time:

\[
T_{sampling,max-clock} = C / (50\times10^6).
\]

The true sampling time is greater than or equal to that value because the clock
must be chosen with respect to p-bit autocorrelation. The model cannot calculate
total wall time because public read, write, serialized I/O, and host-latency
constants are incomplete.

Operation records also preserve logical and physical p-bit counts, participating
free p-bits, clamp-state changes, coupling reflashes, and host round trips. The
latter counts remain audit fields when no defensible energy/latency constant is
published for them. A coupling reflash is an event count, not a node count;
every recorded reflash must include at least one affected full-SRAM node write,
and only those node writes contribute energy in the current model.

## Exclusions

Every projection excludes:

- host energy and latency;
- I/O latency;
- idle energy;
- current core-wide I/O access amplification;
- board and system power.

Current Z1 I/O is described as core-wise and serialized, while the paper's
projection idealizes node-level access. Thermo preserves that caveat.

As a unit check, one modeled complete sweep over all 269,568 p-bits costs about
1.91123712 nJ. Repeating that modeled sampling operation at 50 MHz corresponds
to 95.561856 mW of sampling-only dynamic power. It is not a measurement of total
chip power and must not be compared directly with the announcement's sub-watt
chip/die power claim.
