# M5c synthetic-patch placement probe

Status: **exploration**, `exact_reference`, CPU float64, zero samples. This
follows the [degree and repair probe](2026-09-28-m5c-input-model-probe.md) and
the [bounded degree refit](2026-09-29-m5c-degree-refit-probe.md). It is not a
frozen M5c protocol or a new study gate. The lattice is **synthetic**: it uses
the published local connection rule, not a physical-chip map.

## Question

The repaired kernels satisfy degree 16 and checkerboard parity. Those are
necessary conditions only. This probe asks whether every kernel of the primary
arm embeds on the published offset lattice without chains or routing nodes,
and what the embedding costs in clamped input copies, physical p-bits and
patch area compared with the parity lower bound used so far.

## Choices fixed before placement

- **Scope:** the primary arm, reading B, variational, cap 1: five seeds, 12
  sites, **60 kernels**. The 11 over-degree kernels use the J-only mask of the
  refit note. Placement depends only on which coefficients are nonzero. The
  masked archived vectors and the 20,000-iteration refit endpoints have the
  same nonzero pattern at all 11 sites, so the probe reads the pattern from the
  authenticated M5b archive and needs no refit. Other arms are not placed.
- **Lattice:** integer grid with the 16 offsets of arXiv:2608.01615v2 §II.2.1
  (`(1,0)`, `(2,1)`, `(2,3)`, `(4,1)` and their four rotations). The rule is
  closed under 90° rotation. Every coupler not carrying a model coefficient
  is set to zero.
- **Free p-bits:** the output `y` and each hidden spin occupy one site each.
  There are no chains, no routing nodes and no split outputs. Each nonzero
  `beta` edge must be a lattice edge, so hidden spins sit on neighbors of `y`.
- **Clamped copies:** clamped input model as approved. Each copy site holds
  one blanket input. A nonzero `J` input gets one copy on a neighbor of `y`;
  free neighbor slots of `y` are filled in offset order. `A` copies sit on the
  other color. A copy serves every hidden spin adjacent to it; for each
  nonzero `A[a, i]` exactly one adjacent copy carries the coefficient, and
  the rest of its couplers are zero. Copies are shared only within a kernel.
- **Per-kernel objective:** minimize the number of clamped copies with an
  exact integer program (SciPy `milp`, HiGHS), on the unbounded lattice with
  `y` at the origin. Every archived hidden row has the same input support,
  which the probe asserts, so only the **set** of hidden offsets matters.
  Time limit 120 s per kernel; status and MIP gap are reported. The count is
  the result; the returned layout is one optimum among possibly many.
- **Resident patch:** all 12 kernels of a seed stay resident on one open
  square patch of side `L`, with disjoint site sets and no cross-kernel
  couplers. Only the kernel being updated is sampled; the others are idle.
  Kernels may be rotated by multiples of 90° and translated. Packing is
  greedy first-fit: largest footprint first (ties by site), anchors in
  row-major order, rotations 0 to 3, `L` increasing from
  `ceil(sqrt(total sites))`. The heuristic `L` is an upper bound on the
  smallest resident patch, not an optimum.
- **Time-multiplexed alternative:** one region reused by every kernel in turn,
  sized by the largest single-kernel bounding box. Reprogramming traffic is
  not estimated.
- **Accounting:** clamp writes per outer sweep are the placed copies summed
  over the 12 kernels, one conditional update per site per sweep, replacing
  the parity-only counts. Redraws, reset writes and readout bits are
  unchanged.
- **Verification:** an independent checker rebuilds each placed kernel as a
  site-level coupling matrix, confirms every coupled pair is a lattice edge of
  opposite color, and enumerates the free p-bits for every input row. The
  physical conditional must match `kernel_logit` to within 1e-12. Packed
  layouts are rechecked in patch coordinates with the finite patch's
  neighbor sets.

There is no pass threshold on cost. Feasibility with zero topology
violations is the question; copy, p-bit and area costs are descriptive.
Because the embedding is exact, placed conditionals equal the unplaced ones
and the refit's outer accuracy carries over unchanged.
