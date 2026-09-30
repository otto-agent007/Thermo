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

## Reproduction

```bash
uv run python docs/research/m5c_placement_probe.py > /tmp/m5c-placement.jsonl
uv run python docs/research/m5c_placement_probe.py --seeds 4 > /tmp/m5c-placement-seed4.jsonl
```

The script authenticates the M5b archive, derives each kernel's nonzero
pattern and J-only mask, solves one integer program per kernel, packs each
seed and runs both verification passes. Seeds are independent. The recorded
run covered seeds 0–3 in one invocation, which an outer wrapper timeout
stopped during seed 4, and seed 4 in a separate `--seeds 4` run. A full run
takes about 30 minutes on one CPU, almost all of it in six time-limited
solves. The optimal counts are exact; the six time-limited incumbents depend
on solver speed and may differ by a copy or two on another machine.

## Results (2026-09-30)

**Every kernel embeds.** All 60 kernels of the primary arm place on the
published offset lattice with no chains, routing nodes or topology
violations. The placed site-level models reproduce `kernel_logit` for every
input row to a maximum error of **2.78e-15**, both at the origin and after
packing into the patch. The embedding is exact, so the refit's outer accuracy
carries over unchanged.

**Input copies cost about 1.5× the parity bound.**

| Seed | Free p-bits | Parity-bound copies | Placed copies | Ratio | Physical p-bits (parity bound) | Patch side | Utilization | Largest single-kernel box |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 72 | 182 | 272 | 1.495 | 344 (254) | 27 | 47.2% | 10 × 14 |
| 1 | 72 | 197 | 299 | 1.518 | 371 (269) | 29 | 44.1% | 10 × 14 |
| 2 | 72 | 200 | 306 | 1.530 | 378 (272) | 29 | 44.9% | 12 × 10 |
| 3 | 72 | 193 | 291 | 1.508 | 363 (265) | 28 | 46.3% | 11 × 11 |
| 4 | 72 | 199 | 306 | 1.538 | 378 (271) | 28 | 48.2% | 11 × 10 |

Free p-bits are the 12 outputs and 60 hidden spins. Physical p-bits add every
clamped copy. The parity-bound copy counts equal the J-only clamp-write counts
in the refit note.

- **Why:** every archived `A` row is dense, so each hidden spin needs a copy
  of every blanket input among its 15 other neighbors. A copy serves several
  hidden spins only where their neighborhoods overlap. The overhead on `A`
  copies grows with the hidden count: the per-kernel median ratio to the
  parity bound is 1.57 with 3 hidden spins, 1.71 with 4, 2.0 with 5, 2.11
  with 6 and 2.5 with 7. Kernels with one or two hidden spins meet the bound
  exactly. `J` copies always meet it.
- **Work:** clamp writes per outer sweep rise from **182–200** to
  **272–306**, about 50%. Spin redraws stay at 72K per outer sweep, with 72
  reset writes and 12 readout bits. The largest single kernel uses 45 sites.
- **Resident patch:** a **29 × 29** open square holds every seed's 12
  kernels under the greedy packing (27–29 per seed), at 44–48% utilization.
  This is an upper bound on the smallest resident patch.
- **Time-multiplexed region:** the largest single-kernel bounding box is
  **10 × 14** (140 sites), against 841 for the resident patch. The difference
  would have to be paid in reprogramming traffic, which is not estimated.

**Solver status.** 54 of 60 solves are proven optimal. The other six reached
the 120 s limit. All six are masked degree-16 kernels with 7–8 hidden spins
and 10–11 inputs, and each incumbent is **2 copies** above its dual bound
(gap 7.1–7.7%). Certified lower bounds on per-seed copies are 272, 297, 302,
289 and 302, so the per-seed totals above are within 0–4 copies of optimal.
Median solve time was 11 s; the total was 1,734 s.

## Reading the result

Placement is not the obstacle: once degree is repaired, the published rule
embeds these kernels exactly. The cost is copies. At about 5 physical p-bits
per free p-bit and 1.5× the parity-bound clamp traffic, the dominant term is
the dense input-to-hidden coupling, not the lattice. Two fixed choices bound
this number from above and are the levers if it matters:

- **Dense `A` from the variational fit.** Nothing in the M5a objective rewards
  sparsity. A fit that limits each hidden spin's input support would cut
  copies directly, but it is a training change and needs owner approval.
- **No cross-kernel copy sharing.** Copies of one visible spin in different
  kernels hold the same value between updates of that spin, so sharing is
  semantically possible. It changes the clamp-write accounting and couples
  the packing to the copy problem.

Neither lever is explored here.

## Boundary

The lattice is synthetic: the published local rule on an open square, not a
physical-chip graph, boundary map or placement API. Copy counts are
algorithmic counts under a stated input model, not device
operations. No latency, energy, reprogramming or physical-placement claim
follows. Other arms were not placed. A frozen M5c protocol would need to
record these placements as evidence rather than cite this exploration.
