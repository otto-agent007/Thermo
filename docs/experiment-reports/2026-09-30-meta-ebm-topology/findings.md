# M5c findings: degree repair and placement on a synthetic offset lattice

September 30, 2026 (owner local date; the run log is in UTC, October 1).
Recorded under the [frozen M5c protocol](../../experiments/meta-ebm-synthetic-topology.md).
Exact reference, CPU float64, zero samples. The lattice is **synthetic**:
the published 16-offset rule of arXiv:2608.01615v2 §II.2.1 on an open
square, not a physical-chip graph. Generated tables are in
[summary.md](summary.md); every value is in `study.json.gz`.

## Result

**The primary M5 kernels compile exactly onto the published local rule, at
about 1.5× the parity-bound input copies.**

1. **Degree repair costs almost nothing in accuracy.** The 11 over-degree
   kernels refit under a J-only mask reach a five-seed median stationary
   target bias of **3.47e-5** at K = 4 (seed range 3.46e-6–7.19e-5), and
   4.52e-5 at K = 32 and the exact limit. That is below both reference bars
   (8.3e-3 and 5e-4), which are M5b's 8- and 12-bit rounding biases for this
   arm. The unfitted J-only prune sits at **0.0144** (0.0057–0.0438), so the
   refit, not the mask choice, recovers the accuracy. The original
   unconstrained kernels are at 1.25e-6.
2. **Every kernel embeds without approximation.** All 60 kernels place on
   the lattice with no chains or routing nodes. Rebuilding each placement as
   a site-level model and enumerating it reproduces the kernel conditional to
   **2.55e-15**, both at the origin and in the packed patch. Topology
   violations: zero.
3. **The cost is input copies.** Exact minimum-copy placements need
   **272–306** clamped copies per seed against a parity bound of 182–200,
   all **60/60 proven optimal**. Each seed uses 344–378 physical p-bits for
   72 free ones. Clamp writes per outer sweep rise by the same amount (about
   50%); spin redraws (288 per outer sweep at K = 4), reset writes and
   readouts are unchanged. The overhead comes from dense input-to-hidden
   couplings: every hidden spin needs a copy of every blanket input among its
   own neighbors.
4. **Area:** a 29 × 29 resident patch holds every seed's 12 kernels under
   greedy packing (27–29 per seed, 44–48% used). Time-multiplexing one
   region would need at most a 10 × 14 box, with reprogramming cost not
   estimated.

| Seed | Original bias, K=4 | J-only prune | Refit | Copies (parity bound) | Patch side |
| ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 2.13e-6 | 0.00570 | 3.47e-5 | 272 (182) | 27 |
| 1 | 1.13e-6 | 0.0174 | 3.47e-5 | 299 (197) | 29 |
| 2 | 2.57e-6 | 0.0144 | 3.46e-6 | 306 (200) | 29 |
| 3 | 1.25e-6 | 0.00741 | 7.19e-5 | 291 (193) | 28 |
| 4 | 7.89e-7 | 0.0438 | 3.95e-5 | 306 (199) | 28 |

## Refit details

- 96 of 99 starts and 10 of 11 selected fits terminated successfully at
  `maxiter` 20,000. Seed/site 1/9 reached the limit at objective 1.28e-8; it
  does not set the error floor.
- The floor is seed 3 at 7.19e-5. Its site 3/0 converges to objective
  7.52e-7 under the mask; a lower floor would need a different mask or more
  hidden spins, outside this protocol.
- Local mixing slows at most refit sites: the K for local TV ≤ 1e-3 rises,
  for example from 26 to 79 at 4/10 and from 26 to 67 at 0/1, while 3/0
  speeds up from 77 to 30. The worst case over refit kernels is 79 against
  77 originally. The outer K = 4 result already includes this effect.

## Integrity

All 15 original chains replayed against M5b to 3.95e-16. Refit conditionals
matched brute-force enumeration to 1.22e-15, and the 49 unchanged vectors
were used bitwise. The maximum outer stationary residual was 2.74e-16 and
the row-sum error 5.55e-16. The persisted archive was replayed in full
without refitting or re-solving before `completion.json` was written.

The run took 17.5 minutes on the 8-CPU host (909 s generation, 141 s replay)
with three workers and six placement workers. This is evaluator time, not
device latency.

## Boundary

Copy, p-bit, clamp-write and area counts are algorithmic counts under the
stated clamped-copy input model. They are not device operations, and no
energy, latency, reprogramming or production-embedding claim follows. Only
the primary arm was placed. Refit endpoints depend on the machine; this
record is the evidence, and a rerun elsewhere may select different starts.

## What this closes

M5c answers the topology question M5 set: the compiled kernels fit the
published rule exactly once degree is repaired, and the remaining cost is
measured. Two levers on copy cost, sparse-input training and cross-kernel
copy sharing, are named in the protocol but not planned; either needs owner
approval. The next milestone should come from the charter tracks not yet
started.
