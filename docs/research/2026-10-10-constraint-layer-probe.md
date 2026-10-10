# Constraint layer: archive contracts through a prototype layer (probe, exploration)

Exploration on 2026-10-10 for queue row `constraint-layer` (loop proposal
P-0004). Script: [`constraint_layer_probe.py`](constraint_layer_probe.py); raw
numbers: [`2026-10-10-constraint-layer-probe.json`](2026-10-10-constraint-layer-probe.json).
**Not recorded evidence.** It uses a throwaway prototype layer in the script,
not a module. There is no archive, no replay and no gate. It reads the
hash-bound modules and committed archives and edits none of them. It took
about 22 CPU-minutes in total, including reruns. The full run took 236 s
wall and 865 s user CPU; JAX used several threads in part P3.

## What the prototype layer is

- A hashed constraint request that declares the lattice rules, the codebook,
  the cap rule, the bits, the jitter, the chromatic schedule,
  `replica_exchange: false` and a sorted `relaxed` list.
- The M5b codebook with clipping and any width.
- Static per-site gains.
- A lattice checker. It checks that every coupled pair is a Z1 offset apart,
  that the two sites have opposite colours and that degree is at most 16.
- A lowering from an M5c layout to a pairwise model with clamped copy nodes.
- A generic exact two-colour block-Gibbs kernel.
- A THRML `IsingEBM` lowering with one block per colour.
- Z1 operation counts through the sealed `Z1HardwareProfile`.

## Results against the row's metric

1. **M5b rounding: bitwise.** All 120 reading-B precision cells were
   checked. The layer's codebook equals `round_parameters` on every array,
   and its rounding summary equals the archived `rounding` field in 120/120
   cells. Chain metrics recomputed from layer-rounded vectors (3 cells) match
   to at most 8.3e-17 but not bitwise. `bias` and `sweep_tv_*` are bitwise;
   the stationary-solve fields differ in the last bits. That is BLAS, not the
   layer: the 2026-10-06 lesson says replay must compare such fields
   numerically.
2. **M5c placement: bitwise.** 60/60 archived layouts pass the layer's own
   lattice check. The maximum degree is 16 and there are no offset or colour
   violations. All 60 reproduce `verify_layout`'s archived fields bitwise.
   9/9 re-solved MILP placements (the fastest, at most 5 s each) give the
   archived layout bitwise. Re-solving all 60 costs about 41 CPU-minutes at
   archived timings. The planar-16-offset-ferro graphs also pass: all 18
   rebuild bitwise and lie on the lattice, with maximum degree 9.
3. **am-coupling-bits at 6 bits (step = J): bitwise, with placement
   relaxed.** All 96 archived exact units give `coupling/6` recall bitwise
   through the layer's quantizer. Two archived THRML runs at P = 8 with a
   12-bit cue (jitter 0 and 0.1, layer gains) reproduce the correct-bit
   counts bitwise. The design cannot be placed: hidden degree is 24 and free
   visible degree is 8 to 128, against 16. A chain embedding would need at
   least 28 to 400 p-bits against 20 to 144 logical free spins, and it would
   change the law. This confirms the text fix that D-0001 proposed.
4. **E0 stage B: bitwise only with placement relaxed. With placement
   applied, the metric cannot be met as written.** Unplaced, the layer's
   THRML lowering reproduces E0's archived histograms bitwise in all 3
   cells tried (y0-/K1, y0+/K1, y0-/K2; 65,536 chains per input). The E0
   kernel (seed 0, site 1) has output degree **17**, so the lattice rejects
   it. What M5c placed is its refit, with J[4] masked and the rest refitted.
   Placed on 42 p-bits, the refit's exact chromatic law equals its own
   `powered_rates` law to 1.9e-15. THRML on the placed model passes that
   law within E0's archived tolerances (output 0.0045 against 0.0089, joint
   0.0188 against 0.0197; one cell). But the refit's inner-K law differs from
   the archived E0 law by up to 0.31, 0.40 and 0.38 at K = 1, 2 and 4 (worst
   input). Agreement with E0 "through the layer" holds only for the unplaced
   kernel.
5. **One priced example per constraint.** These are calibrated projections
   of algorithmic counts, not device operations. The relaxed flags are
   hashed, and changing one changes the digest.
   - Placement: one M5c outer sweep at K = 4 on the 344-site patch costs 288
     updates, 12 reads and 344 writes, 272 of them clamp changes. I/O is
     99.996% of modelled energy.
   - Quantization and jitter: one AM recall at P = 8 with a 12-bit cue over
     256 sweeps costs 5,120 updates, 12 reads and 32 writes. I/O is 99.3%.
   - Chromatic schedule with no exchange: one 1024-spin planar chain over
     4,096 sweeps, read every 4 sweeps, costs 4.19M updates, 786k reads and
     1,024 writes. I/O is 98.0%.

## Fixed choices and what each bounds

- **Placement of over-degree kernels.** This bounds the E0 item (4 above).
  E0's kernel has degree 17, so the layer cannot place it unchanged. Any
  bitwise E0 check must run with placement relaxed and say so.
- **Placement of the AM `bias` design.** This bounds the AM item (3). It
  passes only with placement relaxed (D-0001).
- **Cap rule.** "Step = J" exists only for models whose couplings share one
  magnitude, like the AM design. M5b uses an absolute cap of 1.0, and its
  bits are limited to {4, 8, 12} with no clipping. The request has to
  declare the cap rule per model. It does not bound agreement.
- **BLAS-dependent float64 fields.** These bound "bitwise" for stationary
  and trajectory metrics. Compare them numerically at about 1e-15.
- **The definition of a counted update.** It does not bound agreement, but
  it changes the priced numbers. Sequential per-site kernels (M5b schedule)
  update only the active kernel. Whole-array sweeps update everything.
- **MILP re-solve cost.** It does not bound agreement; it bounds CI time.
  A full re-solve takes about 41 CPU-minutes, so use the fast subset in CI.

## What this means for the row

The question is answered by exploration. A study-local layer reproduces the
rounding, placement and AM contracts bitwise. It reproduces E0 bitwise when
unplaced. It cannot reproduce E0 once placement is applied, because the
kernel is over-degree. It prices updates, reads, writes and clamp changes
separately, and records which constraints are relaxed in the hashed request.
None of this is a number anyone designs against. Under amendment 1 the row
is engineering, a module and unit tests, not a recorded study. D-0001 said
the same. The layer still has to land as code before later rows can run
through it. A proposal PR cannot carry it, because those PRs may touch only
`docs/experiments/`, `docs/research/` and the queue.

Evidence classes, if any of this were recorded:
- Rounding, placement and exact laws: `exact_reference`.
- THRML histograms: `software_simulation` (CPU, float32).
- Pricing: `calibrated_projection` of algorithmic counts.

No hardware claim.
