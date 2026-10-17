# dtm-exact-posterior-lattice probe, round 3 (P-0008): layouts and oracle start

Date: 2026-10-17. Label: exploration, not evidence. This round follows the
owner's review of PR #126. It checks the index-4 pixel basis, places relay
paths on the lattice with nothing relaxed, scans the tie coupling on the
carried layout, and tries the oracle start. Earlier rounds:
[round 1](2026-10-16-dtm-exact-posterior-lattice-probe.md),
[round 2](2026-10-17-dtm-exact-posterior-lattice-probe.md).

Scripts and output (round 1 and 2 functions imported unchanged):

- [`dtm_exact_posterior_lattice_layout4.py`](dtm_exact_posterior_lattice_layout4.py)
  builds copy layouts for both pixel bases, routes relay paths by
  breadth-first search over empty sites, checks the result with
  `require_placement` under a request that relaxes nothing (8 bits,
  `split`, 10 percent jitter), computes the tie-limit model, and samples.
  - `--layout-only` ->
    [`...-layout4-layouts.json`](2026-10-17-dtm-exact-posterior-lattice-layout4-layouts.json):
    11 layouts, both families.
  - `t4` run ->
    [`...-layout4-t4.json`](2026-10-17-dtm-exact-posterior-lattice-layout4-t4.json).
  - `smooth8` run ->
    [`...-layout4-smooth8.json`](2026-10-17-dtm-exact-posterior-lattice-layout4-smooth8.json).
- [`dtm_exact_posterior_lattice_oracle_ref.py`](dtm_exact_posterior_lattice_oracle_ref.py)
  ->
  [`...-oracle-ref.json`](2026-10-17-dtm-exact-posterior-lattice-oracle-ref.json):
  exact references for the oracle arm.

CPU only, single-thread BLAS, Python 3.11, probe seed 20261017. Same 200
test images per family as round 2 (10 streams of 20), one sampler seed,
64 chains, K = 1024.

**Run history.** The first full run crashed on the `smooth8` index-4 layout:
the copy-anchor window (6 pixels beyond the image) was too small for index-4
tiles, which reach 7 pixels. A coverage check caught it before any sampling
on that layout. The window became 10 and a per-unit coverage flag was
added. The crashed run's `t4` results are kept as `...-layout4-t4.json`:
under the wider window the `t4` layouts have the same copies per unit, relay
nodes and free nodes (`...-layouts.json`). `smooth8` was rerun in full.

**CPU.** About 12 CPU-minutes for the crashed run (from its cell timings),
516 s for the `smooth8` rerun, 10 s for the oracle references and under a
minute for layout checks: about 22 CPU-minutes, inside the 30-minute probe
budget. The power figures in the protocol come from a 400,000-draw Monte
Carlo of the decision rule (seconds).

Evidence classes: the Bayes posterior, every tie-limit equilibrium,
conditioned targets and n-draw floors are `exact_reference` (float64). Gibbs
errors are `software_simulation` (NumPy two-colour Gibbs on CPU). **This
sampler is not the lowered THRML program.** Joules are
`calibrated_projection` (sealed Z1 profile constants). Nothing here is a
hardware measurement.

## Findings

1. **The index-4 basis places with nothing relaxed, and so does index 5
   once relays are routed.** `require_placement` passes on all 11 layouts:
   no offset, colour or degree violations, distinct sites, and every unit
   couples to every pixel exactly once. Maximum degree is 9 on `t4` and 13
   on `smooth8`. Round 2's direct chain-end ties, which relaxed placement,
   are gone.
   - Index 4 (−4, −1), (0, −1) has three non-pixel cosets touching 4 pixels
     per site. One coset's tile, {(0, 0), (1, −3), (1, −5), (1, −7)}, tiles
     the grid only with period 2, not by a lattice. The layout code handles
     both cases, so any coset can hold the clamps.
2. **Index 4 saves few free nodes.** Border copies are partial, so a unit
   still needs 27 to 32 copies, not 16. Relays add 13 to 38 nodes.

   | Free nodes (relays) | index 5 | index 4, best clamp coset |
   | --- | --- | --- |
   | `t4` | 197 (18) | **189** (14) |
   | `smooth8` | 339 (38) | **332** (32) |

   The review estimated about 192 on `smooth8`; the measured 332 is 1.7x
   that. Index 4 is still the layout with fewer free nodes on both
   families.
3. **`t4` passes stop (a) on both layouts at every tie value.** Tie-limit
   8-bit model on 200 images: MPM gap −0.0006 to +0.0009, exact 64-draw gap
   +0.0015 to +0.0027. The placed field cap is 0.5 (the pixel biases); the
   coupling cap is w.
4. **On index-4 `t4`, w = 2 is the only tie value near Bayes.** Random
   start, K = 1024, 8 bits, jitter 0, 95 percent stream bootstrap:

   | w | 1.5 | **2.0** | 2.5 | 3.0 |
   | --- | --- | --- | --- | --- |
   | Gap | +0.0109 | **+0.0043** [−0.003, +0.011] | +0.0162 | +0.0243 |

   Index 5 at w = 2 is +0.0125 [+0.002, +0.023]. Round 2's index-5 cells,
   with relaxed chain-end ties, were +0.018 or worse. One seed and 200
   images do not separate +0.004 from +0.012; the study does.
5. **`smooth8` is on the stop (b) line before any sampling, on both
   layouts.** Tie-limit 8-bit model, exact 64-draw majority gap: +0.0102,
   +0.0104, +0.0121, +0.0113 on index 4 (w = 1.5 to 3) and +0.0105 to
   +0.0124 on index 5. MPM gaps are +0.007 to +0.010, so stop (a) passes.
   With a standard error near 0.0025 on 200 images, the 1,000 test images
   decide the sign. The protocol now decides it at stage E.
   - Sampled `smooth8` at w = 2, random start: +0.043 (index 4), +0.041
     (index 5), far above the line.
6. **The oracle start holds the latent on both families, and it is not a
   clean mixing test on its own.**
   - Oracle cells at w = 2, K = 1024: `t4` −0.053 (index 4), −0.041 (index
     5); `smooth8` −0.047 (index 4), −0.044 (index 5). All are below Bayes.
   - The start carries the generating template, which the image does not
     (the Bayes component posterior gives the generating component a mean
     weight of 0.54 on `t4`, 0.48 on `smooth8`). The exact tie-limit model
     conditioned on it is −0.068 on `t4` and −0.102 on `smooth8`; the true
     posterior conditioned on it is the same to 0.001.
   - So the chains relax only part way in 1024 sweeps: they keep most of
     the latent on `t4` and about half of it on `smooth8`. A chain that never
     moves would also pass a "within 0.01 of Bayes" test. The protocol
     uses the oracle arm only to rule out finite-w representation loss, and
     reports the retention as a diagnostic.
7. **The decision rule's power.** Pooling three seeds with a per-seed bound
   of 0.015: at a per-seed standard error of 0.004, a true gap of +0.008
   reaches with probability 0.77, +0.010 with 0.46, +0.012 with 0.17. The
   round-2 rule (all three seeds at or below 0.01) had 0.33 at +0.008.

## Cost (calibrated_projection, sealed profile, relay nodes counted)

64-draw estimate at the write constant, against the 3.43e-6 J rung:

| Free nodes | random K = 64 | random K = 1024 | reset/5 K = 1024 | continue K = 1024 |
| --- | --- | --- | --- | --- |
| `t4` index 4 (189) | 2.50e-6 (0.73x) | 2.58e-6 (0.75x) | 1.10e-6 (0.32x) | 7.2e-7 (0.21x) |
| `smooth8` index 4 (332) | 3.91e-6 (1.14x) | 4.05e-6 (1.18x) | 1.44e-6 (0.42x) | 7.9e-7 (0.23x) |
| `smooth8` index 5 (339) | 3.98e-6 (1.16x) | 4.13e-6 (1.20x) | 1.46e-6 (0.43x) | 7.9e-7 (0.23x) |

On `smooth8` a random start fails (c) on both layouts with relays counted.
The index-4 layout does not flip that verdict. `t4` does not decide cost.

## What this means for the protocol

- Carry the layout with fewer free nodes that passes stop (a). Index 4 on
  both families is the prediction.
- Decide `smooth8`'s stop (b) at stage E from the exact tie-limit 64-draw
  gap. A miss there names bias-design representation as the mechanism.
- `t4` stop (b) is open. The best index-4 cell sits at +0.004 with one seed.
- The oracle arm distinguishes finite-w representation loss from mixing.
  It does not show whether a chain would relax away from a wrong mode.
