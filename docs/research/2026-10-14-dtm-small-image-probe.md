# dtm-small-image probe (P-0006)

Date: 2026-10-14. Label: exploration, not evidence. Script:
[`dtm_small_image_probe.py`](dtm_small_image_probe.py). Output:
[`2026-10-14-dtm-small-image-probe.json`](2026-10-14-dtm-small-image-probe.json).
CPU only, single-thread BLAS, Python 3.11, probe seed 20261014 (never reused
by a protocol). One full run takes 325 CPU-s. The whole probe, including two
earlier passes and a THRML timing, used about 20 CPU-minutes.

Evidence classes in this note:

- The data likelihood, the exact samples and the Bayes posterior are
  `exact_reference`.
- Denoiser and baseline errors are `software_simulation` in float64 NumPy,
  on 200 test images.
- Every joule figure is a `calibrated_projection` of algorithmic counts from
  the sealed Z1 profile (`thermo_lab.constraints.price`).

Nothing here is a hardware measurement.

## Findings

1. **The I/O path is per-step host readback and rewrite.** Under the sealed
   profile, on-chip chaining costs more, not less.
   - The profile prices a clamp-state change as a full SRAM write.
   - Chaining T layers therefore adds 2 (T - 1) L^2 clamp changes, which
     host readback avoids.
   - At L = 16, T = 2, K = 64 the projected costs per denoised sample are:

     | Path | Projected energy | I/O share |
     | --- | --- | --- |
     | Host per step | 8.0e-8 J | 99.1 % |
     | Chained, clamp changes as writes | 1.19e-7 J | 99.4 % |
     | Chained, copies free (hypothetical, outside the profile) | 4.05e-8 J | — |

   - Across K = 16 to 1024, energy is set by T x L^2 reads and writes.
     Sweeps barely move it: at T = 2, energy rises only 14 % from K = 16 to
     K = 1024.
   - Bits never enter the price. The quantity to design against is reads
     and writes per sample, and T is the knob that sets it.
2. **The exact reference covers the data, not the denoiser.**
   - The data family is a four-component mixture of nearest-neighbour Ising
     grids with template fields.
   - A site-by-site transfer matrix gives its exact likelihood, exact
     samples and the exact flip-noise posterior. This was checked against
     brute force on a 3 x 3 grid to 1e-15.
   - Cost: 1.7 ms per image at L = 8 and 0.37 s per component at L = 16.
   - The denoiser's lattice graph has latents and long offsets, so its own
     likelihood is not tractable. Held-out quality is measured against the
     data's exact Bayes floor, as conditional log-loss and pixel error.
3. **Synthetic data goes first.**
   - No binarized image set is on disk, and fetching one is a download the
     owner decides on.
   - The synthetic set has an exact floor, which a real set would not.
4. **The lattice layout places with nothing relaxed.**
   - Each pixel uses a 2 x 2 site cell: pixel, its clamp, latent A and
     latent B. Edges are the Z1 offsets on pixel-A and A-B pairs.
   - At L = 8 that gives 256 nodes, 876 edges and maximum degree 16.
     `require_placement` passes.
5. **The task's headroom bounds the metric.** This is the main finding.
   - At the probe's family (J = 0.35, h0 = 0.35, flip rate q = 0.2), the
     noisy input is wrong on 0.201 of pixels and the exact Bayes MPM on
     0.155. Only 0.046 separates doing nothing from doing everything right.
   - Every denoiser landed inside that gap. With 16 draws, majority
     estimates had standard errors of about 0.004:

     | Model | Error |
     | --- | --- |
     | Best DTM, float (T1, K_train 64, K 256) | 0.1725 |
     | 7 x 7 patch logistic baseline (114 parameters) | 0.168 |
     | Quantized DTM, 6 to 8 bits | 0.170 to 0.176 |
     | Quantized DTM, 4 bits | 0.183 to 0.187 |
     | Quantized DTM, 3 bits | 0.206 to 0.221 |

   - Jitter at 0.1 moved no quantized error by more than about one
     standard error.
   - T = 2 and T = 4 were no better than T = 1. Tripling training steps did
     not help.
   - Differences between 6 and 8 bits are therefore smaller than the
     probe's noise.
   - The probe's headroom scan (L = 8, 200 exact images per cell) finds a
     wider gap at stronger coupling. At J = 0.55, h0 = 0.2, q = 0.2 the
     noisy input is wrong on 0.201, Bayes on 0.071, and the headroom is
     0.131 +- 0.005. That is three times the probe family's gap, so a
     protocol should fix this family.
6. **Cap rule.**
   - `split` and `absolute` (cap 2.5) agree within error from 4 bits up.
   - `split` is lossless by construction for trained weights: it uses the
     model's own maximum |w| and |b|.
   - An `absolute` cap needs a number chosen per training run, which is one
     more fixed choice that could bound quality.
   - The protocol should declare `split`.
7. **Sizing.**
   - Float DTM inference in NumPy: 3.3e-4 CPU-s per sample per layer at
     L = 8, K = 64, and about 4x that at L = 16.
   - THRML through `constraint_layer.lower` and `final_states`, warm:
     - L = 8: 8.4 CPU-s per 6,400 chains at K = 64, and 28 at K = 256.
     - L = 16: 40 CPU-s per 6,400 chains at K = 64, and 160 at K = 256.
     - This is about 4e-4 to 6e-3 CPU-s per chain-layer.
   - Training one L = 8 layer for 1,600 Adam steps (batch 64, K_train 16)
     takes 22 CPU-s. At K_train 64 it takes 82 CPU-s.
   - At L = 16, one K_train-16 step at batch 64 takes 0.11 CPU-s, so 1,600
     steps take about 3 CPU-min per layer.
   - The logistic baseline costs 2.2e-5 CPU-s per image, about 1,000 times
     less than 16 DTM draws.

## Fixed choices and whether each could bound the metric

| Choice | Probe setting | Could it bound the metric? |
| --- | --- | --- |
| I/O path | host per step | Sets the joule: energy scales with T L^2. Quality is unaffected |
| Data family (J, h0, q) | J 0.35, h0 0.35, q 0.2 | **Yes, binding.** A 0.046 headroom compresses every comparison below noise |
| Exact reference graph | data grid only | Bounds which log-likelihood is exact. Denoiser likelihood is not reported |
| Image size L | 8 (16 timed) | Cost only, at L <= 16 |
| Steps T | 1, 2, 4 | No quality gain from T > 1 here. T sets I/O cost |
| Latent layout | A and B per pixel | Could bound capacity, but the logistic baseline does not beat it beyond error |
| K_train, training steps | 16 to 64, 400 to 4800 | Small: K_train 64 gained about 0.01, and extra steps did nothing |
| Inference sweeps K | 4 to 256 | K = 4 is clearly worse. K >= 16 is flat within about 0.01 |
| Cap rule | split, absolute 2.5 | Equivalent from 4 bits. `split` adds no free number |
| Bits | 3, 4, 6, 8 | 3 bits clearly worse. 4 bits about 0.01 worse. 6 and 8 tied |
| Jitter | 0, 0.1 | Within noise |
| Test set size | 200 images | Bounds resolution: SE about 0.004 per arm |
| Baseline | 7 x 7 logistic | Matches the DTM |

The fixed choice that bounds the metric is the data family's headroom. The
I/O path sets the projected joule.
