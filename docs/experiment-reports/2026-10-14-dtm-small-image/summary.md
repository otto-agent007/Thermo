# DTM small-image denoiser under the constraint layer: summary

**Outcome: the operating point resolves only at T = 2, and the lattice
denoiser buys no quality over a 114-parameter logistic model.** At T = 2,
4 bits under the `split` cap and K = 64 sweeps per step keep the DTM within
0.01 of its float model in every seed. Static 10 percent beta jitter is
negligible. The DTM is 0.003 to 0.006 behind the logistic baseline in every
seed. That is "matches" under rule 5, but not better. At T = 1 under
continuation, K\*, b\* and the 3-bit control do not resolve: one of three
trained models stays trapped in the carried free-node state.

- **Task.** Exact four-component Ising-grid mixture, L = 8, J = 0.55,
  h0 = 0.2, flip noise q = 0.2. 1,000 held-out images in 50 streams of 20.
  Noisy input 0.199, exact Bayes MPM 0.067.
- **T = 2 (K\* = 64, b\* = 4 at both jitters).** Error at 4 bits, 10 percent
  jitter, K = 64 is 0.075 (seeds 0.075, 0.076, 0.074). That is 0.008 above
  Bayes and 0.124 below the noisy input. The 3-bit control loses 0.009 to
  0.035 in every seed, so the tolerance resolves.
- **T = 1.** K\* is "> 256". Seed 1 sits 0.018 [0.010, 0.027] above its own
  K = 1024 reference at K = 256. b\* is "> 8" without jitter and 4 with it,
  both driven by seed 1's wide intervals. The 3-bit control is not
  detectable in seed 1, so at T = 1 the tolerance is reported as not
  resolving. A random start beats continuation by 0.042 in seed 1 at K = 256.
- **Steps.** Inconclusive under rule 4. T = 2 helps a lot in seed 1
  (-0.045), a little in seed 2 (-0.005) and not at all in seed 0.
- **Projected energy** (`calibrated_projection`, sealed Z1 profile, host path,
  continuation). At T = 2, K = 64 one sample costs 2.01e-8 J, and a 16-draw
  estimate costs 3.2e-7 J. I/O is 99.1 percent of that. With the clamp loads
  at the node-read constant (outside the profile), it is 6.1e-10 J per
  sample. The runner's coded operating point is T = 1, K = 256 at 1.03e-8 J,
  because rule 4 did not return "helps". Its error is 0.088, and seed 1
  reaches 0.105.
- **CPU seconds per estimate** (software on this box): THRML 0.054 (T = 2,
  K = 64), NumPy Gibbs 0.025, exact posterior 0.0021, logistic 5.2e-5.
- **Integrity.** `dtm_small_image_complete`: 3 seeds, 6 trained models, 168
  THRML and 36 NumPy cells, 1,000 test images. Preflight, placement, the
  recomputed references, the prefix check and the full replay all pass.
- **Run.** One attempt, no resume, 54 min wall on 3 workers (22:17 to
  23:11 UTC), 4.8 CPU-hours. Commit c39e91e, clean tree.

Evidence classes:

- Exact reference: the mixture, the posterior and the Bayes floor
  (`exact_reference`, float64).
- Denoiser runs and the logistic baseline: `software_simulation` (CPU).
- Joules: `calibrated_projection`.

There is no hardware claim.

Details: [findings.md](findings.md). Figure: `error-versus-sweeps.png`.
