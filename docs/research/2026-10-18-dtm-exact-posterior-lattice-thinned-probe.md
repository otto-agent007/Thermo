# dtm-exact-posterior-lattice probe, thinned draws (D-0005 premise option 2)

Date: 2026-10-18. Label: exploration, not evidence. Owner-requested, run by
the owner's Claude Code session, outside the research loop. It answers the
estimator half of direction review D-0005's premise option 2
([D-0005](2026-10-18-direction-review.md), "Pattern", item 2): may one clamp
load serve the 64 draws of a majority estimate? The profile half, whether the
Z1 appendix lets a clamped array be read repeatedly without a reload, is the
owner's reading and is not touched here.

Script and output:

- [`dtm_exact_posterior_lattice_thinned_probe.py`](dtm_exact_posterior_lattice_thinned_probe.py)
  runs one chain per image on P-0008's archived model, loads the clamp once,
  reads the pixel states every `lag` sweeps, 64 times, takes the pixelwise
  majority of the 64 reads (ties to +1, as the runner), and prices the path
  with the constraint layer's `count` and `price`.
- [`...-thinned-summary.json`](2026-10-18-dtm-exact-posterior-lattice-thinned-summary.json):
  per-cell and pooled gaps, paired differences against the archived cells,
  prices on both paths.
- [`...-thinned-cells.json.gz`](2026-10-18-dtm-exact-posterior-lattice-thinned-cells.json.gz):
  per-image majority counts, the per-read single-draw error curve, the
  latent flip rate and copy agreement of every cell.

Model, images, quantization and jitter are P-0008's, imported from the merged
runner `thermo_lab.dtm_exact_posterior_lattice` (commit 1d78bae, nothing
edited): `t4_h05_q035`, index-4 layout, w\* = 1.75, 189 free nodes, 8-bit
`split`, 10 percent jitter with the archived per-seed jitter words, the
operating arm's constraint request, `require_placement` with nothing relaxed.
The 1,500 test images are the archived ones (500 per sampler seed, 25 streams
of 20), so every comparison is paired image by image with the archived
`maj_wrong` counts in `evidence.tar.gz`. Fresh RNG seed 20261318. CPU only,
single-thread BLAS, 3 workers, 36 minutes wall, 1.8 CPU-hours. NumPy
two-colour Gibbs, the runner's own loop; not the lowered THRML program.
Nothing ran on hardware.

**Two paths, equal sweeps.** The archived estimate is the majority of 64
independent chains, each run K sweeps from its start with the noisy image
loaded, so 64 clamp loads per estimate. The thinned estimate is one chain,
one clamp load, 64 reads at lag K, so 64 K sweeps and 64 reads either way.
Starts: `random` (a fresh uniform state per image) and `continue` (one chain
per stream carries its state from the last read of the previous image, the
protocol's continuation). Lags 64 (random only), 256, 1024 and 4096. The
lag-4096 random cells pair with the archived oracle pair's random-start
cells.

## Result: thinned draws lose nothing

Pooled 64-draw gap above the Bayes MPM error (mean of three per-seed gaps,
1,500 images), thinned against the archived independent-chain cell at the
same K, and the pooled paired difference (`software_simulation`):

| Start | Lag K | Thinned | Archived, 64 chains | Thinned minus archived |
| --- | --- | --- | --- | --- |
| random | 64 | 0.0235 | 0.0352 | -0.0117 |
| random | 256 | 0.0148 | 0.0256 | -0.0107 |
| continue | 256 | 0.0170 | 0.0409 | -0.0239 |
| random | 1024 | 0.0122 | 0.0166 | -0.0043 |
| continue | 1024 | 0.0122 | 0.0209 | -0.0087 |
| random | 4096 | 0.0128 | 0.0121 (pair, random start) | +0.0007 |
| continue | 4096 | 0.0126 | (no archived cell) | |

Per seed, the thinned estimate is better than the archived one at every lag
at most 1024 in every seed, and the paired 95 percent stream-bootstrap
interval of the difference excludes zero in 14 of those 15 comparisons (seed
2 random at lag 1024 reaches +0.0002). At lag 4096 the three paired intervals
all include zero (-0.0031 to +0.0049). The reason is burn-in: read r of a
thinned chain has had r K sweeps, so only the first read matches an
independent chain at K. The latent decorrelates between reads: the dominant
template unit changes between consecutive reads 18 percent of the time at
lag 64, 31 percent at 256, 49 percent at 1024 and 56 percent at 4096. Copy
agreement at the reads is 0.89 to 0.90 in every cell, as in the archived
cells.

So the estimator half of premise option 2 holds on this model: a majority of
64 thinned reads from one clamped chain is no worse than 64 independent
chains at any lag tested, and better below 4096. The one-sided ending
condition proposed for the probe (thinned no worse than independent by more
than 0.003 at some lag at most 4096) is met at every lag.

**What it does not change.** No thinned cell reaches Bayes + 0.01. Both
starts plateau at 0.012 to 0.013 from lag 1024 on, next to the finite-w
equilibrium the oracle pair measured (0.0117). Per seed at lag 1024: random
0.0097, 0.0128, 0.0142; continue 0.0114, 0.0113, 0.0139. Stop (b) and the
mechanism named in P-0008 stand. The thinned path changes what an estimate
costs, not what it gets.

## Pricing (`calibrated_projection`, sealed Z1 profile)

Per 64-draw estimate at K = 1024, 189 free nodes, the layer's `count` and
`price`. Writes are full SRAM writes (1.536e-10 J), reads 1.692e-12 J,
updates 7.09e-15 J. Band: write constant at the top, clamp loads at the read
constant at the bottom (queue header, standing rule 1).

| Path, start | Clamp loads | Writes | Reads | Updates | J per estimate, write constant | J, clamp at read |
| --- | --- | --- | --- | --- | --- | --- |
| independent, continue (P-0008) | 64 x 64 | 4,096 | 4,096 | 12,386,304 | 7.24e-7 | 1.02e-7 |
| independent, random (P-0008) | 64 x 64 | 16,192 | 4,096 | 12,386,304 | 2.58e-6 | 1.96e-6 |
| thinned, continue | 64 | 64 | 4,096 | 12,386,304 | 1.05e-7 | 9.5e-8 |
| thinned, random | 64 | 253 | 4,096 | 12,386,304 | 1.34e-7 | 1.24e-7 |

Under thinning the clamp floor per estimate falls from 6.3e-7 J (64 loads
of 64 pixels) to 9.8e-9 J (one load), 64x, as D-0005 computed. The band
collapses with it: the thinned continuation estimate is 1.05e-7 J at the
write constant and 9.5e-8 J at the read constant, a 10 percent spread
instead of 7x, because the single load is 9 percent of the total. Updates
are now 84 percent of the projected energy (8.8e-8 J), reads 7 percent.
That is 6.9x below P-0008's independent continuation estimate at the same
K, and it sits at 1.0e5 MACs at 1 pJ per MAC: below the 243,200-MAC net
that misses Bayes + 0.01 on `t4` by 0.0005 (2.3x), the archived `t4` rung
`cnn2_w32_k5` (1.74e6 MACs, 16.6x) and the `smooth8` rung
`cnn2_w32_k5_gpool` (3.43e6 MACs, 32.8x). None of these is a cost verdict:
on `t4` the thinned estimate is 0.002 above the tolerance, and on `smooth8`
the encoding misses before any sampling (D-0005, "The one thing to drop").
The sealed profile prices clamp state changes; whether the array can hold
a clamp through 64 K sweeps and be read 64 times is the profile reading
this probe leaves to the owner.

## For the four-week statement

If the profile reading allows it, clause 3 of the amendment's test changes
from "the write-constant clamp floor reads against the hardware on any L = 8
task under about 6e5 MACs" (D-0005) to: at K = 1024 and 64 draws the
hardware's projected energy per estimate is about 1e-7 J, updates-bound, so
the break-even digital cost is about 1e5 MACs at 1 pJ per MAC; on the one
family with digital headroom that the exact model holds, the lattice is
0.002 short of the quality tolerance, so there is still no task where it
wins on both. P-0006's cost reading (99 percent I/O, 16 draws) would move
the same way and is not recomputed here.

## Not claimed

No hardware measurement, no result from the lowered THRML program, no change
to any archived number, no reaching cell, no cost verdict on `smooth8`. The
thinned estimator was tested on one model, one family and one tie value;
its behaviour on a model that mixes worse (larger w, or `smooth8`) is not
measured.
