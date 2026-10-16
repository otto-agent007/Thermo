# dtm-exact-posterior-lattice probe (P-0008)

Date: 2026-10-16. Label: exploration, not evidence. Scripts:
[`dtm_exact_posterior_lattice_probe.py`](dtm_exact_posterior_lattice_probe.py)
and [`dtm_exact_posterior_lattice_floors.py`](dtm_exact_posterior_lattice_floors.py).
Output: [`2026-10-16-dtm-exact-posterior-lattice-probe.json`](2026-10-16-dtm-exact-posterior-lattice-probe.json)
and [`2026-10-16-dtm-exact-posterior-lattice-floors.json`](2026-10-16-dtm-exact-posterior-lattice-floors.json).
CPU only, single-thread BLAS, Python 3.11, probe seed 20261017 (fresh, not
in any archive or protocol). The recorded run took 1,630 CPU-s and the floors
4 CPU-s. An earlier exact-only pass on `t4` took another 249 CPU-s and is
not recorded separately. Total about 31 CPU-minutes, 1 minute over the
probe budget.

Scope, as the row asks: the **unplaced** model only. Exact enumeration over
hidden configurations times the transfer matrix (`GridIsing`, imported
unchanged), and NumPy chromatic Gibbs. No THRML, no placement, no tied
copies, no training. Family fields are imported from
`thermo_lab.dtm_baseline_headroom` (P-0007's archived generator).
Quantization uses the constraint layer's codebook (`cap_for`,
`quantize_array`) under `split`.

Evidence classes:

- `exact_reference`: the mixture data, the Bayes MPM error, every model's
  exact equilibrium, and every majority-vote floor (float64).
- `software_simulation`: the NumPy Gibbs errors (CPU).
- `calibrated_projection`: every joule (sealed Z1 profile constants).

Nothing here is a hardware measurement.

## The model

The posterior is p(x, c | y) ∝ exp(J Σ x_i x_j + h_c·x − log Z_c + g y·x),
with g = ½ log((1 − q)/q). The template c becomes binary hidden units, one
per template, with pixel couplings d_c = h_c − m and hidden bias
b0 − (log Z_c − mean log Z).

- **bias** design (stage A's): independent hidden units, one scalar b0 set on
  a 300-image development set (fresh salt).
- **one-hot** design: pairwise inhibition λ, set on the same development set.
- **centered** (m = mean template) against **raw** (m = 0) pixel coupling.

The oracle check passes. Restricting the one-hot enumeration to single-on
states reproduces the Bayes posterior to 6.9e-15.

## Findings

1. **`t4` passes stop (a) at 8 and 6 bits.** Exact equilibrium MPM error
   minus Bayes (1,000 test images, standard error 0.0006 to 0.003):

   | Design (t4) | float | 8 bits | 6 bits | 4 bits |
   | --- | --- | --- | --- | --- |
   | bias, centered, b0 = −5 | −0.0003 | +0.0001 | +0.0006 | +0.0327 |
   | bias, raw, b0 = −12 (grid edge) | +0.0396 | +0.0403 | +0.0216 | +0.1563 |
   | one-hot, centered, λ = 8 | 0.0000 | +0.0002 | +0.0164 | +0.2186 |

   - Centering is required for the bias design. Raw couplings put 93
     percent of the hidden mass on several-on states.
   - The bias design at 6 bits is the cheapest arm within 0.01 at
     equilibrium. Its `split` caps are 0.375 on couplings and 3.55 on fields.
     4 bits fails.
   - One-hot needs 8 bits (coupling cap 4.0, field cap 9.0). This is AM A's
     range-hunger again.
2. **`t4` sampling is at its estimator floor.** Bias design, 8 bits, 25
   streams of 20 images, one sampler seed. Error is the 16-draw majority's
   error minus Bayes, with a 95 percent stream bootstrap:

   | Start | K = 64 | K = 256 | K = 1024 | writes per sample (unplaced) |
   | --- | --- | --- | --- | --- |
   | continuation | +0.0199 | +0.0087 [0.0046, 0.0128] | +0.0086 [0.0049, 0.0118] | 64 |
   | random | +0.0131 | +0.0114 | +0.0098 [0.0067, 0.0129] | 132 |
   | reset every 5 | +0.0162 | +0.0092 [0.0056, 0.0127] | +0.0089 [0.0065, 0.0113] | 77.6 |

   - From K = 256, every start is within 0.003 of the quantized model's own
     exact 16-draw equilibrium error. The chains mix: the four-way latent
     switches within 256 sweeps.
   - **Continuation works on `t4`.** It is no worse than a random start at
     K ≥ 256, which reverses P-0007's nine of nine for this model.
   - 10 percent static jitter changes nothing (+0.0096 at K = 1024, random).
   - At 6 bits the random start is +0.0142 at K = 1024, against the model's
     own +0.0122 floor.
   - One-hot at 8 bits does not mix. It sits +0.0196 above its own
     equilibrium at K = 1024 from a random start.
3. **The binding fixed choice is the 16-draw estimator, not the lattice.**
   This is the main finding. Exact floors of an n-draw majority of
   independent draws from the true posterior, as error minus Bayes (floors
   JSON):

   | Draws | 1 | 4 | 16 | 32 | 64 | 128 | 256 |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | `t4` | +0.084 | +0.051 | **+0.0095** | +0.0042 | +0.0018 | +0.0007 | +0.0002 |
   | `smooth8` | +0.085 | +0.051 | **+0.0144** | +0.0080 | +0.0047 | +0.0029 | +0.0018 |

   - On `t4`, a perfect sampler with 16 draws already spends 0.0095 of the
     0.01 tolerance. Every interval in finding 2 straddles +0.01, so
     resolving "within Bayes + 0.01" at 16 draws would need far more test
     images than P-0007 used. The result is decided by the estimator.
   - On `smooth8`, **no sampler can reach Bayes + 0.01 with 16 draws**. The
     exact posterior's own 16-draw majority is +0.0144. As written, with
     (c)'s 16-draw estimate, stop (b) fires on `smooth8` by construction.
     The mechanism would be the estimator's variance, not the lattice or
     chromatic Gibbs.
   - At 32 draws `smooth8`'s floor is +0.008, leaving 0.002 of tolerance.
     At 64 draws it is +0.0047.
4. **`smooth8` stop (a) is not decided.** The probe's budget cut its scan
   short.
   - One-hot, centered, λ = 32 is exact in float (+0.0000) and fails at
     8 bits (+0.0282). Its `split` field cap is 96.8, so an 8-bit field step
     is 0.76.
   - The bias design's b0 scan ran from −10 to 0 to fit the budget. It was
     still falling at its edge (+0.112 at −10 on development). Its 8-bit
     error, +0.1235, is therefore an untuned point, not a verdict. On `t4`
     the optimum was interior (−5).
   - So no 8-bit design reached Bayes + 0.01 on `smooth8` here, and the one
     that is exact in float fails on range.
5. **`smooth8` sampling, on models that do not represent the posterior.**
   - The bias design at b0 = −10 has an exact 16-draw equilibrium error of
     0.347, about the noisy input.
   - Continuation sits 0.05 to 0.07 above that equilibrium, so it is
     trapped.
   - A random start sits 0.03 to 0.04 below it, so it is not at equilibrium
     by K = 1024. Its transient happens to help.
   - One-hot at 8 bits from a random start is +0.057 above its equilibrium.
   - None of this says how a correct `smooth8` model mixes. It does show
     that the eight-way latent is much harder to move than the four-way one.

## Cost arithmetic (calibrated_projection, sealed profile)

These numbers are per sample, host per step, on the **unplaced** model:
64 node reads, 64 clamp loads, (64 + C) free nodes.

- **`t4` per 16-draw estimate at the write constant:** 1.6e-7 J under
  continuation, 3.3e-7 J under a random start.
- **Placement adds copies.** Each hidden unit couples to all 64 pixels, and a
  copy has at most 14 pixel edges beside its two tie edges. That means at
  least 5 copies per unit: 20 or more on `t4` and 40 or more on `smooth8`.
  At M5c's 1.5x it is about 30 and 60. Free-node writes under a random
  start rise to about 94 per sample on `t4` and 124 on `smooth8`. Pixel
  degree is 4 grid neighbours, plus 1 clamp, plus C hidden copies: 9 on
  `t4` and 13 on `smooth8`, within 16.
- **`smooth8` stop (c), rough and placed.** The baseline is 3.43e6 MACs at
  1 pJ, 3.43e-6 J per image.
  - Random start, about 190 writes per sample: about 2.9e-8 J per sample,
    so about 4.7e-7 J per 16-draw estimate, 9.4e-7 J at 32 draws and
    1.9e-6 J at 64 draws.
  - Continuation, 64 writes per sample: about 1.1e-8 J per sample, so
    7e-7 J at 64 draws.
  - A random start at 64 draws is therefore about 1.8x cheaper than the
    rung, close to a tie. Continuation at 64 draws is about 5x cheaper.
  - (c) compares against the archived rung, not the digital minimum. The
    ladder has nothing between 243,200 and 1.74e6 MACs.
- **`t4` does not test cost.** Its likely digital cost is near 2.4e5 MACs
  (2.4e-7 J at 1 pJ), and even its unplaced continuation estimate at 16
  draws (1.6e-7 J) is close to that.

## Fixed choices and whether each could bound the metric

| Choice | Probe setting | Could it bound the metric? |
| --- | --- | --- |
| Draws per estimate (majority vote) | 16, as P-0007 and stop (c) | **Yes, binding.** Floor +0.0095 on `t4`, +0.0144 on `smooth8`, above the 0.01 tolerance |
| Hidden encoding | bias (centered, raw), one-hot | Yes. Raw fails on `t4`. One-hot is range-bound at 8 bits and does not mix |
| Pixel coupling centering | mean template | Yes. Without it the bias design fails by 0.04 on `t4` |
| Bias b0 / inhibition λ | dev-set grid; `smooth8` grid cut to [−10, 0] | **Yes on `smooth8`.** The scan ended at its edge |
| Cap rule and bits | `split`, 4 / 6 / 8 | Yes. 4 bits fails on `t4`. One-hot's 96.8 field cap on `smooth8` |
| Tie coupling and copy count | not probed (unplaced) | Yes, untested. Copies add free-node writes and a mixing barrier |
| Placement layout | not probed | Yes. Pixel grid edges and pixel–copy edges must all be Z1 offsets |
| Sweeps K | 64, 256, 1024 | `t4`: no, at equilibrium by 256. `smooth8`: unknown |
| Start | continuation, random, reset every 5 | `t4`: no. `smooth8`: yes (continuation trapped) |
| Jitter | 0, 10 percent | No on `t4` |
| Test images | 1,000 exact, 500 sampled | Resolution. Intervals ±0.004 straddle 0.01 at 16 draws |
| Sampler seeds | one | Resolution only |

## What this means for the row

- On `t4` the exact posterior can be held at 6 to 8 bits and sampled within
  256 sweeps from any start, on the unplaced model.
- Whether it still holds once the latent is split into tied copies on the
  lattice is the study's question, and the probe did not test it.
- On `smooth8` the row's stop rules, read with P-0007's 16-draw estimate,
  close the line before any lattice question is asked. The exact posterior
  itself misses Bayes + 0.01 with 16 draws.
- The draw count is the owner's choice: it changes stop (c)'s price.
- A protocol needs that decision first. It also needs a full b0 scan for the
  `smooth8` bias design before stop (a) can be judged there.
- Rough production size (NumPy-equivalent, placed, three sampler seeds, both
  families, bits {4, 6, 8} x jitter {0, 0.1} x three starts x K in
  {64, 256, 1024}): 6 to 10 CPU-hours at 64 draws. That may exceed the
  8 CPU-hour flag. Arms whose exact equilibrium already fails (4 bits) can
  skip sampling.
