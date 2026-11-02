# Adsorption precision requirement: findings

**Under class error diffusion on the family caps, the held adsorption model
needs 4 field bits. Under plain nearest rounding on the same caps it needs
9.** The frozen decision rule recommends `diffuse_class_family` at
b_stable = 4. That is an upper bound, because 4 is the lowest bit count the
protocol tested. All 108 frozen cells and 216 secondary cells are exact, so
every arm resolved. On the frozen points, nearest rounding first passes at 6
bits, fails again at 8 (the archived miss, 0.01108 at loading 0.5) and passes
from 9 bits on. The archived 8-bit failure is therefore a real precision
requirement for nearest rounding, not bad luck at one point. Per-point caps
need 8 bits. All four rules generalize: each one also passes the secondary
grid of six fresh chemical potentials at its own b_stable.

Protocol: [docs/experiments/adsorption-precision-requirement.md](../../experiments/adsorption-precision-requirement.md),
approved at 47e20eb (proposal P-0009, PR #135). Runner:
`thermo_lab.adsorption_precision_requirement`, run from c2d37ca with a clean
tree (`git_dirty=false`). Wall time was 11.7 s, of which replay took 5.8 s.
The 324 exact cells took a median of 0.013 CPU-s each.

## Evidence classes

- Every loading, pair occupancy and error is `exact_reference`. Each one is
  the 512-state transfer equilibrium of a rounded model, compared with the
  unrounded model at the same chemical potential. The transfer comes from the
  archived helper, imported unedited and checked by SHA-256.
- The first-order linear-response bound is analysis, not evidence.
- No sampling, jitter, seeds or cost are involved, and nothing here is
  hardware evidence. This is not a task row, so the queue's per-sample I/O
  and per-MAC rules do not apply and no joule is reported.

## What ran

| Stage | Content | Result |
| --- | --- | --- |
| Preflight | three source hashes, the helper's `--self-test`, the control, the hand-built diffusion check | passed. The self-test ran 24 enumeration cases. The control matches the archive to 1.1e-15 at 6 and 8 bits on all three points. |
| Frozen | 4 rules × 9 bit counts × 3 points = 108 cells, plus 3 at full precision | below |
| Secondary | 4 rules × 9 × 6 = 216 cells at loadings 0.15, 0.3, 0.4, 0.6, 0.7, 0.85 (Brent solves) | below |
| Checks | couplings exact (J = 0.25 equals the cap), fields on the codebook, per-class running residual within step/2 wherever no site clips | all 324 cells pass |
| Replay | every cell, full-precision value, solve, verdict, decision and the analysis block recomputed | passed |

## Design numbers (frozen points; the primary result)

| Rule | b\* | b_stable | Bits that fail | Secondary b\* / b_stable | Label |
| --- | --- | --- | --- | --- | --- |
| `nearest_family` (control) | 6 | **9** | 4, 5, 8 | 8 / 8 | passes the frozen points and the secondary grid |
| `diffuse_class_family` | 4 | **4** (grid floor) | none | 4 / 4 | passes the frozen points and the secondary grid |
| `nearest_point` | 8 | **8** | 4 to 7 | 7 / 7 | passes the frozen points and the secondary grid |
| `diffuse_site_family` (diagnostic) | 4 | 4 (grid floor) | none | 4 / 4 | passes the frozen points and the secondary grid; not recommendable |

Under `nearest_point` the host reprograms the field scale for each chemical
potential (protocol, "Assumed"). Its 8 bits depend on that.

The frozen points, max |error| over three points (loading / capacity / pair):

| Bits | `nearest_family` | `diffuse_class_family` | `nearest_point` |
| ---: | --- | --- | --- |
| 4 | 0.0183 / 0.0015 / 0.0175 | 0.0024 / 0.0041 / 0.0028 | 0.0169 / 0.0286 / 0.0170 |
| 5 | 0.0090 / 0.0129 / 0.0127 | 0.0018 / 0.0036 / 0.0021 | 0.0276 / 0.0171 / 0.0284 |
| 6 | 0.0077 / 0.0050 / 0.0106 | 0.0005 / 0.0002 / 0.0006 | 0.0116 / 0.0064 / 0.0115 |
| 7 | 0.0093 / 0.0082 / 0.0098 | 0.0003 / 0.0001 / 0.0003 | 0.0128 / 0.0086 / 0.0133 |
| 8 | **0.0111** / 0.0050 / 0.0116 | 0.0001 / 0.0001 / 0.0001 | 0.0067 / 0.0038 / 0.0071 |
| 9 | 0.0012 / 0.0023 / 0.0016 | 0.0000 / 0.0000 / 0.0000 | 0.0012 / 0.0018 / 0.0016 |

Bits 10 to 12 pass for every rule. The runner's
[summary](summary.md) lists every rule at every bit count on both sets,
and the archive holds every cell.

## Findings

1. **The rounding rule sets the bit budget, and the step size does not.** The
   first-order bound (analysis: per-site error at most step/2 and
   nonnegative susceptibilities) is the same for every rule and first passes
   at 9 bits. At the middle point it gives 0.0122 at 8 bits and 0.0061 at 9.
   Nearest rounding on the family caps tracks the bound: at 8 bits it reaches
   0.0111, 91% of the bound, because all 128 wall fields round the same way
   (archived rounding audit).
   Class diffusion keeps each class's mean field within step/(2 × class size)
   of nominal. That cancels the coherent shift, so at 4 bits its worst
   loading error is 0.0024, against a bound of 0.22.
2. **Nearest rounding is non-monotone in bits, as pre-registered.** On the
   frozen points it passes at 6 and 7 bits, fails at 8 and passes again from
   9. Its b\* of 6 is therefore not a design number; b_stable = 9 is.
   `nearest_point` is also non-monotone below 8 bits (5 is worse than 4 at the
   middle point). The cells are reported as found, not smoothed.
3. **Class diffusion generalizes, with less margin away from the frozen
   points.** On the secondary grid at 4 bits its worst errors are 0.0071
   loading, 0.0071 capacity and 0.0089 pair, against tolerances of 0.01 and
   0.02. On the frozen points they are 0.0024 / 0.0041 / 0.0028.
4. **The secondary 0.85 point clips.** Its max |h| is 2.075, above the
   archived family cap of 1.990. The family-cap rules therefore clip 13 or 14
   sites there at most bit counts: the 14 interior sites of the centre
   column. `nearest_family` clips them from 5 bits, class diffusion from 4
   (13 at 4 bits) and site diffusion fully from 6 (4 and 9 sites before
   that). Site diffusion is
   the rule this hurts. Its residual is carried across all sites, the clipped
   remainder accumulates, and its secondary error flattens at about 0.0027
   loading and 0.0041 pair from 7 to 12 bits instead of falling. It still
   passes. Class diffusion confines the clipped remainder to the clipped
   class and reaches 6e-5 at 12 bits. The frozen points never clip, because
   the family cap is the high point's own max |h|.
5. **The pre-registered expectations missed in three places.** None of these
   misses is an integrity failure:
   - expectation 1, the control, held;
   - expectation 2 predicted b\* = 9 for `nearest_family`. The result is
     b\* = 6 and b_stable = 9, because the rule passes at 6 and 7 bits before
     it fails at 8. The bound did pass first at 9, as expected;
   - expectation 3 predicted b\* = 5 for class diffusion, and it is 4. The
     development points failed 4 bits on capacity (0.0131); the frozen
     points pass at 0.0041;
   - expectation 4 predicted `nearest_point` at 7 or 8, and it is 8;
   - expectation 5 predicted 5 or 6 for site diffusion, and it is 4;
   - expectation 6 predicted non-monotone nearest rules, and they are.

## What a negative would have meant, and what fired

- "`nearest_family` has b\* > 8." Literally this does not fire, because
  b\* = 6. The stable requirement is 9, though, and the archived 8-bit miss
  reproduces. Under plain nearest rounding on family caps, this pore needs
  more than 8 field bits to hold at every higher count. The archived miss is
  a precision requirement, not bad luck at one point.
- "No rule passes within 12 bits." This does not fire.
- "A diffusion rule passes the frozen points but not the secondary grid."
  This does not fire.

## What a reader should use

Use 4 field bits with class error diffusion on the archived family split
caps (coupling cap 0.25, field cap 1.9904094766263793). Error diffusion is a
host-side compile step and needs no hardware feature. Without it, use
9 bits of nearest rounding. With per-point caps and a reprogrammable field
scale, use 8 bits. The 4-bit figure is an upper bound: the study did not
test fewer bits. Couplings are exact at every bit count tested here, because
J equals the coupling cap.

## Not claimed

- No sampled, finite-budget or jittered result, and no hardware or energy
  number.
- No claim beyond this model, these chemical potentials and these
  tolerances, which are screen thresholds rather than validated material
  requirements.
- A pass does not unhold `native-adsorption-eligibility` and does not
  authorize sampling. That decision is the owner's.
- P-0008's tie-limit gaps (0.0752, 0.0025 and 0.0003 at 4, 6 and 8 bits)
  are cited for comparison only. That miss came from finite tie strength,
  not from precision.

## Reproduce and replay

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python -m thermo_lab.adsorption_precision_requirement \
  --output-dir results/adsorption-precision-requirement-new
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python -m thermo_lab.adsorption_precision_requirement \
  --replay docs/experiment-reports/2026-11-02-adsorption-precision-requirement/study.json.gz \
  --output-dir results/adsorption-precision-requirement-replay
```

Archive: `study.json.gz`, 39,680 bytes, SHA-256
`84a7372f29c350d05613a1a722e3a8f6eeb71387bb8b987d388c26d76e6cfb33`.
Result digest
`sha256:0f923a4b69a25b0553ce657d1b134d565b7370fcdc799de2605d0fec3f5cd867`.
The figure is `error-versus-bits.png`, made with
`uv run --no-project --with matplotlib python plot.py`.
