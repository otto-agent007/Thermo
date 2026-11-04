# Adsorption precision requirement: probe (P-0009)

2026-10-23. **Exploration, not a recorded study and not evidence.** Probe
for queue row `adsorption-precision-requirement`, under the loop's
preliminary-probe allowance. Every number here is an `exact_reference`
equilibrium of a rounded model, computed on a CPU. No sampling, no jitter,
no cost, no hardware.

**Finding.** Where rounding errors point decides the bit budget, not their
size. On six development chemical potentials, nearest rounding on the
archived family caps first passes every tolerance at 9 bits. Error diffusion
within field classes on the same caps passes at 5, and nearest rounding on
per-point caps passes at 7. The three frozen chemical potentials were
deliberately not evaluated under any new rule. They are the held-out test
for the protocol, so the probe does not answer the row outright.

## Fixed choices and whether each can bound the metric

| Choice | Value | Can it bound the metric? |
| --- | --- | --- |
| Model | 3×3×16 open lattice gas, beta = epsilon = 1, wall phi = 3, archived spin map (J = 1/4, h = (mu - phi)/2 + degree/4) | Yes. Its susceptibility sets the gain from field error to loading error (below). Frozen by the row. |
| Chemical potentials | The three archived values (loadings 0.2, 0.5, 0.8) | Yes. The middle point has the largest d(loading)/d(mu), 0.780, so it binds. Frozen by the row. |
| Coupling cap | 0.25 = J | No. J lies on the codebook at every bit count, so couplings are exact (checked per cell). |
| Field cap, family | 1.9904094766263793, max \|h\| over the three frozen points | Yes. It sets the step: cap/(2^(b-1) - 1). |
| Field cap, per point | max \|h\| of that point's own model | Yes, by the same mechanism; it only shrinks the step at low mu (1.51 to 1.93 on the development points). |
| Rounding rule | nearest, two error-diffusion variants, per-point nearest | Yes. This is what the probe found binds (below). |
| Bits | 4 to 12 | A 12-bit ceiling could hide a requirement; it does not here (every rule passes by 9). |
| Tolerances | loading and capacity 0.01, pair 0.02, absolute | Yes, by definition. Screen thresholds, not validated material requirements (archived probe). |
| Placement | archived map (x + 4z, y + z), no copies | No. The exact reference is placement-independent; placement changes no coefficient. |
| Reference | 512-state factorized transfer (archived helper, imported unedited, hash-checked) | No. Exact to about 1e-15 against the archive. |

## What ran

[`precision_probe.py`](adsorption_precision_probe/precision_probe.py),
committed as `d7a3bf8` before it was run, with its fixed choices in the
module docstring. Output: [`probe.json`](adsorption_precision_probe/probe.json)
(94,344 bytes, SHA-256
`3e7b53d97e9475324a89d586d08fb988cb1ac29d1563fdfe6b5ba17523f63c33`).
4.7 CPU-seconds in total; 216 exact cells at a median 0.013 CPU-s each.

1. **Control.** The archived rule (codebook nearest on the family cap) at 6
   and 8 bits on the three frozen points reproduces the archived loadings
   and pair occupancies to 1.1e-15. These numbers were already public; no
   other rule or bit count touched the frozen points.
2. **Linear response on the frozen points**, nominal model only (central
   difference, step 1e-5): d(loading)/d(mu) = 0.215, 0.780, 0.355. A field
   error delta acts as a chemical-potential shift 2·delta, and every site
   susceptibility is nonnegative in this ferromagnetic model, so a rounding
   with per-site error at most step/2 moves loading by at most
   step · d(loading)/d(mu) to first order. That bound first meets every
   tolerance at **9 bits** (middle-point loading bound 0.0061; at 8 bits it
   is 0.0122). Nearest rounding at 8 bits reached 0.01108, 91% of its bound,
   because all 128 wall fields rounded upward (archived rounding audit).
3. **Development**: four rules × bits 4 to 12 on six fresh chemical
   potentials solved from the full-precision model at loadings 0.25, 0.35,
   ..., 0.75. Capacity is the 0.75 point minus the 0.25 point.

## Development result (exploration)

Largest absolute error over the six points; tolerance 0.01 loading and
capacity, 0.02 pair. Bold: first passing bit count.

| Bits | nearest, family cap | diffuse in site order, family cap | diffuse within field class, family cap | nearest, per-point cap |
| ---: | --- | --- | --- | --- |
| 4 | 0.151 / 0.099 / 0.169 | 0.013 / 0.012 / 0.012 | 0.007 / 0.013 / 0.009 | 0.148 / 0.037 / 0.163 |
| 5 | 0.052 / 0.030 / 0.065 | **0.003 / 0.004 / 0.004** | **0.001 / 0.002 / 0.001** | 0.050 / 0.058 / 0.064 |
| 6 | 0.040 / 0.019 / 0.043 | 0.008 / 0.000 / 0.008 | 0.001 / 0.000 / 0.001 | 0.019 / 0.011 / 0.021 |
| 7 | 0.011 / 0.001 / 0.011 | 0.001 / 0.001 / 0.001 | 0.000 / 0.000 / 0.000 | **0.007 / 0.008 / 0.009** |
| 8 | 0.010 / 0.007 / 0.011 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.004 / 0.000 / 0.005 |
| 9 | **0.003 / 0.001 / 0.003** | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.003 / 0.003 / 0.003 |

Cells are loading / capacity / pair. Bits 10 to 12 pass for every rule.

- Nearest on the family cap misses at 8 bits by 0.0003 (0.01029 at the
  0.55 point). That is the archive's failure mode on a fresh point.
- Error is non-monotone in bits, reported as found. Nearest at 6 bits misses
  loading on 4 of 6 points and at 8 bits on 1 of 6; site-order diffusion
  gives 0.0032 at 5 bits and 0.0076 at 6.
- Site-order diffusion fails at 4 bits on loading at the 0.35 point
  (0.0133). Class diffusion fails at 4 bits on capacity only (0.0131), with
  every loading inside 0.0071.
- Class diffusion carries each residual within sites that share a nominal
  field, so each class's mean field stays within step/(2·class size) of
  nominal. That cancels the coherent wall shift the rounding audit found.
  Site-order diffusion lets the residual cross from wall to centre classes,
  where susceptibilities differ, so it cancels less well.
- Every rule passes at every bit count from its first pass up to 12 on
  development; no rule passes and then fails again within 12 bits.

## What this sizes

- The fixed choice that bounds the metric is the **rounding rule**, through
  the sign pattern of field errors against site susceptibility. Step size
  alone (the linear bound) predicts 9 bits for any rule; class diffusion
  beats that by four bits.
- The recorded study is tiny: 4 rules × 9 bit counts × 3 frozen points, plus
  a secondary held-out grid, under a CPU-minute. No autosave is needed.
- The probe chose the error-diffusion variant (class, not site order) after
  seeing development errors. That choice is development; the frozen points
  and a second fresh grid test it.

## Comparison, not rerun

P-0008's tie-limit gaps at 4, 6 and 8 bits (0.0752, 0.0025, 0.0003) are a
different model and observable. Its miss was finite tie strength, not
precision ([lessons, 2026-10-17](../knowledge/lessons.md)). Here precision
itself binds, and the rounding rule sets how much.

## Not claimed

No sampled or hardware result. A pass does not unhold
`native-adsorption-eligibility` and does not authorize sampling. The
tolerances are screen thresholds. No claim beyond this model and these
chemical potentials.

Reproduce (refuses an existing output path):

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python docs/research/adsorption_precision_probe/precision_probe.py \
  --output <fresh path>.json
```
