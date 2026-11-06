# Rounding-rule transfer: probe (P-0010)

2026-11-05. **Exploration, not a recorded study and not evidence.** Probe
for queue row `rounding-rule-transfer`. Every number is an
`exact_reference` equilibrium of a rounded model, computed on a CPU. No
sampling, no jitter, no training, no new layout, no cost, no hardware.

**Finding: negative for the transfer, and the probe answers the row.** On
P-0008's archived placed tie-limit model, class error diffusion cuts the
4-bit MPM gap from 0.0752 to 0.0150. That is still above the 0.01 tolerance,
and the per-image bootstrap interval, 0.0111 to 0.0191, lies above it too.
The pre-registered falsifier fires. Both rules first pass at 5 bits (nearest
0.0029, class diffusion 0.0041) and pass at every bit count from 5 up. The
archived "6 under nearest" was a grid artefact: P-0008 never ran 5. So class
diffusion does not move this model's bit requirement. It is 5 under either
rule.

## Fixed choices and whether each can bound the metric

| Choice | Value | Can it bound the metric? |
| --- | --- | --- |
| Model | P-0008's placed tie-limit model of `t4_h05_q035` on `index4` at w\* = 1.75; 189 free nodes | Yes; frozen by the row. Its coefficient classes decide what diffusion can cancel. |
| Images and floor | the 1,000 archived exact test images; Bayes MPM 0.22198 on them, recomputed and equal per image to the archive | No. They are exact and fixed. P-0008's 0.2234 is over all 1,500 images. |
| Caps | archived `split`: one coupling cap 1.75 (= w\*, the tie), one field cap 0.5 | **Yes, and it binds at 4 bits.** The tie sets the coupling step (0.25 at 4 bits), so the grid coupling J = 0.35 sits at 1.4 steps, and no rule that keeps every coupling on the codebook can put a grid edge within 0.1 of J. Changing the cap is a different rule, outside the row. |
| Rounding rule | nearest (archived) and class diffusion, fixed in the committed script before it ran | This is what the row asks. |
| Class definition | equal nominal value to 12 decimals within each `split` vector, any kind (P-0009's `CLASS_DECIMALS`); residual carried in archived vector order | Could. A class rule keyed on kind or on spatial order is a different rule; see "Mechanism". |
| Bits | 4, 5, 6, 8 (row); 3 as exploration | 5 was never run by P-0008; it is the answer. |
| Evaluator | enumeration over 16 hidden states times the transfer matrix | No. The study-local per-edge copy reproduces the imported evaluator bitwise on all four archived rows. |
| Tolerance | MPM gap at most 0.01, point estimate (P-0008 stage E rule) | By definition. |

## What ran

[`probe.py`](rounding_rule_transfer_probe/probe.py), committed as `bb6620c`
before it ran, with every fixed choice in its docstring. Output:
[`probe.json`](rounding_rule_transfer_probe/probe.json) (28,093 bytes,
SHA-256 `98b14a1383ef7235b7cb580ca73bcf87d0b3d0bc177aa174e60054a3c42d5f46`). 307 CPU-seconds, one process, about 10 s per
exact row.

1. **Authentication.** The P-0008 tarball matches `manifest.json`, every
   member matches its hash, every on-disk source matches the archived
   request's hashes, and `make_request()` rebuilds the archived request
   digest. Nothing was edited. `diffuse` is imported unedited from P-0009's
   runner.
2. **Control.** The archived rule reproduces the archived float, 4-, 6- and
   8-bit rows exactly: identical per-image error counts, identical MPM gaps
   and 64-draw gaps to 0.0. The study-local evaluator (per-edge grid and
   per-pixel clamp couplings, which class diffusion needs) matches the
   imported one bitwise on those rows.
3. **Grid.** Both rules at 3, 4, 5, 6 and 8 bits.
4. **Mechanism at 4 bits.** Class tables, and one-kind ablations: one
   coefficient kind rounded, everything else float.
5. **am-coupling-bits class count** on all 16 held sets of the six cells.
6. **Adsorption pore at 2 and 3 bits**, through P-0009's own `evaluate_cell`.

A post-hoc spread check,
[`posthoc_spread.py`](rounding_rule_transfer_probe/posthoc_spread.py), was
written after `probe.json` existed and took about a minute. Output:
[`posthoc_spread.json`](rounding_rule_transfer_probe/posthoc_spread.json).
It recomputes four rows and bootstraps over images (2,000 resamples). It
does not change the reading.

## Result

MPM gap against Bayes on 1,000 exact images; tolerance 0.01.

| Bits | nearest | class diffusion |
| --- | --- | --- |
| 3 (exploration) | 0.0478 | 0.1199 |
| 4 | **0.0752** (archived) | **0.0150** (95% bootstrap 0.0111 to 0.0191) |
| 5 | 0.0029 (0.0008 to 0.0050) | 0.0041 (0.0017 to 0.0066) |
| 6 | 0.0025 (archived) | 0.0003 |
| 8 | 0.0003 (archived) | 0.0004 |

- Smallest passing bit count: 5 under both rules. Both also pass at 6 and
  8, so 5 is the stable count as well.
- Non-monotone, reported as found: nearest is better at 3 bits than at 4.
  Class diffusion is 0.0001 worse at 8 than at 6.
- The 64-draw gaps follow the same pattern (class diffusion 0.0169 at 4,
  0.0081 at 5). They do not gate.

## Mechanism: right about which classes, wrong about the sum

The 4-bit coupling step is 0.25 and the field step is 0.0714. Classes whose
nearest rounding is coherent, largest summed error first:

| Class | Size | Nominal (steps) | Nearest, summed error (steps) | Diffusion (steps) |
| --- | --- | --- | --- | --- |
| grid J | 112 | 1.4 | −44.8 (every edge 0.35 → 0.25) | +0.2 |
| copy ±0.125 | 36 + 36 | ±0.5 | ∓18.0 each (all round to 0, half to even) | 0.0 |
| clamp g | 64 | 1.24 | −15.2 | −0.2 |
| copy bias, four classes | 27 to 28 | −0.16 to −1.84 | −6.4 to +8.3 | at most 0.4 |
| pixel ±0.25 | 12 + 12 | ±3.5 | ±6.0 | 0.0 |
| ties | 121 | 7.0 | 0 (on codebook) | 0 |

So the row's prediction about where the miss comes from holds. Every large
class rounds coherently under nearest, and diffusion brings every class sum
within half a step. The one-kind ablations show which classes matter:

| Kind rounded at 4 bits alone | nearest | class diffusion |
| --- | --- | --- |
| copy couplings | 0.0452 | 0.0028 |
| grid couplings | 0.0202 | **0.0118** |
| copy biases | 0.0092 | −0.0001 |
| clamp couplings | 0.0017 | 0.0043 |
| pixel fields | 0.0018 | 0.0000 |
| ties | 0.0001 | 0.0001 |

Diffusion removes the copy-coupling and copy-bias errors, which act through
sums (each pixel couples to one copy per unit, and the tie limit sums copy
biases). It does not remove the grid error, because the grid does not
respond to the class mean alone. Diffused, the 112 grid
couplings take two levels, 0.25 and 0.5 (three in five at 0.25), in the
archived edge order. Their mean is 0.350 but each edge is off by up to 43
percent, and an Ising grid's
marginals respond to the spatial pattern of J, not to its mean. Diffusion
also makes the clamp worse than nearest (0.0043 against 0.0017), for the
same reason. A class sum within half a step is enough when the observable is
linear in the class sum. On P-0009's pore it was, to first order. Here it is
not for the grid and clamp classes, and the grid alone leaves 0.0118.

What bounds 4 bits is the shared `split` coupling cap. The ties (1.75) set
the step, and at that step J = 0.35 has no codebook level within 0.1. That
is a cap problem, which the row excludes, and not a rounding-rule problem.

## am-coupling-bits: class sizes (counting only, no recall evaluated)

Stage A's bias design, `split` caps, 16 held sets per cell:

- Couplings: two classes (±e/2) of 96 to 1,600 entries. They sit at ±cap, so
  they are exact at every bit count. Diffusion has nothing to do there.
- Fields: 5 to 18 classes. At P = 128 the 128 hidden biases form one class
  that rounds coherently at 6, 8 and 10 bits, summed error 51 to 63 steps.
  At P = 8 and 32 the off-codebook classes have 7 to 11 entries and summed
  error under 4 steps.

So that archive has one large coherent class, the hidden biases at
P = 128. Its `split` requirement of 10 is an interval-width result (D-0008),
and its hidden biases enter recall through each hidden unit's own log cosh
term, not through a sum. Re-reading it would be a new row; this probe
does not.

## Adsorption pore at 2 and 3 bits (exploration)

On the three frozen chemical potentials, under P-0009's tolerances:

| Rule | 2 bits | 3 bits | 4 bits |
| --- | --- | --- | --- |
| class diffusion | fail (loading 0.139) | fail (loading 0.035, pair 0.026) | pass (0.0024) |
| site diffusion | fail | fail (loading 0.014) | pass |
| nearest, family cap | fail | fail (loading 0.194) | fail (0.018) |

P-0009's "at most 4 field bits" is exactly 4 under class diffusion. That
closes D-0008's first caveat.

## Reading

1. **The row's design number.** On P-0008's model the smallest bit count
   within Bayes + 0.01 is 5 under class diffusion and 5 under nearest. It
   is not the archived 6, which came from a grid with no 5. Class diffusion
   does not change it.
2. **The falsifier fires.** The 4-bit gap under class diffusion is 0.0150 and
   its interval excludes 0.01. Under the row's own text, this scopes P-0009's
   rule to models whose observable is close to linear in class sums, like the
   field-dominated pore. It is not a general lever.
3. **Decision the row informs.** Keep nearest rounding as the default of
   `thermo_lab.constraints.quantize`. Class diffusion is a per-model compile
   option. Use it where the large classes act through sums (fields on many
   sites, copy couplings in a tie limit), and check it exactly before use.
   It needs no re-reading of the archived requirements except P-0008's,
   which is 5 bits under either rule, not 6.
4. **What would move 4 bits here** is a cap that does not let the tie set
   the grid step (a per-kind cap), not a rounding rule. That is a
   different question and not a queue row. No proposal is made.

Nothing here is a hardware claim. All rounding is a host-side compile step.

## Not done

No protocol. The probe ran every cell the row names on the archived images,
exactly and deterministically. A recorded study would recompute the same
numbers. No per-kind cap, no spatial-order class rule and no
am-coupling-bits recall were evaluated. No source file or archive was
edited.
