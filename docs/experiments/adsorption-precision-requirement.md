# Adsorption precision requirement: exact protocol (draft, proposal P-0009)

Drafted on 2026-10-23 for queue row `adsorption-precision-requirement`. It
is not frozen until the owner approves it, and no runner exists yet. The
[probe](../research/2026-10-23-adsorption-precision-requirement-probe.md)
is exploration on development chemical potentials only.

## The number this study produces

For the held `native-adsorption-eligibility` model at its three frozen
chemical potentials, per rounding rule: **b\***, the smallest field bit
count in 4 to 12 at which the rounded model's exact equilibrium loading and
working capacity are within 0.01 of full precision and its aggregate pair
occupancy within 0.02, at all three points, or "none within 12 bits". Beside
it, **b_stable**, the smallest bit count from which the rule passes at every
higher count up to 12.

A designer reads it as: "this pore needs b\* field bits under rule R; under
plain nearest rounding it needs b\*(nearest)". The difference between rules
is the value of choosing the rounding rule at compile time.

## Question

How many coefficient bits, and which rounding rule, keep the exact
equilibrium observables of the held adsorption model within tolerance at
every frozen chemical potential?

## Hardware constraints applied and relaxed (rule 8)

- **Applied.** Codebook quantization (`thermo_lab.constraints.quantize`,
  2^(b-1) - 1 symmetric levels per sign, clipping) of every coupling and
  field. Placement is the archived map (x + 4z, y + z) with offsets (1,0),
  (0,1), (4,1), no copies; the exact result does not depend on it, and no
  coefficient changes.
- **Assumed, not a relaxation.** Separate coupling and field scales (split
  caps), as in the archived probe. The per-point-cap rule further assumes
  the host can reprogram the field scale for each chemical potential; the
  report states this beside that rule's number.
- **Relaxed.** Beta jitter (zero) and sampling (none): the row asks for the
  representation requirement alone. The finite-budget sampling question
  stays with the held row.
- **Outside the layer.** Error diffusion is a host-side compile step that
  chooses which codebook level each site gets. It needs no hardware feature.

## Resource accounting (rule 9)

No comparison of costs is made. The only resource is field bits per site
(couplings are exact at every bit count because J = 0.25 equals the coupling
cap). CPU seconds are reported and do not gate. This is not a task row, so
the queue's per-sample I/O and MAC rules do not apply; the study reports no
joule.

## Fixed choices and what they bound

- **Model**, frozen by the row: 3×3×16 open lattice gas, beta = epsilon = 1,
  lateral wall phi = 3, archived spin map J = 1/4,
  h_i = (mu - phi_i)/2 + degree_i/4. *Bounds:* its susceptibility converts
  field error into loading error; the middle point (d loading/d mu = 0.780)
  binds.
- **Chemical potentials**, read from the archived probe JSON (SHA-256
  `c4db957d7f672eb4e22400eaf610a7db549125871fe193e5da1b051f2a12a7ef`): -0.1644052920687611, 0.4847198442982543,
  0.9808189532527588. Capacity is high minus low.
- **Caps.** Coupling 0.25 (exact). Family field cap 1.9904094766263793
  (archived; it is the high point's own max |h|). Per-point field cap: max
  |h| of that point's own model: 1.41780, 1.74236, 1.99041.
- **Rules**, fixed by this protocol from the probe's development points
  (field arrays in site order z, then y, then x):
  1. `nearest_family`: `quantize_array(h, family_cap, b)`. The archived rule
     and the control.
  2. `diffuse_class_family`: the primary error-diffusion rule. Group sites
     by nominal field value (rounded to 12 decimals). In site order, add the
     class's carried residual to h_i, take the family-cap codebook value,
     carry the new residual within the class only.
  3. `nearest_point`: `quantize_array(h, per_point_cap, b)`.
  4. `diffuse_site_family`: diagnostic. The same diffusion with one residual
     carried across all sites. Reported with a b\*, not recommended (the
     probe chose class diffusion over it after seeing development errors).
- **Bits** 4 to 12. *Bounds:* a requirement above 12 bits would read "none";
  the first-order bound (below) passes by 9 for any rule.
- **Tolerances**: loading and capacity 0.01, pair 0.02, absolute, from the
  row. Screen thresholds, not validated material requirements.
- **Reference**: the archived 512-state factorized transfer
  (`docs/research/native_adsorption_probe/adsorption_reference.py`,
  `exact_aggregate`), imported unedited and authenticated by SHA-256.

## Pre-registered expectations (from the probe)

1. The control reproduces the archived 6- and 8-bit loadings and pair
   occupancies at the frozen points to 1e-12 (probe: 1.1e-15).
2. `nearest_family`: b\* = 9 (fails 8 at the middle point by 0.0011, as
   archived). First-order bound: any rule with per-site error at most
   step/2 passes at 9 bits.
3. `diffuse_class_family`: b\* = 5 (development: 5; fails 4 on capacity).
4. `nearest_point`: b\* = 7 or 8 (development: 7). Its high point uses the
   family cap, so it differs from the control only at the low and middle
   points.
5. `diffuse_site_family`: b\* = 5 or 6.
6. Non-monotone error in bits will appear for the nearest rules; the report
   gives every cell, not a smoothed curve.

## Exact anchor and evidence classes

Every observable is `exact_reference`: the exact equilibrium of a rounded
model, compared with the exact equilibrium of the unrounded model at the same
chemical potential. Nothing is `software_simulation`; nothing is hardware
evidence. The linear-response bound is analysis, labelled as such.

## Arms and controls

- 4 rules × 9 bit counts × 3 frozen points = 108 exact cells, plus 3
  full-precision cells.
- **Control (gating):** `nearest_family` at 6 and 8 bits matches the archive
  to 1e-12.
- **Secondary held-out grid (reported, non-gating):** six fresh chemical
  potentials solved at full precision for loadings 0.15, 0.3, 0.4, 0.6, 0.7,
  0.85 (Brent on [-10, 10], xtol 1e-12), disjoint from the probe's
  development loadings and the frozen ones; capacity is 0.85 minus 0.15.
  Family cap stays the archived one; a point whose max |h| exceeds it is
  clipped, and the report counts clipped sites. 216 cells. This tests whether
  a rule's b\* holds away from the points it is read on.
- **Checks:** couplings equal their nominal value in every cell; every
  rounded field lies on its codebook; per-class residual of the diffusion
  rules is at most step/2 when no site clips.

## Primary decision rule (thresholds fixed now)

1. For each rule and bit count, a cell set passes when, over the three
   frozen points, max |loading error| ≤ 0.01, |capacity error| ≤ 0.01 and
   max |pair error| ≤ 0.02.
2. b\* is the smallest passing bit count; b_stable the smallest bit count
   from which every higher count passes; "none" if no count passes.
3. **Recommended rule**: the rule with the smallest b_stable among rules 1
   to 3 whose secondary held-out grid also passes at that b_stable; ties go
   to the lower-numbered rule (the simpler one). If no rule passes the
   secondary grid at its own b_stable, the report recommends
   `nearest_family` at its b_stable and says the other rules did not
   generalize. Every arm is exact, so every arm resolves.
4. A rule that passes on the frozen points but fails the secondary grid at
   the same bits is reported as "passes the frozen points only".

## What one recorded sample means

There are no samples. One recorded cell is one exact transfer evaluation of
(loading, aggregate pair occupancy) for one rule, bit count and chemical
potential.

## Seeds

None. Every computation is deterministic. No archived test seed is used.

## Budgets and CPU estimate

Probe: median 0.013 CPU-s per exact cell. 327 cells plus Brent solves and
replay: **under 1 CPU-minute** (0.02 CPU-hours). No autosave or `--resume`
is needed (rule: studies over about 30 minutes).

## Preflight (gating)

1. Archived probe JSON and reference helper hashes match.
2. The reference helper's `--self-test` passes (24 enumeration cases).
3. Control expectation 1.
4. Diffusion rules: on a hand-built six-site field vector, the class and
   site residual sequences match a hand computation.

## Persistence and archive size

`study.json.gz`: the hashed request (rules, bits, caps, chemical potentials,
tolerances, helper and archive hashes); preflight; every cell's loading,
pair, errors, max and mean field error and clipped-site count; rounded field
arrays by SHA-256 only; verdicts. Under 50 KB gzipped. CPU timings sit
outside the result digest.

Replay recomputes every cell from the request and compares at 1e-12
absolute, then recomputes every verdict. Full replay takes seconds, so CI
replays in full.

## Gate

`completion.json`, written last by the runner, must show:
`status=adsorption_precision_requirement_complete`, `rules=4`, `bits=9`,
`frozen_cells=108`, `secondary_cells=216`, `control_passed`,
`preflight_passed`, `replayed` all `true`. It also carries b\* and b_stable
per rule on both sets, the recommended rule and its b_stable. Those are
scientific results, not integrity checks.

Proposed runner: `uv run python -m thermo_lab.adsorption_precision_requirement
--output-dir results/adsorption-precision-requirement`, study-local code
importing the archived helper.

## Stop rules

- Preflight or control fails: stop before any new cell. Do not tune.
- A replay mismatch is an integrity failure.
- Two consecutive failed runs with the same error: stop and report.

## What a negative result means

- **`nearest_family` has b\* > 8.** The pore needs more than 8 field bits
  under plain nearest rounding on family caps; the archived 8-bit miss is a
  precision requirement, not bad luck at one point.
- **No rule passes within 12 bits.** This pore's observables cannot be held
  to the screen tolerance by codebook fields. Expected to be false (the
  first-order bound passes at 9).
- **A diffusion rule passes the frozen points but not the secondary grid.**
  Its advantage is specific to the points it was read on; the design number
  is the nearest rule's.
- None of these is an integrity failure.

## Not claimed

No sampled, finite-budget or jittered result. No hardware or energy number.
No claim beyond this model, these chemical potentials and these tolerances.
A pass does not unhold `native-adsorption-eligibility` and does not
authorize sampling; the owner decides. P-0008's tie-limit gaps (0.0752,
0.0025, 0.0003 at 4, 6 and 8 bits) are cited for comparison only; that miss
was finite tie strength, not precision.
