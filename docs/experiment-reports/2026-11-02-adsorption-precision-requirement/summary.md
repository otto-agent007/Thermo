# Adsorption precision requirement (P-0009): summary

**Under class error diffusion on the family caps, the held adsorption model
needs 4 field bits. Under plain nearest rounding it needs 9.** The frozen
rule recommends `diffuse_class_family` at b_stable = 4. Every arm is exact
and every arm resolved. 4 bits is the grid floor, so 4 is an upper bound.

- **Design numbers.** These are b_stable values on the three frozen chemical
  potentials. Tolerances are 0.01 for loading and capacity and 0.02 for pair
  occupancy.
  - `nearest_family`: 9. It passes at 6 and 7, fails at 8 and passes from 9.
    The archived 8-bit miss, 0.0111 at loading 0.5, reproduces.
  - `diffuse_class_family`: 4.
  - `nearest_point`: 8. This assumes a per-point field scale.
  - `diffuse_site_family`: 4. It is diagnostic and not recommendable.
- **Generalization.** Every rule also passes six fresh chemical potentials
  (loadings 0.15 to 0.85) at its own b_stable. Class diffusion's worst error
  there at 4 bits is 0.0089 pair and 0.0071 loading.
- **Mechanism.** The first-order bound (analysis) is the same for every rule
  and passes at 9 bits. Nearest rounding reaches 91% of it at 8 bits because
  the wall fields round coherently. Class diffusion cancels that coherent
  shift.
- **Against the pre-registration.** The control and the 9-bit bound held. The
  b\* expectations for nearest (9), class diffusion (5) and site diffusion
  (5 or 6) did not: the results are 6, 4 and 4. Details are in
  [findings](findings.md).
- **Evidence.** Every number is `exact_reference` (512-state transfer, CPU,
  float64). No sampling, jitter, cost or hardware claim. The result does not
  unhold `native-adsorption-eligibility`.
- **Integrity.** Preflight passed, including the 24-case self-test and a
  control at 1.1e-15. All 324 cells pass the coupling, codebook and residual
  checks, and replay passed. The run took 11.7 s wall from clean commit
  c2d37ca.

## Runner tables

Request `sha256:97b2b5ac1ec1951de8f728b08f2df56e20a6bb45f55ecf1272e4b80df7ebfdab`;
result `sha256:0f923a4b69a25b0553ce657d1b134d565b7370fcdc799de2605d0fec3f5cd867`.

| rule | b* frozen | b_stable frozen | b* secondary | b_stable secondary | label |
| --- | --- | --- | --- | --- | --- |
| `nearest_family` | 6 | 9 | 8 | 8 | passes the frozen points and the secondary grid at its b_stable |
| `diffuse_class_family` | 4 | 4 | 4 | 4 | passes the frozen points and the secondary grid at its b_stable |
| `nearest_point` | 8 | 8 | 7 | 7 | passes the frozen points and the secondary grid at its b_stable |
| `diffuse_site_family` | 4 | 4 | 4 | 4 | passes the frozen points and the secondary grid at its b_stable |

## Frozen points: max loading / capacity / max pair error (pass marked *)

| bits | `nearest_family` | `diffuse_class_family` | `nearest_point` | `diffuse_site_family` |
| --- | --- | --- | --- | --- |
| 4 | 0.0183 / 0.0015 / 0.0175 | 0.0024 / 0.0041 / 0.0028* | 0.0169 / 0.0286 / 0.0170 | 0.0021 / 0.0033 / 0.0028* |
| 5 | 0.0090 / 0.0129 / 0.0127 | 0.0018 / 0.0036 / 0.0021* | 0.0276 / 0.0171 / 0.0284 | 0.0013 / 0.0022 / 0.0013* |
| 6 | 0.0077 / 0.0050 / 0.0106* | 0.0005 / 0.0002 / 0.0006* | 0.0116 / 0.0064 / 0.0115 | 0.0005 / 0.0000 / 0.0005* |
| 7 | 0.0093 / 0.0082 / 0.0098* | 0.0003 / 0.0001 / 0.0003* | 0.0128 / 0.0086 / 0.0133 | 0.0002 / 0.0001 / 0.0002* |
| 8 | 0.0111 / 0.0050 / 0.0116 | 0.0001 / 0.0001 / 0.0001* | 0.0067 / 0.0038 / 0.0071* | 0.0001 / 0.0002 / 0.0002* |
| 9 | 0.0012 / 0.0023 / 0.0016* | 0.0000 / 0.0000 / 0.0000* | 0.0012 / 0.0018 / 0.0016* | 0.0001 / 0.0001 / 0.0001* |
| 10 | 0.0009 / 0.0004 / 0.0012* | 0.0000 / 0.0000 / 0.0000* | 0.0009 / 0.0011 / 0.0012* | 0.0001 / 0.0000 / 0.0001* |
| 11 | 0.0005 / 0.0002 / 0.0005* | 0.0000 / 0.0000 / 0.0000* | 0.0010 / 0.0003 / 0.0010* | 0.0000 / 0.0000 / 0.0000* |
| 12 | 0.0001 / 0.0000 / 0.0002* | 0.0000 / 0.0000 / 0.0000* | 0.0002 / 0.0000 / 0.0002* | 0.0000 / 0.0000 / 0.0000* |

## Secondary grid: max loading / capacity / max pair error (pass marked *)

| bits | `nearest_family` | `diffuse_class_family` | `nearest_point` | `diffuse_site_family` |
| --- | --- | --- | --- | --- |
| 4 | 0.1183 / 0.0122 / 0.1260 | 0.0071 / 0.0071 / 0.0089* | 0.1200 / 0.0292 / 0.1194 | 0.0045 / 0.0053 / 0.0060* |
| 5 | 0.0255 / 0.0076 / 0.0216 | 0.0018 / 0.0016 / 0.0024* | 0.0789 / 0.0190 / 0.0868 | 0.0031 / 0.0002 / 0.0035* |
| 6 | 0.0187 / 0.0107 / 0.0149 | 0.0005 / 0.0005 / 0.0004* | 0.0154 / 0.0008 / 0.0157 | 0.0029 / 0.0025 / 0.0044* |
| 7 | 0.0152 / 0.0050 / 0.0145 | 0.0003 / 0.0002 / 0.0003* | 0.0062 / 0.0003 / 0.0073* | 0.0025 / 0.0026 / 0.0040* |
| 8 | 0.0080 / 0.0009 / 0.0096* | 0.0001 / 0.0001 / 0.0002* | 0.0044 / 0.0011 / 0.0049* | 0.0024 / 0.0025 / 0.0039* |
| 9 | 0.0030 / 0.0016 / 0.0034* | 0.0001 / 0.0000 / 0.0001* | 0.0029 / 0.0009 / 0.0027* | 0.0026 / 0.0026 / 0.0041* |
| 10 | 0.0016 / 0.0008 / 0.0016* | 0.0000 / 0.0001 / 0.0001* | 0.0010 / 0.0006 / 0.0011* | 0.0027 / 0.0026 / 0.0041* |
| 11 | 0.0003 / 0.0003 / 0.0002* | 0.0001 / 0.0001 / 0.0001* | 0.0008 / 0.0000 / 0.0010* | 0.0027 / 0.0027 / 0.0042* |
| 12 | 0.0005 / 0.0001 / 0.0005* | 0.0001 / 0.0001 / 0.0001* | 0.0002 / 0.0001 / 0.0003* | 0.0027 / 0.0027 / 0.0041* |
