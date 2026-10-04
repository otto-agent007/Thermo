# THRML execution of an M5a kernel's inner sweep (E0 stage B)

Kernel: reading B, variational, cap 1.0, seed 0, site 1 (blanket 10 inputs, 7 hidden spins, 95 parameters, parameter SHA-256 `84fb742de0a22936...`).

Exact references: the archived M5b inner-K law through the hash-bound `powered_rates`
and a study-local enumeration of the (hidden, output) joint (`exact_reference`, float64).
THRML cells: 0.1.4 on CPU, float32, 65536 independent chains per
input and 2^10 inputs per cell (`software_simulation`). No hardware
evidence anywhere in this report; inner sweeps are not device operations.

Request digest `sha256:b6ade3788db8687fc759209444eaa082ef48cbc5d8338836e8d2fa74960f1e04`; result digest `sha256:b3c328c5694572baa471fbb765cf8c4c7f64d20eb27b79f316a3b70e976f8bfe`.

## Cells

| cell | max |p_hat - p| over inputs | tol (q=0.999) | pass | max joint TV | tol | pass | closest K | max dev to K+1 | to output-first | to negated x | to limit q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y0+/K1 | 0.0051 | 0.0079 | yes | 0.0179 | 0.0220 | yes | K | 0.1942 | 0.9266 | 0.3241 | 0.5326 |
| y0+/K2 | 0.0062 | 0.0085 | yes | 0.0183 | 0.0211 | yes | K | 0.1244 | 0.6040 | 0.4266 | 0.3809 |
| y0+/K4 | 0.0058 | 0.0091 | yes | 0.0178 | 0.0211 | yes | K | 0.0581 | 0.3067 | 0.4919 | 0.2238 |
| y0-/K1 | 0.0055 | 0.0089 | yes | 0.0177 | 0.0197 | yes | K | 0.2418 | 0.8239 | 0.3757 | 0.6491 |
| y0-/K2 | 0.0067 | 0.0089 | yes | 0.0183 | 0.0201 | yes | K | 0.1436 | 0.5725 | 0.4635 | 0.4567 |
| y0-/K4 | 0.0066 | 0.0090 | yes | 0.0173 | 0.0194 | yes | K | 0.0686 | 0.2890 | 0.5302 | 0.2458 |

## Negative controls

Exact-side gate: 24 checks, 0 without separation.

| control | cell | exact separation | max dev of samples to wrong law | tolerance | rejected |
| --- | --- | --- | --- | --- | --- |
| off_by_one | y0-/K1 | 0.2400 | 0.2418 | 0.0089 | yes |
| output_first | y0-/K1 | 0.8228 | 0.8239 | 0.0089 | yes |
| negated_inputs | y0-/K1 | 0.3769 | 0.3757 | 0.0089 | yes |
| marginal_limit | y0-/K1 | 0.6483 | 0.6491 | 0.0089 | yes |
| off_by_one | y0-/K2 | 0.1424 | 0.1436 | 0.0089 | yes |
| output_first | y0-/K2 | 0.5709 | 0.5725 | 0.0089 | yes |
| negated_inputs | y0-/K2 | 0.4625 | 0.4635 | 0.0089 | yes |
| marginal_limit | y0-/K2 | 0.4543 | 0.4567 | 0.0089 | yes |
| off_by_one | y0-/K4 | 0.0661 | 0.0686 | 0.0090 | yes |
| output_first | y0-/K4 | 0.2907 | 0.2890 | 0.0090 | yes |
| negated_inputs | y0-/K4 | 0.5267 | 0.5302 | 0.0090 | yes |
| marginal_limit | y0-/K4 | 0.2468 | 0.2458 | 0.0090 | yes |
| off_by_one | y0+/K1 | 0.1932 | 0.1942 | 0.0079 | yes |
| output_first | y0+/K1 | 0.9261 | 0.9266 | 0.0079 | yes |
| negated_inputs | y0+/K1 | 0.3212 | 0.3241 | 0.0079 | yes |
| marginal_limit | y0+/K1 | 0.5307 | 0.5326 | 0.0079 | yes |
| off_by_one | y0+/K2 | 0.1218 | 0.1244 | 0.0085 | yes |
| output_first | y0+/K2 | 0.6025 | 0.6040 | 0.0085 | yes |
| negated_inputs | y0+/K2 | 0.4257 | 0.4266 | 0.0085 | yes |
| marginal_limit | y0+/K2 | 0.3803 | 0.3809 | 0.0085 | yes |
| off_by_one | y0+/K4 | 0.0551 | 0.0581 | 0.0091 | yes |
| output_first | y0+/K4 | 0.3068 | 0.3067 | 0.0091 | yes |
| negated_inputs | y0+/K4 | 0.4913 | 0.4919 | 0.0091 | yes |
| marginal_limit | y0+/K4 | 0.2228 | 0.2238 | 0.0091 | yes |

## Archived inner contraction of this site

lambda min 0.2074, median 0.5973, max 0.7805; K for local TV below 1e-3: 26, below 1e-6: 54.
