# M5b findings: inner thermalization and precision

The [full exact-reference study](summary.md) contains 80 original M5a chains
and 720 new cells, with zero generated samples. The
[frozen protocol](../../experiments/meta-ebm-finite-thermalization.md) keeps
the M5a parameters fixed. These are CPU calculations of declared 12-spin
models, not physical thermalizer measurements or a hardware performance claim.

## What the comparisons show

Finite inner K can preserve a good stationary law yet delay useful inference
from the common uniform start. For reading A, variational compilation at cap
3, the median stationary TV bias over five seeds is 1.45e-5 with exact hidden
marginalization and 1.40e-5 at K=1. But target TV after 30 outer sweeps is
0.0770 versus 0.3484. At K=32 (2,304 hypothetical spin redraws per outer
sweep), the latter is still 0.0876. K=1 counts 72 redraws per outer sweep;
neither count is an observed hardware cost.

The stronger reading A constructive case at cap 10 makes the distinction
sharper. Its median stationary bias is 1.23e-5 at both exact marginalization
and K=32, while target TV at outer step 30 is 0.0771 versus 0.2018.
K=32 should therefore not be treated as an equilibrium substitute across
this grid. In reading B, variational cap 1, the K=4 median target TV at
step 30 is 1.25e-6, close to the exact-marginal 1.26e-6, with 288
hypothetical spin redraws per outer sweep. These are descriptive contrasts,
not an optimized choice of K.

The separate mathematical rounding experiment exposes a different
sensitivity. At cap 10 across all readings, methods and seeds, the median
stationary TV bias is 0.822 at 4 bits, 0.0881 at 8 bits and 0.00553 at
12 bits. For reading B variational cap 1, the median exact-marginal bias is
1.26e-6, while 12-bit rounding gives 5.00e-4. The bit widths use the
protocol's symmetric codebook and exact marginalization; they do not model a
Z1 encoding or a combined finite-K/precision arm.

These examples do not imply monotonic behavior for every seed or cap. The
[generated summary](summary.md) gives median [min, max] across five seeds
for all 160 groups and every individual cell. The compressed
[record](study.json.gz) holds the per-site contraction, rounding and
t=0..30 measurements. Stationary bias, finite-horizon target TV and
convergence to each chain's own stationary law remain separate throughout.

## Completion and limits

The [completion marker](completion.json) records 80 references, 720 new cells,
zero samples, integrity passed and all 800 units replayed. The archive SHA-256
is `4e5592572f078f1d36ff89928d145be62588ed5fb8c2adc842a403f278c9c674`.
The first execution session ended without a terminal marker after 688 saved
generation units; its exact stop cause is unknown. Recovery used those units,
finished generation and resumed replay from durable receipts. See
[verification](verification.md) and [provenance](provenance.json).
