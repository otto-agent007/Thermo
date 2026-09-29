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

## Answers to the four review questions (added at review)

These were computed by hand from the archived record. Values are medians
over the five seeds. K counts are the largest value over the 12 sites and
are worst-case over all blanket inputs, so they bound the local error and
are not typical costs.

**(a) Inner sweeps needed to come within 1e-3 and 1e-6 of the exact-marginal
kernel.** Under reading B it is about 20 and about 40 at every cap
(7 and 13–15 at cap 0.3, 18–28 and 38–58 at caps 1–10). Under reading A it grows
with the cap: 7–8 and 15 at cap 0.3, 146–197 and 306–404 at cap 1, and
about 4.5e7 and 9.1e7 at cap 10. Reading A at cap 3 is 4.8e4/9.5e4
(variational) and 1.6e6/3.3e6 (constructive).

**(b) Does a larger cap push λ toward 1?** Yes under reading A: the largest
site λ goes from 0.39–0.41 at cap 0.3 to 0.96–0.97 at cap 1 and
0.9999–1.0000 at caps 3 and 10. Not under reading B: it rises from 0.36–0.39 at
cap 0.3 to 0.71–0.80 at cap 1, then stays near 0.70 up to cap 10. The
reading A cost appears in the finite horizon. After 30 outer sweeps, cap 10
at K = 32 is still 0.20 from the target, against 0.077 with exact
marginalization. Under reading B, K = 4 matches the exact-marginal horizon
at every cap (for example 1.25e-6 against 1.26e-6 for variational at
cap 1).

**(c) At what bit width does rounding stop mattering, relative to compile
error?** In this grid, 8–12-bit rounding leaves bias unchanged only where
compile error is already large, for example all of reading A at caps ≤ 1
and reading B at cap 0.3. Wherever compilation is accurate,
12-bit rounding dominates by two to six orders of magnitude. Because the
rounding step is cap/L, the damage grows with the cap. For reading B
variational, 12-bit bias is 5.0e-4 at cap 1, 1.5e-3 at cap 3 and 5.5e-3 at
cap 10, against unrounded bias of 1.3e-6, 3.5e-7 and 2.4e-9. For reading A
variational, 12-bit bias is 2.0e-3 at cap 3 and 6.0e-3 at cap 10. **Under
finite precision the best cap in this grid is the smallest one at which
compilation is already accurate:** cap 1 for reading B variational and cap 3
for reading A variational. Raising the cap further only costs precision.
Precision is a mathematical codebook here, not a Z1 encoding.

**(d) Does finite K shift the stationary law?** Only for variational
kernels at low K. The largest shift is 0.092 (reading A, seed 4, cap 1,
K = 1), and the median at that cell is 0.072. Reading A cap 0.3 and reading
B cap 0.3 shift by 0.020 and 0.012 at K = 1. At K = 32 every median shift is
at most 1.6e-3. All 240 constructive finite-K cells shift by at most
5.7e-13, so they are numerically unchanged at every K. That indicates the
constructive conditionals are compatible with a single joint law, while
the separately fitted variational ones are not. This study does not prove
why.

## Completion and limits

The [completion marker](completion.json) records 80 references, 720 new cells,
zero samples, integrity passed and all 800 units replayed. The archive SHA-256
is `4e5592572f078f1d36ff89928d145be62588ed5fb8c2adc842a403f278c9c674`.
The first execution session ended without a terminal marker after 688 saved
generation units; its exact stop cause is unknown. Recovery used those units,
finished generation and resumed replay from durable receipts. See
[verification](verification.md) and [provenance](provenance.json).
