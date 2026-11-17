# Z1 acceptance kit (P-0015): summary

**The kit holds except for named checks.** The frozen decision rule fails
the collective missing-coupling scan on one target: on
`L16-greedy-long-s4510` the ideal sampler rejected in 76 of 1,000
replicates against a limit of 74, so all three of that target's collective
checks fail. That scan count did not resolve and needs empirical
calibration before use. **Operating point to use for a missing coupling at
an unknown location: the simple matching scan** (8 reads per matching, 7 to
9 programmings, 56 to 72 reads in total; sampled check 3 and 955 of 1,000).
The other 38 of 41 sampled checks pass, all 14 simple checks included, and
all nine collective targets are attributable.

- **Per defect, cheapest exact test** (reads at 5 % false alarm and 5 %
  miss; one read is the full array after one two-colour sweep):
  - stuck p-bit: an unbiased isolated site, 6 reads at a known site, 12, 14
    and 16 scanning 64, 256 and 1,024 sites; 1 programming, 0 state writes;
  - gain spread, uniform ±10 % (row definition): all sites at one bias,
    pooled homogeneity X², 520, 229 and 108 reads at 64, 256 and 1,024
    sites; ±30 %: 59, 27 and 13; 1 programming;
  - one outlying site at 1 ± 10 % (sign unknown): 3,368 reads at a known
    site, 6,456 to 8,377 scanning; at ± 30 %: 469, and 893 to 1,160;
  - missing coupling: 5 reads at a known coupler (pair at |J| = 1, beta 4);
    56 to 72 reads and 7 to 9 programmings at an unknown location;
  - 8-bit rounding: 238, 955 and 3,788 reads at 1,024, 256 and 64 spins
    (pooled matching at weight 0.2, beta 4); 1 programming.
- **Collective scan** at the chosen beta 0.5, one programming: 3,658 to
  5,969 reads at 64 spins, 6,316 to 7,219 at 256 and 9,765 to 11,677 at
  1,024, which is 57 to 182 times the matching scan's reads. Two 1,024-spin
  targets (s4510 at 10,494 and s4512 at 11,677) do not fit the 10,000-read
  limit at this size. The collective test sees no declared defect that the
  simple tests miss; it stays only for its single programming write.
- **Smallest defect within 10,000 reads:** 8-bit rounding at every size;
  uniform gain spread ±2.3 %, ±1.5 % and ±1.0 % at 64, 256 and 1,024 sites;
  an outlying site at 5.7 % (known site) and 8.0 to 9.1 % (scanning); every
  missing coupler; a stuck p-bit in either behaviour.
- **Evidence.** Simple-test counts and the collective means, shifts,
  per-read variances and independent-read counts are `exact_reference`.
  Pooled gain-spread counts and thresholds are Monte Carlo on exact laws.
  Scatter ratios, sampled means and operating characteristics are
  `software_simulation` (NumPy float64, CPU). No chip, no hardware claim, no
  digital baseline (a metrology row).
- **Integrity.** All 18 P-0001 references reproduce bitwise. The reference,
  rank-2 (1e-9), finite-difference, patch and enumeration checks pass;
  23,136 missing-edge correlations are exact. Full replay passed (601 s).
  The run took 2 h 13 min 49 s wall with three workers (5.7 CPU-h) from
  clean commit ba3f4f3; the bounded archive is 858 kB.

## Collective scan by target (beta 0.5)

| Target | Edges | t(127) quantile | n_scan | Independent reads | Ideal false alarm | κ̂ / κ_U | ρ_U | Union bound | Checks (ideal; defect cheapest, median, dearest) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L8 s4510 | 112 | 3.403 | 5,453 | 2,679 | 5.2 % | -0.002 / 0.014 | 0.043 | 0.053 | 52; 1000, 1000, 949 |
| L8 s4511 | 112 | 3.403 | 3,658 | 2,624 | 4.6 % | 0.009 / 0.025 | 0.038 | 0.055 | 46; 1000, 1000, 961 |
| L8 s4512 | 112 | 3.403 | 5,969 | 2,672 | 5.4 % | -0.009 / 0.007 | 0.043 | 0.052 | 54; 1000, 1000, 966 |
| L16 s4510 | 480 | 3.819 | 6,615 | 3,189 | **7.6 %** | 0.006 / 0.014 | 0.039 | 0.053 | **76**; 1000, 1000, 968 |
| L16 s4511 | 480 | 3.819 | 7,219 | 3,152 | 4.4 % | -0.003 / 0.006 | 0.039 | 0.052 | 44; 1000, 1000, 954 |
| L16 s4512 | 474 | 3.816 | 6,316 | 3,126 | 6.7 % | 0.002 / 0.010 | 0.038 | 0.052 | 67; 1000, 1000, 970 |
| L32 s4510 | 1,984 | 4.197 | 10,494 | 3,736 | 6.0 % | 0.001 / 0.005 | 0.034 | 0.051 | 60; 1000, 1000, 961 |
| L32 s4511 | 1,976 | 4.196 | 9,765 | 3,701 | 4.8 % | -0.001 / 0.003 | 0.035 | 0.051 | 48; 1000, 1000, 978 |
| L32 s4512 | 1,982 | 4.197 | 11,677 | 4,013 | 5.6 % | -0.001 / 0.004 | 0.035 | 0.051 | 56; 1000, 1000, 955 |

Pass rule: ideal at most 74 and defect at least 926 of 1,000 replicates.
Attribution limit 0.065. The 1,024-spin n_scan values come from the
16,384-read recalibration (the 8,192-read calibration gave 8,665, 10,479
and 10,662).

## Simple sampled checks (1,000 replicates each)

| Check | Reads | Ideal rejections | Defect rejections |
| --- | --- | --- | --- |
| stuck p-bit, known site | 6 | 29 | 1000 |
| stuck p-bit, scanning 1,024 sites | 16 | 42 | 1000 |
| gain spread ±10 %, 1,024 sites | 108 | 62 | 950 |
| gain spread ±30 %, 1,024 sites | 13 | 45 | 956 |
| outlier +10 %, known site | 3,368 | 52 | 964 |
| outlier −10 %, known site | 3,368 | 45 | 975 |
| outlier +30 %, known site | 469 | 52 | 970 |
| outlier −30 %, known site | 469 | 51 | 997 |
| missing coupling, known coupler | 5 | 0 | 977 |
| missing coupling, one scanned matching at 1,024 spins | 8 | 3 | 955 |
| rounding 4 bits, 1,024-spin matching | 1 | 59 | 963 |
| rounding 6 bits | 46 | 45 | 937 |
| rounding 7 bits | 60 | 41 | 947 |
| rounding 8 bits | 238 | 43 | 959 |

Details, pre-registered expectations and the temperature rule are in
[findings](findings.md). Request
`sha256:3031972bd7cdef5b1013cf963af336b24f1e0a46dc026a4e4dc0201b219cc3f1`.
