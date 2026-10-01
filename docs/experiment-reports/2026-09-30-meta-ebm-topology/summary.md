# M5c study

Exact reference on CPU, float64, zero generated samples. The lattice is synthetic: the published offset rule (arXiv:2608.01615v2 §II.2.1) on an open square.

## Outer chain

Median [min, max] across seeds.

| Method | K | Stationary bias | TV to target at t=30 | TV to own law at t=30 |
|---|---|---|---|---|
| original | 4 | 1.25208e-06 [7.891e-07, 2.56692e-06] | 1.252e-06 [7.89095e-07, 2.56692e-06] | 3.56693e-12 [1.11514e-16, 4.20702e-10] |
| original | 32 | 1.26406e-06 [9.11551e-07, 2.57089e-06] | 1.26406e-06 [9.11551e-07, 2.57089e-06] | 1.81783e-15 [5.68276e-16, 6.08233e-13] |
| original | limit | 1.26406e-06 [9.11636e-07, 2.57089e-06] | 1.26406e-06 [9.11636e-07, 2.57089e-06] | 1.81987e-15 [7.33105e-17, 6.08259e-13] |
| j_prune | 4 | 0.0144047 [0.00569904, 0.0438157] | 0.0144047 [0.00569904, 0.0438157] | 2.3632e-12 [1.13105e-16, 4.19048e-10] |
| j_prune | 32 | 0.0145182 [0.00576595, 0.0491087] | 0.0145182 [0.00576595, 0.0491087] | 1.57974e-15 [4.48095e-16, 3.79331e-13] |
| j_prune | limit | 0.0145182 [0.00576595, 0.0491089] | 0.0145182 [0.00576595, 0.0491089] | 1.60947e-15 [6.72267e-17, 3.79346e-13] |
| refit | 4 | 3.47332e-05 [3.45714e-06, 7.189e-05] | 3.47328e-05 [3.45714e-06, 7.189e-05] | 4.46028e-10 [4.57888e-14, 3.11409e-08] |
| refit | 32 | 4.51509e-05 [3.45607e-06, 7.11344e-05] | 4.51509e-05 [3.45607e-06, 7.11344e-05] | 1.84906e-15 [3.06613e-16, 6.08186e-13] |
| refit | limit | 4.51581e-05 [3.45607e-06, 7.11344e-05] | 4.51581e-05 [3.45607e-06, 7.11344e-05] | 1.78138e-15 [6.70318e-17, 6.0824e-13] |

Refit at K=4 below the 8.3e-3 reference bar: True; below 5e-4: True. Reference points, not gates.

## Placement

| Seed | Free p-bits | Copies (parity bound) | Physical p-bits | Patch side | Utilization | Largest kernel box |
|---|---|---|---|---|---|---|
| 0 | 72 | 272 (182) | 344 | 27 | 47.2% | 10 × 14 |
| 1 | 72 | 299 (197) | 371 | 29 | 44.1% | 10 × 14 |
| 2 | 72 | 306 (200) | 378 | 29 | 44.9% | 12 × 10 |
| 3 | 72 | 291 (193) | 363 | 28 | 46.3% | 11 × 11 |
| 4 | 72 | 306 (199) | 378 | 28 | 48.2% | 11 × 10 |

Placements proven optimal: 60/60. Refits with successful termination: 10/11.

## Integrity

- M5b original replays: 15, maximum error 3.95e-16.
- Placed conditional error: 2.55e-15 at the origin, 2.55e-15 in the patch.
- Refit brute-force error: 1.22e-15.
- Stationary residual: 2.74e-16; topology violations: 0.

Copy, p-bit and clamp-write counts are algorithmic counts under the stated input model, not device operations. Energy, latency and reprogramming are excluded.
