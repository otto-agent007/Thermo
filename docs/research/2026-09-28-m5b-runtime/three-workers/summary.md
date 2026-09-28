# M5b bounded runtime benchmark

Exact reference on CPU, float64, zero generated samples. Finite-K dynamics and exact-marginal precision sensitivity are separate arms.

**Runtime calibration only: this subset is not the full M5b study.**

All persisted units were replayed without refitting before completion.

## Paired summaries over seeds

Values are median [min, max]. Full per-site and t=0..30 summaries are in study.json.gz.

| Reading | Method | Cap | Arm | K/b | Bias | Shift from M5a | TV to target at t=30 |
|---|---|---|---|---|---|---|---|
| B | variational | 1.0 | finite_k | 4 | 2.12633e-06 [2.12633e-06, 2.12633e-06] | 9.86198e-08 [9.86198e-08, 9.86198e-08] | 2.12633e-06 [2.12633e-06, 2.12633e-06] |
| B | variational | 1.0 | precision | 8 | 0.0080237 [0.0080237, 0.0080237] | 0.00802351 [0.00802351, 0.00802351] | 0.0080237 [0.0080237, 0.0080237] |
| B | variational | 1.0 | reference | None | 2.13893e-06 [2.13893e-06, 2.13893e-06] | 0 [0, 0] | 2.13893e-06 [2.13893e-06, 2.13893e-06] |

## Every seed

| Cell | Bias | Shift | Sweep TV from M5a | Algorithmic spin redraws |
|---|---|---|---|---|
| B/0/1.0/variational/finite_k/4 | 2.1263299e-06 | 9.8619771e-08 | 0.25427374 | 288 |
| B/0/1.0/variational/precision/8 | 0.0080237015 | 0.0080235084 | 0.012330192 | n/a |
| B/0/1.0/variational/reference/None | 2.1389299e-06 | 0 | 0 | n/a |

Local detailed balance is retained. Independently compiled conditionals can be incompatible, allowing the outer stationary law to shift. Algorithmic work counts are hypothetical Gibbs redraws, not device time or energy.
