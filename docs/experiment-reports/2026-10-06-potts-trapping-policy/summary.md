# Potts stage C: a reference-free trapping detector chooses the sampler

Exact references: float64 enumeration of all 3^12 states (`exact_reference`). Sampled
cells: THRML 0.1.4 categorical block Gibbs on CPU, float32, 16 independent trials per
target (`software_simulation`). Budgets are algorithmic redraw counts; no wall-clock or
hardware claim.

Request digest `sha256:cde38ab2327ca38ea89676fecd19b08731a9eeaf9159d0c260956cc7e380c080`; result digest `sha256:a18b515f114711633f64dafdb12098ba944ceca6b26670ca82768e3801aa2fd8`.

**Policy succeeds: False.** Total regret (log4 budget steps against
the per-target hindsight oracle; never-qualifying counts as 65536): policy 8, independent 24, tempering 3. Targets qualifying: policy 36, independent 34, tempering 36.

## By temperature

| beta | regret policy | regret independent | regret tempering | qualifying (policy / ind / temp) | mean switch fraction |
| --- | --- | --- | --- | --- | --- |
| 12 | 4 | 8 | 0 | 12 / 12 / 12 | 0.49 |
| 16 | 3 | 16 | 0 | 12 / 10 / 12 | 0.79 |
| 8 | 1 | 0 | 3 | 12 / 12 / 12 | 0.21 |

## Every target

| target | median spread ratio | switched | policy | independent | tempering | oracle | regret policy / ind / temp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n12-g600-b8 | 0.75 | 0.06 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g600-b12 | 1.12 | 0.62 | 4096 | 4096 | 1024 | 1024 | 1 / 1 / 0 |
| n12-g600-b16 | 1.53 | 0.62 | 4096 | 16384 | 1024 | 1024 | 1 / 2 / 0 |
| n12-g601-b8 | 0.78 | 0.12 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g601-b12 | 4.04 | 0.81 | 1024 | 16384 | 1024 | 1024 | 0 / 2 / 0 |
| n12-g601-b16 | 5.74 | 1.00 | 1024 | NR | 1024 | 1024 | 0 / 3 / 0 |
| n12-g602-b8 | 0.63 | 0.06 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g602-b12 | 0.84 | 0.12 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g602-b16 | 0.88 | 0.44 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g603-b8 | 0.89 | 0.44 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g603-b12 | 1.42 | 0.62 | 4096 | 4096 | 1024 | 1024 | 1 / 1 / 0 |
| n12-g603-b16 | 2.74 | 0.94 | 1024 | 16384 | 1024 | 1024 | 0 / 2 / 0 |
| n12-g604-b8 | 1.15 | 0.62 | 4096 | 1024 | 1024 | 1024 | 1 / 0 / 0 |
| n12-g604-b12 | 4.09 | 0.94 | 1024 | 16384 | 1024 | 1024 | 0 / 2 / 0 |
| n12-g604-b16 | 15.41 | 1.00 | 1024 | NR | 1024 | 1024 | 0 / 3 / 0 |
| n12-g605-b8 | 0.78 | 0.12 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g605-b12 | 0.95 | 0.44 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g605-b16 | 1.63 | 0.88 | 1024 | 4096 | 1024 | 1024 | 0 / 1 / 0 |
| n12-g606-b8 | 0.72 | 0.25 | 1024 | 1024 | 4096 | 1024 | 0 / 0 / 1 |
| n12-g606-b12 | 0.95 | 0.44 | 4096 | 4096 | 4096 | 4096 | 0 / 0 / 0 |
| n12-g606-b16 | 2.84 | 1.00 | 4096 | 16384 | 4096 | 4096 | 0 / 1 / 0 |
| n12-g607-b8 | 0.83 | 0.38 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g607-b12 | 1.16 | 0.62 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g607-b16 | 1.03 | 0.50 | 1024 | 4096 | 1024 | 1024 | 0 / 1 / 0 |
| n12-g608-b8 | 0.80 | 0.06 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g608-b12 | 0.76 | 0.12 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g608-b16 | 1.01 | 0.50 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g609-b8 | 0.75 | 0.31 | 1024 | 1024 | 4096 | 1024 | 0 / 0 / 1 |
| n12-g609-b12 | 1.36 | 0.69 | 4096 | 4096 | 1024 | 1024 | 1 / 1 / 0 |
| n12-g609-b16 | 2.67 | 0.94 | 4096 | 16384 | 4096 | 4096 | 0 / 1 / 0 |
| n12-g610-b8 | 0.75 | 0.00 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g610-b12 | 0.83 | 0.12 | 1024 | 1024 | 1024 | 1024 | 0 / 0 / 0 |
| n12-g610-b16 | 1.32 | 0.69 | 4096 | 1024 | 1024 | 1024 | 1 / 0 / 0 |
| n12-g611-b8 | 0.72 | 0.12 | 1024 | 1024 | 4096 | 1024 | 0 / 0 / 1 |
| n12-g611-b12 | 0.85 | 0.38 | 4096 | 4096 | 1024 | 1024 | 1 / 1 / 0 |
| n12-g611-b16 | 2.37 | 0.94 | 4096 | 16384 | 1024 | 1024 | 1 / 2 / 0 |

## Detection

Label: always-independent trial fails the accuracy rule at T = 4096; 133 of 576 trials. AUC: between_spread 0.806, pair_rhat 0.825, spread_ratio 0.808.
Switch rate given the label: 0.80; given its absence: 0.41.
