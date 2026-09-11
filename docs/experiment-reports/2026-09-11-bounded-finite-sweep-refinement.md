# Bounded finite-sweep refinement (M4)

Recorded September 11, 2026 from clean implementation commit
[`63517f9`](https://github.com/otto-agent007/Thermo/commit/63517f949ec2ae58e1c2363b7bfa615a758efcbe).
The [protocol was committed before the study](https://github.com/otto-agent007/Thermo/blob/0a09575c694f187e4ee3652e55cf697ed5ba824f/docs/experiments/bounded-finite-sweep-refinement.md).
All three embedded M1 sources were produced at clean commit
[`2b9b4a2`](https://github.com/otto-agent007/Thermo/commit/2b9b4a2750967ded4613b6c98d9e2d94c1438942).
The training starts from their actual initial parameters.

The complete bounded evidence includes
[seed 0](2026-09-11-bounded-finite-sweep-refinement/seed-0000000000.json),
[seed 1](2026-09-11-bounded-finite-sweep-refinement/seed-0000000001.json),
[seed 2](2026-09-11-bounded-finite-sweep-refinement/seed-0000000002.json),
[sampler checks](2026-09-11-bounded-finite-sweep-refinement/sampler-validation.json),
[requested protocol](2026-09-11-bounded-finite-sweep-refinement/protocol.json), and
[completion](2026-09-11-bounded-finite-sweep-refinement/completion.json).
Each seed artifact embeds its source and sampler validation and retains every
update; it can be independently reconstructed with `FiniteRefinementAudit`.
The directory also contains both JSON Schemas. No raw trajectories are stored.

The fifth update has a lower estimated held-out loss in all three seeds;
two approximate intervals exclude zero and one is inconclusive. The decreases
are small, and final particle leakage remains approximately 77%. This supports
only the reported bounded occupancy-loss comparison; it does not establish
particle conservation, monotonic improvement, or convergence. These K=4 losses
are not directly comparable to the historical equilibrium M1 loss.

Full three-seed release.

Five updates at four complete sweeps, learning rate 0.01, beta 1, float64, and bounds [-2, 2]. Each update uses independent 32,768-trajectory occupancy and gradient roles. The fifth checkpoint is selected in advance; 32,768 fresh paired trajectories evaluate initial versus final parameters.

The initial parameters and target are frozen from each supplied M1 source. Its one-step updated parameters and held-out draws are not reused for M4 training or final evaluation.

## Held-out initial versus fifth update

Signed differences are after minus before. Approximate 95% intervals use the paired delete-one jackknife. Conclusions are descriptive and non-gating, retain small-sample/near-zero coverage limitations, and are not simultaneous confidence bands. This bounded study does not establish convergence.

| Seed | Initial U-loss | Final U-loss | Difference | Approximate 95% interval | Conclusion | Initial leakage | Final leakage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.192684023 | 0.191484833 | -0.00119919016 | [-0.00202080725, -0.000377573078] | improved | 0.769836 | 0.769501 |
| 1 | 0.191412166 | 0.191185754 | -0.000226412107 | [-0.000968673865, 0.000515849651] | inconclusive | 0.772308 | 0.770905 |
| 2 | 0.191476343 | 0.190519455 | -0.00095688785 | [-0.00175762235, -0.000156153353] | improved | 0.772766 | 0.772858 |

## Training diagnostics

These are unbiased order-two occupancy losses from the training role before each update, not held-out checkpoint comparisons. Different rows use fresh randomness and adaptively updated parameters; their fluctuations are not evidence of monotonic learning. No row selects a checkpoint or changes the protocol.

| Seed | Update | Pre-update training U-loss | Parameters clipped |
| --- | --- | --- | --- |
| 0 | 1 | 0.191460603 | 43 |
| 0 | 2 | 0.187724444 | 30 |
| 0 | 3 | 0.189808961 | 28 |
| 0 | 4 | 0.191155184 | 30 |
| 0 | 5 | 0.19246499 | 29 |
| 1 | 1 | 0.192592306 | 35 |
| 1 | 2 | 0.192056042 | 40 |
| 1 | 3 | 0.191291934 | 35 |
| 1 | 4 | 0.190743484 | 28 |
| 1 | 5 | 0.18948547 | 22 |
| 2 | 1 | 0.191790025 | 33 |
| 2 | 2 | 0.192681416 | 28 |
| 2 | 3 | 0.190817832 | 24 |
| 2 | 4 | 0.191828134 | 28 |
| 2 | 5 | 0.190567898 | 25 |

## Sampler validation

The exact estimator mean agrees with M3's independently checked gradient. Exact reference second moments include independent same-parent reference variance and covariance between shared occurrences. The table reports maximum absolute component errors divided by exact standard errors for a fixed reward coefficient. These sampled diagnostics are descriptive, not simultaneous acceptance tests. They do not include uncertainty from an estimated occupancy coefficient.

| Sweeps | Maximum exact mean discrepancy | Seed 0 standardized error | Seed 1 standardized error | Seed 2 standardized error |
| --- | --- | --- | --- | --- |
| 1 | 6.66133815e-16 | 1.09985 | 1.60971 | 0.813156 |
| 2 | 6.10622664e-16 | 0.986281 | 1.59214 | 1.75098 |
| 4 | 1.33226763e-15 | 1.41412 | 1.233 | 1.92481 |
| 8 | 2.2759572e-15 | 0.692238 | 1.06306 | 2.22796 |
| 16 | 5.77315973e-15 | 1.60562 | 1.76726 | 1.91332 |
| 30 | 1.07691633e-14 | 2.20968 | 1.72268 | 1.79817 |

## Evidence and declared work

All sampled outcomes are software_simulation; local probability/derivative tables and microcircuit expectations are exact_reference. NumPy samples precomputed joint endpoints. No hardware measurements are made.

Per study seed: 278,528,000 logical endpoint draws across occupancy, main/reference gradient, and paired final roles. At K=4 this represents 1,114,112,000 declared complete-sweep equivalents and 3,342,336,000 free-pbit-update equivalents. These are algorithmic counts, not executed Gibbs updates or measured latency/energy. They exclude source generation/validation, table construction, microcircuit diagnostics, artifact replay, reset/clamp, and I/O.

JSON evidence retains all five updates and bounded moments, source records, role seeds, and final joined counts/histograms. Reloading replays the supplied sampling inputs and reconstructs the chain. Cached replays retain only immutable summaries; raw trajectories are not persisted.

| Seed | Source summary | Request | Result |
| --- | --- | --- | --- |
| 0 | `sha256:24bd10653d1b7f5e32c14d1959a2f784e690a2eaf58c368e8cd73d6af6cb09bf` | `sha256:5e3b987a60137faf0d75d666f992f8cea9ffc81e01e90eb6257cee80a5ff332b` | `sha256:4f099d1825fc29a8946cbe87061bd2875aba9bce9c392953c1fccc8d7d7db3bd` |
| 1 | `sha256:c43b40ecd64994a250107ca4e3feb3b4fca7be412d4953aef42cbfb501a8ac1c` | `sha256:0faca5c904fa22a6dd2557e17283bf903349c8d81cd309f5d53fa7d509bf421c` | `sha256:2308ad96799bb31b700e75eeb683c26a3b3e0cbb76861138d7419094053ccca1` |
| 2 | `sha256:2b8322b44525a727716e2b73a7ef315c4c2887ff5ff79ab9cbe8827a7b9d5be1` | `sha256:ad1d6beb6e01ead3864e3976c780302c3f9e30949ae4538f1d75007a06ea7866` | `sha256:a860e39d498261e0cd2f1881d2d0ba5c02f592d2bfe35c3564f9ef5fe81cc985` |

Sampler request: `sha256:ba837a2f1a570f66f1d3694711b5be7b0df0d80ab17c5a8976384e0f0a6a7c0a`
Sampler result: `sha256:20574c77d24fcc2ac8c4a8a65ee9f291c0feeaed75e4d186861c3be49b84a41e`
