# Matched-training-budget comparison (M4B)

Full three-seed release.

Five updates per arm from identical archived initial parameters; learning rate 0.01, bounds [-2,2], beta 1, float64. Every update uses independent 32,768-trajectory occupancy and gradient batches. Finite training uses K=4; equilibrium training uses the exact local equilibrium law. All final comparisons execute at K=4.

Updates and logical training draws are matched, not wall-clock time or hardware cost. Both training occupancy and gradient laws change between arms; this is not a score-only ablation. The fifth checkpoint is fixed in advance, with a fresh final stream.

Historical source payloads are authenticated against the complete published archive; unused M1 statistics are not re-executed. Schedule and target are rebuilt independently, and all new M4B computation is replayed.

## Primary contrast: finite minus equilibrium

Negative favors finite training. Approximate paired jackknife normal 95% intervals are conditional, pointwise, descriptive and non-gating. Known small-sample/near-zero undercoverage remains; these are not simultaneous bands or formal acceptance tests. Three independent seeds are the replication units, not the correlated comparisons or trajectories. No convergence or inference-sample saving is established.

| Seed | Equilibrium-trained loss | Finite-trained loss | Finite minus equilibrium | Approximate 95% interval | Conclusion | Equilibrium leakage | Finite leakage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.190478409 | 0.190813715 | 0.000335306482 | [-0.000189953995, 0.000860566959] | inconclusive | 0.767609 | 0.76828 |
| 1 | 0.189387848 | 0.189598014 | 0.000210165482 | [-0.000253837225, 0.00067416819] | inconclusive | 0.772156 | 0.772308 |
| 2 | 0.189647931 | 0.189774937 | 0.000127006238 | [-0.000459932949, 0.000713945424] | inconclusive | 0.767242 | 0.766724 |

Descriptive mean difference across 3 seeds: 0.000224159401. No across-seed confidence claim.

## Each arm versus initial at K=4

| Seed | Arm | Initial loss | Final loss | Difference | Approximate 95% interval | Conclusion | Initial leakage | Final leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | finite_k4 | 0.191457548 | 0.190813715 | -0.000643833007 | [-0.00129846112, 1.07951061e-05] | inconclusive | 0.768555 | 0.76828 |
| 0 | equilibrium | 0.191457548 | 0.190478409 | -0.000979139489 | [-0.00177390203, -0.000184376945] | improved | 0.768555 | 0.767609 |
| 1 | finite_k4 | 0.191300129 | 0.189598014 | -0.00170211548 | [-0.00256402768, -0.00084020329] | improved | 0.773041 | 0.772308 |
| 1 | equilibrium | 0.191300129 | 0.189387848 | -0.00191228097 | [-0.00271017956, -0.00111438237] | improved | 0.773041 | 0.772156 |
| 2 | finite_k4 | 0.190770832 | 0.189774937 | -0.000995894894 | [-0.00172330986, -0.000268479927] | improved | 0.767822 | 0.766724 |
| 2 | equilibrium | 0.190770832 | 0.189647931 | -0.00112290113 | [-0.0019497601, -0.000296042162] | improved | 0.767822 | 0.767242 |

## Projection diagnostics

Counts of clipped parameters by update; training diagnostics do not select checkpoints.

| Seed | Arm | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | finite_k4 | 39 | 39 | 27 | 35 | 28 |
| 0 | equilibrium | 46 | 35 | 39 | 34 | 32 |
| 1 | finite_k4 | 40 | 35 | 28 | 26 | 22 |
| 1 | equilibrium | 51 | 43 | 45 | 38 | 38 |
| 2 | finite_k4 | 41 | 31 | 32 | 32 | 30 |
| 2 | equilibrium | 47 | 47 | 41 | 36 | 30 |

## Declared work per seed

| Quantity | Count |
| --- | --- |
| training_endpoint_draws_per_arm | 245760000 |
| three_pair_evaluation_endpoint_draws | 98304000 |
| total_endpoint_draws | 589824000 |
| finite_training_complete_sweep_equivalents | 983040000 |
| finite_training_free_pbit_update_equivalents | 2949120000 |
| equilibrium_training_complete_sweep_equivalents | undefined: equilibrium oracle |
| evaluation_complete_sweep_equivalents | 393216000 |
| evaluation_free_pbit_update_equivalents | 1179648000 |

Counts describe algorithmic equivalents. NumPy draws joint endpoints; it does not execute these Gibbs sweeps. The three explicitly evaluated pairs repeat work despite sharing randomness. Excludes source preparation, table setup, replay, reset/clamp, host I/O and physical embedding costs. No measured device energy or latency is reported.

Sampled results are software_simulation. Exact local tables are exact_reference. Lower occupancy loss does not imply particle conservation or full-distribution fidelity. Source lineage, every step, final histograms and joined moments are retained in bounded JSON and replayed before reporting.

## Identity and simulator timing

cold synchronous NumPy tables, sampling, reductions and bounded terminal checks; excludes source preparation, sequence replay, enclosing audit validation, persistence and reporting; not hardware latency

| Seed | Source | Request | Result | Simulator seconds |
| --- | --- | --- | --- | --- |
| 0 | `sha256:24bd10653d1b7f5e32c14d1959a2f784e690a2eaf58c368e8cd73d6af6cb09bf` | `sha256:6e33bbd53f07fe81c9947b4f0a6ebf9497db0109f59288f362936cc3b8267584` | `sha256:5461dafe092245fab75bec6eee5cdb9c8431dfb00f159d5d48387126fb70188f` | 69.1264 |
| 1 | `sha256:c43b40ecd64994a250107ca4e3feb3b4fca7be412d4953aef42cbfb501a8ac1c` | `sha256:ab18997f566111799768897191b3c051504976bb141d0573e9a3f6605f9a9a1d` | `sha256:3479258eb9a0d3af0e21ddc176bd351efa05bfe72a8cdc0721b2b01d9dedc390` | 69.389 |
| 2 | `sha256:2b8322b44525a727716e2b73a7ef315c4c2887ff5ff79ab9cbe8827a7b9d5be1` | `sha256:bebc6c4e5c6bff1721f1a32aa8db7e144126678b08b7b38cfaa83212dd1f2291` | `sha256:b9614a542d3d188b030e1e6f099760a9cec6d3c8924c4e874c65e311f6df75f1` | 68.7497 |
