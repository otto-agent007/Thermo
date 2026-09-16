# Frozen-program conservation diagnostic

Full three-seed study.

Frozen initial parameters, beta 1, 25 sites, 500 operations, and 32,768 trajectories per seed/horizon. No training. K4 and equilibrium share fresh uniform streams. Independent source seeds are sampling replications; horizons and operations are correlated.

Exact survival is the probability of never leaving the one-particle sector. It is computed with a killed 25-state recurrence. Final valid paths can include paths that left and returned. Exact columns are exact_reference; sampled columns and occupancy loss are software_simulation. All findings are descriptive.

The three archived sources share the same initial parameter digest; their exact curves therefore coincide. The seeds replicate sampling, not parameter initialization.

| Seed | Horizon | Exact median first exit (operation) | Exact 99% exit (operation) | Exact final survival | Sampled ever exited | Sampled final leakage | Sampled valid after earlier exit | Terminal U-loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4 | 52 | 302 | 0.00040392135 | 0.999542236 | 0.77041626 | 0.229125977 | 0.190020885 |
| 0 | equilibrium | 106 | None | 0.0348647811 | 0.965148926 | 0.594055176 | 0.37109375 | 0.0502825908 |
| 1 | 4 | 52 | 302 | 0.00040392135 | 0.999603271 | 0.77142334 | 0.228179932 | 0.191182814 |
| 1 | equilibrium | 106 | None | 0.0348647811 | 0.966064453 | 0.596099854 | 0.3699646 | 0.0510997262 |
| 2 | 4 | 52 | 302 | 0.00040392135 | 0.999542236 | 0.77130127 | 0.228240967 | 0.192186106 |
| 2 | equilibrium | 106 | None | 0.0348647811 | 0.965942383 | 0.588745117 | 0.377197266 | 0.0496407933 |

## Where the first exit enters

Unconditional exact first-exit mass, summed over all 500 operations, by the active edge's parent state. Parent 00 failures create particles on an empty edge; 01/10 failures either destroy the existing particle or create a second. Parent 11 cannot occur before first exit. These are contributions under the surviving path distribution, not causal interventions or uniform-context averages.

| Seed | Horizon | Parent 00 | Parent 01 | Parent 10 | Parent 11 | Creation | Destruction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4 | 0.252540757 | 0.234344503 | 0.512710819 | 0 | 0.309547716 | 0.690048362 |
| 0 | equilibrium | 0.490363003 | 0.224232858 | 0.250539357 | 0 | 0.534016353 | 0.431118866 |
| 1 | 4 | 0.252540757 | 0.234344503 | 0.512710819 | 0 | 0.309547716 | 0.690048362 |
| 1 | equilibrium | 0.490363003 | 0.224232858 | 0.250539357 | 0 | 0.534016353 | 0.431118866 |
| 2 | 4 | 0.252540757 | 0.234344503 | 0.512710819 | 0 | 0.309547716 | 0.690048362 |
| 2 | equilibrium | 0.490363003 | 0.224232858 | 0.250539357 | 0 | 0.534016353 | 0.431118866 |

## Local empty-edge defect and accumulation

Minimum and maximum exact P(output != 00 | parent 00) across all 37 frozen groups. This describes the current parameter matrices; it is not a search over all feasible parameters or proof of an optimal capacity limit.

| Seed | Horizon | Minimum | Maximum |
| --- | --- | --- | --- |
| 0 | 4 | 0.000470921813 | 0.0255797436 |
| 0 | equilibrium | 0.000470921813 | 0.0255796797 |
| 1 | 4 | 0.000470921813 | 0.0255797436 |
| 1 | equilibrium | 0.000470921813 | 0.0255796797 |
| 2 | 4 | 0.000470921813 | 0.0255797436 |
| 2 | equilibrium | 0.000470921813 | 0.0255796797 |

## Reproduction and limits

Each seed JSON embeds the authenticated original source, protocol, all 500 exact and sampled rows at each horizon, local transition counts and table identities. Reload reconstructs the full diagnostic. Raw trajectories are not persisted. Artifacts require the same numerical table values for exact replay; a different floating-point environment may fail the check rather than silently accept drift.

The new simulation is conditional on archived initialization and is not a new evaluation of trained M4B arms. Terminal occupancy loss measures marginals; neither low marginal loss nor terminal count-one alone establishes pathwise conservation. No inference about optimal architecture, caps, convergence, or hardware performance follows. No confidence intervals or simultaneous ranking claims are asserted.

Per source seed: 2 × 32,768 × 500 = 32,768,000 logical endpoint draws across the two horizons. NumPy draws precomputed endpoints, not live Gibbs updates. Table construction, exact propagation, source authentication and replay are additional host work; no wall-clock, energy or device-cost comparison is made.

| Seed | Request | Result |
| --- | --- | --- |
| 0 | `sha256:2da1c6d00835df62471c277f8fe01e3241361c1c70a1fa6f499ade61cefdebdb` | `sha256:470b623e7c3751f37235808296ee54c0bcfa6611a5d402f0cc4ff1055faaa4ac` |
| 1 | `sha256:803795af94d179f239333206b4ff9dcd024911b2c4e43c88d79c98e1c90b1225` | `sha256:477f0069fc251ad3168193e1baedd54b0d2fa9c41bbe2883183e33d7db1bebec` |
| 2 | `sha256:c18bb591078adca0caebe4fd211599197d03055b99a1dd0773a2deb66004fe98` | `sha256:862cec1455a7e94dbec0f768592c88a7d3d512c4dd5e5b1ccf448ab26ad412c9` |
