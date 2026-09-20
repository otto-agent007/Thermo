# M4G training-runner component preflight

All 21 bounded fixture fits completed five updates: 105 checkpoints independently replayed across K=1,2,4,8,16,30 and equilibrium. **Full M4G readiness remains false.**

Request: `sha256:0a8496a7ace6115e52a1688002335ae906c2cf2eb9285f1d468402b23fe843ae`

Result: `sha256:a5af9d0e0153e336d791f04b738294b342dd9bbe7aa7039ab69cc1661be809ce`

Maximum occupancy replay error: 0. Maximum gradient replay error: 0.

Each fit begins at the fixed three-site fixture and retains law-specific occupancy, realized rewards, gradient sums and sum-squares, projected updates, table identities, and the selected fifth checkpoint. Sampled evidence is `software_simulation`; local tables are `exact_reference`. Independent replay reconstructs every evolving law.

The 21 production requests authenticate the original M1 initial parameters, all 500 operations and all assigned roles. The 210 runner diagnostic roles are distinct from all 213 study roles and 42 prior law-validation roles. Zero production fits or held-out study cells ran. A complete sampling or replay pass accounts for 20,643,840 local endpoint draws; reference draws never propagate as extra trajectories.

This gate validates the shared training engine and source binding. It does not establish task quality, convergence, inference savings or device performance. The complete held-out evaluator, integrated study preflight and independent review must precede full-program training and evaluation.
