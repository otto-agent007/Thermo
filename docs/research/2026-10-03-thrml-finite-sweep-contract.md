# After E0: THRML's sweep is the written kernel

*October 3, 2026. CPU software evidence and exact references only.*

The [E0 study](../experiment-reports/2026-10-03-thrml-finite-sweep-contract/findings.md)
closes the finite-K convention question for THRML 0.1.4 on the five-spin
chain. `SamplingSchedule(n_warmup=K, n_samples=1, steps_per_sample=1)` returns
the state after exactly K ordered block sweeps; each block update is the
product of sigmoid(2 beta h_i) conditionals; the energy sign, beta, block order
and clamping all match Thermo's exact kernel. `hinton_init` draws sites with
P(S_i=1) = sigmoid(beta b_i), a heuristic law, not a conditional.

What changes. Finite-K statements built on THRML (M5 inner thermalization at
K in {1,2,4}, the K=30 PAsymSwap budget) now rest on a checked per-sweep
contract for two-colour spin blocks. Thermo code that initialises with
`hinton_init` must model the initial law with beta b, not 2 beta b.

What stays open. Chromatic schedules with more than two colours, non-spin
nodes, graphs beyond enumeration and float32 at scale are untested. Stage B
(one M5a site through THRML) needs the owner's reading choice. A2 (GPU runtime
contract) and A3 (categorical conditionals) remain the other tier (a) checks.
