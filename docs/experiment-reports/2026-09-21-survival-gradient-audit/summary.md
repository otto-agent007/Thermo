# Exact survival-gradient audit of M4G

**Finding:** 103/105 updates increase survival; two K1/seed0 updates decrease it. All 21 fits finish above initialization. Predicted and actual change signs agree for all 105 updates. K30 survival rises from 3.3211% to 3.3776–3.3803%, a gain of only 0.0564–0.0592 percentage points against the unchanged 95% target. This weakens a general objective-conflict explanation, without establishing that longer training would succeed.

Across saved checkpoints/groups, gradient retention has median 0.9595 and minimum 0.6071. Across updates, directional retention has median 0.5760 and minimum 0.0292. These are distinct diagnostics: group-vector cancellation versus cancellation along each actual displacement. Update displacement norms range from 0.000660 to 0.002725. No causal conclusion about unsharing or optimization budgets follows.

All 21 saved fits and 105 recorded projected updates; no new training or sampling. Each fit is evaluated under its own training law. Exact reference calculations describe the saved software model, not hardware.

| Seed | Training horizon | Initial survival | Final survival | Negative predicted steps | Negative actual steps |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 4.16016043e-09 | 4.16113681e-09 | 2/5 | 2/5 |
| 0 | 2 | 3.60561306e-06 | 3.62359515e-06 | 0/5 | 0/5 |
| 0 | 4 | 0.00040392135 | 0.000409771659 | 0/5 | 0/5 |
| 0 | 8 | 0.0064648929 | 0.00667253696 | 0/5 | 0/5 |
| 0 | 16 | 0.0238625135 | 0.0244672109 | 0/5 | 0/5 |
| 0 | 30 | 0.0332113267 | 0.0338034677 | 0/5 | 0/5 |
| 0 | equilibrium | 0.0348647811 | 0.0354329779 | 0/5 | 0/5 |
| 1 | 1 | 4.16016043e-09 | 4.16382337e-09 | 0/5 | 0/5 |
| 1 | 2 | 3.60561306e-06 | 3.62368845e-06 | 0/5 | 0/5 |
| 1 | 4 | 0.00040392135 | 0.000410067577 | 0/5 | 0/5 |
| 1 | 8 | 0.0064648929 | 0.00667374317 | 0/5 | 0/5 |
| 1 | 16 | 0.0238625135 | 0.0244732594 | 0/5 | 0/5 |
| 1 | 30 | 0.0332113267 | 0.0337895078 | 0/5 | 0/5 |
| 1 | equilibrium | 0.0348647811 | 0.0354353879 | 0/5 | 0/5 |
| 2 | 1 | 4.16016043e-09 | 4.16246423e-09 | 0/5 | 0/5 |
| 2 | 2 | 3.60561306e-06 | 3.62273759e-06 | 0/5 | 0/5 |
| 2 | 4 | 0.00040392135 | 0.000409507109 | 0/5 | 0/5 |
| 2 | 8 | 0.0064648929 | 0.0066785172 | 0/5 | 0/5 |
| 2 | 16 | 0.0238625135 | 0.0244852326 | 0/5 | 0/5 |
| 2 | 30 | 0.0332113267 | 0.0337756647 | 0/5 | 0/5 |
| 2 | equilibrium | 0.0348647811 | 0.03543004 | 0/5 | 0/5 |

The predicted log change is grad(log survival) dotted with the actual projected displacement. Its sign describes the local direction; the observed finite-step change can differ. Raw values and linearization residuals are retained without an acceptance threshold.

Per-operation and per-group directional contributions, displacement norms, first-exit creation/destruction and group gradient retention are in audit.json. Retention is norm(sum contributions) / sum(norm contributions); zero denominators are null. Low retention indicates cancellation, not proof that unsharing would improve quality.

These diagnostics do not identify a global capacity limit, establish convergence, or certify task quality. No checkpoint, objective, training budget or sampling law was selected or changed from these outcomes.

## Reproduce and inspect

`audit.json.gz` contains the complete canonical audit; decompress with Python gzip. `completion.json` binds its canonical scientific digest. Run the command in the linked protocol using a fresh destination. A reloaded record must pass `validate_audit` or the validating `render_report` before use. The compressed archive preserves all occurrence contributions and first-exit masses.

[Protocol](../../experiments/survival-gradient-audit.md). Generated report is supplemented here with descriptive synthesis. No model or checkpoint was selected using these results.
