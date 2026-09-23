# Saved-checkpoint survival-gradient audit

Approved scope: diagnose the 21 saved M4G fits and 105 projected updates without
new fitting, new stochastic evaluation or changes to the original evidence.
All six checkpoints of each fit are examined under its own training horizon
(K1,2,4,8,16,30 or equilibrium). This is a retrospective exact-reference audit,
not a new held-out task-quality claim.

Authenticate the six complete September 20 release files by fixed reviewed byte
SHA-256 pins before using any fit. Different source bytes require a separate
review. The original release already validated the sampled training histories;
this audit does not rerun those stochastic roles.

Use the existing 25-state killed process: only paths that have retained exactly
one particle after every completed operation survive. The initial particle is
at site zero. For each operation, outside-edge mass survives only via 00 -> 00;
on-edge mass survives via the two singly occupied outputs. Internal Gibbs
substeps are not the conservation observation boundary.

Differentiate each exact local endpoint law (uniform reset, hidden-then-output
finite sweeps; equilibrium separately) at beta one and caps [-2,2]. Scaled
forward/backward messages compute grad(log S). Retain occurrence contributions
before summing repeated uses of each of the 37 parameter groups. A three-site,
two-operation fixture checks all nine shared derivatives by central differences
(step 1e-6, absolute tolerance 1e-7) at every horizon before consuming the archive.

For each actual projected displacement delta, record grad(log S) dot delta,
observed log S and S changes, linearization residual, total/group displacement
norms, occurrence/group directional contributions, and cancellation ratios.
Group gradient retention is norm(sum occurrence gradients)/sum(norm gradients).
Directional retention is abs(sum directional contributions)/sum(abs contributions).
Undefined zero-denominator ratios are null; no arbitrary classification threshold
is introduced. First-exit creation/destruction masses are recorded by operation.

No update, checkpoint, horizon or objective is selected from this diagnosis.
Negative local slopes indicate local conflict for those realized directions;
positive small gains do not establish optimizer convergence or a capacity limit.
Cancellation is descriptive and does not prove that unsharing would help.

Run `uv run python -m thermo_lab.survival_gradient_audit --output-dir results/survival-gradient-audit`
in a fresh destination. Persist the complete audit, replay it strictly from the
pinned sources before reporting, and write completion last. Exact computation is
not counted as sampled observations or physical device work. The generating
runtime's strict canonical replay is intentional; cross-runtime bitwise identity
is not promised. Preserve all historical validators and quality thresholds.
