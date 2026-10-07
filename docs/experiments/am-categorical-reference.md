# Associative memory stage A2: a label-first categorical reference

**Status: frozen 2026-10-07 on the owner's instruction ("rerun the reference
with label first updates"), before any cell ran.** It amends the
[stage A protocol](am-binary-emulation.md) for one arm only.

## Why

In stage A the sampled categorical reference updated its visible bits first,
from a uniformly random label. The visible bits copied the wrong pattern and
locked in, so its recall at 256 sweeps (0.64 to 0.93) fell far below its
equilibrium (0.88 to 1.00). The primary "matches the reference" comparison was
therefore uninformative
([findings](../experiment-reports/2026-10-07-am-binary-emulation/findings.md)).
A post-hoc check on probe-only patterns found that updating the label first
removes most of the lock-in.

## What changes and what does not

- **Changed:** the categorical arm's block order per sweep is label first, then
  the missing visible bits. The start is unchanged: missing bits uniform, label
  uniform. The first label update therefore sees the cue plus random missing
  bits.
- **Unchanged from stage A:** N = 24; P ∈ {8, 32, 128}; cues of 12 and 8 bits;
  β ∈ {4, 8, 16}; budgets K ∈ {4, 16, 64, 256}; development seeds 7000–7003;
  held-out seeds 7100–7111; 3 targets and 256 chains per pattern set; THRML
  0.1.4 categorical node with mixed spin–categorical factors; per-budget
  selection on development sets; held-out comparison; ±0.02 matching margin;
  2,000-draw paired bootstrap over held-out sets; the same JAX root seed and
  unit-key derivation.
- **Binary arms are not rerun.** Their held-out per-set recall is read from the
  stage A archive, which is authenticated by the SHA-256 recorded in its
  `completion.json`. Each binary arm keeps its stage A update order (visible
  first, hidden units starting off). That order did not trap them, because no
  hidden unit is on at the start. Giving them a label-first order is not part
  of this amendment.

## Comparisons

1. **Primary:** each binary arm's held-out recall at each K, as a paired
   difference against the label-first reference at the same K. The verdict
   rule is stage A's: matches if the 95% interval lies within ±0.02; falls
   short if it lies below −0.02; exceeds if above +0.02; otherwise
   inconclusive.
2. **Confound size:** the label-first reference against the stage A
   visible-first reference, per cell and K, on the same held-out sets.
3. **Equilibrium:** the reference's exact recall does not depend on update
   order. It is recomputed and must equal stage A's within 1e-12.

## Integrity

- Preflight: one THRML sweep of the label-first program matches the exact
  one-sweep law on the stage A preflight instance (N = 6, P = 3).
- Budgets are prefixes of one run per unit, re-checked at K = 64 for one unit.
- Replay recomputes exact references and the evaluation, compares them
  numerically, and re-authenticates the stage A archive by its SHA-256.

## Outputs and completion

Runner `thermo_lab.am_categorical_reference`, one replay test, one report.
Completion requires `status=am_categorical_reference_complete`, 72 development
and the selected held-out units, `preflight_passed=true`,
`prefix_check_passed=true`, `stage_a_archive_verified=true` and
`replayed=true`. The run should take a few minutes, so there is no autosave
layer.

## Not claimed

Anything about a native multi-state hardware unit. The reference is THRML's
categorical node in software. Also not claimed: other start states, and
label-first updates for the binary arms.
