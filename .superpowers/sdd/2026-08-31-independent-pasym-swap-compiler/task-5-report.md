# Task 5: Deterministic Independent PAsymSwap Compiler

## Commit

- `feat: compile independent PAsymSwap kernels`

## Implementation

- Added the pure float64 `independent_compiler` module with frozen
  `CompilerSettings`, `OptimizationAttempt`, and `CompiledKernelArtifact`
  records.
- Implemented exact target-to-model KL and its nine-sufficient-statistic
  gradient through explicit hidden-state and output-state enumeration.
- Added bounded SciPy L-BFGS-B optimization across three explicit deterministic
  restarts. A winner must have a successful SciPy status, finite observations,
  remain within the cap, and meet the independently calculated projected
  gradient gate; selection is deterministic by objective then parameter tuple.
- Artifact hashes canonicalize the target identity, topology, role and
  parameter orders, dtype, learned parameters, beta, cap, and every compiler
  setting. Optimizer observations are separate immutable records and are not
  part of artifact identity.
- Added artifact-only equilibrium evaluation, which has no optimizer callback,
  trajectory, or target/context argument.
- Post-review, artifact hashes are derived during frozen construction and
  nested attempt/identity collections are tuple-copied, so callers cannot
  supply a stale artifact hash or retain mutable artifact storage.

## Tests

- Exact central-difference gradient agreement and active-bound projected
  gradient behavior.
- Deterministic bounded compilation, immutable artifacts, target isolation,
  artifact-only evaluation, and all non-passing-restart failure gates.
- Table-driven artifact-identity coverage for target, topology, role and
  parameter order, dtype, beta, cap, learned values, and every compiler
  setting; optimizer diagnostic exclusions are checked separately.

## RED / GREEN evidence

1. RED: `uv run pytest tests/unit/test_independent_compiler.py -q` failed at
   collection with `ModuleNotFoundError: No module named
   'thermo_lab.independent_compiler'`.
2. GREEN: after implementing the module, the focused test command reported
   `18 passed in 1.45s`.
3. A follow-up RED for separately recording raw SciPy success versus checked
   success failed with `AttributeError: 'OptimizationAttempt' object has no
   attribute 'passed_checks'`; adding the immutable `passed_checks` observation
   restored the focused suite to green.
4. Review regression GREEN: `uv run pytest
   tests/unit/test_independent_compiler.py -q` — `26 passed in 0.96s`.

## Verification

- Ruff format/check on the two changed source/test files — clean.
- Full suite: `uv run pytest` — `329 passed in 16.67s`.
- `git diff --check` — clean before commit.

## Concerns

None. The compiler intentionally accepts only the checked two-bit PAsymSwap
support and uses the design's fixed beta of 1.0; it is not a generic training
or factor-graph interface.
