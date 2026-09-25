# Raised-cap finite-K4 path-KL screen (M4H)

**Status: draft protocol, not approved.** Nothing below has been fitted or run.
Change any frozen value only by revising this document before the study runs.

## Question and scope

Can the existing five-spin, nine-parameter PAsymSwap kernel family meet the
exact parts of the M4G task-quality contract on the complete 500-operation
program once the field/coupling cap is raised from 2 to 4 or 6, when each of
the 37 grouped kernels is compiled directly against its finite-K4 endpoint law?

The [cap-leakage analysis](../research/2026-09-23-cap-leakage-analysis.md)
finds that the best attainable empty-edge and occupied-edge leakage pair
satisfies L₀₀ · L₀₁ ≈ e^(−4c), both at equilibrium and at K = 4. At c = 2 the
floor is 3.4e-4; 95% survival over 500 operations needs about 1e-7. A
reproduction on 2026-09-25 (numpy 2.4.6, scipy 1.17.1, 8 cores, 1 min 52 s)
matched the published table: the c = 2 and c = 4 cells agree to two or three
digits, and the widest difference, one c = 6, K = 4 cell (8.0e-9 versus
6.4e-9), stays about 1,000 times below the target product. That table
optimizes one kernel for leakage alone. This screen asks whether the full
grouped model, fitted for the logical dynamics, reaches the contract.

Evidence class: `exact_reference`, CPU float64, beta 1, zero samples. This is
not an inference-budget result, a sampled M4G replication, or a hardware
claim. At beta 1, energies are linear in the parameters, so cap c is
equivalent to cap 2 at beta c/2. The cap is a Thermo convention; Z1's
numerical field and coupling limits are unpublished.

## Fixture, arms and objective

Keep the paper fixture from `build_paper_fixture()`: the 5×5 torus, the
initial state, the 500 operations and the 37 canonical targets in
target-hash order (multiplicities 26×10, 9×20, 2×30). Keep the five-spin
graph, the uniform reset over the eight free states, the hidden-then-outputs
sweep order and K = 4 for fitting.

Three arms differ only in the cap: c = 2 (control), c = 4 and c = 6. The c = 2
arm separates the effect of the finite-K4 path objective from the effect of
the cap. The archived M4C/M4E initialization and the recorded M4G fits are
context comparators, not arms or fitting sources.

Each group g minimizes its finite-K4 path-KL contribution

    J_g(θ) = Σ_{x ∈ {00, 01, 10}} W_g(x) · KL(T_g(· | x) ‖ Q_θ^{K4}(· | x))

where T_g is the logical target conditional, Q_θ^{K4} is the visible K4
endpoint conditional with the hidden spin marginalized, and W_g(x) is the
exact target visitation mass of input x summed over that group's occurrences.
Input 11 is never visited by the target and carries zero weight. Because the
target stays in the one-particle sector and each operation changes only its
own edge, Σ_g J_g equals the whole-program target-to-model path KL
D_KL(P ‖ Q) at K4. The groups do not share parameters, so the 37 fits are
independent. By the [September 21 certificate](../research/2026-09-21-after-survival-audit.md),
S(500) ≥ exp(−D_KL(P ‖ Q)); D_KL ≤ 0.0512933 nats is sufficient, not
necessary, for S(500) ≥ 0.95.

## Optimizer and budget

Per group and arm: SciPy L-BFGS-B inside [−c, c]⁹ with the existing compiler
settings (maxiter 2000, maxls 50, ftol 1e-12, gtol 1e-9, projected-gradient
tolerance 1e-6) and the exact analytic gradient from the M3 finite-K law.

Twenty starts, identical in number for every group and arm:

- the four existing compiler start roles (`uniform_baseline_warm_start`,
  `fixed_zero`, `fixed_positive`, `fixed_antithetic_negative`);
- eight uniform draws in [−c, c]⁹;
- eight random corners of the box.

Draw the sixteen random starts from `numpy.random.default_rng([20260925,
arm_index, group_index])`. Keep every start's result. A start is admissible
under the existing rules: SciPy success, finite values, projected gradient
within tolerance and every |θ| ≤ c. Select the admissible start with the lowest
J_g and break ties by parameter vector, as the existing compiler does. A group
with no admissible start is an integrity failure of that arm, not a scientific
result. No other search, restart or post hoc selection is allowed.

## Decision contract

The primary screen for each raised-cap arm applies the exact M4G thresholds at
K4, inclusive, in float64 with no epsilon:

| Quantity | Requirement |
| --- | --- |
| Uninterrupted one-particle survival S(500) | ≥ 0.95 |
| Unconditional hop MAE over 37 groups | ≤ 0.01 |
| Asymmetry MAE over 37 groups | ≤ 0.01 |

Population loss and the sampled terminal-leakage gate are outside this exact
screen. S(500) ≥ 0.95 implies exact terminal leakage ≤ 0.05, but the sampled
leakage gate remains a separate requirement of any later study.

Outcomes, fixed before the run:

- **Pass** at c = 4 or c = 6: write a separate sampled M4G-contract protocol at
  the lowest passing cap, with population loss and sampled leakage.
- **Survival passes, fidelity fails**: survival is feasible but the objective
  trades away rare hops, as in the return-fixture pilots. The next test adds a
  predeclared fidelity constraint at the raised cap.
- **Survival fails at both caps**: the cap alone does not explain the failure.
  The next test is the one-feature capacity screen from the September 21 note
  (one extra hidden spin, or one output–output coupling).

The c = 2 arm is not gated. It is reported beside the raised caps.

## Required report

For every arm:

- S(500), hop MAE and asymmetry MAE at K4;
- D_KL(P ‖ Q), the certificate exp(−D_KL), and each group's J_g;
- a per-group table of L₀₀ = P(output ≠ 00 | 00) and
  L₀₁ = max over inputs 01 and 10 of P(output ∈ {00, 11} | input), plus the
  forward and reverse hop probabilities against target;
- S(500) and the K30-versus-equilibrium conditional TV of the K4-fitted
  parameters at equilibrium and K = 1, 2, 8, 16 and 30, as descriptive
  diagnostics only (the 2026-09-01 cap-4 failure was a K30 residual of
  0.411679 against a 0.05 gate);
- bound activity: parameters at ±c per group;
- admissible starts per group and the selected start's index.

## Implementation gates

The exact evaluators currently reject any parameter outside [−2, 2]. Add a
keyword `parameter_cap` with default 2.0 to `finite_sweep_joint_law`,
`endpoint_tables`, `endpoint_laws` and the table helpers the study calls.
With the default, every existing study, archive replay and test must be
unchanged. Do not modify `QualityBudgetProtocol` or any other frozen design.

Before fitting:

1. Replaying the archived M4C initialization through the generalized
   evaluators at c = 2 reproduces its recorded K4 survival (0.00040392135) within
   the existing replay tolerance.
2. The pooled Σ_g J_g equals the enumerated trajectory KL from
   `return_fixture_objectives` on the three-site fixture at a shared parameter
   vector, within 1e-12 relative.
3. All nine gradient components of J_g match centered differences at two
   interior vectors and one vector with components on the bound, for each cap.
4. Evaluating the K4 law at c = 6 corners gives finite, normalized
   probabilities with no underflow to zero on the target support.

## Persistence

Follow the full-row pilot's conventions. The request binds the fixture
digest, arms, objective, starts and seeds, optimizer settings, gates and
source-file hashes. Persist every start's trace, the selected parameters,
digests, runtime provenance and the report as gzipped canonical JSON. Reload
and replay all three arms before writing `completion.json` last. Add a focused
unit test and a reusable CI workflow joined to `ci.yml`.

## Protocol review record

Draft, 2026-09-25. Awaiting review of the arms, the start budget and the
decision contract before any implementation.
