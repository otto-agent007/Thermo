# One-feature kernel-capacity screen (M4I)

**Status: frozen 2026-09-25 before any fit.** This is the final experiment
in the conservation line that began with M4B. Change a frozen value only by a
new protocol version.

## Question and scope

Does adding one feature to the five-spin PAsymSwap kernel, either a second
hidden spin or a direct output–output coupling, let the 37 grouped kernels
meet the exact parts of the M4G task-quality contract at K4 on the complete
500-operation program?

The [raised-cap screen](../experiment-reports/2026-09-25-raised-cap-path-kl-screen/summary.md)
(M4H) recorded `survival_fails_at_both_caps`. Raising the cap from 2 to 4
lifted exact S(500) at K4 from 3.64% to 58.58%, and cap 6 reached 70.74%.
Empty-edge leakage became negligible, but occupied-edge leakage stayed near
6% and hop and asymmetry errors stayed about 2 and 4 times over their
limits. The predeclared next step was this screen, as proposed in section 3
of the [September 21 note](../research/2026-09-21-after-survival-audit.md):
change one feature at a time, keep caps, targets and budget the same, and
report attained trade-offs rather than a certified optimum.

Evidence class: `exact_reference`, CPU float64, beta 1, zero samples. This is
not a sampled M4G result, an inference-budget result or hardware evidence.
Z1's connectivity and caps are unpublished, so neither added feature is a
claim about what Z1 can realize.

## Kernel families

The base family is the existing nine-parameter kernel. With spins s = 2b − 1,
its log weight is θ · φ over the statistics hidden h, outputs o₀ and o₁,
the four input–output products and the two hidden–output products. It has
no input–hidden and no output–output term, so a sweep updates h given the
outputs, then o₀ and o₁ together given the inputs and h.

| Family | Added statistics | Parameters | Free states | Sweep phases | Spin updates per sweep |
| --- | --- | --- | --- | --- | --- |
| Base | none | 9 | 8 | 2: {h}, {o₀, o₁} | 3 |
| H2 (second hidden spin g) | g, g·o₀, g·o₁ | 12 | 16 | 2: {h, g}, {o₀, o₁} | 4 |
| OO (output–output coupling) | o₀·o₁ | 10 | 8 | 3: {h}, {o₀}, {o₁} | 3 |

H2 keeps the bipartite two-color schedule: given the outputs, h and g are
independent, and given both hiddens and the inputs, the outputs are
independent. OO breaks the output block, so o₀ and o₁ are updated in
separate phases, in that fixed order. Each family resets uniformly over its
free states before every operation, as the base family does.

Implement every family as single-spin Gibbs updates in the fixed order of
the table (h, g, o₀, o₁ for H2; h, o₀, o₁ otherwise) over the enumerated free
states. Within a phase whose spins are conditionally independent, sequential
single-spin updates equal the block update, so the base family in this form
is the M3 law. The exact survival evaluator reads only the visible law, so a
family's visible (4, 4) law is passed to it as a (4, 8) table with the second
hidden half zero.

Both families nest the base: with the added parameters at zero, g is
independent of the outputs and o₀, o₁ are conditionally independent given h,
so the visible K law equals the base law exactly.

The spin-update counts are algorithmic counts per sweep, not measured device
operations. OO's third phase and H2's extra spin mean K4 is not an
equal-cost comparison across families; report the counts beside the results.

## Arms

Four new arms: H2 and OO, each at caps 2 and 4. The M4H base arms at caps 2
and 4 are the comparators. They are authenticated and replayed from the
archive without refitting, not refit here. Cap 6 is omitted because in M4H it
improved survival modestly over cap 4 and left hop and asymmetry errors
unchanged. Cap 2 tests whether capacity alone suffices under the current
convention. Cap 4 tests the feature together with the larger range.

## Objective, optimizer and budget

Unchanged from M4H. Each group g minimizes

    J_g(θ) = Σ_{x ∈ {00, 01, 10}} W_g(x) · KL(T_g(· | x) ‖ Q_θ^{K4}(· | x))

on its own family's visible K4 law, with W_g the exact target visitation.
Σ_g J_g is the whole-program path KL, and S(500) ≥ exp(−Σ_g J_g). Use SciPy
L-BFGS-B inside [−c, c]ⁿ with the M4H settings (maxiter 2000, maxls 50,
ftol 1e-12, gtol 1e-9, projected-gradient tolerance 1e-6) and an exact
analytic or automatic-differentiation gradient.

Twenty-one starts per group and arm:

- `m4h_warm_start`: the M4H selected parameters for that group at the same
  cap, with the added parameters at zero;
- `archived_initial`, `fixed_zero`, `fixed_positive` and
  `fixed_antithetic_negative` as in M4H, with added parameters at zero for
  the first and continuing the ±0.05 alternation for the fixed signs;
- eight uniform draws in [−c, c]ⁿ and eight random box corners from
  `numpy.random.default_rng([20260925, 1, family_index, cap_index, group_index])`.

The warm start is the one addition to the M4H budget. It makes each new arm a
nested local search from the base solution, so a selected objective above the
base group's M4H objective can only come from an inadmissible warm start;
report every such case. Admissibility, selection, integrity failure and "no
post hoc search" follow M4H. Groups may be fitted in parallel processes; the
record must not depend on scheduling.

## Decision contract

This screen closes the conservation line. Whatever the outcome, no further
conservation pilot, fixture study or objective variant follows from it; the
next milestone is M5 (the topology-aware meta-EBM).

For each new arm, the exact M4G thresholds at K4, inclusive, float64, no
epsilon: S(500) ≥ 0.95, hop MAE ≤ 0.01 and asymmetry MAE ≤ 0.01 over the 37
groups. Population loss and sampled leakage stay outside this exact screen.

Outcomes, fixed before the run:

- **Pass** by any arm: one sampled M4G-contract study follows, for the
  passing arm with the lowest cap, then the fewest added parameters (OO before
  H2). It runs alongside M5, not before it.
- **Survival passes, fidelity fails** in some arm and no arm passes: record
  it; no follow-up. The capacity question is answered for this objective.
- **Survival fails in every arm**: a single added feature is not enough at K4
  under this objective. Record it; no follow-up.

Both-features, longer-K and structurally conserving variants stay listed as
open questions in the report, not as scheduled work.

## Required report

For every new arm and each base comparator:

- S(500), hop MAE, asymmetry MAE and path KL with exp(−KL), at K4 and,
  separately, at equilibrium;
- S(500) at K = 1, 2, 8, 16 and 30, as descriptive diagnostics;
- the change in S(500), hop MAE and asymmetry MAE against the base arm at the
  same cap;
- a per-group table of L₀₀, L₀₁, both hops against target, J_g, the base
  J_g at the same cap, and parameters at ±c;
- admissible starts per group, the selected start's role, and every group
  whose selected objective exceeds its M4H base objective;
- spin updates and phases per sweep for each family.

## Implementation gates

Leave the shared evaluators and `raised_cap_screen.py` unchanged; archived
studies bind their hashes. Import M4H's inputs, visitation, constants and
gates from `raised_cap_screen.py` rather than copying them. Before fitting:

1. **Nesting**: with added parameters at zero, each family's visible K4 and
   equilibrium laws equal the M3 base laws within absolute 1e-12 at the
   archived initialization and at the M4H selected fits, and each warm start's
   J_g equals the archived M4H objective within relative 1e-9.
2. **Valid Gibbs kernel**: every single-spin transition row sums to one within
   1e-12, and each family's full sweep leaves its enumerated Boltzmann
   conditional stationary within 1e-12 at three parameter vectors per cap.
3. **Gradients**: all components of J_g match centered differences (step
   1e-6, scaled error at most 1e-6) for groups 0, 18 and 36 at two interior
   vectors and one on the bound, for each new arm.
4. **Base replay**: the M4H cap-2 and cap-4 base metrics replay from their
   stored parameters through `raised_cap_screen.evaluate_arm` within relative
   1e-9 and absolute 1e-12, with the archive pinned by SHA-256.

## Persistence and CI

Follow M4H. The request binds families, arms, objective, starts and seeds,
optimizer settings, gates, the pinned M4H and local-trade-off archives, and
source-file hashes. Persist every start's trace, the selected parameters,
digests, provenance and the report as gzipped canonical JSON, and replay
completely before writing `completion.json` last. Add no workflow: the
existing unit-test CI job runs the focused tests, which replay the archived
evidence without refitting. The full run is a local gate.

## Protocol review record

- 2026-09-25: drafted.
- 2026-09-25, before any fit: the project owner approved the families, caps
  and budget, and made this the final conservation experiment with M5 next
  regardless of outcome. The decision contract and implementation notes
  above reflect that.
