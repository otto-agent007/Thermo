# Exact return-fixture full-row fidelity pilot

Freeze before fitting. This is one deterministic `exact_reference` CPU
comparison, with zero samples. Keep the existing three-site K4 fixture,
initial state 100, operations (0,1), (1,2), (0,1), shared nine parameters,
float64, beta 1 and bounds [-2,2]. The exact base evaluator enumerates all
64 complete visible paths, retains the full terminal law, and kills survival
at the first invalid particle count.

Compare four fixed objective weights 0, 1, 10 and 100. Each arm minimizes
complete-path target-to-model KL plus its weight times the sum of squared
errors over all 16 entries of the four-input-row by four-output-column K4
visible conditional. Give each input row equal weight, including row 11,
which the valid target path never visits. Differentiate the full conditional
through the same shared nine parameters. Weight zero must reproduce the
archived path-KL/backtracking arm; the saved hop-only pilot is a contextual
matched-budget comparator, not an extra arm or fitting source.

Use 201 complete evaluator calls per arm: initialization and 25 rounds of
eight projected proposals. At each round try the fixed steps [1, 1/2, 1/4,
1/8, 1/16, 1/32, 1/64, 1/128] from that round's start. Accept the first
with negative gradient-dot-displacement and objective at most the initial
value plus 0.0001 times that displacement. Evaluate and retain all eight
proposals, including rejected and later ones. No accepted step leaves the
parameters unchanged. Keep all four final arms without post hoc selection.

The pilot gate requires exact uninterrupted survival >= 0.95 and maximum
absolute difference between model and target conditional probabilities
<= 0.005 over *all 16 entries*. Thus a zero forward 10→01 probability
cannot pass: its target is 0.009628878. Report each row's maximum absolute
error, total variation, four-outcome MAE, all-row maximum, the two logical
hop probabilities, path KL, survival, target visitation, bound activity and
accepted rounds. No pass is a negative result at this frozen budget, not a
proof of kernel capacity or convergence.

Check all nine combined objective-gradient components against centered
differences at the initial parameters and a second interior vector.
Bind the request to the circuit, objective, weights, gate, evaluator costs,
source hashes and run selection. Persist the full evaluated trace, digests,
runtime provenance and report; reload and completely reconstruct all arms
before writing completion last. Do not infer 500-operation quality, inference
sample savings or hardware performance.
