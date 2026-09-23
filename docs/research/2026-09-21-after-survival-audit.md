# Improving conservation after the survival-gradient audit

Targeted primary-source research, September 21, 2026. This is a research proposal,
not a new fitted experiment, changed M4G protocol, or hardware result. It builds
on the [previous review](2026-09-20-conservation-directions.md) and
[completed audit](../experiment-reports/2026-09-21-survival-gradient-audit/summary.md).

## What changed our priorities

103 of 105 saved updates increased survival; every fit improved overall. At K30,
survival nevertheless moved only from 3.3211% to 3.3776–3.3803%. Directional
predictions matched all observed signs. The audit weakens the explanation that
occupancy training generally fights conservation. It does not distinguish small
steps, an insufficient objective, or a restrictive model family.

The full computation has 500 operations. For illustration, S=0.0338 corresponds
to a constant-equivalent per-operation failure probability of
1-S^(1/500)=0.00675169, or 0.675169%. S=0.95 corresponds to 0.0102581%.
Total log hazard must fall from about 3.38729 to at most 0.0512933: roughly a
66-fold reduction. These are arithmetic summaries, not measured uniform hazards
or an independence assumption. Actual hazards are conditional on survival so far
and vary by operation. The gap is substantial despite small local-looking errors.

## 1. Add a whole-trajectory error budget and conservation certificate

Extropic derives a chain rule expressing target-to-model trajectory KL as the
sum of local conditional KL errors weighted by target visitation (Appendix C).
Its compiler also distinguishes exact gradients from sampled training.
[Thermalizing Stochastic Programs, v2](https://arxiv.org/pdf/2608.01615).

**Deduction for Thermo:** let P be the logical path distribution and Q the
compiled path distribution at a fixed finite K, with the same initial state and
operation schedule. Let A mean that every completed operation preserves the
particle count. P(A)=1 and S=Q(A). Then

    D_KL(P || Q) = -log S + D_KL(P || Q(. | A)) >= -log S.

Consequently D_KL(P || Q) <= -log(0.95), approximately 0.0512933 nats,
is sufficient for S >= 0.95. The KL direction is essential: reverse KL is
infinite when Q assigns mass to paths forbidden by P. This certificate is
sufficient, not necessary; failure to meet it does not prove failure of survival.
It certifies only survival, not every M4G quality requirement. A production
certificate would need conservative numerical error control at the boundary.

The same KL can be computed as a sum over 500 local endpoint laws using exact
logical visitation. The target has only 25 one-particle locations; we need not
enumerate all invalid model paths. Hidden spins must be marginalized, and finite
K must use its actual reset-and-sweep endpoint law, not equilibrium as a proxy.

This suggests a cheap next diagnostic: compute exact trajectory KL and its
operation/group contributions for the existing checkpoints, compare its
certificate with exact survival, and finite-difference its gradients. Always
sum repeated group contributions before updating shared parameters. Target
visitation weights differ from both uniform row weights and model-survivor
weights. Prior target-context work makes this partly a reuse of existing ideas;
the new value is the full-path budget, explicit certificate and saved-update
comparison, not a claim to invent context matching.

## 2. Test objective choice separately from step size

Trust-region methods constrain distributional change while optimizing an
objective; constrained policy optimization separates an objective from behavior
constraints. Their published RL guarantees do not automatically apply to this
bounded, shared thermodynamic kernel.
[TRPO](https://proceedings.mlr.press/v37/schulman15.html),
[CPO](https://proceedings.mlr.press/v70/achiam17a.html).

**Proposed Thermo pilot:** keep the current architecture, bounds and starting
point fixed. Begin on the enumerable three-site fixture so objective and
optimizer comparisons use exact gradients rather than different noise levels.
Compare the existing occupancy objective, exact trajectory KL, and a joint
valid-terminal objective. Cross objective choice with the same two step rules:
the historical fixed step and a bounded backtracking rule with equal evaluation
budgets. A comparison that changes objective and optimizer simultaneously cannot
identify which helped. Keep an explicit unchanged baseline.

For the joint objective, m_i is unnormalized probability of surviving the entire
path and finishing at i; S=sum_i m_i and r_i=m_i/S. For target terminal law q,

    J = sum_i q_i log(q_i/m_i) = -log S + D_KL(q || r).

This previously proposed objective penalizes lost valid probability and incorrect
valid outputs together. It does not guarantee correct intermediate transitions.
Survival alone is inadequate: a kernel that refuses to move can conserve count
while missing the intended dynamics. Keep unconditional hop/asymmetry, terminal
quality and exact survival as separate reported quantities.

Only after the fixture should a new, predeclared full-program diagnostic compare
these directions. A short objective/step-size study is an optimization diagnosis,
not evidence of inference-sweep savings. Exact oracle work must be charged and
labeled; it is not a free substitute for the sampling-budget experiment.

## 3. Test representational capacity and finite-budget mixing together

Conditional Boltzmann-machine theory establishes that representable conditional
distributions depend on architecture and hidden-unit count. Those results do not
establish feasibility for Thermo's specific graph, parameter caps or finite K.
[Montufar, Ay and Ghazi-Zahedi](https://www.jmlr.org/papers/v16/montufar15b.html).

Our kernel has one hidden spin and no direct output-output interaction. Given
the final hidden spin, outputs are sampled independently. The previous review's
row leakage/fidelity inequality is a candidate mathematical diagnostic, not yet
a reviewed whole-program impossibility result. It should not be used to declare
this architecture incapable of reaching the target.

An exact local capacity screen should compare the existing family with one
additional hidden spin and with one output-output coupling, changing one feature
at a time. Use the same caps, targets and optimization budget; report attained
trade-offs, not a certified global optimum. Evaluate equilibrium and the actual
finite-budget laws separately. An output-output edge changes the legal Gibbs
schedule and operation count; it cannot retain the existing independent output
block unchanged.

Extropic also discusses a mixing/expressivity tension and adaptive correlation
penalties in a separate thermalization construction (Appendix L). Pulling free
spins toward independence may aid mixing, but is not a conservation remedy for
our anticorrelated one-particle outputs. Its conditional-CD caveat is distinct
from Thermo's differentiated finite-K endpoint law. Treat mixing measurements
and equilibrium-versus-finite-K discrepancies as diagnostics before borrowing
that regularizer. [Thermalizing Stochastic Programs](https://arxiv.org/pdf/2608.01615).

## 4. Keep a structurally conserving sampler as a separate execution law

THRML exposes custom conditional samplers and ordered blocks. A SuperBlock
supplies the same preceding global state to its constituent blocks; it does not
itself enforce joint count conservation.
[Conditional samplers](https://docs.thrml.ai/en/latest/api-samplers.html),
[Block sampling](https://docs.thrml.ai/en/latest/api-block-sampling.html).

A count-preserving pair update can prohibit 10/01 from turning into 00/11 at
operation endpoints. We already have an ideal logical reference doing this;
another ideal reference would add little. New value would come from an actual
custom sampler with enumerated transition laws, checked asymmetric-hop fidelity
and explicit costs for gating, randomness, pair updates and state handling.

Finite energy penalties cannot make forbidden outputs exactly impossible. Large
penalties may also make valid transitions difficult under single-bit updates;
checking conservation without checking movement and mixing would be misleading.
A software pair sampler is not evidence of a native Z1 primitive. Extropic's
announcement describes chromatic Gibbs on sparse Ising hardware and prospective
system access. [Official announcement](https://extropic.ai/writing/from-one-to-one-billion).

## 5. Reserve rare-event methods for when exact propagation stops scaling

Controlled sequential Monte Carlo adapts proposal distributions through an
optimal-control formulation. Doob transforms similarly connect conditioned
paths and changed transition laws.
[Heng et al.](https://arxiv.org/abs/1708.08396),
[Chetrite and Touchette](https://arxiv.org/abs/1405.5157).

For our finite-horizon killed matrices Q_t, define h_T=1 and
h_(t-1)=Q_t h_t. Where h_(t-1)(i)>0,

    Q*_t(i,j) = Q_t(i,j) h_t(j) / h_(t-1)(i)

is the transition law conditioned on future survival. This finite-horizon
identity is a direct deduction, not an invocation of the cited paper's
long-time large-deviation theorem. It provides a teacher/diagnostic showing
which transitions would favor surviving paths. It generally depends on time,
global particle location and future schedule; the current shared local kernel
may not represent it. Conditioning preserves the model's survivor distribution,
not necessarily the logical task distribution.

Weighted rare-event proposals can improve estimation when exact state tracking
becomes infeasible. They do not improve the original sampler's success rate.
Counting conditioned samples as ordinary successful runs would invalidate the
comparison. For the present 25-state survival calculation, exact propagation is
the simpler first choice.

## Recommended sequence and decision rules

1. Add and verify the exact trajectory-KL diagnostic/certificate on the fixture,
   then apply it to all saved checkpoints without refitting.
2. Run a small objective-by-step-rule comparison on the fixture. If a better
   step rule helps both objectives similarly, investigate optimization strength;
   if path-aware objectives help under matched step rules, prioritize them.
3. If gains saturate or fidelity deteriorates, run the bounded local capacity
   screen. Failure of local optimization is evidence about attained solutions,
   not proof of a global representational limit.
4. Evaluate a count-preserving sampler under a separate protocol if structural
   conservation is worth changing the execution model.
5. Revisit matched inference-budget training only after a method can satisfy
   the unchanged joint quality contract. Preserve M4G as the historical result.

Predeclare budgets, stopping/selection rules and development/evaluation roles.
Do not recycle M4G's held-out seeds for new tuning or fresh confirmation. Retain
all arms and costs, including negative results. These proposals require a
concrete experiment protocol before implementation; this research adds no new
model fit, sampling run or retrospective selection of a winning checkpoint.
