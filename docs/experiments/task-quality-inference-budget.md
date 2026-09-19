# M4G: task quality versus inference budget

Predeclared September 17, 2026, after M4F and before new M4G fitting or
evaluation. Repository baseline: `7b7b08a7e3c964ff2a103a7adb2156a376ea2855`.
This document freezes the study design. The implementation, uncertainty
validation and full experiment are subsequent deliverables; none has run yet.

## Question and scope

Can training with an explicitly modeled finite Gibbs budget meet the same
task-quality contract at fewer inference sampling operations than
equilibrium-directed training?

The primary budget is **complete Gibbs sweeps per complete program
trajectory**. The number of independent trajectories and local endpoint draws
is fixed. A positive result would establish fewer modeled Gibbs sweeps/p-bit
updates in this benchmark, not fewer independent output samples, actual device
latency or energy savings. Exact endpoint tables simulate the finite sampler;
they do not execute its individual Gibbs transitions.

Use the existing 25-site, 500-operation PAsymSwap task, 37 shared groups of nine
parameters, beta 1, float64, parameter bounds [-2,2], initial one-particle state,
and fixed schedule. Each local draw resets all three free bits uniformly and
uses hidden-then-output complete sweeps. All 500 operations execute even after
particle conservation fails. Do not repair, reject, retry or discard outputs.

The fixed grid is K = 1, 2, 4, 8, 16, 30. Equilibrium is a diagnostic quality
reference with no finite sweep count, never a zero-cost candidate. No
monotonicity assumption or claim about intermediate, untested budgets is allowed.

## Authenticated inputs and matched training

Import the original M1 `source_record` from each committed M4 artifact in
`docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement/seed-*.json`.
Use the M4B archive adapter and its exact whole-record pins, ordered by seed:
hash the validated `RunRecord` canonical representation, as the adapter does,
rather than the unnormalized JSON dictionary or file bytes.

| Seed | Canonical source archive hash |
| --- | --- |
| 0 | `sha256:33796f7354bb72ce1ed0734acea90e1e256d62cd5af826e147c9c433d8cec691` |
| 1 | `sha256:4452312277399f19701bc28914fe607c7749e7644561b2d318803f6a9bdab970` |
| 2 | `sha256:c7edd8fa359d5a47dbf251bb82ead5fc9bcfe5161d6dbb9abd023e8fc4e79f71` |

Preserve archived initial parameters, target and schedule. Independently check
consumed inputs under the existing M4B archive policy; do not rerun unused M1
statistics. The three sources share one initial matrix. New independent
training randomness creates three fitted replications per training condition;
the archives themselves are not independent fitted models. Do not initialize
from M1 updates, M4B trained arms, or M4F endpoints.

For each seed 0,1,2, train six finite-budget models, one at each K, and one
equilibrium-directed model: **21 fits total**. Every model starts from the same
archived initialization and uses the M4B training objective
L = sum over 25 sites of (terminal population occupancy - logical target)^2.
Keep five projected updates, learning rate 0.01, and select update five in
advance. Each update uses independent occupancy and gradient batches of 32,768
complete trajectories. Use reward coefficients 2*(estimated occupancy-target),
the model's own training law for both roles, independent same-parent gradient
reference draws that never propagate, and shared-occurrence gradient summation.
Finite models use the checked finite endpoint scores at their K; the equilibrium
model uses its checked equilibrium estimator. Only the training law differs.

Do not add M4F's local conservation/asymmetry losses to either arm: this study
extends the matched M4B population objective across budgets. Local fidelity and
conservation are acceptance requirements, not new training objectives. Five
updates test the attained bounded procedure, not optimized model capacity.

Randomness: `SeedSequence([0x4D3447, source_seed]).spawn(71)`. Convert each child
with `int(child.generate_state(1, dtype=np.uint64)[0])`, as in M4B, and require
all role integers across the study to be distinct. Children 0:60 comprise ten
roles for each finite K in ascending order; 60:70 are the equilibrium roles.
Within each block alternate occupancy/gradient for updates 1 through 5. Child
70 is exclusively final evaluation. No training, tuning, early stopping or
model selection may consume the final evaluation stream.

## Evaluation matrix and output contract

Freeze every fit before final evaluation. Each cell executes N = 32,768
independent complete trajectories and estimates the terminal occupancy vector
from their binary site states. No majority vote, postselection or extra draws.

| Member | Evaluated horizons per seed | Cells across three seeds |
| --- | --- | ---: |
| Finite-budget-trained | Each model only at its own training K | 18 |
| Equilibrium-trained | All six finite K, plus equilibrium reference | 21 |
| Frozen initialization | All six finite K, plus equilibrium reference | 21 |
| Total | 20 cells per seed | 60 |

This compares a predeclared horizon-specific training procedure with one
equilibrium-trained model per seed. It is not a transfer sweep of one finite
model. Count the entire six-fit finite grid's training cost, not just the fit
at a subsequently passing horizon. Do not add cross-horizon finite-model cells.

For each seed, reset PCG64 to the same final role seed for every cell and use
one length-N uniform vector per occurrence to sample the endpoint categorical
law. Execute each distinct cell once. This preserves paired outcomes across
members/horizons without treating cells as independent replications. Trajectory
rows within a cell remain independent under the declared simulator model.
Do not pool counts across differently fitted seeds.

Persist all 25 terminal occupancy counts, the complete terminal particle-count
histogram and its leakage count, and sufficient joined terminal moments for
paired finite-minus-equilibrium loss diagnostics at each finite K. Retain the
M1 unbiased order-two population-loss estimate and pointwise paired jackknife
diagnostics, explicitly descriptive. They do not determine acceptance.

Compute local visible tables, killed one-particle survival through every
operation, unconditional hop MAE and signed-direction asymmetry MAE exactly
from each cell's frozen parameters and execution law, using the existing
M4C–M4F definitions. Keep all-parent failures, conditional hop errors and empty
edge creation visible as additional diagnostics. Exact calculations carry no
Monte Carlo confidence intervals. Survival excludes paths that leave and later
return to one particle; terminal leakage measures only the final particle count.
Survival is checked after each of the 500 completed logical operations. It
does not certify particle conservation during local resets or intermediate
hidden/output Gibbs transitions.

## Numeric task-quality contract

Every requirement must pass together in each of the three fitted replications.
The thresholds are benchmark requirements selected before M4G outcomes, with
knowledge of the historical failures. They are not externally validated product
requirements or a fidelity guarantee for other tasks.

| Quantity | Requirement | Reason |
| --- | --- | --- |
| Terminal population occupancy L | L <= 0.0625 | RMS probability error across 25 sites <= 0.05 |
| Terminal particle leakage | P(final particle count != 1) <= 0.05 | At least 95% valid terminal particle counts |
| Uninterrupted one-particle survival | S(500) >= 0.95 | At least 95% paths preserve the invariant at every operation endpoint |
| Unconditional hop MAE | <= 0.01 | Mean absolute directional hop error at most one probability point |
| Asymmetry MAE | <= 0.01 | Mean absolute directed-hop difference error at most one probability point |

L is the unnormalized sum-squared population loss used in M4B; its RMS is
sqrt(L/25). It is not a plug-in empirical loss or a full joint-distribution
distance. Hop MAE averages the two directions over all 37 groups; asymmetry MAE
averages the signed forward-minus-reverse difference error over those groups.
Neither is weighted by target-context frequency. All comparisons are inclusive
at equality; exact metrics use literal float64 comparisons without a scientific
epsilon. Historical numerical replay tolerances never relax these thresholds.

Under the same execution law, S(500) >= 0.95 implies terminal leakage <= 0.05.
The separate sampled leakage gate is deliberately retained as a conservative
requirement on the final evaluation evidence; it can remain unresolved even
when exact survival passes. Do not substitute the exact implication for that
gate after observing counts. Nonfinite values, malformed counts, numerical
inversion failures or failed replay are integrity errors, never scientific
fail classifications.

## Simultaneous uncertainty and budget selection

Use a two-sided equal-tailed Clopper–Pearson interval for each terminal binary
mean and the leakage proportion. There are 60*(25+1) = **1,560 intervals**,
including the six equilibrium diagnostic cells. Fix family alpha = 0.05 and
per-interval alpha a = 0.05/1560. Do not recycle unused alpha or switch to a
pointwise rule after seeing results. For count x from N trials:

- lower = 0 if x=0, otherwise BetaQuantile(a/2; x, N-x+1);
- upper = 1 if x=N, otherwise BetaQuantile(1-a/2; x+1, N-x).

This is the exact binomial method implemented by
[SciPy's proportion_ci](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
Each marginal has coverage at least 1-a under binomial sampling; the union
bound gives joint coverage at least 0.95 without requiring independence across
sites, horizons, members or seeds. The statement is conditional on the frozen
fits and simulator sampling assumptions, not a population guarantee over future
training runs. The conservative rule may leave comparisons unresolved.

For occupancy intervals [l_i,u_i] and logical target q_i, construct
L_lower = sum_i distance(q_i,[l_i,u_i])^2 and
L_upper = sum_i max((l_i-q_i)^2,(u_i-q_i)^2).
On the joint coverage event these bound the population loss, regardless of
within-trajectory site dependence. Apply the leakage interval directly.

Cell classification is fixed:

1. **Pass:** L_upper <= 0.0625, leakage upper <= 0.05, and all three exact
   requirements pass.
2. **Fail:** L_lower > 0.0625, leakage lower > 0.05, or any exact requirement
   fails.
3. **Unresolved:** neither pass nor fail.

A method/K passes only if all three seed cells pass; it fails if any seed cell
fails, and is otherwise unresolved. This defines a common budget that works for
all three checked fits, not a pooled-average or majority-vote criterion.
Report each seed's classification alongside the joint one. Frozen controls use
the same rule but never establish trained-method savings.

For each method define K_possible as the smallest finite K not classified fail,
and K_certified as the smallest finite K classified pass, using +infinity when
the corresponding set is empty. On the joint coverage event, the actual
smallest passing tested K lies in [K_possible,K_certified]. If the endpoints
coincide and are finite, the minimum is resolved. If every cell fails, report
quality failure within the grid. If none passes but any is unresolved, report
inconclusive quality, not demonstrated failure or a numerical savings ratio.

Claim a lower-budget result only if both trained methods have a finite
K_certified and K_certified(finite) < K_possible(equilibrium). The sweep-saving
ratio is bracketed by
[K_possible(equilibrium)/K_certified(finite),
 K_certified(equilibrium)/K_possible(finite)]. Report a single minimum-budget
ratio only when both minima resolve. Equal resolved minima mean no demonstrated
savings; overlapping brackets are inconclusive. Do not label the first
certified passing cell the true minimum when cheaper cells remain unresolved.

## Operation accounting

One complete program trajectory contains 500 local endpoint draws/resets,
500*K modeled complete sweeps, and 1500*K modeled p-bit updates. A local reset
uniformly initializes three free bits; record these initialization draws
separately from sweep updates. The fixed output batch multiplies these counts
by 32,768. Endpoint draws and independent trajectory counts do not decrease
when K decreases. Report both the per-trajectory and per-batch costs.

Each fitted model uses 5*N*500*3 = 245,760,000 training endpoint draws, counting
occupancy, gradient main and gradient reference. References are local draws,
not independently propagated full trajectories. A finite fit at K has
245,760,000*K sweep equivalents and three times as many p-bit updates, plus
one free-state reset per endpoint draw. Equilibrium oracle draws have no finite
sweep equivalent; record their draw count and table/enumeration work separately.

Across three seeds, the finite procedure costs six fits per seed and the
equilibrium procedure one: 18 versus 3 fits. Each matched K comparison has equal
updates and training endpoint draws, while the complete finite grid costs six
times as many draws. Total training is 5,160,960,000 endpoint draws. Evaluation
uses 60*N*500 = 983,040,000 endpoint draws, of which 98,304,000 are equilibrium
reference draws. Paired reuse of randomness does not eliminate any execution.

Record exact table/derivative construction, source authentication, full replay,
reporting and I/O work separately from training/evaluation sampling. Record
NumPy timings with included operations and runtime provenance; they are not
device timing. Never call oracle work free, equate endpoint draws to physical
Gibbs steps, or infer hardware energy from these operation counts.

## Implementation and release gates

Before any new fitting or final evaluation:

1. Implement the immutable request, role schedule, cost ledger and acceptance
   calculations from this protocol. Hash all declared inputs and decisions;
   keep observed derived values and runtime metadata out of request identity.
2. Validate intervals independently against binomial-tail inversion, including
   x=0 and x=N, interval nesting, and values immediately around thresholds.
   Enumerate all counts for N=1,2,8,32 over p in {0,0.0001,0.001,0.01,0.05,
   0.5,0.95,0.99,0.999,0.9999,1}; check nominal marginal coverage. Also check
   the production N=32768 interval implementation at those p values. Retain the
   coverage proof; a finite grid alone cannot establish uniform coverage.
3. Use an enumerable joint binary fixture to check the occupancy rectangle
   bounds, deliberately correlated cells to check the simultaneous decision
   logic, and adversarial pass/fail/unresolved horizon patterns to verify
   budget brackets and ratio claims. Never assume monotone quality in K.
4. Validate both training laws on the existing bounded three-site fixture at
   every finite K and preserve the M3 gradient contract. Verify source pins,
   same initialization, law-specific roles, fixed updates and held-out seeds.
5. Obtain independent statistical and implementation review, and pass the
   preflight integrity checks. Record the frozen protocol commit before fitting.

Then execute all 21 fits and 60 evaluation cells with no outcome-driven
budget changes, coefficient searches, restarts or checkpoint selection. A
failed quality screen does not permit early omission of remaining cells.
Infrastructure retries must preserve the original inputs/seeds and record the
retry; they do not add statistical replications. If preflight reveals a design
error, amend and version the protocol before fitting; do not silently change it.

Fresh outputs must retain all steps, role seeds, exact tables, source lineage,
terminal counts/histograms, joined moments, intervals, all 60 classifications,
costs and method-level budget brackets. Reload must reconstruct the request,
replay new numerical evidence under a declared versioned replay policy, and
reject rehashed tampering. Preserve old validators, artifact pins and schemas.
Write completion last, only after complete report validation.

M4G completion requires all cells, reproducible quality-versus-cost curves,
independent evidence review, repository gates and a recorded decision. Sampled
outputs are `software_simulation`; local laws and killed survival are
`exact_reference`. Negative or unresolved scientific outcomes are valid
completions. Record the outcome before moving to M5.

## Protocol review record

Two independent read-only reviews completed September 17 before publication.
The statistical review checked interval coverage, rectangle loss bounds,
nonmonotone budget brackets, equality/infinity cases and ratio conditions. The
second review checked the source pins, actual M4B training logic, matched versus
total work, horizon-specific diagnostics and roadmap requirements. Both found
the design coherent. Clarifications now explicitly retain the conservative
leakage gate and scope survival to completed logical-operation endpoints.

Documentation validation checked local links, unresolved placeholders, the
loss/RMS conversion, fit/cell/interval counts and all declared draw totals.
Ruff formatting/lint and whitespace checks passed. This is documentation-only;
no new experiment, coverage implementation or production acceptance calculation
has been executed. The preflight requirements above remain release gates for
the subsequent implementation, not completed claims.
