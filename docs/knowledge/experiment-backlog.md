# Experiment backlog

*Draft, September 29, 2026; E0 added October 1; native-experiment status updated
October 2. Candidates below come from the [source cards](sources/) and remain
unscheduled/unfrozen. M5a, M5b and M5c are recorded, with complete execution
costs still open. Native Max-Cut, denoising and JAX sampling experiments are now
recorded; Potts, associative memory and Torx Bayesian/state-space inference
remain open charter directions. This backlog does not replace that work.*

## Recorded native-inference findings, October 2

The [sampling synthesis](../research/2026-10-02-sampling-synthesis.md) connects
six reports: optimization pilots, fixed-budget sampling, time to accuracy,
symmetry plus tempering, changing evidence and conditional estimation. They
support target-specific sampler choices and substantial weak-regime gains from
conditional estimates, while leaving strong-coupling mixing and decision
utility unresolved. Grouped spin updates with the improved estimator are the
next proposed native mechanism test, not a completed or frozen study.

E0 remains open. The JAX samplers have their own exact fixtures, but those do
not establish THRML's finite-K behavior on the checked five-spin chain or M5
kernels. Native sampling progress is not evidence that E0's contract passed.

## Scope decision for the owner

Items E1, E2, E4 and E5 introduce a **continuous-variable substrate**
(Langevin and OU dynamics) implemented in NumPy or JAX, outside Torx and
THRML. That is a new line for Thermo. It is closer to Normal Computing's
hardware than to Z1's p-bits, so it can inform methods but not Z1 claims.
E0 and E3 stay discrete and use only dependencies Thermo already pins
(THRML 0.1.4, NumPy and SciPy).

## Open questions

0. At small sweep budgets, where equilibrium no longer hides differences in
   convention, does THRML's block-Gibbs sampler produce the same state
   distribution as Thermo's exact sweep matrix? (E0)
1. Does training for a finite observation time (Whitelam) have the same exact
   gradient structure that M3 verified for finite Gibbs sweeps? (E1)
2. On a target small enough to know exactly, does reverse-trajectory training
   generate the target, and does it emit less heat than a model trained only
   to match the final distribution? (E2)
3. Does online, per-step training reduce heat and precision needs on the same
   target, as Wang and Deng report on MNIST? (E2, secondary arm)
4. How much irreversibility do the compiled M5 sweeps have, and does it track
   cap, stationary bias or inner-sweep count? (E3)
5. For OU-based linear solves, how does error scale with conditioning,
   integration time and sample count, before any hardware assumption? (E4)
6. As the quartic term grows, when does a continuous double-well network
   reproduce the discrete Ising target that THRML samples? (E5)

## Ranked candidates

| # | Experiment | Source | Exact anchor | Cost | Fit |
| --- | --- | --- | --- | --- | --- |
| E0 | THRML finite-sweep contract against Thermo's exact kernel | [THRML](sources/extropic-thrml.md) | Exact 32-state sweep matrix; Ising enumeration | Minutes, CPU | Checks a dependency already in use |
| E1 | Langevin trajectory-gradient contract | [Whitelam, gradient descent](sources/whitelam-gradient-descent.md) | Closed-form gradient and optimum; exact OU output when J₄ = 0 | Seconds, CPU | M3's continuous analogue |
| E2 | Tiny generative thermodynamic model with heat accounting | [Whitelam, generative](sources/whitelam-generative.md); [Wang and Deng](sources/wang-deng-online-learning.md) | Gaussian target with J₄ = 0; exact target density | Minutes, CPU | Tests a stated physical claim |
| E3 | Entropy production of compiled M5 sweeps | [Thermalizers](sources/extropic-thermalizers.md) | Exact 4,096-state sweep matrices | Minutes, zero samples | Reuses M5a/M5b archives |
| E4 | OU linear-algebra warm-up | [Thermodynamic linear algebra](sources/aifer-thermodynamic-linear-algebra.md); [thermox](sources/normal-thermox.md) | Direct solve; closed-form OU transition | Seconds, CPU | Bounded, but a new dependency if thermox is used |
| E5 | Discrete-to-continuous bridge | THRML plus the Langevin model from E1 | Ising enumeration | Minutes, CPU | Connects the two substrates |

**E0** is ranked first. Sun proposed it in review, and it checks an
assumption that every THRML result in this repo already relies on. Today
Thermo compares THRML with exact references only near equilibrium: the smoke
test runs 200 warm-up sweeps, and the PAsymSwap backends compare at K = 30.
At those budgets, a mismatch in initialization, block order or the definition
of a sweep washes out. The two existing paths also initialize differently
(`hinton_init` in `thrml_local.py`, a uniform draw in `thrml_pasym_swap.py`).
M5's finite-K results are exact NumPy/SciPy matrices that have never been run
through THRML.

E0 is not one of the charter's native-THRML algorithm tracks (Potts,
associative memory, Max-Cut), and it doesn't wait for them. It is the check
those tracks, and any later THRML execution of M5's kernels, should be able to cite.
Like E3, it can run beside M5 without touching M5 sources.

**E3** fits beside M5 without new infrastructure. It reads archived M5a/M5b
kernel parameters (import them; don't edit hash-bound sources), builds each
d = 12 sweep matrix, and computes its entropy-production rate exactly. It reports the result beside the recorded stationary bias. It
stays `exact_reference`, and entropy production is a model property, not a
device energy.

**E4** needs only a matrix exponential for its exact reference. Add `thermox`
only if the study compares against its sampler, and then pin it with contract
tests first.

**E5** fixes an Ising target and runs Langevin spins in a double-well
potential. It measures how the distribution of spin signs converges to the
enumerated Ising distribution as J₄ grows, alongside THRML Gibbs on the same
target.

The three draft protocols below follow CLAUDE.md's habit: list the fixed choices
first and ask whether any of them caps the target metric.

---

## Draft protocol E0: THRML finite-sweep contract against Thermo's exact kernel

**Question.** On one tiny Ising model, after exactly K sweeps from a declared
initial distribution, does the state distribution that THRML samples match
the exact distribution p₀Tᴷ from Thermo's own sweep matrix? Do both converge
to the enumerated equilibrium? The question is about matched conventions, not
sampler speed.

**Scope.** For this checked chain, `thrml_local.py` calls THRML and
`finite_sweep_sampling.py` draws from precomputed exact laws. E0 as drafted
checks THRML against Thermo's exact sweep matrix. The later native experiments
add an independent JAX single-site sampler, but do not match its schedule and
initialization to this two-block THRML contract. That independent comparison
remains open (the issue raised in Sun's October 1 review).

**Model.** The checked five-spin chain in
`configs/experiments/thrml-ising-chain.toml`, with its biases, weights,
β = 0.8 and two-block partition {0, 2, 4} | {1, 3}. `exact/ising.py` already
enumerates it, and the smoke test already samples it, so the only new parts
are the finite-K reference and the sampling arm. It has 32 states, so the full
sweep matrix is exact and tiny.

**Exact reference.** Build the 32 × 32 one-sweep matrix T in NumPy from
Thermo's energy convention: update the first block given the second, then the
second given the first. The reference at budget K is p₀Tᴷ. Check that T's
stationary vector equals the enumerated Boltzmann law. The reference is
`exact_reference`, and THRML runs are `software_simulation`.

**Conventions to match.** Each one gets an explicit check, not an assumption.

| Convention | Thermo side | THRML side | How a mismatch shows |
| --- | --- | --- | --- |
| Energy sign | Energy E(s) = −β(Σ bᵢsᵢ + Σ Jᵢⱼsᵢsⱼ), β included (the config's `energy_convention`); log-weight −E, as in `exact/ising.py` | `IsingEBM` biases and weights | One-spin control: p(s = +1) must equal σ(2βb) |
| Temperature | β multiplies the whole energy | `beta` argument | The same control at two β values |
| Spin encoding | ±1 | Boolean nodes | Marginal signs flip |
| Initialization | Declared p₀ | Initial state passed to `sample_states` | Disagreement concentrated at K = 1 |
| Block order | {0, 2, 4} then {1, 3} | Order of the free blocks in the program | Small-K disagreement with this order's reference |
| Clamping | Spin 0 fixed and removed from T | Clamped block with a fixed value | Clamped marginal not exactly ±1, or a wrong conditional |
| What counts as a sweep | One update of every free block from its exact Gibbs conditional | `n_warmup` and `steps_per_sample` semantics in 0.1.4 | THRML at K matches p₀Tᴷ⁻¹ or p₀Tᴷ⁺¹ instead. Composing per-edge two-site kernels is not a sweep: it has the wrong stationary law ([Torx paper](sources/extropic-torx-paper.md), C6) |

**Arms.**
- Budgets K ∈ {1, 2, 3, 4, 8, 16, 30}. K = 30 ties back to the existing
  PAsymSwap comparison budget.
- Initializations: all spins −1 (deterministic, and the most sensitive to an
  off-by-one) and uniform. Include `hinton_init` only if its law can be
  written down from the pinned 0.1.4 source. Otherwise record that it was
  excluded and why.
- A free chain, and spin 0 clamped to each value.
- Both block orders, each against its own reference.

**Sample definition.** One sample is the full five-spin state of one
independent chain after exactly K sweeps. Use N independent chains per cell,
not a time series, so no autocorrelation correction is needed. Use distinct
JAX keys for initialization and sampling. Independently seeded runs are the
replications.

**Acceptance.** For each cell, compare the empirical 32-state distribution
with p₀Tᴷ by total variation. Derive the tolerance exactly: draw multinomial
samples of size N from p₀Tᴷ in NumPy and take a predeclared upper quantile of
their TV. Fix N and the quantile before running. Report per-spin marginal
errors alongside.

**Negative controls.** Each must fail at the chosen N:
1. Coupling sign flipped.
2. β replaced by 1/β.
3. Reversed block order, compared against the forward-order reference.
4. Off-by-one budget: THRML at K compared against p₀Tᴷ⁻¹.

Equilibrium can't detect the last two, so they must fail at K = 1 or 2.
Confirm that before running, using the exact references alone, and increase N
if they wouldn't. If a control is undetectable at every budget, say so in the
report rather than drop it.

**Optional stage B (only after the above passes).** Run one M5a site's inner
sweep through THRML: clamped blanket inputs, a hidden block, then the output
block, for K ∈ {1, 2, 4}. Compare with M5b's exact Tᴷ. This puts M5b's
hidden-reset convention (hidden spins start at −1, and the first hidden block
forgets them) on a real sampler. Import the archived M5a parameters and don't
edit hash-bound sources. Output-first order is the control, because the M5b
protocol says reset and carry-over agree only under hidden-first order.

**Not claimed.** Sampler speed or throughput, any Z1 or TSU hardware
behaviour, agreement between two independent samplers, or agreement on models
larger than those tested.

**Size.** One module, one replay test and one report. It should fit in the
`fixture-study` CI matrix. The existing 0.1.4 contract tests in
`tests/upstream_regressions/` don't fix how many block updates a schedule
performs. If the K = 1 and K = 2 cells agree, add them there as a contract.

---

## Draft protocol E1: Langevin trajectory-gradient contract

**Question.** For a small overdamped Langevin network trained to follow a
target trajectory to a fixed observation time, do the analytic gradient,
autodiff and finite differences of the Euler–Maruyama trajectory
log-likelihood agree? Does gradient descent reach the closed-form optimum? And
how close does the trained network's mean output at the observation time come
to the target's, with sampling and discretization error reported separately?

**Model.** N = 4 real spins: one clamped input, two hidden and one output.
V(x) = Σᵢ (J₂xᵢ² + J₄xᵢ⁴) + Σᵢ bᵢxᵢ + Σ₍ᵢⱼ₎ Jᵢⱼxᵢxⱼ, the family in the
paper's code. Trainable parameters are the biases b and symmetric couplings J.
J₂ and J₄ are fixed.

**Why exact.** V is linear in (b, J), with J₂ and J₄ fixed, so the
discretized log-likelihood of a fixed path is a quadratic function of the
trainable parameters for either J₄ value. Its gradient and maximizer are
available in closed form by linear least squares, independent of autodiff.
With J₄ = 0 and the input clamped, the free spins also follow an OU process,
so the output's mean and covariance at the observation time are exact.

**Fixed choices to predeclare** (check each against the metric):

| Choice | Candidate | Could it cap the result? |
| --- | --- | --- |
| Target trajectory | Hand-set smooth path per input value, not a neural-network teacher | A path the dynamics can't follow bounds the achievable likelihood. Report the optimum's residual. |
| J₂, J₄ | J₂ > 0 fixed. J₄ ∈ {0, one positive value} | Both have exact gradients. Only J₄ = 0 has an exact observation-time distribution. |
| kT, Δt, horizon | One kT, Δt, and a horizon of T steps | A large Δt lets discretization bias dominate. Report it separately at Δt and Δt/2. |
| Parameter bounds | None in the exact arm | Caps limited M4. Don't add one without a reason. |
| Seeds | Distinct JAX keys for initialization and sampling; independently seeded runs as replications | — |

**Checks.**
1. Gradient contract: analytic, autodiff and central finite differences
   agree to a stated tolerance at several random parameter points, for both
   J₄ values.
2. Optimum: gradient descent from a seeded start reaches the least-squares
   optimum within tolerance, for both J₄ values.
3. Observation-time output (J₄ = 0): exact OU mean and covariance
   (`exact_reference`) against sampled Euler–Maruyama trajectories
   (`software_simulation`) at Δt and Δt/2, over independent seeds.
4. Negative controls: a sign-flipped coupling gradient and a dropped pair term
   must fail check 1.

**Sample definition.** One sample is one complete trajectory from t = 0 to
the observation time. Report integration steps as algorithmic counts.

**Not claimed.** The paper's MNIST accuracy or its seven-orders-of-magnitude
estimate, any hardware energy or latency, and anything about Z1.

**Size.** One module, one replay test and one report. It is short enough for
the `fixture-study` CI matrix.

---

## Draft protocol E2: tiny generative thermodynamic model

**Question.** On a target distribution known exactly, does training a
Langevin network to reproduce time-reversed noising trajectories generate
that target? Does it emit less heat than an equally accurate model trained
only on the final distribution?

**Read first.** v3 of arXiv:2506.15121. Record how it parameterizes the
generator (fixed or time-dependent couplings), its noising process and its
heat definition in a short dated research note before freezing anything, as
M5 did for the Thermalizers paper.

**Arms.**
- *A (exact):* Gaussian target in d = 2 with J₄ = 0. Noising and generation
  are both OU, so the optimal parameters, the generated distribution and the
  mean heat are all closed-form.
- *B (sampled):* a two-component Gaussian mixture in d = 2 with J₄ > 0.
  Generated samples against the known target density, with distances
  estimated from independent seeded sample sets (`software_simulation`).
- *C (control):* the same architecture trained on a terminal-distribution
  loss. Compare heat at matched generation quality.
- *D (optional):* online per-step updates (Wang and Deng) against batch
  updates at matched update counts.

**Fixed choices to predeclare.** Target parameters, noising process and
horizon, kT, Δt, parameterization, optimizer, update count, and the quality
metric with its threshold for "matched quality" in arm C. Ask whether the
horizon or parameterization can cap generation quality before running B.

**Heat accounting.** Report heat per generated sample using the paper's
definition with its assumptions and units. It is a model quantity. No device
energy, power or latency follows from it.

**Not claimed.** Image-scale generation, hardware efficiency, or any
comparison with the Extropic denoising-hardware projection beyond method.

## E0 recorded, October 3

E0 is recorded in the [THRML finite-sweep contract](../experiment-reports/2026-10-03-thrml-finite-sweep-contract/findings.md):
64 of 64 finite-K cells match p0 T^K, every convention control separates, and
`hinton_init` is confirmed as the sigmoid(beta b_i) product law. The item is
closed. Stage B (one M5a site through THRML) is recorded in the
[stage B findings](../experiment-reports/2026-10-03-thrml-m5a-kernel-inner-sweep/findings.md):
under reading B, the standing choice, THRML reproduces the archived inner-K
law of the M5c primary-arm kernel at K in {1, 2, 4} in 6 of 6 cells, with 24
of 24 wrong references rejected. Stage B is closed.

## Potts stage A recorded, October 6

The categorical counterpart of E0 (dependency check A3) is recorded in the
[THRML categorical contract](../experiment-reports/2026-10-06-thrml-potts-contract/findings.md):
84 of 84 finite-K cells on a three-label, three-colour Potts patch match
p0 T^K, both categorical factor classes agree, clamping works, a q = 2
encoding of E0's chain matches E0's exact kernel, and 112 of 112 wrong
references are rejected. The Potts charter track can build on it.

## Potts stage B recorded, October 6

The [stage B findings](../experiment-reports/2026-10-06-potts-symmetry-tempering/findings.md)
test whether the October symmetry-plus-tempering result transfers to a new
model family. It does when ordinary chains trap (beta 16) and does not when
they mix (beta 8), where retaining every replica matters more. Candidate
follow-ups: a trapping detector chosen on held-out seeds, reweighting hot
replicas, and larger or field-bearing Potts targets.
